# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check :mod:`kernels.dsv4_moe_layer.reference` against DeepSeek's own reference.

The oracle is the unmodified ``inference/model.py`` from the DeepSeek-V4-Pro
Hugging Face repo, driven through a pure-torch stand-in for its tilelang kernels.
It is not vendored here; point ``DSV4_ORACLE_DIR`` at a checkout containing
``model.py``, ``kernel.py`` and ``fast_hadamard_transform.py``, or the tests skip.

Only the sliding-window layer is covered, matching the golden's current
scope: the oracle's ``Attention``/``MoE`` submodules are driven directly so the
comparison excludes hyper-connections (which land with the mHC work).
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

from kernels.dsv4_moe_layer.config import MoeMode, moe_format
from kernels.dsv4_moe_layer.reference import (
    V4Config,
    compress_step,
    contiguous_pool,
    expert_matrix,
    fp4_row_bytes,
    fp8_mats,
    golden_layer,
    indexer_step,
    layer_idxs,
    make_weights,
    pack_fp4,
    qkv_a_split,
    quant_dequant_fp4,
    rmsnorm,
    unpack_fp4,
    window_idxs,
)
from kernels.mla_moe_layer.reference import bf, dequant, rope_table

ORACLE_DIR = os.environ.get("DSV4_ORACLE_DIR", "/home/weisu/dsv4_oracle")
# top-k margin below which the two implementations may legitimately pick different
# experts; see the note where it is used
NEAR_TIE = 0.05

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]


def _oracle():
    if not os.path.isfile(os.path.join(ORACLE_DIR, "model.py")):
        pytest.skip(f"DeepSeek-V4 oracle not found at {ORACLE_DIR}")
    if ORACLE_DIR not in sys.path:
        sys.path.insert(0, ORACLE_DIR)
    import model as oracle_model

    # model.py keeps scale_fmt as a module global that only Transformer.__init__
    # sets; these tests build its submodules directly, so without this every FP8
    # round trip in the oracle used exact scales whatever ModelArgs said. The
    # checkpoint's is ue8m0 -- power-of-two scales, as the golden and kernel use.
    oracle_model.scale_fmt = "ue8m0"
    return oracle_model


def _cfg(hc_mult=4, compress_ratio=0, max_seq=1024):
    # Small but structurally faithful: nope_dim stays a multiple of 64 and the
    # rope tail keeps V4's own 64 lanes.
    return V4Config(
        heads=8,
        hidden=256,
        q_lora=128,
        head_dim=128,
        rope_dim=64,
        o_groups=2,
        o_lora=64,
        n_experts=8,
        top_k=2,
        inter=128,  # must be a multiple of the 128 FP8 block
        window=32,
        hc_mult=hc_mult,
        compress_ratio=compress_ratio,
        max_seq=max_seq,
    )


def _oracle_modules(om, cfg, device):
    args = om.ModelArgs(
        max_batch_size=1,
        max_seq_len=256,
        dtype="bf16",
        scale_fmt="ue8m0",  # the checkpoint's power-of-two scales, as the golden and the kernel use
        scale_dtype="fp32",
        vocab_size=32,
        dim=cfg.hidden,
        moe_inter_dim=cfg.inter,
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        n_heads=cfg.heads,
        n_routed_experts=cfg.n_experts,
        n_shared_experts=1,
        n_activated_experts=cfg.top_k,
        score_func="sqrtsoftplus",
        route_scale=cfg.route_scale,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.rope_dim,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora,
        window_size=cfg.window,
        compress_ratios=(0,),
        norm_eps=cfg.eps,
        rope_theta=cfg.rope_theta,
        original_seq_len=0,
    )
    with torch.device(device):
        attn = om.Attention(0, args)
        moe = om.MoE(0, args)
    return attn, moe


@torch.no_grad()
def _load_oracle_weights(attn, moe, W, cfg, weight_fmt):
    t = W.t
    dq = {n: dequant(t[f"w_{n}"], t[f"s_{n}"], bk) for n, (_, _, bk) in fp8_mats(cfg).items()}
    bf16 = torch.bfloat16

    attn.wq_a.weight.copy_(dq["qkv_a"][: cfg.q_lora].to(bf16))
    attn.wkv.weight.copy_(dq["qkv_a"][cfg.q_lora : cfg.q_lora + cfg.head_dim].to(bf16))
    attn.wq_b.weight.copy_(dq["q_b"].to(bf16))
    attn.wo_a.weight.copy_(dq["o_a"].to(bf16))
    attn.wo_b.weight.copy_(dq["o_b"].to(bf16))
    attn.q_norm.weight.copy_(t["g_q"].float())
    attn.kv_norm.weight.copy_(t["g_kv"].float())
    attn.attn_sink.copy_(t["attn_sink"].float())

    moe.gate.weight.copy_(t["w_r"].to(bf16))
    moe.gate.bias.copy_(t["bias"].float())
    for e in range(cfg.n_experts + 1):
        ug = expert_matrix(t, "ug", e, cfg, weight_fmt)
        dn = expert_matrix(t, "dn", e, cfg, weight_fmt)
        target = moe.shared_experts if e == cfg.shared_expert else moe.experts[e]
        target.w1.weight.copy_(ug[: cfg.inter].to(bf16))
        target.w3.weight.copy_(ug[cfg.inter :].to(bf16))
        target.w2.weight.copy_(dn.to(bf16))


@torch.no_grad()
def _oracle_step(om, attn, moe, h, pos, g_in, g_post, eps):
    """One layer of the oracle, with the plain residual the golden currently models."""
    # the oracle builds its window/compress index tensors with a bare
    # torch.arange, so it needs the default device pointed at the GPU
    with torch.device(h.device):
        x = bf(rmsnorm(h, g_in, eps)).to(torch.bfloat16)
        o = attn(x.unsqueeze(0), pos).squeeze(0)
        a = (h.float() + o.float()).to(torch.bfloat16)
        x2 = bf(rmsnorm(a, g_post, eps)).to(torch.bfloat16)
        ids = torch.zeros(1, x2.shape[0], dtype=torch.long, device=h.device)
        y = moe(x2.unsqueeze(0), ids).squeeze(0)
    return a, (a.float() + y.float()).to(torch.bfloat16)


@pytest.mark.parametrize("steps", [6])
def test_v4_layer_matches_deepseek_reference(steps):
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1)  # this case drives attn/ffn directly, around mHC

    # BF16 expert activations so both sides feed the experts the same tensor;
    # the weights are FP8 block-scaled here and dequantized into the oracle.
    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    attn, moe = _oracle_modules(om, cfg, device)
    _load_oracle_weights(attn, moe, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(256, theta=cfg.rope_theta, device=device)
    kv_cache = torch.zeros(cfg.window, cfg.head_dim, dtype=torch.bfloat16, device=device)

    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(W, h, [pos], kv_cache, dest, idx, cos, sin, lambda z: z, moe_mode=MoeMode.W8A16)
        a_ref, out_ref = _oracle_step(om, attn, moe, h, pos, W.t["g_in"], W.t["g_post"], cfg.eps)

        da = (res["a"].float() - a_ref.float()).abs().max().item()
        do = (res["x_out"].float() - out_ref.float()).abs().max().item()
        scale = out_ref.float().abs().max().item()
        assert da < 3e-2 * max(scale, 1.0), f"pos={pos} attention half diverges: {da}"
        assert do < 5e-2 * max(scale, 1.0), f"pos={pos} layer output diverges: {do}"


# --------------------------------------------------------------- full block
# With hyper-connections in the golden, the comparison no longer has to drive the
# oracle's submodules around mHC -- it can run DeepSeek's whole Block, which also
# covers hc_pre / hc_post / the Sinkhorn normalisation and the layer's [S, hc, d]
# residual contract.


def _oracle_args(om, cfg):
    return om.ModelArgs(
        max_batch_size=1,
        max_seq_len=256,
        dtype="bf16",
        scale_fmt="ue8m0",
        scale_dtype="fp32",
        vocab_size=32,
        dim=cfg.hidden,
        moe_inter_dim=cfg.inter,
        n_layers=1,
        n_hash_layers=0,
        n_mtp_layers=0,
        n_heads=cfg.heads,
        n_routed_experts=cfg.n_experts,
        n_shared_experts=1,
        n_activated_experts=cfg.top_k,
        score_func="sqrtsoftplus",
        route_scale=cfg.route_scale,
        swiglu_limit=cfg.swiglu_limit,
        q_lora_rank=cfg.q_lora,
        head_dim=cfg.head_dim,
        rope_head_dim=cfg.rope_dim,
        o_groups=cfg.o_groups,
        o_lora_rank=cfg.o_lora,
        window_size=cfg.window,
        compress_ratios=(cfg.compress_ratio,),
        norm_eps=cfg.eps,
        rope_theta=cfg.rope_theta,
        compress_rope_theta=cfg.compress_rope_theta,
        original_seq_len=0,  # YaRN off: it is a host-side rope table, not kernel work
        hc_mult=cfg.hc_mult,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
    )


@torch.no_grad()
def _load_block_weights(block, W, cfg, weight_fmt):
    t = W.t
    _load_oracle_weights(block.attn, block.ffn, W, cfg, weight_fmt)
    if cfg.compress_ratio:
        # the compressors' projections live in our fused qkv_a; split them back out
        dq = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
        cut = qkv_a_split(cfg)
        c = block.attn.compressor
        c.wkv.weight.copy_(dq[slice(*cut["c_kv"])].float())
        c.wgate.weight.copy_(dq[slice(*cut["c_gate"])].float())
        c.ape.copy_(t["ape"].float())
        c.norm.weight.copy_(t["g_ckv"].float())
        if cfg.indexed:
            ix = block.attn.indexer
            ix.compressor.wkv.weight.copy_(dq[slice(*cut["i_kv"])].float())
            ix.compressor.wgate.weight.copy_(dq[slice(*cut["i_gate"])].float())
            ix.compressor.ape.copy_(t["i_ape"].float())
            ix.compressor.norm.weight.copy_(t["g_ickv"].float())
            ix.wq_b.weight.copy_(dequant(t["w_i_q_b"], t["s_i_q_b"], 128))
            ix.weights_proj.weight.copy_(t["i_w"])
            # the model builds this buffer under a bf16 default dtype, and scores
            # its bf16 queries straight against it; these tests default to fp32
            ix.kv_cache = ix.kv_cache.to(torch.bfloat16)
            ix.compressor.kv_cache = None  # re-bound from ix.kv_cache on first use
    block.attn_norm.weight.copy_(t["g_in"].float())
    block.ffn_norm.weight.copy_(t["g_post"].float())
    for side in ("attn", "ffn"):
        # ours is row-padded to the MFMA group; the oracle's is exactly hc_mix
        getattr(block, f"hc_{side}_fn").copy_(t[f"hc_{side}_fn"][: cfg.hc_mix].float())
        getattr(block, f"hc_{side}_base").copy_(t[f"hc_{side}_base"].float())
        getattr(block, f"hc_{side}_scale").copy_(t[f"hc_{side}_scale"].float())


@pytest.mark.parametrize("steps", [5])
def test_v4_block_with_hyper_connections_matches_deepseek(steps):
    """The golden's full layer, hyper-connections included, against DeepSeek's Block."""
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg()
    assert cfg.hc_mult > 1, "this case is about the hyper-connection path"

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    with torch.device(device):
        block = om.Block(0, _oracle_args(om, cfg))
    _load_block_weights(block, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(256, theta=cfg.rope_theta, device=device)
    kv_cache = torch.zeros(cfg.window, cfg.head_dim, dtype=torch.bfloat16, device=device)

    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(W, h, [pos], kv_cache, dest, idx, cos, sin, lambda z: z, moe_mode=MoeMode.W8A16)
        with torch.device(device):
            ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            ref = block(h.unsqueeze(0), pos, ids).squeeze(0)

        assert res["x_out"].shape == h.shape, f"layer must preserve [S, hc, hidden], got {res['x_out'].shape}"
        d = (res["x_out"].float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert d < 5e-2 * max(scale, 1.0), f"pos={pos} block output diverges: {d} (|ref| {scale})"


@pytest.mark.parametrize("steps", [40])
def test_v4_hca_compressor_matches_deepseek(steps):
    """The HCA layer (compress_ratio 128) against DeepSeek's own Block.

    Needs enough steps to cross a compression boundary: only every
    ``compress_ratio``-th token emits a compressed entry, and the attention only
    starts gathering compressed rows once one exists.
    """
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    ratio = 16  # stands in for V4's 128; same code path, far fewer steps to cross
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    cfg.compress_ratio = ratio
    cfg.validate()

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    with torch.device(device):
        block = om.Block(0, _oracle_args(om, cfg))
    _load_block_weights(block, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    kv_cache = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device)
    kv_state = torch.zeros(1, ratio, cfg.head_dim, device=device)
    score_state = torch.zeros(1, ratio, cfg.head_dim, device=device)

    compressed_seen = 0
    near_ties = 0
    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(
            W,
            h,
            [pos],
            kv_cache,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            kv_state=kv_state,
            score_state=score_state,
            cos_c=cos,
            sin_c=sin,
        )
        with torch.device(device):
            ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            ref = block(h.unsqueeze(0), pos, ids).squeeze(0)
        if (pos + 1) % ratio == 0:
            compressed_seen += 1

        # Routing is a discrete top-k, so a near-tie between the last kept expert
        # and the first dropped one can resolve differently here and in the oracle
        # over a difference far smaller than either is accurate to. Scores agree to
        # about 0.005 relative on magnitudes near 3, i.e. ~0.017 absolute, so a
        # margin under NEAR_TIE is genuinely at risk and the comparison past it
        # says nothing about the compressor.
        sc = res["scores"][0].float() + W.t["bias"].float()
        top = torch.topk(sc, cfg.top_k + 1).values
        if (top[cfg.top_k - 1] - top[cfg.top_k]).item() < NEAR_TIE:
            near_ties += 1
            continue

        d = (res["x_out"].float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert d < 5e-2 * max(scale, 1.0), f"pos={pos} diverges: {d} (|ref| {scale})"

    assert compressed_seen >= 2, "the run must cross at least two compression boundaries"
    assert near_ties < steps // 4, f"too many near-ties to have tested much: {near_ties}/{steps}"


@pytest.mark.parametrize("ratio", [4, 8])
def test_v4_compressor_matches_deepseek_directly(ratio):
    """The compressor alone, against DeepSeek's Compressor module.

    ratio 4 is CSA's OVERLAPPING form -- each entry pools 2*ratio tokens at a
    stride of ratio, taking the previous window's overlap channels and the
    current window's normal ones. Tested here in isolation because a whole CSA
    layer also needs the lightning indexer, which is not implemented.
    """
    om = _oracle()
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1, compress_ratio=ratio, max_seq=256)
    cfg.compress_ratio = ratio
    assert cfg.overlap == (ratio == 4), "overlap is tied to ratio 4"

    W = make_weights(rank=0, cfg=cfg, device=device, seed=11, moe_mode=MoeMode.W8A16)
    t = W.t
    args = _oracle_args(om, cfg)
    with torch.device(device):
        comp = om.Compressor(args, ratio, cfg.head_dim)
        comp.kv_cache = torch.zeros(1, cfg.n_compressed, cfg.head_dim, device=device)
        comp.freqs_cis = om.precompute_freqs_cis(cfg.rope_dim, 512, 0, cfg.compress_rope_theta, 1.0, 32, 1)
    dq = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    coff = cfg.c_coff
    # by name: at ratio 4 the fused GEMV also carries the indexer's compressor,
    # so the attention compressor's pair is no longer the tail
    cut = qkv_a_split(cfg)
    with torch.no_grad():
        comp.wkv.weight.copy_(dq[slice(*cut["c_kv"])].float())
        comp.wgate.weight.copy_(dq[slice(*cut["c_gate"])].float())
        comp.ape.copy_(t["ape"].float())
        comp.norm.weight.copy_(t["g_ckv"].float())

    cos, sin = rope_table(512, theta=cfg.compress_rope_theta, device=device)
    cache = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device)
    kv_state = torch.zeros(cfg.c_rows, coff * cfg.head_dim, device=device)
    # -inf, not zero: with overlapping windows the previous window's rows are not
    # written before the first emit, and they must drop out of the softmax
    score_state = torch.full((cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device)

    emitted = 0
    for pos in range(6 * ratio):
        x = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        proj = x.float() @ dq.float().T
        ours = compress_step(
            proj[0, slice(*cut["c_kv"])],
            proj[0, slice(*cut["c_gate"])],
            pos,
            cfg,
            t,
            kv_state,
            score_state,
            cache,
            cos,
            sin,
            dest_row=cfg.window + pos // ratio,
        )
        with torch.device(device):
            theirs = comp(x.unsqueeze(0), pos)
        if (pos + 1) % ratio:
            assert ours is None and theirs is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        d = (ours.float() - theirs.reshape(-1).float()).abs().max().item()
        scale = max(theirs.float().abs().max().item(), 1e-6)
        assert d < 2e-2 * scale, f"ratio={ratio} pos={pos} compressed entry differs: {d / scale:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


def test_v4_indexer_compressor_matches_deepseek():
    """The INDEXER's compressor, against DeepSeek's `Compressor(..., rotate=True)`.

    CSA's indexer runs a second compressor over the same tokens at its own smaller
    head_dim, and finishes differently: a Hadamard rotation over the whole row and
    then FP4, where the attention compressor does FP8 over the nope part only. The
    rotation is what makes FP4 survivable -- it spreads an outlier across every
    lane, so a block's amax stops being set by one coordinate.

    Ratio is 4 here because that is CSA, so this also covers the overlapping
    pooling at the indexer's dimensions.
    """
    om = _oracle()
    device, ratio = "cuda", 4
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1, compress_ratio=ratio, max_seq=256)
    ihd, coff = 128, cfg.c_coff  # index_head_dim; V4 keeps rope_head_dim at 64
    assert cfg.overlap, "ratio 4 is the overlapping form"

    args = _oracle_args(om, cfg)
    with torch.device(device):
        comp = om.Compressor(args, ratio, ihd, True)
        n_comp = cfg.max_seq // ratio
        comp.kv_cache = torch.zeros(1, n_comp, ihd, device=device)
        comp.freqs_cis = om.precompute_freqs_cis(cfg.rope_dim, 512, 0, cfg.compress_rope_theta, 1.0, 32, 1)

    gen = torch.Generator(device=device).manual_seed(5)
    wkv = torch.randn(coff * ihd, cfg.hidden, generator=gen, device=device) / cfg.hidden**0.5
    wgate = torch.randn(coff * ihd, cfg.hidden, generator=gen, device=device) / cfg.hidden**0.5
    ape = 0.5 * torch.randn(ratio, coff * ihd, generator=gen, device=device)
    gamma = (1 + 0.1 * torch.randn(ihd, generator=gen, device=device)).to(torch.bfloat16)
    with torch.no_grad():
        comp.wkv.weight.copy_(wkv.float())
        comp.wgate.weight.copy_(wgate.float())
        comp.ape.copy_(ape.float())
        comp.norm.weight.copy_(gamma.float())

    cos, sin = rope_table(512, theta=cfg.compress_rope_theta, device=device)
    cache = torch.zeros(n_comp, fp4_row_bytes(ihd), dtype=torch.uint8, device=device)
    kv_state = torch.zeros(cfg.c_rows, coff * ihd, device=device)
    score_state = torch.full((cfg.c_rows, coff * ihd), float("-inf"), device=device)

    emitted = 0
    for pos in range(6 * ratio):
        x = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        ours = compress_step(
            (x.float() @ wkv.float().T)[0],
            (x.float() @ wgate.float().T)[0],
            pos,
            cfg,
            None,
            kv_state,
            score_state,
            cache,
            cos,
            sin,
            head_dim=ihd,
            ape=ape,
            gamma=gamma,
            rotate=True,
        )
        with torch.device(device):
            theirs = comp(x.unsqueeze(0), pos)
        if (pos + 1) % ratio:
            assert ours is None and theirs is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        d = (ours.float() - theirs.reshape(-1).float()).abs().max().item()
        scale = max(theirs.float().abs().max().item(), 1e-6)
        assert d < 2e-2 * scale, f"pos={pos} indexer compressed entry differs: {d / scale:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


def test_mxfp8_ceil_scale_never_clips():
    """With the E8M0 scale rounded up, every block's maximum stays in range, so
    each element is off by at most one E4M3 half-ulp (2**-4 relative).

    Rounding the scale to nearest instead clamps a block maximum whose
    ``amax / 448`` has mantissa >= 1.5 by up to a third."""
    from kernels.common.mx_formats import quant_dequant_mxfp8 as nearest_mxfp8
    from kernels.dsv4_moe_layer.reference import quant_dequant_mxfp8

    torch.manual_seed(0)
    x = torch.randn(64, 1024) * torch.exp2(torch.randint(-8, 8, (64, 1)).float())
    x[:, ::32] *= 20.0  # an outlier per block, as real activations have
    q = quant_dequant_mxfp8(x)
    blocks = x.reshape(64, -1, 32)
    amax = blocks.abs().amax(-1, keepdim=True)
    err = (q.reshape(64, -1, 32) - blocks).abs()
    assert (err <= 2**-4 * blocks.abs() + amax * 2**-17).all(), "ceil-scaled MXFP8 clipped or over-rounded"
    assert (q.abs().reshape(64, -1, 32).amax(-1, keepdim=True) >= amax * (1 - 2**-4)).all(), "a block max clipped"
    # the nearest-rounded scale does clip on this input
    assert (nearest_mxfp8(x) - x).norm() > 2 * (q - x).norm()


def test_v4_indexer_cache_packs_fp4_losslessly():
    """The indexer's key cache stores FP4 codes plus e8m0 block scales, not values.

    Two things are pinned. The round trip: what the packed row decodes to is
    exactly ``quant_dequant_fp4``, so the format loses nothing the quantizer
    had not already dropped -- across block scales from tiny to large, with
    zeros and signs. And the byte layout itself, which the kernel writes and
    decodes on its own: a round trip alone would pass with the nibble order or
    the scale bias swapped consistently on both host sides.
    """
    torch.manual_seed(0)
    n = 128
    x = torch.randn(64, n) * torch.logspace(-20, 12, 64, base=2.0)[:, None]
    x[5] = 0  # a zero row: the clamp's smallest scale
    x[6, 32:64] = 0  # one zero block among live ones
    p = pack_fp4(x)
    assert p.dtype == torch.uint8 and p.shape == (64, fp4_row_bytes(n)) == (64, 68)
    assert torch.equal(unpack_fp4(p), quant_dequant_fp4(x))

    # element i in nibble i % 2 of byte i // 2, sign in bit 3; one e8m0 per 32
    y = torch.zeros(n)
    y[0], y[1], y[2], y[33] = 6.0, -0.5, -6.0, 3.0  # block 0 scale 1, block 1 scale 0.5
    q = pack_fp4(y)
    assert q[0].item() == 0x7 | (0x9 << 4), f"byte 0 {q[0].item():#x}"
    assert q[1].item() == 0xF, f"byte 1 {q[1].item():#x}"
    assert q[16].item() == 0x7 << 4, f"byte 16 {q[16].item():#x}"
    assert q[64:].tolist() == [127, 126, 1, 1], q[64:].tolist()


def test_v4_indexer_matches_deepseek():
    """The whole lightning indexer, against DeepSeek's Indexer module.

    What is under test is the SELECTION, not the scores: the kernel will gather
    whatever slots come back, so picking the same set is the thing that matters.
    Scores are compared too, because a near-tie at the top-k boundary is the only
    legitimate way the sets may differ and the score margin is what proves it.

    Single rank, so `allreduce` is the identity -- the cross-rank sum is exercised
    by the TP layer tests, not here.
    """
    om = _oracle()
    device, ratio = "cuda", 4
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1, compress_ratio=ratio, max_seq=256)
    # V4-Pro's 1024 would exceed every compressed entry this run produces, making
    # the top-k select all of them and the comparison vacuous. A small k is the
    # same code path and actually discriminates on score.
    cfg.index_topk = 4
    assert cfg.indexed, "ratio 4 is the indexed form"
    ih, ihd = cfg.index_heads, cfg.index_head_dim

    W = make_weights(rank=0, cfg=cfg, device=device, seed=13, moe_mode=MoeMode.W8A16)
    t = W.t
    args = _oracle_args(om, cfg)
    args.index_n_heads, args.index_head_dim, args.index_topk = ih, ihd, cfg.index_topk
    with torch.device(device):
        idxr = om.Indexer(args, ratio)
        # bf16: the model runs under a bf16 default dtype, and the indexer scores
        # its bf16 queries straight against this buffer
        idxr.kv_cache = torch.zeros(1, cfg.n_compressed, ihd, dtype=torch.bfloat16, device=device)
        idxr.freqs_cis = om.precompute_freqs_cis(cfg.rope_dim, 512, 0, cfg.rope_base, 1.0, 32, 1)

    dq = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    cut = qkv_a_split(cfg)
    with torch.no_grad():
        idxr.compressor.wkv.weight.copy_(dq[slice(*cut["i_kv"])].float())
        idxr.compressor.wgate.weight.copy_(dq[slice(*cut["i_gate"])].float())
        idxr.compressor.ape.copy_(t["i_ape"].float())
        idxr.compressor.norm.weight.copy_(t["g_ickv"].float())
        idxr.wq_b.weight.copy_(dequant(t["w_i_q_b"], t["s_i_q_b"], 128))
        idxr.weights_proj.weight.copy_(t["i_w"])

    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    i_cache = torch.zeros(cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device)
    i_state = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=device)
    i_score = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=device)

    checked = discriminated = 0
    for pos in range(8 * ratio):
        x = (0.5 * torch.randn(1, cfg.hidden, device=device)).to(torch.bfloat16)
        # q_norm returns bf16 in the model, and both wq_b's consume it as such
        q_a_n = rmsnorm(0.5 * torch.randn(1, cfg.q_lora, device=device), t["g_q"], cfg.eps).to(torch.bfloat16)
        proj = x.float() @ dq.float().T
        ours = indexer_step(
            x[0],
            q_a_n[0],
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_state,
            i_score,
            i_cache,
            cos,
            sin,
            lambda z: z,
        )
        with torch.device(device):
            # offset 0: indexer_step now returns compressed ENTRY indices, and
            # the plane row of entry 0 is the caller's to add (contiguous_pool)
            theirs = idxr(x.unsqueeze(0), q_a_n.unsqueeze(0), pos, 0)
        n = (pos + 1) // ratio
        if not n:
            assert int((ours >= 0).sum()) == 0, f"pos={pos}: nothing compressed yet"
            continue
        checked += 1
        a = set(ours[ours >= 0].tolist())
        b = set(theirs.reshape(-1).tolist())
        assert a == b, f"pos={pos} selected {sorted(a)} vs {sorted(b)}"
        if n > cfg.index_topk:
            discriminated += 1
            assert len(a) == cfg.index_topk, f"pos={pos} picked {len(a)}, want {cfg.index_topk}"

    assert checked >= 6, f"expected several scored steps, got {checked}"
    # otherwise every step selected everything and the scores were never tested
    assert discriminated >= 3, f"top-k never had to discard anything ({discriminated})"


@pytest.mark.parametrize("steps", [40])
def test_v4_csa_layer_matches_deepseek(steps):
    """A whole CSA layer (compress_ratio 4) against DeepSeek's own Block.

    This is the case the golden could not run before: overlapping compression,
    the lightning indexer choosing which compressed entries attention sees, and
    the fused qkv_a carrying both compressors' projections at once.

    `index_topk` is reduced so the selection actually discards entries -- at
    V4-Pro's 1024 it would keep every compressed row this run produces and the
    indexer would be untested.
    """
    om = _oracle()
    device, ratio = "cuda", 4
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    cfg.index_topk = 4
    cfg.validate()
    assert cfg.indexed and cfg.overlap, "ratio 4 is the indexed, overlapping form"

    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    args = _oracle_args(om, cfg)
    args.index_n_heads = cfg.index_heads
    args.index_head_dim = cfg.index_head_dim
    args.index_topk = cfg.index_topk
    with torch.device(device):
        block = om.Block(0, args)
    assert block.attn.indexer is not None, "the oracle must have built an indexer"
    _load_block_weights(block, W, cfg, moe_format(MoeMode.W8A16).weight)

    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    ihd, coff = cfg.index_head_dim, cfg.c_coff
    kv_cache = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device)
    kv_state = torch.zeros(1, cfg.c_rows, coff * cfg.head_dim, device=device)
    score_state = torch.full((1, cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device)
    i_cache = torch.zeros(1, cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device)
    i_state = torch.zeros(1, cfg.c_rows, coff * ihd, device=device)
    i_score = torch.full((1, cfg.c_rows, coff * ihd), float("-inf"), device=device)

    selected, near_ties = 0, 0
    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx, dest = contiguous_pool([pos], cfg, device)
        res = golden_layer(
            W,
            h,
            [pos],
            kv_cache,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            kv_state=kv_state,
            score_state=score_state,
            cos_c=cos,
            sin_c=sin,
            i_state=i_state,
            i_score_state=i_score,
            i_cache=i_cache,
        )
        with torch.device(device):
            ids = torch.zeros(1, 1, dtype=torch.long, device=device)
            ref = block(h.unsqueeze(0), pos, ids).squeeze(0)
        if (pos + 1) // ratio > cfg.index_topk:
            selected += 1

        top = res["scores"].reshape(-1).sort(descending=True).values
        if (top[cfg.top_k - 1] - top[cfg.top_k]).item() < NEAR_TIE:
            near_ties += 1
            continue
        d = (res["x_out"].float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        assert d < 5e-2 * max(scale, 1.0), f"pos={pos} diverges: {d} (|ref| {scale})"

    assert selected >= 5, f"the indexer never had to discard anything ({selected})"
    assert near_ties < steps // 4, f"too many near-ties to have tested much: {near_ties}/{steps}"


@pytest.mark.parametrize("ratio", [0, 4])
def test_v4_golden_batches_independent_sequences(ratio):
    """S samples are S independent sequences: batching must change nothing.

    This is the definition of the batch axis, so it is worth asserting directly
    rather than inferring it from a per-stage comparison. Two sequences are run
    together at S=2 and again separately at S=1, each with its own cache and
    compressor state; sample s of the batched run must equal run s.

    It is the test that would catch a stage reading sample 0's state for every
    sample -- a real defect class here, since the whole design keeps rolling
    compressor state per sequence. ratio 4 covers the CSA path, where the
    indexer carries three more per-sequence buffers than the window does.
    """
    device = "cuda"
    torch.manual_seed(0)
    cfg = _cfg(hc_mult=4, compress_ratio=ratio, max_seq=256)
    if ratio:
        cfg.index_topk = 4
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=device, seed=7, moe_mode=MoeMode.W8A16)
    cos, sin = rope_table(512, theta=cfg.rope_base, device=device)
    ihd, coff, S = cfg.index_head_dim, cfg.c_coff, 2

    def state(n):
        """The per-sequence buffers golden_layer threads, for a batch of n."""
        # ONE plane; sample s owns rows [s * cache_rows, (s+1) * cache_rows)
        d = dict(kv_cache=torch.zeros(n * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=device))
        if not ratio:
            return d
        d |= dict(
            kv_state=torch.zeros(n, cfg.c_rows, coff * cfg.head_dim, device=device),
            score_state=torch.full((n, cfg.c_rows, coff * cfg.head_dim), float("-inf"), device=device),
            cos_c=cos,
            sin_c=sin,
        )
        if cfg.indexed:
            d |= dict(
                i_state=torch.zeros(n, cfg.c_rows, coff * ihd, device=device),
                i_score_state=torch.full((n, cfg.c_rows, coff * ihd), float("-inf"), device=device),
                i_cache=torch.zeros(n, cfg.n_compressed, fp4_row_bytes(ihd), dtype=torch.uint8, device=device),
            )
        return d

    def run(st, h, pos):
        n = h.shape[0]
        kw = dict(st)
        idx, dest = contiguous_pool([pos] * n, cfg, device)
        return golden_layer(
            W,
            h,
            [pos] * n,
            kw.pop("kv_cache"),
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=MoeMode.W8A16,
            **kw,
        )

    batched, alone = state(S), [state(1) for _ in range(S)]
    steps = 4 * cfg.window // 3
    for pos in range(steps):
        h = (0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        got = run(batched, h, pos)["x_out"]
        for s in range(S):
            want = run(alone[s], h[s : s + 1], pos)["x_out"]
            d = (got[s].float() - want[0].float()).abs().max().item()
            assert d < 2e-3, f"pos={pos} sample {s}: batched differs from its own run by {d:.3e}"
    # the run has to be deep enough that the compressor and its state actually ran
    if ratio:
        assert steps > 2 * ratio, "too shallow to have crossed a compression boundary"
