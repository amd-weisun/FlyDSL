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
    fp8_mats,
    golden_layer,
    layer_idxs,
    make_weights,
    rmsnorm,
    window_idxs,
)
from kernels.mla_moe_layer.reference import bf, dequant, dequant_expert, rope_table

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
        scale_fmt=None,  # exact amax scales, matching the kernel's FP8 path
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
        ug = dequant_expert(t["w_ug"][e], t["s_ug"][e], weight_fmt)
        dn = dequant_expert(t["w_dn"][e], t["s_dn"][e], weight_fmt)
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
        idx = window_idxs(pos, 1, cfg.window, device)
        res = golden_layer(W, h, pos, kv_cache, idx, cos, sin, lambda z: z, moe_mode=MoeMode.W8A16)
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
        scale_fmt=None,
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
        # the compressor's projections live in our fused qkv_a; split them back out
        hd, ql = cfg.head_dim, cfg.q_lora
        dq = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
        c = block.attn.compressor
        c.wkv.weight.copy_(dq[ql + hd : ql + 2 * hd].float())
        c.wgate.weight.copy_(dq[ql + 2 * hd :].float())
        c.ape.copy_(t["ape"].float())
        c.norm.weight.copy_(t["g_ckv"].float())
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
        idx = window_idxs(pos, 1, cfg.window, device)
        res = golden_layer(W, h, pos, kv_cache, idx, cos, sin, lambda z: z, moe_mode=MoeMode.W8A16)
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
    kv_state = torch.zeros(ratio, cfg.head_dim, device=device)
    score_state = torch.zeros(ratio, cfg.head_dim, device=device)

    compressed_seen = 0
    near_ties = 0
    for pos in range(steps):
        h = (0.5 * torch.randn(1, cfg.hc_mult, cfg.hidden, device=device)).to(torch.bfloat16)
        idx = layer_idxs(pos, 1, cfg, device)
        res = golden_layer(
            W,
            h,
            pos,
            kv_cache,
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
    hd, ql, coff = cfg.head_dim, cfg.q_lora, cfg.c_coff
    with torch.no_grad():
        comp.wkv.weight.copy_(dq[ql + hd : ql + hd + coff * hd].float())
        comp.wgate.weight.copy_(dq[ql + hd + coff * hd :].float())
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
        proj = (x.float() @ dq.float().T)[:, ql + hd :]
        ours = compress_step(
            proj[0, : coff * hd],
            proj[0, coff * hd :],
            pos,
            cfg,
            t,
            kv_state,
            score_state,
            cache,
            cos,
            sin,
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
    cache = torch.zeros(n_comp, ihd, dtype=torch.bfloat16, device=device)
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
            base=0,  # the indexer's cache holds compressed entries only
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
