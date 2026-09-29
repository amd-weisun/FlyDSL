# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the fused DeepSeek-V4 layer kernel against its torch golden, stage by stage.

Covers the sliding-window layer (``compress_ratio == 0``), which is what
:mod:`kernels.dsv4_moe_layer.dsv4_kernel` implements today. Comparing every
mailbox rather than only ``x_out`` means a regression names the stage it broke.

The reduced shard keeps every ratio the kernel's mappings actually depend on --
``head_dim`` stays 512 (the PV MFMA hands each of 8 waves a 32-dim group and the
KV gather hands each of 64 lanes a slice of it) and ``head_dim - rope_dim``
stays a multiple of 64 -- while shrinking hidden/expert counts so the test is
cheap.
"""

from __future__ import annotations

import pytest
import torch

from kernels.dsv4_moe_layer.config import COMPRESS_CSA, COMPRESS_HCA, MoeMode
from kernels.dsv4_moe_layer.reference import (
    V4Config,
    golden_layer,
    golden_moe,
    layer_idxs,
    contiguous_pool,
    make_weights,
    dequant,
    qkv_a_split,
    rmsnorm,
    window_idxs,
)
from kernels.mla_moe_layer.reference import bf, rope_table

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# Calibrated empirically, not guessed: 12 draws x {w8a8, a8w4} x {hc_mult 1, 4}
# x {tp 1, 8}. Each bar is roughly 2-4x the worst relative error seen there.
#
# The attention half does not move with configuration -- every stage stayed under
# 0.0065 in all eight -- so its bars are fixed, and tighter than the blanket 0.02
# they replace. Only `mid` and the end-to-end `x_out` scale, and they do so with
# the arithmetic: mHC mixes hc_mult streams, multi-rank sums bf16 partials, and
# MXFP4 amplifies whatever reaches it. Each of those roughly doubles the error,
# so the bar doubles with each rather than being set to one loose worst case.
STAGE_TOL = {
    "q_a": 1e-3,  # worst seen 1e-4
    "kv": 1e-3,  # worst seen 1e-4
    "q": 0.010,  # worst seen 0.0039
    "o": 0.015,  # worst seen 0.0063
    "o_lora": 0.015,  # worst seen 0.0052
    "a": 0.015,  # worst seen 0.0065
    "scores": 0.012,  # worst seen 0.0052
    "mid": 0.050,  # worst seen 0.0236 at hc=1/tp1, 0.109 at hc=4/tp8/a8w4
}
SCALES_WITH_CONFIG = ("mid",)
# end to end, judged by relative L2: a single rounding flip upstream moves one
# element a long way, and hc_post then mixes it across hc_mult streams
OUT_REL_L2 = 0.050  # worst seen 0.0278 at hc=1/tp1, 0.0914 at hc=4/tp8/a8w4


def _tol(base, hc_mult, npes):
    """Widen for each factor that compounds the error, as measured."""
    return base * (2 if hc_mult > 1 else 1) * (2 if npes > 1 else 1)


def _cfg(hc_mult=1):
    return V4Config(
        heads=8,
        hidden=1024,
        q_lora=512,
        head_dim=512,
        rope_dim=64,
        o_groups=2,
        o_lora=128,
        n_experts=128,
        top_k=6,
        inter=128,
        window=128,
        hc_mult=hc_mult,
    )


def _routing_flipped(got, ref, W, cfg, S):
    """Did the two sides pick different experts, and was it only a near-tie?

    Routing is a discrete top-k over scores that differ by a rounding step, so a
    pair straddling the cut can swap. That makes `mid` and everything after it
    incomparable -- a different expert is a different computation, not a bigger
    error -- so the caller re-bases on the kernel's own routing. The flip must BE
    a near-tie: a wide margin means a real selection bug, which is why the margin
    is asserted rather than assumed. S > 1 hits this more often only because it
    has one cut per sample.
    """
    if got["sel"].tolist() == ref["sel"].tolist():
        return False
    for s in range(S):
        a, b = got["sel"][s].tolist(), ref["sel"][s].tolist()
        if set(a) == set(b):
            continue  # same experts in a different slot order: two near-equal
            # scores traded places. `mid` is per slot so the caller still rebases,
            # but there is no membership change to explain and the cut margin is
            # not the gap that moved.
        sc = (got["scores"][s].float() + W.t["bias"].float()).sort(descending=True).values
        margin = (sc[cfg.top_k - 1] - sc[cfg.top_k]).item()
        assert margin < 1e-4, f"sample {s} chose a different expert SET on a {margin:.3e} margin"
    return True


def _rebase_on_own_routing(got, ref, W, moe_mode):
    """Recompute the golden's MoE half from the routing the kernel actually chose.

    `mid` is deliberately NOT passed back in. Handing the kernel its own `mid`
    would put the up/gate stage on both sides of the comparison and stop checking
    it entirely -- which is how a mutation that let masked-off up/gate tiles
    publish went unnoticed. Passing only `sel`/`prob` keeps `mid` under test while
    removing the one thing a near-tie made incomparable.
    """
    return dict(ref, **golden_moe(W, got["a"], lambda z: z, sel=got["sel"], prob=got["prob"], moe_mode=moe_mode))


def _compare_stages(got, ref, cfg, npes):
    """Per-stage relative error against the golden."""
    for name, base in STAGE_TOL.items():
        tol = _tol(base, cfg.hc_mult, npes) if name in SCALES_WITH_CONFIG else base
        a = got[name].float().reshape(-1)
        b = ref[name].float().reshape(-1)
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"


@pytest.mark.parametrize("moe_mode", [MoeMode.A8W4, MoeMode.W8A8])
@pytest.mark.parametrize("hc_mult", [1, 4])
@pytest.mark.parametrize("S", [1, 2, 4])
def test_dsv4_layer_matches_golden(S, moe_mode, hc_mult):
    """hc_mult=1 is a plain residual; 4 is V4's hyper-connection stream.

    S > 1 is a batch of independent sequences at one shared position, so each
    sample carries its own cache slice. S=2 and 4 also put the ug stage on its
    multi-sample path, where tiles are masked rather than skipped."""
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = _cfg(hc_mult)
    cfg.validate()
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=moe_mode)
    layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=moe_mode)

    hshape = (S, cfg.hidden) if cfg.hc_mult == 1 else (S, cfg.hc_mult, cfg.hidden)
    h = (0.5 * torch.randn(*hshape, device=dev)).bfloat16()
    pos = cfg.window  # ring already wrapped once
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)

    kv_kernel = kv0.clone()
    out = layer.forward(h, cur, kv_kernel, dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()

    kv_ref = kv0.clone()
    ref = golden_layer(W, h, [pos] * S, kv_ref, dest, idx, cos, sin, lambda z: z, moe_mode=moe_mode)

    if _routing_flipped(got, ref, W, cfg, S):
        ref = _rebase_on_own_routing(got, ref, W, moe_mode)
    _compare_stages(got, ref, cfg, 1)
    assert out.shape == h.shape, f"the layer must preserve its input shape, got {out.shape}"

    # end to end, judge by relative L2: one FP8/bf16 rounding flip upstream moves a
    # single element a long way, and hc_post mixes hc_mult streams so it propagates
    a_out, b_out = out.float(), ref["x_out"].float()
    rel_max = (a_out - b_out).abs().max().item() / max(b_out.abs().max().item(), 1e-6)
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    out_tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol} (rel_max {rel_max:.5f})"


@pytest.mark.large_shape
@pytest.mark.parametrize("S", [1, 8])
def test_dsv4_layer_matches_golden_at_real_dims(S):
    """The reduced shard above cannot catch mappings that only break at V4's own
    numbers -- 384 experts overflowed the selection key's id field, which 256 (and
    the reduced 128) fit exactly. Keep a real-shard case.

    S=8 is the only case that puts two up/gate tiles on one CTA: V4-Pro's top-6
    over INTER 384 is 288 tiles for 256 CTAs, while the reduced shard's 96 fit in
    one rep. It is also the only one where the down tile is 7168 / 256 = 28 rows
    before rounding, which is the width emit_dn cannot actually cover."""
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = V4Config(hc_mult=1)  # defaults are DeepSeek-V4-Pro at TP8
    cfg.validate()
    dev, mode = "cuda", MoeMode.A8W4
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=mode)

    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.window
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)

    kv_kernel = kv0.clone()
    out = layer.forward(h, cur, kv_kernel, dest, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(W, h, [pos] * S, kv0.clone(), dest, idx, cos, sin, lambda z: z, moe_mode=mode)

    if _routing_flipped(got, ref, W, cfg, S):
        ref = _rebase_on_own_routing(got, ref, W, mode)
    _compare_stages(got, ref, cfg, 1)
    # an id above 255 is the case the 8-bit key field used to corrupt
    assert max(got["sel"].reshape(-1).tolist()[1:]) > 255 or cfg.n_experts <= 256

    a_out, b_out = out.float(), ref["x_out"].float()
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    out_tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol}"


def test_dsv4_csa_shape_is_the_selected_one():
    """What compress_ratio 4 does and does not promise.

    It used to be rejected outright. The compressor is implemented now, and the
    layer gathers whatever index list it is handed at any ratio, so there is
    nothing left for validate_shard to refuse -- but the SELECTION is still the
    caller's until the in-kernel indexer lands. What the shape does promise is
    that it is sized for a selection: the index list carries index_topk
    compressed slots, not the whole compressed half, which is the difference
    between CSA and HCA.
    """
    from kernels.dsv4_moe_layer.config import COMPRESS_CSA, validate_shard

    # refused by default: nobody should get non-V4 semantics by picking a ratio
    with pytest.raises(ValueError, match="lightning indexer"):
        validate_shard(1, 16, 0, 8, compress_ratio=COMPRESS_CSA)
    validate_shard(1, 16, 0, 8, compress_ratio=COMPRESS_CSA, allow_unindexed_csa=True)

    csa = V4Config(hc_mult=1, compress_ratio=COMPRESS_CSA, max_seq=4096)
    assert csa.indexed and csa.overlap and csa.c_coff == 2
    assert csa.n_index == min(csa.index_topk, csa.n_compressed)
    assert csa.n_keys == csa.window + csa.n_index  # already a multiple of KEY_BLOCK

    # a long enough sequence is where the cap actually bites
    far = V4Config(hc_mult=1, compress_ratio=COMPRESS_CSA, max_seq=4096 * 16)
    assert far.n_compressed > far.index_topk
    assert far.n_index == far.index_topk, "the gather must stay bounded by index_topk"
    assert far.cache_rows > far.n_keys, "the cache still holds every compressed entry"


def test_dsv4_rejects_head_dim_that_would_deadlock():
    """A head_dim below the PV MFMA's per-wave grouping produces no work at all, which
    would surface as an unfillable mailbox rather than an error."""
    from kernels.dsv4_moe_layer.dsv4_kernel import build_dsv4_kernel

    with pytest.raises(AssertionError, match="head_dim"):
        build_dsv4_kernel(S=1, heads=8, npes=1, head_dim=128)


# ---------------------------------------------------------------- multi-rank TP
#   python3 tests/kernels/test_dsv4_layer.py --npes 8
# Routing must agree bit-identically across ranks: every rank sums the peer
# partials in rank order, so all ranks see the same post-attention hidden state
# and therefore select the same experts without a second exchange.

TP_SEED = 1234


def _tp_cfg(real: bool, hc_mult: int = 1, compress_ratio: int = 0, max_seq: int | None = None):
    # NOT a module global: mp.spawn re-imports this module in each child, so
    # anything set under __main__ never reaches the workers
    cfg = V4Config(hc_mult=hc_mult) if real else _cfg(hc_mult)
    if compress_ratio:
        # 256 keeps the correctness runs short. It also decides n_compressed and
        # so the whole compressed shape, so a benchmark MUST pass the length it
        # means to report -- measuring one shape and printing another is a way to
        # publish a number for a configuration that never ran.
        cfg.compress_ratio = compress_ratio
        cfg.max_seq = 256 if max_seq is None else max_seq
    return cfg


def run_rank(rank, npes, real=False, iters=2, group=None, moe_mode=MoeMode.A8W4, hc_mult=1, compress_ratio=0):
    import torch.distributed as dist

    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = _tp_cfg(real, hc_mult, compress_ratio)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    layer = Dsv4MoeLayer(W, samples=1, rank=rank, npes=npes, group=group, moe_mode=moe_mode)
    # one table either way; cfg.rope_base picks compress_rope_theta for a
    # compressing layer and rope_theta for a pure sliding-window one
    cos, sin = rope_table(4096, theta=cfg.rope_base, device=dev)
    gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical inputs everywhere
    # Without compression each step is independent, so the cache is re-seeded from
    # kv0 every iteration. The compressor carries state across steps, so its run has
    # to be a real sequential decode: one cache, one rolling state, advancing pos.
    kv0 = torch.randn(cfg.window, cfg.head_dim, generator=gen, device=dev).to(torch.bfloat16)
    pos = cfg.window
    idx, dest = contiguous_pool([pos], cfg, dev)
    if compress_ratio:
        pos = 0
        kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        ks = torch.zeros(1, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
        ss = torch.full((1, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    if npes == 1:
        allreduce = lambda x: x  # noqa: E731
    else:

        def allreduce(x):
            # peer_reduce rounds each rank's partial to bf16 before pushing it over
            # XGMI, and every rank sums the partials in rank order; model both, or
            # the golden compares exact fp32 sums against 8 rounded ones.
            x = x.to(torch.bfloat16).float()
            parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
            dist.all_gather(parts, x.cpu().contiguous(), group=group)
            return sum(parts[1:], parts[0]).to(x.device)

    ok = True
    boundaries = 0
    for it in range(iters):
        hshape = (1, cfg.hidden) if cfg.hc_mult == 1 else (1, cfg.hc_mult, cfg.hidden)
        h = torch.randn(*hshape, generator=gen, device=dev).to(torch.bfloat16)
        if compress_ratio:
            pos = it
            idx, dest = contiguous_pool([pos], cfg, dev)
            boundaries += (pos + 1) % compress_ratio == 0
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        kv_in = kv_k if compress_ratio else kv0.clone()
        out = layer.forward(h, cur, kv_in, dest, idx, cos, sin)
        torch.cuda.synchronize()
        got = layer.intermediates()

        if npes > 1:
            # every rank must produce the SAME hidden state, bit for bit
            peers = [torch.empty_like(out.cpu()) for _ in range(npes)]
            dist.all_gather(peers, out.cpu().contiguous(), group=group)
            for other in peers[1:]:
                torch.testing.assert_close(other, peers[0], atol=0, rtol=0)
            sels = [torch.empty_like(got["sel"].cpu()) for _ in range(npes)]
            dist.all_gather(sels, got["sel"].cpu().contiguous(), group=group)
            for other in sels[1:]:
                torch.testing.assert_close(other, sels[0], atol=0, rtol=0)

        gkw = dict(kv_state=ks, score_state=ss, cos_c=cos, sin_c=sin) if compress_ratio else {}
        ref = golden_layer(
            W,
            h,
            [pos],
            kv_r if compress_ratio else kv0.clone(),
            dest,
            idx,
            cos,
            sin,
            allreduce,
            moe_mode=moe_mode,
            **gkw,
        )
        if compress_ratio and (pos + 1) % compress_ratio == 0:
            slot = cfg.window + pos // compress_ratio
            a_c, b_c = kv_k[slot].float(), kv_r[slot].float()
            c_rel = (a_c - b_c).abs().max().item() / max(b_c.abs().max().item(), 1e-6)
            if c_rel >= 2e-2:
                print(f"rank {rank}: compressed row at {slot} rel {c_rel:.5f}", flush=True)
                ok = False
        # Routing is a discrete top-k over scores derived from `a`, so a 1-ulp
        # difference there can move a near-tied expert. That makes an end-to-end
        # comparison meaningless -- a different expert gives a different `mid`. So
        # judge the expert math against the kernel's OWN routing (as the MLA suite
        # does), and treat a flip as information rather than a failure.
        flipped = got["sel"].tolist() != ref["sel"].tolist()
        for name, base in STAGE_TOL.items():
            tol = _tol(base, cfg.hc_mult, npes) if name in SCALES_WITH_CONFIG else base
            if flipped and name in SCALES_WITH_CONFIG:
                continue
            a = got[name].float().reshape(-1)
            b = ref[name].float().reshape(-1)
            rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
            l2 = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
            frac = ((a - b).abs() > 1e-3 * max(b.abs().max().item(), 1e-6)).float().mean().item()
            if rel >= tol:
                print(
                    f"rank {rank}: stage {name} rel_max {rel:.5f} rel_l2 {l2:.5f} " f"elems_off {frac * 100:.2f}%",
                    flush=True,
                )
                ok = False
        if flipped:
            print(f"rank {rank}: routing flipped on a near-tie (expected; judging by own routing)", flush=True)
        down = golden_moe(
            W,
            got["a"],
            allreduce,
            mid=got["mid"],
            sel=got["sel"],
            prob=got["prob"],
            moe_mode=moe_mode,
        )
        d_out = down["x_out"].float()
        d_rel = ((out.float() - d_out).norm() / d_out.norm()).item()
        out_tol = _tol(OUT_REL_L2, cfg.hc_mult, npes)
        if d_rel >= out_tol:
            print(f"rank {rank}: x_out vs own-routing golden rel_l2 {d_rel:.5f}", flush=True)
            ok = False
        a_out, b_out = out.float(), ref["x_out"].float()
        rel_max = (a_out - b_out).abs().max().item() / max(b_out.abs().max().item(), 1e-6)
        rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
        if rank == 0:  # pytest captures this; it is what makes a near-miss legible
            print(f"rank {rank}: x_out rel_max {rel_max:.5f}  rel_l2 {rel_l2:.5f}", flush=True)
        if rel_l2 >= out_tol and not flipped:  # end to end only when routing agrees
            ok = False
    layer.close()
    if compress_ratio and boundaries < 2:
        print(f"rank {rank}: only crossed {boundaries} compression boundaries", flush=True)
        ok = False
    return ok


def _free_port():
    """A port the parent has just closed, so the children can bind it.

    A fixed port makes back-to-back TP runs collide on a socket still in
    TIME_WAIT, which fails the rendezvous and looks exactly like a kernel
    regression -- it cost a debugging round already.
    """
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, npes, real, iters, hc_mult, compress_ratio, port, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        results[rank] = run_rank(rank, npes, real=real, iters=iters, hc_mult=hc_mult, compress_ratio=compress_ratio)
    finally:
        dist.destroy_process_group()


def run_tp(npes, real=False, iters=2, hc_mult=1, compress_ratio=0):
    if npes == 1:
        return run_rank(0, 1, real=real, iters=iters, hc_mult=hc_mult, compress_ratio=compress_ratio)
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(
        _worker,
        args=(npes, real, iters, hc_mult, compress_ratio, _free_port(), results),
        nprocs=npes,
    )
    return all(results[r] for r in range(npes))


@pytest.mark.multi_gpu
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_dsv4_layer_tp8(hc_mult):
    """Both residual widths: the tolerances are calibrated per configuration, so
    hyper-connections at TP8 no longer need pinning out."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    assert run_tp(8, hc_mult=hc_mult)


@pytest.mark.multi_gpu
def test_dsv4_hca_layer_tp8():
    """HCA across ranks: a real sequential decode over two compression boundaries.

    The compressor is replicated (wkv is not TP-sharded), so every rank has to
    produce the same compressed row from the same rolling state -- and the ranks
    still have to agree bit-for-bit on routing with those rows in the gather.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    assert run_tp(8, iters=20, compress_ratio=8)


# ------------------------------------------------------------------ benchmark
#   python3 tests/kernels/test_dsv4_layer.py --bench --npes 8 --real
# HIP-graph replay of LAYERS launches per step, so the number is steady-state
# decode latency of one layer, not launch overhead.

BENCH_LAYERS = 16


def bench_rank(
    rank,
    npes,
    real=True,
    iters=320,
    group=None,
    timeline=False,
    moe_mode=MoeMode.A8W4,
    hc_mult=1,
    compress_ratio=0,
    max_seq=None,
    samples=1,
):
    """Returns (us per layer, the config it was measured on).

    ``compress_ratio`` picks the attention variant: 0 sliding-window, 128 HCA,
    4 CSA. The cost is position-independent by construction -- the gather walks a
    compile-time ``n_keys`` and the indexer scores the whole compressed cache with
    the unwritten entries masked -- so one fixed position measures the steady
    state.
    """
    import torch.distributed as dist

    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = _tp_cfg(real, hc_mult, compress_ratio, max_seq)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    cos, sin = rope_table(max(4096, cfg.max_seq), theta=cfg.rope_base, device=dev)
    kv = torch.randn(samples * cfg.cache_rows, cfg.head_dim, device=dev).to(torch.bfloat16)
    # deep enough that the compressed half of the cache is full, which is the
    # steady state a decode spends nearly all of its time in
    pos = (cfg.max_seq - 1) if compress_ratio else cfg.window
    idx, dest = contiguous_pool([pos] * samples, cfg, dev)
    cur = torch.tensor([pos] * samples, dtype=torch.int32, device=dev)
    op = Dsv4MoeLayer(W, samples, rank=rank, npes=npes, group=group, moe_mode=moe_mode, allow_unindexed_csa=True)
    hshape = (samples, cfg.hidden) if cfg.hc_mult == 1 else (samples, cfg.hc_mult, cfg.hidden)
    h = torch.randn(*hshape, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    for _ in range(10):
        op.forward(h, cur, kv, dest, idx, cos, sin, x_out=x)
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()

    if timeline:
        top = Dsv4MoeLayer(
            W,
            samples,
            rank=rank,
            npes=npes,
            group=group,
            timeline=True,
            moe_mode=moe_mode,
            allow_unindexed_csa=True,
        )
        for _ in range(3):
            top.forward(h, cur, kv, dest, idx, cos, sin, x_out=x)
        torch.cuda.synchronize()
        if rank == 0:
            print(top.timeline_report(), flush=True)
        top.close()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(BENCH_LAYERS):
            op.forward(h, cur, kv, dest, idx, cos, sin, x_out=x, layer=layer, advance=False)
        op.advance_step()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()
    t0, t1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters // BENCH_LAYERS):
        graph.replay()
    t1.record()
    torch.cuda.synchronize()
    us = t0.elapsed_time(t1) * 1e3 / (iters // BENCH_LAYERS * BENCH_LAYERS)
    op.close()
    # the shape travels with the number, so a report cannot describe a run that
    # did not happen
    return us, dict(n_keys=cfg.n_keys, n_comp=cfg.n_compressed, max_seq=cfg.max_seq, samples=samples)


def _bench_worker(rank, npes, real, timeline, moe_mode, hc_mult, compress_ratio, max_seq, samples, port, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        results[rank] = bench_rank(
            rank,
            npes,
            real=real,
            timeline=timeline,
            moe_mode=moe_mode,
            hc_mult=hc_mult,
            compress_ratio=compress_ratio,
            max_seq=max_seq,
            samples=samples,
        )
    finally:
        dist.destroy_process_group()


def run_bench(
    npes, real=True, timeline=False, moe_mode=MoeMode.A8W4, hc_mult=1, compress_ratio=0, max_seq=None, samples=1
):
    kw = dict(
        real=real,
        timeline=timeline,
        moe_mode=moe_mode,
        hc_mult=hc_mult,
        compress_ratio=compress_ratio,
        max_seq=max_seq,
        samples=samples,
    )
    if npes == 1:
        return {0: bench_rank(0, 1, **kw)}
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(
        _bench_worker,
        args=(npes, real, timeline, moe_mode, hc_mult, compress_ratio, max_seq, samples, _free_port(), results),
        nprocs=npes,
    )
    return dict(results)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=8)
    ap.add_argument("--real", action="store_true", help="use the real V4-Pro TP8 shard")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--moe-mode", default=MoeMode.A8W4.value, choices=tuple(m.value for m in MoeMode))
    ap.add_argument("--hc-mult", type=int, default=1, help="1 = plain residual, 4 = hyper-connections")
    a = ap.parse_args()
    if a.bench:
        res = run_bench(a.npes, a.real, a.timeline, MoeMode(a.moe_mode), a.hc_mult)
        us = [res[r][0] for r in sorted(res)]
        tag = "real V4-Pro" if a.real else "reduced"
        print(
            f"{tag} shard, {a.moe_mode}, hc={a.hc_mult}, npes={a.npes}: {max(us):7.1f} us/layer  (per rank: "
            + " ".join(f"{v:.1f}" for v in us)
            + ")"
        )
    else:
        print("PASS" if run_tp(a.npes, a.real, a.iters, a.hc_mult) else "FAIL")


@pytest.mark.parametrize("moe_mode", [MoeMode.W8A8, MoeMode.A8W4])
def test_dsv4_hca_layer_matches_golden(moe_mode):
    """The HCA layer end to end: rolling compressor state, compressed entries
    gathered alongside the window, over several compression boundaries.

    A short compress_ratio stands in for V4's 128 -- DeepSeek ties the overlapping
    compressor to ratio 4 specifically, so any other value is the same code path
    and this crosses boundaries in far fewer steps.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    ratio = 16
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio = ratio
    cfg.max_seq = 512
    cfg.validate()
    dev = "cuda"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=moe_mode)
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=moe_mode)

    # ONE table for the whole layer: V4 picks the rope base per LAYER, not per
    # consumer, so a compressing layer rotates its window q/kv on the same table
    # as its compressed rows. cfg.rope_base holds that rule.
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(1, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    # -inf like the layer's own state: unwritten rows must drop out of the softmax
    ss = torch.full((1, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    boundaries = 0
    for pos in range(3 * ratio + 2):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        out = layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()
        ref = golden_layer(
            W,
            h,
            [pos],
            kv_r,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=moe_mode,
            kv_state=ks,
            score_state=ss,
            cos_c=cos,
            sin_c=sin,
        )
        if (pos + 1) % ratio == 0:
            boundaries += 1
            slot = cfg.window + pos // ratio
            # The compressed row is the thing under test. Not bit-exact: the kernel
            # pools with an online softmax on the hardware exp2 while the golden uses
            # a batch softmax, and the result is rounded to bf16, so they can land a
            # few ulp apart. A few percent means the pooling itself is right.
            a_c, b_c = kv_k[slot].float(), kv_r[slot].float()
            rel = (a_c - b_c).abs().max().item() / max(b_c.abs().max().item(), 1e-6)
            assert rel < 2e-2, f"compressed row at slot {slot} differs by rel {rel:.5f}"

        # `a` below is the kernel's own, so attention cancels out of that check --
        # compare it here, where a mis-sized gather over the compressed half shows.
        got = layer.intermediates()
        rel_o = (got["o"].float() - ref["o"].float()).abs().max().item() / max(
            ref["o"].float().abs().max().item(), 1e-6
        )
        assert rel_o < STAGE_TOL["o"], f"pos={pos} stage o diverged: rel {rel_o:.5f}"

        # Over a long sequential run a near-tie will eventually flip an expert, and
        # with 128 experts and top-6 the 6th/7th margin is crowded enough that
        # skipping near-ties would skip most positions. So judge the layer against a
        # golden rebuilt from the kernel's OWN routing, which is flip-immune by
        # construction -- the same structure the multi-rank test uses.
        own = golden_moe(
            W,
            got["a"],
            lambda z: z,
            mid=got["mid"],
            sel=got["sel"],
            prob=got["prob"],
            moe_mode=moe_mode,
        )
        b_out = own["x_out"].float()
        rel_l2 = ((out.float() - b_out).norm() / b_out.norm()).item()
        tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
        assert rel_l2 < tol, f"pos={pos} x_out rel_l2 {rel_l2:.5f} >= {tol}"

    assert boundaries >= 3, "must cross several compression boundaries"


@pytest.mark.large_shape
def test_dsv4_hca_layer_at_real_dims():
    """HCA at V4-Pro's own numbers, which the reduced shard above cannot reach.

    The reduced case stands ratio 16 in for 128, and that is exactly the kind of
    substitution that hid the selection key's 8-bit id field in P1. Here the
    pooling loop really runs 128 rows, ``ape`` is the full 128 x 512 table, the
    compressed half of the cache is addressed at its real offset, and the split
    stage walks the real ``n_keys`` (192, not the window's 128) -- the stride that
    was wrong when the compressed entries were silently never gathered.

    One boundary is enough to exercise all of that, and 128 steps per boundary is
    what makes more expensive.
    """
    from kernels.dsv4_moe_layer.config import COMPRESS_HCA
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = V4Config(hc_mult=1)  # defaults are DeepSeek-V4-Pro at TP8
    cfg.compress_ratio = COMPRESS_HCA
    cfg.validate()
    ratio, dev, mode = cfg.compress_ratio, "cuda", MoeMode.W8A8
    assert cfg.n_keys > cfg.window, "the gather must reach past the window"
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode)

    # one table for the whole layer, at this layer's base
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(1, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((1, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    # Two boundaries, not one: the first anchors at position 0, where every rope
    # table is the identity, so it cannot tell the compressor's base from the
    # window's. The second anchors at `ratio` and does.
    boundaries, checked, top_id = 0, 0, 0
    for pos in range(2 * ratio + 4):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        out = layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()
        ref = golden_layer(
            W,
            h,
            [pos],
            kv_r,
            dest,
            idx,
            cos,
            sin,
            lambda z: z,
            moe_mode=mode,
            kv_state=ks,
            score_state=ss,
            cos_c=cos,
            sin_c=sin,
        )
        if (pos + 1) % ratio == 0:
            boundaries += 1
            slot = cfg.window + pos // ratio
            a_c, b_c = kv_k[slot].float(), kv_r[slot].float()
            rel = (a_c - b_c).abs().max().item() / max(b_c.abs().max().item(), 1e-6)
            assert rel < 2e-2, f"compressed row at slot {slot} differs by rel {rel:.5f}"

        got = layer.intermediates()
        # Stage-by-stage only where it is informative: the steps around the
        # boundary, which is the only place this differs from the plain window.
        if any(abs(pos - (b * ratio - 1)) <= 2 for b in (1, 2)):
            checked += 1
            # `mid` is the one stage that depends on which experts were picked, and
            # over a run this long a near-tie eventually flips one. Every other
            # stage -- the whole attention half, which is what HCA changes -- is
            # compared unconditionally.
            same_route = got["sel"].tolist() == ref["sel"].tolist()
            for name, base in STAGE_TOL.items():
                if name == "mid" and not same_route:
                    continue
                tol = _tol(base, cfg.hc_mult, 1) if name in SCALES_WITH_CONFIG else base
                a = got[name].float().reshape(-1)
                b = ref[name].float().reshape(-1)
                rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
                assert rel < tol, f"pos={pos} stage {name} diverged: rel {rel:.5f} >= {tol}"
        top_id = max(top_id, *got["sel"].reshape(-1).tolist()[1:])

        # flip-immune: judge the layer against a golden rebuilt from the kernel's
        # own routing, as the long reduced run and the multi-rank test do
        own = golden_moe(
            W,
            got["a"],
            lambda z: z,
            mid=got["mid"],
            sel=got["sel"],
            prob=got["prob"],
            moe_mode=mode,
        )
        b_out = own["x_out"].float()
        rel_l2 = ((out.float() - b_out).norm() / b_out.norm()).item()
        tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
        assert rel_l2 < tol, f"pos={pos} x_out rel_l2 {rel_l2:.5f} >= {tol}"

    assert boundaries == 2 and checked == 10
    # an id above 255 is the case the selection key's 8-bit field used to corrupt;
    # over this many steps the routing is certain to reach one
    assert top_id > 255 or cfg.n_experts <= 256


def test_dsv4_csa_compressor_in_kernel():
    """CSA's OVERLAPPING compressor, running in the kernel.

    A whole CSA layer still needs the indexer, so attention here gathers a plain
    prefix and only the compressed row is under test -- compared against
    ``compress_step`` directly, the way the reference suite isolates it against
    DeepSeek's module. What that covers is everything overlap changes: the
    c_coff-wide projections out of the fused GEMV, two windows of state, an entry
    pooled from the previous window's first half and the current window's second,
    and the shift that retires a window.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer
    from kernels.dsv4_moe_layer.reference import compress_step, dequant, qkv_a_split

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    assert cfg.overlap and cfg.c_coff == 2, "ratio 4 is the overlapping form"
    dev, mode = "cuda", MoeMode.W8A8
    W, t = None, None
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)
    cache_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    dq = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    cut = qkv_a_split(cfg)

    emitted = 0
    for pos in range(5 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq.float().T
        ref = compress_step(
            proj[0, slice(*cut["c_kv"])],
            proj[0, slice(*cut["c_gate"])],
            pos,
            cfg,
            t,
            ks,
            ss,
            cache_r,
            cos,
            sin,
            dest_row=cfg.window + pos // ratio,
        )
        if (pos + 1) % ratio:
            assert ref is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        slot = cfg.window + pos // ratio
        a, b = kv_k[slot].float(), cache_r[slot].float()
        # Most boundaries come out bit-exact. The kernel pools with an online
        # softmax on the hardware exp2 and the golden with a batch softmax, so the
        # pre-quantization value can differ in the last fp32 bits, and once in a
        # while that pushes ONE element across an FP8 code boundary -- worth ~9% of
        # that element. A max-abs bar cannot tell that from a broken pooling, so
        # bound both: the bulk must agree tightly, and only a couple of elements
        # may move at all. A wrong overlap moves most of the row.
        n_diff = int((a != b).sum())
        rel_l2 = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
        assert n_diff <= 4, f"pos={pos}: {n_diff} elements differ, not a quantization tie"
        assert rel_l2 < 5e-3, f"pos={pos} compressed row rel_l2 {rel_l2:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


def test_dsv4_indexer_compressor_in_kernel():
    """The INDEXER's compressor in the kernel, against ``compress_step(rotate=True)``.

    Same pooling as the attention compressor, different tail: half the head_dim,
    a Hadamard rotation over the whole row, then FP4 instead of FP8 over the nope
    part. What this covers that the attention one does not is the rotation and the
    FP4 block scale -- and those are the parts with no prior art in this kernel,
    so they get compared against the golden that was itself checked bit-exact
    against DeepSeek's module.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer
    from kernels.dsv4_moe_layer.reference import compress_step, dequant, qkv_a_split

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    assert cfg.indexed, "only CSA runs an indexer"
    ihd, dev, mode = cfg.index_head_dim, "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=dev)
    i_ss = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=dev)
    i_ref = torch.zeros(cfg.n_compressed, ihd, dtype=torch.bfloat16, device=dev)
    dq = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    cut = qkv_a_split(cfg)

    emitted = 0
    for pos in range(5 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq.float().T
        ref = compress_step(
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_ks,
            i_ss,
            i_ref,
            cos,
            sin,
            head_dim=ihd,
            ape=t["i_ape"],
            gamma=t["g_ickv"],
            rotate=True,
        )
        if (pos + 1) % ratio:
            assert ref is None, f"pos={pos} should emit nothing"
            continue
        emitted += 1
        slot = pos // ratio
        a, b = layer.i_cache[0, slot].float(), i_ref[slot].float()
        # FP4's levels are coarse enough that a last-bit difference upstream moves
        # one element a whole step; bound the bulk and the count instead of the max,
        # as the attention compressor's test does.
        n_diff = int((a != b).sum())
        rel_l2 = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
        assert n_diff <= 4, f"pos={pos}: {n_diff}/{ihd} elements differ, not a quantization tie"
        assert rel_l2 < 1e-2, f"pos={pos} indexer compressed row rel_l2 {rel_l2:.5f}"

    assert emitted >= 4, f"expected several compressed entries, got {emitted}"


def test_dsv4_indexer_query_in_kernel():
    """The indexer's QUERY path in the kernel: projection, RoPE, Hadamard, FP4.

    Unlike the main query there is no per-head RMS -- the indexer's query is
    rotated and quantized, not normalised. The golden side is spelled out here
    rather than calling indexer_step, because that also runs the compressor and
    the scoring; what is under test is only the query.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer
    from kernels.dsv4_moe_layer.reference import (
        dequant,
        hadamard,
        quant_dequant_fp4,
        rope,
    )

    torch.manual_seed(0)
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = COMPRESS_CSA, 256
    ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    dq_qkv = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    dq_iqb = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)

    for pos in range(3):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        q_a = (x @ dq_qkv.float().T)[:, : cfg.q_lora]
        q_an = bf(rmsnorm(q_a, t["g_q"], cfg.eps))
        q = (q_an @ dq_iqb.T).view(ih, ihd)
        q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
        ref = quant_dequant_fp4(bf(hadamard(bf(q))))

        got = layer.debug("i_q", (1, ih, ihd))[0]
        # FP4's levels are coarse -- one step is ~0.5 at these magnitudes -- so a
        # last-bit difference upstream moves an element a whole step, and a handful
        # of those already costs a few percent in l2. The RATE is the discriminating
        # signal: a broken rotation or a missing quantization moves most of the
        # vector, not six elements in a thousand.
        n_diff = int((got != ref).sum())
        rel_l2 = ((got - ref).norm() / max(ref.norm().item(), 1e-6)).item()
        assert n_diff <= ih * ihd // 50, f"pos={pos}: {n_diff}/{ih * ihd} elements differ"
        assert rel_l2 < 6e-2, f"pos={pos} indexer query rel_l2 {rel_l2:.5f}"


def test_dsv4_indexer_scoring_in_kernel():
    """The indexer's SCORE for every compressed entry, against the golden.

    score[c] = sum_h relu(q[h] . k[c]) * w[h], over entries the compressor has
    actually written; the rest score NEG so a top-k can never pick them. Single
    rank, so the cross-rank sum the real scale anticipates is the identity here.

    The golden side is spelled out rather than calling indexer_step, because that
    returns the selection and what is under test is the score behind it.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer
    from kernels.dsv4_moe_layer.reference import (
        compress_step,
        dequant,
        hadamard,
        qkv_a_split,
        quant_dequant_fp4,
        rope,
    )

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=dev)
    i_ss = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=dev)
    i_ref = torch.zeros(cfg.n_compressed, ihd, dtype=torch.bfloat16, device=dev)
    dq_qkv = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    dq_iqb = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)
    cut = qkv_a_split(cfg)
    scale = ihd**-0.5 * cfg.index_heads_total**-0.5

    scored = 0
    for pos in range(4 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq_qkv.float().T
        compress_step(
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_ks,
            i_ss,
            i_ref,
            cos,
            sin,
            head_dim=ihd,
            ape=t["i_ape"],
            gamma=t["g_ickv"],
            rotate=True,
        )
        q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
        q = (q_an @ dq_iqb.T).view(ih, ihd)
        q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
        q = quant_dequant_fp4(bf(hadamard(bf(q))))
        w = bf(x @ t["i_w"].float().T)[0] * scale

        n = (pos + 1) // ratio
        got = layer.debug("i_score", (1, cfg.n_compressed))[0]
        if not n:
            assert bool((got < 0).all()), f"pos={pos}: nothing compressed, nothing scorable"
            continue
        scored += 1
        ref = (torch.einsum("hd,td->ht", q, i_ref[:n].float()).relu() * w.view(ih, 1)).sum(0)
        a, b = got[:n], ref
        rel = ((a - b).norm() / max(b.norm().item(), 1e-6)).item()
        assert rel < 2e-2, f"pos={pos} score rel_l2 {rel:.5f}"
        # entries the compressor has not written must be unpickable
        assert bool((got[n:] < 0).all()), f"pos={pos}: unwritten entries are scorable"

    assert scored >= 3, f"expected several scored steps, got {scored}"


def _indexer_score_rank(rank, npes, port, results):
    """One rank of the indexer's score all-reduce; see the test below."""
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer
        from kernels.dsv4_moe_layer.reference import (
            dequant,
            hadamard,
            quant_dequant_fp4,
            rope,
        )

        dev = torch.device("cuda", rank)
        torch.cuda.set_device(dev)
        ratio = COMPRESS_CSA
        cfg = _cfg(hc_mult=1)
        cfg.compress_ratio, cfg.max_seq = ratio, 256
        ih, ihd, rd = cfg.index_heads, cfg.index_head_dim, cfg.rope_dim
        mode = MoeMode.W8A8
        W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=mode)
        t = W.t
        layer = Dsv4MoeLayer(W, samples=1, rank=rank, npes=npes, moe_mode=mode, allow_unindexed_csa=True)
        cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
        kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        dq_qkv = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
        dq_iqb = dequant(t["w_i_q_b"], t["s_i_q_b"], 128)
        scale = ihd**-0.5 * cfg.index_heads_total**-0.5
        gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical everywhere

        ok, scored = True, 0
        for pos in range(3 * ratio):
            h = torch.randn(1, cfg.hidden, generator=gen, device=dev).to(torch.bfloat16)
            cur = torch.tensor([pos], dtype=torch.int32, device=dev)
            idx, dest = contiguous_pool([pos], cfg, dev)
            layer.forward(h, cur, kv_k, dest, idx, cos, sin)
            torch.cuda.synchronize()
            got = layer.debug("i_score", (1, cfg.n_compressed))[0]

            # every rank must see the SAME total, bit for bit, or the top-k below
            # it would pick different keys on different ranks
            peers = [torch.empty_like(got.cpu()) for _ in range(npes)]
            dist.all_gather(peers, got.cpu().contiguous())
            for other in peers[1:]:
                torch.testing.assert_close(other, peers[0], atol=0, rtol=0)

            # This rank's own PARTIAL, summed over ranks the way the kernel does.
            # Scored against the KERNEL's compressed cache, not a golden one: the
            # compressor's FP4 ties move an element by a whole level now and then,
            # which the dot product amplifies and eight ranks then accumulate. That
            # is the compressor's business and it has its own test; what is under
            # test here is the exchange.
            x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
            proj = x @ dq_qkv.float().T
            q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
            q = (q_an @ dq_iqb.T).view(ih, ihd)
            q = torch.stack([torch.cat([q[j, :-rd], bf(rope(q[j, -rd:], cos[pos], sin[pos]))]) for j in range(ih)])
            q_ref = quant_dequant_fp4(bf(hadamard(bf(q))))
            w_ref = bf(x @ t["i_w"].float().T)[0] * scale
            # Both inputs come back from the kernel, and each is checked here on its
            # own bar. They have to: weights_proj rounds to bf16 and the query to
            # FP4, and on either the two sides occasionally land a step apart, which
            # the head-sum's cancellation turns into percents of the score. Feeding
            # the kernel's own q and w leaves the dot product, the ReLU, the
            # head-sum and the EXCHANGE as the only things this can fail on -- which
            # is what this test is for.
            q = layer.debug("i_q", (1, ih, ihd))[0]
            w = layer.debug("i_wp", (1, ih))[0]
            # a proportional bar, not a fixed count: FP4 ties scale with how many
            # elements there are, and a handful in 1024 is the expected rate
            dq_n = int((q != q_ref).sum())
            dq_l2 = ((q - q_ref).norm() / max(q_ref.norm().item(), 1e-6)).item()
            dw = (w - w_ref).abs().max().item() / max(w_ref.abs().max().item(), 1e-6)
            if dq_n > ih * ihd // 50 or dq_l2 >= 6e-2 or dw >= 1e-2:
                print(
                    f"rank {rank} pos={pos} q differs on {dq_n} (l2 {dq_l2:.5f}), " f"w rel {dw:.5f}",
                    flush=True,
                )
                ok = False
            n = (pos + 1) // ratio
            if not n:
                continue
            scored += 1
            kcache = layer.i_cache[0, :n].float()
            part = (torch.einsum("hd,td->ht", q, kcache).relu() * w.view(ih, 1)).sum(0)
            parts = [torch.empty_like(part.cpu()) for _ in range(npes)]
            dist.all_gather(parts, part.cpu().contiguous())
            ref = sum(parts[1:], parts[0]).to(dev)
            rel = ((got[:n] - ref).norm() / max(ref.norm().item(), 1e-6)).item()
            if rel >= 1e-3:
                print(f"rank {rank} pos={pos} score rel_l2 {rel:.5f}", flush=True)
                ok = False
        layer.close()
        results[rank] = ok and scored >= 2
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_dsv4_indexer_score_allreduce_tp8():
    """The indexer's score is a PARTIAL sum on every rank.

    Each rank holds only its shard of the 64 index heads, so the score has to be
    summed across ranks before anything ranks the entries -- and every rank has
    to land on the same total bit for bit, or they would select different keys
    and silently attend to different things. Both halves are checked: agreement
    across ranks, and agreement with the summed golden.
    """
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_indexer_score_rank, args=(8, _free_port(), results), nprocs=8)
    assert all(results[r] for r in range(8))


def test_dsv4_indexer_topk_in_kernel():
    """Which compressed entries the kernel's indexer selects, against the golden.

    The SET is what matters: attention sums over the gathered keys, so their order
    in the index list changes nothing. The selection has to be exact rather than
    approximate, because every rank runs it on bit-identical scores and they must
    land on the same set without a further exchange.

    index_topk is reduced so the top-k actually discards -- at V4-Pro's 1024 it
    would keep every entry these runs produce and the selection would be untested.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer
    from kernels.dsv4_moe_layer.reference import indexer_step

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 256, 4
    ihd, dev, mode = cfg.index_head_dim, "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    t = W.t
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)

    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    i_ks = torch.zeros(cfg.c_rows, cfg.c_coff * ihd, device=dev)
    i_ss = torch.full((cfg.c_rows, cfg.c_coff * ihd), float("-inf"), device=dev)
    i_ref = torch.zeros(cfg.n_compressed, ihd, dtype=torch.bfloat16, device=dev)
    dq_qkv = dequant(t["w_qkv_a"], t["s_qkv_a"], 128)
    cut = qkv_a_split(cfg)

    chose, discarded = 0, 0
    for pos in range(8 * ratio):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        torch.cuda.synchronize()

        x = bf(rmsnorm(h.float(), t["g_in"], cfg.eps))
        proj = x @ dq_qkv.float().T
        q_an = bf(rmsnorm(proj[:, : cfg.q_lora], t["g_q"], cfg.eps))
        ref = indexer_step(
            x[0],
            q_an[0],
            proj[0, slice(*cut["i_kv"])],
            proj[0, slice(*cut["i_gate"])],
            pos,
            cfg,
            t,
            i_ks,
            i_ss,
            i_ref,
            cos,
            sin,
            lambda z: z,
        )
        got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]

        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        assert int((got >= 0).sum()) == k, f"pos={pos}: picked {int((got >= 0).sum())}, want {k}"
        if not k:
            continue
        chose += 1
        discarded += n > cfg.index_topk
        # The golden scores through its own compressor, whose FP4 ties can move a
        # borderline entry, so judge the SELECTION against the kernel's own scores:
        # that is what the top-k is, and the score has its own test.
        sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
        want = set((cfg.window + sc.topk(k).indices).tolist())
        a = set(got[got >= 0].tolist())
        margin = (sc.sort(descending=True).values[k - 1] - sc.sort(descending=True).values[k]).item() if n > k else 1.0
        if a != want and margin > 1e-6:
            raise AssertionError(f"pos={pos} picked {sorted(a)} vs {sorted(want)} (margin {margin:.3e})")
        # and the golden's own pick should agree except where its scores tie-break
        assert len(set(ref[ref >= 0].tolist())) == k

    assert chose >= 6 and discarded >= 3, f"chose {chose}, discarded on {discarded}"


def test_dsv4_indexer_topk_compaction_spans_waves():
    """The compaction writes one slot per pick when the picks span every wave.

    test_dsv4_indexer_topk_in_kernel judges the selection, but at its shape
    (index_topk 4, 64 candidates) every pick lands in wave 0 and at most four
    slots are written, so the block scan that assigns those slots is barely
    used: a scan that drops its cross-wave term, or runs the wrong way round,
    still produces four distinct slots and the same set. Here 512 candidates
    fill all eight waves and ~200 are kept, so a scan that miscounts collides --
    two picks on one slot, which shows up as a short count or a duplicate.

    The picks are judged against the kernel's OWN scores: what is under test is
    the compaction, and the scoring has its own test. No golden is stepped, so
    this can afford the many positions it takes to fill the candidate space.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 2048, 200
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)
    cos, sin = rope_table(4096, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    # positions to judge at: deep enough that the live candidates cover several
    # waves, and that the last two are past index_topk so the pick also discards
    checks = [400, 700, 1100]
    waves_hit = 0
    for pos in range(checks[-1] + 1):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
        sel = got[got >= 0].tolist()
        assert len(sel) == k, f"pos={pos}: wrote {len(sel)} slots, want {k}"
        assert len(set(sel)) == k, f"pos={pos}: {k - len(set(sel))} picks collided on a slot"
        sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
        srt = sc.sort(descending=True).values
        margin = (srt[k - 1] - srt[k]).item() if n > k else 1.0
        want = set((cfg.window + sc.topk(k).indices).tolist())
        if set(sel) != want and margin > 1e-6:
            raise AssertionError(f"pos={pos}: picked {len(set(sel) - want)} entries the scores do not rank")
        waves_hit = max(waves_hit, (max(s - cfg.window for s in sel) // 64) + 1)
    # the point of the shape: without this the cross-wave term is never read
    assert waves_hit >= 5, f"picks only reached wave {waves_hit}, so the scan is still untested"
    layer.close()


def test_dsv4_indexer_topk_spans_many_candidates_per_thread():
    """The top-k holds when a thread owns several candidates, not one.

    Every other indexer test runs at a shape where the candidate count fits the
    block, so each thread owns exactly one and the radix's per-thread scan never
    iterates. That hides the whole strided walk: a loop that ran once, or an
    index that folded back onto j = 0, would still pass them. Here 8192 / 4
    compressed entries give four candidates per thread and the live set reaches
    three of those rounds, so the later rounds carry picks of their own.

    Judged against the kernel's OWN scores, like the other compaction tests --
    what is under test is the selection, and the scoring has its own test.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    ratio = COMPRESS_CSA
    cfg = _cfg(hc_mult=1)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 8192, 300
    dev, mode = "cuda", MoeMode.W8A8
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MoeLayer(W, samples=1, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)
    cos, sin = rope_table(8192, theta=cfg.rope_base, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)

    last = 5999  # 1500 live candidates: rounds 0, 1 and part of 2
    checks = [2500, 4000, last]
    reach = 0
    for pos in range(last + 1):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx, dest = contiguous_pool([pos], cfg, dev)
        layer.forward(h, cur, kv_k, dest, idx, cos, sin)
        if pos not in checks:
            continue
        torch.cuda.synchronize()
        n = (pos + 1) // ratio
        k = min(cfg.index_topk, n)
        got = layer.debug("i_sel", (1, cfg.n_keys - cfg.window), torch.int32)[0]
        sel = got[got >= 0].tolist()
        assert len(sel) == k, f"pos={pos}: wrote {len(sel)} slots, want {k}"
        assert len(set(sel)) == k, f"pos={pos}: {k - len(set(sel))} picks collided on a slot"
        sc = layer.debug("i_score", (1, cfg.n_compressed))[0][:n]
        srt = sc.sort(descending=True).values
        margin = (srt[k - 1] - srt[k]).item() if n > k else 1.0
        want = set((cfg.window + sc.topk(k).indices).tolist())
        if set(sel) != want and margin > 1e-6:
            raise AssertionError(f"pos={pos}: picked {len(set(sel) - want)} entries the scores do not rank")
        reach = max(reach, max(s - cfg.window for s in sel))
    # the point of the shape: without this only the first round is ever read
    assert reach >= 2 * 512, f"picks stopped at candidate {reach}, so the later rounds are untested"
    layer.close()


@pytest.mark.parametrize("ratio", [0, COMPRESS_CSA])
def test_dsv4_batching_is_independent_sequences(ratio):
    """The kernel at S=2 must equal the kernel run twice at S=1.

    This is what the batch axis means, and it is the test that catches a stage
    reading sample 0's state -- or sample 0's POSITION -- for every sample. `i_wp`
    really did the former, by building its address from the Python-level sample
    index (always 0) instead of its task's. Comparing per-stage against the golden
    does not catch that class on its own: at S=1 the two are the same thing, and
    with every sample at the same position a wrong position is invisible too.

    Exact equality, not a tolerance: the samples are independent by construction,
    so batching changes which lanes do the work but not the arithmetic any one
    sample sees. CSA steps far enough to cross compression boundaries, where each
    sequence's rolling compressor state is what keeps them apart.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 2
    cfg = _cfg(hc_mult=4)
    cfg.compress_ratio, cfg.max_seq = ratio, 256
    if ratio:
        cfg.index_topk = 4
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    # Deep enough that the indexer's top-k actually DISCARDS. At 3 * ratio the last
    # step has 3 live compressed entries against index_topk 4, so every candidate is
    # kept whatever it scores -- and a scoring input taken from the wrong sample
    # then changes nothing observable. Asserted below so it cannot drift back.
    steps = 8 * ratio if ratio else 3
    if ratio:
        assert (steps - 1) // ratio > cfg.index_topk, "the top-k must discard, or the scores are untested"

    hs = [(0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=dev)).bfloat16() for _ in range(steps)]
    # STAGGERED starts: sample s begins at OFFSETS[s], as sequences admitted at
    # different times are. Equal positions would let a stage read sample 0's
    # position for every sample and still agree -- the same shape of defect the
    # indexer's score weights actually had. The offsets are not multiples of the
    # ratio, so the samples also cross their compression boundaries on different
    # steps, which is what makes the per-sample boundary test meaningful.
    OFFSETS = (0, 3)
    assert len(OFFSETS) == S and (not ratio or OFFSETS[1] % ratio), "offsets must stagger the boundary"

    def run(rows):
        """Decode `steps` steps for the given sample rows; per-stage results."""
        n = len(rows)
        layer = Dsv4MoeLayer(W, samples=n, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)
        kv = torch.zeros(n * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        seq = []
        for step in range(steps):
            ps = [OFFSETS[r] + step for r in rows]
            cur = torch.tensor(ps, dtype=torch.int32, device=dev)
            idx, dest = contiguous_pool(ps, cfg, dev)
            h = torch.cat([hs[step][r : r + 1] for r in rows])
            out = layer.forward(h, cur, kv, dest, idx, cos, sin).clone()
            torch.cuda.synchronize()
            seq.append((out, {k: v.clone() for k, v in layer.intermediates().items()}))
        layer.close()
        return seq

    batched = run(list(range(S)))
    for s in range(S):
        alone = run([s])
        for pos in range(steps):
            # Attention and the residual are per-sample work either way, so they are
            # bit-exact; a stage that leaked another sample's state shows up here.
            for name in ("q_a", "kv", "q", "o", "o_lora", "a", "scores"):
                d = (batched[pos][1][name][s].float() - alone[pos][1][name][0].float()).abs().max().item()
                assert d == 0.0, f"pos={pos} sample {s}: stage {name} moved by {d:.3e} when batched"
            assert batched[pos][1]["sel"][s].tolist() == alone[pos][1]["sel"][0].tolist()
            # x_out is not bit-exact and must not be asserted so: `down` folds K over
            # S * MOE_SLOTS units, so batching regroups the f32 summation tree and
            # the last bf16 rounding can differ by an ulp. A leaked sample would be
            # wrong by far more than this bar.
            a, b = batched[pos][0][s].float(), alone[pos][0][0].float()
            rel = ((a - b).norm() / b.norm().clamp(min=1e-6)).item()
            assert rel < 1e-3, f"pos={pos} sample {s}: x_out rel {rel:.3e} when batched"


def test_dsv4_state_slots_place_the_rolling_state():
    """The compressor's state lives where `state_slots` says, not at sample s.

    Every other test hands the kernel the identity assignment, so a kernel that
    ignored the slot vector entirely would pass all of them. Here the same two
    sequences run twice: once in a pool of exactly S slots taking 0 and 1, and
    once in a larger pool taking 3 and 1 -- non-identity, non-contiguous, and
    not even in order. The answers must be identical, because which slot a
    sequence's window lives in is bookkeeping, not arithmetic.

    That is the property a serving pool needs: it hands slots out per request,
    they are not contiguous, and it relocates them.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    dev, mode, S, ratio = "cuda", MoeMode.W8A8, 2, COMPRESS_CSA
    cfg = _cfg(hc_mult=4)
    cfg.compress_ratio, cfg.max_seq, cfg.index_topk = ratio, 256, 4
    cfg.validate()
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    cos, sin = rope_table(2048, theta=cfg.rope_base, device=dev)
    steps, coff, ihd = 6 * ratio, cfg.c_coff, cfg.index_head_dim
    hs = [(0.5 * torch.randn(S, cfg.hc_mult, cfg.hidden, device=dev)).bfloat16() for _ in range(steps)]

    def run(slots, pool):
        layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)
        if pool != S:  # repoint at a wider pool, as a runtime's allocator would
            layer.state_slots = torch.tensor(slots, dtype=torch.int32, device=dev)
            # POISON every slot, then initialise only the two that were handed out.
            # Without this the test cannot tell "uses the slot vector" from "uses
            # the sample index": any injective map onto private, zeroed slots gives
            # the same answer. Reading the wrong slot has to read somebody's data.
            layer.kv_state = torch.randn(pool, cfg.c_rows, coff * cfg.head_dim, device=dev)
            layer.score_state = torch.randn(pool, cfg.c_rows, coff * cfg.head_dim, device=dev)
            layer.i_kv_state = torch.randn(pool, cfg.c_rows, coff * ihd, device=dev)
            layer.i_score_state = torch.randn(pool, cfg.c_rows, coff * ihd, device=dev)
            layer.i_cache = torch.randn(pool, cfg.n_compressed, ihd, device=dev).bfloat16()
            for sl in slots:
                layer.kv_state[sl] = 0
                layer.score_state[sl] = float("-inf")
                layer.i_kv_state[sl] = 0
                layer.i_score_state[sl] = float("-inf")
                layer.i_cache[sl] = 0
        kv = torch.zeros(S * cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        outs = []
        for pos in range(steps):
            cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
            idx, dest = contiguous_pool([pos] * S, cfg, dev)
            outs.append(layer.forward(hs[pos], cur, kv, dest, idx, cos, sin).clone())
        torch.cuda.synchronize()
        layer.close()
        return outs

    # the scattered run takes slots that are neither the sample indices nor in
    # order, in a pool whose other slots hold noise
    packed, scattered = run([0, 1], S), run([3, 1], S + 3)
    for pos in range(steps):
        d = (packed[pos].float() - scattered[pos].float()).abs().max().item()
        assert d == 0.0, f"pos={pos}: moving the state to other slots changed the answer by {d:.3e}"


def test_dsv4_compress_schedule_is_the_checkpoints():
    """The per-layer variant schedule, against DeepSeek-V4-Pro's own config.

    Structural first, so this runs anywhere: 62 entries for 61 main layers plus
    MTP, 31 HCA and 30 CSA among the main ones and no ratio-0 there, layers 0 and
    1 both HCA, and strict parity from id 2. Then, when a copy of the config
    happens to be on the box, byte-for-byte -- the artifact is not always
    present, and a test that silently skipped its only real check would be worse
    than one that states what it checked.
    """
    import json
    import os

    from kernels.dsv4_moe_layer.config import COMPRESS_CSA as CSA
    from kernels.dsv4_moe_layer.config import COMPRESS_HCA as HCA
    from kernels.dsv4_moe_layer.config import compress_ratios

    r = compress_ratios()
    assert len(r) == 62, f"61 layers plus one MTP entry, got {len(r)}"
    main, mtp = r[:61], r[61:]
    assert main.count(HCA) == 31 and main.count(CSA) == 30, f"31 HCA + 30 CSA, got {main}"
    assert 0 not in main, "V4-Pro has no sliding-window-only main layer"
    assert mtp == (0,), "the MTP block is the ratio-0 entry"
    assert main[0] == main[1] == HCA, "layers 0 and 1 are the one break in the alternation"
    for i in range(2, 61):
        want = CSA if i % 2 == 0 else HCA
        assert main[i] == want, f"layer {i} should be {want}, schedule says {main[i]}"

    path = "/tmp/dsv4_ref/config.json"
    if os.path.exists(path):
        assert list(r) == json.load(open(path))["compress_ratios"], "schedule differs from the config"


def test_dsv4_split_merge_spans_the_block():
    """More key splits than one wave holds, against the golden.

    The flash merge in `uv` gives each split one thread. While they fit in a
    wave that reduction is register-only; past 64 it has to run over the whole
    block, and the two paths are selected at trace time, so nothing else in the
    suite compiles the wide one -- every other shape here has at most 18 splits.

    Overrunning the width is SILENT rather than a crash: the splits above it get
    no weight, so the softmax simply normalises over a prefix of the keys and the
    answer is a plausible number that attends to the wrong set. That is why this
    checks the output against the golden instead of merely that it runs.

    max_seq is the knob: n_keys = window + max_seq // ratio, and 64 splits is
    4096 keys, so a HCA layer needs more than ~508K of context to get here.
    """
    from kernels.dsv4_moe_layer.dsv4_kernel import SPLIT_KEYS
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    cfg = _cfg()
    cfg.compress_ratio, cfg.max_seq = COMPRESS_HCA, 655360
    cfg.validate()
    splits = cfg.n_keys // SPLIT_KEYS
    assert splits > 64, f"this shape must exercise the wide path, got {splits} splits"

    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=mode)
    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.max_seq - 1  # every compressed entry live
    cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(S * cfg.cache_rows, cfg.head_dim, device=dev)).bfloat16()
    idx, dest = contiguous_pool([pos] * S, cfg, dev)
    cos, sin = rope_table(cfg.max_seq, theta=cfg.rope_theta, device=dev)

    out = layer.forward(h, cur, kv0.clone(), dest, idx, cos, sin)
    torch.cuda.synchronize()
    ref = golden_layer(
        W,
        h,
        [pos] * S,
        kv0.clone(),
        dest,
        idx,
        cos,
        sin,
        lambda z: z,
        moe_mode=mode,
        kv_state=torch.zeros(S, cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev),
        score_state=torch.full(
            (S, cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev
        ),
        cos_c=cos,
        sin_c=sin,
    )

    a_out, b_out = out.float(), ref["x_out"].float()
    assert torch.isfinite(a_out).all(), "the wide merge produced non-finite output"
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    assert rel_l2 < _tol(OUT_REL_L2, cfg.hc_mult, 1), f"x_out diverged: rel_l2 {rel_l2:.5f}"


@pytest.mark.parametrize("n_layers", [4])
def test_dsv4_alternating_stack_matches_golden(n_layers):
    """A stack whose attention variant changes per layer, against the golden.

    This is what per-layer variants is for. Layers of one variant SHARE a
    compiled kernel, its scratch and its symmetric buffer -- only the weights and
    the rolling compressor state are per layer -- so the stack builds one
    Dsv4Variant per distinct ratio and hands it to every layer that uses it.
    Sharing scratch is why the layers must pass distinct `layer` tags.

    Four layers is the smallest prefix of V4-Pro's own schedule that exercises
    the alternation AND the repeat: [HCA, HCA, CSA, HCA], so the HCA variant is
    shared by three layers with three different weight sets and three different
    rolling states, which is exactly the case a per-layer object would get right
    by accident and a shared one has to get right on purpose.
    """
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer, Dsv4Variant

    torch.manual_seed(0)
    dev, mode, S = "cuda", MoeMode.W8A8, 1
    base = _cfg(hc_mult=4)
    base.max_seq = 256
    cfgs = []
    for i in range(n_layers):
        c = base.for_layer(i)
        if c.indexed:
            c.index_topk = 4
        c.validate()
        cfgs.append(c)
    assert {c.compress_ratio for c in cfgs} == {COMPRESS_HCA, COMPRESS_CSA}, "the prefix must alternate"

    Ws = [make_weights(rank=0, cfg=c, device=dev, seed=100 + i, moe_mode=mode) for i, c in enumerate(cfgs)]
    variants, layers = {}, []
    for i, c in enumerate(cfgs):
        if c.compress_ratio not in variants:
            variants[c.compress_ratio] = Dsv4Variant(c, S, rank=0, npes=1, moe_mode=mode, allow_unindexed_csa=True)
        layers.append(
            Dsv4MoeLayer(
                Ws[i],
                S,
                rank=0,
                npes=1,
                moe_mode=mode,
                allow_unindexed_csa=True,
                variant=variants[c.compress_ratio],
            )
        )
    assert len(variants) == 2, "three HCA layers must share one compiled kernel"

    tables = {c.rope_base: rope_table(2048, theta=c.rope_base, device=dev) for c in cfgs}
    # separate caches: the two sides each evolve their own, or the golden would
    # gather rows the kernel wrote and the comparison would stop being one
    kvs_k = [torch.zeros(c.cache_rows, c.head_dim, dtype=torch.bfloat16, device=dev) for c in cfgs]
    kvs_r = [torch.zeros(c.cache_rows, c.head_dim, dtype=torch.bfloat16, device=dev) for c in cfgs]

    # each layer keeps its OWN rolling compressor state, as every layer of a real
    # stack does; sharing the variant shares the kernel, never the state
    def fresh_states():
        st_all = []
        for c in cfgs:
            coff, ihd = c.c_coff, c.index_head_dim
            st = dict(
                kv_state=torch.zeros(S, c.c_rows, coff * c.head_dim, device=dev),
                score_state=torch.full((S, c.c_rows, coff * c.head_dim), float("-inf"), device=dev),
            )
            if c.indexed:
                st |= dict(
                    i_state=torch.zeros(S, c.c_rows, coff * ihd, device=dev),
                    i_score_state=torch.full((S, c.c_rows, coff * ihd), float("-inf"), device=dev),
                    i_cache=torch.zeros(S, c.n_compressed, ihd, dtype=torch.bfloat16, device=dev),
                )
            st_all.append(st)
        return st_all

    # the kernel's rolling state is the layer object's; the golden threads its own
    k_states, states = fresh_states(), fresh_states()
    for lay, st in zip(layers, k_states):
        for k, v in st.items():
            setattr(lay, {"i_state": "i_kv_state"}.get(k, k), v)

    # Each layer is judged against the golden fed THIS layer's kernel input, not
    # against a golden chained end to end. The chain is still what produced that
    # input, so variant dispatch, weights and state are all under test -- but a
    # single routing flip at any layer's top-k cut would otherwise dominate the
    # end-to-end number and stop it measuring any of them. Same reason the
    # single-layer tests rebase on the kernel's own routing.
    for pos in range(3 * COMPRESS_CSA):
        h = (0.5 * torch.randn(S, base.hc_mult, base.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos] * S, dtype=torch.int32, device=dev)
        hk = h
        for i, (c, lay) in enumerate(zip(cfgs, layers)):
            cos, sin = tables[c.rope_base]
            idx, dest = contiguous_pool([pos] * S, c, dev)
            h_in = hk
            hk = lay.forward(h_in, cur, kvs_k[i], dest, idx, cos, sin, layer=i, advance=False)
            torch.cuda.synchronize()
            got = lay.intermediates()
            kw = dict(states[i])
            if c.compress_ratio:
                kw |= dict(cos_c=cos, sin_c=sin)
            ref = golden_layer(Ws[i], h_in, [pos] * S, kvs_r[i], dest, idx, cos, sin, lambda z: z, moe_mode=mode, **kw)
            if _routing_flipped(got, ref, Ws[i], c, S):
                ref = _rebase_on_own_routing(got, ref, Ws[i], mode)
            a, b = hk.float(), ref["x_out"].float()
            rel = ((a - b).norm() / b.norm()).item()
            tol = _tol(OUT_REL_L2, c.hc_mult, 1)
            assert rel < tol, f"pos={pos} layer {i} (ratio {c.compress_ratio}): rel_l2 {rel:.4f} >= {tol}"
        for v in variants.values():
            v.advance_step()
    for v in variants.values():
        v.close()
