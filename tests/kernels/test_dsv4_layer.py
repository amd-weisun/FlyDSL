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

from kernels.dsv4_moe_layer.config import MoeMode
from kernels.dsv4_moe_layer.reference import (
    V4Config,
    golden_layer,
    golden_moe,
    layer_idxs,
    make_weights,
    window_idxs,
)
from kernels.mla_moe_layer.reference import rope_table

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


@pytest.mark.parametrize("moe_mode", [MoeMode.A8W4, MoeMode.W8A8])
@pytest.mark.parametrize("hc_mult", [1, 4])
def test_dsv4_layer_matches_golden(moe_mode, hc_mult):
    """hc_mult=1 is a plain residual; 4 is V4's hyper-connection stream."""
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = _cfg(hc_mult)
    cfg.validate()
    dev, S = "cuda", 1
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=moe_mode)
    layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=moe_mode)

    hshape = (S, cfg.hidden) if cfg.hc_mult == 1 else (S, cfg.hc_mult, cfg.hidden)
    h = (0.5 * torch.randn(*hshape, device=dev)).bfloat16()
    pos = cfg.window  # ring already wrapped once
    cur = torch.tensor([pos], dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx = window_idxs(pos, S, cfg.window, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)

    kv_kernel = kv0.clone()
    out = layer.forward(h, cur, kv_kernel, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()

    kv_ref = kv0.clone()
    ref = golden_layer(W, h, pos, kv_ref, idx, cos, sin, lambda z: z, moe_mode=moe_mode)

    for name, base in STAGE_TOL.items():
        tol = _tol(base, cfg.hc_mult, 1) if name in SCALES_WITH_CONFIG else base
        a = got[name].float().reshape(-1)
        b = ref[name].float().reshape(-1)
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"

    # routing must agree exactly: a different expert set is not a rounding artefact
    assert got["sel"].tolist() == ref["sel"].tolist(), "expert selection differs"
    assert out.shape == h.shape, f"the layer must preserve its input shape, got {out.shape}"

    # end to end, judge by relative L2: one FP8/bf16 rounding flip upstream moves a
    # single element a long way, and hc_post mixes hc_mult streams so it propagates
    a_out, b_out = out.float(), ref["x_out"].float()
    rel_max = (a_out - b_out).abs().max().item() / max(b_out.abs().max().item(), 1e-6)
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    out_tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol} (rel_max {rel_max:.5f})"


@pytest.mark.large_shape
def test_dsv4_layer_matches_golden_at_real_dims():
    """The reduced shard above cannot catch mappings that only break at V4's own
    numbers -- 384 experts overflowed the selection key's id field, which 256 (and
    the reduced 128) fit exactly. Keep a real-shard case."""
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = V4Config(hc_mult=1)  # defaults are DeepSeek-V4-Pro at TP8
    cfg.validate()
    dev, S, mode = "cuda", 1, MoeMode.A8W4
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=mode)
    layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=mode)

    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
    pos = cfg.window
    cur = torch.tensor([pos], dtype=torch.int32, device=dev)
    kv0 = (0.3 * torch.randn(cfg.window, cfg.head_dim, device=dev)).bfloat16()
    idx = window_idxs(pos, S, cfg.window, dev)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)

    kv_kernel = kv0.clone()
    out = layer.forward(h, cur, kv_kernel, idx, cos, sin)
    torch.cuda.synchronize()
    got = layer.intermediates()
    ref = golden_layer(W, h, pos, kv0.clone(), idx, cos, sin, lambda z: z, moe_mode=mode)

    for name, base in STAGE_TOL.items():
        tol = _tol(base, cfg.hc_mult, 1) if name in SCALES_WITH_CONFIG else base
        a = got[name].float().reshape(-1)
        b = ref[name].float().reshape(-1)
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"
    assert got["sel"].tolist() == ref["sel"].tolist(), "expert selection differs"
    # an id above 255 is the case the 8-bit key field used to corrupt
    assert max(got["sel"].reshape(-1).tolist()[1:]) > 255 or cfg.n_experts <= 256

    a_out, b_out = out.float(), ref["x_out"].float()
    rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
    out_tol = _tol(OUT_REL_L2, cfg.hc_mult, 1)
    assert rel_l2 < out_tol, f"x_out diverged: rel_l2 {rel_l2:.5f} >= {out_tol}"


def test_dsv4_rejects_unsupported_compress_ratio():
    """The KV compressor (HCA) and lightning indexer (CSA) are not implemented yet."""
    from kernels.dsv4_moe_layer.config import COMPRESS_CSA, validate_shard

    with pytest.raises(ValueError, match="compress_ratio"):
        validate_shard(1, 16, 0, 8, compress_ratio=COMPRESS_CSA)


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


def _tp_cfg(real: bool, hc_mult: int = 1, compress_ratio: int = 0):
    # NOT a module global: mp.spawn re-imports this module in each child, so
    # anything set under __main__ never reaches the workers
    cfg = V4Config(hc_mult=hc_mult) if real else _cfg(hc_mult)
    if compress_ratio:
        cfg.compress_ratio, cfg.max_seq = compress_ratio, 256
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
    # a compressing layer rotates everything on compress_rope_theta, a pure
    # sliding-window layer on rope_theta -- one table either way
    theta = cfg.compress_rope_theta if compress_ratio else cfg.rope_theta
    cos, sin = rope_table(4096, theta=theta, device=dev)
    gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical inputs everywhere
    # Without compression each step is independent, so the cache is re-seeded from
    # kv0 every iteration. The compressor carries state across steps, so its run has
    # to be a real sequential decode: one cache, one rolling state, advancing pos.
    kv0 = torch.randn(cfg.window, cfg.head_dim, generator=gen, device=dev).to(torch.bfloat16)
    pos = cfg.window
    idx = window_idxs(pos, 1, cfg.window, dev)
    if compress_ratio:
        pos = 0
        kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        ks = torch.zeros(cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
        ss = torch.full((cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

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
            idx = layer_idxs(pos, 1, cfg, dev)
            boundaries += (pos + 1) % compress_ratio == 0
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        kv_in = kv_k if compress_ratio else kv0.clone()
        out = layer.forward(h, cur, kv_in, idx, cos, sin)
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
            pos,
            kv_r if compress_ratio else kv0.clone(),
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


def _worker(rank, npes, real, iters, hc_mult, compress_ratio, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29551", rank=rank, world_size=npes)
    try:
        results[rank] = run_rank(rank, npes, real=real, iters=iters, hc_mult=hc_mult, compress_ratio=compress_ratio)
    finally:
        dist.destroy_process_group()


def run_tp(npes, real=False, iters=2, hc_mult=1, compress_ratio=0):
    if npes == 1:
        return run_rank(0, 1, real=real, iters=iters, hc_mult=hc_mult, compress_ratio=compress_ratio)
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_worker, args=(npes, real, iters, hc_mult, compress_ratio, results), nprocs=npes)
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


def bench_rank(rank, npes, real=True, iters=320, group=None, timeline=False, moe_mode=MoeMode.A8W4, hc_mult=1):
    """Returns us per layer."""
    import torch.distributed as dist

    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = _tp_cfg(real, hc_mult)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    kv = torch.randn(cfg.window, cfg.head_dim, device=dev).to(torch.bfloat16)
    pos = cfg.window
    idx = window_idxs(pos, 1, cfg.window, dev)
    cur = torch.tensor([pos], dtype=torch.int32, device=dev)
    op = Dsv4MoeLayer(W, 1, rank=rank, npes=npes, group=group, moe_mode=moe_mode)
    hshape = (1, cfg.hidden) if cfg.hc_mult == 1 else (1, cfg.hc_mult, cfg.hidden)
    h = torch.randn(*hshape, device=dev).to(torch.bfloat16)
    x = torch.empty_like(h)
    for _ in range(10):
        op.forward(h, cur, kv, idx, cos, sin, x_out=x)
    torch.cuda.synchronize()
    if npes > 1:
        dist.barrier()

    if timeline:
        top = Dsv4MoeLayer(W, 1, rank=rank, npes=npes, group=group, timeline=True, moe_mode=moe_mode)
        for _ in range(3):
            top.forward(h, cur, kv, idx, cos, sin, x_out=x)
        torch.cuda.synchronize()
        if rank == 0:
            print(top.timeline_report(), flush=True)
        top.close()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for layer in range(BENCH_LAYERS):
            op.forward(h, cur, kv, idx, cos, sin, x_out=x, layer=layer, advance=False)
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
    return us


def _bench_worker(rank, npes, real, timeline, moe_mode, hc_mult, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29552", rank=rank, world_size=npes)
    try:
        results[rank] = bench_rank(rank, npes, real=real, timeline=timeline, moe_mode=moe_mode, hc_mult=hc_mult)
    finally:
        dist.destroy_process_group()


def run_bench(npes, real=True, timeline=False, moe_mode=MoeMode.A8W4, hc_mult=1):
    if npes == 1:
        return {0: bench_rank(0, 1, real=real, timeline=timeline, moe_mode=moe_mode, hc_mult=hc_mult)}
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_bench_worker, args=(npes, real, timeline, moe_mode, hc_mult, results), nprocs=npes)
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
        us = [res[r] for r in sorted(res)]
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

    # ONE table for the whole layer, on compress_rope_theta: V4 picks the rope base
    # per LAYER, not per consumer -- a compressing layer rotates its window q/kv on
    # the same table as its compressed rows.
    cos, sin = rope_table(2048, theta=cfg.compress_rope_theta, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    # -inf like the layer's own state: unwritten rows must drop out of the softmax
    ss = torch.full((cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    boundaries = 0
    for pos in range(3 * ratio + 2):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx = layer_idxs(pos, 1, cfg, dev)
        out = layer.forward(h, cur, kv_k, idx, cos, sin)
        torch.cuda.synchronize()
        ref = golden_layer(
            W,
            h,
            pos,
            kv_r,
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

    # one table for the whole layer, on the compressing layer's base
    cos, sin = rope_table(2048, theta=cfg.compress_rope_theta, device=dev)
    kv_k = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    kv_r = torch.zeros(cfg.cache_rows, cfg.head_dim, dtype=torch.bfloat16, device=dev)
    ks = torch.zeros(cfg.c_rows, cfg.c_coff * cfg.head_dim, device=dev)
    ss = torch.full((cfg.c_rows, cfg.c_coff * cfg.head_dim), float("-inf"), device=dev)

    # Two boundaries, not one: the first anchors at position 0, where every rope
    # table is the identity, so it cannot tell the compressor's base from the
    # window's. The second anchors at `ratio` and does.
    boundaries, checked, top_id = 0, 0, 0
    for pos in range(2 * ratio + 4):
        h = (0.5 * torch.randn(1, cfg.hidden, device=dev)).bfloat16()
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        idx = layer_idxs(pos, 1, cfg, dev)
        out = layer.forward(h, cur, kv_k, idx, cos, sin)
        torch.cuda.synchronize()
        ref = golden_layer(
            W,
            h,
            pos,
            kv_r,
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
