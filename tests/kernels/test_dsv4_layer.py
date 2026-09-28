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
    make_weights,
    window_idxs,
)
from kernels.mla_moe_layer.reference import rope_table

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

# bf16 MFMA activations, so exact equality is not the bar; the attention and MoE
# halves each round several times before x_out
STAGE_TOL = {
    "q_a": 1e-3,
    "kv": 1e-3,
    "q": 0.02,
    "o": 0.02,
    "o_lora": 0.02,
    "a": 0.02,
    "scores": 0.02,
    "mid": 0.02,
}
OUT_TOL = 0.02
# End to end, a single FP8/bf16 rounding flip upstream moves one element a long
# way, so max-abs is the wrong statistic there -- judge the final hidden state by
# relative L2, as the MLA kernel's own suite does for x_out_e2e.
OUT_REL_L2 = 0.05


def _cfg():
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
    )


@pytest.mark.parametrize("moe_mode", [MoeMode.A8W4, MoeMode.W8A8])
def test_dsv4_layer_matches_golden(moe_mode):
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = _cfg()
    cfg.validate()
    dev, S = "cuda", 1
    W = make_weights(rank=0, cfg=cfg, device=dev, seed=3, moe_mode=moe_mode)
    layer = Dsv4MoeLayer(W, samples=S, rank=0, npes=1, moe_mode=moe_mode)

    h = (0.5 * torch.randn(S, cfg.hidden, device=dev)).bfloat16()
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

    for name, tol in STAGE_TOL.items():
        a = got[name].float().reshape(-1)
        b = ref[name].float().reshape(-1)
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"

    # routing must agree exactly: a different expert set is not a rounding artefact
    assert got["sel"].tolist() == ref["sel"].tolist(), "expert selection differs"

    rel_out = (out.float() - ref["x_out"].float()).abs().max().item() / max(
        ref["x_out"].float().abs().max().item(), 1e-6
    )
    assert rel_out < OUT_TOL, f"x_out diverged: rel {rel_out:.5f}"


@pytest.mark.large_shape
def test_dsv4_layer_matches_golden_at_real_dims():
    """The reduced shard above cannot catch mappings that only break at V4's own
    numbers -- 384 experts overflowed the selection key's id field, which 256 (and
    the reduced 128) fit exactly. Keep a real-shard case."""
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    torch.manual_seed(0)
    cfg = V4Config()  # defaults are DeepSeek-V4-Pro at TP8
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

    for name, tol in STAGE_TOL.items():
        a = got[name].float().reshape(-1)
        b = ref[name].float().reshape(-1)
        rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
        assert rel < tol, f"stage {name} diverged: rel {rel:.5f} >= {tol}"
    assert got["sel"].tolist() == ref["sel"].tolist(), "expert selection differs"
    # an id above 255 is the case the 8-bit key field used to corrupt
    assert max(got["sel"].reshape(-1).tolist()[1:]) > 255 or cfg.n_experts <= 256

    rel_out = (out.float() - ref["x_out"].float()).abs().max().item() / max(
        ref["x_out"].float().abs().max().item(), 1e-6
    )
    assert rel_out < OUT_TOL, f"x_out diverged: rel {rel_out:.5f}"


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


def _tp_cfg(real: bool):
    return V4Config() if real else _cfg()


def run_rank(rank, npes, real=False, iters=2, group=None, moe_mode=MoeMode.A8W4):
    import torch.distributed as dist

    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = _tp_cfg(real)
    cfg.validate()
    W = make_weights(rank, cfg=cfg, device=dev, seed=TP_SEED, moe_mode=moe_mode)
    layer = Dsv4MoeLayer(W, samples=1, rank=rank, npes=npes, group=group, moe_mode=moe_mode)
    cos, sin = rope_table(4096, theta=cfg.rope_theta, device=dev)
    gen = torch.Generator(device=dev).manual_seed(TP_SEED + 99)  # identical inputs everywhere
    kv0 = torch.randn(cfg.window, cfg.head_dim, generator=gen, device=dev).to(torch.bfloat16)
    pos = cfg.window
    idx = window_idxs(pos, 1, cfg.window, dev)

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
    for _ in range(iters):
        h = torch.randn(1, cfg.hidden, generator=gen, device=dev).to(torch.bfloat16)
        cur = torch.tensor([pos], dtype=torch.int32, device=dev)
        out = layer.forward(h, cur, kv0.clone(), idx, cos, sin)
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

        ref = golden_layer(W, h, pos, kv0.clone(), idx, cos, sin, allreduce, moe_mode=moe_mode)
        for name, tol in STAGE_TOL.items():
            a = got[name].float().reshape(-1)
            b = ref[name].float().reshape(-1)
            rel = (a - b).abs().max().item() / max(b.abs().max().item(), 1e-6)
            if rel >= tol:
                print(f"rank {rank}: stage {name} rel {rel:.5f} >= {tol}", flush=True)
                ok = False
        if got["sel"].tolist() != ref["sel"].tolist():
            print(f"rank {rank}: expert selection differs", flush=True)
            ok = False
        a_out, b_out = out.float(), ref["x_out"].float()
        rel_max = (a_out - b_out).abs().max().item() / max(b_out.abs().max().item(), 1e-6)
        rel_l2 = ((a_out - b_out).norm() / b_out.norm()).item()
        if rel_l2 >= OUT_REL_L2:
            print(f"rank {rank}: x_out rel_max {rel_max:.5f}  rel_l2 {rel_l2:.5f}", flush=True)
            ok = False
    layer.close()
    return ok


def _worker(rank, npes, real, iters, results):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method="tcp://127.0.0.1:29551", rank=rank, world_size=npes)
    try:
        results[rank] = run_rank(rank, npes, real=real, iters=iters)
    finally:
        dist.destroy_process_group()


def run_tp(npes, real=False, iters=2):
    if npes == 1:
        return run_rank(0, 1, real=real, iters=iters)
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_worker, args=(npes, real, iters, results), nprocs=npes)
    return all(results[r] for r in range(npes))


@pytest.mark.multi_gpu
def test_dsv4_layer_tp8():
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    assert run_tp(8)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--npes", type=int, default=8)
    ap.add_argument("--real", action="store_true", help="use the real V4-Pro TP8 shard")
    ap.add_argument("--iters", type=int, default=2)
    a = ap.parse_args()
    print("PASS" if run_tp(a.npes, a.real, a.iters) else "FAIL")
