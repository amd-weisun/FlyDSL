# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the fused Qwen3.5 layer kernel (both mixers) against its torch golden, stage by stage.

One TP4 rank's real shard (GDN hk 4 / hv 16; 8 query heads + 1 K / V head; 512 experts, 256 local
intermediates) on one GPU, over several decode steps, so the state the kernel writes (conv and
recurrent state, the paged FP8 K / V cache) feeds its next step. TP4 cases run under ``-m multi_gpu``.
"""

from __future__ import annotations

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.monokernel.qwen35.config import Q35Config
from kernels.monokernel.qwen35.op import Qwen35MonoKernel
from kernels.monokernel.qwen35.reference import (
    golden_layer,
    golden_moe,
    kv_encode,
    kv_quant_dequant,
    make_weights,
    paged_kv_read,
    paged_kv_write,
    rope_tables,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not str(get_rocm_arch() or "").startswith("gfx95"):
    pytest.skip("the MonoKernel's MFMA paths are written for gfx950", allow_module_level=True)

STEPS = 3


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _states(cfg, n_slots, gen):
    conv = (0.5 * torch.randn(n_slots, cfg.conv_kernel - 1, cfg.conv_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    ssm = (0.1 * torch.randn(n_slots, cfg.hv, cfg.linear_v_dim, cfg.linear_k_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    return conv, ssm


@pytest.mark.parametrize("S", [1, 2, 8])
@pytest.mark.parametrize("shift", [False, True], ids=["in_place", "new_slots"])
def test_qwen35_gdn_layer_matches_golden(S, shift):
    cfg = Q35Config(tp=4, rank=0)
    W = make_weights(cfg, full=False)
    gen = torch.Generator(device="cuda").manual_seed(7)
    n_slots = 2 * S + 1
    conv_k, ssm_k = _states(cfg, n_slots, gen)
    conv_g, ssm_g = conv_k.clone(), ssm_k.clone()
    # samples read slot 2s + 1 and write 2s + 2 (new_slots) or the same slot (in_place)
    slots_in = torch.arange(S, dtype=torch.int32, device="cuda") * 2 + 1
    slots_out = slots_in + 1 if shift else slots_in.clone()
    layer = Qwen35MonoKernel(W, S)
    try:
        for step in range(STEPS):
            h = (torch.randn(S, cfg.hidden, generator=gen, device="cuda")).to(torch.bfloat16)
            ref = golden_layer(W, h, lambda y: y, gdn_state=(conv_g, ssm_g, slots_in, slots_out))
            out = layer.forward(h, gdn_state=(conv_k, ssm_k, slots_in, slots_out))
            torch.cuda.synchronize()
            got = layer.intermediates()
            errs = {
                "conv": _rel(got["conv"], ref["conv_out"]),
                "z": _rel(got["z"], ref["z"].flatten(1)),
                "b": _rel(got["ba"][:, : cfg.hv], ref["beta"].logit()),
                "gated": _rel(got["gated"], ref["gated"].flatten(1)),
                "a": _rel(got["a"], ref["a"]),
                "logits": _rel(got["scores"][:, : cfg.n_experts], ref["logits"]),
                "conv_state": _rel(conv_k[slots_out.long()], conv_g[slots_out.long()]),
                "ssm_state": _rel(ssm_k[slots_out.long()], ssm_g[slots_out.long()]),
            }
            same_pick = torch.equal(got["sel"].long(), ref["sel"].long())
            if same_pick:  # a near-tie flips an expert: then only the stages before it are comparable
                errs["prob"] = _rel(got["prob"], ref["prob"])
                errs["xq"] = _rel(got["xq"], ref["xq"])
                # the experts from the kernel's own residual and FP8 input (midpoint roundings differ)
                own = golden_moe(W, got["a"].to(torch.bfloat16), lambda y: y, xq=got["xq"])
                errs["mid"] = _rel(got["mid"], own["mid"])
                errs["x_out_own"] = _rel(out, own["x_out"])
                errs["x_out"] = _rel(out, ref["x_out"])
            print(f"S={S} shift={shift} step {step} pick={same_pick}", {k: f"{v:.2e}" for k, v in errs.items()})
            bad = {k: v for k, v in errs.items() if v > 2e-2}
            assert not bad, f"step {step}: {bad} (all: {errs})"
        # with new slots the input slots were only read: the kernel must not have written them
        if shift:
            assert torch.equal(conv_k[slots_in.long()], conv_g[slots_in.long()])
    finally:
        layer.close()


# per-sample positions of the first step: a partial block, block / tile edges, many splits
FULL_POS = {1: [1000], 2: [37, 300], 8: [5, 15, 16, 63, 64, 200, 777, 2049]}


@pytest.mark.parametrize("S", [1, 2, 8])
def test_qwen35_full_attention_layer_matches_golden(S):
    cfg = Q35Config(tp=4, rank=0)
    W = make_weights(cfg, full=True)
    gen = torch.Generator(device="cuda").manual_seed(13)
    pos0 = FULL_POS[S]
    max_seq = max(pos0) + STEPS + 1
    max_blocks = -(-max_seq // 16)
    # every sample its own blocks, in a shuffled order (paging must matter)
    perm = torch.randperm(S * max_blocks, generator=torch.Generator().manual_seed(3)).view(S, max_blocks)
    bt = perm.to(torch.int32).cuda().contiguous()
    nb = S * max_blocks
    kc = torch.zeros(nb, 1, cfg.head_dim // 16, 16, 16, dtype=torch.float8_e4m3fn, device="cuda")
    vc = torch.zeros(nb, 1, 1, cfg.head_dim, 16, dtype=torch.float8_e4m3fn, device="cuda")
    ks = torch.zeros(nb, 1, 16, device="cuda")
    vs = torch.zeros(nb, 1, 16, device="cuda")
    k_g = torch.zeros(S, max_seq, 1, cfg.head_dim, device="cuda")
    v_g = torch.zeros_like(k_g)
    for s in range(S):
        hist_k = torch.randn(pos0[s], 1, cfg.head_dim, generator=gen, device="cuda")
        hist_v = torch.randn(pos0[s], 1, cfg.head_dim, generator=gen, device="cuda")
        k_g[s, : pos0[s]] = kv_quant_dequant(hist_k)
        v_g[s, : pos0[s]] = kv_quant_dequant(hist_v)
        for p in range(pos0[s]):
            paged_kv_write(kc, vc, ks, vs, bt[s].tolist(), p, hist_k[p], hist_v[p])
    cos, sin = rope_tables(cfg, max_seq)
    layer = Qwen35MonoKernel(W, S)
    try:
        for step in range(STEPS):
            pos = torch.tensor([p + step for p in pos0], dtype=torch.int32, device="cuda")
            h = torch.randn(S, cfg.hidden, generator=gen, device="cuda").to(torch.bfloat16)
            ref = golden_layer(W, h, lambda y: y, kv=(k_g, v_g, pos))
            out = layer.forward(h, kv=(kc, vc, ks, vs, pos, bt), rope=(cos, sin))
            torch.cuda.synchronize()
            got = layer.intermediates()
            new_k = torch.stack([paged_kv_read(kc, vc, ks, vs, bt[s].tolist(), int(pos[s]))[0] for s in range(S)])
            new_v = torch.stack([paged_kv_read(kc, vc, ks, vs, bt[s].tolist(), int(pos[s]))[1] for s in range(S)])
            # the stored per-token scales are ATOM's convention (amax / 448), not just self-consistent
            blk = torch.stack([bt[s, int(pos[s]) // 16] for s in range(S)]).long()
            tok = (pos % 16).long()
            errs = {
                "k_scale": _rel(ks[blk, 0, tok], kv_encode(ref["k"])[1][:, 0]),
                "v_scale": _rel(vs[blk, 0, tok], kv_encode(ref["v"])[1][:, 0]),
                "q": _rel(got["q"], ref["q"]),
                "k_cache": _rel(new_k, k_g[torch.arange(S), pos.long()]),
                "v_cache": _rel(new_v, v_g[torch.arange(S), pos.long()]),
                "gated": _rel(got["gated"], ref["gated"].flatten(1)),
                "a": _rel(got["a"], ref["a"]),
                "logits": _rel(got["scores"][:, : cfg.n_experts], ref["logits"]),
            }
            own = golden_moe(W, got["a"].to(torch.bfloat16), lambda y: y, xq=got["xq"])
            if torch.equal(got["sel"].long(), own["sel"].long()):
                errs["mid"] = _rel(got["mid"], own["mid"])
                errs["x_out_own"] = _rel(out, own["x_out"])
            print(f"full S={S} step {step}", {k: f"{v:.2e}" for k, v in errs.items()})
            bad = {k: v for k, v in errs.items() if v > 2e-2}
            assert not bad, f"step {step}: {bad} (all: {errs})"
    finally:
        layer.close()


def _run_rank_tp(rank, npes, S=2, iters=3):
    import torch.distributed as dist

    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    cfg = Q35Config(tp=npes, rank=rank)
    W = make_weights(cfg, full=False, device=dev)
    gen = torch.Generator(device=dev).manual_seed(11)  # identical residual inputs on every rank
    conv_k, ssm_k = _states(cfg, S, torch.Generator(device=dev).manual_seed(100 + rank))
    conv_g, ssm_g = conv_k.clone(), ssm_k.clone()
    sl = torch.arange(S, dtype=torch.int32, device=dev)

    def allreduce(x):
        # the kernel's peer reduce: bf16-rounded partials summed in rank order
        x = x.to(torch.bfloat16).float()
        parts = [torch.empty_like(x.cpu()) for _ in range(npes)]
        dist.all_gather(parts, x.cpu().contiguous())
        return sum(parts[1:], parts[0]).to(x.device)

    def same_everywhere(t):
        peers = [torch.empty_like(t.cpu()) for _ in range(npes)]
        dist.all_gather(peers, t.cpu().contiguous())
        return all(torch.equal(p, peers[0]) for p in peers[1:])

    layer = Qwen35MonoKernel(W, S, rank=rank, npes=npes)
    ok = True
    try:
        for step in range(iters):
            h = torch.randn(S, cfg.hidden, generator=gen, device=dev).to(torch.bfloat16)
            ref = golden_layer(W, h, allreduce, gdn_state=(conv_g, ssm_g, sl, sl))
            out = layer.forward(h, gdn_state=(conv_k, ssm_k, sl, sl))
            torch.cuda.synchronize()
            got = layer.intermediates()
            if not (same_everywhere(out) and same_everywhere(got["sel"])):
                print(f"rank {rank} step {step}: ranks disagree on x_out or routing", flush=True)
                ok = False
            errs = {"a": _rel(got["a"], ref["a"]), "ssm_state": _rel(ssm_k, ssm_g), "conv_state": _rel(conv_k, conv_g)}
            own = golden_moe(W, got["a"].to(torch.bfloat16), allreduce, xq=got["xq"])
            if torch.equal(got["sel"].long(), own["sel"].long()):
                errs["mid"] = _rel(got["mid"], own["mid"])
                errs["x_out_own"] = _rel(out, own["x_out"])
            if rank == 0:
                print(f"rank 0 step {step}", {k: f"{v:.2e}" for k, v in errs.items()}, flush=True)
            if any(v > 2e-2 for v in errs.values()):
                print(f"rank {rank} step {step}: {errs}", flush=True)
                ok = False
    finally:
        layer.close()
    return ok


def _free_port():
    """A fresh port: a fixed one collides with a socket still in TIME_WAIT on back-to-back runs."""
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, fn, npes, port, results, kw):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=npes)
    try:
        results[rank] = fn(rank, npes, **kw)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.parametrize("S", [1, 8])
def test_qwen35_gdn_layer_tp4(S):
    """The recipe's TP4: every rank's shard against the golden, cross-GPU reduces, identical outputs."""
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    import torch.multiprocessing as mp

    results = mp.Manager().dict()
    mp.spawn(_worker, args=(_run_rank_tp, 4, _free_port(), results, dict(S=S)), nprocs=4)
    assert all(results[r] for r in range(4))
