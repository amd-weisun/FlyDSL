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
    ssm = (0.1 * torch.randn(n_slots, cfg.hv, cfg.linear_k_dim, cfg.linear_v_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    return conv, ssm


# 5 and 7: an engine captures every batch size, and an odd S leaves a router task a sample short
# 12 and 16: past the 8 waves (two samples a wave in the router of the MoE, chunked staging)
@pytest.mark.parametrize("S", [1, 2, 5, 7, 8, 12, 16])
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
FULL_POS = {
    1: [1000],
    2: [37, 300],
    5: [3, 100, 129, 640, 1500],
    8: [5, 15, 16, 63, 64, 200, 777, 2049],
    16: [5, 15, 16, 63, 64, 200, 777, 2049, 1, 31, 32, 127, 128, 300, 1023, 1500],
}


@pytest.mark.parametrize("S", [1, 2, 5, 8, 16])
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
            slots = (bt[torch.arange(S), pos.long() // 16] * 16 + pos % 16).to(torch.int32)
            out = layer.forward(h, kv=(kc, vc, ks, vs, pos, bt, slots), rope=(cos, sin))
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


# MTP verify: T tokens per sequence in one launch == T sequential decode steps of the golden
MTP_CASES = [(2, 1), (2, 4), (3, 2), (4, 2), (4, 4)]  # (tokens per sequence, sequences)


@pytest.mark.parametrize("T,Q", MTP_CASES)
def test_qwen35_gdn_mtp_verify_matches_sequential_golden(T, Q):
    cfg = Q35Config(tp=4, rank=0)
    W = make_weights(cfg, full=False)
    gen = torch.Generator(device="cuda").manual_seed(17)
    S = Q * T
    n_slots = S + 3
    # per sequence T state slots (ATOM's spec_state_indices), shuffled; the accepted counts of the last step
    spec = (torch.randperm(n_slots - 3, generator=torch.Generator().manual_seed(5))[:S] + 2).view(Q, T)
    n_acc = torch.tensor([1 + (3 * q + 1) % T for q in range(Q)], dtype=torch.int32)
    conv_k = (0.5 * torch.randn(n_slots, T + 2, cfg.conv_dim, generator=gen, device="cuda")).to(torch.bfloat16)
    ssm_k = (0.1 * torch.randn(n_slots, cfg.hv, cfg.linear_k_dim, cfg.linear_v_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    # the golden's per-sequence state: the window from n_accepted - 1 and the state after the last accepted token
    win0 = torch.stack([conv_k[spec[q, 0], int(n_acc[q]) - 1 : int(n_acc[q]) + 2] for q in range(Q)])
    conv_g = win0.clone()
    ssm_g = torch.stack([ssm_k[spec[q, int(n_acc[q]) - 1]] for q in range(Q)]).clone()
    sl = torch.arange(Q, dtype=torch.int32, device="cuda")
    old_window = conv_k[spec[:, 0]].clone()
    h = torch.randn(S, cfg.hidden, generator=gen, device="cuda").to(torch.bfloat16)
    layer = Qwen35MonoKernel(W, S, tokens_per_seq=T)
    try:
        out = layer.forward(
            h, gdn_state=(conv_k, ssm_k, n_acc.cuda(), spec.flatten().to(torch.int32).cuda().contiguous())
        )
        torch.cuda.synchronize()
        got = layer.intermediates()
        rows = torch.arange(Q) * T
        for j in range(T):
            ref = golden_layer(W, h[rows + j], lambda y: y, gdn_state=(conv_g, ssm_g, sl, sl))
            errs = {
                "conv": _rel(got["conv"][rows + j], ref["conv_out"]),
                "gated": _rel(got["gated"][rows + j], ref["gated"].flatten(1)),
                "a": _rel(got["a"][rows + j], ref["a"]),
                "ssm_snapshot": _rel(ssm_k[spec[:, j]], ssm_g),
            }
            if torch.equal(got["sel"][rows + j].long(), ref["sel"].long()):
                errs["x_out"] = _rel(out[rows + j], ref["x_out"])
            print(f"MTP T={T} Q={Q} token {j}", {k: f"{v:.2e}" for k, v in errs.items()})
            bad = {k: v for k, v in errs.items() if v > 2e-2}
            assert not bad, f"token {j}: {bad} (all: {errs})"
        # the window shifted by one: two kept inputs from n_accepted, then this launch's T inputs
        new_window = conv_k[spec[:, 0]]
        for q in range(Q):
            na = int(n_acc[q])
            assert torch.equal(new_window[q, :2], old_window[q, na : na + 2])
        assert _rel(new_window[:, T - 1 :], conv_g) < 1e-2, "the last three inputs are the golden's conv state"
    finally:
        layer.close()


# ReplaySSM verify (ATOM's default GDN state under speculative decoding): a checkpoint per request plus
# CAP records (k, u, g) a v-head; (tokens per sequence, sequences, write_pos): no fold (h + 2 T <= CAP), a
# fold (the checkpoint absorbs the h records, this step writes from 0), the prefill sentinel -1
REPLAY_CAP = 16
REPLAY_CASES = [(2, 1, 5), (4, 1, 3), (4, 2, 9), (4, 2, -1), (2, 4, 13), (4, 4, 9)]


def _rebuild(ck, k, u, g, n):
    """A v-head stack's state from a checkpoint [hv, K, V] and the first ``n`` records (ATOM's replay)."""
    S = ck.float()
    if n == 0:
        return S
    gg = g[:, :n].float()
    c = gg.cumsum(1)
    ctot = c[:, -1:]
    w = torch.exp(ctot - c)
    S = S * torch.exp(ctot)[:, :, None]
    return S + torch.einsum("hjk,hjv->hkv", k[:, :n].float() * w[:, :, None], u[:, :n].float())


@pytest.mark.parametrize("T,Q,wp", REPLAY_CASES)
def test_qwen35_gdn_replay_verify_matches_sequential_golden(T, Q, wp):
    from kernels.monokernel.qwen35.op import Qwen35Variant

    cfg = Q35Config(tp=4, rank=0)
    W = make_weights(cfg, full=False)
    gen = torch.Generator(device="cuda").manual_seed(23)
    S, cap, hv = Q * T, REPLAY_CAP, cfg.hv
    n_slots = Q + 3
    slot = (torch.randperm(n_slots, generator=torch.Generator().manual_seed(7))[:Q]).to(torch.int32)
    n_acc = torch.tensor([1 + (3 * q + 1) % T for q in range(Q)], dtype=torch.int32)
    conv_k = (0.5 * torch.randn(n_slots, T + 2, cfg.conv_dim, generator=gen, device="cuda")).to(torch.bfloat16)
    ckpt = (0.1 * torch.randn(n_slots, hv, cfg.linear_k_dim, cfg.linear_v_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    rk = torch.randn(n_slots, hv, cap, cfg.linear_k_dim, generator=gen, device="cuda")
    rk = (rk * torch.rsqrt(rk.square().sum(-1, keepdim=True))).to(torch.bfloat16)
    ru = (0.1 * torch.randn(n_slots, hv, cap, cfg.linear_v_dim, generator=gen, device="cuda")).to(torch.bfloat16)
    rg = -0.05 - 0.4 * torch.rand(n_slots, hv, cap, generator=gen, device="cuda")
    wpos = torch.zeros(n_slots, dtype=torch.int32, device="cuda")
    wpos[slot.long()] = wp
    hp = max(wp, 0)
    flush = hp + 2 * T > cap
    base = 0 if flush else hp
    ck0 = ckpt.clone()
    # the golden's per-sequence state: the window from n_accepted - 1 and the replayed state S_h (f32)
    conv_g = torch.stack([conv_k[slot[q], int(n_acc[q]) - 1 : int(n_acc[q]) + 2] for q in range(Q)]).clone()
    ssm_g = torch.stack([_rebuild(ckpt[slot[q]], rk[slot[q]], ru[slot[q]], rg[slot[q]], hp) for q in range(Q)])
    s_h = ssm_g.clone()
    sl = torch.arange(Q, dtype=torch.int32, device="cuda")
    old_window = conv_k[slot.long()].clone()
    h = torch.randn(S, cfg.hidden, generator=gen, device="cuda").to(torch.bfloat16)
    variant = Qwen35Variant(cfg, S, False, tokens_per_seq=T, replay_cap=cap)
    layer = Qwen35MonoKernel(W, S, variant=variant, tokens_per_seq=T)
    conv_rows = slot.repeat_interleave(T).cuda().contiguous()
    try:
        out = layer.forward(
            h,
            gdn_state=(conv_k, ckpt, n_acc.cuda(), conv_rows, slot.cuda(), wpos, rk, ru, rg),
        )
        torch.cuda.synchronize()
        got = layer.intermediates()
        rows = torch.arange(Q) * T
        for j in range(T):
            ref = golden_layer(W, h[rows + j], lambda y: y, gdn_state=(conv_g, ssm_g, sl, sl))
            errs = {
                "conv": _rel(got["conv"][rows + j], ref["conv_out"]),
                "gated": _rel(got["gated"][rows + j], ref["gated"].flatten(1)),
                "a": _rel(got["a"][rows + j], ref["a"]),
            }
            if torch.equal(got["sel"][rows + j].long(), ref["sel"].long()):
                errs["x_out"] = _rel(out[rows + j], ref["x_out"])
            print(f"replay T={T} Q={Q} wp={wp} token {j}", {k: f"{v:.2e}" for k, v in errs.items()})
            bad = {k: v for k, v in errs.items() if not v <= 2e-2}
            assert not bad, f"token {j}: {bad} (all: {errs})"
        assert torch.equal(wpos[slot.long()], torch.full((Q,), wp, dtype=torch.int32, device="cuda")), "cursor moved"
        for q in range(Q):
            s_ = int(slot[q])
            if flush:  # the checkpoint absorbed the h records
                e = _rel(ckpt[s_], s_h[q])
                assert e < 1e-2, f"sequence {q}: folded checkpoint {e:.2e}"
            else:
                assert torch.equal(ckpt[s_], ck0[s_]), f"sequence {q}: checkpoint written without a fold"
            # the checkpoint and the records through this step's replay to the golden's final state
            e = _rel(_rebuild(ckpt[s_], rk[s_], ru[s_], rg[s_], base + T), ssm_g[q])
            print(f"replay T={T} Q={Q} wp={wp} sequence {q}: replayed final state {e:.2e}")
            assert e < 1e-2, f"sequence {q}: replayed final state {e:.2e}"
        new_window = conv_k[slot.long()]
        for q in range(Q):
            na = int(n_acc[q])
            assert torch.equal(new_window[q, :2], old_window[q, na : na + 2])
    finally:
        layer.close()


MTP_POS = {1: [700], 2: [15, 300], 4: [62, 128, 1000, 33]}  # (4, 4): 16 rows


@pytest.mark.parametrize("T,Q", MTP_CASES)
def test_qwen35_full_attention_mtp_verify_matches_sequential_golden(T, Q):
    cfg = Q35Config(tp=4, rank=0)
    W = make_weights(cfg, full=True)
    gen = torch.Generator(device="cuda").manual_seed(19)
    S = Q * T
    pos0 = MTP_POS[Q]
    max_seq = max(pos0) + T + 1
    max_blocks = -(-max_seq // 16)
    perm = torch.randperm(Q * max_blocks, generator=torch.Generator().manual_seed(3)).view(Q, max_blocks)
    bt = perm.to(torch.int32).cuda().contiguous()
    nb = Q * max_blocks
    kc = torch.zeros(nb, 1, cfg.head_dim // 16, 16, 16, dtype=torch.float8_e4m3fn, device="cuda")
    vc = torch.zeros(nb, 1, 1, cfg.head_dim, 16, dtype=torch.float8_e4m3fn, device="cuda")
    ks = torch.zeros(nb, 1, 16, device="cuda")
    vs = torch.zeros(nb, 1, 16, device="cuda")
    k_g = torch.zeros(Q, max_seq, 1, cfg.head_dim, device="cuda")
    v_g = torch.zeros_like(k_g)
    for q in range(Q):
        hist_k = torch.randn(pos0[q], 1, cfg.head_dim, generator=gen, device="cuda")
        hist_v = torch.randn(pos0[q], 1, cfg.head_dim, generator=gen, device="cuda")
        k_g[q, : pos0[q]] = kv_quant_dequant(hist_k)
        v_g[q, : pos0[q]] = kv_quant_dequant(hist_v)
        for p in range(pos0[q]):
            paged_kv_write(kc, vc, ks, vs, bt[q].tolist(), p, hist_k[p], hist_v[p])
    cos, sin = rope_tables(cfg, max_seq)
    h = torch.randn(S, cfg.hidden, generator=gen, device="cuda").to(torch.bfloat16)
    seq = torch.arange(S, device="cuda") // T
    pos = (torch.tensor(pos0, device="cuda")[seq] + torch.arange(S, device="cuda") % T).to(torch.int32)
    bt_rows = bt[seq].contiguous()
    slots = (bt_rows[torch.arange(S), pos.long() // 16] * 16 + pos % 16).to(torch.int32)
    layer = Qwen35MonoKernel(W, S, tokens_per_seq=T)
    try:
        out = layer.forward(h, kv=(kc, vc, ks, vs, pos, bt_rows, slots), rope=(cos, sin))
        torch.cuda.synchronize()
        got = layer.intermediates()
        rows = torch.arange(Q, device="cuda") * T
        for j in range(T):
            ref = golden_layer(W, h[rows + j], lambda y: y, kv=(k_g, v_g, pos[rows + j]))
            errs = {
                "q": _rel(got["q"][rows + j], ref["q"]),
                "gated": _rel(got["gated"][rows + j], ref["gated"].flatten(1)),
                "a": _rel(got["a"][rows + j], ref["a"]),
            }
            if torch.equal(got["sel"][rows + j].long(), ref["sel"].long()):
                errs["x_out"] = _rel(out[rows + j], ref["x_out"])
            print(f"full MTP T={T} Q={Q} token {j}", {k: f"{v:.2e}" for k, v in errs.items()})
            bad = {k: v for k, v in errs.items() if v > 2e-2}
            assert not bad, f"token {j}: {bad} (all: {errs})"
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


def test_qwen35_full_attention_padding_row_writes_nothing():
    """A CUDA-graph padding row (ATOM: slot_mapping -1, a placeholder block table) must not write the
    cache, whatever its block table points at; the live row is unaffected."""
    cfg = Q35Config(tp=4, rank=0)
    W = make_weights(cfg, full=True)
    gen = torch.Generator(device="cuda").manual_seed(21)
    S, pos0 = 2, 50
    bt = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32, device="cuda")  # row 1 aliases row 0
    kc = torch.zeros(4, 1, cfg.head_dim // 16, 16, 16, dtype=torch.float8_e4m3fn, device="cuda")
    vc = torch.zeros(4, 1, 1, cfg.head_dim, 16, dtype=torch.float8_e4m3fn, device="cuda")
    ks, vs = torch.zeros(4, 1, 16, device="cuda"), torch.zeros(4, 1, 16, device="cuda")
    k_g = torch.zeros(1, 64, 1, cfg.head_dim, device="cuda")
    v_g = torch.zeros_like(k_g)
    hist_k = torch.randn(pos0, 1, cfg.head_dim, generator=gen, device="cuda")
    hist_v = torch.randn(pos0, 1, cfg.head_dim, generator=gen, device="cuda")
    k_g[0, :pos0], v_g[0, :pos0] = kv_quant_dequant(hist_k), kv_quant_dequant(hist_v)
    for p in range(pos0):
        paged_kv_write(kc, vc, ks, vs, bt[0].tolist(), p, hist_k[p], hist_v[p])
    before = (kc.clone(), vc.clone(), ks.clone(), vs.clone())
    pos = torch.tensor([pos0, 7], dtype=torch.int32, device="cuda")  # the padding row points into live rows
    slots = torch.tensor([pos0, -1], dtype=torch.int32, device="cuda")
    h = torch.randn(S, cfg.hidden, generator=gen, device="cuda").to(torch.bfloat16)
    cos, sin = rope_tables(cfg, 64)
    layer = Qwen35MonoKernel(W, S)
    try:
        layer.forward(h, kv=(kc, vc, ks, vs, pos, bt, slots), rope=(cos, sin))
        torch.cuda.synchronize()
        got = layer.intermediates()
    finally:
        layer.close()
    ref = golden_layer(W, h[:1], lambda y: y, kv=(k_g, v_g, pos[:1]))
    # only the live row's slot changed
    blk, t = pos0 // 16, pos0 % 16
    for new, old in zip((kc, vc, ks, vs), before):
        mask = torch.ones_like(new, dtype=torch.bool)
        if new.dim() == 5 and new.shape[2] == cfg.head_dim // 16:
            mask[blk, :, :, t, :] = False
        elif new.dim() == 5:
            mask[blk, :, 0, :, t] = False
        else:
            mask[blk, :, t] = False
        assert torch.equal(
            new[mask].view(torch.uint8) if new.dtype == torch.float8_e4m3fn else new[mask],
            old[mask].view(torch.uint8) if old.dtype == torch.float8_e4m3fn else old[mask],
        )
    assert _rel(got["gated"][:1], ref["gated"].flatten(1)) < 2e-2
