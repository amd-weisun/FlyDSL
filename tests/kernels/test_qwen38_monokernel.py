# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the Qwen3.8-Flash-Next layer kernel against its torch golden, stage by stage.

Real TP1 dims (hidden 2560, 4 streams, 48 v-heads, 512 experts x 640), several decode steps so the state the
kernel writes (conv, fp32 recurrent) feeds its next step.
"""

from __future__ import annotations

import pytest
import torch

from flydsl.runtime.device import get_rocm_arch
from kernels.monokernel.qwen38.config import Q38Config
from kernels.monokernel.qwen38.op import Qwen38MonoKernel, rope_tables
from kernels.monokernel.qwen38.reference import (
    apply_rope,
    golden_layer,
    golden_moe,
    hc_combine,
    make_weights,
    rmsnorm1p,
    rope_cos_sin,
)

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not str(get_rocm_arch() or "").startswith("gfx95"):
    pytest.skip("the MonoKernel's MFMA paths are written for gfx950", allow_module_level=True)

STEPS = 3


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _combine_err(out, x1, y, r, hc):
    """The bundle increment's error, x_out - x1 against the combine of the golden MoE output, in units of the
    floor bf16 storage of x_out alone puts on it (the bundle is rounded at its own magnitude): <= 1 is exact."""
    S = x1.shape[0]
    inj = 2.0 * torch.sigmoid((r.float() / hc).to(torch.bfloat16).float())
    exact = (x1.float().view(S, hc, -1) + y.float()[:, None, :] * inj[:, :, None]).view(S, -1)
    inc = exact - x1.float()
    floor = float((exact.to(torch.bfloat16).float() - exact).norm() / inc.norm())
    err = float((out.float() - exact).norm() / inc.norm())
    return err / max(floor, 1e-6)


@pytest.mark.parametrize("S", [1, 2, 5, 8])
def test_qwen38_gdn_layer_matches_golden(S):
    cfg = Q38Config()
    W = make_weights(cfg, full=False)
    gen = torch.Generator(device="cuda").manual_seed(7)
    n_slots = 2 * S + 1
    conv_k = (0.5 * torch.randn(n_slots, cfg.conv_kernel - 1, cfg.conv_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    ssm_k = 0.1 * torch.randn(
        n_slots, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim, generator=gen, device="cuda"
    )
    conv_g, ssm_g = conv_k.clone(), ssm_k.clone()
    slots_in = torch.arange(S, dtype=torch.int32, device="cuda") * 2 + 1
    slots_out = slots_in + 1
    layer = Qwen38MonoKernel(W, S)
    for step in range(STEPS):
        X = torch.randn(S, cfg.streams, generator=gen, device="cuda").to(torch.bfloat16)
        ref = golden_layer(W, X, gdn_state=(conv_g, ssm_g, slots_in, slots_out))
        out = layer.forward(X, gdn_state=(conv_k, ssm_k, slots_in, slots_out))
        torch.cuda.synchronize()
        got = layer.intermediates()
        errs = {
            "m": _rel(got["m"], ref["m"]),
            "r": _rel(got["r"], ref["r"]),
            "conv": _rel(got["conv"], ref["conv_out"]),
            "z": _rel(got["z"], ref["z"].flatten(1)),
            "gated": _rel(got["gated"], ref["gated"].flatten(1)),
            "x1_delta": _rel(got["x1"] - X.float(), ref["x1"] - X.float()),
            "m2": _rel(got["m2"], ref["m2"]),
            "logits": _rel(got["scores"][:, : cfg.n_experts], ref["logits"]),
            "conv_state": _rel(conv_k[slots_out.long()], conv_g[slots_out.long()]),
            "ssm_state": _rel(ssm_k[slots_out.long()], ssm_g[slots_out.long()]),
        }
        # the MoE from the kernel's own input (FP8 midpoints and expert near-ties differ otherwise)
        own = golden_moe(W, got["m2"].to(torch.bfloat16), xq=got["xq"])
        if torch.equal(got["sel"].long().sort(1).values, own["sel"].long().sort(1).values):
            errs["xq"] = _rel(got["xq"], own["xq"])
            errs["mid"] = _rel(got["mid"], own["mid"])
            errs["mid_sh"] = _rel(got["mid_sh"], own["mid_sh"])
            # in units of the bf16 storage floor (<= 1: the increment is exact to storage); budget 2
            errs["x_out_vs_floor"] = 1e-2 * _combine_err(out, got["x1"], own["y_moe"], got["r2"], cfg.hc)
        print(f"S={S} step {step}", {k: f"{v:.2e}" for k, v in errs.items()})
        bad = {k: v for k, v in errs.items() if v > 2e-2}
        assert not bad, f"step {step}: {bad} (all: {errs})"


# per-sample first positions: dense (<= 2051 tokens), the budget edge, sparse (top 512 of > 512 groups)
QSA_POS = {1: [2900], 2: [37, 2500], 5: [3, 100, 2046, 2051, 2900], 8: [5, 15, 16, 63, 600, 2047, 2700, 3100]}
MAX_SEQ = 4096


@pytest.mark.parametrize("S", [1, 2, 5, 8])
def test_qwen38_qsa_layer_matches_golden(S):
    cfg = Q38Config()
    W = make_weights(cfg, full=True)
    t = W.t
    gen = torch.Generator(device="cuda").manual_seed(13)
    pos0 = QSA_POS[S]
    nb = MAX_SEQ // 16
    bt = torch.randperm(S * nb, generator=torch.Generator().manual_seed(3)).view(S, nb).to(torch.int32).cuda()
    bfl = torch.bfloat16
    kc = torch.zeros(S * nb, 16, cfg.kv_heads, cfg.head_dim, dtype=bfl, device="cuda")
    vc = torch.zeros_like(kc)
    ic = torch.zeros(S * nb, 16, 1, cfg.index_dim, dtype=bfl, device="cuda")
    cc = torch.zeros(S * nb, 4, 1, cfg.index_dim, dtype=bfl, device="cuda")
    # the golden's contiguous histories: random K / V / raw index keys, compressed keys as the model forms them
    kg = torch.zeros(S, MAX_SEQ, cfg.kv_heads, cfg.head_dim, device="cuda")
    vg, ig = torch.zeros_like(kg), torch.zeros(S, MAX_SEQ, cfg.index_dim, device="cuda")
    cg = torch.zeros(S, MAX_SEQ // 4, cfg.index_dim, device="cuda")
    for s in range(S):
        p = pos0[s]
        kg[s, :p] = torch.randn(p, cfg.kv_heads, cfg.head_dim, generator=gen, device="cuda").to(bfl).float()
        vg[s, :p] = torch.randn(p, cfg.kv_heads, cfg.head_dim, generator=gen, device="cuda").to(bfl).float()
        ig[s, :p] = torch.randn(p, cfg.index_dim, generator=gen, device="cuda").to(bfl).float()
        ng = p // 4
        if ng:
            pk = ig[s, : ng * 4].view(ng, 4, -1).mean(1).to(bfl).float()
            c0, s0 = rope_cos_sin(torch.arange(ng, device="cuda") * 4, cfg)
            cg[s, :ng] = apply_rope(rmsnorm1p(pk, t["g_ik"], cfg.eps).to(bfl).float()[:, None], c0, s0)[:, 0]
        tok = torch.arange(p, device="cuda")
        blk, off = bt[s, tok // 16].long(), tok % 16
        kc[blk, off], vc[blk, off], ic[blk, off, 0] = kg[s, :p].to(bfl), vg[s, :p].to(bfl), ig[s, :p].to(bfl)
        grp = torch.arange(ng, device="cuda")
        cc[bt[s, grp // 4].long(), grp % 4, 0] = cg[s, :ng].to(bfl)
    cos, sin = rope_tables(cfg, MAX_SEQ)
    layer = Qwen38MonoKernel(W, S, max_seq=MAX_SEQ)
    for step in range(STEPS):
        pos = torch.tensor([p + step for p in pos0], dtype=torch.int32, device="cuda")
        X = torch.randn(S, cfg.streams, generator=gen, device="cuda").to(torch.bfloat16)
        ref = golden_layer(W, X, qsa_state=(kg, vg, ig, cg, pos))
        slots = (bt[torch.arange(S), pos.long() // 16] * 16 + pos % 16).to(torch.int32)
        out = layer.forward(X, kv=(kc, vc, ic, cc, pos, bt, slots), rope=(cos, sin))
        torch.cuda.synchronize()
        got = layer.intermediates()
        new_k = torch.stack([kc[bt[s, int(pos[s]) // 16], int(pos[s]) % 16] for s in range(S)]).float()
        errs = {
            "m": _rel(got["m"], ref["m"]),
            "q": _rel(got["q"], ref["q"]),
            "iq": _rel(got["iq"], ref["iq"]),
            "k_cache": _rel(new_k, ref["k"]),
        }
        # Past the budget the selection depends on near-tied scores (the kernel's index query is ~3e-3 off the
        # golden's): check it against scores from the kernel's own query, and the attention on its own selection.
        gated_own = torch.zeros(S, cfg.heads, cfg.head_dim, device="cuda")
        kv_of = torch.arange(cfg.heads, device="cuda") // (cfg.heads // cfg.kv_heads)
        for s in range(S):
            p = int(pos[s])
            ng = (p + 1) // 4
            if (p + 1) % 4 == 0:
                errs[f"ck{s}"] = _rel(cc[bt[s, (ng - 1) // 4], (ng - 1) % 4, 0], cg[s, ng - 1])
            n_sel = int(got["gcnt"][s])
            assert n_sel == min(ng, cfg.block_topk), f"sample {s}: {n_sel} groups selected of {ng}"
            sel = got["gsel"][s, :n_sel].long()
            assert len(set(sel.tolist())) == n_sel and int(sel.min()) >= 0 and int(sel.max()) < ng
            if ng > cfg.block_topk:
                sc = torch.relu(got["iq"][s] @ cg[s, :ng].T).sum(0) / cfg.index_dim**0.5
                out_ = torch.ones(ng, dtype=torch.bool, device="cuda")
                out_[sel] = False
                tol = 1e-4 * float(sc.abs().max())
                assert float(sc[sel].min()) >= float(sc[out_].max()) - tol, f"sample {s}: not a top-{n_sel}"
            toks = torch.cat(
                [
                    (sel[:, None] * 4 + torch.arange(4, device="cuda")).flatten(),
                    torch.arange(ng * 4, p + 1, device="cuda"),
                ]
            )
            lg = torch.einsum("hd,thd->ht", ref["q"][s], kg[s, toks][:, kv_of]) * cfg.head_dim**-0.5
            o_ = torch.einsum("ht,thd->hd", torch.softmax(lg, -1), vg[s, toks][:, kv_of])
            gated_own[s] = (o_.to(bfl).float() * torch.sigmoid(ref["gate"][s]).to(bfl).float()).to(bfl).float()
        errs["gated_own_sel"] = _rel(got["gated"], gated_own.flatten(1))
        y_own = (got["gated"].to(bfl).float() @ t["w_o"].float().T).to(bfl)
        x1_own = hc_combine(X, y_own, got["r"], cfg.hc)
        errs["x1_delta"] = _rel(got["x1"] - X.float(), x1_own.float() - X.float())
        own = golden_moe(W, got["m2"].to(torch.bfloat16), xq=got["xq"])
        if torch.equal(got["sel"].long().sort(1).values, own["sel"].long().sort(1).values):
            # in units of the bf16 storage floor (<= 1: the increment is exact to storage); budget 2
            errs["x_out_vs_floor"] = 1e-2 * _combine_err(out, got["x1"], own["y_moe"], got["r2"], cfg.hc)
        print(f"QSA S={S} step {step} pos {pos.tolist()}", {k: f"{v:.2e}" for k, v in errs.items()})
        bad = {k: v for k, v in errs.items() if v > 2e-2}
        assert not bad, f"step {step}: {bad} (all: {errs})"
