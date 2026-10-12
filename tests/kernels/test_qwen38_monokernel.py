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
from kernels.monokernel.qwen38.op import Qwen38MonoKernel
from kernels.monokernel.qwen38.reference import golden_layer, golden_moe, hc_combine, make_weights

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

if not str(get_rocm_arch() or "").startswith("gfx95"):
    pytest.skip("the MonoKernel's MFMA paths are written for gfx950", allow_module_level=True)

STEPS = 3


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


@pytest.mark.parametrize("S", [1, 2, 5, 8])
def test_qwen38_gdn_layer_matches_golden(S):
    cfg = Q38Config()
    W = make_weights(cfg, full=False)
    gen = torch.Generator(device="cuda").manual_seed(7)
    n_slots = 2 * S + 1
    conv_k = (0.5 * torch.randn(n_slots, cfg.conv_kernel - 1, cfg.conv_dim, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    ssm_k = 0.1 * torch.randn(n_slots, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim, generator=gen, device="cuda")
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
            own_out = hc_combine(got["x1"].to(torch.bfloat16), own["y_moe"], got["r2"], cfg.hc)
            errs["x_out_delta"] = _rel(out.float() - got["x1"], own_out.float() - got["x1"])
        print(f"S={S} step {step}", {k: f"{v:.2e}" for k, v in errs.items()})
        bad = {k: v for k, v in errs.items() if v > 2e-2}
        assert not bad, f"step {step}: {bad} (all: {errs})"
