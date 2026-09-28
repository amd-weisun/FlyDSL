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
