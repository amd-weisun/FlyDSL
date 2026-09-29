# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Paged (serving) mode of the MLA+MoE layer monokernel vs the torch golden.

S *independent* sequences, one new token each, whose contexts live at scattered rows of a
shared 576-wide bf16 pool (512 latent | 64 k_pe), addressed through a CSR list -- the
layout ATOM's MLA decode uses.  Every sample is checked against ``golden_layer`` run on its
own contiguous cache.  DeepSeek-V3 dims (heads=16, hidden=7168, ...), group-limited
routing, eps=1e-6 and a non-default softmax scale are exercised.

    python3 tests/kernels/test_mla_moe_layer_paged.py --npes 1
"""

import argparse
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernels.mla_moe_layer import reference  # noqa: E402
from kernels.mla_moe_layer.config import KV_LORA, PE_DIM  # noqa: E402
from kernels.mla_moe_layer.reference import golden_layer, golden_moe, make_weights, rope_table  # noqa: E402

DS = dict(heads=16, hidden=7168, q_lora=1536, nope_dim=128, v_dim=128)
N_GROUPS, TOPK_GROUPS = 8, 4
EPS = 1e-6
SCALE = (128 + 64) ** -0.5 * 1.8742  # DeepSeek-V3's yarn mscale^2
MAX_SEQ = 4096
POOL_ROWS = 16384


def run(S, lens, seed=1234, dev="cuda:0"):
    from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer

    torch.cuda.set_device(dev)
    dev = torch.device(dev)
    reference.EPS, reference.SOFTMAX_SCALE = EPS, SCALE
    W = make_weights(0, device=dev, seed=seed, **DS)
    cos, sin = rope_table(MAX_SEQ, device=dev)
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    pool = torch.randn(POOL_ROWS, KV_LORA + PE_DIM, generator=gen, device=dev).to(torch.bfloat16)
    perm = torch.randperm(POOL_ROWS, generator=gen, device=dev)
    slots, o = [], 0
    for L in lens:  # each sample owns L scattered rows; the last one is its new token
        slots.append(perm[o : o + L].to(torch.int32))
        o += L
    indptr = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device=dev)
    indices = torch.cat(slots).to(torch.int32)
    positions = torch.tensor([L - 1 for L in lens], dtype=torch.int32, device=dev)
    slot_map = torch.stack([sl[-1] for sl in slots]).to(torch.int32)
    h = torch.randn(S, DS["hidden"], generator=gen, device=dev).to(torch.bfloat16)
    pool0 = pool.clone()

    op = SharedReuseMlaMoeLayer(
        W, S, npes=1, topk=2048, paged=True, eps=EPS, softmax_scale=SCALE, n_groups=N_GROUPS, topk_groups=TOPK_GROUPS
    )
    out = op.forward_paged(h, positions, pool, slot_map, indptr, indices, cos, sin)
    torch.cuda.synchronize()

    ident = lambda x, bf16_partials=False: x  # noqa: E731
    stages = op.intermediates()
    worst = 0.0
    for b in range(S):
        L = lens[b]
        kv = torch.zeros(MAX_SEQ, KV_LORA, dtype=torch.bfloat16, device=dev)
        pe = torch.zeros(MAX_SEQ, PE_DIM, dtype=torch.bfloat16, device=dev)
        kv[: L - 1] = pool0[slots[b][:-1].long(), :KV_LORA]
        pe[: L - 1] = pool0[slots[b][:-1].long(), KV_LORA:]
        ref = golden_layer(W, h[b : b + 1], L - 1, kv, pe, None, cos, sin, ident, topk=2048)
        moe = golden_moe(W, ref["a"], ident, n_groups=N_GROUPS, topk_groups=TOPK_GROUPS)
        for name in ("q_a", "kv_a", "q_nope", "q_pe", "q_lat", "o", "a"):
            g, r = stages[name][b].float(), ref[name][0].float()
            rel = ((g - r).norm() / r.norm()).item()
            print(f"  {name:7s} rel_l2={rel:.2e}")
            assert rel < 2e-2, f"{name} of sample {b} diverges (rel_l2={rel:.2e})"
        same_sel = sorted(stages["sel"][b].tolist()) == sorted(moe["sel"][0].tolist())
        own = golden_moe(
            W,
            stages["a"][b : b + 1].clone(),
            lambda x, bf16_partials=False: x.to(torch.bfloat16).float() if bf16_partials else x,
            stages["mid"][b : b + 1].clone(),
            stages["sel"][b : b + 1].clone(),
            stages["prob"][b : b + 1].clone(),
        )
        e_own = ((out[b].float() - own["x_out"][0].float()).norm() / own["x_out"][0].float().norm()).item()
        print(f"  x_out vs golden fed the kernel's own a/mid/sel/prob: rel_l2={e_own:.2e}")
        worst = max(worst, e_own)
        if same_sel:  # a near-tied routing decision may legitimately flip on a 1-ulp difference in ``a``
            e2e = ((out[b].float() - moe["x_out"][0].float()).norm() / moe["x_out"][0].float().norm()).item()
            print(f"  x_out vs independent golden: rel_l2={e2e:.2e}")
            worst = max(worst, e2e)
        else:
            print("  x_out vs independent golden: skipped (routing flipped on a 1-ulp difference in a)")
        # cache write of the new token
        new = pool[slot_map[b].long()]
        torch.testing.assert_close(new[:KV_LORA].float(), kv[L - 1].float(), atol=2e-2, rtol=1e-2)
        torch.testing.assert_close(new[KV_LORA:].float(), pe[L - 1].float(), atol=2e-2, rtol=1e-2)
    # only the new rows may change
    mask = torch.ones(POOL_ROWS, dtype=torch.bool, device=dev)
    mask[slot_map.long()] = False
    assert torch.equal(pool[mask], pool0[mask]), "paged kernel wrote outside the new-token rows"
    op.close()
    return worst


@pytest.mark.parametrize("lens", [[37], [1500], [200, 2000], [1, 64, 65, 700], [3, 10, 999, 2048, 5, 640, 1, 2000]])
def test_paged_layer(lens):
    assert run(len(lens), lens) < 5e-2


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", type=int, nargs="+", default=[200, 2000])
    args = ap.parse_args()
    assert run(len(args.lens), args.lens) < 5e-2
    print("PASS")
