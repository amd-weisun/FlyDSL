# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Does our V4 attention agree with ATOM's, given the same inputs?

ATOM (`/app/ATOM`, an AITER-based serving engine) implements DeepSeek V4 as a
chain of per-op kernels. If our fused layer is ever to replace that chain, the
two have to compute the same thing -- so this checks the piece where they share
a contract exactly, the gather-sparse attention: a query, a KV plane, a list of
absolute rows, and a per-head sink. No weight mapping is needed, which is what
makes this the cheapest real equivalence check available.

It compares against ``reference.sparse_attention`` -- the same function
``golden_layer`` calls, which the kernel tests already pin the kernel to within
0.002 -- so agreement here covers the kernel transitively.

Skipped unless ATOM is importable, so it runs where ATOM is installed and stays
out of the way elsewhere.
"""

from __future__ import annotations

import pytest
import torch

from kernels.dsv4_moe_layer.reference import sparse_attention

pytestmark = [
    pytest.mark.l2_device,
    pytest.mark.rocm_lower,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

H, D = 16, 512  # one TP8 rank's heads; K and V share head_dim
SCALE = D**-0.5
# bf16 rounding of the probabilities before P V differs slightly between the two
# implementations; 2.3e-3 is the worst seen across these cases, and the
# sensitivity test below shows a real disagreement lands orders above it.
TOL = 6e-3


def _atom_sparse_attn():
    atom = pytest.importorskip("atom", reason="ATOM is not installed here")
    from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode

    assert atom is not None
    return sparse_attn_v4_paged_decode


def _both(n_keys, rows, n_masked=0, sink_val=0.0, seed=0):
    """Run the same problem through ours and ATOM's; return (ours, theirs)."""
    run = _atom_sparse_attn()
    dev = "cuda"
    torch.manual_seed(seed)
    q = (0.3 * torch.randn(H, D, device=dev)).bfloat16()
    kv = (0.3 * torch.randn(rows, D, device=dev)).bfloat16()
    keys = torch.randperm(rows, device=dev)[:n_keys].to(torch.int32)
    if n_masked:  # slots the cache has not written yet
        keys[-n_masked:] = -1
    sink = torch.full((H,), sink_val, dtype=torch.float32, device=dev)
    ours = sparse_attention(q.float(), kv, keys, sink, SCALE)
    # ATOM has no -1 convention: it masks by OMISSION, so its gather list is the
    # valid rows. Same key set, expressed differently.
    valid = keys[keys >= 0].contiguous()
    ptr = torch.tensor([0, valid.numel()], dtype=torch.int32, device=dev)
    theirs = run(q.unsqueeze(0), kv, valid, ptr, sink, SCALE)[0]
    torch.cuda.synchronize()
    return ours.float(), theirs.float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize(
    "name,kw",
    [
        ("csa key list", dict(n_keys=1152, rows=8192)),
        ("window only", dict(n_keys=128, rows=4096)),
        ("long hca list", dict(n_keys=4096, rows=16384)),
        ("unwritten slots", dict(n_keys=1152, rows=8192, n_masked=200)),
        ("positive sink", dict(n_keys=1152, rows=8192, sink_val=2.0)),
        ("negative sink", dict(n_keys=1152, rows=8192, sink_val=-3.0)),
        # few keys is where the sink actually MOVES the denominator: at 1152 keys
        # exp(sink - M) is a rounding error next to the sum, so a sink bug hides.
        ("sink with few keys", dict(n_keys=8, rows=4096, sink_val=3.0)),
        ("another draw", dict(n_keys=1152, rows=8192, seed=7)),
    ],
)
def test_dsv4_attention_matches_atom(name, kw):
    """Same inputs, same answer -- masking, the sink, and the softmax scale."""
    ours, theirs = _both(**kw)
    rel = _rel(ours, theirs)
    assert rel < TOL, f"{name}: rel_l2 {rel:.3e} >= {TOL}"


def test_dsv4_attention_comparison_would_notice_a_difference():
    """The agreement above is only worth what this test says it is.

    Every case lands at ~2e-3, which could equally mean "both are roughly this
    shape". Perturb OUR side and require the error to move well clear of the
    bar; otherwise the comparison is not measuring agreement at all.
    """
    run = _atom_sparse_attn()
    dev = "cuda"
    torch.manual_seed(0)
    # few keys, big sink: the regime where every term below actually matters
    q = (0.3 * torch.randn(H, D, device=dev)).bfloat16()
    kv = (0.3 * torch.randn(4096, D, device=dev)).bfloat16()
    keys = torch.randperm(4096, device=dev)[:64].to(torch.int32)
    keys[-8:] = -1
    sink = torch.full((H,), 3.0, dtype=torch.float32, device=dev)
    valid = keys[keys >= 0].contiguous()
    ptr = torch.tensor([0, valid.numel()], dtype=torch.int32, device=dev)
    theirs = run(q.unsqueeze(0), kv, valid, ptr, sink, SCALE)[0].float()

    base = _rel(sparse_attention(q.float(), kv, keys, sink, SCALE), theirs)
    assert base < TOL, f"unperturbed should agree, got {base:.3e}"

    worse = {
        "sink removed": sparse_attention(q.float(), kv, keys, torch.full_like(sink, -60.0), SCALE),
        "scale doubled": sparse_attention(q.float(), kv, keys, sink, 2 * SCALE),
        "unwritten slots attended": sparse_attention(q.float(), kv, keys.clamp(min=0), sink, SCALE),
        "a key dropped": sparse_attention(q.float(), kv, keys[:-9], sink, SCALE),
    }
    for what, got in worse.items():
        r = _rel(got, theirs)
        assert r > 10 * base, f"{what} moved the error only {r / base:.1f}x ({r:.3e}); too blunt to trust"
