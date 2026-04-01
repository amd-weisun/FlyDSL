"""MoE Sorting kernel — correctness test + optional AITER (CK) benchmark.

Tests FlyDSL MoeSorting against a CPU reference and optionally benchmarks
against the AITER CK baseline (`aiter.moe_sorting_fwd`).

Matches: MoeSortingProblemEx<int, float, 1, true, false, false, true, 0>
  num_experts=128, topk=4, unit_size=32 (GPT-OSS 120B)

Usage:
    # Correctness only (no GPU needed for mock path):
    PYTHONPATH=./ pytest tests/kernels/test_moe_sorting.py -v -s

    # With benchmark + AITER (CK) comparison:
    FLYDSL_BENCH=1 AITER_REPO=../aiter PYTHONPATH=./ pytest tests/kernels/test_moe_sorting.py -v -s

    # CLI — all token sizes:
    FLYDSL_BENCH=1 AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/test_moe_sorting.py
"""

import os
import sys
import torch
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
for _p in [_REPO_ROOT, os.path.join(_REPO_ROOT, "build", "python_packages")]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from kernels.moe_sorting_kernel import (
    build_moe_sorting_module,
    compute_sub_tokens,
    compute_max_tokens_padded,
    compute_max_m_blocks,
)

try:
    from tests.kernels.benchmark_common import bench_gpu_us_torch, maybe_enable_aiter
    HAS_BENCH = True
except ImportError:
    try:
        from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter
        HAS_BENCH = True
    except ImportError:
        HAS_BENCH = False

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

# ── GPT-OSS 120B config ──────────────────────────────────────────────────────
NUM_EXPERTS = 128
TOPK = 4
UNIT_SIZE = 32   # M_a block size
MODEL_DIM = 2880  # hidden dim (for moe_buf sizing, must match fused-moe kernel)

# Cache compiled modules: key = (num_experts, topk)
_module_cache: dict = {}


def _get_launch_fn(num_experts=NUM_EXPERTS, topk=TOPK):
    key = (num_experts, topk)
    if key not in _module_cache:
        _module_cache[key] = build_moe_sorting_module(num_experts, topk, max_tokens=4096)
    return _module_cache[key]


# ── CPU reference ────────────────────────────────────────────────────────────

def _mock_id(token_id: int, topk_id: int) -> int:
    return int(((token_id & 0x00FFFFFF) | ((topk_id & 0xFF) << 24)))


def reference_moe_sorting(
    topk_ids: torch.Tensor,      # [tokens, topk] int32
    topk_weights: torch.Tensor,  # [tokens, topk] float32
    num_experts: int,
    unit_size: int,
) -> tuple:
    """CPU reference — matches CK reference_moe_sorting with MOCK_ID encoding
    and SkipExpertsWithZeroTokens=true."""
    tokens, topk = topk_ids.shape
    invalid = _mock_id(tokens, topk)

    expert_tokens = [[invalid] * unit_size for _ in range(num_experts)]
    expert_weights = [[0.0] * unit_size for _ in range(num_experts)]
    expert_slices = [1] * num_experts
    expert_idx = [0] * num_experts

    for t in range(tokens):
        for k in range(topk):
            e = int(topk_ids[t, k])
            w = float(topk_weights[t, k])
            i = expert_idx[e]
            if i > expert_slices[e] * unit_size - 1:
                expert_slices[e] += 1
                new_size = expert_slices[e] * unit_size
                expert_tokens[e].extend([invalid] * (new_size - len(expert_tokens[e])))
                expert_weights[e].extend([0.0] * (new_size - len(expert_weights[e])))
            expert_tokens[e][i] = _mock_id(t, k)
            expert_weights[e][i] = w
            expert_idx[e] += 1

    out_ids, out_wts, out_eids, unit_cnt = [], [], [], 0
    for e in range(num_experts):
        if expert_idx[e] == 0:   # SkipExpertsWithZeroTokens
            continue
        s = expert_slices[e]
        out_ids.extend(expert_tokens[e][: s * unit_size])
        out_wts.extend(expert_weights[e][: s * unit_size])
        for _ in range(s):
            out_eids.append(e)
            unit_cnt += 1

    total = unit_cnt * unit_size
    return (
        torch.tensor(out_ids, dtype=torch.int32),
        torch.tensor(out_wts, dtype=torch.float32),
        torch.tensor(out_eids, dtype=torch.int32),
        total,
    )


# ── Allocate output tensors matching AITER _moe_sorting_impl ────────────────

def _alloc_outputs(tokens: int, num_experts: int, topk: int, unit_size: int, model_dim: int):
    dev = torch.device("cuda")
    max_pad = compute_max_tokens_padded(tokens, num_experts, topk, unit_size)
    max_blk = compute_max_m_blocks(tokens, num_experts, topk, unit_size)
    sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
    sorted_wts = torch.empty(max_pad, dtype=torch.float32, device=dev)
    sorted_exp = torch.empty(max_blk, dtype=torch.int32, device=dev)
    num_valid = torch.empty(2, dtype=torch.int32, device=dev)
    # moe_buf: [tokens, model_dim] bf16 (flat int32 view for zeroing)
    moe_buf_2d = torch.ones(tokens, model_dim, dtype=torch.bfloat16, device=dev)
    return sorted_ids, sorted_wts, sorted_exp, num_valid, moe_buf_2d


# ── Core test function ───────────────────────────────────────────────────────

def run_moe_sorting_test(
    tokens: int,
    num_experts: int = NUM_EXPERTS,
    topk: int = TOPK,
    unit_size: int = UNIT_SIZE,
    model_dim: int = MODEL_DIM,
    seed: int = 42,
) -> dict:
    """Run FlyDSL MoeSorting, verify against CPU reference, optionally benchmark.

    Returns dict with correctness flag and timing info (if FLYDSL_BENCH=1).
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda")

    # Random routing: each token selects topk distinct experts (sorted)
    topk_ids_2d = torch.zeros(tokens, topk, dtype=torch.int32)
    for t in range(tokens):
        topk_ids_2d[t] = torch.randperm(num_experts)[:topk].sort().values
    topk_wts_2d = torch.rand(tokens, topk, dtype=torch.float32)

    # FlyDSL expects flat [tokens * topk] layout
    topk_ids_flat = topk_ids_2d.reshape(-1).contiguous().to(dev)
    topk_wts_flat = topk_wts_2d.reshape(-1).contiguous().to(dev)

    sorted_ids, sorted_wts, sorted_exp, num_valid, moe_buf_2d = _alloc_outputs(
        tokens, num_experts, topk, unit_size, model_dim
    )
    # Pass moe_buf as int32 view (flat)
    moe_buf_flat = moe_buf_2d.view(torch.int32).reshape(-1)
    moe_buf_elems = moe_buf_flat.numel()

    sub_tokens_val = compute_sub_tokens(tokens, num_experts)
    launch_fn = _get_launch_fn(num_experts, topk)

    def _run_flydsl():
        launch_fn(
            topk_ids_flat, topk_wts_flat,
            sorted_ids, sorted_wts, sorted_exp, num_valid,
            moe_buf_flat,
            tokens, unit_size, sub_tokens_val, moe_buf_elems,
        )

    _run_flydsl()
    torch.cuda.synchronize()

    # ── CPU reference ────────────────────────────────────────────────────────
    ref_ids, ref_wts, ref_eids, ref_total = reference_moe_sorting(
        topk_ids_2d, topk_wts_2d, num_experts, unit_size
    )

    # ── Verify total_tokens_post_pad ─────────────────────────────────────────
    gpu_total = int(num_valid[0].item())
    assert gpu_total == ref_total, (
        f"total_tokens_post_pad[0]: got {gpu_total}, expected {ref_total}"
    )
    assert int(num_valid[1].item()) == tokens, (
        f"total_tokens_post_pad[1]: got {num_valid[1]}, expected {tokens}"
    )

    # ── Verify sorted_expert_ids ─────────────────────────────────────────────
    n_blocks = ref_total // unit_size
    gpu_eids = sorted_exp[:n_blocks].cpu()
    assert torch.equal(gpu_eids, ref_eids), (
        f"sorted_expert_ids mismatch\n"
        f"  gpu={gpu_eids[:20].tolist()}\n  ref={ref_eids[:20].tolist()}"
    )

    # ── Verify sorted_token_ids / sorted_weights per expert ──────────────────
    # Token order within an expert can differ (thread scheduling), so compare sets.
    gpu_sids = sorted_ids[:ref_total].cpu()
    gpu_swts = sorted_wts[:ref_total].cpu()
    invalid = _mock_id(tokens, topk)

    # Reconstruct expert boundaries from ref_eids
    cur = 0
    expert_ranges = {}  # e -> (start, end)
    for blk, e in enumerate(ref_eids.tolist()):
        s = blk * unit_size
        if e not in expert_ranges:
            expert_ranges[e] = (s, s + unit_size)
        else:
            expert_ranges[e] = (expert_ranges[e][0], s + unit_size)

    for e, (es, ee) in expert_ranges.items():
        ref_slice = ref_ids[es:ee]
        gpu_slice = gpu_sids[es:ee]
        ref_wt = ref_wts[es:ee]
        gpu_wt = gpu_swts[es:ee]

        ref_real = ref_slice != invalid
        gpu_real = gpu_slice != invalid

        assert ref_real.sum() == gpu_real.sum(), (
            f"Expert {e}: real token count ref={ref_real.sum()} gpu={gpu_real.sum()}"
        )

        ref_set = set(zip(ref_slice[ref_real].tolist(), ref_wt[ref_real].tolist()))
        gpu_set = set(zip(gpu_slice[gpu_real].tolist(), gpu_wt[gpu_real].tolist()))
        assert ref_set == gpu_set, (
            f"Expert {e}: (token_id, weight) set mismatch\n"
            f"  ref={sorted(ref_set)[:4]}\n  gpu={sorted(gpu_set)[:4]}"
        )

        gpu_pad_ids = gpu_slice[~gpu_real]
        gpu_pad_wts = gpu_wt[~gpu_real]
        assert (gpu_pad_ids == invalid).all(), (
            f"Expert {e}: padding IDs not all invalid: {gpu_pad_ids.tolist()}"
        )
        assert (gpu_pad_wts == 0.0).all(), (
            f"Expert {e}: padding weights not all zero: {gpu_pad_wts.tolist()}"
        )

    # ── Verify moe_buf is zeroed ─────────────────────────────────────────────
    assert (moe_buf_2d.cpu() == 0).all(), "moe_buf not fully zeroed"

    result = {"ok": True, "total_pad": ref_total, "flydsl_us": None, "aiter_us": None}

    # ── Optional benchmark ───────────────────────────────────────────────────
    run_bench = HAS_BENCH and os.environ.get("FLYDSL_BENCH", "0") == "1"
    if run_bench:
        flydsl_us = bench_gpu_us_torch(_run_flydsl, warmup=20, iters=200)
        result["flydsl_us"] = flydsl_us
        print(f"  [flyc ] tokens={tokens:5d}  total_pad={ref_total:6d}  {flydsl_us:.1f} us")

        # ── AITER (CK) comparison ────────────────────────────────────────────
        if maybe_enable_aiter():
            try:
                import aiter

                # AITER moe_sorting_fwd expects [tokens, topk] tensors
                topk_ids_aiter = topk_ids_2d.to(dev)
                topk_wts_aiter = topk_wts_2d.to(dev)

                max_pad = compute_max_tokens_padded(tokens, num_experts, topk, unit_size)
                max_blk = compute_max_m_blocks(tokens, num_experts, topk, unit_size)
                a_sorted_ids = torch.empty(max_pad, dtype=torch.int32, device=dev)
                a_sorted_wts = torch.empty(max_pad, dtype=torch.float32, device=dev)
                a_sorted_exp = torch.empty(max_blk, dtype=torch.int32, device=dev)
                a_num_valid = torch.empty(2, dtype=torch.int32, device=dev)
                a_moe_buf = torch.empty(tokens, model_dim, dtype=torch.bfloat16, device=dev)

                def _run_aiter():
                    aiter.moe_sorting_fwd(
                        topk_ids_aiter,
                        topk_wts_aiter,
                        a_sorted_ids,
                        a_sorted_wts,
                        a_sorted_exp,
                        a_num_valid,
                        a_moe_buf,
                        num_experts,
                        unit_size,
                    )

                _run_aiter()  # warmup compile
                torch.cuda.synchronize()
                aiter_us = bench_gpu_us_torch(_run_aiter, warmup=20, iters=200)
                result["aiter_us"] = aiter_us
                speedup = aiter_us / flydsl_us if flydsl_us > 0 else 0
                print(
                    f"  [aiter] tokens={tokens:5d}  {aiter_us:.1f} us  "
                    f"→ FlyDSL/AITER speedup: {speedup:.2f}x"
                )
            except Exception as e:
                print(f"  [aiter] skipped: {e}")

    return result


# ── pytest parametrized tests ────────────────────────────────────────────────

@pytest.mark.parametrize("tokens", [1, 4, 16, 32, 64])
def test_moe_sorting_decode(tokens):
    r = run_moe_sorting_test(tokens)
    assert r["ok"], f"FAILED tokens={tokens}"


@pytest.mark.parametrize("tokens", [128, 256, 512, 1024, 2048])
def test_moe_sorting_prefill(tokens):
    r = run_moe_sorting_test(tokens)
    assert r["ok"], f"FAILED tokens={tokens}"


@pytest.mark.parametrize("tokens", [4096])
def test_moe_sorting_large(tokens):
    r = run_moe_sorting_test(tokens)
    assert r["ok"], f"FAILED tokens={tokens}"


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MoeSorting correctness + benchmark")
    parser.add_argument(
        "--tokens",
        type=str,
        default="1,4,16,32,64,128,256,512,1024,2048,4096",
        help="Comma-separated list of token counts to test",
    )
    args = parser.parse_args()
    token_list = [int(x) for x in args.tokens.split(",")]

    print(
        f"MoeSorting test  num_experts={NUM_EXPERTS}  topk={TOPK}  "
        f"unit_size={UNIT_SIZE}  model_dim={MODEL_DIM}"
    )
    run_bench = os.environ.get("FLYDSL_BENCH", "0") == "1"
    if run_bench:
        print("Benchmarking enabled (FLYDSL_BENCH=1)")
        if maybe_enable_aiter():
            print("AITER comparison enabled")
        else:
            print("AITER not available (set AITER_REPO=../aiter to enable)")
    print()

    passed = failed = 0
    rows = []
    for tokens in token_list:
        try:
            r = run_moe_sorting_test(tokens)
            status = "PASS"
            passed += 1
        except Exception as e:
            status = f"FAIL: {e}"
            failed += 1
            r = {"ok": False, "flydsl_us": None, "aiter_us": None, "total_pad": 0}
        if not run_bench:
            print(f"  tokens={tokens:5d}  total_pad={r.get('total_pad', '?'):6}  {status}")
        rows.append((tokens, r))

    if run_bench and rows:
        print("\n" + "=" * 70)
        print(f"{'tokens':>8s}  {'total_pad':>10s}  {'FlyDSL(us)':>12s}  {'AITER(us)':>12s}  {'speedup':>10s}")
        print("=" * 70)
        for tokens, r in rows:
            fu = r.get("flydsl_us")
            au = r.get("aiter_us")
            sp = f"{au/fu:.2f}x" if fu and au else "-"
            fu_s = f"{fu:.1f}" if fu else "-"
            au_s = f"{au:.1f}" if au else "-"
            print(f"  {tokens:6d}  {r.get('total_pad', 0):10d}  {fu_s:>12s}  {au_s:>12s}  {sp:>10s}")
        print("=" * 70)

    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
