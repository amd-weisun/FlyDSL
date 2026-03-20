#!/usr/bin/env python3
"""Fused RoPE + KV Cache benchmark: FlyDSL vs AITER Triton.

This is the kernel that takes 4.1% GPU time in GPT-OSS 120B (3.50 ms).
Apples-to-apples comparison: both do RoPE on Q/K + KV cache write in one call.

Usage (from FlyDSL/ directory):
    # FlyDSL vs AITER
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --sweep

    # FlyDSL only
    PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --flydsl-only --sweep
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_FLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_FLYDSL_SRC) and _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

import torch

from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter

# GPT-OSS 120B model config
HEAD_DIM = 64
TOTAL_Q_HEADS = 64
TOTAL_KV_HEADS = 8
BLOCK_SIZE = 16
MAX_POS = 8192

# Derived per-GPU heads (set by --tp flag)
NUM_Q_HEADS = 8   # default TP=8
NUM_KV_HEADS = 1  # default TP=8

SWEEP_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 1024]


@dataclass
class BenchRow:
    label: str
    tokens: int
    us: Optional[float]

    @property
    def bytes_total(self) -> int:
        eb = 2  # bf16
        # Read: Q + K + V + cos + sin + positions + slot_mapping
        read = self.tokens * (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM * eb  # Q + K
        read += self.tokens * NUM_KV_HEADS * HEAD_DIM * eb  # V
        read += self.tokens * (HEAD_DIM // 2) * 2 * eb  # cos + sin
        read += self.tokens * 4 * 2  # positions + slot_mapping (i32)
        # Write: Q_out + K_out + key_cache + value_cache
        write = self.tokens * (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM * eb  # Q_out + K_out
        write += self.tokens * NUM_KV_HEADS * HEAD_DIM * eb * 2  # key_cache + value_cache
        return read + write

    @property
    def bw_gbps(self) -> Optional[float]:
        if self.us is None:
            return None
        return self.bytes_total / 1e9 / (self.us / 1e6)


def _create_test_tensors(tokens, device):
    """Create test tensors matching GPT-OSS shapes."""
    num_blocks = max(32, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    q = torch.randn(tokens, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos_cache = torch.randn(MAX_POS, HEAD_DIM // 2, device=device, dtype=torch.bfloat16)
    sin_cache = torch.randn(MAX_POS, HEAD_DIM // 2, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    slot_mapping = torch.arange(tokens, device=device, dtype=torch.int32)
    # Flash layout: [num_blocks, block_size, KH, D]
    key_cache = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                             device=device, dtype=torch.bfloat16)
    value_cache = torch.zeros(num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM,
                               device=device, dtype=torch.bfloat16)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    return q, k, v, cos_cache, sin_cache, positions, slot_mapping, key_cache, value_cache, q_out, k_out


def bench_flydsl(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark FlyDSL fused RoPE + KV cache kernel (two launches)."""
    try:
        from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
    except ImportError:
        print("  [FlyDSL] Could not import fused_rope_cache_kernel")
        return None

    device = torch.device("cuda")
    launch_fn = build_fused_rope_cache_module(
        head_dim=HEAD_DIM, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE, is_neox=True, flash_layout=True, dtype_str="bf16",
    )

    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_test_tensors(tokens, device)
    stream = torch.cuda.current_stream()

    def run():
        launch_fn(q, k, v, pos, cos, sin, slots, kc, vc, qo, ko, tokens, stream=stream)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] Failed: {e}")
        return None


def bench_flydsl_single(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark FlyDSL fused RoPE + KV cache kernel (SINGLE launch, like Triton)."""
    try:
        from kernels.fused_rope_cache_single_kernel import build_fused_rope_cache_single_module
    except ImportError:
        print("  [FlyDSL-1K] Could not import fused_rope_cache_single_kernel")
        return None

    device = torch.device("cuda")
    launch_fn = build_fused_rope_cache_single_module(
        head_dim=HEAD_DIM, num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        block_size=BLOCK_SIZE, is_neox=True, flash_layout=True, dtype_str="bf16",
    )

    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_test_tensors(tokens, device)
    stream = torch.cuda.current_stream()
    n_q_progs = tokens * NUM_Q_HEADS
    n_total = n_q_progs + tokens * NUM_KV_HEADS
    nqp_tensor = torch.tensor([n_q_progs], device=device, dtype=torch.int32)

    def run():
        launch_fn(q, k, v, pos, cos, sin, slots, kc, vc, qo, ko,
                  nqp_tensor, n_total, stream=stream)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL-1K] Failed: {e}")
        return None


def bench_aiter(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark AITER Triton fused_qk_rope_reshape_and_cache."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
    except ImportError:
        try:
            from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
        except ImportError:
            print("  [AITER] Could not import fused_qk_rope_reshape_and_cache")
            return None

    device = torch.device("cuda")
    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_test_tensors(tokens, device)
    # AITER expects int64 positions
    pos_i64 = pos.to(torch.int64)
    k_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    v_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    def run():
        fused_qk_rope_reshape_and_cache(
            q, k, v, kc, vc, slots, pos_i64, cos, sin,
            k_scale, v_scale,
            is_neox=True, flash_layout=True,
            apply_scale=False, q_out=qo, k_out=ko, output_zeros=False,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [AITER] Failed: {e}")
        return None


def print_results(rows: List[BenchRow]) -> None:
    print()
    print("=" * 90)
    print(f"Fused RoPE+KVCache — GPT-OSS 120B (D={HEAD_DIM}, QH={NUM_Q_HEADS}, "
          f"KH={NUM_KV_HEADS}, BS={BLOCK_SIZE})")
    print("=" * 90)
    hdr = f"{'M(conc)':>7s}  {'Kernel':<16s}  {'Latency(us)':>12s}  {'BW(GB/s)':>10s}"
    print(hdr)
    print("-" * 90)

    by_tokens: dict[int, List[BenchRow]] = {}
    for r in rows:
        by_tokens.setdefault(r.tokens, []).append(r)

    for tokens in sorted(by_tokens.keys()):
        group = by_tokens[tokens]
        for i, r in enumerate(group):
            tok_col = f"{tokens:7d}" if i == 0 else f"{'':7s}"
            us_s = f"{r.us:12.1f}" if r.us else f"{'N/A':>12s}"
            bw_s = f"{r.bw_gbps:10.2f}" if r.bw_gbps else f"{'N/A':>10s}"
            print(f"{tok_col}  {r.label:<16s}  {us_s}  {bw_s}")

        aiter_row = next((r for r in group if r.label == "Triton" and r.us), None)
        fly2_row = next((r for r in group if r.label == "FlyDSL-2K" and r.us), None)
        fly1_row = next((r for r in group if r.label == "FlyDSL-1K" and r.us), None)
        if aiter_row and fly2_row:
            sp = aiter_row.us / fly2_row.us
            print(f"{'':7s}  {'2K/Triton':<16s}  {'':12s}  {sp:9.2f}x")
        if aiter_row and fly1_row:
            sp = aiter_row.us / fly1_row.us
            print(f"{'':7s}  {'1K/Triton':<16s}  {'':12s}  {sp:9.2f}x")
        print()

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Fused RoPE+KVCache: FlyDSL vs Triton")
    parser.add_argument("--concurrency", type=str, default="1,4,32,128",
                        help="Comma-separated M values")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true")
    parser.add_argument("--tp", type=str, default="8",
                        help="Tensor parallelism: 1,2,4,8 or 'all'")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    token_list = SWEEP_CONCURRENCY if args.sweep else [int(t) for t in args.concurrency.split(",")]

    if args.tp == "all":
        tp_list = [1, 2, 4, 8]
    else:
        tp_list = [int(t) for t in args.tp.split(",")]

    for tp in tp_list:
        global NUM_Q_HEADS, NUM_KV_HEADS
        NUM_Q_HEADS = TOTAL_Q_HEADS // tp
        NUM_KV_HEADS = TOTAL_KV_HEADS // tp

        print(f"\n{'='*90}")
        print(f"TP={tp} → QH={NUM_Q_HEADS}, KH={NUM_KV_HEADS}")
        print(f"{'='*90}")

        rows: List[BenchRow] = []
        for tokens in token_list:
            print(f"Benchmarking concurrency={tokens} ...")

            if not args.flydsl_only:
                aiter_us = bench_aiter(tokens, args.warmup, args.iters)
                rows.append(BenchRow(label="Triton", tokens=tokens, us=aiter_us))

            fly_us = bench_flydsl(tokens, args.warmup, args.iters)
            rows.append(BenchRow(label="FlyDSL-2K", tokens=tokens, us=fly_us))

            fly1_us = bench_flydsl_single(tokens, args.warmup, args.iters)
            rows.append(BenchRow(label="FlyDSL-1K", tokens=tokens, us=fly1_us))

        print_results(rows)


if __name__ == "__main__":
    main()
