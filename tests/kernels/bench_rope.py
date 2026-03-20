#!/usr/bin/env python3
"""RoPE kernel benchmark: FlyDSL vs AITER Triton baseline.

Usage (from FlyDSL/ directory):
    # FlyDSL only
    PYTHONPATH=./ python tests/kernels/bench_rope.py --flydsl-only

    # FlyDSL vs AITER
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_rope.py

    # Sweep all concurrencies
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_rope.py --sweep
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

# GPT-OSS 120B config (TP=8)
HEAD_DIM = 64
ROTARY_DIM = 64
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
MAX_POS = 8192

SWEEP_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128]


@dataclass
class RopeBenchRow:
    label: str
    tokens: int
    us: Optional[float]

    @property
    def bytes_total(self) -> int:
        elem_bytes = 2  # bf16
        read = self.tokens * (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM * elem_bytes
        read += self.tokens * (ROTARY_DIM // 2) * 2 * elem_bytes  # cos + sin
        read += self.tokens * 4  # positions i32
        write = self.tokens * (NUM_Q_HEADS + NUM_KV_HEADS) * HEAD_DIM * elem_bytes
        return read + write

    @property
    def bw_gbps(self) -> Optional[float]:
        if self.us is None:
            return None
        return self.bytes_total / 1e9 / (self.us / 1e6)


def bench_flydsl_rope(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark FlyDSL RoPE kernel."""
    try:
        from kernels.rope_kernel import build_rope_module
    except ImportError:
        print("  [FlyDSL] Could not import rope_kernel")
        return None

    device = torch.device("cuda")
    launch_fn = build_rope_module(
        head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        is_neox=True, dtype_str="bf16",
    )

    q = torch.randn(tokens, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos_cache = torch.randn(MAX_POS, ROTARY_DIM // 2, device=device, dtype=torch.bfloat16)
    sin_cache = torch.randn(MAX_POS, ROTARY_DIM // 2, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int32)
    stream = torch.cuda.current_stream()

    def run():
        launch_fn(q, k, cos_cache, sin_cache, positions, q, k, tokens, stream=stream)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] RoPE failed: {e}")
        return None


def bench_aiter_rope(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark AITER Triton RoPE baseline."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.triton.rope.rope import rope_cached_thd_positions_2c_fwd_inplace
    except ImportError:
        print("  [AITER] Could not import rope_cached_thd_positions_2c_fwd_inplace")
        return None

    device = torch.device("cuda")
    q = torch.randn(tokens, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
    cos_cache = torch.randn(MAX_POS, ROTARY_DIM // 2, device=device, dtype=torch.bfloat16)
    sin_cache = torch.randn(MAX_POS, ROTARY_DIM // 2, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)

    def run():
        rope_cached_thd_positions_2c_fwd_inplace(
            q, k, cos_cache, sin_cache, positions,
            rotate_style=0,  # NeoX
            reuse_freqs_front_part=True,
            nope_first=False,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [AITER] RoPE failed: {e}")
        return None


def print_results(rows: List[RopeBenchRow]) -> None:
    print()
    print("=" * 90)
    print(f"RoPE Benchmark — GPT-OSS 120B (head_dim={HEAD_DIM}, "
          f"Q_heads={NUM_Q_HEADS}, KV_heads={NUM_KV_HEADS})")
    print("=" * 90)
    hdr = f"{'M(conc)':>7s}  {'Kernel':<16s}  {'Latency(us)':>12s}  {'BW(GB/s)':>10s}"
    print(hdr)
    print("-" * 90)

    by_tokens: dict[int, List[RopeBenchRow]] = {}
    for r in rows:
        by_tokens.setdefault(r.tokens, []).append(r)

    for tokens in sorted(by_tokens.keys()):
        group = by_tokens[tokens]
        for i, r in enumerate(group):
            tok_col = f"{tokens:7d}" if i == 0 else f"{'':7s}"
            us_s = f"{r.us:12.1f}" if r.us else f"{'N/A':>12s}"
            bw_s = f"{r.bw_gbps:10.2f}" if r.bw_gbps else f"{'N/A':>10s}"
            print(f"{tok_col}  {r.label:<16s}  {us_s}  {bw_s}")

        aiter_row = next((r for r in group if r.label == "AITER" and r.us), None)
        fly_row = next((r for r in group if r.label == "FlyDSL" and r.us), None)
        if aiter_row and fly_row:
            sp = aiter_row.us / fly_row.us
            print(f"{'':7s}  {'FlyDSL/AITER':<16s}  {'':12s}  {sp:9.2f}x")
        print()

    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="RoPE benchmark: FlyDSL vs AITER")
    parser.add_argument("--concurrency", type=str, default="1,4,32,128",
                        help="Comma-separated M values (decode concurrency)")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    token_list = SWEEP_CONCURRENCY if args.sweep else [int(t) for t in args.concurrency.split(",")]

    rows: List[RopeBenchRow] = []
    for tokens in token_list:
        print(f"Benchmarking concurrency={tokens} ...")

        if not args.flydsl_only:
            aiter_us = bench_aiter_rope(tokens, args.warmup, args.iters)
            rows.append(RopeBenchRow(label="AITER", tokens=tokens, us=aiter_us))

        fly_us = bench_flydsl_rope(tokens, args.warmup, args.iters)
        rows.append(RopeBenchRow(label="FlyDSL", tokens=tokens, us=fly_us))

    print_results(rows)


if __name__ == "__main__":
    main()
