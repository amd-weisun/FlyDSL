#!/usr/bin/env python3
"""Benchmark: Triton _fused_qk_rope_reshape_and_cache kernel.

This kernel is 4.1% of GPU time in GPT-OSS 120B. It fuses:
  RoPE application + Q/K reshape + KV cache write (+ optional FP8 quantization)

FlyDSL does NOT have this kernel yet — this benchmark establishes the Triton
baseline performance to set the target for FlyDSL implementation.

Usage (from FlyDSL/ directory):
    # Default GPT-OSS shapes
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_qk_rope_cache.py

    # Sweep batch sizes
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_qk_rope_cache.py --sweep
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

# --- path setup ---
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_FLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_FLYDSL_SRC) and _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

import torch

from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter

# GPT-OSS 120B attention config
NUM_Q_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 64
KV_BLOCK_SIZE = 128  # paged attention block size

# Inference scenario: prompt=1K, output=8K, decode-dominated
DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_TOKENS = [1024]
SWEEP_TOKENS = DECODE_TOKENS + PREFILL_TOKENS


@dataclass
class RopeCacheBenchRow:
    tokens: int
    kv_dtype: str  # "bf16" or "fp8"
    triton_us: Optional[float]
    flydsl_us: Optional[float]

    @property
    def speedup(self) -> Optional[float]:
        if self.triton_us is None or self.flydsl_us is None:
            return None
        return self.triton_us / self.flydsl_us

    @property
    def bandwidth_gb(self) -> float:
        """Approximate memory traffic in GB."""
        elem = 2  # bf16
        T = self.tokens
        # Read: Q[T, QH, D] + K[T, KH, D] + V[T, KH, D] + cos[T, D/2] + sin[T, D/2] + pos[T]
        # Write: Q_out[T, QH, D] + K_out[T, KH, D] + KV_cache_write[T, KH, D]*2
        read_bytes = (T * NUM_Q_HEADS * HEAD_DIM + T * NUM_KV_HEADS * HEAD_DIM * 2
                      + T * HEAD_DIM + T * 4) * elem
        write_bytes = (T * NUM_Q_HEADS * HEAD_DIM + T * NUM_KV_HEADS * HEAD_DIM * 3) * elem
        return (read_bytes + write_bytes) / 1e9


def bench_triton_fused_qk_rope_cache(
    tokens: int, kv_dtype: str, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark Triton fused_qk_rope_reshape_and_cache."""
    if not maybe_enable_aiter():
        print("  [Triton] AITER not available")
        return None
    try:
        from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
    except ImportError:
        print("  [Triton] Could not import fused_qk_rope_reshape_and_cache")
        return None

    device = torch.device("cuda")
    T = tokens

    # Input tensors
    q = torch.randn(T, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    k = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    v = torch.randn(T, NUM_KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)

    # RoPE cos/sin
    pos = torch.arange(T, dtype=torch.int64, device=device)
    cos = torch.randn(T, HEAD_DIM // 2, dtype=torch.float32, device=device)
    sin = torch.randn(T, HEAD_DIM // 2, dtype=torch.float32, device=device)

    # KV cache (paged) — allocate enough blocks for T tokens
    num_blocks = (T + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE + 1

    if kv_dtype == "fp8":
        try:
            cache_dtype = torch.float8_e4m3fn
        except AttributeError:
            cache_dtype = torch.int8
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        apply_scale = True
    else:
        cache_dtype = torch.bfloat16
        k_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
        apply_scale = False

    # Non-flash layout: (num_blocks, KH, D//X, block_size, X) where X=16
    X = 16
    key_cache = torch.zeros(
        num_blocks, NUM_KV_HEADS, HEAD_DIM // X, KV_BLOCK_SIZE, X,
        dtype=cache_dtype, device=device,
    )
    value_cache = torch.zeros(
        num_blocks, NUM_KV_HEADS, HEAD_DIM, KV_BLOCK_SIZE,
        dtype=cache_dtype, device=device,
    )

    # Slot mapping: token i -> slot i (simple linear mapping)
    slot_mapping = torch.arange(T, dtype=torch.int64, device=device)

    # Output buffers
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    def run():
        fused_qk_rope_reshape_and_cache(
            q=q, k=k, v=v,
            key_cache=key_cache, value_cache=value_cache,
            slot_mapping=slot_mapping,
            pos=pos, cos=cos, sin=sin,
            k_scale=k_scale, v_scale=v_scale,
            is_neox=True,
            flash_layout=False,
            apply_scale=apply_scale,
            q_out=q_out, k_out=k_out,
            output_zeros=False,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [Triton] Failed: {e}")
        return None


def print_results(rows: List[RopeCacheBenchRow]) -> None:
    print()
    print("=" * 100)
    print(f"Fused QK RoPE + KV Cache Benchmark — GPT-OSS 120B "
          f"(QH={NUM_Q_HEADS}, KH={NUM_KV_HEADS}, D={HEAD_DIM})")
    print("=" * 100)
    hdr = (
        f"{'Tokens':>7s}  {'KV dtype':>8s}  "
        f"{'Triton(us)':>12s}  {'BW(GB/s)':>10s}  "
        f"{'FlyDSL(us)':>12s}  {'Speedup':>8s}"
    )
    print(hdr)
    print("-" * 100)

    for r in rows:
        tri_us = f"{r.triton_us:12.1f}" if r.triton_us else f"{'N/A':>12s}"
        bw = r.bandwidth_gb / (r.triton_us / 1e6) if r.triton_us else None
        bw_s = f"{bw:10.1f}" if bw else f"{'N/A':>10s}"
        fly_us = f"{r.flydsl_us:12.1f}" if r.flydsl_us else f"{'N/A':>12s}"
        sp = f"{r.speedup:7.2f}x" if r.speedup else f"{'N/A':>8s}"
        print(f"{r.tokens:7d}  {r.kv_dtype:>8s}  {tri_us}  {bw_s}  {fly_us}  {sp}")

    print("=" * 100)
    print()
    print("NOTE: FlyDSL does NOT have this kernel yet. This establishes the Triton baseline.")
    print("      The kernel fuses: RoPE + Q/K reshape + KV cache write (+ optional FP8 quant)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Fused QK RoPE + KV cache benchmark (Triton baseline)")
    parser.add_argument("--tokens", type=str, default="1,32,128,1024")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--kv-dtype", type=str, default="both",
                        choices=["bf16", "fp8", "both"])
    args = parser.parse_args()

    torch.set_default_device("cuda")

    if args.sweep:
        token_list = SWEEP_TOKENS
    else:
        token_list = [int(t.strip()) for t in args.tokens.split(",")]

    kv_dtypes = ["bf16", "fp8"] if args.kv_dtype == "both" else [args.kv_dtype]

    rows: List[RopeCacheBenchRow] = []
    for tokens in token_list:
        for kv_dt in kv_dtypes:
            print(f"Benchmarking tokens={tokens}, kv_dtype={kv_dt} ...")
            tri_us = bench_triton_fused_qk_rope_cache(tokens, kv_dt, args.warmup, args.iters)
            rows.append(RopeCacheBenchRow(
                tokens=tokens, kv_dtype=kv_dt,
                triton_us=tri_us, flydsl_us=None,
            ))

    print_results(rows)


if __name__ == "__main__":
    main()
