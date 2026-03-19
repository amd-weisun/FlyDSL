#!/usr/bin/env python3
"""Benchmark: Triton gemm_afp4wfp4_preshuffle vs FlyDSL compile_preshuffle_gemm_w4.

Usage (from FlyDSL/ directory):
    # Quick test (default shapes)
    PYTHONPATH=./ python tests/kernels/bench_fp4_gemm_triton_vs_flydsl.py

    # Custom shape
    PYTHONPATH=./ python tests/kernels/bench_fp4_gemm_triton_vs_flydsl.py -M 128 -N 8192 -K 8192

    # Sweep GPT-OSS inference shapes
    PYTHONPATH=./ python tests/kernels/bench_fp4_gemm_triton_vs_flydsl.py --sweep

Requirements:
    - gfx950 GPU (MXFP4 hardware support)
    - FlyDSL installed (pip install -e .)
    - AITER importable (set AITER_REPO env var or pip install -e ../aiter)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

# --- path setup (same as benchmark_common.py) ---
_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_FLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_FLYDSL_SRC) and _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)
_EMBEDDED_FLYDSL = os.path.join(_REPO_ROOT, ".flydsl", "build", "python_packages", "flydsl")
if os.path.isdir(_EMBEDDED_FLYDSL) and _EMBEDDED_FLYDSL not in sys.path:
    sys.path.insert(0, _EMBEDDED_FLYDSL)

import torch

from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter


# ---------------------------------------------------------------------------
# Data types & shapes representative of GPT-OSS 120B inference
# ---------------------------------------------------------------------------
# GPT-OSS 120B: hidden=8192, inter=28672, heads=64, kv_heads=8
# QKV proj:   M=batch*seq, N=10240 (8192+1024+1024), K=8192
# O proj:     M=batch*seq, N=8192, K=8192
# Gate proj:  M=batch*seq, N=1 (scalar), K=8192
# Up/Down:    M=batch*seq, N=28672, K=8192  /  N=8192, K=28672

GPT_OSS_SHAPES: List[Tuple[int, int, int]] = [
    # (M, N, K) — typical inference batch sizes
    # Decode (small M)
    (1, 10240, 8192),
    (1, 8192, 8192),
    (8, 10240, 8192),
    (8, 8192, 8192),
    (32, 10240, 8192),
    (32, 8192, 8192),
    # Prefill (larger M)
    (64, 10240, 8192),
    (64, 8192, 8192),
    (128, 8192, 8192),
    (256, 8192, 8192),
    (512, 8192, 8192),
    (1024, 8192, 8192),
]

DEFAULT_SHAPES: List[Tuple[int, int, int]] = [
    (32, 8192, 8192),
    (64, 8192, 8192),
    (128, 8192, 8192),
]

SCALE_GROUP_SIZE = 32


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class BenchRow:
    M: int
    N: int
    K: int
    flydsl_us: Optional[float]
    triton_us: Optional[float]

    @property
    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    @property
    def flydsl_tflops(self) -> Optional[float]:
        if self.flydsl_us is None:
            return None
        return self.flops / (self.flydsl_us / 1e6) / 1e12

    @property
    def triton_tflops(self) -> Optional[float]:
        if self.triton_us is None:
            return None
        return self.flops / (self.triton_us / 1e6) / 1e12

    @property
    def speedup(self) -> Optional[float]:
        """FlyDSL speedup over Triton (>1 means FlyDSL faster)."""
        if self.flydsl_us is None or self.triton_us is None:
            return None
        return self.triton_us / self.flydsl_us


def print_results(rows: List[BenchRow]) -> None:
    print()
    print("=" * 110)
    print("FP4 GEMM Benchmark: FlyDSL vs Triton (gemm_afp4wfp4_preshuffle)")
    print("=" * 110)
    hdr = (
        f"{'M':>6s}  {'N':>6s}  {'K':>6s}  "
        f"{'FlyDSL(us)':>12s}  {'TFLOPS':>8s}  "
        f"{'Triton(us)':>12s}  {'TFLOPS':>8s}  "
        f"{'Speedup':>8s}"
    )
    print(hdr)
    print("-" * 110)
    for r in rows:
        fly_us = f"{r.flydsl_us:12.1f}" if r.flydsl_us else f"{'N/A':>12s}"
        fly_tf = f"{r.flydsl_tflops:8.2f}" if r.flydsl_tflops else f"{'N/A':>8s}"
        tri_us = f"{r.triton_us:12.1f}" if r.triton_us else f"{'N/A':>12s}"
        tri_tf = f"{r.triton_tflops:8.2f}" if r.triton_tflops else f"{'N/A':>8s}"
        sp = f"{r.speedup:7.2f}x" if r.speedup else f"{'N/A':>8s}"
        print(f"{r.M:6d}  {r.N:6d}  {r.K:6d}  {fly_us}  {fly_tf}  {tri_us}  {tri_tf}  {sp}")
    print("=" * 110)
    # Summary
    valid = [r for r in rows if r.speedup is not None]
    if valid:
        avg_sp = sum(r.speedup for r in valid) / len(valid)
        geo_sp = 1.0
        for r in valid:
            geo_sp *= r.speedup
        geo_sp = geo_sp ** (1.0 / len(valid))
        print(f"Average speedup (FlyDSL/Triton): {avg_sp:.2f}x  |  Geomean: {geo_sp:.2f}x")
    print()


# ---------------------------------------------------------------------------
# FlyDSL FP4 GEMM
# ---------------------------------------------------------------------------
def _make_fp4_inputs_flydsl(M: int, N: int, K: int, device: torch.device):
    """Create FP4 quantized inputs using FlyDSL's fp4_utils."""
    from tests.kernels.utils import fp4_utils

    M_align = (M + 31) // 32 * 32
    N_align = (N + 31) // 32 * 32

    a_fp32 = torch.randn(M, K, device=device, dtype=torch.float32)
    b_fp32 = torch.randn(N, K, device=device, dtype=torch.float32)

    a_padded = torch.zeros(M_align, K, device=device, dtype=torch.float32)
    b_padded = torch.zeros(N_align, K, device=device, dtype=torch.float32)
    a_padded[:M] = a_fp32
    b_padded[:N] = b_fp32

    a_q, scale_a_orig, _ = fp4_utils.per_1x32_f4_quant(a_padded)
    a_q = a_q[:M]
    scale_a = fp4_utils.shuffle_scale_w4(scale_a_orig, 1, False)

    b_q, scale_b, _ = fp4_utils.per_1x32_f4_quant(b_padded)
    b_q = b_q[:N]
    b_shuffled = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    scale_b_shuffled = fp4_utils.shuffle_scale_w4(scale_b, 1, False)

    return a_q, b_shuffled, scale_a, scale_b_shuffled


def bench_flydsl_fp4_gemm(
    M: int, N: int, K: int,
    tile_m: int = 64, tile_n: int = 128, tile_k: int = 128,
    warmup: int = 20, iters: int = 200,
) -> Optional[float]:
    """Benchmark FlyDSL FP4 GEMM. Returns latency in microseconds."""
    try:
        from kernels.preshuffle_gemm import compile_preshuffle_gemm_w4
    except ImportError:
        print("  [FlyDSL] Could not import compile_preshuffle_gemm_w4")
        return None

    device = torch.device("cuda")

    # Adjust tile_m for small M
    actual_tile_m = min(tile_m, max(16, (M + 15) // 16 * 16))
    # tile_m must be power-of-2-ish and <= M (rounded up)
    for tm in [16, 32, 64, 128]:
        if tm >= M or tm == 128:
            actual_tile_m = tm
            break

    try:
        launch_fn = compile_preshuffle_gemm_w4(
            M=M, N=N, K=K,
            tile_m=actual_tile_m, tile_n=tile_n, tile_k=tile_k,
            a_dtype="fp4", b_dtype="fp4",
            out_dtype="bf16", lds_stage=2,
        )
    except Exception as e:
        print(f"  [FlyDSL] Compile failed: {e}")
        return None

    a_q, b_shuffled, scale_a, scale_b = _make_fp4_inputs_flydsl(M, N, K, device)
    c_out = torch.zeros((M, N), dtype=torch.bfloat16, device=device)

    def _to_bytes(t):
        if t.dtype in (torch.uint8, torch.int8):
            return t
        return t.view(torch.uint8)

    stream = torch.cuda.current_stream()

    def run():
        launch_fn(
            c_out.contiguous().view(-1),
            _to_bytes(a_q).contiguous().view(-1),
            _to_bytes(b_shuffled).contiguous().view(-1),
            _to_bytes(scale_a).contiguous().view(-1),
            _to_bytes(scale_b).contiguous().view(-1),
            M, N,
            stream,
        )

    return bench_gpu_us_torch(run, warmup=warmup, iters=iters)


# ---------------------------------------------------------------------------
# Triton FP4 GEMM
# ---------------------------------------------------------------------------
def _make_fp4_inputs_triton(M: int, N: int, K: int, device: torch.device):
    """Create FP4 quantized inputs using Triton's expected format."""
    # Random FP4-packed uint8 tensors
    torch.manual_seed(42)
    x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=device)
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=device)

    # E8M0 scales (values near 127 = scale ~1.0)
    M_pad = (M + 255) // 256 * 256
    x_scales = torch.randint(124, 130, (K // SCALE_GROUP_SIZE, M_pad),
                             dtype=torch.uint8, device=device).T[:M]
    w_scales = torch.randint(124, 130, (K // SCALE_GROUP_SIZE, N),
                             dtype=torch.uint8, device=device).T

    # Shuffle scales for preshuffle variant
    def shuffle_scales(scales):
        sm, sn = scales.shape
        if sm < 32:
            return scales.contiguous()
        s = scales.clone().view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
        s = s.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
        return s.view(sm // 32, sn * 32)

    x_scales_shuffled = shuffle_scales(x_scales)
    w_scales_shuffled = shuffle_scales(w_scales)

    # Shuffle weights for preshuffle layout: (N, K//2) -> (N//16, K//2*16)
    from aiter.ops.shuffle import shuffle_weight
    w_shuffled = shuffle_weight(w, layout=(16, 16), use_int4=False)
    w_shuffled = w_shuffled.reshape(N // 16, (K // 2) * 16)

    return x, w_shuffled, x_scales_shuffled, w_scales_shuffled


def bench_triton_fp4_gemm(
    M: int, N: int, K: int,
    warmup: int = 20, iters: int = 200,
) -> Optional[float]:
    """Benchmark Triton FP4 GEMM. Returns latency in microseconds."""
    if not maybe_enable_aiter():
        print("  [Triton] AITER not available (set AITER_REPO env var)")
        return None

    try:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
    except ImportError:
        print("  [Triton] Could not import gemm_afp4wfp4_preshuffle")
        return None

    device = torch.device("cuda")
    x, w_shuffled, x_scales, w_scales = _make_fp4_inputs_triton(M, N, K, device)
    y = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    def run():
        gemm_afp4wfp4_preshuffle(
            x, w_shuffled, x_scales, w_scales,
            dtype=torch.bfloat16, y=y,
        )

    return bench_gpu_us_torch(run, warmup=warmup, iters=iters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_benchmark(
    shapes: List[Tuple[int, int, int]],
    tile_m: int = 64, tile_n: int = 128, tile_k: int = 128,
    warmup: int = 20, iters: int = 200,
) -> List[BenchRow]:
    rows = []
    for M, N, K in shapes:
        print(f"Benchmarking M={M}, N={N}, K={K} ...")
        fly_us = bench_flydsl_fp4_gemm(M, N, K, tile_m, tile_n, tile_k, warmup, iters)
        tri_us = bench_triton_fp4_gemm(M, N, K, warmup, iters)
        rows.append(BenchRow(M=M, N=N, K=K, flydsl_us=fly_us, triton_us=tri_us))
    return rows


def main():
    parser = argparse.ArgumentParser(description="FP4 GEMM: FlyDSL vs Triton benchmark")
    parser.add_argument("-M", type=int, default=64)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--tile_m", type=int, default=64)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--sweep", action="store_true",
                        help="Run GPT-OSS inference shapes sweep")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    if args.sweep:
        shapes = GPT_OSS_SHAPES
    else:
        shapes = [(args.M, args.N, args.K)]

    rows = run_benchmark(
        shapes,
        tile_m=args.tile_m, tile_n=args.tile_n, tile_k=args.tile_k,
        warmup=args.warmup, iters=args.iters,
    )
    print_results(rows)


if __name__ == "__main__":
    main()
