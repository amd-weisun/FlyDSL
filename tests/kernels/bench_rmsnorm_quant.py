#!/usr/bin/env python3
"""Benchmark: AITER add_rmsnorm_quant (HIP/CK) vs FlyDSL rmsnorm_kernel.

add_rmsnorm_quant is 8.2% of GPU time in GPT-OSS 120B. It fuses:
  residual_add + rmsnorm + dynamic_quantization

Current FlyDSL has basic RMSNorm only (no add or quant fusion).
This benchmark establishes the baseline and measures the FlyDSL RMSNorm portion.

Usage (from FlyDSL/ directory):
    # Default GPT-OSS shapes
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_rmsnorm_quant.py

    # Sweep batch sizes
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_rmsnorm_quant.py --sweep

    # FlyDSL only
    PYTHONPATH=./ python tests/kernels/bench_rmsnorm_quant.py --flydsl-only
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
_EMBEDDED_FLYDSL = os.path.join(_REPO_ROOT, ".flydsl", "build", "python_packages", "flydsl")
if os.path.isdir(_EMBEDDED_FLYDSL) and _EMBEDDED_FLYDSL not in sys.path:
    sys.path.insert(0, _EMBEDDED_FLYDSL)

import torch

from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter

# GPT-OSS 120B: hidden_size=2880
HIDDEN_DIM = 2880
EPSILON = 1e-5

# Inference scenario: prompt=1K, output=8K, decode-dominated
DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]
PREFILL_TOKENS = [1024]
SWEEP_TOKENS = DECODE_TOKENS + PREFILL_TOKENS


@dataclass
class NormBenchRow:
    tokens: int
    variant: str  # "rmsnorm" or "add_rmsnorm_quant"
    aiter_us: Optional[float]
    flydsl_us: Optional[float]

    @property
    def speedup(self) -> Optional[float]:
        if self.aiter_us is None or self.flydsl_us is None:
            return None
        return self.aiter_us / self.flydsl_us

    @property
    def bandwidth_gb(self) -> float:
        """Approximate memory traffic in GB (read input + write output)."""
        elem_bytes = 2  # bf16
        # rmsnorm: read input[M,N] + weight[N] + write output[M,N]
        # add_rmsnorm_quant: + read residual_in[M,N] + write residual_out[M,N] + write scale[M]
        if self.variant == "rmsnorm":
            return (2 * self.tokens * HIDDEN_DIM + HIDDEN_DIM) * elem_bytes / 1e9
        else:
            return (4 * self.tokens * HIDDEN_DIM + HIDDEN_DIM + self.tokens * 4) * elem_bytes / 1e9

    def aiter_bw(self) -> Optional[float]:
        if self.aiter_us is None:
            return None
        return self.bandwidth_gb / (self.aiter_us / 1e6)

    def flydsl_bw(self) -> Optional[float]:
        if self.flydsl_us is None:
            return None
        return self.bandwidth_gb / (self.flydsl_us / 1e6)


# ---------------------------------------------------------------------------
# AITER benchmarks
# ---------------------------------------------------------------------------
def bench_aiter_rmsnorm(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark basic AITER rmsnorm (CK or HIP backend)."""
    if not maybe_enable_aiter():
        return None
    try:
        import aiter
    except ImportError:
        return None

    device = torch.device("cuda")
    x = torch.randn(tokens, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(x)

    def run():
        aiter.rmsnorm(out, x, weight, EPSILON)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [AITER] rmsnorm failed: {e}")
        return None


def bench_aiter_add_rmsnorm_quant(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark AITER add_rmsnorm_quant (fused residual+norm+quant)."""
    if not maybe_enable_aiter():
        return None
    try:
        import aiter
    except ImportError:
        return None

    device = torch.device("cuda")
    x = torch.randn(tokens, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    residual_in = torch.randn(tokens, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    residual_out = torch.empty_like(x)
    weight = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    # FP8 quantized output
    try:
        out = torch.empty(tokens, HIDDEN_DIM, dtype=torch.float8_e4m3fn, device=device)
    except Exception:
        out = torch.empty(tokens, HIDDEN_DIM, dtype=torch.int8, device=device)
    scale = torch.empty(tokens, 1, dtype=torch.float32, device=device)

    def run():
        aiter.add_rmsnorm_quant(out, x, residual_in, residual_out, scale, weight, EPSILON)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [AITER] add_rmsnorm_quant failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FlyDSL benchmark
# ---------------------------------------------------------------------------
def bench_flydsl_rmsnorm(tokens: int, warmup: int, iters: int) -> Optional[float]:
    """Benchmark FlyDSL RMSNorm kernel."""
    try:
        from kernels.rmsnorm_kernel import build_rmsnorm_module
    except ImportError:
        print("  [FlyDSL] Could not import build_rmsnorm_module")
        return None

    device = torch.device("cuda")

    try:
        launch_fn = build_rmsnorm_module(tokens, HIDDEN_DIM, "bf16")
    except Exception as e:
        print(f"  [FlyDSL] Compile failed: {e}")
        return None

    x = torch.randn(tokens, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    y = torch.empty_like(x)
    stream = torch.cuda.current_stream()

    def run():
        launch_fn(x, gamma, y, tokens, stream=stream)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] rmsnorm failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_results(rows: List[NormBenchRow]) -> None:
    print()
    print("=" * 120)
    print(f"RMSNorm Benchmark: AITER vs FlyDSL — GPT-OSS 120B (hidden={HIDDEN_DIM})")
    print("=" * 120)
    hdr = (
        f"{'Variant':<22s}  {'Tokens':>7s}  "
        f"{'AITER(us)':>10s}  {'BW(GB/s)':>10s}  "
        f"{'FlyDSL(us)':>12s}  {'BW(GB/s)':>10s}  "
        f"{'Speedup':>8s}"
    )
    print(hdr)
    print("-" * 120)

    for r in rows:
        a_us = f"{r.aiter_us:10.1f}" if r.aiter_us else f"{'N/A':>10s}"
        a_bw = f"{r.aiter_bw():10.1f}" if r.aiter_bw() else f"{'N/A':>10s}"
        f_us = f"{r.flydsl_us:12.1f}" if r.flydsl_us else f"{'N/A':>12s}"
        f_bw = f"{r.flydsl_bw():10.1f}" if r.flydsl_bw() else f"{'N/A':>10s}"
        sp = f"{r.speedup:7.2f}x" if r.speedup else f"{'N/A':>8s}"
        print(f"{r.variant:<22s}  {r.tokens:7d}  {a_us}  {a_bw}  {f_us}  {f_bw}  {sp}")

    print("=" * 120)
    valid = [r for r in rows if r.speedup is not None]
    if valid:
        geo = 1.0
        for r in valid:
            geo *= r.speedup
        geo = geo ** (1.0 / len(valid))
        print(f"Geomean speedup (FlyDSL/AITER): {geo:.2f}x")
    print()
    print("NOTE: FlyDSL currently benchmarks basic RMSNorm only.")
    print("      To match add_rmsnorm_quant, FlyDSL needs: residual_add + quant fusion.")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RMSNorm benchmark: AITER vs FlyDSL")
    parser.add_argument("--tokens", type=str, default="1,32,128,1024",
                        help="Comma-separated token counts")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--flydsl-only", action="store_true")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    if args.sweep:
        token_list = SWEEP_TOKENS
    else:
        token_list = [int(t.strip()) for t in args.tokens.split(",")]

    rows: List[NormBenchRow] = []

    for tokens in token_list:
        print(f"Benchmarking tokens={tokens} ...")

        # 1. Basic RMSNorm (apples-to-apples comparison)
        aiter_us = None if args.flydsl_only else bench_aiter_rmsnorm(tokens, args.warmup, args.iters)
        flydsl_us = bench_flydsl_rmsnorm(tokens, args.warmup, args.iters)
        rows.append(NormBenchRow(
            tokens=tokens, variant="rmsnorm",
            aiter_us=aiter_us, flydsl_us=flydsl_us,
        ))

        # 2. Fused add_rmsnorm_quant (AITER only — FlyDSL doesn't have this yet)
        if not args.flydsl_only:
            fused_us = bench_aiter_add_rmsnorm_quant(tokens, args.warmup, args.iters)
            rows.append(NormBenchRow(
                tokens=tokens, variant="add_rmsnorm_quant",
                aiter_us=fused_us, flydsl_us=None,
            ))

    print_results(rows)


if __name__ == "__main__":
    main()
