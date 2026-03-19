#!/usr/bin/env python3
"""GPT-OSS kernel-level benchmark: AITER (baseline) vs FlyDSL.

Uses gpt_oss_kernel_shapes.py to derive realistic kernel input shapes
parameterized by TP and concurrency, then benchmarks each kernel
with both AITER and FlyDSL backends.

Usage (from FlyDSL/ directory):
    # Benchmark all GEMM kernels at TP=8
    PYTHONPATH=./ python tests/kernels/bench_gpt_oss_kernels.py --tp 8

    # Benchmark specific kernel types
    PYTHONPATH=./ python tests/kernels/bench_gpt_oss_kernels.py --tp 8 --kernel gemm

    # Benchmark specific concurrencies
    PYTHONPATH=./ python tests/kernels/bench_gpt_oss_kernels.py --tp 8 --tokens 1,32,128,1024

    # Only FlyDSL (no AITER needed)
    PYTHONPATH=./ python tests/kernels/bench_gpt_oss_kernels.py --tp 8 --flydsl-only

Requirements:
    - GPU (gfx942 or gfx950)
    - FlyDSL installed
    - AITER importable (for baseline comparison)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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
from gpt_oss_kernel_shapes import (
    GptOssConfig,
    KernelShape,
    generate_shapes,
    MIXED_TOKENS,
)


# ============================================================================
# Result tracking
# ============================================================================
@dataclass
class BenchResult:
    shape: KernelShape
    aiter_us: Optional[float] = None
    flydsl_us: Optional[float] = None
    aiter_err: str = ""
    flydsl_err: str = ""

    @property
    def speedup(self) -> Optional[float]:
        """FlyDSL speedup over AITER (>1 = FlyDSL faster)."""
        if self.flydsl_us and self.aiter_us:
            return self.aiter_us / self.flydsl_us
        return None

    @property
    def aiter_tflops(self) -> Optional[float]:
        if self.aiter_us and self.shape.flops > 0:
            return self.shape.flops / (self.aiter_us / 1e6) / 1e12
        return None

    @property
    def flydsl_tflops(self) -> Optional[float]:
        if self.flydsl_us and self.shape.flops > 0:
            return self.shape.flops / (self.flydsl_us / 1e6) / 1e12
        return None


# ============================================================================
# AITER kernel benchmarks (baseline)
# ============================================================================

def _bench_aiter_rmsnorm(s: KernelShape, warmup: int, iters: int) -> float:
    """Benchmark AITER rmsnorm2d_fwd."""
    import aiter
    x = torch.randn(s.M, s.N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(s.N, dtype=torch.bfloat16, device="cuda")
    return bench_gpu_us_torch(
        lambda: aiter.rmsnorm2d_fwd(x, w, epsilon=1e-5),
        warmup=warmup, iters=iters,
    )


def _bench_aiter_gemm_bf16(s: KernelShape, warmup: int, iters: int) -> float:
    """Benchmark AITER tgemm.mm (BF16, hipblasLt path)."""
    from aiter.tuned_gemm import TunedGemm
    tgemm = TunedGemm()
    a = torch.randn(s.M, s.K, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(s.N, s.K, dtype=torch.bfloat16, device="cuda")  # (N, K) for TN layout
    c = torch.empty(s.M, s.N, dtype=torch.bfloat16, device="cuda")
    return bench_gpu_us_torch(
        lambda: tgemm.mm(a, b, c),
        warmup=warmup, iters=iters,
    )


def _bench_aiter_gemm_mxfp4(s: KernelShape, warmup: int, iters: int) -> float:
    """Benchmark AITER gemm_a4w4 or Triton gemm_afp4wfp4_preshuffle (MXFP4)."""
    use_triton = os.environ.get("ATOM_USE_TRITON_GEMM", "0") == "1"
    if use_triton:
        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
        from aiter.ops.shuffle import shuffle_weight

        SCALE_GROUP = 32
        x = torch.randint(0, 256, (s.M, s.K // 2), dtype=torch.uint8, device="cuda")
        w = torch.randint(0, 256, (s.N, s.K // 2), dtype=torch.uint8, device="cuda")
        w_shuf = shuffle_weight(w, layout=(16, 16), use_int4=False)
        w_shuf = w_shuf.reshape(s.N // 16, (s.K // 2) * 16)

        M_pad = (s.M + 255) // 256 * 256
        x_sc = torch.randint(124, 130, (s.K // SCALE_GROUP, M_pad),
                             dtype=torch.uint8, device="cuda").T[:s.M]
        w_sc = torch.randint(124, 130, (s.K // SCALE_GROUP, s.N),
                             dtype=torch.uint8, device="cuda").T

        def _shuffle_scales(sc):
            sm, sn = sc.shape
            if sm < 32:
                return sc.contiguous()
            t = sc.clone().view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
            return t.permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm // 32, sn * 32)

        x_sc_s = _shuffle_scales(x_sc)
        w_sc_s = _shuffle_scales(w_sc)
        y = torch.empty(s.M, s.N, dtype=torch.bfloat16, device="cuda")

        return bench_gpu_us_torch(
            lambda: gemm_afp4wfp4_preshuffle(x, w_shuf, x_sc_s, w_sc_s,
                                              dtype=torch.bfloat16, y=y),
            warmup=warmup, iters=iters,
        )
    else:
        import aiter
        x = torch.randint(0, 256, (s.M, s.K // 2), dtype=torch.uint8, device="cuda")
        w = torch.randint(0, 256, (s.N, s.K // 2), dtype=torch.uint8, device="cuda")
        x_sc = torch.ones(s.M, s.K // 32, dtype=torch.float32, device="cuda")
        w_sc = torch.ones(s.N, s.K // 32, dtype=torch.float32, device="cuda")
        return bench_gpu_us_torch(
            lambda: aiter.gemm_a4w4(x, w, x_sc, w_sc),
            warmup=warmup, iters=iters,
        )


AITER_BENCH_DISPATCH: Dict[Tuple[str, str], Callable] = {
    ("gemm", "hipblaslt"): _bench_aiter_gemm_bf16,
    ("gemm", "ck"): _bench_aiter_gemm_mxfp4,  # MoE GEMM path
    ("rmsnorm", "ck"): _bench_aiter_rmsnorm,
}


def bench_aiter(s: KernelShape, warmup: int, iters: int) -> Tuple[Optional[float], str]:
    """Dispatch to appropriate AITER benchmark. Returns (latency_us, error_msg)."""
    key = (s.kernel_type, s.backend)
    fn = AITER_BENCH_DISPATCH.get(key)
    if fn is None:
        return None, f"no AITER bench for {key}"
    try:
        return fn(s, warmup, iters), ""
    except Exception as e:
        return None, str(e)[:80]


# ============================================================================
# FlyDSL kernel benchmarks
# ============================================================================

def _bench_flydsl_rmsnorm(s: KernelShape, warmup: int, iters: int) -> float:
    """Benchmark FlyDSL rmsnorm_kernel."""
    from kernels.rmsnorm_kernel import build_rmsnorm_module

    launch_fn = build_rmsnorm_module(s.M, s.N, "bf16")
    x = torch.randn(s.M, s.N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(s.N, dtype=torch.bfloat16, device="cuda")
    y = torch.empty(s.M, s.N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()

    return bench_gpu_us_torch(
        lambda: launch_fn(x, w, y, s.M, stream=stream),
        warmup=warmup, iters=iters,
    )


def _bench_flydsl_gemm_bf16(s: KernelShape, warmup: int, iters: int) -> float:
    """Benchmark FlyDSL preshuffle GEMM (BF16)."""
    from kernels.preshuffle_gemm import compile_preshuffle_gemm_a8

    tile_m = min(64, max(16, s.M))
    for tm in [16, 32, 64, 128]:
        if tm >= s.M or tm == 128:
            tile_m = tm
            break

    launch_fn = compile_preshuffle_gemm_a8(
        M=s.M, N=s.N, K=s.K,
        tile_m=tile_m, tile_n=128, tile_k=256,
        in_dtype="bf16", out_dtype="bf16",
    )
    a = torch.randn(s.M, s.K, dtype=torch.bfloat16, device="cuda")
    # B needs preshuffle — for now use random bytes in the right shape
    b = torch.randint(0, 256, (s.N, s.K), dtype=torch.uint8, device="cuda")
    scale_a = torch.ones(s.M, dtype=torch.float32, device="cuda")
    scale_b = torch.ones(s.N, dtype=torch.float32, device="cuda")
    c = torch.empty(s.M, s.N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()

    return bench_gpu_us_torch(
        lambda: launch_fn(
            c.view(-1), a.view(-1), b.view(-1),
            scale_a.view(-1), scale_b.view(-1),
            s.M, s.N, stream,
        ),
        warmup=warmup, iters=iters,
    )


def _bench_flydsl_gemm_mxfp4(s: KernelShape, warmup: int, iters: int) -> float:
    """Benchmark FlyDSL preshuffle GEMM (MXFP4)."""
    from kernels.preshuffle_gemm import compile_preshuffle_gemm_w4
    from tests.kernels.utils import fp4_utils

    tile_m = min(64, max(16, s.M))
    for tm in [16, 32, 64, 128]:
        if tm >= s.M or tm == 128:
            tile_m = tm
            break

    launch_fn = compile_preshuffle_gemm_w4(
        M=s.M, N=s.N, K=s.K,
        tile_m=tile_m, tile_n=128, tile_k=128,
        a_dtype="fp4", b_dtype="fp4",
        out_dtype="bf16", lds_stage=2,
    )

    M_align = (s.M + 31) // 32 * 32
    N_align = (s.N + 31) // 32 * 32

    a_fp32 = torch.randn(s.M, s.K, device="cuda", dtype=torch.float32)
    b_fp32 = torch.randn(s.N, s.K, device="cuda", dtype=torch.float32)
    a_pad = torch.zeros(M_align, s.K, device="cuda", dtype=torch.float32)
    b_pad = torch.zeros(N_align, s.K, device="cuda", dtype=torch.float32)
    a_pad[:s.M] = a_fp32
    b_pad[:s.N] = b_fp32

    a_q, sa_orig, _ = fp4_utils.per_1x32_f4_quant(a_pad)
    a_q = a_q[:s.M]
    sa = fp4_utils.shuffle_scale_w4(sa_orig, 1, False)
    b_q, sb, _ = fp4_utils.per_1x32_f4_quant(b_pad)
    b_q = b_q[:s.N]
    b_shuf = fp4_utils.shuffle_weight_w4(b_q, 16, False, False)
    sb_shuf = fp4_utils.shuffle_scale_w4(sb, 1, False)

    c = torch.zeros(s.M, s.N, dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.current_stream()

    def _to_bytes(t):
        return t if t.dtype in (torch.uint8, torch.int8) else t.view(torch.uint8)

    return bench_gpu_us_torch(
        lambda: launch_fn(
            c.view(-1),
            _to_bytes(a_q).view(-1), _to_bytes(b_shuf).view(-1),
            _to_bytes(sa).view(-1), _to_bytes(sb_shuf).view(-1),
            s.M, s.N, stream,
        ),
        warmup=warmup, iters=iters,
    )


FLYDSL_BENCH_DISPATCH: Dict[str, Callable] = {
    "rmsnorm": _bench_flydsl_rmsnorm,
    "gemm_bf16": _bench_flydsl_gemm_bf16,
    "gemm_mxfp4": _bench_flydsl_gemm_mxfp4,
}


def bench_flydsl(s: KernelShape, warmup: int, iters: int) -> Tuple[Optional[float], str]:
    """Dispatch to appropriate FlyDSL benchmark. Returns (latency_us, error_msg)."""
    if s.kernel_type == "rmsnorm":
        key = "rmsnorm"
    elif s.kernel_type == "gemm" and s.dtype == "mxfp4":
        key = "gemm_mxfp4"
    elif s.kernel_type == "gemm" and s.dtype == "bf16":
        key = "gemm_bf16"
    else:
        return None, f"no FlyDSL bench for {s.kernel_type}/{s.dtype}"

    fn = FLYDSL_BENCH_DISPATCH.get(key)
    if fn is None:
        return None, f"no FlyDSL bench for {key}"
    try:
        return fn(s, warmup, iters), ""
    except Exception as e:
        return None, str(e)[:80]


# ============================================================================
# Printing
# ============================================================================
def _fmt(val: Optional[float], width: int = 10) -> str:
    if val is None:
        return f"{'N/A':>{width}s}"
    return f"{val:{width}.1f}"


def _fmt_tf(val: Optional[float], width: int = 8) -> str:
    if val is None:
        return f"{'':>{width}s}"
    return f"{val:{width}.2f}"


def _fmt_sp(val: Optional[float], width: int = 8) -> str:
    if val is None:
        return f"{'':>{width}s}"
    return f"{val:{width-1}.2f}x"


def print_results(results: List[BenchResult], tp_size: int) -> None:
    print(f"\n{'='*130}")
    print(f"GPT-OSS 120B Kernel Benchmark — TP={tp_size} — AITER (baseline) vs FlyDSL")
    print(f"{'='*130}")
    hdr = (f"{'M':>6s} {'Layer':<20s} {'Type':<8s} {'Shape':<20s} "
           f"{'AITER(us)':>10s} {'TFLOPS':>8s} "
           f"{'FlyDSL(us)':>10s} {'TFLOPS':>8s} "
           f"{'Speedup':>8s} {'Errors'}")
    print(hdr)
    print("-" * 130)

    prev_m = None
    for r in results:
        if r.shape.M != prev_m and prev_m is not None:
            print()  # visual separator between token groups
        prev_m = r.shape.M

        err_parts = []
        if r.aiter_err:
            err_parts.append(f"A:{r.aiter_err}")
        if r.flydsl_err:
            err_parts.append(f"F:{r.flydsl_err}")
        err = " | ".join(err_parts) if err_parts else ""

        print(f"{r.shape.M:6d} {r.shape.layer:<20s} {r.shape.kernel_type:<8s} "
              f"{r.shape.shape_str:<20s} "
              f"{_fmt(r.aiter_us)} {_fmt_tf(r.aiter_tflops)} "
              f"{_fmt(r.flydsl_us)} {_fmt_tf(r.flydsl_tflops)} "
              f"{_fmt_sp(r.speedup)} {err}")

    print(f"{'='*130}")

    # Per kernel-type summary
    by_type: Dict[str, List[BenchResult]] = {}
    for r in results:
        key = f"{r.shape.kernel_type}/{r.shape.dtype}"
        by_type.setdefault(key, []).append(r)

    print(f"\nSummary by kernel type:")
    for key, rs in sorted(by_type.items()):
        valid = [r for r in rs if r.speedup is not None]
        if valid:
            avg = sum(r.speedup for r in valid) / len(valid)
            geo = 1.0
            for r in valid:
                geo *= r.speedup
            geo = geo ** (1.0 / len(valid))
            print(f"  {key:<20s}: {len(valid):2d} shapes, avg speedup={avg:.2f}x, geomean={geo:.2f}x")
        else:
            tested = len([r for r in rs if r.aiter_us is not None or r.flydsl_us is not None])
            print(f"  {key:<20s}: {tested:2d} shapes tested, no comparison available")
    print()


# ============================================================================
# Main benchmark loop
# ============================================================================
def run_benchmark(
    tp_size: int = 8,
    tokens_list: Optional[List[int]] = None,
    kernel_filter: Optional[str] = None,
    flydsl_only: bool = False,
    warmup: int = 10,
    iters: int = 100,
) -> List[BenchResult]:
    """Run the full benchmark sweep."""
    cfg = GptOssConfig()
    if tokens_list is None:
        tokens_list = MIXED_TOKENS

    has_aiter = False
    if not flydsl_only:
        has_aiter = maybe_enable_aiter()
        if not has_aiter:
            print("[WARN] AITER not available. Set AITER_REPO env var. Running FlyDSL only.")

    results = []
    for batch_tokens in tokens_list:
        shapes = generate_shapes(cfg, tp_size, batch_tokens)

        # Filter kernel types if requested
        if kernel_filter:
            shapes = [s for s in shapes if s.kernel_type == kernel_filter
                      or s.layer.startswith(kernel_filter)]

        for s in shapes:
            print(f"  [{s.layer}] M={s.M} N={s.N} K={s.K} ({s.dtype}) ...", end=" ", flush=True)

            aiter_us, aiter_err = (None, "skipped")
            if has_aiter and not flydsl_only:
                aiter_us, aiter_err = bench_aiter(s, warmup, iters)

            flydsl_us, flydsl_err = bench_flydsl(s, warmup, iters)

            status_parts = []
            if aiter_us is not None:
                status_parts.append(f"AITER={aiter_us:.1f}us")
            if flydsl_us is not None:
                status_parts.append(f"FlyDSL={flydsl_us:.1f}us")
            if not status_parts:
                status_parts.append("N/A")
            print(" | ".join(status_parts))

            results.append(BenchResult(
                shape=s,
                aiter_us=aiter_us, flydsl_us=flydsl_us,
                aiter_err=aiter_err, flydsl_err=flydsl_err,
            ))

    return results


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS kernel benchmark: AITER vs FlyDSL")
    parser.add_argument("--tp", type=int, default=8, choices=[1, 2, 4, 8])
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated token counts (default: 1,8,32,64,128,256,512,1024)")
    parser.add_argument("--kernel", type=str, default=None,
                        help="Filter by kernel type: gemm, rmsnorm, attention, topk, moe, embedding")
    parser.add_argument("--flydsl-only", action="store_true",
                        help="Only run FlyDSL benchmarks (no AITER needed)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    if args.tokens:
        tokens_list = [int(t.strip()) for t in args.tokens.split(",")]
    else:
        tokens_list = None

    torch.set_default_device("cuda")
    results = run_benchmark(
        tp_size=args.tp,
        tokens_list=tokens_list,
        kernel_filter=args.kernel,
        flydsl_only=args.flydsl_only,
        warmup=args.warmup,
        iters=args.iters,
    )
    print_results(results, args.tp)


if __name__ == "__main__":
    main()
