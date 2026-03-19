#!/usr/bin/env python3
"""Benchmark: CK fused_moe vs FlyDSL MoE (Stage1 + Stage2).

MoE kernels are the #1 GPU time consumer in GPT-OSS 120B (35% of GPU time).
This benchmark compares CK (via AITER) vs FlyDSL for both MoE stages.

Usage (from FlyDSL/ directory):
    # Default GPT-OSS shapes (TP=8)
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py

    # Sweep decode + prefill batch sizes
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py --sweep

    # FlyDSL only (no AITER required)
    PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py --flydsl-only

    # Custom shape
    PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py --tokens 128 --stage 1

Requirements:
    - gfx942/gfx950 GPU
    - FlyDSL installed (pip install -e .)
    - AITER importable for CK baseline (set AITER_REPO env var or pip install -e ../aiter)
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

# ---------------------------------------------------------------------------
# GPT-OSS 120B MoE config
# ---------------------------------------------------------------------------
MODEL_DIM = 2880
INTER_DIM = 2880
NUM_EXPERTS = 128
TOPK = 4
BLOCK_M = 32

# Inference scenario: prompt=1K, output=8K, decode-dominated
# M = concurrency in decode (1 token/request/step), or prompt_len in prefill
DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]   # concurrency points
PREFILL_TOKENS = [1024]                            # 1K prompt (processed once)
SWEEP_TOKENS = DECODE_TOKENS + PREFILL_TOKENS


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
@dataclass
class MoeBenchRow:
    stage: int
    tokens: int
    ck_us: Optional[float]
    flydsl_us: Optional[float]

    @property
    def speedup(self) -> Optional[float]:
        if self.ck_us is None or self.flydsl_us is None:
            return None
        return self.ck_us / self.flydsl_us

    @property
    def flops(self) -> int:
        # MoE GEMM: each token hits topk experts
        # Stage1: gate+up fused = 2 * tokens * topk * (2*inter_dim) * model_dim
        # Stage2: down = 2 * tokens * topk * model_dim * inter_dim
        if self.stage == 1:
            return 2 * self.tokens * TOPK * (2 * INTER_DIM) * MODEL_DIM
        else:
            return 2 * self.tokens * TOPK * MODEL_DIM * INTER_DIM

    @property
    def ck_tflops(self) -> Optional[float]:
        if self.ck_us is None:
            return None
        return self.flops / (self.ck_us / 1e6) / 1e12

    @property
    def flydsl_tflops(self) -> Optional[float]:
        if self.flydsl_us is None:
            return None
        return self.flops / (self.flydsl_us / 1e6) / 1e12


# ---------------------------------------------------------------------------
# Routing buffer setup (shared between CK and FlyDSL)
# ---------------------------------------------------------------------------
def build_routing(tokens: int, device: torch.device):
    """Build MoE routing buffers using pure-torch fallback."""
    topk_ids = torch.randint(0, NUM_EXPERTS, (tokens, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(tokens, TOPK, device=device, dtype=torch.float32), dim=1
    )

    # Try AITER moe_sorting first, fallback to torch implementation
    try:
        from tests.kernels.test_moe_gemm import build_routing_buffers
        sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, sorted_size, blocks = (
            build_routing_buffers(
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                experts=NUM_EXPERTS,
                model_dim=MODEL_DIM,
                tile_m=BLOCK_M,
            )
        )
        return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids
    except Exception as e:
        print(f"  [routing] build_routing_buffers failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CK MoE benchmark (via AITER)
# ---------------------------------------------------------------------------
def bench_ck_moe_stage1(
    tokens: int, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark CK MoE stage1 (gate+up GEMM)."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.moe_op import ck_moe_stage1_fwd
        from aiter.ops.enum import QuantType, ActivationType
    except ImportError:
        print("  [CK] Could not import ck_moe_stage1_fwd")
        return None

    device = torch.device("cuda")
    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing

    hidden = torch.randn(tokens, MODEL_DIM, dtype=torch.float16, device=device)
    w1 = torch.randn(NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM, dtype=torch.float16, device=device)
    w2 = torch.randn(NUM_EXPERTS, MODEL_DIM, INTER_DIM, dtype=torch.float16, device=device)
    out = torch.empty(tokens, TOPK, INTER_DIM, dtype=torch.float16, device=device)

    def run():
        ck_moe_stage1_fwd(
            hidden_states=hidden, w1=w1, w2=w2,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=out, topk=TOPK,
            quant_type=QuantType.No,
            activation=ActivationType.Silu,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [CK] Stage1 failed: {e}")
        return None


def bench_ck_moe_stage2(
    tokens: int, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark CK MoE stage2 (down GEMM)."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.moe_op import ck_moe_stage2_fwd
        from aiter.ops.enum import QuantType, ActivationType
    except ImportError:
        print("  [CK] Could not import ck_moe_stage2_fwd")
        return None

    device = torch.device("cuda")
    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing

    inter_states = torch.randn(tokens, TOPK, INTER_DIM, dtype=torch.float16, device=device)
    w1 = torch.randn(NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM, dtype=torch.float16, device=device)
    w2 = torch.randn(NUM_EXPERTS, MODEL_DIM, INTER_DIM, dtype=torch.float16, device=device)
    out = torch.zeros(tokens, MODEL_DIM, dtype=torch.float16, device=device)

    def run():
        out.zero_()
        ck_moe_stage2_fwd(
            inter_states=inter_states, w1=w1, w2=w2,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            out=out, topk=TOPK,
            sorted_weights=sorted_w,
            quant_type=QuantType.No,
            activation=ActivationType.Silu,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [CK] Stage2 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FlyDSL MoE benchmark
# ---------------------------------------------------------------------------
def bench_flydsl_moe_stage1(
    tokens: int, warmup: int, iters: int,
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
    in_dtype: str = "fp8",
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage1 (gate+up GEMM)."""
    try:
        from kernels.moe_gemm_2stage import compile_moe_gemm1
    except ImportError:
        print("  [FlyDSL] Could not import compile_moe_gemm1")
        return None

    device = torch.device("cuda")

    try:
        from flydsl.runtime.device import get_rocm_arch
        arch = str(get_rocm_arch())
        dtype_fp8 = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz
    except Exception:
        dtype_fp8 = torch.float8_e4m3fn

    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing

    from tests.utils import pertoken_quant, shuffle_weight

    # Create FP8 inputs with per-token quantization
    a_fp32 = torch.randn(tokens, MODEL_DIM, device=device, dtype=torch.float32)
    a_q, scale_a = pertoken_quant(a_fp32, quant_dtype=dtype_fp8)

    w1_fp32 = torch.randn(NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM, device=device, dtype=torch.float32)
    # Per-row quantize and shuffle each expert's weights
    w1_list = []
    scale_w1_list = []
    for e in range(NUM_EXPERTS):
        wq, sw = pertoken_quant(w1_fp32[e], quant_dtype=dtype_fp8)
        w1_list.append(shuffle_weight(wq, layout=(16, 16)))
        scale_w1_list.append(sw)
    w1_shuffled = torch.stack(w1_list)
    scale_w1 = torch.stack(scale_w1_list).squeeze(-1)

    sorted_size = int(sorted_ids.numel())
    blocks = int(sorted_expert_ids.numel())

    try:
        launch_fn = compile_moe_gemm1(
            M=sorted_size, N=2 * INTER_DIM, K=MODEL_DIM,
            E=NUM_EXPERTS, topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype=in_dtype, out_dtype="bf16",
            num_blocks=blocks,
        )
    except Exception as e:
        print(f"  [FlyDSL] Stage1 compile failed: {e}")
        return None

    out = torch.empty(tokens * TOPK, INTER_DIM, dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream()

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def run():
        launch_fn(
            out.contiguous().view(-1),
            _as_i8(a_q).contiguous().view(-1),
            _as_i8(w1_shuffled).contiguous().view(-1),
            scale_a.contiguous().view(-1),
            scale_w1.contiguous().view(-1),
            sorted_ids.contiguous().view(-1),
            sorted_expert_ids.contiguous().view(-1),
            num_valid_ids.contiguous().view(-1),
            sorted_size, blocks,
            stream,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] Stage1 failed: {e}")
        return None


def bench_flydsl_moe_stage2(
    tokens: int, warmup: int, iters: int,
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
    in_dtype: str = "fp8",
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage2 (down GEMM)."""
    try:
        from kernels.moe_gemm_2stage import compile_moe_gemm2, MoeGemm2Mode
    except ImportError:
        print("  [FlyDSL] Could not import compile_moe_gemm2")
        return None

    device = torch.device("cuda")

    try:
        from flydsl.runtime.device import get_rocm_arch
        arch = str(get_rocm_arch())
        dtype_fp8 = torch.float8_e4m3fn if "gfx95" in arch else torch.float8_e4m3fnuz
    except Exception:
        dtype_fp8 = torch.float8_e4m3fn

    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing

    from tests.utils import pertoken_quant, shuffle_weight

    # Inter states (stage1 output) — FP8 quantized
    inter_fp32 = torch.randn(tokens * TOPK, INTER_DIM, device=device, dtype=torch.float32)
    inter_q, scale_a2 = pertoken_quant(inter_fp32, quant_dtype=dtype_fp8)

    # W2 weights — per-row quantized and shuffled
    w2_fp32 = torch.randn(NUM_EXPERTS, MODEL_DIM, INTER_DIM, device=device, dtype=torch.float32)
    w2_list = []
    scale_w2_list = []
    for e in range(NUM_EXPERTS):
        wq, sw = pertoken_quant(w2_fp32[e], quant_dtype=dtype_fp8)
        w2_list.append(shuffle_weight(wq, layout=(16, 16)))
        scale_w2_list.append(sw)
    w2_shuffled = torch.stack(w2_list)
    scale_w2 = torch.stack(scale_w2_list).squeeze(-1)

    sorted_size = int(sorted_ids.numel())
    blocks = int(sorted_expert_ids.numel())

    try:
        launch_fn = compile_moe_gemm2(
            M=sorted_size, N=MODEL_DIM, K=INTER_DIM,
            E=NUM_EXPERTS, topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            in_dtype=in_dtype, out_dtype="bf16",
            num_blocks=blocks,
            mode=MoeGemm2Mode.ATOMIC,
        )
    except Exception as e:
        print(f"  [FlyDSL] Stage2 compile failed: {e}")
        return None

    out = torch.zeros(tokens, MODEL_DIM, dtype=torch.bfloat16, device=device)
    stream = torch.cuda.current_stream()

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def run():
        out.zero_()
        launch_fn(
            out.contiguous().view(-1),
            _as_i8(inter_q).contiguous().view(-1),
            _as_i8(w2_shuffled).contiguous().view(-1),
            scale_a2.contiguous().view(-1),
            scale_w2.contiguous().view(-1),
            sorted_ids.contiguous().view(-1),
            sorted_expert_ids.contiguous().view(-1),
            num_valid_ids.contiguous().view(-1),
            sorted_w.contiguous().view(-1),
            sorted_size, blocks,
            stream,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] Stage2 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_results(rows: List[MoeBenchRow]) -> None:
    print()
    print("=" * 110)
    print(f"MoE Benchmark: CK (AITER) vs FlyDSL — GPT-OSS 120B (E={NUM_EXPERTS}, topk={TOPK}, "
          f"hidden={MODEL_DIM}, inter={INTER_DIM})")
    print("=" * 110)
    hdr = (
        f"{'Stage':>6s}  {'Tokens':>7s}  "
        f"{'CK(us)':>10s}  {'TFLOPS':>8s}  "
        f"{'FlyDSL(us)':>12s}  {'TFLOPS':>8s}  "
        f"{'Speedup':>8s}"
    )
    print(hdr)
    print("-" * 110)

    for r in rows:
        ck_us = f"{r.ck_us:10.1f}" if r.ck_us else f"{'N/A':>10s}"
        ck_tf = f"{r.ck_tflops:8.2f}" if r.ck_tflops else f"{'N/A':>8s}"
        fly_us = f"{r.flydsl_us:12.1f}" if r.flydsl_us else f"{'N/A':>12s}"
        fly_tf = f"{r.flydsl_tflops:8.2f}" if r.flydsl_tflops else f"{'N/A':>8s}"
        sp = f"{r.speedup:7.2f}x" if r.speedup else f"{'N/A':>8s}"
        print(f"{'S'+str(r.stage):>6s}  {r.tokens:7d}  {ck_us}  {ck_tf}  {fly_us}  {fly_tf}  {sp}")

    print("=" * 110)
    valid = [r for r in rows if r.speedup is not None]
    if valid:
        geo = 1.0
        for r in valid:
            geo *= r.speedup
        geo = geo ** (1.0 / len(valid))
        avg = sum(r.speedup for r in valid) / len(valid)
        print(f"Geomean speedup (FlyDSL/CK): {geo:.2f}x  |  Average: {avg:.2f}x")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MoE benchmark: CK vs FlyDSL")
    parser.add_argument("--tokens", type=str, default="1,32,128",
                        help="Comma-separated token counts")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all decode+prefill batch sizes")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2],
                        help="Stage to benchmark (0=both, 1=stage1, 2=stage2)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true",
                        help="Skip CK baseline (no AITER needed)")
    parser.add_argument("--in-dtype", type=str, default="fp8",
                        choices=["fp8", "fp16", "bf16"],
                        help="Input dtype for FlyDSL kernels")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    if args.sweep:
        token_list = SWEEP_TOKENS
    else:
        token_list = [int(t.strip()) for t in args.tokens.split(",")]

    rows: List[MoeBenchRow] = []
    stages = [1, 2] if args.stage == 0 else [args.stage]

    for tokens in token_list:
        for stage in stages:
            print(f"Benchmarking Stage{stage}, tokens={tokens} ...")

            ck_us = None
            if not args.flydsl_only:
                if stage == 1:
                    ck_us = bench_ck_moe_stage1(tokens, args.warmup, args.iters)
                else:
                    ck_us = bench_ck_moe_stage2(tokens, args.warmup, args.iters)

            if stage == 1:
                fly_us = bench_flydsl_moe_stage1(
                    tokens, args.warmup, args.iters, in_dtype=args.in_dtype,
                )
            else:
                fly_us = bench_flydsl_moe_stage2(
                    tokens, args.warmup, args.iters, in_dtype=args.in_dtype,
                )

            rows.append(MoeBenchRow(
                stage=stage, tokens=tokens, ck_us=ck_us, flydsl_us=fly_us,
            ))

    print_results(rows)


if __name__ == "__main__":
    main()
