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
    label: str       # "CK fused", "FlyDSL S1", "FlyDSL S2", "FlyDSL S1+S2"
    tokens: int
    us: Optional[float]

    @property
    def flops(self) -> int:
        # Total MoE FLOPs (both stages combined):
        # S1: 2 * tokens * topk * (2*inter) * model
        # S2: 2 * tokens * topk * model * inter
        s1 = 2 * self.tokens * TOPK * (2 * INTER_DIM) * MODEL_DIM
        s2 = 2 * self.tokens * TOPK * MODEL_DIM * INTER_DIM
        if "S1+S2" in self.label or "fused" in self.label:
            return s1 + s2
        elif "S1" in self.label:
            return s1
        elif "S2" in self.label:
            return s2
        return s1 + s2

    @property
    def tflops(self) -> Optional[float]:
        if self.us is None:
            return None
        return self.flops / (self.us / 1e6) / 1e12


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
def bench_ck_moe_fused(
    tokens: int, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark CK fused_moe (stage1+stage2 combined) with MXFP4 (A4W4).

    GPT-OSS uses MXFP4 for MoE: per-1x32 block scales, FP4 packed weights.
    CK's fused_moe handles sorting + stage1 + stage2 internally.
    """
    if not maybe_enable_aiter():
        return None
    try:
        import aiter
        from aiter.fused_moe import fused_moe
        from aiter.ops.enum import QuantType, ActivationType
    except ImportError:
        print("  [CK] Could not import aiter.fused_moe")
        return None

    device = torch.device("cuda")

    # MXFP4 (float4_e2m1fn_x2) weights — packed FP4, 2 elements per byte
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    w1 = torch.randint(0, 256, (NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w2 = torch.randint(0, 256, (NUM_EXPERTS, MODEL_DIM, INTER_DIM // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    # E8M0 block scales (per 32 elements)
    w1_scale = torch.randint(124, 130, (NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM // 32),
                              dtype=torch.uint8, device=device)
    w2_scale = torch.randint(124, 130, (NUM_EXPERTS, MODEL_DIM, INTER_DIM // 32),
                              dtype=torch.uint8, device=device)

    # Input: BF16 hidden states (A4 quantization happens inside fused_moe)
    hidden = torch.randn(tokens, MODEL_DIM, dtype=torch.bfloat16, device=device)

    # Router: compute topk routing outside fused_moe
    router_logits = torch.randn(tokens, NUM_EXPERTS, dtype=torch.float32, device=device)
    topk_vals, topk_ids = torch.topk(router_logits, k=TOPK, dim=1)
    topk_weight = torch.softmax(topk_vals, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    def run():
        fused_moe(
            hidden_states=hidden,
            w1=w1, w2=w2,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            w1_scale=w1_scale, w2_scale=w2_scale,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        import traceback
        print(f"  [CK] fused_moe failed: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# FlyDSL MoE benchmark
# ---------------------------------------------------------------------------
def bench_flydsl_moe_stage1(
    tokens: int, warmup: int, iters: int,
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
    in_dtype: str = "fp8",
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage1 (gate+up GEMM).

    Follows the same setup as tests/kernels/test_moe_gemm.py::run_moe_stage1.
    """
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
    sorted_size = int(sorted_ids.numel())
    blocks = int(sorted_expert_ids.numel())

    from tests.utils import pertoken_quant, shuffle_weight

    # Quantize and preshuffle (matching test_moe_gemm.py setup)
    x_fp32 = torch.randn(tokens, MODEL_DIM, device=device, dtype=torch.float32)
    w1_fp32 = torch.randn(NUM_EXPERTS, 2 * INTER_DIM, MODEL_DIM, device=device, dtype=torch.float32)

    if in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=dtype_fp8)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=dtype_fp8)
    elif in_dtype in ("fp16", "bf16"):
        torch_dt = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        x_q = x_fp32.to(torch_dt)
        w1_q = w1_fp32.to(torch_dt)
        scale_x = None
        scale_w1 = None
    else:
        print(f"  [FlyDSL] Unsupported in_dtype={in_dtype}")
        return None

    # Preshuffle weights and flatten expert dim
    w1_shuffled = shuffle_weight(w1_q)
    w_kernel = w1_shuffled.view(NUM_EXPERTS * (2 * INTER_DIM), MODEL_DIM).contiguous()

    # Flatten scales to 1D
    if scale_x is None:
        scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_x_1d = scale_x.view(-1).contiguous()
    if scale_w1 is None:
        scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_w1_1d = scale_w1.view(-1).contiguous()
    sorted_w_1d = sorted_w.contiguous().view(-1)

    try:
        exe = compile_moe_gemm1(
            model_dim=MODEL_DIM,
            inter_dim=INTER_DIM,
            experts=NUM_EXPERTS,
            topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage1=False,
            in_dtype=in_dtype,
            out_dtype="f16",
        )
    except Exception as e:
        print(f"  [FlyDSL] Stage1 compile failed: {e}")
        return None

    out = torch.empty(tokens, TOPK, INTER_DIM, dtype=torch.float16, device=device)

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def run():
        stream = torch.cuda.current_stream()
        exe(
            out, _as_i8(x_q), _as_i8(w_kernel),
            scale_x_1d, scale_w1_1d,
            sorted_ids, sorted_expert_ids, sorted_w_1d,
            num_valid_ids,
            tokens, INTER_DIM, MODEL_DIM, blocks,
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
    """Benchmark FlyDSL MoE stage2 (down GEMM).

    Follows the same setup as tests/kernels/test_moe_gemm.py::run_moe_stage2.
    """
    try:
        from kernels.moe_gemm_2stage import compile_moe_gemm2
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
    sorted_size = int(sorted_ids.numel())
    blocks = int(sorted_expert_ids.numel())

    from tests.utils import pertoken_quant, shuffle_weight

    # Stage2 input: inter_states from stage1 output
    inter_fp32 = torch.randn(tokens * TOPK, INTER_DIM, device=device, dtype=torch.float32)
    w2_fp32 = torch.randn(NUM_EXPERTS, MODEL_DIM, INTER_DIM, device=device, dtype=torch.float32)

    if in_dtype == "fp8":
        a2_q, scale_a2 = pertoken_quant(inter_fp32, quant_dtype=dtype_fp8)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=dtype_fp8)
    elif in_dtype in ("fp16", "bf16"):
        torch_dt = torch.float16 if in_dtype == "fp16" else torch.bfloat16
        a2_q = inter_fp32.to(torch_dt)
        w2_q = w2_fp32.to(torch_dt)
        scale_a2 = None
        scale_w2 = None
    else:
        print(f"  [FlyDSL] Unsupported in_dtype={in_dtype}")
        return None

    # Preshuffle W2 and flatten
    w2_shuffled = shuffle_weight(w2_q)
    w2_kernel = w2_shuffled.view(NUM_EXPERTS * MODEL_DIM, INTER_DIM).contiguous()

    if scale_a2 is None:
        scale_a2_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_a2_1d = scale_a2.view(-1).contiguous()
    if scale_w2 is None:
        scale_w2_1d = torch.empty((0,), device=device, dtype=torch.float32)
    else:
        scale_w2_1d = scale_w2.view(-1).contiguous()
    sorted_w_1d = sorted_w.contiguous().view(-1)

    try:
        exe = compile_moe_gemm2(
            model_dim=MODEL_DIM,
            inter_dim=INTER_DIM,
            experts=NUM_EXPERTS,
            topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage2=True,
            in_dtype=in_dtype,
            out_dtype="f16",
            accumulate=True,
        )
    except Exception as e:
        print(f"  [FlyDSL] Stage2 compile failed: {e}")
        return None

    out = torch.zeros(tokens, MODEL_DIM, dtype=torch.float16, device=device)

    def _as_i8(t):
        return t.view(torch.int8) if "float8" in str(t.dtype) else t

    def run():
        out.zero_()
        stream = torch.cuda.current_stream()
        exe(
            out, _as_i8(a2_q), _as_i8(w2_kernel),
            scale_a2_1d, scale_w2_1d,
            sorted_ids, sorted_expert_ids, sorted_w_1d,
            num_valid_ids,
            tokens, MODEL_DIM, INTER_DIM, blocks,
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
    print("=" * 90)
    print(f"MoE Benchmark — GPT-OSS 120B A4W4 (E={NUM_EXPERTS}, topk={TOPK}, "
          f"hidden={MODEL_DIM}, inter={INTER_DIM})")
    print("=" * 90)
    hdr = f"{'Tokens':>7s}  {'Kernel':<16s}  {'Latency(us)':>12s}  {'TFLOPS':>8s}"
    print(hdr)
    print("-" * 90)

    # Group by token count
    by_tokens: dict[int, List[MoeBenchRow]] = {}
    for r in rows:
        by_tokens.setdefault(r.tokens, []).append(r)

    for tokens in sorted(by_tokens.keys()):
        group = by_tokens[tokens]
        for i, r in enumerate(group):
            tok_col = f"{tokens:7d}" if i == 0 else f"{'':7s}"
            us_s = f"{r.us:12.1f}" if r.us else f"{'N/A':>12s}"
            tf_s = f"{r.tflops:8.2f}" if r.tflops else f"{'N/A':>8s}"
            print(f"{tok_col}  {r.label:<16s}  {us_s}  {tf_s}")
        # Print speedup if we have both CK fused and FlyDSL S1+S2
        ck_row = next((r for r in group if "CK" in r.label and r.us), None)
        s1_row = next((r for r in group if "S1" in r.label and "S2" not in r.label and r.us), None)
        s2_row = next((r for r in group if "S2" in r.label and "S1" not in r.label and r.us), None)
        if ck_row and s1_row and s2_row:
            fly_total = s1_row.us + s2_row.us
            sp = ck_row.us / fly_total
            print(f"{'':7s}  {'→ FlyDSL/CK':<16s}  {fly_total:12.1f}  {sp:7.2f}x")
        print()

    print("=" * 90)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MoE benchmark: CK A4W4 vs FlyDSL")
    parser.add_argument("--tokens", type=str, default="1,32,128",
                        help="Comma-separated token counts")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all decode+prefill batch sizes")
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

    for tokens in token_list:
        print(f"Benchmarking tokens={tokens} ...")

        # CK: fused_moe (stage1+stage2 combined, MXFP4 A4W4)
        if not args.flydsl_only:
            ck_us = bench_ck_moe_fused(tokens, args.warmup, args.iters)
            rows.append(MoeBenchRow(label="CK fused", tokens=tokens, us=ck_us))

        # FlyDSL: stage1 and stage2 separately
        s1_us = bench_flydsl_moe_stage1(
            tokens, args.warmup, args.iters, in_dtype=args.in_dtype,
        )
        rows.append(MoeBenchRow(label="FlyDSL S1", tokens=tokens, us=s1_us))

        s2_us = bench_flydsl_moe_stage2(
            tokens, args.warmup, args.iters, in_dtype=args.in_dtype,
        )
        rows.append(MoeBenchRow(label="FlyDSL S2", tokens=tokens, us=s2_us))

    print_results(rows)


if __name__ == "__main__":
    main()
