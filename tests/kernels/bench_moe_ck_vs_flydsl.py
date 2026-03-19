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

# MXFP4 requires 256-byte alignment; Mxfp4MoEMethod pads dimensions
MXFP4_PAD_ALIGN = 256
MODEL_DIM_PADDED = (MODEL_DIM + MXFP4_PAD_ALIGN - 1) // MXFP4_PAD_ALIGN * MXFP4_PAD_ALIGN  # 3072
INTER_DIM_PADDED = (INTER_DIM + MXFP4_PAD_ALIGN - 1) // MXFP4_PAD_ALIGN * MXFP4_PAD_ALIGN  # 3072

# MoE is a per-token FFN — performance depends only on M (number of tokens),
# not on sequence length. Prefill M=1024 and decode concurrency=1024 are identical
# from MoE's perspective. In decode-dominated serving (1K prompt, 8K output),
# M = concurrency (number of concurrent requests generating 1 token each).
SWEEP_CONCURRENCY = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


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
# Sorting + activation quant benchmark (overhead shared by both CK and FlyDSL)
# ---------------------------------------------------------------------------
def bench_sorting_overhead(
    tokens: int, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark MoE sorting + MXFP4 activation quantization overhead."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.fused_moe import moe_sorting
    except ImportError:
        return None

    device = torch.device("cuda")
    topk_ids = torch.randint(0, NUM_EXPERTS, (tokens, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(tokens, TOPK, device=device, dtype=torch.float32), dim=1
    )

    def run():
        moe_sorting(topk_ids, topk_weights, NUM_EXPERTS, MODEL_DIM, torch.float16, BLOCK_M)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CK MoE benchmark (via AITER)
# ---------------------------------------------------------------------------
def _prepare_ck_moe_inputs(tokens: int, device: torch.device):
    """Prepare CK MoE inputs: sorting + MXFP4 activation quantization.

    Returns all tensors needed to call ck_moe_stage1_fwd / ck_moe_stage2_fwd
    separately, matching the exact flow in aiter/fused_moe.py::fused_moe_2stages.
    """
    import aiter
    from aiter.fused_moe import moe_sorting
    from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort

    H = MODEL_DIM_PADDED   # 3072
    I = INTER_DIM_PADDED   # 3072
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    # Weights (MXFP4)
    w1 = torch.randint(0, 256, (NUM_EXPERTS, 2 * I, H // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w2 = torch.randint(0, 256, (NUM_EXPERTS, H, I // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w1_scale = torch.randint(124, 130, (NUM_EXPERTS, 2 * I, H // 32),
                              dtype=torch.uint8, device=device)
    w2_scale = torch.randint(124, 130, (NUM_EXPERTS, H, I // 32),
                              dtype=torch.uint8, device=device)

    # Hidden states (BF16)
    hidden = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)

    # Routing
    router_logits = torch.randn(tokens, NUM_EXPERTS, dtype=torch.float32, device=device)
    topk_vals, topk_ids = torch.topk(router_logits, k=TOPK, dim=1)
    topk_weight = torch.softmax(topk_vals, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    # Sorting (same as fused_moe_2stages)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weight, NUM_EXPERTS, H, torch.bfloat16, BLOCK_M,
    )

    # MXFP4 activation quantization (BF16 → FP4 + E8M0 scales)
    a1, a1_scale = fused_dynamic_mxfp4_quant_moe_sort(
        hidden, sorted_ids=sorted_ids, num_valid_ids=num_valid_ids,
        token_num=tokens, topk=1, block_size=BLOCK_M,
    )

    return {
        "w1": w1, "w2": w2,
        "w1_scale": w1_scale.view(fp8_e8m0), "w2_scale": w2_scale.view(fp8_e8m0),
        "sorted_ids": sorted_ids, "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids, "num_valid_ids": num_valid_ids,
        "a1": a1, "a1_scale": a1_scale,
        "tokens": tokens, "H": H, "I": I,
    }


def bench_ck_moe_fused(
    tokens: int, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark CK fused_moe (sorting + quant + S1 + S2 combined)."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.fused_moe import fused_moe
        from aiter.ops.enum import QuantType, ActivationType
    except ImportError:
        print("  [CK] Could not import aiter.fused_moe")
        return None

    device = torch.device("cuda")
    H = MODEL_DIM_PADDED
    I = INTER_DIM_PADDED
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)

    w1 = torch.randint(0, 256, (NUM_EXPERTS, 2 * I, H // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w2 = torch.randint(0, 256, (NUM_EXPERTS, H, I // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w1_scale = torch.randint(124, 130, (NUM_EXPERTS, 2 * I, H // 32),
                              dtype=torch.uint8, device=device)
    w2_scale = torch.randint(124, 130, (NUM_EXPERTS, H, I // 32),
                              dtype=torch.uint8, device=device)
    hidden = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(tokens, NUM_EXPERTS, dtype=torch.float32, device=device)
    topk_vals, topk_ids = torch.topk(router_logits, k=TOPK, dim=1)
    topk_weight = torch.softmax(topk_vals, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    def run():
        fused_moe(
            hidden_states=hidden, w1=w1, w2=w2,
            topk_weight=topk_weight, topk_ids=topk_ids,
            w1_scale=w1_scale, w2_scale=w2_scale,
            quant_type=QuantType.per_1x32, activation=ActivationType.Silu,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        import traceback
        print(f"  [CK] fused_moe failed: {e}")
        traceback.print_exc()
        return None


def bench_ck_moe_stage1(
    tokens: int, warmup: int, iters: int,
) -> Optional[float]:
    """Benchmark CK MoE stage1 only (kernel-only, pre-quantized inputs)."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.moe_op import ck_moe_stage1_fwd
        from aiter.ops.enum import QuantType, ActivationType
    except ImportError:
        print("  [CK] Could not import ck_moe_stage1_fwd")
        return None

    device = torch.device("cuda")
    try:
        inp = _prepare_ck_moe_inputs(tokens, device)
    except Exception as e:
        print(f"  [CK] Input preparation failed: {e}")
        return None

    H, I = inp["H"], inp["I"]
    out = torch.empty(tokens, TOPK, I, dtype=torch.bfloat16, device=device)

    def run():
        ck_moe_stage1_fwd(
            hidden_states=inp["a1"], w1=inp["w1"], w2=inp["w2"],
            sorted_token_ids=inp["sorted_ids"],
            sorted_expert_ids=inp["sorted_expert_ids"],
            num_valid_ids=inp["num_valid_ids"],
            out=out, topk=TOPK,
            w1_scale=inp["w1_scale"], a1_scale=inp["a1_scale"],
            quant_type=QuantType.per_1x32,
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
    """Benchmark CK MoE stage2 only (kernel-only, pre-quantized inputs)."""
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.moe_op import ck_moe_stage2_fwd
        from aiter.ops.enum import QuantType, ActivationType
        from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
    except ImportError:
        print("  [CK] Could not import ck_moe_stage2_fwd")
        return None

    device = torch.device("cuda")
    try:
        inp = _prepare_ck_moe_inputs(tokens, device)
    except Exception as e:
        print(f"  [CK] Input preparation failed: {e}")
        return None

    H, I = inp["H"], inp["I"]

    # Stage2 input: simulated stage1 output (BF16 intermediate), then quantized to FP4
    a2_bf16 = torch.randn(tokens, TOPK, I, dtype=torch.bfloat16, device=device)
    a2_flat = a2_bf16.view(-1, I)
    a2, a2_scale = fused_dynamic_mxfp4_quant_moe_sort(
        a2_flat, sorted_ids=inp["sorted_ids"], num_valid_ids=inp["num_valid_ids"],
        token_num=tokens, topk=TOPK, block_size=BLOCK_M,
    )
    a2 = a2.view(tokens, TOPK, -1)

    out = torch.zeros(tokens, H, dtype=torch.bfloat16, device=device)

    def run():
        out.zero_()
        ck_moe_stage2_fwd(
            inter_states=a2, w1=inp["w1"], w2=inp["w2"],
            sorted_token_ids=inp["sorted_ids"],
            sorted_expert_ids=inp["sorted_expert_ids"],
            num_valid_ids=inp["num_valid_ids"],
            out=out, topk=TOPK,
            w2_scale=inp["w2_scale"], a2_scale=a2_scale,
            sorted_weights=inp["sorted_weights"],
            quant_type=QuantType.per_1x32,
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
    hdr = f"{'M(conc)':>7s}  {'Kernel':<16s}  {'Latency(us)':>12s}  {'TFLOPS':>8s}"
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
        # Print speedup comparisons
        ck_s1 = next((r for r in group if r.label == "CK S1" and r.us), None)
        ck_s2 = next((r for r in group if r.label == "CK S2" and r.us), None)
        fly_s1 = next((r for r in group if r.label == "FlyDSL S1" and r.us), None)
        fly_s2 = next((r for r in group if r.label == "FlyDSL S2" and r.us), None)

        # S1 vs S1
        if ck_s1 and fly_s1:
            sp = ck_s1.us / fly_s1.us
            print(f"{'':7s}  {'S1: FlyDSL/CK':<16s}  {'':12s}  {sp:7.2f}x")
        # S2 vs S2
        if ck_s2 and fly_s2:
            sp = ck_s2.us / fly_s2.us
            print(f"{'':7s}  {'S2: FlyDSL/CK':<16s}  {'':12s}  {sp:7.2f}x")
        print()

    print("=" * 90)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MoE benchmark: CK A4W4 vs FlyDSL")
    parser.add_argument("--concurrency", type=str, default="1,32,128",
                        help="Comma-separated M values (decode concurrency or prefill tokens)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all concurrency points (1..1024)")
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
        token_list = SWEEP_CONCURRENCY
    else:
        token_list = [int(t.strip()) for t in args.concurrency.split(",")]

    rows: List[MoeBenchRow] = []

    for tokens in token_list:
        print(f"Benchmarking tokens={tokens} ...")

        if not args.flydsl_only:
            # CK stage1 kernel-only (pre-quantized MXFP4 inputs)
            ck_s1 = bench_ck_moe_stage1(tokens, args.warmup, args.iters)
            rows.append(MoeBenchRow(label="CK S1", tokens=tokens, us=ck_s1))

            # CK stage2 kernel-only
            ck_s2 = bench_ck_moe_stage2(tokens, args.warmup, args.iters)
            rows.append(MoeBenchRow(label="CK S2", tokens=tokens, us=ck_s2))

        # FlyDSL: stage1 and stage2 separately (kernel-only)
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
