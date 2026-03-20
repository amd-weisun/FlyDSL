#!/usr/bin/env python3
"""Benchmark: CK fused_moe vs FlyDSL MoE (Stage1 + Stage2).

MoE kernels are the #1 GPU time consumer in GPT-OSS 120B (35% of GPU time).
This benchmark compares CK (via AITER) vs FlyDSL for both MoE stages.

Profiling trace confirms GPT-OSS uses A16W4 (BF16 activations × MXFP4 weights):
  - CK kernel: ck_tile::MoeFlatmmKernel<F16xMXF4FlatmmPipeline...>
  - Tile shape: 16×128×256
  - Both stages use BF16 activations (no re-quant between stages for Swiglu+FP4)
  - Padded dimensions: 3072×3072 (from 2880, 256-aligned for MXFP4)

Usage (from FlyDSL/ directory):
    # Default: A16W4 (matches production trace)
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py

    # Sweep all concurrency points
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py --sweep

    # Compare all precision variants
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py --precision all

    # FlyDSL only (no AITER required)
    PYTHONPATH=./ python tests/kernels/bench_moe_ck_vs_flydsl.py --flydsl-only

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

# Reuse battle-tested helpers from the existing MoE test harness.
# We do NOT modify test_moe_gemm.py — only import from it.
from tests.kernels.test_moe_gemm import (
    build_routing_buffers,
    run_moe_stage1,
    run_moe_stage2,
)

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
# Routing buffer setup (delegates to test_moe_gemm.py)
# ---------------------------------------------------------------------------
def build_routing(tokens: int, device: torch.device, model_dim: int = None):
    """Build MoE routing buffers (reuses test_moe_gemm.py::build_routing_buffers)."""
    if model_dim is None:
        model_dim = MODEL_DIM_PADDED
    topk_ids = torch.randint(0, NUM_EXPERTS, (tokens, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.softmax(
        torch.randn(tokens, TOPK, device=device, dtype=torch.float32), dim=1
    )
    try:
        routing = build_routing_buffers(
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            experts=NUM_EXPERTS,
            model_dim=model_dim,
            tile_m=BLOCK_M,
        )
        sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids, sorted_size, blocks = routing
        return sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids
    except Exception as e:
        print(f"  [routing] build_routing_buffers failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FP8 benchmark via existing test_moe_gemm.py (reuse, not reinvent)
# ---------------------------------------------------------------------------
def bench_fp8_via_test_harness(
    tokens: int, stage: int, iters: int, warmup: int,
) -> Tuple[Optional[float], Optional[float]]:
    """Run FlyDSL + CK comparison for FP8 using existing test_moe_gemm.py.

    Returns (flydsl_us, ck_us) or (None, None) on failure.
    Uses run_moe_stage1/run_moe_stage2 which already handles:
    - pertoken_quant, shuffle_weight, preshuffle
    - FlyDSL kernel compile + benchmark
    - CK (ck_moe_stage1_fwd/ck_moe_stage2_fwd) comparison
    - Correctness verification (FlyDSL vs torch ref, FlyDSL vs CK)
    """
    H = MODEL_DIM_PADDED
    I = INTER_DIM_PADDED
    runner = run_moe_stage1 if stage == 1 else run_moe_stage2
    try:
        result = runner(
            tokens=tokens,
            model_dim=H, inter_dim=I,
            experts=NUM_EXPERTS, topk=TOPK,
            tile_m=BLOCK_M, tile_n=128, tile_k=128,
            doweight_stage1=True,
            in_dtype="fp8",
            num_iters=iters, num_warmup=warmup,
            compare_aiter_ck=True,
            return_outputs=True,
        )
        if result is not None:
            _, us = result
            return us, None  # CK timing is printed by the harness but not returned
        return None, None
    except Exception as e:
        print(f"  [FP8 harness] Stage{stage} failed: {e}")
        return None, None


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
# FlyDSL MoE standalone benchmark helpers (fallback when mixed_moe_gemm is unavailable)
# ---------------------------------------------------------------------------
def _bench_flydsl_moe_stage1_standalone(
    tokens: int, warmup: int, iters: int,
    in_dtype: str = "fp8",
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage1 using the standalone compile_moe_gemm1 kernel."""
    from tests.utils import pertoken_quant, shuffle_weight
    from kernels.moe_gemm_2stage import compile_moe_gemm1

    device = torch.device("cuda")
    H = MODEL_DIM_PADDED
    I = INTER_DIM_PADDED

    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing
    blocks = int(sorted_expert_ids.numel())

    is_f16_or_bf16 = in_dtype in ("fp16", "bf16")
    x_fp32 = torch.randn(tokens, H, device=device, dtype=torch.float32)
    w1_fp32 = torch.randn(NUM_EXPERTS, 2 * I, H, device=device, dtype=torch.float32)

    if in_dtype == "fp8":
        x_q, scale_x = pertoken_quant(x_fp32, quant_dtype=torch.float8_e4m3fn)
        w1_q, scale_w1 = pertoken_quant(w1_fp32, quant_dtype=torch.float8_e4m3fn)
    elif in_dtype == "fp16":
        x_q = x_fp32.to(torch.float16)
        w1_q = w1_fp32.to(torch.float16)
        scale_x, scale_w1 = None, None
    elif in_dtype == "bf16":
        x_q = x_fp32.to(torch.bfloat16)
        w1_q = w1_fp32.to(torch.bfloat16)
        scale_x, scale_w1 = None, None
    else:
        print(f"  [FlyDSL standalone] Unsupported in_dtype={in_dtype}")
        return None

    w1_shuffled = shuffle_weight(w1_q)
    w1_flat = w1_shuffled.view(NUM_EXPERTS * (2 * I), H)
    scale_w1_flat = None if scale_w1 is None else scale_w1.view(NUM_EXPERTS * (2 * I), 1)
    x_q = x_q.contiguous().view(tokens, H)

    scale_x_1d = torch.empty((0,), device=device, dtype=torch.float32) if scale_x is None else scale_x.view(-1).contiguous()
    scale_w1_1d = torch.empty((0,), device=device, dtype=torch.float32) if scale_w1_flat is None else scale_w1_flat.view(-1).contiguous()
    sorted_w_1d = sorted_w.contiguous().view(-1)

    out = torch.empty(tokens, TOPK, I, dtype=torch.float16, device=device)

    try:
        exe = compile_moe_gemm1(
            model_dim=H, inter_dim=I,
            experts=NUM_EXPERTS, topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage1=True,
            in_dtype=in_dtype,
            use_cshuffle_epilog=False,
        )
    except Exception as e:
        print(f"  [FlyDSL standalone] Stage1 compile failed: {e}")
        return None

    def run():
        stream = torch.cuda.current_stream()
        exe(
            out, x_q, w1_flat,
            scale_x_1d, scale_w1_1d,
            sorted_ids, sorted_expert_ids, sorted_w_1d,
            num_valid_ids,
            tokens, I, H, blocks,
            stream,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL standalone] Stage1 failed: {e}")
        return None


def _bench_flydsl_moe_stage2_standalone(
    tokens: int, warmup: int, iters: int,
    in_dtype: str = "fp8",
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 128,
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage2 using the standalone compile_moe_gemm2 kernel."""
    from tests.utils import pertoken_quant, shuffle_weight
    from kernels.moe_gemm_2stage import compile_moe_gemm2

    device = torch.device("cuda")
    H = MODEL_DIM_PADDED
    I = INTER_DIM_PADDED

    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing
    blocks = int(sorted_expert_ids.numel())

    w2_fp32 = torch.randn(NUM_EXPERTS, H, I, device=device, dtype=torch.float32)

    if in_dtype == "fp8":
        a2_fp32 = torch.randn(tokens, TOPK, I, device=device, dtype=torch.float32)
        a2_q, a2_scale = pertoken_quant(a2_fp32, quant_dtype=torch.float8_e4m3fn)
        w2_q, scale_w2 = pertoken_quant(w2_fp32, quant_dtype=torch.float8_e4m3fn)
    elif in_dtype == "fp16":
        a2_q = torch.randn(tokens, TOPK, I, device=device, dtype=torch.float16)
        w2_q = w2_fp32.to(torch.float16)
        a2_scale, scale_w2 = None, None
    elif in_dtype == "bf16":
        a2_q = torch.randn(tokens, TOPK, I, device=device, dtype=torch.bfloat16)
        w2_q = w2_fp32.to(torch.bfloat16)
        a2_scale, scale_w2 = None, None
    else:
        print(f"  [FlyDSL standalone] Unsupported in_dtype={in_dtype}")
        return None

    w2_shuffled = shuffle_weight(w2_q)
    w2_flat = w2_shuffled.view(NUM_EXPERTS * H, I)
    scale_w2_flat = None if scale_w2 is None else scale_w2.view(NUM_EXPERTS * H, 1)

    a2_scale_1d = torch.empty((0,), device=device, dtype=torch.float32) if a2_scale is None else a2_scale.view(-1).contiguous()
    scale_w2_1d = torch.empty((0,), device=device, dtype=torch.float32) if scale_w2_flat is None else scale_w2_flat.view(-1).contiguous()
    sorted_w_1d = sorted_w.contiguous().view(-1)

    out = torch.zeros(tokens, H, dtype=torch.float16, device=device)

    try:
        exe = compile_moe_gemm2(
            model_dim=H, inter_dim=I,
            experts=NUM_EXPERTS, topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage2=True,
            in_dtype=in_dtype,
        )
    except Exception as e:
        print(f"  [FlyDSL standalone] Stage2 compile failed: {e}")
        return None

    def run():
        stream = torch.cuda.current_stream()
        exe(
            out, a2_q, w2_flat,
            a2_scale_1d, scale_w2_1d,
            sorted_ids, sorted_expert_ids, sorted_w_1d,
            num_valid_ids,
            tokens, H, I, blocks,
            stream,
        )

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL standalone] Stage2 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# FlyDSL MoE benchmark
# ---------------------------------------------------------------------------
def bench_flydsl_moe_stage1(
    tokens: int, warmup: int, iters: int,
    a_dtype: str = "fp16", b_dtype: str = "fp4",
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 256,
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage1 (gate+up GEMM) using mixed-precision kernel.

    Uses compile_mixed_moe_gemm1 from AITER's FlyDSL kernels, which supports
    mixed A×W precision (A16W4, A8W4, A4W4).
    """
    try:
        from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
    except ImportError:
        # Fallback: try the FlyDSL standalone kernel for homogeneous dtypes
        try:
            from kernels.moe_gemm_2stage import compile_moe_gemm1 as _compile
            # Map to homogeneous in_dtype for the standalone kernel
            if a_dtype == b_dtype or b_dtype in ("fp4",):
                in_dtype = {"fp16": "fp16", "fp8": "fp8", "fp4": "fp8", "bf16": "bf16"}.get(a_dtype, "fp8")
            else:
                print(f"  [FlyDSL] mixed_moe_gemm not available and standalone doesn't support a={a_dtype}/w={b_dtype}")
                return None
            return _bench_flydsl_moe_stage1_standalone(
                tokens, warmup, iters, in_dtype=in_dtype,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            )
        except ImportError:
            print("  [FlyDSL] Could not import compile_mixed_moe_gemm1")
            return None

    device = torch.device("cuda")
    H = MODEL_DIM_PADDED
    I = INTER_DIM_PADDED

    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing
    blocks = int(sorted_expert_ids.numel())

    # Create weight tensors in MXFP4 format (packed fp4x2)
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    w1 = torch.randint(0, 256, (NUM_EXPERTS, 2 * I, H // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w1_scale = torch.randint(124, 130, (NUM_EXPERTS, 2 * I, H // 32),
                              dtype=torch.uint8, device=device).view(fp8_e8m0)

    # Create activation tensors based on a_dtype
    if a_dtype == "fp16":
        x = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
        a1_scale = None
    elif a_dtype == "fp8":
        fp8_dtype = torch.float8_e4m3fn
        x = torch.randn(tokens, H, dtype=torch.float32, device=device).to(fp8_dtype)
        a1_scale = torch.ones(tokens, H // 32, dtype=fp8_e8m0, device=device)
    elif a_dtype == "fp4":
        x = torch.randint(0, 256, (tokens, H // 2),
                           dtype=torch.uint8, device=device).view(fp4_dtype)
        a1_scale = torch.randint(124, 130, (tokens, H // 32),
                                  dtype=torch.uint8, device=device).view(fp8_e8m0)
    else:
        print(f"  [FlyDSL] Unsupported a_dtype={a_dtype}")
        return None

    try:
        exe = compile_mixed_moe_gemm1(
            model_dim=H, inter_dim=I,
            experts=NUM_EXPERTS, topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage1=True,
            a_dtype=a_dtype, b_dtype=b_dtype,
            out_dtype="bf16", act="swiglu",
        )
    except Exception as e:
        print(f"  [FlyDSL] Stage1 compile failed: {e}")
        return None

    out = torch.empty(tokens, TOPK, I, dtype=torch.bfloat16, device=device)
    sorted_w_1d = sorted_w.contiguous().view(-1)

    def _as_i8(t):
        if hasattr(t, 'dtype') and "float8" in str(t.dtype):
            return t.view(torch.int8)
        if hasattr(t, 'dtype') and "float4" in str(t.dtype):
            return t.view(torch.uint8)
        return t

    def run():
        stream = torch.cuda.current_stream()
        args = [
            out, _as_i8(x), _as_i8(w1),
        ]
        # Scale args depend on precision
        if a1_scale is not None:
            args.append(_as_i8(a1_scale).view(-1))
        else:
            args.append(torch.empty((0,), device=device, dtype=torch.float32))
        args.append(_as_i8(w1_scale).view(-1))
        args.extend([
            sorted_ids, sorted_expert_ids, sorted_w_1d,
            num_valid_ids,
            tokens, I, H, blocks,
            stream,
        ])
        exe(*args)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] Stage1 failed: {e}")
        return None


def bench_flydsl_moe_stage2(
    tokens: int, warmup: int, iters: int,
    a_dtype: str = "fp16", b_dtype: str = "fp4",
    tile_m: int = 32, tile_n: int = 128, tile_k: int = 256,
) -> Optional[float]:
    """Benchmark FlyDSL MoE stage2 (down GEMM) using mixed-precision kernel.

    Uses compile_mixed_moe_gemm2 from AITER's FlyDSL kernels.
    """
    try:
        from aiter.ops.flydsl.kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2
    except ImportError:
        try:
            from kernels.moe_gemm_2stage import compile_moe_gemm2 as _compile
            in_dtype = {"fp16": "fp16", "fp8": "fp8", "fp4": "fp8", "bf16": "bf16"}.get(a_dtype, "fp8")
            return _bench_flydsl_moe_stage2_standalone(
                tokens, warmup, iters, in_dtype=in_dtype,
                tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            )
        except ImportError:
            print("  [FlyDSL] Could not import compile_mixed_moe_gemm2")
            return None

    device = torch.device("cuda")
    H = MODEL_DIM_PADDED
    I = INTER_DIM_PADDED

    routing = build_routing(tokens, device)
    if routing is None:
        return None
    sorted_ids, sorted_w, sorted_expert_ids, num_valid_ids = routing
    blocks = int(sorted_expert_ids.numel())

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    w2 = torch.randint(0, 256, (NUM_EXPERTS, H, I // 2),
                        dtype=torch.uint8, device=device).view(fp4_dtype)
    w2_scale = torch.randint(124, 130, (NUM_EXPERTS, H, I // 32),
                              dtype=torch.uint8, device=device).view(fp8_e8m0)

    # Stage2 activation: BF16 intermediate from stage1 (for A16W4)
    if a_dtype == "fp16":
        a2 = torch.randn(tokens, TOPK, I, dtype=torch.bfloat16, device=device)
        a2_scale = None
    elif a_dtype == "fp8":
        fp8_dtype = torch.float8_e4m3fn
        a2 = torch.randn(tokens * TOPK, I, dtype=torch.float32, device=device).to(fp8_dtype)
        a2 = a2.view(tokens, TOPK, I)
        a2_scale = torch.ones(tokens * TOPK, I // 32, dtype=fp8_e8m0, device=device)
    elif a_dtype == "fp4":
        a2 = torch.randint(0, 256, (tokens * TOPK, I // 2),
                            dtype=torch.uint8, device=device).view(fp4_dtype)
        a2 = a2.view(tokens, TOPK, -1)
        a2_scale = torch.randint(124, 130, (tokens * TOPK, I // 32),
                                  dtype=torch.uint8, device=device).view(fp8_e8m0)
    else:
        print(f"  [FlyDSL] Unsupported a_dtype={a_dtype}")
        return None

    try:
        exe = compile_mixed_moe_gemm2(
            model_dim=H, inter_dim=I,
            experts=NUM_EXPERTS, topk=TOPK,
            tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
            doweight_stage2=True,
            a_dtype=a_dtype, b_dtype=b_dtype,
            out_dtype="bf16",
        )
    except Exception as e:
        print(f"  [FlyDSL] Stage2 compile failed: {e}")
        return None

    out = torch.zeros(tokens, H, dtype=torch.bfloat16, device=device)
    sorted_w_1d = sorted_w.contiguous().view(-1)

    def _as_i8(t):
        if hasattr(t, 'dtype') and "float8" in str(t.dtype):
            return t.view(torch.int8)
        if hasattr(t, 'dtype') and "float4" in str(t.dtype):
            return t.view(torch.uint8)
        return t

    def run():
        out.zero_()
        stream = torch.cuda.current_stream()
        args = [
            out, _as_i8(a2), _as_i8(w2),
        ]
        if a2_scale is not None:
            args.append(_as_i8(a2_scale).view(-1))
        else:
            args.append(torch.empty((0,), device=device, dtype=torch.float32))
        args.append(_as_i8(w2_scale).view(-1))
        args.extend([
            sorted_ids, sorted_expert_ids, sorted_w_1d,
            num_valid_ids,
            tokens, H, I, blocks,
            stream,
        ])
        exe(*args)

    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] Stage2 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
def print_results(rows: List[MoeBenchRow], precision: str = "A16W4") -> None:
    print()
    print("=" * 90)
    print(f"MoE Benchmark — GPT-OSS 120B {precision} (E={NUM_EXPERTS}, topk={TOPK}, "
          f"hidden={MODEL_DIM}→{MODEL_DIM_PADDED}, inter={INTER_DIM}→{INTER_DIM_PADDED})")
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
PRECISION_CONFIGS = {
    "A16W4": ("fp16", "fp4"),   # BF16 act × MXFP4 weight (confirmed from trace)
    "A8W4":  ("fp8",  "fp4"),   # FP8 act × MXFP4 weight
    "A4W4":  ("fp4",  "fp4"),   # FP4 act × MXFP4 weight
    "FP8":   ("fp8",  "fp8"),   # FP8 homogeneous (uses existing test_moe_gemm.py harness)
}


def main():
    parser = argparse.ArgumentParser(description="MoE benchmark: CK vs FlyDSL (A16W4/A8W4/A4W4)")
    parser.add_argument("--concurrency", type=str, default="4,32,128",
                        help="Comma-separated M values (decode concurrency)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all concurrency points (1..1024)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true",
                        help="Skip CK baseline (no AITER needed)")
    parser.add_argument("--precision", type=str, default="A16W4",
                        help="Precision config: A16W4 (default, matches trace), A8W4, A4W4, or 'all'")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    if args.sweep:
        token_list = SWEEP_CONCURRENCY
    else:
        token_list = [int(t.strip()) for t in args.concurrency.split(",")]

    if args.precision == "all":
        precisions = list(PRECISION_CONFIGS.keys())
    else:
        precisions = [p.strip() for p in args.precision.split(",")]

    for prec in precisions:
        if prec not in PRECISION_CONFIGS:
            print(f"Unknown precision '{prec}'. Choose from: {list(PRECISION_CONFIGS.keys())}")
            return
        a_dtype, b_dtype = PRECISION_CONFIGS[prec]
        rows: List[MoeBenchRow] = []

        print(f"\n{'='*90}")
        print(f"Precision: {prec} (A={a_dtype}, W={b_dtype})")
        print(f"{'='*90}")

        for tokens in token_list:
            print(f"Benchmarking concurrency={tokens} ...")

            if prec == "FP8":
                # Delegate to existing test_moe_gemm.py harness (includes CK comparison)
                s1_us, _ = bench_fp8_via_test_harness(tokens, stage=1, iters=args.iters, warmup=args.warmup)
                rows.append(MoeBenchRow(label="FlyDSL S1", tokens=tokens, us=s1_us))
                s2_us, _ = bench_fp8_via_test_harness(tokens, stage=2, iters=args.iters, warmup=args.warmup)
                rows.append(MoeBenchRow(label="FlyDSL S2", tokens=tokens, us=s2_us))
            else:
                # Mixed-precision path (A16W4 / A8W4 / A4W4)
                if not args.flydsl_only:
                    ck_s1 = bench_ck_moe_stage1(tokens, args.warmup, args.iters)
                    rows.append(MoeBenchRow(label="CK S1", tokens=tokens, us=ck_s1))

                    ck_s2 = bench_ck_moe_stage2(tokens, args.warmup, args.iters)
                    rows.append(MoeBenchRow(label="CK S2", tokens=tokens, us=ck_s2))

                s1_us = bench_flydsl_moe_stage1(
                    tokens, args.warmup, args.iters,
                    a_dtype=a_dtype, b_dtype=b_dtype,
                )
                rows.append(MoeBenchRow(label="FlyDSL S1", tokens=tokens, us=s1_us))

                s2_us = bench_flydsl_moe_stage2(
                    tokens, args.warmup, args.iters,
                    a_dtype=a_dtype, b_dtype=b_dtype,
                )
                rows.append(MoeBenchRow(label="FlyDSL S2", tokens=tokens, us=s2_us))

        print_results(rows, prec)


if __name__ == "__main__":
    main()
