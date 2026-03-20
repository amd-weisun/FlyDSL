#!/usr/bin/env python3
"""Fused RoPE + KV Cache benchmark: FlyDSL vs AITER Triton.

Usage (from FlyDSL/ directory):
    # Default (GPT-OSS-120B, TP=8)
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --sweep

    # All models, multiple TPs
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --sweep --model all --tp 1,8

    # FlyDSL only
    PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --flydsl-only --sweep --model all
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_THIS = os.path.abspath(__file__)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_FLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_FLYDSL_SRC) and _FLYDSL_SRC not in sys.path:
    sys.path.insert(0, _FLYDSL_SRC)

import torch

from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter

# ---------------------------------------------------------------------------
# Model configs: (head_dim, total_q_heads, total_kv_heads)
# ---------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Tuple[int, int, int]] = {
    "GPT-OSS-120B":   (64,  64,   8),
    "Qwen3-235B-MoE": (64,  64,   4),
    "Llama-3.1-8B":   (128, 32,   8),
    "Llama-3.1-70B":  (128, 64,   8),
    "Qwen3-72B":      (128, 64,   8),
    "Llama-3.1-405B": (128, 128,  8),
}

BLOCK_SIZE = 16
MAX_POS = 8192
SWEEP_CONCURRENCY = [1, 4, 32, 128,  1024]

# Active config (set per benchmark iteration)
_CFG = {"head_dim": 64, "num_q_heads": 8, "num_kv_heads": 1}


def _create_tensors(tokens, device):
    """Create tensors matching ATOM's actual shapes and dtypes.

    Key ATOM conventions (from atom/model_ops/attention_mha.py):
      - cos/sin: [max_pos, 1, 1, D//2] (4D, from RotaryEmbedding)
      - positions: int64 (from backends.py)
      - slot_mapping: int64 (from backends.py)
      - flash_layout=False for standard MHA: key_cache [T_cache, KH, D//x, BS, x]
      - flash_layout=True for plugin mode: key_cache [T_cache, BS, KH, D]
    """
    hd = _CFG["head_dim"]
    num_q_heads = _CFG["num_q_heads"]
    num_kv_heads = _CFG["num_kv_heads"]
    nb = max(32, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    x_size = 16  # packing factor for non-flash key_cache layout
    q = torch.randn(tokens, num_q_heads, hd, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    # cos/sin: 4D to match RotaryEmbedding output
    cos = torch.randn(MAX_POS, 1, 1, hd // 2, device=device, dtype=torch.bfloat16)
    sin = torch.randn(MAX_POS, 1, 1, hd // 2, device=device, dtype=torch.bfloat16)
    # positions and slot_mapping: int64 to match ATOM
    pos = torch.arange(tokens, device=device, dtype=torch.int64)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    # Non-flash layout (ATOM standard MHA default):
    #   key_cache: [T_cache, KH, D//x, BS, x]
    #   value_cache: [T_cache, KH, D, BS]
    kc = torch.zeros(nb, num_kv_heads, hd // x_size, BLOCK_SIZE, x_size,
                      device=device, dtype=torch.bfloat16)
    vc = torch.zeros(nb, num_kv_heads, hd, BLOCK_SIZE,
                      device=device, dtype=torch.bfloat16)
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    return q, k, v, cos, sin, pos, slots, kc, vc, qo, ko


def bench_triton(tokens, warmup, iters):
    if not maybe_enable_aiter():
        return None
    try:
        from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
    except ImportError:
        try:
            from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
        except ImportError:
            return None
    device = torch.device("cuda")
    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_tensors(tokens, device)
    ks = torch.tensor([1.0], device=device, dtype=torch.float32)
    vs = torch.tensor([1.0], device=device, dtype=torch.float32)

    def run():
        fused_qk_rope_reshape_and_cache(
            q, k, v, kc, vc, slots, pos, cos, sin,
            ks, vs, is_neox=True, flash_layout=False,
            apply_scale=False, q_out=qo, k_out=ko, output_zeros=False,
        )
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [Triton] {e}")
        return None


def bench_flydsl(tokens, warmup, iters):
    from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
    device = torch.device("cuda")
    hd = _CFG["head_dim"]
    num_q_heads = _CFG["num_q_heads"]
    num_kv_heads = _CFG["num_kv_heads"]

    # FlyDSL kernel currently only supports flash_layout=True.
    # Create flash-layout tensors for FlyDSL (separate from Triton's non-flash tensors).
    nb = max(32, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    q = torch.randn(tokens, num_q_heads, hd, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    cos = torch.randn(MAX_POS, hd // 2, device=device, dtype=torch.bfloat16)
    sin = torch.randn(MAX_POS, hd // 2, device=device, dtype=torch.bfloat16)
    pos = torch.arange(tokens, device=device, dtype=torch.int32)
    slots = torch.arange(tokens, device=device, dtype=torch.int32)
    kc = torch.zeros(nb, BLOCK_SIZE, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    vc = torch.zeros(nb, BLOCK_SIZE, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)

    launch_fn = build_fused_rope_cache_module(
        head_dim=hd, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        block_size=BLOCK_SIZE, is_neox=True, flash_layout=True, dtype_str="bf16",
    )
    stream = torch.cuda.current_stream()

    def run():
        launch_fn(q, k, v, pos, cos, sin, slots, kc, vc, qo, ko, tokens, stream=stream)
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] {e}")
        return None


@dataclass
class Row:
    model: str
    tp: int
    tokens: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    triton_us: Optional[float]
    flydsl_us: Optional[float]


def print_summary(results: List[Row]):
    print(f"\n{'='*105}")
    print(f"{'Model':<18s} {'TP':>2s} {'Q heads':>7s} {'KV heads':>8s} {'head_dim':>8s} {'M':>5s}  "
          f"{'Triton(us)':>10s} {'FlyDSL(us)':>10s}  {'Speedup':>8s}")
    print(f"{'='*105}")
    for r in results:
        tri_s = f"{r.triton_us:.1f}" if r.triton_us else "N/A"
        fly_s = f"{r.flydsl_us:.1f}" if r.flydsl_us else "N/A"
        sp = f"{r.triton_us / r.flydsl_us:.2f}x" if r.triton_us and r.flydsl_us else ""
        print(f"{r.model:<18s} {r.tp:>2d} {r.num_q_heads:>7d} {r.num_kv_heads:>8d} {r.head_dim:>8d} {r.tokens:>5d}  "
              f"{tri_s:>10s} {fly_s:>10s}  {sp:>8s}")
    print(f"{'='*105}")


def main():
    parser = argparse.ArgumentParser(description="Fused RoPE+KVCache: FlyDSL vs Triton")
    parser.add_argument("--concurrency", type=str, default="1,4,32,128")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true")
    parser.add_argument("--tp", type=str, default="8",
                        help="1,2,4,8 or 'all'")
    parser.add_argument("--model", type=str, default="GPT-OSS-120B",
                        help=f"{', '.join(MODEL_CONFIGS.keys())} or 'all'")
    args = parser.parse_args()

    torch.set_default_device("cuda")
    token_list = SWEEP_CONCURRENCY if args.sweep else [int(t) for t in args.concurrency.split(",")]
    tp_list = [1, 2, 4, 8] if args.tp == "all" else [int(t) for t in args.tp.split(",")]
    model_list = list(MODEL_CONFIGS.keys()) if args.model == "all" else [m.strip() for m in args.model.split(",")]

    results: List[Row] = []

    for model_name in model_list:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model '{model_name}'. Options: {list(MODEL_CONFIGS.keys())}")
            continue
        hd, total_qh, total_kh = MODEL_CONFIGS[model_name]

        for tp in tp_list:
            num_q_heads = total_qh // tp
            num_kv_heads = max(1, total_kh // tp)
            if num_q_heads < 1:
                continue
            _CFG["head_dim"] = hd
            _CFG["num_q_heads"] = num_q_heads
            _CFG["num_kv_heads"] = num_kv_heads

            print(f"\n{'='*80}")
            print(f"{model_name} TP={tp}: num_q_heads={num_q_heads}, "
                  f"num_kv_heads={num_kv_heads}, head_dim={hd}")
            print(f"{'='*80}")

            for tokens in token_list:
                print(f"  M={tokens:>4d} ... ", end="", flush=True)

                tri = bench_triton(tokens, args.warmup, args.iters) if not args.flydsl_only else None
                fly = bench_flydsl(tokens, args.warmup, args.iters)

                results.append(Row(model_name, tp, tokens, num_q_heads, num_kv_heads, hd, tri, fly))

                parts = []
                if tri: parts.append(f"Tri={tri:.1f}")
                if fly: parts.append(f"FlyDSL={fly:.1f}")
                if tri and fly: parts.append(f"({tri/fly:.2f}x)")
                print("  ".join(parts))

    print_summary(results)


if __name__ == "__main__":
    main()
