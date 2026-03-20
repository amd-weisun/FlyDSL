#!/usr/bin/env python3
"""Fused RoPE + KV Cache benchmark: FlyDSL vs AITER Triton.

This is the kernel that takes 4.1% GPU time in GPT-OSS 120B (3.50 ms).
Compares Triton (single launch, 40+ args) vs FlyDSL-2K (two launches)
vs FlyDSL-1K (single launch with runtime Q/K dispatch).

Usage (from FlyDSL/ directory):
    # GPT-OSS 120B default (TP=8)
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --sweep

    # All TP values
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --sweep --tp all

    # Multiple models (stress test larger configs)
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/bench_fused_rope_cache.py --sweep --model all

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

SWEEP_CONCURRENCY = [1, 4, 32, 128, 1024]

# Active config (set per benchmark iteration)
_CFG = {"head_dim": 64, "qh": 8, "kh": 1}


def _create_tensors(tokens, device):
    hd = _CFG["head_dim"]
    qh = _CFG["qh"]
    kh = _CFG["kh"]
    num_blocks = max(32, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    q = torch.randn(tokens, qh, hd, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, kh, hd, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, kh, hd, device=device, dtype=torch.bfloat16)
    cos = torch.randn(MAX_POS, hd // 2, device=device, dtype=torch.bfloat16)
    sin = torch.randn(MAX_POS, hd // 2, device=device, dtype=torch.bfloat16)
    pos = torch.arange(tokens, device=device, dtype=torch.int32)
    slots = torch.arange(tokens, device=device, dtype=torch.int32)
    kc = torch.zeros(num_blocks, BLOCK_SIZE, kh, hd, device=device, dtype=torch.bfloat16)
    vc = torch.zeros(num_blocks, BLOCK_SIZE, kh, hd, device=device, dtype=torch.bfloat16)
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
    pos_i64 = pos.to(torch.int64)
    k_scale = torch.tensor([1.0], device=device, dtype=torch.float32)
    v_scale = torch.tensor([1.0], device=device, dtype=torch.float32)

    def run():
        fused_qk_rope_reshape_and_cache(
            q, k, v, kc, vc, slots, pos_i64, cos, sin,
            k_scale, v_scale, is_neox=True, flash_layout=True,
            apply_scale=False, q_out=qo, k_out=ko, output_zeros=False,
        )
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [Triton] {e}")
        return None


def bench_fly2k(tokens, warmup, iters):
    from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
    device = torch.device("cuda")
    launch_fn = build_fused_rope_cache_module(
        head_dim=_CFG["head_dim"], num_q_heads=_CFG["qh"], num_kv_heads=_CFG["kh"],
        block_size=BLOCK_SIZE, is_neox=True, flash_layout=True, dtype_str="bf16",
    )
    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_tensors(tokens, device)
    stream = torch.cuda.current_stream()

    def run():
        launch_fn(q, k, v, pos, cos, sin, slots, kc, vc, qo, ko, tokens, stream=stream)
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [2K] {e}")
        return None


def bench_fly1k(tokens, warmup, iters):
    from kernels.fused_rope_cache_single_kernel import build_fused_rope_cache_single_module
    device = torch.device("cuda")
    launch_fn = build_fused_rope_cache_single_module(
        head_dim=_CFG["head_dim"], num_q_heads=_CFG["qh"], num_kv_heads=_CFG["kh"],
        block_size=BLOCK_SIZE, is_neox=True, flash_layout=True, dtype_str="bf16",
    )
    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_tensors(tokens, device)
    stream = torch.cuda.current_stream()
    n_q = tokens * _CFG["qh"]
    n_total = n_q + tokens * _CFG["kh"]
    nqp = torch.tensor([n_q], device=device, dtype=torch.int32)

    def run():
        launch_fn(q, k, v, pos, cos, sin, slots, kc, vc, qo, ko, nqp, n_total, stream=stream)
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [1K] {e}")
        return None


@dataclass
class Row:
    model: str
    tp: int
    tokens: int
    qh: int
    kh: int
    hd: int
    triton_us: Optional[float]
    fly2k_us: Optional[float]
    fly1k_us: Optional[float]


def print_summary(results: List[Row]):
    print(f"\n{'='*115}")
    print(f"{'Model':<18s} {'TP':>2s} {'M':>5s} {'progs':>7s}  "
          f"{'Triton(us)':>10s} {'2K(us)':>8s} {'1K(us)':>8s}  "
          f"{'2K/Tri':>7s} {'1K/Tri':>7s} {'1K/2K':>6s}")
    print(f"{'='*115}")

    for r in results:
        progs = r.tokens * (r.qh + r.kh)
        tri_s = f"{r.triton_us:.1f}" if r.triton_us else "N/A"
        f2_s = f"{r.fly2k_us:.1f}" if r.fly2k_us else "N/A"
        f1_s = f"{r.fly1k_us:.1f}" if r.fly1k_us else "N/A"

        sp_2k = f"{r.triton_us / r.fly2k_us:.2f}x" if r.triton_us and r.fly2k_us else ""
        sp_1k = f"{r.triton_us / r.fly1k_us:.2f}x" if r.triton_us and r.fly1k_us else ""
        sp_1v2 = f"{r.fly2k_us / r.fly1k_us:.2f}x" if r.fly2k_us and r.fly1k_us else ""

        print(f"{r.model:<18s} {r.tp:>2d} {r.tokens:>5d} {progs:>7d}  "
              f"{tri_s:>10s} {f2_s:>8s} {f1_s:>8s}  "
              f"{sp_2k:>7s} {sp_1k:>7s} {sp_1v2:>6s}")

    print(f"{'='*115}")


def main():
    parser = argparse.ArgumentParser(description="Fused RoPE+KVCache: FlyDSL vs Triton")
    parser.add_argument("--concurrency", type=str, default="1,4,32,128",
                        help="Comma-separated M values")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep concurrency: 1,4,32,128,1024")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true")
    parser.add_argument("--tp", type=str, default="8",
                        help="Tensor parallelism: 1,2,4,8 or 'all'")
    parser.add_argument("--model", type=str, default="GPT-OSS-120B",
                        help=f"Model config: {', '.join(MODEL_CONFIGS.keys())} or 'all'")
    args = parser.parse_args()

    torch.set_default_device("cuda")

    token_list = SWEEP_CONCURRENCY if args.sweep else [int(t) for t in args.concurrency.split(",")]

    if args.tp == "all":
        tp_list = [1, 2, 4, 8]
    else:
        tp_list = [int(t) for t in args.tp.split(",")]

    if args.model == "all":
        model_list = list(MODEL_CONFIGS.keys())
    else:
        model_list = [m.strip() for m in args.model.split(",")]

    results: List[Row] = []

    for model_name in model_list:
        if model_name not in MODEL_CONFIGS:
            print(f"Unknown model '{model_name}'. Options: {list(MODEL_CONFIGS.keys())}")
            continue
        hd, total_qh, total_kh = MODEL_CONFIGS[model_name]

        for tp in tp_list:
            qh = total_qh // tp
            kh = max(1, total_kh // tp)
            if qh < 1:
                continue

            _CFG["head_dim"] = hd
            _CFG["qh"] = qh
            _CFG["kh"] = kh
            progs_per_tok = qh + kh

            print(f"\n{'='*90}")
            print(f"{model_name} TP={tp}: QH={qh}, KH={kh}, D={hd}, progs/tok={progs_per_tok}")
            print(f"{'='*90}")

            for tokens in token_list:
                total_progs = tokens * progs_per_tok
                print(f"  M={tokens:>4d} ({total_progs:>7d} progs) ... ", end="", flush=True)

                tri = bench_triton(tokens, args.warmup, args.iters) if not args.flydsl_only else None
                f2 = bench_fly2k(tokens, args.warmup, args.iters)
                f1 = bench_fly1k(tokens, args.warmup, args.iters)

                results.append(Row(model_name, tp, tokens, qh, kh, hd, tri, f2, f1))

                parts = []
                if tri: parts.append(f"Tri={tri:.1f}")
                if f2: parts.append(f"2K={f2:.1f}")
                if f1: parts.append(f"1K={f1:.1f}")
                print("  ".join(parts))

    print_summary(results)


if __name__ == "__main__":
    main()
