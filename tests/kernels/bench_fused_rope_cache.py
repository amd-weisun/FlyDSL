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


def _create_tensors(tokens, device, use_flash_layout=True):
    """Create tensors for benchmarking.

    Args:
        use_flash_layout: True for flash layout [T,BS,KH,D], False for non-flash [T,KH,D//x,BS,x]
    """
    hd = _CFG["head_dim"]
    num_q_heads = _CFG["num_q_heads"]
    num_kv_heads = _CFG["num_kv_heads"]
    nb = max(32, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    q = torch.randn(tokens, num_q_heads, hd, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    cos = torch.randn(MAX_POS, hd // 2, device=device, dtype=torch.bfloat16)
    sin = torch.randn(MAX_POS, hd // 2, device=device, dtype=torch.bfloat16)
    pos = torch.arange(tokens, device=device, dtype=torch.int64)
    slots = torch.arange(tokens, device=device, dtype=torch.int64)
    if use_flash_layout:
        kc = torch.zeros(nb, BLOCK_SIZE, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
        vc = torch.zeros(nb, BLOCK_SIZE, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    else:
        x_size = 16
        kc = torch.zeros(nb, num_kv_heads, hd // x_size, BLOCK_SIZE, x_size,
                          device=device, dtype=torch.bfloat16)
        vc = torch.zeros(nb, num_kv_heads, hd, BLOCK_SIZE,
                          device=device, dtype=torch.bfloat16)
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    return q, k, v, cos, sin, pos, slots, kc, vc, qo, ko


def bench_triton(tokens, warmup, iters, flash_layout=False):
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
    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_tensors(tokens, device, use_flash_layout=flash_layout)
    ks = torch.tensor([1.0], device=device, dtype=torch.float32)
    vs = torch.tensor([1.0], device=device, dtype=torch.float32)

    def run():
        fused_qk_rope_reshape_and_cache(
            q, k, v, kc, vc, slots, pos, cos, sin,
            ks, vs, is_neox=True, flash_layout=flash_layout,
            apply_scale=False, q_out=qo, k_out=ko, output_zeros=False,
        )
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [Triton] {e}")
        return None


def bench_flydsl(tokens, warmup, iters, flash_layout=False):
    from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
    device = torch.device("cuda")

    q, k, v, cos, sin, pos, slots, kc, vc, qo, ko = _create_tensors(tokens, device, use_flash_layout=flash_layout)
    pos_i32 = pos.to(torch.int32)
    slots_i32 = slots.to(torch.int32)

    launch_fn = build_fused_rope_cache_module(
        head_dim=_CFG["head_dim"], num_q_heads=_CFG["num_q_heads"], num_kv_heads=_CFG["num_kv_heads"],
        block_size=BLOCK_SIZE, is_neox=True, flash_layout=flash_layout, dtype_str="bf16",
    )
    stream = torch.cuda.current_stream()

    def run():
        launch_fn(q, k, v, pos_i32, cos, sin, slots_i32, kc, vc, qo, ko, tokens, stream=stream)
    try:
        return bench_gpu_us_torch(run, warmup=warmup, iters=iters)
    except Exception as e:
        print(f"  [FlyDSL] {e}")
        return None


def verify_flydsl(launch_fn, tokens, flash_layout):
    """Quick correctness check against PyTorch reference. Returns (ok, error_str)."""
    device = torch.device("cuda")
    hd = _CFG["head_dim"]
    num_q_heads = _CFG["num_q_heads"]
    num_kv_heads = _CFG["num_kv_heads"]
    half_dim = hd // 2

    torch.manual_seed(42)
    q = torch.randn(tokens, num_q_heads, hd, device=device, dtype=torch.bfloat16)
    k = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    v = torch.randn(tokens, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    cos = torch.randn(MAX_POS, half_dim, device=device, dtype=torch.bfloat16)
    sin = torch.randn(MAX_POS, half_dim, device=device, dtype=torch.bfloat16)
    pos = torch.randint(0, MAX_POS, (tokens,), device=device, dtype=torch.int32)
    slots = torch.arange(tokens, device=device, dtype=torch.int32)
    nb = max(32, (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)

    if flash_layout:
        kc = torch.zeros(nb, BLOCK_SIZE, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
        vc = torch.zeros(nb, BLOCK_SIZE, num_kv_heads, hd, device=device, dtype=torch.bfloat16)
    else:
        kc = torch.zeros(nb, num_kv_heads, hd // 16, BLOCK_SIZE, 16, device=device, dtype=torch.bfloat16)
        vc = torch.zeros(nb, num_kv_heads, hd, BLOCK_SIZE, device=device, dtype=torch.bfloat16)
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)

    # PyTorch reference (Q/K rotation only — layout-independent)
    c = cos[pos.long()].unsqueeze(1).float()
    s = sin[pos.long()].unsqueeze(1).float()
    q1, q2 = q.float()[..., :half_dim], q.float()[..., half_dim:]
    q_ref = torch.cat([q1 * c - q2 * s, q2 * c + q1 * s], dim=-1).to(torch.bfloat16)
    k1, k2 = k.float()[..., :half_dim], k.float()[..., half_dim:]
    k_ref = torch.cat([k1 * c - k2 * s, k2 * c + k1 * s], dim=-1).to(torch.bfloat16)

    launch_fn(q, k, v, pos, cos, sin, slots, kc, vc, qo, ko, tokens,
              stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    q_err = (qo.float() - q_ref.float()).abs().max().item()
    k_err = (ko.float() - k_ref.float()).abs().max().item()
    ok = q_err < 0.1 and k_err < 0.1
    return ok, f"Q={q_err:.2e} K={k_err:.2e}"


@dataclass
class Row:
    model: str
    tp: int
    tokens: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    layout: str  # "flash" or "non-flash"
    triton_us: Optional[float]
    flydsl_us: Optional[float]


def print_summary(results: List[Row]):
    print(f"\n{'='*115}")
    print("Fused RoPE + KV Cache Benchmark")
    print(f"  flash:     key_cache [T,BS,KH,D],            value_cache [T,BS,KH,D]")
    print(f"  non-flash: key_cache [T,KH,D//16,BS,16],     value_cache [T,KH,D,BS]  (ATOM default)")
    print(f"  Both Triton and FlyDSL use same layout per row")
    print(f"{'='*115}")
    print(f"{'Model':<18s} {'TP':>2s} {'Q heads':>7s} {'KV heads':>8s} {'D':>3s} {'M':>5s} {'Layout':>9s}  "
          f"{'Triton(us)':>10s} {'FlyDSL(us)':>10s}  {'Speedup':>8s}")
    print(f"{'-'*115}")
    for r in results:
        tri_s = f"{r.triton_us:.1f}" if r.triton_us else "N/A"
        fly_s = f"{r.flydsl_us:.1f}" if r.flydsl_us else "N/A"
        sp = f"{r.triton_us / r.flydsl_us:.2f}x" if r.triton_us and r.flydsl_us else ""
        print(f"{r.model:<18s} {r.tp:>2d} {r.num_q_heads:>7d} {r.num_kv_heads:>8d} {r.head_dim:>3d} {r.tokens:>5d} {r.layout:>9s}  "
              f"{tri_s:>10s} {fly_s:>10s}  {sp:>8s}")
    print(f"{'='*115}")


def main():
    parser = argparse.ArgumentParser(description="Fused RoPE+KVCache: FlyDSL vs Triton")
    parser.add_argument("--concurrency", type=str, default="1,4,32,128")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--flydsl-only", action="store_true")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip correctness verification")
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

            # Verify correctness once per layout (reuses bench_flydsl's compiled kernel)
            if not args.no_verify:
                from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
                for layout_name, flash in [("non-flash", False), ("flash", True)]:
                    fn = build_fused_rope_cache_module(
                        head_dim=hd, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                        block_size=BLOCK_SIZE, is_neox=True, flash_layout=flash, dtype_str="bf16",
                    )
                    ok, err_str = verify_flydsl(fn, min(32, token_list[0]), flash_layout=flash)
                    status = "PASS" if ok else "FAIL"
                    print(f"  [verify {layout_name}] {status}: {err_str}")
                    if not ok:
                        print(f"  CORRECTNESS FAILED for {layout_name} — skipping")

            for tokens in token_list:
                for layout_name, flash in [("non-flash", False), ("flash", True)]:
                    print(f"  M={tokens:>4d} {layout_name:>9s} ... ", end="", flush=True)

                    tri = bench_triton(tokens, args.warmup, args.iters, flash_layout=flash) if not args.flydsl_only else None
                    fly = bench_flydsl(tokens, args.warmup, args.iters, flash_layout=flash)

                    results.append(Row(model_name, tp, tokens, num_q_heads, num_kv_heads, hd, layout_name, tri, fly))

                    parts = []
                    if tri: parts.append(f"Tri={tri:.1f}")
                    if fly: parts.append(f"FlyDSL={fly:.1f}")
                    if tri and fly: parts.append(f"({tri/fly:.2f}x)")
                    print("  ".join(parts))

    print_summary(results)


if __name__ == "__main__":
    main()
