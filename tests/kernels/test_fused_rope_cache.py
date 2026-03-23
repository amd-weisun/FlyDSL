#!/usr/bin/env python3
"""Fused RoPE + KV Cache kernel test.

Tests correctness of the fused kernel against PyTorch reference.
Supports both flash and non-flash KV cache layouts.

Usage:
    # Fast CI (GPT-OSS 120B TP=8, 10 tests):
    PYTHONPATH=./ pytest tests/kernels/test_fused_rope_cache.py -v -s

    # All models × TPs (multi-model sweep):
    PYTHONPATH=./ pytest tests/kernels/test_fused_rope_cache.py -v -s -k multi_model

    # With AITER performance comparison:
    AITER_REPO=../aiter PYTHONPATH=./ pytest tests/kernels/test_fused_rope_cache.py -v -s

    # CLI — all models:
    PYTHONPATH=./ python tests/kernels/test_fused_rope_cache.py --all-models

    # CLI — with AITER comparison:
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/test_fused_rope_cache.py --all-models
"""

import os
import sys
import logging

import torch
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if os.path.isdir(_PYFLYDSL_SRC) and _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
from tests.test_common import run_perftest

logging.basicConfig(level=logging.INFO)

try:
    from tests.kernels.benchmark_common import bench_gpu_us_torch, maybe_enable_aiter
    HAS_BENCH = True
except ImportError:
    try:
        from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter
        HAS_BENCH = True
    except ImportError:
        HAS_BENCH = False

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

BLOCK_SIZE = 16
MAX_POS = 8192

# Model configs: (head_dim, total_q_heads, total_kv_heads)
MODEL_CONFIGS = {
    "GPT-OSS-120B":   (64, 64, 8),
    "Qwen3-235B-MoE": (64, 64, 4),
    "Llama-3.1-8B":   (128, 32, 8),
    "Llama-3.1-70B":  (128, 64, 8),
    "Qwen3-72B":      (128, 64, 8),
    "Llama-3.1-405B": (128, 128, 8),
}

# Default: GPT-OSS 120B TP=8 (fast CI)
HEAD_DIM = 64
ROTARY_DIM = 64
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1


def fused_rope_cache_ref(q, k, v, cos_cache, sin_cache, positions, slot_mapping,
                          key_cache, value_cache, block_size, flash_layout=True):
    """PyTorch reference for fused RoPE + KV cache."""
    half_dim = cos_cache.shape[-1]
    cos = cos_cache[positions.long()].unsqueeze(1).float()
    sin = sin_cache[positions.long()].unsqueeze(1).float()

    q_f32 = q.float()
    q1, q2 = q_f32[..., :half_dim], q_f32[..., half_dim:]
    q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).to(q.dtype)

    k_f32 = k.float()
    k1, k2 = k_f32[..., :half_dim], k_f32[..., half_dim:]
    k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).to(k.dtype)

    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()
    for i in range(slot_mapping.shape[0]):
        slot = slot_mapping[i].item()
        if slot >= 0:
            bi = slot // block_size
            bp = slot % block_size
            if flash_layout:
                # [num_blocks, block_size, KH, D]
                key_cache_out[bi, bp] = k_out[i]
                value_cache_out[bi, bp] = v[i]
            else:
                # key_cache: [num_blocks, KH, D//x, block_size, x]
                x = 16
                for d in range(k_out.shape[-1]):
                    dg, dw = d // x, d % x
                    key_cache_out[bi, k_out.shape[1] * 0 // 1, dg, bp, dw] = k_out[i, 0, d]
                    # For multi-head: iterate over heads
                for h in range(k_out.shape[1]):
                    for d in range(k_out.shape[-1]):
                        dg, dw = d // x, d % x
                        key_cache_out[bi, h, dg, bp, dw] = k_out[i, h, d]
                # value_cache: [num_blocks, KH, D, block_size]
                for h in range(v.shape[1]):
                    for d in range(v.shape[-1]):
                        value_cache_out[bi, h, d, bp] = v[i, h, d]

    return q_out, k_out, key_cache_out, value_cache_out


def run_fused_test(num_tokens, head_dim=HEAD_DIM, num_q_heads=NUM_Q_HEADS,
                   num_kv_heads=NUM_KV_HEADS, block_size=BLOCK_SIZE,
                   max_pos=MAX_POS, flash_layout=True):
    """Run fused RoPE + KV cache kernel test."""
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
    num_blocks = max(32, (num_tokens + block_size - 1) // block_size + 1)
    rotary_dim = head_dim  # full rotation

    layout_name = "flash" if flash_layout else "non-flash"
    print(f"[fused_rope_cache] M={num_tokens}, BS={block_size}, "
          f"QH={num_q_heads}, KH={num_kv_heads}, D={head_dim}, layout={layout_name}")

    launch_fn = build_fused_rope_cache_module(
        head_dim=head_dim, rotary_dim=rotary_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        block_size=block_size, is_neox=True,
        flash_layout=flash_layout, dtype_str="bf16",
    )

    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_q_heads, head_dim, device=device, dtype=torch_dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch_dtype)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch_dtype)
    cos_cache = torch.randn(max_pos, rotary_dim // 2, device=device, dtype=torch_dtype)
    sin_cache = torch.randn(max_pos, rotary_dim // 2, device=device, dtype=torch_dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)

    x_size = 16
    if flash_layout:
        key_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                                 device=device, dtype=torch_dtype)
        value_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim,
                                   device=device, dtype=torch_dtype)
    else:
        key_cache = torch.zeros(num_blocks, num_kv_heads, head_dim // x_size, block_size, x_size,
                                 device=device, dtype=torch_dtype)
        value_cache = torch.zeros(num_blocks, num_kv_heads, head_dim, block_size,
                                   device=device, dtype=torch_dtype)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Reference
    q_ref, k_ref, kc_ref, vc_ref = fused_rope_cache_ref(
        q, k, v, cos_cache, sin_cache, positions, slot_mapping,
        key_cache.clone(), value_cache.clone(), block_size, flash_layout=flash_layout,
    )

    # Launch FlyDSL kernel — correctness run
    stream = torch.cuda.current_stream()
    launch_fn(q, k, v, positions, cos_cache, sin_cache, slot_mapping,
              key_cache, value_cache, q_out, k_out, num_tokens, stream=stream)
    torch.cuda.synchronize()

    # Perf measurement using bench_gpu_us_torch (same timer used for AITER comparison)
    if HAS_BENCH:
        def run_flydsl():
            launch_fn(q, k, v, positions, cos_cache, sin_cache, slot_mapping,
                      key_cache, value_cache, q_out, k_out, num_tokens, stream=stream)
        us = bench_gpu_us_torch(run_flydsl, warmup=10, iters=100)
    else:
        us = 0.0

    # Compute bandwidth
    total_bytes = (q.nelement() + k.nelement() + v.nelement()) * 2 * 2  # read+write bf16
    total_bytes += cos_cache[0:1].nelement() * 2 * 2 * num_tokens  # cos+sin per token
    bw_gbs = total_bytes / (us * 1e-6) / 1e9 if us > 0 else 0
    print(f"  [flyc] {us:.1f} us, BW: {bw_gbs:.2f} GB/s")

    # Verify
    atol = 0.1
    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()

    # Compare full KV cache tensors (same layout for ref and kernel)
    kc_err = (key_cache.float() - kc_ref.float()).abs().max().item()
    vc_err = (value_cache.float() - vc_ref.float()).abs().max().item()

    print(f"  q_err={q_err:.6f}, k_err={k_err:.6f}, kc_err={kc_err:.6f}, vc_err={vc_err:.6f}")

    # Optional AITER comparison
    if HAS_BENCH and maybe_enable_aiter():
        try:
            from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache
        except ImportError:
            try:
                from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
            except ImportError:
                fused_qk_rope_reshape_and_cache = None

        if fused_qk_rope_reshape_and_cache is not None:
            cos_4d = cos_cache.unsqueeze(1).unsqueeze(1)
            sin_4d = sin_cache.unsqueeze(1).unsqueeze(1)
            pos_i64 = positions.to(torch.int64)
            slots_i64 = slot_mapping.to(torch.int64)
            kc_aiter = torch.zeros_like(key_cache)
            vc_aiter = torch.zeros_like(value_cache)
            qo_aiter = torch.empty_like(q)
            ko_aiter = torch.empty_like(k)
            # Pre-clone inputs so clone overhead is NOT in timed region
            q_aiter = q.clone()
            k_aiter = k.clone()
            v_aiter = v.clone()
            ks = torch.tensor([1.0], device=device, dtype=torch.float32)
            vs = torch.tensor([1.0], device=device, dtype=torch.float32)

            def launch_aiter():
                fused_qk_rope_reshape_and_cache(
                    q_aiter, k_aiter, v_aiter, kc_aiter, vc_aiter,
                    slots_i64, pos_i64, cos_4d, sin_4d, ks, vs,
                    is_neox=True, flash_layout=flash_layout,
                    apply_scale=False, q_out=qo_aiter, k_out=ko_aiter,
                    output_zeros=False,
                )

            aiter_us = bench_gpu_us_torch(launch_aiter, warmup=10, iters=100)
            speedup = aiter_us / us if us > 0 else 0

            # Cross-validate: AITER Q/K output must match FlyDSL Q/K output
            torch.cuda.synchronize()
            q_cross_err = (qo_aiter.float() - q_out.float()).abs().max().item()
            k_cross_err = (ko_aiter.float() - k_out.float()).abs().max().item()
            cross_ok = q_cross_err < 0.1 and k_cross_err < 0.1
            cross_status = "MATCH" if cross_ok else "MISMATCH"
            print(f"  [aiter] {aiter_us:.1f} us → FlyDSL/AITER: {speedup:.2f}x "
                  f"(cross-check: {cross_status}, Q={q_cross_err:.2e}, K={k_cross_err:.2e})")

    ok = q_err < atol and k_err < atol and kc_err < atol and vc_err < atol
    return ok, q_err, k_err, kc_err, vc_err


# --- Default tests: GPT-OSS 120B TP=8 (fast CI) ---

@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32, 128])
def test_fused_rope_cache_flash(num_tokens):
    ok, q_err, k_err, kc_err, vc_err = run_fused_test(num_tokens, flash_layout=True)
    assert ok, f"FAILED: q={q_err:.2e} k={k_err:.2e} kc={kc_err:.2e} vc={vc_err:.2e}"


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32, 128])
def test_fused_rope_cache_nonflash(num_tokens):
    ok, q_err, k_err, kc_err, vc_err = run_fused_test(num_tokens, flash_layout=False)
    assert ok, f"FAILED: q={q_err:.2e} k={k_err:.2e} kc={kc_err:.2e} vc={vc_err:.2e}"


# --- Multi-model tests (larger, run with --run-all-models) ---

_MULTI_MODEL_CASES = []
for _model, (_hd, _total_qh, _total_kh) in MODEL_CONFIGS.items():
    for _tp in [1, 8]:
        _qh = _total_qh // _tp
        _kh = max(1, _total_kh // _tp)
        if _qh >= 1:
            _MULTI_MODEL_CASES.append(
                pytest.param(_model, _hd, _qh, _kh, id=f"{_model}-TP{_tp}")
            )


@pytest.mark.parametrize("model,head_dim,num_q_heads,num_kv_heads", _MULTI_MODEL_CASES)
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
def test_fused_rope_cache_multi_model(model, head_dim, num_q_heads, num_kv_heads,
                                       num_tokens, flash_layout):
    ok, q_err, k_err, kc_err, vc_err = run_fused_test(
        num_tokens, head_dim=head_dim,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
        flash_layout=flash_layout,
    )
    assert ok, f"FAILED ({model}): q={q_err:.2e} k={k_err:.2e} kc={kc_err:.2e} vc={vc_err:.2e}"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-models", action="store_true",
                        help="Test all model configs (default: GPT-OSS-120B TP=8 only)")
    args = parser.parse_args()

    configs = []
    if args.all_models:
        for model, (hd, total_qh, total_kh) in MODEL_CONFIGS.items():
            for tp in [1, 8]:
                qh = total_qh // tp
                kh = max(1, total_kh // tp)
                if qh >= 1:
                    configs.append((model, tp, hd, qh, kh))
    else:
        configs = [("GPT-OSS-120B", 8, HEAD_DIM, NUM_Q_HEADS, NUM_KV_HEADS)]

    for model, tp, hd, qh, kh in configs:
        print(f"\n{'='*60}")
        print(f"{model} TP={tp}: QH={qh}, KH={kh}, D={hd}")
        print(f"{'='*60}")
        for flash_layout in [True, False]:
            layout = "flash" if flash_layout else "non-flash"
            for m in [1, 4, 32, 128]:
                ok, q_err, k_err, kc_err, vc_err = run_fused_test(
                    m, head_dim=hd, num_q_heads=qh, num_kv_heads=kh,
                    flash_layout=flash_layout,
                )
                status = "PASS" if ok else "FAIL"
                print(f"  [{status}] {layout:>9s} M={m:>4d} "
                      f"q={q_err:.2e} k={k_err:.2e} kc={kc_err:.2e} vc={vc_err:.2e}")
    print("\nDone.")
