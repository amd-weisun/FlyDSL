#!/usr/bin/env python3
"""RoPE Kernel Test — NeoX-style rotation with cos/sin cache and position indexing.

Kernel implementation lives in `kernels/rope_kernel.py`.
This file is the correctness + perf harness.

Usage:
    # Fast CI (GPT-OSS 120B TP=8, 12 tests):
    PYTHONPATH=./ pytest tests/kernels/test_rope.py -v -s

    # All models × TPs (multi-model sweep):
    PYTHONPATH=./ pytest tests/kernels/test_rope.py -v -s -m multi_model

    # CLI — single config:
    PYTHONPATH=./ python tests/kernels/test_rope.py --num-tokens 32

    # CLI — all models:
    PYTHONPATH=./ python tests/kernels/test_rope.py --all-models

    # CLI — with AITER comparison:
    AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/test_rope.py --all-models
"""

import os
import sys
import logging

import torch
import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_PYFLYDSL_SRC = os.path.join(_REPO_ROOT, "flydsl", "src")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _PYFLYDSL_SRC not in sys.path:
    sys.path.insert(0, _PYFLYDSL_SRC)

from kernels.rope_kernel import build_rope_module
from tests.test_common import run_perftest, verify_output

logging.basicConfig(level=logging.INFO)

try:
    from benchmark_common import bench_gpu_us_torch, maybe_enable_aiter
    HAS_BENCH = True
except ImportError:
    HAS_BENCH = False

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available. Skipping GPU tests.", allow_module_level=True)

MAX_POS = 8192
DEFAULT_BENCH_ITERS = 20
DEFAULT_BENCH_WARMUP = 3

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


def rope_neox_ref(q, k, cos_cache, sin_cache, positions):
    """PyTorch reference for NeoX-style RoPE.

    Args:
        q: [M, num_q_heads, head_dim] bf16/f16/f32
        k: [M, num_kv_heads, head_dim] bf16/f16/f32
        cos_cache: [max_pos, head_dim//2] bf16/f16/f32
        sin_cache: [max_pos, head_dim//2] bf16/f16/f32
        positions: [M] int32/int64

    Returns:
        q_out, k_out: same shapes as input
    """
    half_dim = cos_cache.shape[-1]
    cos = cos_cache[positions.long()].unsqueeze(1).to(torch.float32)  # [M, 1, half_dim]
    sin = sin_cache[positions.long()].unsqueeze(1).to(torch.float32)  # [M, 1, half_dim]

    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)

    q1, q2 = q_f32[..., :half_dim], q_f32[..., half_dim:]
    q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).to(q.dtype)

    k1, k2 = k_f32[..., :half_dim], k_f32[..., half_dim:]
    k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).to(k.dtype)

    return q_out, k_out


def run_rope_test(
    num_tokens: int,
    head_dim: int = HEAD_DIM,
    rotary_dim: int = ROTARY_DIM,
    num_q_heads: int = NUM_Q_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    max_pos: int = MAX_POS,
    dtype_str: str = "bf16",
    num_iters: int = DEFAULT_BENCH_ITERS,
    num_warmup: int = DEFAULT_BENCH_WARMUP,
):
    """Run RoPE kernel test: compile, execute, verify against PyTorch reference."""
    torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_str]
    device = torch.device("cuda")

    print("=" * 80)
    print(f"[flyc] RoPE Test: M={num_tokens}, head_dim={head_dim}, "
          f"Q_heads={num_q_heads}, KV_heads={num_kv_heads}, dtype={dtype_str}")
    print("=" * 80)

    # Build kernel
    launch_fn = build_rope_module(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        is_neox=True,
        dtype_str=dtype_str,
    )
    print("✓ Kernel compiled")

    # Create test data
    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_q_heads, head_dim, device=device, dtype=torch_dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=device, dtype=torch_dtype)
    cos_cache = torch.randn(max_pos, rotary_dim // 2, device=device, dtype=torch_dtype)
    sin_cache = torch.randn(max_pos, rotary_dim // 2, device=device, dtype=torch_dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Reference
    q_ref, k_ref = rope_neox_ref(q, k, cos_cache, sin_cache, positions)

    # Launch FlyDSL kernel
    def launch(qo, ko, qi, ki, cc, sc, pos):
        launch_fn(qi, ki, cc, sc, pos, qo, ko, num_tokens, stream=torch.cuda.current_stream())

    _, us = run_perftest(
        launch, q_out, k_out, q, k, cos_cache, sin_cache, positions,
        num_iters=num_iters, num_warmup=num_warmup,
    )
    torch.cuda.synchronize()

    # Verify
    rtol = 0.05 if dtype_str == "f32" else 0.1
    atol = 0.05 if dtype_str == "f32" else 0.1

    q_ok = verify_output(q_out.to(torch.float32), q_ref.to(torch.float32), rtol=rtol, atol=atol, msg="Q RoPE")
    k_ok = verify_output(k_out.to(torch.float32), k_ref.to(torch.float32), rtol=rtol, atol=atol, msg="K RoPE")

    assert q_ok, "Q RoPE verification failed"
    assert k_ok, "K RoPE verification failed"

    # Performance stats
    total_heads = num_q_heads + num_kv_heads
    # Read: Q + K + cos + sin + positions
    # Write: Q_out + K_out
    bytes_read = num_tokens * (num_q_heads + num_kv_heads) * head_dim * 2  # bf16
    bytes_read += num_tokens * rotary_dim // 2 * 2 * 2  # cos + sin bf16
    bytes_read += num_tokens * 4  # positions i32
    bytes_written = num_tokens * (num_q_heads + num_kv_heads) * head_dim * 2
    total_bytes = bytes_read + bytes_written
    bw_gbps = total_bytes / 1e9 / (us / 1e6)

    print(f"[flyc] RoPE: {us:.1f} us, BW: {bw_gbps:.2f} GB/s "
          f"(M={num_tokens}, heads={total_heads})")


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32, 64, 128, 256, 1024])
def test_rope_correctness(num_tokens):
    """Test RoPE correctness at various batch sizes (GPT-OSS decode concurrencies)."""
    run_rope_test(num_tokens)


def test_rope_inplace():
    """Test inplace mode: Q_out=Q, K_out=K."""
    torch_dtype = torch.bfloat16
    device = torch.device("cuda")
    num_tokens = 32

    launch_fn = build_rope_module(
        head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        is_neox=True, dtype_str="bf16",
    )

    torch.manual_seed(42)
    q = torch.randn(num_tokens, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch_dtype)
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch_dtype)
    cos_cache = torch.randn(MAX_POS, ROTARY_DIM // 2, device=device, dtype=torch_dtype)
    sin_cache = torch.randn(MAX_POS, ROTARY_DIM // 2, device=device, dtype=torch_dtype)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device=device, dtype=torch.int32)

    q_ref, k_ref = rope_neox_ref(q.clone(), k.clone(), cos_cache, sin_cache, positions)

    # Inplace: pass same tensor for input and output
    launch_fn(q, k, cos_cache, sin_cache, positions, q, k,
              num_tokens, stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    assert verify_output(q.float(), q_ref.float(), rtol=0.1, atol=0.1, msg="Q inplace")
    assert verify_output(k.float(), k_ref.float(), rtol=0.1, atol=0.1, msg="K inplace")


def test_rope_single_token():
    """Regression test for M=1 (single decode step)."""
    run_rope_test(num_tokens=1)


@pytest.mark.parametrize("dtype_str", ["bf16", "f16"])
def test_rope_dtypes(dtype_str):
    """Test RoPE with different data types."""
    run_rope_test(num_tokens=32, dtype_str=dtype_str)


# --- Multi-model tests (run with -m multi_model) ---

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
@pytest.mark.multi_model
def test_rope_multi_model(model, head_dim, num_q_heads, num_kv_heads, num_tokens):
    run_rope_test(num_tokens, head_dim=head_dim, rotary_dim=head_dim,
                  num_q_heads=num_q_heads, num_kv_heads=num_kv_heads)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="RoPE kernel test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Single config:
  PYTHONPATH=./ python tests/kernels/test_rope.py --num-tokens 32

  # All models × TPs:
  PYTHONPATH=./ python tests/kernels/test_rope.py --all-models

  # With AITER comparison:
  AITER_REPO=../aiter PYTHONPATH=./ python tests/kernels/test_rope.py --all-models
""",
    )
    parser.add_argument("--num-tokens", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["f32", "f16", "bf16"])
    parser.add_argument("--num-iters", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=3)
    parser.add_argument("--all-models", action="store_true",
                        help="Test all model configs × TP values")
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
        for m in [1, 4, 32, 128]:
            run_rope_test(m, head_dim=hd, rotary_dim=hd,
                          num_q_heads=qh, num_kv_heads=kh,
                          dtype_str=args.dtype,
                          num_iters=args.num_iters,
                          num_warmup=args.num_warmup)
    print("\nDone.")
