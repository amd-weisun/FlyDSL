#!/usr/bin/env python3
"""Fused RoPE + KV Cache kernel test.

Tests correctness of the fused kernel against PyTorch reference.
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

logging.basicConfig(level=logging.INFO)

if not torch.cuda.is_available():
    pytest.skip("CUDA/ROCm not available.", allow_module_level=True)

# GPT-OSS 120B config (TP=8)
HEAD_DIM = 64
ROTARY_DIM = 64
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
BLOCK_SIZE = 16
MAX_POS = 8192


def fused_rope_cache_ref(q, k, v, cos_cache, sin_cache, positions, slot_mapping,
                          key_cache, value_cache, block_size):
    """PyTorch reference for fused RoPE + KV cache."""
    half_dim = cos_cache.shape[-1]
    cos = cos_cache[positions.long()].unsqueeze(1).float()
    sin = sin_cache[positions.long()].unsqueeze(1).float()

    # RoPE Q
    q_f32 = q.float()
    q1, q2 = q_f32[..., :half_dim], q_f32[..., half_dim:]
    q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1).to(q.dtype)

    # RoPE K
    k_f32 = k.float()
    k1, k2 = k_f32[..., :half_dim], k_f32[..., half_dim:]
    k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + k1 * sin], dim=-1).to(k.dtype)

    # KV cache write (flash layout: [num_blocks, block_size, KH, D])
    key_cache_out = key_cache.clone()
    value_cache_out = value_cache.clone()
    for i in range(slot_mapping.shape[0]):
        slot = slot_mapping[i].item()
        if slot >= 0:
            block_idx = slot // block_size
            block_pos = slot % block_size
            key_cache_out[block_idx, block_pos] = k_out[i]
            value_cache_out[block_idx, block_pos] = v[i]

    return q_out, k_out, key_cache_out, value_cache_out


def run_fused_test(num_tokens, block_size=BLOCK_SIZE, max_pos=MAX_POS):
    """Run fused RoPE + KV cache kernel test."""
    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
    num_blocks = 32  # enough blocks for test

    print(f"[fused_rope_cache] M={num_tokens}, BS={block_size}, "
          f"QH={NUM_Q_HEADS}, KH={NUM_KV_HEADS}, D={HEAD_DIM}")

    launch_fn = build_fused_rope_cache_module(
        head_dim=HEAD_DIM, rotary_dim=ROTARY_DIM,
        num_q_heads=NUM_Q_HEADS, num_kv_heads=NUM_KV_HEADS,
        block_size=block_size, is_neox=True,
        flash_layout=True, dtype_str="bf16",
    )

    torch.manual_seed(42)
    q = torch.randn(num_tokens, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=torch_dtype)
    k = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch_dtype)
    v = torch.randn(num_tokens, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=torch_dtype)
    cos_cache = torch.randn(max_pos, ROTARY_DIM // 2, device=device, dtype=torch_dtype)
    sin_cache = torch.randn(max_pos, ROTARY_DIM // 2, device=device, dtype=torch_dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device=device, dtype=torch.int32)
    # Slot mapping: assign sequential slots
    slot_mapping = torch.arange(num_tokens, device=device, dtype=torch.int32)

    key_cache = torch.zeros(num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM,
                             device=device, dtype=torch_dtype)
    value_cache = torch.zeros(num_blocks, block_size, NUM_KV_HEADS, HEAD_DIM,
                               device=device, dtype=torch_dtype)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Reference
    q_ref, k_ref, kc_ref, vc_ref = fused_rope_cache_ref(
        q, k, v, cos_cache, sin_cache, positions, slot_mapping,
        key_cache.clone(), value_cache.clone(), block_size,
    )

    # Launch FlyDSL kernel
    stream = torch.cuda.current_stream()
    launch_fn(q, k, v, positions, cos_cache, sin_cache, slot_mapping,
              key_cache, value_cache, q_out, k_out, num_tokens, stream=stream)
    torch.cuda.synchronize()

    # Verify
    atol = 0.1
    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()

    # Check KV cache for written slots
    kc_err = 0.0
    vc_err = 0.0
    for i in range(num_tokens):
        slot = slot_mapping[i].item()
        bi, bp = slot // block_size, slot % block_size
        kc_err = max(kc_err, (key_cache[bi, bp].float() - kc_ref[bi, bp].float()).abs().max().item())
        vc_err = max(vc_err, (value_cache[bi, bp].float() - vc_ref[bi, bp].float()).abs().max().item())

    print(f"  q_err={q_err:.6f}, k_err={k_err:.6f}, kc_err={kc_err:.6f}, vc_err={vc_err:.6f}")

    ok = q_err < atol and k_err < atol and kc_err < atol and vc_err < atol
    return ok, q_err, k_err, kc_err, vc_err


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32, 128])
def test_fused_rope_cache(num_tokens):
    ok, q_err, k_err, kc_err, vc_err = run_fused_test(num_tokens)
    assert ok, f"FAILED: q={q_err:.2e} k={k_err:.2e} kc={kc_err:.2e} vc={vc_err:.2e}"


if __name__ == "__main__":
    for m in [1, 4, 16, 32, 128]:
        ok, q_err, k_err, kc_err, vc_err = run_fused_test(m)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] M={m}")
    print("Done.")
