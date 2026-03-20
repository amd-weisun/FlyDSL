"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.

Fuses 4 operations (matching aiter's _fused_qk_rope_reshape_and_cache_kernel):
  1. RoPE on Q → q_out                     (kernel 1: q_rope_kernel)
  2. RoPE on K → k_out + key_cache write   (kernel 2: k_rope_cache_kernel)
  3. V → value_cache write                 (kernel 2: k_rope_cache_kernel)

Uses two kernel launches for simplicity (can be fused into one later).
Each thread handles one vec8 chunk (8 bf16 elements) of one head.

GPT-OSS 120B config (TP=8):
  Q: [T, 8, 64] bf16, K: [T, 1, 64] bf16, V: [T, 1, 64] bf16
  key_cache:   flash: [num_blocks, block_size, KH, D]
               paged: [num_blocks, KH, D//x, block_size, x]
  value_cache: flash: [num_blocks, block_size, KH, D]
               paged: [num_blocks, KH, D, block_size]
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops


KERNEL_NAME = "fused_rope_cache"
WARP_SIZE = 64
VEC_WIDTH = 8


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported dtype: {dtype_str}")


def build_fused_rope_cache_module(
    head_dim: int = 64,
    rotary_dim: int = 64,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    is_neox: bool = True,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
):
    """Build fused RoPE + KV cache kernel.

    Args:
        head_dim: dimension per attention head
        rotary_dim: dimensions to rotate (== head_dim for full rotation)
        num_q_heads: query heads per rank
        num_kv_heads: KV heads per rank
        block_size: paged attention block size
        is_neox: True for NeoX-style rotation
        flash_layout: True for [num_blocks, block_size, KH, D] cache layout
        dtype_str: element dtype ("bf16", "f16")

    Returns:
        launch_fn(Q, K, V, Positions, CosCache, SinCache, SlotMapping,
                  KeyCache, ValueCache, Q_out, K_out, num_tokens, stream)
    """
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation not yet supported")

    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    half_dim = rotary_dim // 2
    elem_bytes = 4 if dtype_str == "f32" else 2
    vec_dwords = (VEC_WIDTH * elem_bytes) // 4
    vecs_per_half = half_dim // VEC_WIDTH   # 4
    vecs_per_head = head_dim // VEC_WIDTH   # 8

    BLOCK_THREADS = WARP_SIZE

    # ----- Kernel 1: Q RoPE -----
    # Grid: (T * QH, 1, 1), one program per (token, q_head)
    # Each program: vecs_per_head threads process head_dim elements
    @flyc.kernel
    def q_rope_kernel(
        Q: fx.Tensor,            # [T, QH, D]
        Positions: fx.Tensor,    # [T] int32
        CosCache: fx.Tensor,     # [max_pos, half_dim]
        SinCache: fx.Tensor,     # [max_pos, half_dim]
        Q_out: fx.Tensor,        # [T, QH, D]
    ):
        from flydsl.expr.arith import ArithValue

        pid = fx.block_idx.x    # program id: 0..T*QH-1
        tid = fx.thread_idx.x   # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        vec_type_c = T.vec(VEC_WIDTH, compute_type)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        qo_rsrc = buffer_ops.create_buffer_resource(Q_out, max_size=True)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            pid_t = pid // num_q_heads
            pid_hq = pid % num_q_heads

            # Load position
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # Load cos/sin for this position
            # vec_idx 0..3 → first half, vec_idx 4..7 → second half (same cos/sin)
            cos_vec_idx = tid % vecs_per_half
            cos_bytes = ArithValue(pos_val) * (half_dim * elem_bytes) + ArithValue(cos_vec_idx) * (VEC_WIDTH * elem_bytes)
            cos_dw = cos_bytes >> fx.Int32(2)

            cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
            sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)
            cos_f32 = cos_e.extf(vec_type_c) if dtype_str != "f32" else cos_e
            sin_f32 = sin_e.extf(vec_type_c) if dtype_str != "f32" else sin_e

            # Load Q element
            q_bytes = ArithValue(pid_t) * (num_q_heads * head_dim * elem_bytes) + ArithValue(pid_hq) * (head_dim * elem_bytes) + ArithValue(tid) * (VEC_WIDTH * elem_bytes)
            q_dw = q_bytes >> fx.Int32(2)
            q_raw = buffer_ops.buffer_load(q_rsrc, q_dw, vec_width=vec_dwords, dtype=T.i32)
            q_e = vector.bitcast(vec_type_e, q_raw) if vec_dwords != VEC_WIDTH else q_raw.bitcast(vec_type_e)
            q_f32 = q_e.extf(vec_type_c) if dtype_str != "f32" else q_e

            # Load paired half for rotation
            if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half)):
                # First half — pair is second half (offset +half_dim)
                pair_bytes = q_bytes + (half_dim * elem_bytes)
            else:
                # Second half — pair is first half (offset -half_dim)
                pair_bytes = q_bytes - (half_dim * elem_bytes)
            pair_dw = pair_bytes >> fx.Int32(2)
            pair_raw = buffer_ops.buffer_load(q_rsrc, pair_dw, vec_width=vec_dwords, dtype=T.i32)
            pair_e = vector.bitcast(vec_type_e, pair_raw) if vec_dwords != VEC_WIDTH else pair_raw.bitcast(vec_type_e)
            pair_f32 = pair_e.extf(vec_type_c) if dtype_str != "f32" else pair_e

            # NeoX rotation
            if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half)):
                # first_half: out = q*cos - pair*sin
                rot_f32 = ArithValue(q_f32) * ArithValue(cos_f32) - ArithValue(pair_f32) * ArithValue(sin_f32)
            else:
                # second_half: out = q*cos + pair*sin
                rot_f32 = ArithValue(q_f32) * ArithValue(cos_f32) + ArithValue(pair_f32) * ArithValue(sin_f32)

            # Store
            if dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32:
                rot_e = rot_f32.truncf(vec_type_e)
            elif dtype_str == "f32":
                rot_e = rot_f32
            else:
                rot_e = rot_f32.truncf(vec_type_e)

            rot_i32 = vector.bitcast(i32_vec_ty, rot_e) if vec_dwords != VEC_WIDTH else rot_e.bitcast(i32_vec_ty)
            buffer_ops.buffer_store(rot_i32, qo_rsrc, q_dw)

    # ----- Kernel 2: K RoPE + KV cache write -----
    # Grid: (T * KH, 1, 1), one program per (token, kv_head)
    # Each program: vecs_per_head threads process head_dim elements
    # Writes: k_out (rotated K), key_cache (rotated K to paged cache), value_cache (V to paged cache)
    @flyc.kernel
    def k_cache_kernel(
        K: fx.Tensor,            # [T, KH, D]
        V: fx.Tensor,            # [T, KH, D]
        Positions: fx.Tensor,    # [T] int32
        CosCache: fx.Tensor,     # [max_pos, half_dim]
        SinCache: fx.Tensor,     # [max_pos, half_dim]
        SlotMapping: fx.Tensor,  # [T] int32
        KeyCache: fx.Tensor,     # flash: [T_cache, BS, KH, D]
        ValueCache: fx.Tensor,   # flash: [T_cache, BS, KH, D]
        K_out: fx.Tensor,        # [T, KH, D]
    ):
        from flydsl.expr.arith import ArithValue

        pid = fx.block_idx.x    # program id: 0..T*KH-1
        tid = fx.thread_idx.x   # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        vec_type_c = T.vec(VEC_WIDTH, compute_type)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        slot_rsrc = buffer_ops.create_buffer_resource(SlotMapping, max_size=True)
        kc_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
        vc_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)
        ko_rsrc = buffer_ops.create_buffer_resource(K_out, max_size=True)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            pid_t = pid // num_kv_heads
            pid_hk = pid % num_kv_heads

            # Load position
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # Load cos/sin
            cos_vec_idx = tid % vecs_per_half
            cos_bytes = ArithValue(pos_val) * (half_dim * elem_bytes) + ArithValue(cos_vec_idx) * (VEC_WIDTH * elem_bytes)
            cos_dw = cos_bytes >> fx.Int32(2)
            cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
            sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)
            cos_f32 = cos_e.extf(vec_type_c) if dtype_str != "f32" else cos_e
            sin_f32 = sin_e.extf(vec_type_c) if dtype_str != "f32" else sin_e

            # Load K
            k_bytes = ArithValue(pid_t) * (num_kv_heads * head_dim * elem_bytes) + ArithValue(pid_hk) * (head_dim * elem_bytes) + ArithValue(tid) * (VEC_WIDTH * elem_bytes)
            k_dw = k_bytes >> fx.Int32(2)
            k_raw = buffer_ops.buffer_load(k_rsrc, k_dw, vec_width=vec_dwords, dtype=T.i32)
            k_e = vector.bitcast(vec_type_e, k_raw) if vec_dwords != VEC_WIDTH else k_raw.bitcast(vec_type_e)
            k_f32 = k_e.extf(vec_type_c) if dtype_str != "f32" else k_e

            # Load K paired half
            if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half)):
                pair_bytes = k_bytes + (half_dim * elem_bytes)
            else:
                pair_bytes = k_bytes - (half_dim * elem_bytes)
            pair_dw = pair_bytes >> fx.Int32(2)
            pair_raw = buffer_ops.buffer_load(k_rsrc, pair_dw, vec_width=vec_dwords, dtype=T.i32)
            pair_e = vector.bitcast(vec_type_e, pair_raw) if vec_dwords != VEC_WIDTH else pair_raw.bitcast(vec_type_e)
            pair_f32 = pair_e.extf(vec_type_c) if dtype_str != "f32" else pair_e

            # K RoPE rotation
            if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half)):
                k_rot_f32 = ArithValue(k_f32) * ArithValue(cos_f32) - ArithValue(pair_f32) * ArithValue(sin_f32)
            else:
                k_rot_f32 = ArithValue(k_f32) * ArithValue(cos_f32) + ArithValue(pair_f32) * ArithValue(sin_f32)

            if dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32:
                k_rot_e = k_rot_f32.truncf(vec_type_e)
            elif dtype_str == "f32":
                k_rot_e = k_rot_f32
            else:
                k_rot_e = k_rot_f32.truncf(vec_type_e)

            # Store k_out
            k_rot_i32 = vector.bitcast(i32_vec_ty, k_rot_e) if vec_dwords != VEC_WIDTH else k_rot_e.bitcast(i32_vec_ty)
            buffer_ops.buffer_store(k_rot_i32, ko_rsrc, k_dw)

            # --- KV Cache write ---
            # Load slot_mapping[pid_t] → slot index
            slot_val = buffer_ops.buffer_load(slot_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # slot_val >= 0 means valid slot
            if arith.cmpi(arith.CmpIPredicate.sge, slot_val, fx.Int32(0)):
                # Decompose slot into block_idx and position within block
                pid_t_slot = ArithValue(slot_val) // block_size
                pid_b = ArithValue(slot_val) % block_size

                if flash_layout:
                    # key_cache: [T_cache, BS, KH, D]
                    # byte offset = pid_t_slot * BS * KH * D * elem_bytes
                    #              + pid_b * KH * D * elem_bytes
                    #              + pid_hk * D * elem_bytes
                    #              + tid * VEC_WIDTH * elem_bytes
                    kc_bytes = (
                        ArithValue(pid_t_slot) * (block_size * num_kv_heads * head_dim * elem_bytes)
                        + ArithValue(pid_b) * (num_kv_heads * head_dim * elem_bytes)
                        + ArithValue(pid_hk) * (head_dim * elem_bytes)
                        + ArithValue(tid) * (VEC_WIDTH * elem_bytes)
                    )
                    kc_dw = kc_bytes >> fx.Int32(2)
                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw)

                    # value_cache: [T_cache, BS, KH, D]
                    # Same layout as key_cache for flash
                    v_raw = buffer_ops.buffer_load(v_rsrc, k_dw, vec_width=vec_dwords, dtype=T.i32)
                    vc_bytes = (
                        ArithValue(pid_t_slot) * (block_size * num_kv_heads * head_dim * elem_bytes)
                        + ArithValue(pid_b) * (num_kv_heads * head_dim * elem_bytes)
                        + ArithValue(pid_hk) * (head_dim * elem_bytes)
                        + ArithValue(tid) * (VEC_WIDTH * elem_bytes)
                    )
                    vc_dw = vc_bytes >> fx.Int32(2)
                    buffer_ops.buffer_store(v_raw, vc_rsrc, vc_dw)

    @flyc.jit
    def launch_fused_rope_cache(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        Positions: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        SlotMapping: fx.Tensor,
        KeyCache: fx.Tensor,
        ValueCache: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # Kernel 1: Q RoPE
        n_q = ArithValue(num_tokens) * num_q_heads
        q_launcher = q_rope_kernel(Q, Positions, CosCache, SinCache, Q_out)
        q_launcher.launch(
            grid=(n_q, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

        # Kernel 2: K RoPE + KV cache write
        n_k = ArithValue(num_tokens) * num_kv_heads
        k_launcher = k_cache_kernel(
            K, V, Positions, CosCache, SinCache, SlotMapping,
            KeyCache, ValueCache, K_out,
        )
        k_launcher.launch(
            grid=(n_k, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_rope_cache
