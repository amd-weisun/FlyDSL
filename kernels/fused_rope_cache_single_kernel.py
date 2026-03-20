"""Fused RoPE + KV Cache — SINGLE kernel launch (matches Triton's design).

Grid: (T * QH + T * KH, 1, 1)
  - pid < T*QH: Q programs (RoPE on Q → q_out)
  - pid >= T*QH: K programs (RoPE on K → k_out + key_cache + value_cache)

Dispatch is branchless via arith.select (pid determines Q vs K path).
Block_idx is uniform across wavefront → no divergence.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T, Int32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops


KERNEL_NAME = "fused_rope_cache_v2"
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


def build_fused_rope_cache_single_module(
    head_dim: int = 64,
    rotary_dim: int = 64,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    block_size: int = 16,
    is_neox: bool = True,
    flash_layout: bool = True,
    dtype_str: str = "bf16",
):
    """Build single-kernel fused RoPE + KV cache (matches Triton's design)."""
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation not yet supported")

    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    half_dim = rotary_dim // 2
    elem_bytes = 4 if dtype_str == "f32" else 2
    vec_dwords = (VEC_WIDTH * elem_bytes) // 4
    vecs_per_half = half_dim // VEC_WIDTH
    vecs_per_head = head_dim // VEC_WIDTH

    q_head_stride = head_dim * elem_bytes           # bytes per Q head
    k_head_stride = head_dim * elem_bytes           # bytes per K head
    q_token_stride = num_q_heads * q_head_stride    # bytes per Q token row
    k_token_stride = num_kv_heads * k_head_stride   # bytes per K token row

    BLOCK_THREADS = WARP_SIZE

    @flyc.kernel
    def fused_rope_cache_kernel(
        Q: fx.Tensor,            # [T, QH, D]
        K: fx.Tensor,            # [T, KH, D]
        V: fx.Tensor,            # [T, KH, D]
        Positions: fx.Tensor,    # [T] int32
        CosCache: fx.Tensor,     # [max_pos, half_dim]
        SinCache: fx.Tensor,     # [max_pos, half_dim]
        SlotMapping: fx.Tensor,  # [T] int32
        KeyCache: fx.Tensor,     # [T_cache, BS, KH, D] (flash layout)
        ValueCache: fx.Tensor,   # [T_cache, BS, KH, D] (flash layout)
        Q_out: fx.Tensor,        # [T, QH, D]
        K_out: fx.Tensor,        # [T, KH, D]
        NumQPrograms: fx.Tensor, # [1] int32 — T * QH (passed as 1-element tensor)
    ):
        # ArithValue imported at module level

        pid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        vec_type_c = T.vec(VEC_WIDTH, compute_type)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        # All buffer resources at top level
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(V, max_size=True)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        slot_rsrc = buffer_ops.create_buffer_resource(SlotMapping, max_size=True)
        kc_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
        vc_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)
        qo_rsrc = buffer_ops.create_buffer_resource(Q_out, max_size=True)
        ko_rsrc = buffer_ops.create_buffer_resource(K_out, max_size=True)
        nqp_rsrc = buffer_ops.create_buffer_resource(NumQPrograms, max_size=True)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            # Load num_q_programs (T * QH) — uniform across block
            num_q_progs = buffer_ops.buffer_load(nqp_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)

            # --- Runtime dispatch: Q or K program? ---
            is_q = arith.cmpi(arith.CmpIPredicate.slt, pid, num_q_progs)

            # Compute token/head for BOTH paths, then select
            q_pid_t = pid // num_q_heads
            q_pid_h = pid % num_q_heads
            k_pid_offset = ArithValue(pid) - ArithValue(num_q_progs)
            # Clamp k_pid_offset to 0 for Q programs (avoids negative division)
            k_pid_safe = arith.select(is_q, fx.Int32(0), k_pid_offset)
            k_pid_t = ArithValue(k_pid_safe) // num_kv_heads
            k_pid_h = ArithValue(k_pid_safe) % num_kv_heads

            pid_t = arith.select(is_q, q_pid_t, k_pid_t)
            pid_h = arith.select(is_q, q_pid_h, k_pid_h)

            # --- Load position (same for both paths) ---
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # --- Load cos/sin ---
            cos_vec_idx = tid % vecs_per_half
            cos_bytes = ArithValue(pos_val) * (half_dim * elem_bytes) + ArithValue(cos_vec_idx) * (VEC_WIDTH * elem_bytes)
            cos_dw = cos_bytes >> fx.Int32(2)
            cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
            cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
            sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)
            cos_f32 = cos_e.extf(vec_type_c) if dtype_str != "f32" else cos_e
            sin_f32 = sin_e.extf(vec_type_c) if dtype_str != "f32" else sin_e

            # --- Load input (Q or K selected by byte offset) ---
            vec_off = ArithValue(tid) * (VEC_WIDTH * elem_bytes)
            q_in_bytes = ArithValue(pid_t) * q_token_stride + ArithValue(pid_h) * q_head_stride + vec_off
            k_in_bytes = ArithValue(pid_t) * k_token_stride + ArithValue(pid_h) * k_head_stride + vec_off
            in_bytes = arith.select(is_q, q_in_bytes, k_in_bytes)
            in_dw = in_bytes >> fx.Int32(2)

            # Load from Q (for Q programs) or K (for K programs)
            q_raw = buffer_ops.buffer_load(q_rsrc, in_dw, vec_width=vec_dwords, dtype=T.i32)
            k_raw = buffer_ops.buffer_load(k_rsrc, in_dw, vec_width=vec_dwords, dtype=T.i32)
            in_raw = arith.select(is_q, q_raw, k_raw)
            in_e = vector.bitcast(vec_type_e, in_raw) if vec_dwords != VEC_WIDTH else in_raw.bitcast(vec_type_e)
            in_f32 = in_e.extf(vec_type_c) if dtype_str != "f32" else in_e

            # --- Load paired half (branchless) ---
            is_first_half = arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half))
            pair_off_first = in_bytes + (half_dim * elem_bytes)
            pair_off_second = in_bytes - (half_dim * elem_bytes)
            pair_bytes = arith.select(is_first_half, pair_off_first, pair_off_second)
            pair_dw = pair_bytes >> fx.Int32(2)

            q_pair_raw = buffer_ops.buffer_load(q_rsrc, pair_dw, vec_width=vec_dwords, dtype=T.i32)
            k_pair_raw = buffer_ops.buffer_load(k_rsrc, pair_dw, vec_width=vec_dwords, dtype=T.i32)
            pair_raw = arith.select(is_q, q_pair_raw, k_pair_raw)
            pair_e = vector.bitcast(vec_type_e, pair_raw) if vec_dwords != VEC_WIDTH else pair_raw.bitcast(vec_type_e)
            pair_f32 = pair_e.extf(vec_type_c) if dtype_str != "f32" else pair_e

            # --- NeoX rotation (branchless) ---
            in_cos = ArithValue(in_f32) * ArithValue(cos_f32)
            pair_sin = ArithValue(pair_f32) * ArithValue(sin_f32)
            neg_pair_sin = arith.negf(pair_sin)
            sin_term = arith.select(is_first_half, neg_pair_sin, pair_sin)
            rot_f32 = ArithValue(in_cos) + ArithValue(sin_term)

            # --- Truncate to output dtype ---
            if dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32:
                rot_e = rot_f32.truncf(vec_type_e)
            elif dtype_str == "f32":
                rot_e = rot_f32
            else:
                rot_e = rot_f32.truncf(vec_type_e)
            rot_i32 = vector.bitcast(i32_vec_ty, rot_e) if vec_dwords != VEC_WIDTH else rot_e.bitcast(i32_vec_ty)

            # --- Store rotated output ---
            # Q programs → Q_out, K programs → K_out
            # Since block_idx is uniform, this branch is non-divergent
            if arith.cmpi(arith.CmpIPredicate.slt, pid, num_q_progs):
                # Q program: store to Q_out
                qo_bytes = ArithValue(pid_t) * q_token_stride + ArithValue(pid_h) * q_head_stride + vec_off
                qo_dw = qo_bytes >> fx.Int32(2)
                buffer_ops.buffer_store(rot_i32, qo_rsrc, qo_dw)
            else:
                # K program: store to K_out + KV cache
                ko_bytes = ArithValue(pid_t) * k_token_stride + ArithValue(pid_h) * k_head_stride + vec_off
                ko_dw = ko_bytes >> fx.Int32(2)
                buffer_ops.buffer_store(rot_i32, ko_rsrc, ko_dw)

                # KV cache write
                slot_val = buffer_ops.buffer_load(slot_rsrc, pid_t, vec_width=1, dtype=T.i32)
                if arith.cmpi(arith.CmpIPredicate.sge, slot_val, fx.Int32(0)):
                    pid_t_slot = ArithValue(slot_val) // block_size
                    pid_b = ArithValue(slot_val) % block_size

                    if flash_layout:
                        # key_cache: [T_cache, BS, KH, D]
                        kc_bytes = (
                            ArithValue(pid_t_slot) * (block_size * num_kv_heads * head_dim * elem_bytes)
                            + ArithValue(pid_b) * (num_kv_heads * head_dim * elem_bytes)
                            + ArithValue(pid_h) * (head_dim * elem_bytes)
                            + vec_off
                        )
                        kc_dw = kc_bytes >> fx.Int32(2)
                        buffer_ops.buffer_store(rot_i32, kc_rsrc, kc_dw)

                        # value_cache: [T_cache, BS, KH, D] — load V and store
                        v_raw = buffer_ops.buffer_load(v_rsrc, ko_dw, vec_width=vec_dwords, dtype=T.i32)
                        vc_dw = kc_dw  # Same layout as key_cache for flash
                        buffer_ops.buffer_store(v_raw, vc_rsrc, vc_dw)

    @flyc.jit
    def launch_fused_rope_cache_single(
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
        NumQPrograms: fx.Tensor,
        num_total_programs: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_rope_cache_kernel(
            Q, K, V, Positions, CosCache, SinCache, SlotMapping,
            KeyCache, ValueCache, Q_out, K_out, NumQPrograms,
        )
        launcher.launch(
            grid=(num_total_programs, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_rope_cache_single
