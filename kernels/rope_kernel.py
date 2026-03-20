"""RoPE (Rotary Position Embedding) kernel builder using the @flyc.kernel API.

NeoX-style rotation:
  first_half  = x[..., :D//2]
  second_half = x[..., D//2:]
  out[..., :D//2]  = first_half * cos[pos] - second_half * sin[pos]
  out[..., D//2:]  = second_half * cos[pos] + first_half * sin[pos]

Processes Q (multi-head) and K (fewer heads) together with shared cos/sin lookup.
Supports inplace (Q_out=Q, K_out=K) since each element is written by exactly one thread.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops


KERNEL_NAME = "rope"

WARP_SIZE = 64
VEC_WIDTH = 8  # 8 x bf16 = 128 bits = buffer_load dwordx4


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported dtype: {dtype_str}")


def build_rope_module(
    head_dim: int = 64,
    rotary_dim: int = 64,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
    is_neox: bool = True,
    dtype_str: str = "bf16",
):
    """Build RoPE kernel for Q [M, num_q_heads, head_dim] and K [M, num_kv_heads, head_dim].

    Args:
        head_dim: dimension per attention head (64 for GPT-OSS 120B)
        rotary_dim: number of dimensions to rotate (== head_dim for full rotation)
        num_q_heads: query heads per rank (8 at TP=8 for GPT-OSS 120B)
        num_kv_heads: key/value heads per rank (1 at TP=8 for GPT-OSS 120B)
        is_neox: True for NeoX-style (split at midpoint), False for GPT-J interleaved
        dtype_str: element dtype ("bf16", "f16", "f32")

    Returns:
        launch_rope: @flyc.jit launcher function
    """
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation (rotary_dim != head_dim) not yet supported")

    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    half_dim = rotary_dim // 2  # 32 for GPT-OSS
    total_heads = num_q_heads + num_kv_heads  # 9
    elem_bytes = 4 if dtype_str == "f32" else 2
    vec_dwords = (VEC_WIDTH * elem_bytes) // 4  # 4 for bf16
    vecs_per_half = half_dim // VEC_WIDTH  # 4

    # Thread mapping: each thread handles one vec8 chunk of one head
    # threads_needed = total_heads * vecs_per_half = 9 * 4 = 36
    BLOCK_THREADS = WARP_SIZE  # 64 threads = one full wavefront

    @flyc.kernel
    def rope_kernel(
        Q: fx.Tensor,            # [M, num_q_heads, head_dim]
        K: fx.Tensor,            # [M, num_kv_heads, head_dim]
        CosCache: fx.Tensor,     # [max_pos, half_dim]
        SinCache: fx.Tensor,     # [max_pos, half_dim]
        Positions: fx.Tensor,    # [M] int32
        Q_out: fx.Tensor,        # [M, num_q_heads, head_dim]
        K_out: fx.Tensor,        # [M, num_kv_heads, head_dim]
    ):
        from flydsl.expr.arith import ArithValue

        bid = fx.block_idx.x   # token index
        tid = fx.thread_idx.x  # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32
        vec_type_c = T.vec(VEC_WIDTH, compute_type)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)

        # Thread-to-work mapping
        head_idx_i32 = tid // vecs_per_half   # which head (0..8 active, 9..15 inactive)
        vec_idx_i32 = tid % vecs_per_half     # which vec8 chunk within half (0..3)

        # Guard: only threads_needed threads are active
        is_active = head_idx_i32 < total_heads

        if is_active:
            # --- 1. Load position for this token ---
            pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
            pos_val = buffer_ops.buffer_load(pos_rsrc, bid, vec_width=1, dtype=T.i32)

            # --- 2. Load cos/sin for this position ---
            cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
            sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)

            # cos/sin row offset = pos * half_dim * elem_bytes
            cos_row_soff = ArithValue(pos_val) * (half_dim * elem_bytes)
            # cos/sin col offset = vec_idx * VEC_WIDTH * elem_bytes
            cos_col_bytes = ArithValue(vec_idx_i32) * (VEC_WIDTH * elem_bytes)
            cos_col_dw = cos_col_bytes >> fx.Int32(2)

            cos_raw = buffer_ops.buffer_load(
                cos_rsrc, cos_col_dw, vec_width=vec_dwords, dtype=T.i32,
                soffset_bytes=cos_row_soff,
            )
            sin_raw = buffer_ops.buffer_load(
                sin_rsrc, cos_col_dw, vec_width=vec_dwords, dtype=T.i32,
                soffset_bytes=cos_row_soff,
            )
            cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
            sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)
            cos_f32 = cos_e.extf(vec_type_c) if dtype_str != "f32" else cos_e
            sin_f32 = sin_e.extf(vec_type_c) if dtype_str != "f32" else sin_e

            # --- 3. Determine Q or K head ---
            is_q = head_idx_i32 < num_q_heads

            if is_q:
                # Q head
                q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
                qo_rsrc = buffer_ops.create_buffer_resource(Q_out, max_size=True)
                local_head = head_idx_i32

                # Q row offset = bid * num_q_heads * head_dim * elem_bytes
                q_row_soff = ArithValue(bid) * (num_q_heads * head_dim * elem_bytes)
                # Head offset = local_head * head_dim * elem_bytes
                head_off = ArithValue(local_head) * (head_dim * elem_bytes)
                # Vec offset within half
                vec_off = ArithValue(vec_idx_i32) * (VEC_WIDTH * elem_bytes)

                # Load first_half: x[..., 0:half_dim]
                first_byte_off = ArithValue(head_off) + vec_off
                first_dw = first_byte_off >> fx.Int32(2)
                first_raw = buffer_ops.buffer_load(
                    q_rsrc, first_dw, vec_width=vec_dwords, dtype=T.i32,
                    soffset_bytes=q_row_soff,
                )
                first_e = vector.bitcast(vec_type_e, first_raw) if vec_dwords != VEC_WIDTH else first_raw.bitcast(vec_type_e)
                first_f32 = first_e.extf(vec_type_c) if dtype_str != "f32" else first_e

                # Load second_half: x[..., half_dim:head_dim]
                second_byte_off = ArithValue(head_off) + vec_off + (half_dim * elem_bytes)
                second_dw = second_byte_off >> fx.Int32(2)
                second_raw = buffer_ops.buffer_load(
                    q_rsrc, second_dw, vec_width=vec_dwords, dtype=T.i32,
                    soffset_bytes=q_row_soff,
                )
                second_e = vector.bitcast(vec_type_e, second_raw) if vec_dwords != VEC_WIDTH else second_raw.bitcast(vec_type_e)
                second_f32 = second_e.extf(vec_type_c) if dtype_str != "f32" else second_e

                # --- 4. Rotate ---
                # out_first = first_half * cos - second_half * sin
                out_first_f32 = ArithValue(first_f32) * ArithValue(cos_f32) - ArithValue(second_f32) * ArithValue(sin_f32)
                # out_second = second_half * cos + first_half * sin
                out_second_f32 = ArithValue(second_f32) * ArithValue(cos_f32) + ArithValue(first_f32) * ArithValue(sin_f32)

                # --- 5. Truncate and store ---
                i32_vec_ty = T.vec(vec_dwords, T.i32)

                if dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32:
                    out_first_e = out_first_f32.truncf(vec_type_e)
                    out_second_e = out_second_f32.truncf(vec_type_e)
                elif dtype_str == "bf16":
                    # Manual round-to-nearest-even for bf16 on pre-gfx950
                    out_first_e = _f32_to_bf16_manual(out_first_f32, vec_type_e)
                    out_second_e = _f32_to_bf16_manual(out_second_f32, vec_type_e)
                elif dtype_str == "f32":
                    out_first_e = out_first_f32
                    out_second_e = out_second_f32
                else:
                    out_first_e = out_first_f32.truncf(vec_type_e)
                    out_second_e = out_second_f32.truncf(vec_type_e)

                out_first_i32 = vector.bitcast(i32_vec_ty, out_first_e) if vec_dwords != VEC_WIDTH else out_first_e.bitcast(i32_vec_ty)
                out_second_i32 = vector.bitcast(i32_vec_ty, out_second_e) if vec_dwords != VEC_WIDTH else out_second_e.bitcast(i32_vec_ty)

                first_store_dw = first_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(out_first_i32, qo_rsrc, first_store_dw, soffset_bytes=q_row_soff)
                second_store_dw = second_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(out_second_i32, qo_rsrc, second_store_dw, soffset_bytes=q_row_soff)

            else:
                # K head
                k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
                ko_rsrc = buffer_ops.create_buffer_resource(K_out, max_size=True)
                local_head = head_idx_i32 - num_q_heads

                k_row_soff = ArithValue(bid) * (num_kv_heads * head_dim * elem_bytes)
                head_off = ArithValue(local_head) * (head_dim * elem_bytes)
                vec_off = ArithValue(vec_idx_i32) * (VEC_WIDTH * elem_bytes)

                first_byte_off = ArithValue(head_off) + vec_off
                first_dw = first_byte_off >> fx.Int32(2)
                first_raw = buffer_ops.buffer_load(
                    k_rsrc, first_dw, vec_width=vec_dwords, dtype=T.i32,
                    soffset_bytes=k_row_soff,
                )
                first_e = vector.bitcast(vec_type_e, first_raw) if vec_dwords != VEC_WIDTH else first_raw.bitcast(vec_type_e)
                first_f32 = first_e.extf(vec_type_c) if dtype_str != "f32" else first_e

                second_byte_off = ArithValue(head_off) + vec_off + (half_dim * elem_bytes)
                second_dw = second_byte_off >> fx.Int32(2)
                second_raw = buffer_ops.buffer_load(
                    k_rsrc, second_dw, vec_width=vec_dwords, dtype=T.i32,
                    soffset_bytes=k_row_soff,
                )
                second_e = vector.bitcast(vec_type_e, second_raw) if vec_dwords != VEC_WIDTH else second_raw.bitcast(vec_type_e)
                second_f32 = second_e.extf(vec_type_c) if dtype_str != "f32" else second_e

                out_first_f32 = ArithValue(first_f32) * ArithValue(cos_f32) - ArithValue(second_f32) * ArithValue(sin_f32)
                out_second_f32 = ArithValue(second_f32) * ArithValue(cos_f32) + ArithValue(first_f32) * ArithValue(sin_f32)

                i32_vec_ty = T.vec(vec_dwords, T.i32)

                if dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32:
                    out_first_e = out_first_f32.truncf(vec_type_e)
                    out_second_e = out_second_f32.truncf(vec_type_e)
                elif dtype_str == "bf16":
                    out_first_e = _f32_to_bf16_manual(out_first_f32, vec_type_e)
                    out_second_e = _f32_to_bf16_manual(out_second_f32, vec_type_e)
                elif dtype_str == "f32":
                    out_first_e = out_first_f32
                    out_second_e = out_second_f32
                else:
                    out_first_e = out_first_f32.truncf(vec_type_e)
                    out_second_e = out_second_f32.truncf(vec_type_e)

                out_first_i32 = vector.bitcast(i32_vec_ty, out_first_e) if vec_dwords != VEC_WIDTH else out_first_e.bitcast(i32_vec_ty)
                out_second_i32 = vector.bitcast(i32_vec_ty, out_second_e) if vec_dwords != VEC_WIDTH else out_second_e.bitcast(i32_vec_ty)

                first_store_dw = first_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(out_first_i32, ko_rsrc, first_store_dw, soffset_bytes=k_row_soff)
                second_store_dw = second_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(out_second_i32, ko_rsrc, second_store_dw, soffset_bytes=k_row_soff)

    def _f32_to_bf16_manual(val, vec_bf16_ty):
        """Manual f32→bf16 round-to-nearest-even (pre-gfx950 fallback)."""
        from flydsl.expr.arith import ArithValue

        vec_i32_ty = T.vec(VEC_WIDTH, T.i32)
        vec4_i32_ty = T.vec(VEC_WIDTH // 2, T.i32)
        c16_i32 = arith.constant(16, type=T.i32)
        c16_v = vector.broadcast(vec_i32_ty, c16_i32)
        u = val.bitcast(vec_i32_ty)
        upper = u.shrui(c16_v)
        c1_v = vector.broadcast(vec_i32_ty, arith.constant(1, type=T.i32))
        lsb = upper & c1_v
        c7fff_v = vector.broadcast(vec_i32_ty, arith.constant(0x7FFF, type=T.i32))
        bias = ArithValue(c7fff_v) + ArithValue(lsb)
        u_round = ArithValue(u) + bias
        bf16_bits = u_round.shrui(c16_v)
        even = vector.shuffle(bf16_bits, bf16_bits, [0, 2, 4, 6])
        odd = vector.shuffle(bf16_bits, bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << vector.broadcast(vec4_i32_ty, c16_i32)
        packed = even | odd_sh
        return vector.bitcast(vec_bf16_ty, packed)

    @flyc.jit
    def launch_rope(
        Q: fx.Tensor,
        K: fx.Tensor,
        CosCache: fx.Tensor,
        SinCache: fx.Tensor,
        Positions: fx.Tensor,
        Q_out: fx.Tensor,
        K_out: fx.Tensor,
        num_tokens: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        idx_m = arith.index_cast(T.index, num_tokens)
        launcher = rope_kernel(Q, K, CosCache, SinCache, Positions, Q_out, K_out)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rope
