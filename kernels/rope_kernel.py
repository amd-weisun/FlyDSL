"""Standalone RoPE (Rotary Position Embedding) kernel using the @flyc.kernel API.

NeoX-style rotation:
  out[..., :D//2]  = x[..., :D//2] * cos - x[..., D//2:] * sin
  out[..., D//2:]  = x[..., D//2:] * cos + x[..., :D//2] * sin

Input shapes:
  Q: [M, QH, D],  K: [M, KH, D]
  CosCache/SinCache: [max_pos, D//2]
  Positions: [M] int32

Supports inplace (Q_out=Q, K_out=K).
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
    elem_bytes = 4 if dtype_str == "f32" else 2
    vec_dwords = (VEC_WIDTH * elem_bytes) // 4  # 4 for bf16
    vecs_per_half = half_dim // VEC_WIDTH  # 4

    # Grid: one block per token. Block: one wavefront (64 threads).
    # All 64 threads participate — each handles one vec8 chunk.
    # Thread-to-work mapping: work_id = tid → (head, vec_within_half)
    # Each work item loads first_half[vec] AND second_half[vec], rotates, stores both.
    BLOCK_THREADS = WARP_SIZE  # 64

    # Work items: each head needs vecs_per_half work items (load both halves per item)
    total_heads = num_q_heads + num_kv_heads
    q_work = num_q_heads * vecs_per_half   # e.g. 8*4=32 for TP=8
    k_work = num_kv_heads * vecs_per_half  # e.g. 1*4=4 for TP=8
    total_work = q_work + k_work           # e.g. 36 for TP=8
    import math
    iters_per_thread = math.ceil(total_work / BLOCK_THREADS)  # 1 for TP=8, 9 for TP=1

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
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        # --- Create all buffer resources at top level ---
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        q_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        qo_rsrc = buffer_ops.create_buffer_resource(Q_out, max_size=True)
        ko_rsrc = buffer_ops.create_buffer_resource(K_out, max_size=True)

        # --- Load position for this token (uniform across block) ---
        pos_val = buffer_ops.buffer_load(pos_rsrc, bid, vec_width=1, dtype=T.i32)

        def _truncf_out(val_f32):
            if dtype_str == "bf16" and USE_HW_CVT_PK_BF16_F32:
                return val_f32.truncf(vec_type_e)
            elif dtype_str == "bf16":
                return _f32_to_bf16_manual(val_f32, vec_type_e)
            elif dtype_str == "f32":
                return val_f32
            else:
                return val_f32.truncf(vec_type_e)

        def _to_store_i32(val_e):
            if vec_dwords != VEC_WIDTH:
                return vector.bitcast(i32_vec_ty, val_e)
            return val_e.bitcast(i32_vec_ty)

        def _rotate_and_store(in_rsrc, out_rsrc, row_bytes, head_idx_ct, vec_idx_rt,
                              cos_f32, sin_f32):
            """Rotate one vec8 chunk of one head and store both halves."""
            vec_off_bytes = ArithValue(vec_idx_rt) * (VEC_WIDTH * elem_bytes)
            head_off_first = head_idx_ct * head_dim * elem_bytes
            head_off_second = head_off_first + half_dim * elem_bytes

            first_dw = (row_bytes + head_off_first + vec_off_bytes) >> fx.Int32(2)
            first_raw = buffer_ops.buffer_load(in_rsrc, first_dw, vec_width=vec_dwords, dtype=T.i32)
            first_e = vector.bitcast(vec_type_e, first_raw) if vec_dwords != VEC_WIDTH else first_raw.bitcast(vec_type_e)
            first_f32 = first_e.extf(vec_type_c) if dtype_str != "f32" else first_e

            second_dw = (row_bytes + head_off_second + vec_off_bytes) >> fx.Int32(2)
            second_raw = buffer_ops.buffer_load(in_rsrc, second_dw, vec_width=vec_dwords, dtype=T.i32)
            second_e = vector.bitcast(vec_type_e, second_raw) if vec_dwords != VEC_WIDTH else second_raw.bitcast(vec_type_e)
            second_f32 = second_e.extf(vec_type_c) if dtype_str != "f32" else second_e

            out_first_f32 = ArithValue(first_f32) * ArithValue(cos_f32) - ArithValue(second_f32) * ArithValue(sin_f32)
            out_second_f32 = ArithValue(second_f32) * ArithValue(cos_f32) + ArithValue(first_f32) * ArithValue(sin_f32)

            buffer_ops.buffer_store(_to_store_i32(_truncf_out(out_first_f32)), out_rsrc, first_dw)
            buffer_ops.buffer_store(_to_store_i32(_truncf_out(out_second_f32)), out_rsrc, second_dw)

        # --- Process Q heads: all 64 threads participate ---
        q_row_bytes = ArithValue(bid) * (num_q_heads * head_dim * elem_bytes)
        q_iters = math.ceil(q_work / BLOCK_THREADS)

        for i in range_constexpr(q_iters):
            work_id = tid + i * BLOCK_THREADS  # Python int * BLOCK_THREADS
            # work_id maps to (head, vec_within_half)
            head_i = work_id // vecs_per_half   # runtime i32 division
            vec_idx = work_id % vecs_per_half   # runtime i32 modulo

            if arith.cmpi(arith.CmpIPredicate.ult, work_id, fx.Int32(q_work)):
                # Load cos/sin for this vec position
                cos_bytes = ArithValue(pos_val) * (half_dim * elem_bytes) + ArithValue(vec_idx) * (VEC_WIDTH * elem_bytes)
                cos_dw = cos_bytes >> fx.Int32(2)
                cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
                sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
                cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
                sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)
                cos_f32 = cos_e.extf(vec_type_c) if dtype_str != "f32" else cos_e
                sin_f32 = sin_e.extf(vec_type_c) if dtype_str != "f32" else sin_e

                _rotate_and_store(q_rsrc, qo_rsrc, q_row_bytes, head_i, vec_idx,
                                  cos_f32, sin_f32)

        # --- Process K heads: all 64 threads participate ---
        k_row_bytes = ArithValue(bid) * (num_kv_heads * head_dim * elem_bytes)
        k_iters = math.ceil(k_work / BLOCK_THREADS)

        for i in range_constexpr(k_iters):
            work_id = tid + i * BLOCK_THREADS
            head_i = work_id // vecs_per_half
            vec_idx = work_id % vecs_per_half

            if arith.cmpi(arith.CmpIPredicate.ult, work_id, fx.Int32(k_work)):
                cos_bytes = ArithValue(pos_val) * (half_dim * elem_bytes) + ArithValue(vec_idx) * (VEC_WIDTH * elem_bytes)
                cos_dw = cos_bytes >> fx.Int32(2)
                cos_raw = buffer_ops.buffer_load(cos_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
                sin_raw = buffer_ops.buffer_load(sin_rsrc, cos_dw, vec_width=vec_dwords, dtype=T.i32)
                cos_e = vector.bitcast(vec_type_e, cos_raw) if vec_dwords != VEC_WIDTH else cos_raw.bitcast(vec_type_e)
                sin_e = vector.bitcast(vec_type_e, sin_raw) if vec_dwords != VEC_WIDTH else sin_raw.bitcast(vec_type_e)
                cos_f32 = cos_e.extf(vec_type_c) if dtype_str != "f32" else cos_e
                sin_f32 = sin_e.extf(vec_type_c) if dtype_str != "f32" else sin_e

                _rotate_and_store(k_rsrc, ko_rsrc, k_row_bytes, head_i, vec_idx,
                                  cos_f32, sin_f32)

    def _f32_to_bf16_manual(val, vec_bf16_ty):
        """Manual f32->bf16 round-to-nearest-even (pre-gfx950 fallback)."""
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
        launcher = rope_kernel(Q, K, CosCache, SinCache, Positions, Q_out, K_out)
        launcher.launch(
            grid=(num_tokens, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rope
