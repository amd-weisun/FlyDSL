# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.

Fuses 3 operations into two kernel launches:
  Kernel 1 (Q RoPE):     Q → rotate → Q_out
  Kernel 2 (K+V cache):  K → rotate → K_out + key_cache;  V → value_cache

Input shapes:
  Q: [T, QH, D],  K: [T, KH, D],  V: [T, KH, D]
  CosCache/SinCache: [max_pos, D//2]  (must be 2-D contiguous)
  Positions: [T] int32,  SlotMapping: [T] int32

KV cache layouts:
  flash_layout=True:
    KeyCache:   [num_blocks, block_size, KH, D]
    ValueCache: [num_blocks, block_size, KH, D]
  flash_layout=False (ATOM default):
    KeyCache:   [num_blocks, KH, D//x, block_size, x]  (x=16, x-packed)
    ValueCache: [num_blocks, KH, D, block_size]         (dim-major)


"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import arith, vector, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.expr import buffer_ops
from kernels.kernels_common import dtype_to_elem_type
from kernels.mfma_preshuffle_pipeline import crd2idx


WARP_SIZE = 64
VEC_WIDTH = 8


def _layout_to_dword_off(coord, layout, elem_bytes):
    """Coordinate → dword offset for buffer_load/buffer_store.

    crd2idx(coord, layout) → element offset (index) → byte offset (i32) → dword offset (i32).
    """
    elem_off = arith.index_cast(T.i32, crd2idx(coord, layout))
    return (ArithValue(elem_off) * elem_bytes) >> fx.Int32(2)


def _coalesced_buffer_tensor(tensor):
    """Return a buffer tensor view using the coalesced layout of ``tensor``."""
    layout = fx.coalesce(fx.get_layout(tensor))
    coalesced_view = fx.make_view(fx.get_iter(tensor), layout)
    coalesced_tensor = fx.Tensor(coalesced_view)
    return fx.rocdl.make_buffer_tensor(coalesced_tensor)


def _apply_neox_rope(qk_e, cos_rsrc, sin_rsrc, cos_dw,
                     pair_rsrc, pair_dw, is_first_half,
                     vec_dwords, vec_type_e):
    """Rotate qk_e (pre-loaded element vec) using NeoX-style RoPE.

    qk_e is pre-loaded via the layout API.  cos/sin and the paired-half
    vector are loaded via buffer_load (runtime-computed / non-affine indices).

    Returns rot_e in element dtype — caller bitcasts to i32 when needed for
    buffer_store (KV cache writes).
    """
    def _load_e(rsrc, dw):
        raw = buffer_ops.buffer_load(rsrc, dw, vec_width=vec_dwords, dtype=T.i32)
        return vector.bitcast(vec_type_e, raw)

    cos_e  = _load_e(cos_rsrc,  cos_dw)
    sin_e  = _load_e(sin_rsrc,  cos_dw)
    pair_e = _load_e(pair_rsrc, pair_dw)

    # NeoX sign: first half uses -sin, second half uses +sin
    qk_cos   = ArithValue(qk_e) * ArithValue(cos_e)
    pair_sin = ArithValue(pair_e) * ArithValue(sin_e)
    sin_term = arith.select(is_first_half, arith.negf(pair_sin), pair_sin)
    rot_e    = ArithValue(qk_cos) + ArithValue(sin_term)

    return rot_e


def build_fused_rope_cache_module(
    head_dim: int = 64,
    rotary_dim: int = -1,
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
        dtype_str: element dtype ("bf16" or "f16")

    Returns:
        launch_fn(Q, K, V, Positions, CosCache, SinCache, SlotMapping,
                  KeyCache, ValueCache, Q_out, K_out, num_tokens, stream)
    """
    if rotary_dim == -1:
        rotary_dim = head_dim
    if not is_neox:
        raise NotImplementedError("Only NeoX-style RoPE is supported")
    if rotary_dim != head_dim:
        raise NotImplementedError("Partial rotation not yet supported")
    if dtype_str not in ("bf16", "f16"):
        raise ValueError(
            f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r} "
            f"(f32 is not supported: kernel uses 2-byte elem_bytes and vec8 vectorization)"
        )
    half_dim = rotary_dim // 2
    elem_bytes = 2  # bf16 and f16 are both 2 bytes
    vec_dwords = (VEC_WIDTH * elem_bytes) // 4  # 4 dwords for vec8 of 2-byte elements
    vecs_per_half = half_dim // VEC_WIDTH   # number of VEC_WIDTH-wide vectors covering half_dim
    vecs_per_head = head_dim // VEC_WIDTH   # number of VEC_WIDTH-wide vectors covering head_dim
    x_size = 16  # x-packing factor for non-flash key_cache

    # Validate vectorization and layout assumptions to avoid silent truncation.
    if head_dim % VEC_WIDTH != 0:
        raise ValueError(
            f"head_dim must be a multiple of VEC_WIDTH ({VEC_WIDTH}), "
            f"got head_dim={head_dim}"
        )
    if rotary_dim % 2 != 0:
        raise ValueError(
            f"rotary_dim must be even so that half_dim=rotary_dim//2 is integral, "
            f"got rotary_dim={rotary_dim}"
        )
    if half_dim % VEC_WIDTH != 0:
        raise ValueError(
            f"half_dim (rotary_dim//2) must be a multiple of VEC_WIDTH "
            f"({VEC_WIDTH}), got half_dim={half_dim} (rotary_dim={rotary_dim})"
        )
    if not flash_layout and head_dim % x_size != 0:
        raise ValueError(
            f"With flash_layout=False, head_dim must be a multiple of the "
            f"key_cache packing factor x_size ({x_size}), got head_dim={head_dim}"
        )
    if vecs_per_head > WARP_SIZE:
        max_head_dim = WARP_SIZE * VEC_WIDTH
        raise ValueError(
            f"Unsupported head_dim={head_dim}: with WARP_SIZE={WARP_SIZE} and "
            f"VEC_WIDTH={VEC_WIDTH}, head_dim must satisfy "
            f"head_dim <= {max_head_dim} to avoid incomplete coverage "
            f"(got vecs_per_head={vecs_per_head} > WARP_SIZE)"
        )
    BLOCK_THREADS = WARP_SIZE

    # Layout shape/stride tuples (plain Python ints) — materialized as
    # fx.make_layout inside each kernel where an MLIR context is active.
    # These are ONLY used for the manual-path offsets (pair, cos/sin, KV cache).
    _q_shape = (None, num_q_heads, vecs_per_head)
    _q_stride = (num_q_heads * head_dim, head_dim, VEC_WIDTH)
    _kv_shape = (None, num_kv_heads, vecs_per_head)
    _kv_stride = (num_kv_heads * head_dim, head_dim, VEC_WIDTH)
    _cos_shape = (None, vecs_per_half)
    _cos_stride = (half_dim, VEC_WIDTH)

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
        bid_x, _, _ = fx.block_idx  # program id: 0..T*QH-1
        tid = fx.thread_idx.x       # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        # --- Buffer resources for scalar / runtime-indexed / non-affine loads ---
        # Must be created BEFORE make_buffer_tensor rebinds the Python name.
        q_pair_rsrc = buffer_ops.create_buffer_resource(Q, max_size=True)
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)

        # --- Convert to buffer tensors for layout API (Step 1) ---
        Q = fx.rocdl.make_buffer_tensor(Q)
        Q_out = fx.rocdl.make_buffer_tensor(Q_out)

        q_tiles = fx.logical_divide(_coalesced_buffer_tensor(Q), fx.make_layout(head_dim, 1))
        qo_tiles = fx.logical_divide(_coalesced_buffer_tensor(Q_out), fx.make_layout(head_dim, 1))

        # --- Layouts for manual-path offsets (pair, cos/sin) ---
        q_layout = fx.make_layout(_q_shape, _q_stride)
        cos_sin_layout = fx.make_layout(_cos_shape, _cos_stride)

        # --- Tiled copy setup (before guard — Step 9) ---
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_type)
        thr_layout = fx.make_layout((1, vecs_per_head), (1, 1))
        val_layout = fx.make_layout((VEC_WIDTH, 1), (1, 1))
        layout_tv = fx.raked_product(thr_layout, val_layout)
        tile_copy = fx.make_tile(VEC_WIDTH, vecs_per_head)
        tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_copy)
        thr_copy = tiled_copy.get_slice(tid)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            pid_t = bid_x // num_q_heads
            pid_hq = bid_x % num_q_heads

            # Select this program's head_dim slice and partition for vector copy.
            gQ_2d = fx.logical_divide(fx.slice(q_tiles, (None, bid_x)), fx.make_layout(VEC_WIDTH, 1))
            gQo_2d = fx.logical_divide(fx.slice(qo_tiles, (None, bid_x)), fx.make_layout(VEC_WIDTH, 1))

            # Load Q via layout API (global → register — Step 4)
            src_Q = thr_copy.partition_S(gQ_2d)
            frag_Q = fx.make_fragment_like(src_Q)
            fx.copy(copy_atom, src_Q, frag_Q)
            q_e = fx.memref_load_vec(frag_Q)

            # Load position (scalar, runtime index — Step 8 exception)
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # cos/sin address (runtime pos_val index — no layout-API equivalent)
            cos_vec_idx = tid % vecs_per_half
            cos_coord = (pos_val, fx.Int32(cos_vec_idx))
            cos_dw = _layout_to_dword_off(cos_coord, cos_sin_layout, elem_bytes)

            # Paired-half (non-affine access — no layout-API equivalent)
            is_first_half = arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half))
            pair_tid = arith.select(is_first_half, tid + vecs_per_half, tid - vecs_per_half)
            pair_coord = (pid_t, fx.Int32(pid_hq), pair_tid)
            pair_dw = _layout_to_dword_off(pair_coord, q_layout, elem_bytes)

            # RoPE rotation (q_e pre-loaded via layout API — Step 6)
            rot_e = _apply_neox_rope(
                q_e, cos_rsrc, sin_rsrc, cos_dw,
                q_pair_rsrc, pair_dw, is_first_half,
                vec_dwords, vec_type_e,
            )

            # Store rotated Q via layout API (register → global — Step 4)
            dst_Qo = thr_copy.partition_D(gQo_2d)
            frag_Qo = fx.make_fragment_like(dst_Qo)
            fx.memref_store_vec(rot_e, frag_Qo)
            fx.copy(copy_atom, frag_Qo, dst_Qo)

    # ----- Kernel 2: K RoPE + KV cache write -----
    # Grid: (T * KH, 1, 1), one program per (token, kv_head)
    # Each program: vecs_per_head threads process head_dim elements
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
        bid_x, _, _ = fx.block_idx  # program id: 0..T*KH-1
        tid = fx.thread_idx.x       # 0..63

        elem_type = dtype_to_elem_type(dtype_str)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)
        i32_vec_ty = T.vec(vec_dwords, T.i32)

        # --- Buffer resources (before make_buffer_tensor) ---
        # Pair load (non-affine)
        k_pair_rsrc = buffer_ops.create_buffer_resource(K, max_size=True)
        # Scalar / runtime-indexed
        pos_rsrc = buffer_ops.create_buffer_resource(Positions, max_size=True)
        cos_rsrc = buffer_ops.create_buffer_resource(CosCache, max_size=True)
        sin_rsrc = buffer_ops.create_buffer_resource(SinCache, max_size=True)
        slot_rsrc = buffer_ops.create_buffer_resource(SlotMapping, max_size=True)
        # KV cache (runtime slot index)
        kc_rsrc = buffer_ops.create_buffer_resource(KeyCache, max_size=True)
        vc_rsrc = buffer_ops.create_buffer_resource(ValueCache, max_size=True)

        # --- Buffer tensors for layout API (Step 1) ---
        K = fx.rocdl.make_buffer_tensor(K)
        V = fx.rocdl.make_buffer_tensor(V)
        K_out = fx.rocdl.make_buffer_tensor(K_out)

        # View [T,KH,D] as [num_tiles, head_dim] for layout-friendly slicing.
        k_tiles = fx.logical_divide(_coalesced_buffer_tensor(K), fx.make_layout(head_dim, 1))
        v_tiles = fx.logical_divide(_coalesced_buffer_tensor(V), fx.make_layout(head_dim, 1))
        ko_tiles = fx.logical_divide(_coalesced_buffer_tensor(K_out), fx.make_layout(head_dim, 1))

        # --- Layouts for manual-path offsets (pair, cos/sin, KV cache) ---
        kv_layout = fx.make_layout(_kv_shape, _kv_stride)
        cos_sin_layout = fx.make_layout(_cos_shape, _cos_stride)

        # --- Tiled copy setup (before guard — Step 9) ---
        copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_type)
        thr_layout = fx.make_layout((1, vecs_per_head), (1, 1))
        val_layout = fx.make_layout((VEC_WIDTH, 1), (1, 1))
        layout_tv = fx.raked_product(thr_layout, val_layout)
        tile_copy = fx.make_tile(VEC_WIDTH, vecs_per_head)
        tiled_copy = fx.make_tiled_copy(copy_atom, layout_tv, tile_copy)
        thr_copy = tiled_copy.get_slice(tid)

        if arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_head)):
            pid_t = bid_x // num_kv_heads
            pid_hk = bid_x % num_kv_heads

            # gK/gV/gKo tiles: bid_x selects the program's slice.
            gK_2d = fx.logical_divide(fx.slice(k_tiles, (None, bid_x)), fx.make_layout(VEC_WIDTH, 1))
            gV_2d = fx.logical_divide(fx.slice(v_tiles, (None, bid_x)), fx.make_layout(VEC_WIDTH, 1))
            gKo_2d = fx.logical_divide(fx.slice(ko_tiles, (None, bid_x)), fx.make_layout(VEC_WIDTH, 1))

            # --- Load K via layout API (global → register) ---
            src_K = thr_copy.partition_S(gK_2d)
            frag_K = fx.make_fragment_like(src_K)
            fx.copy(copy_atom, src_K, frag_K)
            k_e = fx.memref_load_vec(frag_K)

            # --- Position (scalar, runtime) ---
            pos_val = buffer_ops.buffer_load(pos_rsrc, pid_t, vec_width=1, dtype=T.i32)

            # --- cos/sin address (runtime pos_val index) ---
            cos_vec_idx = tid % vecs_per_half
            cos_coord = (pos_val, fx.Int32(cos_vec_idx))
            cos_dw = _layout_to_dword_off(cos_coord, cos_sin_layout, elem_bytes)

            # --- Paired-half (non-affine) ---
            is_first_half = arith.cmpi(arith.CmpIPredicate.ult, tid, fx.Int32(vecs_per_half))
            pair_tid = arith.select(is_first_half, tid + vecs_per_half, tid - vecs_per_half)
            pair_coord = (pid_t, fx.Int32(pid_hk), pair_tid)
            pair_dw = _layout_to_dword_off(pair_coord, kv_layout, elem_bytes)

            # --- K RoPE rotation ---
            k_rot_e = _apply_neox_rope(
                k_e, cos_rsrc, sin_rsrc, cos_dw,
                k_pair_rsrc, pair_dw, is_first_half,
                vec_dwords, vec_type_e,
            )

            # --- Store rotated K via layout API (register → global) ---
            dst_Ko = thr_copy.partition_D(gKo_2d)
            frag_Ko = fx.make_fragment_like(dst_Ko)
            fx.memref_store_vec(k_rot_e, frag_Ko)
            fx.copy(copy_atom, frag_Ko, dst_Ko)

            # --- KV Cache write (runtime slot index — stays manual, Step 8) ---
            slot_val = buffer_ops.buffer_load(slot_rsrc, pid_t, vec_width=1, dtype=T.i32)

            if arith.cmpi(arith.CmpIPredicate.sge, slot_val, fx.Int32(0)):
                pid_t_slot = ArithValue(slot_val) // block_size
                pid_b = ArithValue(slot_val) % block_size

                # Load V via layout API (global → register)
                src_V = thr_copy.partition_S(gV_2d)
                frag_V = fx.make_fragment_like(src_V)
                fx.copy(copy_atom, src_V, frag_V)
                v_e = fx.memref_load_vec(frag_V)
                v_raw = vector.bitcast(i32_vec_ty, v_e)

                # K rotated as i32 for KV cache buffer_store
                k_rot_i32 = vector.bitcast(i32_vec_ty, k_rot_e)

                if flash_layout:
                    # -- Flash KV cache: [num_blocks, BS, KH, D] via crd2idx --
                    kc_flash_layout = fx.make_layout(
                        (None, block_size, num_kv_heads, vecs_per_head),
                        (block_size * num_kv_heads * head_dim,
                         num_kv_heads * head_dim,
                         head_dim,
                         VEC_WIDTH),
                    )
                    kc_coord = (pid_t_slot, pid_b, pid_hk, tid)
                    kc_dw = _layout_to_dword_off(kc_coord, kc_flash_layout, elem_bytes)

                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw)
                    # value_cache: same layout
                    buffer_ops.buffer_store(v_raw, vc_rsrc, kc_dw)
                else:
                    # -- Non-flash key_cache: [num_blocks, KH, D//x, BS, x] via crd2idx --
                    d_start = ArithValue(tid) * VEC_WIDTH
                    dim_group = d_start // x_size
                    dim_within = d_start % x_size

                    kc_nf_layout = fx.make_layout(
                        (None, num_kv_heads, head_dim // x_size, block_size, x_size),
                        (num_kv_heads * (head_dim // x_size) * block_size * x_size,
                         (head_dim // x_size) * block_size * x_size,
                         block_size * x_size,
                         x_size,
                         1),
                    )
                    kc_coord_nf = (pid_t_slot, pid_hk, dim_group, pid_b, dim_within)
                    kc_dw_nf = _layout_to_dword_off(kc_coord_nf, kc_nf_layout, elem_bytes)

                    buffer_ops.buffer_store(k_rot_i32, kc_rsrc, kc_dw_nf)

                    # -- Non-flash value_cache: [num_blocks, KH, D, BS] scalar stores --
                    # Scalar stores required: stride-1 dim is block_size (not D), so
                    # consecutive elements along D are block_size apart in memory —
                    # a vector store would scatter across non-contiguous addresses.
                    vc_nf_layout = fx.make_layout(
                        (None, num_kv_heads, head_dim, block_size),
                        (num_kv_heads * head_dim * block_size,
                         head_dim * block_size,
                         block_size,
                         1),
                    )
                    for vi in range_constexpr(VEC_WIDTH):
                        v_scalar = vector.extract(v_e, static_position=[vi])
                        d_idx = ArithValue(tid) * VEC_WIDTH + vi
                        vc_coord = (pid_t_slot, pid_hk, d_idx, pid_b)
                        vc_elem_off = arith.index_cast(T.i32, crd2idx(vc_coord, vc_nf_layout))
                        buffer_ops.buffer_store(v_scalar, vc_rsrc, vc_elem_off)

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
