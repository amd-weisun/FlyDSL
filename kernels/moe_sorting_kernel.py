# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MoE Sorting kernel — replaces CK MoeSortingKernel.

Implements token-expert sorting needed before MoE GEMM:
  - Groups tokens by expert assignment, pads to unit_size multiples
  - Packs token_id + topk_id into sorted_token_ids (bits 24-31 = topk_id)
  - Outputs: sorted_token_ids, sorted_weights, sorted_expert_ids,
             total_tokens_post_pad[2]
  - Fused: zeros moe_buf

Matches CK:
    MoeSortingKernel<MoeSortingProblemEx<int, float, 1, true, false, false, true, 0>>
    SubTokenTile=1, SubTokenOneShot=true
    LocalExpertMasking=false, LocalToken=false
    SkipExpertsWithZeroTokens=true, ExpertTile=0

Algorithm (CK EX kernel translated to FlyDSL):
  Block 0  → sorting (1 block, all phases below)
  Blocks 1+ → moe_buf zero-fill (fused)

  LDS layout  (all int32, smem_cols = num_experts + 1):
    smem_cumsum  [0 .. smem_cols)             – per-expert counts, then prefix sums
    smem_cumdup  [smem_cols .. 2*smem_cols)   – duplicate of cumsum start for padding fill
    smem_tokens  [2*smem_cols .. smem_rows*smem_cols) – [sub_tokens × smem_cols] histogram

  Phase 0: clear smem_tokens, zero smem_cumsum
  Phase 1: for each batch of sub_tokens tokens, mark smem_tokens[sub_tok, expert]
  Phase 2: count per expert using 8-lane groups + wave_cumsum prefix in scatter
  Phase 3: wave_cumsum over experts (wave 0), compute padded prefix sum → smem_cumsum
  Phase 4: fill sorted_expert_ids
  Phase 5: scatter tokens via smem_tokens + 8-lane intra-group prefix scan
  Phase 6: fill padding entries with invalid token ID
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, Int32
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl._mlir import ir

BLOCK_SIZE = 256
WARP_SIZE = 64
LANE_GROUP_SZ = 8  # threads per expert group in scatter
LANE_GROUPS = BLOCK_SIZE // LANE_GROUP_SZ  # 32 groups


def _smem_dims(max_tokens: int, num_experts: int):
    """Compute (smem_rows, smem_cols) matching CK's moe_sorting_get_smem_row_col."""
    smem_cols = num_experts + 1
    total_ints = 65536 // 4  # 64 KB LDS / 4 B per int32
    occupancy = 2
    sub_unroll = 8
    cumsum_bufs = 2

    r = total_ints // occupancy // smem_cols
    if r < cumsum_bufs + sub_unroll:
        return cumsum_bufs, smem_cols

    r_sub = (r - cumsum_bufs) // sub_unroll * sub_unroll
    r_token_min = (max_tokens + sub_unroll - 1) // sub_unroll * sub_unroll
    r_sub = min(r_sub, r_token_min)
    return r_sub + cumsum_bufs, smem_cols


def build_moe_sorting_module(
    num_experts: int,
    topk: int,
    max_tokens: int = 4096,
):
    """Build the MoeSorting JIT module.

    Args:
        num_experts: total experts (e.g. 128 for GPT-OSS 120B)
        topk:        top-k expert selection (e.g. 4 for GPT-OSS 120B)
        max_tokens:  upper bound on token count used to size LDS (default 4096)

    Returns:
        launch_moe_sorting: compiled JIT launch function
    """
    arch = get_rocm_arch()
    smem_cols = num_experts + 1
    smem_rows, _ = _smem_dims(max_tokens, num_experts)
    sub_tokens_max = smem_rows - 2  # max sub_tokens per batch

    # LDS regions (int32 elements):
    #   [CUMSUM_BASE .. CUMSUM_BASE + smem_cols)  = smem_cumsum
    #   [CUMDUP_BASE .. CUMDUP_BASE + smem_cols)  = smem_cumdup
    #   [TOKENS_BASE .. smem_rows * smem_cols)    = smem_tokens[sub_tokens_max, smem_cols]
    CUMSUM_BASE = 0
    CUMDUP_BASE = smem_cols
    TOKENS_BASE = 2 * smem_cols
    total_smem_ints = smem_rows * smem_cols

    allocator = SmemAllocator(None, arch=arch)
    _smem_arr = allocator.allocate_array(T.i32, total_smem_ints)

    @flyc.kernel
    def moe_sorting_kernel(
        topk_ids: fx.Tensor,               # [tokens * topk]  int32  (row-major: [tokens][topk])
        topk_weights: fx.Tensor,           # [tokens * topk]  float32
        sorted_token_ids: fx.Tensor,       # [max_tokens_padded]  int32
        sorted_weights: fx.Tensor,         # [max_tokens_padded]  float32
        sorted_expert_ids: fx.Tensor,      # [max_m_blocks]  int32
        total_tokens_post_pad: fx.Tensor,  # [2]  int32
        moe_buf: fx.Tensor,                # flat int32 buffer to zero (may be None-sentinel)
        tokens: Int32,                     # actual token count (runtime)
        unit_size: Int32,                  # M_a block size (e.g. 32)
        sub_tokens: Int32,                 # actual sub_tokens for this token count
        moe_buf_elems: Int32,              # total int32 elements in moe_buf to zero
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        # ── Create buffer resources (top-level, before any scf.if) ──────────
        topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
        topk_wts_rsrc = buffer_ops.create_buffer_resource(topk_weights, max_size=True)
        sorted_ids_rsrc = buffer_ops.create_buffer_resource(sorted_token_ids, max_size=True)
        sorted_wts_rsrc = buffer_ops.create_buffer_resource(sorted_weights, max_size=True)
        expert_ids_rsrc = buffer_ops.create_buffer_resource(sorted_expert_ids, max_size=True)
        total_rsrc = buffer_ops.create_buffer_resource(total_tokens_post_pad, max_size=True)
        moe_rsrc = buffer_ops.create_buffer_resource(moe_buf, max_size=True)

        # ── LDS views ───────────────────────────────────────────────────────
        base_ptr = allocator.get_base()
        smem_cumsum = SmemPtr(base_ptr, CUMSUM_BASE * 4, T.i32, shape=(smem_cols,))
        smem_cumdup = SmemPtr(base_ptr, CUMDUP_BASE * 4, T.i32, shape=(smem_cols,))
        smem_tokens_ptr = SmemPtr(
            base_ptr, TOKENS_BASE * 4, T.i32, shape=(sub_tokens_max * smem_cols,)
        )
        cs = smem_cumsum.get()   # memref for cumsum
        cd = smem_cumdup.get()   # memref for cumdup
        st = smem_tokens_ptr.get()  # memref for tokens histogram (flat: [sub_tok * smem_cols])

        zero_i32 = arith.constant(0, type=T.i32)
        zero_f32 = arith.constant(0.0, type=T.f32)

        # ── Blocks 1+: zero moe_buf ─────────────────────────────────────────
        is_moe_block = arith.cmpi(arith.CmpIPredicate.ne, bid, fx.Int32(0))
        if is_moe_block:
            # Each non-sorting block zeros a strided slice of moe_buf (int32 elements)
            block_m1 = arith.subi(bid, fx.Int32(1))
            block_offset = arith.addi(arith.muli(block_m1, fx.Int32(BLOCK_SIZE)), tid)
            n_blocks_minus1 = arith.subi(fx.grid_dim.x, fx.Int32(1))
            stride = arith.muli(n_blocks_minus1, fx.Int32(BLOCK_SIZE))
            for i in range(block_offset, moe_buf_elems, stride):
                buffer_ops.buffer_store(zero_i32, moe_rsrc, i)

        else:
            # ── Block 0: sorting ────────────────────────────────────────────

            # ── Phase 0: Clear smem_cumsum and smem_tokens ──────────────────
            for i in range(tid, fx.Int32(smem_cols), fx.Int32(BLOCK_SIZE)):
                idx = arith.index_cast(T.index, i)
                smem_cumsum.store(zero_i32, [idx])
                smem_cumdup.store(zero_i32, [idx])

            sub_tok_x_cols = arith.muli(sub_tokens, fx.Int32(smem_cols))
            for i in range(tid, sub_tok_x_cols, fx.Int32(BLOCK_SIZE)):
                idx = arith.index_cast(T.index, i)
                smem_tokens_ptr.store(zero_i32, [idx])

            gpu.barrier()

            # ── Phase 1: Count tokens per expert (sub_tokens batching) ──────
            # For each batch [i_token, i_token+sub_tokens), mark
            #   smem_tokens[curr_token_id * smem_cols + eid] = curr_topk_id + 1
            # SubTokenOneShot=true → plain assignment (no atomic needed:
            # a given (sub_tok_id, expert) is written by at most one topk slot
            # since each token maps each expert at most once in typical MoE).
            for i_token in range(fx.Int32(0), tokens, sub_tokens):
                loop_end_1 = arith.muli(sub_tokens, fx.Int32(topk))
                for i in range(tid, loop_end_1, fx.Int32(BLOCK_SIZE)):
                    curr_token_id = arith.divsi(i, fx.Int32(topk))
                    curr_topk_id = arith.remsi(i, fx.Int32(topk))
                    i_t = arith.addi(i_token, curr_token_id)
                    in_range = arith.cmpi(arith.CmpIPredicate.slt, i_t, tokens)
                    # safe load index (clamp to 0 when out of range)
                    safe_i_t = arith.select(in_range, i_t, fx.Int32(0))
                    flat_idx = arith.addi(arith.muli(safe_i_t, fx.Int32(topk)), curr_topk_id)
                    eid = buffer_ops.buffer_load(topk_ids_rsrc, flat_idx, vec_width=1, dtype=T.i32)
                    st_flat = arith.addi(arith.muli(curr_token_id, fx.Int32(smem_cols)), eid)
                    val_store = arith.addi(curr_topk_id, fx.Int32(1))
                    if in_range:
                        st_idx = arith.index_cast(T.index, st_flat)
                        smem_tokens_ptr.store(val_store, [st_idx])

                gpu.barrier()

                # ── Phase 2 (inline per batch): accumulate expert counts ────
                # 8-lane groups: group_id = tid // 8, group_os = tid % 8
                # Each group handles experts [group_id, group_id + LANE_GROUPS, ...]
                lane_group_id = arith.divsi(tid, fx.Int32(LANE_GROUP_SZ))
                lane_group_os = arith.remsi(tid, fx.Int32(LANE_GROUP_SZ))
                for i_e in range(lane_group_id, fx.Int32(num_experts), fx.Int32(LANE_GROUPS)):
                    # Sum non-zero entries for this expert in smem_tokens[:,i_e]
                    # lane_group_os strides within the sub_tokens dimension
                    cnt = fx.Int32(0)
                    for j in range(lane_group_os, sub_tokens, fx.Int32(LANE_GROUP_SZ)):
                        st_flat_j = arith.addi(arith.muli(j, fx.Int32(smem_cols)), i_e)
                        st_flat_j_idx = arith.index_cast(T.index, st_flat_j)
                        val = smem_tokens_ptr.load([st_flat_j_idx])
                        is_nz = arith.cmpi(arith.CmpIPredicate.ne, val, zero_i32)
                        one_if_nz = arith.select(is_nz, fx.Int32(1), fx.Int32(0))
                        cnt = arith.addi(cnt, one_if_nz)

                    # Reduce cnt across 8 lanes of this group via XOR shuffle
                    width_i32 = fx.Int32(WARP_SIZE)
                    # Shuffle within 8-lane group: offsets 4, 2, 1
                    for sh in range_constexpr(3):
                        off = fx.Int32(1 << (2 - sh))  # 4, 2, 1
                        peer = cnt.shuffle_xor(off, width_i32)
                        cnt = arith.addi(cnt, peer)

                    # Lane 0 of group accumulates into smem_cumsum[i_e + 1]
                    if arith.cmpi(arith.CmpIPredicate.eq, lane_group_os, fx.Int32(0)):
                        cs_idx = arith.index_cast(T.index, arith.addi(i_e, fx.Int32(1)))
                        existing = smem_cumsum.load([cs_idx])
                        smem_cumsum.store(arith.addi(existing, cnt), [cs_idx])

                gpu.barrier()

                # Clear smem_tokens for next batch (except last — Phase 5 reuses it)
                for i in range(tid, sub_tok_x_cols, fx.Int32(BLOCK_SIZE)):
                    idx = arith.index_cast(T.index, i)
                    smem_tokens_ptr.store(zero_i32, [idx])

                gpu.barrier()

            # ── Phase 3: Padded prefix sum over experts (wave 0 only) ───────
            # ExpertTile==0 + SkipExpertsWithZeroTokens=true:
            #   padded[e] = ceil(cnt[e]/unit_size)*unit_size  (0 if cnt==0)
            # Uses wave_cumsum (DPP row_shr) across 64 lanes, one expert per lane.
            # We implement via ds_bpermute for portability.
            #
            # Each lane handles one expert in the current stripe [i_e_, i_e_+64).
            wid = arith.divsi(tid, fx.Int32(WARP_SIZE))
            lid = arith.remsi(tid, fx.Int32(WARP_SIZE))
            if arith.cmpi(arith.CmpIPredicate.eq, wid, fx.Int32(0)):
                # Process experts in stripes of WARP_SIZE
                for i_e_ in range_constexpr(0, num_experts, WARP_SIZE):
                    e_lane = arith.addi(fx.Int32(i_e_), lid)
                    in_expert_range = arith.cmpi(
                        arith.CmpIPredicate.slt, e_lane, fx.Int32(num_experts)
                    )
                    safe_e_lane = arith.select(in_expert_range, e_lane, fx.Int32(0))

                    # Load pre_cumsum from lane 0's previous stripe result
                    pre_cs = smem_cumsum.load(
                        [arith.index_cast(T.index, fx.Int32(i_e_))]
                    ) if lid == fx.Int32(0) else zero_i32
                    # NOTE: the above Python-level `if` is compile-time — lid is runtime.
                    # We use arith.select instead:
                    pre_cs_lane0 = smem_cumsum.load(
                        [arith.index_cast(T.index, fx.Int32(i_e_))]
                    )
                    pre_cs = arith.select(
                        arith.cmpi(arith.CmpIPredicate.eq, lid, fx.Int32(0)),
                        pre_cs_lane0,
                        zero_i32,
                    )

                    # Load this expert's raw count
                    cnt_e = smem_cumsum.load(
                        [arith.index_cast(T.index, arith.addi(safe_e_lane, fx.Int32(1)))]
                    )
                    cnt_e = arith.select(in_expert_range, cnt_e, zero_i32)

                    # SkipExpertsWithZeroTokens: padded = 0 when cnt==0
                    blocks_per_expert = arith.divsi(
                        arith.addi(cnt_e, arith.subi(unit_size, fx.Int32(1))), unit_size
                    )
                    padded = arith.muli(blocks_per_expert, unit_size)
                    is_zero = arith.cmpi(arith.CmpIPredicate.eq, cnt_e, zero_i32)
                    padded_final = arith.select(is_zero, zero_i32, padded)

                    # In-wave exclusive prefix scan (wave_cumsum):
                    # local_cumsum = padded_final + pre_cs, then scan
                    local_cs = arith.addi(padded_final, pre_cs)
                    width_i32 = fx.Int32(WARP_SIZE)
                    for sh in range_constexpr(6):  # log2(64)=6 stages
                        off = fx.Int32(1 << sh)
                        peer = local_cs.shuffle_up(off, width_i32)
                        # Only lanes >= off receive the peer value
                        lane_ge = arith.cmpi(arith.CmpIPredicate.sge, lid, off)
                        local_cs = arith.select(lane_ge, arith.addi(local_cs, peer), local_cs)

                    # Store inclusive cumsum to smem_cumsum[i_e_+lid+1]
                    # (exclusive for expert i_e_+lid is the value at smem_cumsum[i_e_+lid])
                    if in_expert_range:
                        smem_cumsum.store(
                            local_cs,
                            [arith.index_cast(T.index, arith.addi(e_lane, fx.Int32(1)))],
                        )

                    # Last lane in the stripe writes total so far (for next stripe's pre_cs)
                    last_lid = arith.cmpi(
                        arith.CmpIPredicate.eq,
                        arith.addi(fx.Int32(i_e_), lid),
                        fx.Int32(num_experts - 1),
                    )
                    if last_lid:
                        # Write total_tokens_post_pad[0], [1]
                        buffer_ops.buffer_store(local_cs, total_rsrc, fx.Int32(0))
                        buffer_ops.buffer_store(tokens, total_rsrc, fx.Int32(1))

                    # s_waitcnt lgkmcnt(0) equivalent — barrier covers this below

            gpu.barrier()

            # ── Phase 4: Fill sorted_expert_ids & duplicate cumsum ──────────
            # smem_cumsum[e]   = exclusive prefix (start offset for expert e)
            # smem_cumsum[e+1] = exclusive prefix (end offset for expert e)
            # sorted_expert_ids[block] = e  for block in [start/unit_size, end/unit_size)
            #
            # Also: smem_cumdup[e] = smem_cumsum[e] (write position tracker)
            for i_e in range(tid, fx.Int32(num_experts), fx.Int32(BLOCK_SIZE)):
                e_idx = arith.index_cast(T.index, i_e)
                ep1_idx = arith.index_cast(T.index, arith.addi(i_e, fx.Int32(1)))
                e_start = smem_cumsum.load([e_idx])
                e_end = smem_cumsum.load([ep1_idx])

                # SkipExpertsWithZeroTokens: skip if e_start == e_end
                has_tokens = arith.cmpi(arith.CmpIPredicate.ne, e_start, e_end)

                # Duplicate start for scatter phase
                smem_cumdup.store(e_start, [e_idx])

                if has_tokens:
                    block_start = arith.divsi(e_start, unit_size)
                    block_end = arith.divsi(e_end, unit_size)
                    for b in range(block_start, block_end, fx.Int32(1)):
                        buffer_ops.buffer_store(i_e, expert_ids_rsrc, b)

            gpu.barrier()

            # smem_cumdup[num_experts] = smem_cumsum[num_experts]  (used in padding fill)
            if arith.cmpi(arith.CmpIPredicate.eq, tid, fx.Int32(0)):
                ne_idx = arith.index_cast(T.index, fx.Int32(num_experts))
                total_val = smem_cumsum.load([ne_idx])
                smem_cumdup.store(total_val, [ne_idx])

            gpu.barrier()

            # ── Phase 5: Scatter tokens ─────────────────────────────────────
            # Re-run the same sub_tokens batching from Phase 1.
            # For each batch, 8-lane groups scatter tokens using intra-group
            # prefix scan (wave_cumsum<8>) + ds_bpermute for the group total.
            #
            # smem_cumdup[e] acts as the running write position for expert e.
            for i_token in range(fx.Int32(0), tokens, sub_tokens):
                # 5a: Re-populate smem_tokens (same as Phase 1)
                loop_end_5 = arith.muli(sub_tokens, fx.Int32(topk))
                for i in range(tid, loop_end_5, fx.Int32(BLOCK_SIZE)):
                    curr_token_id = arith.divsi(i, fx.Int32(topk))
                    curr_topk_id = arith.remsi(i, fx.Int32(topk))
                    i_t = arith.addi(i_token, curr_token_id)
                    in_range = arith.cmpi(arith.CmpIPredicate.slt, i_t, tokens)
                    safe_i_t = arith.select(in_range, i_t, fx.Int32(0))
                    flat_idx = arith.addi(arith.muli(safe_i_t, fx.Int32(topk)), curr_topk_id)
                    eid = buffer_ops.buffer_load(topk_ids_rsrc, flat_idx, vec_width=1, dtype=T.i32)
                    st_flat = arith.addi(arith.muli(curr_token_id, fx.Int32(smem_cols)), eid)
                    val_store = arith.addi(curr_topk_id, fx.Int32(1))
                    if in_range:
                        st_idx = arith.index_cast(T.index, st_flat)
                        smem_tokens_ptr.store(val_store, [st_idx])

                gpu.barrier()

                # 5b: Scatter — each 8-lane group handles experts in strides
                # lane_group_id = tid // 8, lane_group_os = tid % 8
                lane_group_id_5 = arith.divsi(tid, fx.Int32(LANE_GROUP_SZ))
                lane_group_os_5 = arith.remsi(tid, fx.Int32(LANE_GROUP_SZ))
                width_i32 = fx.Int32(WARP_SIZE)

                for eid in range(lane_group_id_5, fx.Int32(num_experts), fx.Int32(LANE_GROUPS)):
                    cd_eid_idx = arith.index_cast(T.index, eid)
                    position = smem_cumdup.load([cd_eid_idx])

                    # Stride over sub_tokens rows in steps of LANE_GROUP_SZ
                    for i_sub in range(lane_group_os_5, sub_tokens, fx.Int32(LANE_GROUP_SZ)):
                        st_flat_s = arith.addi(arith.muli(i_sub, fx.Int32(smem_cols)), eid)
                        st_flat_s_idx = arith.index_cast(T.index, st_flat_s)
                        x = smem_tokens_ptr.load([st_flat_s_idx])

                        # x != 0 means this sub-token is assigned to this expert
                        has_tok = arith.cmpi(arith.CmpIPredicate.ne, x, zero_i32)
                        local_cnt_cache = arith.select(has_tok, fx.Int32(1), fx.Int32(0))

                        # In-group exclusive prefix scan over 8 lanes (wave_cumsum<8>)
                        local_cnt = local_cnt_cache
                        for sh in range_constexpr(3):  # log2(8)=3 stages
                            off = fx.Int32(1 << sh)  # 1, 2, 4
                            peer = local_cnt.shuffle_up(off, width_i32)
                            lane_ge = arith.cmpi(arith.CmpIPredicate.sge, lane_group_os_5, off)
                            local_cnt = arith.select(
                                lane_ge, arith.addi(local_cnt, peer), local_cnt
                            )
                        # local_cnt is now INCLUSIVE prefix within the 8-lane group

                        if has_tok:
                            # Write position = position + local_cnt - 1
                            write_pos = arith.addi(position, arith.subi(local_cnt, fx.Int32(1)))
                            i_global = arith.addi(i_token, i_sub)

                            # Encode: (token_id & 0x00FFFFFF) | (topk_id << 24)
                            topk_id_val = arith.subi(x, fx.Int32(1))  # recover topk_id from x=topk+1
                            encoded_id = arith.ori(
                                arith.andi(i_global, fx.Int32(0x00FFFFFF)),
                                arith.shli(arith.andi(topk_id_val, fx.Int32(0xFF)), fx.Int32(24)),
                            )

                            # Load weight: topk_ids layout is [tokens, topk] → [i_t * topk + topk_id]
                            wt_flat = arith.addi(
                                arith.muli(i_global, fx.Int32(topk)), topk_id_val
                            )
                            weight = buffer_ops.buffer_load(
                                topk_wts_rsrc, wt_flat, vec_width=1, dtype=T.f32
                            )

                            buffer_ops.buffer_store(encoded_id, sorted_ids_rsrc, write_pos)
                            buffer_ops.buffer_store(weight, sorted_wts_rsrc, write_pos)

                        # Advance position by the group total (last lane's inclusive sum)
                        # Use ds_bpermute to broadcast lane 7's value to all lanes in group
                        last_lane_in_group = arith.addi(
                            arith.muli(
                                arith.addi(lane_group_id_5, fx.Int32(1)), fx.Int32(LANE_GROUP_SZ)
                            ),
                            fx.Int32(-1),
                        )
                        bperm_addr = arith.shli(last_lane_in_group, fx.Int32(2))
                        remote_cnt = rocdl.ds_bpermute(bperm_addr, local_cnt)
                        position = arith.addi(position, remote_cnt)

                    # Write back updated position for this expert
                    smem_cumdup.store(position, [cd_eid_idx])

                gpu.barrier()

                # Clear smem_tokens for next batch
                for i in range(tid, sub_tok_x_cols, fx.Int32(BLOCK_SIZE)):
                    idx = arith.index_cast(T.index, i)
                    smem_tokens_ptr.store(zero_i32, [idx])

                gpu.barrier()

            # ── Phase 6: Fill padding entries with invalid token ID ──────────
            # After scatter, smem_cumdup[e] = next free position for expert e.
            # smem_cumsum[e+1] = end of expert e's padded region.
            # Fill [smem_cumdup[e], smem_cumsum[e+1]) with MOE_SORTING_MOCK_ID(tokens, topk).
            invalid_id = arith.ori(
                arith.andi(tokens, fx.Int32(0x00FFFFFF)),
                arith.shli(fx.Int32(topk), fx.Int32(24)),
            )
            for i_e in range(tid, fx.Int32(num_experts), fx.Int32(BLOCK_SIZE)):
                e_idx = arith.index_cast(T.index, i_e)
                ep1_idx = arith.index_cast(T.index, arith.addi(i_e, fx.Int32(1)))
                write_pos = smem_cumdup.load([e_idx])
                e_end = smem_cumsum.load([ep1_idx])

                # SkipExpertsWithZeroTokens: skip if write_pos == e_end (no tokens)
                has_pad = arith.cmpi(arith.CmpIPredicate.ne, write_pos, e_end)
                if has_pad:
                    for p in range(write_pos, e_end, fx.Int32(1)):
                        buffer_ops.buffer_store(invalid_id, sorted_ids_rsrc, p)
                        buffer_ops.buffer_store(zero_f32, sorted_wts_rsrc, p)

    @flyc.jit
    def launch_moe_sorting(
        topk_ids: fx.Tensor,
        topk_weights: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        sorted_weights: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        total_tokens_post_pad: fx.Tensor,
        moe_buf: fx.Tensor,
        tokens: fx.Int32,
        unit_size: fx.Int32,
        sub_tokens: fx.Int32,
        moe_buf_elems: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # Grid: block 0 = sorting, blocks 1..num_cu*OCCUPANCY = moe_buf zero-fill
        import subprocess

        try:
            result = subprocess.run(
                ["rocm-smi", "--showmemuse"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            # Default to MI300X (228 CUs) if query fails
            num_cu = 228
        except Exception:
            num_cu = 228

        OCCUPANCY = 2
        grid_x = 1 + num_cu * OCCUPANCY  # block 0: sort, rest: moe_buf

        moe_sorting_kernel(
            topk_ids,
            topk_weights,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            total_tokens_post_pad,
            moe_buf,
            tokens,
            unit_size,
            sub_tokens,
            moe_buf_elems,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_moe_sorting


def compute_sub_tokens(tokens: int, num_experts: int) -> int:
    """Host-side helper: compute sub_tokens for a given token count.

    Call this before launching to get the sub_tokens runtime parameter.
    Matches CK's moe_sorting_get_sub_token().
    """
    smem_rows, _ = _smem_dims(tokens, num_experts)
    return max(smem_rows - 2, 0)


def compute_max_tokens_padded(tokens: int, num_experts: int, topk: int, unit_size: int) -> int:
    """Compute upper bound on sorted_token_ids / sorted_weights size.

    Matches CK: topk * tokens + num_experts * (unit_size - 1)
    """
    return topk * tokens + num_experts * (unit_size - 1)


def compute_max_m_blocks(tokens: int, num_experts: int, topk: int, unit_size: int) -> int:
    """Compute upper bound on sorted_expert_ids size (number of unit_size blocks)."""
    return (compute_max_tokens_padded(tokens, num_experts, topk, unit_size) + unit_size - 1) // unit_size
