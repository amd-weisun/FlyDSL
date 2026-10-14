# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Qwen3.5-MoE decoder layer (Gated DeltaNet or gated full attention) in ONE persistent launch per rank.

One launch of ``grid = 256 CTAs x 512 threads`` runs the whole layer for this rank's shard. GDN::

    input RMSNorm (Gemma 1 + w) -> qkv | z FP8 GEMV (+ causal conv1d, SiLU, conv-state shift)
      -> b | a BF16 GEMV -> per (sample, v-head): L2 norms, gates, delta rule on the bf16 state,
         gated RMSNorm x silu(z) -> out_proj -> attention TP peer reduce + residual   (sym attn)
      -> post-attention RMSNorm -> router (512 experts + the shared-expert gate) + FP8 MoE input
      -> softmax top-10 -> shared + 10 routed expert up/gate (W8A8) + SiLU
      -> expert down + route weighting -> MoE TP peer reduce + residual -> x_out     (sym ffn)

Full attention replaces the mixer: q | gate, k, v FP8 GEMV -> Gemma q / k norm + partial RoPE, K / V into
ATOM's paged FP8 cache -> split attention (one wave per query head) -> merge x sigmoid(gate) -> out_proj.

Scheduling, mailboxes and the GEMV units follow the DeepSeek-V4 MonoKernel (see its ``build.py``):
task ``t`` of a stage runs on CTA ``(stage_base + t) % 256``, every CTA walks the stages in order,
hand-offs are tagged ``(value, tag)`` pairs polled until the tag is this launch's epoch.
"""

import enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Int32, Int64, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import SHARED_SOURCE_KEY
from kernels.monokernel.ops import rsrc, uniform
from kernels.monokernel.qwen35.kernel.attn import attn_stages
from kernels.monokernel.qwen35.kernel.common import common_defs
from kernels.monokernel.qwen35.kernel.ffn import ffn_stages
from kernels.monokernel.qwen35.kernel.gdn import gdn_stages
from kernels.monokernel.qwen35.kernel.mixer import mixer_defs, out_stages
from kernels.monokernel.qwen35.kernel.plan import (
    BLOCKS,
    BS,
    KT,
    LAYER_SLOTS,
    MOE_MISC,
    PROJ_GROUPS,
    PROJ_TILE,
    ROUTER_TILE,
    ROW_TILE,
    THREADS,
    UG8,
    WAVES,
    dn_tile,
    gdn_dims,
    layout,
    n_split,
    out_k,
    router_rows,
    stage_tasks,
)


def build_qwen35_kernel(
    S: int,
    npes: int,
    full: bool,
    *,
    hidden,
    hk,
    hv,
    k_dim,
    v_dim,
    heads,
    head_dim,
    n_experts,
    top_k,
    inter,
    eps,
    rope_dim,
    tokens_per_seq=1,
    replay_cap=0,
    timeline=False,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole layer: full attention when ``full``, else
    Gated DeltaNet. ``heads`` / ``inter`` are this rank's (one K / V head per rank). ``tokens_per_seq`` > 1:
    an MTP verify step, rows ``q * tokens_per_seq + j`` = token j of sequence q (consecutive positions).
    ``replay_cap`` > 0: the verify runs on ReplaySSM GDN state (a checkpoint per request and ``replay_cap``
    records of (k, u, g) a v-head), as ATOM keeps it by default under speculative decoding."""
    assert 1 <= S <= 8, "dn_route routes sample s on wave s; the samples sit in the MFMA's B columns"
    assert k_dim == v_dim == 128, "the recurrence maps 8 K lanes x 4 values and 64 V rows per pass"
    assert hv % hk == 0
    assert n_experts % 64 == 0 and top_k < 16
    assert inter % 128 == 0 and hidden % 128 == 0
    HIDDEN = hidden
    T_SEQ = tokens_per_seq
    assert S % T_SEQ == 0
    REPLAY = replay_cap > 0
    CAP = replay_cap
    assert not REPLAY or (not full and CAP >= 2 * T_SEQ), "two verify windows fit the record buffer"
    HK, HV, K_DIM, V_DIM = hk, hv, k_dim, v_dim
    VPK = HV // HK
    g = gdn_dims(dict(hk=hk, hv=hv, k_dim=k_dim, v_dim=v_dim))
    KEY_DIM, VALUE_DIM, CONV_DIM = g["key_dim"], g["value_dim"], g["conv_dim"]
    assert (CONV_DIM + VALUE_DIM) % (PROJ_TILE * PROJ_GROUPS) == 0 and CONV_DIM % (PROJ_TILE * PROJ_GROUPS) == 0
    assert (2 * HV) % (PROJ_TILE * PROJ_GROUPS) == 0, "b | a rows must fill whole proj tasks"
    assert VALUE_DIM % 256 == 0 and HIDDEN % ROW_TILE == 0
    N_EXPERTS, TOP_K, INTER = n_experts, top_k, inter
    EPS = eps
    MOE_SLOTS = TOP_K + 1
    SHARED_EXPERT = N_EXPERTS
    ROUTER_ROWS = router_rows(N_EXPERTS)
    N_ROUTER = ROUTER_ROWS // ROUTER_TILE
    ID_BITS = max(8, (N_EXPERTS - 1).bit_length())
    ID_MASK = (1 << ID_BITS) - 1
    _off, TOPK_SUM_OFFS = 1, []
    while _off < TOP_K:
        TOPK_SUM_OFFS.append(_off)
        _off *= 2
    TOPK_SUM_OFFS = tuple(TOPK_SUM_OFFS)
    XQ_BLOCKS = HIDDEN // 128
    PUBLISH_BLOCKS = XQ_BLOCKS
    XQ_WAVES = (PUBLISH_BLOCKS + N_ROUTER - 1) // N_ROUTER
    assert XQ_WAVES <= WAVES
    UG_PER_SLOT = INTER // UG8
    N_UG_TASKS = TOP_K * UG_PER_SLOT
    DN_TILE = dn_tile(S, HIDDEN)
    N_DN_TILES = HIDDEN // DN_TILE
    # full attention: this rank's query heads share its one K / V head
    FULL = full
    H, HD, ROPE = heads, head_dim, rope_dim
    KVD = HD
    QG = H * 2 * HD
    SCALE = HD**-0.5
    NSPLIT = n_split(S)
    KROWE = HD + 8  # bf16 elements per query row in kvt, padded off the bank
    if FULL:
        assert H == WAVES, "split attention runs one query head per wave"
        assert HD % 64 == 0 and HD <= 512 and ROPE % 2 == 0 and ROPE <= HD
        assert BS == 16 and KT == 64, "a tile is 4 cache blocks: a key row group per wave quarter"
        assert (H * HD) % 512 == 0 and NSPLIT <= 512
    OUT_K = out_k(full, dict(hk=hk, hv=hv, k_dim=k_dim, v_dim=v_dim, heads=heads, head_dim=head_dim))
    assert OUT_K % 256 == 0
    W = npes
    G = BLOCKS
    dims = dict(
        hidden=hidden,
        hk=hk,
        hv=hv,
        k_dim=k_dim,
        v_dim=v_dim,
        heads=heads,
        head_dim=head_dim,
        n_experts=n_experts,
        top_k=top_k,
        inter=inter,
    )
    SC, SY = layout(S, W, full, **dims)
    stages = stage_tasks(S, full, tokens_per_seq=T_SEQ, **dims)
    base, first, acc = {}, {}, 0
    for name, n in stages:
        first[name] = acc
        acc += n
    acc = 0
    for name, n in stages:
        base[name] = acc % G
        acc += n
    XN = max(S * HIDDEN // 2, 1024, (16 * KT + 32 + 2 * HD + 2) if FULL else 0)  # split: P | rescale | new K / V
    ON = max(S * max(ROW_TILE, PROJ_TILE * PROJ_GROUPS, DN_TILE), V_DIM, HD)
    MISC = max(MOE_MISC + S * XQ_BLOCKS, S * MOE_SLOTS * INTER // 128, 2 * NSPLIT) + 8

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 / FP8 activations, the recurrence's inputs
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, MISC, 16]
        keys: fx.Array[fx.Int32, S * MOE_SLOTS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]
        kvt: fx.Array[fx.BFloat16, (16 * KROWE) if FULL else 8, 16]  # the attention's query, 16 padded rows

    # the stages read this build's constants from bc; _cache_tag puts them in the JIT cache key,
    # with the digest of the shared helper sources (outside this directory, so outside the key)
    bc = dict(locals())
    _cache_tag = (SHARED_SOURCE_KEY,) + tuple(
        (n, v) for n, v in sorted(bc.items()) if isinstance(v, (bool, int, float, str, tuple, type(None), enum.Enum))
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def qwen35_kernel(
        h_in: Int64,
        x_out: Int64,
        slots_in: Int64,
        slots_out: Int64,
        n_accepted: Int64,
        write_pos: Int64,
        rec_k: Int64,
        rec_u: Int64,
        rec_g: Int64,
        conv_state: Int64,
        ssm_state: Int64,
        positions: Int64,
        block_tables: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_qn1: Int64,
        g_kn1: Int64,
        g_in1: Int64,
        w_qkvz: Int64,
        s_qkvz: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        g_norm: Int64,
        w_out: Int64,
        s_out: Int64,
        g_post1: Int64,
        w_r: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        bt_stride: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        outs = lds.out.ptr
        red = lds.red.ptr
        misc = lds.misc.ptr
        keys = lds.keys.ptr
        dnw = lds.dnw.ptr
        kvt = lds.kvt.ptr
        # the full-attention projection's matrix comes in the GDN qkvz slot
        w_qkv, s_qkv = w_qkvz, s_qkvz
        # this launch's epoch: a per-step device counter, unique per layer
        tag = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32)) * LAYER_SLOTS + layer + 1
        r_peers = rsrc(peers)
        # each wave sends to one peer
        pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
        peer_dst = (fx.Int64(uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(uniform(pv[0])))

        _ = _cache_tag
        ctx = dict(bc)
        ctx.update(locals())
        ctx.update(common_defs(ctx))
        ctx.update(mixer_defs(ctx))
        if const_expr(FULL):
            attn_stages(ctx)
        else:
            gdn_stages(ctx)
        out_stages(ctx)
        ffn_stages(ctx)

    @flyc.jit
    def launch_qwen35(
        h_in: Int64,
        x_out: Int64,
        slots_in: Int64,
        slots_out: Int64,
        n_accepted: Int64,
        write_pos: Int64,
        rec_k: Int64,
        rec_u: Int64,
        rec_g: Int64,
        conv_state: Int64,
        ssm_state: Int64,
        positions: Int64,
        block_tables: Int64,
        slot_mapping: Int64,
        k_cache: Int64,
        v_cache: Int64,
        k_scale: Int64,
        v_scale: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_qn1: Int64,
        g_kn1: Int64,
        g_in1: Int64,
        w_qkvz: Int64,
        s_qkvz: Int64,
        w_ba: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        g_norm: Int64,
        w_out: Int64,
        s_out: Int64,
        g_post1: Int64,
        w_r: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        rank: Int32,
        layer: Int32,
        bt_stride: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        qwen35_kernel(
            h_in,
            x_out,
            slots_in,
            slots_out,
            n_accepted,
            write_pos,
            rec_k,
            rec_u,
            rec_g,
            conv_state,
            ssm_state,
            positions,
            block_tables,
            slot_mapping,
            k_cache,
            v_cache,
            k_scale,
            v_scale,
            rope_cos,
            rope_sin,
            g_qn1,
            g_kn1,
            g_in1,
            w_qkvz,
            s_qkvz,
            w_ba,
            w_conv,
            a_log,
            dt_bias,
            g_norm,
            w_out,
            s_out,
            g_post1,
            w_r,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            rank,
            layer,
            bt_stride,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return flyc.compile[{"llvm_options": {"disable-vector-combine": True}}](launch_qwen35)
