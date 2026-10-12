# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Qwen3.8-Flash-Next decoder layer in ONE persistent launch (TP1).

One launch of ``grid = 256 CTAs x 512 threads`` maps the hyper-connection bundle X [S, hc * hidden] to the next::

    HC read (down | inject over K splits -> rstd-weighted reduce -> up + gated stream mean) -> m
      -> [q | k | v | z | b | a] BF16 GEMV (+ causal conv1d, SiLU, conv-state shift)
      -> per (sample, v-head): L2 norms, gates, delta rule on the fp32 state, gated RMSNorm x sigmoid(z)
      -> out_proj -> combine: x1 = X + y * 2 sigmoid(r / hc)                                  (mailbox x1)
      -> HC read of x1 -> m2 -> router (512 experts + shared gate) + FP8 MoE input
      -> BF16 shared expert up/gate; softmax top-10 routed up/gate (W8A8) + SiLU
      -> routed + shared down -> combine into x_out

Scheduling, mailboxes and GEMV units follow the Qwen3.5 / DeepSeek-V4 MonoKernels: task ``t`` of a stage runs
on CTA ``(stage_base + t) % 256``, hand-offs are tagged ``(value, tag)`` pairs polled until the tag is this
launch's epoch. No TP reduce: every weight is whole.
"""

import enum

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64, T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import SHARED_SOURCE_KEY
from kernels.monokernel.ops import rsrc, uniform
from kernels.monokernel.qwen35.kernel.common import common_defs
from kernels.monokernel.qwen38.kernel.ffn import ffn_stages
from kernels.monokernel.qwen38.kernel.gdn import gdn_stages
from kernels.monokernel.qwen38.kernel.hc import hc_stages
from kernels.monokernel.qwen38.kernel.out import out_stage
from kernels.monokernel.qwen38.kernel.plan import (
    BLOCKS,
    HC_KS,
    LAYER_SLOTS,
    MOE_MISC,
    PROJ_GROUPS,
    PROJ_TILE,
    THREADS,
    WAVES,
    dn_tile,
    gdn_dims,
    hc_ksplit,
    hc_rows,
    layout,
    out_k,
    router_rows,
    stage_tasks,
)

ARGS = (
    "h_in x_out slots_in slots_out conv_state ssm_state "
    "hca_g hca_down hca_up hca_inj hcm_g hcm_down hcm_up hcm_inj "
    "w_proj w_conv a_log dt_bias g_norm w_out w_r w_ug s_ug w_dn s_dn w_sh_ug w_sh_dn "
    "scratch timeline_buf step"
).split()


def build_qwen38_kernel(S: int, full: bool, *, timeline=False, **dims):
    """Return the ``@flyc.jit`` launcher for one whole layer (GDN; QSA to come)."""
    assert not full, "the QSA layer is not built yet"
    assert 1 <= S <= 8
    assert dims["k_dim"] == dims["v_dim"] == 128 and dims["hv"] % dims["hk"] == 0
    HIDDEN = dims["hidden"]
    HC = dims["hc"]
    LOWRANK = dims["hc_lowrank"]
    KSPLIT, RR = hc_ksplit(dims), hc_rows(dims)
    assert HIDDEN % HC_KS == 0 and LOWRANK % 64 == 0 and HIDDEN % 32 == 0
    HK, HV, K_DIM, V_DIM = dims["hk"], dims["hv"], dims["k_dim"], dims["v_dim"]
    VPK = HV // HK
    g = gdn_dims(dims)
    KEY_DIM, VALUE_DIM, CONV_DIM = g["key_dim"], g["value_dim"], g["conv_dim"]
    assert CONV_DIM % (PROJ_TILE * PROJ_GROUPS) == 0
    N_EXPERTS, TOP_K, INTER, SH_INTER = dims["n_experts"], dims["top_k"], dims["inter"], dims["shared_inter"]
    assert N_EXPERTS % 64 == 0 and TOP_K < 16 and INTER % 128 == 0 and SH_INTER % 64 == 0
    EPS = dims["eps"]
    ROUTER_ROWS = router_rows(N_EXPERTS)
    ID_BITS = max(8, (N_EXPERTS - 1).bit_length())
    ID_MASK = (1 << ID_BITS) - 1
    _off, TOPK_SUM_OFFS = 1, []
    while _off < TOP_K:
        TOPK_SUM_OFFS.append(_off)
        _off *= 2
    TOPK_SUM_OFFS = tuple(TOPK_SUM_OFFS)
    XQ_BLOCKS = HIDDEN // 128
    DN_TILE = dn_tile(S, HIDDEN)
    OUT_K = out_k(full, dims)
    W = 1
    G = BLOCKS
    SC = layout(S, full, **dims)
    SY = {"_bytes": 0}
    stages = stage_tasks(S, full, **dims)
    base, first, acc = {}, {}, 0
    for name, n in stages:
        first[name] = acc
        acc += n
    acc = 0
    for name, n in stages:
        base[name] = acc % G
        acc += n
    XN = max(
        S * HIDDEN // 2,
        S * OUT_K // 2,
        S * TOP_K * INTER // 4 + S * SH_INTER // 2,
        2 * K_DIM + 2 * V_DIM + 2,
        1024,
    )
    ON = max(S * PROJ_TILE * PROJ_GROUPS, S * WAVES * 16, 2 * S * DN_TILE, V_DIM)
    MISC = max(MOE_MISC + S * XQ_BLOCKS, S * TOP_K * INTER // 128, KSPLIT) + 8

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, MISC, 16]
        keys: fx.Array[fx.Int32, S * TOP_K, 16]
        dnw: fx.Array[fx.Float32, S * TOP_K, 16]

    bc = dict(locals())
    bc.pop("dims")
    _cache_tag = (SHARED_SOURCE_KEY,) + tuple(
        (n, v) for n, v in sorted(bc.items()) if isinstance(v, (bool, int, float, str, tuple, type(None), enum.Enum))
    )

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def qwen38_kernel(
        h_in: Int64,
        x_out: Int64,
        slots_in: Int64,
        slots_out: Int64,
        conv_state: Int64,
        ssm_state: Int64,
        hca_g: Int64,
        hca_down: Int64,
        hca_up: Int64,
        hca_inj: Int64,
        hcm_g: Int64,
        hcm_down: Int64,
        hcm_up: Int64,
        hcm_inj: Int64,
        w_proj: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        g_norm: Int64,
        w_out: Int64,
        w_r: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        w_sh_dn: Int64,
        scratch: Int64,
        timeline_buf: Int64,
        step: Int64,
        layer: Int32,
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
        tag = uniform(bo.buffer_load(rsrc(step), 0, vec_width=1, dtype=T.i32)) * LAYER_SLOTS + layer + 1
        rank = fx.Int32(0)
        sym = fx.Int64(0)
        peer_dst = fx.Int64(0)

        _ = _cache_tag
        ctx = dict(bc)
        ctx.update(locals())
        ctx.update(common_defs(ctx))
        ctx["unit_bf16_rows"] = ctx["st"].unit_bf16_rows
        hc_stages(ctx, "a", False)
        gdn_stages(ctx)
        out_stage(ctx)
        hc_stages(ctx, "m", True)
        ffn_stages(ctx)

    @flyc.jit
    def launch_qwen38(
        h_in: Int64,
        x_out: Int64,
        slots_in: Int64,
        slots_out: Int64,
        conv_state: Int64,
        ssm_state: Int64,
        hca_g: Int64,
        hca_down: Int64,
        hca_up: Int64,
        hca_inj: Int64,
        hcm_g: Int64,
        hcm_down: Int64,
        hcm_up: Int64,
        hcm_inj: Int64,
        w_proj: Int64,
        w_conv: Int64,
        a_log: Int64,
        dt_bias: Int64,
        g_norm: Int64,
        w_out: Int64,
        w_r: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        w_sh_ug: Int64,
        w_sh_dn: Int64,
        scratch: Int64,
        timeline_buf: Int64,
        step: Int64,
        layer: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        qwen38_kernel(
            h_in,
            x_out,
            slots_in,
            slots_out,
            conv_state,
            ssm_state,
            hca_g,
            hca_down,
            hca_up,
            hca_inj,
            hcm_g,
            hcm_down,
            hcm_up,
            hcm_inj,
            w_proj,
            w_conv,
            a_log,
            dt_bias,
            g_norm,
            w_out,
            w_r,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            w_sh_ug,
            w_sh_dn,
            scratch,
            timeline_buf,
            step,
            layer,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return flyc.compile[{"llvm_options": {"disable-vector-combine": True}}](launch_qwen38)
