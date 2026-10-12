# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The Gated DeltaNet mixer of the Qwen3.5 MonoKernel: projections (+ causal conv) and the delta-rule
recurrence with its gated norm (publishing ``gated`` for the shared out_proj)."""

import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import bf16_round, exp, ld_bf16, ld_f32, lds_ld, lds_st, rsq, rsrc, uniform, xshfl
from kernels.monokernel.qwen35.kernel.common import sigmoid, silu
from kernels.monokernel.qwen35.kernel.plan import PROJ_TILE, WAVES

# recurrence thread mapping (as Kimi-K3's kda_recurrence): 8 lanes split K, 64 V rows per pass
K_LANES = 8
VPT = 4  # K values per thread per pass
K_TILE = K_LANES * VPT


def _subgroup_sum(v):
    for off in (4, 2, 1):
        v = v + xshfl(v, off)
    return v


@traced
def gdn_stages(ctx):
    """1-3. qkv|z and b|a projections (conv fused), recurrence + gated norm."""
    CONV_DIM = ctx["CONV_DIM"]
    EPS = ctx["EPS"]
    G = ctx["G"]
    HV = ctx["HV"]
    KEY_DIM = ctx["KEY_DIM"]
    K_DIM = ctx["K_DIM"]
    S = ctx["S"]
    VALUE_DIM = ctx["VALUE_DIM"]
    VPK = ctx["VPK"]
    V_DIM = ctx["V_DIM"]
    a_log = ctx["a_log"]
    block_sum = ctx["block_sum"]
    conv_state = ctx["conv_state"]
    dt_bias = ctx["dt_bias"]
    g_norm = ctx["g_norm"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    proj_stage = ctx["proj_stage"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    s_qkvz = ctx["s_qkvz"]
    slots_in = ctx["slots_in"]
    slots_out = ctx["slots_out"]
    ssm_state = ctx["ssm_state"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    w_ba = ctx["w_ba"]
    w_conv = ctx["w_conv"]
    w_qkvz = ctx["w_qkvz"]
    wave = ctx["wave"]
    xs = ctx["xs"]

    r_si, r_so = rsrc(slots_in), rsrc(slots_out)
    r_conv, r_wc = rsrc(conv_state), rsrc(w_conv)

    # ================================ 1. qkv | z (FP8) and b | a (BF16) projections
    def conv_epilogue(s, ch, v):
        """Channel ``ch`` of sample ``s``: causal conv over the 3 stored inputs and ``v``, SiLU;
        the state shifts to slots_out. A padded sample (negative slot) publishes 0."""
        si = fx.Int32(bo.buffer_load(r_si, s, vec_width=1, dtype=T.i32))
        so = fx.Int32(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
        ok = (si >= 0) & (so >= 0)
        x = bf16_round(v)
        cs = [ld_bf16(r_conv, (fx.max(si, 0) * 3 + j) * CONV_DIM + ch) for j in range(3)]
        wv = fx.Vector(bo.buffer_load(r_wc, ch * 4, vec_width=4, dtype=T.bf16)).to(fx.Float32)
        acc = cs[0] * wv[0] + cs[1] * wv[1] + cs[2] * wv[2] + x * wv[3]
        if ok:
            for j, val in ((0, cs[1]), (1, cs[2]), (2, x)):
                bo.buffer_store(val.to(fx.BFloat16), r_conv, (so * 3 + j) * CONV_DIM + ch)
        put(mb("conv"), s * CONV_DIM + ch, ok.select(bf16_round(silu(acc)), fx.Float32(0.0)))

    def qkvz_epilogue(s, row, v):
        if row < CONV_DIM:
            conv_epilogue(s, row, v)
        else:
            put(mb("z"), s * VALUE_DIM + row - CONV_DIM, bf16_round(v))

    proj_stage("proj", (CONV_DIM + VALUE_DIM) // PROJ_TILE, w_qkvz, s_qkvz, False, qkvz_epilogue)
    proj_stage(
        "proj_ba",
        2 * HV // PROJ_TILE,
        w_ba,
        None,
        True,
        lambda s, row, v: put(mb("ba"), s * 2 * HV + row, bf16_round(v)),
    )

    # ================================ 2. gated delta rule + gated RMSNorm (one task per sample x v-head)
    V_TILE = WAVES * (64 // K_LANES)
    V_ITERS = V_DIM // V_TILE
    K_ITERS = K_DIM // K_TILE
    HEAD_BYTES = V_DIM * K_DIM * 2
    k_lane = lane % K_LANES
    v_lane = lane // K_LANES
    r_alog, r_dtb, r_gn = rsrc(a_log), rsrc(dt_bias), rsrc(g_norm)
    v4f = fx.Vector.make_type(VPT, fx.Float32)
    # LDS (f32 words of xs): q 0, k K_DIM, v 2*K_DIM, z 2*K_DIM + V_DIM, b / a after
    XQ, XK, XV = 0, K_DIM, 2 * K_DIM
    XZ = XV + V_DIM
    XB = XZ + V_DIM
    for tt in range(start("gdn"), S * HV, G):
        tt = fx.Int32(tt)
        s = tt // HV
        h = tt % HV
        kh = h // VPK
        stamp("gdn", tt, 0)
        gpu.barrier()
        si = uniform(bo.buffer_load(r_si, s, vec_width=1, dtype=T.i32))
        so = uniform(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
        ok = (si >= 0) & (so >= 0)
        # the state does not depend on this launch: issue its loads before waiting for the inputs
        r_in = rsrc(ssm_state + (fx.Int64(fx.max(si, 0)) * HV + fx.Int64(h)) * HEAD_BYTES)
        sv = []
        for vi in range_constexpr(V_ITERS):
            v_idx = wave * (64 // K_LANES) + v_lane + vi * V_TILE
            for ki in range_constexpr(K_ITERS):
                k_base = k_lane * VPT + ki * K_TILE
                sv.append(
                    fx.Vector(bo.buffer_load(r_in, v_idx * K_DIM + k_base, vec_width=VPT, dtype=T.bf16)).to(fx.Float32)
                )
        alog = ld_f32(r_alog, h)
        dtb = ld_f32(r_dtb, h)
        if tid < K_DIM:
            got = poll(
                [
                    (mb("conv"), s * CONV_DIM + kh * K_DIM + tid, 1),
                    (mb("conv"), s * CONV_DIM + KEY_DIM + kh * K_DIM + tid, 1),
                    (mb("conv"), s * CONV_DIM + 2 * KEY_DIM + h * V_DIM + tid, 1),
                    (mb("z"), s * VALUE_DIM + h * V_DIM + tid, 1),
                ]
            )
            for j, off in ((0, XQ), (1, XK), (2, XV), (3, XZ)):
                lds_st(xs, off + tid, got[j][0].bitcast(fx.Float32))
        if tid == K_DIM:
            got = poll([(mb("ba"), s * 2 * HV + h, 1), (mb("ba"), s * 2 * HV + HV + h, 1)])
            lds_st(xs, XB, got[0][0].bitcast(fx.Float32))
            lds_st(xs, XB + 1, got[1][0].bitcast(fx.Float32))
        gpu.barrier()
        stamp("gdn", tt, 2)
        b_in = lds_ld(xs, XB)
        a_in = lds_ld(xs, XB + 1)
        beta = sigmoid(b_in)
        x_sp = a_in + dtb
        softplus = (x_sp > 20.0).select(x_sp, fmath.log1p(exp(x_sp)))
        decay = exp(-exp(alog) * softplus)
        qv, kv = [], []
        qsq = fx.Float32(0.0)
        ksq = fx.Float32(0.0)
        for ki in range_constexpr(K_ITERS):
            k_base = k_lane * VPT + ki * K_TILE
            qx = fx.Vector(fx.ptr_load(xs + (XQ + k_base), result_type=v4f))
            kx = fx.Vector(fx.ptr_load(xs + (XK + k_base), result_type=v4f))
            qv.append(qx)
            kv.append(kx)
            qsq = qsq + (qx * qx).reduce(fx.ReductionOp.ADD)
            ksq = ksq + (kx * kx).reduce(fx.ReductionOp.ADD)
        q_scale = rsq(_subgroup_sum(qsq) + 1.0e-6) * (K_DIM**-0.5)
        k_scale = rsq(_subgroup_sum(ksq) + 1.0e-6)
        qv = [x * fx.Vector.filled(VPT, q_scale, fx.Float32) for x in qv]
        kv = [x * fx.Vector.filled(VPT, k_scale, fx.Float32) for x in kv]
        kq = fx.Vector.filled(VPT, 0.0, fx.Float32)
        for ki in range_constexpr(K_ITERS):
            kq = fx.math.fma(kv[ki], qv[ki], kq)
        dot_kq = _subgroup_sum(kq.reduce(fx.ReductionOp.ADD))
        dvec = fx.Vector.filled(VPT, decay, fx.Float32)
        outs_v = []
        for vi in range_constexpr(V_ITERS):
            v_idx = wave * (64 // K_LANES) + v_lane + vi * V_TILE
            skp = fx.Vector.filled(VPT, 0.0, fx.Float32)
            sqp = fx.Vector.filled(VPT, 0.0, fx.Float32)
            for ki in range_constexpr(K_ITERS):
                dec = sv[vi * K_ITERS + ki] * dvec
                sv[vi * K_ITERS + ki] = dec
                skp = fx.math.fma(dec, kv[ki], skp)
                sqp = fx.math.fma(dec, qv[ki], sqp)
            s_k = _subgroup_sum(skp.reduce(fx.ReductionOp.ADD))
            s_q = _subgroup_sum(sqp.reduce(fx.ReductionOp.ADD))
            v_new = (lds_ld(xs, XV + v_idx) - s_k) * beta
            vn = fx.Vector.filled(VPT, v_new, fx.Float32)
            for ki in range_constexpr(K_ITERS):
                sv[vi * K_ITERS + ki] = fx.math.fma(kv[ki], vn, sv[vi * K_ITERS + ki])
            outs_v.append(bf16_round(s_q + v_new * dot_kq))
        if ok:
            r_o = rsrc(ssm_state + (fx.Int64(so) * HV + fx.Int64(h)) * HEAD_BYTES)
            for vi in range_constexpr(V_ITERS):
                v_idx = wave * (64 // K_LANES) + v_lane + vi * V_TILE
                for ki in range_constexpr(K_ITERS):
                    k_base = k_lane * VPT + ki * K_TILE
                    bo.buffer_store(sv[vi * K_ITERS + ki].to(fx.BFloat16), r_o, v_idx * K_DIM + k_base)
        stamp("gdn", tt, 3)
        # gated RMSNorm over the head's V_DIM outputs (k_lane 0 holds each once)
        sq = fx.Float32(0.0)
        for vi in range_constexpr(V_ITERS):
            sq = sq + outs_v[vi] * outs_v[vi]
        rstd = rsq(block_sum((k_lane == 0).select(sq, fx.Float32(0.0))) * (1.0 / V_DIM) + EPS)
        if k_lane == 0:
            for vi in range_constexpr(V_ITERS):
                v_idx = wave * (64 // K_LANES) + v_lane + vi * V_TILE
                nrm = bf16_round(outs_v[vi] * rstd)
                gated = bf16_round(bf16_round(nrm * ld_f32(r_gn, v_idx)) * silu(lds_ld(xs, XZ + v_idx)))
                lds_st(outs, v_idx, ok.select(gated, fx.Float32(0.0)))
        gpu.barrier()
        if tid < V_DIM // 2:
            put_bf(mb("gated"), s * VALUE_DIM + h * V_DIM + 2 * tid, [lds_ld(outs, 2 * tid), lds_ld(outs, 2 * tid + 1)])
        stamp("gdn", tt, 4)

    return {}
