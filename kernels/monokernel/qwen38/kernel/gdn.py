# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The Gated DeltaNet mixer of the Qwen3.8 MonoKernel: the BF16 [q | k | v | z | b | a] projection of the sublayer
input ``m_a`` (causal conv fused), and the delta-rule recurrence on the fp32 state with its gated norm (sigmoid
gate), publishing ``gated`` for out_proj. The Qwen3.5 recurrence's thread mapping, at 48 v-heads per k-head x 3."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import bf16_round, exp, ld_bf16, ld_f32, lds_ld, lds_st, rsq, rsrc, uniform, wave_sum, xshfl
from kernels.monokernel.qwen35.kernel.common import sigmoid, silu
from kernels.monokernel.qwen38.kernel.plan import PROJ_GROUPS, PROJ_TILE, WAVES

# recurrence thread mapping over ATOM's K-major state [K][V]: lane % 8 takes 16 consecutive k, lane // 8 and the
# wave take 2 consecutive v; k-reductions span 8 lanes
KPT = 16
VPT = 2
KL = 8


def _sum8(v):
    for off in (1, 2, 4):
        v = v + xshfl(v, off)
    return v


@traced
def proj_stage_rows(ctx, name, mats, src, epilogue):
    """Stage ``name``: the rows of row-major BF16 matrices ``mats`` = [(pointer, rows)] (one or two, read as one
    row range; the first's rows a multiple of 16) times the bf16 mailbox ``src`` [S, hidden], PROJ_GROUPS row
    groups per task; ``epilogue(s, row, value)`` per output."""
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    S = ctx["S"]
    emit_out = ctx["emit_out"]
    lds_outs = ctx["outs"]
    n_sel = ctx["n_sel"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    stage_x_pairs = ctx["stage_x_pairs"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16_rows = ctx["unit_bf16_rows"]
    wave = ctx["wave"]
    NKC = HIDDEN // 64
    WPR = WAVES // PROJ_GROUPS
    UPW = NKC // WPR
    ROWS = PROJ_TILE * PROJ_GROUPS
    assert NKC % WPR == 0
    n_rows = sum(r for _, r in mats)
    rg0 = mats[0][1] // 16
    assert mats[0][1] % 16 == 0 and len(mats) <= 2
    batch = UPW  # every unit of the task in flight before the input wait
    for t in range(start(name), -(-n_rows // ROWS), G):
        t = fx.Int32(t)
        stamp(name, t, 0)
        rg = t * PROJ_GROUPS + wave // WPR
        if const_expr(len(mats) == 1):
            r_w = bo.create_buffer_resource_from_addr(mats[0][0], num_records_bytes=fx.Int32(n_rows * HIDDEN * 2))
            rg_w = rg
        else:
            first = rg < rg0
            r_w = bo.create_buffer_resource_from_addr(
                first.select(mats[0][0], mats[1][0]),
                num_records_bytes=first.select(fx.Int32(mats[0][1] * HIDDEN * 2), fx.Int32(mats[1][1] * HIDDEN * 2)),
            )
            rg_w = first.select(rg, rg - rg0)

        def u(c):
            kc = (wave % WPR) * UPW + c
            return unit_bf16_rows(r_w, rg_w, kc, HIDDEN, (n_sel() * HIDDEN + kc * 64) // 2)

        pre = [u(c) for c in range(batch)]
        gpu.barrier()
        stage_x_pairs(src, S * HIDDEN, lambda k: k)
        gpu.barrier()
        stamp(name, t, 2)
        acc = run_units(u, UPW, batch, pre)
        reduce_rows(PROJ_GROUPS, acc, emit_out(ROWS))
        stamp(name, t, 3)
        gpu.barrier()
        if (tid < S * ROWS) & (t * ROWS + tid % ROWS < n_rows):
            epilogue(tid // ROWS, t * ROWS + tid % ROWS, lds_ld(lds_outs, tid))
        stamp(name, t, 4)


@traced
def gdn_stages(ctx):
    """1-2. [q | k | v | z | b | a] projection (conv fused), recurrence + gated norm."""
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
    conv_state = ctx["conv_state"]
    dt_bias = ctx["dt_bias"]
    g_norm = ctx["g_norm"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    red = ctx["red"]
    slots_in = ctx["slots_in"]
    slots_out = ctx["slots_out"]
    ssm_state = ctx["ssm_state"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    w_conv = ctx["w_conv"]
    w_proj = ctx["w_proj"]
    wave = ctx["wave"]
    xs = ctx["xs"]

    r_si, r_so = rsrc(slots_in), rsrc(slots_out)
    r_conv, r_wc = rsrc(conv_state), rsrc(w_conv)

    # ================================ 1. [q | k | v | z | b | a] (BF16), conv fused
    def conv_epilogue(s, ch, v):
        """Channel ``ch`` of sample ``s``: causal conv over the 3 stored inputs and ``v``, SiLU; the state shifts
        to slots_out. A padded sample (negative slot) publishes 0."""
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

    def proj_epilogue(s, row, v):
        if row < CONV_DIM:
            conv_epilogue(s, row, v)
        elif row < CONV_DIM + VALUE_DIM:
            put(mb("z"), s * VALUE_DIM + row - CONV_DIM, bf16_round(v))
        else:
            put(mb("ba"), s * 2 * HV + row - CONV_DIM - VALUE_DIM, bf16_round(v))

    proj_stage_rows(ctx, "proj", [(w_proj, CONV_DIM + VALUE_DIM + 2 * HV)], "m_a", proj_epilogue)

    # ================================ 2. gated delta rule + gated RMSNorm (one task per sample x GDN_HPT v-heads;
    # two heads -- 256 threads each -- when one head per task would take more than one round of the grid)
    HPT = ctx["GDN_HPT"]
    WPH = WAVES // HPT  # waves per head
    VPTH = V_DIM // (WPH * (64 // KL))  # v per thread
    assert K_DIM == KL * KPT and HV % HPT == 0 and VPTH in (2, 4), "one 128 x 128 head over 512 / HPT threads"
    HEAD_BYTES = V_DIM * K_DIM * 4  # fp32 state
    kg = lane % KL
    k0_ = kg * KPT
    hl = wave // WPH  # this thread's head within the task
    v0_ = ((wave % WPH) * (64 // KL) + lane // KL) * VPTH
    r_alog, r_dtb, r_gn = rsrc(a_log), rsrc(dt_bias), rsrc(g_norm)
    v4f = fx.Vector.make_type(4, fx.Float32)
    vvf = fx.Vector.make_type(VPTH, fx.Float32)
    REG = 2 * K_DIM + 2 * V_DIM + 2  # LDS words per head: q, k, v, z, b, a
    XQ, XK, XV = 0, K_DIM, 2 * K_DIM
    XZ = XV + V_DIM
    XB = XZ + V_DIM
    lb = hl * REG
    for tt in range(start("gdn"), S * HV // HPT, G):
        tt = fx.Int32(tt)
        s = tt * HPT // HV
        h0 = tt * HPT % HV
        h = h0 + hl
        stamp("gdn", tt, 0)
        gpu.barrier()
        si = uniform(bo.buffer_load(r_si, s, vec_width=1, dtype=T.i32))
        so = uniform(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
        ok = (si >= 0) & (so >= 0)
        # the state does not depend on this launch: issue its loads before waiting for the inputs
        r_in = rsrc(ssm_state + (fx.Int64(fx.max(si, 0)) * HV + fx.Int64(h)) * HEAD_BYTES)
        sv = [fx.Vector(bo.buffer_load(r_in, (k0_ + j) * V_DIM + v0_, vec_width=VPTH, dtype=T.f32)) for j in range(KPT)]
        alog = ld_f32(r_alog, h)
        dtb = ld_f32(r_dtb, h)
        if tid < HPT * K_DIM:
            hs = h0 + tid // K_DIM  # the staging thread's head
            d = tid % K_DIM
            got = poll(
                [
                    (mb("conv"), s * CONV_DIM + hs // VPK * K_DIM + d, 1),
                    (mb("conv"), s * CONV_DIM + KEY_DIM + hs // VPK * K_DIM + d, 1),
                    (mb("conv"), s * CONV_DIM + 2 * KEY_DIM + hs * V_DIM + d, 1),
                    (mb("z"), s * VALUE_DIM + hs * V_DIM + d, 1),
                ]
            )
            for j, off in ((0, XQ), (1, XK), (2, XV), (3, XZ)):
                lds_st(xs, tid // K_DIM * REG + off + d, got[j][0].bitcast(fx.Float32))
        if (tid >= HPT * K_DIM) & (tid < HPT * K_DIM + HPT):
            jh = tid - HPT * K_DIM
            got = poll([(mb("ba"), s * 2 * HV + h0 + jh, 1), (mb("ba"), s * 2 * HV + HV + h0 + jh, 1)])
            lds_st(xs, jh * REG + XB, got[0][0].bitcast(fx.Float32))
            lds_st(xs, jh * REG + XB + 1, got[1][0].bitcast(fx.Float32))
        gpu.barrier()
        stamp("gdn", tt, 2)
        beta = sigmoid(lds_ld(xs, lb + XB))
        x_sp = lds_ld(xs, lb + XB + 1) + dtb
        softplus = (x_sp > 20.0).select(x_sp, fmath.log1p(exp(x_sp)))
        decay = exp(-exp(alog) * softplus)
        qx = [fx.Vector(fx.ptr_load(xs + (lb + XQ + k0_ + 4 * i), result_type=v4f)) for i in range(KPT // 4)]
        kx = [fx.Vector(fx.ptr_load(xs + (lb + XK + k0_ + 4 * i), result_type=v4f)) for i in range(KPT // 4)]
        qq = [qx[j // 4][j % 4] for j in range(KPT)]
        kk = [kx[j // 4][j % 4] for j in range(KPT)]
        qsq = fx.Float32(0.0)
        ksq = fx.Float32(0.0)
        kq = fx.Float32(0.0)
        for j in range_constexpr(KPT):
            qsq = qsq + qq[j] * qq[j]
            ksq = ksq + kk[j] * kk[j]
            kq = kq + kk[j] * qq[j]
        q_scale = rsq(_sum8(qsq) + 1.0e-6) * (K_DIM**-0.5)
        k_scale = rsq(_sum8(ksq) + 1.0e-6)
        qq = [x * q_scale for x in qq]
        kk = [x * k_scale for x in kk]
        dot_kq = _sum8(kq) * q_scale * k_scale
        dvec = fx.Vector.filled(VPTH, decay, fx.Float32)
        skp = fx.Vector.filled(VPTH, 0.0, fx.Float32)
        sqp = fx.Vector.filled(VPTH, 0.0, fx.Float32)
        for j in range_constexpr(KPT):
            sv[j] = sv[j] * dvec
            skp = fx.math.fma(sv[j], fx.Vector.filled(VPTH, kk[j], fx.Float32), skp)
            sqp = fx.math.fma(sv[j], fx.Vector.filled(VPTH, qq[j], fx.Float32), sqp)
        vin = fx.Vector(fx.ptr_load(xs + (lb + XV + v0_), result_type=vvf))
        delta = [(vin[e] - _sum8(skp[e])) * beta for e in range(VPTH)]
        dv = fx.Vector.from_elements(delta, fx.Float32)
        for j in range_constexpr(KPT):
            sv[j] = fx.math.fma(fx.Vector.filled(VPTH, kk[j], fx.Float32), dv, sv[j])
        outs_v = [bf16_round(_sum8(sqp[e]) + delta[e] * dot_kq) for e in range(VPTH)]
        stamp("gdn", tt, 3)
        # gated RMSNorm over the head's V_DIM outputs (lane kg 0 of each group holds its VPTH once), sigmoid gate;
        # the sum over the head's WPH waves through LDS
        ssq = fx.Float32(0.0)
        for e in range_constexpr(VPTH):
            ssq = ssq + outs_v[e] * outs_v[e]
        wsq = wave_sum((kg == 0).select(ssq, fx.Float32(0.0)))
        if lane == 0:
            lds_st(red, wave, wsq)
        gpu.barrier()
        tot = fx.Float32(0.0)
        for i in range_constexpr(WPH):
            tot = tot + lds_ld(red, hl * WPH + i)
        rstd = rsq(tot * (1.0 / V_DIM) + EPS)
        if kg == 0:
            for e in range_constexpr(VPTH):
                v_idx = v0_ + e
                nrm = bf16_round(outs_v[e] * rstd)
                gated = bf16_round(bf16_round(nrm * ld_f32(r_gn, v_idx)) * sigmoid(lds_ld(xs, lb + XZ + v_idx)))
                lds_st(outs, hl * V_DIM + v_idx, ok.select(gated, fx.Float32(0.0)))
        gpu.barrier()
        if tid < HPT * V_DIM // 2:
            hp = tid // (V_DIM // 2)
            jv = tid % (V_DIM // 2)
            put_bf(
                mb("gated"),
                s * VALUE_DIM + (h0 + hp) * V_DIM + 2 * jv,
                [lds_ld(outs, hp * V_DIM + 2 * jv), lds_ld(outs, hp * V_DIM + 2 * jv + 1)],
            )
        # the new state after the publish: out_proj waits on the gated row, not on this store
        if ok:
            r_o = rsrc(ssm_state + (fx.Int64(so) * HV + fx.Int64(h)) * HEAD_BYTES)
            for j in range_constexpr(KPT):
                bo.buffer_store(sv[j], r_o, (k0_ + j) * V_DIM + v0_)
        stamp("gdn", tt, 4)

    return {}
