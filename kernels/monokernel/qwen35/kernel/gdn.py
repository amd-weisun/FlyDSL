# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The Gated DeltaNet mixer of the Qwen3.5 MonoKernel: projections (+ causal conv) and the delta-rule
recurrence with its gated norm (publishing ``gated`` for the shared out_proj)."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import bf16_round, exp, ld_bf16, ld_f32, lds_ld, lds_st, rsq, rsrc, uniform, xshfl
from kernels.monokernel.qwen35.kernel.common import sigmoid, silu
from kernels.monokernel.qwen35.kernel.plan import PROJ_GROUPS, PROJ_TILE, WAVES

# recurrence thread mapping over ATOM's K-major state [K][V]: lane % 8 takes 16 consecutive k, lane // 8
# and the wave take 2 consecutive v (one 4-byte vector per k); k-reductions span 8 lanes (3 DPP levels)
KPT = 16  # k per thread
VPT = 2  # v per thread
KL = 8  # lanes sharing a v pair (over k)


def _sum8(v):
    for off in (1, 2, 4):
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
    T_SEQ = ctx["T_SEQ"]
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
    n_accepted = ctx["n_accepted"]
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

    r_si, r_so, r_na = rsrc(slots_in), rsrc(slots_out), rsrc(n_accepted)
    r_conv, r_wc = rsrc(conv_state), rsrc(w_conv)
    # MTP verify (T_SEQ > 1): rows q * T_SEQ + j are token j of sequence q; slots_out[row] = the state slot
    # after the row's token (ATOM's spec_state_indices), n_accepted[q] = the tokens the last step accepted.
    # The sequence's conv window lives in token 0's slot (CW = T_SEQ + 2 inputs, read from n_accepted - 1).
    CW = T_SEQ + 2
    ROWS = PROJ_TILE * PROJ_GROUPS

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

    def conv_epilogue_mtp(s, ch, v):
        """Channel ``ch`` of every token of sequence ``s // T_SEQ`` (on token 0's thread): the window
        [3 stored inputs from n_accepted - 1, this launch's T_SEQ] -> T_SEQ outputs; the stored
        inputs shift by one (ATOM's speculative causal_conv1d_update)."""
        if s % T_SEQ == 0:
            so = fx.Int32(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
            na = fx.Int32(bo.buffer_load(r_na, s // T_SEQ, vec_width=1, dtype=T.i32))
            ok = (so >= 0) & (na >= 1)
            base = fx.max(so, 0) * CW + fx.max(na - 1, 0)
            win = [ld_bf16(r_conv, (base + j) * CONV_DIM + ch) for j in range(3)]
            win += [bf16_round(lds_ld(outs, (s + j) * ROWS + ch % ROWS)) for j in range(T_SEQ)]
            wv = fx.Vector(bo.buffer_load(r_wc, ch * 4, vec_width=4, dtype=T.bf16)).to(fx.Float32)
            for j in range_constexpr(T_SEQ):
                acc = win[j] * wv[0] + win[j + 1] * wv[1] + win[j + 2] * wv[2] + win[j + 3] * wv[3]
                put(mb("conv"), (s + j) * CONV_DIM + ch, ok.select(bf16_round(silu(acc)), fx.Float32(0.0)))
            if ok:
                for i in range_constexpr(CW):
                    bo.buffer_store(win[i + 1].to(fx.BFloat16), r_conv, (fx.max(so, 0) * CW + i) * CONV_DIM + ch)

    def qkvz_epilogue(s, row, v):
        if row < CONV_DIM:
            if const_expr(T_SEQ == 1):
                conv_epilogue(s, row, v)
            else:
                conv_epilogue_mtp(s, row, v)
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
    assert K_DIM == KL * KPT and V_DIM == WAVES * (64 // KL) * VPT, "one 128 x 128 head over 512 threads"
    HEAD_BYTES = V_DIM * K_DIM * 2
    kg = lane % KL
    k0_ = kg * KPT
    v0_ = (wave * (64 // KL) + lane // KL) * VPT
    r_alog, r_dtb, r_gn = rsrc(a_log), rsrc(dt_bias), rsrc(g_norm)
    v4f = fx.Vector.make_type(4, fx.Float32)
    vvf = fx.Vector.make_type(VPT, fx.Float32)
    # LDS (f32 words of xs): q 0, k K_DIM, v 2*K_DIM, z 2*K_DIM + V_DIM, b / a after
    XQ, XK, XV = 0, K_DIM, 2 * K_DIM
    XZ = XV + V_DIM
    XB = XZ + V_DIM
    for tt in range(start("gdn"), S // T_SEQ * HV, G):
        tt = fx.Int32(tt)
        sq = tt // HV
        h = tt % HV
        kh = h // VPK
        stamp("gdn", tt, 0)
        gpu.barrier()
        if const_expr(T_SEQ == 1):
            si = uniform(bo.buffer_load(r_si, sq, vec_width=1, dtype=T.i32))
            live = si >= 0
        else:
            # MTP verify: the state after the last accepted token of the previous step
            na = uniform(bo.buffer_load(r_na, sq, vec_width=1, dtype=T.i32))
            si = uniform(bo.buffer_load(r_so, sq * T_SEQ + fx.max(na - 1, 0), vec_width=1, dtype=T.i32))
            live = (si >= 0) & (na >= 1)
        # the state does not depend on this launch: issue its loads before waiting for the inputs
        r_in = rsrc(ssm_state + (fx.Int64(fx.max(si, 0)) * HV + fx.Int64(h)) * HEAD_BYTES)
        sv = [
            fx.Vector(bo.buffer_load(r_in, (k0_ + j) * V_DIM + v0_, vec_width=VPT, dtype=T.bf16)).to(fx.Float32)
            for j in range(KPT)
        ]
        alog = ld_f32(r_alog, h)
        dtb = ld_f32(r_dtb, h)
        for tok in range_constexpr(T_SEQ):  # the sequence's tokens in order, the state in registers
            s = sq * T_SEQ + tok
            so = uniform(bo.buffer_load(r_so, s, vec_width=1, dtype=T.i32))
            ok = live & (so >= 0)
            if const_expr(tok > 0):
                gpu.barrier()  # the previous token's LDS inputs are read
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
            # this thread's KPT q / k (L2-normalized over the head: sums over the KL lanes)
            qx = [fx.Vector(fx.ptr_load(xs + (XQ + k0_ + 4 * i), result_type=v4f)) for i in range(KPT // 4)]
            kx = [fx.Vector(fx.ptr_load(xs + (XK + k0_ + 4 * i), result_type=v4f)) for i in range(KPT // 4)]
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
            # decay, S^T k and S^T q over this thread's k (VPT v at once), summed over the KL lanes
            dvec = fx.Vector.filled(VPT, decay, fx.Float32)
            skp = fx.Vector.filled(VPT, 0.0, fx.Float32)
            sqp = fx.Vector.filled(VPT, 0.0, fx.Float32)
            for j in range_constexpr(KPT):
                sv[j] = sv[j] * dvec
                skp = fx.math.fma(sv[j], fx.Vector.filled(VPT, kk[j], fx.Float32), skp)
                sqp = fx.math.fma(sv[j], fx.Vector.filled(VPT, qq[j], fx.Float32), sqp)
            vin = fx.Vector(fx.ptr_load(xs + (XV + v0_), result_type=vvf))
            delta = [(vin[e] - _sum8(skp[e])) * beta for e in range(VPT)]
            dv = fx.Vector.from_elements(delta, fx.Float32)
            for j in range_constexpr(KPT):
                sv[j] = fx.math.fma(fx.Vector.filled(VPT, kk[j], fx.Float32), dv, sv[j])
            outs_v = [bf16_round(_sum8(sqp[e]) + delta[e] * dot_kq) for e in range(VPT)]
            stamp("gdn", tt, 3)
            # gated RMSNorm over the head's V_DIM outputs (lane kg 0 of each group holds its VPT once)
            ssq = fx.Float32(0.0)
            for e in range_constexpr(VPT):
                ssq = ssq + outs_v[e] * outs_v[e]
            rstd = rsq(block_sum((kg == 0).select(ssq, fx.Float32(0.0))) * (1.0 / V_DIM) + EPS)
            if kg == 0:
                for e in range_constexpr(VPT):
                    v_idx = v0_ + e
                    nrm = bf16_round(outs_v[e] * rstd)
                    gated = bf16_round(bf16_round(nrm * ld_f32(r_gn, v_idx)) * silu(lds_ld(xs, XZ + v_idx)))
                    lds_st(outs, v_idx, ok.select(gated, fx.Float32(0.0)))
            gpu.barrier()
            if tid < V_DIM // 2:
                put_bf(mb("gated"), s * VALUE_DIM + h * V_DIM + 2 * tid, [lds_ld(outs, 2 * tid), lds_ld(outs, 2 * tid + 1)])
            # the new state after the publish: out_proj waits on the gated row, not on this store
            if ok:
                r_o = rsrc(ssm_state + (fx.Int64(so) * HV + fx.Int64(h)) * HEAD_BYTES)
                for j in range_constexpr(KPT):
                    bo.buffer_store(sv[j].to(fx.BFloat16), r_o, (k0_ + j) * V_DIM + v0_)
            stamp("gdn", tt, 4)

    return {}
