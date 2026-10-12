# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device helpers shared by every stage of the Qwen3.5 MonoKernel."""

import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.dsv4.kernel.common import sort_network
from kernels.monokernel.helpers import SHARED_SOURCE_KEY, LaunchState, traced
from kernels.monokernel.ops import (
    bf16_pair,
    exp,
    f8_word,
    lds_st,
    rcp,
    read_lane_i32,
    rsq,
    rsrc,
    wave_max,
    wave_umax,
    write_lane_i32,
    xred,
)
from kernels.monokernel.qwen35.config import FP8_MAX
from kernels.monokernel.qwen35.kernel.plan import MOE_MISC, TL_COLS


def silu(x):
    return x * rcp(1.0 + exp(-x))


def sigmoid(x):
    return rcp(1.0 + exp(-x))


def quant_floor(a0, a1):
    """``helpers.quant_scaled`` with aiter's group-quant floor: the block max is at least 1e-10, so a
    near-zero block gets a finite scale. -> (q0, q1, scale) of a 128-block held as 2 f32 per lane."""
    amax = fx.max(wave_max(fx.max(fmath.absf(a0), fmath.absf(a1))), fx.Float32(1.0e-10))
    qs = amax * (1.0 / FP8_MAX)
    inv = rcp(amax) * FP8_MAX
    q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
    q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
    return q0, q1, qs


@traced
def common_defs(ctx):
    """The shared helpers bound to this launch, plus the Qwen-specific ones."""
    EPS = ctx["EPS"]
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    ID_BITS = ctx["ID_BITS"]
    ID_MASK = ctx["ID_MASK"]
    N_EXPERTS = ctx["N_EXPERTS"]
    ROUTER_ROWS = ctx["ROUTER_ROWS"]
    S = ctx["S"]
    SC = ctx["SC"]
    SY = ctx["SY"]
    TOPK_SUM_OFFS = ctx["TOPK_SUM_OFFS"]
    TOP_K = ctx["TOP_K"]
    W = ctx["W"]
    XQ_BLOCKS = ctx["XQ_BLOCKS"]
    base = ctx["base"]
    bid = ctx["bid"]
    first = ctx["first"]
    lane = ctx["lane"]
    misc = ctx["misc"]
    outs = ctx["outs"]
    peer_dst = ctx["peer_dst"]
    rank = ctx["rank"]
    red = ctx["red"]
    scratch = ctx["scratch"]
    sym = ctx["sym"]
    tag = ctx["tag"]
    tid = ctx["tid"]
    timeline = ctx["timeline"]
    timeline_buf = ctx["timeline_buf"]
    wave = ctx["wave"]
    xs = ctx["xs"]

    st = LaunchState(
        source_key=SHARED_SOURCE_KEY,
        S=S,
        tid=tid,
        lane=lane,
        wave=wave,
        bid=bid,
        G=G,
        base=base,
        scratch=scratch,
        SC=SC,
        tag=tag,
        red=red,
        outs=outs,
        xs=xs,
        timeline=timeline,
        timeline_buf=timeline_buf,
        first=first,
        tl_cols=TL_COLS,
        rank=rank,
        W=W,
        HIDDEN=HIDDEN,
        peer_dst=peer_dst,
        sym=sym,
        SY=SY,
        eps=EPS,
    )
    mb = st.mb
    poll = st.poll
    block_sums = st.block_sums
    rmsnorm_tail_ks = st.rmsnorm_tail_ks

    def stage_x_gemma(ld4s, n, gamma, count=None):
        """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for ``count`` samples, ``gamma`` an f32
        vector (the Gemma ``1 + w``, formed on the host); ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])]."""
        count = S if count is None else count
        ks, active = rmsnorm_tail_ks(n)
        per = len(ks)
        rg_ = rsrc(gamma)
        gs = []
        for k in ks:
            g = fx.Vector(bo.buffer_load(rg_, k, vec_width=4, dtype=T.f32))
            gs.append([g[j] for j in range(4)])
        vals = ld4s([(s, k) for s in range(count) for k in ks])
        sss = []
        for s in range_constexpr(count):
            ss = fx.Float32(0.0)
            for i in range_constexpr(per):
                for a in vals[s * per + i]:
                    term = a * a
                    if const_expr(active is not None and i == per - 1):
                        term = active.select(term, fx.Float32(0.0))
                    ss = ss + term
            sss.append(ss)
        rstds = [rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
        for s in range_constexpr(count):
            for i in range_constexpr(per):
                a = vals[s * per + i]
                for j in range_constexpr(2):
                    lds_st(
                        xs,
                        (s * n + ks[i]) // 2 + j,
                        bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                    )
        return rstds, vals, gs, ks

    def stage_moe_input(samples):
        """LDS FP8 MoE inputs (``f8_word`` order) and their block scales (misc[MOE_MISC + ...])."""
        nq = HIDDEN // 4
        full = nq // 512
        xk = [tid + i * 512 for i in range(full)]
        if const_expr(nq % 512):
            xk.append(fx.min(tid + full * 512, nq - 1))
        nxw = len(xk)
        got = poll(
            [(mb("xq"), sx * nq + k, 1) for sx in samples for k in xk]
            + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
        )
        for j in range_constexpr(len(samples)):
            for i in range_constexpr(nxw):
                lds_st(xs, j * nq + f8_word(xk[i] * 4), got[j * nxw + i][0].bitcast(fx.Float32))
            if tid < XQ_BLOCKS:
                lds_st(misc, MOE_MISC + j * XQ_BLOCKS + tid, got[len(samples) * nxw + j][0].bitcast(fx.Float32))

    def route_softmax(s):
        """Softmax top-``TOP_K`` of sample ``s``'s router logits (one whole wave). Renormalized
        top-k of a softmax is the softmax of the picked logits. Returns (expert id, weight) of
        pick ``lane``, valid in lanes < TOP_K; picks are in descending logit order, ties to the
        lower expert id."""
        KPL = N_EXPERTS // 64
        raws = st.getf_many([(mb("scores"), s * ROUTER_ROWS + lane + i * 64) for i in range(KPL)])
        ks = []
        for i in range_constexpr(KPL):
            kb = raws[i].bitcast(fx.Int32)
            ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
            ks.append(fx.Uint32((ok & fx.Int32(-(1 << ID_BITS))) | (ID_MASK - (lane + i * 64))))
        for a, b in sort_network(KPL):
            ks[a], ks[b] = fx.max(ks[a], ks[b]), fx.min(ks[a], ks[b])
        ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
        mv = fx.Int32(0)  # lane k: the key of pick k
        for k in range_constexpr(TOP_K):
            m = wave_umax(ks[0])
            hit = ks[0] == m
            ks = [hit.select(ks[i + 1], ks[i]) for i in range(KPL)] + [ks[KPL]]
            mv = write_lane_i32(m, k, mv)
        e = ID_MASK - (mv & ID_MASK)
        logit = st.getf(mb("scores"), s * ROUTER_ROWS + fx.min(e, N_EXPERTS - 1))
        top = read_lane_i32(logit.bitcast(fx.Int32), 0).bitcast(fx.Float32)
        p = (lane < TOP_K).select(exp(logit - top), fx.Float32(0.0))
        tot = p
        for off in TOPK_SUM_OFFS:
            tot = xred(tot, off, lambda a, b: a + b)
        return e, p * rcp(tot)

    return dict(
        st=st,
        mb=mb,
        poll=poll,
        put=st.put,
        put2=st.put2,
        put_bf=st.put_bf,
        get=st.get,
        getf=st.getf,
        getf_many=st.getf_many,
        get2_many=st.get2_many,
        hint_wait=st.hint_wait,
        block_sum=st.block_sum,
        block_sums=block_sums,
        unit_fp8=st.unit_fp8,
        unit_bf16=st.unit_bf16,
        unit_f8f8=st.unit_f8f8,
        mma_units=st.mma_units,
        run_units=st.run_units,
        reduce_rows=st.reduce_rows,
        emit_out=st.emit_out,
        n_sel=st.n_sel,
        stage_x_pairs=st.stage_x_pairs,
        quant_scaled=st.quant_scaled,
        st_f8=st.st_f8,
        peer_reduce=st.peer_reduce,
        stamp=st.stamp,
        start=st.start,
        stage_x_gemma=stage_x_gemma,
        stage_moe_input=stage_moe_input,
        route_softmax=route_softmax,
    )
