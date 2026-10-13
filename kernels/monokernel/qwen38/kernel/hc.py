# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The hyper-connection read of the Qwen3.8 MonoKernel: from the bundle X [S, hc * hidden] to the sublayer input
``m`` = mean_s(sigmoid(up(silu(down(Xn) / hc)))_s * Xn_s) and the inject logits ``r`` = inject(Xn), where Xn is
the per-stream RMSNorm (1 + w) of X.

Three stages per read:
- ``hcd``: the down | inject GEMV over K splits of 512 (inside one stream). A task stages its slice of X * (1 + w)
  (the per-stream rstd factors out of the K split) and publishes partial rows and the slice's sum of squares.
- ``hcr``: per sample and 64-row block, sums the partials scaled by each stream's rstd -> silu(bf16(d) / hc)
  for the low-rank rows, the inject logits, the rstds.
- ``hcu``: the up GEMV (K = low rank) for 32 columns of every stream, then the gated mean over the streams.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import bf2_f32, bf16_pair, bf16_round, ld_f32, lds_ld, lds_st, rsq, rsrc
from kernels.monokernel.qwen35.kernel.common import sigmoid, silu
from kernels.monokernel.qwen38.kernel.plan import HC_KS, HC_RB, THREADS, WAVES


@traced
def hc_stages(ctx, side, from_mailbox):
    """The read on side ``side`` ("a": before the mixer, from ``h_in``; "m": before the MoE, from mailbox x1)."""
    EPS = ctx["EPS"]
    G = ctx["G"]
    HC = ctx["HC"]
    HIDDEN = ctx["HIDDEN"]
    LOWRANK = ctx["LOWRANK"]
    KSPLIT = ctx["KSPLIT"]
    RR = ctx["RR"]
    S = ctx["S"]
    block_sums = ctx["block_sums"]
    h_in = ctx["h_in"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    put_bf = ctx["put_bf"]
    red = ctx["red"]
    reduce_rows = ctx["reduce_rows"]
    emit_out = ctx["emit_out"]
    run_units = ctx["run_units"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16_rows = ctx["unit_bf16_rows"]
    wave = ctx["wave"]
    xs = ctx["xs"]
    gamma = ctx[f"hc{side}_g"]
    w_down = ctx[f"hc{side}_down"]
    w_up = ctx[f"hc{side}_up"]
    w_inj = ctx[f"hc{side}_inj"]
    W4 = HC * HIDDEN
    SPLITS_PER_STREAM = HIDDEN // HC_KS
    DOWN_RG = LOWRANK // 16  # row groups of the down matrix; the next one is the inject rows
    r_h, r_g = rsrc(h_in), rsrc(gamma)

    def x_pairs(idx):
        """Bundle elements (2 i, 2 i + 1) for flat pair indices ``idx`` (``s * W4 / 2 + k / 2``) -> [(x0, x1)]."""
        if const_expr(from_mailbox):
            return [bf2_f32(v[0]) for v in poll([(mb("x1"), i, 1) for i in idx])]
        return [bf2_f32(fx.Int32(bo.buffer_load(r_h, i, vec_width=1, dtype=T.i32))) for i in idx]

    # ================================ hcd: down | inject partials over a 512-wide K split
    NPAIR = HC_KS // 2  # pairs of one sample's slice
    n_it = -(-S * NPAIR // THREADS)
    for tt in range(start(f"hcd_{side}"), RR // 32 * KSPLIT, G):
        tt = fx.Int32(tt)
        rp = tt // KSPLIT
        ks = tt % KSPLIT
        stamp(f"hcd_{side}", tt, 0)
        rg = rp * 2 + wave // 4
        is_down = rg < DOWN_RG
        is_inj = rg == DOWN_RG
        w_addr = is_down.select(w_down, w_inj)
        nrec = is_down.select(fx.Int32(LOWRANK * W4 * 2), is_inj.select(fx.Int32(HC * W4 * 2), fx.Int32(0)))
        r_w = bo.create_buffer_resource_from_addr(w_addr, num_records_bytes=nrec)
        rg_w = is_down.select(rg, fx.Int32(0))

        def u(c):
            kl = (wave % 4) * 2 + c
            return unit_bf16_rows(r_w, rg_w, ks * (HC_KS // 64) + kl, W4, (n_sel() * HC_KS + kl * 64) // 2)

        pre = [u(c) for c in range(2)]
        gpu.barrier()
        # this split's X * (1 + w) as bf16 in LDS, and its sum of squares per sample
        sqs = [fx.Float32(0.0) for _ in range(S)]
        for i in range_constexpr(n_it):
            p = fx.min(tid + i * THREADS, S * NPAIR - 1)
            live = tid + i * THREADS < S * NPAIR
            s = p // NPAIR
            j = (p % NPAIR) * 2
            x0, x1 = x_pairs([(s * W4 + ks * HC_KS + j) // 2])[0]
            g0, g1 = ld_f32(r_g, ks * HC_KS + j), ld_f32(r_g, ks * HC_KS + j + 1)
            if live:
                lds_st(xs, (s * HC_KS + j) // 2, bf16_pair(x0 * g0, x1 * g1))
            sq = live.select(x0 * x0 + x1 * x1, fx.Float32(0.0))
            for sm in range_constexpr(S):
                sqs[sm] = sqs[sm] + (s == sm).select(sq, fx.Float32(0.0))
        tots = block_sums(sqs)
        gpu.barrier()
        stamp(f"hcd_{side}", tt, 2)
        acc = run_units(u, 2, 2, pre)

        def emit(rl, n, v):
            row = rp * 32 + rl
            if row < LOWRANK + HC:
                put(mb(f"hcp_{side}"), (n * KSPLIT + ks) * RR + row, v)

        reduce_rows(2, acc, emit)
        if (rp == 0) & (tid < S):
            mine = fx.Float32(0.0)
            for sm in range_constexpr(S):
                mine = (tid == sm).select(tots[sm], mine)
            put(mb(f"hcss_{side}"), tid * KSPLIT + ks, mine)
        stamp(f"hcd_{side}", tt, 4)

    # ================================ hcr: per sample and 64-row block, the rstd-weighted sum of the splits
    NRB = -(-RR // HC_RB)
    KG = THREADS // HC_RB  # split groups per thread row
    for tt in range(start(f"hcr_{side}"), S * NRB, G):
        tt = fx.Int32(tt)
        s = tt // NRB
        rb = tt % NRB
        stamp(f"hcr_{side}", tt, 0)
        gpu.barrier()
        if tid < KSPLIT:
            lds_st(misc, tid, poll([(mb(f"hcss_{side}"), s * KSPLIT + tid, 1)])[0][0].bitcast(fx.Float32))
        gpu.barrier()
        rstds = []
        for st in range_constexpr(HC):
            ss = fx.Float32(0.0)
            for j in range_constexpr(SPLITS_PER_STREAM):
                ss = ss + lds_ld(misc, st * SPLITS_PER_STREAM + j)
            rstds.append(rsq(ss * (1.0 / HIDDEN) + EPS))
        row = rb * HC_RB + tid % HC_RB
        row_c = fx.min(row, LOWRANK + HC - 1)
        kg = tid // HC_RB
        NI = -(-KSPLIT // KG)
        got = poll(
            [(mb(f"hcp_{side}"), (s * KSPLIT + fx.min(kg + i * KG, KSPLIT - 1)) * RR + row_c, 1) for i in range(NI)]
        )
        part = fx.Float32(0.0)
        for i in range_constexpr(NI):
            k_ = kg + i * KG
            rs = rstds[0]
            for st in range_constexpr(1, HC):
                rs = (k_ >= st * SPLITS_PER_STREAM).select(rstds[st], rs)
            part = part + (k_ < KSPLIT).select(got[i][0].bitcast(fx.Float32) * rs, fx.Float32(0.0))
        lds_st(red, tid, part)
        gpu.barrier()
        stamp(f"hcr_{side}", tt, 2)
        if tid < HC_RB:
            d = fx.Float32(0.0)
            for k_ in range_constexpr(KG):
                d = d + lds_ld(red, k_ * HC_RB + tid)
            d = bf16_round(d)
            if row < LOWRANK:
                put(mb(f"hsd_{side}"), s * LOWRANK + row, bf16_round(silu(bf16_round(d * (1.0 / HC)))))
            elif row < LOWRANK + HC:
                put(mb(f"hr_{side}"), s * HC + row - LOWRANK, d)
        if (rb == 0) & (tid < HC):
            mine = rstds[0]
            for st in range_constexpr(1, HC):
                mine = (tid == st).select(rstds[st], mine)
            put(mb(f"hrs_{side}"), s * HC + tid, mine)
        stamp(f"hcr_{side}", tt, 4)

    # ================================ hcu: up GEMV for 32 columns of every stream, gated mean -> m
    UP_NKC = LOWRANK // 64
    r_up = rsrc(w_up)
    NP_SD = S * LOWRANK // 2
    for t in range(start(f"hcu_{side}"), HIDDEN // 32, G):
        t = fx.Int32(t)
        stamp(f"hcu_{side}", t, 0)
        # wave w: stream w // 2, columns 32 t + 16 (w % 2) ..
        rg_u = ((wave // 2) * HIDDEN + t * 32 + (wave % 2) * 16) // 16

        def u_up(kc):
            return unit_bf16_rows(r_up, rg_u, kc, LOWRANK, (n_sel() * LOWRANK + kc * 64) // 2)

        pre = [u_up(kc) for kc in range(UP_NKC)]
        # the epilogue's bundle and gamma values do not depend on the reduce: load them before waiting
        n_e = fx.min(tid // 16, S - 1)
        cp_e = (tid % 16) * 2
        col_e = t * 32 + cp_e
        xv = x_pairs([(n_e * W4 + st * HIDDEN + col_e) // 2 for st in range(HC)])
        gv = [[ld_f32(r_g, st * HIDDEN + col_e + e) for e in range(2)] for st in range(HC)]
        gpu.barrier()
        for i in range_constexpr(-(-NP_SD // THREADS)):
            p = fx.min(tid + i * THREADS, NP_SD - 1)
            v = poll([(mb(f"hsd_{side}"), 2 * p, 2)])[0]
            if tid + i * THREADS < NP_SD:
                lds_st(xs, p, bf16_pair(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)))
        gpu.barrier()
        stamp(f"hcu_{side}", t, 2)
        acc = run_units(u_up, UP_NKC, UP_NKC, pre)
        reduce_rows(WAVES, acc, emit_out(WAVES * 16))
        gpu.barrier()
        stamp(f"hcu_{side}", t, 3)
        if tid < S * 16:
            n = n_e
            cp = cp_e
            col = col_e
            rs = poll([(mb(f"hrs_{side}"), n * HC + st, 1) for st in range(HC)])
            m = [fx.Float32(0.0), fx.Float32(0.0)]
            for st in range_constexpr(HC):
                rstd = rs[st][0].bitcast(fx.Float32)
                for e in range_constexpr(2):
                    xn = bf16_round(xv[st][e] * rstd * gv[st][e])
                    u_ = bf16_round(lds_ld(outs, n * WAVES * 16 + st * 32 + cp + e))
                    m[e] = m[e] + sigmoid(u_) * xn
            put_bf(mb(f"m_{side}"), n * HIDDEN + col, [bf16_round(m[0] * (1.0 / HC)), bf16_round(m[1] * (1.0 / HC))])
        stamp(f"hcu_{side}", t, 4)
    return {}
