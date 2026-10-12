# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The MoE of the Qwen3.8 MonoKernel on the sublayer input ``m_m``: router (512 experts + the shared-expert gate,
row-major BF16), W8A8 routed experts (softmax top-k), the BF16 shared expert, and the MLP-side hyper-connection
combine into ``x_out``. Routed up/gate and down follow the Qwen3.5 kernel (ATOM's shuffled FP8 bank)."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import (
    bf2_f32,
    bf16_pair,
    bf16_round,
    fp8_roundtrip,
    ld_f32,
    lds_ld,
    lds_st,
    rsrc,
    uniform,
    uniform_f32,
    xshfl,
)
from kernels.monokernel.qwen35.kernel.common import quant_floor, sigmoid, silu
from kernels.monokernel.qwen38.kernel.plan import MOE_MISC, SH_TILE, UG8, WAVES


@traced
def ffn_stages(ctx):
    """Router + MoE input, shared expert up/gate, routed up/gate, down + combine."""
    DN_TILE = ctx["DN_TILE"]
    G = ctx["G"]
    HC = ctx["HC"]
    HIDDEN = ctx["HIDDEN"]
    INTER = ctx["INTER"]
    N_EXPERTS = ctx["N_EXPERTS"]
    ROUTER_ROWS = ctx["ROUTER_ROWS"]
    S = ctx["S"]
    SH_INTER = ctx["SH_INTER"]
    TOP_K = ctx["TOP_K"]
    XQ_BLOCKS = ctx["XQ_BLOCKS"]
    dnw = ctx["dnw"]
    emit_out = ctx["emit_out"]
    get2_many = ctx["get2_many"]
    getf = ctx["getf"]
    hint_wait = ctx["hint_wait"]
    keys = ctx["keys"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    mma_units = ctx["mma_units"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    put = ctx["put"]
    put2 = ctx["put2"]
    put_bf = ctx["put_bf"]
    reduce_rows = ctx["reduce_rows"]
    route_softmax = ctx["route_softmax"]
    run_units = ctx["run_units"]
    s_dn = ctx["s_dn"]
    s_ug = ctx["s_ug"]
    st_f8 = ctx["st_f8"]
    stage_moe_input = ctx["stage_moe_input"]
    stage_x_pairs = ctx["stage_x_pairs"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16_rows = ctx["unit_bf16_rows"]
    unit_f8f8 = ctx["unit_f8f8"]
    w_dn = ctx["w_dn"]
    w_r = ctx["w_r"]
    w_sh_dn = ctx["w_sh_dn"]
    w_sh_ug = ctx["w_sh_ug"]
    w_ug = ctx["w_ug"]
    wave = ctx["wave"]
    x_out = ctx["x_out"]
    xs = ctx["xs"]
    NKC = HIDDEN // 64
    W4 = HC * HIDDEN

    def stage_m():
        stage_x_pairs("m_m", S * HIDDEN, lambda k: k)

    # ====== router logits (+ shared gate) over 16-row groups; tasks < XQ_BLOCKS also publish the FP8 MoE input
    assert NKC % WAVES == 0
    R_UPW = NKC // WAVES
    r_wr = bo.create_buffer_resource_from_addr(w_r, num_records_bytes=fx.Int32((N_EXPERTS + 1) * HIDDEN * 2))
    for t in range(start("router"), ROUTER_ROWS // 16, G):
        t = fx.Int32(t)
        stamp("router", t, 0)

        def u_r(c):
            kc = wave * R_UPW + c
            return unit_bf16_rows(r_wr, t, kc, HIDDEN, (n_sel() * HIDDEN + kc * 64) // 2)

        pre = [u_r(c) for c in range(R_UPW)]
        gpu.barrier()
        stage_m()
        gpu.barrier()
        stamp("router", t, 2)
        # block t of sample `wave`: the expert input is m, block-quantized to FP8
        if (t < XQ_BLOCKS) & (wave < S):
            xk = t * 128 + lane * 2
            a0, a1 = bf2_f32(lds_ld(xs, (wave * HIDDEN + xk) // 2).bitcast(fx.Int32))
            q0, q1, qs = quant_floor(a0, a1)
            w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
            w8n = xshfl(w8, 1)
            if lane % 2 == 0:
                put(mb("xq"), (wave * HIDDEN + xk) // 4, w8 | (w8n << 16))
            d0, d1 = fp8_roundtrip(q0, q1)
            if lane == 0:
                put(mb("xqs"), wave * XQ_BLOCKS + t, qs)
            bo.buffer_store(
                fx.Vector.from_elements([d0 * qs, d1 * qs], fx.Float32), rsrc(mb("xqd")), wave * HIDDEN + xk
            )
        acc = run_units(u_r, R_UPW, R_UPW, pre)
        reduce_rows(1, acc, emit_out(16))
        gpu.barrier()
        if tid < S * 16:
            n = tid // 16
            row = t * 16 + tid % 16
            if row <= N_EXPERTS:
                put(mb("scores"), n * ROUTER_ROWS + row, bf16_round(lds_ld(outs, tid)))  # bf16, as ATOM
        stamp("router", t, 4)

    # ====== shared expert up/gate (BF16 on m): SH_TILE intermediates per task -> mid_sh (bf16)
    r_shug = rsrc(w_sh_ug)
    SH_UPW = NKC // (WAVES // 2)
    for t in range(start("sh_ug"), SH_INTER // SH_TILE, G):
        t = fx.Int32(t)
        stamp("sh_ug", t, 0)
        # waves 0-3: gate rows 16 t .., waves 4-7: up rows SH_INTER + 16 t ..
        rg_s = (wave // 4) * (SH_INTER // 16) + t

        def u_s(c):
            kc = (wave % 4) * SH_UPW + c
            return unit_bf16_rows(r_shug, rg_s, kc, HIDDEN, (n_sel() * HIDDEN + kc * 64) // 2)

        pre = [u_s(c) for c in range(SH_UPW // 2)]
        gpu.barrier()
        stage_m()
        gpu.barrier()
        stamp("sh_ug", t, 2)
        acc = run_units(u_s, SH_UPW, SH_UPW // 2, pre)
        reduce_rows(2, acc, emit_out(32))
        gpu.barrier()
        if tid < S * SH_TILE // 2:
            n = tid // (SH_TILE // 2)
            r = (tid % (SH_TILE // 2)) * 2
            mid = []
            for e in range_constexpr(2):
                gt = bf16_round(lds_ld(outs, n * 32 + r + e))
                up = bf16_round(lds_ld(outs, n * 32 + 16 + r + e))
                mid.append(silu(gt) * up)
            put_bf(mb("mid_sh"), n * SH_INTER + t * SH_TILE + r, mid)
        stamp("sh_ug", t, 4)

    def dn_route():
        """Wave s routes sample s: slots 0..TOP_K-1 = the picks -> keys[s * TOP_K + slot], dnw[...]."""
        if wave < S:
            e, w = route_softmax(wave)
            if lane < TOP_K:
                lds_st(keys, wave * TOP_K + lane, e)
                lds_st(dnw, wave * TOP_K + lane, w)

    # ================================ routed expert up/gate + SiLU (W8A8)
    UG_NKC = HIDDEN // 64
    UG_W_BYTES = 2 * INTER * HIDDEN
    UG_S_BYTES = 2 * INTER // 128 * (HIDDEN // 128) * 4
    XB = HIDDEN // 128  # 128-k blocks of the MoE input
    UG8_UNITS = -(-XB // WAVES)
    UG_PER_SLOT = INTER // UG8
    N_UG_TASKS = TOP_K * UG_PER_SLOT

    def up_gate():
        """In its own scope: its unit lists must not join `down`'s scf.for-carried state."""
        XW = HIDDEN // 4
        u0 = fx.Int32(start("ug"))

        def ug8_units(w_rg, w_ln, s_rg, e, sample, live):
            rw = bo.create_buffer_resource_from_addr(
                w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES),
                num_records_bytes=live.select(fx.Int32(UG_W_BYTES), fx.Int32(0)),
            )
            rs = bo.create_buffer_resource_from_addr(
                s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES),
                num_records_bytes=live.select(fx.Int32(UG_S_BYTES), fx.Int32(0)),
            )
            sn = fx.Int32(sample)
            units = []
            for cc in range_constexpr(UG8_UNITS):
                # block cc * WAVES + wave; blocks past the input read block XB - 1 against zero weights
                blk = cc * WAVES + wave
                ok = blk < XB
                kb = fx.min(blk, XB - 1)
                kc = kb * 2
                wv = [
                    fx.Vector(bo.buffer_load(rw, ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4, vec_width=4, dtype=T.i32))
                    for j in range(2)
                ]
                sc = ok.select(ld_f32(rs, (s_rg * 16 // 128) * (HIDDEN // 128) + kb), fx.Float32(0.0))

                def coefficient(sc=sc, kb=kb, sn=sn):
                    return sc * lds_ld(misc, MOE_MISC + sn * XQ_BLOCKS + kb)

                units.append(("f8f8", wv, coefficient, sn * XW + kc * 16 + (lane // 16) * 4))
            return units

        def ug8_emit(c, slot, sample, live):
            if tid < UG8 // 2:
                r = tid * 2
                g0, g1 = lds_ld(outs, r), lds_ld(outs, r + 1)
                v0, v1 = lds_ld(outs, UG8 + r), lds_ld(outs, UG8 + r + 1)
                if live:
                    put2(mb("mid"), (sample * TOP_K + slot) * INTER + c * UG8 + r, silu(g0) * v0, silu(g1) * v1)
            if (c == 0) & (tid == 0) & live:
                put(mb("sel"), sample * TOP_K + slot, lds_ld(keys, sample * TOP_K + slot))
                put(mb("prob"), sample * TOP_K + slot, lds_ld(dnw, sample * TOP_K + slot))

        N_UG_ITEMS = S * N_UG_TASKS
        UG_ITEMS = (N_UG_ITEMS + G - 1) // G

        def ug8_item(k):
            w = u0 + k * G
            live = w < N_UG_ITEMS
            ww = fx.min(w, fx.Int32(N_UG_ITEMS - 1))
            uu = ww % N_UG_TASKS
            sample = ww // N_UG_TASKS
            c = uu % UG_PER_SLOT
            slot = uu // UG_PER_SLOT
            w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2
            w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (INTER // 16) + c // 2
            return live, uu, sample, c, slot, w_rg, w_ln, s_rg

        dn_route()
        gpu.barrier()
        items = [ug8_item(0)]
        lv, uu_, sm, c_, sl, wr, wl, sr = items[0]
        cur = ug8_units(wr, wl, sr, uniform(lds_ld(keys, sm * TOP_K + sl)), sm, lv)
        stage_moe_input(list(range(S)))
        gpu.barrier()
        for k in range_constexpr(UG_ITEMS):
            live, uu, sample, c, slot, w_rg, w_ln, s_rg = items[k]
            stamp("ug", sample * N_UG_TASKS + uu, 0, pred=live)
            pre = cur
            if const_expr(k + 1 < UG_ITEMS):
                items.append(ug8_item(k + 1))
                lv, uu_, sm, c_, sl, wr, wl, sr = items[k + 1]
                cur = ug8_units(wr, wl, sr, uniform(lds_ld(keys, sm * TOP_K + sl)), sm, lv)
            reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16), count=1)
            gpu.barrier()
            ug8_emit(c, slot, sample, live)
            stamp("ug", sample * N_UG_TASKS + uu, 4, pred=live)

    up_gate()

    # =============== expert down (routed W8A8 + shared BF16) + the MLP-side combine -> x_out
    DN_NKC = INTER // 64
    assert DN_TILE % 16 == 0 and HIDDEN % DN_TILE == 0
    DN_R = DN_TILE // 16
    DN_WPR = WAVES // DN_R
    DN_UNITS_PER_SLOT = INTER // 128
    DN_NU = S * TOP_K * DN_UNITS_PER_SLOT
    DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
    DN_BLK = S * TOP_K * INTER // 128
    DN_W_BYTES = HIDDEN * INTER
    DN_S_BYTES = HIDDEN // 128 * (INTER // 128) * 4
    DN_BATCH = 9
    SH_NKC = SH_INTER // 64
    SH_UPW = -(-SH_NKC // DN_WPR)
    XSH = S * TOP_K * INTER // 4  # LDS word of the shared-expert activation (after the routed FP8 mids)
    NOUT = S * DN_TILE
    for t in range(start("down"), HIDDEN // DN_TILE, G):
        t = fx.Int32(t)
        stamp("down", t, 0)
        gpu.barrier()
        gu = wave // DN_WPR
        dn_rg = t * DN_TILE // 16
        dn_off = t * DN_TILE % 16
        dn_lr = gu * 16 + lane % 16 - dn_off
        dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

        def u_dn(cc):
            qu = (wave % DN_WPR) * DN_UPW + cc
            live = qu < DN_NU
            r = fx.min(qu, DN_NU - 1)
            s_q = r // (TOP_K * DN_UNITS_PER_SLOT)
            slot_q = (r // DN_UNITS_PER_SLOT) % TOP_K
            kc = r % DN_UNITS_PER_SLOT
            q = (s_q * TOP_K + slot_q) * DN_UNITS_PER_SLOT + kc
            e = uniform(lds_ld(keys, s_q * TOP_K + slot_q))
            masked = DN_NU % DN_WPR != 0
            wb = bo.create_buffer_resource_from_addr(
                w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                num_records_bytes=live.select(fx.Int32(DN_W_BYTES), fx.Int32(0)) if masked else None,
            )
            sb = bo.create_buffer_resource_from_addr(
                s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                num_records_bytes=live.select(fx.Int32(DN_S_BYTES), fx.Int32(0)) if masked else None,
            )

            def coef():  # mid block scale * route weight, only in this sample's column
                return (lane % 16 == s_q).select(uniform_f32(lds_ld(misc, q)), fx.Float32(0.0))

            return unit_f8f8(wb, sb, dn_rg + gu, kc * 2, DN_NKC, INTER, q * 32, coef, dn_ln)

        def u_sh(cc):
            # chunks past K read the last one against zero weights
            kc_raw = (wave % DN_WPR) * SH_UPW + cc
            ok = kc_raw < SH_NKC
            kc = fx.min(kc_raw, SH_NKC - 1)
            rb = bo.create_buffer_resource_from_addr(
                w_sh_dn, num_records_bytes=ok.select(fx.Int32(HIDDEN * SH_INTER * 2), fx.Int32(0))
            )
            return unit_bf16_rows(rb, dn_rg + gu, kc, SH_INTER, XSH + (n_sel() * SH_INTER + kc * 64) // 2, dn_ln)

        pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
        pre_sh = [u_sh(cc) for cc in range(SH_UPW)]
        hint_wait(0, None, mark=("down", t))
        mids = get2_many(
            [
                (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                for b in range((DN_BLK + WAVES - 1) // WAVES)
            ]
        )
        stamp("down", t, 2)
        for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
            blk = wave + b * WAVES
            if blk < DN_BLK:
                q0, q1, qs = quant_floor(mids[b][0], mids[b][1])
                st_f8(blk * 128 + lane * 2, q0, q1, natural=True)
                if lane == 0:
                    lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
        # the shared expert's activation (bf16 pairs) after the routed FP8 bytes
        NP_SH = S * SH_INTER // 2
        for i in range_constexpr(-(-NP_SH // 512)):
            p = fx.min(tid + i * 512, NP_SH - 1)
            v = poll([(mb("mid_sh"), p, 1)])[0][0]
            if tid + i * 512 < NP_SH:
                lds_st(xs, XSH + p, v.bitcast(fx.Float32))
        gpu.barrier()
        acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)
        acc_sh = mma_units([fx.Float32(0.0) for _ in range(4)], pre_sh)

        def emit_dn(base):
            def f(rl, n, v):
                if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                    lds_st(outs, base + n * DN_TILE + rl - dn_off, v)

            return f

        reduce_rows(DN_R, acc, emit_dn(0))
        gpu.barrier()
        reduce_rows(DN_R, acc_sh, emit_dn(NOUT))
        stamp("down", t, 3)
        gpu.barrier()
        # thread (sample n, stream st, row pair): y = bf16(bf16(shared) * bf16(sigmoid(gate))) + bf16(routed)
        if tid < S * HC * DN_TILE // 2:
            n = tid // (HC * DN_TILE // 2)
            st = (tid // (DN_TILE // 2)) % HC
            rp = (tid % (DN_TILE // 2)) * 2
            sg = bf16_round(sigmoid(getf(mb("scores"), n * ROUTER_ROWS + N_EXPERTS)))
            r = poll([(mb("hr_m"), n * HC + st, 1)])[0][0].bitcast(fx.Float32)
            inj = 2.0 * sigmoid(bf16_round(r * (1.0 / HC)))
            idx = n * W4 + st * HIDDEN + t * DN_TILE + rp
            xv = bf2_f32(poll([(mb("x1"), idx // 2, 1)])[0][0])
            outv = []
            for e in range_constexpr(2):
                routed = bf16_round(lds_ld(outs, n * DN_TILE + rp + e))
                shared = bf16_round(lds_ld(outs, NOUT + n * DN_TILE + rp + e))
                y = bf16_round(bf16_round(shared * sg) + routed)
                outv.append(xv[e] + y * inj)
            bo.buffer_store(bf16_pair(outv[0], outv[1]), rsrc(x_out), idx // 2)
        gpu.barrier()
        stamp("down", t, 4)
    return {}
