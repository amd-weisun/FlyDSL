# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The MoE FFN of the Qwen3.5 MonoKernel: router (+ shared-expert gate), W8A8 expert up/gate and
down, and the FFN-side TP reduce. Adapted from DeepSeek-V4's ``ffn.py``: softmax top-k routing, the
shared expert as a sigmoid-gated slot of the bank, plain SwiGLU."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import (
    bf2_f32,
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
from kernels.monokernel.qwen35.kernel.plan import MOE_MISC, ROUTER_TILE, UG8, WAVES, router_spt


@traced
def ffn_stages(ctx):
    """5-7. Router + MoE input, expert up/gate + SiLU, expert down + route weighting + TP reduce."""
    DN_TILE = ctx["DN_TILE"]
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    INTER = ctx["INTER"]
    MOE_SLOTS = ctx["MOE_SLOTS"]
    N_DN_TILES = ctx["N_DN_TILES"]
    N_EXPERTS = ctx["N_EXPERTS"]
    N_ROUTER = ctx["N_ROUTER"]
    N_UG_TASKS = ctx["N_UG_TASKS"]
    PUBLISH_BLOCKS = ctx["PUBLISH_BLOCKS"]
    ROUTER_ROWS = ctx["ROUTER_ROWS"]
    S = ctx["S"]
    SHARED_EXPERT = ctx["SHARED_EXPERT"]
    TOP_K = ctx["TOP_K"]
    UG_PER_SLOT = ctx["UG_PER_SLOT"]
    XQ_BLOCKS = ctx["XQ_BLOCKS"]
    XQ_WAVES = ctx["XQ_WAVES"]
    dnw = ctx["dnw"]
    emit_out = ctx["emit_out"]
    g_post1 = ctx["g_post1"]
    getf = ctx["getf"]
    get2_many = ctx["get2_many"]
    hint_wait = ctx["hint_wait"]
    keys = ctx["keys"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    mma_units = ctx["mma_units"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    peer_reduce = ctx["peer_reduce"]
    poll = ctx["poll"]
    put = ctx["put"]
    put2 = ctx["put2"]
    red = ctx["red"]
    reduce_rows = ctx["reduce_rows"]
    route_softmax = ctx["route_softmax"]
    run_units = ctx["run_units"]
    s_dn = ctx["s_dn"]
    s_ug = ctx["s_ug"]
    st_f8 = ctx["st_f8"]
    stage_moe_input = ctx["stage_moe_input"]
    stage_x_gemma = ctx["stage_x_gemma"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_f8f8 = ctx["unit_f8f8"]
    w_dn = ctx["w_dn"]
    w_r = ctx["w_r"]
    w_ug = ctx["w_ug"]
    wave = ctx["wave"]
    x_out = ctx["x_out"]

    # ====== 5. post-attn RMSNorm -> router logits (+ shared gate) + this task's FP8 MoE input blocks
    r_wr = rsrc(w_r)
    R_NKC = HIDDEN // 64
    SPT = router_spt(S, N_EXPERTS)
    # tasks cover ceil(S / SPT) sample groups; a group's samples past S read sample S - 1 and publish nothing
    for tt in range(start("router"), -(-S // SPT) * N_ROUTER, G):
        tt = fx.Int32(tt)
        t = tt % N_ROUTER
        rs0 = (tt // N_ROUTER) * SPT
        stamp("router", tt, 0)
        # K-fold: rows / columns 0..7 take the first K half, 8..15 the second
        r_sub = t * ROUTER_TILE % 16
        r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
        R_CPW = R_NKC // WAVES // 2
        r_fold = (lane % 16) // ROUTER_TILE
        r_ns = fx.min(lane % 8, fx.Int32(SPT - 1))

        def u_r(c):
            kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
            return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

        pre = [u_r(c) for c in range(R_CPW)]
        hint_wait(0, None, mark=("router", tt))
        x_blk = wave * N_ROUTER + t
        xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 128 + lane * 2
        x_ok = (wave < XQ_WAVES) & (x_blk < PUBLISH_BLOCKS)
        r_gp = rsrc(g_post1)
        xg = (ld_f32(r_gp, xk), ld_f32(r_gp, xk + 1))
        xa = []

        def ld_a(sks):
            # sks = (local sample s, k) pairs; local sample s is sample rs0 + s
            n4 = len(sks)
            specs = [(mb("a"), (fx.min(rs0 + s, S - 1) * HIDDEN + k) // 2, 2) for s, k in sks]
            specs += [(mb("a"), (fx.min(rs0 + s, S - 1) * HIDDEN + xk) // 2, 1) for s in range(SPT)]
            v = poll(specs, batch=len(specs))
            for s in range_constexpr(SPT):
                xa.append(bf2_f32(v[n4 + s][0]))
            return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:n4]]

        rstds = stage_x_gemma(ld_a, HIDDEN, g_post1, count=SPT)[0]
        stamp("router", tt, 2)
        for s_l in range_constexpr(SPT):
            if x_ok & (rs0 + s_l < S):
                x_s = rs0 + s_l
                a0, a1 = xa[s_l]
                # the expert input is the bf16 normed activation, block-quantized to FP8
                v0, v1 = bf16_round(a0 * rstds[s_l] * xg[0]), bf16_round(a1 * rstds[s_l] * xg[1])
                q0, q1, qs = quant_floor(v0, v1)
                w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                w8n = xshfl(w8, 1)
                if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                    put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                d0, d1 = fp8_roundtrip(q0, q1)
                if lane == 0:
                    put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                bo.buffer_store(
                    fx.Vector.from_elements([d0 * qs, d1 * qs], fx.Float32), rsrc(mb("xqd")), x_s * HIDDEN + xk
                )
        gpu.barrier()
        acc = run_units(u_r, R_CPW, R_CPW, pre)
        fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
        gpu.barrier()
        stamp("router", tt, 3)
        if (tid < ROUTER_TILE * SPT) & (rs0 + tid // ROUTER_TILE < S):
            r = tid % ROUTER_TILE
            n = tid // ROUTER_TILE
            logit = fx.Float32(0.0)
            for w in range_constexpr(WAVES):
                for f in range_constexpr(2):
                    m = f * ROUTER_TILE + r
                    logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
            put(mb("scores"), (rs0 + n) * ROUTER_ROWS + t * ROUTER_TILE + r, bf16_round(logit))  # bf16, as ATOM
        stamp("router", tt, 4)

    def dn_route():
        """Wave s routes sample s: slot 0 the shared expert (weight sigmoid(gate)), slots 1..TOP_K the
        picks -> keys[s * MOE_SLOTS + slot], dnw[...]; the scores must have landed."""
        if wave < S:
            e, w = route_softmax(wave)
            sg = sigmoid(getf(mb("scores"), wave * ROUTER_ROWS + N_EXPERTS))
            if lane < MOE_SLOTS:
                q = wave * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                lds_st(dnw, q, (lane == TOP_K).select(sg, w))

    # ================================ 6. expert up/gate + SiLU (W8A8)
    UG_NKC = HIDDEN // 64
    UG_W_BYTES = 2 * INTER * HIDDEN
    UG_S_BYTES = 2 * INTER // 128 * (HIDDEN // 128) * 4

    def up_gate():
        """In its own scope: its unit lists must not join `down`'s scf.for-carried state."""
        UG8_UNITS = (HIDDEN // 128) // WAVES
        XW = HIDDEN // 4
        u0 = fx.Int32(start("ug"))

        def ug8_units(c, w_rg, w_ln, s_rg, e, sample, live=None):
            nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
            ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
            rw = bo.create_buffer_resource_from_addr(w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw)
            rs = bo.create_buffer_resource_from_addr(s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns)
            sn = n_sel() if sample is None else fx.Int32(sample)
            units = []
            for cc in range_constexpr(UG8_UNITS):
                kc = (wave * UG8_UNITS + cc) * 2
                wv = [
                    fx.Vector(bo.buffer_load(rw, ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4, vec_width=4, dtype=T.i32))
                    for j in range(2)
                ]
                sc = ld_f32(rs, (s_rg * 16 // 128) * (HIDDEN // 128) + kc // 2)

                def coefficient(sc=sc, kb=kc // 2, sn=sn):
                    return sc * lds_ld(misc, MOE_MISC + sn * XQ_BLOCKS + kb)

                units.append(("f8f8", wv, coefficient, sn * XW + kc * 16 + (lane // 16) * 4))
            return units

        def ug8_emit(c, slot, sample, shared, live):
            if tid < (S if shared else 1) * UG8 // 2:
                n = tid // (UG8 // 2)
                r = (tid % (UG8 // 2)) * 2
                g0, g1 = lds_ld(outs, n * 16 + r), lds_ld(outs, n * 16 + r + 1)
                v0, v1 = lds_ld(outs, n * 16 + UG8 + r), lds_ld(outs, n * 16 + UG8 + r + 1)
                sn = n if shared else fx.Int32(sample)
                sl = fx.Int32(0) if shared else slot
                if live:
                    put2(mb("mid"), (sn * MOE_SLOTS + sl) * INTER + c * UG8 + r, silu(g0) * v0, silu(g1) * v1)
            if (c == 0) & (tid < S if shared else tid == 0):
                sn = tid if shared else fx.Int32(sample)
                sl = fx.Int32(0) if shared else slot
                if live:
                    put(mb("sel"), sn * MOE_SLOTS + sl, lds_ld(keys, sn * MOE_SLOTS + sl))
                    put(mb("prob"), sn * MOE_SLOTS + sl, lds_ld(dnw, sn * MOE_SLOTS + sl))

        def ug8_tile(u):
            """Tile ``u`` (clamped; ``live`` masks a dead one): (live, uu, c, slot, has_sh, w_rg, w_ln, s_rg).
            Tiles of pick slot 0 run as slot TOP_K and also carry the shared expert (slot 0)."""
            live = u < N_UG_TASKS
            uu = fx.min(u, fx.Int32(N_UG_TASKS - 1))
            c = uu % UG_PER_SLOT
            has_sh = uu < UG_PER_SLOT
            slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), uu // UG_PER_SLOT)
            w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2
            w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
            s_rg = (lane // 32) * (INTER // 16) + c // 2
            return live, uu, c, slot, has_sh, w_rg, w_ln, s_rg

        # a shared-expert tile serves every sample at once (B columns); routed work is items
        N_UG_ITEMS = S * N_UG_TASKS
        UG_ITEMS = (N_UG_ITEMS + G - 1) // G

        def ug8_item(k):
            w = u0 + k * G
            live = w < N_UG_ITEMS
            ww = fx.min(w, fx.Int32(N_UG_ITEMS - 1))
            uu = ww % N_UG_TASKS
            sample = ww // N_UG_TASKS
            _l, _u, c, slot, _h, w_rg, w_ln, s_rg = ug8_tile(uu)
            return live, uu, sample, c, slot, w_rg, w_ln, s_rg

        live0, uu0, c0, slot0, has_sh0, wr0, wl0, sr0 = ug8_tile(u0)
        shared_pre = ug8_units(c0, wr0, wl0, sr0, fx.Int32(SHARED_EXPERT), None, has_sh0 & live0)
        dn_route()
        gpu.barrier()
        items = [ug8_item(0)]
        lv, uu_, sm, c_, sl, wr, wl, sr = items[0]
        cur = ug8_units(c_, wr, wl, sr, uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv)
        stage_moe_input(list(range(S)))
        gpu.barrier()
        if has_sh0 & live0:
            reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre), emit_out(16))
            gpu.barrier()
            ug8_emit(c0, slot0, 0, True, live0)
        for k in range_constexpr(UG_ITEMS):
            live, uu, sample, c, slot, w_rg, w_ln, s_rg = items[k]
            stamp("ug", sample * N_UG_TASKS + uu, 0, pred=live)
            pre = cur
            if const_expr(k + 1 < UG_ITEMS):
                items.append(ug8_item(k + 1))
                lv, uu_, sm, c_, sl, wr, wl, sr = items[k + 1]
                cur = ug8_units(c_, wr, wl, sr, uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv)
            reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
            gpu.barrier()
            ug8_emit(c, slot, sample, False, live)
            stamp("ug", sample * N_UG_TASKS + uu, 4, pred=live)

    up_gate()

    # =============== 7. expert down + route weighting + MoE TP reduce -> x_out
    DN_NKC = INTER // 64
    assert DN_TILE % 16 == 0 and HIDDEN % DN_TILE == 0
    DN_R = DN_TILE // 16
    DN_WPR = WAVES // DN_R
    DN_UNITS_PER_SLOT = INTER // 128
    DN_NU = S * MOE_SLOTS * DN_UNITS_PER_SLOT
    DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
    DN_BLK = S * MOE_SLOTS * INTER // 128
    DN_W_BYTES = HIDDEN * INTER
    DN_S_BYTES = HIDDEN // 128 * (INTER // 128) * 4
    DN_BATCH = 9  # 128-k chunks per wave in flight / prefetched before the mid wait
    for t in range(start("down"), N_DN_TILES, G):
        t = fx.Int32(t)
        stamp("down", t, 0)
        gpu.barrier()
        gu = wave // DN_WPR
        dn_rg = t * DN_TILE // 16
        dn_off = t * DN_TILE % 16
        # rows outside the tile load their lane ^ 8 twin (same lines) and are dropped
        dn_lr = gu * 16 + lane % 16 - dn_off
        dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

        def u_dn(cc):
            qu = (wave % DN_WPR) * DN_UPW + cc
            live = qu < DN_NU
            r = fx.min(qu, DN_NU - 1)
            s_q = r // (MOE_SLOTS * DN_UNITS_PER_SLOT)
            slot_q = (r // DN_UNITS_PER_SLOT) % MOE_SLOTS
            kc = r % DN_UNITS_PER_SLOT
            q = (s_q * MOE_SLOTS + slot_q) * DN_UNITS_PER_SLOT + kc
            e = uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
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

        # the experts are known: stream their down weights while up/gate finishes
        pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
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
                st_f8(blk * 128 + lane * 2, q0, q1)
                if lane == 0:
                    lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
        gpu.barrier()
        acc = run_units(u_dn, DN_UPW, DN_BATCH, pre)

        def emit_dn(rl, n, v):
            if (rl >= dn_off) & (rl < dn_off + DN_TILE):
                lds_st(outs, n * DN_TILE + rl - dn_off, v)

        reduce_rows(DN_R, acc, emit_dn)
        stamp("down", t, 3)
        gpu.barrier()

        def store_x(s, row, v0, v1):
            bo.buffer_store(
                fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), rsrc(x_out), s * HIDDEN + row
            )

        peer_reduce("ffn", t, mb("a"), store_x, tile=DN_TILE)
        gpu.barrier()
        stamp("down", t, 4)
    return {}
