# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The QSA mixer of the Qwen3.8 MonoKernel: sparse gated GQA over the groups an indexer selects.

1. ``proj``: [gate | q | k | v] (ATOM's ``qkv_proj`` row order) then the indexer's [q heads | key], BF16;
2. ``qk``: per sample, the 24 query heads, 2 kv heads (K / V into the paged BF16 cache), 4 index heads (RMSNorm
   (1 + w), partial RoPE) and the index key (raw into the index cache; the token closing a group of 4 writes the
   group's compressed key: fp32 mean -> bf16 -> RMSNorm (1 + w) -> RoPE at the group's first position);
3. ``i_score``: score[g] = sum_h relu(iq_h . ck_g) / sqrt(index_dim) for every complete group (only past budget);
4. ``i_topk``: the top budget / 4 groups (exact radix select, ties to the lower group id), or all of them;
5. ``split``: attention over the selected groups' tokens plus the causal tail: 64-key tiles of 16 entries on the
   matrix cores (keys x query heads), V transposed through LDS; this launch's token from mailbox ``kvnew``;
6. ``merge``: the splits' softmax merge x sigmoid(gate) -> ``gated`` for the shared out_proj.
"""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import (
    bf16_round,
    exp,
    ld_bf16,
    ld_f32,
    lds_ld,
    lds_st,
    rcp,
    rsq,
    rsrc,
    uniform,
    wave_max,
    wave_sum,
    xshfl,
)
from kernels.monokernel.qwen35.kernel.common import sigmoid
from kernels.monokernel.qwen38.kernel.gdn import proj_stage_rows
from kernels.monokernel.qwen38.kernel.plan import BS, KT, THREADS, WAVES

NEG = -1.0e30
MIN_I32 = -(2**31)


@traced
def qsa_stages(ctx):
    EPS = ctx["EPS"]
    G = ctx["G"]
    HD = ctx["HD"]
    IH = ctx["IH"]
    IHD = ctx["IHD"]
    KVH = ctx["KVH"]
    MAXG = ctx["MAXG"]
    NQ = ctx["NQ"]
    NS = ctx["NS"]
    OUT_K = ctx["OUT_K"]
    RATIO = ctx["RATIO"]
    ROPE = ctx["ROPE"]
    S = ctx["S"]
    SEL = ctx["SEL"]
    block_sum = ctx["block_sum"]
    block_tables = ctx["block_tables"]
    bt_stride = ctx["bt_stride"]
    c_cache = ctx["c_cache"]
    ent = ctx["ent"]
    g_ik1 = ctx["g_ik1"]
    g_iq1 = ctx["g_iq1"]
    g_kn1 = ctx["g_kn1"]
    g_qn1 = ctx["g_qn1"]
    hist = ctx["hist"]
    i_cache = ctx["i_cache"]
    k_cache = ctx["k_cache"]
    kvt = ctx["kvt"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    positions = ctx["positions"]
    put = ctx["put"]
    put2 = ctx["put2"]
    put_bf = ctx["put_bf"]
    red = ctx["red"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    slot_mapping = ctx["slot_mapping"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    v_cache = ctx["v_cache"]
    vt = ctx["vt"]
    w_idx = ctx["w_idx"]
    w_qkv = ctx["w_qkv"]
    wave = ctx["wave"]
    xs = ctx["xs"]
    NQH = NQ // KVH  # query heads per kv head
    assert NQH <= 16 and HD % 64 == 0 and IHD <= THREADS and IH * IHD <= THREADS and IH == 4
    QR, GR = NQ * HD, NQ * HD  # gate rows, then query rows
    KVR = 2 * KVH * HD
    r_pos, r_bt, r_slot = rsrc(positions), rsrc(block_tables), rsrc(slot_mapping)
    r_kc, r_vc, r_ic, r_cc = rsrc(k_cache), rsrc(v_cache), rsrc(i_cache), rsrc(c_cache)
    r_cos, r_sin = rsrc(rope_cos), rsrc(rope_sin)
    HALF = ROPE // 2

    def ld_pos(s):
        return uniform(bo.buffer_load(r_pos, s, vec_width=1, dtype=T.i32))

    def ld_block(s, i):
        return fx.Int32(bo.buffer_load(r_bt, s * bt_stride + i, vec_width=1, dtype=T.i32))

    # ================================ 1. projection
    QKV_ROWS = GR + QR + KVR

    def proj_epilogue(s, row, v):
        v = bf16_round(v)
        if row < GR:
            put(mb("gate"), s * NQ * HD + row, v)
        if (row >= GR) & (row < GR + QR):
            put(mb("qr"), s * NQ * HD + row - GR, v)
        if (row >= GR + QR) & (row < QKV_ROWS):
            put(mb("kvr"), s * KVR + row - GR - QR, v)
        if (row >= QKV_ROWS) & (row < QKV_ROWS + IH * IHD):
            put(mb("iqr"), s * IH * IHD + row - QKV_ROWS, v)
        if row >= QKV_ROWS + IH * IHD:
            put(mb("rkr"), s * IHD + row - QKV_ROWS - IH * IHD, v)

    proj_stage_rows(ctx, "proj", [(w_qkv, QKV_ROWS), (w_idx, (IH + 1) * IHD)], "m_a", proj_epilogue)

    # ================================ 2. norms, RoPE, cache writes
    NQK = NQ + KVH + IH + 1

    def norm_rope(x, live, n_dim, r_gam, pos, d):
        """RMSNorm (1 + w) over ``n_dim`` (threads ``live``) then RoPE on the first ROPE dims at ``pos``, with
        ATOM's rounding (x cos and rot(x) sin each to bf16, then their sum)."""
        rstd = rsq(block_sum(live.select(x * x, fx.Float32(0.0))) * (1.0 / n_dim) + EPS)
        xn = bf16_round(x * rstd * ld_f32(r_gam, fx.min(d, n_dim - 1)))
        lds_st(outs, d, xn)
        gpu.barrier()
        partner = lds_ld(outs, (d < HALF).select(d + HALF, d - HALF))
        cs = ld_f32(r_cos, pos * HALF + d % HALF)
        sn = ld_f32(r_sin, pos * HALF + d % HALF)
        rot = (d < HALF).select(-partner, partner)
        y = (d < ROPE).select(bf16_round(bf16_round(xn * cs) + bf16_round(rot * sn)), xn)
        gpu.barrier()
        return y

    r_gq, r_gk, r_giq, r_gik = rsrc(g_qn1), rsrc(g_kn1), rsrc(g_iq1), rsrc(g_ik1)
    for tt in range(start("qk"), S * NQK, G):
        tt = fx.Int32(tt)
        s = tt // NQK
        j = tt % NQK
        stamp("qk", tt, 0)
        gpu.barrier()
        pos = ld_pos(s)
        d = fx.min(tid, HD - 1)
        slot = uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32))
        blk = fx.max(slot, 0) // BS
        off = slot % BS
        if j < NQ:
            live = tid < HD
            x = poll([(mb("qr"), s * NQ * HD + j * HD + d, 1)])[0][0].bitcast(fx.Float32)
            y = norm_rope(x, live, HD, r_gq, pos, d)
            if live:
                put(mb("q"), (s * NQ + j) * HD + d, y)
        if (j >= NQ) & (j < NQ + KVH):
            kh = j - NQ
            live = tid < HD
            got = poll([(mb("kvr"), s * KVR + kh * HD + d, 1), (mb("kvr"), s * KVR + KVH * HD + kh * HD + d, 1)])
            y = norm_rope(got[0][0].bitcast(fx.Float32), live, HD, r_gk, pos, d)
            v = got[1][0].bitcast(fx.Float32)
            if live & (slot >= 0):
                e_off = ((blk * BS + off) * KVH + kh) * HD + d
                bo.buffer_store(y.to(fx.BFloat16), r_kc, e_off)
                bo.buffer_store(v.to(fx.BFloat16), r_vc, e_off)
            if live:
                put(mb("kvnew"), (s * KVH + kh) * 2 * HD + d, y)
                put(mb("kvnew"), (s * KVH + kh) * 2 * HD + HD + d, v)
        if (j >= NQ + KVH) & (j < NQK - 1):
            ih = j - NQ - KVH
            live = tid < IHD
            dd = fx.min(tid, IHD - 1)
            x = poll([(mb("iqr"), s * IH * IHD + ih * IHD + dd, 1)])[0][0].bitcast(fx.Float32)
            y = norm_rope(x, live, IHD, r_giq, pos, dd)
            if live:
                put(mb("iq"), (s * IH + ih) * IHD + dd, y)
        if j == NQK - 1:
            live = tid < IHD
            dd = fx.min(tid, IHD - 1)
            x = poll([(mb("rkr"), s * IHD + dd, 1)])[0][0].bitcast(fx.Float32)
            if live & (slot >= 0):
                bo.buffer_store(x.to(fx.BFloat16), r_ic, (blk * BS + off) * IHD + dd)
            if (pos + 1) % RATIO == 0:
                # this token closes group pos // RATIO: mean of its raw keys (earlier ones from the cache)
                tot = x
                for i in range_constexpr(1, RATIO):
                    pi = fx.max(pos - i, 0)
                    tot = tot + ld_bf16(r_ic, (ld_block(s, pi // BS) * BS + pi % BS) * IHD + dd)
                pk = bf16_round(tot * (1.0 / RATIO))
                c = norm_rope(pk, live, IHD, r_gik, pos - (RATIO - 1), dd)
                grp = pos // RATIO
                if live & (slot >= 0):
                    cb = ld_block(s, grp // (BS // RATIO))
                    bo.buffer_store(c.to(fx.BFloat16), r_cc, (cb * (BS // RATIO) + grp % (BS // RATIO)) * IHD + dd)
                if live:
                    put(mb("cnew"), s * IHD + dd, c)
        stamp("qk", tt, 4)

    # ================================ 3. indexer scores (only when more groups than the budget), on the matrix
    # cores: a wave's 64 groups as 4 x 16 A rows (compressed keys straight from the cache), the index heads as B
    # columns; relu, then the heads summed across their lanes
    NSC = -(-MAXG // THREADS)
    ISCALE = IHD**-0.5
    IQROW = IHD + 8  # bf16 elements per index-query row in kvt (heads >= IH stay zero)
    CNB = 16 * IQROW  # kvt offset of this token's compressed key (bf16)
    v8b_ = fx.Vector.make_type(8, fx.BFloat16)
    for tt in range(start("i_score"), S * NSC, G):
        tt = fx.Int32(tt)
        s = tt // NSC
        sb = tt % NSC
        stamp("i_score", tt, 0)
        pos = ld_pos(s)
        n = (pos + 1) // RATIO
        if (n > SEL) & (sb * THREADS < n):
            gpu.barrier()
            hq = tid // IHD
            iqv = poll([(mb("iq"), s * IH * IHD + fx.min(tid, IH * IHD - 1), 1)])[0][0].bitcast(fx.Float32)
            fx.ptr_store((tid < IH * IHD).select(iqv, fx.Float32(0.0)).to(fx.BFloat16), kvt + (hq * IQROW + tid % IHD))
            for i in range_constexpr(-(-(16 - IH) * IHD // THREADS)):
                z = fx.min(tid + i * THREADS, (16 - IH) * IHD - 1)
                fx.ptr_store(fx.Float32(0.0).to(fx.BFloat16), kvt + ((IH + z // IHD) * IQROW + z % IHD))
            closing = (pos + 1) % RATIO == 0
            if closing & (tid < IHD):
                cv = poll([(mb("cnew"), s * IHD + tid, 1)])[0][0].bitcast(fx.Float32)
                fx.ptr_store(cv.to(fx.BFloat16), kvt + (CNB + tid))
            gpu.barrier()
            stamp("i_score", tt, 2)
            qbi = [
                fx.Vector(fx.ptr_load(kvt + ((lane % 16) * IQROW + 32 * kb + 8 * (lane // 16)), result_type=v8b_))
                for kb in range(IHD // 32)
            ]
            for rg in range_constexpr(4):
                g0 = sb * THREADS + wave * 64 + 16 * rg
                ga = fx.min(g0 + lane % 16, n - 1)  # this lane's A row (group)
                use_new = closing & (ga == n - 1)
                cb = ld_block(s, ga // (BS // RATIO))
                base = (cb * (BS // RATIO) + ga % (BS // RATIO)) * IHD
                c4 = fx.Vector.filled(4, 0.0, fx.Float32)
                for kb in range_constexpr(IHD // 32):
                    k0 = 32 * kb + 8 * (lane // 16)
                    w4 = fx.Vector(bo.buffer_load(r_cc, (base + k0) // 2, vec_width=4, dtype=T.i32))
                    nw = fx.Vector(fx.ptr_load(kvt + (CNB + k0), result_type=v8b_)).bitcast(fx.Int32)
                    w4 = fx.Vector.from_elements([use_new.select(nw[e], w4[e]) for e in range(4)], fx.Int32)
                    c4 = fx.Vector(
                        rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [w4.bitcast(fx.BFloat16), qbi[kb], c4])
                    )
                # C[group 4 (lane // 16) + e][head lane % 16]: relu, sum the IH head lanes
                for e in range_constexpr(4):
                    v = fx.max(c4[e], fx.Float32(0.0))
                    v = (lane % 16 < IH).select(v, fx.Float32(0.0))
                    for o_ in (1, 2):
                        v = v + xshfl(v, o_)
                    g = g0 + 4 * (lane // 16) + e
                    if (lane % 16 == 0) & (g < n):
                        put(mb("isc"), s * MAXG + g, v * ISCALE)
        stamp("i_score", tt, 4)

    # ================================ 4. top-SEL groups (exact radix select; ties to the lower group id)
    def block_excl_scan(v):
        """Exclusive prefix sum of a per-thread int32 over the block, and the block total."""
        x = fx.Float32(v)
        pre = fx.Float32(0.0)
        for sh in range_constexpr(6):
            o = 1 << sh
            p = xshfl(x, o)
            pre = ((lane & o) != 0).select(pre + p, pre)
            x = x + p
        if lane == 0:
            lds_st(red, wave, x)
        gpu.barrier()
        tot = fx.Float32(0.0)
        for i in range_constexpr(WAVES):
            t = lds_ld(red, i)
            pre = (fx.Int32(i) < wave).select(pre + t, pre)
            tot = tot + t
        gpu.barrier()
        return fx.Int32(pre), fx.Int32(tot)

    TRIPS = -(-MAXG // THREADS)
    for s in range(start("i_topk"), S, G):
        s = fx.Int32(s)
        stamp("i_topk", s, 0)
        pos = ld_pos(s)
        n = (pos + 1) // RATIO
        k_want = fx.min(n, fx.Int32(SEL))
        if n <= SEL:
            for i in range_constexpr(-(-SEL // THREADS)):
                e = tid + i * THREADS
                if e < n:
                    put(mb("gsel"), s * SEL + e, e)
        else:
            keys = []
            got = poll([(mb("isc"), s * MAXG + fx.min(tid * TRIPS + j, n - 1), 1) for j in range(TRIPS)])
            for j in range_constexpr(TRIPS):
                kb = got[j][0]
                keys.append((kb >= 0).select(kb ^ fx.Int32(MIN_I32), ~kb))
            stamp("i_topk", s, 2)
            pfx = fx.Int32(0)
            gt = fx.Int32(0)
            for dg in range_constexpr(4):
                sh = 24 - 8 * dg
                mk = (~((1 << (sh + 8)) - 1)) & 0xFFFFFFFF
                hi = fx.Int32(mk - (1 << 32) if mk >= (1 << 31) else mk)
                if tid < 256:
                    lds_st(hist, tid, fx.Int32(0))
                gpu.barrier()
                for j in range_constexpr(TRIPS):
                    c = tid * TRIPS + j
                    u = keys[j]
                    if (c < n) & (((u ^ pfx) & hi) == 0):
                        fx.atomic_add(hist + ((u >> sh) & 255), fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup)
                gpu.barrier()
                need = k_want - gt
                bn = 255 - fx.min(tid, 255)
                cnt = (tid < 256).select(lds_ld(hist, bn), fx.Int32(0))
                above, _t = block_excl_scan(cnt)
                if (tid < 256) & (above < need) & ((above + cnt) >= need):
                    lds_st(hist, 256, bn)
                    lds_st(hist, 257, above)
                gpu.barrier()
                pfx = pfx | (lds_ld(hist, 256) << sh)
                gt = gt + lds_ld(hist, 257)
                gpu.barrier()
            thr = pfx ^ MIN_I32
            # compaction in group order (thread order, then j): above the threshold to [0, gt), the first ties
            # to [gt, k_want); one scan per class over the per-thread counts
            ca = fx.Int32(0)
            ct = fx.Int32(0)
            for j in range_constexpr(TRIPS):
                c = tid * TRIPS + j
                sk = keys[j] ^ MIN_I32
                ca = ca + ((c < n) & (sk > thr)).select(fx.Int32(1), fx.Int32(0))
                ct = ct + ((c < n) & (sk == thr)).select(fx.Int32(1), fx.Int32(0))
            pa, _ta = block_excl_scan(ca)
            pt, _tt = block_excl_scan(ct)
            for j in range_constexpr(TRIPS):
                c = tid * TRIPS + j
                sk = keys[j] ^ MIN_I32
                is_a = (c < n) & (sk > thr)
                is_t = (c < n) & (sk == thr)
                if is_a:
                    put(mb("gsel"), s * SEL + pa, c)
                if is_t & (gt + pt < k_want):
                    put(mb("gsel"), s * SEL + gt + pt, c)
                pa = pa + is_a.select(fx.Int32(1), fx.Int32(0))
                pt = pt + is_t.select(fx.Int32(1), fx.Int32(0))
        if tid == 0:
            put(mb("gcnt"), s, k_want)
        stamp("i_topk", s, 4)

    # ================================ 5. split attention on the matrix cores
    EPT = KT // RATIO  # entries (groups) per tile
    max_tiles = -(-(SEL + 1) // EPT)
    MAX_SPAN = -(-max_tiles // NS)
    QROWE = HD + 8
    VROWE = KT + 8
    XP = 0  # P [16 heads][KT] f32
    XC = XP + 16 * KT  # per-head rescale [16] | this token's weight [16]
    XN = XC + 32  # this token's K then V
    WRG = wave % 4
    WKH = wave // 4
    KSTEPS = HD // 32 // 2
    NG = HD // 16 // WAVES
    assert NG == 2
    g4 = lane // 16
    SCALE = HD**-0.5
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v4f = fx.Vector.make_type(4, fx.Float32)
    v4z = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma(a, b, c):
        return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))

    def key_token(e_val, r, pos):
        """(token, live) of key r of entry value ``e_val`` (a group id, -1 = none); this launch's token is not
        live (it comes from kvnew)."""
        tok = e_val * RATIO + r
        return tok, (e_val >= 0) & (tok <= pos) & (tok != pos)

    def kv_row(s, tok, kvh):
        """Element offset of token ``tok``'s row of kv head ``kvh`` (NHD paged cache)."""
        b = ld_block(s, tok // BS)
        return ((b * BS + tok % BS) * KVH + kvh) * HD

    for tt in range(start("split"), S * KVH * NS, G):
        tt = fx.Int32(tt)
        s = tt // (KVH * NS)
        kvh = (tt // NS) % KVH
        js = tt % NS
        stamp("split", tt, 0)
        gpu.barrier()
        pos = ld_pos(s)
        n = (pos + 1) // RATIO
        cnt = uniform(poll([(mb("gcnt"), s, 1)])[0][0])
        n_ent = cnt + 1  # + the causal tail (group n, tokens n * RATIO .. pos)
        ntile = (n_ent + EPT - 1) // EPT
        span = (ntile + NS - 1) // NS
        t0 = js * span
        ntl = fx.max(fx.min(t0 + span, ntile) - t0, 0)
        e0 = t0 * EPT
        # this split's entries -> LDS: group ids, the tail as group n, -1 past the end
        own = fx.Int32(0)
        for i in range_constexpr(-(-MAX_SPAN * EPT // THREADS)):
            q = tid + i * THREADS
            e = e0 + q
            if q < MAX_SPAN * EPT:
                gv = fx.Int32(-1)
                if (q < ntl * EPT) & (e < cnt):
                    gv = poll([(mb("gsel"), s * SEL + e, 1)])[0][0]
                gv = ((q < ntl * EPT) & (e == cnt)).select(n, gv)
                lds_st(ent, q, gv)
                # this launch's token belongs to this split: its group was selected, or it is in the tail
                mine = ((gv == pos // RATIO) & (gv >= 0)).select(fx.Int32(1), fx.Int32(0))
                own = own + mine
        own_p = block_sum(fx.Float32(own)) > 0.5
        # q -> kvt as bf16 [16 rows][QROWE]; rows >= NQH are zero, and so are P rows >= NQH
        for i in range_constexpr(-(-16 * HD // THREADS)):
            idx = tid + i * THREADS
            hh = idx // HD
            qv = poll([(mb("q"), (s * NQ + kvh * NQH + fx.min(hh, NQH - 1)) * HD + idx % HD, 1)])[0][0]
            val = (hh < NQH).select(qv.bitcast(fx.Float32), fx.Float32(0.0))
            fx.ptr_store(val.to(fx.BFloat16), kvt + (hh * QROWE + idx % HD))
        for i in range_constexpr(-(-(16 - NQH) * KT // THREADS)):
            lds_st(xs, XP + NQH * KT + fx.min(tid + i * THREADS, (16 - NQH) * KT - 1), fx.Float32(0.0))
        gpu.barrier()
        stamp("split", tt, 2)
        qb = [
            fx.Vector(fx.ptr_load(kvt + ((lane % 16) * QROWE + (WKH * KSTEPS + c) * 32 + 8 * g4), result_type=v8b))
            for c in range(KSTEPS)
        ]
        m0 = fx.Float32(NEG)
        l0 = fx.Float32(0.0)
        m1 = fx.Float32(NEG)
        l1 = fx.Float32(0.0)
        a0 = fx.Float32(0.0)
        a1 = fx.Float32(0.0)
        a2 = fx.Float32(0.0)
        a3 = fx.Float32(0.0)
        a4 = fx.Float32(0.0)
        a5 = fx.Float32(0.0)
        a6 = fx.Float32(0.0)
        a7 = fx.Float32(0.0)
        for it in range(0, ntl):
            # S^T partial: keys of row group WRG x heads, half WKH of the head dim
            kq = 16 * WRG + lane % 16
            tok, ok = key_token(lds_ld(ent, it * EPT + kq // RATIO), kq % RATIO, pos)
            row = kv_row(s, ok.select(tok, fx.Int32(0)), kvh)
            c4 = v4z
            for c in range_constexpr(KSTEPS):
                ks = WKH * KSTEPS + c
                w4 = fx.Vector(bo.buffer_load(r_kc, (row + 32 * ks + 8 * g4) // 2, vec_width=4, dtype=T.i32))
                c4 = mfma(w4.bitcast(fx.BFloat16), qb[c], c4)
            fx.ptr_store(c4, red + (wave * 64 + lane) * 4)
            # V rows of the tile -> vt [dim][key] (bf16): 8 dims per load
            for i in range_constexpr(KT * (HD // 8) // THREADS):
                ch = tid + i * THREADS
                kk2 = ch // (HD // 8)
                dc = ch % (HD // 8)
                tk2, ok2 = key_token(lds_ld(ent, it * EPT + kk2 // RATIO), kk2 % RATIO, pos)
                vrow = kv_row(s, ok2.select(tk2, fx.Int32(0)), kvh)
                w4 = fx.Vector(bo.buffer_load(r_vc, (vrow + 8 * dc) // 2, vec_width=4, dtype=T.i32))
                vb = w4.bitcast(fx.BFloat16)
                for e in range_constexpr(8):
                    fx.ptr_store(vb[e], vt + ((8 * dc + e) * VROWE + kk2))
            gpu.barrier()
            # softmax over the tile: lane = key; head wave (pass 0) and wave + 8 (pass 1)
            kk = lane % 16
            tk_l, ok_l = key_token(lds_ld(ent, it * EPT + lane // RATIO), lane % RATIO, pos)
            for pss in range_constexpr(2):
                hh = wave + 8 * pss
                src = (fx.min(hh, 15) + 16 * (kk // 4)) * 4 + kk % 4
                dot = lds_ld(red, (lane // 16) * 256 + src) + lds_ld(red, (lane // 16 + 4) * 256 + src)
                sc_ = ok_l.select(dot * SCALE, fx.Float32(NEG))
                mo = m0 if pss == 0 else m1
                lo = l0 if pss == 0 else l1
                m_new = fx.max(mo, wave_max(sc_))
                p = exp(sc_ - m_new)
                corr = exp(mo - m_new)
                l_new = lo * corr + wave_sum(p)
                if hh < NQH:
                    lds_st(xs, XP + hh * KT + lane, ok_l.select(p, fx.Float32(0.0)))
                    if lane == 0:
                        lds_st(xs, XC + hh, corr)
                if const_expr(pss == 0):
                    m0 = m_new
                    l0 = l_new
                else:
                    m1 = (hh < NQH).select(m_new, m1)
                    l1 = (hh < NQH).select(l_new, l1)
            gpu.barrier()
            # P V: heads (rows 4 g4 + e) x this wave's dim groups
            cr = fx.Vector.from_elements([lds_ld(xs, XC + fx.min(4 * g4 + e, NQH - 1)) for e in range(4)], fx.Float32)
            acc = [fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32) * cr]
            acc.append(fx.Vector.from_elements([a4, a5, a6, a7], fx.Float32) * cr)
            for ks in range_constexpr(2):
                pa = fx.Vector(fx.ptr_load(xs + (XP + (lane % 16) * KT + 32 * ks + 8 * g4), result_type=v4f))
                pb = fx.Vector(fx.ptr_load(xs + (XP + (lane % 16) * KT + 32 * ks + 8 * g4 + 4), result_type=v4f))
                pf = fx.Vector.from_elements([pa[e] for e in range(4)] + [pb[e] for e in range(4)], fx.Float32)
                pbf = pf.to(fx.BFloat16)
                for ng in range_constexpr(NG):
                    dim = (wave * NG + ng) * 16 + lane % 16
                    vfr = fx.Vector(fx.ptr_load(vt + (dim * VROWE + 32 * ks + 8 * g4), result_type=v8b))
                    acc[ng] = mfma(pbf, vfr, acc[ng])
            a0, a1, a2, a3 = acc[0][0], acc[0][1], acc[0][2], acc[0][3]
            a4, a5, a6, a7 = acc[1][0], acc[1][1], acc[1][2], acc[1][3]
            gpu.barrier()
        # this launch's token: one more key, in the split that holds it
        if own_p:
            if tid < 2 * HD:
                nv = poll([(mb("kvnew"), (s * KVH + kvh) * 2 * HD + tid, 1)])[0][0].bitcast(fx.Float32)
                lds_st(xs, XN + tid, nv)
            gpu.barrier()
            for pss in range_constexpr(2):
                hh = wave + 8 * pss
                part = fx.Float32(0.0)
                for e in range_constexpr(HD // 64):
                    dd = lane * (HD // 64) + e
                    qd = fx.Float32(fx.ptr_load(kvt + (fx.min(hh, 15) * QROWE + dd)))
                    part = part + qd * lds_ld(xs, XN + dd)
                s_new = wave_sum(part) * SCALE
                mo = m0 if pss == 0 else m1
                lo = l0 if pss == 0 else l1
                m_new = fx.max(mo, s_new)
                corr = exp(mo - m_new)
                p_new = exp(s_new - m_new)
                l_new = lo * corr + p_new
                if (hh < NQH) & (lane == 0):
                    lds_st(xs, XC + hh, corr)
                    lds_st(xs, XC + 16 + hh, p_new)
                if const_expr(pss == 0):
                    m0 = m_new
                    l0 = l_new
                else:
                    m1 = (hh < NQH).select(m_new, m1)
                    l1 = (hh < NQH).select(l_new, l1)
            gpu.barrier()
            cr = fx.Vector.from_elements([lds_ld(xs, XC + fx.min(4 * g4 + e, NQH - 1)) for e in range(4)], fx.Float32)
            pn = fx.Vector.from_elements(
                [lds_ld(xs, XC + 16 + fx.min(4 * g4 + e, NQH - 1)) for e in range(4)], fx.Float32
            )
            vd0 = fx.Vector.filled(4, lds_ld(xs, XN + HD + (wave * NG) * 16 + lane % 16), fx.Float32)
            vd1 = fx.Vector.filled(4, lds_ld(xs, XN + HD + (wave * NG + 1) * 16 + lane % 16), fx.Float32)
            n0 = fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32) * cr + pn * vd0
            n1 = fx.Vector.from_elements([a4, a5, a6, a7], fx.Float32) * cr + pn * vd1
            a0, a1, a2, a3 = n0[0], n0[1], n0[2], n0[3]
            a4, a5, a6, a7 = n1[0], n1[1], n1[2], n1[3]
        # publish: rows 4 g4 + e < NQH are this kv head's query heads
        accs = [a0, a1, a2, a3, a4, a5, a6, a7]
        for ng in range_constexpr(NG):
            dim = (wave * NG + ng) * 16 + lane % 16
            for e in range_constexpr(4):
                hl = 4 * g4 + e
                if hl < NQH:
                    put(mb("sp_acc"), ((s * NS + js) * NQ + kvh * NQH + hl) * HD + dim, accs[4 * ng + e])
        if lane == 0:
            put2(mb("sp_ml"), ((s * NS + js) * NQ + kvh * NQH + wave) * 2, m0, l0)
            if wave + 8 < NQH:
                put2(mb("sp_ml"), ((s * NS + js) * NQ + kvh * NQH + wave + 8) * 2, m1, l1)
        stamp("split", tt, 4)

    # ================================ 6. merge the splits, sigmoid gate -> gated
    MT = 64
    GRP = THREADS // MT
    for tt in range(start("merge"), S * NQ * (HD // MT), G):
        tt = fx.Int32(tt)
        s = tt // (NQ * (HD // MT))
        h = (tt // (HD // MT)) % NQ
        dt = tt % (HD // MT)
        stamp("merge", tt, 0)
        gpu.barrier()
        jt = fx.min(tid, NS - 1)
        ml = poll([(mb("sp_ml"), ((s * NS + jt) * NQ + h) * 2, 2)])[0]
        mj = (tid < NS).select(ml[0].bitcast(fx.Float32), fx.Float32(NEG))
        w_ = wave_max(mj)
        if lane == 0:
            lds_st(red, wave, w_)
        gpu.barrier()
        mx = lds_ld(red, 0)
        for i in range_constexpr(1, WAVES):
            mx = fx.max(mx, lds_ld(red, i))
        gpu.barrier()
        wj = exp(mj - mx)
        lsum = block_sum((tid < NS).select(wj * ml[1].bitcast(fx.Float32), fx.Float32(0.0)))
        if tid < NS:
            lds_st(misc, tid, wj)
        gpu.barrier()
        stamp("merge", tt, 2)
        dim = tid % MT
        grp = tid // MT
        NQQ = -(-NS // GRP)
        accs = poll(
            [
                (mb("sp_acc"), ((s * NS + fx.min(grp + qq * GRP, NS - 1)) * NQ + h) * HD + dt * MT + dim, 1)
                for qq in range(NQQ)
            ]
        )
        part = fx.Float32(0.0)
        for qq in range_constexpr(NQQ):
            jj = grp + qq * GRP
            w = lds_ld(misc, fx.min(jj, NS - 1))
            part = part + (jj < NS).select(w * accs[qq][0].bitcast(fx.Float32), fx.Float32(0.0))
        lds_st(red, tid, part)
        gpu.barrier()
        if tid < MT:
            num = fx.Float32(0.0)
            for gq in range_constexpr(GRP):
                num = num + lds_ld(red, gq * MT + tid)
            o = bf16_round(num * rcp(lsum))
            gate = poll([(mb("gate"), s * NQ * HD + h * HD + dt * MT + tid, 1)])[0][0].bitcast(fx.Float32)
            lds_st(outs, tid, bf16_round(o * bf16_round(sigmoid(gate))))
        gpu.barrier()
        if tid < MT // 2:
            put_bf(
                mb("gated"), s * OUT_K + h * HD + dt * MT + 2 * tid, [lds_ld(outs, 2 * tid), lds_ld(outs, 2 * tid + 1)]
            )
        stamp("merge", tt, 4)
    return {}
