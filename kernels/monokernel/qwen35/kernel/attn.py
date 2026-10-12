# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""The gated full-attention mixer of the Qwen3.5 MonoKernel: q | gate, k, v projection; q / k Gemma
norm + partial RoPE and the K / V write into ATOM's paged FP8 cache; split (flash-decoding) attention;
the merge with the sigmoid output gate, publishing ``gated`` for the shared out_proj.

The cache is ATOM's MHA pool layout (block size BS = 16): K ``[blocks, kv_heads, head_dim / 16, BS, 16]``,
V ``[blocks, kv_heads, BS / 16, head_dim, 16]``, one fp32 scale per (kv head, token) for each. A tile
keeps the FP8 codes (exact in bf16) and applies a key's scale to its score / softmax weight.
"""

import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.layout import NEG
from kernels.monokernel.ops import (
    bf16_round,
    exp,
    fp8_to_bf16x8,
    ld_f32,
    lds_ld,
    lds_st,
    rcp,
    rsq,
    rsrc,
    uniform,
    wave_max,
    wave_sum,
)
from kernels.monokernel.qwen35.config import FP8_MAX
from kernels.monokernel.qwen35.kernel.common import sigmoid
from kernels.monokernel.qwen35.kernel.plan import BS, KT, PROJ_TILE, WAVES


@traced
def attn_stages(ctx):
    """1-4. q | gate, k, v projection; norm + RoPE + cache write; split attention; merge + gate."""
    EPS = ctx["EPS"]
    G = ctx["G"]
    H = ctx["H"]
    HD = ctx["HD"]
    KROWE = ctx["KROWE"]
    KVD = ctx["KVD"]
    NSPLIT = ctx["NSPLIT"]
    OUT_K = ctx["OUT_K"]
    QG = ctx["QG"]
    ROPE = ctx["ROPE"]
    S = ctx["S"]
    T_SEQ = ctx["T_SEQ"]
    SCALE = ctx["SCALE"]
    block_sum = ctx["block_sum"]
    block_tables = ctx["block_tables"]
    slot_mapping = ctx["slot_mapping"]
    bt_stride = ctx["bt_stride"]
    g_kn1 = ctx["g_kn1"]
    g_qn1 = ctx["g_qn1"]
    k_cache = ctx["k_cache"]
    k_scale = ctx["k_scale"]
    kvt = ctx["kvt"]
    lane = ctx["lane"]
    mb = ctx["mb"]
    misc = ctx["misc"]
    outs = ctx["outs"]
    poll = ctx["poll"]
    positions = ctx["positions"]
    proj_stage = ctx["proj_stage"]
    put = ctx["put"]
    put2 = ctx["put2"]
    put_bf = ctx["put_bf"]
    red = ctx["red"]
    rope_cos = ctx["rope_cos"]
    rope_sin = ctx["rope_sin"]
    s_qkv = ctx["s_qkv"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    v_cache = ctx["v_cache"]
    v_scale = ctx["v_scale"]
    w_qkv = ctx["w_qkv"]
    wave = ctx["wave"]
    xs = ctx["xs"]
    r_pos, r_bt, r_slot = rsrc(positions), rsrc(block_tables), rsrc(slot_mapping)
    r_kc, r_vc, r_ks, r_vs = rsrc(k_cache), rsrc(v_cache), rsrc(k_scale), rsrc(v_scale)

    def ld_pos(s):
        return uniform(bo.buffer_load(r_pos, s, vec_width=1, dtype=T.i32))

    def ld_block(s, i):
        return fx.Int32(bo.buffer_load(r_bt, s * bt_stride + i, vec_width=1, dtype=T.i32))

    # ================================ 1. q | gate, k, v projection (FP8)
    def qkv_epilogue(s, row, v):
        # ATOM's row order: every local head's gate, then every head's q; mailbox qg holds [q, gate] per head
        if row < QG:
            hq = row % (QG // 2)
            put(mb("qg"), s * QG + (hq // HD) * 2 * HD + (row < QG // 2).select(fx.Int32(HD), fx.Int32(0)) + hq % HD, bf16_round(v))
        else:
            put(mb("kvr"), s * 2 * KVD + row - QG, bf16_round(v))

    proj_stage("proj", (QG + 2 * KVD) // PROJ_TILE, w_qkv, s_qkv, False, qkv_epilogue)

    # ================================ 2. Gemma q / k norm, partial RoPE; K / V -> the paged FP8 cache
    # task j < H: query head j; j == H: this rank's K / V head (one: local_kv_heads == 1)
    r_cos, r_sin, r_gq, r_gk = rsrc(rope_cos), rsrc(rope_sin), rsrc(g_qn1), rsrc(g_kn1)
    HALF = ROPE // 2

    def block_max(v):
        w = wave_max(v)
        if lane == 0:
            lds_st(red, wave, w)
        gpu.barrier()
        t = lds_ld(red, 0)
        for i in range_constexpr(1, WAVES):
            t = fx.max(t, lds_ld(red, i))
        gpu.barrier()
        return t

    def kv_quant(x, live):
        """Per-token FP8 E4M3 over the head (``amax / 448``) -> (FP8 value as f32, scale)."""
        amax = block_max(live.select(fmath.absf(x), fx.Float32(0.0)))
        sc = (amax > 0.0).select(amax / FP8_MAX, fx.Float32(1.0))
        q = fx.min(fx.max(x / sc, -FP8_MAX), FP8_MAX)
        word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
        v2 = fx.Vector.make_type(2, fx.Float32)
        q8 = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
        return q8, sc, word

    for tt in range(start("qk"), S * (H + 1), G):
        tt = fx.Int32(tt)
        s = tt // (H + 1)
        j = tt % (H + 1)
        is_kv = j == H
        stamp("qk", tt, 0)
        gpu.barrier()
        pos = ld_pos(s)
        live = tid < HD
        d = fx.min(tid, HD - 1)
        src = is_kv.select(mb("kvr"), mb("qg"))
        idx = is_kv.select(s * 2 * KVD + d, s * QG + j * 2 * HD + d)
        gam = is_kv.select(ld_f32(r_gk, d), ld_f32(r_gq, d))
        got = poll([(src, idx, 1), (mb("kvr"), s * 2 * KVD + KVD + d, 1)])
        x = got[0][0].bitcast(fx.Float32)
        vraw = got[1][0].bitcast(fx.Float32)
        rstd = rsq(block_sum(live.select(x * x, fx.Float32(0.0))) * (1.0 / HD) + EPS)
        xn = bf16_round(x * rstd * gam)
        lds_st(outs, d, xn)
        gpu.barrier()
        partner = lds_ld(outs, (d < HALF).select(d + HALF, d - HALF))
        cs = ld_f32(r_cos, pos * HALF + d % HALF)
        sn = ld_f32(r_sin, pos * HALF + d % HALF)
        rot = (d < HALF).select(xn * cs - partner * sn, xn * cs + partner * sn)
        y = bf16_round((d < ROPE).select(rot, xn))
        if live & (j < H):
            put(mb("q"), (s * H + j) * HD + d, y)
        if is_kv:
            k8, ksc, kword = kv_quant(y, live)
            v8, vsc, vword = kv_quant(vraw, live)
            # the token's cache slot (block * BS + offset) is the engine's; -1 = a padding row: no write
            slot = uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32))
            blk = slot // BS
            t = slot % BS
            if live & (slot >= 0):
                kb = arith.trunci(T.i8, fx.Int32(kword).ir_value())
                vb = arith.trunci(T.i8, fx.Int32(vword).ir_value())
                bo.buffer_store(kb, r_kc, ((blk * (HD // 16) + d // 16) * BS + t) * 16 + d % 16)
                bo.buffer_store(vb, r_vc, ((blk * (BS // 16) + t // 16) * HD + d) * 16 + t % 16)
            if live:
                # this launch's row for the split tasks: other CTAs may not see the cache stores yet
                put(mb("kvnew"), s * (2 * HD + 2) + d, k8)
                put(mb("kvnew"), s * (2 * HD + 2) + HD + d, v8)
            if (tid == 0) & (slot >= 0):
                bo.buffer_store(ksc, r_ks, blk * BS + t)
                bo.buffer_store(vsc, r_vs, blk * BS + t)
            if tid == 0:
                put(mb("kvnew"), s * (2 * HD + 2) + 2 * HD, ksc)
                put(mb("kvnew"), s * (2 * HD + 2) + 2 * HD + 1, vsc)
        stamp("qk", tt, 4)

    # ================================ 3. split attention on the matrix cores
    # A tile is KT = 64 keys = 4 cache blocks. S^T = K q^T: 16x16x32 MFMAs with keys as M (a row group is
    # one block) and the query heads as N, the head dim split over two wave halves; then P V with the heads
    # as M and 16-dim groups as N. K / V fragments come straight from the cache (FP8 codes are exact in
    # bf16); a key's scales fold into its score and its softmax weight. This launch's token is masked
    # from the tiles (other CTAs may not see its cache store yet) and added once, after them.
    QROWE = KROWE  # q rows (bf16) in kvt, padded off the bank
    XP = 0  # P [16 rows][KT] f32 (rows >= H stay 0)
    XC = XP + 16 * KT  # per-head rescale [16] | this token's weight [16]
    XN = XC + 32  # this token's K then V codes (f32) | its two scales
    WRG = wave % 4  # QK: this wave's key row group
    WKH = wave // 4  # QK: this wave's half of the head dim
    NG = HD // 16 // WAVES  # PV: 16-dim groups per wave
    assert NG == 2, "the P V accumulators are carried as two 16-dim groups per wave"
    KSTEPS = HD // 32 // 2  # QK k-steps per wave
    g4 = lane // 16
    v8b = fx.Vector.make_type(8, fx.BFloat16)
    v4f = fx.Vector.make_type(4, fx.Float32)
    v4z = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma(a, b, c):
        return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))

    for tt in range(start("split"), S * NSPLIT, G):
        tt = fx.Int32(tt)
        s = tt // NSPLIT
        js = tt % NSPLIT
        stamp("split", tt, 0)
        gpu.barrier()
        pos = ld_pos(s)
        ctx_len = pos + 1
        # keys from pos0 on are this launch's (the sequence's earlier MTP tokens and this one): from kvnew
        pos0 = pos - s % T_SEQ
        span = (ctx_len + NSPLIT * KT - 1) // (NSPLIT * KT) * KT  # keys per split, whole tiles
        k0 = js * span
        k1 = fx.min(k0 + span, ctx_len)
        ntile = fx.max(k1 - k0 + KT - 1, 0) // KT
        last_blk = (ctx_len - 1) // BS
        # q -> kvt as bf16 [16 rows][QROWE]; rows >= H and P rows >= H are zero
        qv = poll([(mb("q"), s * H * HD + tid + i * 512, 1) for i in range((H * HD) // 512)])
        for i in range_constexpr((H * HD) // 512):
            idx = tid + i * 512
            fx.ptr_store(qv[i][0].bitcast(fx.Float32).to(fx.BFloat16), kvt + ((idx // HD) * QROWE + idx % HD))
        for i in range_constexpr(((16 - H) * HD + 511) // 512):
            idx = fx.min(tid + i * 512, (16 - H) * HD - 1)
            fx.ptr_store(fx.Float32(0.0).to(fx.BFloat16), kvt + ((H + idx // HD) * QROWE + idx % HD))
        for i in range_constexpr(((16 - H) * KT + 511) // 512):
            lds_st(xs, XP + H * KT + fx.min(tid + i * 512, (16 - H) * KT - 1), fx.Float32(0.0))
        gpu.barrier()
        stamp("split", tt, 2)
        # this wave's q B-fragments (head lane % 16, dims 32 ks + 8 g4 ..), fixed over the tiles
        qb = [
            fx.Vector(fx.ptr_load(kvt + ((lane % 16) * QROWE + (WKH * KSTEPS + c) * 32 + 8 * g4), result_type=v8b))
            for c in range(KSTEPS)
        ]
        m = fx.Float32(NEG)
        lsum = fx.Float32(0.0)
        # P V accumulators, carried as named scalars (the loop rewriter carries names, not list items):
        # group n, row e is a{4 n + e}
        a0 = fx.Float32(0.0)
        a1 = fx.Float32(0.0)
        a2 = fx.Float32(0.0)
        a3 = fx.Float32(0.0)
        a4 = fx.Float32(0.0)
        a5 = fx.Float32(0.0)
        a6 = fx.Float32(0.0)
        a7 = fx.Float32(0.0)
        def tile_loads(kb_):
            """A tile's K fragments (this wave's row group and head-dim half), V fragments (dims 16 n + lane
            % 16, keys 32 ks + 8 g4 ..) and the lane's key scales: issued a tile ahead (blocks past the
            context clamp to the last one; those keys are masked)."""
            b0 = kb_ // BS
            kblk = uniform(bo.buffer_load(r_bt, s * bt_stride + fx.min(b0 + WRG, last_blk), vec_width=1, dtype=T.i32))
            kw = []
            for c in range_constexpr(KSTEPS):
                ks = WKH * KSTEPS + c
                off = kblk * (HD * BS) + ((2 * ks + g4 // 2) * BS + lane % 16) * 16 + 8 * (g4 % 2)
                w2 = fx.Vector(bo.buffer_load(r_kc, off // 4, vec_width=2, dtype=T.i32))
                kw += [w2[0], w2[1]]
            vw = []
            for n in range_constexpr(NG):
                for ks in range_constexpr(2):
                    vblk = ld_block(s, fx.min(b0 + 2 * ks + g4 // 2, last_blk))
                    dim = (wave * NG + n) * 16 + lane % 16
                    w2 = fx.Vector(
                        bo.buffer_load(
                            r_vc, (vblk * (HD * BS) + dim * 16 + 8 * (g4 % 2)) // 4, vec_width=2, dtype=T.i32
                        )
                    )
                    vw += [w2[0], w2[1]]
            sblk = ld_block(s, fx.min(b0 + lane // 16, last_blk))
            return kw + vw + [ld_f32(r_ks, sblk * BS + lane % 16), ld_f32(r_vs, sblk * BS + lane % 16)]

        # the next tile's fragments, carried as named scalars (the loop rewriter carries names, not lists)
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17 = tile_loads(k0)
        for it in range(0, ntile):
            kb = k0 + it * KT
            cur = [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15]
            ksc, vsc = f16, f17
            # S^T partial: keys of row group WRG (= block kb / BS + WRG) x heads, half WKH of the head dim
            c4 = v4z
            for c in range_constexpr(KSTEPS):
                c4 = mfma(fp8_to_bf16x8(cur[2 * c], cur[2 * c + 1]), qb[c], c4)
            fx.ptr_store(c4, red + (wave * 64 + lane) * 4)
            f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17 = tile_loads(kb + KT)
            # this lane's V fragments (masked keys' bytes 0)
            vfr = []
            for n in range_constexpr(NG):
                for ks in range_constexpr(2):
                    kfirst = kb + 32 * ks + 8 * g4
                    words = []
                    for half in range_constexpr(2):
                        mask = fx.Int32(0)
                        for j in range_constexpr(4):
                            key = kfirst + half * 4 + j
                            ok = (key < k1) & (key < pos0)
                            mask = mask | ok.select(fx.Int32(0xFF << (8 * j) if j < 3 else -(1 << 24)), fx.Int32(0))
                        words.append(cur[2 * KSTEPS + (n * 2 + ks) * 2 + half] & mask)
                    vfr.append(fp8_to_bf16x8(words[0], words[1]))
            gpu.barrier()
            # softmax over the tile: wave = head, lane = key
            kk = lane % 16
            src = (wave + 16 * (kk // 4)) * 4 + kk % 4
            dot = lds_ld(red, (lane // 16) * 256 + src) + lds_ld(red, (lane // 16 + 4) * 256 + src)
            key_abs = kb + lane
            ok = (key_abs < k1) & (key_abs < pos0)
            sc_ = ok.select(dot * ksc * SCALE, fx.Float32(NEG))
            m_new = fx.max(m, wave_max(sc_))
            p = exp(sc_ - m_new)
            corr = exp(m - m_new)
            lsum = lsum * corr + wave_sum(p)
            m = m_new
            lds_st(xs, XP + wave * KT + lane, ok.select(p * vsc, fx.Float32(0.0)))
            if lane == 0:
                lds_st(xs, XC + wave, corr)
            gpu.barrier()
            # P V: heads (rows 4 g4 + e) x this wave's dim groups
            cr = [lds_ld(xs, XC + fx.min(4 * g4 + e, H - 1)) for e in range(4)]
            crv = fx.Vector.from_elements(cr, fx.Float32)
            acc = [fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32) * crv]
            acc.append(fx.Vector.from_elements([a4, a5, a6, a7], fx.Float32) * crv)
            for ks in range_constexpr(2):
                pa = fx.Vector(fx.ptr_load(xs + (XP + (lane % 16) * KT + 32 * ks + 8 * g4), result_type=v4f))
                pb = fx.Vector(fx.ptr_load(xs + (XP + (lane % 16) * KT + 32 * ks + 8 * g4 + 4), result_type=v4f))
                pf = fx.Vector.from_elements([pa[e] for e in range(4)] + [pb[e] for e in range(4)], fx.Float32)
                pbf = pf.to(fx.BFloat16)
                for n in range_constexpr(NG):
                    acc[n] = mfma(pbf, vfr[n * 2 + ks], acc[n])
            a0, a1, a2, a3 = acc[0][0], acc[0][1], acc[0][2], acc[0][3]
            a4, a5, a6, a7 = acc[1][0], acc[1][1], acc[1][2], acc[1][3]
            gpu.barrier()
        # this launch's tokens of the sequence (positions pos0 .. pos): one more key each, in the split
        # that holds it
        for tk in range_constexpr(T_SEQ):
            kpos = pos0 + tk
            krow = s - s % T_SEQ + tk
            if const_expr(tk > 0):
                gpu.barrier()  # the previous key's LDS row is read
            if (kpos <= pos) & (kpos >= k0) & (kpos < k1):
                if tid < 2 * HD:
                    nv = poll([(mb("kvnew"), krow * (2 * HD + 2) + tid, 1)])[0][0].bitcast(fx.Float32)
                    lds_st(xs, XN + tid, nv)
                # its two scales (2 * HD may be every thread, so not a thread past the codes)
                if tid < 2:
                    sc2 = poll([(mb("kvnew"), krow * (2 * HD + 2) + 2 * HD + tid, 1)])[0][0].bitcast(fx.Float32)
                    lds_st(xs, XN + 2 * HD + tid, sc2)
                gpu.barrier()
                part = fx.Float32(0.0)
                for e in range_constexpr(HD // 64):
                    d = lane * (HD // 64) + e
                    qd = fx.Float32(fx.ptr_load(kvt + (wave * QROWE + d)))
                    part = part + qd * lds_ld(xs, XN + d)
                s_new = wave_sum(part) * lds_ld(xs, XN + 2 * HD) * SCALE
                m_new = fx.max(m, s_new)
                corr = exp(m - m_new)
                p_new = exp(s_new - m_new)
                lsum = lsum * corr + p_new
                m = m_new
                if lane == 0:
                    lds_st(xs, XC + wave, corr)
                    lds_st(xs, XC + 16 + wave, p_new * lds_ld(xs, XN + 2 * HD + 1))
                gpu.barrier()
                cr = fx.Vector.from_elements([lds_ld(xs, XC + fx.min(4 * g4 + e, H - 1)) for e in range(4)], fx.Float32)
                pn = fx.Vector.from_elements(
                    [lds_ld(xs, XC + 16 + fx.min(4 * g4 + e, H - 1)) for e in range(4)], fx.Float32
                )
                vd0 = fx.Vector.filled(4, lds_ld(xs, XN + HD + (wave * NG) * 16 + lane % 16), fx.Float32)
                vd1 = fx.Vector.filled(4, lds_ld(xs, XN + HD + (wave * NG + 1) * 16 + lane % 16), fx.Float32)
                n0 = fx.Vector.from_elements([a0, a1, a2, a3], fx.Float32) * cr + pn * vd0
                n1 = fx.Vector.from_elements([a4, a5, a6, a7], fx.Float32) * cr + pn * vd1
                a0, a1, a2, a3 = n0[0], n0[1], n0[2], n0[3]
                a4, a5, a6, a7 = n1[0], n1[1], n1[2], n1[3]
        # publish: rows 4 g4 + e < H are the query heads
        if lane < 4 * H:
            accs = [a0, a1, a2, a3, a4, a5, a6, a7]
            for n in range_constexpr(NG):
                dim = (wave * NG + n) * 16 + lane % 16
                for e in range_constexpr(4):
                    head = 4 * g4 + e
                    put(mb("sp_acc"), ((s * NSPLIT + js) * H + head) * HD + dim, accs[4 * n + e])
        if lane == 0:
            put2(mb("sp_ml"), ((s * NSPLIT + js) * H + wave) * 2, m, lsum)
        stamp("split", tt, 4)

    # ================================ 4. merge the splits, output gate -> gated
    MT = 64  # dims per merge task
    GRP = 512 // MT  # split groups per task, one thread per (group, dim)
    for tt in range(start("merge"), S * H * (HD // MT), G):
        tt = fx.Int32(tt)
        s = tt // (H * (HD // MT))
        h = (tt // (HD // MT)) % H
        dt = tt % (HD // MT)
        stamp("merge", tt, 0)
        gpu.barrier()
        jt = fx.min(tid, NSPLIT - 1)
        ml = poll([(mb("sp_ml"), ((s * NSPLIT + jt) * H + h) * 2, 2)])[0]
        mj = (tid < NSPLIT).select(ml[0].bitcast(fx.Float32), fx.Float32(NEG))
        mx = block_max(mj)
        wj = exp(mj - mx)
        lsum = block_sum((tid < NSPLIT).select(wj * ml[1].bitcast(fx.Float32), fx.Float32(0.0)))
        if tid < NSPLIT:
            lds_st(misc, tid, wj)
        gpu.barrier()
        stamp("merge", tt, 2)
        dim = tid % MT
        grp = tid // MT
        NQ = -(-NSPLIT // GRP)
        accs = poll(
            [
                (mb("sp_acc"), ((s * NSPLIT + fx.min(grp + q * GRP, NSPLIT - 1)) * H + h) * HD + dt * MT + dim, 1)
                for q in range(NQ)
            ]
        )
        part = fx.Float32(0.0)
        for q in range_constexpr(NQ):
            jj = grp + q * GRP
            w = lds_ld(misc, fx.min(jj, NSPLIT - 1))
            part = part + (jj < NSPLIT).select(w * accs[q][0].bitcast(fx.Float32), fx.Float32(0.0))
        lds_st(red, tid, part)
        gpu.barrier()
        if tid < MT:
            num = fx.Float32(0.0)
            for gq in range_constexpr(GRP):
                num = num + lds_ld(red, gq * MT + tid)
            o = bf16_round(num * rcp(lsum))
            gate = poll([(mb("qg"), s * QG + h * 2 * HD + HD + dt * MT + tid, 1)])[0][0].bitcast(fx.Float32)
            lds_st(outs, tid, bf16_round(o * sigmoid(gate)))
        gpu.barrier()
        if tid < MT // 2:
            put_bf(
                mb("gated"), s * OUT_K + h * HD + dt * MT + 2 * tid, [lds_ld(outs, 2 * tid), lds_ld(outs, 2 * tid + 1)]
            )
        stamp("merge", tt, 4)
    return {}
