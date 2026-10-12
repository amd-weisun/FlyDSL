# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""What both token mixers of the Qwen3.5 MonoKernel share: the input-RMSNorm projection GEMV and
out_proj with the attention-side TP reduce."""

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu
from flydsl.expr.typing import T
from kernels.common import buffer_ops as bo
from kernels.monokernel.helpers import traced
from kernels.monokernel.ops import bf2_f32, lds_ld, rsrc
from kernels.monokernel.qwen35.kernel.plan import PROJ_GROUPS, PROJ_TILE, ROW_TILE, WAVES


@traced
def mixer_defs(ctx):
    """``proj_stage``: a GEMV over ``rmsnorm(h) * (1 + g_in)`` whose rows go to an epilogue."""
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    S = ctx["S"]
    emit_out = ctx["emit_out"]
    g_in1 = ctx["g_in1"]
    h_in = ctx["h_in"]
    n_sel = ctx["n_sel"]
    outs = ctx["outs"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    stage_x_gemma = ctx["stage_x_gemma"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    tid = ctx["tid"]
    unit_bf16 = ctx["unit_bf16"]
    unit_fp8 = ctx["unit_fp8"]
    wave = ctx["wave"]
    r_h = rsrc(h_in)
    NKC = HIDDEN // 64
    WPR = WAVES // PROJ_GROUPS
    UPW = NKC // WPR
    ROWS = PROJ_TILE * PROJ_GROUPS

    def ld_h(sks):
        res = []
        for s, k in sks:
            w = fx.Vector(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
            v = w.bitcast(fx.BFloat16).to(fx.Float32)
            res.append([v[j] for j in range(4)])
        return res

    def proj_stage(name, n_groups, w, sc, bf16_w, epilogue):
        """Stage ``name``: ``n_groups`` 16-row groups of FP8 (scales ``sc``) or BF16 matrix ``w``,
        PROJ_GROUPS per task; ``epilogue(s, row, value)`` takes each output (``tid`` < S * rows)."""
        r_w, r_s = rsrc(w), (None if bf16_w else rsrc(sc))
        batch = max(1, NKC // WAVES // (4 if bf16_w else 1))  # units in flight per wave
        for t in range(start(name), n_groups // PROJ_GROUPS, G):
            t = fx.Int32(t)
            stamp(name, t, 0)
            gpu.barrier()
            rg = t * PROJ_GROUPS + wave // WPR

            def u(c):
                kc = (wave % WPR) * UPW + c
                bw = (n_sel() * HIDDEN + kc * 64) // 2
                if const_expr(bf16_w):
                    return unit_bf16(r_w, rg, kc, NKC, bw)
                return unit_fp8(r_w, r_s, rg, kc, NKC, HIDDEN, 128, bw)

            pre = [u(c) for c in range(batch)]
            stage_x_gemma(ld_h, HIDDEN, g_in1)
            gpu.barrier()
            stamp(name, t, 2)
            acc = run_units(u, UPW, batch, pre)
            reduce_rows(PROJ_GROUPS, acc, emit_out(ROWS))
            stamp(name, t, 3)
            gpu.barrier()
            if tid < S * ROWS:
                epilogue(tid // ROWS, t * ROWS + tid % ROWS, lds_ld(outs, tid))
            stamp(name, t, 4)

    return dict(proj_stage=proj_stage)


@traced
def out_stages(ctx):
    """out_proj over the ``gated`` mailbox (the mixer's [S, OUT_K] bf16 output), the attention-side TP
    peer reduce and the residual ``h`` -> mailbox ``a``."""
    G = ctx["G"]
    HIDDEN = ctx["HIDDEN"]
    OUT_K = ctx["OUT_K"]
    S = ctx["S"]
    emit_out = ctx["emit_out"]
    h_in = ctx["h_in"]
    mb = ctx["mb"]
    n_sel = ctx["n_sel"]
    peer_reduce = ctx["peer_reduce"]
    put_bf = ctx["put_bf"]
    reduce_rows = ctx["reduce_rows"]
    run_units = ctx["run_units"]
    s_out = ctx["s_out"]
    stage_x_pairs = ctx["stage_x_pairs"]
    stamp = ctx["stamp"]
    start = ctx["start"]
    unit_fp8 = ctx["unit_fp8"]
    w_out = ctx["w_out"]
    wave = ctx["wave"]
    r_h = rsrc(h_in)
    OUT_NKC = OUT_K // 64
    OUT_R = ROW_TILE // 16
    OUT_WPR = WAVES // OUT_R
    OUT_UPW = OUT_NKC // OUT_WPR
    r_wo, r_so = rsrc(w_out), rsrc(s_out)

    def residual(s, row):
        return bf2_f32(fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32)))

    for t in range(start("out"), HIDDEN // ROW_TILE, G):
        t = fx.Int32(t)
        stamp("out", t, 0)
        rg = t * OUT_R + wave // OUT_WPR

        def u_o(c):
            kc = (wave % OUT_WPR) * OUT_UPW + c
            return unit_fp8(r_wo, r_so, rg, kc, OUT_NKC, OUT_K, 128, (n_sel() * OUT_K + kc * 64) // 2)

        pre = [u_o(c) for c in range(OUT_UPW)]
        gpu.barrier()
        stage_x_pairs("gated", S * OUT_K, lambda k: k)
        gpu.barrier()
        stamp("out", t, 2)
        acc = run_units(u_o, OUT_UPW, OUT_UPW, pre)
        reduce_rows(OUT_R, acc, emit_out(ROW_TILE))
        stamp("out", t, 3)
        gpu.barrier()
        peer_reduce(
            "attn",
            t,
            residual,
            lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
            tile=ROW_TILE,
        )
        gpu.barrier()
        stamp("out", t, 4)
    return {}
