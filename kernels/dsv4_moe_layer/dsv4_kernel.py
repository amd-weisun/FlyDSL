# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""DeepSeek-V4 attention + MoE layer in ONE persistent launch per rank (TP8 decode).

Forked from ``kernels/mla_moe_layer/shared_reuse_moe_kernel.py``. The scheduler,
tagged-pair mailboxes, MFMA GEMV machinery, quantization and TP peer-reduce are
carried over unchanged; the attention half is rewritten because V4 is not MLA.

V4 is shared-KV (MQA): ``wkv`` emits ONE ``HEAD_DIM`` vector per token, K and V
are the *same* tensor, and RoPE occupies its last ``ROPE_DIM`` lanes. So there is
no ``kv_lora`` / nope-vs-pe split and no absorbed ``W_UK``/``W_UV``:

* ``q_b`` emits the full ``HEAD_DIM`` query directly -- MLA's ``uk`` stage is gone.
* ``split`` contracts over ``HEAD_DIM`` out of a single LDS tile (MLA needed two,
  one for the latent and one for k_pe), and adds a per-head learnable softmax
  ``sink`` to the denominator.
* ``uv`` keeps MLA's flash-decode merge but drops the ``W_UV`` GEMV, and instead
  de-rotates the output's RoPE lanes (needed because V shares the RoPE'd K).
* ``o`` becomes the grouped low-rank pair ``o_a`` (per ``O_GROUPS``) then ``o_b``.

One launch of ``grid = 256 CTAs x 512 threads`` (one CTA per MI355X CU) runs the
whole layer body for this rank's TP shard::

    input RMSNorm -> q_a / kv projection -> q_a RMSNorm -> q_b (+ head RMS, RoPE)
      -> KV RMSNorm / RoPE / FP8 round-trip -> sliding-window KV ring publish
      -> gather-sparse split softmax (+ sink) -> merge -> inverse RoPE
      -> o_a (grouped low rank) -> o_b
      -> attention TP peer reduce + residual                       (sym_attn)
      -> post-attention RMSNorm -> router sqrt-softplus + expert activation
      -> flat top-6 -> 1 shared + 6 routed expert up/gate/clamped SwiGLU
      -> expert down + route weighting
      -> MoE TP peer reduce + residual -> x_out                    (sym_ffn)

This module covers the **sliding-window-only** layer (``compress_ratio == 0``),
which is the MTP block's shape and the first milestone of the staged V4 plan. The
KV compressor (HCA, ratio 128) and lightning indexer (CSA, ratio 4) add stages
later; hyper-connections replace the plain residual at that point too.

Scheduling: every stage is a list of tasks; task ``t`` of a stage runs on CTA
``(stage_base + t) % 256`` and every CTA walks the stages in order.  There is
no grid-wide barrier: dependencies only point to earlier stages and all CTAs
are co-resident, so every spin wait makes progress.

Mailboxes are *tagged pairs*: every 32-bit value a task hands to another CTA
(or GPU) is stored next to this launch's epoch tag, ``(value, tag)``, with
device- (``sc1``) or system-coherent (``sc0 sc1``) 8 / 16-byte stores.  A
consumer polls the payload itself until the tags match, so a hand-off costs
one memory round trip: no store drain, no separate flag, no second load.

GEMVs run on the matrix cores: ``packing.py`` arranges weights so one wave
loads 16 rows x 64 k as one contiguous 1 KB. FP8 is widened exactly to bf16 and
fed to ``mfma_f32_16x16x32_bf16`` with the samples as the N dimension.  Each
64-k chunk's partial is scaled by its f32 block scale (times any activation
scale / route weight) into the accumulator, so the math is exact block-scaled
FP8 on bf16 activations.  Weight loads that do not depend on upstream results
are issued before the task waits for its inputs.

Cross-GPU: each rank rounds partial rows to BF16 and pushes packed pairs plus
an epoch tag into every peer's symmetric buffer and polls its own; every rank
sums the 8 partials in rank order, so all ranks produce bit-identical hidden
states (and routing).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T, as_ir_value
from kernels.common import buffer_ops as bo
from kernels.common.dpp_utils import update_dpp_i32
from kernels.dsv4_moe_layer.config import (
    EPS,
    FP8_MAX,
    HC_EPS,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    MAX_LAYERS_PER_STEP,
    MOE_SLOTS,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    ROUTE_SCALE,
    SCALE_BM,
    SOFTMAX_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    moe_format,
)

BLOCKS = 256
LAYER_SLOTS = MAX_LAYERS_PER_STEP
THREADS = 512
WAVES = THREADS // 64
QKV_A_TILE = 16
Q_B_TILE = 16
UV_TILE = 64  # output dims merged per uv task
ROW_TILE = 32  # rows per o_a / o_b / attention peer-reduce tile
ROUTER_TILE = 8  # experts per router task (a part of a 16-row MFMA group)
UG_TILE = 16  # intermediates per up/gate task (16 gate rows + 16 up rows)
UG8 = 8  # intermediates one up/gate task actually owns
SPLIT_KEYS = 64
HC_CPW = 2  # hc_pre 64-K chunks per wave; sets the K split across tasks
NEG = -1.0e30
MIN_I32 = -(1 << 31)  # flips the sign bit: signed-ordered <-> unsigned-ordered

# task counts per stage
QKV_A_ROWS = Q_LORA + HEAD_DIM
N_QKV_A = QKV_A_ROWS // QKV_A_TILE
N_ROW_TILES = HIDDEN // ROW_TILE


def dn_tile(S: int, hidden: int = HIDDEN) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1 (placed off
    the router CTAs, whose up/gate task finishes last, so every down task streams
    its weights during the mid wait); about one task per CTA above.

    A MULTIPLE OF 16, which the plain ``hidden // BLOCKS`` is not. ``emit_dn``
    keeps tile rows ``dn_off .. dn_off + DN_TILE`` of the ``DN_R * 16`` rows
    ``reduce_rows`` produces, where ``dn_off = t * DN_TILE % 16``; that only covers
    the tile when ``dn_off + DN_TILE <= DN_R * 16``. GLM-5's 6144 / 256 = 24 gives
    dn_off in {0, 8} and just fits, which is why the MLA parent never saw this.
    V4-Pro's 7168 / 256 = 28 gives dn_off in {0, 4, 8, 12}, and a tile at 12 needs
    40 rows out of 32 -- eight hidden rows never written, then pushed to the peers
    as whatever LDS held. Rounding up makes every tile 16-aligned, so dn_off is 0.
    """
    return 32 if S == 1 else max(16, -(-(hidden // BLOCKS) // 16) * 16)


N_ROUTER = N_EXPERTS // ROUTER_TILE
N_UG_PER_SLOT = INTER // UG_TILE


# gfx94x/95x cache policy bits (LLVM CPol): SC0 = 1, NT = 2, SC1 = 16.  SC1:SC0 is
# the coherence scope of the access itself: SC1 = device (past the per-XCD
# non-coherent caches), SC0|SC1 = system (peer GPUs over XGMI).
CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12  # mailbox specs polled per batch
TL_COLS = 8  # timeline stamps per task: 5 phases + 3 free debug marks


def _align(n, a=256):
    return (n + a - 1) // a * a


def hc_shape(hc_mult: int, hidden: int):
    """(tasks per side, K per task, rows, values published per task) for hc_pre.

    The projection has only (2 + hc) * hc rows but K = hc * hidden, so its cost is
    weight bytes. K is split across tasks to keep the per-task volume in line with
    the other GEMVs instead of parking megabytes on one CU. Each task also carries
    a partial sum of squares for hc_pre's weightless RMS, published as one extra row.
    """
    if hc_mult <= 1:
        return 0, 0, 0, 0
    rows = ((2 + hc_mult) * hc_mult + 15) // 16 * 16
    k_total = hc_mult * hidden
    n_tasks = k_total // (64 * WAVES * HC_CPW)
    return n_tasks, k_total // n_tasks, rows, rows + 1


def layout(
    S: int,
    heads: int,
    npes: int,
    window: int = WINDOW,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    hc_mult: int = 1,
    compress_ratio: int = 0,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    # this and index_topk are unused here, and stay in the signature on purpose:
    # layout(), stage_tasks() and build_dsv4_kernel() are all fed from ONE dims
    # dict, and letting them drift apart is what caused three separate bugs
    index_heads_total: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
):
    """Byte offsets of the per-rank scratch and of the symmetric buffer.

    Every mailbox holds ``(value, tag)`` int32 pairs (8 bytes per element)."""
    fmt = moe_format(moe_mode)
    quant_group = fmt.activation_group
    xq_blocks = 0 if quant_group is None else hidden // quant_group
    n_split = (window if n_keys is None else n_keys) // SPLIT_KEYS
    hc_tasks, _, hc_rows, hc_vals = hc_shape(hc_mult, hidden)
    hc_coef = 2 * hc_mult + hc_mult * hc_mult  # pre | post | comb
    pr = 8
    items = [
        # two sides (attention, ffn) of hyper-connection coefficients per sample
        ("hc_d", S * 2 * max(hc_tasks, 1) * max(hc_vals, 1) * pr),
        ("hc_c", S * 2 * max(hc_coef, 1) * pr),
        # hc_pre's output: the hc_mult streams contracted by `pre` to one. An
        # explicit mailbox rather than contracting at each consumer -- `a` alone is
        # read in six places and inline contraction would quadruple that traffic.
        ("xin", S * hidden * pr if hc_mult > 1 else pr),
        ("ain", S * hidden * pr if hc_mult > 1 else pr),
        ("q_a", S * q_lora * pr),
        ("kv_a", S * head_dim * pr),  # the single shared KV row, pre-norm
        # The compressor's own kv / gate, split out of the same fused qkv_a GEMV.
        # c_coff-wide: overlapping windows carry a previous-window half too.
        ("c_kv", S * c_coff * head_dim * pr if compress_ratio else pr),
        ("c_gate", S * c_coff * head_dim * pr if compress_ratio else pr),
        # ... and the indexer's own compressor, out of the same GEMV
        ("i_kv", S * c_coff * index_head_dim * pr if index_head_dim else pr),
        ("i_gate", S * c_coff * index_head_dim * pr if index_head_dim else pr),
        # the compressed row this launch just wrote, for the same reason kvnew
        # exists: the gather cannot rely on seeing our own global store
        ("cnew", S * head_dim * pr if compress_ratio else pr),
        ("kvnew", S * head_dim * pr),  # this launch's KV ring rows (bf16 values)
        ("q_raw", S * heads * head_dim * pr),  # q_b output, before the per-head RMS
        # the indexer's query, raw then rotated/rotated-into-Hadamard/FP4
        ("i_q_raw", S * index_heads * index_head_dim * pr if index_head_dim else pr),
        ("i_q", S * index_heads * index_head_dim * pr if index_head_dim else pr),
        ("i_wp", S * index_heads * pr if index_head_dim else pr),
        # the row this launch just compressed, for the same reason cnew exists:
        # the scorer cannot rely on seeing our own global store
        ("i_cnew", S * index_head_dim * pr if index_head_dim else pr),
        # one score per compressed entry; entries not yet written score NEG
        ("i_score", S * max(n_compressed(max_seq, compress_ratio), 1) * pr),
        # the slots the indexer picked, which the attention gather reads instead of
        # the caller's index list for the compressed half
        ("i_sel", S * max((n_keys or window) - window, 1) * pr),
        ("q", S * heads * head_dim * pr),  # full per-head query: rope is inside it
        ("sp_acc", S * n_split * heads * head_dim * pr),
        ("sp_m", S * n_split * heads * pr),
        ("sp_l", S * n_split * heads * pr),
        ("o", S * heads * head_dim * pr),  # merged, de-rotated attention output
        ("o_lora", S * o_groups * o_lora * pr),
        ("a", S * max(hc_mult, 1) * hidden * pr),  # post-attention residual stream (bf16)
        ("scores", S * N_EXPERTS * pr),
        ("xq", S * hidden // (4 if quant_group is not None else 2) * pr),
        ("xqs", S * xq_blocks * pr),
        ("sel", S * MOE_SLOTS * pr),
        ("prob", S * MOE_SLOTS * pr),
        ("mid", S * MOE_SLOTS * INTER * pr),
        ("xqd", S * hidden * 4),  # debug: dequantized MoE activation (plain f32)
    ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * hidden * pr
    # the indexer's score partials get their own region: every rank scores every
    # compressed entry with its OWN index heads, so the sum spans ranks
    iscore = npes * S * max(n_compressed(max_seq, compress_ratio), 1) * pr
    sym = {"attn": 0, "ffn": part, "iscore": 2 * part, "_bytes": 2 * part + _align(iscore)}
    return scratch, sym


def _rsrc(addr):
    return bo.create_buffer_resource_from_addr(addr)


def _uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _uniform_f32(v):
    return _uniform(fx.Float32(v).bitcast(fx.Int32)).bitcast(fx.Float32)


def _hw_f32(name, x):
    """One hardware transcendental (v_rsq / v_rcp / v_exp): ~1 ulp, no libm range fixups."""
    return fx.Float32(llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], []))


def _rsq(x):
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def _rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def _exp(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", fx.Float32(x) * 1.4426950408889634)


def _sqrt_softplus(x):
    """V4's router score: sqrt(softplus(x)), in the numerically stable branchless form
    softplus(x) = max(x, 0) + log1p(exp(-|x|))."""
    sp = fx.max(x, fx.Float32(0.0)) + fmath.log1p(_exp(-fmath.absf(x)))
    return fmath.sqrt(sp)


def _swiglu(g, u, limit):
    """SwiGLU with V4's clamp. Note the asymmetry: ``up`` is clamped on both sides,
    ``gate`` only from above (``limit <= 0`` disables both)."""
    if const_expr(limit > 0):
        g = fx.min(g, fx.Float32(limit))
        u = fx.min(fx.max(u, fx.Float32(-limit)), fx.Float32(limit))
    return g * _rcp(1.0 + _exp(-g)) * u


def _sort_network(n):
    """Odd-even transposition network: a provably correct (if not minimal) list of
    compare-exchange pairs for ``n`` elements, unrolled at build time."""
    pairs = []
    for r in range(n):
        for i in range(r % 2, n - 1, 2):
            pairs.append((i, i + 1))
    return pairs


def _xshfl(v, off):
    """Value of lane ``lane ^ off``.  Offsets 32 / 16 lower to v_permlane*_swap; the
    in-row offsets use DPP (VALU latency) instead of ds_swizzle (LDS latency)."""
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:  # row_shr:8 into banks 2-3, row_shl:8 into banks 0-1
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:  # quad_perm [2, 3, 0, 1]
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:  # quad_perm [1, 0, 3, 2]
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def _wave_umax(v):
    """Unsigned max over the fully active wave as a wave-uniform Int32."""
    return fx.Int32(fx.coop.warp_reduce(fx.Uint32(v), fx.ReductionOp.MAX, width=64))


def _xred(v, off, op):
    """op(v, value of lane ``lane ^ off``) for a symmetric op.  Offsets 32 / 16 take
    both halves of one v_permlane*_swap as the operands (no select needed)."""
    if off < 16:
        return op(v, _xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = as_ir_value(v.bitcast(fx.Int32) if is_f else fx.Int32(v))
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(llvm.StructType.get_literal([T.i32, T.i32]), x, x, False, False)
    a, b = (fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


FP4_MAX = 6.0


def _fp4_roundtrip(a, b):
    """f32 pair -> E2M1 -> f32 pair (inputs already scaled into range).

    The scale operand is 1.0 and the scaling is done in f32 around this, as
    ``_fp8_to_bf16x8`` does: the hardware honours only the scale's exponent, and
    keeping the arithmetic explicit means this does not depend on which direction
    the instruction applies it.
    """
    one = as_ir_value(fx.Float32(1.0))
    word = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, as_ir_value(fx.Int32(0)), a, b, one, 0))
    v2 = fx.Vector.make_type(2, fx.Float32)
    out = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp4(res=v2, src=as_ir_value(word), scale=one, src_sel_index=0))
    return out[0], out[1]


def _pow2_ceil(x):
    """Smallest power of two >= x, by exponent arithmetic.

    V4's FP4 block scale is rounded UP to a power of two, so it is exact in the
    exponent and costs no mantissa. Matches reference.quant_dequant_fp4.
    """
    bits = fx.Float32(x).bitcast(fx.Int32)
    man = bits & ((1 << 23) - 1)
    e = ((bits >> 23) & 0xFF) - 127 + (man != 0).select(fx.Int32(1), fx.Int32(0))
    return ((e + 127) << 23).bitcast(fx.Float32)


def _fp8_roundtrip(a, b):
    """f32 pair -> E4M3FN -> f32 pair (inputs already scaled into range)."""
    word = rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False)
    v2 = fx.Vector.make_type(2, fx.Float32)
    lo = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))
    return lo[0], lo[1]


def f8_word(k):
    """LDS word of FP8 activation byte k: each 64-k chunk is stored so that the 16 bytes
    lane group g needs (k = 8 g + [0, 8) and 32 + 8 g + [0, 8), the packed weight order)
    are contiguous."""
    return (k // 64) * 16 + ((k % 32) // 8) * 4 + ((k % 64) // 32) * 2 + (k % 8) // 4


def _fp8_to_bf16x8(w0, w1):
    """Two dwords of 8 FP8 -> vector<8 x bf16> (exact: E4M3 is a subset of bf16).

    ``cvt_scalef32_pk_bf16_fp8`` only honours the scale's exponent, so the scale
    is 1 here and the f32 block scale is applied to the MFMA partials instead.
    """
    one = as_ir_value(fx.Float32(1.0))
    parts = []
    for w in (w0, w1):
        for half in range_constexpr(2):
            pr = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), as_ir_value(w), one, bool(half)))
            parts += [pr[0], pr[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


def _mxfp4_to_bf16x8(word, scale):
    """One packed dword of eight E2M1 values -> scaled BF16 MFMA operand."""

    parts = []
    for select in range_constexpr(4):
        pair = fx.Vector(
            rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), as_ir_value(word), as_ir_value(scale), select)
        )
        parts += [pair[0], pair[1]]
    return fx.Vector.from_elements(parts, fx.BFloat16)


SCORE_TILE = 512  # one candidate per thread


def n_compressed(max_seq: int, compress_ratio: int) -> int:
    return max_seq // compress_ratio if compress_ratio else 0


def n_index(max_seq: int, compress_ratio: int, index_head_dim: int, index_topk: int) -> int:
    """Compressed slots the attention can gather: the indexer's pick, capped."""
    if not index_head_dim:
        return 0
    return min(index_topk, n_compressed(max_seq, compress_ratio))


def n_score_tiles(max_seq: int, compress_ratio: int, index_head_dim: int) -> int:
    """Tiles of compressed entries the indexer scores, sized for the WHOLE cache.

    The count that matters at runtime is how many entries exist so far, which
    grows with position; a monokernel stage cannot be sized from that, so it is
    sized for the maximum and entries past the end score NEG.
    """
    if not index_head_dim:
        return 0
    n = n_compressed(max_seq, compress_ratio)
    return (n + SCORE_TILE - 1) // SCORE_TILE


def qkv_a_rows(q_lora: int, head_dim: int, compress_ratio: int, c_coff: int, index_head_dim: int = 0) -> int:
    """Rows of the fused qkv_a GEMV.

    q_a, kv, the compressor's c_coff-wide pair, and -- when the indexer runs --
    its own compressor's pair as well. Mirrors reference.qkv_a_tail(). Both the
    task count and the kernel's loop bound come from here; they disagreed before,
    so the kernel ran qkv_a tasks the CTA placement had not accounted for.
    """
    if not compress_ratio:
        return q_lora + head_dim
    tail = 2 * c_coff * (head_dim + index_head_dim)
    return q_lora + head_dim + tail


def stage_tasks(
    S: int,
    heads: int,
    window: int = WINDOW,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    top_k: int = TOP_K,
    inter: int = INTER,
    hc_mult: int = 1,
    compress_ratio: int = 0,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    index_heads_total: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
):
    """[(stage name, task count)] in execution order.

    MLA's ``uk`` stage has no V4 counterpart (no absorbed W_UK) and its ``o``
    GEMV splits into the grouped low-rank pair ``o_a`` / ``o_b``."""
    n_qkv_a = qkv_a_rows(q_lora, head_dim, compress_ratio, c_coff, index_head_dim) // QKV_A_TILE
    hc_tasks, _, _, _ = hc_shape(hc_mult, hidden)
    return [
        # hyper-connection pre-mix for the attention side, before anything reads
        # the (hc_mult-wide) residual stream
        ("hcd_a", S * hc_tasks),
        ("hcc_a", (hidden // ROW_TILE) if hc_mult > 1 else 0),
        ("qkv_a", n_qkv_a),
        ("cache", 1),
        # the compressed entry has to land before the attention gathers it
        ("cmp", S if compress_ratio else 0),
        # the indexer's compressed entry, for the scoring that selects keys
        ("i_cmp", S if index_head_dim else 0),
        ("q_b", heads * head_dim // Q_B_TILE),
        ("q_norm", S * heads),
        # the indexer's query, and its rope / rotation / FP4 tail
        ("i_q_b", index_heads * index_head_dim // Q_B_TILE if index_head_dim else 0),
        ("i_q", S if index_head_dim else 0),
        # the per-head score weights, then the score of every compressed entry
        ("i_wp", S if index_head_dim else 0),
        ("i_score", S * n_score_tiles(max_seq, compress_ratio, index_head_dim)),
        # ... and the top-k over them, which the split stage waits on
        ("i_topk", S if index_head_dim else 0),
        ("split", S * ((window if n_keys is None else n_keys) // SPLIT_KEYS)),
        ("uv", S * (heads * head_dim // UV_TILE)),
        ("o_a", S * o_groups * o_lora // ROW_TILE),
        ("o_b", hidden // ROW_TILE),
        # ... and for the ffn side, once o_b has produced the new residual stream
        ("hcd_f", S * hc_tasks),
        ("hcc_f", (hidden // ROW_TILE) if hc_mult > 1 else 0),
        ("router", S * N_ROUTER),
        # one tile per (routed slot, 8 intermediates); tasks below INTER / UG8 also
        # carry the shared expert.  GLM-5/V3 happened to make this exactly BLOCKS
        # (8 slots x 32 tiles); V4's top-6 over INTER 384 does not.
        ("ug", S * top_k * (inter // UG8)),
        ("down", hidden // dn_tile(S, hidden)),
    ]


def build_dsv4_kernel(
    S: int = 1,
    heads: int = 16,
    npes: int = 8,
    window: int = WINDOW,
    scale: float = SOFTMAX_SCALE,
    timeline: bool = False,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    hidden: int = HIDDEN,
    q_lora: int = Q_LORA,
    head_dim: int = HEAD_DIM,
    o_groups: int = O_GROUPS,
    o_lora: int = O_LORA,
    n_experts: int = N_EXPERTS,
    top_k: int = TOP_K,
    inter: int = INTER,
    swiglu_limit: float = SWIGLU_LIMIT,
    hc_mult: int = 1,
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS,
    hc_eps: float = HC_EPS,
    compress_ratio: int = 0,
    window_rows: int | None = None,
    n_keys: int | None = None,
    c_coff: int = 1,
    index_head_dim: int = 0,
    index_heads: int = 0,
    index_heads_total: int = 0,
    index_topk: int = 0,
    max_seq: int = 0,
):
    """Return the ``@flyc.jit`` launcher for one rank's whole V4 layer.

    Covers the sliding-window-only layer (``compress_ratio == 0``); ``window``
    keys are gathered per sample from a ring cache of that size, so unlike the
    MLA kernel's ``topk`` there is no dense-vs-sparse switch -- V4's window is
    always exactly ``window`` slots, with -1 marking ones not yet written.

    ``timeline=True`` records ``s_memrealtime`` (100 MHz) at the start and end of
    every task, and once its inputs have arrived, into the ``timeline`` buffer:
    int64 ``[sum(task counts), TL_COLS]`` (start, hint seen, inputs staged, compute
    done, end, then free debug marks) in ``stage_tasks`` order.
    """
    # The split-attention score MFMA's N width is the hardware 16 (hn clamps to
    # heads - 1 below it); heads > 16 would need a wider MFMA tiling, not just more
    # head-groups per wave, so this only covers heads in (8, 16) today.
    assert (
        heads % WAVES == 0 and heads <= 16
    ), "the split-attention mapping needs a whole number of wave-groups per head, heads <= 16"
    assert window % SPLIT_KEYS == 0
    # What actually bounds the sample count, rather than one opaque `S <= 8`. Each
    # is satisfied at every shipped shape; they are here so a future one fails the
    # build instead of hanging or quietly producing zeros on the GPU.
    assert 1 <= S <= 16, "n_sel / reduce_rows put the samples in the MFMA's 16 B columns"
    assert S <= WAVES, f"dn_route routes sample s on wave s, and there are {WAVES} waves"
    assert S * (1 + top_k) <= SPLIT_KEYS, (
        f"dn_route packs S * MOE_SLOTS = {S * (1 + top_k)} expert ids into Smem.keys, "
        f"which the split stage sizes at SPLIT_KEYS = {SPLIT_KEYS}"
    )
    assert head_dim % 64 == 0, "the score MFMA walks HEAD_DIM in 32-wide steps over 2 wave halves"
    # The PV MFMA gives each wave HEAD_DIM / 32 / WAVES dim-pair groups and the KV
    # gather gives each lane HEAD_DIM / 64 elements; both silently produce no work at
    # all (-> an unfillable mailbox -> a poll that never retires) if HEAD_DIM is too
    # small, so reject that here rather than hanging on the GPU.
    assert (
        head_dim % (32 * WAVES) == 0
    ), f"head_dim must be a multiple of {32 * WAVES} for the PV MFMA's per-wave dim groups, got {head_dim}"
    assert head_dim >= 128, "the KV gather needs at least one packed word per lane"
    assert (head_dim - ROPE_DIM) % 64 == 0, "the KV row's FP8 round-trip blocks the nope part by 64"
    assert heads % o_groups == 0, "o_a groups partition the concatenated heads"
    assert o_lora % ROW_TILE == 0 and (heads * head_dim // o_groups) % 64 == 0
    assert head_dim <= THREADS, "the cache / q_norm stages map one thread per head dim"
    assert (head_dim - ROPE_DIM) % 2 == 0, "interleaved RoPE pairs must align to the nope boundary"
    assert n_experts % 64 == 0, "route_topk packs n_experts // 64 selection keys per lane"
    assert n_experts <= 1 << 16, "the packed selection key needs room for the score above the id field"
    assert inter % UG_TILE == 0
    # Shadow the module-level fixed-shard constants of the same name with this
    # build's overrides before anything else in this function reads them; every
    # later reference to the bare names in this function (including inside the
    # kernel and its nested helpers below) resolves to these, not the module
    # constants, because Python treats a name assigned anywhere in a function as
    # local to the whole function (so this block must run first, or any earlier
    # read raises UnboundLocalError instead of silently using the module default).
    HIDDEN = hidden
    Q_LORA = q_lora
    HEAD_DIM = head_dim
    O_GROUPS = o_groups
    O_LORA = o_lora
    N_EXPERTS = n_experts
    TOP_K = top_k
    INTER = inter
    NOPE_DIM = HEAD_DIM - ROPE_DIM
    MOE_SLOTS = 1 + TOP_K
    SHARED_EXPERT = N_EXPERTS
    QKV_A_ROWS = qkv_a_rows(Q_LORA, HEAD_DIM, compress_ratio, c_coff, index_head_dim)
    N_QKV_A = QKV_A_ROWS // QKV_A_TILE
    N_ROW_TILES = HIDDEN // ROW_TILE
    N_ROUTER = N_EXPERTS // ROUTER_TILE
    N_UG_PER_SLOT = INTER // UG_TILE
    fmt = moe_format(moe_mode)
    use_fp8_block128 = fmt.activation is ExpertActivation.FP8_BLOCK128
    use_mxfp8_block32 = fmt.activation is ExpertActivation.MXFP8_BLOCK32
    use_mxfp4_weight = fmt.weight is ExpertWeight.MXFP4_BLOCK32
    XQ_BLOCKS = 0 if fmt.activation_group is None else HIDDEN // fmt.activation_group
    PUBLISH_BLOCKS = HIDDEN // (32 if use_mxfp8_block32 else 128)
    XQ_WAVES = (
        (PUBLISH_BLOCKS + N_ROUTER * 4 - 1) // (N_ROUTER * 4)
        if use_mxfp8_block32
        else (PUBLISH_BLOCKS + N_ROUTER - 1) // N_ROUTER
    )
    assert XQ_WAVES <= WAVES
    # --- KV compression. CR == 0 is sliding-window only and compiles out.
    # CSA's windows OVERLAP: an entry pools 2*CR tokens at a stride of CR, so the
    # projections carry two halves (C_COFF) and the state holds two windows.
    CR = compress_ratio
    C_COFF = c_coff
    # The lightning indexer keeps a SECOND compressor over the same tokens at its
    # own smaller head_dim, Hadamard-rotated and FP4-quantized. IHD == 0 is "no
    # indexer" and compiles the whole thing out.
    IHD = index_head_dim
    IW = C_COFF * IHD
    CW = C_COFF * HEAD_DIM  # width of one state row
    C_ROWS = C_COFF * CR  # rows of state
    OVERLAP = C_COFF > 1
    CBASE = CR if OVERLAP else 0  # the current window fills the second half
    # the window and the compressed entries share one cache, the compressed half
    # starting at `window`, so the attention gathers both from one index list
    CACHE_ROWS = window if window_rows is None else window_rows
    # the split walks the whole index list, not just the window: with compression
    # the gather reaches past the window into the compressed half of the cache
    N_KEYS = window if n_keys is None else n_keys
    if CR:
        assert CACHE_ROWS > window, "a compressing layer needs cache rows past the window"
        assert HEAD_DIM <= THREADS, "the compressor maps one thread per channel"

    # --- hyper-connections. HC == 1 is a plain residual and compiles all of this out.
    HC = hc_mult
    HC_TASKS, HC_KSLICE, HC_ROWS, HC_VALS = hc_shape(HC, HIDDEN)
    HC_MIX = (2 + HC) * HC if HC > 1 else 0
    HC_COEF = 2 * HC + HC * HC if HC > 1 else 0
    # comb lane = j * HC + k, so XORing the low bits walks a row and the high bits
    # a column -- the whole Sinkhorn is cross-lane inside one wave
    HC_ROW_OFFS = tuple(1 << b for b in range(HC.bit_length() - 1)) if HC > 1 else ()
    HC_COL_OFFS = tuple(HC << b for b in range(HC.bit_length() - 1)) if HC > 1 else ()
    HC_NKC = HC_KSLICE // 64 if HC > 1 else 0
    HC_RG = HC_ROWS // 16 if HC > 1 else 0
    HC_WPR = WAVES // HC_RG if HC > 1 else 0
    HC_NKC_FULL = (HC * HIDDEN) // 64 if HC > 1 else 0
    if HC > 1:
        assert HC & (HC - 1) == 0, "hc_mult must be a power of two for the cross-lane Sinkhorn"
        assert HC * HC <= 64, "the Sinkhorn matrix must fit one wave"
        assert HC_NKC % HC_WPR == 0, "hc_pre chunks must divide over the waves of a row group"

    down_scale_words = 0 if fmt.activation_group is None else S * MOE_SLOTS * INTER // fmt.activation_group
    # the router and up/gate read the contracted stream, which is a separate
    # mailbox once hyper-connections widen `a`
    A_IN = "ain" if hc_mult > 1 else "a"
    HC_MISC = 8 + max(S * XQ_BLOCKS, down_scale_words)
    # `uv` stores one per-split weight in misc, so it must hold N_SPLIT of them
    misc_words = max(HC_MISC + S * max(HC_COEF, 1), n_keys // SPLIT_KEYS)
    H = heads
    W = npes
    G = BLOCKS
    SC, SY = layout(
        S,
        H,
        W,
        window,
        moe_mode,
        hidden=HIDDEN,
        q_lora=Q_LORA,
        head_dim=HEAD_DIM,
        o_groups=O_GROUPS,
        o_lora=O_LORA,
        hc_mult=HC,
        compress_ratio=CR,
        n_keys=N_KEYS,
        c_coff=C_COFF,
        index_head_dim=IHD,
        index_heads=index_heads,
        index_topk=index_topk,
        max_seq=max_seq,
    )
    assert N_KEYS % SPLIT_KEYS == 0, "the index list must be a whole number of key tiles"
    N_SPLIT = N_KEYS // SPLIT_KEYS
    # The flash merge in `uv` gives each split ONE LANE of one wave: wave 0
    # computes every split's exp(m - M)/L into misc[split], and the merge then
    # reads misc[0 .. N_SPLIT). Past 64 splits the lanes run out, so the splits
    # above 64 get no weight, the softmax normalises over a prefix, and the read
    # runs past misc -- NaN at every position, silently. Reject it here.
    # This binds the context: n_keys <= 4096, so HCA (whose key list grows with
    # the sequence) tops out near max_seq 508K, while CSA is unaffected at any
    # length because index_topk pins its n_keys at 1152.
    # Largest chunk up to 8 that divides the split count: bounds how many
    # per-split words the `uv` merge holds live at once (see the loop below).
    UV_CHUNK = max(c for c in range(1, 9) if N_SPLIT % c == 0)
    assert N_SPLIT <= THREADS // WAVES, (
        f"{N_SPLIT} key splits needs a block-wide merge in `uv`; one wave holds {THREADS // WAVES}. "
        f"n_keys {N_KEYS} = window {window} + the compressed list, so this is a max_seq limit"
    )
    N_QB = H * HEAD_DIM // Q_B_TILE
    QB_PER_HEAD = HEAD_DIM // Q_B_TILE
    # the indexer's query: its own per-head projection off the SAME normed q_lora
    IH = index_heads if IHD else 0
    N_IQB = IH * IHD // Q_B_TILE
    IQB_PER_HEAD = IHD // Q_B_TILE if IHD else 0
    IH_TOTAL = index_heads_total or index_heads
    N_COMP = (max_seq // CR) if (IHD and CR) else 0
    N_INDEX = n_index(max_seq, compress_ratio, index_head_dim, index_topk)
    # the top-k holds every candidate in registers across the radix passes, so one
    # thread's share is bounded rather than the sequence length
    TOPK_PER = max(1, (N_COMP + THREADS - 1) // THREADS) if N_COMP else 1
    # the index list pads to a whole key tile, so its compressed half has room for
    # more than the indexer will ever pick; the surplus is filled with -1
    N_ISEL = N_KEYS - window
    N_ISCORE = n_score_tiles(max_seq, compress_ratio, index_head_dim)
    if IHD:
        assert window % SPLIT_KEYS == 0, "a key tile must fall wholly inside or outside the window"
        assert N_KEYS >= window + N_INDEX, "the index list must hold the window and the pick"
        assert IHD % 8 == 0 and N_COMP > 0
        assert IHD == 128, "the indexer's Hadamard is written for a 128-wide head"
        assert IH % WAVES == 0, "one wave takes a whole index head"
        assert IHD - ROPE_DIM == 64, "rope must fall entirely in the head's second half"
    N_UV = H * HEAD_DIM // UV_TILE
    UV_PER_HEAD = HEAD_DIM // UV_TILE
    OA_K = H * HEAD_DIM // O_GROUPS  # one group's slice of the concatenated heads
    N_OA = S * O_GROUPS * O_LORA // ROW_TILE
    OA_PER_GROUP = O_LORA // ROW_TILE
    OB_K = O_GROUPS * O_LORA
    # Packed selection key: the score's order-preserving bits with the low ID_BITS
    # replaced by ID_MASK - expert id, so keys are unique and near-ties go to the
    # lower id.  GLM-5/V3's 256 experts fit an 8-bit field exactly (255 - 255 == 0);
    # V4's 384 would make that term negative and flood the key with ones, so the
    # field is sized to the expert count.
    ID_BITS = max(8, (N_EXPERTS - 1).bit_length())
    ID_MASK = (1 << ID_BITS) - 1
    UG_PER_SLOT = INTER // UG8
    N_UG_TASKS = TOP_K * UG_PER_SLOT
    N_UG = S * MOE_SLOTS * N_UG_PER_SLOT
    # route weight normalisation sums lanes 0..TOP_K-1; lanes >= TOP_K hold 0, so the
    # butterfly just has to span the next power of two
    _off, TOPK_SUM_OFFS = 1, []
    while _off < TOP_K:
        TOPK_SUM_OFFS.append(_off)
        _off *= 2
    TOPK_SUM_OFFS = tuple(TOPK_SUM_OFFS)
    # V4's query is one contiguous HEAD_DIM vector per head with RoPE inside it,
    # so unlike MLA there is a single KV tile (no separate k_pe tile).
    QK_DIM = HEAD_DIM
    QS = QK_DIM // 2 + 4
    KS = HEAD_DIM // 2 + 4
    KT_OFF = H * QS
    XN = max(S * HIDDEN // 2, KT_OFF + SPLIT_KEYS * KS)
    ON = S * max(ROW_TILE, UV_TILE, QKV_A_TILE, UG_TILE * 2)
    DN_TILE = dn_tile(S, HIDDEN)
    N_DN_TILES = HIDDEN // DN_TILE

    st_args = dict(
        window=window,
        hidden=HIDDEN,
        q_lora=Q_LORA,
        head_dim=HEAD_DIM,
        o_groups=O_GROUPS,
        o_lora=O_LORA,
        top_k=TOP_K,
        inter=INTER,
        hc_mult=HC,
        compress_ratio=CR,
        n_keys=N_KEYS,
        c_coff=C_COFF,
        index_head_dim=IHD,
        index_heads=index_heads,
        index_topk=index_topk,
        max_seq=max_seq,
    )
    base, first, acc = {}, {}, 0
    for name, n in stage_tasks(S, H, **st_args):
        first[name] = acc
        acc += n
    # CTA placement: split before the q_b tiles it waits on land, so every split
    # tile starts on a CTA already freed by qkv_a
    tasks = dict(stage_tasks(S, H, **st_args))
    acc = 0
    for name in (
        "hcd_a",
        "hcc_a",
        "qkv_a",
        "cache",
        "cmp",
        "i_cmp",
        "split",
        "q_b",
        "q_norm",
        "i_q_b",
        "i_q",
        "i_wp",
        "i_score",
        "i_topk",
        "uv",
        "o_a",
        "o_b",
        "hcd_f",
        "hcc_f",
        "router",
        "ug",
        "down",
    ):
        base[name] = acc % G
        acc += tasks[name]

    @fx.struct
    class Smem:
        x: fx.Array[fx.Float32, XN, 16]  # bf16 activations (pairs) / split q + KV tile
        out: fx.Array[fx.Float32, ON, 16]
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, misc_words, 16]
        p: fx.Array[fx.Float32, H * SPLIT_KEYS, 16]
        keys: fx.Array[fx.Int32, SPLIT_KEYS, 16]
        dnw: fx.Array[fx.Float32, S * MOE_SLOTS, 16]  # expert-down route weights

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def dsv4_kernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        dest_rows: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        attn_sink: Int64,
        ape: Int64,
        g_ckv: Int64,
        kv_state: Int64,
        score_state: Int64,
        i_ape: Int64,
        g_ickv: Int64,
        i_kv_state: Int64,
        i_score_state: Int64,
        i_cache: Int64,
        hc_attn_fn: Int64,
        hc_attn_sb: Int64,
        hc_ffn_fn: Int64,
        hc_ffn_sb: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_i_q_b: Int64,
        s_i_q_b: Int64,
        i_w: Int64,
        w_o_a: Int64,
        s_o_a: Int64,
        w_o_b: Int64,
        s_o_b: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        state_slots: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        st_ic: Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64
        lds = fx.SharedAllocator().allocate(Smem).peek()
        xs = lds.x.ptr
        outs = lds.out.ptr
        red = lds.red.ptr
        misc = lds.misc.ptr
        pl = lds.p.ptr
        keys = lds.keys.ptr
        dnw = lds.dnw.ptr
        ktile = xs + KT_OFF  # f32-typed view holding raw bf16 pairs
        v4f = fx.Vector.make_type(4, fx.Float32)

        r_h = _rsrc(h_in)
        # this launch's epoch: every mailbox tag must equal it.  ``step`` is a
        # device counter bumped once per decode step (graph friendly); ``layer``
        # makes it unique per layer within the step.
        tag = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32)) * LAYER_SLOTS + layer + 1
        r_pos = _rsrc(cur_pos)

        def ld_pos(s):
            """Sample ``s``'s position. One entry per sample, not one scalar: the
            samples are independent sequences and a real batch has each at its own
            offset. Every caller derives ``s`` from its task index (or a constexpr
            loop), so the value is wave-uniform and the readfirstlane is honest."""
            return _uniform(bo.buffer_load(r_pos, s, vec_width=1, dtype=T.i32))

        r_dest = _rsrc(dest_rows)

        def ld_dest(j, s):
            """Plane rows for sample ``s``: j=0 the row this token's KV goes to,
            j=1 the row its compressed entry ZERO lives at (entry c is row1 + c).

            Supplied rather than derived, because the cache is ONE plane and which
            rows a sequence owns is the pool's arithmetic, not the kernel's -- a
            serving pool interleaves layers and relocates slots, so
            ``base + pos % window`` is not a formula this side can know. Both the
            write and the split's 'did we just write this row' test read the same
            value, so they cannot drift apart."""
            return _uniform(bo.buffer_load(r_dest, j * S + s, vec_width=1, dtype=T.i32))

        r_slot = _rsrc(state_slots)

        def ld_slot(s, stride):
            """Element offset of sample ``s``'s slice of a rolling-state pool.

            The slot is supplied, not the sample index: a pool hands slots out per
            sequence, they are not contiguous, and they move -- a request can be
            relocated, or fork its state. The stride is supplied too and is the
            whole ENTRY, not this field: a pool interleaves several fields inside
            one slot, so a kernel that assumed contiguity walks into its neighbour.
            """
            return _uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32)) * stride

        r_peers = _rsrc(peers)
        # Each wave sends to one peer, so retain only that wave's destination.
        pv = fx.Vector(bo.buffer_load(r_peers, fx.min(wave, W - 1) * 2, vec_width=2, dtype=T.i32))
        peer_dst = (fx.Int64(_uniform(pv[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv[0])))

        # ------------------------------------------------------------ helpers
        def ld_f32(r, i):
            return fx.Float32(bo.buffer_load(r, i, vec_width=1, dtype=T.f32))

        def ld_bf16(r, i):
            return fx.Float32(fx.BFloat16(bo.buffer_load(r, i, vec_width=1, dtype=T.bf16)))

        def lds_ld(ptr, i):
            return fx.ptr_load(ptr + i)

        def lds_st(ptr, i, v):
            fx.ptr_store(v, ptr + i)

        def bf16_pair(a, b):
            """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
            return fx.Vector.from_elements([a, b], fx.Float32).to(fx.BFloat16).bitcast(fx.Float32)[0]

        def bf16_round(a):
            return fx.Float32(fx.Float32(a).to(fx.BFloat16))

        # ---- tagged-pair mailboxes
        def mb(name):
            return scratch + fx.Int64(SC[name])

        def put(base_addr, i, v, cm=CM_DEV):
            """Pair i := (v, tag); ``v`` f32 (or int32 bits)."""
            bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
            bo.buffer_store(fx.Vector.from_elements([bits, tag], fx.Int32), _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put2(base_addr, i, v0, v1, cm=CM_DEV):
            """Pairs i, i+1 (i even) in one 16-byte store."""
            vec = fx.Vector.from_elements(
                [fx.Float32(v0).bitcast(fx.Int32), tag, fx.Float32(v1).bitcast(fx.Int32), tag], fx.Int32
            )
            bo.buffer_store(vec, _rsrc(base_addr), i * 2, cache_modifier=cm)

        def put_bf(base_addr, i, vs, cm=CM_DEV):
            """Elements i .. i + len(vs) (2 or 4, i aligned) as packed bf16 pairs: pair
            i / 2 + j := (bf16(vs[2j]) | bf16(vs[2j + 1]) << 16, tag), one 8 / 16-byte store."""
            words = []
            for j in range_constexpr(len(vs) // 2):
                words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), tag]
            bo.buffer_store(fx.Vector.from_elements(words, fx.Int32), _rsrc(base_addr), i, cache_modifier=cm)

        def bf2_f32(w):
            """Packed bf16 pair word -> (f32 low, f32 high)."""
            return (w << 16).bitcast(fx.Float32), (w & fx.Int32(-65536)).bitcast(fx.Float32)

        def _qptr(addr):
            return fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), fx.Int64(addr))

        def _ld_pair(addr, scope):
            """One (value, tag) pair as a single 64-bit relaxed atomic load: never hoisted,
            coherent at ``scope`` (agent -> sc1, system -> sc0 sc1)."""
            return fx.generic_load(_qptr(addr), memory_order=fx.AtomicOrdering.Monotonic, syncscope=scope)

        def poll(specs, scope="agent", batch=POLL_MAX):
            """Batched poll of mailbox pairs: ``specs`` = [(base_addr, pair index, npairs in {1, 2})].

            All pairs are loaded together with plain 8 / 16-byte coherent buffer loads
            (sc1 locally, sc0 sc1 for peer memory); while any tag is not this launch's
            the whole batch is re-loaded, so a batch costs one round trip after its
            last producer lands.  A side-effecting scheduling op in the retry loop
            keeps the loads from being hoisted.  Returns one list of Int32 value bits
            per spec."""
            if const_expr(len(specs) == 0):
                return []
            if const_expr(len(specs) > batch):  # bound live registers
                return poll(specs[:batch], scope, batch) + poll(specs[batch:], scope, batch)
            cm = CM_DEV if const_expr(scope == "agent") else CM_SYS

            def load_all():
                words = []
                for b, i, n in specs:
                    w = fx.Vector(
                        bo.buffer_load(_rsrc(b), fx.Int32(i) * 2, vec_width=2 * n, dtype=T.i32, cache_modifier=cm)
                    )
                    words += [w[e] for e in range(2 * n)]
                return fx.Vector.from_elements(words, fx.Int32)

            nw = sum(2 * n for _, _, n in specs)

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            v = load_all()
            while pending(v):
                rocdl.s_nop(0)
                v = load_all()
            outs_, e = [], 0
            for _, _, n in specs:
                outs_.append([v[e + 2 * q] for q in range(n)])
                e += 2 * n
            return outs_

        def hint_wait(n, addr_of, mark=None):
            """Consumers poll their payload directly (tight per-wave spins); a wave-0
            pre-poll of each producer's last pair only added a hop of latency."""
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 1)
            gpu.barrier()

        def pre_poll(n, addr_of):
            """Wave 0 spins on one small pair per producer (lane j -> producer j < n <= 64)
            before a large payload poll, so waiting CTAs do not flood memory."""
            if wave == 0:
                b, i = addr_of(fx.min(lane, n - 1))
                poll([(b, i, 1)])
            gpu.barrier()

        def get(base_addr, i):
            return poll([(base_addr, i, 1)])[0][0]

        def getf(base_addr, i):
            return get(base_addr, i).bitcast(fx.Float32)

        def getf_many(specs):
            """[(base, i)] single pairs -> list of f32."""
            return [v[0].bitcast(fx.Float32) for v in poll([(b, i, 1) for b, i in specs])]

        def get2_many(specs):
            """[(base, i)] double pairs (i even) -> list of (f32, f32)."""
            return [(v[0].bitcast(fx.Float32), v[1].bitcast(fx.Float32)) for v in poll([(b, i, 2) for b, i in specs])]

        def get2(base_addr, i):
            return get2_many([(base_addr, i)])[0]

        def get_bf2_many(specs):
            """[(base, i)] packed bf16 elements i, i + 1 (i even) -> list of (f32, f32)."""
            return [bf2_f32(v[0]) for v in poll([(b, i // 2, 1) for b, i in specs])]

        # ---- wave reductions
        def wave_sum(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, lambda a, b: a + b)
            return v

        def wave_max(v):
            for sh in range_constexpr(6):
                v = _xred(v, 32 >> sh, fx.max)
            return v

        def subgroup16_max(v):
            for off in (8, 4, 2, 1):
                v = _xred(v, off, fx.max)
            return v

        def block_isum(v):
            """Block-wide sum of a per-thread int32."""
            w = wave_sum(fx.Float32(v))
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return fx.Int32(t)

        def block_excl_scan(v):
            """Exclusive prefix sum of a per-thread int32 over the block, and the
            block total.

            Butterfly scan: before step ``off`` every lane holds the sum of its
            aligned ``off``-wide block, so a lane in the upper half of the next
            block adds the lower half's sum to its prefix, and both halves add each
            other to stay a block sum. Six xor shuffles cover the wave; the WAVES
            wave totals then combine through LDS. This replaces the O(THREADS) form
            -- one LDS write, then every thread walking all the counts before it --
            which at THREADS = 512 was a 512-deep unrolled LDS walk per thread and
            cost 139 us of the layer.
            """
            x = fx.Float32(v)
            pre = fx.Float32(0.0)
            for sh in range_constexpr(6):
                off = 1 << sh
                p = _xshfl(x, off)
                pre = ((lane & off) != 0).select(pre + p, pre)
                x = x + p  # every lane now holds the sum of its 2 * off block
            if lane == 0:  # x is the wave total in every lane
                lds_st(red, wave, x)
            gpu.barrier()
            tot = fx.Float32(0.0)
            for i in range_constexpr(WAVES):
                t = lds_ld(red, i)
                pre = (fx.Int32(i) < wave).select(pre + t, pre)
                tot = tot + t
            gpu.barrier()
            return fx.Int32(pre), fx.Int32(tot)

        def block_sums(vs):
            """Block-wide sums of several per-thread values with one LDS exchange."""
            ws = [wave_sum(v) for v in vs]
            if lane == 0:
                for i in range_constexpr(len(vs)):
                    lds_st(red, i * WAVES + wave, ws[i])
            gpu.barrier()
            tots = []
            for i in range_constexpr(len(vs)):
                t = lds_ld(red, i * WAVES)
                for w in range_constexpr(1, WAVES):
                    t = t + lds_ld(red, i * WAVES + w)
                tots.append(t)
            gpu.barrier()
            return tots

        def block_sum(v):
            w = wave_sum(v)
            if lane == 0:
                lds_st(red, wave, w)
            gpu.barrier()
            t = lds_ld(red, 0)
            for i in range_constexpr(1, WAVES):
                t = t + lds_ld(red, i)
            gpu.barrier()
            return t

        # ------------------------------------------------ MFMA GEMV machinery
        def unit_fp8(w_rsrc, s_rsrc, rg, kc, NKC, K, BK, b_word, coef=None, ln=None):
            """Issue one 64-k chunk of row group ``rg`` of a packed FP8 matrix; the
            bf16 activation chunk starts at LDS word ``b_word``."""
            ln = lane if ln is None else ln
            wv = fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // BK) + kc * 64 // BK)
            if const_expr(callable(coef)):  # factor known only after a later wait
                return ("fp8", [wv], lambda: s * coef(), b_word + (lane // 16) * 4)
            if const_expr(coef is not None):
                s = s * coef
            return ("fp8", [wv], s, b_word + (lane // 16) * 4)

        def unit_f8f8(w_rsrc, s_rsrc, rg, kc, NKC, K, b_word, coef, ln=None):
            """Issue one 128-k chunk (packed 64-k chunks kc, kc + 1; kc even) of row group
            ``rg`` against the FP8 activation of LDS words ``b_word`` + [0, 32) (``f8_word``
            order); ``coef()`` = activation block scale (times route weight).  ``ln``
            = the lane whose weights are loaded (default: own lane)."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, ((rg * NKC + kc + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (rg * 16 // SCALE_BM) * (K // 128) + kc // 2)
            return ("f8f8", wv, lambda: s * coef(), b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
            """Issue one packed 128-K MXFP4 tile and its four per-row E8M0 scales."""

            ln = lane if ln is None else ln
            raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            row = rg * 16 + ln % 16
            packed_scale = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32))
            scales = [
                ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                for sp in range_constexpr(4)
            ]
            return ("mxfp4", (raw, scales), coef, b_word + (lane // 16) * 4)

        def unit_bf16(w_rsrc, rg, kc, NKC, b_word, ln=None):
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(bo.buffer_load(w_rsrc, (((rg * NKC + kc) * 2 + sp) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
                for sp in range(2)
            ]
            return ("bf16", wv, None, b_word + (lane // 16) * 4)

        def mma_units(acc, units):
            """acc[4] += coef * (W_chunk @ X_chunk) for every issued unit."""
            for unit_format, wv, coef, bw in units:
                if const_expr(callable(coef) and unit_format != "mxfp4"):
                    coef = coef()
                if const_expr(unit_format == "mxfp4"):
                    raw, scales = wv
                    for sp in range_constexpr(4):
                        a = _mxfp4_to_bf16x8(raw[sp], scales[sp])
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector.filled(4, 0.0, fx.Float32)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                        part_coef = coef[sp] if const_expr(isinstance(coef, list)) else coef
                        if const_expr(callable(part_coef)):
                            part_coef = part_coef()
                        if const_expr(part_coef is None):
                            acc = [acc[e] + c[e] for e in range(4)]
                        else:
                            acc = [acc[e] + c[e] * part_coef for e in range(4)]
                    continue
                c = fx.Vector.filled(4, 0.0, fx.Float32)
                if const_expr(unit_format == "f8f8"):  # one FP8 x FP8 MFMA (E8M0 scales = 1)
                    a = fx.Vector.from_elements([wv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                    bv = [
                        fx.Vector(fx.ptr_load(xs + (bw + h * 16), result_type=v4f)).bitcast(fx.Int32) for h in range(2)
                    ]
                    b = fx.Vector.from_elements([bv[h][e] for h in range(2) for e in range(4)], fx.Int32)
                    one = fx.Int32(127)
                    c = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(T.vec(4, T.f32), [a, b, c, 0, 0, 0, one, 0, one])
                    )
                for sp in range_constexpr(2 if unit_format != "f8f8" else 0):
                    if const_expr(unit_format == "fp8"):
                        a = _fp8_to_bf16x8(wv[0][sp * 2], wv[0][sp * 2 + 1])
                    else:
                        a = wv[sp].bitcast(fx.BFloat16)
                    b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                    c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                if const_expr(coef is None):
                    acc = [acc[e] + c[e] for e in range(4)]
                else:
                    acc = [acc[e] + c[e] * coef for e in range(4)]
            return acc

        def run_units(make_unit, cpw, batch, pre=None):
            """Software pipelined: issue batch b+1's loads before computing batch b.
            ``pre`` = the already-issued first batch (prefetched before a wait)."""
            acc = [fx.Float32(0.0) for _ in range(4)]
            starts = list(range(0, cpw, batch))
            cur = pre if pre is not None else [make_unit(c) for c in range(0, min(batch, cpw))]
            for bi in range_constexpr(len(starts)):
                nxt = None
                if const_expr(bi + 1 < len(starts)):
                    n0 = starts[bi + 1]
                    nxt = [make_unit(c) for c in range(n0, min(n0 + batch, cpw))]
                acc = mma_units(acc, cur)
                cur = nxt
            return acc

        def reduce_rows(R, acc, emit):
            """Sum the per-wave MFMA tiles of each of R row groups; emit(row_local, n, v) for n < S."""
            wpr = WAVES // R
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            n_out = R * 16 * S
            for i in range_constexpr((n_out + THREADS - 1) // THREADS):
                t = tid + i * THREADS
                if t < n_out:
                    rl = t % (R * 16)
                    n = t // (R * 16)
                    r = rl % 16
                    tot = fx.Float32(0.0)
                    for j in range_constexpr(wpr):
                        ww = (rl // 16) * wpr + j
                        tot = tot + lds_ld(red, (ww * 64 + n + 16 * (r // 4)) * 4 + r % 4)
                    emit(rl, n, tot)

        def emit_out(stride):
            def f(rl, n, v):
                lds_st(outs, n * stride + rl, v)

            return f

        def _rmsnorm_tail_ks(n):
            """This thread's n // 4 group-of-4 starting element indices.

            n need only be a multiple of 4 (not 4 * THREADS): when n // 4 isn't a
            multiple of THREADS, the last entry is clamped to the final real group
            instead of running out of bounds, so threads past the end redundantly
            reread/rewrite that group (idempotent: same k, same value, everywhere).
            ``active`` masks that lane's contribution to a cross-lane sum (writes
            need no mask since the redundant write is idempotent); None when n // 4
            is an exact multiple of THREADS (no tail)."""
            nq = n // 4
            full = nq // THREADS
            ks = [(tid + i * THREADS) * 4 for i in range(full)]
            active = None
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                active = w < nq
                ks.append(fx.min(w, nq - 1) * 4)
            return ks, active

        def stage_x_rmsnorm(ld4s, n, gamma, mark=None, loaded=None, count=S):
            """LDS bf16 X[s][0:n] = bf16(rmsnorm(x_s) * gamma) for every sample s, where
            ld4s([(s, k)]) -> [(x_s[k], .., x_s[k+3])] (one batched load); returns the rstds.
            ``loaded``: the (gamma, x) loads already issued by load_x_rmsnorm."""
            ks, active = _rmsnorm_tail_ks(n)
            per = len(ks)
            gs, vals = loaded if loaded is not None else load_x_rmsnorm(ld4s, n, gamma, count)
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
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 6)
            rstds = [_rsq(tot * (1.0 / n) + EPS) for tot in block_sums(sss)]
            if const_expr(mark is not None):
                stamp(mark[0], mark[1], 7)
            for s in range_constexpr(count):
                for i in range_constexpr(per):
                    a = vals[s * per + i]
                    for j in range_constexpr(2):
                        lds_st(
                            xs,
                            (s * n + ks[i]) // 2 + j,
                            bf16_pair(a[2 * j] * rstds[s] * gs[i][2 * j], a[2 * j + 1] * rstds[s] * gs[i][2 * j + 1]),
                        )
            return rstds

        def load_x_rmsnorm(ld4s, n, gamma, count=S):
            """The gamma loads (issued ahead of the wait), then ld4s -> (gammas, x values)."""
            rg_ = _rsrc(gamma)
            ks, _ = _rmsnorm_tail_ks(n)
            gs = []
            for k in ks:
                g = fx.Vector(bo.buffer_load(rg_, k // 2, vec_width=2, dtype=T.i32)).bitcast(fx.BFloat16).to(fx.Float32)
                gs.append([g[j] for j in range(4)])
            return gs, ld4s([(s, k) for s in range(count) for k in ks])

        def stage_x_pairs(name, n_total, src_of):
            """LDS bf16 X[k] = packed bf16 mailbox ``name`` element src_of(k) for k < n_total
            (src_of contiguous over aligned groups of 4): one 16-byte poll per 4 elements."""
            nq = n_total // 4
            full = nq // THREADS
            vals = poll([(mb(name), src_of((tid + i * THREADS) * 4) // 2, 2) for i in range(full)])
            for i in range_constexpr(full):
                for j in range_constexpr(2):
                    lds_st(xs, (tid + i * THREADS) * 2 + j, vals[i][j].bitcast(fx.Float32))
            if const_expr(nq % THREADS):
                w = tid + full * THREADS
                if w < nq:
                    v = poll([(mb(name), src_of(w * 4) // 2, 2)])[0]
                    for j in range_constexpr(2):
                        lds_st(xs, w * 2 + j, v[j].bitcast(fx.Float32))

        def quant_scaled(a0, a1):
            """Per-wave FP8 quant of a 128-block held as 2 f32 per lane -> (scaled q0, q1, scale)."""
            amax = wave_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
            inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))  # hardware rcp, no IEEE divide
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            return q0, q1, qs

        def quant_block(a0, a1):
            """quant_scaled, values returned as the FP8-rounded f32s."""
            q0, q1, qs = quant_scaled(a0, a1)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, qs

        def quant_mxfp8(a0, a1):
            """Per-16-lane/32-value MXFP8 quantization with an E8M0 scale."""

            amax = subgroup16_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            raw_scale = amax * (1.0 / FP8_MAX)
            bits = raw_scale.bitcast(fx.Int32)
            exponent = (bits.shrui(fx.Int32(23))) & fx.Int32(0xFF)
            round_up = ((bits & fx.Int32(0x400000)) != 0) & (
                ((bits & fx.Int32(0x200000)) != 0) | ((bits & fx.Int32(0x1FFFFF)) != 0) | (exponent > 0)
            )
            exponent = exponent + round_up.select(fx.Int32(1), fx.Int32(0))
            scale = nz.select((exponent << fx.Int32(23)).bitcast(fx.Float32), fx.Float32(1.0))
            inv = nz.select(_rcp(scale), fx.Float32(1.0))
            q0 = fx.min(fx.max(a0 * inv, -FP8_MAX), FP8_MAX)
            q1 = fx.min(fx.max(a1 * inv, -FP8_MAX), FP8_MAX)
            d0, d1 = _fp8_roundtrip(q0, q1)
            return d0, d1, scale

        def stage_moe_input(samples):
            """Stage normalized expert inputs published by the router into LDS."""
            if const_expr(use_fp8_block128):
                # HIDDEN // 4 slots (one poll pair each, 4 packed FP8 bytes per slot);
                # when that isn't a multiple of THREADS the last slot is clamped
                # (redundant, idempotent re-read/rewrite -- see _rmsnorm_tail_ks).
                nq = HIDDEN // 4
                full = nq // THREADS
                xk = [tid + i * THREADS for i in range(full)]
                if const_expr(nq % THREADS):
                    xk.append(fx.min(tid + full * THREADS, nq - 1))
                nxw = len(xk)
                got = poll(
                    [(mb("xq"), sx * nq + k, 1) for sx in samples for k in xk]
                    + [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
                )
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(nxw):
                        wd = f8_word(xk[i] * 4)
                        lds_st(xs, j * nq + wd, got[j * nxw + i][0].bitcast(fx.Float32))
                    if tid < XQ_BLOCKS:
                        lds_st(
                            misc,
                            8 + j * XQ_BLOCKS + tid,
                            got[len(samples) * nxw + j][0].bitcast(fx.Float32),
                        )
            elif const_expr(use_mxfp8_block32):
                chunks = HIDDEN // 8
                per_thread = (chunks + THREADS - 1) // THREADS
                data_specs = []
                for sx in samples:
                    for i in range_constexpr(per_thread):
                        chunk = fx.min(tid + i * THREADS, chunks - 1)
                        data_specs.append((mb("xq"), sx * (HIDDEN // 4) + chunk * 2, 2))
                scale_specs = [(mb("xqs"), sx * XQ_BLOCKS + fx.min(tid, XQ_BLOCKS - 1), 1) for sx in samples]
                got = poll(data_specs + scale_specs)
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(per_thread):
                        chunk = tid + i * THREADS
                        if chunk < chunks:
                            words = got[j * per_thread + i]
                            values = _fp8_to_bf16x8(words[0], words[1])
                            for pair in range_constexpr(4):
                                lds_st(
                                    xs,
                                    j * (HIDDEN // 2) + chunk * 4 + pair,
                                    fx.Vector.from_elements(
                                        [values[2 * pair], values[2 * pair + 1]], fx.BFloat16
                                    ).bitcast(fx.Float32)[0],
                                )
                    if tid < XQ_BLOCKS:
                        lds_st(
                            misc,
                            8 + j * XQ_BLOCKS + tid,
                            got[len(samples) * per_thread + j][0].bitcast(fx.Float32),
                        )
            else:
                nxw = HIDDEN // 2 // THREADS
                got = poll(
                    [(mb("xq"), sx * (HIDDEN // 2) + tid + i * THREADS, 1) for sx in samples for i in range(nxw)]
                )
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(nxw):
                        lds_st(
                            xs,
                            j * (HIDDEN // 2) + tid + i * THREADS,
                            got[j * nxw + i][0].bitcast(fx.Float32),
                        )

        def st_f8(k, q0, q1):
            """LDS FP8 activation bytes k, k + 1 (k even, held by this lane; lane ^ 1 holds
            k ^ 2) in ``f8_word`` order.  Call from the whole wave."""
            w = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
            nb = _xshfl(w, 1)
            if lane % 2 == 0:
                lds_st(xs, f8_word(k), (w | (nb << 16)).bitcast(fx.Float32))

        def load_bias():
            """This lane's 4 expert biases (issue before the scores wait)."""
            return [ld_f32(_rsrc(bias), lane + i * 64) for i in range(N_EXPERTS // 64)]

        def route_topk(s, raws=None, bs=None):
            """Top-``TOP_K`` of sample s (call from one whole wave, after the router
            scores landed).

            Packed-key argmax: key = order-preserving bits of (score + bias) with the
            low ID_BITS replaced by ID_MASK - expert id (unique; near-ties go to the
            lower id),
            so each round is one u32 wave max (candidate i of this lane is expert
            lane + 64 i).  V4 has no group-limited routing -- selection is flat over all
            experts.  Returns (expert id, route weight = raw score / sum of the TOP_K
            raw scores * ROUTE_SCALE) of pick ``lane`` in score order, valid in
            lanes < TOP_K."""
            KPL = N_EXPERTS // 64  # selection keys held per lane
            if const_expr(bs is None):
                bs = load_bias()
            if const_expr(raws is None):
                raws = getf_many([(mb("scores"), s * N_EXPERTS + lane + i * 64) for i in range(KPL)])
                stamp("ug", bid, 7)
            ks = []
            for i in range_constexpr(KPL):
                kb = (raws[i] + bs[i]).bitcast(fx.Int32)
                ok = (kb >= 0).select(kb ^ fx.Int32(-(2**31)), ~kb)
                ks.append(fx.Uint32((ok & fx.Int32(-(1 << ID_BITS))) | (ID_MASK - (lane + i * 64))))
            # sort this lane's KPL keys descending; each round then takes the wave max of
            # the lane heads and shifts the winning lane's list (0 is below every key)
            for a, b in _sort_network(KPL):
                ks[a], ks[b] = fx.max(ks[a], ks[b]), fx.min(ks[a], ks[b])
            ks = [fx.Int32(k) for k in ks] + [fx.Int32(0)]
            mv = fx.Int32(0)  # lane k: the key of pick k
            for k in range_constexpr(TOP_K):
                m = _wave_umax(ks[0])
                hit = ks[0] == m
                ks = [hit.select(ks[i + 1], ks[i]) for i in range(KPL)] + [ks[KPL]]
                mv = fx.Int32(
                    llvm.call_intrinsic(
                        T.i32,
                        "llvm.amdgcn.writelane.i32",
                        [m.ir_value(), fx.Int32(k).ir_value(), mv.ir_value()],
                        [],
                        [],
                    )
                )
            e = ID_MASK - (mv & ID_MASK)
            src = (e % 64) * 4
            got = [fx.Int32(rocdl.ds_bpermute(T.i32, src.ir_value(), r.bitcast(fx.Int32).ir_value())) for r in raws]
            raw = got[0]
            for i in range_constexpr(1, N_EXPERTS // 64):
                raw = (e // 64 == i).select(got[i], raw)
            raw = (lane < TOP_K).select(raw.bitcast(fx.Float32), fx.Float32(0.0))
            tot = raw
            for off in TOPK_SUM_OFFS:
                tot = _xred(tot, off, lambda a, b: a + b)
            return e, raw * (_rcp(tot) * ROUTE_SCALE)

        def peer_reduce(region, t, residual, out_fn, tile=ROW_TILE):
            """Push BF16 partials in tagged pairs to every peer, then sum all
            ranks' pairs from the own symmetric buffer in rank order (W = 1: no exchange).  ``residual`` is
            either fn(s, row) -> (r0, r1) (plain loads, issued first), a mailbox base
            (pairs s * HIDDEN + row, polled in the same batch as the peers), or None
            when the caller adds its own -- hyper-connection hc_post does."""
            if const_expr(W > 1):
                # One wave per destination: peer pointers are wave-uniform, and the
                # destinations progress concurrently instead of eight serial stores
                # from the output wave. All waves consume outs before it is reused.
                if wave < W:
                    pair_count = S * tile // 2
                    for batch in range_constexpr((pair_count + 63) // 64):
                        pair = lane + batch * 64
                        if pair < pair_count:
                            si = pair // (tile // 2)
                            ri = (pair % (tile // 2)) * 2
                            put_bf(
                                peer_dst + fx.Int64(SY[region]),
                                (rank * S + si) * HIDDEN + t * tile + ri,
                                [lds_ld(outs, si * tile + ri), lds_ld(outs, si * tile + ri + 1)],
                                CM_SYS,
                            )
                gpu.barrier()
            if tid < S * tile // 2:
                s = tid // (tile // 2)
                r = (tid % (tile // 2)) * 2
                row = t * tile + r
                r0 = fx.Float32(0.0)
                r1 = fx.Float32(0.0)
                if const_expr(callable(residual)):
                    r0, r1 = residual(s, row)
                v0 = lds_ld(outs, s * tile + r)
                v1 = lds_ld(outs, s * tile + r + 1)
                if const_expr(W == 1):  # no TP peers: the sum is the local value
                    parts = [(v0, v1)]
                    got = []
                    if const_expr(residual is not None and not callable(residual)):
                        got = poll([(residual, (s * HIDDEN + row) // 2, 1)])
                else:
                    own = sym + fx.Int64(SY[region])
                    specs = [(own, ((src * S + s) * HIDDEN + row) // 2, 1) for src in range(W)]
                    if const_expr(residual is not None and not callable(residual)):  # packed bf16 pair
                        specs.append((residual, (s * HIDDEN + row) // 2, 1))
                    got = poll(specs, "one-as")
                    parts = [bf2_f32(v[0]) for v in got[:W]]
                    got = got[W:]
                if const_expr(residual is not None and not callable(residual)):
                    r0, r1 = bf2_f32(got[0][0])
                t0 = fx.Float32(0.0)
                t1 = fx.Float32(0.0)
                for src in range_constexpr(W):
                    t0 = t0 + parts[src][0]
                    t1 = t1 + parts[src][1]
                out_fn(s, row, r0 + t0, r1 + t1)

        def start(name):
            return (bid + (G - base[name])) & (G - 1)

        def stamp(name, t, which, lead=0, pred=None):
            if const_expr(timeline):
                # Nothing may cross the clock read. Without this the scheduler
                # sinks plain VALU work past the stamp -- it only has to respect
                # memory side effects -- and the phase that follows is charged for
                # it. That reported i_score's exchange at 233 us when the exchange
                # was 8 and the 233 was the scoring loop's own tail. Compiler-only,
                # and the whole body is compiled out when timeline is off -- but it
                # does constrain the scheduler in the timeline build, so read the
                # phase SPLIT from here and the layer total from a timeline-off run.
                rocdl.sched_barrier(0)
                # ``pred`` false means this task is a masked-off rep that shares a
                # row with a live one; letting it stamp would overwrite that task's
                # real timing with a time it never spent.
                ok = (tid == lead) if const_expr(pred is None) else ((tid == lead) & pred)
                if ok:
                    now = fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                    fx.generic_store(
                        fx.inttoptr(
                            fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8),
                            timeline_buf + fx.Int64((first[name] + t) * TL_COLS + which) * 8,
                        ),
                        now,
                    )

        def n_sel():
            """This lane's MFMA B column (sample); columns >= S duplicate the last one."""
            return fx.min(lane % 16, S - 1)

        # ======== 0. hyper-connection pre-mix: partial dots + partial sum of squares
        # hc_pre projects the whole hc*hidden residual stream onto (2+hc)*hc mixes.
        # Only 24 rows but K = hc*hidden, so K is split across tasks; each task also
        # carries a partial sum of squares for hc_pre's weightless RMS, published as
        # one extra row so the reduce below is a single poll per row.
        def hc_stage_coef(sd):
            """This sample's pre | post | comb into LDS, so the row loops below read
            coefficients from LDS instead of re-polling the mailbox per row."""
            if tid < S * HC_COEF:
                s_ = tid // HC_COEF
                lds_st(misc, HC_MISC + tid, getf(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + tid % HC_COEF))
            gpu.barrier()

        def hc_post(s, row, v0, v1, res_word, emit):
            """out[k] = post[k] * x + sum_j comb[j, k] * residual[j], for a row pair.

            The residual streams are read once and reused across k, so this costs
            hc reads rather than hc * hc.

            ``v0``/``v1`` arrive as the f32 sum of the peer partials, but the model
            this reproduces rounds there: its row-parallel projection all-reduces in
            f32 and returns bf16, so hc_post sees bf16. Keeping f32 here is *more*
            precise than the reference and shows up as a growing mismatch against it
            as the rank count rises, so round to match.
            """
            v0 = bf16_round(v0)
            v1 = bf16_round(v1)
            rj = [bf2_f32(res_word(s, j, row)) for j in range_constexpr(HC)]
            for k in range_constexpr(HC):
                pk = lds_ld(misc, HC_MISC + s * HC_COEF + HC + k)
                o0 = pk * v0
                o1 = pk * v1
                for j in range_constexpr(HC):
                    cjk = lds_ld(misc, HC_MISC + s * HC_COEF + 2 * HC + j * HC + k)
                    o0 = o0 + cjk * rj[j][0]
                    o1 = o1 + cjk * rj[j][1]
                emit(s, k, row, o0, o1)

        def hc_pre_stages(side, sd, fn_ptr, sb_ptr, src_word, out_name):
            r_fn = _rsrc(fn_ptr)
            r_sb = _rsrc(sb_ptr)  # [3 scales | HC_MIX bases]
            for tt in range(start(f"hcd_{side}"), S * HC_TASKS, G):
                tt = fx.Int32(tt)
                stamp(f"hcd_{side}", tt, 0)
                s = tt // HC_TASKS
                t = tt % HC_TASKS
                w = src_word(s, t * HC_KSLICE + tid * 2)
                lds_st(xs, tid, w.bitcast(fx.Float32))
                x0, x1 = bf2_f32(w)
                ssq = block_sum(x0 * x0 + x1 * x1)
                stamp(f"hcd_{side}", tt, 2)
                gpu.barrier()

                def u_hc(c, t=t):
                    kc = (wave % HC_WPR) * (HC_NKC // HC_WPR) + c
                    return unit_bf16(r_fn, wave // HC_WPR, t * HC_NKC + kc, HC_NKC_FULL, kc * 32)

                acc = run_units(u_hc, HC_NKC // HC_WPR, HC_NKC // HC_WPR)
                reduce_rows(HC_RG, acc, emit_out(HC_ROWS))
                stamp(f"hcd_{side}", tt, 3)
                gpu.barrier()
                base_i = ((s * 2 + sd) * HC_TASKS + t) * HC_VALS
                if tid < HC_ROWS:
                    put(mb("hc_d"), base_i + tid, lds_ld(outs, tid))
                if tid == HC_ROWS:
                    put(mb("hc_d"), base_i + HC_ROWS, ssq)
                stamp(f"hcd_{side}", tt, 4)

            # --- coefficients: reduce the partials, take the RMS scale, run the
            # Sinkhorn.  Computed inside hcc rather than as its own stage: mHC's cost
            # is dominated by the number of pipeline stages it adds (hc_mult=2 costs
            # 84% of hc_mult=4 despite half the data), so paying this redundantly per
            # hcc task -- they run concurrently, so it costs latency once -- is
            # cheaper than an extra grid-wide dependency.  Task 0 publishes them for
            # the hc_post epilogues, which need post / comb but no rows of their own.
            def hc_coefficients(sd, publish):
                """Reduce the hcd partials, take the RMS scale, run the Sinkhorn.

                Computed inside hcc rather than as its own stage: mHC's cost is
                dominated by how many pipeline stages it adds (hc_mult=2 costs 84% of
                hc_mult=4 despite carrying half the data), so paying this redundantly
                per hcc task -- they run concurrently, so it costs latency once -- is
                cheaper than another grid-wide dependency. Task 0 also publishes the
                coefficients for the hc_post epilogues, which need post / comb but
                have no rows of their own.

                comb must sit on lanes [0, HC * HC) so the Sinkhorn's XOR offsets stay
                inside it -- the low bits walk a row, the high bits a column.
                """
                sc0 = ld_f32(r_sb, 0)
                sc1 = ld_f32(r_sb, 1)
                sc2 = ld_f32(r_sb, 2)
                for blk in range_constexpr((S * HC_VALS + 63) // 64):
                    idx = lane + blk * 64
                    if wave == 0 and idx < S * HC_VALS:
                        s_ = idx // HC_VALS
                        j = idx % HC_VALS
                        parts = poll(
                            [(mb("hc_d"), ((s_ * 2 + sd) * HC_TASKS + i) * HC_VALS + j, 1) for i in range(HC_TASKS)]
                        )
                        tot = fx.Float32(0.0)
                        for i in range_constexpr(HC_TASKS):
                            tot = tot + parts[i][0].bitcast(fx.Float32)
                        lds_st(red, idx, tot)
                gpu.barrier()
                if wave == 0:
                    for s_ in range_constexpr(S):
                        rstd = _rsq(lds_ld(red, s_ * HC_VALS + HC_ROWS) * (1.0 / (HC * HIDDEN)) + EPS)

                        def coef(i, s_=s_, rstd=rstd):
                            return lds_ld(red, s_ * HC_VALS + i) * rstd

                        if lane < 2 * HC:  # pre then post share these lanes
                            m = coef(fx.min(lane, fx.Int32(HC_MIX - 1)))
                            b = ld_f32(r_sb, 3 + lane)
                            v = (lane < HC).select(
                                _rcp(1.0 + _exp(-(m * sc0 + b))) + hc_eps,
                                2.0 * _rcp(1.0 + _exp(-(m * sc1 + b))),
                            )
                            lds_st(misc, HC_MISC + s_ * HC_COEF + lane, v)
                            if publish:
                                put(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + lane, v)
                        if lane < HC * HC:
                            cb = coef(2 * HC + lane) * sc2 + ld_f32(r_sb, 3 + 2 * HC + lane)
                            rmax = cb
                            for off in HC_ROW_OFFS:
                                rmax = _xred(rmax, off, fx.max)
                            c = _exp(cb - rmax)
                            rsum = c
                            for off in HC_ROW_OFFS:
                                rsum = _xred(rsum, off, lambda a, b: a + b)
                            c = c * _rcp(rsum) + hc_eps
                            csum = c
                            for off in HC_COL_OFFS:
                                csum = _xred(csum, off, lambda a, b: a + b)
                            c = c * _rcp(csum + hc_eps)
                            for _ in range_constexpr(hc_sinkhorn_iters - 1):
                                rsum = c
                                for off in HC_ROW_OFFS:
                                    rsum = _xred(rsum, off, lambda a, b: a + b)
                                c = c * _rcp(rsum + hc_eps)
                                csum = c
                                for off in HC_COL_OFFS:
                                    csum = _xred(csum, off, lambda a, b: a + b)
                                c = c * _rcp(csum + hc_eps)
                            lds_st(misc, HC_MISC + s_ * HC_COEF + 2 * HC + lane, c)
                            if publish:
                                put(mb("hc_c"), (s_ * 2 + sd) * HC_COEF + 2 * HC + lane, c)
                gpu.barrier()

            # --- contract the hc_mult streams by `pre` into the single-width input
            for t in range(start(f"hcc_{side}"), N_ROW_TILES, G):
                t = fx.Int32(t)
                stamp(f"hcc_{side}", t, 0)
                hc_coefficients(sd, t == 0)
                stamp(f"hcc_{side}", t, 2)
                if tid < S * ROW_TILE // 2:
                    s_ = tid // (ROW_TILE // 2)
                    r = (tid % (ROW_TILE // 2)) * 2
                    row = t * ROW_TILE + r
                    a0 = fx.Float32(0.0)
                    a1 = fx.Float32(0.0)
                    for j in range_constexpr(HC):
                        pj = lds_ld(misc, HC_MISC + s_ * HC_COEF + j)
                        x0, x1 = bf2_f32(src_word(s_, j * HIDDEN + row))
                        a0 = a0 + pj * x0
                        a1 = a1 + pj * x1
                    put_bf(mb(out_name), s_ * HIDDEN + row, [a0, a1])
                stamp(f"hcc_{side}", t, 4)

        if const_expr(HC > 1):
            hc_pre_stages(
                "a",
                0,
                hc_attn_fn,
                hc_attn_sb,
                lambda s, k: fx.Int32(bo.buffer_load(r_h, (s * HC * HIDDEN + k) // 2, vec_width=1, dtype=T.i32)),
                "xin",
            )

        # ================================================= 1. q_a / kv GEMV
        # 1 row group x (HIDDEN / 64) chunks: 8 waves split K (all prefetched)
        r_wqa, r_sqa = _rsrc(w_qkv_a), _rsrc(s_qkv_a)
        QA_NKC = HIDDEN // 64
        for t in range(start("qkv_a"), N_QKV_A, G):
            t = fx.Int32(t)
            stamp("qkv_a", t, 0)

            def u_qa(c):
                kc = wave * (QA_NKC // WAVES) + c
                return unit_fp8(r_wqa, r_sqa, t, kc, QA_NKC, HIDDEN, 128, (n_sel() * HIDDEN + kc * 64) // 2)

            def ld_h(sks):
                if const_expr(HC > 1):  # hc_pre already contracted the streams
                    vals = poll([(mb("xin"), (s * HIDDEN + k) // 2, 2) for s, k in sks])
                    res = []
                    for i in range_constexpr(len(sks)):
                        a0, a1 = bf2_f32(vals[i][0])
                        b0, b1 = bf2_f32(vals[i][1])
                        res.append([a0, a1, b0, b1])
                    return res
                res = []
                for s, k in sks:
                    w = fx.Vector(bo.buffer_load(r_h, (s * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                    v = w.bitcast(fx.BFloat16).to(fx.Float32)
                    res.append([v[j] for j in range(4)])
                return res

            # the (small) input loads go out before the weight stream: loads complete in order
            h_ld = load_x_rmsnorm(ld_h, HIDDEN, g_in)
            pre = [u_qa(c) for c in range(QA_NKC // WAVES)]
            stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld)
            gpu.barrier()
            stamp("qkv_a", t, 2)
            acc = run_units(u_qa, QA_NKC // WAVES, QA_NKC // WAVES, pre)
            reduce_rows(1, acc, emit_out(QKV_A_TILE))
            stamp("qkv_a", t, 3)
            gpu.barrier()
            if tid < S * QKV_A_TILE:
                s = tid // QKV_A_TILE
                row = t * QKV_A_TILE + tid % QKV_A_TILE
                v = lds_ld(outs, tid)
                # Column ranges of the fused GEMV, in layout order. Must match
                # reference.qkv_a_split(): the compressor's pair is C_COFF-wide, so
                # this is not "one head_dim each" once CSA overlaps.
                if row < Q_LORA:
                    put(mb("q_a"), s * Q_LORA + row, v)
                elif row < Q_LORA + HEAD_DIM:
                    put(mb("kv_a"), s * HEAD_DIM + row - Q_LORA, v)
                elif row < Q_LORA + HEAD_DIM + CW:
                    put(mb("c_kv"), s * CW + row - Q_LORA - HEAD_DIM, v)
                elif row < Q_LORA + HEAD_DIM + 2 * CW:
                    put(mb("c_gate"), s * CW + row - Q_LORA - HEAD_DIM - CW, v)
                elif row < Q_LORA + HEAD_DIM + 2 * CW + IW:
                    put(mb("i_kv"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW, v)
                else:
                    put(mb("i_gate"), s * IW + row - Q_LORA - HEAD_DIM - 2 * CW - IW, v)
            stamp("qkv_a", t, 4)

        # ====== 2. KV RMSNorm + RoPE + FP8 round trip -> sliding-window ring cache
        # V4's K and V are the same HEAD_DIM row: RoPE occupies its last ROPE_DIM
        # lanes and the leading NOPE_DIM is FP8 round-tripped in 64-wide blocks
        # (one block per wave), matching the checkpoint's QAT.
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
            r_kv = _rsrc(kv_cache)
            # gamma and the RoPE factors are issued ahead of the wait
            g = ld_bf16(_rsrc(g_kv), fx.min(tid, HEAD_DIM - 1))
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            # one task covers every sample, so each loads its own position
            ps = [ld_pos(sx) for sx in range(S)]
            cs = [ld_f32(_rsrc(rope_cos), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)]
            sns = [ld_f32(_rsrc(rope_sin), ps[sx] * (ROPE_DIM // 2) + ri) for sx in range(S)]
            hint_wait(
                HEAD_DIM // QKV_A_TILE,
                lambda k: (mb("kv_a"), (S - 1) * HEAD_DIM + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("cache", t),
            )
            vs = getf_many([(mb("kv_a"), s * HEAD_DIM + fx.min(tid, HEAD_DIM - 1)) for s in range(S)])
            stamp("cache", t, 2)
            live = tid < HEAD_DIM
            ssq = block_sums([live.select(v * v, fx.Float32(0.0)) for v in vs])
            for s in range_constexpr(S):
                nv = vs[s] * _rsq(ssq[s] * (1.0 / HEAD_DIM) + EPS) * g
                # rope tail: lane ^ 1 is the other half of this interleaved (2i, 2i+1) pair
                partner = _xshfl(nv, 1)
                even = tid % 2 == 0
                rot = even.select(nv * cs[s] - partner * sns[s], partner * sns[s] + nv * cs[s])
                # nope head: one 64-wide FP8 block per wave
                amax = wave_max(fmath.absf(nv))
                nz = amax > 0.0
                qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
                inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))
                d0, _ = _fp8_roundtrip(fx.min(fx.max(nv * inv, -FP8_MAX), FP8_MAX), fx.Float32(0.0))
                kvn = bf16_round((tid < NOPE_DIM).select(d0 * qs, rot))
                if live:
                    bo.buffer_store(kvn.to(fx.BFloat16), r_kv, ld_dest(0, s) * HEAD_DIM + tid)
                    put(mb("kvnew"), s * HEAD_DIM + tid, kvn)
            stamp("cache", t, 4)

        # ============= 2b. KV compressor (HCA): rolling state, pooled on the boundary
        # Every token feeds a window of CR positions; only the last of each window
        # emits a compressed entry, pooled by a PER-CHANNEL softmax over the window's
        # positions. The pooled row then gets the same norm / RoPE / FP8 round trip
        # the window KV does, rotated at the window's FIRST position, and lands in
        # the compressed half of the same cache.
        if const_expr(CR):
            r_kvst = _rsrc(kv_state)
            r_scst = _rsrc(score_state)
            for tt in range(start("cmp"), S, G):
                tt = fx.Int32(tt)
                stamp("cmp", tt, 0)
                ch = fx.min(tid, HEAD_DIM - 1)
                live = tid < HEAD_DIM
                # tt is the sample: its own sequence, at the shared position, with
                # its own rolling state. That per-sample state is also what keeps
                # these S tasks -- one per CTA, nothing ordering them -- from racing.
                p = ld_pos(tt)
                sb = ld_slot(tt, st_kv)  # this sequence's slice of the rolling state
                slot = p % CR
                ap0 = [ld_f32(_rsrc(ape), slot * CW + j * HEAD_DIM + ch) for j in range(C_COFF)]
                g = ld_bf16(_rsrc(g_ckv), ch)
                # anchor: the window's FIRST position. Only read on boundary steps,
                # but the load is unconditional, so clamp it.
                anchor = fx.max(p + 1 - CR, fx.Int32(0))
                ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
                # One table per layer, indexed at the window's FIRST position: a
                # compressing layer's table is built on compress_rope_theta (and
                # YaRN) for EVERYTHING it rotates, not just the compressed rows.
                rc = ld_f32(_rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ri)
                rs = ld_f32(_rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ri)
                kvv = [getf(mb("c_kv"), (tt * C_COFF + j) * HEAD_DIM + ch) for j in range(C_COFF)]
                gtv = [getf(mb("c_gate"), (tt * C_COFF + j) * HEAD_DIM + ch) for j in range(C_COFF)]
                stamp("cmp", tt, 2)
                if live:
                    for j in range_constexpr(C_COFF):
                        w = sb + (CBASE + slot) * CW + j * HEAD_DIM + ch
                        bo.buffer_store(kvv[j], r_kvst, w)
                        bo.buffer_store(gtv[j] + ap0[j], r_scst, w)
                if (p + 1) % CR == 0:  # uniform across the CTA
                    # online softmax over the window, one channel per thread, so the
                    # CR positions are a loop rather than CR unrolled copies
                    for _i, acc in range(
                        0,
                        C_ROWS,
                        fx.Int32(1),
                        init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
                    ):
                        m = fx.Float32(acc[0])
                        den = fx.Float32(acc[1])
                        num = fx.Float32(acc[2])
                        # an overlapped entry takes the previous window's rows from
                        # their FIRST half and the current window's from their SECOND
                        i = fx.Int32(_i)
                        coff = (i >= CR).select(fx.Int32(HEAD_DIM), fx.Int32(0)) if OVERLAP else 0
                        wi = sb + i * CW + coff + ch
                        sv = ld_f32(r_scst, wi)
                        kv_i = ld_f32(r_kvst, wi)
                        m_new = fx.max(m, sv)
                        rescale = _exp(m - m_new)
                        w = _exp(sv - m_new)
                        res = yield [m_new, den * rescale + w, num * rescale + w * kv_i]
                    pooled = fx.Float32(res[2]) * _rcp(fx.Float32(res[1]))
                    pooled = bf16_round(pooled)
                    ssq = block_sum(live.select(pooled * pooled, fx.Float32(0.0)))
                    # the model this reproduces returns bf16 from the norm, so the
                    # RoPE and FP8 round trip below see bf16
                    nv = bf16_round(pooled * _rsq(ssq * (1.0 / HEAD_DIM) + EPS) * g)
                    partner = _xshfl(nv, 1)
                    even = tid % 2 == 0
                    rot = even.select(nv * rc - partner * rs, partner * rs + nv * rc)
                    amax = wave_max(fmath.absf(nv))
                    nz = amax > 0.0
                    qs = nz.select(amax * (1.0 / FP8_MAX), fx.Float32(1.0))
                    inv = nz.select(_rcp(amax) * FP8_MAX, fx.Float32(1.0))
                    d0, _ = _fp8_roundtrip(fx.min(fx.max(nv * inv, -FP8_MAX), FP8_MAX), fx.Float32(0.0))
                    cv = bf16_round((tid < NOPE_DIM).select(d0 * qs, rot))
                    if live:
                        crow = ld_dest(1, tt) + p // CR  # this sequence's entry p // CR
                        bo.buffer_store(cv.to(fx.BFloat16), _rsrc(kv_cache), crow * HEAD_DIM + tid)
                        put(mb("cnew"), tt * HEAD_DIM + tid, cv)
                    if const_expr(OVERLAP):
                        # the current window becomes the previous one. Each thread
                        # owns channels {ch, HEAD_DIM + ch} of every row it touches,
                        # and read both before writing either, so no barrier is owed.
                        for i in range_constexpr(CR):
                            for j in range_constexpr(C_COFF):
                                o = j * HEAD_DIM + ch
                                if live:
                                    bo.buffer_store(ld_f32(r_kvst, sb + (CR + i) * CW + o), r_kvst, sb + i * CW + o)
                                    bo.buffer_store(ld_f32(r_scst, sb + (CR + i) * CW + o), r_scst, sb + i * CW + o)
                stamp("cmp", tt, 4)

        def had_pair(v0, v1, ln):
            """FWHT over IHD channels held two per lane, scaled by IHD**-0.5.

            Lane ln owns channels ln and ln + 64, so every butterfly below stride 64
            is an xor shuffle inside the wave and the stride-64 one is the pair
            this lane already holds -- no LDS, no barrier.
            """
            # an explicit sequence, NOT `while h < 64`: a Python while over a
            # value the tracer can see becomes a device scf.while, and the
            # shuffle offset then stops being a compile-time constant
            for h in range_constexpr(6):
                st = 1 << h
                p0, p1 = _xshfl(v0, st), _xshfl(v1, st)
                hi = (ln & st) != 0
                v0 = hi.select(p0 - v0, v0 + p0)
                v1 = hi.select(p1 - v1, v1 + p1)
            v0, v1 = v0 + v1, v0 - v1
            sc = float(IHD) ** -0.5
            return v0 * sc, v1 * sc

        def fp4_block(v):
            """FP4 round trip over aligned 32-lane blocks, power-of-two scale."""
            amax = fmath.absf(v)
            for off in (16, 8, 4, 2, 1):
                amax = _xred(amax, off, fx.max)
            sc = _pow2_ceil(fx.max(amax, fx.Float32(FP4_MAX * 2.0**-126)) * (1.0 / FP4_MAX))
            q = fx.min(fx.max(v * _rcp(sc), -FP4_MAX), FP4_MAX)
            d, _ = _fp4_roundtrip(q, fx.Float32(0.0))
            return d * sc

        # ========== 2c. the indexer's compressor: same pooling, different tail
        # Half the head_dim, and it finishes with a Hadamard rotation over the whole
        # row followed by FP4 instead of FP8 over the nope part. The rotation is what
        # makes FP4 survivable: it spreads an outlier across every lane, so a block's
        # amax stops being set by one coordinate, and being orthonormal it leaves the
        # scores the indexer ranks unchanged.
        if const_expr(IHD):
            r_ikvst = _rsrc(i_kv_state)
            r_iscst = _rsrc(i_score_state)

            for tt in range(start("i_cmp"), S, G):
                tt = fx.Int32(tt)
                stamp("i_cmp", tt, 0)
                # one WAVE covers the row: lane ln holds channels ln and ln + 64
                ln = lane
                ilive = wave == 0
                p = ld_pos(tt)  # tt is the sample: its own sequence, its own position
                isb = ld_slot(tt, st_i)  # this sequence's slice of the rolling state
                icb = ld_slot(tt, st_ic)  # ... and of the indexer's key cache
                slot = p % CR
                chs = [ln, ln + 64]
                ap0 = [[ld_f32(_rsrc(i_ape), slot * IW + j * IHD + c) for j in range(C_COFF)] for c in chs]
                gg = [ld_bf16(_rsrc(g_ickv), c) for c in chs]
                anchor = fx.max(p + 1 - CR, fx.Int32(0))
                kvv = [[getf(mb("i_kv"), (tt * C_COFF + j) * IHD + c) for j in range(C_COFF)] for c in chs]
                gtv = [[getf(mb("i_gate"), (tt * C_COFF + j) * IHD + c) for j in range(C_COFF)] for c in chs]
                stamp("i_cmp", tt, 2)
                if ilive:
                    for e in range_constexpr(2):
                        for j in range_constexpr(C_COFF):
                            w = isb + (CBASE + slot) * IW + j * IHD + chs[e]
                            bo.buffer_store(kvv[e][j], r_ikvst, w)
                            bo.buffer_store(gtv[e][j] + ap0[e][j], r_iscst, w)
                if (p + 1) % CR == 0:  # uniform across the CTA
                    pooled = []
                    for e in range_constexpr(2):
                        for _i, acc in range(
                            0,
                            C_ROWS,
                            fx.Int32(1),
                            init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
                        ):
                            m = fx.Float32(acc[0])
                            den = fx.Float32(acc[1])
                            num = fx.Float32(acc[2])
                            i = fx.Int32(_i)
                            coff = (i >= CR).select(fx.Int32(IHD), fx.Int32(0)) if OVERLAP else 0
                            wi = isb + i * IW + coff + chs[e]
                            sv = ld_f32(r_iscst, wi)
                            kv_i = ld_f32(r_ikvst, wi)
                            m_new = fx.max(m, sv)
                            rescale = _exp(m - m_new)
                            w = _exp(sv - m_new)
                            res = yield [m_new, den * rescale + w, num * rescale + w * kv_i]
                        pooled.append(bf16_round(fx.Float32(res[2]) * _rcp(fx.Float32(res[1]))))
                    # RMS over the whole IHD row: both halves, one wave
                    sq = pooled[0] * pooled[0] + pooled[1] * pooled[1]
                    for off in range_constexpr(6):
                        sq = _xred(sq, 32 >> off, lambda a, b: a + b)
                    rs = _rsq(sq * (1.0 / IHD) + EPS)
                    nv = [bf16_round(pooled[e] * rs * gg[e]) for e in range(2)]
                    # rope lives in the TAIL of the row, i.e. entirely in the second
                    # half (IHD - ROPE_DIM == 64), so only channel ln + 64 rotates
                    rc = ld_f32(_rsrc(rope_cos), anchor * (ROPE_DIM // 2) + ln // 2)
                    rs2 = ld_f32(_rsrc(rope_sin), anchor * (ROPE_DIM // 2) + ln // 2)
                    partner = _xshfl(nv[1], 1)
                    even = ln % 2 == 0
                    nv[1] = bf16_round(even.select(nv[1] * rc - partner * rs2, partner * rs2 + nv[1] * rc))
                    h0, h1 = had_pair(nv[0], nv[1], ln)
                    q0, q1 = bf16_round(h0), bf16_round(h1)
                    o0, o1 = fp4_block(q0), fp4_block(q1)
                    if ilive:
                        r_ic = _rsrc(i_cache)
                        row = icb + (p // CR) * IHD
                        bo.buffer_store(bf16_round(o0).to(fx.BFloat16), r_ic, row + ln)
                        bo.buffer_store(bf16_round(o1).to(fx.BFloat16), r_ic, row + ln + 64)
                        put(mb("i_cnew"), tt * IHD + ln, bf16_round(o0))
                        put(mb("i_cnew"), tt * IHD + ln + 64, bf16_round(o1))
                    if const_expr(OVERLAP):
                        # retire the window, as the attention compressor does
                        for i in range_constexpr(CR):
                            for e in range_constexpr(2):
                                for j in range_constexpr(C_COFF):
                                    o = j * IHD + chs[e]
                                    if ilive:
                                        bo.buffer_store(
                                            ld_f32(r_ikvst, isb + (CR + i) * IW + o), r_ikvst, isb + i * IW + o
                                        )
                                        bo.buffer_store(
                                            ld_f32(r_iscst, isb + (CR + i) * IW + o), r_iscst, isb + i * IW + o
                                        )
                stamp("i_cmp", tt, 4)

        # ==================================== 3. q_a RMSNorm -> q_b (raw f32 query)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        for t in range(start("q_b"), N_QB, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)

            def u_qb(c):
                kc = wave * (QB_NKC // WAVES) + c
                return unit_fp8(r_wqb, r_sqb, t, kc, QB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

            pre = [u_qb(c) for c in range(QB_NKC // WAVES)]
            hint_wait(
                Q_LORA // QKV_A_TILE,
                lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
                mark=("q_b", t),
            )

            def ld_qa(sks):
                v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
                return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

            stage_x_rmsnorm(ld_qa, Q_LORA, g_q)
            stamp("q_b", t, 2)
            gpu.barrier()
            acc = run_units(u_qb, QB_NKC // WAVES, QB_NKC // WAVES, pre)
            reduce_rows(1, acc, emit_out(Q_B_TILE))
            stamp("q_b", t, 3)
            gpu.barrier()
            # published as f32: the per-head RMS below is taken on the unrounded GEMV
            # output, so rounding to bf16 happens only once, after RoPE
            head = t // QB_PER_HEAD
            hoff = (t % QB_PER_HEAD) * Q_B_TILE
            if tid < S * Q_B_TILE:
                s = tid // Q_B_TILE
                r = tid % Q_B_TILE
                put(mb("q_raw"), (s * H + head) * HEAD_DIM + hoff + r, lds_ld(outs, tid))
            stamp("q_b", t, 4)

        # ========= 3b. the indexer's query: its own per-head projection off q_a
        # Same normed q_lora the main q_b reads, so this is a second weight stream
        # but no second dependency. NOTE there is no per-head RMS here -- the
        # indexer's query is rotated and quantized, not normalised.
        if const_expr(IHD):
            r_wiqb, r_siqb = _rsrc(w_i_q_b), _rsrc(s_i_q_b)
            IQB_NKC = Q_LORA // 64
            for t in range(start("i_q_b"), N_IQB, G):
                t = fx.Int32(t)
                stamp("i_q_b", t, 0)

                def u_iqb(c):
                    kc = wave * (IQB_NKC // WAVES) + c
                    return unit_fp8(r_wiqb, r_siqb, t, kc, IQB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

                pre = [u_iqb(c) for c in range(IQB_NKC // WAVES)]
                hint_wait(
                    Q_LORA // QKV_A_TILE,
                    lambda k: (mb("q_a"), (S - 1) * Q_LORA + k * QKV_A_TILE + QKV_A_TILE - 1),
                    mark=("i_q_b", t),
                )

                def ld_iqa(sks):
                    v = get2_many([(mb("q_a"), s * Q_LORA + k + j) for s, k in sks for j in (0, 2)])
                    return [list(v[2 * i]) + list(v[2 * i + 1]) for i in range(len(sks))]

                stage_x_rmsnorm(ld_iqa, Q_LORA, g_q)
                stamp("i_q_b", t, 2)
                gpu.barrier()
                acc = run_units(u_iqb, IQB_NKC // WAVES, IQB_NKC // WAVES, pre)
                reduce_rows(1, acc, emit_out(Q_B_TILE))
                stamp("i_q_b", t, 3)
                gpu.barrier()
                ihead = t // IQB_PER_HEAD
                ihoff = (t % IQB_PER_HEAD) * Q_B_TILE
                if tid < S * Q_B_TILE:
                    s = tid // Q_B_TILE
                    r = tid % Q_B_TILE
                    put(mb("i_q_raw"), (s * IH + ihead) * IHD + ihoff + r, lds_ld(outs, tid))
                stamp("i_q_b", t, 4)

            # ---- rope -> Hadamard -> FP4, one whole index head per wave
            # The rotation has to see the rope'd lanes, and the Hadamard mixes the
            # whole head, so this cannot ride in a 16-row tile either.
            for tt in range(start("i_q"), S, G):
                tt = fx.Int32(tt)
                stamp("i_q", tt, 0)
                ln = lane
                chs = [ln, ln + 64]
                ip = ld_pos(tt)
                rc = ld_f32(_rsrc(rope_cos), ip * (ROPE_DIM // 2) + ln // 2)
                rs2 = ld_f32(_rsrc(rope_sin), ip * (ROPE_DIM // 2) + ln // 2)
                for k in range_constexpr(IH // WAVES):
                    ihead = wave + k * WAVES
                    base_i = (tt * IH + ihead) * IHD
                    v = [getf(mb("i_q_raw"), base_i + c) for c in chs]
                    # rope falls entirely in the head's second half (IHD - ROPE_DIM == 64)
                    partner = _xshfl(v[1], 1)
                    even = ln % 2 == 0
                    v[1] = bf16_round(even.select(v[1] * rc - partner * rs2, partner * rs2 + v[1] * rc))
                    v[0] = bf16_round(v[0])
                    h0, h1 = had_pair(v[0], v[1], ln)
                    o0, o1 = fp4_block(bf16_round(h0)), fp4_block(bf16_round(h1))
                    put(mb("i_q"), base_i + chs[0], o0)
                    put(mb("i_q"), base_i + chs[1], o1)
                stamp("i_q", tt, 4)

        # ===== 3c. weights_proj: the per-head weight the score's head-sum uses
        # bf16 and tiny (IH outputs), so no MFMA: wave w accumulates head w over
        # this thread's slice of the normed layer input, then one block reduction.
        # Scaled by the GLOBAL head count -- the sum is finished by the all-reduce.
        if const_expr(IHD):
            for tt in range(start("i_wp"), S, G):
                tt = fx.Int32(tt)
                stamp("i_wp", tt, 0)
                r_iw = _rsrc(i_w)

                def ld_hw(sks):
                    # `tt` IS this task's sample, so the pair's own `s` (always 0,
                    # since count=1 below asks for one sample's worth of offsets)
                    # is discarded for it -- the same substitution `router`'s ld_a
                    # makes. Taking `s` here gave every sample sample 0's score
                    # weights, which is invisible at S == 1 and silent above it.
                    if const_expr(HC > 1):
                        vals = poll([(mb("xin"), (tt * HIDDEN + k) // 2, 2) for _s, k in sks])
                        res = []
                        for i in range_constexpr(len(sks)):
                            a0, a1 = bf2_f32(vals[i][0])
                            b0, b1 = bf2_f32(vals[i][1])
                            res.append([a0, a1, b0, b1])
                        return res
                    res = []
                    for _s, k in sks:
                        w = fx.Vector(bo.buffer_load(r_h, (tt * HIDDEN + k) // 2, vec_width=2, dtype=T.i32))
                        v = w.bitcast(fx.BFloat16).to(fx.Float32)
                        res.append([v[j] for j in range(4)])
                    return res

                # the same normed input qkv_a reads, recomputed rather than
                # republished: one extra pass over HIDDEN is cheaper than the traffic
                ks, act = _rmsnorm_tail_ks(HIDDEN)
                gs, xv = load_x_rmsnorm(ld_hw, HIDDEN, g_in, count=1)
                ssq = fx.Float32(0.0)
                for i in range_constexpr(len(ks)):
                    for a in xv[i]:
                        term = a * a
                        if const_expr(act is not None and i == len(ks) - 1):
                            term = act.select(term, fx.Float32(0.0))
                        ssq = ssq + term
                rstd = _rsq(block_sums([ssq])[0] * (1.0 / HIDDEN) + EPS)
                stamp("i_wp", tt, 2)
                parts = []
                for hh in range_constexpr(IH):
                    acc = fx.Float32(0.0)
                    for i in range_constexpr(len(ks)):
                        wv = (
                            fx.Vector(bo.buffer_load(r_iw, (hh * HIDDEN + ks[i]) // 2, vec_width=2, dtype=T.i32))
                            .bitcast(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        for j in range_constexpr(4):
                            term = xv[i][j] * rstd * gs[i][j] * wv[j]
                            if const_expr(act is not None and i == len(ks) - 1):
                                term = act.select(term, fx.Float32(0.0))
                            acc = acc + term
                    parts.append(acc)
                tots = block_sums(parts)
                # the GLOBAL head count: this rank holds IH of them and the
                # all-reduce finishes the sum, so the scale cannot be local
                sc = float(IHD) ** -0.5 * float(IH_TOTAL) ** -0.5
                for hh in range_constexpr(IH):
                    if tid == 0:
                        put(mb("i_wp"), tt * IH + hh, bf16_round(tots[hh]) * sc)
                stamp("i_wp", tt, 4)

            # ===== 3d. score every compressed entry against the indexer's queries
            # score[c] = sum_h relu(q[h] . k[c]) * w[h]. One candidate per thread;
            # the queries go to LDS once, where every thread reads the SAME element
            # at a time, so those reads broadcast rather than conflict.
            for tt in range(start("i_score"), S * N_ISCORE, G):
                tt = fx.Int32(tt)
                stamp("i_score", tt, 0)
                s = tt // N_ISCORE
                blk = tt % N_ISCORE
                for i in range_constexpr((IH * IHD + THREADS - 1) // THREADS):
                    w4 = tid + i * THREADS
                    if w4 < IH * IHD:
                        lds_st(xs, w4, getf(mb("i_q"), s * IH * IHD + w4))
                wv = [getf(mb("i_wp"), s * IH + hh) for hh in range(IH)]
                gpu.barrier()
                stamp("i_score", tt, 2)
                c = blk * SCORE_TILE + tid
                # entries the compressor has not written yet must never be picked
                sp = ld_pos(s)
                n_live = (sp + 1) // CR
                r_ic2 = _rsrc(i_cache)
                # This sequence's slice of the key cache. Folded into the row base
                # ONCE, outside the loop below -- `s` is loop-invariant, and adding
                # it per iteration would put another live value in the body that
                # was just trimmed to stop it spilling.
                icb = ld_slot(s, st_ic) // 2
                # the entry this launch just wrote is not reliably visible in the
                # cache yet, so take it from the mailbox instead. The outer test is
                # CTA-uniform, so every thread reaches the poll; only the thread
                # holding that candidate uses the value.
                is_new = c == sp // CR
                # A RUNTIME loop, not range_constexpr. Unrolled, the compiler hoists
                # all IHD / 8 key loads and every query read to the top of the stage;
                # that is ~130 values more than the budget, so it spilled them and
                # then fed the FMAs back one scratch load and one full vmcnt wait at
                # a time -- 274 us of stall to do 1024 FMAs. Carrying the
                # accumulators keeps a single iteration live and the spills go away.
                for _d0, acc in range(
                    0,
                    IHD // 8,
                    fx.Int32(1),
                    init=[fx.Float32(0.0) for _ in range(IH)],
                ):
                    d0 = fx.Int32(_d0)
                    kw = [
                        v
                        for v in fx.Vector(
                            bo.buffer_load(
                                r_ic2,
                                icb + fx.min(c, N_COMP - 1) * (IHD // 2) + d0 * 4,
                                vec_width=4,
                                dtype=T.i32,
                            )
                        )
                        .bitcast(fx.BFloat16)
                        .to(fx.Float32)
                    ]
                    if (sp + 1) % CR == 0:
                        nv = [getf(mb("i_cnew"), s * IHD + d0 * 8 + e) for e in range(8)]
                        kw = [is_new.select(nv[e], kw[e]) for e in range(8)]
                    accs = [fx.Float32(acc[hh]) for hh in range(IH)]
                    for e in range_constexpr(8):
                        kv = kw[e]
                        for hh in range_constexpr(IH):
                            accs[hh] = accs[hh] + kv * lds_ld(xs, hh * IHD + d0 * 8 + e)
                    res = yield accs
                sc_t = fx.Float32(0.0)
                for hh in range_constexpr(IH):
                    sc_t = sc_t + fx.max(fx.Float32(res[hh]), fx.Float32(0.0)) * wv[hh]
                live = (c < n_live) & (c < N_COMP)
                if const_expr(W > 1):
                    # This rank holds only IH of the 64 index heads, so its score is
                    # a PARTIAL sum -- without this exchange the ranks would rank the
                    # entries differently and silently attend to different keys.
                    # Push to every peer, then sum all ranks' partials in rank order
                    # from our own buffer: same order everywhere, so the totals are
                    # bit-identical and the top-k below cannot disagree.
                    # mark 3, not 5: the report reads marks 0..4, so this is what
                    # splits `compute` (the scoring loop) from `epi` (the exchange).
                    # Stamping outside that range lumped them together.
                    gpu.barrier()  # TEMP probe fence
                    stamp("i_score", tt, 3)
                    for p in range_constexpr(W):
                        pv2 = fx.Vector(bo.buffer_load(r_peers, p * 2, vec_width=2, dtype=T.i32))
                        dst = (fx.Int64(_uniform(pv2[1])) << 32) | fx.Int64(fx.Uint32(_uniform(pv2[0])))
                        if c < N_COMP:
                            put(
                                dst + fx.Int64(SY["iscore"]),
                                (rank * S + s) * N_COMP + c,
                                sc_t,
                                CM_SYS,
                            )
                    stamp("i_score", tt, 5)  # TEMP probe: push issued
                    own = sym + fx.Int64(SY["iscore"])
                    ci = fx.min(c, N_COMP - 1)
                    got = poll([(own, (src * S + s) * N_COMP + ci, 1) for src in range(W)], "one-as")
                    sc_t = fx.Float32(0.0)
                    for src in range_constexpr(W):
                        sc_t = sc_t + got[src][0].bitcast(fx.Float32)
                if c < N_COMP:
                    put(mb("i_score"), s * N_COMP + c, live.select(sc_t, fx.Float32(NEG)))
                stamp("i_score", tt, 4)

        # ============ 3e. top-k: which compressed entries the attention gathers
        # Exact and deterministic, which is the point: every rank all-reduced to
        # bit-identical scores above, so an exact selection gives every rank the
        # same set without a second exchange. An approximate or order-dependent
        # pick would let the ranks attend to different skey.
        #
        # Radix select, MSB first, on the score's order-preserving bits. Each pass
        # counts the candidates whose remaining prefix matches a trial, so it needs
        # only equality tests -- no unsigned compare.
        if const_expr(IHD):
            for tt in range(start("i_topk"), S, G):
                tt = fx.Int32(tt)
                stamp("i_topk", tt, 0)
                s = tt
                c0 = ld_dest(1, s)  # plane row of this sequence's compressed entry 0
                n_live = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
                k_want = fx.min(n_live, fx.Int32(N_INDEX))
                mine = [fx.Int32(tid) * TOPK_PER + j for j in range(TOPK_PER)]
                ok_c = [(c < n_live) for c in mine]
                # f32 bits -> signed int32 whose ORDER matches the float's: flip the
                # low 31 bits of negatives, which puts -2 below -1 and both below 0
                skey = []
                for j in range_constexpr(TOPK_PER):
                    b = getf(mb("i_score"), s * N_COMP + fx.min(mine[j], fx.Int32(N_COMP - 1)))
                    bi = b.bitcast(fx.Int32)
                    skey.append(bi ^ ((bi >> 31) & 0x7FFFFFFF))
                # ... and to the unsigned domain, where MSB-first prefixes work
                us = [ok_c[j].select(skey[j] ^ MIN_I32, fx.Int32(0)) for j in range(TOPK_PER)]
                stamp("i_topk", tt, 2)

                pfx = fx.Int32(0)
                gt = fx.Int32(0)
                for bit in range_constexpr(32):
                    b = 31 - bit
                    trial = pfx | fx.Int32(1 << b)
                    cnt = fx.Int32(0)
                    for j in range_constexpr(TOPK_PER):
                        hit = ok_c[j] & (((us[j] ^ trial) >> b) == 0)
                        cnt = cnt + hit.select(fx.Int32(1), fx.Int32(0))
                    tot = block_isum(cnt)
                    take = (gt + tot) >= k_want
                    pfx = take.select(trial, pfx)
                    gt = take.select(gt, gt + tot)
                thr = pfx ^ MIN_I32  # back to the signed-comparable domain
                need = k_want - gt  # how many of the ties at thr to keep

                # Two compactions in index order: everything strictly above the
                # threshold, then as many of the ties as the count still needs. One
                # pass over `>=` would not do -- ties early in index order would
                # crowd out strict-greaters that must all be kept.
                # Each phase appends from wherever the last one ended, and the -1
                # fill starts from what was ACTUALLY written, not from k_want. That
                # matters beyond tidiness: the gather polls every slot of this
                # mailbox, so a slot the compaction skips is not a wrong answer, it
                # is a kernel that never finishes. Deriving the fill from the real
                # count makes any error in the threshold above show up as a bad
                # selection, which a test can see.
                stamp("i_topk", tt, 3)
                w_base = fx.Int32(0)
                for phase in range_constexpr(2):
                    sel = []
                    for j in range_constexpr(TOPK_PER):
                        hit = skey[j] > thr if phase == 0 else skey[j] == thr
                        sel.append(ok_c[j] & hit)
                    mycnt = fx.Int32(0)
                    for j in range_constexpr(TOPK_PER):
                        mycnt = mycnt + sel[j].select(fx.Int32(1), fx.Int32(0))
                    off, tot = block_excl_scan(mycnt)
                    room = k_want - w_base
                    w = w_base + off
                    for j in range_constexpr(TOPK_PER):
                        if sel[j] & ((w - w_base) < room):
                            put(mb("i_sel"), s * N_ISEL + w, c0 + mine[j])
                        w = w + sel[j].select(fx.Int32(1), fx.Int32(0))
                    w_base = w_base + fx.min(tot, room)
                    gpu.barrier()
                for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                    o = fx.Int32(tid) + j * THREADS
                    if o < N_ISEL:
                        if o >= w_base:
                            put(mb("i_sel"), s * N_ISEL + o, fx.Int32(-1))
                stamp("i_topk", tt, 4)

        # ============== 4. per-head query RMS (no weight) + RoPE -> bf16 query
        # V4 scales each head's whole HEAD_DIM query by rsqrt(mean(q^2) + eps) before
        # rotating its tail; that spans all of a head's dims, so it cannot ride in a
        # 16-row q_b tile and gets its own (tiny) stage, one task per (sample, head).
        for tt in range(start("q_norm"), S * H, G):
            tt = fx.Int32(tt)
            stamp("q_norm", tt, 0)
            s = tt // H
            head = tt % H
            ri = fx.max(tid - NOPE_DIM, fx.Int32(0)) // 2
            sp = ld_pos(s)
            c = ld_f32(_rsrc(rope_cos), sp * (ROPE_DIM // 2) + ri)
            sn = ld_f32(_rsrc(rope_sin), sp * (ROPE_DIM // 2) + ri)
            hint_wait(
                QB_PER_HEAD,
                lambda k: (mb("q_raw"), (s * H + head) * HEAD_DIM + k * Q_B_TILE + Q_B_TILE - 1),
                mark=("q_norm", tt),
            )
            qv = getf(mb("q_raw"), (s * H + head) * HEAD_DIM + fx.min(tid, HEAD_DIM - 1))
            stamp("q_norm", tt, 2)
            live = tid < HEAD_DIM
            ssq = block_sum(live.select(qv * qv, fx.Float32(0.0)))
            nv = qv * _rsq(ssq * (1.0 / HEAD_DIM) + EPS)
            partner = _xshfl(nv, 1)
            even = tid % 2 == 0
            rot = even.select(nv * c - partner * sn, partner * sn + nv * c)
            qn = (tid < NOPE_DIM).select(nv, rot)
            # repack as bf16 pairs for the split stage's MFMA operand
            other = _xshfl(qn, 1)
            if live & (tid % 2 == 0):
                put(mb("q"), ((s * H + head) * HEAD_DIM + tid) // 2, bf16_pair(qn, other))
            stamp("q_norm", tt, 4)

        # ============== 5. gather-sparse sliding-window split: 64 keys x H heads
        r_kv = _rsrc(kv_cache)
        r_idx = _rsrc(indices)
        KPW = SPLIT_KEYS // WAVES
        EPL = HEAD_DIM // 64  # KV elements one lane owns of a key's row
        WPL = EPL // 2  # ... as packed bf16 words

        def split_keys(t, s):
            """Wave 0 writes this split's 64 ring slots to LDS keys; -1 = not yet written.

            With an indexer the compressed half is not the caller's to give: it is
            whatever the top-k just chose. The window is a whole number of tiles, so
            which side a tile falls on is uniform across the CTA.
            """
            if wave == 0:
                k_pos = t * SPLIT_KEYS + lane
                if const_expr(IHD):
                    if t * SPLIT_KEYS >= window:  # uniform across the CTA
                        lds_st(keys, lane, get(mb("i_sel"), s * N_ISEL + k_pos - window))
                    else:
                        lds_st(
                            keys,
                            lane,
                            fx.Int32(bo.buffer_load(r_idx, s * N_KEYS + k_pos, vec_width=1, dtype=T.i32)),
                        )
                else:
                    lds_st(
                        keys,
                        lane,
                        fx.Int32(bo.buffer_load(r_idx, s * N_KEYS + k_pos, vec_width=1, dtype=T.i32)),
                    )

        def gather_old_kv():
            """Each wave copies its KPW keys' shared KV row (HEAD_DIM bf16) into the tile.
            Unwritten slots (-1) are clamped to 0 here and masked in the softmax.

            The index list holds ABSOLUTE rows of one cache plane, so there is no
            per-sample or per-layer term here: whoever owns the pool folds its base
            into the indices (or hands this layer a view), as a paged runtime must
            anyway to express a row two sequences share."""
            krows = [fx.max(lds_ld(keys, wave * KPW + jj), fx.Int32(0)) for jj in range(KPW)]
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kv8 = fx.Vector(
                    bo.buffer_load(r_kv, krows[jj] * (HEAD_DIM // 2) + lane * WPL, vec_width=WPL, dtype=T.i32)
                )
                fx.ptr_store(kv8.bitcast(fx.Float32), ktile + (j * KS + lane * WPL))

        def patch_new_kv(s):
            """Rows this launch wrote come from mailboxes, not the cache: the split
            cannot rely on seeing our own global store. That covers the window row
            (kvnew) and, on a compression boundary, the compressed row (cnew).

            Only sample ``s``'s own rows: the samples are separate sequences, so
            another sample's new row is not in this one's cache and must not be
            patched into its tile. ``kr`` is wave-uniform, so the polls below are
            reached by a whole wave or none of it."""
            sp = ld_pos(s)
            w_row = ld_dest(0, s)
            c_row = (ld_dest(1, s) + sp // CR) if const_expr(CR) else fx.Int32(0)
            for jj in range_constexpr(KPW):
                j = wave * KPW + jj
                kr = lds_ld(keys, j)
                if kr == w_row:
                    kvp = get2_many([(mb("kvnew"), s * HEAD_DIM + lane * EPL + m * 2) for m in range(WPL)])
                    w = [bf16_pair(a0, a1) for a0, a1 in kvp]
                    fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * WPL))
                if const_expr(CR):
                    # only a boundary step writes one, and then it is the newest
                    # compressed slot
                    if ((sp + 1) % CR == 0) & (kr == c_row):
                        cvp = get2_many([(mb("cnew"), s * HEAD_DIM + lane * EPL + m * 2) for m in range(WPL)])
                        w = [bf16_pair(a0, a1) for a0, a1 in cvp]
                        fx.ptr_store(fx.Vector.from_elements(w, fx.Float32), ktile + (j * KS + lane * WPL))

        for tt in range(start("split"), S * N_SPLIT, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // N_SPLIT  # sample
            t = tt % N_SPLIT  # 64-key chunk
            split_keys(t, s)
            gpu.barrier()
            gather_old_kv()  # before waiting for q: these rows are from earlier launches
            hint_wait(
                H,
                lambda k: (mb("q"), ((s * H + k) * HEAD_DIM + HEAD_DIM - 2) // 2),
                mark=("split", tt),
            )
            # q of all heads -> bf16 Q[h][HEAD_DIM] (words h * QS + d / 2). The head
            # stride QS is padded, so this cannot use stage_x_pairs' flat layout.
            NQ_TOT = H * HEAD_DIM // 4  # aligned groups of 4 elements
            NQ = NQ_TOT // THREADS

            def stage_q(w4, v):
                qw = (w4 // (HEAD_DIM // 4)) * QS + (w4 % (HEAD_DIM // 4)) * 2
                lds_st(xs, qw, v[0].bitcast(fx.Float32))
                lds_st(xs, qw + 1, v[1].bitcast(fx.Float32))

            qv = poll([(mb("q"), (s * H * HEAD_DIM + (tid + i * THREADS) * 4) // 2, 2) for i in range(NQ)])
            for i in range_constexpr(NQ):
                stage_q(tid + i * THREADS, qv[i])
            if const_expr(NQ_TOT % THREADS):
                w4 = tid + NQ * THREADS
                if w4 < NQ_TOT:
                    stage_q(w4, poll([(mb("q"), (s * H * HEAD_DIM + w4 * 4) // 2, 2)])[0])
            patch_new_kv(s)
            stamp("split", tt, 2)
            gpu.barrier()
            stamp("split", tt, 5)
            # scores = K Q^T on MFMA: keys are M (4 row groups), HEAD_DIM is K (split in
            # two wave halves), heads N.  Unlike MLA there is a single tile: V4's RoPE
            # lanes live inside the same HEAD_DIM row, so no latent/pe base select.
            hn = fx.min(lane % 16, H - 1)
            rgk = wave % 4
            c = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(QK_DIM // 32 // 2):
                kst = (wave // 4) * (QK_DIM // 32 // 2) + st
                key = rgk * 16 + lane % 16
                kw = KT_OFF + key * KS + kst * 16
                a = fx.ptr_load(xs + (kw + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                b = fx.ptr_load(xs + (hn * QS + kst * 16 + (lane // 16) * 4), result_type=v4f).bitcast(fx.BFloat16)
                c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
            fx.ptr_store(c, red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("split", tt, 6)
            # split-local softmax: wave h, lane = key j (score = sum of the two K halves).
            for hg in range_constexpr(H // WAVES):
                h = wave + hg * WAVES
                valid = lds_ld(keys, lane) >= 0
                r16 = lane % 16
                cl = h + 16 * (r16 // 4)
                raw = lds_ld(red, ((lane // 16) * 64 + cl) * 4 + r16 % 4) + lds_ld(
                    red, ((lane // 16 + 4) * 64 + cl) * 4 + r16 % 4
                )
                sc_v = valid.select(raw * scale, fx.Float32(NEG))
                m = wave_max(sc_v)
                p = valid.select(_exp(sc_v - m), fx.Float32(0.0))
                lsum = wave_sum(p)
                p_n = _xshfl(p, 1)
                if lane % 2 == 0:  # P^T bf16 [h][64 keys] (words h * 32 + j / 2)
                    lds_st(pl, h * (SPLIT_KEYS // 2) + lane // 2, bf16_pair(p, p_n))
                if lane == 0:  # written last: the merge's readiness hint
                    put(mb("sp_m"), (s * N_SPLIT + t) * H + h, m)
                    put(mb("sp_l"), (s * N_SPLIT + t) * H + h, lsum)
            gpu.barrier()
            stamp("split", tt, 3)
            # O = P V on MFMA: heads M, keys K (2 steps), dims N.  V is the same tile the
            # scores read -- in V4 as in MLA there is no separate V tensor.
            for g in range_constexpr(HEAD_DIM // 32 // WAVES):
                dw = (wave * (HEAD_DIM // 32 // WAVES) + g) * 16 + lane % 16  # dim pair word
                c0 = fx.Vector.filled(4, 0.0, fx.Float32)
                c1 = fx.Vector.filled(4, 0.0, fx.Float32)
                for js in range_constexpr(SPLIT_KEYS // 32):
                    a = fx.ptr_load(
                        pl + (hn * (SPLIT_KEYS // 2) + js * 16 + (lane // 16) * 4), result_type=v4f
                    ).bitcast(fx.BFloat16)
                    ws = [
                        fx.ptr_load(ktile + ((js * 32 + (lane // 16) * 8 + i) * KS + dw)).bitcast(fx.Int32)
                        for i in range(8)
                    ]
                    w_lo = [(ws[2 * i] & 0xFFFF) | (ws[2 * i + 1] << 16) for i in range(4)]
                    w_hi = [fx.Int32(fx.Uint32(ws[2 * i]) >> 16) | (ws[2 * i + 1] & -65536) for i in range(4)]
                    b0 = fx.Vector.from_elements(w_lo, fx.Int32).bitcast(fx.BFloat16)
                    b1 = fx.Vector.from_elements(w_hi, fx.Int32).bitcast(fx.BFloat16)
                    c0 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b0, c0]))
                    c1 = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b1, c1]))
                if lane < 16 * (H // 4):  # rows (heads) 4 * (lane // 16) + e < H
                    for e in range_constexpr(4):
                        hh = (lane // 16) * 4 + e
                        put_bf(mb("sp_acc"), ((s * N_SPLIT + t) * H + hh) * HEAD_DIM + dw * 2, [c0[e], c1[e]])
            stamp("split", tt, 4)

        # ============ 6. split merge (+ attention sink) + inverse RoPE -> o
        # No W_UV: V4's attention output already lives in the output space. The merge
        # is MLA's flash-decode merge plus the per-head learnable sink, which enters
        # the denominator only (it contributes no value), and the output's RoPE lanes
        # are de-rotated because V shares the RoPE'd K.
        UV_PAIRS = UV_TILE // 2
        for tt in range(start("uv"), S * N_UV, G):
            tt = fx.Int32(tt)
            stamp("uv", tt, 0)
            s = tt // N_UV  # sample
            t = tt % N_UV  # UV_TILE-dim tile
            head = t // UV_PER_HEAD
            doff = (t % UV_PER_HEAD) * UV_TILE
            sink = ld_f32(_rsrc(attn_sink), head)
            hint_wait(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head), mark=("uv", tt))
            pre_poll(N_SPLIT, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            dp = fx.min(tid, UV_PAIRS - 1)  # dim pair within this tile
            spi = fx.min(lane, N_SPLIT - 1)
            ml = (s * N_SPLIT + spi) * H + head
            got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)], batch=2)
            if wave == 0:  # per-split weights exp(m - M) / L for this head -> misc[sp]
                ok_sp = lane < N_SPLIT
                m_sp = ok_sp.select(got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
                l_sp = ok_sp.select(got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
                mx = wave_max(m_sp)
                w_sp = _exp(m_sp - mx)
                den = wave_sum(l_sp * w_sp) + _exp(sink - mx)
                if ok_sp:
                    lds_st(misc, lane, w_sp * _rcp(den))
            stamp("uv", tt, 2)
            gpu.barrier()
            # Accumulate the per-split values a CHUNK at a time, in a runtime
            # loop. Polling all N_SPLIT of them up front and consuming them in a
            # constexpr loop keeps N_SPLIT + 2 words live at once: fine at 18
            # splits, but at 64 it overruns the 256-VGPR budget and spills 113
            # registers into 368 bytes of scratch (measured, and worth ~26 us).
            # The chunk keeps one iteration live while still issuing UV_CHUNK
            # loads at a time, so the waits do not serialise. Same shape of fix
            # as the i_score loop above.
            for _c, acc in range(
                0, N_SPLIT // UV_CHUNK, fx.Int32(1), init=[fx.Float32(0.0), fx.Float32(0.0)]
            ):
                cb = fx.Int32(_c) * UV_CHUNK
                gc = poll(
                    [
                        (
                            mb("sp_acc"),
                            ((s * N_SPLIT + cb + e) * H + head) * (HEAD_DIM // 2) + doff // 2 + dp,
                            1,
                        )
                        for e in range(UV_CHUNK)
                    ],
                    batch=UV_CHUNK,
                )
                o0 = fx.Float32(acc[0])
                o1 = fx.Float32(acc[1])
                for e in range_constexpr(UV_CHUNK):
                    wj = lds_ld(misc, cb + e)
                    a0, a1 = bf2_f32(gc[e][0])
                    o0 = o0 + a0 * wj
                    o1 = o1 + a1 * wj
                res = yield [o0, o1]
            o0 = fx.Float32(res[0])
            o1 = fx.Float32(res[1])
            # de-rotate the RoPE lanes (inverse rotation: sin negated)
            d0 = doff + dp * 2
            ri = fx.max(d0 - NOPE_DIM, fx.Int32(0)) // 2
            sp = ld_pos(s)
            c = ld_f32(_rsrc(rope_cos), sp * (ROPE_DIM // 2) + ri)
            sn = ld_f32(_rsrc(rope_sin), sp * (ROPE_DIM // 2) + ri)
            rot = d0 >= NOPE_DIM
            v0 = rot.select(o0 * c + o1 * sn, o0)
            v1 = rot.select(o1 * c - o0 * sn, o1)
            if tid < UV_PAIRS:
                put(mb("o"), ((s * H + head) * HEAD_DIM + doff) // 2 + dp, bf16_pair(v0, v1))
            stamp("uv", tt, 4)

        # ================= 7a. o_a: grouped low-rank output projection
        # Each of O_GROUPS groups consumes its own OA_K slice of the concatenated
        # heads, so a task's K window depends on its group.
        r_woa, r_soa = _rsrc(w_o_a), _rsrc(s_o_a)
        OA_NKC = OA_K // 64
        OA_R = ROW_TILE // 16
        OA_WPR = WAVES // OA_R
        for tt in range(start("o_a"), N_OA, G):
            tt = fx.Int32(tt)
            stamp("o_a", tt, 0)
            s = tt // (O_GROUPS * OA_PER_GROUP)
            t = tt % (O_GROUPS * OA_PER_GROUP)
            grp = t // OA_PER_GROUP

            def u_oa(c):
                kc = (wave % OA_WPR) * (OA_NKC // OA_WPR) + c
                return unit_fp8(r_woa, r_soa, t * OA_R + wave // OA_WPR, kc, OA_NKC, OA_K, 128, (kc * 64) // 2)

            pre = [u_oa(c) for c in range(OA_NKC // OA_WPR)]
            hint_wait(
                H // O_GROUPS,
                lambda k: (mb("o"), ((s * H + grp * (H // O_GROUPS) + k) * HEAD_DIM + HEAD_DIM - 2) // 2),
                mark=("o_a", tt),
            )
            stage_x_pairs("o", OA_K, lambda k: (s * H) * HEAD_DIM + grp * OA_K + k)
            stamp("o_a", tt, 2)
            gpu.barrier()
            acc = run_units(u_oa, OA_NKC // OA_WPR, OA_NKC // OA_WPR, pre)
            reduce_rows(OA_R, acc, emit_out(ROW_TILE))
            stamp("o_a", tt, 3)
            gpu.barrier()
            if tid < ROW_TILE // 4:
                r = tid * 4
                put_bf(
                    mb("o_lora"),
                    s * OB_K + t * ROW_TILE + r,
                    [lds_ld(outs, r + j) for j in range(4)],
                )
            stamp("o_a", tt, 4)

        # ============= 7b. o_b + attention TP peer reduce + residual -> a
        r_wob, r_sob = _rsrc(w_o_b), _rsrc(s_o_b)
        OB_NKC = OB_K // 64
        OB_R = ROW_TILE // 16
        OB_WPR = WAVES // OB_R
        for t in range(start("o_b"), N_ROW_TILES, G):
            t = fx.Int32(t)
            stamp("o_b", t, 0)

            def u_ob(c):
                kc = (wave % OB_WPR) * (OB_NKC // OB_WPR) + c
                return unit_fp8(
                    r_wob, r_sob, t * OB_R + wave // OB_WPR, kc, OB_NKC, OB_K, 128, (n_sel() * OB_K + kc * 64) // 2
                )

            pre = [u_ob(c) for c in range(OB_NKC // OB_WPR)]
            hint_wait(
                S * O_GROUPS * OA_PER_GROUP,
                lambda k: (
                    mb("o_lora"),
                    (k // (O_GROUPS * OA_PER_GROUP)) * OB_K + (k % (O_GROUPS * OA_PER_GROUP)) * ROW_TILE + ROW_TILE - 1,
                ),
                mark=("o_b", t),
            )
            stage_x_pairs("o_lora", S * OB_K, lambda k: k)
            stamp("o_b", t, 2)
            gpu.barrier()
            acc = run_units(u_ob, OB_NKC // OB_WPR, OB_NKC // OB_WPR, pre)
            reduce_rows(OB_R, acc, emit_out(ROW_TILE))
            stamp("o_b", t, 3)
            gpu.barrier()

            def resid_h(s, row):
                w = fx.Vector.from_elements(
                    [fx.Int32(bo.buffer_load(r_h, (s * HIDDEN + row) // 2, vec_width=1, dtype=T.i32))], fx.Int32
                )
                v = w.bitcast(fx.BFloat16).to(fx.Float32)
                return v[0], v[1]

            if const_expr(HC > 1):
                hc_stage_coef(0)
                peer_reduce(
                    "attn",
                    t,
                    None,  # hc_post owns the combination
                    lambda s, row, v0, v1: hc_post(
                        s,
                        row,
                        v0,
                        v1,
                        lambda s_, j, r_: fx.Int32(
                            bo.buffer_load(r_h, ((s_ * HC + j) * HIDDEN + r_) // 2, vec_width=1, dtype=T.i32)
                        ),
                        lambda s_, k, r_, o0, o1: put_bf(mb("a"), (s_ * HC + k) * HIDDEN + r_, [o0, o1]),
                    ),
                )
            else:
                peer_reduce(
                    "attn",
                    t,
                    resid_h,
                    lambda s, row, v0, v1: put_bf(mb("a"), s * HIDDEN + row, [v0, v1]),
                )
            stamp("o_b", t, 4)
        if const_expr(HC > 1):
            hc_pre_stages(
                "f",
                1,
                hc_ffn_fn,
                hc_ffn_sb,
                lambda s, k: get(mb("a"), (s * HC * HIDDEN + k) // 2),
                "ain",
            )

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        for tt in range(start("router"), S * N_ROUTER, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            router_sample = tt // N_ROUTER
            stamp("router", tt, 0)

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n]
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.Int32(0)

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(
                N_ROW_TILES,
                lambda k: (mb(A_IN), router_sample * HIDDEN + k * ROW_TILE + ROW_TILE - 1),
                mark=("router", tt),
            )
            # This task's expert-activation block inputs ride along with the staging
            # loads. MXFP8 uses four independent 16-lane groups per wave.
            r_gp = _rsrc(g_post)
            if const_expr(use_mxfp8_block32):
                x_blk = (wave * N_ROUTER + t) * 4 + lane // 16
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 32 + lane % 16 * 2
            else:
                x_blk = wave * N_ROUTER + t
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 128 + lane * 2
            x_s = router_sample
            x_ok = (wave < XQ_WAVES) & (x_blk < PUBLISH_BLOCKS)
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                specs = [(mb(A_IN), (router_sample * HIDDEN + k) // 2, 2) for s, k in sks]
                specs.append((mb(A_IN), (x_s * HIDDEN + xk) // 2, 1))
                v = poll(specs, batch=len(specs))
                stamp("router", tt, 5, lead=THREADS - 64)
                xa.append(bf2_f32(v[-1][0]))
                return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:-1]]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=1)
            stamp("router", tt, 2)
            # This task's normalized expert input goes out ahead of the gate GEMV.
            if x_ok:
                x_rstd = rstds[0]
                a0, a1 = xa[0]
                v0, v1 = a0 * x_rstd * xg[0], a1 * x_rstd * xg[1]
                if const_expr(use_fp8_block128):
                    q0, q1, qs = quant_scaled(v0, v1)
                    w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q0, q1, fx.Int32(0), False)) & 0xFFFF
                    w8n = _xshfl(w8, 1)
                    if lane % 2 == 0:  # FP8 bytes k .. k + 3 in one tagged word
                        put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                    d0, d1 = _fp8_roundtrip(q0, q1)
                    d0, d1 = d0 * qs, d1 * qs
                    if lane == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                elif const_expr(use_mxfp8_block32):
                    d0, d1, qs = quant_mxfp8(v0, v1)
                    w8 = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, d0, d1, fx.Int32(0), False)) & 0xFFFF
                    w8n = _xshfl(w8, 1)
                    if lane % 2 == 0:
                        put(mb("xq"), (x_s * HIDDEN + xk) // 4, w8 | (w8n << 16))
                    if lane % 16 == 0:
                        put(mb("xqs"), x_s * XQ_BLOCKS + x_blk, qs)
                    d0, d1 = d0 * qs, d1 * qs
                else:
                    d0, d1 = bf16_round(v0), bf16_round(v1)
                    put(mb("xq"), (x_s * HIDDEN + xk) // 2, bf16_pair(d0, d1))
                bo.buffer_store(fx.Vector.from_elements([d0, d1], fx.Float32), _rsrc(mb("xqd")), x_s * HIDDEN + xk)
            gpu.barrier()
            acc = run_units(u_r, R_CPW, R_CPW, pre)
            fx.ptr_store(fx.Vector.from_elements(acc, fx.Float32), red + (wave * 64 + lane) * 4)
            gpu.barrier()
            stamp("router", tt, 3)
            if tid < ROUTER_TILE:
                r = tid % ROUTER_TILE
                n = fx.Int32(0)
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
                put(mb("scores"), router_sample * N_EXPERTS + t * ROUTER_TILE + r, _sqrt_softplus(logit))
            stamp("router", tt, 4)

        def dn_route(bs):
            """Expert-down routing (wave s -> sample s): expert ids -> keys[s * 9 + slot],
            route weights -> dnw[]; the scores must have landed."""
            if wave < S:
                e, w = route_topk(wave, bs=bs)
                if lane < MOE_SLOTS:  # slot 0: the shared expert, then pick lane (slot lane + 1)
                    q = wave * MOE_SLOTS + (lane + 1) % MOE_SLOTS
                    lds_st(keys, q, (lane == TOP_K).select(fx.Int32(SHARED_EXPERT), e))
                    lds_st(dnw, q, (lane == TOP_K).select(fx.Float32(1.0), w))

        # ================================ 9. expert up/gate + SiLU
        # One 16-row group (8 gate + 8 up rows), with all eight waves splitting K.
        UG_NKC = HIDDEN // 64
        UG_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
        UG_W_BYTES = 2 * INTER * HIDDEN // (2 if use_mxfp4_weight else 1)
        UG_S_BYTES = 2 * INTER * (HIDDEN // 32) if use_mxfp4_weight else 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4

        if const_expr(S == 1):
            # task u takes intermediates (u % UG_PER_SLOT) * UG8 of routed slot
            # u // UG_PER_SLOT (the last slot for u < UG_PER_SLOT, which also take the
            # shared expert's); the 8 gate + 8 up rows are one MFMA row group and all
            # waves split K.  There are TOP_K * UG_PER_SLOT such tasks, which is only
            # BLOCKS for GLM-5/V3's own top-8 over INTER 256.
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            for u in range(start("ug"), N_UG_TASKS, G):
                u = fx.Int32(u)
                stamp("ug", u, 0)
                s_u, c = fx.Int32(0), u % UG_PER_SLOT
                has_sh = u < UG_PER_SLOT
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), u // UG_PER_SLOT)
                # The expert activation is recomputed here in parallel with the router.
                # MXFP8 uses the four independent 16-lane groups in each wave.
                if const_expr(use_mxfp8_block32):
                    NB = XQ_BLOCKS // (WAVES * 4)
                    ks_ = [((wave + j * WAVES) * 4 + lane // 16) * 32 + lane % 16 * 2 for j in range(NB)]
                else:
                    NB = PUBLISH_BLOCKS // WAVES
                    ks_ = [(wave + j * WAVES) * 128 + lane * 2 for j in range(NB)]
                r_gp = _rsrc(g_post)
                gps = [(ld_bf16(r_gp, k), ld_bf16(r_gp, k + 1)) for k in ks_]  # issued ahead of the wait
                bs = load_bias()
                w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2  # MFMA rows 0-7 gate, 8-15 up
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (INTER // 16) + c // 2  # this lane's output rows

                def u_ug8(cc, e, live=None):  # expert e's weights (loads return 0 unless live)
                    nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                    ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                    r_wug = bo.create_buffer_resource_from_addr(
                        w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                    )
                    r_sug = bo.create_buffer_resource_from_addr(
                        s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                    )
                    unit = wave * UG8_UNITS + cc
                    if const_expr(use_mxfp4_weight):
                        coefficients = None
                        if const_expr(use_mxfp8_block32):
                            coefficients = []
                            for sp in range_constexpr(4):

                                def coefficient(sp=sp, unit=unit):
                                    return _uniform_f32(lds_ld(misc, 8 + unit * 4 + sp))

                                coefficients.append(coefficient)
                        return unit_mxfp4(
                            r_wug,
                            r_sug,
                            w_rg,
                            unit,
                            HIDDEN,
                            unit * 64,
                            coefficients,
                            w_ln,
                        )
                    kc = unit * (2 if use_fp8_block128 else 1)
                    nwc = 2 if use_fp8_block128 else 1
                    wv = [
                        fx.Vector(
                            bo.buffer_load(r_wug, ((w_rg * UG_NKC + kc + h) * 64 + w_ln) * 4, vec_width=4, dtype=T.i32)
                        )
                        for h in range(nwc)
                    ]
                    sc = ld_f32(r_sug, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)
                    if const_expr(use_fp8_block128):
                        return (
                            "f8f8",
                            wv,
                            lambda: sc * _uniform_f32(lds_ld(misc, 8 + kc // 2)),
                            kc * 16 + (lane // 16) * 4,
                        )
                    return ("fp8", wv, sc, kc * 32 + (lane // 16) * 4)

                # the shared expert's weights do not depend on routing: prefetch them (the
                # later zero-weight MMAs of the other tasks are cheaper than a branch)
                pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_UNITS)]
                hint_wait(N_ROW_TILES, lambda k: (mb(A_IN), s_u * HIDDEN + k * ROW_TILE + ROW_TILE - 1), mark=("ug", u))
                # the sum of squares takes the router's element partition and order
                # (stage_x_rmsnorm, via the same _rmsnorm_tail_ks), so rstd -- and
                # every FP8 rounding -- is bit-identical
                nq4_ks, nq4_active = _rmsnorm_tail_ks(HIDDEN)
                NQ4 = len(nq4_ks)
                got = poll(
                    [(mb(A_IN), (s_u * HIDDEN + k) // 2, 2) for k in nq4_ks]
                    + [(mb(A_IN), (s_u * HIDDEN + k) // 2, 1) for k in ks_]
                )
                av = [bf2_f32(w[0]) for w in got[NQ4:]]
                ss = fx.Float32(0.0)
                for i in range_constexpr(NQ4):
                    for a in list(bf2_f32(got[i][0])) + list(bf2_f32(got[i][1])):
                        term = a * a
                        if const_expr(nq4_active is not None and i == NQ4 - 1):
                            term = nq4_active.select(term, fx.Float32(0.0))
                        ss = ss + term
                rstd = _rsq(block_sum(ss) * (1.0 / HIDDEN) + EPS)
                for j in range_constexpr(NB):
                    v0, v1 = av[j][0] * rstd * gps[j][0], av[j][1] * rstd * gps[j][1]
                    if const_expr(use_fp8_block128):
                        q0, q1, qs = quant_scaled(v0, v1)
                        st_f8(ks_[j], q0, q1)
                        if lane == 0:
                            lds_st(misc, 8 + wave + j * WAVES, qs)
                    elif const_expr(use_mxfp8_block32):
                        d0, d1, qs = quant_mxfp8(v0, v1)
                        lds_st(xs, ks_[j] // 2, bf16_pair(d0, d1))
                        if lane % 16 == 0:
                            block = (wave + j * WAVES) * 4 + lane // 16
                            lds_st(misc, 8 + block, qs)
                    else:
                        lds_st(xs, ks_[j] // 2, bf16_pair(v0, v1))
                if wave == 0:
                    e, w = route_topk(s_u, bs=bs)
                    if lane == slot - 1:
                        lds_st(keys, 0, e)
                        lds_st(misc, 0, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, 0))
                post = [u_ug8(cc, e_sel) for cc in range(UG8_UNITS)]
                reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                gpu.barrier()
                reduce_rows(
                    1, mma_units([fx.Float32(0.0) for _ in range(4)], post), lambda rl, n, v: lds_st(outs, 16 + rl, v)
                )
                stamp("ug", u, 3)
                gpu.barrier()
                if tid < UG8:  # threads 0-3: the shared expert's rows, 4-7: the routed slot's
                    r = (tid % (UG8 // 2)) * 2
                    o = (tid // (UG8 // 2)) * 16
                    g0, g1 = lds_ld(outs, o + r), lds_ld(outs, o + r + 1)
                    u0, u1 = lds_ld(outs, o + UG8 + r), lds_ld(outs, o + UG8 + r + 1)
                    if has_sh | (tid >= UG8 // 2):
                        put2(
                            mb("mid"),
                            (tid < UG8 // 2).select(fx.Int32(0), slot) * INTER + c * UG8 + r,
                            _swiglu(g0, u0, swiglu_limit),
                            _swiglu(g1, u1, swiglu_limit),
                        )
                if (c == 0) & (tid == 0):  # routing record (debug / tests)
                    put(mb("sel"), slot, e_sel)
                    put(mb("prob"), slot, lds_ld(misc, 0))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                        put(mb("prob"), 0, fx.Float32(1.0))
                stamp("ug", u, 4)
        elif const_expr(S > 1):
            # One eight-intermediate tile per CTA and sample. Shared weights use
            # the sample columns of one MFMA; routed tiles pipeline over samples.
            #
            # REP tiles per CTA, unrolled at BUILD time rather than walked with a
            # runtime `range(start("ug"), N_UG_TASKS, G)` the way S == 1 does: the
            # unit lists below are Python lists of traced operands that mma_units
            # consumes while tracing, so they cannot be carried by an scf.for.
            # N_UG_TASKS need not divide BLOCKS -- V4-Pro's top-6 over INTER 384
            # gives 288 tiles for 256 CTAs -- so a rep past the end is MASKED by
            # `live` rather than skipped, which keeps every barrier below reached
            # by the whole workgroup. `live` zeroes that rep's buffer descriptors'
            # num_records so its weight loads are killed in the texture unit, and
            # suppresses its publishes. The mask is redundant-work suppression, not
            # a correctness guard: `uu` clamps a dead rep onto a real tile, so
            # removing the publish mask alone changes no output (measured). What IS
            # load-bearing is the loop itself -- capping REP at 1 leaves tiles
            # 256..287 uncomputed and `down` then polls their `mid` slots forever.
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            XW = HIDDEN // (4 if use_fp8_block128 else 2)
            REP = (N_UG_TASKS + G - 1) // G
            u0 = fx.Int32(start("ug"))

            def ug8_units(c, w_rg, w_ln, s_rg, e, sample, live=None):
                nw = None if live is None else live.select(fx.Int32(UG_W_BYTES), fx.Int32(0))
                ns = None if live is None else live.select(fx.Int32(UG_S_BYTES), fx.Int32(0))
                rw = bo.create_buffer_resource_from_addr(
                    w_ug + fx.Int64(e) * fx.Int64(UG_W_BYTES), num_records_bytes=nw
                )
                rs = bo.create_buffer_resource_from_addr(
                    s_ug + fx.Int64(e) * fx.Int64(UG_S_BYTES), num_records_bytes=ns
                )
                sn = n_sel() if sample is None else fx.Int32(sample)
                units = []
                for cc in range_constexpr(UG8_UNITS):
                    unit = wave * UG8_UNITS + cc
                    if const_expr(use_mxfp4_weight):
                        coefficients = None
                        if const_expr(use_mxfp8_block32):
                            coefficients = []
                            for sp in range_constexpr(4):

                                def coefficient(sp=sp, unit=unit, sn=sn):
                                    return lds_ld(misc, 8 + sn * XQ_BLOCKS + unit * 4 + sp)

                                coefficients.append(coefficient)
                        units.append(
                            unit_mxfp4(
                                rw,
                                rs,
                                w_rg,
                                unit,
                                HIDDEN,
                                sn * XW + unit * 64,
                                coefficients,
                                w_ln,
                            )
                        )
                        continue
                    kc = unit * (2 if use_fp8_block128 else 1)
                    nwc = 2 if use_fp8_block128 else 1
                    wv = [
                        fx.Vector(
                            bo.buffer_load(rw, ((w_rg * UG_NKC + kc + j) * 64 + w_ln) * 4, vec_width=4, dtype=T.i32)
                        )
                        for j in range(nwc)
                    ]
                    sc = ld_f32(rs, (s_rg * 16 // SCALE_BM) * (HIDDEN // 128) + kc // 2)

                    # Bind each chunk's operands; the deferred scale follows staging.
                    if const_expr(use_fp8_block128):

                        def coefficient(sc=sc, kb=kc // 2, sn=sn):
                            return sc * lds_ld(misc, 8 + sn * XQ_BLOCKS + kb)

                        units.append(("f8f8", wv, coefficient, sn * XW + kc * 16 + (lane // 16) * 4))
                    else:
                        units.append(("fp8", wv, sc, sn * XW + kc * 32 + (lane // 16) * 4))
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
                        put2(
                            mb("mid"),
                            (sn * MOE_SLOTS + sl) * INTER + c * UG8 + r,
                            _swiglu(g0, v0, swiglu_limit),
                            _swiglu(g1, v1, swiglu_limit),
                        )
                if (c == 0) & (tid < S if shared else tid == 0):
                    sn = tid if shared else fx.Int32(sample)
                    sl = fx.Int32(0) if shared else slot
                    if live:
                        put(mb("sel"), sn * MOE_SLOTS + sl, lds_ld(keys, sn * MOE_SLOTS + sl))
                        put(mb("prob"), sn * MOE_SLOTS + sl, lds_ld(dnw, sn * MOE_SLOTS + sl))

            def ug8_tile(u):
                """This CTA's tile for one rep: (live, c, slot, has_sh, w_rg, w_ln, s_rg).

                ``u`` is clamped so a dead rep's LDS and descriptor indices stay in
                range; `live` is what actually suppresses its effects."""
                live = u < N_UG_TASKS
                uu = fx.min(u, fx.Int32(N_UG_TASKS - 1))
                c = uu % UG_PER_SLOT
                has_sh = uu < UG_PER_SLOT
                slot = has_sh.select(fx.Int32(MOE_SLOTS - 1), uu // UG_PER_SLOT)
                w_rg = ((lane % 16) // 8) * (INTER // 16) + c // 2
                w_ln = (lane & -16) | ((c % 2) * 8 + lane % 8)
                s_rg = (lane // 32) * (INTER // 16) + c // 2
                return live, uu, c, slot, has_sh, w_rg, w_ln, s_rg

            # Rep 0's weights are prefetched ahead of the routing wait and the input
            # staging, as the single-tile version did; dn_route and stage_moe_input
            # are CTA-global, so they stay out of the rep loop and every rep reads
            # the one staged input.
            live0, uu0, c0, slot0, has_sh0, wr0, wl0, sr0 = ug8_tile(u0)
            shared_pre = ug8_units(c0, wr0, wl0, sr0, fx.Int32(SHARED_EXPERT), None, has_sh0 & live0)
            dn_route(load_bias())
            gpu.barrier()
            cur = ug8_units(c0, wr0, wl0, sr0, _uniform(lds_ld(keys, slot0)), 0, live0)
            stage_moe_input(list(range(S)))
            gpu.barrier()
            for rep in range_constexpr(REP):
                if const_expr(rep == 0):
                    live, uu, c, slot, has_sh = live0, uu0, c0, slot0, has_sh0
                    w_rg, w_ln, s_rg = wr0, wl0, sr0
                else:
                    live, uu, c, slot, has_sh, w_rg, w_ln, s_rg = ug8_tile(u0 + rep * G)
                    shared_pre = ug8_units(c, w_rg, w_ln, s_rg, fx.Int32(SHARED_EXPERT), None, has_sh & live)
                    cur = ug8_units(c, w_rg, w_ln, s_rg, _uniform(lds_ld(keys, slot)), 0, live)
                if has_sh:
                    reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], shared_pre), emit_out(16))
                    gpu.barrier()
                    ug8_emit(c, slot, 0, True, live)
                for sample in range_constexpr(S):
                    stamp("ug", sample * N_UG_TASKS + uu, 0, pred=live)
                    pre = cur
                    if const_expr(sample + 1 < S):
                        cur = ug8_units(
                            c,
                            w_rg,
                            w_ln,
                            s_rg,
                            _uniform(lds_ld(keys, (sample + 1) * MOE_SLOTS + slot)),
                            sample + 1,
                            live,
                        )
                    reduce_rows(1, mma_units([fx.Float32(0.0) for _ in range(4)], pre), emit_out(16))
                    gpu.barrier()
                    ug8_emit(c, slot, sample, False, live)
                    stamp("ug", sample * N_UG_TASKS + uu, 4, pred=live)

        # =============== 10. expert down + route weighting + MoE TP reduce
        DN_NKC = INTER // 64
        # dn_tile keeps this a multiple of 16, so every tile starts at row 0 of a
        # group (dn_off == 0) and reduce_rows produces exactly the tile's rows.
        assert DN_TILE % 16 == 0 and HIDDEN % DN_TILE == 0, f"down tile {DN_TILE} must divide {HIDDEN} by 16s"
        DN_R = DN_TILE // 16  # 16-row groups touched by a tile
        DN_WPR = WAVES // DN_R
        DN_UNIT_K = 128 if (use_fp8_block128 or use_mxfp4_weight) else 64
        DN_UNITS_PER_SLOT = INTER // DN_UNIT_K
        DN_NU = S * MOE_SLOTS * DN_UNITS_PER_SLOT
        DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
        DN_BLK = S * MOE_SLOTS * INTER // 128
        DN_W_BYTES = HIDDEN * INTER // (2 if use_mxfp4_weight else 1)
        DN_S_BYTES = HIDDEN * (INTER // 32) if use_mxfp4_weight else HIDDEN // SCALE_BM * (INTER // 128) * 4
        DN_BATCH = 9  # 128-k chunks per wave in flight / prefetched before the mid wait
        for t in range(start("down"), N_DN_TILES, G):
            t = fx.Int32(t)
            stamp("down", t, 0)
            if const_expr(S == 1):  # multi-sample routing was staged before up/gate
                dn_route(load_bias())
            gpu.barrier()
            gu = wave // DN_WPR
            dn_rg = t * DN_TILE // 16
            dn_off = t * DN_TILE % 16
            # this lane's row, as a tile row; rows outside the tile load their lane ^ 8 twin
            # (same cache lines) and are dropped in the output
            dn_lr = gu * 16 + lane % 16 - dn_off
            dn_ln = ((dn_lr >= 0) & (dn_lr < DN_TILE)).select(lane, lane ^ 8)

            def u_dn(cc):  # cc: 128-k chunk of this wave
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                q = fx.min(qu, DN_NU - 1)  # unit index over (sample, slot, K chunk)
                s_q = q // (MOE_SLOTS * DN_UNITS_PER_SLOT)
                slot_q = (q // DN_UNITS_PER_SLOT) % MOE_SLOTS
                kc = q % DN_UNITS_PER_SLOT
                e = _uniform(lds_ld(keys, s_q * MOE_SLOTS + slot_q))
                wb = bo.create_buffer_resource_from_addr(
                    w_dn + fx.Int64(e) * fx.Int64(DN_W_BYTES),
                    num_records_bytes=None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_W_BYTES), fx.Int32(0)),
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_dn + fx.Int64(e) * fx.Int64(DN_S_BYTES),
                    num_records_bytes=None if DN_NU % DN_WPR == 0 else live.select(fx.Int32(DN_S_BYTES), fx.Int32(0)),
                )

                if const_expr(use_mxfp4_weight):
                    coefficients = []
                    if const_expr(use_mxfp8_block32):
                        for sp in range_constexpr(4):

                            def coefficient(sp=sp, q=q, s_q=s_q):
                                return (lane % 16 == s_q).select(
                                    _uniform_f32(lds_ld(misc, q * 4 + sp)), fx.Float32(0.0)
                                )

                            coefficients.append(coefficient)
                    else:

                        def coefficient():
                            return (lane % 16 == s_q).select(
                                _uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), fx.Float32(0.0)
                            )

                        coefficients = coefficient
                    return unit_mxfp4(
                        wb,
                        sb,
                        dn_rg + gu,
                        kc,
                        INTER,
                        q * 64,
                        coefficients,
                        dn_ln,
                    )

                kc64 = kc * (2 if use_fp8_block128 else 1)
                if const_expr(use_fp8_block128):

                    def coef():  # mid block scale * route weight, only in this sample's column
                        return (lane % 16 == s_q).select(_uniform_f32(lds_ld(misc, q)), fx.Float32(0.0))

                    return unit_f8f8(wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, q * 32, coef, dn_ln)

                def coef():
                    return (lane % 16 == s_q).select(_uniform_f32(lds_ld(dnw, s_q * MOE_SLOTS + slot_q)), 0.0)

                return unit_fp8(wb, sb, dn_rg + gu, kc64, DN_NKC, INTER, 128, q * 32, coef, dn_ln)

            # the experts are known: stream their down weights while up/gate finishes
            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_UPW))]
            hint_wait(
                N_UG,
                lambda k: (
                    mb("mid"),
                    (k // (MOE_SLOTS * N_UG_PER_SLOT) * MOE_SLOTS + (k // N_UG_PER_SLOT) % MOE_SLOTS) * INTER
                    + (k % N_UG_PER_SLOT) * UG_TILE
                    + UG_TILE
                    - 1,
                ),
                mark=("down", t),
            )
            mids = get2_many(
                [
                    (mb("mid"), fx.min(wave + b * WAVES, DN_BLK - 1) * 128 + lane * 2)
                    for b in range((DN_BLK + WAVES - 1) // WAVES)
                ]
            )
            stamp("down", t, 2)
            # Stage expert intermediates in the activation format selected for this mode.
            for b in range_constexpr((DN_BLK + WAVES - 1) // WAVES):
                blk = wave + b * WAVES
                if blk < DN_BLK:
                    if const_expr(use_fp8_block128):
                        q0, q1, qs = quant_scaled(mids[b][0], mids[b][1])
                        st_f8(blk * 128 + lane * 2, q0, q1)
                        if lane == 0:
                            lds_st(misc, blk, qs * lds_ld(dnw, blk // (INTER // 128)))
                    elif const_expr(use_mxfp8_block32):
                        d0, d1, qs = quant_mxfp8(mids[b][0], mids[b][1])
                        lds_st(xs, blk * 64 + lane, bf16_pair(d0, d1))
                        if lane % 16 == 0:
                            scale_group = blk * 4 + lane // 16
                            lds_st(misc, scale_group, qs * lds_ld(dnw, blk // (INTER // 128)))
                    else:
                        lds_st(xs, blk * 64 + lane, bf16_pair(mids[b][0], mids[b][1]))
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
                    fx.Vector.from_elements([v0, v1], fx.Float32).to(fx.BFloat16), _rsrc(x_out), s * HIDDEN + row
                )

            if const_expr(HC > 1):
                hc_stage_coef(1)
                peer_reduce(
                    "ffn",
                    t,
                    None,
                    lambda s, row, v0, v1: hc_post(
                        s,
                        row,
                        v0,
                        v1,
                        lambda s_, j, r_: get(mb("a"), ((s_ * HC + j) * HIDDEN + r_) // 2),
                        lambda s_, k, r_, o0, o1: bo.buffer_store(
                            fx.Vector.from_elements([o0, o1], fx.Float32).to(fx.BFloat16),
                            _rsrc(x_out),
                            (s_ * HC + k) * HIDDEN + r_,
                        ),
                    ),
                    tile=DN_TILE,
                )
            else:
                peer_reduce("ffn", t, mb("a"), store_x, tile=DN_TILE)
            gpu.barrier()
            stamp("down", t, 4)

    @flyc.jit
    def launch_dsv4(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        dest_rows: Int64,
        indices: Int64,
        rope_cos: Int64,
        rope_sin: Int64,
        g_in: Int64,
        g_q: Int64,
        g_kv: Int64,
        g_post: Int64,
        attn_sink: Int64,
        ape: Int64,
        g_ckv: Int64,
        kv_state: Int64,
        score_state: Int64,
        i_ape: Int64,
        g_ickv: Int64,
        i_kv_state: Int64,
        i_score_state: Int64,
        i_cache: Int64,
        hc_attn_fn: Int64,
        hc_attn_sb: Int64,
        hc_ffn_fn: Int64,
        hc_ffn_sb: Int64,
        w_qkv_a: Int64,
        s_qkv_a: Int64,
        w_q_b: Int64,
        s_q_b: Int64,
        w_i_q_b: Int64,
        s_i_q_b: Int64,
        i_w: Int64,
        w_o_a: Int64,
        s_o_a: Int64,
        w_o_b: Int64,
        s_o_b: Int64,
        w_r: Int64,
        bias: Int64,
        w_ug: Int64,
        s_ug: Int64,
        w_dn: Int64,
        s_dn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        state_slots: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        st_ic: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        dsv4_kernel(
            h_in,
            x_out,
            cur_pos,
            kv_cache,
            dest_rows,
            indices,
            rope_cos,
            rope_sin,
            g_in,
            g_q,
            g_kv,
            g_post,
            attn_sink,
            ape,
            g_ckv,
            kv_state,
            score_state,
            i_ape,
            g_ickv,
            i_kv_state,
            i_score_state,
            i_cache,
            hc_attn_fn,
            hc_attn_sb,
            hc_ffn_fn,
            hc_ffn_sb,
            w_qkv_a,
            s_qkv_a,
            w_q_b,
            s_q_b,
            w_i_q_b,
            s_i_q_b,
            i_w,
            w_o_a,
            s_o_a,
            w_o_b,
            s_o_b,
            w_r,
            bias,
            w_ug,
            s_ug,
            w_dn,
            s_dn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            state_slots,
            rank,
            layer,
            st_kv,
            st_i,
            st_ic,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    return launch_dsv4
