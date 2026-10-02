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
    BLOCK_TOKENS,
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


def qkv_a_groups(n_tiles: int) -> int:
    """16-row groups per qkv_a task. Each task stages every sample's normalized input,
    and that staging is most of its time, so when the tiles outnumber the grid (CSA's
    compressor rows make 288) a second round would stage it all again: two groups per
    task instead, four waves each splitting K."""
    return 2 if n_tiles > BLOCKS and n_tiles % 2 == 0 else 1


def q_b_groups(n_tiles: int) -> int:
    """16-row groups per q_b task, by qkv_a_groups' rule: every task stages each
    sample's normalized q_a, and H * HEAD_DIM / Q_B_TILE = 512 tiles took two rounds
    of 256 CTAs, staging it all twice per CTA (S times the work at batch S)."""
    return qkv_a_groups(n_tiles)


def router_spt(S: int) -> int:
    """Samples per router task. One per task until S * N_ROUTER outgrows the grid
    (S = 8: 384 tasks, two rounds of ~13 us on the FFN's critical path); then two
    share a task, in the two B columns of each K-fold half that held copies."""
    return 2 if S * N_ROUTER > BLOCKS else 1
N_UG_PER_SLOT = INTER // UG_TILE


# gfx94x/95x cache policy bits (LLVM CPol): SC0 = 1, NT = 2, SC1 = 16.  SC1:SC0 is
# the coherence scope of the access itself: SC1 = device (past the per-XCD
# non-coherent caches), SC0|SC1 = system (peer GPUs over XGMI).
CM_DEV = 16
CM_SYS = 17
POLL_MAX = 12  # mailbox specs polled per batch
# Opt-in (build_dsv4_kernel's poll_timeout_us; off by default -- the bound costs ~3%
# of a layer and no hang has been seen in serving): a poll that has not seen its
# producer for this long gives up instead of spinning forever, flagging the launch
# (see `hang` below). 10 s is far past any real wait -- a whole layer is well under
# a millisecond, and TP ranks launch within milliseconds of each other.
POLL_TIMEOUT_US = 10_000_000
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
    kv_fp8: bool = False,
    indexer_hadamard: bool = True,
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
        # the top-k parts' bins, one set per radix digit
        ("tk_hist", S * 4 * n_topk_parts(max_seq, compress_ratio, index_head_dim) * 256 * pr or pr),
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
    """f32 pair -> E2M1 -> f32 pair (inputs already scaled into range), plus the
    codes themselves: ``a`` in the low nibble of the returned word, ``b`` above it.

    The scale operand is 1.0 and the scaling is done in f32 around this, as
    ``_fp8_to_bf16x8`` does: the hardware honours only the scale's exponent, and
    keeping the arithmetic explicit means this does not depend on which direction
    the instruction applies it.
    """
    one = as_ir_value(fx.Float32(1.0))
    word = fx.Int32(rocdl.cvt_scalef32_pk_fp4_f32(T.i32, as_ir_value(fx.Int32(0)), a, b, one, 0))
    v2 = fx.Vector.make_type(2, fx.Float32)
    out = fx.Vector(rocdl.cvt_scalef32_pk_f32_fp4(res=v2, src=as_ir_value(word), scale=one, src_sel_index=0))
    return out[0], out[1], word


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


def _fp8_to_bf16x8(w0, w1, scale=None):
    """Two dwords of 8 FP8 -> vector<8 x bf16> (exact: E4M3 is a subset of bf16).

    ``cvt_scalef32_pk_bf16_fp8`` only honours the scale's exponent, so ``scale``
    must be a power of two (an E8M0 block scale, applied exactly); a non-power-of-
    two block scale stays 1 here and goes on the MFMA partials instead.
    """
    one = as_ir_value(fx.Float32(1.0) if scale is None else scale)
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


def n_topk_parts(max_seq: int, compress_ratio: int, index_head_dim: int) -> int:
    """CTAs the indexer's top-k splits its candidates over.

    The select is one task per sample, so at batch 1 it ran on ONE CTA while 255
    idled -- at a 1M context that is 6 MB read at 19 GB/s, single-CU territory,
    and 72% of the layer. Each part scans its own share and the parts agree on
    each radix digit by summing one another's bins.

    Sized to leave a part a couple of trips of its own, and capped: the bins the
    parts trade cost TK_PARTS * (TK_PARTS - 1) reads per digit, which grows
    faster than the scan it is splitting, so past a point more parts is more
    exchange for less work. ONE part compiles to exactly the single-CTA code,
    which is why every shape below the threshold is untouched by any of this.
    """
    if not index_head_dim:
        return 0
    return min(16, max(1, n_compressed(max_seq, compress_ratio) // 4096))


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
    kv_fp8: bool = False,
    indexer_hadamard: bool = True,
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
        ("qkv_a", n_qkv_a // qkv_a_groups(n_qkv_a)),
        ("cache", 1),
        # the compressed entry has to land before the attention gathers it
        ("cmp", S if compress_ratio else 0),
        # the indexer's compressed entry, for the scoring that selects keys
        ("i_cmp", S if index_head_dim else 0),
        ("q_b", heads * head_dim // Q_B_TILE // q_b_groups(heads * head_dim // Q_B_TILE)),
        ("q_norm", S * heads),
        # the indexer's query, and its rope / rotation / FP4 tail
        ("i_q_b", index_heads * index_head_dim // Q_B_TILE if index_head_dim else 0),
        ("i_q", S if index_head_dim else 0),
        # the per-head score weights, then the score of every compressed entry
        ("i_wp", S if index_head_dim else 0),
        ("i_score", S * n_score_tiles(max_seq, compress_ratio, index_head_dim)),
        # ... and the top-k over them, which the split stage waits on
        ("i_topk", S * n_topk_parts(max_seq, compress_ratio, index_head_dim)),
        ("split", S * ((window if n_keys is None else n_keys) // SPLIT_KEYS)),
        ("uv", S * (heads * head_dim // UV_TILE)),
        ("o_a", S * o_groups * o_lora // ROW_TILE),
        ("o_b", hidden // ROW_TILE),
        # ... and for the ffn side, once o_b has produced the new residual stream
        ("hcd_f", S * hc_tasks),
        # none: the router contracts the FFN side's streams itself
        ("hcc_f", 0),
        ("router", S * N_ROUTER // router_spt(S)),
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
    poll_timeout_us: int | None = None,
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
    kv_fp8: bool = False,
    indexer_hadamard: bool = True,
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

    ``hang`` is one int32: with ``poll_timeout_us`` set, a poll that waits longer
    stores the launch's tag there and the launch winds down (see ``poll``); left
    None, polls spin unbounded and ``hang`` is never written.
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
    assert TOP_K <= 8, "ug keeps the routing weights in misc[:8], below the quant scales"
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
    # An MXFP4 bank holds only the routed experts; the shared expert stays FP8 128x128
    # beside it (w_sug / w_sdn), as the checkpoint stores it and ATOM runs it. Its
    # activation is the routed experts' (MXFP8 or bf16).
    SHARED_FP8 = use_mxfp4_weight
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
    # KV rows in ATOM's fp8 layout: a NoPE plane of KV_ROW_BYTES per row (NOPE_DIM
    # FP8 bytes, then each 64-wide group's E8M0 byte twice, then padding) and a
    # bf16 RoPE plane [rows, ROPE_DIM]. Otherwise one bf16 plane [rows, HEAD_DIM].
    KV_FP8 = kv_fp8
    BOUNDED_POLL = poll_timeout_us is not None
    POLL_TIMEOUT_TICKS = (poll_timeout_us or 0) * 100  # s_memrealtime runs at 100 MHz
    INDEXER_HADAMARD = indexer_hadamard
    if CR:
        assert BLOCK_TOKENS % CR == 0, "a block holds a whole number of compressed entries"
    KV_ROW_BYTES = 512
    if KV_FP8:
        assert NOPE_DIM % 64 == 0 and HEAD_DIM - NOPE_DIM == ROPE_DIM and THREADS == HEAD_DIM
    IW = C_COFF * IHD
    # Compressed entries are PAGED, as ATOM keeps them: entry e of sequence s lives
    # in block block_tables[s][e // K_PB], K_PB = BLOCK_TOKENS // CR entries a block.
    # A compressed KV row is dest_rows[1, s] + block * env_rows + e % K_PB (ATOM:
    # 0, its tables, its envelope height; a contiguous pool: an identity table
    # with env_rows = K_PB gives dest_rows[1, s] + e). The indexer's key cache is
    # ATOM's FP4 pool: per block, codes [IHD / 32 groups][K_PB][16 bytes] and E8M0
    # scales [IHD / 32][K_PB] with the entry axis interleaved (16-row runs:
    # byte (e % 16) * 4 + (e % K_PB) // 16), at a per-sequence base
    # state_slots[s] * st_ic bytes (the scale pool's base is 1/16 of it).
    K_PB = BLOCK_TOKENS // CR if CR else 1
    IC_GRP_WORDS = K_PB * 16 // 4  # one 32-element group of one block, in dwords
    IC_BLK_WORDS = (IHD // 32) * IC_GRP_WORDS if IHD else 0
    IC_S_BLK = (IHD // 32) * K_PB if IHD else 0  # scale bytes of one block
    CW = C_COFF * HEAD_DIM  # width of one state row
    # Rows of state, a RING indexed by absolute position (row = pos % C_ROWS), as
    # ATOM's compressor keeps it: the window pooled at position p is rows
    # (p + 1 + i) % C_ROWS for i in [0, C_ROWS), oldest first. No row ever moves.
    C_ROWS = C_COFF * CR
    OVERLAP = C_COFF > 1
    # Rows of state the pooling loop folds per trip. Its online softmax carries
    # (max, den, num) serially, but the LOADS do not depend on the carry, so
    # issuing a trip's worth together is what keeps the loop off memory latency:
    # one row at a time, HCA's 128 rows cost 270 ns each and 37% of the layer.
    CMP_CHUNK = max(c for c in range(1, 9) if C_ROWS % c == 0) if C_ROWS else 1
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
    # waves that share the coefficient poll, and the hcd tasks each takes: one
    # poll batch apiece (POLL_MAX)
    HC_PW = min(WAVES, -(-HC_TASKS // 12)) if HC > 1 else 1
    HC_TPW = -(-HC_TASKS // HC_PW) if HC > 1 else 1
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
    # split/merge sized per launch by the live keys (see live_splits): HCA's list
    # grows with max_seq, CSA's is the window plus a fixed index_topk
    LIVE_SPLITS = bool(compress_ratio) and not index_head_dim and N_KEYS > window
    # The flash merge in `uv` gives each split ONE THREAD, which computes that
    # split's exp(m - M)/L into misc[split]; the merge then reads
    # misc[0 .. N_SPLIT). Run it over a single wave while the splits fit in one
    # (the reduction is then register-only, no LDS and no barrier) and over the
    # whole block past that. Overrunning it is silent -- the splits above the
    # width get no weight, the softmax normalises over a prefix, and the read
    # runs past misc -- so it is asserted rather than left to NaN.
    UV_WIDE = N_SPLIT > THREADS // WAVES
    # Largest chunk up to 8 that divides the split count: bounds how many
    # per-split words the `uv` merge holds live at once (see the loop below).
    UV_CHUNK = max(c for c in range(1, 9) if N_SPLIT % c == 0)
    assert N_SPLIT <= THREADS, (
        f"{N_SPLIT} key splits exceeds the {THREADS} threads the block-wide `uv` merge has. "
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
    # The top-k re-reads its candidates each radix pass rather than holding them,
    # so these bound a loop trip count, not a register budget. A thread takes
    # TK_PER CONSECUTIVE candidates per trip -- two 16-byte loads, the widest the
    # ISA has -- and the threads stride over those groups, so a wave's trip is one
    # contiguous run.
    TK_PER = 4
    TK_BINS = 256  # 8-bit radix digit; 4 passes cover the key
    # Copies of the bin array, one per lane group. A score's top digit is its sign
    # and exponent, which barely vary across a context's worth of logits, so nearly
    # every lane of a wave bins to the SAME counter and the read-modify-writes
    # serialize -- that one conflict was 700 of the 1113 us the select took at 1M.
    # Splitting by lane spreads them over consecutive words, so over banks.
    TK_REP = 16
    TK_BC = TK_BINS * TK_REP  # the four words past the bins that broadcast a pass's result
    TK_PARTS = n_topk_parts(max_seq, compress_ratio, index_head_dim)
    # trips PER PART: part q takes every TK_PARTS-th trip, starting at q
    TK_TRIPS = max(1, -(-N_COMP // (THREADS * TK_PER * max(TK_PARTS, 1)))) if N_COMP else 1
    # the index list pads to a whole key tile, so its compressed half has room for
    # more than the indexer will ever pick; the surplus is filled with -1
    N_ISEL = N_KEYS - window
    N_ISCORE = n_score_tiles(max_seq, compress_ratio, index_head_dim)
    if IHD:
        assert window % SPLIT_KEYS == 0, "a key tile must fall wholly inside or outside the window"
        assert N_KEYS >= window + N_INDEX, "the index list must hold the window and the pick"
        assert IHD % 8 == 0 and N_COMP > 0
        # Load-bearing twice: it makes the clamped base 16-byte aligned for the
        # wide read, and it puts a clamped group wholly past n_live, so the wrong
        # candidates it then reads are all masked off.
        assert N_COMP % TK_PER == 0, "the top-k reads whole groups of TK_PER candidates"
        # Every part polls every other part's bins, so they must run CONCURRENTLY.
        # Two parts sharing a CTA would have the first spin for a second that the
        # same CTA has not started: a hang, not a wrong answer.
        assert S * TK_PARTS <= BLOCKS, "each top-k part needs its own CTA"
        # every thread holds its share of the candidates in registers; 32 is a 1M
        # context, V4's longest
        assert TK_TRIPS * TK_PER <= 32, "the top-k's candidates per thread must fit in registers"
        assert IHD == 128, "the indexer's Hadamard is written for a 128-wide head"
        assert K_PB == 64, "the FP4 pool's scale interleave is written for 64 entries a block (4 runs of 16)"
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
        # radix-select bins, replicated, plus two words to broadcast the winning digit
        hist: fx.Array[fx.Int32, (TK_BC + 4) if IHD else 1, 16]

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def dsv4_kernel(
        h_in: Int64,
        x_out: Int64,
        cur_pos: Int64,
        kv_cache: Int64,
        kv_rope: Int64,
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
        w_sug: Int64,
        s_sug: Int64,
        w_sdn: Int64,
        s_sdn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        hang: Int64,
        state_slots: Int64,
        tok_ids: Int64,
        tid2eid: Int64,
        block_tables: Int64,
        i_cache_s: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        st_ic: Int32,
        use_hash: Int32,
        bt_stride: Int32,
        env_rows: Int32,
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
        hist = lds.hist.ptr
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

        def bt_block(s, e):
            """The physical block holding sequence ``s``'s compressed entry ``e``."""
            return fx.Int32(bo.buffer_load(_rsrc(block_tables), s * bt_stride + e // K_PB, vec_width=1, dtype=T.i32))

        def comp_row(s, e):
            """Plane row of sequence ``s``'s compressed entry ``e`` (see K_PB)."""
            return ld_dest(1, s) + bt_block(s, e) * env_rows + e % K_PB

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

        # A serving pool is one plane of hundreds of millions of rows, with the
        # per-request slots at its top: ATOM's V4 pool puts a row 115 GB past a
        # layer's view and a slot's state tens of GB past slot 0. Any 32-bit offset
        # from the pool base wraps there, and a buffer resource cannot reach past
        # 4 GB of its base anyway. So the large part of every such address is
        # 64-bit and becomes the resource's base; offsets inside a row or a slot
        # stay small. (Every test pool fits in 4 GB, which is how this hid.)
        def slot_rsrc(ptr, s, stride):
            """A resource at sample ``s``'s slot of an f32 rolling-state pool."""
            slot = _uniform(bo.buffer_load(r_slot, s, vec_width=1, dtype=T.i32))
            return _rsrc(ptr + fx.Int64(slot) * fx.Int64(stride) * 4)

        def row_rsrc(ptr, row, row_bytes):
            """A resource at plane row ``row`` (wave-uniform) of rows ``row_bytes`` wide."""
            return _rsrc(ptr + fx.Int64(_uniform(row)) * row_bytes)

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

        def now_ticks():
            """``s_memrealtime``: a 100 MHz clock, the same on every CU."""
            return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))

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

            def unpack(v):
                outs_, e = [], 0
                for _, _, n in specs:
                    outs_.append([v[e + 2 * q] for q in range(n)])
                    e += 2 * n
                return outs_

            def pending(v):
                bad = v[1] != tag
                for e in range_constexpr(3, nw, 2):
                    bad = bad | (v[e] != tag)
                return bad

            # Unbounded by default, as the GLM kernel this grew from. Bounded
            # (poll_timeout_us set): past the timeout the poll stores this
            # launch's tag into ``hang`` and gives up on garbage, and every other
            # poll of the launch that sees that word gives up at once -- a CTA
            # facing a dead peer would otherwise wait out the timeout on each of
            # its polls in turn. The launch then runs to its end and the host
            # reads ``hang`` (Dsv4Variant.hang_detected). stop: 0 waiting, 1
            # flagged, 2 timed out. The bound costs ~3% of an S=8 layer (HCA 149
            # -> 153 us, CSA 166 -> 171), the same without the flag read or with
            # a spin count for the clock: it is the exit itself, at every inlined
            # poll site.
            v = load_all()
            if const_expr(BOUNDED_POLL):
                t0 = now_ticks()
                stop = fx.Int32(0)
                while pending(v) & (stop == 0):
                    rocdl.s_nop(0)
                    v = load_all()
                    flagged = _uniform(bo.buffer_load(_rsrc(hang), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) == tag
                    late = (now_ticks() - t0) > fx.Int64(POLL_TIMEOUT_TICKS)
                    stop = late.select(fx.Int32(2), flagged.select(fx.Int32(1), fx.Int32(0)))
                if stop == 2:
                    bo.buffer_store(tag, _rsrc(hang), 0, cache_modifier=CM_DEV)
            else:
                while pending(v):
                    rocdl.s_nop(0)
                    v = load_all()
            return unpack(v)
            t0 = now_ticks()
            stop = fx.Int32(0)
            while pending(v) & (stop == 0):
                rocdl.s_nop(0)
                v = load_all()
                flagged = _uniform(bo.buffer_load(_rsrc(hang), 0, vec_width=1, dtype=T.i32, cache_modifier=CM_DEV)) == tag
                late = (now_ticks() - t0) > fx.Int64(POLL_TIMEOUT_TICKS)
                stop = late.select(fx.Int32(2), flagged.select(fx.Int32(1), fx.Int32(0)))
            if stop == 2:
                bo.buffer_store(tag, _rsrc(hang), 0, cache_modifier=CM_DEV)
            return unpack(v)

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

        def _other_parts(part):
            """Every top-k part but this one, starting just after it.

            Walking from ``part + 1`` rather than from 0 means a part never reads a
            slot it wrote itself -- which would be a bet on seeing your own global
            store, the very thing the ``cnew`` mailbox exists because you cannot
            make."""
            return [(part + 1 + k) % TK_PARTS for k in range(TK_PARTS - 1)]

        def part_keys(sbase, part, n_live, n_parts):
            """This thread's candidates in top-k part ``part``, as order-preserving keys.

            f32 bits -> a signed int32 whose ORDER matches the float's (flip the low
            31 bits of negatives, which puts -2 below -1 and both below 0), then ^
            MIN_I32 into the unsigned domain, where MSB-first prefixes work.

            Returns (trip bases, keys): trip j covers ``TK_PER`` consecutive
            candidates from its base, and the threads stride over those groups, so a
            wave's trip is one contiguous run. Polled ONCE, all trips in one batched
            poll, and held in registers for every radix pass and the compaction --
            re-walking them per pass was eight serial round trips a pass. Each base
            is clamped so a whole vector stays in range; the caller masks on the
            UNCLAMPED candidate index, which is past the end exactly when it was
            clamped, so a clamped trip contributes nothing.

            A trip past ``n_live`` polls candidate 0's vector instead: i_score skips
            the tiles past the live entries, so their slots are never written this
            launch, and the caller masks those candidates anyway.

            ``n_parts`` is how many parts share the candidates this launch: TK_PARTS,
            or 1 when part 0 holds them all (see i_topk's ``solo``)."""
            cbs = [((fx.Int32(j) * n_parts + part) * THREADS + tid) * TK_PER for j in range(TK_TRIPS)]
            specs = []
            for cb in cbs:
                a = (cb < n_live).select(fx.min(cb, fx.Int32(N_COMP - TK_PER)), fx.Int32(0))
                specs += [(mb("i_score"), sbase + a + 2 * q, 2) for q in range(TK_PER // 2)]
            ws = [w[e] for w in poll(specs) for e in range(2)]
            return cbs, [(w ^ ((w >> 31) & 0x7FFFFFFF)) ^ MIN_I32 for w in ws]

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

        def unit_fp8mx(w_rsrc, s_rsrc, rg, s_rg, kc, K, b_word, coef=None, ln=None):
            """One 128-K chunk ``kc`` of row group ``rg`` of a packed FP8 matrix (128x128
            block scales; ``s_rg`` = the row group of this lane's OUTPUT rows) against the
            bf16 activation at LDS word ``b_word`` (any MXFP8 scale already folded in);
            ``coef`` = one extra factor for the whole unit (None = 1)."""
            ln = lane if ln is None else ln
            wv = [
                fx.Vector(
                    bo.buffer_load(w_rsrc, ((rg * (K // 64) + kc * 2 + h) * 64 + ln) * 4, vec_width=4, dtype=T.i32)
                )
                for h in range(2)
            ]
            s = ld_f32(s_rsrc, (s_rg * 16 // SCALE_BM) * (K // 128) + kc)
            return ("fp8mx", wv, (s, coef), b_word + (lane // 16) * 4)

        def unit_mxfp4(w_rsrc, s_rsrc, rg, kc, K, b_word, coef=None, ln=None):
            """Issue one packed 128-K MXFP4 tile and its four per-row E8M0 scales."""

            ln = lane if ln is None else ln
            raw = fx.Vector(bo.buffer_load(w_rsrc, ((rg * (K // 128) + kc) * 64 + ln) * 4, vec_width=4, dtype=T.i32))
            row = rg * 16 + ln % 16
            packed_scale = fx.Int32(bo.buffer_load(s_rsrc, row * (K // 128) + kc, vec_width=1, dtype=T.i32))
            # the four E8M0 scales stay packed in one register until their conversion
            # (unpacked here they held four VGPRs per in-flight unit, and the kernel is
            # at its VGPR ceiling: they spilled)
            return ("mxfp4", (raw, packed_scale), coef, b_word + (lane // 16) * 4)

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
                if const_expr(unit_format == "fp8mx"):
                    # one factor per unit (the weight's block scale, times a route
                    # weight if any), so the four K32 MFMAs chain into one partial
                    ws, f = coef
                    c = fx.Vector.filled(4, 0.0, fx.Float32)
                    for sp in range_constexpr(4):
                        a = _fp8_to_bf16x8(wv[sp // 2][(sp % 2) * 2], wv[sp // 2][(sp % 2) * 2 + 1])
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    f = ws if const_expr(f is None) else ws * (f() if const_expr(callable(f)) else f)
                    acc = [acc[e] + c[e] * f for e in range(4)]
                    continue
                if const_expr(callable(coef) and unit_format != "mxfp4"):
                    coef = coef()
                if const_expr(unit_format == "mxfp4"):
                    # The four K32 MFMAs chain: with no factor straight into the running
                    # sum, else into one partial that takes its factor once per unit.
                    raw, packed_scale = wv
                    assert not isinstance(coef, list), "per-K32 factors are folded into the operands"
                    c = fx.Vector.from_elements(acc, fx.Float32) if coef is None else fx.Vector.filled(4, 0.0, fx.Float32)
                    for sp in range_constexpr(4):
                        sc = ((packed_scale.shrui(fx.Int32(sp * 8)) & fx.Int32(0xFF)) << fx.Int32(23)).bitcast(fx.Float32)
                        a = _mxfp4_to_bf16x8(raw[sp], sc)
                        b = fx.ptr_load(xs + (bw + sp * 16), result_type=v4f).bitcast(fx.BFloat16)
                        c = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))
                    if const_expr(coef is None):
                        acc = [c[e] for e in range(4)]
                    else:
                        f = coef() if const_expr(callable(coef)) else coef
                        acc = [acc[e] + c[e] * f for e in range(4)]
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
            """Per-16-lane/32-value MXFP8 quantization with an E8M0 scale.

            The scale is rounded UP to a power of two, so the block max never clips:
            rounding it to nearest let amax / scale reach 672 and clamp to 448, which on
            real activations (outlier-heavy blocks) cost 7-9% of the input's norm."""

            amax = subgroup16_max(fx.max(fmath.absf(a0), fmath.absf(a1)))
            nz = amax > 0.0
            scale = nz.select(_pow2_ceil(amax * (1.0 / FP8_MAX)), fx.Float32(1.0))
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
                # each 8-byte chunk with its own 32-block's scale (chunk // 4): the
                # power-of-two scale folds into the conversion exactly, so the expert
                # MFMAs read finished activations and need no per-block factor
                data_specs, scale_specs = [], []
                for sx in samples:
                    for i in range_constexpr(per_thread):
                        chunk = fx.min(tid + i * THREADS, chunks - 1)
                        data_specs.append((mb("xq"), sx * (HIDDEN // 4) + chunk * 2, 2))
                        scale_specs.append((mb("xqs"), sx * XQ_BLOCKS + chunk // 4, 1))
                got = poll(data_specs + scale_specs)
                nd = len(data_specs)
                for j in range_constexpr(len(samples)):
                    for i in range_constexpr(per_thread):
                        chunk = tid + i * THREADS
                        if chunk < chunks:
                            words = got[j * per_thread + i]
                            qs = got[nd + j * per_thread + i][0].bitcast(fx.Float32)
                            values = _fp8_to_bf16x8(words[0], words[1], qs)
                            for pair in range_constexpr(4):
                                lds_st(
                                    xs,
                                    j * (HIDDEN // 2) + chunk * 4 + pair,
                                    fx.Vector.from_elements(
                                        [values[2 * pair], values[2 * pair + 1]], fx.BFloat16
                                    ).bitcast(fx.Float32)[0],
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
            lanes < TOP_K.

            A hash-routed layer (``use_hash``, V4's first ``num_hash_layers``) takes the
            ids from ``tid2eid[token]`` instead of selecting them, and weights them the
            same way. Decided at run time, not build time, so hash and scored layers
            of one attention variant still share a kernel; the selection is a few
            wave-max rounds either way."""
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
            # A scored layer passes null for both tables: zero records makes these
            # loads return 0 in the texture unit instead of faulting on the address.
            nrec = (use_hash != 0).select(fx.Int32(0x7FFFFFF0), fx.Int32(0))
            r_tok = bo.create_buffer_resource_from_addr(tok_ids, num_records_bytes=nrec)
            r_t2e = bo.create_buffer_resource_from_addr(tid2eid, num_records_bytes=nrec)
            tok = fx.Int32(bo.buffer_load(r_tok, s, vec_width=1, dtype=T.i32))
            hk = tok * TOP_K + fx.min(lane, fx.Int32(TOP_K - 1))
            e_hash = fx.Int32(bo.buffer_load(r_t2e, hk, vec_width=1, dtype=T.i32))
            e = (use_hash != 0).select(e_hash, e)
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

        def hc_pre_stages(side, sd, fn_ptr, sb_ptr, src_word, out_name, contract=True):
            """hcd (the mixing projection's partials) and, with ``contract``, hcc (the
            streams contracted into one input). Returns the coefficient routine, for a
            consumer that contracts the streams itself."""
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
                # The partials are split over HC_PW waves so each polls one batch:
                # all HC_TASKS on wave 0 was three dependent round trips (POLL_MAX
                # is 12), and this routine sits on the FFN's critical path -- the
                # router runs it before it can contract its input. Each wave sums
                # its share, then wave 0 adds the shares in wave order, which every
                # rank does identically.
                NBLK = (S * HC_VALS + 63) // 64
                for blk in range_constexpr(NBLK):
                    idx = lane + blk * 64
                    if (wave < HC_PW) & (idx < S * HC_VALS):
                        s_ = idx // HC_VALS
                        j = idx % HC_VALS
                        tasks = [fx.min(wave * HC_TPW + i, HC_TASKS - 1) for i in range(HC_TPW)]
                        parts = poll([(mb("hc_d"), ((s_ * 2 + sd) * HC_TASKS + ti) * HC_VALS + j, 1) for ti in tasks])
                        tot = fx.Float32(0.0)
                        for i in range_constexpr(HC_TPW):
                            ok = (wave * HC_TPW + i) < HC_TASKS
                            tot = tot + ok.select(parts[i][0].bitcast(fx.Float32), fx.Float32(0.0))
                        lds_st(red, (wave * NBLK + blk) * 64 + lane, tot)
                gpu.barrier()
                for blk in range_constexpr(NBLK):
                    idx = lane + blk * 64
                    if (wave == 0) & (idx < S * HC_VALS):
                        tot = fx.Float32(0.0)
                        for w_ in range_constexpr(HC_PW):
                            tot = tot + lds_ld(red, (w_ * NBLK + blk) * 64 + lane)
                        lds_st(red, idx, tot)
                gpu.barrier()
                # one wave per sample (S <= WAVES): the Sinkhorn is a long dependent chain,
                # and running every sample's on wave 0 in turn cost S chains per hcc_a and
                # router task -- 8 of them back to back at S=8
                if wave < S:
                    s_ = wave
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

            if const_expr(not contract):
                return hc_coefficients

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
            return hc_coefficients

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
        QA_R = qkv_a_groups(N_QKV_A)  # row groups per task; WAVES // QA_R waves split K for each
        QA_WPR = WAVES // QA_R
        QA_UPW = QA_NKC // QA_WPR  # K chunks per wave
        QA_BATCH = QA_NKC // WAVES
        QA_ROWS = QKV_A_TILE * QA_R
        for t in range(start("qkv_a"), N_QKV_A // QA_R, G):
            t = fx.Int32(t)
            stamp("qkv_a", t, 0)
            qa_rg = t * QA_R + wave // QA_WPR

            def u_qa(c):
                kc = (wave % QA_WPR) * QA_UPW + c
                return unit_fp8(r_wqa, r_sqa, qa_rg, kc, QA_NKC, HIDDEN, 128, (n_sel() * HIDDEN + kc * 64) // 2)

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
            pre = [u_qa(c) for c in range(QA_BATCH)]
            stage_x_rmsnorm(ld_h, HIDDEN, g_in, loaded=h_ld)
            gpu.barrier()
            stamp("qkv_a", t, 2)
            acc = run_units(u_qa, QA_UPW, QA_BATCH, pre)
            reduce_rows(QA_R, acc, emit_out(QA_ROWS))
            stamp("qkv_a", t, 3)
            gpu.barrier()
            if tid < S * QA_ROWS:
                s = tid // QA_ROWS
                row = t * QA_ROWS + tid % QA_ROWS
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

        def kv_quant(nv):
            """This thread's KV channel through the NoPE FP8 round trip, one 64-wide
            group per wave with a power-of-two scale, 2**ceil(log2(amax / 448)) (the
            checkpoint's ue8m0; what ATOM's writers use).

            Returns (dequantized value, FP8 byte, the group's E8M0 byte)."""
            amax = wave_max(fmath.absf(nv))
            sc = _pow2_ceil(fx.max(amax, fx.Float32(FP8_MAX * 2.0**-126)) * (1.0 / FP8_MAX))
            q = fx.min(fx.max(nv * _rcp(sc), -FP8_MAX), FP8_MAX)  # _rcp is exact on a power of two
            word = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, q, q, fx.Int32(0), False))
            v2 = fx.Vector.make_type(2, fx.Float32)
            d = fx.Vector(rocdl.cvt_pk_f32_fp8(res=v2, src=word, word_sel=False))[0]
            return d * sc, word & 0xFF, (sc.bitcast(fx.Int32) >> 23) & 0xFF

        def put_kv_row(row, kvn, byte, e8):
            """Write one KV row (thread = channel; ``kvn`` its value, ``byte`` / ``e8``
            from kv_quant). CTA-uniform: the fp8 layout trades scale bytes through LDS."""
            if const_expr(KV_FP8):
                r_nope, r_rope = row_rsrc(kv_cache, row, KV_ROW_BYTES), row_rsrc(kv_rope, row, ROPE_DIM * 2)
                # four lanes' FP8 bytes to one dword, channel tid in byte tid % 4
                wb = byte << ((tid % 4) * 8)
                for off in (1, 2):
                    wb = _xred(wb, off, lambda a, b: a | b)
                if (tid < NOPE_DIM) & (tid % 4 == 0):
                    bo.buffer_store(wb, r_nope, tid // 4)
                if tid >= NOPE_DIM:
                    bo.buffer_store(kvn.to(fx.BFloat16), r_rope, tid - NOPE_DIM)
                if (lane == 0) & (tid < NOPE_DIM):
                    lds_st(misc, wave, e8.bitcast(fx.Float32))
                gpu.barrier()
                # scale dword k: groups 2k and 2k + 1, each byte twice (the last
                # dword's high half is padding)
                NG = NOPE_DIM // 64
                if tid < (NG + 1) // 2:
                    lo = lds_ld(misc, fx.min(2 * tid, fx.Int32(NG - 1))).bitcast(fx.Int32)
                    hi = (2 * tid + 1 < NG).select(
                        lds_ld(misc, fx.min(2 * tid + 1, fx.Int32(NG - 1))).bitcast(fx.Int32), fx.Int32(0)
                    )
                    sw = lo | (lo << 8) | (hi << 16) | (hi << 24)
                    bo.buffer_store(sw, r_nope, NOPE_DIM // 4 + tid)
                gpu.barrier()
            else:
                if tid < HEAD_DIM:
                    bo.buffer_store(kvn.to(fx.BFloat16), row_rsrc(kv_cache, row, HEAD_DIM * 2), tid)

        # ====== 2. KV RMSNorm + RoPE + FP8 round trip -> sliding-window ring cache
        # V4's K and V are the same HEAD_DIM row: RoPE occupies its last ROPE_DIM
        # lanes and the leading NOPE_DIM is FP8 round-tripped in 64-wide blocks
        # (one block per wave), matching the checkpoint's QAT.
        for t in range(start("cache"), 1, G):
            stamp("cache", t, 0)
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
                # nope head: one 64-wide FP8 group per wave
                dq, byte, e8 = kv_quant(nv)
                kvn = bf16_round((tid < NOPE_DIM).select(dq, rot))
                put_kv_row(ld_dest(0, s), kvn, byte, e8)
                if live:
                    put(mb("kvnew"), s * HEAD_DIM + tid, kvn)
            stamp("cache", t, 4)

        # ============= 2b. KV compressor (HCA): rolling state, pooled on the boundary
        # Every token feeds a window of CR positions; only the last of each window
        # emits a compressed entry, pooled by a PER-CHANNEL softmax over the window's
        # positions. The pooled row then gets the same norm / RoPE / FP8 round trip
        # the window KV does, rotated at the window's FIRST position, and lands in
        # the compressed half of the same cache.
        if const_expr(CR):
            for tt in range(start("cmp"), S, G):
                tt = fx.Int32(tt)
                stamp("cmp", tt, 0)
                ch = fx.min(tid, HEAD_DIM - 1)
                live = tid < HEAD_DIM
                # tt is the sample: its own sequence, at the shared position, with
                # its own rolling state. That per-sample state is also what keeps
                # these S tasks -- one per CTA, nothing ordering them -- from racing.
                p = ld_pos(tt)
                # this sequence's slice of the rolling state (64-bit: see slot_rsrc)
                rs_kv, rs_sc, sb = slot_rsrc(kv_state, tt, st_kv), slot_rsrc(score_state, tt, st_kv), 0
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
                        w = sb + (p % C_ROWS) * CW + j * HEAD_DIM + ch
                        bo.buffer_store(kvv[j], rs_kv, w)
                        bo.buffer_store(gtv[j] + ap0[j], rs_sc, w)
                if (p + 1) % CR == 0:  # uniform across the CTA
                    # Online softmax over the window, one channel per thread, so the
                    # CR positions are a loop rather than CR unrolled copies -- a
                    # CMP_CHUNK of them per trip, all their loads issued before any
                    # is consumed. Keeping the loop runtime is what stops the whole
                    # window being hoisted into registers; keeping a chunk of loads
                    # in flight is what stops each row costing a round trip.
                    for _i, acc in range(
                        0,
                        C_ROWS // CMP_CHUNK,
                        fx.Int32(1),
                        init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
                    ):
                        ib = fx.Int32(_i) * CMP_CHUNK
                        svs, kvs = [], []
                        for e in range_constexpr(CMP_CHUNK):
                            # an overlapped entry takes the previous window's rows
                            # from their FIRST half and the current window's from
                            # their SECOND
                            i = ib + e
                            coff = (i >= CR).select(fx.Int32(HEAD_DIM), fx.Int32(0)) if OVERLAP else 0
                            wi = sb + ((p + 1 + i) % C_ROWS) * CW + coff + ch
                            svs.append(ld_f32(rs_sc, wi))
                            kvs.append(ld_f32(rs_kv, wi))
                        m = fx.Float32(acc[0])
                        den = fx.Float32(acc[1])
                        num = fx.Float32(acc[2])
                        for e in range_constexpr(CMP_CHUNK):
                            m_new = fx.max(m, svs[e])
                            rescale = _exp(m - m_new)
                            w = _exp(svs[e] - m_new)
                            den = den * rescale + w
                            num = num * rescale + w * kvs[e]
                            m = m_new
                        res = yield [m, den, num]
                    pooled = fx.Float32(res[2]) * _rcp(fx.Float32(res[1]))
                    pooled = bf16_round(pooled)
                    ssq = block_sum(live.select(pooled * pooled, fx.Float32(0.0)))
                    # the model this reproduces returns bf16 from the norm, so the
                    # RoPE and FP8 round trip below see bf16
                    nv = bf16_round(pooled * _rsq(ssq * (1.0 / HEAD_DIM) + EPS) * g)
                    partner = _xshfl(nv, 1)
                    even = tid % 2 == 0
                    rot = even.select(nv * rc - partner * rs, partner * rs + nv * rc)
                    dq, byte, e8 = kv_quant(nv)
                    cv = bf16_round((tid < NOPE_DIM).select(dq, rot))
                    put_kv_row(comp_row(tt, p // CR), cv, byte, e8)  # this sequence's entry p // CR
                    if live:
                        put(mb("cnew"), tt * HEAD_DIM + tid, cv)
                stamp("cmp", tt, 4)

        def had_pair(v0, v1, ln):
            """FWHT over IHD channels held two per lane, scaled by IHD**-0.5.

            Lane ln owns channels ln and ln + 64, so every butterfly below stride 64
            is an xor shuffle inside the wave and the stride-64 one is the pair
            this lane already holds -- no LDS, no barrier.

            Without INDEXER_HADAMARD (ATOM's indexer: neither side rotated) it is the
            identity, on queries and keys alike.
            """
            if const_expr(not INDEXER_HADAMARD):
                return v0, v1
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
            """FP4 round trip over aligned 32-lane blocks, power-of-two scale.

            Returns (value, the lane's E2M1 code, the block's scale).
            """
            amax = fmath.absf(v)
            for off in (16, 8, 4, 2, 1):
                amax = _xred(amax, off, fx.max)
            sc = _pow2_ceil(fx.max(amax, fx.Float32(FP4_MAX * 2.0**-126)) * (1.0 / FP4_MAX))
            q = fx.min(fx.max(v * _rcp(sc), -FP4_MAX), FP4_MAX)
            d, _, word = _fp4_roundtrip(q, fx.Float32(0.0))
            return d * sc, word & 0xF, sc

        # ========== 2c. the indexer's compressor: same pooling, different tail
        # Half the head_dim, and it finishes with a Hadamard rotation over the whole
        # row followed by FP4 instead of FP8 over the nope part. The rotation is what
        # makes FP4 survivable: it spreads an outlier across every lane, so a block's
        # amax stops being set by one coordinate, and being orthonormal it leaves the
        # scores the indexer ranks unchanged.
        if const_expr(IHD):
            for tt in range(start("i_cmp"), S, G):
                tt = fx.Int32(tt)
                stamp("i_cmp", tt, 0)
                # one WAVE covers the row: lane ln holds channels ln and ln + 64
                ln = lane
                ilive = wave == 0
                p = ld_pos(tt)  # tt is the sample: its own sequence, its own position
                # this sequence's slice of the rolling state (64-bit: see slot_rsrc)
                rs_ikv, rs_isc, isb = slot_rsrc(i_kv_state, tt, st_i), slot_rsrc(i_score_state, tt, st_i), 0
                icb = ld_slot(tt, st_ic)  # ... and of the indexer's key cache (bytes)
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
                            w = isb + (p % C_ROWS) * IW + j * IHD + chs[e]
                            bo.buffer_store(kvv[e][j], rs_ikv, w)
                            bo.buffer_store(gtv[e][j] + ap0[e][j], rs_isc, w)
                if (p + 1) % CR == 0:  # uniform across the CTA
                    pooled = []
                    for e in range_constexpr(2):
                        # a chunk of rows per trip, loads first -- see the KV
                        # compressor's loop, which this one mirrors
                        for _i, acc in range(
                            0,
                            C_ROWS // CMP_CHUNK,
                            fx.Int32(1),
                            init=[fx.Float32(NEG), fx.Float32(0.0), fx.Float32(0.0)],
                        ):
                            ib = fx.Int32(_i) * CMP_CHUNK
                            svs, kvs = [], []
                            for z in range_constexpr(CMP_CHUNK):
                                i = ib + z
                                coff = (i >= CR).select(fx.Int32(IHD), fx.Int32(0)) if OVERLAP else 0
                                wi = isb + ((p + 1 + i) % C_ROWS) * IW + coff + chs[e]
                                svs.append(ld_f32(rs_isc, wi))
                                kvs.append(ld_f32(rs_ikv, wi))
                            m = fx.Float32(acc[0])
                            den = fx.Float32(acc[1])
                            num = fx.Float32(acc[2])
                            for z in range_constexpr(CMP_CHUNK):
                                m_new = fx.max(m, svs[z])
                                rescale = _exp(m - m_new)
                                w = _exp(svs[z] - m_new)
                                den = den * rescale + w
                                num = num * rescale + w * kvs[z]
                                m = m_new
                            res = yield [m, den, num]
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
                    (o0, k0, s0), (o1, k1, s1) = fp4_block(q0), fp4_block(q1)
                    # The cache keeps the codes, not their values: eight lanes' codes
                    # OR into one word (channel 8w + j in nibble j), and the four
                    # groups' exponents into one word, group g in byte g.
                    cw = [k0 << ((ln % 8) * 4), k1 << ((ln % 8) * 4)]
                    for off in (1, 2, 4):
                        cw = [_xred(w, off, lambda a, b: a | b) for w in cw]
                    e8 = [(sc.bitcast(fx.Int32) >> 23) & 0xFF for sc in (s0, s1)]
                    sw = (e8[0] << ((ln // 32) * 8)) | (e8[1] << ((ln // 32 + 2) * 8))
                    sw = _xred(sw, 32, lambda a, b: a | b)
                    if ilive:
                        # ATOM's FP4 pool (see K_PB): word w of the entry is group w // 4,
                        # dword w % 4 of the entry's 16 bytes in that group
                        e_i = p // CR
                        blk_i = bt_block(tt, e_i)
                        sl = e_i % K_PB
                        dbase = icb // 4 + blk_i * IC_BLK_WORDS + sl * 4
                        if ln % 8 == 0:
                            for h in range_constexpr(2):
                                w = ln // 8 + 8 * h
                                bo.buffer_store(cw[h], _rsrc(i_cache), dbase + (w // 4) * IC_GRP_WORDS + w % 4)
                        if ln == 0:
                            sbase = icb // 16 + blk_i * IC_S_BLK + (sl % 16) * 4 + (sl % K_PB) // 16
                            for g in range_constexpr(IHD // 32):
                                bo.buffer_store(fx.Int8((sw >> (8 * g)) & 0xFF), _rsrc(i_cache_s), sbase + g * K_PB)
                        put(mb("i_cnew"), tt * IHD + ln, bf16_round(o0))
                        put(mb("i_cnew"), tt * IHD + ln + 64, bf16_round(o1))
                stamp("i_cmp", tt, 4)

        # ==================================== 3. q_a RMSNorm -> q_b (raw f32 query)
        r_wqb, r_sqb = _rsrc(w_q_b), _rsrc(s_q_b)
        QB_NKC = Q_LORA // 64
        QB_R = q_b_groups(N_QB)  # row groups per task; WAVES // QB_R waves split K for each
        QB_WPR = WAVES // QB_R
        QB_UPW = QB_NKC // QB_WPR  # K chunks per wave
        QB_BATCH = QB_NKC // WAVES
        QB_ROWS = Q_B_TILE * QB_R
        for t in range(start("q_b"), N_QB // QB_R, G):
            t = fx.Int32(t)
            stamp("q_b", t, 0)
            qb_rg = t * QB_R + wave // QB_WPR

            def u_qb(c):
                kc = (wave % QB_WPR) * QB_UPW + c
                return unit_fp8(r_wqb, r_sqb, qb_rg, kc, QB_NKC, Q_LORA, 128, (n_sel() * Q_LORA + kc * 64) // 2)

            pre = [u_qb(c) for c in range(QB_BATCH)]
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
            acc = run_units(u_qb, QB_UPW, QB_BATCH, pre)
            reduce_rows(QB_R, acc, emit_out(QB_ROWS))
            stamp("q_b", t, 3)
            gpu.barrier()
            # published as f32: the per-head RMS below is taken on the unrounded GEMV
            # output, so rounding to bf16 happens only once, after RoPE
            if tid < S * QB_ROWS:
                s = tid // QB_ROWS
                row = t * QB_ROWS + tid % QB_ROWS  # a head's rows are whole tiles, so whole groups
                put(mb("q_raw"), (s * H + row // HEAD_DIM) * HEAD_DIM + row % HEAD_DIM, lds_ld(outs, tid))
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
                    o0, o1 = fp4_block(bf16_round(h0))[0], fp4_block(bf16_round(h1))[0]
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
            #
            # The stage is sized for max_seq -- at ATOM's default 1M, 512 tiles a
            # sample -- but only the tiles holding live entries are walked: the
            # tasks are (sample, tile) over the batch's LARGEST live-tile count, so
            # an 8K context at 1M is 4 tiles a sample, not 512 rounds of skipped
            # tasks on every CTA (~50 us that held up everything behind them).
            # Every CTA computes the same count from the same positions.
            ISC_L = fx.Int32(0)
            for s_ in range_constexpr(S):
                nl = fx.min((ld_pos(s_) + 1) // CR, fx.Int32(N_COMP))
                ISC_L = fx.max(ISC_L, (nl > N_INDEX).select((nl + SCORE_TILE - 1) // SCORE_TILE, fx.Int32(0)))
            for tt in range(start("i_score"), S * ISC_L, G):
                tt = fx.Int32(tt)
                stamp("i_score", tt, 0)
                s = tt // ISC_L
                blk = tt % ISC_L
                # a sequence with no more than N_INDEX live entries keeps all of them:
                # its scores are never read (see i_topk), so they are not computed;
                # nor is a tile past its own live entries (another sample's may be longer)
                n_live_s = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
                if (n_live_s > N_INDEX) & (blk * SCORE_TILE < n_live_s):
                    # the query as f32 (words [0, IH * IHD)) and as bf16 pairs after it
                    # (QBF): the bf16 copy is the scoring MFMA's B operand -- exact, the
                    # values are FP4 codes times a power-of-two scale
                    QBF = IH * IHD
                    for i in range_constexpr((IH * IHD // 2 + THREADS - 1) // THREADS):
                        w2 = tid + i * THREADS
                        if w2 < IH * IHD // 2:
                            q0 = getf(mb("i_q"), s * IH * IHD + 2 * w2)
                            q1 = getf(mb("i_q"), s * IH * IHD + 2 * w2 + 1)
                            lds_st(xs, 2 * w2, q0)
                            lds_st(xs, 2 * w2 + 1, q1)
                            lds_st(xs, QBF + w2, bf16_pair(q0, q1))
                    wv = [getf(mb("i_wp"), s * IH + hh) for hh in range(IH)]
                    gpu.barrier()
                    stamp("i_score", tt, 2)
                    c = blk * SCORE_TILE + tid
                    # entries the compressor has not written yet must never be picked
                    sp = ld_pos(s)
                    n_live = (sp + 1) // CR
                    r_ic2 = _rsrc(i_cache)
                    r_ics = _rsrc(i_cache_s)
                    icb = ld_slot(s, st_ic)  # bytes; the scale pool's base is 1/16 of it
                    ne = sp // CR  # the entry this launch wrote, if (sp + 1) % CR == 0
                    has_new = ((sp + 1) % CR == 0) & (blk == ne // SCORE_TILE)
                    # Scores on MFMA: a wave takes 64 entries, 4 groups of 16 as the A rows
                    # (K = IHD, one 32-wide block per MFMA), the IH heads as the B columns.
                    # A lane's 8 key codes of a block are one dword of ATOM's FP4 pool (the
                    # entry's 16 bytes of that block, dword lane // 16), converted to bf16
                    # with the block's E8M0 scale -- exact, as is the bf16 query. The scalar
                    # version (one entry per thread, 8 heads x 128 FMAs, a query broadcast
                    # per 4) was ~14 us of the CSA layer at S = 8.
                    hn = lane % 16
                    wcol = wv[0]
                    for hh in range_constexpr(1, IH):
                        wcol = (hn == hh).select(wv[hh], wcol)
                    wcol = (hn < IH).select(wcol, fx.Float32(0.0))
                    qb = QBF + (fx.min(hn, fx.Int32(IH - 1)) * IHD + (lane // 16) * 8) // 2
                    for g in range_constexpr(4):
                        ec = fx.min(blk * SCORE_TILE + wave * 64 + g * 16 + hn, fx.Int32(N_COMP - 1))
                        blk_e = bt_block(s, ec)
                        sl_e = ec % K_PB
                        db = icb // 4 + blk_e * IC_BLK_WORDS + sl_e * 4 + lane // 16
                        sdb = icb // 64 + blk_e * (IC_S_BLK // 4) + sl_e % 16
                        kds = [bo.buffer_load(r_ic2, db + kb * IC_GRP_WORDS, vec_width=1, dtype=T.i32) for kb in range(IHD // 32)]
                        sds = [bo.buffer_load(r_ics, sdb + kb * (K_PB // 4), vec_width=1, dtype=T.i32) for kb in range(IHD // 32)]
                        acc = fx.Vector.filled(4, 0.0, fx.Float32)
                        for kb in range_constexpr(IHD // 32):
                            bsc = (((fx.Int32(sds[kb]) >> ((sl_e // 16) * 8)) & 0xFF) << 23).bitcast(fx.Float32)
                            a = _mxfp4_to_bf16x8(fx.Int32(kds[kb]), bsc)
                            b = fx.ptr_load(xs + (qb + kb * 16), result_type=v4f).bitcast(fx.BFloat16)
                            acc = fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, acc]))
                        # C[entry 4 * (lane // 16) + i][head lane % 16]: relu, weight, sum heads
                        for i in range_constexpr(4):
                            v = fx.max(acc[i], fx.Float32(0.0)) * wcol
                            for off in (1, 2, 4, 8):
                                v = _xred(v, off, lambda x, y: x + y)
                            if hn == i:
                                lds_st(red, wave * 64 + g * 16 + 4 * (lane // 16) + i, v)
                    gpu.barrier()
                    # the entry this launch just wrote is not reliably visible in the cache
                    # yet: its owning wave rescores it from the mailbox copy, lane j taking
                    # dims j and j + 64
                    if has_new & (wave == (ne - blk * SCORE_TILE) // 64):
                        nv = getf_many([(mb("i_cnew"), s * IHD + lane + 64 * hf) for hf in range(IHD // 64)])
                        sc_n = fx.Float32(0.0)
                        for hh in range_constexpr(IH):
                            part = fx.Float32(0.0)
                            for hf in range_constexpr(IHD // 64):
                                part = part + nv[hf] * lds_ld(xs, hh * IHD + lane + 64 * hf)
                            sc_n = sc_n + fx.max(wave_sum(part), fx.Float32(0.0)) * wv[hh]
                        if lane == 0:
                            lds_st(red, ne - blk * SCORE_TILE, sc_n)
                    gpu.barrier()
                    sc_t = lds_ld(red, tid)
                    live = (c < n_live) & (c < N_COMP)
                    # Splits `compute` (the scoring loop) from `epi` (the exchange), and
                    # it is stamped whether or not there IS an exchange: inside the
                    # W > 1 branch a single-rank run left mark 3 unwritten, the report
                    # inherited mark 2 for it, and the whole scoring loop was reported
                    # as epilogue -- which read as 28 us of write overhead that was
                    # really the arithmetic, and sent one round of tuning the wrong way.
                    stamp("i_score", tt, 3)
                    if const_expr(W > 1):
                        # This rank holds only IH of the 64 index heads, so its score is
                        # a PARTIAL sum -- without this exchange the ranks would rank the
                        # entries differently and silently attend to different keys.
                        # Push to every peer, then sum all ranks' partials in rank order
                        # from our own buffer: same order everywhere, so the totals are
                        # bit-identical and the top-k below cannot disagree.
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
        # Radix select over 8-bit digits, most significant first. Each pass
        # histograms the digit of every candidate whose higher bits already match
        # the winning prefix, then walks the 256 bins downwards to the one where the
        # running count crosses what the pick still needs. Four passes cover a
        # 32-bit key where a bit-at-a-time select needs 32.
        #
        # How the scores are read carries the cost, not the select. A thread takes
        # TK_PER consecutive candidates and the threads stride over those groups, so
        # a wave's trip is one contiguous run: mailbox loads bypass the cache by
        # design, since they have to observe a remote CTA's write, so a wave whose
        # lanes sit 4 KB apart pays a whole line per lane. And a thread's share is
        # polled once and held in registers (part_keys) -- which is only possible
        # because the parts split the candidates: on ONE CTA a thread held 512 of
        # them, and holding those spilled 5387 slots at 1M.
        if const_expr(IHD):
            for tt in range(start("i_topk"), S * TK_PARTS, G):
                tt = fx.Int32(tt)
                stamp("i_topk", tt, 0)
                s = tt // TK_PARTS
                part = tt % TK_PARTS  # which share of the candidates this CTA scans
                n_live = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_COMP))
                k_want = fx.min(n_live, fx.Int32(N_INDEX))
                sbase = s * N_COMP
                # Up to N_INDEX live entries the pick is every one of them -- the set
                # does not depend on the scores, and the gather is order-blind -- so a
                # short sequence skips the select (and i_score skips its scores).
                # One part holds TK_TRIPS * THREADS * TK_PER candidates, so while the
                # live ones fit it, part 0 selects alone and the others sit out: no
                # bins traded. At 1M max-len that is any context up to 64K tokens,
                # where 16 parts were trading bins four times over 2K candidates.
                n_parts = (n_live <= TK_TRIPS * THREADS * TK_PER).select(fx.Int32(1), fx.Int32(TK_PARTS))
                if (n_live > N_INDEX) & (part < n_parts):
                    tk_cbs, tk_keys = part_keys(sbase, part, n_live, n_parts)

                    def trip_live(j):
                        """Whether trip ``j`` of this part holds any live candidate --
                        CTA-uniform, so a dead trip's work is branched over, not masked.
                        A part is sized for the longest context it may take (8 trips at
                        1M), and the radix walks every trip four times plus once more to
                        compact: at an 8K context 7 of the 8 are dead, ~15 us at S=8."""
                        return (fx.Int32(j) * n_parts + part) * (THREADS * TK_PER) < n_live

                    stamp("i_topk", tt, 2)

                    pfx = fx.Int32(0)  # the digits already fixed, in the unsigned domain
                    gt = fx.Int32(0)  # candidates ranking strictly above that prefix
                    # ... of them, how many the parts before this one hold, and after the
                    # last digit how many of the ties they hold: where this part's picks go
                    gt_b = fx.Int32(0)
                    eq_b = fx.Int32(0)
                    for d in range_constexpr(4):
                        sh = 24 - 8 * d
                        # the bits above this digit: zero on the first pass, where every
                        # candidate is still in the running. Folded at trace time, so no
                        # 32-wide shift ever reaches the ISA.
                        mk = (~((1 << (sh + 8)) - 1)) & 0xFFFFFFFF
                        hi = fx.Int32(mk - (1 << 32) if mk >= (1 << 31) else mk)
                        for z in range_constexpr(-(-(TK_BC + 4) // THREADS)):
                            zi = fx.Int32(tid) + z * THREADS
                            if zi < TK_BC + 4:
                                lds_st(hist, zi, fx.Int32(0))
                        gpu.barrier()
                        for j in range_constexpr(TK_TRIPS):
                            if trip_live(j):
                                for q in range_constexpr(TK_PER):
                                    c = tk_cbs[j] + q
                                    ok = c < n_live
                                    # a dead candidate keys as 0, the very bottom
                                    u = ok.select(tk_keys[j * TK_PER + q], fx.Int32(0))
                                    if ok & (((u ^ pfx) & hi) == 0):
                                        fx.atomic_add(
                                            hist + ((u >> sh) & (TK_BINS - 1)) * TK_REP + (tid & (TK_REP - 1)),
                                            fx.Int32(1),
                                            syncscope=fx.rocdl.SyncScope.Workgroup,
                                        )
                        gpu.barrier()
                        # Thread t takes bin TK_BINS - 1 - t, so an ascending exclusive
                        # scan over threads is a descending suffix sum over bins: every
                        # thread learns how many candidates outrank its own bin. Exactly
                        # one bin has that count below what is still needed and its own
                        # count enough to reach it -- unless nothing is wanted at all,
                        # which the pre-zeroed broadcast slots cover.
                        need = k_want - gt
                        cnt = fx.Int32(0)
                        bn = TK_BINS - 1 - tid
                        if tid < TK_BINS:
                            for r in range_constexpr(TK_REP):
                                cnt = cnt + lds_ld(hist, bn * TK_REP + r)
                        if const_expr(TK_PARTS > 1):
                            # These bins only count THIS part's share, so the parts
                            # trade them and every one picks the same digit from the
                            # same totals. Each reads the OTHERS and adds its own from
                            # the register it already holds -- walking from part + 1
                            # means no part ever polls a slot it wrote itself, which
                            # would be a bet on seeing your own global store.
                            if (tid < TK_BINS) & (n_parts > 1):
                                put(mb("tk_hist"), ((s * 4 + d) * TK_PARTS + part) * TK_BINS + bn, cnt)
                            tot = cnt
                            before = fx.Int32(0)  # this bin's count in the parts before this one
                            if (tid < TK_BINS) & (n_parts > 1):
                                # ONE batch: polled one at a time these are TK_PARTS - 1
                                # dependent round trips, since each tag has to be
                                # compared before the next load can issue
                                vs = poll(
                                    [
                                        (mb("tk_hist"), ((s * 4 + d) * TK_PARTS + pp) * TK_BINS + bn, 1)
                                        for pp in _other_parts(part)
                                    ]
                                )
                                others = _other_parts(part)
                                for k in range_constexpr(TK_PARTS - 1):
                                    tot = tot + vs[k][0]
                                    before = before + (others[k] < part).select(vs[k][0], fx.Int32(0))
                            cnt = tot
                        above, _tot = block_excl_scan(cnt)
                        if const_expr(TK_PARTS > 1):
                            # the same scan over the earlier parts' counts: at the chosen
                            # bin it is how many of the candidates above it they hold
                            above_b, _tb = block_excl_scan(before)
                        if (tid < TK_BINS) & (above < need) & ((above + cnt) >= need):
                            lds_st(hist, TK_BC, bn)
                            lds_st(hist, TK_BC + 1, above)
                            if const_expr(TK_PARTS > 1):
                                lds_st(hist, TK_BC + 2, above_b)
                                lds_st(hist, TK_BC + 3, before)
                        gpu.barrier()
                        pfx = pfx | (lds_ld(hist, TK_BC) << sh)
                        gt = gt + lds_ld(hist, TK_BC + 1)
                        if const_expr(TK_PARTS > 1):
                            gt_b = gt_b + lds_ld(hist, TK_BC + 2)
                            eq_b = lds_ld(hist, TK_BC + 3)  # the last digit's is the ties'
                        gpu.barrier()
                    thr = pfx ^ MIN_I32  # back to the signed-comparable domain

                    # One compaction pass writes both: everything strictly above the
                    # threshold, then as many of the ties as the count still needs. The
                    # radix already fixed where every part's picks go -- all of them sit
                    # at [0, gt) and the ties at [gt, k_want), and gt_b / eq_b are this
                    # part's offsets into each, from the bins the parts traded -- so no
                    # part counts its hits first or trades a count. That was three more
                    # scans of the candidates and two more exchanges, ~10 us at 1M.
                    #
                    # The cost is where a radix bug surfaces. The gather polls every
                    # slot of i_sel, so a slot nobody writes is not a wrong answer but a
                    # kernel that never finishes, and the slots are now fixed by the
                    # histograms rather than by counting the writes. The histograms and
                    # this pass read the same keys from the same thread, so they cannot
                    # disagree short of a bug in the select itself.
                    stamp("i_topk", tt, 3)
                    if tid == 0:
                        lds_st(hist, TK_BC, fx.Int32(0))
                        lds_st(hist, TK_BC + 1, fx.Int32(0))
                    gpu.barrier()
                    for j in range_constexpr(TK_TRIPS):
                        if trip_live(j):
                            for q in range_constexpr(TK_PER):
                                c = tk_cbs[j] + q
                                ok = c < n_live
                                sk = tk_keys[j * TK_PER + q] ^ MIN_I32  # back to the signed-comparable domain
                                if ok & (sk > thr):
                                    w = fx.Int32(
                                        fx.atomic_add(hist + TK_BC, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup)
                                    )
                                    put(mb("i_sel"), s * N_ISEL + gt_b + w, comp_row(s, c))
                                if ok & (sk == thr):
                                    w = fx.Int32(
                                        fx.atomic_add(hist + TK_BC + 1, fx.Int32(1), syncscope=fx.rocdl.SyncScope.Workgroup)
                                    )
                                    if (gt + eq_b + w) < k_want:
                                        put(mb("i_sel"), s * N_ISEL + gt + eq_b + w, comp_row(s, c))
                    # the next task's first digit re-zeroes these counters
                    gpu.barrier()
                    # One part fills the tail, and it is the part that cannot collide:
                    # every part only ever writes below k_want.
                    if part == 0:
                        for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                            o = fx.Int32(tid) + j * THREADS
                            if o < N_ISEL:
                                if o >= k_want:
                                    put(mb("i_sel"), s * N_ISEL + o, fx.Int32(-1))
                else:
                    if part == 0:
                        for j in range_constexpr((N_ISEL + THREADS - 1) // THREADS):
                            o = fx.Int32(tid) + j * THREADS
                            if o < N_ISEL:
                                row = comp_row(s, fx.max(fx.min(o, n_live - 1), fx.Int32(0)))
                                put(mb("i_sel"), s * N_ISEL + o, (o < n_live).select(row, fx.Int32(-1)))
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
            if const_expr(KV_FP8):
                # ATOM's fp8 attention takes the NoPE query as FP8 too: the KV's 64-wide
                # power-of-two groups (one per wave), from the fp32 normed value
                nv = kv_quant(nv)[0]
            qn = (tid < NOPE_DIM).select(nv, rot)
            # repack as bf16 pairs for the split stage's MFMA operand
            other = _xshfl(qn, 1)
            if live & (tid % 2 == 0):
                put(mb("q"), ((s * H + head) * HEAD_DIM + tid) // 2, bf16_pair(qn, other))
            stamp("q_norm", tt, 4)

        # ============== 5. gather-sparse sliding-window split: 64 keys x H heads
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
                if const_expr(KV_FP8):
                    # lanes < NOPE_DIM / EPL take 8 FP8 bytes of the NoPE plane and
                    # their group's E8M0 byte; the rest read the bf16 RoPE plane.
                    # FP8 times a power of two is exact in bf16, so the tile holds
                    # the very values the model would.
                    r_row = row_rsrc(kv_cache, krows[jj], KV_ROW_BYTES)
                    q8 = fx.Vector(
                        bo.buffer_load(r_row, fx.min(lane, fx.Int32(NOPE_DIM // EPL - 1)) * 2, vec_width=2, dtype=T.i32)
                    )
                    g = fx.min(lane, fx.Int32(NOPE_DIM // EPL - 1)) // (64 // EPL)  # this lane's 64-group
                    sw = fx.Int32(bo.buffer_load(r_row, NOPE_DIM // 4 + g // 2, vec_width=1, dtype=T.i32))
                    sc = (((sw >> ((g % 2) * 16)) & 0xFF) << 23).bitcast(fx.Float32)
                    nope = (_fp8_to_bf16x8(q8[0], q8[1]).to(fx.Float32) * sc).to(fx.BFloat16)
                    rope = fx.Vector(
                        bo.buffer_load(
                            row_rsrc(kv_rope, krows[jj], ROPE_DIM * 2),
                            fx.max(lane - NOPE_DIM // EPL, fx.Int32(0)) * WPL,
                            vec_width=WPL,
                            dtype=T.i32,
                        )
                    )
                    nw = nope.bitcast(fx.Int32)
                    is_n = lane < NOPE_DIM // EPL
                    kv8 = fx.Vector.from_elements([is_n.select(nw[m], rope[m]) for m in range(WPL)], fx.Int32)
                else:
                    kv8 = fx.Vector(
                        bo.buffer_load(
                            row_rsrc(kv_cache, krows[jj], HEAD_DIM * 2), lane * WPL, vec_width=WPL, dtype=T.i32
                        )
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
            c_row = comp_row(s, sp // CR) if const_expr(CR) else fx.Int32(0)
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

        def live_splits(s):
            """Splits of sample ``s`` that can hold a live key: the window's, then
            the compressed keys that exist so far -- ``(pos + 1) // CR`` of them
            (ATOM's ``n_hca`` / ``n_csa`` too), capped by the list. Every split past
            them is all -1. N_SPLIT is sized for max_seq: at 1M an HCA list is 130
            splits, of which an 8K context fills 3.

            HCA only (LIVE_SPLITS): a CSA list is the window plus index_topk picks,
            18 splits that any context past 4K fills, so it would pay the
            bookkeeping (~3 us at S=8, measured) for nothing; it keeps N_SPLIT."""
            n = N_SPLIT
            if const_expr(LIVE_SPLITS):
                nc = fx.min((ld_pos(s) + 1) // CR, fx.Int32(N_KEYS - window))
                n = (window + nc + SPLIT_KEYS - 1) // SPLIT_KEYS
            return n

        # (sample, split) over the batch's LARGEST live-split count, so a long
        # max-len does not cost its empty splits; a shorter sample's extra tasks
        # just score all -1 keys, and the merge reads only its own live ones.
        SPL_L = N_SPLIT
        if const_expr(LIVE_SPLITS):
            SPL_L = fx.Int32(0)
            for s_ in range_constexpr(S):
                SPL_L = fx.max(SPL_L, live_splits(s_))
        for tt in range(start("split"), S * SPL_L, G):
            tt = fx.Int32(tt)
            stamp("split", tt, 0)
            s = tt // SPL_L  # sample
            t = tt % SPL_L  # 64-key chunk
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
            n_sp = live_splits(s)  # only these were written: see the split stage
            hint_wait(n_sp, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head), mark=("uv", tt))
            pre_poll(n_sp, lambda k: (mb("sp_l"), (s * N_SPLIT + k) * H + head))
            stamp("uv", tt, 5)
            dp = fx.min(tid, UV_PAIRS - 1)  # dim pair within this tile
            # which thread owns a split: a lane of wave 0 while they fit in one
            # wave, otherwise one thread of the whole block
            sp_id = tid if UV_WIDE else lane
            spi = fx.min(sp_id, n_sp - 1)  # clamped so the spare threads read a live slot
            ml = (s * N_SPLIT + spi) * H + head
            got = poll([(mb("sp_m"), ml, 1), (mb("sp_l"), ml, 1)], batch=2)
            # per-split weights exp(m - M) / L for this head -> misc[split]
            ok_sp = sp_id < n_sp
            m_sp = ok_sp.select(got[0][0].bitcast(fx.Float32), fx.Float32(NEG))
            l_sp = ok_sp.select(got[1][0].bitcast(fx.Float32), fx.Float32(0.0))
            if UV_WIDE:  # trace-time: a plain Python bool, not a traced value
                mx = block_max(m_sp)
                w_sp = _exp(m_sp - mx)
                den = block_sum(l_sp * w_sp) + _exp(sink - mx)
                if ok_sp:
                    lds_st(misc, sp_id, w_sp * _rcp(den))
            else:
                if wave == 0:
                    mx = wave_max(m_sp)
                    w_sp = _exp(m_sp - mx)
                    den = wave_sum(l_sp * w_sp) + _exp(sink - mx)
                    if ok_sp:
                        lds_st(misc, sp_id, w_sp * _rcp(den))
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
            # Over the live splits only; a chunk's entries past them re-read the last
            # live split (never poll one nobody wrote this launch) at weight 0.
            # (With N_SPLIT -- CSA -- this folds to the full, unmasked loop.)
            for _c, acc in range(
                0, (n_sp + UV_CHUNK - 1) // UV_CHUNK, fx.Int32(1), init=[fx.Float32(0.0), fx.Float32(0.0)]
            ):
                cb = fx.Int32(_c) * UV_CHUNK
                gc = poll(
                    [
                        (
                            mb("sp_acc"),
                            ((s * N_SPLIT + (fx.min(cb + e, n_sp - 1) if const_expr(LIVE_SPLITS) else cb + e)) * H + head) * (HEAD_DIM // 2) + doff // 2 + dp,
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
                    if const_expr(LIVE_SPLITS):
                        wj = (cb + e < n_sp).select(wj, fx.Float32(0.0))
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
        # The FFN side has no hcc stage: its only consumer, the router, contracts
        # the streams itself (below), which takes a whole grid-wide handoff off the
        # FFN's chain -- hcc publishing x and the router then polling it.
        hc_coef_f = None
        if const_expr(HC > 1):
            hc_coef_f = hc_pre_stages(
                "f",
                1,
                hc_ffn_fn,
                hc_ffn_sb,
                lambda s, k: get(mb("a"), (s * HC * HIDDEN + k) // 2),
                None,
                contract=False,
            )

        # ====== 8. post-attn RMSNorm -> router scores + this task's FP8 activation blocks
        # One sample per CTA: 1 row group x 96 chunks (bf16), 8 waves split K
        r_wr = _rsrc(w_r)
        R_NKC = HIDDEN // 64
        SPT = router_spt(S)
        assert SPT <= ROUTER_TILE
        for tt in range(start("router"), S * N_ROUTER // SPT, G):
            tt = fx.Int32(tt)
            t = tt % N_ROUTER
            rs0 = (tt // N_ROUTER) * SPT  # this task's first sample; it takes SPT of them
            stamp("router", tt, 0)

            # K-fold: MFMA rows / B columns 0..7 take this wave's first K half, rows /
            # columns 8..15 the second, so every loaded weight row is distinct and the
            # whole K slice is prefetched; logit = C[r][n] + C[8 + r][8 + n], where
            # column n of each half is local sample min(n, SPT - 1)
            r_sub = t * ROUTER_TILE % 16  # this task's rows of the 16-row group
            r_ln = (lane & -16) | (r_sub + lane % ROUTER_TILE)
            R_CPW = R_NKC // WAVES // 2
            r_fold = (lane % 16) // ROUTER_TILE
            r_ns = fx.min(lane % 8, fx.Int32(SPT - 1))

            def u_r(c):
                kc = wave * (R_NKC // WAVES) + r_fold * R_CPW + c
                return unit_bf16(r_wr, t * ROUTER_TILE // 16, kc, R_NKC, (r_ns * HIDDEN + kc * 64) // 2, r_ln)

            pre = [u_r(c) for c in range(R_CPW)]
            hint_wait(0, None, mark=("router", tt))
            # This task's expert-activation block inputs ride along with the staging
            # loads. MXFP8 uses four independent 16-lane groups per wave.
            r_gp = _rsrc(g_post)
            if const_expr(use_mxfp8_block32):
                x_blk = (wave * N_ROUTER + t) * 4 + lane // 16
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 32 + lane % 16 * 2
            else:
                x_blk = wave * N_ROUTER + t
                xk = fx.min(x_blk, PUBLISH_BLOCKS - 1) * 128 + lane * 2
            x_ok = (wave < XQ_WAVES) & (x_blk < PUBLISH_BLOCKS)
            xg = (ld_bf16(r_gp, xk), ld_bf16(r_gp, xk + 1))
            xa = []

            def ld_a(sks):
                # sks = (local sample s, k) pairs; local sample s is sample rs0 + s
                n4 = len(sks)
                if const_expr(HC == 1):
                    specs = [(mb("a"), ((rs0 + s) * HIDDEN + k) // 2, 2) for s, k in sks]
                    specs += [(mb("a"), ((rs0 + s) * HIDDEN + xk) // 2, 1) for s in range(SPT)]
                    v = poll(specs, batch=len(specs))
                    stamp("router", tt, 5, lead=THREADS - 64)
                    for s in range_constexpr(SPT):
                        xa.append(bf2_f32(v[n4 + s][0]))
                    return [list(bf2_f32(w[0])) + list(bf2_f32(w[1])) for w in v[:n4]]
                # x = sum_j pre[j] * stream j, contracted here from the streams and
                # the hcd partials rather than polled from an hcc stage. Same order
                # and the same bf16 rounding hcc used, so x is bit-identical.
                specs = [
                    (mb("a"), ((rs0 + s) * HC * HIDDEN + j * HIDDEN + k) // 2, 2) for s, k in sks for j in range(HC)
                ]
                specs += [
                    (mb("a"), ((rs0 + s) * HC * HIDDEN + j * HIDDEN + xk) // 2, 1) for s in range(SPT) for j in range(HC)
                ]
                v = poll(specs)  # the streams land before the partials do
                stamp("router", tt, 5, lead=THREADS - 64)
                hc_coef_f(1, tt == 0)  # task 0 also publishes post / comb for down
                pjs = [[lds_ld(misc, HC_MISC + (rs0 + s) * HC_COEF + j) for j in range(HC)] for s in range(SPT)]

                def mix(words, pj):
                    """bf16(sum_j pre[j] * x_j) for each element of the words' streams."""
                    xs_ = [list(bf2_f32(w[0])) + (list(bf2_f32(w[1])) if len(w) > 1 else []) for w in words]
                    out = []
                    for e in range_constexpr(len(xs_[0])):
                        acc = fx.Float32(0.0)
                        for j in range_constexpr(HC):
                            acc = acc + pj[j] * xs_[j][e]
                        out.append(bf16_round(acc))
                    return out

                for s in range_constexpr(SPT):
                    xa.append(tuple(mix(v[(n4 + s) * HC : (n4 + s + 1) * HC], pjs[s])))
                return [mix(v[i * HC : (i + 1) * HC], pjs[sks[i][0]]) for i in range(n4)]

            rstds = stage_x_rmsnorm(ld_a, HIDDEN, g_post, mark=("router", tt), count=SPT)
            stamp("router", tt, 2)
            # This task's normalized expert inputs go out ahead of the gate GEMV.
            for s_l in range_constexpr(SPT):
                if x_ok:
                    x_s = rs0 + s_l
                    x_rstd = rstds[s_l]
                    a0, a1 = xa[s_l]
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
            if tid < ROUTER_TILE * SPT:
                r = tid % ROUTER_TILE
                n = tid // ROUTER_TILE  # local sample = B column n of each K-fold half
                logit = fx.Float32(0.0)
                for w in range_constexpr(WAVES):
                    for f in range_constexpr(2):
                        m = f * ROUTER_TILE + r
                        logit = logit + lds_ld(red, (w * 64 + f * ROUTER_TILE + n + 16 * (m // 4)) * 4 + m % 4)
                put(mb("scores"), (rs0 + n) * N_EXPERTS + t * ROUTER_TILE + r, _sqrt_softplus(bf16_round(logit)))  # the gate's logits are bf16, as ATOM's
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
        SUG_S_BYTES = 2 * INTER // SCALE_BM * (HIDDEN // 128) * 4  # the FP8 shared expert's scales

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
                        coefficients = None  # MXFP8 scales are folded into the staged activations
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
                if const_expr(SHARED_FP8):

                    def u_ug8_sh(cc, live):
                        unit = wave * UG8_UNITS + cc
                        coefficients = None  # MXFP8 scales are folded into the staged activations
                        return unit_fp8mx(
                            bo.create_buffer_resource_from_addr(
                                w_sug, num_records_bytes=live.select(fx.Int32(2 * INTER * HIDDEN), fx.Int32(0))
                            ),
                            bo.create_buffer_resource_from_addr(
                                s_sug, num_records_bytes=live.select(fx.Int32(SUG_S_BYTES), fx.Int32(0))
                            ),
                            w_rg,
                            s_rg,
                            unit,
                            HIDDEN,
                            unit * 64,
                            coefficients,
                            w_ln,
                        )

                    pre = [u_ug8_sh(cc, has_sh) for cc in range(UG8_UNITS)]
                else:
                    pre = [u_ug8(cc, fx.Int32(SHARED_EXPERT), has_sh) for cc in range(UG8_UNITS)]
                # The normed, quantized input and the routing are the SAME for every
                # task of the sample, and S == 1 has one sample, so only a CTA's first
                # task stages them; the ones after it find them in LDS (xs, the
                # scales in misc[8:], the picks in keys / misc[:TOP_K]), which
                # nothing between two of its tasks writes. Staging them per task was
                # 4.6 us of every task, and the 32 CTAs that take a second tile are
                # the tail `down` waits on. The input is the router's quantized copy,
                # published ahead of its scores, which routing waits for anyway --
                # so up/gate never needs x itself, and no stage has to publish x.
                if u == fx.Int32(start("ug")):
                    hint_wait(0, None, mark=("ug", u))
                    stage_moe_input([0])
                    if wave == 0:
                        e, w = route_topk(s_u, bs=bs)
                        # every pick, not just this task's slot: a later task on this
                        # CTA reads its own from here
                        if lane < TOP_K:
                            lds_st(keys, lane, e)
                            lds_st(misc, lane, w)
                stamp("ug", u, 2)
                gpu.barrier()
                e_sel = _uniform(lds_ld(keys, slot - 1))
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
                    put(mb("prob"), slot, lds_ld(misc, slot - 1))
                    if has_sh:
                        put(mb("sel"), 0, fx.Int32(SHARED_EXPERT))
                        put(mb("prob"), 0, fx.Float32(1.0))
                stamp("ug", u, 4)
        elif const_expr(S > 1):
            # One eight-intermediate tile per CTA and sample. Shared weights use
            # the sample columns of one MFMA; routed tiles pipeline over samples.
            #
            # UG_ITEMS (tile, sample) items per CTA, unrolled at BUILD time rather than
            # walked with a runtime `range(...)` the way S == 1 does: the unit lists
            # below are Python lists of traced operands that mma_units consumes while
            # tracing, so they cannot be carried by an scf.for. The item count need
            # not divide BLOCKS, so an item past the end is MASKED by `live` rather
            # than skipped, which keeps every barrier below reached by the whole
            # workgroup. `live` zeroes that item's buffer descriptors' num_records so
            # its weight loads are killed in the texture unit, and suppresses its
            # publishes; its index is clamped onto a real item. What IS load-bearing
            # is covering every item -- one left out leaves its `mid` slots unwritten
            # and `down` polls them forever.
            UG8_UNITS = (HIDDEN // UG_UNIT_K) // WAVES
            XW = HIDDEN // (4 if use_fp8_block128 else 2)
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
                        coefficients = None  # MXFP8 scales are folded into the staged activations
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

            def ug8_units_sh(c, w_rg, w_ln, s_rg, live):
                """The FP8 shared expert's units: every sample at once (lane column = sample)."""
                rw = bo.create_buffer_resource_from_addr(
                    w_sug, num_records_bytes=live.select(fx.Int32(2 * INTER * HIDDEN), fx.Int32(0))
                )
                rs = bo.create_buffer_resource_from_addr(
                    s_sug, num_records_bytes=live.select(fx.Int32(SUG_S_BYTES), fx.Int32(0))
                )
                sn = n_sel()
                units = []
                for cc in range_constexpr(UG8_UNITS):
                    unit = wave * UG8_UNITS + cc
                    coefficients = None  # MXFP8 scales are folded into the staged activations
                    units.append(
                        unit_fp8mx(rw, rs, w_rg, s_rg, unit, HIDDEN, sn * XW + unit * 64, coefficients, w_ln)
                    )
                return units

            def shared_units(c, w_rg, w_ln, s_rg, live):
                if const_expr(SHARED_FP8):
                    return ug8_units_sh(c, w_rg, w_ln, s_rg, live)
                return ug8_units(c, w_rg, w_ln, s_rg, fx.Int32(SHARED_EXPERT), None, live)

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

            # The shared expert's tiles (uu < UG_PER_SLOT) are computed once each, for
            # every sample at once, by the CTA whose first tile it is. The routed work is
            # S * N_UG_TASKS (tile, sample) items spread evenly over the CTAs: as whole
            # tiles (each carrying all S samples) the 288 tiles left 32 CTAs with two,
            # 2 * S items, while the rest had S -- the stage's tail. Its weights are
            # prefetched ahead of the routing wait and the input staging, as the
            # single-tile version did; dn_route and stage_moe_input are CTA-global.
            N_UG_ITEMS = S * N_UG_TASKS
            UG_ITEMS = (N_UG_ITEMS + G - 1) // G

            def ug8_item(k):
                """This CTA's k-th routed item: (live, uu, sample, c, slot, w_rg, w_ln, s_rg).
                The index is clamped so a dead item's LDS and descriptor indices stay in
                range; `live` is what suppresses its effects."""
                w = u0 + k * G
                live = w < N_UG_ITEMS
                ww = fx.min(w, fx.Int32(N_UG_ITEMS - 1))
                uu = ww % N_UG_TASKS
                sample = ww // N_UG_TASKS
                _l, _u, c, slot, _h, w_rg, w_ln, s_rg = ug8_tile(uu)
                return live, uu, sample, c, slot, w_rg, w_ln, s_rg

            live0, uu0, c0, slot0, has_sh0, wr0, wl0, sr0 = ug8_tile(u0)
            shared_pre = shared_units(c0, wr0, wl0, sr0, has_sh0 & live0)
            dn_route(load_bias())
            gpu.barrier()
            items = [ug8_item(0)]
            lv, uu_, sm, c_, sl, wr, wl, sr = items[0]
            cur = ug8_units(c_, wr, wl, sr, _uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv)
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
                    cur = ug8_units(c_, wr, wl, sr, _uniform(lds_ld(keys, sm * MOE_SLOTS + sl)), sm, lv)
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
        # routed units over (sample, slot, K chunk); with SHARED_FP8 the shared slot 0 is
        # its own group of FP8 units after them (a unit's format is compile-time)
        DN_SLOTS = TOP_K if SHARED_FP8 else MOE_SLOTS
        DN_NU = S * DN_SLOTS * DN_UNITS_PER_SLOT
        DN_UPW = (DN_NU + DN_WPR - 1) // DN_WPR
        DN_SH_NU = S * DN_UNITS_PER_SLOT if SHARED_FP8 else 0
        DN_SH_UPW = (DN_SH_NU + DN_WPR - 1) // DN_WPR
        DN_CPW = DN_UPW + DN_SH_UPW
        DN_BLK = S * MOE_SLOTS * INTER // 128
        DN_W_BYTES = HIDDEN * INTER // (2 if use_mxfp4_weight else 1)
        DN_S_BYTES = HIDDEN * (INTER // 32) if use_mxfp4_weight else HIDDEN // SCALE_BM * (INTER // 128) * 4
        SDN_S_BYTES = HIDDEN // SCALE_BM * (INTER // 128) * 4  # the FP8 shared expert's scales
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

            def dn_coefficients(q, s_q, slot_q):
                """A unit's factor: its route weight, in its sample's column only (an
                MXFP8 mid's scale is folded into the staged mid). A plain VGPR read of
                the one LDS word -- every lane reads the same address, a broadcast --
                not a readfirstlane: an SGPR copy of it per unit forced lgkmcnt(0) waits
                and pushed the kernel, already at its VGPR / SGPR ceilings, into more
                spills (measured: ~12 us of the S = 8 stage)."""

                def coefficient():
                    return (lane % 16 == s_q).select(lds_ld(dnw, s_q * MOE_SLOTS + slot_q), fx.Float32(0.0))

                return coefficient

            def u_dn_sh(cc):  # cc: this wave's cc-th unit of the FP8 shared expert
                qs = (wave % DN_WPR) * DN_SH_UPW + cc
                live = qs < DN_SH_NU
                r = fx.min(qs, DN_SH_NU - 1)
                s_q, kc = r // DN_UNITS_PER_SLOT, r % DN_UNITS_PER_SLOT
                q = s_q * MOE_SLOTS * DN_UNITS_PER_SLOT + kc  # slot 0's mid
                masked = DN_SH_NU % DN_WPR != 0
                wb = bo.create_buffer_resource_from_addr(
                    w_sdn, num_records_bytes=live.select(fx.Int32(HIDDEN * INTER), fx.Int32(0)) if masked else None
                )
                sb = bo.create_buffer_resource_from_addr(
                    s_sdn, num_records_bytes=live.select(fx.Int32(SDN_S_BYTES), fx.Int32(0)) if masked else None
                )
                rg = dn_rg + gu
                return unit_fp8mx(wb, sb, rg, rg, kc, INTER, q * 64, dn_coefficients(q, s_q, fx.Int32(0)), dn_ln)

            def u_dn(cc):  # cc: 128-k chunk of this wave
                if const_expr(cc >= DN_UPW):
                    return u_dn_sh(cc - DN_UPW)
                qu = (wave % DN_WPR) * DN_UPW + cc
                live = qu < DN_NU
                r = fx.min(qu, DN_NU - 1)
                s_q = r // (DN_SLOTS * DN_UNITS_PER_SLOT)
                slot_q = (r // DN_UNITS_PER_SLOT) % DN_SLOTS + (MOE_SLOTS - DN_SLOTS)
                kc = r % DN_UNITS_PER_SLOT
                q = (s_q * MOE_SLOTS + slot_q) * DN_UNITS_PER_SLOT + kc  # unit over (sample, slot, K chunk)
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
                    coefficients = dn_coefficients(q, s_q, slot_q)
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
            pre = [u_dn(cc) for cc in range(min(DN_BATCH, DN_CPW))]
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
                        # the power-of-two scale folds in exactly (an FP8 value times
                        # 2**k is a bf16), so the MFMAs need no per-block factor
                        lds_st(xs, blk * 64 + lane, bf16_pair(d0 * qs, d1 * qs))
                    else:
                        lds_st(xs, blk * 64 + lane, bf16_pair(mids[b][0], mids[b][1]))
            gpu.barrier()
            acc = run_units(u_dn, DN_CPW, DN_BATCH, pre)

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
        kv_rope: Int64,
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
        w_sug: Int64,
        s_sug: Int64,
        w_sdn: Int64,
        s_sdn: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        timeline_buf: Int64,
        step: Int64,
        hang: Int64,
        state_slots: Int64,
        tok_ids: Int64,
        tid2eid: Int64,
        block_tables: Int64,
        i_cache_s: Int64,
        rank: Int32,
        layer: Int32,
        st_kv: Int32,
        st_i: Int32,
        st_ic: Int32,
        use_hash: Int32,
        bt_stride: Int32,
        env_rows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        dsv4_kernel(
            h_in,
            x_out,
            cur_pos,
            kv_cache,
            kv_rope,
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
            w_sug,
            s_sug,
            w_sdn,
            s_sdn,
            scratch,
            sym,
            peers,
            timeline_buf,
            step,
            hang,
            state_slots,
            tok_ids,
            tid2eid,
            block_tables,
            i_cache_s,
            rank,
            layer,
            st_kv,
            st_i,
            st_ic,
            use_hash,
            bt_stride,
            env_rows,
        ).launch(grid=(G,), block=(THREADS,), stream=stream)

    # LLVM's VectorCombine (foldShuffleToIdentity) goes exponential on the S == 1
    # up/gate's FP8 shared-expert units: a real-dims compile went from ~6 s to >24 min,
    # past the 600 s compile lock other ranks wait on.
    return flyc.compile[{"llvm_options": {"disable-vector-combine": True}}](launch_dsv4)


# ---------------------------------------------------------------- step advance
SCRUB_PAIRS = THREADS  # mailbox pairs each step advance checks: one per thread, one round trip


def scrub_period(n_pairs: int) -> int:
    """Steps for the step advance to visit every one of ``n_pairs`` pairs: a power
    of two, so ``step % period`` stays continuous through the int32 wrap."""
    return 1 << max(0, -(-n_pairs // SCRUB_PAIRS) - 1).bit_length()


def build_advance_step(scr_pairs: int, sym_pairs: int):
    """The ``@flyc.jit`` step advance for one scratch: ``step += 1`` plus a scrub.

    Tags are ``step * LAYER_SLOTS + layer + 1`` in int32, so they come round again
    after 2**32 / LAYER_SLOTS steps (~46 h at 5 ms a step), and a mailbox left
    unwritten that long would then read as fresh -- many are written only when
    their data calls for it (``cnew`` on compression steps, the indexer's live
    tiles, the picked experts' ``mid``). So each advance also zeroes the stale
    pairs of one slice of the scratch's mailboxes and its symmetric buffer,
    ``scrub_period`` slices in turn: no pair stays stale for more than a period,
    far short of the wrap, and a zeroed pair matches no tag (tags are never 0).

    Stale is older than the step just done (which stays readable, for
    ``Dsv4MoeLayer.debug``): those launches have all completed on this rank, and
    no consumer accepts an old tag. A peer can be one launch ahead, writing
    next-step tags into the symmetric buffer meanwhile; those are newer, so kept,
    and the compare-and-swap leaves a pair alone if one lands between the read
    and the clear. Traffic is push-only
    (ranks poll their own memory), so no peer is reading what is cleared."""
    n = scr_pairs + sym_pairs
    period = scrub_period(n)
    chunk = -(-n // period)
    per_thread = -(-chunk // THREADS)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def advance_kernel(step: Int64, scratch: Int64, sym: Int64):
        tid = fx.thread_idx.x

        def scrub(addr, new_base):
            """Zero the pair at ``addr`` if its tag is nonzero and at most ``new_base``."""
            ptr = fx.inttoptr(fx.PointerType.get(fx.Int64.ir_type, fx.AddressSpace.Global, 8), addr)
            old = fx.Int64(fx.generic_load(ptr, memory_order=fx.AtomicOrdering.Monotonic, syncscope="agent"))
            tg = fx.Int32(old >> 32)  # a pair is (value, tag): the tag is the high word
            if (tg != 0) & ((new_base - tg) >= 0):
                fx.atomic_cas(ptr, old, fx.Int64(0))

        s = _uniform(bo.buffer_load(_rsrc(step), 0, vec_width=1, dtype=T.i32))
        new_base = s * LAYER_SLOTS  # every tag of steps < s is at most this
        base = (s & (period - 1)) * chunk
        for j in range_constexpr(per_thread):
            o = j * THREADS + tid
            i = base + o
            if (o < chunk) & (i < n):
                if i < scr_pairs:
                    scrub(scratch + fx.Int64(i) * 8, new_base)
                else:
                    scrub(sym + fx.Int64(i - scr_pairs) * 8, new_base)
        gpu.barrier()  # every thread has read the step
        if tid == 0:
            bo.buffer_store(s + 1, _rsrc(step), 0)

    @flyc.jit
    def advance_step(step: Int64, scratch: Int64, sym: Int64, stream: fx.Stream = fx.Stream(None)):
        advance_kernel(step, scratch, sym).launch(grid=(1,), block=(THREADS,), stream=stream)

    return advance_step
