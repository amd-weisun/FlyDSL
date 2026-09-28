# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, layouts and the torch golden for the DeepSeek-V4 attention+MoE layer.

One rank's TP shard of one fused layer, matching what the monokernel will fuse.
DeepSeek-V4 drops MLA's absorbed ``W_UK``/``W_UV`` for shared-KV (MQA) attention:
``wkv`` emits a single ``head_dim`` vector per token and K and V are the *same*
tensor, with RoPE occupying its last ``rope_dim`` lanes. The output projection is
grouped low-rank (``o_a`` per group, then a row-parallel ``o_b``).

This module currently covers the **sliding-window-only** layer (``compress_ratio
== 0``): the attention rewrite plus the V4 MoE, without the KV compressor or the
lightning indexer. Those extend :func:`golden_layer` for the HCA (ratio 128)
and CSA (ratio 4) variants respectively; hyper-connections replace the plain
residual at that point too (``hc_mult`` > 1).

Quantization mirrors the kernel, not the published checkpoint: every attention
matrix is row-major FP8 E4M3FN with FP32 block scales, and GEMV activations are
rounded to bf16 (the MFMA operand precision). ``tests/kernels/`` checks the
*algorithm* against DeepSeek's own reference implementation separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from kernels.common.mx_formats import quant_dequant_mxfp8, quantize_mxfp4
from kernels.dsv4_moe_layer.config import (
    COMPRESS_CSA,
    COMPRESS_ROPE_THETA,
    COMPRESS_SWA,
    EPS,
    HC_EPS,
    HC_MULT,
    HC_SINKHORN_ITERS,
    HEAD_DIM,
    HIDDEN,
    INTER,
    KEY_BLOCK,
    N_EXPERTS,
    O_GROUPS,
    O_LORA,
    Q_LORA,
    ROPE_DIM,
    ROUTE_SCALE,
    SWIGLU_LIMIT,
    TOP_K,
    WINDOW,
    ExpertActivation,
    ExpertWeight,
    MoeMode,
    as_moe_mode,
    moe_format,
)
from kernels.mla_moe_layer.reference import (
    _rand_fp8,
    bf,
    dequant,
    dequant_expert,
    quant_dequant,
    rope,
    scale_shape,
)


@dataclass
class V4Config:
    """One rank's shard of a DeepSeek-V4 layer. Defaults are V4-Pro at TP8."""

    heads: int = 16  # local; 128 global / 8 ranks
    hidden: int = HIDDEN
    q_lora: int = Q_LORA
    head_dim: int = HEAD_DIM  # K and V share this; rope lives in its tail
    rope_dim: int = ROPE_DIM
    o_groups: int = O_GROUPS  # local; 16 global / 8 ranks
    o_lora: int = O_LORA
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER  # local; 3072 global / 8 ranks
    window: int = WINDOW
    route_scale: float = ROUTE_SCALE
    swiglu_limit: float = SWIGLU_LIMIT
    eps: float = EPS
    rope_theta: float = 1.0e4
    compress_ratio: int = COMPRESS_SWA  # 0 = sliding window only, 128 = HCA
    compress_rope_theta: float = COMPRESS_ROPE_THETA
    max_seq: int = 4096  # sizes the compressed half of the KV cache
    hc_mult: int = HC_MULT  # 1 = plain residual
    hc_sinkhorn_iters: int = HC_SINKHORN_ITERS
    hc_eps: float = HC_EPS

    @property
    def hc_mix(self) -> int:
        return (2 + self.hc_mult) * self.hc_mult

    @property
    def rope_base(self) -> float:
        """The rope base for THIS layer.

        V4 chooses per layer, not per consumer: a compressing layer rotates
        everything -- window q/kv, the compressed rows, and the indexer -- on
        ``compress_rope_theta`` (with YaRN), and a pure sliding-window layer on
        ``rope_theta`` (without). Getting this per-consumer instead of per-layer
        is an easy and invisible mistake, so it lives in one place.
        """
        return self.compress_rope_theta if self.compress_ratio else self.rope_theta

    @property
    def overlap(self) -> bool:
        """CSA compresses with overlapping windows -- DeepSeek ties that to ratio 4.
        Each entry then pools 2*ratio tokens at a stride of ratio."""
        return self.compress_ratio == COMPRESS_CSA

    @property
    def c_coff(self) -> int:
        """Channel multiplier on the compressor's projections: overlapping windows
        carry two halves, one for the previous window and one for the current."""
        return 2 if self.overlap else 1

    @property
    def c_rows(self) -> int:
        """Rows of compressor state: two windows' worth when overlapping."""
        return self.compress_ratio * self.c_coff

    @property
    def n_compressed(self) -> int:
        """Compressed cache slots. The window and the compressed entries share one
        cache, the compressed half starting at ``window`` -- which is what lets the
        attention gather span both from a single index list."""
        return 0 if self.compress_ratio == 0 else self.max_seq // self.compress_ratio

    @property
    def cache_rows(self) -> int:
        return self.window + self.n_compressed

    @property
    def n_keys(self) -> int:
        """Length of the attention's index list, padded to the key tile. The split
        stage walks this, NOT the window: with compression the gather has to reach
        past the window into the compressed half of the cache."""
        rows = self.cache_rows
        return (rows + KEY_BLOCK - 1) // KEY_BLOCK * KEY_BLOCK

    @property
    def hc_rows(self) -> int:
        """hc_mix padded to the MFMA row group (24 -> 32 at hc_mult 4)."""
        return (self.hc_mix + 15) // 16 * 16

    @property
    def nope_dim(self) -> int:
        return self.head_dim - self.rope_dim

    @property
    def group_dim(self) -> int:
        """Slice of the concatenated heads that one ``o_a`` group consumes."""
        return self.heads * self.head_dim // self.o_groups

    @property
    def shared_expert(self) -> int:
        """The shared expert sits last in the bank, as in the GLM-5/V3 layout."""
        return self.n_experts

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    def validate(self) -> None:
        # NOTE: CSA is gated at the layer (validate_shard), not here. The
        # overlapping compressor it needs IS modelled, and is tested on its own; what
        # is missing is the lightning indexer, which only a whole layer needs.
        assert self.compress_ratio == 0 or self.window % self.compress_ratio == 0
        assert self.nope_dim % 64 == 0, "act_quant blocks the nope part by 64"
        assert self.heads % self.o_groups == 0
        assert self.head_dim % 2 == 0 and self.rope_dim % 2 == 0


def fp8_mats(cfg: V4Config):
    """(rows, K, BK) of every FP8 attention matrix in one rank's shard."""
    return {
        # wq_a, wkv and -- when the layer compresses -- the compressor's own wkv and
        # wgate all read the same normed input, so they fuse into one GEMV rather
        # than costing a second weight stream and another dependency
        "qkv_a": (
            cfg.q_lora + cfg.head_dim + (2 * cfg.c_coff * cfg.head_dim if cfg.compress_ratio else 0),
            cfg.hidden,
            128,
        ),
        "q_b": (cfg.heads * cfg.head_dim, cfg.q_lora, 128),
        "o_a": (cfg.o_groups * cfg.o_lora, cfg.group_dim, 128),
        "o_b": (cfg.hidden, cfg.o_groups * cfg.o_lora, 128),
    }


@dataclass
class LayerWeights:
    cfg: V4Config
    t: dict  # name -> tensor


def make_weights(
    rank: int,
    cfg: V4Config | None = None,
    device="cuda",
    seed: int = 1234,
    moe_mode: MoeMode | str = MoeMode.A8W4,
) -> LayerWeights:
    """Replicated tensors share ``seed``; TP shards add ``rank`` to it.

    ``moe_mode`` defaults to A8W4 because V4 ships native MXFP4 expert weights.
    """
    cfg = cfg or V4Config()
    cfg.validate()
    expert_weight = moe_format(moe_mode).weight
    rep = torch.Generator(device=device).manual_seed(seed)
    shd = torch.Generator(device=device).manual_seed(seed + 1 + rank)
    t = {}
    bfl = torch.bfloat16

    t["g_in"] = (1 + 0.1 * torch.randn(cfg.hidden, generator=rep, device=device)).to(bfl)
    t["g_q"] = (1 + 0.1 * torch.randn(cfg.q_lora, generator=rep, device=device)).to(bfl)
    t["g_kv"] = (1 + 0.1 * torch.randn(cfg.head_dim, generator=rep, device=device)).to(bfl)
    t["g_post"] = (1 + 0.1 * torch.randn(cfg.hidden, generator=rep, device=device)).to(bfl)
    # per-head learnable softmax sink, fp32
    t["attn_sink"] = 0.5 * torch.randn(cfg.heads, generator=shd, device=device)

    for name, (rows, k, bk) in fp8_mats(cfg).items():
        # qkv_a is replicated (wkv is not TP-sharded in V4); the rest are shards
        gen = rep if name == "qkv_a" else shd
        t[f"w_{name}"], t[f"s_{name}"] = _rand_fp8(rows, k, bk, gen, device)

    if cfg.compress_ratio:
        # The compressor runs in fp32, and is replicated: it consumes the same
        # layer input on every rank.
        r = cfg.compress_ratio
        t["ape"] = 0.5 * torch.randn(r, cfg.c_coff * cfg.head_dim, generator=rep, device=device)
        t["g_ckv"] = (1 + 0.1 * torch.randn(cfg.head_dim, generator=rep, device=device)).to(bfl)

    if cfg.hc_mult > 1:
        # hyper-connection mixers, fp32 in the checkpoint. Replicated: every rank
        # must derive the same pre/post/comb or the residual streams diverge.
        for side in ("attn", "ffn"):
            # stored bf16 and row-padded: the kernel consumes this as a packed
            # MFMA operand, and halving the bytes matters because K is hc*hidden
            fn = torch.zeros(cfg.hc_rows, cfg.hc_mult * cfg.hidden, device=device)
            fn[: cfg.hc_mix] = (
                torch.randn(cfg.hc_mix, cfg.hc_mult * cfg.hidden, generator=rep, device=device)
                / (cfg.hc_mult * cfg.hidden) ** 0.5
            )
            t[f"hc_{side}_fn"] = fn.to(bfl)
            t[f"hc_{side}_base"] = torch.randn(cfg.hc_mix, generator=rep, device=device) * 0.5
            t[f"hc_{side}_scale"] = torch.rand(3, generator=rep, device=device) + 0.5

    t["w_r"] = (torch.randn(cfg.n_experts, cfg.hidden, generator=rep, device=device) / cfg.hidden**0.5 * 4).to(bfl)
    t["bias"] = torch.randn(cfg.n_experts, generator=rep, device=device) * 0.1

    n_bank = cfg.n_experts + 1
    if expert_weight is ExpertWeight.FP8_BLOCK128:
        ug_q = torch.empty(n_bank, 2 * cfg.inter, cfg.hidden, dtype=torch.float8_e4m3fn, device=device)
        ug_s = torch.empty(n_bank, *scale_shape(2 * cfg.inter, cfg.hidden, 128), device=device)
        dn_q = torch.empty(n_bank, cfg.hidden, cfg.inter, dtype=torch.float8_e4m3fn, device=device)
        dn_s = torch.empty(n_bank, *scale_shape(cfg.hidden, cfg.inter, 128), device=device)
        for e in range(n_bank):
            ug_q[e], ug_s[e] = _rand_fp8(2 * cfg.inter, cfg.hidden, 128, shd, device)
            dn_q[e], dn_s[e] = _rand_fp8(cfg.hidden, cfg.inter, 128, shd, device)
    else:
        ug_q = torch.empty(n_bank, 2 * cfg.inter, cfg.hidden // 2, dtype=torch.uint8, device=device)
        ug_s = torch.empty(n_bank, 2 * cfg.inter, cfg.hidden // 32, dtype=torch.uint8, device=device)
        dn_q = torch.empty(n_bank, cfg.hidden, cfg.inter // 2, dtype=torch.uint8, device=device)
        dn_s = torch.empty(n_bank, cfg.hidden, cfg.inter // 32, dtype=torch.uint8, device=device)
        for e in range(n_bank):
            ug = torch.randn(2 * cfg.inter, cfg.hidden, generator=shd, device=device) / cfg.hidden**0.5
            dn = torch.randn(cfg.hidden, cfg.inter, generator=shd, device=device) / cfg.inter**0.5
            ug_q[e], ug_s[e] = quantize_mxfp4(ug)
            dn_q[e], dn_s[e] = quantize_mxfp4(dn)
    t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = ug_q, ug_s, dn_q, dn_s
    return LayerWeights(cfg, t)


def rmsnorm(x: torch.Tensor, g: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm; ``g=None`` is the weightless per-head scale applied to the query."""
    x = x.float()
    y = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    return y if g is None else y * g.float()


FP4_MAX = 6.0
# e2m1 representable magnitudes in code order; the code IS the index
FP4_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_BLOCK = 32


def hadamard(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh-Hadamard transform over the last dim, scaled by n**-0.5.

    V4 rotates the indexer's queries and compressed keys into the Hadamard basis
    before quantizing them to FP4: the transform spreads any outlier across all
    lanes, so a block's amax stops being set by one coordinate. Scoring is a dot
    product and the scaled transform is orthonormal, so it leaves scores intact.
    """
    n = x.shape[-1]
    assert n & (n - 1) == 0, f"hadamard size must be a power of two, got {n}"
    y = x.float().reshape(-1, n)
    h = 1
    while h < n:
        y = y.reshape(-1, n // (2 * h), 2, h)
        a, b = y[:, :, 0, :].clone(), y[:, :, 1, :].clone()
        y[:, :, 0, :], y[:, :, 1, :] = a + b, a - b
        y = y.reshape(-1, n)
        h *= 2
    return (y * n**-0.5).reshape(x.shape)


def quant_dequant_fp4(x: torch.Tensor, block: int = FP4_BLOCK) -> torch.Tensor:
    """FP4 (e2m1) round trip in blocks of ``block``, with power-of-2 scales.

    Unlike the FP8 path, the scale is rounded UP to a power of two, so it is
    exact in the exponent and costs no mantissa. Ties round to the even code.
    """
    n = x.shape[-1]
    xb = x.float().reshape(*x.shape[:-1], n // block, block)
    amax = xb.abs().amax(-1, keepdim=True).clamp(min=FP4_MAX * 2.0**-126)
    # ceil(log2(amax / FP4_MAX)) by exponent arithmetic, as the model does
    bits = (amax / FP4_MAX).view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & ((1 << 23) - 1)) != 0).to(torch.int32)
    s = torch.ldexp(torch.ones_like(amax), e)
    q = (xb / s).clamp(-FP4_MAX, FP4_MAX)
    lv = torch.tensor(FP4_LEVELS, device=x.device, dtype=torch.float32)
    mid = (lv[1:] + lv[:-1]) / 2
    mag = q.abs()
    down, up = torch.bucketize(mag, mid, right=False), torch.bucketize(mag, mid, right=True)
    idx = torch.where(up != down, torch.where(down % 2 == 0, down, up), down)
    return (torch.sign(q) * lv[idx] * s).reshape(x.shape).to(x.dtype)


def hc_split_sinkhorn(mixes: torch.Tensor, scale, base, cfg: V4Config):
    """Split the mix vector into hyper-connection coefficients.

    ``mixes`` [..., (2 + hc) * hc], packed pre | post | comb. ``comb`` is
    row-softmaxed then Sinkhorn-normalised towards doubly stochastic, which is
    the "manifold constraint" in mHC.
    """
    hc, eps = cfg.hc_mult, cfg.hc_eps
    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(cfg.hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x: torch.Tensor, fn: torch.Tensor, scale, base, cfg: V4Config):
    """[S, hc, hidden] -> ([S, hidden], post [S, hc], comb [S, hc, hc]).

    The mixes come from a weightless RMS scale over the *whole* hc*hidden stream,
    so this is a reduction over 4x the layer width before anything else runs.
    """
    shape, dtype = x.size(), x.dtype
    xf = x.flatten(-2).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.eps)
    # bf16 weights, matching what the kernel's MFMA actually consumes
    mixes = torch.nn.functional.linear(xf, fn[: cfg.hc_mix].float()) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, cfg)
    y = (pre.unsqueeze(-1) * xf.view(shape)).sum(dim=-2)
    return y.to(dtype), post, comb


def hc_post(x: torch.Tensor, residual: torch.Tensor, post, comb):
    """([S, hidden], residual [S, hc, hidden]) -> [S, hc, hidden].

    out[k] = post[k] * x + sum_j comb[j, k] * residual[j]
    """
    mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2).float()).sum(dim=-3)
    return (post.unsqueeze(-1) * x.unsqueeze(-2).float() + mixed).type_as(x)


def compress_step(
    kv,
    score,
    cur_pos,
    cfg,
    t,
    kv_state,
    score_state,
    cache,
    cos_c,
    sin_c,
    head_dim=None,
    ape=None,
    gamma=None,
    base=None,
    rotate=False,
):
    """One decode step of the KV compressor.

    Every token feeds a rolling window of ``compress_ratio`` positions; only the
    last of each window emits a compressed entry. The pooling weight is a
    **per-channel** softmax over the window's positions -- not one scalar per
    token -- plus a learned absolute-position bias ``ape``. Mutates
    ``kv_state`` / ``score_state`` / ``cache``, as the kernel does. ``kv`` and
    ``score`` come from the fused qkv_a projection.

    When ``cfg.overlap`` (CSA), the windows overlap: each entry pools 2*ratio
    tokens at a stride of ratio, taking the previous window's overlap channels and
    the current window's normal ones.

    The indexer runs a SECOND compressor over the same tokens, at its own (smaller)
    ``head_dim`` and with its own ``ape`` / ``gamma`` / cache, so those are
    overridable. ``rotate`` picks that one's tail: Hadamard over the whole row and
    then FP4, where the attention compressor does FP8 on the nope part only.
    """
    r, rd = cfg.compress_ratio, cfg.rope_dim
    d = cfg.head_dim if head_dim is None else head_dim
    ape = t["ape"] if ape is None else ape
    gamma = t["g_ckv"] if gamma is None else gamma
    base = cfg.window if base is None else base
    kv = kv.float()
    score = score.float() + ape[cur_pos % r]
    if cfg.overlap:
        # the current window fills the second half of the state; the first half
        # still holds the previous window, and a pooled entry spans both
        kv_state[r + cur_pos % r] = kv
        score_state[r + cur_pos % r] = score
        if (cur_pos + 1) % r:
            return None
        ks = torch.cat([kv_state[:r, :d], kv_state[r:, d:]], dim=0)
        ss = torch.cat([score_state[:r, :d], score_state[r:, d:]], dim=0)
        pooled = (ks * ss.softmax(dim=0)).sum(dim=0)
        kv_state[:r] = kv_state[r:]  # the current window becomes the previous one
        score_state[:r] = score_state[r:]
    else:
        kv_state[cur_pos % r] = kv
        score_state[cur_pos % r] = score
        if (cur_pos + 1) % r:
            return None
        pooled = (kv_state * score_state.softmax(dim=0)).sum(dim=0)
    # the norm returns bf16 in the model this reproduces, so the RoPE and the FP8
    # round trip below see bf16 -- keeping fp32 here is more precise than the
    # reference and shows up as a systematic offset in the compressed entry
    v = bf(rmsnorm(pooled.to(torch.bfloat16), gamma, cfg.eps))
    anchor = cur_pos + 1 - r  # the window's FIRST position carries the rotation
    # bf(): the model rotates in place inside a bf16 tensor, so the rotated lanes
    # are rounded before anything downstream sees them
    v = torch.cat([v[:-rd], bf(rope(v[-rd:], cos_c[anchor], sin_c[anchor]))])
    # The rotation spans the rope lanes too, so it has to follow them. bf() is
    # load-bearing: the model rotates a bf16 tensor and its transform casts back
    # before quantizing, so FP4 sees bf16 -- staying in fp32 here moves elements
    # that sit near a level boundary by a whole FP4 step.
    v = quant_dequant_fp4(bf(hadamard(v))) if rotate else torch.cat([quant_dequant(v[:-rd], 64), v[-rd:]])
    cache[base + cur_pos // r] = v.to(torch.bfloat16)
    return v


def layer_idxs(cur_pos: int, samples: int, cfg: V4Config, device) -> torch.Tensor:
    """Ring slots for the sliding window, then the compressed entries so far.

    Both live in one cache -- window first, compressed after -- which is what lets
    the attention gather span both from a single index list. Unwritten slots are -1.
    """
    win = window_idxs(cur_pos, samples, cfg.window, device)
    if not cfg.compress_ratio:
        return win
    rows = []
    for s in range(samples):
        n = (cur_pos + s + 1) // cfg.compress_ratio
        rows.append([cfg.window + i for i in range(n)] + [-1] * (cfg.n_compressed - n))
    comp = torch.tensor(rows, dtype=torch.int32, device=device)
    idx = torch.cat([win, comp], dim=1)
    pad = cfg.n_keys - idx.shape[1]
    if pad:
        idx = torch.cat([idx, torch.full((samples, pad), -1, dtype=torch.int32, device=device)], dim=1)
    return idx


def window_idxs(cur_pos: int, samples: int, window: int, device) -> torch.Tensor:
    """Ring-buffer slots for the sliding window, oldest first, -1 where unfilled.

    Mirrors ``get_window_topk_idxs`` in DeepSeek's reference: once the ring is
    full the slots are a rotation of ``range(window)``; before that they are the
    written prefix, right-padded with -1.
    """
    rows = []
    for s in range(samples):
        p = cur_pos + s
        if p + 1 >= window:
            start = (p + 1) % window
            rows.append([(start + i) % window for i in range(window)])
        else:
            rows.append(list(range(p + 1)) + [-1] * (window - p - 1))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def route(scores: torch.Tensor, bias: torch.Tensor, cfg: V4Config):
    """sqrt-softplus scores [E] -> (indices [top_k], probs [top_k]) in score order.

    Flat top-k over all experts: V4 drops V3's group-limited routing. As in the
    kernel's packed-key argmax, the selection key is the order-preserving bits of
    the f32 ``score + bias`` with the low byte replaced by ``255 - expert id``, so
    keys are unique and near-ties go to the lower expert id. The returned weight
    comes from the *unbiased* score (``noaux_tc``).
    """
    n = cfg.n_experts
    bits = (scores.float() + bias.float()).view(torch.int32).long()
    okey = torch.where(bits >= 0, bits ^ (1 << 31), ~bits & 0xFFFFFFFF) & 0xFFFFFFFF
    # the id field is sized to the expert count: V4's 384 does not fit the 8 bits
    # that GLM-5/V3's 256 experts filled exactly
    id_bits = max(8, (n - 1).bit_length())
    id_mask = (1 << id_bits) - 1
    key = (okey & (0xFFFFFFFF ^ id_mask)) | (id_mask - torch.arange(n, device=scores.device))
    idx = torch.argsort(key, descending=True)[: cfg.top_k]
    p = scores[idx]
    return idx, p / p.sum() * cfg.route_scale


def golden_layer(
    W: LayerWeights,
    h,
    cur_pos: int,
    kv_cache,
    indices,
    cos,
    sin,
    allreduce,
    moe_mode: MoeMode | str = MoeMode.A8W4,
    kv_state=None,
    score_state=None,
    cos_c=None,
    sin_c=None,
):
    """One rank's view of a V4 layer. Mutates ``kv_cache`` (and the compressor state).

    ``h`` is [S, hidden] when ``cfg.hc_mult == 1`` (plain residual) and
    [S, hc_mult, hidden] otherwise -- V4 carries hc_mult parallel residual
    streams, contracted to one by ``hc_pre`` and re-expanded by ``hc_post``.
    ``kv_cache`` is a ring of ``cfg.window`` rows of ``head_dim`` (K and V both).
    ``indices`` [S, n_keys] are ring slots, -1 meaning "not yet written".
    Returns a dict of intermediates keyed like the kernel's debug scratch.
    """
    cfg, t = W.cfg, W.t
    H, S = cfg.heads, h.shape[0]
    rd, hd = cfg.rope_dim, cfg.head_dim
    dq = {n: dequant(t[f"w_{n}"], t[f"s_{n}"], bk) for n, (_, _, bk) in fp8_mats(cfg).items()}

    if cfg.hc_mult > 1:
        xin, post_a, comb_a = hc_pre(h, t["hc_attn_fn"], t["hc_attn_scale"], t["hc_attn_base"], cfg)
    else:
        xin, post_a, comb_a = h, None, None
    x = bf(rmsnorm(xin, t["g_in"], cfg.eps))
    qkv = x @ dq["qkv_a"].T
    hd = cfg.head_dim
    q_a = qkv[:, : cfg.q_lora]
    kv = qkv[:, cfg.q_lora : cfg.q_lora + hd]
    c_kv = qkv[:, cfg.q_lora + hd : cfg.q_lora + 2 * hd] if cfg.compress_ratio else None
    c_gate = qkv[:, cfg.q_lora + 2 * hd :] if cfg.compress_ratio else None

    # query: lora -> per-head, then a weightless RMS over the whole head, then rope
    q = (bf(rmsnorm(q_a, t["g_q"], cfg.eps)) @ dq["q_b"].T).view(S, H, hd)
    q = rmsnorm(q, None, cfg.eps)
    pos = [cur_pos + s for s in range(S)]
    q = torch.stack(
        [torch.cat([q[s, :, :-rd], rope(q[s, :, -rd:], cos[pos[s]], sin[pos[s]])], dim=-1) for s in range(S)]
    )

    # shared KV: one row per token, rope in the tail, nope part FP8 round-tripped
    for s in range(S):
        v = rmsnorm(kv[s], t["g_kv"], cfg.eps)
        v = torch.cat([quant_dequant(v[:-rd], 64), rope(v[-rd:], cos[pos[s]], sin[pos[s]])])
        kv_cache[pos[s] % cfg.window] = v.to(torch.bfloat16)
        if cfg.compress_ratio:
            # the compressor sees the same normed input the projections do, and
            # writes into the compressed half of the same cache
            compress_step(c_kv[s], c_gate[s], pos[s], cfg, t, kv_state, score_state, kv_cache, cos_c, sin_c)
    kvf = kv_cache.float()

    # gather-sparse attention with a per-head sink in the denominator
    sink = t["attn_sink"].float()
    o = torch.empty(S, H, hd, device=h.device)
    for s in range(S):
        keys = indices[s].long()
        valid = keys >= 0
        k = kvf[keys.clamp(min=0)]
        sc = (bf(q[s]) @ k.T) * cfg.softmax_scale
        sc = sc.masked_fill(~valid.unsqueeze(0), float("-inf"))
        # split softmax over 64-key splits: bf16 unnormalized probs feed P V (MFMA)
        ms, ls, accs = [], [], []
        for k0 in range(0, keys.numel(), 64):
            scs = sc[:, k0 : k0 + 64]
            m = scs.amax(-1, keepdim=True)
            m = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
            p = torch.exp(scs - m)
            ms.append(m)
            ls.append(p.sum(-1, keepdim=True))
            accs.append(bf(p) @ k[k0 : k0 + 64])
        mx = torch.stack(ms).amax(0)
        w = [torch.exp(m - mx) for m in ms]
        denom = sum(li * wi for li, wi in zip(ls, w)) + torch.exp(sink.unsqueeze(-1) - mx)
        o[s] = sum(a * wi for a, wi in zip(accs, w)) / denom

    # V shares the RoPE'd K, so the output has to be de-rotated
    o = torch.stack(
        [torch.cat([o[s, :, :-rd], rope(o[s, :, -rd:], cos[pos[s]], sin[pos[s]], True)], dim=-1) for s in range(S)]
    )

    # grouped low-rank output projection: per-group o_a, then a row-parallel o_b
    og = bf(o).reshape(S, cfg.o_groups, cfg.group_dim)
    wa = dq["o_a"].view(cfg.o_groups, cfg.o_lora, cfg.group_dim)
    o_lora = torch.einsum("sgd,grd->sgr", og, wa).reshape(S, cfg.o_groups * cfg.o_lora)
    attn_out = allreduce(bf(o_lora) @ dq["o_b"].T)
    if cfg.hc_mult > 1:
        a = hc_post(attn_out.to(torch.bfloat16), h, post_a, comb_a)
    else:
        a = (h.float() + attn_out).to(torch.bfloat16)

    moe = golden_moe(W, a, allreduce, moe_mode=moe_mode)
    res = dict(q_a=q_a, kv=kv, q=q, o=o, o_lora=o_lora, a=a, xin=xin)
    res.update(moe)
    return res


def golden_moe(
    W: LayerWeights,
    a,
    allreduce,
    mid=None,
    sel=None,
    prob=None,
    xq=None,
    hash_ids=None,
    moe_mode: MoeMode | str = MoeMode.A8W4,
):
    """MoE half from the post-attention hidden state ``a`` [S, hidden] (bf16).

    ``xq`` overrides the quant-dequantized activation and ``mid``/``sel``/``prob``
    the down-projection inputs, so each stage can be checked from the kernel's own
    inputs. ``hash_ids`` [S, top_k] replaces scored routing with V4's hash routing
    (the first ``num_hash_layers`` layers look expert ids up by token id).
    """
    cfg, t = W.cfg, W.t
    mode = as_moe_mode(moe_mode)
    fmt = moe_format(mode)
    S = a.shape[0]
    out = {k: [] for k in ("sel", "prob", "mid")}

    if cfg.hc_mult > 1:
        ain, post_f, comb_f = hc_pre(a, t["hc_ffn_fn"], t["hc_ffn_scale"], t["hc_ffn_base"], cfg)
    else:
        ain, post_f, comb_f = a, None, None
    x2 = rmsnorm(ain, t["g_post"], cfg.eps)
    # V4 scores with sqrt(softplus(.)) instead of V3/GLM-5's sigmoid
    scores = torch.nn.functional.softplus(bf(x2) @ t["w_r"].float().T).sqrt()

    if fmt.activation is ExpertActivation.FP8_BLOCK128:
        xq_ref = quant_dequant(x2)
    elif fmt.activation is ExpertActivation.MXFP8_BLOCK32:
        xq_ref = quant_dequant_mxfp8(x2)
    else:
        xq_ref = bf(x2)
    xq = xq_ref if xq is None else xq.float()

    lim = cfg.swiglu_limit
    y = torch.zeros(S, cfg.hidden, device=a.device)
    for s in range(S):
        if hash_ids is None:
            idx, p = route(scores[s], t["bias"], cfg)
        else:
            idx = hash_ids[s].long()
            raw = scores[s][idx]
            p = raw / raw.sum() * cfg.route_scale
        experts = [cfg.shared_expert] + idx.tolist()
        weights = [1.0] + p.tolist()
        mids = []
        for e in experts:
            ug = dequant_expert(t["w_ug"][e], t["s_ug"][e], fmt.weight) @ xq[s]
            gate, up = ug[: cfg.inter], ug[cfg.inter :]
            if lim > 0:
                # note the asymmetry: up is clamped both sides, gate only above
                gate = gate.clamp(max=lim)
                up = up.clamp(min=-lim, max=lim)
            value = torch.nn.functional.silu(gate) * up
            mids.append(bf(value) if fmt.activation is ExpertActivation.BF16 else value)
        out["sel"].append(torch.tensor(experts, device=a.device, dtype=torch.int32))
        out["prob"].append(torch.tensor(weights, device=a.device))
        out["mid"].append(torch.stack(mids))

    for s in range(S):
        experts = out["sel"][s].tolist() if sel is None else sel[s].tolist()
        weights = out["prob"][s].tolist() if prob is None else prob[s].tolist()
        for j, (e, wgt) in enumerate(zip(experts, weights)):
            m = out["mid"][s][j] if mid is None else mid[s, j].float()
            if fmt.activation is ExpertActivation.FP8_BLOCK128:
                activation = quant_dequant(m)
            elif fmt.activation is ExpertActivation.MXFP8_BLOCK32:
                activation = quant_dequant_mxfp8(m)
            else:
                activation = bf(m)
            y[s] += wgt * (dequant_expert(t["w_dn"][e], t["s_dn"][e], fmt.weight) @ activation)

    ffn_out = allreduce(y)
    if cfg.hc_mult > 1:
        x_out = hc_post(ffn_out.to(torch.bfloat16), a, post_f, comb_f)
    else:
        x_out = (a.float() + ffn_out).to(torch.bfloat16)
    return dict(
        scores=scores,
        xq=xq_ref,
        x_out=x_out,
        sel=torch.stack(out["sel"]),
        prob=torch.stack(out["prob"]),
        mid=torch.stack(out["mid"]),
    )
