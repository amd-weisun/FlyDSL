# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, state layouts and the torch golden for one rank's Qwen3.5-MoE decoder layer.

Math follows HF transformers ``modeling_qwen3_5_moe`` (checked by the reference test); rounding
points follow what baseline ATOM stores between its kernels (bf16 projections, conv output, state
and attention output). State layouts are ATOM's native ones:

- conv state ``[slots, conv_kernel - 1, conv_dim]`` bf16, oldest row first;
- recurrent state ``[slots, hv, v_dim, k_dim]`` bf16;
- full-attention K/V as a dequantized history ``[samples, max_seq, local_kv_heads, head_dim]`` fp32.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kernels.monokernel.qwen35.config import FP8_MAX, Q35Config
from kernels.monokernel.reference import _rand_fp8, bf, dequant, quant_dequant


@dataclass
class LayerWeights:
    cfg: Q35Config
    full: bool  # gated full attention; else Gated DeltaNet
    t: dict  # name -> tensor


def fp8_mats(cfg: Q35Config, full: bool):
    """(rows, K) of every FP8 block-128 matrix of the mixer, one rank's shard."""
    if full:
        hd = cfg.head_dim
        return {
            "q": (cfg.local_heads * 2 * hd, cfg.hidden),  # per head [q | gate]
            "k": (cfg.local_kv_heads * hd, cfg.hidden),
            "v": (cfg.local_kv_heads * hd, cfg.hidden),
            "o": (cfg.hidden, cfg.local_heads * hd),
        }
    return {
        "qkv": (cfg.conv_dim, cfg.hidden),  # [q | k | v]
        "z": (cfg.value_dim, cfg.hidden),
        "out": (cfg.hidden, cfg.value_dim),
    }


def make_weights(cfg: Q35Config, full: bool, device="cuda", seed: int = 1234) -> LayerWeights:
    """Random weights at checkpoint formats. Replicated tensors share ``seed``; shards add the rank."""
    cfg.validate()
    rep = torch.Generator(device=device).manual_seed(seed)
    shd = torch.Generator(device=device).manual_seed(seed + 1 + cfg.rank)
    bfl = torch.bfloat16
    t = {}
    # Gemma-style norms: the checkpoint holds w, the norm applies (1 + w)
    t["g_in"] = (0.1 * torch.randn(cfg.hidden, generator=rep, device=device)).to(bfl)
    t["g_post"] = (0.1 * torch.randn(cfg.hidden, generator=rep, device=device)).to(bfl)
    for name, (rows, k) in fp8_mats(cfg, full).items():
        t[f"w_{name}"], t[f"s_{name}"] = _rand_fp8(rows, k, 128, shd, device)
    if full:
        t["g_qn"] = (0.1 * torch.randn(cfg.head_dim, generator=rep, device=device)).to(bfl)
        t["g_kn"] = (0.1 * torch.randn(cfg.head_dim, generator=rep, device=device)).to(bfl)
    else:
        # in_proj_b then in_proj_a rows, one per local v-head (BF16 in the checkpoint)
        t["w_ba"] = (torch.randn(2 * cfg.hv, cfg.hidden, generator=shd, device=device) / cfg.hidden**0.5).to(bfl)
        t["w_conv"] = (0.5 * torch.randn(cfg.conv_dim, cfg.conv_kernel, generator=shd, device=device)).to(bfl)
        t["a_log"] = torch.log(torch.rand(cfg.hv, generator=shd, device=device) * 15.99 + 0.01)
        t["dt_bias"] = torch.randn(cfg.hv, generator=shd, device=device).to(bfl)
        # the gated norm applies its weight directly (not 1 + w); fp32 in the checkpoint
        t["g_norm"] = 1 + 0.1 * torch.randn(cfg.linear_v_dim, generator=rep, device=device)
    t["w_r"] = (torch.randn(cfg.n_experts, cfg.hidden, generator=rep, device=device) / cfg.hidden**0.5 * 4).to(bfl)
    t["w_sg"] = (torch.randn(1, cfg.hidden, generator=rep, device=device) / cfg.hidden**0.5).to(bfl)
    # expert bank: routed experts then the shared one; ug rows are [gate | up] of this rank's slice
    n, inter = cfg.n_experts + 1, cfg.local_inter
    t["w_ug"], t["s_ug"] = _rand_fp8(2 * inter, cfg.hidden, 128, shd, device, lead=(n,))
    t["w_dn"], t["s_dn"] = _rand_fp8(cfg.hidden, inter, 128, shd, device, lead=(n,))
    return LayerWeights(cfg, full, t)


def gemma_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * (1 + w)`` in fp32 (Qwen3_5MoeRMSNorm)."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def linear(W: LayerWeights, name: str, x: torch.Tensor) -> torch.Tensor:
    """FP8 block-128 matrix ``name`` times the bf16 activation ``x`` [S, K], fp32 out."""
    return bf(x) @ dequant(W.t[f"w_{name}"], W.t[f"s_{name}"], 128).T


def rope_cos_sin(positions: torch.Tensor, cfg: Q35Config):
    """cos/sin [S, rope_dim] (halves repeated, rotate-half convention)."""
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rope_dim, 2, device=positions.device).float() / cfg.rope_dim))
    f = positions.float()[:, None] * inv[None]
    f = torch.cat([f, f], -1)
    return f.cos(), f.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the first ``rope_dim`` dims of ``x`` [S, H, D]; the rest passes."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([xr * cos[:, None] + rot * sin[:, None], xp], -1)


def kv_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Per-token-per-head FP8 E4M3 (``amax / 448`` scale) over the last dim, dequantized."""
    amax = x.float().abs().amax(-1, keepdim=True)
    s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    return (x.float() / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s


def golden_gdn(
    W: LayerWeights,
    xn: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    slots_in: torch.Tensor,
    slots_out: torch.Tensor | None = None,
):
    """Gated DeltaNet on the normed input ``xn`` [S, hidden] (bf16 values); updates both states in place.

    Sample ``s`` reads state slot ``slots_in[s]`` and writes ``slots_out[s]`` (default: the same); a
    negative slot skips the sample. Returns this rank's ``out_proj`` partial (before the reduce).
    """
    cfg, t = W.cfg, W.t
    S = xn.shape[0]
    slots_out = slots_in if slots_out is None else slots_out
    qkv = bf(linear(W, "qkv", xn))
    z = bf(linear(W, "z", xn)).view(S, cfg.hv, cfg.linear_v_dim)
    ba = bf(bf(xn) @ t["w_ba"].float().T)
    b, a = ba[:, : cfg.hv], ba[:, cfg.hv :]
    beta = torch.sigmoid(b)
    g = -torch.exp(t["a_log"].float()) * F.softplus(a + t["dt_bias"].float())
    conv_out = torch.zeros_like(qkv)
    core = torch.zeros(S, cfg.hv, cfg.linear_v_dim, device=xn.device)
    for s in range(S):
        si, so = int(slots_in[s]), int(slots_out[s])
        if si < 0 or so < 0:
            continue
        window = torch.cat([conv_state[si].float(), qkv[s][None]], 0)  # [conv_kernel, conv_dim]
        conv_out[s] = bf(F.silu((window * t["w_conv"].float().T).sum(0)))
        conv_state[so] = window[1:].to(conv_state.dtype)
        q, k, v = conv_out[s].split([cfg.key_dim, cfg.key_dim, cfg.value_dim])
        q = q.view(cfg.hk, -1).repeat_interleave(cfg.v_per_k, 0)
        k = k.view(cfg.hk, -1).repeat_interleave(cfg.v_per_k, 0)
        v = v.view(cfg.hv, -1)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6) * cfg.linear_k_dim**-0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
        h = ssm_state[si].float() * torch.exp(g[s])[:, None, None]  # [hv, V, K]
        delta = (v - torch.einsum("hvk,hk->hv", h, k)) * beta[s][:, None]
        h = h + delta[:, :, None] * k[:, None, :]
        core[s] = torch.einsum("hvk,hk->hv", h, q)
        ssm_state[so] = h.to(ssm_state.dtype)
    core = bf(core)
    # Qwen3_5MoeRMSNormGated: norm in fp32, to bf16, times the weight, times silu(z)
    nrm = bf(core * torch.rsqrt(core.square().mean(-1, keepdim=True) + cfg.eps))
    gated = bf(bf(nrm * t["g_norm"].to(torch.bfloat16).float()) * F.silu(z))
    partial = linear(W, "out", gated.flatten(1))
    return dict(qkv=qkv, z=z, beta=beta, g=g, conv_out=conv_out, core=core, gated=gated, partial=partial)


def golden_full_attn(
    W: LayerWeights,
    xn: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    positions: torch.Tensor,
):
    """Gated GQA on ``xn`` [S, hidden]: sample ``s`` writes its K/V at ``positions[s]`` of row ``s`` of the
    caches (in place) and attends to positions ``0..positions[s]``. Returns the ``o_proj`` partial."""
    cfg, t = W.cfg, W.t
    S, hd, H, KH = xn.shape[0], cfg.head_dim, cfg.local_heads, cfg.local_kv_heads
    qg = bf(linear(W, "q", xn)).view(S, H, 2 * hd)
    q, gate = qg[..., :hd], qg[..., hd:]
    k = bf(linear(W, "k", xn)).view(S, KH, hd)
    v = bf(linear(W, "v", xn)).view(S, KH, hd)
    q = bf(gemma_rmsnorm(q, t["g_qn"], cfg.eps))
    k = bf(gemma_rmsnorm(k, t["g_kn"], cfg.eps))
    cos, sin = rope_cos_sin(positions, cfg)
    q, k = bf(apply_rope(q, cos, sin)), bf(apply_rope(k, cos, sin))
    out = torch.zeros(S, H, hd, device=xn.device)
    for s in range(S):
        p = int(positions[s])
        k_cache[s, p] = kv_quant_dequant(k[s]) if cfg.kv_fp8 else k[s]
        v_cache[s, p] = kv_quant_dequant(v[s]) if cfg.kv_fp8 else v[s]
        keys, vals = k_cache[s, : p + 1].float(), v_cache[s, : p + 1].float()  # [T, KH, hd]
        kv_of = torch.arange(H, device=xn.device) // (H // KH)
        logits = torch.einsum("hd,thd->ht", q[s], keys[:, kv_of]) * hd**-0.5
        out[s] = torch.einsum("ht,thd->hd", torch.softmax(logits, -1), vals[:, kv_of])
    gated = bf(bf(out) * torch.sigmoid(gate))
    partial = linear(W, "o", gated.flatten(1))
    return dict(q=q, gate=gate, k=k, v=v, attn=bf(out), gated=gated, partial=partial)


def route(logits: torch.Tensor, top_k: int):
    """Softmax over all experts, top-k, renormalized: (indices [top_k], weights [top_k])."""
    p = torch.softmax(logits.float(), -1)
    # bf16 logits tie often: equal values go to the lower expert id (stable), as the kernel's keys
    w, idx = torch.sort(p, descending=True, stable=True)
    w, idx = w[:top_k], idx[:top_k]
    return idx, w / w.sum()


def golden_moe(W: LayerWeights, a: torch.Tensor, allreduce, xq: torch.Tensor | None = None):
    """MoE half from the post-attention residual ``a`` [S, hidden] (bf16); returns ``x_out`` and stages.

    ``xq`` [S, hidden] overrides the quant-dequantized expert input, so the experts can be checked from
    a kernel's own input (FP8 rounding of a value on a midpoint differs between implementations)."""
    cfg, t = W.cfg, W.t
    S = a.shape[0]
    x2 = bf(gemma_rmsnorm(a, t["g_post"], cfg.eps))
    logits = bf(x2 @ t["w_r"].float().T)
    shared_gate = torch.sigmoid(bf(x2 @ t["w_sg"].float().T))[:, 0]
    xq_ref = quant_dequant(x2) if cfg.act_fp8 else x2
    xq = xq_ref if xq is None else xq.float()
    inter = cfg.local_inter
    y = torch.zeros(S, cfg.hidden, device=a.device)
    sel, prob, mids = [], [], []
    for s in range(S):
        idx, w = route(logits[s], cfg.top_k)
        # slot 0 the shared expert, then the picks in descending order (the kernel's slots)
        experts = [cfg.shared_expert] + idx.tolist()
        weights = [float(shared_gate[s])] + w.tolist()
        for e, wgt in zip(experts, weights):
            ug = dequant(t["w_ug"][e], t["s_ug"][e], 128) @ xq[s]
            mid = F.silu(ug[:inter]) * ug[inter:]
            mids.append(mid)
            mid = quant_dequant(mid) if cfg.act_fp8 else bf(mid)
            y[s] += wgt * (dequant(t["w_dn"][e], t["s_dn"][e], 128) @ mid)
        sel.append(torch.tensor(experts, device=a.device, dtype=torch.int32))
        prob.append(torch.tensor(weights, device=a.device))
    x_out = (a.float() + allreduce(y)).to(torch.bfloat16)
    return dict(
        x2=x2,
        xq=xq_ref,
        logits=logits,
        shared_gate=shared_gate,
        sel=torch.stack(sel),
        prob=torch.stack(prob),
        mid=torch.stack(mids).view(S, cfg.top_k + 1, inter),
        x_out=x_out,
    )


def golden_layer(W: LayerWeights, h: torch.Tensor, allreduce, *, gdn_state=None, kv=None):
    """One decoder layer on the residual ``h`` [S, hidden] (bf16).

    ``gdn_state`` = (conv_state, ssm_state, slots_in, slots_out) for a GDN layer; ``kv`` =
    (k_cache, v_cache, positions) for a full-attention layer. ``allreduce`` sums a partial over ranks.
    """
    xn = bf(gemma_rmsnorm(h, W.t["g_in"], W.cfg.eps))
    mix = golden_full_attn(W, xn, *kv) if W.full else golden_gdn(W, xn, *gdn_state)
    a = (h.float() + allreduce(mix["partial"])).to(torch.bfloat16)
    moe = golden_moe(W, a, allreduce)
    return {**mix, **moe, "xn": xn, "a": a}


def rope_tables(cfg: Q35Config, max_pos: int, device="cuda"):
    """The kernel's RoPE tables: cos, sin f32 [max_pos, rope_dim / 2] (the rotate-half halves are equal)."""
    cos, sin = rope_cos_sin(torch.arange(max_pos, device=device), cfg)
    half = cfg.rope_dim // 2
    return cos[:, :half].contiguous(), sin[:, :half].contiguous()


def kv_encode(x: torch.Tensor):
    """Per-token-per-head FP8 E4M3 of ``x`` [..., head_dim] -> (codes, f32 scales [...]), as ``kv_quant_dequant``."""
    amax = x.float().abs().amax(-1)
    s = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    return (x.float() / s[..., None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn), s


def paged_kv_write(k_cache, v_cache, k_scale, v_scale, block_table, pos: int, k, v):
    """Store one token's K / V ([kv_heads, head_dim], any float) at ``pos`` of the sequence whose blocks
    are ``block_table`` (ATOM's MHA pool: block size 16; K [blocks, kvh, hd / 16, 16, 16], V [blocks, kvh,
    1, hd, 16], scales [blocks, kvh, 16])."""
    blk, t = int(block_table[pos // 16]), pos % 16
    kq, ks = kv_encode(k)
    vq, vs = kv_encode(v)
    kvh, hd = kq.shape
    k_cache[blk, :, :, t, :] = kq.view(kvh, hd // 16, 16).to(k_cache.dtype)
    v_cache[blk, :, 0, :, t] = vq.to(v_cache.dtype)
    k_scale[blk, :, t] = ks
    v_scale[blk, :, t] = vs


def paged_kv_read(k_cache, v_cache, k_scale, v_scale, block_table, pos: int):
    """The dequantized K / V [kv_heads, head_dim] stored at ``pos`` (see ``paged_kv_write``)."""
    blk, t = int(block_table[pos // 16]), pos % 16
    kvh = k_cache.shape[1]
    k = k_cache[blk, :, :, t, :].float().reshape(kvh, -1) * k_scale[blk, :, t, None]
    v = v_cache[blk, :, 0, :, t].float() * v_scale[blk, :, t, None]
    return k, v
