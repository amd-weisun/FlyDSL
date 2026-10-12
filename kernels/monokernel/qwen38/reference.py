# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Weights, state layouts and the torch golden for one Qwen3.8-Flash-Next decoder layer (TP1).

Math follows HF transformers ``modeling_qwen4_exp`` (checked by the reference test); rounding points follow
baseline ATOM's kernels. A layer maps the hyper-connection bundle ``X`` [S, 4 * hidden] (bf16) to the next one:
mix -> token mixer -> combine -> mix -> MoE -> combine. PLE (layers.1 only) is not part of a layer here.
State layouts are ATOM's native ones:

- GDN conv state ``[slots, conv_kernel - 1, conv_dim]`` bf16, oldest row first;
- GDN recurrent state ``[slots, hv, k_dim, v_dim]`` **fp32** (``mamba_ssm_dtype``), K-major in memory;
- QSA caches as contiguous per-sample histories: K / V ``[samples, max_seq, kv_heads, head_dim]``, the raw
  index keys ``[samples, max_seq, index_dim]`` and the compressed keys ``[samples, max_seq // 4, index_dim]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kernels.monokernel.qwen35.reference import act_quant, route
from kernels.monokernel.qwen38.config import Q38Config
from kernels.monokernel.reference import _rand_fp8, bf, dequant


@dataclass
class LayerWeights:
    cfg: Q38Config
    full: bool  # QSA; else Gated DeltaNet
    t: dict  # name -> tensor


def _bf16_mat(rows, k, gen, device, scale=1.0):
    return (scale * torch.randn(rows, k, generator=gen, device=device) / k**0.5).to(torch.bfloat16)


def make_weights(cfg: Q38Config, full: bool, device="cuda", seed: int = 1234) -> LayerWeights:
    """Random weights at checkpoint formats: every matrix BF16 except the routed experts (FP8 128 x 128)."""
    cfg.validate()
    g = torch.Generator(device=device).manual_seed(seed)
    bfl = torch.bfloat16
    t = {}
    H, W4, R = cfg.hidden, cfg.streams, cfg.hc_lowrank
    for hc in ("hca", "hcm"):  # the attention-side and MLP-side hyper-connections
        t[f"{hc}_norm"] = (0.1 * torch.randn(W4, generator=g, device=device)).to(bfl)
        t[f"{hc}_down"] = _bf16_mat(R, W4, g, device)
        t[f"{hc}_up"] = _bf16_mat(W4, R, g, device, 2.0)
        t[f"{hc}_inj"] = _bf16_mat(cfg.hc, W4, g, device, 2.0)
    if full:
        hd = cfg.head_dim
        t["w_q"] = _bf16_mat(cfg.heads * 2 * hd, H, g, device)  # per head [q | gate], as the checkpoint
        t["w_k"] = _bf16_mat(cfg.kv_heads * hd, H, g, device)
        t["w_v"] = _bf16_mat(cfg.kv_heads * hd, H, g, device)
        t["w_o"] = _bf16_mat(H, cfg.heads * hd, g, device)
        t["g_qn"] = (0.1 * torch.randn(hd, generator=g, device=device)).to(bfl)
        t["g_kn"] = (0.1 * torch.randn(hd, generator=g, device=device)).to(bfl)
        t["w_idx"] = _bf16_mat((cfg.index_heads + 1) * cfg.index_dim, H, g, device, 4.0)  # [q heads | key]
        t["g_iq"] = (0.1 * torch.randn(cfg.index_dim, generator=g, device=device)).to(bfl)
        t["g_ik"] = (0.1 * torch.randn(cfg.index_dim, generator=g, device=device)).to(bfl)
    else:
        hv = cfg.linear_v_heads
        t["w_qkv"] = _bf16_mat(cfg.conv_dim, H, g, device)  # [q | k | v]
        t["w_z"] = _bf16_mat(cfg.value_dim, H, g, device)
        t["w_ba"] = _bf16_mat(2 * hv, H, g, device)  # in_proj_b rows then in_proj_a rows
        t["w_conv"] = (0.5 * torch.randn(cfg.conv_dim, cfg.conv_kernel, generator=g, device=device)).to(bfl)
        t["a_log"] = torch.log(torch.rand(hv, generator=g, device=device) * 15.99 + 0.01).to(bfl)
        t["dt_bias"] = torch.randn(hv, generator=g, device=device).to(bfl)
        t["g_norm"] = (1 + 0.1 * torch.randn(cfg.linear_v_dim, generator=g, device=device)).to(bfl)  # plain w
        t["w_out"] = _bf16_mat(H, cfg.value_dim, g, device)
    t["w_r"] = _bf16_mat(cfg.n_experts, H, g, device, 4.0)
    t["w_sg"] = _bf16_mat(1, H, g, device)
    t["w_ug"], t["s_ug"] = _rand_fp8(2 * cfg.inter, H, 128, g, device, lead=(cfg.n_experts,))  # [gate | up]
    t["w_dn"], t["s_dn"] = _rand_fp8(H, cfg.inter, 128, g, device, lead=(cfg.n_experts,))
    t["w_sh_ug"] = _bf16_mat(2 * cfg.shared_inter, H, g, device)  # shared expert [gate | up], BF16
    t["w_sh_dn"] = _bf16_mat(H, cfg.shared_inter, g, device)
    return LayerWeights(cfg, full, t)


def rmsnorm1p(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """``x * rsqrt(mean(x^2) + eps) * (1 + w)`` over the last dim, fp32 (Qwen4ExpTextRMSNorm)."""
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1.0 + w.float())


def mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """bf16 activation ``x`` [..., K] times BF16 matrix ``w`` [N, K], fp32 out."""
    return bf(x) @ w.float().T


def rope_cos_sin(positions: torch.Tensor, cfg: Q38Config):
    """cos / sin [S, rope_dim] (halves repeated, rotate-half), rounded to bf16 as ATOM's tables."""
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.rope_dim, 2, device=positions.device).float() / cfg.rope_dim))
    f = positions.float()[:, None] * inv[None]
    f = torch.cat([f, f], -1)
    return bf(f.cos()), bf(f.sin())


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate-half RoPE on the first ``rope_dim`` dims of the bf16 ``x`` [S, heads, D]: ``x * cos`` and
    ``rot(x) * sin`` each rounded to bf16, then their sum (ATOM's rounding)."""
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    x1, x2 = xr[..., : r // 2], xr[..., r // 2 :]
    rot = torch.cat([-x2, x1], -1)
    return torch.cat([bf(bf(xr * cos[:, None]) + bf(rot * sin[:, None])), xp], -1)


def hc_mix(W: LayerWeights, hc: str, X: torch.Tensor):
    """Hyper-connection read of the bundle ``X`` [S, 4 * hidden]: (sublayer input [S, hidden], inject
    logits [S, 4]), with ATOM's rounding points (gate products summed in fp32, one rounding)."""
    cfg, t = W.cfg, W.t
    S = X.shape[0]
    xn = bf(rmsnorm1p(X.view(S, cfg.hc, cfg.hidden), t[f"{hc}_norm"].view(cfg.hc, cfg.hidden), cfg.eps))
    flat = xn.reshape(S, -1)
    d = bf(mm(flat, t[f"{hc}_down"]))
    u = bf(mm(bf(F.silu(bf(d / cfg.hc))), t[f"{hc}_up"])).view(S, cfg.hc, cfg.hidden)
    m = bf((torch.sigmoid(u) * xn).sum(1) / cfg.hc)
    r = bf(mm(flat, t[f"{hc}_inj"]))
    return m, r


def hc_combine(X: torch.Tensor, y: torch.Tensor, r: torch.Tensor, hc: int) -> torch.Tensor:
    """``X_s + y * 2 sigmoid(r_s / 4)`` per stream (fp32, one bf16 rounding)."""
    S = X.shape[0]
    inj = 2.0 * torch.sigmoid(bf(r / hc))  # [S, hc]
    out = X.float().view(S, hc, -1) + y.float()[:, None, :] * inj[:, :, None]
    return bf(out.view(S, -1))


def golden_gdn(W: LayerWeights, m, conv_state, ssm_state, slots_in, slots_out=None):
    """Gated DeltaNet on the sublayer input ``m`` [S, hidden]; updates both states in place. A negative
    slot skips the sample. Returns the stages and ``y`` (out_proj, bf16)."""
    cfg, t = W.cfg, W.t
    S, hv = m.shape[0], cfg.linear_v_heads
    slots_out = slots_in if slots_out is None else slots_out
    qkv = bf(mm(m, t["w_qkv"]))
    z = bf(mm(m, t["w_z"])).view(S, hv, cfg.linear_v_dim)
    ba = bf(mm(m, t["w_ba"]))
    b, a = ba[:, :hv], ba[:, hv:]
    beta = torch.sigmoid(b)
    g = -torch.exp(t["a_log"].float()) * F.softplus(a + t["dt_bias"].float())
    conv_out = torch.zeros_like(qkv)
    core = torch.zeros(S, hv, cfg.linear_v_dim, device=m.device)
    for s in range(S):
        si, so = int(slots_in[s]), int(slots_out[s])
        if si < 0 or so < 0:
            continue
        window = torch.cat([conv_state[si].float(), qkv[s][None]], 0)
        conv_out[s] = bf(F.silu((window * t["w_conv"].float().T).sum(0)))
        conv_state[so] = window[1:].to(conv_state.dtype)
        q, k, v = conv_out[s].split([cfg.key_dim, cfg.key_dim, cfg.value_dim])
        q = q.view(cfg.linear_k_heads, -1).repeat_interleave(cfg.v_per_k, 0)
        k = k.view(cfg.linear_k_heads, -1).repeat_interleave(cfg.v_per_k, 0)
        v = v.view(hv, -1)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1.0e-6) * cfg.linear_k_dim**-0.5
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1.0e-6)
        h = ssm_state[si].float() * torch.exp(g[s])[:, None, None]  # [hv, K, V]
        delta = (v - torch.einsum("hkv,hk->hv", h, k)) * beta[s][:, None]
        h = h + k[:, :, None] * delta[:, None, :]
        core[s] = torch.einsum("hkv,hk->hv", h, q)
        ssm_state[so] = h.to(ssm_state.dtype)
    core = bf(core)
    # Qwen4ExpTextRMSNormGated: norm -> bf16, times the plain weight -> bf16, times sigmoid(z)
    nrm = bf(core * torch.rsqrt(core.square().mean(-1, keepdim=True) + cfg.eps))
    gated = bf(bf(nrm * t["g_norm"].float()) * torch.sigmoid(z))
    y = bf(mm(gated.flatten(1), t["w_out"]))
    return dict(qkv=qkv, z=z, beta=beta, g=g, conv_out=conv_out, core=core, gated=gated, y=y)


def golden_qsa(W: LayerWeights, m, k_cache, v_cache, idx_cache, cmp_cache, positions):
    """QSA on ``m`` [S, hidden]: sample ``s`` writes its K / V / raw index key at ``positions[s]`` of row ``s``
    of the caches (and the compressed key of the group it closes), selects ``block_topk`` groups by the
    indexer and attends to them plus the causal tail. Returns the stages and ``y`` (o_proj, bf16)."""
    cfg, t = W.cfg, W.t
    S, hd, H, KH = m.shape[0], cfg.head_dim, cfg.heads, cfg.kv_heads
    R, D = cfg.index_ratio, cfg.index_dim
    qg = bf(mm(m, t["w_q"])).view(S, H, 2, hd)
    q, gate = qg[:, :, 0], qg[:, :, 1]
    k = bf(mm(m, t["w_k"])).view(S, KH, hd)
    v = bf(mm(m, t["w_v"])).view(S, KH, hd)
    cos, sin = rope_cos_sin(positions, cfg)
    q = apply_rope(bf(rmsnorm1p(q, t["g_qn"], cfg.eps)), cos, sin)
    k = apply_rope(bf(rmsnorm1p(k, t["g_kn"], cfg.eps)), cos, sin)
    iqk = bf(mm(m, t["w_idx"]))
    iq = apply_rope(bf(rmsnorm1p(iqk[:, : cfg.index_heads * D].view(S, -1, D), t["g_iq"], cfg.eps)), cos, sin)
    raw = iqk[:, cfg.index_heads * D :]
    out = torch.zeros(S, H, hd, device=m.device)
    picked = []
    for s in range(S):
        p = int(positions[s])
        k_cache[s, p], v_cache[s, p], idx_cache[s, p] = k[s], v[s], raw[s]
        if (p + 1) % R == 0:  # this token closes group p // R: mean-pool, norm, RoPE at the group's first token
            G = p // R
            pk = bf(idx_cache[s, G * R : G * R + R].float().mean(0))
            c0, s0 = rope_cos_sin(torch.tensor([G * R], device=m.device), cfg)
            cmp_cache[s, G] = apply_rope(bf(rmsnorm1p(pk, t["g_ik"], cfg.eps))[None, None], c0, s0)[0, 0]
        n = (p + 1) // R
        scores = torch.relu(iq[s].float() @ cmp_cache[s, :n].float().T).sum(0) / D**0.5  # [n]
        order = torch.sort(scores, descending=True, stable=True).indices[: min(cfg.block_topk, n)]
        toks = torch.cat(
            [
                (order[:, None] * R + torch.arange(R, device=m.device)).flatten(),
                torch.arange(n * R, p + 1, device=m.device),
            ]
        )
        picked.append(order)
        keys, vals = k_cache[s, toks].float(), v_cache[s, toks].float()  # [T, KH, hd]
        kv_of = torch.arange(H, device=m.device) // (H // KH)
        logits = torch.einsum("hd,thd->ht", q[s], keys[:, kv_of]) * hd**-0.5
        out[s] = torch.einsum("ht,thd->hd", torch.softmax(logits, -1), vals[:, kv_of])
    gated = bf(bf(out) * bf(torch.sigmoid(gate)))
    y = bf(mm(gated.flatten(1), t["w_o"]))
    return dict(q=q, gate=gate, k=k, v=v, iq=iq, raw=raw, picked=picked, attn=bf(out), gated=gated, y=y)


def golden_moe(W: LayerWeights, m: torch.Tensor, xq: torch.Tensor | None = None):
    """MoE on the sublayer input ``m`` [S, hidden] (bf16): routed FP8 experts (W8A8 when ``act_fp8``) plus the
    sigmoid-gated BF16 shared expert. ``xq`` overrides the routed experts' quantized input."""
    cfg, t = W.cfg, W.t
    S, inter = m.shape[0], cfg.inter
    logits = bf(mm(m, t["w_r"]))
    sg = bf(mm(m, t["w_sg"]))[:, 0]
    xq_ref = act_quant(m) if cfg.act_fp8 else bf(m)
    xq = xq_ref if xq is None else xq.float()
    routed = torch.zeros(S, cfg.hidden, device=m.device)
    sel, prob, mids = [], [], []
    for s in range(S):
        idx, w = route(logits[s], cfg.top_k)
        for e, wgt in zip(idx.tolist(), w.tolist()):
            ug = dequant(t["w_ug"][e], t["s_ug"][e], 128) @ xq[s]
            mid = F.silu(ug[:inter]) * ug[inter:]
            mids.append(mid)
            mid = act_quant(mid) if cfg.act_fp8 else bf(mid)
            routed[s] += wgt * (dequant(t["w_dn"][e], t["s_dn"][e], 128) @ mid)
        sel.append(idx.to(torch.int32))
        prob.append(w)
    routed = bf(routed)
    gu = bf(mm(m, t["w_sh_ug"]))
    mid_sh = bf(F.silu(gu[:, : cfg.shared_inter]) * gu[:, cfg.shared_inter :])
    sh = bf(mm(mid_sh, t["w_sh_dn"]))
    y = bf(bf(sh * bf(torch.sigmoid(sg))[:, None]) + routed)
    return dict(
        logits=logits,
        shared_gate=sg,
        xq=xq_ref,
        sel=torch.stack(sel),
        prob=torch.stack(prob),
        mid=torch.stack(mids).view(S, cfg.top_k, inter),
        routed=routed,
        mid_sh=mid_sh,
        shared=sh,
        y_moe=y,
    )


def golden_layer(W: LayerWeights, X: torch.Tensor, *, gdn_state=None, qsa_state=None):
    """One decoder layer on the bundle ``X`` [S, 4 * hidden] (bf16) -> ``x_out``.

    ``gdn_state`` = (conv_state, ssm_state, slots_in, slots_out); ``qsa_state`` = (k_cache, v_cache, idx_cache,
    cmp_cache, positions)."""
    cfg = W.cfg
    m, r = hc_mix(W, "hca", X)
    mix = golden_qsa(W, m, *qsa_state) if W.full else golden_gdn(W, m, *gdn_state)
    X1 = hc_combine(X, mix["y"], r, cfg.hc)
    m2, r2 = hc_mix(W, "hcm", X1)
    moe = golden_moe(W, m2)
    x_out = hc_combine(X1, moe["y_moe"], r2, cfg.hc)
    return {**mix, **moe, "m": m, "r": r, "x1": X1, "m2": m2, "r2": r2, "x_out": x_out}
