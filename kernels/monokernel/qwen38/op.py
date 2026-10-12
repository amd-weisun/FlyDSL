# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host side of the Qwen3.8-Flash-Next decode MonoKernel (TP1): weights in ATOM's layouts, the shared launch
state per sample count, and one layer's launch."""

from __future__ import annotations

import torch

from kernels.monokernel.dsv4.kernel import build_advance_step
from kernels.monokernel.packing import shuffle_fp8
from kernels.monokernel.qwen38.kernel import build_qwen38_kernel
from kernels.monokernel.qwen38.kernel.build import ARGS
from kernels.monokernel.qwen38.kernel.plan import TL_COLS, layout, router_rows, stage_tasks
from kernels.monokernel.qwen38.reference import LayerWeights, rope_cos_sin

__all__ = ["Qwen38MonoKernel", "Qwen38Variant", "pack_layer_weights"]


def shape_dims(cfg, max_seq: int = 4096) -> dict:
    """The shape arguments layout(), stage_tasks() and the builder must share identically; ``max_seq`` sizes
    the indexer's score mailbox (the longest context a launch serves)."""
    return dict(
        max_seq=max_seq,
        kv_heads=cfg.kv_heads,
        rope_dim=cfg.rope_dim,
        index_heads=cfg.index_heads,
        index_dim=cfg.index_dim,
        index_budget=cfg.index_budget,
        index_ratio=cfg.index_ratio,
        hidden=cfg.hidden,
        hc=cfg.hc,
        hc_lowrank=cfg.hc_lowrank,
        hk=cfg.linear_k_heads,
        hv=cfg.linear_v_heads,
        k_dim=cfg.linear_k_dim,
        v_dim=cfg.linear_v_dim,
        heads=cfg.heads,
        head_dim=cfg.head_dim,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        inter=cfg.inter,
        shared_inter=cfg.shared_inter,
        eps=cfg.eps,
    )


def pack_layer_weights(W: LayerWeights) -> dict:
    """The kernel's view of a layer's weights, in ATOM's layouts: BF16 matrices row-major (GDN [q|k|v|z|b|a] as
    one matrix, the router with the shared-expert gate as row n_experts), routed experts' FP8 in aiter's
    ``shuffle_weight`` (16, 16) with f32 block scales; f32 Gemma gammas ``1 + w`` of the HC norms."""
    cfg, t = W.cfg, W.t
    f32 = lambda x: x.float().contiguous()  # noqa: E731
    p = {}
    for hc in ("hca", "hcm"):
        p[f"{hc}_g"] = (1.0 + t[f"{hc}_norm"].float()).contiguous()
        for k in ("down", "up", "inj"):
            p[f"{hc}_{k}"] = t[f"{hc}_{k}"].contiguous()
    if W.full:
        H, hd = cfg.heads, cfg.head_dim
        wq = t["w_q"].view(H, 2, hd, -1)  # per head [q | gate] in the checkpoint
        p["w_qkv"] = torch.cat(
            [wq[:, 1].reshape(H * hd, -1), wq[:, 0].reshape(H * hd, -1), t["w_k"], t["w_v"]]
        ).contiguous()  # ATOM's qkv_proj: every head's gate, then every head's q
        p["w_idx"] = t["w_idx"].contiguous()
        p["w_out"] = t["w_o"].contiguous()
        for k in ("g_qn", "g_kn", "g_iq", "g_ik"):
            p[k + "1"] = (1.0 + t[k].float()).contiguous()
        p.update(_moe(t))
        return p
    p["w_proj"] = torch.cat([t["w_qkv"], t["w_z"], t["w_ba"]]).contiguous()
    p["w_conv"] = t["w_conv"].to(torch.bfloat16).contiguous()
    p["a_log"] = f32(t["a_log"])
    p["dt_bias"] = f32(t["dt_bias"])
    p["g_norm"] = f32(t["g_norm"])
    p["w_out"] = t["w_out"].contiguous()
    p.update(_moe(t))
    return p


def _moe(t) -> dict:
    return dict(
        w_r=torch.cat([t["w_r"], t["w_sg"]]).contiguous(),
        w_ug=shuffle_fp8(t["w_ug"]),
        s_ug=t["s_ug"].float().contiguous(),
        w_dn=shuffle_fp8(t["w_dn"]),
        s_dn=t["s_dn"].float().contiguous(),
        w_sh_ug=t["w_sh_ug"].contiguous(),
        w_sh_dn=t["w_sh_dn"].contiguous(),
    )


def rope_tables(cfg, max_pos: int, device="cuda"):
    """The kernel's RoPE tables: cos, sin f32 [max_pos, rope_dim / 2] of bf16-rounded values (ATOM's)."""
    cos, sin = rope_cos_sin(torch.arange(max_pos, device=device), cfg)
    half = cfg.rope_dim // 2
    return cos[:, :half].contiguous(), sin[:, :half].contiguous()


class Qwen38Variant:
    """The compiled launch, scratch and step counter that every layer of one kind shares at one sample count.
    Fresh epoch tags (``step * LAYER_SLOTS + layer + 1``) keep layers apart in the shared scratch: one
    ``advance_step`` per decode step, distinct ``layer`` values per launch."""

    _cache: dict = {}
    _adv_cache: dict = {}

    def __init__(self, cfg, samples: int, full: bool, timeline=False, max_seq: int = 4096):
        self.cfg, self.S, self.full = cfg, samples, full
        dims = shape_dims(cfg, max_seq)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout = layout(samples, full, **dims)
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        self.key = (samples, full, timeline, tuple(sorted(dims.items())))
        if self.key not in Qwen38Variant._cache:
            Qwen38Variant._cache[self.key] = build_qwen38_kernel(samples, full, timeline=timeline, **dims)
        self.launch = Qwen38Variant._cache[self.key]
        adv_key = self.scr_layout["xqd"] // 8
        if adv_key not in Qwen38Variant._adv_cache:
            Qwen38Variant._adv_cache[adv_key] = build_advance_step(adv_key, 0)
        self._advance = Qwen38Variant._adv_cache[adv_key]
        self.stages = stage_tasks(samples, full, **dims)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs; graph-capturable."""
        self._advance(self.step.data_ptr(), self.scratch.data_ptr(), 0, stream=torch.cuda.current_stream())


class Qwen38MonoKernel:
    """One decoder layer: its weights (``packed``, ATOM's tensors in serving) on a ``Qwen38Variant``. State belongs
    to the caller: a GDN layer's conv / fp32 recurrent state with per-sample ``slots_in`` / ``slots_out``."""

    def __init__(self, W: LayerWeights, samples: int, timeline=False, variant=None, packed=None, max_seq=4096):
        self.W, self.S, self.full = W, samples, W.full
        self.packed = pack_layer_weights(W) if packed is None else packed
        if variant is None:
            variant = Qwen38Variant(W.cfg, samples, W.full, timeline, max_seq)
        elif (variant.S, variant.full) != (samples, W.full):
            raise ValueError("the variant was built for another sample count or layer kind")
        self.variant = variant
        self.scr_layout, self.scratch = variant.scr_layout, variant.scratch
        self.stages, self.timeline = variant.stages, variant.timeline

    def forward(self, X, *, gdn_state=None, kv=None, rope=None, x_out=None, layer=0, advance=True):
        """One layer on the bundle ``X`` [S, hc * hidden] bf16 -> the next bundle; updates the state in place.

        GDN: ``gdn_state = (conv_state [slots, 3, conv_dim] bf16, ssm_state [slots, hv, k, v] fp32, slots_in,
        slots_out)``. QSA: ``kv = (k_cache, v_cache [blocks, 16, kv_heads, head_dim] bf16, index_cache
        [blocks, 16, 1, index_dim] bf16, compressed_cache [blocks, 4, 1, index_dim] bf16, positions, block_tables,
        slot_mapping)`` (ATOM's QSA pool) and ``rope`` = ``rope_tables(...)``."""
        cfg, S = self.W.cfg, self.S
        if tuple(X.shape) != (S, cfg.streams) or X.dtype != torch.bfloat16 or not X.is_contiguous():
            raise ValueError(f"X must be contiguous bfloat16 {(S, cfg.streams)}")

        def i32(name, t, shape):
            if tuple(t.shape[: len(shape)]) != shape or t.dtype != torch.int32 or t.stride(-1) != 1:
                raise ValueError(f"{name} must be int32 {shape} with unit-stride rows")

        if x_out is None:
            x_out = torch.empty_like(X)
        a = dict(h_in=X, x_out=x_out)
        if self.full:
            kc, vc, ic, cc, pos, bt, slots = kv
            for name, t, shp in (
                ("k_cache", kc, (16, cfg.kv_heads, cfg.head_dim)),
                ("index_cache", ic, (16, 1, cfg.index_dim)),
                ("compressed_cache", cc, (16 // cfg.index_ratio, 1, cfg.index_dim)),
            ):
                if t.dtype != torch.bfloat16 or tuple(t.shape[1:]) != shp or not t.is_contiguous():
                    raise ValueError(f"{name} must be contiguous bf16 [blocks, {shp}] (ATOM's QSA pool)")
            i32("positions", pos, (S,))
            i32("slot_mapping", slots, (S,))
            i32("block_tables", bt, (S,))
            a.update(
                positions=pos,
                block_tables=bt,
                slot_mapping=slots,
                k_cache=kc,
                v_cache=vc,
                i_cache=ic,
                c_cache=cc,
                rope_cos=rope[0],
                rope_sin=rope[1],
                bt_stride=bt.stride(0),
            )
        else:
            conv, ssm, sl_in, sl_out = gdn_state
            sl_out = sl_in if sl_out is None else sl_out
            if conv.dtype != torch.bfloat16 or tuple(conv.shape[1:]) != (cfg.conv_kernel - 1, cfg.conv_dim):
                raise ValueError(f"conv_state must be bf16 [slots, {cfg.conv_kernel - 1}, {cfg.conv_dim}]")
            if ssm.dtype != torch.float32 or tuple(ssm.shape[1:]) != (
                cfg.linear_v_heads,
                cfg.linear_k_dim,
                cfg.linear_v_dim,
            ):
                raise ValueError("ssm_state must be fp32 [slots, hv, k_dim, v_dim] (K-major, as ATOM)")
            i32("slots_in", sl_in, (S,))
            i32("slots_out", sl_out, (S,))
            a.update(slots_in=sl_in, slots_out=sl_out, conv_state=conv, ssm_state=ssm)
        v = self.variant
        a.update(self.packed)
        a.update(scratch=v.scratch, timeline_buf=v.timeline, step=v.step, layer=layer)
        args = []
        for name in ARGS:
            x = a.get(name)
            if name in ("layer", "bt_stride"):
                args.append(int(x or 0))
            else:
                args.append(0 if x is None else x.data_ptr())
        v.launch(*args, stream=torch.cuda.current_stream())
        if advance:
            v.advance_step()
        return x_out

    def debug(self, name: str, shape, dtype=torch.float32, bf2=False) -> torch.Tensor:
        """A scratch mailbox's values (``bf2``: two bf16 per value word)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if bf2:
            words = self.scratch[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
            return words.view(torch.bfloat16).float().view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
        return words.view(dtype).view(shape)

    def intermediates(self):
        cfg, S = self.W.cfg, self.S
        return dict(
            m=self.debug("m_a", (S, cfg.hidden), bf2=True),
            r=self.debug("hr_a", (S, cfg.hc)),
            **(
                dict(
                    q=self.debug("q", (S, cfg.heads, cfg.head_dim)),
                    iq=self.debug("iq", (S, cfg.index_heads, cfg.index_dim)),
                    gsel=self.debug("gsel", (S, cfg.block_topk), torch.int32),
                    gcnt=self.debug("gcnt", (S,), torch.int32),
                    gated=self.debug("gated", (S, cfg.heads * cfg.head_dim), bf2=True),
                )
                if self.full
                else dict(
                    conv=self.debug("conv", (S, cfg.conv_dim)),
                    z=self.debug("z", (S, cfg.value_dim)),
                    gated=self.debug("gated", (S, cfg.value_dim), bf2=True),
                )
            ),
            x1=self.debug("x1", (S, cfg.streams), bf2=True),
            m2=self.debug("m_m", (S, cfg.hidden), bf2=True),
            r2=self.debug("hr_m", (S, cfg.hc)),
            scores=self.debug("scores", (S, router_rows(cfg.n_experts))),
            sel=self.debug("sel", (S, cfg.top_k), torch.int32),
            prob=self.debug("prob", (S, cfg.top_k)),
            mid=self.debug("mid", (S, cfg.top_k, cfg.inter)),
            mid_sh=self.debug("mid_sh", (S, cfg.shared_inter), bf2=True),
            xq=self.scratch[self.scr_layout["xqd"] :].view(torch.float32)[: S * cfg.hidden].view(S, cfg.hidden),
        )

    def timeline_report(self) -> str:
        tl = self.timeline[:, :5].cpu().double() / 100.0
        live = tl[:, 0] > 0
        t0 = tl[live, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n].clone()
            i += n
            st = st[st[:, 0] > 0]
            if not len(st):
                continue
            for c in (1, 2, 3):
                st[:, c] = torch.where(st[:, c] > 0, st[:, c], st[:, c - 1])
            d = (st[:, 1:] - st[:, :-1]).median(0).values
            rows.append(
                f"{name:8s} x{n:4d}  [{(st[:, 0].min() - t0):6.1f} .. end {(st[:, 4].max() - t0):6.1f}]  "
                f"hint {d[0]:5.1f}  stage {d[1]:5.1f}  compute {d[2]:5.1f}  epi {d[3]:5.1f}"
            )
        return "\n".join(rows)
