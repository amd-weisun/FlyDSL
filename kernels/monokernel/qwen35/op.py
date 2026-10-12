# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: packed weights, scratch, symmetric peer buffers and the launch of one rank's
Qwen3.5 decoder layer (Gated DeltaNet or gated full attention)."""

from __future__ import annotations

import torch

from kernels.monokernel.dsv4.kernel import build_advance_step
from kernels.monokernel.packing import pack_bf16, pack_fp8
from kernels.monokernel.qwen35.kernel import build_qwen35_kernel
from kernels.monokernel.qwen35.kernel.plan import TL_COLS, layout, n_split, router_rows, stage_tasks
from kernels.monokernel.qwen35.reference import LayerWeights
from kernels.monokernel.runtime import SymmetricPeerBuffer

__all__ = ["Qwen35MonoKernel", "pack_layer_weights"]


def shape_dims(cfg) -> dict:
    """The shape arguments layout(), stage_tasks() and the builder must share identically."""
    return dict(
        hidden=cfg.hidden,
        hk=cfg.hk,
        hv=cfg.hv,
        k_dim=cfg.linear_k_dim,
        v_dim=cfg.linear_v_dim,
        heads=cfg.local_heads,
        head_dim=cfg.head_dim,
        n_experts=cfg.n_experts,
        top_k=cfg.top_k,
        inter=cfg.local_inter,
    )


def pack_layer_weights(W: LayerWeights) -> dict:
    """The kernel's layouts of a layer's weights (FP8 / BF16 MFMA tiles, f32 Gemma gammas ``1 + w``).
    The full-attention layer's q | gate, k, v matrix takes the GDN qkvz slot, its o_proj the out slot."""
    cfg, t = W.cfg, W.t
    rr = router_rows(cfg.n_experts)
    w_r = torch.zeros(rr, cfg.hidden, dtype=torch.bfloat16, device=t["w_r"].device)
    w_r[: cfg.n_experts] = t["w_r"]
    w_r[cfg.n_experts] = t["w_sg"][0]
    if W.full:
        mixer = dict(
            w_qkvz=pack_fp8(torch.cat([t["w_q"], t["w_k"], t["w_v"]])),
            s_qkvz=torch.cat([t["s_q"], t["s_k"], t["s_v"]]).float().contiguous(),
            g_qn1=(1.0 + t["g_qn"].float()).contiguous(),
            g_kn1=(1.0 + t["g_kn"].float()).contiguous(),
            w_out=pack_fp8(t["w_o"]),
            s_out=t["s_o"].float().contiguous(),
        )
    else:
        mixer = dict(
            w_qkvz=pack_fp8(torch.cat([t["w_qkv"], t["w_z"]])),
            s_qkvz=torch.cat([t["s_qkv"], t["s_z"]]).float().contiguous(),
            w_ba=pack_bf16(t["w_ba"]),
            w_conv=t["w_conv"].to(torch.bfloat16).contiguous(),
            a_log=t["a_log"].float().contiguous(),
            dt_bias=t["dt_bias"].float().contiguous(),
            g_norm=t["g_norm"].to(torch.bfloat16).float().contiguous(),  # the bf16 weight, as the model applies it
            w_out=pack_fp8(t["w_out"]),
            s_out=t["s_out"].float().contiguous(),
        )
    return dict(
        mixer,
        g_in1=(1.0 + t["g_in"].float()).contiguous(),
        g_post1=(1.0 + t["g_post"].float()).contiguous(),
        w_r=pack_bf16(w_r),
        w_ug=pack_fp8(t["w_ug"]),
        s_ug=t["s_ug"].float().contiguous(),
        w_dn=pack_fp8(t["w_dn"]),
        s_dn=t["s_dn"].float().contiguous(),
    )


class Qwen35MonoKernel:
    """One rank of one decoder layer: its packed weights, scratch and the compiled launch.

    State belongs to the caller (ATOM's pools): a GDN layer's conv / recurrent state with per-sample
    ``slots_in`` / ``slots_out``; a full-attention layer's paged FP8 K / V cache, its scales, block
    tables and positions, and the RoPE tables."""

    _cache: dict = {}
    _adv_cache: dict = {}

    def __init__(self, W: LayerWeights, samples: int, rank: int = 0, npes: int = 1, group=None, timeline=False):
        cfg = W.cfg
        if W.full and cfg.local_kv_heads != 1:
            raise ValueError("the full-attention kernel holds one K / V head per rank")
        self.W, self.S, self.rank, self.npes, self.full = W, samples, rank, npes, W.full
        self.packed = pack_layer_weights(W)
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(samples, npes, W.full, **dims)
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.sym, self.peers = self.peer_buffer.local_address, self.peer_buffer.addresses
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)
        key = (samples, npes, W.full, timeline, cfg.eps, cfg.rope_dim, tuple(sorted(dims.items())))
        if key not in Qwen35MonoKernel._cache:
            Qwen35MonoKernel._cache[key] = build_qwen35_kernel(
                samples, npes, W.full, eps=cfg.eps, rope_dim=cfg.rope_dim, timeline=timeline, **dims
            )
        self.launch = Qwen35MonoKernel._cache[key]
        # the step advance clears mailbox pairs; the plain-f32 "xqd" region sits last and is skipped
        adv_key = (self.scr_layout["xqd"] // 8, self.sym_layout["_bytes"] // 8)
        if adv_key not in Qwen35MonoKernel._adv_cache:
            Qwen35MonoKernel._adv_cache[adv_key] = build_advance_step(*adv_key)
        self._advance = Qwen35MonoKernel._adv_cache[adv_key]
        self.stages = stage_tasks(samples, W.full, **dims)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None

    def forward(self, h, *, gdn_state=None, kv=None, rope=None, x_out=None, layer=0, advance=True):
        """One layer on the residual ``h`` [S, hidden] bf16; updates the state in place.

        GDN: ``gdn_state = (conv_state, ssm_state, slots_in, slots_out)``. Full attention: ``kv = (k_cache,
        v_cache, k_scale, v_scale, positions, block_tables)`` in ATOM's MHA pool layout and ``rope = (cos,
        sin)``, f32 ``[positions, rope_dim / 2]``."""
        cfg, S = self.W.cfg, self.S
        if tuple(h.shape) != (S, cfg.hidden) or h.dtype != torch.bfloat16 or not h.is_contiguous():
            raise ValueError(f"h must be contiguous bfloat16 {(S, cfg.hidden)}, got {tuple(h.shape)} {h.dtype}")

        def i32(name, t, shape):
            if tuple(t.shape[: len(shape)]) != shape or t.dtype != torch.int32 or t.stride(-1) != 1:
                raise ValueError(f"{name} must be int32 {shape} with unit-stride rows")

        z = torch.zeros(1, dtype=torch.int32, device=h.device)
        conv = ssm = sl_in = sl_out = z
        kc = vc = ks = vs = pos = bt = cos = sin = z
        bt_stride = 0
        if self.full:
            kc, vc, ks, vs, pos, bt = kv
            cos, sin = rope
            i32("positions", pos, (S,))
            i32("block_tables", bt, (S,))
            if kc.dtype not in (torch.float8_e4m3fn, torch.uint8) or kc.shape[1:] != (1, cfg.head_dim // 16, 16, 16):
                raise ValueError("k_cache must be FP8 [blocks, 1, head_dim / 16, 16, 16] (ATOM's MHA pool)")
            if ks.dtype != torch.float32 or vs.dtype != torch.float32:
                raise ValueError("k_scale / v_scale must be float32 [blocks, 1, 16]")
            bt_stride = bt.stride(0)
        else:
            conv, ssm, sl_in, sl_out = gdn_state
            sl_out = sl_in if sl_out is None else sl_out
            if conv.dtype != torch.bfloat16 or tuple(conv.shape[1:]) != (cfg.conv_kernel - 1, cfg.conv_dim):
                raise ValueError(f"conv_state must be bf16 [slots, {cfg.conv_kernel - 1}, {cfg.conv_dim}]")
            if ssm.dtype != torch.bfloat16 or tuple(ssm.shape[1:]) != (cfg.hv, cfg.linear_v_dim, cfg.linear_k_dim):
                raise ValueError(f"ssm_state must be bf16 [slots, {cfg.hv}, {cfg.linear_v_dim}, {cfg.linear_k_dim}]")
            i32("slots_in", sl_in, (S,))
            i32("slots_out", sl_out, (S,))
        if x_out is None:
            x_out = torch.empty_like(h)
        p = self.packed
        ptr = lambda x: x.data_ptr()  # noqa: E731
        ptr0 = lambda k: p[k].data_ptr() if k in p else 0  # noqa: E731
        self.launch(
            ptr(h),
            ptr(x_out),
            ptr(sl_in),
            ptr(sl_out),
            ptr(conv),
            ptr(ssm),
            ptr(pos),
            ptr(bt),
            ptr(kc),
            ptr(vc),
            ptr(ks),
            ptr(vs),
            ptr(cos),
            ptr(sin),
            ptr0("g_qn1"),
            ptr0("g_kn1"),
            ptr(p["g_in1"]),
            ptr(p["w_qkvz"]),
            ptr(p["s_qkvz"]),
            ptr0("w_ba"),
            ptr0("w_conv"),
            ptr0("a_log"),
            ptr0("dt_bias"),
            ptr0("g_norm"),
            ptr(p["w_out"]),
            ptr(p["s_out"]),
            ptr(p["g_post1"]),
            ptr(p["w_r"]),
            ptr(p["w_ug"]),
            ptr(p["s_ug"]),
            ptr(p["w_dn"]),
            ptr(p["s_dn"]),
            ptr(self.scratch),
            self.sym,
            ptr(self.peers),
            0 if self.timeline is None else ptr(self.timeline),
            ptr(self.step),
            self.rank,
            layer,
            bt_stride,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self):
        """``step += 1`` and clear a slice of stale mailbox pairs; graph-capturable."""
        self._advance(self.step.data_ptr(), self.scratch.data_ptr(), self.sym, stream=torch.cuda.current_stream())

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
        slots = cfg.top_k + 1
        if self.full:
            mixer = dict(
                q=self.debug("q", (S, cfg.local_heads, cfg.head_dim)),
                gated=self.debug("gated", (S, cfg.local_heads * cfg.head_dim), bf2=True),
                split_ml=self.debug("sp_ml", (S, n_split(S), cfg.local_heads, 2)),
            )
        else:
            mixer = dict(
                conv=self.debug("conv", (S, cfg.conv_dim)),
                z=self.debug("z", (S, cfg.value_dim)),
                ba=self.debug("ba", (S, 2 * cfg.hv)),
                gated=self.debug("gated", (S, cfg.value_dim), bf2=True),
            )
        return dict(
            mixer,
            a=self.debug("a", (S, cfg.hidden), bf2=True),
            scores=self.debug("scores", (S, router_rows(cfg.n_experts))),
            sel=self.debug("sel", (S, slots), torch.int32),
            prob=self.debug("prob", (S, slots)),
            mid=self.debug("mid", (S, slots, cfg.local_inter)),
            xq=self.scratch[self.scr_layout["xqd"] :].view(torch.float32)[: S * cfg.hidden].view(S, cfg.hidden),
        )

    def timeline_report(self) -> str:
        """Per stage, in us from launch start: span and median per-task phases (``timeline=True``)."""
        if self.timeline is None:
            raise RuntimeError("timeline collection was not enabled")
        tl = self.timeline[:, :5].cpu().double() / 100.0  # s_memrealtime ticks at 100 MHz
        live = tl[:, 0] > 0
        t0 = tl[live, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n].clone()
            i += n
            st = st[st[:, 0] > 0]
            if not len(st):
                continue
            for c in (1, 2, 3):  # missing marks inherit the previous one
                st[:, c] = torch.where(st[:, c] > 0, st[:, c], st[:, c - 1])
            d = (st[:, 1:] - st[:, :-1]).median(0).values
            rows.append(
                f"{name:8s} x{n:4d}  [{(st[:, 0].min() - t0):6.1f} .. end {(st[:, 4].max() - t0):6.1f}]  "
                f"hint {d[0]:5.1f}  stage {d[1]:5.1f}  compute {d[2]:5.1f}  epi {d[3]:5.1f}"
            )
        return "\n".join(rows)

    def close(self):
        self.peer_buffer.close()
