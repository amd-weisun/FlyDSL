# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's V4 layer."""

from __future__ import annotations

import torch

from kernels.dsv4_moe_layer.config import (
    MAX_LAYERS_PER_STEP,
    ExpertActivation,
    MoeMode,
    as_moe_mode,
    moe_format,
    validate_shard,
)
from kernels.dsv4_moe_layer.dsv4_kernel import (
    TL_COLS,
    build_dsv4_kernel,
    layout,
    stage_tasks,
)
from kernels.dsv4_moe_layer.packing import pack_layer_weights
from kernels.dsv4_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer

__all__ = ["MoeMode", "Dsv4MoeLayer"]


class Dsv4MoeLayer:
    """One rank of the TP layer. ``group`` is a torch.distributed group (None for npes=1).

    The symmetric buffer is a torch allocation exported to every peer through
    HIP IPC; scratch and symmetric buffers may be shared by all
    layers because every launch uses a fresh ``tag``.
    """

    def __init__(
        self,
        W: LayerWeights,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
    ):
        cfg = W.cfg
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio)
        if cfg.hc_mult > 1 and cfg.hc_mult & (cfg.hc_mult - 1):
            raise ValueError(f"hc_mult must be 1 or a power of two, got {cfg.hc_mult}")
        self.moe_mode = as_moe_mode(moe_mode)
        self.W, self.S, self.rank, self.npes = W, samples, rank, npes
        self.window = cfg.window
        self.packed = pack_layer_weights(W.t, self.moe_mode)
        # [3 scales | hc_mix bases] per side, as one f32 vector the kernel indexes
        dev0 = torch.device("cuda", torch.cuda.current_device())
        self.hc_sb = {}
        for side in ("attn", "ffn"):
            if f"hc_{side}_scale" in W.t:
                self.hc_sb[side] = torch.cat(
                    [W.t[f"hc_{side}_scale"].float(), W.t[f"hc_{side}_base"].float()]
                ).contiguous()
            else:
                self.hc_sb[side] = torch.zeros(1, device=dev0)
        dims = dict(
            hidden=cfg.hidden,
            q_lora=cfg.q_lora,
            head_dim=cfg.head_dim,
            o_groups=cfg.o_groups,
            o_lora=cfg.o_lora,
        )
        # layout() and build_dsv4_kernel() MUST see identical shape arguments: the
        # host derives scratch offsets and its size from one and the kernel from the
        # other, so any drift both misreads every mailbox and undersizes the buffer.
        dims["hc_mult"] = cfg.hc_mult
        dims["compress_ratio"] = cfg.compress_ratio
        self.scr_layout, self.sym_layout = layout(samples, cfg.heads, npes, cfg.window, self.moe_mode, **dims)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses
        self.launch = build_dsv4_kernel(
            samples,
            cfg.heads,
            npes,
            cfg.window,
            scale=cfg.softmax_scale,
            timeline=timeline,
            moe_mode=self.moe_mode,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            inter=cfg.inter,
            swiglu_limit=cfg.swiglu_limit,
            hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
            hc_eps=cfg.hc_eps,
            window_rows=cfg.cache_rows,
            n_keys=cfg.n_keys,
            **dims,
        )
        self.stages = stage_tasks(samples, cfg.heads, window=cfg.window, top_k=cfg.top_k, inter=cfg.inter, **dims)
        # the compressor carries a rolling window across decode steps, so its state
        # lives here rather than being rebuilt per call
        if cfg.compress_ratio:
            shape = (cfg.c_rows, cfg.c_coff * cfg.head_dim)
            self.kv_state = torch.zeros(*shape, dtype=torch.float32, device=dev)
            # -inf, not zero: with overlapping windows (CSA) the previous window's
            # rows are unwritten before the first emit and must drop out of the
            # softmax. Harmless for the non-overlapping case, which fills them all.
            self.score_state = torch.full(shape, float("-inf"), dtype=torch.float32, device=dev)
        else:
            self.kv_state = self.score_state = torch.zeros(1, device=dev)
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter

    def debug(self, name: str, shape, dtype=torch.float32, pairs=True, bf2=False) -> torch.Tensor:
        """Values of a scratch mailbox (``(value, tag)`` pairs unless ``pairs=False``;
        ``bf2``: each pair's value word packs two bf16 elements)."""
        off = self.scr_layout[name]
        n = 1
        for d in shape:
            n *= d
        if not pairs:
            return self.scratch[off : off + n * 4].view(dtype).view(shape)
        if bf2:
            words = self.scratch[off : off + n * 4].view(torch.int32).view(n // 2, 2)[:, 0].contiguous()
            return words.view(torch.bfloat16).float().view(shape)
        words = self.scratch[off : off + n * 8].view(torch.int32).view(n, 2)[:, 0].contiguous()
        return words.view(dtype).view(shape)

    def forward(self, h, cur_pos, kv_cache, indices, cos, sin, x_out=None, layer=0, advance=True):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < MAX_LAYERS_PER_STEP:
            raise ValueError(f"layer must be in [0, {MAX_LAYERS_PER_STEP}), got {layer}")
        t = dict(self.W.t, **self.packed)
        if x_out is None:
            shape = (self.S, self.W.cfg.hidden)
            if self.W.cfg.hc_mult > 1:  # the layer preserves the whole residual stream
                shape = (self.S, self.W.cfg.hc_mult, self.W.cfg.hidden)
            x_out = torch.empty(*shape, dtype=torch.bfloat16, device=h.device)
        p = lambda x: x.data_ptr()  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_cache),
            p(indices),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["attn_sink"]),
            p(t["ape"]) if "ape" in t else 0,
            p(t["g_ckv"]) if "g_ckv" in t else 0,
            p(self.kv_state),
            p(self.score_state),
            p(t["hc_attn_fn"]) if "hc_attn_fn" in t else 0,
            p(self.hc_sb["attn"]),
            p(t["hc_ffn_fn"]) if "hc_ffn_fn" in t else 0,
            p(self.hc_sb["ffn"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            p(t["w_o_a"]),
            p(t["s_o_a"]),
            p(t["w_o_b"]),
            p(t["s_o_b"]),
            p(t["w_r"]),
            p(t["bias"]),
            p(t["w_ug"]),
            p(t["s_ug"]),
            p(t["w_dn"]),
            p(t["s_dn"]),
            p(self.scratch),
            self.sym,
            p(self.peers),
            0 if self.timeline is None else p(self.timeline),
            p(self.step),
            self.rank,
            layer,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self):
        self.step.add_(1)

    def close(self):
        """Release this rank's remote HIP IPC mappings."""

        self.peer_buffer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def timeline_report(self) -> str:
        """Per stage, in us from launch start: [first start, median hint seen, last end]
        and median per-task phases (hint wait, payload staging, compute, epilogue)."""
        if self.timeline is None:
            raise RuntimeError("timeline collection was not enabled")
        tl = self.timeline[:, :5].cpu().double() / 100.0  # s_memrealtime ticks at 100 MHz
        t0 = tl[:, 0].min()
        rows, i = [], 0
        for name, n in self.stages:
            st = tl[i : i + n].clone()
            i += n
            for c in (1, 2, 3):  # missing marks inherit the previous one
                st[:, c] = torch.where(st[:, c] > 0, st[:, c], st[:, c - 1])
            d = (st[:, 1:] - st[:, :-1]).median(0).values
            rows.append(
                f"{name:7s} x{n:4d}  [{(st[:, 0].min() - t0):6.1f} | hint {(st[:, 1].median() - t0):6.1f} | "
                f"end {(st[:, 4].max() - t0):6.1f}]  hint {d[0]:5.1f}  stage {d[1]:5.1f}  "
                f"compute {d[2]:5.1f}  epi {d[3]:5.1f}"
            )
        return "\n".join(rows)

    def intermediates(self):
        cfg = self.W.cfg
        S, H = self.S, cfg.heads
        mid = self.debug("mid", (S, cfg.top_k + 1, cfg.inter))
        if moe_format(self.moe_mode).activation is ExpertActivation.BF16:
            mid = mid.to(torch.bfloat16).float()
        return dict(
            q_a=self.debug("q_a", (S, cfg.q_lora)),
            kv=self.debug("kv_a", (S, cfg.head_dim)),
            q=self.debug("q", (S, H, cfg.head_dim), bf2=True),
            o=self.debug("o", (S, H, cfg.head_dim), bf2=True),
            o_lora=self.debug("o_lora", (S, cfg.o_groups * cfg.o_lora), bf2=True),
            a=self.debug(
                "a",
                (S, cfg.hidden) if cfg.hc_mult == 1 else (S, cfg.hc_mult, cfg.hidden),
                bf2=True,
            ).to(torch.bfloat16),
            scores=self.debug("scores", (S, cfg.n_experts)),
            sel=self.debug("sel", (S, cfg.top_k + 1), torch.int32),
            prob=self.debug("prob", (S, cfg.top_k + 1)),
            mid=mid,
            xq=self.debug("xqd", (S, cfg.hidden), pairs=False),
        )
