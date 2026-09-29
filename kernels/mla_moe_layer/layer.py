# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host wrapper: scratch, symmetric peer buffers and the launch of one rank's layer."""

from __future__ import annotations

import torch

from kernels.mla_moe_layer.config import (
    EPS,
    KV_LORA,
    INTER,
    MAX_LAYERS_PER_STEP,
    MOE_SLOTS,
    N_EXPERTS,
    SOFTMAX_SCALE,
    ExpertActivation,
    MoeMode,
    as_moe_mode,
    moe_format,
    validate_shard,
)
from kernels.mla_moe_layer.packing import pack_layer_weights
from kernels.mla_moe_layer.reference import LayerWeights
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer
from kernels.mla_moe_layer.shared_reuse_moe_kernel import (
    TL_COLS,
    build_shared_reuse_kernel,
    layout,
    stage_tasks,
)

__all__ = ["MoeMode", "SharedReuseMlaMoeLayer"]


class SharedReuseMlaMoeLayer:
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
        topk: int = 2048,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.W8A8,
        n_groups: int | None = None,
        topk_groups: int | None = None,
        paged: bool = False,
        eps: float = EPS,
        softmax_scale: float = SOFTMAX_SCALE,
        free_unpacked: bool = False,
        reuse: "SharedReuseMlaMoeLayer | None" = None,
    ):
        """``paged``: serving mode (S independent sequences over a paged 576-wide cache, see
        ``build_shared_reuse_kernel``).  ``reuse``: another layer built with identical
        static arguments whose compiled launcher, scratch, symmetric buffer and step
        counter are shared (they are safe to share: every launch uses a fresh tag).
        ``free_unpacked`` drops the row-major weight copies once packed."""
        validate_shard(samples, W.heads, rank, npes, topk)
        self.moe_mode = as_moe_mode(moe_mode)
        self.W, self.S, self.rank, self.npes, self.topk = W, samples, rank, npes, topk
        self.paged = paged
        self.packed = pack_layer_weights(W.t, self.moe_mode)
        if free_unpacked:
            for name in self.packed:
                W.t.pop(name, None)
        dims = dict(hidden=W.hidden, q_lora=W.q_lora, nope_dim=W.nope_dim, v_dim=W.v_dim)
        dev = torch.device("cuda", torch.cuda.current_device())
        if reuse is not None:
            assert (reuse.S, reuse.W.heads, reuse.npes, reuse.topk, reuse.paged) == (
                samples,
                W.heads,
                npes,
                topk,
                paged,
            ), "reuse= needs identical static arguments"
            self.scr_layout, self.sym_layout = reuse.scr_layout, reuse.sym_layout
            self.scratch, self.peer_buffer = reuse.scratch, reuse.peer_buffer
            self.launch, self.stages, self.timeline, self.step = reuse.launch, reuse.stages, reuse.timeline, reuse.step
        else:
            self.scr_layout, self.sym_layout = layout(samples, W.heads, npes, topk, self.moe_mode, **dims)
            self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
            self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
            self.launch = build_shared_reuse_kernel(
                samples,
                W.heads,
                npes,
                topk,
                scale=softmax_scale,
                timeline=timeline,
                moe_mode=self.moe_mode,
                n_groups=n_groups,
                topk_groups=topk_groups,
                paged=paged,
                eps=eps,
                **dims,
            )
            self.stages = stage_tasks(samples, W.heads, topk, **dims)
            n_tasks = sum(n for _, n in self.stages)
            self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
            self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses

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

    def forward_paged(
        self, h, positions, kv_pool, slot_map, kv_indptr, kv_indices, cos, sin, x_out=None, layer=0, advance=True
    ):
        """Serving decode step of one layer: ``h`` [S, hidden] bf16 (S independent sequences, one
        new token each), ``positions``/``slot_map`` int32 [S], ``kv_indptr`` int32 [S+1] and
        ``kv_indices`` int32 CSR rows of each sample's whole context (new token included) into
        ``kv_pool`` (a [rows, 576] bf16 pool: 512 latent | 64 k_pe), ``cos``/``sin`` f32 [max_pos, 32]."""
        assert self.paged, "layer was not built with paged=True"
        return self.forward(
            h,
            positions,
            kv_pool,
            kv_pool.data_ptr() + KV_LORA * kv_pool.element_size(),
            kv_indices,
            cos,
            sin,
            x_out=x_out,
            layer=layer,
            advance=advance,
            slot_map=slot_map,
            kv_indptr=kv_indptr,
        )

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        pe_cache,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
        slot_map=None,
        kv_indptr=None,
    ):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < MAX_LAYERS_PER_STEP:
            raise ValueError(f"layer must be in [0, {MAX_LAYERS_PER_STEP}), got {layer}")
        t = dict(self.W.t, **self.packed)
        if x_out is None:
            x_out = torch.empty(self.S, self.W.hidden, dtype=torch.bfloat16, device=h.device)
        p = lambda x: x.data_ptr()  # noqa: E731
        pe = pe_cache if isinstance(pe_cache, int) else p(pe_cache)
        opt = lambda x: 0 if x is None else p(x)  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_cache),
            pe,
            p(indices),
            opt(slot_map),
            opt(kv_indptr),
            p(cos),
            p(sin),
            p(t["g_in"]),
            p(t["g_q"]),
            p(t["g_kv"]),
            p(t["g_post"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            p(t["w_uk"]),
            p(t["s_uk"]),
            p(t["w_uv"]),
            p(t["s_uv"]),
            p(t["w_o"]),
            p(t["s_o"]),
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
        S, H = self.S, self.W.heads
        from kernels.mla_moe_layer.config import KV_LORA, PE_DIM

        hidden, q_lora, nope_dim, v_dim = self.W.hidden, self.W.q_lora, self.W.nope_dim, self.W.v_dim
        mid = self.debug("mid", (S, MOE_SLOTS, INTER))
        if moe_format(self.moe_mode).activation is ExpertActivation.BF16:
            mid = mid.to(torch.bfloat16).float()
        return dict(
            q_a=self.debug("q_a", (S, q_lora)),
            kv_a=self.debug("kv_a", (S, KV_LORA + PE_DIM)),
            q_nope=self.debug("q_nope", (S, H, nope_dim), bf2=True),
            q_pe=self.debug("q_pe", (S, H, PE_DIM), bf2=True),
            q_lat=self.debug("q_lat", (S, H, KV_LORA), bf2=True),
            o=self.debug("o", (S, H * v_dim), bf2=True),
            a=self.debug("a", (S, hidden), bf2=True).to(torch.bfloat16),
            scores=self.debug("scores", (S, N_EXPERTS)),
            sel=self.debug("sel", (S, MOE_SLOTS), torch.int32),
            prob=self.debug("prob", (S, MOE_SLOTS)),
            mid=mid,
            xq=self.debug("xqd", (S, hidden), pairs=False),
        )
