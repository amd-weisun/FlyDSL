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
    build_advance_step,
    build_dsv4_kernel,
    layout,
    stage_tasks,
)
from kernels.dsv4_moe_layer.packing import pack_layer_weights
from kernels.dsv4_moe_layer.config import BLOCK_TOKENS
from kernels.dsv4_moe_layer.reference import LayerWeights, fp4_pool_shapes
from kernels.mla_moe_layer.runtime import SymmetricPeerBuffer

__all__ = ["MoeMode", "Dsv4MoeLayer", "Dsv4Variant", "shape_dims"]


def shape_dims(cfg) -> dict:
    """The shape arguments layout(), stage_tasks() and build_dsv4_kernel() share.

    They MUST see identical values: the host derives scratch offsets and its size
    from one and the kernel from the other, so any drift both misreads every
    mailbox and undersizes the buffer. Deriving them once, here, is what stops a
    new shape reaching two of the three. Note the JIT disk cache does not key on
    layout()/stage_tasks(), so editing either needs FLYDSL_RUNTIME_ENABLE_CACHE=0
    to take effect.
    """
    return dict(
        hidden=cfg.hidden,
        q_lora=cfg.q_lora,
        head_dim=cfg.head_dim,
        o_groups=cfg.o_groups,
        o_lora=cfg.o_lora,
        hc_mult=cfg.hc_mult,
        compress_ratio=cfg.compress_ratio,
        n_keys=cfg.n_keys,
        c_coff=cfg.c_coff,
        # 0 means "no indexer"; only CSA runs one
        index_head_dim=cfg.index_head_dim if cfg.indexed else 0,
        index_heads=cfg.index_heads if cfg.indexed else 0,
        max_seq=cfg.max_seq,
        index_heads_total=cfg.index_heads_total if cfg.indexed else 0,
        index_topk=cfg.index_topk if cfg.indexed else 0,
        kv_fp8=cfg.kv_fp8,
        indexer_hadamard=cfg.indexer_hadamard,
    )


def _variant_key(cfg, samples, npes, moe_mode, timeline, poll_timeout_us, tokens_per_seq=1):
    """Everything the compiled kernel and the scratch layout depend on."""
    return (
        poll_timeout_us,
        tokens_per_seq,
        tuple(sorted(shape_dims(cfg).items())),
        samples,
        cfg.heads,
        npes,
        cfg.window,
        cfg.cache_rows,
        cfg.n_experts,
        cfg.top_k,
        cfg.inter,
        cfg.softmax_scale,
        cfg.swiglu_limit,
        cfg.hc_sinkhorn_iters,
        cfg.hc_eps,
        moe_mode,
        timeline,
    )


class Dsv4Variant:
    """The compiled kernel and the buffers every layer of one attention variant shares.

    V4 alternates its attention variant per layer -- at V4-Pro, HCA on layers 0, 1
    and then odd ids, CSA on even ids from 2 -- and the shape is structural, not a
    runtime branch: the stage task counts differ, which moves every stage's
    CTA-to-task base, and the mailbox offsets differ, which moves every scratch
    region. So a stack needs one compiled kernel per variant.

    What it does NOT need is one per LAYER. The kernel, the scratch and the
    symmetric buffer depend only on the shape, so all 31 HCA layers share one of
    these and all 30 CSA layers another; only the weights and the rolling
    compressor state are per layer. That matters at 61 layers, where a symmetric
    buffer each would mean 61 IPC exchanges, in the same order on every rank.

    Scratch CANNOT be shared ACROSS variants: the layouts differ, so two kernels
    would read the same bytes as different mailboxes. Within a variant it is safe
    because every launch stamps a fresh epoch tag -- which is why ``step`` lives
    here too, one per scratch, advanced once per decode step. A second step
    counter on the same scratch would let two layers mint the same tag.
    """

    _cache: dict = {}
    _adv_cache: dict = {}

    def __init__(
        self,
        cfg,
        samples: int,
        rank: int = 0,
        npes: int = 1,
        group=None,
        timeline: bool = False,
        moe_mode: MoeMode | str = MoeMode.A8W4,
        allow_unindexed_csa: bool = False,
        poll_timeout_us: int | None = None,
        tokens_per_seq: int = 1,
    ):
        moe_mode = as_moe_mode(moe_mode)
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio, allow_unindexed_csa)
        if samples % tokens_per_seq:
            raise ValueError(f"{samples} samples are not whole runs of {tokens_per_seq} tokens per sequence")
        self.poll_timeout_us = poll_timeout_us
        self.tokens_per_seq = tokens_per_seq
        self.key = _variant_key(cfg, samples, npes, moe_mode, timeline, poll_timeout_us, tokens_per_seq)
        dims = shape_dims(cfg)
        dev = torch.device("cuda", torch.cuda.current_device())
        self.scr_layout, self.sym_layout = layout(samples, cfg.heads, npes, cfg.window, moe_mode, **dims)
        self.scratch = torch.zeros(self.scr_layout["_bytes"], dtype=torch.uint8, device=dev)
        self.peer_buffer = SymmetricPeerBuffer(self.sym_layout["_bytes"], rank=rank, npes=npes, group=group)
        self.sym_storage = self.peer_buffer.storage
        self.sym = self.peer_buffer.local_address
        self.peers = self.peer_buffer.addresses
        self.step = torch.zeros(1, dtype=torch.int32, device=dev)  # decode-step counter
        # Nonzero once any launch on this scratch gave up a poll (the kernel's
        # bounded wait): its outputs, and every later one, are garbage.
        self.hang = torch.zeros(1, dtype=torch.int32, device=dev)
        # Tracing is the expensive part and depends only on the key, so two variants
        # of the same shape (or a stack built one layer at a time) reuse it.
        built = Dsv4Variant._cache.get(self.key)
        if built is None:
            built = build_dsv4_kernel(
                samples,
                cfg.heads,
                npes,
                cfg.window,
                scale=cfg.softmax_scale,
                timeline=timeline,
                moe_mode=moe_mode,
                poll_timeout_us=poll_timeout_us,
                tokens_per_seq=tokens_per_seq,
                n_experts=cfg.n_experts,
                top_k=cfg.top_k,
                inter=cfg.inter,
                swiglu_limit=cfg.swiglu_limit,
                hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
                hc_eps=cfg.hc_eps,
                window_rows=cfg.cache_rows,
                **dims,
            )
            Dsv4Variant._cache[self.key] = built
        self.launch = built
        # the step advance clears stale mailbox pairs (see build_advance_step);
        # the debug-only plain-f32 "xqd" region sits last and is left alone
        self.scr_pairs = self.scr_layout["xqd"] // 8
        self.sym_pairs = self.sym_layout["_bytes"] // 8
        adv_key = (self.scr_pairs, self.sym_pairs)
        if adv_key not in Dsv4Variant._adv_cache:
            Dsv4Variant._adv_cache[adv_key] = build_advance_step(*adv_key)
        self._advance = Dsv4Variant._adv_cache[adv_key]
        self.stages = stage_tasks(samples, cfg.heads, window=cfg.window, top_k=cfg.top_k, inter=cfg.inter, **dims)

    def hang_detected(self) -> bool:
        """Whether a launch on this scratch timed out a poll (only a variant built
        with ``poll_timeout_us`` ever does). Synchronizes."""
        return bool(self.hang.item())

    def advance_step(self):
        """One decode step done on this scratch: ``step += 1``, and a slice of the
        stale mailbox pairs cleared so tags can wrap. Stream-ordered, so
        graph-capturable."""
        self._advance(
            self.step.data_ptr(), self.scratch.data_ptr(), self.sym, stream=torch.cuda.current_stream()
        )

    def close(self):
        self.peer_buffer.close()


class Dsv4MoeLayer:
    """One rank of one transformer layer: its weights and its rolling state.

    Everything that depends only on the SHAPE -- the compiled kernel, the scratch
    and the symmetric buffer -- lives in a ``Dsv4Variant``, which layers of the
    same attention variant share; pass one to build a stack. Left out, the layer
    makes its own, which is what a single-layer test or benchmark wants.

    Layers sharing a variant share its scratch, so they must pass distinct
    ``layer`` values (the epoch tag is ``step * MAX_LAYERS_PER_STEP + layer + 1``)
    and advance the step exactly once per decode step. Note ``debug()`` and
    ``intermediates()`` then read whichever layer ran last.
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
        allow_unindexed_csa: bool = False,
        variant: "Dsv4Variant | None" = None,
        packed: dict | None = None,
        tokens_per_seq: int = 1,
    ):
        """``packed``: another layer object's ``packed`` weights to share -- one set
        of weights serving several sample counts (one object, and one compiled
        variant, per count). ``tokens_per_seq`` > 1: an MTP verify step, samples in
        runs of that many tokens of one sequence (see build_dsv4_kernel); taken
        from ``variant`` when one is given."""
        cfg = W.cfg
        validate_shard(samples, cfg.heads, rank, npes, cfg.window, cfg.compress_ratio, allow_unindexed_csa)
        if cfg.hc_mult > 1 and cfg.hc_mult & (cfg.hc_mult - 1):
            raise ValueError(f"hc_mult must be 1 or a power of two, got {cfg.hc_mult}")
        self.moe_mode = as_moe_mode(moe_mode)
        # the kernel reads the router bias as N_EXPERTS f32: a bf16 one (torch's default
        # dtype in a serving process) is half that, and the read runs off its end
        if W.t["bias"].dtype != torch.float32:
            raise ValueError(f"the router bias must be float32, got {W.t['bias'].dtype}")
        self.W, self.S, self.rank, self.npes = W, samples, rank, npes
        self.window = cfg.window
        self.packed = pack_layer_weights(W.t, self.moe_mode) if packed is None else packed
        # [3 scales | hc_mix bases] per side, as one f32 vector the kernel indexes
        dev0 = torch.device("cuda", torch.cuda.current_device())
        self.hc_sb = {}
        for side in ("attn", "ffn"):
            if f"hc_{side}_scale" in W.t:
                self.hc_sb[side] = torch.cat(
                    [W.t[f"hc_{side}_scale"].float(), W.t[f"hc_{side}_base"].float()]
                ).contiguous()
            else:
                self.hc_sb[side] = torch.zeros(1, dtype=torch.float32, device=dev0)
        dims = shape_dims(cfg)
        # layout() and build_dsv4_kernel() MUST see identical shape arguments: the
        # host derives scratch offsets and its size from one and the kernel from the
        # other, so any drift both misreads every mailbox and undersizes the buffer.
        # build_dsv4_kernel() then re-derives both from ITS arguments, so a new shape
        # has to reach all three -- which is why every one of them is fed from `dims`.
        # Note the JIT disk cache does not key on layout()/stage_tasks(), so a change
        # to either needs FLYDSL_RUNTIME_ENABLE_CACHE=0 to actually take effect.
        #
        # `cos`/`sin` are ONE table for the whole layer -- window, compressor and
        # (later) indexer alike. V4 picks its base per layer, not per consumer: a
        # compressing layer builds the table on compress_rope_theta with YaRN, a
        # pure sliding-window layer on rope_theta without. That is the caller's
        # choice; see V4Config.compress_rope_theta.
        dev = torch.device("cuda", torch.cuda.current_device())
        if variant is None:
            variant = Dsv4Variant(
                cfg,
                samples,
                rank=rank,
                npes=npes,
                group=group,
                timeline=timeline,
                moe_mode=self.moe_mode,
                allow_unindexed_csa=allow_unindexed_csa,
                tokens_per_seq=tokens_per_seq,
            )
        elif variant.key != _variant_key(
            cfg, samples, npes, self.moe_mode, timeline, variant.poll_timeout_us, variant.tokens_per_seq
        ):
            raise ValueError(
                "this layer's shape is not the one the variant was compiled for; a variant is shared "
                "only by layers of the SAME attention variant (compress_ratio, n_keys, max_seq, ...)"
            )
        self.variant = variant
        self.scr_layout, self.sym_layout = variant.scr_layout, variant.sym_layout
        self.scratch, self.sym, self.peers = variant.scratch, variant.sym, variant.peers
        self.launch, self.stages = variant.launch, variant.stages
        tok = self.tokens_per_seq = variant.tokens_per_seq
        n_seq = samples // tok
        c_ring = cfg.c_rows + tok - 1  # the kernel's C_RING: the window plus a step's drafts
        # the compressor carries a rolling window across decode steps, so its state
        # lives here rather than being rebuilt per call. The leading `samples` is
        # the batch axis: each sample is its own sequence, so it carries its own
        # rolling state -- which is also what keeps the S compressor tasks, one per
        # CTA with nothing ordering them, from racing on a shared one.
        if cfg.indexed:
            ishape = (n_seq, c_ring, cfg.c_coff * cfg.index_head_dim)
            self.i_kv_state = torch.zeros(*ishape, dtype=torch.float32, device=dev)
            self.i_score_state = torch.full(ishape, float("-inf"), dtype=torch.float32, device=dev)
            # the indexer's key cache: ATOM's paged FP4 pool (codes, scales), each
            # sample its own run of blocks here (reference.fp4_pool_shapes)
            dshape, sshape = fp4_pool_shapes(cfg, n_seq)
            self.i_cache = torch.zeros(*dshape, dtype=torch.uint8, device=dev)
            self.i_cache_s = torch.zeros(*sshape, dtype=torch.uint8, device=dev)
        else:
            self.i_kv_state = self.i_score_state = self.i_cache = self.i_cache_s = torch.zeros(1, dtype=torch.float32, device=dev)
        if cfg.compress_ratio:
            shape = (n_seq, c_ring, cfg.c_coff * cfg.head_dim)
            self.kv_state = torch.zeros(*shape, dtype=torch.float32, device=dev)
            # -inf, not zero: with overlapping windows (CSA) the previous window's
            # rows are unwritten before the first emit and must drop out of the
            # softmax. Harmless for the non-overlapping case, which fills them all.
            self.score_state = torch.full(shape, float("-inf"), dtype=torch.float32, device=dev)
        else:
            self.kv_state = self.score_state = torch.zeros(1, dtype=torch.float32, device=dev)
        # The trivial state pool: slot s is sample s and each field is contiguous,
        # so the strides are just the per-sample sizes. A serving pool hands out
        # slots per sequence and interleaves these fields inside one entry, which
        # is why both the slot and the stride cross the boundary rather than being
        # derived from the sample index here.
        # (a sequence's tokens share its slot)
        self.state_slots = torch.arange(samples, dtype=torch.int32, device=dev) // tok
        self.st_kv = self.kv_state[0].numel() if cfg.compress_ratio else 0
        self.st_i = self.i_kv_state[0].numel() if cfg.indexed else 0
        self.st_ic = self.i_cache[0].numel() if cfg.indexed else 0  # bytes per sample
        # Compressed entries are paged (see the kernel's K_PB). The trivial table:
        # every sample's block b is b, one entry per plane row, so entry e of
        # sample s sits at dest_rows[1, s] + e -- a contiguous pool. ATOM passes
        # its own tables and envelope height to forward().
        self.k_pb = BLOCK_TOKENS // cfg.compress_ratio if cfg.compress_ratio else 1
        n_blocks = max(1, -(-cfg.n_compressed // self.k_pb)) if cfg.compress_ratio else 1
        self.block_tables = torch.arange(n_blocks, dtype=torch.int32, device=dev).repeat(samples, 1)
        self.env_rows = self.k_pb
        n_tasks = sum(n for _, n in self.stages)
        self.timeline = torch.zeros(n_tasks, TL_COLS, dtype=torch.int64, device=dev) if timeline else None
        self.step = variant.step

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

    def forward(
        self,
        h,
        cur_pos,
        kv_cache,
        dest_rows,
        indices,
        cos,
        sin,
        x_out=None,
        layer=0,
        advance=True,
        tokens=None,
        block_tables=None,
        env_rows=None,
        state=None,
    ):
        """One layer.  Mailbox epochs are ``step * 128 + layer + 1``: layers sharing
        this scratch within a decode step need distinct ``layer``; call
        ``advance_step`` (or pass ``advance=True``) once per step.  Both are
        stream-ordered device ops, so the sequence can be captured in a HIP graph."""
        if not 0 <= layer < MAX_LAYERS_PER_STEP:
            raise ValueError(f"layer must be in [0, {MAX_LAYERS_PER_STEP}), got {layer}")
        cfg = self.W.cfg
        # The kernel derives every per-sample offset from self.S, so a leading
        # dimension that disagrees is not a wrong answer, it is a read past the end
        # of somebody's buffer. Cheap to check once per layer against a launch that
        # would otherwise corrupt memory.
        hshape = (self.S, cfg.hidden) if cfg.hc_mult == 1 else (self.S, cfg.hc_mult, cfg.hidden)
        if tuple(h.shape) != hshape:
            raise ValueError(f"h must be {hshape}, got {tuple(h.shape)}")
        if cfg.kv_fp8:
            # ATOM's fp8 layout: (NoPE plane uint8 [rows, 512], RoPE plane bf16 [rows, rope_dim])
            kv_nope, kv_rope = kv_cache
            if kv_nope.dtype != torch.uint8 or kv_nope.shape[-1] != 512 or kv_rope.shape[-1] != cfg.rope_dim:
                raise ValueError(
                    f"kv_fp8 wants (uint8 [rows, 512], bf16 [rows, {cfg.rope_dim}]), "
                    f"got {kv_nope.dtype} {tuple(kv_nope.shape)}, {tuple(kv_rope.shape)}"
                )
        else:
            kv_nope, kv_rope = kv_cache, None
            if kv_cache.ndim != 2 or kv_cache.shape[1] != cfg.head_dim:
                raise ValueError(
                    f"kv_cache must be one plane [rows, {cfg.head_dim}], got {tuple(kv_cache.shape)} -- "
                    "which rows a sequence owns is the caller's, supplied through indices and dest_rows"
                )
        if tuple(dest_rows.shape) != (2, self.S):
            raise ValueError(f"dest_rows must be {(2, self.S)}, got {tuple(dest_rows.shape)}")
        if tuple(indices.shape) != (self.S, cfg.n_keys):
            raise ValueError(f"indices must be {(self.S, cfg.n_keys)}, got {tuple(indices.shape)}")
        if cur_pos.numel() != self.S:
            raise ValueError(
                f"cur_pos must hold one position per sample ({self.S}), got {cur_pos.numel()} -- "
                "the samples are independent sequences, each at its own offset"
            )
        t = dict(self.W.t, **self.packed)
        bt = self.block_tables if block_tables is None else block_tables
        # `state` hands the layer someone else's per-sequence state in place of its
        # own -- a serving engine's views, slots and strides (keys: kv_state,
        # score_state, i_kv_state, i_score_state, i_cache, i_cache_s, state_slots,
        # st_kv, st_i, st_ic). What is not given stays the layer's.
        st = {
            k: getattr(self, k)
            for k in (
                "kv_state",
                "score_state",
                "i_kv_state",
                "i_score_state",
                "i_cache",
                "i_cache_s",
                "state_slots",
                "st_kv",
                "st_i",
                "st_ic",
            )
        }
        if state:
            unknown = set(state) - set(st)
            if unknown:
                raise ValueError(f"unknown state keys {sorted(unknown)}")
            st.update(state)
        # A hash-routed layer picks its experts by token id: tid2eid [vocab, top_k].
        use_hash = "tid2eid" in t
        if use_hash and (tokens is None or tokens.numel() != self.S or tokens.dtype != torch.int32):
            raise ValueError(f"a hash-routed layer needs tokens: int32, one per sample ({self.S})")
        if x_out is None:
            x_out = torch.empty(*hshape, dtype=torch.bfloat16, device=h.device)
        p = lambda x: x.data_ptr()  # noqa: E731
        self.launch(
            p(h),
            p(x_out),
            p(cur_pos),
            p(kv_nope),
            p(kv_rope) if kv_rope is not None else 0,
            p(dest_rows),
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
            p(st["kv_state"]),
            p(st["score_state"]),
            p(t["i_ape"]) if "i_ape" in t else 0,
            p(t["g_ickv"]) if "g_ickv" in t else 0,
            p(st["i_kv_state"]),
            p(st["i_score_state"]),
            p(st["i_cache"]),
            p(t["hc_attn_fn"]) if "hc_attn_fn" in t else 0,
            p(self.hc_sb["attn"]),
            p(t["hc_ffn_fn"]) if "hc_ffn_fn" in t else 0,
            p(self.hc_sb["ffn"]),
            p(t["w_qkv_a"]),
            p(t["s_qkv_a"]),
            p(t["w_q_b"]),
            p(t["s_q_b"]),
            p(t["w_i_q_b"]) if "w_i_q_b" in t else 0,
            p(t["s_i_q_b"]) if "s_i_q_b" in t else 0,
            p(t["i_w"]) if "i_w" in t else 0,
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
            p(t["w_sug"]) if "w_sug" in t else 0,
            p(t["s_sug"]) if "s_sug" in t else 0,
            p(t["w_sdn"]) if "w_sdn" in t else 0,
            p(t["s_sdn"]) if "s_sdn" in t else 0,
            p(self.scratch),
            self.sym,
            p(self.peers),
            0 if self.timeline is None else p(self.timeline),
            p(self.step),
            p(self.variant.hang),
            p(st["state_slots"]),
            p(tokens) if use_hash else 0,
            p(t["tid2eid"]) if use_hash else 0,
            p(bt),
            p(st["i_cache_s"]),
            self.rank,
            layer,
            st["st_kv"],
            st["st_i"],
            st["st_ic"],
            int(use_hash),
            bt.stride(0),
            self.env_rows if env_rows is None else env_rows,
            stream=torch.cuda.current_stream(),
        )
        if advance:
            self.advance_step()
        return x_out

    def advance_step(self):
        self.variant.advance_step()

    def close(self):
        """Release this rank's remote HIP IPC mappings.

        The mappings belong to the variant, so this is only the layer's to call
        when it made its own; a shared variant is the caller's to close.
        """
        self.variant.close()

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
            if not n:  # a stage this configuration compiles out has no rows to report
                continue
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
