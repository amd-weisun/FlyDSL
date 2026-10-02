# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""One DeepSeek-V4 decoder layer's weights, from the HF checkpoint, as ``LayerWeights``.

The checkpoint uses DeepSeek's own ``inference/model.py`` names (``layers.N.attn.wq_a``,
``layers.N.ffn.experts.E.w1``, ...), the same modules the golden was validated against.
Each rank takes its tensor-parallel shard: heads, output groups, index heads, the MoE
intermediate and ``wo_b``'s K are split; everything else is replicated.

Most tensors map across unchanged -- FP8 E4M3 with E8M0 128x128 block scales, and the
routed experts' packed FP4 with E8M0 per-32 scales, are the kernel's own formats. Two
are converted, and each conversion loses information the checkpoint has:

- the compressors' ``wkv`` / ``wgate`` are BF16 in the checkpoint, and are requantized
  to FP8 128x128 blocks here because the kernel fuses them into the ``qkv_a`` GEMV;
The hyper-connection mixers (``hc_*_fn``) stay FP32; packing splits them into a bf16
hi / lo pair, as ATOM's aiter mHC does, so they keep ~16 mantissa bits.

The shared expert stays FP8 128x128, beside the routed experts' MXFP4 bank.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace

import torch
from safetensors import safe_open

from kernels.dsv4_moe_layer.config import ExpertWeight, MoeMode, moe_format
from kernels.dsv4_moe_layer.reference import LayerWeights, V4Config, fp8_mats

FP8_MAX = 448.0


def config_for_layer(path: str, layer: int, tp: int) -> V4Config:
    """This rank's ``V4Config`` for ``layer``, from the checkpoint's own config.json."""
    with open(os.path.join(path, "config.json")) as f:
        c = json.load(f)
    cfg = V4Config(
        heads=c["num_attention_heads"] // tp,
        hidden=c["hidden_size"],
        q_lora=c["q_lora_rank"],
        head_dim=c["head_dim"],
        rope_dim=c["qk_rope_head_dim"],
        o_groups=c["o_groups"] // tp,
        o_lora=c["o_lora_rank"],
        n_experts=c["n_routed_experts"],
        top_k=c["num_experts_per_tok"],
        inter=c["moe_intermediate_size"] // tp,
        window=c["sliding_window"],
        route_scale=c["routed_scaling_factor"],
        swiglu_limit=c["swiglu_limit"],
        eps=c["rms_norm_eps"],
        rope_theta=c["rope_theta"],
        index_heads=c["index_n_heads"] // tp,
        index_heads_total=c["index_n_heads"],
        index_head_dim=c["index_head_dim"],
        index_topk=c["index_topk"],
        compress_rope_theta=c["compress_rope_theta"],
        hc_mult=c["hc_mult"],
        hc_sinkhorn_iters=c["hc_sinkhorn_iters"],
        hc_eps=c["hc_eps"],
    )
    return replace(cfg, compress_ratio=c["compress_ratios"][layer])


def is_hash_layer(path: str, layer: int) -> bool:
    with open(os.path.join(path, "config.json")) as f:
        return layer < json.load(f)["num_hash_layers"]


class Checkpoint:
    """Lazy, sliced reads of a sharded safetensors checkpoint."""

    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.index = json.load(f)["weight_map"]
        self._files = {}

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str, rows=None, cols=None) -> torch.Tensor:
        fname = self.index[name]
        if fname not in self._files:
            self._files[fname] = safe_open(os.path.join(self.path, fname), "pt")
        sl = self._files[fname].get_slice(name)
        r = slice(*rows) if rows else slice(None)
        if len(sl.get_shape()) == 1:
            return sl[r]
        return sl[r, slice(*cols) if cols else slice(None)]


def e8m0_float(s: torch.Tensor) -> torch.Tensor:
    """E8M0 scale bytes -> their float32 values, 2**(byte - 127)."""
    return torch.exp2(s.view(torch.uint8).float() - 127.0)


def fp8_block_quant(w: torch.Tensor, bk: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Float [rows, K] -> FP8 E4M3 plus one float32 scale per 128 x ``bk`` block (amax / 448)."""
    rows, k = w.shape
    b = w.float().reshape(rows // 128, 128, k // bk, bk)
    s = (b.abs().amax(dim=(1, 3)) / FP8_MAX).clamp_min(torch.finfo(torch.float32).tiny)
    q = (b / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.reshape(rows, k), s


def load_layer(
    ck: Checkpoint,
    layer: int,
    rank: int,
    tp: int,
    device="cuda",
    moe_mode: MoeMode | str = MoeMode.A8W4,
) -> LayerWeights:
    """``layer``'s weights for TP rank ``rank`` of ``tp``, in ``make_weights``' layout.

    A hash-routed layer also carries ``tid2eid`` [vocab, top_k] (int32), the token-id ->
    expert-id table that replaces scored routing there.
    """
    cfg = config_for_layer(ck.path, layer, tp)
    cfg.validate()
    p = f"layers.{layer}."
    t = {}

    def fp8(name, rows=None, cols=None):
        """An FP8 matrix and its block scales, sliced to this rank's shard."""
        q = ck.get(name + ".weight", rows, cols).view(torch.float8_e4m3fn)
        sr = (rows[0] // 128, rows[1] // 128) if rows else None
        sc = (cols[0] // 128, cols[1] // 128) if cols else None
        return q.to(device), e8m0_float(ck.get(name + ".scale", sr, sc)).to(device)

    def shard(n):
        return (n * rank, n * (rank + 1))

    t["g_in"] = ck.get(p + "attn_norm.weight").to(device)
    t["g_q"] = ck.get(p + "attn.q_norm.weight").to(device)
    t["g_kv"] = ck.get(p + "attn.kv_norm.weight").to(device)
    t["g_post"] = ck.get(p + "ffn_norm.weight").to(device)
    t["attn_sink"] = ck.get(p + "attn.attn_sink", shard(cfg.heads)).float().to(device)

    # qkv_a, in qkv_a_split's order: wq_a | wkv | compressor wkv | wgate | indexer's pair
    parts = [fp8(p + "attn.wq_a"), fp8(p + "attn.wkv")]
    bf16_rows = []
    if cfg.compress_ratio:
        bf16_rows += [p + "attn.compressor.wkv.weight", p + "attn.compressor.wgate.weight"]
        if cfg.indexed:
            bf16_rows += [p + "attn.indexer.compressor.wkv.weight", p + "attn.indexer.compressor.wgate.weight"]
    for name in bf16_rows:
        q, s = fp8_block_quant(ck.get(name).to(device))
        parts.append((q, s))
    t["w_qkv_a"] = torch.cat([q for q, _ in parts])
    t["s_qkv_a"] = torch.cat([s for _, s in parts])

    t["w_q_b"], t["s_q_b"] = fp8(p + "attn.wq_b", rows=shard(cfg.heads * cfg.head_dim))
    t["w_o_a"], t["s_o_a"] = fp8(p + "attn.wo_a", rows=shard(cfg.o_groups * cfg.o_lora))
    t["w_o_b"], t["s_o_b"] = fp8(p + "attn.wo_b", cols=shard(cfg.o_groups * cfg.o_lora))

    if cfg.compress_ratio:
        t["ape"] = ck.get(p + "attn.compressor.ape").float().to(device)
        t["g_ckv"] = ck.get(p + "attn.compressor.norm.weight").to(device)
        if cfg.indexed:
            t["w_i_q_b"], t["s_i_q_b"] = fp8(
                p + "attn.indexer.wq_b", rows=shard(cfg.index_heads * cfg.index_head_dim)
            )
            t["i_ape"] = ck.get(p + "attn.indexer.compressor.ape").float().to(device)
            t["g_ickv"] = ck.get(p + "attn.indexer.compressor.norm.weight").to(device)
            t["i_w"] = ck.get(p + "attn.indexer.weights_proj.weight", shard(cfg.index_heads)).to(device)

    if cfg.hc_mult > 1:
        for side in ("attn", "ffn"):
            fn = torch.zeros(cfg.hc_rows, cfg.hc_mult * cfg.hidden, dtype=torch.float32, device=device)
            fn[: cfg.hc_mix] = ck.get(p + f"hc_{side}_fn").float().to(device)
            t[f"hc_{side}_fn"] = fn  # fp32: packing splits it into bf16 hi / lo
            t[f"hc_{side}_base"] = ck.get(p + f"hc_{side}_base").float().to(device)
            t[f"hc_{side}_scale"] = ck.get(p + f"hc_{side}_scale").float().to(device)

    t["w_r"] = ck.get(p + "ffn.gate.weight").to(device)
    bias = p + "ffn.gate.bias"
    # explicit dtypes throughout: a serving process (ATOM) sets torch's default dtype
    # to bf16, and a bf16 bias is half the bytes the kernel reads
    t["bias"] = (
        ck.get(bias).float().to(device)
        if ck.has(bias)
        else torch.zeros(cfg.n_experts, dtype=torch.float32, device=device)
    )
    if ck.has(p + "ffn.gate.tid2eid"):
        t["tid2eid"] = ck.get(p + "ffn.gate.tid2eid").to(torch.int32).to(device)

    if moe_format(moe_mode).weight is not ExpertWeight.MXFP4_BLOCK32:
        raise NotImplementedError("the checkpoint's routed experts are MXFP4: load them with MoeMode.A8W4")
    inter, hidden = cfg.inter, cfg.hidden
    n_bank = cfg.n_experts  # the shared expert stays FP8, beside the bank
    ug_q = torch.empty(n_bank, 2 * inter, hidden // 2, dtype=torch.uint8, device=device)
    ug_s = torch.empty(n_bank, 2 * inter, hidden // 32, dtype=torch.uint8, device=device)
    dn_q = torch.empty(n_bank, hidden, inter // 2, dtype=torch.uint8, device=device)
    dn_s = torch.empty(n_bank, hidden, inter // 32, dtype=torch.uint8, device=device)
    rows = shard(inter)
    for e in range(cfg.n_experts):
        x = p + f"ffn.experts.{e}."
        for h, w in enumerate(("w1", "w3")):  # gate rows, then up rows
            ug_q[e, h * inter : (h + 1) * inter] = ck.get(x + w + ".weight", rows).view(torch.uint8).to(device)
            ug_s[e, h * inter : (h + 1) * inter] = ck.get(x + w + ".scale", rows).view(torch.uint8).to(device)
        dn_q[e] = ck.get(x + "w2.weight", cols=shard(inter // 2)).view(torch.uint8).to(device)
        dn_s[e] = ck.get(x + "w2.scale", cols=shard(inter // 32)).view(torch.uint8).to(device)
    # the shared expert: FP8 128x128 in the checkpoint, kept so (as ATOM runs it)
    sh = p + "ffn.shared_experts."
    (gq, gs), (uq, us) = fp8(sh + "w1", rows=rows), fp8(sh + "w3", rows=rows)
    t["w_sug"], t["s_sug"] = torch.cat([gq, uq]), torch.cat([gs, us])
    # a column slice: the kernel reads both as dense row-major
    t["w_sdn"], t["s_sdn"] = (v.contiguous() for v in fp8(sh + "w2", cols=rows))
    t["w_ug"], t["s_ug"], t["w_dn"], t["s_dn"] = ug_q, ug_s, dn_q, dn_s

    for name, (r, k, _bk) in fp8_mats(cfg).items():
        assert t[f"w_{name}"].shape == (r, k), f"{name}: {tuple(t[f'w_{name}'].shape)} != {(r, k)}"
    return LayerWeights(cfg, t)
