# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side shapes of the Qwen3.5 MonoKernel: execution constants, per-stage task counts and the
mailbox layout."""

from kernels.monokernel.dsv4.kernel.plan import LAYER_SLOTS  # noqa: F401  (one step-advance tag scheme)

BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64
PROJ_TILE = 16  # rows of one MFMA row group
PROJ_GROUPS = 2  # row groups per proj task (four waves each split K)
ROW_TILE = 32  # rows per out_proj / attention peer-reduce tile
ROUTER_TILE = 8  # router rows per task (a part of a 16-row MFMA group)
UG8 = 8  # intermediates one up/gate task owns (8 gate + 8 up rows)
MOE_MISC = 16  # misc words before the MoE input's FP8 block scales
KT = 64  # keys per attention tile
BS = 16  # tokens per KV-cache block (ATOM's kv_cache_block_size)
TL_COLS = 5  # timeline stamps per task: start, hint seen, inputs staged, compute done, end


def router_rows(n_experts: int) -> int:
    """Router GEMV rows: the experts, the shared-expert gate, padded to a 16-row group."""
    return -(-(n_experts + 1) // 16) * 16


def router_spt(S: int, n_experts: int) -> int:
    """Samples per router task: the fewest (at most 8) that fit the grid in one round."""
    n_router = router_rows(n_experts) // ROUTER_TILE
    spt = 1
    while S * n_router > BLOCKS * spt and spt < ROUTER_TILE:
        spt *= 2
    return spt


def dn_tile(S: int, hidden: int) -> int:
    """Hidden rows per expert-down / FFN peer-reduce task: 32 at S = 1, else about one task per CTA,
    a multiple of 16 so a tile never straddles the row groups ``reduce_rows`` emits."""
    return 32 if S == 1 else max(16, -(-(hidden // BLOCKS) // 16) * 16)


def n_split(S: int) -> int:
    """Key splits per sample of the full-attention layer: about one split task per CTA."""
    return max(32, 256 // S)


def gdn_dims(dims: dict) -> dict:
    """Derived GDN shard sizes from the build dims."""
    key_dim = dims["hk"] * dims["k_dim"]
    value_dim = dims["hv"] * dims["v_dim"]
    return dict(key_dim=key_dim, value_dim=value_dim, conv_dim=2 * key_dim + value_dim)


def _align(n, a=256):
    return (n + a - 1) // a * a


def out_k(full: bool, dims: dict) -> int:
    """K of out_proj: the mixer's output width (GDN value dim / heads x head dim)."""
    return dims["heads"] * dims["head_dim"] if full else gdn_dims(dims)["value_dim"]


def layout(S: int, npes: int, full: bool, *, hidden, hk, hv, k_dim, v_dim, heads, head_dim, n_experts, top_k, inter):
    """Byte offsets of the per-rank scratch and of the symmetric buffer. Every mailbox holds
    ``(value, tag)`` int32 pairs (8 bytes); bf16 mailboxes pack two elements per pair."""
    dims = dict(hk=hk, hv=hv, k_dim=k_dim, v_dim=v_dim, heads=heads, head_dim=head_dim)
    pr = 8
    slots = top_k + 1
    if full:
        mixer = [
            ("qg", S * heads * 2 * head_dim * pr),  # q | gate per head (bf16 values)
            ("kvr", S * 2 * head_dim * pr),  # raw k then v of this rank's kv head
            ("q", S * heads * head_dim * pr),  # normed, rotated query
            ("kvnew", S * (2 * head_dim + 2) * pr),  # this token's K / V FP8 codes + their scales
            ("sp_acc", S * n_split(S) * heads * head_dim * pr),
            ("sp_ml", S * n_split(S) * heads * 2 * pr),
        ]
    else:
        g = gdn_dims(dims)
        mixer = [
            ("conv", S * g["conv_dim"] * pr),  # conv(qkv) output, one f32 per pair
            ("z", S * g["value_dim"] * pr),
            ("ba", S * 2 * hv * pr),  # b then a, per v-head
        ]
    items = mixer + [
        ("gated", S * out_k(full, dims) // 2 * pr),  # the mixer's output, bf16 pairs
        ("a", S * hidden // 2 * pr),  # post-mixer residual, bf16 pairs
        ("scores", S * router_rows(n_experts) * pr),  # router logits (bf16-rounded) + shared gate
        ("xq", S * hidden // 4 * pr),  # MoE input, four FP8 bytes per pair
        ("xqs", S * hidden // 128 * pr),
        ("sel", S * slots * pr),
        ("prob", S * slots * pr),
        ("mid", S * slots * inter * pr),
        ("xqd", S * hidden * 4),  # debug: dequantized MoE input (plain f32), last: not scrubbed
    ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    part = npes * S * hidden * pr
    sym = {"attn": 0, "ffn": part, "_bytes": 2 * part}
    return scratch, sym


def stage_tasks(S: int, full: bool, *, hidden, hk, hv, k_dim, v_dim, heads, head_dim, n_experts, top_k, inter):
    """[(stage name, task count)] in execution order."""
    if full:
        mixer = [
            ("proj", (heads * 2 * head_dim + 2 * head_dim) // PROJ_TILE // PROJ_GROUPS),
            ("qk", S * (heads + 1)),
            ("split", S * n_split(S)),
            ("merge", S * heads * (head_dim // 64)),
        ]
    else:
        g = gdn_dims(dict(hk=hk, hv=hv, k_dim=k_dim, v_dim=v_dim))
        mixer = [
            ("proj", (g["conv_dim"] + g["value_dim"]) // PROJ_TILE // PROJ_GROUPS),
            ("proj_ba", -(-(2 * hv // PROJ_TILE) // PROJ_GROUPS)),
            ("gdn", S * hv),
        ]
    return mixer + [
        ("out", hidden // ROW_TILE),
        ("router", S * (router_rows(n_experts) // ROUTER_TILE) // router_spt(S, n_experts)),
        # one tile per (routed slot, 8 intermediates); the first inter / UG8 also carry the shared expert
        ("ug", S * top_k * (inter // UG8)),
        ("down", hidden // dn_tile(S, hidden)),
    ]
