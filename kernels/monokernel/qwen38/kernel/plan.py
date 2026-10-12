# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side shapes of the Qwen3.8 MonoKernel (TP1): execution constants, per-stage task counts and the
mailbox layout."""

from kernels.monokernel.dsv4.kernel.plan import LAYER_SLOTS  # noqa: F401  (one step-advance tag scheme)
from kernels.monokernel.qwen35.kernel.plan import (  # noqa: F401
    BLOCKS,
    MOE_MISC,
    PROJ_GROUPS,
    PROJ_TILE,
    THREADS,
    TL_COLS,
    UG8,
    WAVES,
    dn_tile,
)

HC_KS = 512  # K of one hyper-connection down task (8 chunks of 64, inside one stream)
HC_RB = 64  # rows of one hyper-connection reduce task
OUT_TILE = 16  # out_proj rows per task
SH_TILE = 16  # shared-expert intermediates per up/gate task


def hc_rows(dims: dict) -> int:
    """Rows of the down + inject GEMV (low rank, then the inject logits), padded to a 32-row task pair."""
    return -(-(dims["hc_lowrank"] + dims["hc"]) // 32) * 32


def hc_ksplit(dims: dict) -> int:
    return dims["hc"] * dims["hidden"] // HC_KS


def router_rows(n_experts: int) -> int:
    """Router GEMV rows: the experts, the shared-expert gate, padded to a 16-row group."""
    return -(-(n_experts + 1) // 16) * 16


def gdn_dims(dims: dict) -> dict:
    key_dim = dims["hk"] * dims["k_dim"]
    value_dim = dims["hv"] * dims["v_dim"]
    return dict(key_dim=key_dim, value_dim=value_dim, conv_dim=2 * key_dim + value_dim)


def proj_rows(full: bool, dims: dict) -> int:
    """Rows of the token mixer's input projection: GDN [q | k | v | z | b | a]."""
    g = gdn_dims(dims)
    return g["conv_dim"] + g["value_dim"] + 2 * dims["hv"]


def out_k(full: bool, dims: dict) -> int:
    return dims["heads"] * dims["head_dim"] if full else gdn_dims(dims)["value_dim"]


def _align(n, a=256):
    return (n + a - 1) // a * a


def layout(S: int, full: bool, **dims):
    """Byte offsets of the scratch mailboxes; every mailbox holds ``(value, tag)`` int32 pairs (8 bytes),
    bf16 mailboxes pack two elements per pair."""
    pr = 8
    H, W4 = dims["hidden"], dims["hc"] * dims["hidden"]
    KS, RR = hc_ksplit(dims), hc_rows(dims)
    hc = []
    for side in ("a", "m"):
        hc += [
            (f"hcp_{side}", S * KS * RR * pr),  # down / inject partials per K split
            (f"hcss_{side}", S * KS * pr),  # partial sums of squares per K split
            (f"hsd_{side}", S * dims["hc_lowrank"] * pr),  # silu(down / hc)
            (f"hr_{side}", S * dims["hc"] * pr),  # inject logits (bf16 values)
            (f"hrs_{side}", S * dims["hc"] * pr),  # per-stream rstd
            (f"m_{side}", S * H // 2 * pr),  # the sublayer input, bf16 pairs
        ]
    g = gdn_dims(dims)
    mixer = [
        ("conv", S * g["conv_dim"] * pr),
        ("z", S * g["value_dim"] * pr),
        ("ba", S * 2 * dims["hv"] * pr),
    ]
    slots = dims["top_k"]
    items = hc + mixer + [
        ("gated", S * out_k(full, dims) // 2 * pr),
        ("x1", S * W4 // 2 * pr),  # the bundle after the mixer's combine, bf16 pairs
        ("scores", S * router_rows(dims["n_experts"]) * pr),
        ("xq", S * H // 4 * pr),
        ("xqs", S * H // 128 * pr),
        ("sel", S * slots * pr),
        ("prob", S * slots * pr),
        ("mid", S * slots * dims["inter"] * pr),
        ("mid_sh", S * dims["shared_inter"] // 2 * pr),  # shared-expert SwiGLU, bf16 pairs
        ("xqd", S * H * 4),  # debug: dequantized MoE input (plain f32), last: not scrubbed
    ]
    off, scratch = 0, {}
    for name, size in items:
        scratch[name] = off
        off += _align(size)
    scratch["_bytes"] = off
    return scratch


def stage_tasks(S: int, full: bool, **dims):
    """[(stage name, task count)] in execution order."""
    H = dims["hidden"]
    KS, RR = hc_ksplit(dims), hc_rows(dims)

    def hc(side):
        return [
            (f"hcd_{side}", RR // 32 * KS),
            (f"hcr_{side}", S * -(-RR // HC_RB)),
            (f"hcu_{side}", H // 32),
        ]

    mixer = [
        ("proj", -(-proj_rows(full, dims) // (PROJ_TILE * PROJ_GROUPS))),
        ("gdn", S * dims["hv"]),
    ]
    return (
        hc("a")
        + mixer
        + [("out", H // OUT_TILE)]
        + hc("m")
        + [
            ("router", router_rows(dims["n_experts"]) // 16),
            ("sh_ug", dims["shared_inter"] // SH_TILE),
            ("ug", S * dims["top_k"] * (dims["inter"] // UG8)),
            ("down", H // dn_tile(S, H)),
        ]
    )
