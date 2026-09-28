# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Static DeepSeek-V4 shard dimensions.

Defaults are V4-Pro at TP8. The MoE arithmetic-mode enums are shared with
:mod:`kernels.mla_moe_layer.config` -- the expert formats are identical, only
the dimensions and the routing differ.
"""

from __future__ import annotations

from kernels.mla_moe_layer.config import (  # noqa: F401  (re-exported)
    MOE_FORMATS,
    ExpertActivation,
    ExpertWeight,
    MoeFormat,
    MoeMode,
    as_moe_mode,
    moe_format,
)

# Attention. V4 is shared-KV (MQA): wkv emits ONE head_dim vector per token and
# K and V are the same tensor, with RoPE in its tail. There is no kv_lora /
# nope-vs-pe split and no absorbed W_UK / W_UV.
HIDDEN = 7168
Q_LORA = 1536
HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM  # 448, the FP8-quantized part of the KV row
O_LORA = 1024
O_GROUPS = 2  # local; 16 global / 8 ranks
WINDOW = 128  # sliding-window KV ring

# MoE. V4 drops V3's group-limited routing: flat top-k over all experts, scored
# with sqrt(softplus(.)) instead of sigmoid, and a clamped SwiGLU.
N_EXPERTS = 384
TOP_K = 6
MOE_SLOTS = 1 + TOP_K
SHARED_EXPERT = N_EXPERTS  # the shared expert sits last in the bank
INTER = 384  # local; 3072 global / 8 ranks
ROUTE_SCALE = 2.5
SWIGLU_LIMIT = 10.0

# Manifold-Constrained Hyper-Connections: the residual stream carries HC_MULT
# parallel copies instead of one, mixed per token through a Sinkhorn-normalised
# HC_MULT x HC_MULT matrix. HC_MULT == 1 means a plain residual.
HC_MULT = 4
HC_SINKHORN_ITERS = 20
HC_EPS = 1e-6
HC_MIX = (2 + HC_MULT) * HC_MULT  # pre | post | comb, packed in that order

EPS = 1e-6
SCALE_BM = 128
FP8_MAX = 448.0
SOFTMAX_SCALE = HEAD_DIM**-0.5

SUPPORTED_SAMPLES = (1, 2, 4, 8)
SUPPORTED_PEERS = (1, 2, 4, 8)
# 128 total heads / 8 ranks; the split-attention kernel requires
# `heads % WAVES == 0 and heads <= 16`.
SUPPORTED_HEADS = (8, 16)
MAX_LAYERS_PER_STEP = 128

# Per-layer attention variants. V4-Pro ships 31 HCA + 30 CSA layers and one
# ratio-0 entry for the MTP block; `compress_ratios` in the HF config carries
# one entry per layer PLUS one for MTP.
COMPRESS_SWA = 0  # sliding window only (the MTP block's shape)
COMPRESS_CSA = 4  # compressed sparse attention, needs the lightning indexer
COMPRESS_HCA = 128  # heavily compressed attention, dense over compressed
# HCA needs only the KV compressor; CSA additionally needs the lightning indexer
# (and the compressor in its overlapping form), which is not implemented.
SUPPORTED_COMPRESS = (COMPRESS_SWA, COMPRESS_HCA)  # informational; see V4Config.validate
COMPRESS_ROPE_THETA = 1.6e5  # compressed layers use their own rope base


def validate_shard(
    samples: int,
    heads: int,
    rank: int,
    npes: int,
    window: int = WINDOW,
    compress_ratio: int = COMPRESS_SWA,
) -> None:
    """Validate the V4 shard contract before allocating GPU buffers."""

    if samples not in SUPPORTED_SAMPLES:
        raise ValueError(f"samples must be one of {SUPPORTED_SAMPLES}, got {samples}")
    if heads not in SUPPORTED_HEADS:
        raise ValueError(f"heads must be one of {SUPPORTED_HEADS}, got {heads}")
    if npes not in SUPPORTED_PEERS:
        raise ValueError(f"npes must be one of {SUPPORTED_PEERS}, got {npes}")
    if not 0 <= rank < npes:
        raise ValueError(f"rank must be in [0, {npes}), got {rank}")
    if window <= 0 or window % 64:
        raise ValueError(f"window must be a positive multiple of 64, got {window}")
    if compress_ratio not in SUPPORTED_COMPRESS:
        raise ValueError(
            f"compress_ratio must be one of {SUPPORTED_COMPRESS}, got {compress_ratio}; "
            "the KV compressor (HCA) and lightning indexer (CSA) are not implemented yet"
        )
