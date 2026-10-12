# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Qwen3.8-Flash-Next (HF ``qwen4_exp``) dimensions, served at TP1.

Three Gated DeltaNet layers per QSA (sparse full-attention) layer; every layer carries four hyper-connection
residual streams (no layer norms) and ends in a 512-expert top-10 MoE with a sigmoid-gated BF16 shared expert.
"""

from __future__ import annotations

from dataclasses import dataclass

HIDDEN = 2560
N_LAYERS = 48
FULL_ATTENTION_INTERVAL = 4
EPS = 1.0e-6

# hyper-connections
HC = 4
HC_LOWRANK = 320

# Gated DeltaNet
LINEAR_K_HEADS = 16
LINEAR_V_HEADS = 48
LINEAR_K_DIM = 128
LINEAR_V_DIM = 128
CONV_KERNEL = 4

# QSA
HEADS = 24
KV_HEADS = 2
HEAD_DIM = 256
ROPE_DIM = 64  # partial_rotary_factor 0.25
ROPE_THETA = 1.0e7
INDEX_HEADS = 4
INDEX_DIM = 128
INDEX_BUDGET = 2048  # tokens
INDEX_RATIO = 4  # tokens per compressed key

# MoE
N_EXPERTS = 512
TOP_K = 10
INTER = 640
SHARED_INTER = 640


def is_full_attention(layer_id: int) -> bool:
    return (layer_id + 1) % FULL_ATTENTION_INTERVAL == 0


@dataclass(frozen=True)
class Q38Config:
    hidden: int = HIDDEN
    hc: int = HC
    hc_lowrank: int = HC_LOWRANK
    linear_k_heads: int = LINEAR_K_HEADS
    linear_v_heads: int = LINEAR_V_HEADS
    linear_k_dim: int = LINEAR_K_DIM
    linear_v_dim: int = LINEAR_V_DIM
    conv_kernel: int = CONV_KERNEL
    heads: int = HEADS
    kv_heads: int = KV_HEADS
    head_dim: int = HEAD_DIM
    rope_dim: int = ROPE_DIM
    rope_theta: float = ROPE_THETA
    index_heads: int = INDEX_HEADS
    index_dim: int = INDEX_DIM
    index_budget: int = INDEX_BUDGET
    index_ratio: int = INDEX_RATIO
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER
    shared_inter: int = SHARED_INTER
    eps: float = EPS
    # FP8 per-1x128 activations into the routed experts (W8A8, as ATOM); False = bf16
    act_fp8: bool = True

    @property
    def streams(self) -> int:
        """Width of the hyper-connection bundle."""
        return self.hc * self.hidden

    @property
    def v_per_k(self) -> int:
        return self.linear_v_heads // self.linear_k_heads

    @property
    def key_dim(self) -> int:
        return self.linear_k_heads * self.linear_k_dim

    @property
    def value_dim(self) -> int:
        return self.linear_v_heads * self.linear_v_dim

    @property
    def conv_dim(self) -> int:
        return 2 * self.key_dim + self.value_dim

    @property
    def block_topk(self) -> int:
        """Compressed groups the indexer keeps."""
        return self.index_budget // self.index_ratio

    def validate(self) -> None:
        assert self.linear_v_heads % self.linear_k_heads == 0 and self.heads % self.kv_heads == 0
        assert self.inter % 128 == 0 and self.hidden % 128 == 0
