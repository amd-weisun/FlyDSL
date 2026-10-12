# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Qwen3.5-397B-A17B dimensions and one rank's tensor-parallel shard of a decoder layer.

The model alternates three Gated DeltaNet (linear attention) layers with one gated full-attention
layer; every layer ends in a 512-expert top-10 MoE with a sigmoid-gated shared expert.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

HIDDEN = 4096
N_LAYERS = 60
FULL_ATTENTION_INTERVAL = 4
EPS = 1.0e-6
FP8_MAX = 448.0

# Gated DeltaNet
LINEAR_K_HEADS = 16
LINEAR_V_HEADS = 64
LINEAR_K_DIM = 128
LINEAR_V_DIM = 128
CONV_KERNEL = 4

# gated full attention
HEADS = 32
KV_HEADS = 2
HEAD_DIM = 256
ROPE_DIM = 64  # partial_rotary_factor 0.25
ROPE_THETA = 1.0e7

# MoE
N_EXPERTS = 512
TOP_K = 10
INTER = 1024
SHARED_INTER = 1024


def is_full_attention(layer_id: int) -> bool:
    return (layer_id + 1) % FULL_ATTENTION_INTERVAL == 0


@dataclass(frozen=True)
class Q35Config:
    """Rank ``rank`` of a ``tp``-way shard. Dimension fields are model-global; properties give the shard."""

    tp: int = 4
    rank: int = 0
    hidden: int = HIDDEN
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
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    inter: int = INTER
    shared_inter: int = SHARED_INTER
    eps: float = EPS
    # FP8 per-1x128 activations into the expert GEMMs (W8A8, as ATOM); False = bf16. Projections take bf16.
    act_fp8: bool = True
    # full-attention K/V rounded through per-token-per-head FP8, as ATOM's fp8 KV cache stores them
    kv_fp8: bool = True

    def at_rank(self, rank: int) -> "Q35Config":
        return replace(self, rank=rank)

    # --- Gated DeltaNet shard
    @property
    def hk(self) -> int:
        return self.linear_k_heads // self.tp

    @property
    def hv(self) -> int:
        return self.linear_v_heads // self.tp

    @property
    def v_per_k(self) -> int:
        return self.linear_v_heads // self.linear_k_heads

    @property
    def key_dim(self) -> int:
        return self.hk * self.linear_k_dim

    @property
    def value_dim(self) -> int:
        return self.hv * self.linear_v_dim

    @property
    def conv_dim(self) -> int:
        """Channels of the local ``[q | k | v]`` conv input."""
        return 2 * self.key_dim + self.value_dim

    # --- full-attention shard
    @property
    def local_heads(self) -> int:
        return self.heads // self.tp

    @property
    def local_kv_heads(self) -> int:
        """K/V heads are replicated once ``tp`` exceeds them: each rank keeps one."""
        return max(1, self.kv_heads // self.tp)

    @property
    def kv_head(self) -> int:
        """First global K/V head this rank holds."""
        return self.rank * self.kv_heads // self.tp if self.tp >= self.kv_heads else self.rank * self.local_kv_heads

    # --- MoE shard (tensor-parallel over the intermediate dim)
    @property
    def local_inter(self) -> int:
        return self.inter // self.tp

    @property
    def local_shared_inter(self) -> int:
        return self.shared_inter // self.tp

    @property
    def shared_expert(self) -> int:
        """The shared expert is the last entry of the expert bank, as ATOM fuses it."""
        return self.n_experts

    def validate(self) -> None:
        assert self.linear_k_heads % self.tp == 0 and self.linear_v_heads % self.linear_k_heads == 0
        assert self.heads % self.tp == 0
        assert self.tp % self.kv_heads == 0 or self.kv_heads % self.tp == 0
        assert self.local_inter == self.local_shared_inter, "the shared expert shares the bank's shape"
        assert self.local_inter % 128 == 0 and self.hidden % 128 == 0
