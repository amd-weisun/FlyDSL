# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side weight packing for the DeepSeek-V4 layer.

The MFMA tile swizzles are identical to the MLA layer's, so they are reused
directly; only the set of attention matrices differs (V4 has no absorbed
``W_UK``/``W_UV``, and its output projection is the grouped low-rank pair
``o_a`` / ``o_b``).
"""

from __future__ import annotations

import torch

from kernels.dsv4_moe_layer.config import ExpertWeight, MoeMode, moe_format
from kernels.mla_moe_layer.packing import pack_bf16, pack_fp8, pack_mxfp4

__all__ = ["pack_bf16", "pack_fp8", "pack_mxfp4", "pack_layer_weights"]

ATTENTION_NAMES = ("w_qkv_a", "w_q_b", "w_o_a", "w_o_b")
# the indexer's query projection, present only on a CSA layer
OPTIONAL_ATTENTION_NAMES = ("w_i_q_b",)
EXPERT_NAMES = ("w_ug", "w_dn")
# hyper-connection mixers: bf16, K = hc_mult * hidden, rows already padded
HC_NAMES = ("hc_attn_fn", "hc_ffn_fn")


def pack_layer_weights(
    tensors: dict[str, torch.Tensor], moe_mode: MoeMode | str = MoeMode.A8W4
) -> dict[str, torch.Tensor]:
    """Pack every matrix consumed by the fused V4 layer kernel."""

    missing = [name for name in (*ATTENTION_NAMES, *EXPERT_NAMES, "w_r") if name not in tensors]
    if missing:
        raise ValueError(f"missing layer weights: {', '.join(missing)}")
    packed = {name: pack_fp8(tensors[name]) for name in ATTENTION_NAMES}
    packed.update({name: pack_fp8(tensors[name]) for name in OPTIONAL_ATTENTION_NAMES if name in tensors})
    weight = moe_format(moe_mode).weight
    pack_expert = pack_mxfp4 if weight is ExpertWeight.MXFP4_BLOCK32 else pack_fp8
    packed.update({name: pack_expert(tensors[name]) for name in EXPERT_NAMES})
    packed["w_r"] = pack_bf16(tensors["w_r"])
    for name in HC_NAMES:
        if name in tensors:
            packed[name] = pack_bf16(tensors[name])
    return packed
