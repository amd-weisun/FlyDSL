# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the Qwen3.5 decode golden against HF transformers' ``Qwen3_5MoeDecoderLayer``.

The golden runs T single-token decode steps from empty state; HF runs the same T tokens as one causal
prefill (chunked delta rule, full causal attention). Every step's output must agree. Reduced TP1 shape,
bf16 activations, fp32 HF weights from the golden's dequantized ones.
"""

from __future__ import annotations

import pytest
import torch

from kernels.monokernel.qwen35.config import Q35Config
from kernels.monokernel.qwen35.reference import golden_layer, make_weights
from kernels.monokernel.reference import dequant

modeling = pytest.importorskip("transformers.models.qwen3_5_moe.modeling_qwen3_5_moe")
configuration = pytest.importorskip("transformers.models.qwen3_5_moe.configuration_qwen3_5_moe")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

STEPS = 7
CFG = Q35Config(
    tp=1,
    hidden=256,
    linear_k_heads=2,
    linear_v_heads=8,
    heads=4,
    kv_heads=2,
    n_experts=16,
    top_k=4,
    inter=128,
    shared_inter=128,
    act_fp8=False,
    kv_fp8=False,
)


def _hf_layer(W, layer_idx):
    cfg, t = W.cfg, W.t
    hf = configuration.Qwen3_5MoeTextConfig(
        hidden_size=cfg.hidden,
        num_hidden_layers=4,
        num_attention_heads=cfg.heads,
        num_key_value_heads=cfg.kv_heads,
        head_dim=cfg.head_dim,
        linear_num_key_heads=cfg.linear_k_heads,
        linear_num_value_heads=cfg.linear_v_heads,
        linear_key_head_dim=cfg.linear_k_dim,
        linear_value_head_dim=cfg.linear_v_dim,
        linear_conv_kernel_dim=cfg.conv_kernel,
        num_experts=cfg.n_experts,
        num_experts_per_tok=cfg.top_k,
        moe_intermediate_size=cfg.inter,
        shared_expert_intermediate_size=cfg.shared_inter,
        rms_norm_eps=cfg.eps,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": cfg.rope_theta,
            "partial_rotary_factor": cfg.rope_dim / cfg.head_dim,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    hf._attn_implementation = "sdpa"
    layer = modeling.Qwen3_5MoeDecoderLayer(hf, layer_idx).cuda().float()
    deq = lambda n: dequant(t[f"w_{n}"], t[f"s_{n}"], 128)  # noqa: E731
    sd = {"input_layernorm.weight": t["g_in"], "post_attention_layernorm.weight": t["g_post"]}
    if W.full:
        sd |= {f"self_attn.{n}_proj.weight": deq(n) for n in "qkvo"}
        sd |= {"self_attn.q_norm.weight": t["g_qn"], "self_attn.k_norm.weight": t["g_kn"]}
    else:
        la = "linear_attn."
        sd |= {
            la + "in_proj_qkv.weight": deq("qkv"),
            la + "in_proj_z.weight": deq("z"),
            la + "in_proj_b.weight": t["w_ba"][: cfg.hv],
            la + "in_proj_a.weight": t["w_ba"][cfg.hv :],
            la + "conv1d.weight": t["w_conv"][:, None, :],
            la + "A_log": t["a_log"],
            la + "dt_bias": t["dt_bias"],
            la + "norm.weight": t["g_norm"],
            la + "out_proj.weight": deq("out"),
        }
    n, inter = cfg.n_experts, cfg.local_inter
    ug = torch.stack([dequant(t["w_ug"][e], t["s_ug"][e], 128) for e in range(n + 1)])
    dn = torch.stack([dequant(t["w_dn"][e], t["s_dn"][e], 128) for e in range(n + 1)])
    sd |= {
        "mlp.gate.weight": t["w_r"],
        "mlp.experts.gate_up_proj": ug[:n],
        "mlp.experts.down_proj": dn[:n],
        "mlp.shared_expert.gate_proj.weight": ug[n, :inter],
        "mlp.shared_expert.up_proj.weight": ug[n, inter:],
        "mlp.shared_expert.down_proj.weight": dn[n],
        "mlp.shared_expert_gate.weight": t["w_sg"],
    }
    missing, unexpected = layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    rotary = modeling.Qwen3_5MoeTextRotaryEmbedding(hf).cuda()
    return layer, rotary


@pytest.mark.parametrize("full", [False, True], ids=["gdn", "full_attention"])
def test_qwen35_golden_matches_hf_prefill(full):
    torch.manual_seed(0)
    W = make_weights(CFG, full)
    h = torch.randn(STEPS, CFG.hidden, device="cuda").to(torch.bfloat16)

    layer, rotary = _hf_layer(W, 3 if full else 0)
    pos = torch.arange(STEPS, device="cuda")[None]
    x = h.float()[None]
    with torch.no_grad():
        ref = layer(x, position_embeddings=rotary(x, pos), position_ids=pos)
    ref = (ref[0] if isinstance(ref, tuple) else ref)[0]

    cfg = W.cfg
    conv = torch.zeros(1, cfg.conv_kernel - 1, cfg.conv_dim, device="cuda", dtype=torch.bfloat16)
    ssm = torch.zeros(1, cfg.hv, cfg.linear_v_dim, cfg.linear_k_dim, device="cuda", dtype=torch.bfloat16)
    kc = torch.zeros(1, STEPS, cfg.local_kv_heads, cfg.head_dim, device="cuda")
    vc = torch.zeros_like(kc)
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    errs = []
    for step in range(STEPS):
        kw = dict(kv=(kc, vc, torch.tensor([step], device="cuda"))) if full else dict(gdn_state=(conv, ssm, slot, slot))
        out = golden_layer(W, h[step : step + 1], lambda y: y, **kw)["x_out"][0].float()
        # compare the layer's increment, not the residual it carries through
        delta, ref_delta = out - h[step].float(), ref[step] - h[step].float()
        errs.append(float((delta - ref_delta).norm() / ref_delta.norm()))
    assert max(errs) < 2e-2, errs
