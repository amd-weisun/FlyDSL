# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Check the Qwen3.8-Flash-Next decode golden against HF transformers' ``Qwen4ExpTextDecoderLayer``.

The golden runs T single-token decode steps from empty state; HF runs the same T tokens as one causal
prefill (chunked delta rule; QSA with its indexer over the prefix). Every step's bundle increment must agree.
Reduced shape with a small indexer budget so selection is sparse; bf16 activations, fp32 HF weights.
"""

from __future__ import annotations

import pytest
import torch

import kernels.monokernel.qwen38.reference as ref_mod
from kernels.monokernel.qwen38.config import Q38Config
from kernels.monokernel.qwen38.reference import golden_layer, make_weights
from kernels.monokernel.reference import dequant

modeling = pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
configuration = pytest.importorskip("transformers.models.qwen4_exp.configuration_qwen4_exp")

pytestmark = [pytest.mark.l2_device, pytest.mark.rocm_lower]

STEPS = 40
CFG = Q38Config(
    hidden=256,
    hc_lowrank=32,
    linear_k_heads=2,
    linear_v_heads=6,
    heads=6,
    kv_heads=2,
    index_budget=16,  # 4 groups of 4: sparse once the prefix passes 19 tokens
    n_experts=16,
    top_k=4,
    inter=128,
    shared_inter=128,
    act_fp8=False,
)


def _hf_layer(W, layer_idx):
    cfg, t = W.cfg, W.t
    hf = configuration.Qwen4ExpTextConfig(
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
        hc_count=cfg.hc,
        hc_lowrank=cfg.hc_lowrank,
        ple_layer_ids=[],
        indexer_n_heads=cfg.index_heads,
        indexer_kv_heads=1,
        indexer_head_dim=cfg.index_dim,
        indexer_budget=cfg.index_budget,
        indexer_compress_ratio=cfg.index_ratio,
        output_gate_type="sigmoid",
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
    layer = modeling.Qwen4ExpTextDecoderLayer(hf, layer_idx).cuda().float()
    sd = {}
    for ours, theirs in (("hca", "attn_hyper_connection."), ("hcm", "mlp_hyper_connection.")):
        sd |= {
            theirs + "hc_norm.weight": t[f"{ours}_norm"],
            theirs + "input_mix_weight_down.weight": t[f"{ours}_down"],
            theirs + "input_mix_weight_up.weight": t[f"{ours}_up"],
            theirs + "block_inject_weight.weight": t[f"{ours}_inj"],
        }
    if W.full:
        a = "self_attn."
        sd |= {a + f"{n}_proj.weight": t[f"w_{n}"] for n in "qkvo"}
        sd |= {a + "q_norm.weight": t["g_qn"], a + "k_norm.weight": t["g_kn"]}
        sd |= {
            a + "indexer.index_qk_proj.weight": t["w_idx"],
            a + "indexer.q_layernorm.weight": t["g_iq"],
            a + "indexer.k_layernorm.weight": t["g_ik"],
        }
    else:
        la, hv = "linear_attn.", cfg.linear_v_heads
        sd |= {
            la + "in_proj_qkv.weight": t["w_qkv"],
            la + "in_proj_z.weight": t["w_z"],
            la + "in_proj_b.weight": t["w_ba"][:hv],
            la + "in_proj_a.weight": t["w_ba"][hv:],
            la + "conv1d.weight": t["w_conv"][:, None, :],
            la + "A_log": t["a_log"],
            la + "dt_bias": t["dt_bias"],
            la + "norm.weight": t["g_norm"],
            la + "out_proj.weight": t["w_out"],
        }
    n, inter = cfg.n_experts, cfg.inter
    sd |= {
        "mlp.gate.weight": t["w_r"],
        "mlp.experts.gate_up_proj": torch.stack([dequant(t["w_ug"][e], t["s_ug"][e], 128) for e in range(n)]),
        "mlp.experts.down_proj": torch.stack([dequant(t["w_dn"][e], t["s_dn"][e], 128) for e in range(n)]),
        "mlp.shared_expert.gate_proj.weight": t["w_sh_ug"][:inter],
        "mlp.shared_expert.up_proj.weight": t["w_sh_ug"][inter:],
        "mlp.shared_expert.down_proj.weight": t["w_sh_dn"],
        "mlp.shared_expert_gate.weight": t["w_sg"],
    }
    missing, unexpected = layer.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    assert not unexpected and not [m for m in missing if "rotary" not in m], (missing, unexpected)
    rotary = modeling.Qwen4ExpTextRotaryEmbedding(hf).cuda()
    return layer, rotary


# exact: the golden with its bf16 roundings off (fp32 throughout) must match HF's fp32 prefill to fp32
# accuracy -- the math check; bf16: the golden as the kernel follows it (ATOM's rounding points)
@pytest.mark.parametrize("exact", [True, False], ids=["fp32", "bf16"])
@pytest.mark.parametrize("full", [False, True], ids=["gdn", "qsa"])
def test_qwen38_golden_matches_hf_prefill(full, exact, monkeypatch):
    if exact:
        monkeypatch.setattr(ref_mod, "bf", lambda x: x.float())
    torch.manual_seed(0)
    W = make_weights(CFG, full)
    cfg = W.cfg
    X = torch.randn(STEPS, cfg.streams, device="cuda").to(torch.bfloat16)

    layer, rotary = _hf_layer(W, 3 if full else 0)
    pos = torch.arange(STEPS, device="cuda")[None]
    x = X.float()[None]
    causal = torch.ones(STEPS, STEPS, device="cuda", dtype=torch.bool).tril()[None, None]
    with torch.no_grad():
        ref = layer(x, position_embeddings=rotary(x, pos), attention_mask=causal, position_ids=pos)
    ref = (ref[0] if isinstance(ref, tuple) else ref)[0]

    conv = torch.zeros(
        1, cfg.conv_kernel - 1, cfg.conv_dim, device="cuda", dtype=torch.float32 if exact else torch.bfloat16
    )
    ssm = torch.zeros(1, cfg.linear_v_heads, cfg.linear_k_dim, cfg.linear_v_dim, device="cuda")
    kc = torch.zeros(1, STEPS, cfg.kv_heads, cfg.head_dim, device="cuda")
    vc = torch.zeros_like(kc)
    ic = torch.zeros(1, STEPS, cfg.index_dim, device="cuda")
    cc = torch.zeros(1, STEPS // cfg.index_ratio + 1, cfg.index_dim, device="cuda")
    slot = torch.zeros(1, dtype=torch.int32, device="cuda")
    errs = []
    for step in range(STEPS):
        if full:
            kw = dict(qsa_state=(kc, vc, ic, cc, torch.tensor([step], device="cuda")))
        else:
            kw = dict(gdn_state=(conv, ssm, slot, slot))
        out = golden_layer(W, X[step : step + 1], **kw)["x_out"][0].float()
        # compare the layer's increment, not the bundle it carries through
        delta, ref_delta = out - X[step].float(), ref[step] - X[step].float()
        errs.append(float((delta - ref_delta).norm() / ref_delta.norm()))
    print("max rel err of the increment", max(errs), "per step", [f"{e:.1e}" for e in errs[::8]])
    # bf16: X is rounded at its own magnitude every combine while the increment is a small part of it, so
    # the increment's relative error carries ~|X| / |increment| x bf16 epsilon; the fp32 case checks the math
    assert max(errs) < (1e-4 if exact else 8e-2), errs
