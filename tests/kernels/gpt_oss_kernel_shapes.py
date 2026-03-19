#!/usr/bin/env python3
"""GPT-OSS 120B kernel shape catalog.

Derives the exact kernel input shapes for every layer operation as a function of:
  - tp_size: tensor parallel degree (1, 2, 4, 8)
  - batch_tokens (M): number of tokens in the batch (concurrency)

These shapes are used to benchmark AITER vs FlyDSL kernel-by-kernel under
realistic inference workloads.

Usage:
    # Print all shapes for TP=8
    python gpt_oss_kernel_shapes.py --tp 8

    # Print shapes for specific concurrencies
    python gpt_oss_kernel_shapes.py --tp 8 --tokens 1,32,128,1024
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================================
# GPT-OSS 120B Model Config
# ============================================================================
@dataclass(frozen=True)
class GptOssConfig:
    """GPT-OSS 120B model dimensions (from HuggingFace config)."""
    hidden_size: int = 2880
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 2880      # MoE expert FFN intermediate dim
    num_hidden_layers: int = 36
    num_local_experts: int = 128
    num_experts_per_tok: int = 4       # top-k
    vocab_size: int = 201088

    @property
    def q_output_size(self) -> int:
        return self.num_attention_heads * self.head_dim  # 4096

    @property
    def kv_output_size(self) -> int:
        return self.num_key_value_heads * self.head_dim  # 512

    @property
    def qkv_output_size(self) -> int:
        return self.q_output_size + 2 * self.kv_output_size  # 5120

    @property
    def moe_gate_up_size(self) -> int:
        return 2 * self.intermediate_size  # 5760 (gate + up fused)


# ============================================================================
# Kernel Shape Entry
# ============================================================================
@dataclass
class KernelShape:
    """A single kernel invocation with its shape and metadata."""
    layer: str          # e.g., "qkv_proj", "moe_w13", "rmsnorm"
    kernel_type: str    # e.g., "gemm", "rmsnorm", "attention", "topk", "moe"
    backend: str        # current backend: "hipblaslt", "ck", "triton", "asm"
    M: int              # batch dimension (tokens)
    N: int              # output dimension
    K: int              # input/reduction dimension
    dtype: str          # "bf16", "fp8", "mxfp4"
    count_per_layer: int = 1   # how many times per transformer layer
    notes: str = ""

    @property
    def flops(self) -> int:
        """FLOPs for GEMM (2*M*N*K). 0 for non-GEMM."""
        if self.kernel_type == "gemm":
            return 2 * self.M * self.N * self.K
        return 0

    @property
    def shape_str(self) -> str:
        return f"{self.M}x{self.N}x{self.K}"


# ============================================================================
# Shape Catalog Generator
# ============================================================================
def generate_shapes(
    cfg: GptOssConfig,
    tp_size: int,
    batch_tokens: int,
    ep_size: int = 1,
) -> List[KernelShape]:
    """Generate all kernel shapes for one forward pass.

    Args:
        cfg: Model config
        tp_size: Tensor parallel degree
        batch_tokens: Number of tokens in batch (M dimension)
        ep_size: Expert parallel degree (default 1 = no EP)

    Returns:
        List of KernelShape for every kernel in the forward pass.
    """
    M = batch_tokens
    shapes = []

    # --- Per-rank dimensions ---
    q_size_local = cfg.q_output_size // tp_size
    kv_size_local = cfg.kv_output_size // tp_size
    qkv_size_local = q_size_local + 2 * kv_size_local
    num_heads_local = cfg.num_attention_heads // tp_size
    num_kv_heads_local = cfg.num_key_value_heads // tp_size
    num_experts_local = cfg.num_local_experts // max(ep_size, 1)

    # Avg tokens per expert in MoE (after top-k routing)
    total_routed = M * cfg.num_experts_per_tok
    tokens_per_expert = max(1, total_routed // num_experts_local)

    # =====================================================================
    # 1. Embedding (once, not per layer)
    # =====================================================================
    shapes.append(KernelShape(
        layer="embedding", kernel_type="embedding", backend="triton",
        M=M, N=cfg.hidden_size, K=cfg.vocab_size,
        dtype="bf16", count_per_layer=0,
        notes="masked_embedding_kernel (TP>1) or F.embedding",
    ))

    # =====================================================================
    # 2. Per-layer kernels (×num_hidden_layers)
    # =====================================================================

    # --- Input RMSNorm ---
    shapes.append(KernelShape(
        layer="input_rmsnorm", kernel_type="rmsnorm", backend="ck",
        M=M, N=cfg.hidden_size, K=0,
        dtype="bf16", count_per_layer=1,
        notes="rmsnorm2d_fwd or fused variants",
    ))

    # --- QKV Projection (ColumnParallelLinear) ---
    # Each rank computes full QKV but with TP-split output
    shapes.append(KernelShape(
        layer="qkv_proj", kernel_type="gemm", backend="hipblaslt",
        M=M, N=qkv_size_local, K=cfg.hidden_size,
        dtype="bf16", count_per_layer=1,
        notes=f"tgemm.mm, Q={q_size_local}+K={kv_size_local}+V={kv_size_local}",
    ))

    # --- Attention (prefill: flash_attn, decode: PA) ---
    shapes.append(KernelShape(
        layer="attention_prefill", kernel_type="attention", backend="ck",
        M=M, N=num_heads_local, K=cfg.head_dim,
        dtype="bf16", count_per_layer=1,
        notes=f"flash_attn_varlen_func, heads={num_heads_local}, kv_heads={num_kv_heads_local}",
    ))

    shapes.append(KernelShape(
        layer="attention_decode", kernel_type="attention", backend="asm",
        M=M, N=num_heads_local, K=cfg.head_dim,
        dtype="fp8", count_per_layer=1,
        notes=f"pa_fwd_asm or pa_persistent_fwd, kv_block=16|1024",
    ))

    # --- O Projection (RowParallelLinear) ---
    shapes.append(KernelShape(
        layer="o_proj", kernel_type="gemm", backend="hipblaslt",
        M=M, N=cfg.hidden_size, K=q_size_local,
        dtype="bf16", count_per_layer=1,
        notes="tgemm.mm + AllReduce",
    ))

    # --- Post-attention RMSNorm ---
    shapes.append(KernelShape(
        layer="post_attn_rmsnorm", kernel_type="rmsnorm", backend="ck",
        M=M, N=cfg.hidden_size, K=0,
        dtype="bf16", count_per_layer=1,
        notes="rmsnorm2d_fwd_with_add (residual)",
    ))

    # --- MoE Router Gate (ReplicatedLinear) ---
    shapes.append(KernelShape(
        layer="moe_gate", kernel_type="gemm", backend="hipblaslt",
        M=M, N=cfg.num_local_experts, K=cfg.hidden_size,
        dtype="bf16", count_per_layer=1,
        notes="tgemm.mm, replicated across TP ranks",
    ))

    # --- MoE TopK ---
    shapes.append(KernelShape(
        layer="moe_topk", kernel_type="topk", backend="asm",
        M=M, N=cfg.num_local_experts, K=cfg.num_experts_per_tok,
        dtype="bf16", count_per_layer=1,
        notes="topk_softmax or grouped_topk",
    ))

    # --- MoE W13 Expert Up-Projection (gate+up fused) ---
    shapes.append(KernelShape(
        layer="moe_w13", kernel_type="gemm", backend="ck",
        M=tokens_per_expert, N=cfg.moe_gate_up_size, K=cfg.hidden_size,
        dtype="mxfp4", count_per_layer=1,
        notes=f"fused_moe stage1, {num_experts_local} experts × ~{tokens_per_expert} tok/expert",
    ))

    # --- MoE W2 Expert Down-Projection ---
    shapes.append(KernelShape(
        layer="moe_w2", kernel_type="gemm", backend="ck",
        M=tokens_per_expert, N=cfg.hidden_size, K=cfg.intermediate_size,
        dtype="mxfp4", count_per_layer=1,
        notes=f"fused_moe stage2, {num_experts_local} experts × ~{tokens_per_expert} tok/expert",
    ))

    # =====================================================================
    # 3. Final norm + LM head (once, not per layer)
    # =====================================================================
    shapes.append(KernelShape(
        layer="final_rmsnorm", kernel_type="rmsnorm", backend="ck",
        M=M, N=cfg.hidden_size, K=0,
        dtype="bf16", count_per_layer=0,
        notes="rmsnorm2d_fwd",
    ))

    shapes.append(KernelShape(
        layer="lm_head", kernel_type="gemm", backend="hipblaslt",
        M=M, N=cfg.vocab_size // tp_size, K=cfg.hidden_size,
        dtype="bf16", count_per_layer=0,
        notes="tgemm.mm + AllGather",
    ))

    return shapes


# ============================================================================
# Convenience: typical inference scenarios
# ============================================================================

# Inference scenario: prompt=1K, output=8K
# Decode: M = concurrency (1 token/request/step), decode-dominated (8x more steps)
DECODE_TOKENS = [1, 2, 4, 8, 16, 32, 64, 128]

# Prefill: M = prompt length (processed once per request)
PREFILL_TOKENS = [1024]

# Full sweep: decode concurrency points + prefill
MIXED_TOKENS = DECODE_TOKENS + PREFILL_TOKENS


def generate_sweep(
    tp_size: int = 8,
    tokens_list: Optional[List[int]] = None,
    cfg: Optional[GptOssConfig] = None,
) -> Dict[int, List[KernelShape]]:
    """Generate shapes for multiple concurrency levels.

    Returns:
        Dict mapping batch_tokens -> list of KernelShape.
    """
    if cfg is None:
        cfg = GptOssConfig()
    if tokens_list is None:
        tokens_list = MIXED_TOKENS

    return {t: generate_shapes(cfg, tp_size, t) for t in tokens_list}


# ============================================================================
# Pretty-print
# ============================================================================
def print_shapes(shapes: List[KernelShape], tp_size: int, batch_tokens: int) -> None:
    print(f"\n{'='*100}")
    print(f"GPT-OSS 120B Kernel Shapes — TP={tp_size}, M={batch_tokens} tokens")
    print(f"{'='*100}")
    hdr = f"{'Layer':<22s} {'Type':<12s} {'Backend':<10s} {'M':>6s} {'N':>6s} {'K':>6s} {'Dtype':<6s} {'GFLOPS':>10s} {'Notes'}"
    print(hdr)
    print("-" * 100)
    for s in shapes:
        gf = f"{s.flops / 1e9:10.1f}" if s.flops > 0 else f"{'—':>10s}"
        print(f"{s.layer:<22s} {s.kernel_type:<12s} {s.backend:<10s} "
              f"{s.M:6d} {s.N:6d} {s.K:6d} {s.dtype:<6s} {gf} {s.notes}")
    print(f"{'='*100}")

    # Summary: total GFLOPS per forward pass
    per_layer = [s for s in shapes if s.count_per_layer > 0]
    once = [s for s in shapes if s.count_per_layer == 0]
    layer_flops = sum(s.flops for s in per_layer)
    once_flops = sum(s.flops for s in once)
    total = layer_flops * 36 + once_flops
    print(f"Per-layer GEMM: {layer_flops/1e9:.1f} GFLOPS × 36 layers = {layer_flops*36/1e9:.1f} GFLOPS")
    print(f"One-time GEMM:  {once_flops/1e9:.1f} GFLOPS")
    print(f"Total forward:  {total/1e9:.1f} GFLOPS")


def print_sweep(sweep: Dict[int, List[KernelShape]], tp_size: int) -> None:
    """Print a compact sweep table focusing on GEMM shapes."""
    print(f"\n{'='*120}")
    print(f"GPT-OSS 120B GEMM Shape Sweep — TP={tp_size}")
    print(f"{'='*120}")
    hdr = (f"{'Tokens':>7s} | {'Layer':<18s} | {'M':>6s} {'N':>6s} {'K':>6s} | "
           f"{'Dtype':<6s} | {'Backend':<10s} | {'GFLOPS':>10s}")
    print(hdr)
    print("-" * 120)
    for tokens, shapes in sorted(sweep.items()):
        gemms = [s for s in shapes if s.kernel_type == "gemm"]
        for i, s in enumerate(gemms):
            tok_col = f"{tokens:7d}" if i == 0 else f"{'':7s}"
            gf = f"{s.flops / 1e9:10.1f}"
            print(f"{tok_col} | {s.layer:<18s} | {s.M:6d} {s.N:6d} {s.K:6d} | "
                  f"{s.dtype:<6s} | {s.backend:<10s} | {gf}")
        if tokens != max(sweep.keys()):
            print(f"{'':7s} |{'':19s} |{'':20s} |{'':8s} |{'':12s} |")
    print(f"{'='*120}")


# ============================================================================
# CLI
# ============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="GPT-OSS 120B kernel shape catalog")
    parser.add_argument("--tp", type=int, default=8, choices=[1, 2, 4, 8],
                        help="Tensor parallel degree")
    parser.add_argument("--tokens", type=str, default=None,
                        help="Comma-separated list of token counts (default: mixed scenario)")
    parser.add_argument("--sweep", action="store_true",
                        help="Print compact GEMM sweep table")
    args = parser.parse_args()

    if args.tokens:
        tokens_list = [int(t.strip()) for t in args.tokens.split(",")]
    else:
        tokens_list = MIXED_TOKENS

    cfg = GptOssConfig()

    if args.sweep:
        sweep = generate_sweep(tp_size=args.tp, tokens_list=tokens_list, cfg=cfg)
        print_sweep(sweep, args.tp)
    else:
        for t in tokens_list:
            shapes = generate_shapes(cfg, args.tp, t)
            print_shapes(shapes, args.tp, t)


if __name__ == "__main__":
    main()
