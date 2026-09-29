# SPDX-License-Identifier: Apache-2.0
"""Per-layer latency of the paged (serving) MLA+MoE layer kernel vs batch size, context length and split count.

    python3 tests/kernels/bench_mla_moe_layer_paged.py --splits 32 64 --ctx 1024 9216 131072 --samples 1 4 8
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from kernels.mla_moe_layer.config import KV_LORA, PE_DIM  # noqa: E402
from kernels.mla_moe_layer.layer import SharedReuseMlaMoeLayer  # noqa: E402
from kernels.mla_moe_layer.reference import make_weights, rope_table  # noqa: E402

DS = dict(heads=16, hidden=7168, q_lora=1536, nope_dim=128, v_dim=128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, nargs="+", default=[32])
    ap.add_argument("--ctx", type=int, nargs="+", default=[1024, 9216, 32768, 131072])
    ap.add_argument("--samples", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    W = make_weights(0, device=dev, **DS)
    max_ctx = max(args.ctx) + 8
    cos, sin = rope_table(max_ctx, device=dev)
    rows = 8 * max_ctx
    pool = torch.randn(rows, KV_LORA + PE_DIM, device=dev).to(torch.bfloat16)
    for ns in args.splits:
        for S in args.samples:
            op = SharedReuseMlaMoeLayer(
                W, S, npes=1, topk=ns * 64, paged=True, eps=1e-6, softmax_scale=0.1, n_groups=8, topk_groups=4
            )
            for ctx in args.ctx:
                lens = [ctx] * S
                indptr = torch.tensor([0] + [ctx * (i + 1) for i in range(S)], dtype=torch.int32, device=dev)
                indices = torch.stack([torch.arange(ctx, dtype=torch.int32, device=dev) + i * ctx for i in range(S)]).view(-1)
                pos = torch.full((S,), ctx - 1, dtype=torch.int32, device=dev)
                slot = torch.tensor([i * ctx + ctx - 1 for i in range(S)], dtype=torch.int32, device=dev)
                h = torch.randn(S, DS["hidden"], device=dev).to(torch.bfloat16)
                x = torch.empty_like(h)
                LAYERS = 16  # one decode step's worth of launches per graph; per-layer time = replay / LAYERS
                for _ in range(3):
                    for li in range(LAYERS):
                        op.forward_paged(h, pos, pool, slot, indptr, indices, cos, sin, x_out=x, layer=li, advance=(li == LAYERS - 1))
                torch.cuda.synchronize()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for li in range(LAYERS):
                        op.forward_paged(h, pos, pool, slot, indptr, indices, cos, sin, x_out=x, layer=li, advance=(li == LAYERS - 1))
                for _ in range(3):
                    g.replay()
                torch.cuda.synchronize()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(args.iters):
                    g.replay()
                b.record()
                torch.cuda.synchronize()
                print(f"splits={ns:3d} S={S} ctx={ctx:6d}: {a.elapsed_time(b) / args.iters * 1000 / LAYERS:8.1f} us/layer", flush=True)
            op.close()


if __name__ == "__main__":
    main()
