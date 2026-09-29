# DeepSeek-R1 / V3 decode on ATOM with the FlyDSL MLA+MoE mega kernel

Decode of DeepSeek-R1-0528 (TP8) in ATOM, with layers 3-60 (MLA attention + MoE, including both TP all-reduces) — and,
for MTP, the draft layer — executed as **one persistent FlyDSL kernel per layer**. Prefill, the 3 dense layers, embedding,
LM head and sampling are unchanged ATOM. Decode batches the kernel does not cover (see *Limits*) run ATOM's stock path.

## What was implemented

**FlyDSL** (`kernels/mla_moe_layer/`): the persistent MLA+MoE layer kernel, extended for serving:
- *paged mode*: S independent sequences (one new token each) over ATOM's paged MLA KV pool — rows of 576 values
  (512 latent + 64 rope), addressed by ATOM's CSR `kv_indptr`/`kv_indices`, with per-sample RoPE positions and cache slots;
  CUDA-graph padded samples (slot -1, empty range) are handled.
- *dynamic context up to 128K*: each of a fixed number of KV splits walks its share of 64-key chunks with an online-softmax
  merge, so one compiled kernel / captured graph serves any length.
- *fp8 KV*: E4M3FN rows (unit scale) are dequantized on gather; the new token's row is stored saturated.
- *speculative-decode verification*: Q consecutive tokens per sequence; later tokens see the sequence's earlier new rows.
- DeepSeek-V3 parameters: 16 heads per rank, group-limited routing, `eps`, YaRN softmax scale; and a fix for hidden size 7168 at
  more than one sample per launch (the original down-projection tiling was only valid for hidden size 6144).

**ATOM** (`atom/model_ops/dsv3_megakernel.py` plus small hooks in `deepseek_v2.py`, `deepseek_mtp.py`, `model_runner.py`,
`utils/envs.py`): reads the checkpoint shards for the TP shard of each MoE layer, packs them into the kernel's layout (cached
on node-local disk), and replaces the decode forward of layers 3-60 with one kernel launch per layer. The MTP draft layer is
routed through the kernel via a custom op. Prefill and uncovered batch shapes take ATOM's normal path. Enabled with
`ATOM_DSV3_MEGAKERNEL=1`.

## Setup

| | |
|---|---|
| Hardware | 8x AMD MI355X (gfx950), TP8, one node |
| Container | `rocm/atom-dev:latest` (ROCm 7.2.4, torch 2.10, aiter bundled with the image) |
| Model | `deepseek-ai/DeepSeek-R1-0528` (FP8 128x128 block weights) |
| ATOM | branch `atom_dsv3_megakernel`, based on the ATOM commit shipped in the image (`5b78f17b`) |
| FlyDSL | branch `atom_dsv3_megakernel` (kernel: `kernels/mla_moe_layer/`) |
| KV cache | fp8 (E4M3FN, unit scale) — the recipe default. bf16 KV is also supported |
| Method | ATOM `benchmark_serving` and `lm_eval` exactly as in `recipes/DeepSeek-R1.md`; baseline = the same ATOM without the kernel |

Metrics: **TPOT** = median time per output token (ms); **out tok/s** = output throughput over the run. Every run
uses 4 warmup requests. ISL/OSL use `--random-range-ratio=0.8` (lengths in [0.8x, 1x]), `--num-prompts = 10 x concurrency`.

## 1. Decode performance, no MTP (fp8 KV, ISL 8192 / OSL 1024)

| concurrency | baseline TPOT | mega TPOT | speedup | baseline out tok/s | mega out tok/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 12.70 ms | 4.18 ms | 3.04x | 76 | 210 | 2.76x |
| 2 | 12.09 ms | 5.33 ms | 2.27x | 155 | 334 | 2.15x |
| 4 | 12.49 ms | 6.90 ms | 1.81x | 310 | 541 | 1.74x |
| 8 | 13.81 ms | 11.07 ms | 1.25x | 555 | 694 | 1.25x |

Time to first token is unchanged (~230-250 ms median): prefill is not touched.

## 2. Decode performance with MTP3 (3 speculative tokens, fp8 KV, ISL 8192 / OSL 1024)

Target verification (4 tokens per sequence) **and** the MTP draft layer run on the kernel.

| concurrency | baseline MTP3 TPOT | mega MTP3 TPOT | speedup | baseline out tok/s | mega out tok/s | speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.11 ms | 2.17 ms | 2.82x | 154 | 392 | 2.55x |
| 2 | 4.73 ms | 3.05 ms | 1.55x | 359 | 525 | 1.46x |
| 4 | 5.12 ms | 4.84 ms | 1.06x | 672 | 702 | 1.04x |
| 8 | 6.62 ms | 6.57 ms | 1.01x | 1042 | 1061 | 1.02x |

The kernel handles at most 8 tokens per step, so MTP3 (4 tokens per sequence) is covered for 1-2 sequences; concurrency 4
and 8 run the stock path (no gain, no loss). Draft acceptance rate: 76.4% (mega) vs 74.6% (baseline).

## 3. Long context (fp8 KV, no MTP)

**ISL 32768 / OSL 1024**

| concurrency | baseline TPOT | mega TPOT | speedup | baseline out tok/s | mega out tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 12.92 ms | 7.44 ms | 1.74x | 72 | 115 |
| 2 | 13.37 ms | 9.24 ms | 1.45x | 134 | 190 |
| 4 | 15.50 ms | 12.53 ms | 1.24x | 233 | 278 |

The gain shrinks with context because the baseline's step time is nearly flat while the kernel's attention part grows with
context. The kernel's context length is dynamic (any length up to 128K, one compiled kernel / CUDA graph).
Per-layer kernel time at batch 1 (64 KV splits): 40 us @1K, 50 us @9K, 76 us @32K, 212 us @128K.

**Needle-in-a-haystack retrieval** (greedy; a code hidden mid-prompt, asked for at the end): baseline and mega both pass at
6.6K, 30K, 61K and 116K prompt tokens, with identical answers.

## 4. Accuracy — GSM8K 5-shot (`lm_eval`, recipe command), stderr about +-0.0056

| configuration | num_concurrent | flexible-extract | strict-match |
|---|---:|---:|---:|
| baseline, fp8 KV | 8 | 0.9560 | 0.9530 |
| baseline, fp8 KV | 64 | 0.9530 | 0.9515 |
| **mega**, fp8 KV | 8 | 0.9560 | 0.9553 |
| baseline MTP3, fp8 KV | 2 | 0.9560 | 0.9568 |
| **mega** MTP3 (target + draft on the kernel), fp8 KV | 2 | 0.9583 | 0.9545 |
| recipe reference (fp8 KV) / CI threshold | 64 | 0.9553 / >= 0.94 | 0.9538 |

Mega runs use concurrency <= 8 (MTP3: 2) so that decode steps actually run on the kernel; at concurrency 64 nearly every
step is above the kernel's batch limit and takes the stock path.

## 5. Limits

- Kernel batch limit: at most **8 tokens per step**, in {1, 2, 4, 8}. Plain decode: up to 8 sequences; MTP3 (4 tokens/seq): up to 2
  sequences; MTP1: up to 4. MTP2 (3 tokens/seq) is not supported. Larger batches fall back to ATOM's stock path automatically.
- KV cache: bf16, or fp8 with unit scale (ATOM's default scale). TP8 only. DeepSeek-V3-style dims (heads 128, 256+1 experts, group routing).
- The kernel is not NaN-tolerant across samples: padded rows of a CUDA-graph batch can hold NaN/Inf, and the ATOM integration
  zeroes non-finite hidden values before the kernel layers (a kernel unit test records this as an expected failure).
- Occasional single-step stalls of 0.6-2 s were seen in some earlier mega runs (cause unknown); the runs above show none that
  affect the median, and mean TPOT stays within ~13% of the median.

## 6. Reproduce

### 6.1 Checkout, container, environment

```bash
export WORK=<directory for the checkouts>  MODEL_ROOT=<directory containing the DeepSeek-R1-0528 checkpoint>
# Branch `atom_dsv3_megakernel` of each repo (ATOM: ROCm/ATOM based on the image's commit; FlyDSL: ROCm/FlyDSL)
git clone -b atom_dsv3_megakernel <ATOM-repo-url>    $WORK/ATOM
git clone -b atom_dsv3_megakernel <FlyDSL-repo-url>  $WORK/FlyDSL
# FlyDSL's compiled MLIR bindings: build once as described in the FlyDSL docs (scripts/build_llvm.sh, scripts/build.sh).
# This work changes only Python kernels, not the C++ dialects, so any FlyDSL build of the same base works.

docker run -d --name atom-dsv3 --network=host --ipc=host --privileged \
  --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v $WORK:$WORK -v $MODEL_ROOT:$MODEL_ROOT --shm-size=64G --ulimit memlock=-1 --ulimit stack=67108864 \
  --entrypoint bash rocm/atom-dev:latest -c 'sleep infinity'
docker exec -it atom-dsv3 bash

# inside the container
export MODEL=$MODEL_ROOT/DeepSeek-R1-0528     # local copy of deepseek-ai/DeepSeek-R1-0528 (or the Hugging Face id)
export FLYDSL_BUILD=$WORK/FlyDSL/build-fly/python_packages
export PYTHONPATH=$WORK/ATOM:$WORK/FlyDSL:$FLYDSL_BUILD:$PYTHONPATH
export LD_LIBRARY_PATH=$FLYDSL_BUILD/flydsl/_mlir/_mlir_libs:$LD_LIBRARY_PATH
export AITER_LOG_LEVEL=WARNING
rm -rf /root/.cache/atom/torch_compile_cache        # after changing model code, ATOM reuses cached compiled graphs
```

### 6.2 Servers (one at a time; port 8000)

```bash
# baseline (stock ATOM)
python -m atom.entrypoints.openai_server --model $MODEL --kv_cache_dtype fp8 -tp 8

# mega kernel
ATOM_DSV3_MEGAKERNEL=1 python -m atom.entrypoints.openai_server --model $MODEL --kv_cache_dtype fp8 -tp 8

# MTP3: add to either command
    --method mtp --num-speculative-tokens 3
```

The first mega launch builds each rank's packed kernel weights from the checkpoint and caches them on node-local disk
(`ATOM_DSV3_MEGAKERNEL_CACHE`, default `/root/mega_cache`, ~1.5 GB per layer per rank, ~700 GB in total): ~13 min once, then
~2.5 min per launch. Other switches: `ATOM_DSV3_MEGAKERNEL_SPLITS` (KV splits, default `auto`), `ATOM_DSV3_MEGAKERNEL_CHECK=1`
(debug: compare the kernel against ATOM's own layer, eager mode only).

### 6.3 Performance (recipe benchmark command; ISL=8192 OSL=1024, CONC in 1 2 4 8; ISL=32768 for section 3)

```bash
ISL=8192; OSL=1024
for CONC in 1 2 4 8; do
  python -m atom.benchmarks.benchmark_serving \
    --model=$MODEL --backend=vllm --base-url=http://localhost:8000 \
    --dataset-name=random --random-input-len=$ISL --random-output-len=$OSL \
    --random-range-ratio=0.8 --num-prompts=$(( CONC * 10 )) --max-concurrency=$CONC \
    --request-rate=inf --ignore-eos --num-warmups 4 \
    --save-result --result-dir ./bench --result-filename isl${ISL}_osl${OSL}_c${CONC}.json \
    --percentile-metrics="ttft,tpot,itl,e2el"
done
```

### 6.4 Accuracy (recipe lm_eval command)

```bash
NC=8      # baseline also 64; MTP3: 2
lm_eval --model local-completions \
  --model_args model=$MODEL,base_url=http://localhost:8000/v1/completions,num_concurrent=$NC,max_retries=3,tokenized_requests=False \
  --tasks gsm8k --num_fewshot 5
```

### 6.5 Needle-in-a-haystack (section 3), against the running server

```python
# needle.py — python needle.py
import json, time, urllib.request
from transformers import AutoTokenizer
MODEL = "deepseek-ai/DeepSeek-R1-0528"   # or the local path used for --model; the server must serve the same name
tok = AutoTokenizer.from_pretrained(MODEL)
FILL = ("The grass is green and the sky is blue. Rivers flow to the sea while clouds drift above quiet hills. "
        "Farmers plant seeds in spring and harvest in autumn, and children play near the old stone bridge. ")
def build(n_tokens, code):
    per = len(tok.encode(FILL, add_special_tokens=False))
    reps = max(n_tokens // per - 30, 1); half = reps // 2
    text = FILL * half + f" IMPORTANT: The secret code is {code}. Remember it. " + FILL * (reps - half)
    return text + "\n\nQuestion: What is the secret code mentioned in the text above?\nAnswer: The secret code is"
def ask(prompt):
    req = urllib.request.Request("http://localhost:8000/v1/completions", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": 12, "temperature": 0}).encode())
    return json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["text"]
for n, code in ((8000, "48213"), (32000, "90517"), (64000, "27364"), (120000, "73856")):
    t = time.time(); out = ask(build(n, code))
    print(n, "PASS" if code in out else "FAIL", repr(out), f"{time.time()-t:.1f}s")
```

### 6.6 Kernel tests and micro-benchmark (FlyDSL checkout; a single GPU is enough for the paged tests)

```bash
cd $WORK/FlyDSL
python3 -m pytest tests/kernels/test_shared_reuse_mla_moe_layer.py tests/kernels/test_mla_moe_layer_host.py \
                  tests/kernels/test_mla_moe_layer_paged.py -q        # 61 passed, 2 xfailed
python3 tests/kernels/bench_mla_moe_layer_paged.py --splits 32 64 --ctx 1024 9216 32768 131072 --samples 1 4 8
```

The paged tests cover independent sequences, mixed and padded batches, contexts up to 131072, the fp8 KV pool and MTP
verification (several tokens per sequence), each against a torch golden.
