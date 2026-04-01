---
name: review-kernel-style
description: >
  Code review for FlyDSL kernel implementation style. Checks that kernels use
  the layout API (make_buffer_tensor, flat_divide/zipped_divide, tiled_copy,
  copy_atom_call) for all address computation and data movement, and flags
  unnecessary use of raw buffer_load/buffer_store for vector access.
  Use when implementing new kernels , extending existing kernels  and before a merge request.
  Usage: /review-kernel-style [file_or_kernel_name]
tools: Read,Grep,Glob,Agent
---

# FlyDSL Kernel Style Review

Review the target kernel file against the FlyDSL preferred implementation style.

**Canonical references** (read these first if unsure about a pattern):
- `FlyDSL/examples/01-vectorAdd.py` — 1D copy: `logical_divide` + `slice` + `copy_atom_call`
- `FlyDSL/examples/02-tiledCopy.py` — 2D copy: `zipped_divide` + `make_tiled_copy` + `partition_S/D`
- `FlyDSL/examples/03-tiledMma.py` — GEMM: `zipped_divide` + `TiledMma` + `retile` + `fx.gemm`
- `FlyDSL/examples/04-preshuffle_gemm.py` — pipelined GEMM: `flat_divide` + multi-stage copy

If no file is specified, ask the user which kernel to review before proceeding.

---

## Step 0: Read the Target File

Read the kernel file in full. Identify:
- All `@flyc.kernel` functions and their argument lists
- All `@flyc.jit` launcher functions
- All module-level helper functions

---

## Step 1: Tensor-to-Buffer-Tensor Conversion

**Rule**: All `fx.Tensor` kernel arguments accessed for vector loads or stores must be
converted with `fx.rocdl.make_buffer_tensor` at the top of the kernel body, before any
layout operations.

**Canonical pattern** (`04-preshuffle_gemm.py:27-29`, `03-tiledMma.py:27-29`,
`02-tiledCopy.py:22-23`, `01-vectorAdd.py:21`):
```python
A = fx.rocdl.make_buffer_tensor(A)
B = fx.rocdl.make_buffer_tensor(B)
C = fx.rocdl.make_buffer_tensor(C)
```
All four examples do this for every vector-accessed tensor.

**What to flag**:
- `buffer_ops.create_buffer_resource(tensor, ...)` used for vector loads/stores on a
  regular stride-contiguous tensor — replace with `make_buffer_tensor`.
- `make_buffer_tensor` called *after* `create_buffer_resource` on the same Python name
  — the resource captures the wrong type. `create_buffer_resource` must come before
  rebinding.

**Correct exception**: `create_buffer_resource` + `buffer_load(vec_width=1)` is the
established pattern for tensors where only a single scalar element per thread is loaded
at a runtime-computed index. Confirmed in `kernels/pa_decode_fp8.py` (e.g., `bt_rsrc`
for block-table lookup, `cl_rsrc` for context-length lookup). The examples/ directory
has no scalar-load tensors. Do NOT flag scalar-load tensors.




---

## Step 2: Block Index Access

**Rule**: When multiple block dimensions are used (e.g., both M and N tile indices),
use tuple destructuring for `fx.block_idx`.

**Canonical pattern** (`04-preshuffle_gemm.py:25`):
```python
bid_x, bid_y, _ = fx.block_idx
```

For kernels with a single block dimension, `bid = fx.block_idx.x` is also used across
the examples (`01-vectorAdd.py:17`, `02-tiledCopy.py:16`, `03-tiledMma.py:21`) and
is acceptable.

Block index decomposition into semantic coordinates stays as plain `//` and `%`
arithmetic. `fx.idx2crd` is for **thread** index decomposition (wave_id / lane_id
from `tid`), as shown in `kernels/preshuffle_gemm.py:407-410`. Do NOT use `idx2crd`
on block indices.

---

## Step 3: Tile Selection via Divide Operations

**Rule**: Use layout-API divide operations to select the current program's tile from a
buffer tensor. Do not compute byte or dword offsets manually.

**Canonical patterns**:

```python
# zipped_divide: select a 2D block tile at a static block index (bid)
# (02-tiledCopy.py:25-27, 03-tiledMma.py:31-35)
tile = fx.make_tile(BLOCK_M, BLOCK_K)
bA   = fx.zipped_divide(A, tile)
bA   = fx.slice(bA, (None, bid))        # pin this block's tile

# flat_divide: select tile leaving K-outer as a loop dimension
# (04-preshuffle_gemm.py:31-33)
gA_k = fx.flat_divide(A, fx.make_tile(BLOCK_M, BLOCK_K))[None, None, bid_x, None]
gC   = fx.flat_divide(C, fx.make_tile(BLOCK_M, BLOCK_N))[None, None, bid_x, bid_y]
```

**What to flag**:
- Manual byte or dword offset computation (`* elem_bytes`, `>> 2`, `// 4`)
- Helper functions that convert multi-dim coordinates to dword offsets
- `buffer_ops.buffer_load(rsrc, dword_offset, vec_width=N)` for vector-width loads
  where a contiguous tile could be selected with a divide operation instead

**Rank-matching requirement**: `flat_divide` and `zipped_divide` require the tile rank
to match the tensor rank. A 2D tensor needs a 2D tile; a 3D tensor needs a 3D tile.
Using a lower-rank tile (e.g., `make_tile(head_dim)` on a 3D tensor) causes a C++
assertion failure: `intTupleZip2By expects rank-2 tuple at terminal`. Consequence: for
a 3D tensor `[T, QH, D]` where each program selects by two block-derived indices
`(pid_t, pid_hq)`, you need `fx.make_tile(1, 1, head_dim)` (3D tile), which produces a
6-mode result requiring 6-element slice indexing `[None, pid_t, None, pid_hq, None, 0]`.
No canonical example of this pattern exists — flag and escalate to human reviewers.

**`logical_divide` on multi-dimensional tensors**: `logical_divide` divides the
**first** mode of the tensor, not the flat linear address space. Applying
`logical_divide(Q[T,QH,D], layout(block_dim, 1))` divides the T dimension, not the
flattened T×QH space. Only use `logical_divide` on 1D tensors (as in `01-vectorAdd.py`)
or when explicitly dividing the outermost mode.

**Current limitation**: All divide + index examples in the codebase use static block
indices (`bid`, `bid_x`, `bid_y`). If the tile index is computed from a runtime-loaded
value, there is no canonical example; flag the manual offset and note it for the author
to discuss with human reviewers.

---

## Step 4: Tiled Copy for Vec Load/Store

**Rule**: Use `make_tiled_copy` → `get_slice(tid)` → `partition_S/D` →
`make_fragment_like` → `fx.copy` for vectorized global-memory loads and stores.

**Canonical pattern** (`02-tiledCopy.py:30-46`):
```python
copy_atom    = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
thr_layout   = fx.make_layout((4, 1), (1, 1))
val_layout   = fx.make_layout((1, 8), (1, 1))
layout_tv    = fx.raked_product(thr_layout, val_layout)
tiled_copy   = fx.make_tiled_copy(copy_atom, layout_tv, fx.make_tile(4, 8))
thr_copy     = tiled_copy.get_slice(tid)

partition_src = thr_copy.partition_S(bA)
partition_dst = thr_copy.partition_D(bB)
frag          = fx.make_fragment_like(partition_src)

fx.copy(copy_atom, partition_src, frag)   # global → register
fx.copy(copy_atom, frag, partition_dst)   # register → global
```

**What to flag**:
- `buffer_ops.buffer_load(rsrc, dw_offset, vec_width=N)` for regular-stride vector
  loads — replace with `tiled_copy` + `fx.copy`
- `buffer_ops.buffer_store(data, rsrc, dw_offset)` for regular-stride vector stores
  — replace with `tiled_copy` + `fx.copy`

---

## Step 5: Per-Thread Scalar Copy with copy_atom_call

**Rule**: For simple per-element copies where each thread handles one scalar or one
fixed-width slice at its own thread index, use `logical_divide` + `slice(t, (None, tid))`
+ `copy_atom_call` — the simpler alternative to a full `make_tiled_copy`.

**Canonical pattern** (`01-vectorAdd.py:23-44`):
```python
# Divide the block tile into per-thread slices, then copy_atom_call at tid
tA = fx.logical_divide(A, fx.make_layout(block_dim, 1))
tA = fx.slice(tA, (None, bid))                          # select this block
tA = fx.logical_divide(tA, fx.make_layout(1, 1))        # per-element partition

rA = fx.memref_alloca(RABMemRefTy, fx.make_layout(1, 1))
fx.copy_atom_call(copyAtomBuffer, fx.slice(tA, (None, tid)), rA)  # load element at tid
```

`copy_atom_call` works directly on any tensor slice without needing the full
`make_tiled_copy` infrastructure. It is appropriate when the slice index is a plain
thread or block index.

**What to flag**:
- `buffer_load(rsrc, manual_dword_offset, vec_width=N)` where the offset is derived
  from `tid` or `bid` via byte arithmetic — replace with `logical_divide` + `slice`
  + `copy_atom_call`.

---

## Step 6: Fragment-Centric Math Pattern

**Rule**: After loading data into a register fragment, extract a vector for arithmetic
with `memref_load_vec`, compute, then store back with `memref_store_vec` before copying
to the destination.

**Canonical pattern** (`01-vectorAdd.py:46-47`):
```python
fx.copy_atom_call(copyAtomBuffer, fx.slice(tA, (None, tid)), rA)  # global → register
fx.copy_atom_call(copyAtom,       fx.slice(tB, (None, tid)), rB)

vC = fx.arith.addf(fx.memref_load_vec(rA), fx.memref_load_vec(rB))  # math on vectors
fx.memref_store_vec(vC, rC)                                           # result → register

fx.copy_atom_call(copyAtom, rC, fx.slice(tC, (None, tid)))           # register → global
```

`memref_load_vec` and `memref_store_vec` operate on **register fragments** (allocated
with `make_fragment_like` or `memref_alloca`), not on global buffer tensors.

**What to flag**: `memref_load_vec` / `memref_store_vec` called on a global buffer
tensor or a slice of a buffer tensor — these ops are for register space only. Global
loads/stores go through `fx.copy` or `copy_atom_call`.

**Contrast with MMA**: For matrix multiply-accumulate, `fx.gemm` operates directly on
register fragments — no `memref_load_vec` extraction needed. `memref_load_vec` +
`arith.*` is only appropriate for **elementwise** operations on register data; it is
not a substitute for the layout API and should not appear on any global tensor.

```
Global memory ←→ register fragment   : layout API only (flat_divide / tiled_copy / fx.copy / copy_atom_call)
Register fragment ←→ MMA result      : fx.gemm  ← layout API, no memref extraction
Register fragment ←→ elementwise math: memref_load_vec + arith.* + memref_store_vec ← register space only
```

---

## Step 7: arith / vector / memref Usage

### Scope of "avoid arith/vector/memref"

The general guidance to avoid `arith`, `vector`, and `memref` targets **address and
offset computation** — not arithmetic on data values. Specifically, it prohibits using
these low-level ops to bypass the layout API for global memory access (manual dword
offsets, bitcasts, raw buffer loads/stores).

`arith.*` on data values and `memref_load_vec`/`memref_store_vec` on register fragments
are still appropriate in the contexts shown below, because the layout API has already
handled all global addressing.

### Permitted uses (from examples)

| Usage | Context | Reference |
|-------|---------|-----------|
| `arith.addf`, `arith.mulf` etc. | Math on vectors extracted from register fragments | `01-vectorAdd.py:46` |
| `ArithValue` arithmetic | Block/thread index decomposition (`bid // heads`, `tid % half`) | `04-preshuffle_gemm.py` |
| `memref_load_vec` / `memref_store_vec` | Elementwise read/write of register fragments only | `01-vectorAdd.py:46-47` |
| `fx.gemm(mma_atom, ...)` | MMA on register fragments — layout API, no extraction needed | `03-tiledMma.py:67` |
| `range_constexpr` | Compile-time-unrolled loops over fragment elements | `04-preshuffle_gemm.py` |

### Flag these

| Pattern | Flag | Preferred replacement |
|---------|------|-----------------------|
| `buffer_load(rsrc, dw, vec_width=N)` for vector loads | **FLAG** | `tiled_copy` + `fx.copy` (Step 4) or `copy_atom_call` (Step 5) |
| `buffer_store(data, rsrc, dw)` for vector stores | **FLAG** | `tiled_copy` + `fx.copy` |
| Byte/dword offset helpers (`* elem_bytes`, `>> 2`) | **FLAG** | Layout-API divide + index is byte-width-agnostic |
| `vector.bitcast` to/from `i32` around `buffer_load`/`buffer_store` | **FLAG** | Use copy_atom with elem_type directly; no bitcast needed |
| `memref_load_vec` / `memref_store_vec` on global tensors | **FLAG** | Use `fx.copy` or `copy_atom_call` |
| `rocdl.mfma_*` intrinsics instead of `fx.gemm` | **FLAG** | `fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)` (Step 14) |

---

## Step 8: Scalar buffer_load / buffer_store (Permitted)

`buffer_ops.buffer_load` and `buffer_ops.buffer_store` with `vec_width=1` are the
correct pattern for tensors where a single scalar is loaded or stored at a
runtime-computed index (block tables, index arrays, context lengths, non-contiguous
cache layouts). Confirmed in `pa_decode_fp8.py` (`bt_rsrc`, `cl_rsrc`, `ml_rsrc`).
No example in `examples/` covers this case. Do NOT flag scalar-load/store tensors.

**Rule for coordinate offset computation**: Even within the scalar
`buffer_load`/`buffer_store` path, the flat element offset must be computed using
`fx.make_layout(shape, stride)` + `crd2idx(coord, layout)` — not manual arithmetic
(`pid * (N * D * BS) + head * (D * BS) + ...`).

**Canonical pattern** (non-flash `value_cache` in `kernels/fused_rope_cache_kernel.py`):
```python
vc_nf_shape  = (None, num_kv_heads, head_dim, block_size)
vc_nf_stride = (num_kv_heads * head_dim * block_size,
                head_dim * block_size, block_size, 1)
vc_nf_layout = fx.make_layout(vc_nf_shape, vc_nf_stride)
from kernels.mfma_preshuffle_pipeline import crd2idx
for vi in range_constexpr(VEC_WIDTH):
    v_scalar = vector.extract(v_e, static_position=[vi])
    d_idx    = ArithValue(tid) * VEC_WIDTH + vi
    vc_coord = (pid_t_slot, pid_hk, d_idx, pid_b)
    vc_off   = arith.index_cast(T.i32, crd2idx(vc_coord, vc_nf_layout))
    buffer_ops.buffer_store(v_scalar, vc_rsrc, vc_off)
```

**What to flag**:
- Manual multi-dimensional product arithmetic for scalar offsets:
  `pid_t * (KH * D * BS) + pid_h * (D * BS) + d * BS + b` — replace with
  `fx.make_layout` + `crd2idx`.

---

## Step 9: tiled_copy and copy_atom Setup Location

**Rule**: Build `copy_atom`, `tiled_copy`, and `thr_copy` before any runtime thread
guard. They are static descriptors with no side effects and must be visible to all
uses in the kernel body.

This follows from MLIR SSA dominance requirements — values defined inside a branch
are not visible outside it. No example explicitly shows the anti-pattern, but the
rule applies to any SSA-based IR.

---

## Step 10: Buffer Resource Scope

**Rule**: `buffer_ops.create_buffer_resource` must be called at the top level of the
kernel function, not inside `scf.if` branches or loops.

Buffer resources are SGPR descriptors. In MLIR SSA form, a value defined inside a
branch does not dominate uses outside it. No example explicitly shows this anti-pattern,
but the MLIR dominance requirement is the basis for the rule.


---

## Step 11: 2D Tile Selection — zipped_divide vs flat_divide

**Rule**: For 2D tiles, choose the divide operation based on whether a K-loop outer
dimension needs to be kept for pipelining.

**Canonical patterns**:
```python
# zipped_divide: both dimensions tiled together, one block per tile
# (02-tiledCopy.py:25-27, 03-tiledMma.py:31-35)
tile = fx.make_tile(block_m, block_n)
bA   = fx.zipped_divide(A, tile)
bA   = fx.slice(bA, (None, bid))

# flat_divide: K-outer kept as a separate trailing dimension for the K-loop
# (04-preshuffle_gemm.py:31-32)
gA_k = fx.flat_divide(A, fx.make_tile(BLOCK_M, BLOCK_K))[None, None, bid_x, None]
gB_k = fx.flat_divide(B, fx.make_tile(BLOCK_N, BLOCK_K))[None, None, bid_y, None]
```

**When to use which**:
- No K-loop (single tile, epilogue writes): `zipped_divide` + `slice`
- Multi-step K-loop with software pipelining: `flat_divide` with K-outer as `None`

---

## Step 12: TiledMma Setup

**Rule**: GEMM kernels using MFMA must set up `TiledMma` through the layout API.

**Canonical pattern** (`03-tiledMma.py:39-41`):
```python
mma_atom  = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 4, fx.Float32))
tiled_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
thr_mma   = tiled_mma.thr_slice(tid)
```

**What to flag**:
- Direct `fx.rocdl.mfma_*` calls with manually packed operand vectors — bypasses the
  layout system entirely
- `make_tiled_mma` used without `thr_slice` before `make_fragment_A/B/C`

---

## Step 13: MMA-Matched TiledCopy for Operands

**Rule**: For A, B, C operands of an MFMA kernel, derive the copy layout from
`tiled_mma` using `make_tiled_copy_A/B/C` — not a manually specified layout.

**Canonical pattern** (`03-tiledMma.py:43-50`):
```python
copy_atom    = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
tiled_copy_A = fx.make_tiled_copy_A(copy_atom, tiled_mma)
tiled_copy_B = fx.make_tiled_copy_B(copy_atom, tiled_mma)
tiled_copy_C = fx.make_tiled_copy_C(copy_atom, tiled_mma)
thr_copy_A   = tiled_copy_A.get_slice(tid)
thr_copy_B   = tiled_copy_B.get_slice(tid)
thr_copy_C   = tiled_copy_C.get_slice(tid)
```

**What to flag**: `make_tiled_copy` with a manually chosen `thr_layout` / `val_layout`
for an A, B, or C MFMA operand — the layout must be derived from `tiled_mma` to match
the operand register layout expected by `fx.gemm`.

**Correct exception**: Non-operand tensors (bias, scales, output not consumed by MFMA)
may use a manually specified `make_tiled_copy`.

---

## Step 14: Fragment Allocation, Retiling, and gemm Invocation

**Rule**: Allocate MMA fragments with `thr_mma.make_fragment_A/B/C`, retile with
`thr_copy.retile`, and compute with `fx.gemm`.

**Canonical pattern** (`03-tiledMma.py:56-69`):
```python
frag_A = thr_mma.make_fragment_A(bA)
frag_B = thr_mma.make_fragment_B(bB)
frag_C = thr_mma.make_fragment_C(bC)

copy_frag_A = thr_copy_A.retile(frag_A)
copy_frag_B = thr_copy_B.retile(frag_B)
copy_frag_C = thr_copy_C.retile(frag_C)

fx.copy(copy_atom, copy_src_A, copy_frag_A)
fx.copy(copy_atom, copy_src_B, copy_frag_B)

fx.gemm(mma_atom, frag_C, frag_A, frag_B, frag_C)

fx.copy(copy_atom, copy_frag_C, copy_dst_C)
```

**What to flag**:
- `make_fragment_like(thr_copy.partition_S(...))` for MMA operands — must use
  `make_fragment_A/B/C` to get the correct register layout for MFMA
- Passing an MMA fragment directly to `fx.copy` without `retile` first
- `rocdl.mfma_*` intrinsic calls in place of `fx.gemm`

---

## Reporting

For each issue found, report:
1. **File and line number** (or approximate location)
2. **Rule violated** (reference the Step number above)
3. **Current code** (the flagged snippet)
4. **Suggested replacement** (the preferred pattern, citing the canonical example file and line)

Format the report as a table followed by per-issue details:

```
## Style Review: <filename>

| Line | Rule | Summary |
|------|------|---------|
| 42   | Step 3 | Manual dword offset instead of flat_divide |
| 87   | Step 10 | create_buffer_resource inside scf.if branch |

### Issue 1 (line 42, Step 3)
Current:
  kc_bytes = pid_t_slot * (BS * KH * D * elem_bytes) + ...
  buffer_ops.buffer_store(rot_i32, kc_rsrc, kc_bytes >> 2)

Suggested (04-preshuffle_gemm.py:31-33 pattern):
  gA_k = fx.flat_divide(A, fx.make_tile(BLOCK_M, BLOCK_K))[None, None, bid_x, None]
  dst  = thr_copy.partition_D(gA_k[..., k_iter])
  fx.copy(copy_atom, frag, dst)
```

If no issues are found, say so explicitly and confirm the kernel matches the canonical style.
