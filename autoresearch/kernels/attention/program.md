# Attention Kernel Autoresearch

Autonomous optimization of attention computation across all backends.

## Objective

**Minimize attention forward pass latency** — scaled dot-product attention, rotary position
embeddings, QK normalization, GQA head expansion, and causal masking. Attention is the
single most expensive operation in every transformer forward pass. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/attention/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/attention/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/models/common/attention.rs` — CausalSelfAttention: Q/K/V projection,
     rotary position embeddings (RoPE), optional QK normalization, grouped-query attention
     (GQA) head repeat, scaled dot-product attention dispatch, output projection.
   - `cake-core/src/models/common/cache.rs` — KV cache: cosine/sine tables for RoPE,
     causal masks, KV concatenation, sliding window support.
   - `cake-core/src/backends/cpu/mod.rs` — CPU attention: manual matmul + softmax + mask.
   - `cake-core/src/backends/cuda/ops.cu` — CUDA attention kernels (if any custom ones).
   - `cake-core/src/backends/metal/ops.msl` — Metal attention shaders.
   - `cake-core/src/backends/vulkan/ops.wgsl` — Vulkan attention shaders.
   - `cake-core/src/models/qwen3_5/` — GatedDeltaNet linear attention (an alternative
     attention mechanism with recurrent state).
   - `cake-core/benches/bench_attention.rs` — Attention benchmarks.
   - `cake-core/benches/bench_linear_attn.rs` — GatedDeltaNet linear attention benchmarks.
   - `cake-core/benches/bench_cache.rs` — Cache benchmarks.
4. **Run prepare.sh**: `bash autoresearch/kernels/attention/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/models/common/attention.rs` — CausalSelfAttention implementation. Projection
  computation, RoPE application, QK norm, GQA expansion, attention score computation.
- `cake-core/src/models/common/cache.rs` — KV cache implementation. Cos/sin precomputation,
  mask generation, KV concat/update, sliding window.
- `cake-core/src/backends/cpu/mod.rs` — CPU attention implementation.

## Files You Must NOT Modify

- `autoresearch/kernels/attention/benchmark.sh`
- `autoresearch/kernels/attention/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_attention.rs`
- `cake-core/benches/bench_linear_attn.rs`
- `cake-core/benches/bench_cache.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused attention optimization hypothesis.
2. **Implement** the change.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol`
5. **Benchmark**: `bash autoresearch/kernels/attention/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Architecture Constraints

### Attention Variants
Three attention mechanisms exist:
1. **CausalSelfAttention** — standard scaled dot-product with causal masking (most models)
2. **Sliding window attention** — windowed causal attention (Gemma3, Exaone4)
3. **GatedDeltaNet** — linear attention with recurrent state (Qwen3.5 linear layers)

All three must remain correct. Optimizations to one must not break the others.

### GQA Head Repeat
Grouped-query attention uses fewer K/V heads than Q heads. The code uses
`repeat_interleave` to expand K/V to match Q head count. This is mathematically
equivalent to tiled matmul — a direct tiled implementation could avoid the memory copy.

### RoPE Application
Rotary position embeddings are applied to Q and K tensors. The cos/sin tables come from
the cache. RoPE involves element-wise operations: `q * cos + rotate_half(q) * sin`.
The `rotate_half` creates a new tensor by splitting and negating — this allocation may
be avoidable.

### Backend Dispatch
The `backend.attention()` call dispatches to the appropriate GPU kernel. On CPU, attention
is computed manually: `Q @ K^T`, scale, mask, softmax, `@ V`. On GPU, flash-attention or
fused kernels are used. Backend-agnostic optimizations (reducing Q/K/V computation,
caching, shape optimization) benefit all backends.

## Promising Areas to Explore

1. **Skip mask for generation** — during token generation (seq_len=1), the causal mask is
   trivially all-true. Detecting this case and skipping mask computation + application
   could save significant time for the most common inference path.

2. **RoPE without rotate_half allocation** — `rotate_half` splits the tensor, negates half,
   and concatenates. An in-place or fused implementation could eliminate the intermediate
   tensor allocation.

3. **GQA without repeat_interleave** — instead of expanding K/V heads, compute attention
   with grouped matmul. This avoids O(seq_len * num_q_heads * head_dim) memory copy.

4. **Fused QKV projection** — compute Q, K, V from a single large matmul instead of three
   separate ones. This is a single memory pass over the input.

5. **Precomputed cos/sin tables** — RoPE cos/sin depend only on position and head_dim.
   Can they be computed once at model load and indexed rather than sliced each step?

6. **Cache-friendly KV layout** — the KV cache is concatenated along seq_len each step.
   A pre-allocated ring buffer with pointer update could avoid reallocation.

7. **Softmax numerical stability** — the max-subtract-exp-sum pattern in softmax involves
   multiple passes. Online softmax (single pass) may be faster for small sequences.

8. **GatedDeltaNet optimization** — the linear attention mechanism has a different
   computational profile (matrix state updates). The recurrence may benefit from
   different parallelization strategies.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## NEVER STOP

Run autonomously until manually interrupted.
