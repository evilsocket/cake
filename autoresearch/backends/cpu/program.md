# CPU Backend Autoresearch

Autonomous optimization of CPU compute operations for x86_64 and aarch64.

## Objective

**Minimize CPU execution time** for all fused operations: silu_mul, stable_softplus,
rms_norm_gated, add_rms_norm, rms_norm_channel, add3, add_scaled, exp_mul, sub_mul,
depthwise_conv1d_silu, depthwise_conv1d_bias, adaln_modulate, attention, and the 16
inference primitive methods (linear_forward, rms_norm, layer_norm, group_norm, softmax,
sdpa, rope, silu, gelu, sigmoid, embedding, causal_mask, topk, conv1d, conv_transpose1d,
conv2d). These are the inner loop of every forward pass when no GPU is available.
Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/backends/cpu`
2. **Read the in-scope files**:
   - `cake-core/src/backends/cpu/mod.rs` — CpuBackend struct, ComputeBackend trait
     implementation. **Primary target.** Contains all fused operation implementations:
     silu_mul, stable_softplus, rms_norm_gated, add_rms_norm, rms_norm_channel, add3,
     add_scaled, exp_mul, sub_mul, depthwise_conv1d_silu, depthwise_conv1d_bias,
     depthwise_conv1d_bias_ctx, adaln_modulate, attention, f8e4m3 conversions.
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait definition with default
     implementations for the 16 inference primitives. The CPU backend inherits all
     defaults — override any where a CPU-specific implementation is faster.
   - `cake-core/benches/bench_utils.rs` — Fused ops benchmarks (CPU and GPU variants).
   - `cake-core/benches/bench_backend_ops.rs` — Inference primitive benchmarks.
3. **Run prepare.sh**: `bash autoresearch/backends/cpu/prepare.sh`
4. **Confirm baseline** and start experimenting.

**NOTE**: This task requires no special hardware or feature flags. The CPU backend is the
default and always available. Build with `cargo build -p cake-core` (no `--features`).

## Files You May Modify

- `cake-core/src/backends/cpu/mod.rs` — CpuBackend implementation. All fused operations,
  attention, FP8 conversion. Override inherited inference primitive defaults when a
  CPU-specific path is measurably faster.

## Files You Must NOT Modify

- `autoresearch/backends/cpu/benchmark.sh`
- `autoresearch/backends/cpu/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_utils.rs`
- `cake-core/benches/bench_backend_ops.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in `cake-core/src/backends/cpu/mod.rs`.
3. **Commit**: `git add cake-core/src/backends/cpu/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol` — must pass.
5. **Benchmark**: `bash autoresearch/backends/cpu/benchmark.sh`
6. **Parse** the BENCH_RESULT line.
7. **If build/test failed**: quick fix (2 tries), then revert and log as `crash`.
8. **Record** in `experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND all gates pass.
   - **DISCARD** otherwise. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## CPU Architecture Constraints

### Memory Hierarchy
CPU performance is dominated by memory access patterns. L1 cache (32-64KB per core),
L2 (256KB-1MB), L3 (shared, 8-32MB). Operations that fit in cache are fast; those that
spill to main memory are 10-100x slower. Keep working sets small and access patterns
sequential.

### SIMD
Modern CPUs have SIMD units: SSE4.2 (128-bit), AVX2 (256-bit), AVX-512 (512-bit) on
x86_64; NEON (128-bit) on aarch64. Candle uses rayon for parallelism and BLAS for matmul
but does not always vectorize elementwise ops. Manual restructuring of operations to be
SIMD-friendly (contiguous, aligned, batch-processed) can help the compiler auto-vectorize.

### Allocation Pressure
Every `Tensor::new`, `zeros`, `ones`, and intermediate tensor result allocates. The CPU
backend creates many temporaries (e.g., `contiguous()` calls, broadcast results). Reducing
allocation count is often more impactful than reducing ALU instructions.

### Candle CPU Backend
The CPU backend delegates to candle's internal CPU ops, which use rayon for parallelism
and link against BLAS (OpenBLAS, MKL, or Accelerate on macOS) for matmul. The fused ops
in `cpu/mod.rs` compose multiple candle ops — each intermediate step allocates a new tensor.
Fusing these into fewer operations reduces allocation overhead.

### Contiguity
Many candle operations require contiguous tensors. The `.contiguous()` call is cheap for
already-contiguous tensors (returns self) but allocates a full copy for strided views.
Avoid unnecessary `.contiguous()` calls, and structure operations to maintain contiguity
through the pipeline.

## Promising Areas to Explore

1. **Reduce intermediate allocations** — Many fused ops chain 2-3 candle ops, each creating
   a new tensor. Can operations be restructured to reduce the number of intermediates?
   For example, `silu_mul` currently does `silu(gate) * up` — two allocations. A single-pass
   implementation over raw data could eliminate one.

2. **Avoid unnecessary contiguous() calls** — Several methods call `.contiguous()` on tensors
   that may already be contiguous. Guard with `is_contiguous()` checks or restructure to
   avoid needing contiguity.

3. **Batch processing for depthwise_conv1d_bias** — The current implementation loops over
   `output_time` positions, creating and concatenating individual slices. A single batched
   operation could eliminate O(output_time) tensor allocations.

4. **Causal mask precomputation** — `attention()` rebuilds the causal mask from scratch every
   call using a `Vec<u8>` loop. Cache common mask sizes or use candle's `Tensor::tril` to
   build masks more efficiently.

5. **Fused attention softmax** — The attention implementation creates separate tensors for
   `attn * scale`, mask application, and softmax. Fusing scale+mask+softmax could reduce
   three allocations to one.

6. **Optimized topk** — The default `topk` (inherited from the trait) uses a full sort.
   A CPU-specific partial selection algorithm (O(N + K log K) instead of O(N log N))
   could be significantly faster for MoE routing with many experts.

7. **In-place operations** — Where candle supports it, use in-place tensor operations
   (e.g., `add_` instead of `add`) to avoid allocating result tensors.

8. **Better RoPE implementation** — Override the default `rope()` with a CPU-optimized
   version that processes cos/sin rotations in a SIMD-friendly batch pattern.

9. **Fused linear+bias** — Override `linear_forward` to fuse matmul and bias addition
   into a single pass, eliminating the intermediate tensor from broadcast_add.

10. **stable_softplus simplification** — The current implementation creates two temporaries
    (clamped and exp result). A direct computation path could reduce allocations.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails
- If **3 consecutive discards**: move to a different area
- After every **keep**: update baseline to new score

## Safety

- Do not change the ComputeBackend trait signature
- Do not use `unsafe` unless you can prove correctness and the improvement is >5%
- Maintain numerical equivalence with the original implementations — the unit tests
  and backend_ops equivalence tests verify correctness
- Do not add external crate dependencies without justification

## NEVER STOP

Once the loop begins, do NOT pause. Run autonomously until manually interrupted.
Each experiment takes ~1-2 minutes.
