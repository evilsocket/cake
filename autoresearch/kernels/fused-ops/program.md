# Fused Operations Autoresearch

Autonomous optimization of fused compute operations across all backends.

## Objective

**Minimize fused operation latency** — silu_mul, stable_softplus, rms_norm_gated, add_rms_norm,
rms_norm_channel, add3, add_scaled, exp_mul, sub_mul, depthwise_conv1d_silu, depthwise_conv1d_bias,
and adaln_modulate. These operations run inside every transformer layer and are the primary targets
for micro-optimization. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/fused-ops/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/fused-ops/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/backends/cpu/mod.rs` — CPU implementations of all fused ops. These are the
     default fallback and what benchmarks measure without GPU features.
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait defining the fused op interface.
   - `cake-core/benches/bench_utils.rs` — **Primary benchmark file.** Contains benchmarks for
     every fused operation on both CPU and GPU paths: `fused_silu_mul_cpu`, `fused_stable_softplus_cpu`,
     `fused_rms_norm_gated_cpu`, `fused_add3_cpu`, `fused_add_rms_norm_cpu`, `fused_add_scaled_cpu`,
     `fused_rms_norm_channel_cpu`, `fused_depthwise_conv1d_silu_cpu`, `fused_depthwise_conv1d_bias_cpu`,
     and GPU variants.
   - `cake-core/benches/bench_helpers.rs` — Shared helpers.
4. **Run prepare.sh**: `bash autoresearch/kernels/fused-ops/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/backends/cpu/mod.rs` — CPU fused operation implementations. Algorithm choice,
  vectorization hints, memory access patterns, temporary allocation reduction.

## Files You Must NOT Modify

- `autoresearch/kernels/fused-ops/benchmark.sh`
- `autoresearch/kernels/fused-ops/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_utils.rs`
- `cake-core/benches/bench_helpers.rs`
- GPU backend files (cuda/, metal/, vulkan/) — use the backend-specific tasks for those
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization for a CPU fused operation.
2. **Implement** the change in `cpu/mod.rs`.
3. **Commit**: `git add cake-core/src/backends/cpu/ && git commit -m "<description>"`
4. **Quality gate**: `cargo test -p cake-core --lib --test unit`
5. **Benchmark**: `bash autoresearch/kernels/fused-ops/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## CPU Backend Constraints

### Candle Tensor Operations
The CPU backend uses candle's tensor operations (element-wise, matmul, etc.) which map to
optimized BLAS/LAPACK under the hood. Custom Rust loops are unlikely to beat candle's
vectorized operations. Focus on reducing the *number* of operations rather than making
individual operations faster.

### Temporary Tensor Allocation
Each tensor operation creates a new tensor. Multi-step fusions (e.g., `a + b` then `* c`)
create intermediate tensors. Reducing the number of intermediate allocations is the primary
lever for CPU optimization.

### Zero-Copy Where Possible
Candle supports broadcasting and views without copying. Use `broadcast_as`, `reshape`, and
`narrow` instead of `expand` + operations when possible.

## Promising Areas to Explore

1. **Reduce intermediate tensors in rms_norm** — RMS norm does: square → mean → add_eps →
   sqrt → reciprocal → multiply. Each step creates a tensor. Can operations be combined?

2. **Fuse add + rms_norm more tightly** — `add_rms_norm` returns both the residual and the
   normalized result. Can the residual computation share work with normalization?

3. **silu_mul without intermediate** — `silu(gate) * up` computes silu(gate) as an
   intermediate. Can the sigmoid + multiply + multiply chain be done in fewer passes?

4. **stable_softplus clamp optimization** — the clamp + exp + ln + max chain may have
   fast-path shortcuts for common value ranges.

5. **Three-way add fusion** — `add3(a, b, c)` currently does `(a + b)? + c`. A single-pass
   element-wise operation over three inputs could halve memory traffic.

6. **Depthwise conv1d optimization** — the convolution uses explicit loops. Can it leverage
   candle's built-in conv operations or be restructured for better cache behavior?

7. **add_scaled as fused multiply-add** — `a + b * c` is a classic FMA pattern. Ensure the
   implementation maps to hardware FMA instructions.

8. **exp_mul and sub_mul fusion** — these simple two-op patterns may benefit from combined
   element-wise kernels using candle's custom ops.

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
