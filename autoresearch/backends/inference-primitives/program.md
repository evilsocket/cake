# Backend Inference Primitives Autoresearch

Autonomous optimization of the ComputeBackend trait's inference primitive methods.

## Objective

**Minimize latency of the 16 backend inference primitive methods** — linear_forward, rms_norm,
layer_norm, group_norm, softmax, sdpa, rope, silu, gelu, sigmoid, embedding, causal_mask, topk,
conv1d, conv_transpose1d, conv2d. These are the core building blocks called by every model
during inference. Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/inference-primitives`
2. **Read the in-scope files**:
3. **Read in-scope files**:
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait with default implementations.
   - `cake-core/src/backends/cpu/mod.rs` — CPU backend (inherits most defaults).
   - `cake-core/benches/bench_backend_ops.rs` — Benchmarks for all primitive methods.
4. **Run prepare.sh**: `bash autoresearch/backends/inference-primitives/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/backends/mod.rs` — Default implementations of all 16 methods.
- `cake-core/src/backends/cpu/mod.rs` — CPU-specific overrides.

## Files You Must NOT Modify

- `autoresearch/backends/inference-primitives/benchmark.sh`
- `autoresearch/backends/inference-primitives/prepare.sh`
- `cake-core/benches/bench_backend_ops.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/backends/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol` — must pass.
5. **Benchmark**: `bash autoresearch/backends/inference-primitives/benchmark.sh`
6. **Parse** the BENCH_RESULT line.
7. **If build/test failed**: quick fix (2 tries), then revert and log as `crash`.
8. **Record** in `experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND all gates pass.
   - **DISCARD** otherwise. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## Architecture Constraints

### Default Implementations
The default trait methods in `mod.rs` use candle_nn ops (rms_norm, softmax_last_dim, sigmoid,
silu, rope). These are already reasonably optimized but can be improved.

### No Breaking Changes
All 16 methods have strict API contracts. Do not change method signatures. The existing
equivalence tests in `test_backend_ops.rs` verify numerical correctness against candle_nn.

### CPU Backend
The CPU backend inherits all defaults. Override specific methods only when you can prove
the override is faster than the default (e.g., manual SIMD, better memory access pattern).

## Promising Areas to Explore

1. **Fused linear+bias** — The default `linear_forward` does matmul then broadcast_add.
   A fused kernel could eliminate the intermediate tensor allocation.

2. **Optimized topk** — Default uses full sort O(N log N). Partial sort (selection algorithm)
   is O(N + K log K) which matters for MoE with 128-256 experts and K=8.

3. **Fused softmax+mask** — Combine causal masking with softmax to avoid materializing the
   full mask tensor. Saves one tensor allocation + one elementwise op per attention head.

4. **Batched RoPE** — The rope implementation processes elements individually. SIMD-friendly
   batch processing of cos/sin rotations could improve throughput.

5. **Memory-efficient group_norm** — Current implementation reshapes to (B, groups, hidden),
   computes stats, normalizes, reshapes back. An in-place variant could reduce allocations.

6. **causal_mask caching** — The mask is recomputed per call. Cache common sizes (1, 128, 512, 2048).

7. **Contiguity-aware softmax** — The default checks `rank()-1` to dispatch to fused path.
   For non-last-dim softmax, a transpose+fused+transpose pattern might beat the generic path.

8. **embedding with F16 weights** — Direct F16 index_select avoids dtype conversion.

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

## NEVER STOP

Once the loop begins, do NOT pause. Run autonomously until manually interrupted.
