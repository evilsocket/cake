# MoE Autoresearch

Autonomous optimization of Mixture-of-Experts inference components.

## Objective

**Minimize MoE forward pass latency** — expert routing, gating, weight access, and
combine operations. The MoE layer is the bottleneck in Qwen3-MoE and Qwen3.5-MoE models
where K experts are selected from N total for each token. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/moe/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/moe/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/models/common/mod.rs` — MoE layer: router, top-k gating, expert dispatch, combine.
   - `cake-core/src/models/common/expert_provider.rs` — ExpertProvider trait + StackedResidentProvider + IndividualResidentProvider.
   - `cake-core/src/models/common/disk_expert_provider.rs` — DiskExpertProvider for memory-mapped expert weights.
   - `cake-core/src/models/qwen3_moe/mod.rs` — Qwen3 MoE model integration.
   - `cake-core/src/models/qwen3_5_moe/mod.rs` — Qwen3.5 MoE model integration.
   - `cake-core/benches/bench_moe.rs` — MoE forward benchmarks.
   - `cake-core/benches/bench_expert_provider.rs` — Expert weight access benchmarks.
   - `cake-core/benches/bench_blocks.rs` — MoE block benchmarks (qwen3_moe_block, qwen3_5_moe_block).
4. **Run prepare.sh**: `bash autoresearch/models/moe/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/models/common/mod.rs` — MoE layer implementation (gating, routing, expert dispatch, output combination).
- `cake-core/src/models/common/expert_provider.rs` — ExpertProvider trait, StackedResidentProvider, IndividualResidentProvider.
- `cake-core/src/models/common/disk_expert_provider.rs` — DiskExpertProvider, memory-mapped weight access patterns.
- `cake-core/src/models/qwen3_moe/mod.rs` — Qwen3 MoE architecture.
- `cake-core/src/models/qwen3_5_moe/mod.rs` — Qwen3.5 MoE architecture.

## Files You Must NOT Modify

- `autoresearch/models/moe/benchmark.sh`
- `autoresearch/models/moe/prepare.sh`
- `cake-core/benches/bench_moe.rs`
- `cake-core/benches/bench_expert_provider.rs`
- `cake-core/benches/bench_blocks.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files (`cake-core/tests/`)

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol` — must pass.
5. **Benchmark**: `bash autoresearch/models/moe/benchmark.sh`
6. **Parse** the BENCH_RESULT line.
7. **If build/test failed**: quick fix (2 tries), then revert and log as `crash`.
8. **Record** in `experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND all gates pass.
   - **DISCARD** otherwise. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## Architecture Constraints

### Expert Provider Abstraction
The `ExpertProvider` trait separates weight storage from computation. Three implementations
exist: StackedResident (single stacked tensor), IndividualResident (per-expert tensors), and
Disk (memory-mapped safetensors). Do not break this abstraction — models use it polymorphically.

### Top-K Routing
MoE uses top-k softmax routing with optional `norm_topk_prob`. The router output is a
`(batch*seq, num_experts)` tensor, top-k selects experts per token. The gating weights
are used to combine expert outputs. This is a numerically sensitive path — do not change
the routing algorithm without verifying output quality.

### Shared MLP
Some MoE architectures include a shared expert that runs for every token alongside the
routed experts. Do not remove or skip shared expert computation.

## Promising Areas to Explore

1. **Expert batching** — currently experts are dispatched sequentially. Can tokens routed
   to the same expert be batched into a single matmul?

2. **Stacked weight layout** — StackedResidentProvider stores all expert weights in one tensor.
   Can the stacking layout be optimized for the access pattern (e.g., strided vs. contiguous)?

3. **Router optimization** — top-k selection uses argsort. For small K (2-4) and moderate N
   (8-64), a partial sort or selection network may be faster.

4. **Expert output combination** — the weighted sum of K expert outputs may benefit from
   fused multiply-accumulate instead of separate multiply + reduce.

5. **Memory access patterns** — for DiskExpertProvider, pread patterns and OS page cache
   interaction may have optimization opportunities.

6. **Sparse activation** — if most tokens route to the same few experts, can we skip
   weight loading for unused experts earlier?

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
