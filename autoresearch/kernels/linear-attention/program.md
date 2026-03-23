# Linear Attention (GatedDeltaNet) Autoresearch

Autonomous optimization of the GatedDeltaNet recurrent linear attention mechanism.

## Objective

**Minimize GatedDeltaNet forward pass latency** — the recurrent state update, gating,
depthwise convolution, and output projection. This is the dominant attention variant in
Qwen3.5 models (18 of 24 layers use GDN). Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/linear-attention`
2. **Read the in-scope files**:
   - `cake-core/src/models/qwen3_5/linear_attention.rs` — GatedDeltaNet implementation (~470 lines).
   - `cake-core/src/models/qwen3_5/block.rs` — Block dispatch (linear vs full attention).
   - `cake-core/benches/bench_linear_attn.rs` — GDN benchmarks.
   - `cake-core/benches/bench_blocks.rs` — Qwen3.5 block benchmarks.
4. **Run prepare.sh**: `bash autoresearch/kernels/linear-attention/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/models/qwen3_5/linear_attention.rs` — GatedDeltaNet forward pass, recurrent update, gating.
- `cake-core/src/models/qwen3_5/block.rs` — Block-level dispatch and residual logic.

## Files You Must NOT Modify

- `autoresearch/kernels/linear-attention/benchmark.sh`
- `autoresearch/kernels/linear-attention/prepare.sh`
- `cake-core/benches/bench_linear_attn.rs`
- `cake-core/benches/bench_blocks.rs`
- `cake-core/benches/bench_helpers.rs`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/models/qwen3_5/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol` — must pass.
5. **Benchmark**: `bash autoresearch/kernels/linear-attention/benchmark.sh`
6. **Parse** the BENCH_RESULT line.
7. **If build/test failed**: quick fix (2 tries), then revert and log as `crash`.
8. **Record** in `experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND all gates pass.
   - **DISCARD** otherwise. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## Architecture Constraints

### Recurrent State
GDN maintains a per-layer recurrent state matrix `S` of shape `(batch, heads, head_dim, head_dim)`.
The update rule is: `S_new = S_old * decay + outer(key, value * gate)`. This must be numerically
identical after optimization — the state is carried across tokens.

### Generation vs Prefill
- **Generation (seq_len=1)**: Single-token recurrent update. This is the critical hot path.
- **Prefill (seq_len>1)**: Vectorized using unfold for the conv1d window. Less critical but still important.

### Backend Methods Used
The implementation calls: `backend.depthwise_conv1d_silu()`, `backend.stable_softplus()`,
`backend.rms_norm_gated()`, `backend.sigmoid()`, `backend.silu()`, `backend.rms_norm()`,
`backend.linear_forward()`. Do not bypass these — they enable GPU acceleration.

## Promising Areas to Explore

1. **Recurrent update fusion** — The state update `S = S * decay + k_outer * v_gate` involves
   3 separate tensor ops. A fused backend method could do this in one pass.

2. **Gate computation batching** — Currently alpha, beta, gate are computed separately from the
   fused projection. Pre-computing all gate values before the recurrent loop could improve cache locality.

3. **Conv1d window reuse** — For generation, the conv1d window is rebuilt each step from cached context.
   Pre-rolling the window could save a narrow+cat operation.

4. **F32 computation scope** — Bulk F32 conversion at layer entry may be wider than needed.
   Narrowing the F32 scope could reduce dtype conversion overhead.

5. **State matrix memory layout** — Transposing the state for better matmul alignment.

6. **Output projection fusion** — `rms_norm_gated` + `linear_forward` could be fused if
   the backend supports it.

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
