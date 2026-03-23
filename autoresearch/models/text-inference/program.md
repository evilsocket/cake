# Text Inference Autoresearch

Autonomous optimization of text model inference components on CPU.

## Objective

**Minimize forward pass latency** for core text model building blocks: attention, MLP,
transformer blocks, and KV cache operations. All benchmarks use synthetic tensors on CPU
with small dimensions (hidden=64) for fast iteration. Lower is better.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/text-inference/<tag>` from the current branch.
3. **Read the in-scope files** for full context:
   - `cake-core/src/models/common/attention.rs` — CausalSelfAttention: Q/K/V projection, rotary embeddings, scaled dot-product attention, output projection.
   - `cake-core/src/models/common/mlp.rs` — MLP: gate/up projection, SiLU activation, down projection.
   - `cake-core/src/models/common/transformer.rs` — TransformerBlock: attention + MLP + residual + RmsNorm composition.
   - `cake-core/src/models/common/cache.rs` — KV cache: cosine/sine tables, causal masks, KV concatenation, sliding window.
   - `cake-core/src/models/common/text_model.rs` — TextModel: embedding, N layers, final norm, LM head.
   - `cake-core/src/models/common/config.rs` — Config struct and loading.
   - `cake-core/src/backends/cpu/mod.rs` — CPU backend (default implementations used by benchmarks).
   - `cake-core/benches/bench_attention.rs` — Attention benchmarks.
   - `cake-core/benches/bench_mlp.rs` — MLP benchmarks.
   - `cake-core/benches/bench_blocks.rs` — Transformer block benchmarks.
   - `cake-core/benches/bench_cache.rs` — Cache benchmarks.
   - `cake-core/benches/bench_helpers.rs` — Shared benchmark helpers (test_config, make_tensor, etc.).
4. **Run prepare.sh**: `bash autoresearch/models/text-inference/prepare.sh`
5. **Confirm baseline**: Check that `baseline.txt` and `experiments.tsv` were created.
6. **Start experimenting**.

## Files You May Modify

- `cake-core/src/models/common/attention.rs` — CausalSelfAttention implementation. Projection layout, RoPE application, attention computation, output projection.
- `cake-core/src/models/common/mlp.rs` — MLP implementation. Gate/up/down projections, activation fusion.
- `cake-core/src/models/common/transformer.rs` — TransformerBlock composition. Layer ordering, residual connections, norm placement.
- `cake-core/src/models/common/cache.rs` — KV cache. Cosine/sine precomputation, mask generation, KV concatenation, sliding window logic.
- `cake-core/src/models/common/text_model.rs` — TextModel forward pass. Embedding lookup, layer iteration, final head.
- `cake-core/src/backends/cpu/mod.rs` — CPU backend default implementations.

## Files You Must NOT Modify

- `autoresearch/models/text-inference/benchmark.sh` — the measurement harness is sacred
- `autoresearch/models/text-inference/prepare.sh`
- `cake-core/benches/bench_attention.rs` — benchmark definitions
- `cake-core/benches/bench_mlp.rs`
- `cake-core/benches/bench_blocks.rs`
- `cake-core/benches/bench_cache.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files (`cake-core/tests/`)

## The Experiment Loop

Each experiment: modify → build → test → benchmark → decide → repeat.

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol` — must pass with zero failures.
5. **Benchmark**: `bash autoresearch/models/text-inference/benchmark.sh`
6. **Parse** the BENCH_RESULT line: `score=X tests=PASS/FAIL clippy=PASS/FAIL status=OK/...`
7. **If build or test failed**: attempt a quick fix. If unfixable after 2 tries, revert and log as `crash`.
8. **Record** in `autoresearch/models/text-inference/experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND tests PASS AND clippy PASS. (Lower score is better; 0.5% noise margin.)
   - **DISCARD** if `score > previous_best * 1.005` OR any gate fails. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## Architecture Constraints

### Tensor Layout
All tensors use candle's row-major layout. Attention inputs are `(batch, seq_len, hidden)`,
internally reshaped to `(batch, heads, seq_len, head_dim)`. Do not change these conventions —
every model architecture depends on them.

### Backend Abstraction
Models call `ctx.backend().method()` for GPU-accelerated operations. Do NOT add device-specific
code in model files. The CPU backend is used for benchmarks; optimizations must be
backend-agnostic (algorithmic, not hardware-specific).

### Config Compatibility
The `Config` struct is deserialized from HuggingFace model configs. Do not change field names
or remove fields — this would break model loading for all supported architectures.

### Test Compatibility
765+ tests must pass. The benchmarks use `test_config()` with hidden_size=64, num_heads=4,
num_kv_heads=2, num_layers=2. Optimizations must work at all scales, not just the benchmark
dimensions.

## Promising Areas to Explore

1. **RoPE precomputation** — the cosine/sine tables are recomputed per forward call. Can they
   be cached more aggressively or computed with fewer operations?

2. **KV cache memory layout** — the cache concatenates along the sequence dimension each step.
   Can a ring-buffer or pre-allocated layout avoid repeated allocations?

3. **Attention mask optimization** — causal masks are regenerated each step. For generation
   (seq_len=1), the mask is trivial. Can we skip mask computation entirely for single-token steps?

4. **MLP activation fusion** — `silu_mul` is already fused in GPU backends but the CPU path
   may have opportunities for better vectorization or reduced allocations.

5. **Residual connection pattern** — the add + norm pattern in transformer blocks creates
   intermediate tensors. Can in-place operations reduce allocation pressure?

6. **GQA optimization** — grouped-query attention repeats K/V heads. The repeat_interleave
   may be avoidable with careful indexing.

7. **Sliding window attention** — for models like Gemma3/Exaone4, the windowed attention
   path may have redundant computation that can be eliminated.

8. **Cache clearing** — `cache.as_new()` and `cache.clear()` patterns may be optimizable
   for generation loops.

## Recording Results

Append to `experiments.tsv` (tab-separated):

```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

Example:
```
001	2026-03-23T01:30:00	abc1234	485320	PASS	PASS	keep	Skip mask computation for seq_len=1 generation steps
002	2026-03-23T01:32:00	def5678	502100	PASS	PASS	discard	Ring buffer KV cache — extra bookkeeping overhead
003	2026-03-23T01:34:00	ghi9012	0	SKIP	SKIP	crash	Pre-allocated cache — shape mismatch in sliding window
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails
- If **3 consecutive discards** with similar approaches: move to a different area
- If **build fails**: quick fix (2 tries max), then revert
- After every **keep**: update your mental baseline to the new score
- Always **revert before starting a new experiment** (clean slate from last keep)

## Simplicity Criterion

All else being equal, simpler is better. A 0.5% improvement that adds 50 lines of complexity?
Borderline. A 0.5% improvement from *removing* code? Definitely keep. The goal is lean, fast
inference — not a pile of micro-optimizations.

## Safety

- Do not change the `ComputeBackend` trait signature — all backends depend on it
- Do not change the `Config` struct field names — model loading depends on them
- Do not add new dependencies to `Cargo.toml`
- If build fails, revert immediately
- Do not modify benchmark or test files

## NEVER STOP

Once the loop begins, do NOT pause to ask the human for permission. The human may be asleep.
You run autonomously until manually interrupted. If you run out of ideas, re-read the source
for new angles, try combining near-misses, try more radical restructuring. Each experiment
takes ~30-60 seconds, so you can run ~60-120 per hour.
