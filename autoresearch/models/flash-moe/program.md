# Flash-MoE Autoresearch

Autonomous optimization of disk-backed expert offloading for large MoE models.

## Objective

**Minimize expert-offload latency** — pread I/O, dtype conversion, tensor construction,
and page cache utilization for disk-backed MoE expert weights. Flash-MoE enables running
models with hundreds of experts (Qwen3-MoE 128, Qwen3.5-MoE 256) without loading all
expert weights into RAM. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/flash-moe/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/flash-moe/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/utils/tensor_storage.rs` — TensorStorageProvider trait + SafetensorsStorage (pread, header parsing, offset calculation).
   - `cake-core/src/models/common/disk_expert_provider.rs` — DiskExpertProvider (tensor reads, dtype conversion, device transfer).
   - `cake-core/src/models/common/expert_provider.rs` — ExpertProvider trait + resident providers (comparison targets).
   - `cake-core/src/models/qwen3_moe/block.rs` — Block-level wiring for expert offload.
   - `cake-core/src/models/qwen3_5_moe/block.rs` — Block-level wiring for expert offload.
   - `cake-core/benches/bench_flash_moe.rs` — Flash-MoE specific benchmarks.
   - `cake-core/benches/bench_expert_provider.rs` — Provider comparison benchmarks.
4. **Run prepare.sh**: `bash autoresearch/models/flash-moe/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/utils/tensor_storage.rs` — SafetensorsStorage: pread implementation, header parsing, offset computation, buffer management.
- `cake-core/src/models/common/disk_expert_provider.rs` — DiskExpertProvider: read_weight, dtype conversion, buffer reuse, prefetching.
- `cake-core/src/models/common/expert_provider.rs` — ExpertProvider trait, ExpertWeights struct.

## Files You Must NOT Modify

- `autoresearch/models/flash-moe/benchmark.sh`
- `autoresearch/models/flash-moe/prepare.sh`
- `cake-core/benches/bench_flash_moe.rs`
- `cake-core/benches/bench_expert_provider.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files (`cake-core/tests/`)
- MoE forward logic (`cake-core/src/models/qwen3_moe/moe.rs`, `cake-core/src/models/qwen3_5_moe/moe.rs`)

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol` — must pass.
5. **Benchmark**: `bash autoresearch/models/flash-moe/benchmark.sh`
6. **Parse** the BENCH_RESULT line.
7. **If build/test failed**: quick fix (2 tries), then revert and log as `crash`.
8. **Record** in `experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND all gates pass.
   - **DISCARD** otherwise. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## Architecture Constraints

### pread Thread Safety
SafetensorsStorage uses `pread()` (Unix `read_at()`) for thread-safe concurrent reads from
the same file descriptor. Do not replace with `seek()+read()` — that would break concurrent
access from multiple inference threads.

### OS Page Cache
The design intentionally relies on the OS page cache instead of an application-level LRU.
Flash-MoE paper showed OS page cache is 38% faster. Do not add application-level caching
unless benchmarks prove it faster.

### Dtype Conversion Path
`read_weight()` reads in storage dtype, then converts to inference dtype. The hot path is
F16→F32 or BF16→F32 conversion. Optimizations must not change the numeric result.

### ExpertProvider Contract
`get_expert(idx)` returns `ExpertWeights { gate_proj, up_proj, down_proj }`. Each is a 2D
tensor. The MoE forward pass relies on this contract — do not change the interface.

## Promising Areas to Explore

1. **Buffer reuse** — `read_weight()` allocates a new `Vec<u8>` per tensor read. A reusable
   buffer pool could reduce allocation pressure (3 tensors × K experts per token).

2. **Prefetching** — after the router selects top-K experts, issue `posix_fadvise(WILLNEED)`
   or background reads for the K experts while the shared expert runs.

3. **Batch pread** — read gate_proj+up_proj+down_proj in a single pread if they're contiguous
   in the safetensors file (they often are within the same shard).

4. **Header indexing** — `from_model_path()` re-parses shard headers every time. Cache parsed
   headers or use a more compact in-memory index.

5. **Dtype conversion fusion** — combine pread + dtype conversion in a single pass instead
   of constructing an intermediate tensor.

6. **Memory-mapped fallback** — for hot experts that are accessed every token, mmap might
   beat pread. Detect hot experts and switch strategy.

7. **Parallel expert reads** — read K expert weight sets in parallel using thread pool or
   io_uring (on Linux).

8. **Tensor construction** — `Tensor::from_raw_buffer` does a memcpy. Can we construct
   zero-copy tensors from the read buffer?

9. **Batched read_tensors()** — `TensorStorageProvider::read_tensors()` now detects contiguous
   tensors in the same shard and reads them in a single pread. Optimize the contiguity detection
   and buffer splitting for the gate+up+down pattern.

10. **load_tensor() convenience** — `SafetensorsStorage::load_tensor()` reads + converts dtype +
    moves to device. The dtype conversion could be fused with the read for common F16→F32 paths.

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
