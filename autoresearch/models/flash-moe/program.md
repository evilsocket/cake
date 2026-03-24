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

- `cake-core/src/utils/tensor_storage.rs` — SafetensorsStorage: mmap-based reads, read_tensor_slice for byte-range access, tensor_meta for metadata-only queries.
- `cake-core/src/models/common/disk_expert_provider.rs` — DiskExpertProvider: read_expert_weight (GPTQ path), read_stacked_expert_slice (stacked switch_mlp format with per-expert byte-range pread + affine dequant), new_stacked constructor.
- `cake-core/src/models/common/expert_provider.rs` — ExpertProvider trait, ExpertWeights struct.
- `cake-core/src/utils/gptq.rs` — dequantize_packed_4bit (affine 4-bit dequant, handles 2D and 3D), dequantize_gptq_4bit (standard GPTQ).

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

### High Priority (biggest impact on 35B model at 0.22 tok/s)

1. **GPTQ affine dequantization speed** — Each expert read requires dequantizing a 2D
   slice from the stacked 3D tensor: `read_stacked_expert_slice()` reads raw bytes via mmap,
   builds a Tensor, then calls `dequantize_packed_4bit()` (CPU-bound, uses rayon). With 8
   active experts × 3 projections × 40 layers = 960 dequantizations per token. Profile and
   optimize the inner loop (bit extraction, scale/bias multiply, F32→F16 conversion).

2. **Expert weight caching** — Currently every token re-reads and re-dequantizes all 8
   active experts per layer from disk. An LRU cache of recently-used dequantized expert
   weights (in GPU memory) could eliminate redundant dequantizations. MoE models often
   route to the same popular experts — cache hit rate could be 50-80%.

3. **Parallel expert dequantization** — The 8 active experts per layer are dequantized
   sequentially. Launch dequantization on a thread pool while the shared expert runs
   (shared expert always executes, providing a natural overlap window).

4. **Stacked tensor byte-range optimization** — `read_stacked_expert_slice()` calls
   `tensor_meta()` + `read_tensor_slice()` separately. Fuse into a single operation that
   computes the byte range and reads in one step. Also consider reading all 3 projections
   (gate+up+down) for one expert in a single contiguous read if they're adjacent in the shard.

5. **Prefetching with madvise** — After the router selects top-K experts, issue
   `posix_madvise(WILLNEED)` on the mmap'd regions for the K experts' weight slices
   while the shared expert runs. This triggers OS readahead on NVMe.

### Medium Priority

6. **Buffer reuse** — `read_stacked_expert_slice()` allocates new `Vec<u8>` per read.
   A reusable buffer pool could reduce allocation pressure.

7. **Batch pread for individual experts** — For non-stacked format, read
   gate_proj+up_proj+down_proj in a single pread if they're contiguous in the shard.
   Already partially implemented via `read_tensors()` contiguity detection.

8. **Dtype conversion fusion** — Combine mmap read + dequantization + F16 conversion
   in a single pass instead of materializing intermediate F32 tensors.

### Lower Priority (already fast enough)

9. **Parallel expert reads** — read K expert weight sets in parallel using thread pool or
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
