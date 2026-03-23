# ROCm Backend Autoresearch

Autonomous optimization of the ROCm/HIP backend for AMD GPUs.

## Objective

**Minimize ROCm backend operation latency** — HIP kernel dispatch, rocBLAS GEMM tuning,
buffer management, and fused op implementations. Targets AMD GPUs including Steam Deck
(RDNA 2, Van Gogh APU) and discrete AMD cards. Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/rocm`
2. **Read the in-scope files**:
   - `cake-core/src/backends/rocm/mod.rs` — RocmBackend struct, ComputeBackend impl, buffer cache.
   - `cake-core/src/backends/rocm/ffi.rs` — HIP + rocBLAS FFI bindings via libloading.
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait (reference for method contracts).
4. **Run prepare.sh**: `bash autoresearch/backends/rocm/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/backends/rocm/mod.rs` — Backend implementation, buffer cache, dispatch logic.
- `cake-core/src/backends/rocm/ffi.rs` — FFI function loading, type definitions.

## Files You Must NOT Modify

- `autoresearch/backends/rocm/benchmark.sh`
- `autoresearch/backends/rocm/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- Any test or bench files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/backends/rocm/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --features rocm --lib --tests -- -D warnings && cargo test -p cake-core --features rocm --lib && cargo test -p cake-core --features rocm --test unit && cargo test -p cake-core --features rocm --test protocol` — must pass.
5. **Benchmark**: `bash autoresearch/backends/rocm/benchmark.sh`
6. **Parse** the BENCH_RESULT line.
7. **If build/test failed**: quick fix (2 tries), then revert and log as `crash`.
8. **Record** in `experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND all gates pass.
   - **DISCARD** otherwise. Revert: `git reset --hard HEAD~1`
10. **Repeat**

## Architecture Constraints

### Runtime Library Loading
ROCm uses `libloading` to dlopen `libamdhip64.so` and `librocblas.so` at runtime.
Do not add compile-time HIP dependencies — the backend must remain optional.

### Buffer Cache
The backend maintains a buffer cache for GPU-side tensors. Changes to the cache eviction
policy or sizing affect memory pressure on UMA devices (Steam Deck has shared 16GB).

### rocBLAS GEMM
Matrix multiplication uses rocBLAS sgemm. The Tensile kernel library selection depends
on the GPU architecture (gfx1030/gfx1033 for RDNA 2). Not all matrix sizes have optimized
kernels — benchmark various (M,N,K) combinations.

## Promising Areas to Explore

1. **Buffer pool sizing** — Tune the cache capacity for different model sizes.
2. **rocBLAS workspace** — Pre-allocate workspace for GEMM operations.
3. **HIP stream management** — Overlap compute with memory transfers.
4. **Batch dispatch** — Combine multiple small ops into fewer HIP launches.
5. **Memory layout** — Column-major vs row-major for rocBLAS compatibility.

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
