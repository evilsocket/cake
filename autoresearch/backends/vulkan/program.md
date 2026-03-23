# Vulkan Backend Autoresearch

Autonomous optimization of Vulkan/wgpu compute shaders.

## Objective

**Minimize Vulkan kernel execution time** for all fused operations on Vulkan-capable GPUs
(AMD, Intel, Steam Deck, any Vulkan 1.3+ device). The Vulkan backend uses wgpu with WGSL
compute shaders. Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/backends/vulkan`
2. **Read the in-scope files**:
   - `cake-core/src/backends/vulkan/ops.wgsl` — WGSL compute shaders. **Primary target.**
     Contains: elementwise ops (silu_mul, add3, add_scaled, rms_norm_gated, add_rms_norm,
     stable_softplus, exp_mul, sub_mul) and matrix operations (GEMV with tiled 4×64 layout,
     GEMM with 16×16 tiles).
   - `cake-core/src/backends/vulkan/mod.rs` — VulkanBackend struct. GPU buffer cache
     (weight cache by TensorId, output pool by power-of-2 size), pipeline management,
     dispatch logic, wgpu device/queue management.
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait definition.
   - `cake-core/benches/bench_vulkan.rs` — Vulkan-specific benchmarks (GEMV, GEMM, silu_mul,
     add3, rms_norm, MLP, dispatch overhead).
   - `cake-core/benches/bench_utils.rs` — Backend operation benchmarks.
3. **Run prepare.sh**: `bash autoresearch/backends/vulkan/prepare.sh`
4. **Confirm baseline** and start experimenting.

**IMPORTANT**: This task requires a Vulkan 1.3+ GPU. Build with `--features vulkan`.

## Files You May Modify

- `cake-core/src/backends/vulkan/ops.wgsl` — WGSL compute shaders. Workgroup sizes,
  tiling strategy, shared memory (workgroup var) usage, vectorized loads (vec4),
  reduction patterns, memory access coalescing.
- `cake-core/src/backends/vulkan/mod.rs` — VulkanBackend implementation. Buffer cache
  strategy, pipeline creation, dispatch dimensions, uniform buffer packing, wgpu
  configuration.

## Files You Must NOT Modify

- `autoresearch/backends/vulkan/benchmark.sh`
- `autoresearch/backends/vulkan/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_vulkan.rs`
- `cake-core/benches/bench_utils.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused shader optimization hypothesis.
2. **Implement** the change in ops.wgsl / mod.rs.
3. **Commit**: `git add cake-core/src/backends/vulkan/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --features vulkan --lib --tests -- -D warnings && cargo test -p cake-core --features vulkan --lib && cargo test -p cake-core --features vulkan --test unit && cargo test -p cake-core --features vulkan --test protocol`
5. **Benchmark**: `bash autoresearch/backends/vulkan/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Vulkan/wgpu Architecture Constraints

### WGSL Limitations
WGSL is more restrictive than CUDA/MSL:
- No recursive functions
- No function pointers
- No dynamic indexing of workgroup arrays in some implementations
- Limited atomic operations
- No native half-precision in many implementations (f16 extension not universal)

### Workgroup Size
The current GEMV uses workgroup_size(256, 1, 1) split into 4 row-groups of 64 threads.
This is tuned for RDNA 2 (Steam Deck) with 64-wide wavefronts. AMD GPUs process 64
threads per wavefront; NVIDIA processes 32 per warp. Workgroup sizes should be multiples
of 64 for AMD and 32 for NVIDIA.

### Buffer Cache
The Vulkan backend caches weight buffers by TensorId (stable across forwards) and pools
output buffers by power-of-2 size. This avoids repeated GPU memory allocation. Do not
break the cache invalidation logic.

### wgpu Abstraction
The backend uses wgpu (version 24), which abstracts over Vulkan, Metal, DX12, and WebGPU.
However, optimizations should target Vulkan specifically (the primary use case is
Linux/Steam Deck with AMD GPUs).

### Dispatch Overhead
wgpu has higher per-dispatch overhead than native Vulkan/Metal due to validation and
abstraction layers. Minimizing the number of dispatches (via kernel fusion) is more
impactful here than on other backends.

## Promising Areas to Explore

1. **GEMV tiling** — the current 4×64 tiling in GEMV was tuned for RDNA 2. Experiment
   with different tile sizes (2×128, 8×32) for different matrix shapes.

2. **Shared memory (workgroup var) utilization** — the input vector is loaded into
   workgroup memory. Can more data be shared to reduce global memory reads?

3. **vec4 vectorized loads** — use `vec4<f32>` for memory reads where alignment permits.
   This can 4x effective bandwidth for large contiguous reads.

4. **Kernel fusion** — combine sequential elementwise operations into single dispatches.
   Each wgpu dispatch has ~50-100µs overhead; fusing reduces dispatch count.

5. **Output buffer pool tuning** — the power-of-2 pooling strategy may waste memory for
   odd-sized tensors. A buddy allocator or size-class pool might be more efficient.

6. **Reduction optimization** — rms_norm and attention softmax use sequential reductions.
   Tree-style parallel reduction within workgroups can be faster.

7. **Subgroup operations** — WGSL's `subgroupAdd`, `subgroupBroadcast` (where available)
   can replace workgroup memory reductions. Check for `subgroups` feature support.

8. **Pipeline caching** — wgpu recompiles shaders on pipeline creation. Can more
   aggressive pipeline caching reduce cold-start overhead?

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## Safety

- Do not allocate more than 256MB of new GPU buffers
- Do not change the ComputeBackend trait signature
- Do not break the buffer cache invalidation logic
- Test on actual Vulkan hardware if possible

## NEVER STOP

Run autonomously until manually interrupted.
