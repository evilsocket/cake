# CUDA Backend Autoresearch

Autonomous optimization of CUDA compute kernels for NVIDIA GPUs.

## Objective

**Minimize CUDA kernel execution time** for all fused operations: silu_mul, stable_softplus,
rms_norm_gated, add_rms_norm, add3, add_scaled, exp_mul, sub_mul, depthwise_conv1d, FP8
dequantization, and attention. These kernels are the inner loop of every forward pass on
NVIDIA hardware. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/cuda/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/cuda/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/backends/cuda/ops.cu` — Raw CUDA PTX kernels. **Primary target.**
     Contains all fused operation kernels: silu_mul, stable_softplus, rms_norm_gated,
     add_rms_norm, rms_norm_channel, add3, add_scaled, depthwise_conv1d_silu,
     depthwise_conv1d_bias, exp_mul, sub_mul, adaln_modulate, f8e4m3 dequantization.
   - `cake-core/src/backends/cuda/ops.rs` — Rust-side CustomOp2 adapters that dispatch
     to CUDA kernels. Handles dtype dispatch, grid/block sizing, kernel launch.
   - `cake-core/src/backends/cuda/mod.rs` — CudaBackend struct, ComputeBackend trait
     implementation, attention dispatch (flash-attn vs manual).
   - `cake-core/src/backends/cuda/compat/` — CUDA version compatibility layer.
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait definition.
   - `cake-core/benches/bench_utils.rs` — Backend operation benchmarks (fused_*_gpu variants).
4. **Run prepare.sh**: `bash autoresearch/backends/cuda/prepare.sh`
5. **Confirm baseline** and start experimenting.

**IMPORTANT**: This task requires an NVIDIA GPU with CUDA support. Build with `--features cuda`.

## Files You May Modify

- `cake-core/src/backends/cuda/ops.cu` — CUDA kernel source. Thread/block dimensions, shared
  memory usage, register pressure, loop unrolling, warp-level primitives, memory coalescing,
  FMA chains, vectorized loads.
- `cake-core/src/backends/cuda/ops.rs` — Kernel dispatch: grid/block dimensions, dtype routing,
  CustomOp2 shape inference. Kernel launch parameters.
- `cake-core/src/backends/cuda/mod.rs` — CudaBackend implementation. Attention dispatch logic,
  weight preprocessing, synchronization.

## Files You Must NOT Modify

- `autoresearch/backends/cuda/benchmark.sh`
- `autoresearch/backends/cuda/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_utils.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused kernel optimization hypothesis.
2. **Implement** the change in ops.cu / ops.rs / mod.rs.
3. **Commit**: `git add cake-core/src/backends/cuda/ && git commit -m "<description>"`
4. **Quality gate**: `cargo test -p cake-core --features cuda --lib --test unit`
5. **Benchmark**: `bash autoresearch/backends/cuda/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## GPU Architecture Constraints

### Memory Bandwidth Bound
Most fused kernels are **memory-bandwidth-bound**, not compute-bound. On modern NVIDIA GPUs
(Ampere, Ada, Hopper), global memory bandwidth is the bottleneck. Pure ALU tricks that don't
reduce memory traffic are unlikely to help.

### Warp Size
NVIDIA warp size is 32 threads. All reductions, shuffles, and synchronization must respect
warp boundaries. Use `__shfl_xor_sync` and `__shfl_down_sync` for warp-level reductions
instead of shared memory when possible.

### PTX Compilation
`ops.cu` is compiled to PTX at build time by NVCC (via `bindgen_cuda`). The PTX is embedded
as a string constant `FUSED_OPS_PTX` and loaded by candle's CUDA runtime. Changes to ops.cu
require a full rebuild.

### Dtype Support
Kernels must support F32, F16, and BF16. Use C++ templates or `if constexpr` patterns.
The Rust side dispatches to the correct kernel variant based on tensor dtype.

### Flash Attention
When the `flash-attn` feature is enabled, attention uses candle's flash-attn implementation
(optimized Triton-style kernels). The manual attention path is a fallback. Focus optimization
on the fused ops kernels rather than reimplementing attention.

## Promising Areas to Explore

1. **Vectorized loads** — use `float4`/`half2` for coalesced global memory reads. Many
   kernels process elements one at a time; vectorized loads can 2-4x bandwidth utilization.

2. **Warp-level reductions** — replace shared memory reductions with `__shfl_down_sync`
   in rms_norm kernels. Eliminates shared memory bank conflicts and sync barriers.

3. **Thread coarsening** — process multiple output elements per thread. Amortizes the
   cost of loading shared data (e.g., normalization weight vectors, scale factors).

4. **Register blocking for convolution** — the depthwise conv1d kernels can keep the
   kernel weights in registers instead of repeatedly loading from global memory.

5. **FP8 dequant fusion** — combine FP8 dequantization with the subsequent matmul or
   activation. Eliminates an intermediate tensor write/read.

6. **Occupancy tuning** — experiment with block sizes (64, 128, 256, 512) and shared
   memory allocation to maximize SM occupancy for different kernel shapes.

7. **Persistent kernels** — for small tensors, kernel launch overhead dominates. Can
   multiple operations be fused into a single persistent kernel?

8. **FMA instruction chains** — reorder multiply-add operations to maximize FMA utilization.
   NVIDIA GPUs have dedicated FMA units that are faster than separate mul + add.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## Safety

- Do not allocate more than 256MB of GPU memory in new buffers
- Do not change the ComputeBackend trait signature
- Do not modify the PTX loading mechanism in ops.rs
- Kernel changes must be backward-compatible with SM 7.0+ (Volta and newer)

## NEVER STOP

Run autonomously until manually interrupted. Each experiment takes ~1-2 minutes.
