# Metal Backend Autoresearch

Autonomous optimization of Metal compute shaders for Apple Silicon.

## Objective

**Minimize Metal kernel execution time** for all fused operations on Apple Silicon GPUs.
Metal shaders are the inner loop of every forward pass on Mac/iOS. The GPU is typically
memory-bandwidth-bound at ~200-400 GiB/s depending on the chip. Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/backends/metal`
2. **Read the in-scope files**:
   - `cake-core/src/backends/metal/ops.msl` — Metal Shading Language compute kernels.
     **Primary target.** Contains all fused ops: silu_mul, stable_softplus, rms_norm_gated,
     add_rms_norm, rms_norm_channel, add3, add_scaled, depthwise_conv1d_silu,
     depthwise_conv1d_bias, exp_mul, sub_mul, adaln_modulate, f8e4m3 dequantization.
   - `cake-core/src/backends/metal/mod.rs` — MetalBackend struct, ComputeBackend trait
     implementation, MSL compilation, command buffer management, synchronization logic.
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait definition.
   - `cake-core/benches/bench_utils.rs` — Backend operation benchmarks.
3. **Run prepare.sh**: `bash autoresearch/backends/metal/prepare.sh`
4. **Confirm baseline** and start experimenting.

**IMPORTANT**: This task requires a Mac with Apple Silicon or Intel GPU. Build with `--features metal`.

## Files You May Modify

- `cake-core/src/backends/metal/ops.msl` — Metal shaders. Threadgroup sizes, SIMD group
  operations, memory access patterns, half-precision ALU, register pressure, kernel fusion.
- `cake-core/src/backends/metal/mod.rs` — MetalBackend implementation. Command buffer
  management, pipeline state caching, dispatch configuration, synchronization frequency.

## Files You Must NOT Modify

- `autoresearch/backends/metal/benchmark.sh`
- `autoresearch/backends/metal/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_utils.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused shader optimization hypothesis.
2. **Implement** the change in ops.msl / mod.rs.
3. **Commit**: `git add cake-core/src/backends/metal/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --features metal --lib --tests -- -D warnings && cargo test -p cake-core --features metal --lib && cargo test -p cake-core --features metal --test unit && cargo test -p cake-core --features metal --test protocol`
5. **Benchmark**: `bash autoresearch/backends/metal/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Apple Silicon GPU Constraints

### Unified Memory Architecture
Apple Silicon shares memory between CPU and GPU. There is no discrete VRAM — all tensors
live in unified memory. This means no PCIe transfer overhead, but also means CPU and GPU
compete for memory bandwidth.

### SIMD Width
Apple GPU SIMD group width is 32 threads (like NVIDIA warps). Use `simd_sum`, `simd_shuffle`,
and other SIMD group operations for fast reductions instead of threadgroup memory.

### Command Buffer Accumulation
Metal command buffers must be flushed periodically. The existing code flushes every ~25
operations via `synchronize()`. If >25 commands accumulate without a flush, performance
drops ~50x. Do not remove synchronization calls.

### Half-Precision ALU
Apple GPU has native half-precision (float16) ALU at 2x the throughput of float32.
Using `half` types for intermediate values can double effective compute throughput
when the operation is ALU-bound.

### MSL Compilation
Metal shaders are compiled from MSL source at runtime (not ahead-of-time). The source
is embedded as a Rust string constant. Changes to ops.msl require a rebuild but not
a separate shader compilation step.

## Promising Areas to Explore

1. **Half-precision intermediates** — use `half` for intermediate values in silu_mul,
   rms_norm, and add_rms_norm where full f32 precision isn't needed. 2x ALU throughput.

2. **SIMD group reductions** — replace threadgroup-memory-based reductions with
   `simd_sum` / `simd_shuffle_xor` for rms_norm and attention softmax. Eliminates
   threadgroup barrier overhead.

3. **Threadgroup size tuning** — experiment with threadgroup sizes for different
   operation types. Small elementwise ops may benefit from smaller threadgroups
   (less launch overhead), while reductions need larger threadgroups.

4. **Kernel fusion opportunities** — can sequential operations (e.g., norm → projection
   or dequant → activation) be combined into a single kernel dispatch?

5. **Memory access coalescing** — ensure threads in a SIMD group access consecutive
   memory addresses. Strided access patterns waste bandwidth.

6. **Register pressure management** — too many live variables cause register spilling
   to device memory. Profile register usage and reduce live variable count.

7. **Texture memory** — for read-only weight data, texture samplers may provide
   better cache behavior than raw buffer reads.

8. **Command buffer batching** — reduce the number of command buffer commits by
   encoding more operations per buffer. But respect the 25-command limit.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## Safety

- Do not allocate more than 200MB of new Metal buffers
- Do not remove `synchronize()` calls — this causes 50x slowdowns
- Do not change the ComputeBackend trait signature
- Do not modify pipeline state creation patterns without testing

## NEVER STOP

Run autonomously until manually interrupted.
