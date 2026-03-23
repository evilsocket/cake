# Quantization Kernel Autoresearch

Autonomous optimization of quantization and dequantization operations.

## Objective

**Minimize dequantization latency** — FP8 (E4M3) dequantization, GPTQ 4-bit dequantization,
blockwise FP8, and weight preprocessing. Dequantization runs on every linear layer forward pass
for quantized models, making it one of the highest-frequency operations. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/quantization/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/quantization/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/utils/quantization.rs` — GPTQ 4-bit dequantization, FP8 blockwise
     dequantization, weight format detection, quantization utilities.
   - `cake-core/src/backends/cpu/mod.rs` — CPU FP8 dequant: software decoding of E4M3
     format to F32/F16/BF16.
   - `cake-core/src/backends/cuda/ops.cu` — CUDA FP8 dequant kernels (hardware path for
     SM 8.9+, software for older).
   - `cake-core/src/backends/metal/ops.msl` — Metal FP8 dequant shaders.
   - `cake-core/src/backends/vulkan/ops.wgsl` — Vulkan FP8 dequant shaders.
   - `cake-core/src/models/flux/` — Fp8Linear layer (FP8 weights + block scaling).
   - `cake-core/benches/bench_quantization.rs` — Dequantization benchmarks (GPTQ 4-bit,
     FP8 blockwise).
   - `cake-core/benches/bench_flux.rs` — FP8 linear forward, F8 dequant benchmarks.
   - `cake-core/benches/bench_utils.rs` — fp8_linear_forward_cpu, fp8_linear_forward_with_bias_cpu.
4. **Run prepare.sh**: `bash autoresearch/kernels/quantization/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/utils/quantization.rs` — GPTQ dequantization algorithm, FP8 blockwise
  dequant, bit manipulation, scaling factor application.
- `cake-core/src/backends/cpu/mod.rs` — CPU FP8 dequant (f8e4m3_to_f32/f16/bf16).
  Software E4M3 decoding: exponent extraction, mantissa recovery, denormal handling.
- `cake-core/src/models/flux/fp8_linear.rs` (or equivalent) — Fp8Linear forward pass:
  dequant → matmul → optional bias. Fusion opportunities.

## Files You Must NOT Modify

- `autoresearch/kernels/quantization/benchmark.sh`
- `autoresearch/kernels/quantization/prepare.sh`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- `cake-core/benches/bench_quantization.rs`
- `cake-core/benches/bench_flux.rs`
- `cake-core/benches/bench_utils.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused quantization optimization hypothesis.
2. **Implement** the change.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo test -p cake-core --lib --test unit`
5. **Benchmark**: `bash autoresearch/kernels/quantization/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Quantization Format Constraints

### FP8 E4M3 Format
- 1 sign bit, 4 exponent bits, 3 mantissa bits
- Bias = 7, range ≈ ±448, precision ≈ 3 decimal digits
- Denormals: exponent=0, mantissa encodes sub-normal values
- Special: no infinity, NaN only when all bits set
- Block scaling: per-block scale factor (typically 32 or 128 elements per block)

### GPTQ 4-bit Format
- 4 bits per weight, packed 8 weights per u32
- Group quantization: shared zero point + scale per group (typically 128 elements)
- Bit extraction: shift + mask to extract nibbles
- Dequant: `(nibble - zero_point) * scale`

### Numerical Correctness
Dequantization must be bit-exact. The FP8 E4M3 software decoder must produce identical
results to hardware FP8 on SM 8.9+ GPUs. Do not introduce approximations.

## Promising Areas to Explore

1. **Lookup table for FP8** — the 256 possible E4M3 values have fixed F32 equivalents.
   A 256-entry LUT (1KB) may be faster than bit manipulation for CPU dequant.

2. **Vectorized nibble extraction** — GPTQ dequant extracts 4-bit values from packed u32.
   SIMD instructions can extract multiple nibbles in parallel.

3. **Fused dequant + matmul** — instead of dequantizing the entire weight matrix, dequantize
   rows on-the-fly during the matrix multiply. This trades compute for memory bandwidth.

4. **Block-level parallelism** — FP8 blockwise dequant processes blocks independently.
   Ensure blocks are processed in parallel (rayon) with optimal chunk sizes.

5. **Cache-friendly dequant ordering** — process weights in the order they'll be consumed
   by the subsequent matmul to maximize L1/L2 cache hits.

6. **Reduce branch density in E4M3** — the denormal handling creates branches. Branch-free
   formulations using conditional moves or bit tricks may be faster.

7. **Half-precision output path** — when the target dtype is F16, avoid the F32 intermediate.
   Decode directly to F16 to halve memory writes.

8. **Batch dequantization** — process multiple weight tensors in a single pass when they
   share the same quantization parameters.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## NEVER STOP

Run autonomously until manually interrupted.
