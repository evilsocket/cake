# Cake Autoresearch

Autonomous optimization of inference throughput across models, backends, and kernels.

## Concept

Each autoresearch task is a self-contained, agent-driven optimization loop. An AI agent
reads the `program.md` for a task, runs `prepare.sh` to establish a baseline, then
enters an infinite loop: **hypothesize → implement → benchmark → decide → repeat**.

This is adapted from [flash-moe/autoresearch](https://github.com/Alexintosh/flash-moe/tree/main/autoresearch).

## Structure

```
autoresearch/
├── models/                     # Model-specific optimization
│   ├── text-inference/         # Attention, MLP, cache, transformer blocks
│   ├── moe/                    # Mixture-of-Experts routing and dispatch
│   └── image-generation/       # FLUX/Stable Diffusion components
├── backends/                   # Backend-specific GPU/CPU optimization
│   ├── cuda/                   # NVIDIA CUDA kernels (ops.cu, ops.rs)
│   ├── metal/                  # Apple Metal shaders (ops.msl)
│   └── vulkan/                 # Vulkan/wgpu compute shaders (ops.wgsl)
└── kernels/                    # Kernel-specific fused operation optimization
    ├── attention/              # Scaled dot-product attention
    ├── fused-ops/              # silu_mul, rms_norm, add3, add_scaled, etc.
    └── quantization/           # FP8 dequantization and weight preprocessing
```

Each task directory contains:

| File | Purpose |
|------|---------|
| `program.md` | Agent instructions — objective, constraints, experiment loop, ideas |
| `benchmark.sh` | Fixed benchmark harness — **do not modify** |
| `prepare.sh` | One-time setup: build, run baseline, initialize experiment log |

Generated at runtime (gitignored):

| File | Purpose |
|------|---------|
| `baseline.txt` | Baseline benchmark score from `prepare.sh` |
| `experiments.tsv` | Tab-separated experiment log |

## Quick Start

```bash
# 1. Pick a task
cd autoresearch/kernels/fused-ops

# 2. Run setup (builds, benchmarks baseline)
bash prepare.sh

# 3. Start the agent with program.md
#    The agent reads program.md and enters the optimization loop.
```

## How It Works

1. **Prepare**: `prepare.sh` builds the project, runs the benchmark suite for the task,
   saves the baseline score to `baseline.txt`, and initializes `experiments.tsv`.

2. **Loop** (from `program.md`):
   - **Propose** a single, focused optimization hypothesis
   - **Implement** the change in the allowed source files
   - **Build**: `cargo build -p cake-core` (with backend features if needed)
   - **Test**: `cargo test -p cake-core` — quality gate must pass
   - **Benchmark**: `bash benchmark.sh` — measures performance
   - **Decide**: KEEP if faster (within noise margin) and tests pass; DISCARD otherwise
   - **Record** result in `experiments.tsv`
   - **Repeat forever** until manually interrupted

3. **Metrics**: Each benchmark harness outputs a `BENCH_RESULT` line:
   ```
   BENCH_RESULT score=<lower_is_better_ns> tests=PASS/FAIL clippy=PASS/FAIL status=OK/FAIL
   ```

## Task Overview

### Models

| Task | Target | Primary Metric | Benchmarks |
|------|--------|----------------|------------|
| text-inference | Common text model components | Forward pass time | attention, mlp, blocks, cache |
| moe | MoE routing and expert dispatch | MoE forward time | moe, expert_provider |
| image-generation | FLUX/SD image pipeline | Component time | flux |

### Backends

| Task | Target | Hardware | Benchmarks |
|------|--------|----------|------------|
| cuda | CUDA kernels (ops.cu) | NVIDIA GPU | fused ops (GPU path) |
| metal | Metal shaders (ops.msl) | Apple Silicon | fused ops (GPU path) |
| vulkan | WGSL compute shaders (ops.wgsl) | Vulkan 1.3+ GPU | vulkan benchmarks |

### Kernels

| Task | Target | Primary Metric | Benchmarks |
|------|--------|----------------|------------|
| attention | Attention implementations | Attention forward time | attention, linear_attn |
| fused-ops | Fused activation/norm ops | Per-op latency | bench_utils fused ops |
| quantization | FP8/GPTQ dequantization | Dequant throughput | quantization, flux dequant |

## Prerequisites

- Rust toolchain (stable)
- `cargo bench` support (divan benchmarks)
- For backend tasks: appropriate GPU + feature flags (cuda, metal, vulkan)
- No model downloads required — all benchmarks use synthetic tensors
