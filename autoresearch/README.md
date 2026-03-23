# Cake Autoresearch

Autonomous optimization of inference throughput across models, backends, kernels, and network.

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
│   ├── tts/                    # TTS: diffusion head, DDPM, VAE, vocoder, mel
│   └── image-generation/       # FLUX/Stable Diffusion components
├── backends/                   # Backend-specific GPU/CPU optimization
│   ├── cuda/                   # NVIDIA CUDA kernels (ops.cu, ops.rs)
│   ├── metal/                  # Apple Metal shaders (ops.msl)
│   └── vulkan/                 # Vulkan/wgpu compute shaders (ops.wgsl)
├── kernels/                    # Kernel-specific fused operation optimization
│   ├── attention/              # Scaled dot-product attention
│   ├── fused-ops/              # silu_mul, rms_norm, add3, add_scaled, etc.
│   └── quantization/           # FP8 dequantization and weight preprocessing
└── network/                    # Distributed inference protocol
    └── protocol/               # Serialization, topology, auth, discovery
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

## Branching

Each task uses a **fixed branch name** matching its directory path:

```bash
git checkout -b autoresearch/backends/cuda       # CUDA kernels
git checkout -b autoresearch/models/tts          # TTS models
git checkout -b autoresearch/network/protocol    # Network protocol
# etc.
```

No date tags — one branch per task. Merge to main when done.

## Quick Start

```bash
# 1. Pick a task
cd autoresearch/network/protocol

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
   - **Clippy**: `cargo clippy -p cake-core --lib --tests -- -D warnings` (with features)
   - **Test**: `cargo test -p cake-core --lib`, `--test unit`, `--test protocol` — all three suites
   - **Benchmark**: `bash benchmark.sh` — measures performance
   - **Decide**: KEEP if faster (within noise margin) and all quality gates pass; DISCARD otherwise
   - **Record** result in `experiments.tsv`
   - **Repeat forever** until manually interrupted

3. **Metrics**: Each benchmark harness outputs a `BENCH_RESULT` line:
   ```
   BENCH_RESULT score=<lower_is_better_ns> tests=PASS/FAIL clippy=PASS/FAIL status=OK/FAIL
   ```

## Task Overview

### Models

| Task | Branch | Target | Benchmarks |
|------|--------|--------|------------|
| text-inference | `autoresearch/models/text-inference` | Attention, MLP, cache, blocks | attention, mlp, blocks, cache |
| moe | `autoresearch/models/moe` | MoE routing and dispatch | moe, expert_provider |
| tts | `autoresearch/models/tts` | Diffusion head, DDPM, VAE, vocoder | prediction_head, ddpm, connectors, mel, wav |
| image-generation | `autoresearch/models/image-generation` | FLUX/SD pipeline | flux |

### Backends

| Task | Branch | Hardware | Benchmarks |
|------|--------|----------|------------|
| cuda | `autoresearch/backends/cuda` | NVIDIA GPU | fused ops (GPU path) |
| metal | `autoresearch/backends/metal` | Apple Silicon | fused ops (GPU path) |
| vulkan | `autoresearch/backends/vulkan` | Vulkan 1.3+ GPU | vulkan benchmarks |

### Kernels

| Task | Branch | Target | Benchmarks |
|------|--------|--------|------------|
| attention | `autoresearch/kernels/attention` | Attention implementations | attention, linear_attn |
| fused-ops | `autoresearch/kernels/fused-ops` | Fused activation/norm ops | bench_utils fused ops |
| quantization | `autoresearch/kernels/quantization` | FP8/GPTQ dequantization | quantization, flux dequant |

### Network

| Task | Branch | Target | Benchmarks |
|------|--------|--------|------------|
| protocol | `autoresearch/network/protocol` | Serialization, topology, auth, discovery | protocol, serialization, discovery, topology, auth |

## Prerequisites

- Rust toolchain (stable)
- `cargo bench` support (divan benchmarks)
- For backend tasks: appropriate GPU + feature flags (cuda, metal, vulkan)
- For TTS tasks: `--features vibevoice,luxtts`
- No model downloads required — all benchmarks use synthetic tensors
