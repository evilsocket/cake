# Cake Development Guide

## Cluster Machines

| Machine | Role | GPU | VRAM | OS | Work Dir | SSH |
|---------|------|-----|------|----|----------|-----|
| **blade.local** | Master (local) | RTX 3080 Laptop | 16 GB | Linux | `/home/evilsocket/Lab/cake` | N/A |
| **bahamut.local** | Worker | 2× TITAN X Pascal | 2×12 GB | Linux | `~/Lab/cake` | `ssh bahamut.local` |
| **stevie.local** | Worker | Apple M3 Pro | 36 GB unified | macOS | `~/Lab/cake` | `ssh stevie.local` |

## Build Commands

```bash
# blade.local (local, CUDA)
cargo build --release --features cuda

# bahamut.local (CUDA — MUST use cuda-12.4, driver only supports up to 12.4)
CUDA_HOME=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 cargo build --release --features cuda

# stevie.local (Metal)
cargo build --release --features metal

# Vulkan (Steam Deck, AMD GPUs, any Vulkan 1.3+ device)
cargo build --release --features vulkan
```

## Acceleration Features

| Feature | Platform | Backend | Best For | Notes |
|---------|----------|---------|----------|-------|
| `metal` | macOS (Apple Silicon) | GPU via MPS + custom MSL kernels | Primary inference on Mac | Fastest option on Apple Silicon (~42 tok/s on M3 Pro) |
| `cuda` | Linux (NVIDIA GPU) | GPU via cuBLAS/cuDNN | Primary inference on Linux | Requires CUDA toolkit matching driver version |
| `accelerate` | macOS | CPU via Apple Accelerate (AMX) | CPU-only F32 inference on Mac | 2.7x faster than pure-Rust for F32 matmul; no F16 support |
| `vulkan` | Any (Vulkan 1.3+) | GPU via Vulkan compute shaders | Steam Deck, AMD GPUs | Portable but less optimized than Metal/CUDA |
| (none) | Any | CPU via pure-Rust `gemm` | Portable CPU fallback | F16 weights stay F16, avoids bandwidth doubling |

**When to use which:**
- **Apple Silicon (stevie.local):** Use `--features metal`. Metal is 1.6x faster than CPU F16 (42 vs 26 tok/s). The `accelerate` feature doesn't help with Metal and doesn't support F16 matmul, so CPU F16 (default, no features) is actually faster than `accelerate` with F32 (26 vs 23 tok/s).
- **NVIDIA GPU (blade/bahamut):** Use `--features cuda`. Add `flash-attn` for flash attention support.
- **CPU-only with F32 models:** Use `--features accelerate` on macOS for 2.7x faster F32 matmul. On Linux, consider linking against MKL or OpenBLAS.
- **CPU-only with F16 models:** Use no features — pure-Rust `gemm` with F16 avoids the 2x memory bandwidth penalty of converting to F32.

## Interactive Chat

```bash
# Local mode (loads model, starts TUI chat — no server needed)
cake chat evilsocket/Qwen3-0.6B

# Remote mode (connects to a running API server)
cake chat --server http://localhost:8080
```

The TUI streams tokens in real time and detects `<think>` tags — showing a "thinking..." indicator with reasoning in gray, then the final response in white.

## Model Management

```bash
cake pull evilsocket/Qwen3-0.6B       # download from HuggingFace
cake list                              # show all cached models
cake rm evilsocket/Qwen3-0.6B         # delete a cached model (with confirmation)
cake rm Qwen3-0.6B                    # short name also works (suffix match)
```

`cake rm` shows the model name, path, and size before asking for confirmation. For HF cache models it removes the entire `models--org--name/` directory. Refuses to delete local (non-cache) models.

## Run Commands (Qwen3.5-0.8B cluster)

```bash
# Workers first (on each machine):
# bahamut:
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ./target/release/cake run \
  evilsocket/Qwen3.5-0.8B --name bahamut \
  --topology topology-0.8B.yml --address 0.0.0.0:10128

# stevie:
./target/release/cake run \
  evilsocket/Qwen3.5-0.8B --name stevie \
  --topology topology-0.8B.yml --address 0.0.0.0:10128

# Master (blade, local):
./target/release/cake run evilsocket/Qwen3.5-0.8B \
  "Explain quantum computing in simple terms" \
  --topology topology-0.8B.yml
```

## Model: evilsocket/Qwen3.5-0.8B

- **Architecture**: Qwen3_5ForConditionalGeneration
- **Layers**: 24 (18 GatedDeltaNet linear attn + 6 full attn)
- **Hidden size**: 1024
- **Layer prefix**: `model.language_model.layers.{N}`
- **Location**: HuggingFace cache on all 3 machines (`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/`)
- **Size**: ~1.6 GB in F16

## Topology: `topology-0.8B.yml`

24 layers split evenly: bahamut 0-7, stevie 8-15, blade master keeps 16-23.

## Self Improving Loop

This is an iterative optimization process for maximizing inference speed:

### Process

1. **Instrument**: Add timing/profiling logs to hot code paths (forward pass, attention, MLP, network, serialization)
2. **Commit & push**: Commit changes, push to origin
3. **Deploy**: Pull on all 3 machines via SSH, rebuild with appropriate features (cuda/metal)
4. **Run experiment**: Start workers on bahamut and stevie, then master on blade with a test prompt
5. **Collect metrics**: Capture tok/s, per-layer timing, network latency from logs
6. **Analyze**: Identify the current bottleneck (slowest component)
7. **Optimize**: Make targeted code changes to address the bottleneck
8. **Verify**: Run tests (`cargo test --features cuda`), verify coherent output
9. **Compare**: Compare metrics with previous iteration
10. **Repeat**: Go to step 1 until no further gains are possible

### Key metrics to track
- **tok/s** (tokens per second) — the primary metric
- **per-layer forward time** (ms) — identifies slow layers
- **network round-trip time** (ms) — identifies network bottlenecks
- **embedding + lm_head time** (ms) — head/tail overhead
- **total forward pass time** (ms) — end-to-end per token

### Deploy script pattern
```bash
# Push from blade
git push

# Pull & build on bahamut
ssh bahamut.local "cd ~/Lab/cake && git pull && CUDA_HOME=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 cargo build --release --features cuda"

# Pull & build on stevie
ssh stevie.local "cd ~/Lab/cake && git pull && cargo build --release --features metal"
```

### Run experiment pattern
```bash
# Start workers (background SSH sessions)
ssh bahamut.local "cd ~/Lab/cake && LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ./target/release/cake run evilsocket/Qwen3.5-0.8B --name bahamut --topology topology-0.8B.yml --address 0.0.0.0:10128"
ssh stevie.local "cd ~/Lab/cake && ./target/release/cake run evilsocket/Qwen3.5-0.8B --name stevie --topology topology-0.8B.yml --address 0.0.0.0:10128"

# Run master (blade, local)
./target/release/cake run evilsocket/Qwen3.5-0.8B "Explain quantum computing in simple terms" --topology topology-0.8B.yml
```

## Testing

### Unit Tests (765+ tests, <3s)

All tests run offline — no model downloads, no GPU, no network servers required.

```bash
# Run all tests
cargo test -p cake-core

# Run only inline library tests (fastest, ~0.02s)
cargo test -p cake-core --lib

# Run external unit tests (attention, blocks, MoE, topology, client/worker mock)
cargo test -p cake-core --test unit

# Run protocol tests
cargo test -p cake-core --test protocol

# Measure coverage (requires cargo-llvm-cov)
cargo llvm-cov --test unit --test protocol --lib -p cake-core --summary-only
```

Test helpers are in `cake-core/tests/unit_tests/helpers.rs` — reuse `test_config()`, `make_tensor()`,
`make_vb_attention()`, `make_vb_mlp()`, `make_vb_transformer_block()`, `make_cache()` for any new tests.

For model block tests, use the `make_context()` + `Forwarder::load()` pattern in `test_blocks.rs`.

### Benchmarks (divan)

```bash
# Run all benchmarks
cargo bench -p cake-core

# Run specific benchmark group
cargo bench -p cake-core -- attention
cargo bench -p cake-core -- protocol
cargo bench -p cake-core -- moe

# Quick smoke test (1 sample per benchmark)
DIVAN_SAMPLE_COUNT=1 cargo bench -p cake-core
```

Benchmarks use divan and are in `cake-core/benches/`. They share helpers with unit tests
and run on CPU with small dimensions (hidden=64) for fast iteration. ~130 benchmarks covering:
attention, MLP, GatedDeltaNet, MoE, FLUX (timestep embed, pos embed, Fp8Linear, VAE ResnetBlock),
full blocks, cache, serialization, protocol (tensor encode/decode, model data transfer with
zstd compression and CRC32 checksums), auth, discovery, topology, quantization, backend ops
(silu_mul, stable_softplus, rms_norm_gated, add3, add_rms_norm, add_scaled, depthwise_conv1d,
rms_norm_channel — CPU and GPU variants), Fp8Linear forward, transformer block forward,
native dtype backend, and SD utilities.

### Rules

1. **Any change to existing code MUST pass all tests.** Run `cargo test -p cake-core` before committing. No exceptions.
2. **Any change to existing code MUST NOT create a performance regression.** Run `cargo bench -p cake-core` before and after, compare results. If a benchmark regresses, fix the regression or justify it.
3. **New models require full test and benchmark coverage:**
   - Config parsing test (inline `#[cfg(test)]` in `config.rs` — deserialize sample JSON, verify `into_config()`)
   - Block forward test (in `tests/unit_tests/test_blocks.rs` — load via VarBuilder, assert output shape for prefill and generation)
   - Block forward benchmark (in `benches/bench_blocks.rs` — `args = [1, 8, 64]` for seq_len)
   - If the model introduces new attention/MLP/MoE variants, add dedicated tests and benchmarks for those components
   - Chat history encoder test if a new history format is added
4. **New components (protocol messages, discovery features, utilities) require:**
   - Unit tests covering all code paths (happy path + error cases)
   - Benchmarks for any function on a hot path or that processes data proportional to model/tensor size
5. **Clippy must pass:** `cargo clippy --all-targets -- -D warnings` with zero warnings.
