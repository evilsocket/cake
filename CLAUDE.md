# Cake Development Guide

## Build Commands

```bash
# Linux (CUDA)
cargo build --release --features cuda

# Linux (CUDA, specific version)
CUDA_HOME=/usr/local/cuda-12.4 LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 cargo build --release --features cuda

# macOS (Metal)
cargo build --release --features metal

# CPU only
cargo build --release
```

## Run Commands (example: Qwen3.5-0.8B cluster)

```bash
# Worker (on each worker machine):
./target/release/cake worker \
  --model Qwen/Qwen3.5-0.8B --name worker1 \
  --topology topology.yml --address 0.0.0.0:10128

# Master:
./target/release/cake master \
  --model Qwen/Qwen3.5-0.8B \
  --topology topology.yml \
  --prompt "Explain quantum computing in simple terms"
```

## Testing

```bash
# Run all tests (no model files required)
cargo test --features cuda

# Integration tests require model files — set env vars:
#   CAKE_TEST_MODEL=./path/to/Llama-3.2-1B-Instruct/
#   CAKE_TEST_QWEN2_MODEL=./path/to/Qwen2-0.5B/
# Tests skip gracefully when model paths are not available.

# Run protocol benchmarks
cargo test --test protocol -- --ignored --nocapture
```

## Model: Qwen/Qwen3.5-0.8B

- **Architecture**: Qwen3_5ForConditionalGeneration
- **Layers**: 24
- **Hidden size**: 1024
- **Layer prefix**: `model.language_model.layers.{N}`
- **Location**: HuggingFace cache (`~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/`)
- **Size**: ~1.6 GB in F16

## Self Improving Loop

Iterative optimization process for maximizing inference speed:

### Process

1. **Instrument**: Add timing/profiling logs to hot code paths (forward pass, attention, MLP, network, serialization)
2. **Commit & push**: Commit changes, push to origin
3. **Deploy**: Pull on all machines, rebuild with appropriate features (cuda/metal)
4. **Run experiment**: Start workers, then master with a test prompt
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
