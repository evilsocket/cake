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
```

## Run Commands (Qwen3.5-0.8B cluster)

```bash
# Workers first (on each machine):
# bahamut:
LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ./target/release/cake worker \
  --model Qwen/Qwen3.5-0.8B --name bahamut \
  --topology topology-0.8B.yml --address 0.0.0.0:10128

# stevie:
./target/release/cake worker \
  --model Qwen/Qwen3.5-0.8B --name stevie \
  --topology topology-0.8B.yml --address 0.0.0.0:10128

# Master (blade, local):
./target/release/cake master \
  --model Qwen/Qwen3.5-0.8B \
  --topology topology-0.8B.yml \
  --prompt "Explain quantum computing in simple terms"
```

## Model: Qwen/Qwen3.5-0.8B

- **Architecture**: Qwen3_5ForConditionalGeneration
- **Layers**: 24 (48 GatedDeltaNet linear attn + 16 full attn... wait, 0.8B has 24 total)
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
ssh bahamut.local "cd ~/Lab/cake && LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64 ./target/release/cake worker --model Qwen/Qwen3.5-0.8B --name bahamut --topology topology-0.8B.yml --address 0.0.0.0:10128"
ssh stevie.local "cd ~/Lab/cake && ./target/release/cake worker --model Qwen/Qwen3.5-0.8B --name stevie --topology topology-0.8B.yml --address 0.0.0.0:10128"

# Run master (blade, local)
./target/release/cake master --model Qwen/Qwen3.5-0.8B --topology topology-0.8B.yml --prompt "Explain quantum computing in simple terms"
```
