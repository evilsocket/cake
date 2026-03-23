# Image Generation Autoresearch

Autonomous optimization of FLUX and Stable Diffusion inference components.

## Objective

**Minimize image generation component latency** — FLUX transformer blocks, VAE encode/decode,
text encoders (T5, CLIP), timestep embedding, and FP8 linear layers. Lower is better.

## Setup

1. **Agree on a run tag**: e.g. `mar23`. Branch: `autoresearch/image-generation/<tag>`.
2. **Create the branch**: `git checkout -b autoresearch/image-generation/<tag>`
3. **Read the in-scope files**:
   - `cake-core/src/models/flux/` — FLUX transformer, T5/CLIP encoders, VAE, shardable variants.
   - `cake-core/src/models/sd/` — Stable Diffusion: CLIP, VAE, U-Net, scheduler.
   - `cake-core/benches/bench_flux.rs` — FLUX component benchmarks.
   - `cake-core/benches/bench_helpers.rs` — Shared helpers.
4. **Run prepare.sh**: `bash autoresearch/models/image-generation/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/models/flux/*.rs` — All FLUX model components (transformer blocks, encoders, VAE, FP8 linear).
- `cake-core/src/models/sd/*.rs` — All Stable Diffusion components (CLIP, VAE, U-Net, scheduler).

## Files You Must NOT Modify

- `autoresearch/models/image-generation/benchmark.sh`
- `autoresearch/models/image-generation/prepare.sh`
- `cake-core/benches/bench_flux.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files (`cake-core/tests/`)

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single optimization hypothesis.
2. **Implement** the change.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol`
5. **Benchmark**: `bash autoresearch/models/image-generation/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Architecture Constraints

### FLUX Feature Gate
FLUX code is behind `#[cfg(feature = "flux")]`. The benchmarks enable this feature.
Do not remove or change feature gates.

### FP8 Precision
FLUX uses FP8 (E4M3) quantized weights with per-block scaling. The dequantization path
is performance-critical. Do not change the quantization format — only optimize the
dequant computation.

### Shardable Trait
FLUX models implement `Shardable` for distributed inference. Do not break the sharding
interface — layer splits must remain valid.

### AdaLN Modulation
FLUX transformer blocks use Adaptive Layer Normalization (AdaLN) for conditioning.
The `adaln_modulate` backend operation fuses norm + scale + shift. Optimizations should
work through the backend abstraction.

## Promising Areas to Explore

1. **FP8 linear optimization** — the Fp8Linear layer does `dequant(weight) @ input`.
   Can the dequant be fused with the matmul more tightly?

2. **Timestep embedding** — sinusoidal embedding with MLP. The MLP has SiLU activation
   that could benefit from fusion.

3. **VAE ResNet blocks** — the VAE decoder uses ResNet blocks with GroupNorm. GroupNorm
   may have optimization opportunities (vectorization, reduced passes).

4. **Position embedding** — FLUX uses 2D rotary position embedding. The precomputation
   of cos/sin tables may be cacheable.

5. **Text encoder optimization** — T5 and CLIP encoders run once per generation.
   Attention in these encoders may have different optimal patterns than the main
   transformer (longer sequences, no KV cache).

6. **Memory layout for image tensors** — image tensors are (batch, channels, height, width).
   Channel-first vs channel-last layout may affect vectorization.

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
