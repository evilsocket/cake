# TTS Models Autoresearch

Autonomous optimization of text-to-speech model inference components.

## Objective

**Minimize forward pass latency** for TTS model building blocks: diffusion prediction head,
DPM-Solver++ scheduler, VAE decoder, acoustic/semantic connectors, Zipformer layers,
Vocos vocoder, and mel spectrogram computation. All benchmarks use synthetic tensors on CPU
with reduced dimensions for fast iteration. Lower is better.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar23`). The branch `autoresearch/tts/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/tts/<tag>` from the current branch.
3. **Read the in-scope files** for full context:

   ### VibeVoice (primary target — diffusion TTS)
   - `cake-core/src/models/vibevoice/prediction_head.rs` — DiT-style diffusion head: 4 layers
     with AdaLN modulation, SwiGLU FFN, timestep embedding. **Hot inner loop** — called
     20× per speech frame (one per diffusion step).
   - `cake-core/src/models/vibevoice/ddpm.rs` — DPM-Solver++ multistep scheduler.
     v-prediction → x0 conversion, cosine beta schedule, 2nd-order midpoint solver.
     Called 20× per speech frame.
   - `cake-core/src/models/vibevoice/vae_decoder.rs` — σ-VAE decoder: 7-stage Conv1d with
     depthwise conv mixers, RMSNorm, FFN, ConvTranspose1d upsampling. StreamingConvCache
     for frame-by-frame streaming.
   - `cake-core/src/models/vibevoice/vae_encoder.rs` — Tokenizer encoder (acoustic/semantic).
   - `cake-core/src/models/vibevoice/acoustic_connector.rs` — fc1 → RmsNorm → fc2
     (VAE latent to LM hidden dimension).
   - `cake-core/src/models/vibevoice/eos_classifier.rs` — End-of-speech detection.
   - `cake-core/src/models/vibevoice/vibevoice.rs` — VibeVoice-0.5B main generation loop.
   - `cake-core/src/models/vibevoice/vibevoice_1_5b.rs` — VibeVoice-1.5B variant.
   - `cake-core/src/models/vibevoice/voice_prompt.rs` — Voice prompt KV cache loading.
   - `cake-core/src/models/vibevoice/config.rs` — Configuration structs.

   ### LuxTTS (flow-matching TTS)
   - `cake-core/src/models/luxtts/zipformer_layer.rs` — Zipformer encoder layer:
     3× FeedforwardModule, RelPositionMultiheadAttention, NonlinAttention,
     2× ConvolutionModule, 2× BypassModule. **Hot path** — 16 layers in FM decoder.
   - `cake-core/src/models/luxtts/rel_pos_attention.rs` — Relative position multihead
     attention with compact positional encoding.
   - `cake-core/src/models/luxtts/vocos.rs` — ISTFT vocoder: Conv1d → 8× ConvNeXt → ISTFT head.
   - `cake-core/src/models/luxtts/mel.rs` — STFT-based mel spectrogram extraction.
   - `cake-core/src/models/luxtts/euler_solver.rs` — Euler flow matching solver.
   - `cake-core/src/models/luxtts/model.rs` — LuxTTS main generation pipeline.
   - `cake-core/src/models/luxtts/activations.rs` — swoosh_r/swoosh_l activations.
   - `cake-core/src/models/luxtts/bias_norm.rs` — BiasNorm layer.
   - `cake-core/src/models/luxtts/convolution_module.rs` — Depthwise conv + GLU.
   - `cake-core/src/models/luxtts/feedforward.rs` — Simple FFN.
   - `cake-core/src/models/luxtts/nonlin_attention.rs` — Nonlinear attention variant.

   ### Shared infrastructure
   - `cake-core/src/backends/mod.rs` — ComputeBackend trait (read-only).
   - `cake-core/src/backends/cpu/mod.rs` — CPU backend (used by benchmarks).
   - `cake-core/src/utils/wav.rs` — WAV encode/decode/resample.

   ### Benchmarks
   - `cake-core/benches/bench_vibevoice.rs` — VibeVoice component benchmarks.
   - `cake-core/benches/bench_luxtts.rs` — LuxTTS component benchmarks.
   - `cake-core/benches/bench_wav.rs` — WAV I/O benchmarks.
   - `cake-core/benches/bench_helpers.rs` — Shared helpers.

4. **Run prepare.sh**: `bash autoresearch/models/tts/prepare.sh`
5. **Confirm baseline** and start experimenting.

## Features

VibeVoice benchmarks require no feature flags (always compiled). LuxTTS benchmarks require
`--features luxtts`. The benchmark harness enables both automatically.

## Files You May Modify

### VibeVoice
- `cake-core/src/models/vibevoice/prediction_head.rs` — Diffusion head forward pass.
  Timestep embedding, AdaLN modulation, SwiGLU FFN, layer composition.
- `cake-core/src/models/vibevoice/ddpm.rs` — DPM-Solver++ step function. Coefficient
  precomputation, v-prediction conversion, multistep buffers.
- `cake-core/src/models/vibevoice/vae_decoder.rs` — VAE decoder forward pass. Conv
  block composition, streaming cache, upsampling stages.
- `cake-core/src/models/vibevoice/vae_encoder.rs` — Encoder forward pass.
- `cake-core/src/models/vibevoice/acoustic_connector.rs` — Connector forward pass.
- `cake-core/src/models/vibevoice/eos_classifier.rs` — EOS classifier forward.
- `cake-core/src/models/vibevoice/vibevoice.rs` — Generation loop logic.
- `cake-core/src/models/vibevoice/vibevoice_1_5b.rs` — 1.5B generation loop.

### LuxTTS
- `cake-core/src/models/luxtts/zipformer_layer.rs` — Zipformer layer forward pass.
- `cake-core/src/models/luxtts/rel_pos_attention.rs` — Attention computation.
- `cake-core/src/models/luxtts/vocos.rs` — Vocoder forward pass.
- `cake-core/src/models/luxtts/mel.rs` — Mel spectrogram computation.
- `cake-core/src/models/luxtts/euler_solver.rs` — Euler step.
- `cake-core/src/models/luxtts/model.rs` — Pipeline orchestration.
- `cake-core/src/models/luxtts/activations.rs` — Activation functions.
- `cake-core/src/models/luxtts/convolution_module.rs` — Conv module.
- `cake-core/src/models/luxtts/nonlin_attention.rs` — Nonlinear attention.

### Shared
- `cake-core/src/utils/wav.rs` — WAV encoding/decoding.
- `cake-core/src/backends/cpu/mod.rs` — CPU backend.

## Files You Must NOT Modify

- `autoresearch/models/tts/benchmark.sh` — the measurement harness is sacred
- `autoresearch/models/tts/prepare.sh`
- `cake-core/benches/bench_vibevoice.rs`
- `cake-core/benches/bench_luxtts.rs`
- `cake-core/benches/bench_wav.rs`
- `cake-core/benches/bench_helpers.rs`
- `cake-core/src/backends/mod.rs` — the ComputeBackend trait
- Any test files (`cake-core/tests/`)
- Config files (`config.rs`, `config_1_5b.rs`) — model loading depends on field names

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis. Write it down.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --features vibevoice,luxtts --lib --tests -- -D warnings && cargo test -p cake-core --features vibevoice,luxtts --lib && cargo test -p cake-core --features vibevoice,luxtts --test unit && cargo test -p cake-core --features vibevoice,luxtts --test protocol`
5. **Benchmark**: `bash autoresearch/models/tts/benchmark.sh`
6. **Parse** the BENCH_RESULT line: `score=X tests=PASS/FAIL clippy=PASS/FAIL status=OK/...`
7. **If build or test failed**: attempt a quick fix. If unfixable after 2 tries, revert and log.
8. **Record** in `autoresearch/models/tts/experiments.tsv`
9. **Decide**:
   - **KEEP** if `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
   - **DISCARD** if `score > previous_best * 1.005` OR any gate fails
10. **Repeat**

## TTS-Specific Architecture Constraints

### Diffusion Loop (VibeVoice)
The prediction head is called **20× per speech frame** (one per DPM-Solver++ step).
Each call involves: timestep embedding (sinusoidal → MLP), condition projection,
4× DiT layers (AdaLN modulation + SwiGLU FFN), final layer. Any per-call overhead
is multiplied 20×. **Precomputation across steps is the highest-leverage optimization.**

### Streaming Decode (VibeVoice)
The VAE decoder operates frame-by-frame with `StreamingConvCache`. Each conv layer
maintains context from the previous frame. Cache slot allocation (`take_slot`) and
tensor management are per-frame overhead. Minimize allocations in the streaming path.

### Flow Matching (LuxTTS)
The Euler solver runs 4 steps through 16 Zipformer layers (64 layer forwards total).
Each Zipformer layer has 3 FFN + 2 attention + 2 conv = 7 sublayers. Total: 448
sublayer forwards per utterance. Micro-optimizations compound significantly.

### Mel Spectrogram
Computed once per reference audio (LuxTTS speaker conditioning). Uses STFT with
Hann window, mel filterbanks, log scaling. The FFT dominates — consider whether
a real-FFT implementation or windowed overlap-add can reduce work.

### Backend Abstraction
Models call `backend.silu_mul()`, `backend.add_rms_norm()` etc. for fused operations.
On CPU (benchmark target), these use default implementations. Optimizations should be
algorithmic (reducing operations, precomputation, better math) not hardware-specific.

## Promising Areas to Explore

### VibeVoice — High Impact

1. **Timestep embedding caching** — The sinusoidal embedding (`arange → exp → mul → cos/sin → cat`)
   is recomputed every diffusion step. The set of timesteps is known at scheduler creation.
   Precompute and cache the embeddings at init time.

2. **AdaLN modulation fusion** — Each DiT layer computes `adaLN_modulation(silu(t_emb))` which
   involves a SiLU activation + linear projection + chunk into 3 parts. The SiLU can be
   fused with the projection, and the chunk can be done via narrow (zero-copy).

3. **DPM-Solver++ coefficient precomputation** — The scheduler recomputes `alpha_t`, `sigma_t`,
   `lambda_t` lookups each step. These are indexed by fixed timesteps known at init.
   Precompute per-step coefficients at construction time.

4. **Batch diffusion CFG** — Classifier-Free Guidance runs the prediction head 2× (conditional +
   unconditional). If the head supports batch>1, both can be computed in a single forward pass
   with a single set of kernel launches. The benchmark already tests batch=4.

5. **VAE decoder conv fusion** — The decoder has repeated patterns of depthwise_conv + norm + FFN.
   Reducing intermediate tensor allocations in the conv chain would help streaming latency.

6. **StreamingConvCache allocation** — `take_slot` + `set` pattern creates/replaces Option<Tensor>.
   Pre-allocating fixed-size buffers and writing in-place would avoid allocation churn.

### LuxTTS — High Impact

7. **Zipformer FFN reduction** — Each Zipformer layer has 3 FFN modules with different widths.
   Can any of the matmul pairs be combined (e.g., concatenated weights for parallel FFNs)?

8. **Relative position attention precomputation** — The compact positional encoding
   (`get_pos_emb`) may recompute position tables unnecessarily across layers.

9. **Vocos ConvNeXt optimization** — 8 identical ConvNeXt blocks with depthwise_conv →
   LayerNorm → SwiGLU FFN → gamma scale → residual. The depthwise conv is the bottleneck.

10. **Mel spectrogram: real FFT** — Current implementation uses full complex FFT. A real-FFT
    would halve the computation since mel inputs are real-valued.

### Cross-Cutting

11. **Reduce tensor allocations** — Many forward passes create intermediate tensors that could
    be computed in-place or with pre-allocated buffers.

12. **Activation function optimization** — `swoosh_r`/`swoosh_l` use `log(1 + exp(x))` which
    can use the same fast-path optimization as stable_softplus.

13. **WAV encoding optimization** — `encode_wav_bytes` converts f32 → i16 sample by sample.
    SIMD-friendly batch conversion could help for large audio outputs.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails
- If **3 consecutive discards** with similar approaches: move to a different area
- After every **keep**: update your mental baseline to the new score
- Always **revert before starting a new experiment** (clean slate from last keep)

## Safety

- Do not change the `ComputeBackend` trait signature
- Do not change config struct field names (model loading depends on them)
- Do not add new dependencies to `Cargo.toml`
- Do not modify benchmark, prepare, or test files
- Ensure streaming decode correctness — VAE cache state must be preserved exactly

## NEVER STOP

Once the loop begins, do NOT pause to ask the human for permission. The human may be asleep.
You run autonomously until manually interrupted. If you run out of ideas, re-read the source
for new angles. Each experiment takes ~30-60 seconds, so you can run ~60-120 per hour.
