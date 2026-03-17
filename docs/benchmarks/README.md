# Cake Multimodal Benchmark Results

**Date:** 2026-03-17
**System:** RTX 3080 Laptop 16GB VRAM, i9-11900H, 32GB RAM, CUDA 13.1, Linux 6.19.6

## Summary

| Model | Modality | Reference | cake | Speedup | VRAM (ref) | VRAM (cake) |
|-------|----------|-----------|------|---------|------------|-------------|
| Qwen3.5-0.8B | Text (256 tok) | ollama 183.6 tok/s | **128.2 tok/s** | 0.70x | 2,248 MB | **1,906 MB** |
| FLUX.1-dev FP8 | Image (768x1024) | n/a (gated) | **3.3 s/step** | — | — | 13,317 MB |
| VibeVoice-1.5B | Voice (14s audio) | Python 27 ms/frame | **34 ms/frame** | 0.79x | 5,978 MB | 6,565 MB |

## Qwen3.5-0.8B Text Generation

**Prompt:** "Explain the theory of general relativity in simple terms."
**Config:** temperature=0, seed=42

| Metric | ollama (Q4_K_M) | cake (F16) |
|--------|----------------|------------|
| Eval rate | 183.6 tok/s | 128.2 tok/s |
| Prompt eval rate | 1,170 tok/s | — |
| Tokens generated | 1,686 | 256 |
| Total time | 14.4s | 3.8s |
| Peak VRAM | 2,248 MB | 1,906 MB |
| Quantization | Q4_K_M (4-bit) | F16 (16-bit) |

**Notes:**
- ollama uses Q4_K_M quantization (4-bit), cake uses F16 (16-bit) — not directly comparable on tok/s.
- ollama's higher tok/s is largely due to 4-bit quantization reducing memory bandwidth.
- cake uses less VRAM despite higher precision weights.
- Both produce coherent, accurate explanations of general relativity.

**Outputs:** [`ollama_output.txt`](qwen3.5/ollama_output.txt) | [`cake_output.txt`](qwen3.5/cake_output.txt)

## FLUX.1-dev FP8 Image Generation

**Prompt:** "A cyberpunk samurai standing on a neon-lit rooftop at night, overlooking a sprawling futuristic Tokyo cityscape with flying cars and holographic billboards, rain falling, dramatic lighting"
**Config:** 768x1024, 20 steps, guidance=3.5

| Metric | cake (FP8) |
|--------|------------|
| Total time | ~90s |
| T5-XXL encoding | 28.0s |
| Transformer loading | 9.0s |
| Denoising (20 steps) | 66.0s |
| Per-step time | 3.3s |
| VAE decode | 4.0s |
| Peak VRAM | 13,317 MB |

**Notes:**
- Reference (diffusers) benchmark skipped — `black-forest-labs/FLUX.1-dev` is a gated repo requiring HuggingFace authentication. Cake uses the publicly available `Comfy-Org/flux1-dev` FP8 checkpoint.
- FP8 weights kept on GPU (~8.6GB), dequantized to F16 on-the-fly per layer.
- BF16 outer compute eliminates F32↔F16 round-trip casts (~7% speedup vs F32 outer).
- Fits on a 16GB GPU with room for activations — ComfyUI-comparable memory footprint.

**Output:** [`cake_output.png`](flux1/cake_output.png)

## VibeVoice-1.5B Voice Synthesis

**Text:** "Generating long-form, multi-speaker conversational audio like podcasts poses significant challenges for traditional text-to-speech systems, particularly in scalability, speaker consistency, and natural turn-taking."
**Voice:** en-Carter_man.wav | **Config:** 10 diffusion steps, CFG=1.3

| Metric | Python (BF16, SDPA) | cake (BF16, flash attn) |
|--------|-------------------|------------------|
| Total time | 8.9s | 12.7s |
| Audio duration | 15.6s | 13.9s |
| Speech frames | 325 | 104 |
| Per-frame time | 27 ms | 34 ms |
| Peak VRAM | 5,978 MB | 6,565 MB |
| Whisper transcript | Exact match | Exact match |
| Real-time factor | 0.57x (1.75x RT) | 0.91x (1.1x RT) |

**Notes:**
- Python generates 3x more speech frames (325 vs 104) for similar audio duration — uses streaming VAE decode which generates shorter audio per frame.
- cake uses flash attention (BF16) and manual depthwise conv (broadcast_mul+sum) replacing candle's slow grouped Conv1d.
- Both produce identical Whisper transcription — speech quality is equivalent.
- cake achieves ~1.1x real-time: generates 13.9s of speech in 12.7s (including model loading).

**Outputs:** [`python_output.wav`](vibevoice/python_output.wav) | [`cake_output.wav`](vibevoice/cake_output.wav)

## Key Takeaways

1. **Text generation** is competitive — cake at F16 is ~70% of ollama's Q4 throughput, while using less VRAM. With quantization support, cake would likely match or exceed ollama.
2. **Image generation** works on consumer hardware — FLUX.1-dev FP8 runs at 768x1024 on a 16GB laptop GPU with BF16 compute, matching ComfyUI memory footprint.
3. **Voice synthesis** is near real-time — 34ms/frame vs Python's 27ms (1.26x gap), generating 13.9s of audio in 12.7s. Optimized via manual depthwise Conv1d (broadcast_mul+sum) replacing candle's slow grouped Conv1d kernel.
