# Supported Models

## Text Models

| Model | HuggingFace ID (example) | Feature Flag | Notes |
|-------|--------------------------|-------------|-------|
| LLaMA 3.x | `meta-llama/Llama-3.2-1B-Instruct` | `llama` (default) | Also covers SmolLM2 and DeepSeek-R1 distilled variants |
| SmolLM2 | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `llama` (default) | LLaMA architecture, 135M–1.7B |
| Qwen2 / Qwen2.5 | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | `qwen2` (default) | |
| Qwen3 (dense) | `Qwen/Qwen3-0.6B` | `qwen3` (default) | GQA + QK-norm, thinking mode via `/think` |
| Qwen3 MoE | `Qwen/Qwen3-30B-A3B` | `qwen3_moe` (default) | Sparse MoE FFN, 128 experts / top-8 per token |
| Qwen3.5 | `Qwen/Qwen3.5-0.8B` | `qwen3_5` (default) | Hybrid GDN linear + full attention |
| Qwen3.5 MoE | `Qwen/Qwen3.5-35B-A3B-GPTQ-Int4` | `qwen3_5_moe` (default) | Hybrid GDN+full-attn + sparse MoE FFN, 256 experts / top-8 per token; GPTQ-Int4 dequantized at load time |
| Phi-4-mini | `microsoft/Phi-4-mini-instruct` | `phi4` (default) | 3.8B, partial RoPE, 200K vocab |
| Phi-4 | `microsoft/phi-4` | `phi4` (default) | 14B, same family as Phi-4-mini |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | `mistral` (default) | Standard GQA, optional sliding window |
| Gemma 3 | `google/gemma-3-1b-it` | `gemma3` (default) | Interleaved local/global attention, GELU-tanh MLP |
| Falcon3 | `tiiuae/Falcon3-1B-Instruct` | `falcon3` (default) | Standard GQA, Apache 2.0 |
| OLMo 2 | `allenai/OLMo-2-1124-7B` | `olmo2` (default) | Post-norm, QK-norm, fully open weights+data |
| EXAONE 4.0 | `LGAI-EXAONE/EXAONE-4.0-1.2B-Instruct` | `exaone4` (default) | 3:1 local/global hybrid, QK-norm |
| DeepSeek-R1 (distilled) | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` | `llama` (default) | LLaMA or Qwen2.5 base — no new code needed |

## Image Models

| Model | Status |
|-------|--------|
| Stable Diffusion 1.5 | Supported |
| Stable Diffusion 2.1 | Supported |
| Stable Diffusion XL | Supported |
| Stable Diffusion XL Turbo | Supported |

See [Stable Diffusion](stable_diffusion.md) for image generation usage.

## Architecture Detection

Text model architecture is auto-detected from `config.json` in the model directory. You can also set it explicitly:

```sh
cake master --model /path/to/model --text-model-arch auto|llama|qwen2|qwen3|qwen3-moe|qwen3-5|qwen3-5moe|phi4|mistral|gemma3|falcon3|ol-mo2|exaone4
```

## Model Notes

### LLaMA 3.x / SmolLM2 / DeepSeek-R1 Distilled

SmolLM2 uses the LLaMA architecture (`model_type: "llama"` in config.json) and loads automatically via the `llama` feature. DeepSeek-R1 distilled variants built on LLaMA 3 or Qwen2.5 also work with no extra code.

### Qwen3 (dense)

Qwen3 dense models (0.6B–32B) extend the Qwen2.5 design with QK-norm on query and key projections. They support a dual-mode thinking toggle: prefix your prompt with `/think` or `/no_think` to enable or disable the reasoning chain.

### Qwen3.5

Qwen3.5 is a hybrid linear/full attention model using Gated DeltaNet (GDN). It uses recurrent (linear) attention for 18 of 24 layers and standard softmax attention for the remaining 6, in a repeating 3:1 pattern. It is a "thinking" model — responses begin with a `<think>...</think>` reasoning block before the answer.

### Phi-4-mini / Phi-4

Both models share the `phi4` feature and are loaded via the same code path. They use pre-fused QKV and gate+up projections, a 200K-token vocabulary, and partial RoPE (`partial_rotary_factor: 0.25`).

### Gemma 3

Gemma 3 uses an interleaved local/global attention pattern: every 6th layer is a global attention layer with full context and RoPE, while the rest use sliding-window (local) attention without RoPE. The MLP uses GELU-tanh activation (not SiLU), and embeddings are scaled by `sqrt(hidden_size)`. All norms use `GemmaRMSNorm` (weights initialized to zero, forward = `(1+weight) * norm(x)`).

The Gemma 3 IT chat template has no separate `system` role — the system prompt is prepended to the first user turn.

**Note:** The 1B model does not benefit from a system prompt. Use `--system-prompt ""` for best results with Gemma 3 1B.

### Mistral

Mistral models use standard GQA and optionally sliding-window attention (4096-token window on Mistral Small). They load via the `mistral` feature.

### Falcon3

Falcon3 models use standard GQA + SwiGLU, similar to LLaMA 3. They use ChatML-style tokenization. Released under Apache 2.0.

### OLMo 2

OLMo 2 uses post-norm (RMSNorm applied after the residual add, not before) and QK-norm. It is fully open: weights, training data, and code are all public. The 7B model requires a cluster to run (does not fit on a single 16 GB GPU).

### EXAONE 4.0

EXAONE 4.0 uses a 3:1 local/global hybrid attention pattern where global layers use full context without RoPE, similar to Gemma 3. It includes QK-norm and is strong on multilingual and reasoning tasks.

### Qwen3 MoE

Qwen3 MoE (30B-A3B and 235B-A22B) uses the same attention block as dense Qwen3 (GQA + QK-norm) but replaces the dense FFN with a Sparse Mixture-of-Experts layer. Each layer has 128 experts; the router selects the top-8 per token using softmax → top-K → renormalize. This makes these models ideal cluster targets: the 30B model activates only 3B parameters per token, while the 235B model activates 22B — both require multiple nodes to hold all expert weights.

The router matches HuggingFace `Qwen3MoeTopKRouter` exactly: softmax over all experts first, then top-K, then renormalize the selected weights to sum to 1.
