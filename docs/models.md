# Supported Models

## Text Models

| Model | Feature Flag | Status |
|-------|-------------|--------|
| LLaMA 3.x | `llama` (default) | Supported |
| Qwen2 / Qwen2.5 | `qwen2` (default) | Supported |
| Qwen3.5 | `qwen3_5` (default) | Supported |

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
cake master --model /path/to/model --text-model-arch auto|llama|qwen2|qwen3_5
```

## Qwen3.5

Qwen3.5 is a hybrid linear/full attention model using Gated DeltaNet. It uses recurrent (linear) attention for 18 of 24 layers and standard softmax attention for the remaining 6, in a repeating 3:1 pattern.

It's a "thinking" model — responses start with a `<think>...</think>` reasoning block before the actual answer.
