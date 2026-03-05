<div align="center">

# `cake`

[![Documentation](https://img.shields.io/badge/docs-blue)](https://github.com/evilsocket/cake/blob/main/docs/index.md)
[![License](https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square)](https://github.com/evilsocket/cake/blob/master/LICENSE.md)

  <small>Join the project community on our server!</small>
  <br/><br/>
  <a href="https://discord.gg/https://discord.gg/btZpkp45gQ" target="_blank" title="Join our community!">
    <img src="https://dcbadge.limes.pink/api/server/https://discord.gg/btZpkp45gQ"/>
  </a>

</div>

Cake is a Rust framework for distributed inference of large language models and image generation models based on [Candle](https://github.com/huggingface/candle). The goal is to run big (70B+) models by repurposing consumer hardware into a heterogeneous cluster of iOS, Android, macOS, Linux and Windows devices, effectively leveraging [planned obsolescence](https://en.wikipedia.org/wiki/Planned_obsolescence) as a tool to make AI more accessible and democratic.

<p align="center">
  <strong>
  This is experimental code that's being actively developed and changed very quickly.
  </strong>
</p>

## Key Features

- **Distributed Inference** — Shard transformer blocks across multiple devices to run models that don't fit on a single GPU. [Learn more](https://github.com/evilsocket/cake/blob/main/docs/clustering.md).
- **Multi Model** — Support for [LLaMA 3.x, SmolLM2, Qwen2/2.5/3/3.5, Phi-4, Mistral, Gemma 3, Falcon3, OLMo 2, EXAONE 4.0](https://github.com/evilsocket/cake/blob/main/docs/models.md) and [Stable Diffusion](https://github.com/evilsocket/cake/blob/main/docs/stable_diffusion.md).
- **Multi Platform** — CUDA, Metal, and CPU backends across [Linux, macOS, Windows, iOS, and Android](https://github.com/evilsocket/cake/blob/main/docs/install.md).
- **Zero-Config Clustering** — mDNS discovery, automatic layer assignment, and model data push with a single `--cluster-key` flag. [Learn more](https://github.com/evilsocket/cake/blob/main/docs/clustering.md#zero-config-cluster-mdns-discovery).
- **OpenAI-Compatible API** — REST API with streaming support, plus a [built-in web UI and TUI chat client](https://github.com/evilsocket/cake/blob/main/docs/usage.md#web-ui).
- **Docker** — [Container builds](https://github.com/evilsocket/cake/blob/main/docs/docker.md) for Linux/NVIDIA with docker-compose cluster support.

### Platform Support

| OS | Architectures | Acceleration | Status |
|----|---------------|--------------|--------|
| GNU/Linux | arm, arm64, x86_64 | - | ✅ |
| GNU/Linux | arm, arm64, x86_64 | CUDA | ✅ |
| GNU/Linux | arm, arm64, x86_64 | BLAS | ✅ |
| Windows | x86_64 | BLAS | [⚠️](https://github.com/evilsocket/cake/issues/7) |
| Windows | x86_64 | CUDA | ✅ |
| macOS | x86_64 | - | ✅ |
| macOS | aarch64 | - | ✅ |
| macOS | aarch64 | Metal | ✅ |
| Android | arm, arm64, x86_64 | - | ✅ |
| Android | arm, arm64, x86_64 | CUDA | [⚠️](https://docs.nvidia.com/gameworks/content/technologies/mobile/cuda_android_main.htm) |
| iOS / iPadOS | aarch64 | - | ✅ |
| iOS / iPadOS | aarch64 | Metal | ✅ (A13+ / M-series) |

### Models

| Model | Type | Feature Flag | Status |
|-------|------|-------------|--------|
| LLaMA 3.x | Text | `llama` (default) | ✅ |
| SmolLM2 | Text | `llama` (default) | ✅ |
| Qwen2 / Qwen2.5 | Text | `qwen2` (default) | ✅ |
| Qwen3 (dense) | Text | `qwen3` (default) | ✅ |
| Qwen3 MoE | Text | `qwen3_moe` (default) | ✅ |
| Qwen3.5 | Text | `qwen3_5` (default) | ✅ |
| Qwen3.5 MoE (GPTQ-Int4) | Text | `qwen3_5_moe` (default) | ✅ |
| Phi-4-mini / Phi-4 | Text | `phi4` (default) | ✅ |
| Mistral | Text | `mistral` (default) | ✅ |
| Gemma 3 | Text | `gemma3` (default) | ✅ |
| Falcon3 | Text | `falcon3` (default) | ✅ |
| OLMo 2 | Text | `olmo2` (default) | ✅ |
| EXAONE 4.0 | Text | `exaone4` (default) | ✅ |
| DeepSeek-R1 (distilled) | Text | `llama` / `qwen2` (default) | ✅ |
| Stable Diffusion (1.5, 2.1, XL, XL Turbo) | Image | - | ✅ |

## Quick Start

```sh
cargo build --release --features cuda  # or: --features metal
cake download Qwen/Qwen2.5-Coder-1.5B-Instruct
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt "Hello!"
```

To start the API server and web UI:

```sh
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --api 0.0.0.0:8080
```

For the full usage guide and API reference, [check the project documentation](https://github.com/evilsocket/cake/blob/main/docs/index.md).

## Contributors

<a href="https://github.com/evilsocket/cake/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=evilsocket/cake" alt="Cake project contributors" />
</a>

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=evilsocket/cake&type=Timeline)](https://www.github.com/evilsocket/cake&Timeline)

## License

Released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.
