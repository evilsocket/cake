<div align="center">

# `cake`

[![Documentation](https://img.shields.io/badge/docs-blue)](https://github.com/evilsocket/cake/blob/main/docs/index.md)
[![License](https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square)](https://github.com/evilsocket/cake/blob/master/LICENSE.md)

</div>

Cake is a Rust framework for **multimodal distributed inference** based on [Candle](https://github.com/huggingface/candle). It shards models across a heterogeneous cluster of consumer devices — iOS, Android, macOS, Linux, Windows — to run workloads that wouldn't fit on a single GPU, effectively leveraging [planned obsolescence](https://en.wikipedia.org/wiki/Planned_obsolescence) to make AI more accessible and democratic.

<p align="center">
  <strong>
  This is experimental code that's being actively developed and changed very quickly.
  </strong>
</p>

## Key Features

- **Multi Modal** — [Text generation](docs/models.md), [image generation](docs/image_generation.md) (Stable Diffusion, FLUX), and [voice synthesis](docs/voice_generation.md) (VibeVoice TTS with voice cloning).
- **Multi Model** — [15 text model families](docs/models.md), 6 image model variants, and 2 TTS models. Architecture auto-detected from HuggingFace checkpoints.
- **Multi Platform** — CUDA, Metal, and CPU backends across [Linux, macOS, Windows, iOS, and Android](docs/install.md).
- **Multi Node** — Shard transformer blocks across devices with [zero-config mDNS clustering](docs/clustering.md) or manual topology. Also runs entirely on a single machine.
- **OpenAI-Compatible API** — REST API with streaming, plus a [built-in web UI and TUI chat client](docs/usage.md#web-ui).
- **Docker** — [Container builds](docs/docker.md) for Linux/NVIDIA with docker-compose cluster support.

## Quick Start

```sh
cargo build --release --features cuda  # or: --features metal

# Text generation
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --prompt "Hello!"

# API server + web UI
cake master --model Qwen/Qwen2.5-Coder-1.5B-Instruct --api 0.0.0.0:8080

# Image generation (FLUX.1-dev FP8, 1024x768)
cake master --model-type image-model --image-model-arch flux1 \
  --sd-image-prompt "a cyberpunk cityscape at night" --flux-height 768 --flux-width 1024

# Voice synthesis (VibeVoice-1.5B with voice cloning)
cake master --model-type audio-model --model microsoft/VibeVoice-1.5B \
  --voice-prompt voice.wav --prompt "Hello world, this is a test."
```

Models are downloaded automatically from HuggingFace. For the full usage guide and API reference, [check the project documentation](docs/index.md).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=evilsocket/cake&type=Timeline)](https://star-history.com/#evilsocket/cake&Timeline)

## License

Released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.
