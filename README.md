<div align="center">

# `cake`

[![Documentation](https://img.shields.io/badge/docs-blue)](https://github.com/evilsocket/cake/blob/main/docs/index.md)
[![Release](https://img.shields.io/github/release/evilsocket/cake.svg?style=flat-square)](https://github.com/evilsocket/cake/releases/latest)
[![Rust Report](https://rust-reportcard.xuri.me/badge/github.com/evilsocket/cake)](https://rust-reportcard.xuri.me/report/github.com/evilsocket/cake)
[![CI](https://img.shields.io/github/actions/workflow/status/evilsocket/cake/ci.yml)](https://github.com/evilsocket/cake/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-GPL3-brightgreen.svg?style=flat-square)](https://github.com/evilsocket/cake/blob/master/LICENSE.md)

</div>

Cake is a **multimodal AI inference server** written in Rust that can run models as a single node, or shard them across a heterogeneous cluster of devices — iOS, Android, macOS, Linux, Windows — to run workloads that wouldn't fit on a single GPU, effectively leveraging [planned obsolescence](https://en.wikipedia.org/wiki/Planned_obsolescence) to make AI more accessible and democratic.

<p align="center">
  <strong>
  This is experimental code that's being actively developed and changed very quickly.
  </strong>
</p>

## Key Features

- **Multi Modal** — [Text generation](docs/models.md), [image generation](docs/image_generation.md) (Stable Diffusion, FLUX), and [voice synthesis](docs/voice_generation.md) (VibeVoice TTS with voice cloning).
- **Multi Model** — [15 text model families](docs/models.md), 6 image model variants, and 2 TTS models. Architecture auto-detected from HuggingFace checkpoints.
- **Multi Platform** — CUDA, Metal, Vulkan, and CPU backends across [Linux, macOS, Windows, iOS, and Android](docs/install.md).
- **Multi Node** — Shard transformer blocks across devices with [zero-config mDNS clustering](docs/clustering.md) or manual topology. Also runs entirely on a single machine.
- **OpenAI-Compatible API** — REST API with streaming, plus a [built-in web UI and TUI chat client](docs/usage.md#web-ui).
- **Docker** — [Container builds](docs/docker.md) for Linux/NVIDIA with docker-compose cluster support.

## Quick Start

### Build

```sh
cargo build --release --features cuda   # Linux (NVIDIA)
cargo build --release --features metal  # macOS (Apple Silicon)
cargo build --release --features vulkan # Linux (AMD/Intel/Steam Deck)
cargo build --release                   # CPU only
```

### Models

Download models from HuggingFace with `cake pull`. Models are stored in the standard HuggingFace cache directory (`~/.cache/huggingface/hub/`) and are shared with any other tools that use the same cache (transformers, huggingface-cli, etc.).

```sh
cake pull evilsocket/Qwen3-0.6B             # text model (600M params)
cake pull evilsocket/flux1-dev               # image model (FLUX.1-dev FP8)
cake pull evilsocket/VibeVoice-1.5B          # voice synthesis model

cake list                                    # show all locally available models
```

Models are also downloaded automatically on first use if not already cached.

### Single Node

Run any model locally on a single machine — architecture is auto-detected from the model's `config.json`:

```sh
# Text generation
cake run evilsocket/Qwen3-0.6B "Explain quantum computing in simple terms"

# Start an API server + web UI
cake serve evilsocket/Qwen3-0.6B

# Image generation (FLUX.1-dev FP8)
cake run evilsocket/flux1-dev --model-type image-model --image-model-arch flux1 \
  --sd-image-prompt "a cyberpunk cityscape at night"

# Voice synthesis with voice cloning
cake run evilsocket/VibeVoice-1.5B --model-type audio-model \
  --voice-prompt voice.wav "Hello world"
```

### Distributed

Shard a model across multiple machines using `--cluster-key`. Workers don't need the model data — the master automatically streams the required tensor weights over the network (compressed with zstd, verified with CRC32 checksums). Workers cache received data locally for subsequent runs.

```sh
# Start workers on any machines (no model needed)
cake run --cluster-key mysecret --name gpu-server-1    # machine A
cake run --cluster-key mysecret --name macbook          # machine B

# Run inference from the master (has the model)
cake run evilsocket/Qwen3-0.6B "Hello" --cluster-key mysecret

# Or start an API server as the master
cake serve evilsocket/Qwen3-0.6B --cluster-key mysecret
```

The master discovers workers via mDNS, assigns layers proportionally to each device's VRAM/compute, and pushes only the required weight shards. See the [clustering documentation](docs/clustering.md) for manual topology files and advanced configuration.

For the full usage guide and API reference, [check the project documentation](docs/index.md).

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=evilsocket/cake&type=Timeline)](https://star-history.com/#evilsocket/cake&Timeline)

## License

Released under the GPL 3 license. To see the licenses of the project dependencies, install cargo license with `cargo install cargo-license` and then run `cargo license`.
