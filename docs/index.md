# Cake Documentation

Cake is a Rust framework for multimodal distributed inference. It shards models across consumer devices — iOS, Android, macOS, Linux, Windows — to run workloads that wouldn't fit on a single GPU.

Built on [Candle](https://github.com/huggingface/candle) with support for CUDA, Metal, and CPU backends.

## Table of Contents

- [Installation](install.md) — Building from source, platform support, acceleration backends
- [Models](models.md) — Supported text, image, and voice model architectures
- [Usage](usage.md) — Downloading models, running inference, Web UI, TUI chat, API reference
- [Clustering](clustering.md) — Zero-config mDNS discovery, manual topology, model splitting
- [Image Generation](image_generation.md) — FLUX and Stable Diffusion image synthesis
- [Voice Generation](voice_generation.md) — VibeVoice TTS with voice cloning
- [Docker](docker.md) — Container builds for Linux/NVIDIA
