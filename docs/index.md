# Cake Documentation

Cake is a Rust framework for distributed inference of large language models and image generation models. It shards transformer blocks across multiple devices — iOS, Android, macOS, Linux, Windows — to run models that wouldn't fit on a single GPU.

Built on [Candle](https://github.com/huggingface/candle) with support for CUDA, Metal, and CPU backends.

## Table of Contents

- [Installation](install.md) — Building from source, platform support, acceleration backends
- [Models](models.md) — Supported model architectures and feature flags
- [Usage](usage.md) — Downloading models, running inference, Web UI, TUI chat, API reference
- [Clustering](clustering.md) — Zero-config mDNS discovery, manual topology, model splitting
- [Docker](docker.md) — Container builds for Linux/NVIDIA
- [Stable Diffusion](stable_diffusion.md) — Distributed image generation
