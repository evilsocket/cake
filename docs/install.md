# Installation

## Platform Support

| OS | Architectures | Acceleration | Status |
|----|---------------|--------------|--------|
| GNU/Linux | arm, arm64, x86_64 | - | Supported |
| GNU/Linux | arm, arm64, x86_64 | CUDA | Supported |
| GNU/Linux | arm, arm64, x86_64 | BLAS | Supported |
| Windows | x86_64 | BLAS | [Untested](https://github.com/evilsocket/cake/issues/7) |
| Windows | x86_64 | CUDA | Supported |
| macOS | x86_64 | - | Supported |
| macOS | aarch64 | - | Supported |
| macOS | aarch64 | Metal | Supported |
| Android | arm, arm64, x86_64 | - | Supported |
| Android | arm, arm64, x86_64 | CUDA | [Untested](https://docs.nvidia.com/gameworks/content/technologies/mobile/cuda_android_main.htm) |
| iOS / iPadOS | aarch64 | - | Supported |
| iOS / iPadOS | aarch64 | Metal | [WIP](https://github.com/huggingface/candle/issues/2322) |

CUDA >= 12.2 is required for CUDA accelerated systems.

## Building from Source

With [Rust installed](https://www.rust-lang.org/tools/install):

**CPU only (no acceleration):**

```sh
cargo build --release
```

**Metal (Apple Silicon):**

```sh
cargo build --release --features metal
```

**CUDA:**

```sh
cargo build --release --features cuda
```

If your system has **multiple CUDA toolkit versions** installed, set `CUDA_HOME` to the
version supported by your driver to avoid library version mismatches:

```sh
CUDA_HOME=/usr/local/cuda-12.4 cargo build --release --features cuda
```

### Pre-Volta NVIDIA GPUs

For Pascal, Maxwell, or other GPUs with compute capability < 7.0, the upstream `candle-kernels` crate requires patches. See [`cuda-compat/`](../cuda-compat/) for a one-command fix:

```sh
./cuda-compat/patch.sh
cargo build --release --features cuda
```

### iOS Bindings

To generate the iOS bindings that can then be [compiled and deployed via Xcode](https://github.com/evilsocket/cake/tree/main/cake-ios-worker-app):

```sh
make ios
```

## Feature Flags

By default, all text model architectures are compiled in. To build only for specific models:

```sh
# Only LLaMA support
cargo build --release --no-default-features --features llama

# Only Qwen2 support
cargo build --release --no-default-features --features qwen2

# Only Qwen3.5 support
cargo build --release --no-default-features --features qwen3_5
```
