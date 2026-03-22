# cuda-compat: Pre-Volta NVIDIA GPU Support

This directory contains patches for [`candle-kernels`](https://github.com/huggingface/candle) that enable Cake to build and run on **pre-Volta NVIDIA GPUs** (compute capability < 7.0), such as Pascal-era cards (GTX 1080, TITAN X, Tesla P100, etc.).

## Why This Exists

The upstream `candle-kernels` crate uses CUDA intrinsics and headers that require Volta (sm_70) or newer hardware:

- **`atomicAdd(__half)`** requires sm_70+; Pascal (sm_61) needs a CAS-based fallback.
- **`__hmax` / `__hmin`** intrinsics are not available on pre-Volta; we rewrite them using float comparisons.
- **`cuda_fp8.h`** was introduced in CUDA 11.8; older toolkits need a preprocessor guard.
- **MOE WMMA kernels** require sm_80+ (Ampere); they are disabled on older GPUs.

Without these patches, `cargo build --features cuda` fails with compilation errors on pre-Volta hardware.

## What's Included

| File | Purpose |
|------|---------|
| `compatibility.cuh` | Patched version of `candle-kernels/src/compatibility.cuh` with atomicAdd fallback, `__hmax_nan`/`__hmin_nan` rewrites, and FP8 header guards |
| `candle-kernels-build.rs` | Modified `build.rs` that conditionally includes WMMA kernels only when the source files exist |
| `patch.sh` | Shell script that finds `candle-kernels` in the Cargo registry and applies all patches automatically |

## Usage

First, trigger a download of `candle-kernels` (the build will fail, that's expected):

```sh
cargo build --features cuda || true
```

Then apply the patches:

```sh
./cake-core/src/backends/cuda/compat/patch.sh
```

Now build normally:

```sh
cargo build --release --features cuda
```

## Supported Configurations

Tested with:
- NVIDIA TITAN X (Pascal, sm_61) + CUDA 12.4
- Should work with any sm_50+ GPU and CUDA >= 11.5

## Notes

- CUDA 13.x dropped support for sm < 75. Use CUDA 12.x for Pascal/Maxwell GPUs.
- These patches are applied to the local Cargo registry copy of `candle-kernels`. They need to be re-applied after `cargo update` pulls a new version.
