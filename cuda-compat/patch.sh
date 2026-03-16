#!/bin/bash
#
# Patch candle-kernels for pre-Volta NVIDIA GPUs (compute capability < 7.0).
#
# Fixes:
#   - atomicAdd(__half) fallback for sm < 70 (Pascal, Maxwell)
#   - __hmax/__hmin rewrites using float comparisons for old CUDA toolkits
#   - cuda_fp8.h guarded for CUDA < 11.8
#   - MOE WMMA kernels disabled (require sm_80+, Ampere)
#   - build.rs patched to conditionally include WMMA kernels
#
# Usage:
#   ./cuda-compat/patch.sh
#
# Run this before `cargo build --features cuda` on pre-Volta GPUs.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find candle-kernels source in cargo registry
CK_DIR=""
for d in "$HOME"/.cargo/registry/src/*/candle-kernels-*; do
    if [ -d "$d/src" ]; then
        CK_DIR="$d"
    fi
done

if [ -z "$CK_DIR" ]; then
    echo "error: candle-kernels not found in cargo registry."
    echo "run 'cargo build --features cuda' once first (it will fail, but downloads the crate)."
    exit 1
fi

echo "patching $CK_DIR ..."

# 1. Replace compatibility.cuh with our patched version
cp "$SCRIPT_DIR/compatibility.cuh" "$CK_DIR/src/compatibility.cuh"
echo "  patched compatibility.cuh"

# 2. Replace build.rs with conditional WMMA inclusion
cp "$SCRIPT_DIR/candle-kernels-build.rs" "$CK_DIR/build.rs"
echo "  patched build.rs"

# 3. Disable MOE WMMA kernels (require sm_80+)
for f in moe_wmma.cu moe_wmma_gguf.cu; do
    src="$CK_DIR/src/moe/$f"
    if [ -f "$src" ]; then
        mv "$src" "$src.bak"
        echo "  disabled $f (renamed to .bak)"
    fi
done

# 4. Fix F8E4M3 CUDA kernel naming mismatch in candle-core.
#    candle-core builds kernel name as "cast_f8e4m3_*" but the CUDA kernels
#    are named "cast_f8_e4m3_*" (with underscore). This breaks F8→BF16 on GPU.
CC_DIR=""
for d in "$HOME"/.cargo/registry/src/*/candle-core-*; do
    if [ -d "$d/src/cuda_backend" ]; then
        CC_DIR="$d"
    fi
done

if [ -n "$CC_DIR" ]; then
    CUDA_MOD="$CC_DIR/src/cuda_backend/mod.rs"
    if grep -q 'let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());' "$CUDA_MOD" 2>/dev/null; then
        sed -i 's|let kernel_name = format!("cast_{}_{}", self.dtype().as_str(), dtype.as_str());|// Fix F8E4M3 kernel naming: as_str() returns "f8e4m3" but CUDA kernels use "f8_e4m3"\n        let src_str = if self.dtype() == DType::F8E4M3 { "f8_e4m3" } else { self.dtype().as_str() };\n        let dst_str = if dtype == DType::F8E4M3 { "f8_e4m3" } else { dtype.as_str() };\n        let kernel_name = format!("cast_{}_{}", src_str, dst_str);|' "$CUDA_MOD"
        echo "  patched candle-core F8E4M3 kernel naming in $CC_DIR"
    else
        echo "  candle-core F8E4M3 patch: already applied or pattern not found"
    fi
else
    echo "  warning: candle-core not found in cargo registry, skipping F8 patch"
fi

# 5. Clean stale build artifacts so cargo picks up the changes
rm -rf "$REPO_DIR/target/release/build/candle-kernels-"* 2>/dev/null || true
rm -rf "$REPO_DIR/target/release/build/candle-core-"* 2>/dev/null || true
echo "  cleaned build cache"

echo "done. you can now build with: cargo build --release --features cuda"
