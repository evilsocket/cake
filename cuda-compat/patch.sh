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

# 4. Clean stale build artifacts so cargo picks up the changes
rm -rf "$REPO_DIR/target/release/build/candle-kernels-"* 2>/dev/null || true
echo "  cleaned build cache"

echo "done. you can now build with: cargo build --release --features cuda"
