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
#   ./cake-core/src/backends/cuda/compat/patch.sh
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

# 3.5. Guard FP8 cast kernels in cast.cu for GPUs without cuda_fp8.h / __nv_fp8_e4m3.
#      We wrap the FP8 code in TWO guards:
#      Guard A: FP8 template functions + FP8 macro defs (F8E4M3_TO_FLOAT through CAST_OP_FP8_INTO)
#      Guard B: FP8 CAST_OP instantiations (CAST_OP with f8_e4m3 args)
CAST_CU="$CK_DIR/src/cast.cu"
FP8_GUARD='#if (__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)'
if [ -f "$CAST_CU" ] && ! grep -q 'CAKE_COMPAT_FP8_GUARD' "$CAST_CU"; then
    # Guard A: from #define F8E4M3_TO_FLOAT to end of CAST_OP_FP8_INTO macro def.
    # Find the line AFTER the CAST_OP_FP8_INTO closing "} \"
    GUARD_A_START=$(grep -n '#define F8E4M3_TO_FLOAT' "$CAST_CU" | head -1 | cut -d: -f1)
    # CAST_OP_FP8_INTO ends with "} \" — find the line number of the closing brace-backslash
    GUARD_A_END=$(grep -n 'CAST_OP_FP8_INTO' "$CAST_CU" | head -1 | cut -d: -f1)
    if [ -n "$GUARD_A_END" ]; then
        # Find the "} \" line after CAST_OP_FP8_INTO
        GUARD_A_END=$(tail -n "+$GUARD_A_END" "$CAST_CU" | grep -n '^} \\' | head -1 | cut -d: -f1)
        GUARD_A_END=$((GUARD_A_END + $(grep -n 'CAST_OP_FP8_INTO' "$CAST_CU" | head -1 | cut -d: -f1) - 1))
    fi

    # Guard B: all CAST_OP/CAST_THROUGH_OP lines with f8_e4m3
    GUARD_B_START=$(grep -n 'CAST_OP.*__nv_fp8_e4m3' "$CAST_CU" | head -1 | cut -d: -f1)
    GUARD_B_END=$(grep -n 'f8_e4m3' "$CAST_CU" | tail -1 | cut -d: -f1)

    if [ -n "$GUARD_A_START" ] && [ -n "$GUARD_A_END" ] && [ -n "$GUARD_B_START" ] && [ -n "$GUARD_B_END" ]; then
        # Apply all guards in a single awk pass to avoid line-number shift issues
        awk -v a_start="$GUARD_A_START" -v a_end="$GUARD_A_END" \
            -v b_start="$GUARD_B_START" -v b_end="$GUARD_B_END" '
        NR == a_start { print "// CAKE_COMPAT_FP8_GUARD A"; print "#if (__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)" }
        { print }
        NR == a_end   { print "#endif // CAKE_COMPAT_FP8_GUARD A" }
        NR == b_start { print "// CAKE_COMPAT_FP8_GUARD B"; print "#if (__CUDACC_VER_MAJOR__ > 11) || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 8)" }
        NR == b_end   { print "#endif // CAKE_COMPAT_FP8_GUARD B" }
        ' "$CAST_CU" > "$CAST_CU.tmp" && mv "$CAST_CU.tmp" "$CAST_CU"
        echo "  patched cast.cu: guarded FP8 kernels (A:${GUARD_A_START}-${GUARD_A_END}, B:${GUARD_B_START}-${GUARD_B_END})"
    else
        echo "  cast.cu FP8 guard: could not find FP8 markers (A:${GUARD_A_START}-${GUARD_A_END}, B:${GUARD_B_START}-${GUARD_B_END})"
    fi
else
    echo "  cast.cu FP8 guard: already applied or file not found"
fi

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
