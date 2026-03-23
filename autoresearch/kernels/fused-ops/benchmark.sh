#!/bin/bash
# benchmark.sh — Fixed harness for fused-ops kernel autoresearch.
# DO NOT MODIFY. This is the ground truth measurement.
#
# Measures: All CPU-path fused operations (silu_mul, rms_norm, add3, conv1d, etc.).
# Metric: total benchmark wall-clock time in nanoseconds (lower is better).
#
# Usage: bash autoresearch/kernels/fused-ops/benchmark.sh
# Exit codes: 0=all pass, 1=build fail, 2=test fail, 3=bench fail

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$PROJECT_DIR"

# --- Step 1: BUILD ---
echo "=== BUILD ===" >&2
if ! cargo build -p cake-core 2>&1 | tail -5 >&2; then
    echo "BENCH_RESULT score=0 tests=SKIP clippy=SKIP status=BUILD_FAIL"
    exit 1
fi
echo "Build OK" >&2

# --- Step 2: CLIPPY GATE ---
echo "=== CLIPPY ===" >&2
CLIPPY_PASS="FAIL"
if cargo clippy -p cake-core --lib --tests -- -D warnings 2>&1 | tail -5 >&2; then
    CLIPPY_PASS="PASS"
fi
echo "Clippy: $CLIPPY_PASS" >&2

# --- Step 3: TEST GATE ---
echo "=== TESTS ===" >&2
TEST_PASS="FAIL"
if cargo test -p cake-core --lib 2>&1 | tail -10 >&2 \
   && cargo test -p cake-core --test unit 2>&1 | tail -10 >&2 \
   && cargo test -p cake-core --test protocol 2>&1 | tail -10 >&2; then
    TEST_PASS="PASS"
fi
echo "Tests: $TEST_PASS" >&2

# --- Step 4: BENCHMARK ---
echo "=== BENCHMARK ===" >&2
BENCH_OUT=$(mktemp)
trap "rm -f $BENCH_OUT 2>/dev/null" EXIT

# Match all fused CPU ops + fp8 linear + native dtype benchmarks from bench_utils
BENCH_FILTER="fused_.*_cpu|fp8_linear|native_dtype|rms_norm_forward|silu_forward|softmax_last_dim|matmul_square|tensor_cat"
START_NS=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")

if ! DIVAN_SAMPLE_COUNT=3 cargo bench -p cake-core -- "$BENCH_FILTER" > "$BENCH_OUT" 2>&1; then
    echo "BENCH_RESULT score=0 tests=$TEST_PASS clippy=$CLIPPY_PASS status=BENCH_FAIL"
    cat "$BENCH_OUT" >&2
    exit 3
fi

END_NS=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
SCORE=$((END_NS - START_NS))

echo "Benchmark output:" >&2
cat "$BENCH_OUT" >&2
echo "" >&2
echo "Score: ${SCORE} ns (wall-clock)" >&2

# --- Step 5: STRUCTURED OUTPUT ---
if [ "$TEST_PASS" = "PASS" ] && [ "$CLIPPY_PASS" = "PASS" ]; then
    STATUS="OK"
    EXIT_CODE=0
else
    STATUS="QUALITY_FAIL"
    EXIT_CODE=2
fi

echo "BENCH_RESULT score=$SCORE tests=$TEST_PASS clippy=$CLIPPY_PASS status=$STATUS"
exit $EXIT_CODE
