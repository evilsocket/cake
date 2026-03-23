#!/bin/bash
# benchmark.sh — Fixed harness for ROCm backend autoresearch.
# DO NOT MODIFY. This is the ground truth measurement.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$PROJECT_DIR"
FEATURES="rocm"
echo "=== BUILD ===" >&2
if ! cargo build -p cake-core --features "$FEATURES" 2>&1 | tail -5 >&2; then
    echo "BENCH_RESULT score=0 tests=SKIP clippy=SKIP status=BUILD_FAIL"; exit 1
fi
echo "Build OK" >&2
echo "=== CLIPPY ===" >&2
CLIPPY_PASS="FAIL"
if cargo clippy -p cake-core --features "$FEATURES" --lib --tests -- -D warnings 2>&1 | tail -5 >&2; then CLIPPY_PASS="PASS"; fi
echo "Clippy: $CLIPPY_PASS" >&2
echo "=== TESTS ===" >&2
TEST_PASS="FAIL"
if cargo test -p cake-core --features "$FEATURES" --lib 2>&1 | tail -10 >&2 \
   && cargo test -p cake-core --features "$FEATURES" --test unit 2>&1 | tail -10 >&2 \
   && cargo test -p cake-core --features "$FEATURES" --test protocol 2>&1 | tail -10 >&2; then TEST_PASS="PASS"; fi
echo "Tests: $TEST_PASS" >&2
echo "=== BENCHMARK ===" >&2
BENCH_OUT=$(mktemp)
trap "rm -f $BENCH_OUT 2>/dev/null" EXIT
BENCH_FILTER="backend_ops|attention"
START_NS=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
if ! DIVAN_SAMPLE_COUNT=3 cargo bench -p cake-core --features "$FEATURES" -- "$BENCH_FILTER" > "$BENCH_OUT" 2>&1; then
    echo "BENCH_RESULT score=0 tests=$TEST_PASS clippy=$CLIPPY_PASS status=BENCH_FAIL"
    cat "$BENCH_OUT" >&2; exit 3
fi
END_NS=$(date +%s%N 2>/dev/null || python3 -c "import time; print(int(time.time()*1e9))")
SCORE=$((END_NS - START_NS))
echo "Benchmark output:" >&2; cat "$BENCH_OUT" >&2; echo "" >&2
echo "Score: ${SCORE} ns (wall-clock)" >&2
if [ "$TEST_PASS" = "PASS" ] && [ "$CLIPPY_PASS" = "PASS" ]; then STATUS="OK"; EXIT_CODE=0
else STATUS="QUALITY_FAIL"; EXIT_CODE=2; fi
echo "BENCH_RESULT score=$SCORE tests=$TEST_PASS clippy=$CLIPPY_PASS status=$STATUS"
exit $EXIT_CODE
