#!/bin/bash
# prepare.sh — One-time setup for image-generation autoresearch.
#
# Usage: bash autoresearch/models/image-generation/prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "========================================"
echo "  Cake Autoresearch — Image Generation"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

echo "Building cake-core with flux feature..."
if ! cargo build -p cake-core --features flux 2>&1 | tail -5; then
    echo "ERROR: Build failed"
    exit 1
fi
echo "Build: OK"
echo ""

echo "Running tests..."
if ! cargo test -p cake-core --lib --test unit 2>&1 | tail -10; then
    echo "ERROR: Tests failed"
    exit 1
fi
echo "Tests: OK"
echo ""

BASELINE_FILE="$SCRIPT_DIR/baseline.txt"
TSV_FILE="$SCRIPT_DIR/experiments.tsv"

if [ -f "$BASELINE_FILE" ]; then
    EXISTING=$(cat "$BASELINE_FILE")
    echo "Existing baseline found: $EXISTING ns"
    echo -n "Overwrite? [y/N] "
    read -r REPLY
    if [ "$REPLY" != "y" ] && [ "$REPLY" != "Y" ]; then
        echo "Keeping existing baseline."
        exit 0
    fi
fi

echo "Running baseline benchmark..."
echo ""

RESULT=$(bash "$SCRIPT_DIR/benchmark.sh")
echo "$RESULT"

SCORE=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'score=[0-9]+' | cut -d= -f2)
TESTS=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'tests=[A-Z]+' | cut -d= -f2)
CLIPPY=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'clippy=[A-Z]+' | cut -d= -f2)
STATUS=$(echo "$RESULT" | grep "BENCH_RESULT" | grep -oE 'status=[A-Z_]+' | cut -d= -f2)

if [ "$STATUS" != "OK" ]; then
    echo ""
    echo "ERROR: Baseline failed with status=$STATUS"
    exit 1
fi

echo "$SCORE" > "$BASELINE_FILE"

COMMIT=$(git -C "$PROJECT_DIR" rev-parse --short HEAD)
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%S)

echo -e "id\ttimestamp\tcommit\tscore_ns\ttests\tclippy\tstatus\tnotes" > "$TSV_FILE"
echo -e "000\t$TIMESTAMP\t$COMMIT\t$SCORE\t$TESTS\t$CLIPPY\tbaseline\tUnmodified baseline" >> "$TSV_FILE"

echo ""
echo "========================================"
echo "  Baseline: $SCORE ns"
echo "  Tests: $TESTS  Clippy: $CLIPPY"
echo "  Saved to: autoresearch/models/image-generation/baseline.txt"
echo "  TSV: autoresearch/models/image-generation/experiments.tsv"
echo "========================================"
echo ""
echo "Ready. Start the agent with program.md."
