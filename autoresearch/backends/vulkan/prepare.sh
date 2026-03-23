#!/bin/bash
# prepare.sh — One-time setup for Vulkan backend autoresearch.
# Requires: Vulkan 1.3+ GPU.
#
# Usage: bash autoresearch/backends/vulkan/prepare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "========================================"
echo "  Cake Autoresearch — Vulkan Backend"
echo "========================================"
echo ""

cd "$PROJECT_DIR"

FEATURES="vulkan"

# --- Check Vulkan ---
if command -v vulkaninfo &>/dev/null; then
    echo "Vulkan device detected:"
    vulkaninfo --summary 2>/dev/null | head -10 || true
else
    echo "WARNING: vulkaninfo not found. Ensure Vulkan drivers are installed."
fi
echo ""

echo "Building cake-core with Vulkan..."
if ! cargo build -p cake-core --features "$FEATURES" 2>&1 | tail -5; then
    echo "ERROR: Build failed"
    exit 1
fi
echo "Build: OK"
echo ""

echo "Running tests..."
if ! cargo test -p cake-core --features "$FEATURES" --lib --test unit 2>&1 | tail -10; then
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
echo "  Saved to: autoresearch/backends/vulkan/baseline.txt"
echo "  TSV: autoresearch/backends/vulkan/experiments.tsv"
echo "========================================"
echo ""
echo "Ready. Start the agent with program.md."
