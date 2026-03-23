#!/usr/bin/env bash
# install-dev.sh — Bootstrap a fresh GPU instance for cake development.
# Usage: ssh user@host 'bash -s' < install-dev.sh
#    or: scp install-dev.sh user@host: && ssh user@host bash install-dev.sh
set -euo pipefail

echo "=== Installing system dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq pkg-config libssl-dev build-essential git

echo "=== Installing Rust ==="
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
echo "Rust: $(rustc --version)"

echo "=== Detecting CUDA ==="
CUDA_HOME=""
for p in /usr/local/cuda-12.4 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -x "$p/bin/nvcc" ]; then
        CUDA_HOME="$p"
        break
    fi
done
if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: No CUDA installation found in /usr/local/cuda*"
    exit 1
fi
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
echo "CUDA: $($CUDA_HOME/bin/nvcc --version | tail -1)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo "=== Persisting environment ==="
cat > "$HOME/.cake-env" << 'ENVEOF'
export CUDA_HOME="__CUDA_HOME__"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
source "$HOME/.cargo/env" 2>/dev/null || true
ENVEOF
sed -i "s|__CUDA_HOME__|$CUDA_HOME|g" "$HOME/.cake-env"
grep -q '.cake-env' "$HOME/.bashrc" 2>/dev/null || echo 'source "$HOME/.cake-env"' >> "$HOME/.bashrc"

echo "=== Cloning cake ==="
if [ ! -d "$HOME/cake" ]; then
    git clone https://github.com/evilsocket/cake.git "$HOME/cake"
else
    cd "$HOME/cake" && git pull
fi

echo "=== Building cake (release, CUDA) ==="
cd "$HOME/cake"
cargo build --release --features cuda

echo "=== Running tests ==="
cargo test -p cake-core --features cuda 2>&1 | grep "^test result"

echo ""
echo "=== Done ==="
echo "cake binary: $HOME/cake/target/release/cake"
echo "To use: source ~/.cake-env && cd ~/cake"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
