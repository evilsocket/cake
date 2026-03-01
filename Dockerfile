# =============================================================================
# Cake: Distributed LLM Inference - Docker Build (NVIDIA CUDA)
# =============================================================================
# Build:  docker build -t cake .
# Run:    docker run --rm --gpus all -v /path/to/model:/model:ro cake --model /model
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Chef — base image with Rust, cargo-chef, and system dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04 AS chef

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    libonig-dev \
    cmake \
    clang \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN cargo install cargo-chef

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 2: Planner — generate dependency recipe
# ---------------------------------------------------------------------------
FROM chef AS planner

COPY . .
RUN cargo chef prepare --recipe-path recipe.json

# ---------------------------------------------------------------------------
# Stage 3: Builder — cook dependencies (cached), then build workspace
# ---------------------------------------------------------------------------
FROM chef AS builder

COPY --from=planner /app/recipe.json recipe.json

# CUDA compute capability — set to match your target GPU(s)
# Common values: 75 (RTX 2000), 80 (A100), 86 (RTX 3000), 89 (RTX 4000), 90 (H100)
ARG CUDA_COMPUTE_CAP=80

# Cook dependencies — this layer is cached until Cargo.toml/Cargo.lock change
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}
RUN cargo chef cook --release --recipe-path recipe.json -p cake-cli -p cake-split-model --features cuda

# Copy full source and build
COPY . .
RUN cargo build --release -p cake-cli -p cake-split-model --features cuda

# ---------------------------------------------------------------------------
# Stage 4: Runtime — minimal image with just the binaries
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.6.0-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3t64 \
    libonig5 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/cake-cli /usr/local/bin/cake
COPY --from=builder /app/target/release/cake-split-model /usr/local/bin/cake-split-model

# Worker port
EXPOSE 10128
# API port (master mode)
EXPOSE 8080

ENTRYPOINT ["cake"]
