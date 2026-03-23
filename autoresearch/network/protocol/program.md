# Network Protocol Autoresearch

Autonomous optimization of the distributed inference protocol: message serialization,
tensor wire format, model data transfer, authentication, discovery, and topology management.

## Objective

**Minimize protocol overhead** for distributed inference across a heterogeneous GPU cluster.
The protocol carries tensor data between master and workers on every forward pass — any
per-message overhead is multiplied by (num_layers × num_tokens). Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/network/protocol`
2. **Read the in-scope files** for full context:

   ### Wire Protocol
   - `cake-core/src/cake/sharding/proto/message.rs` — Message enum: Hello, WorkerInfo,
     LayerAssignment, SingleOp, Batch, Tensor, ModelDataChunk, etc. Serialization via
     `speedy` with BigEndian encoding. Wire format: 4-byte magic + 4-byte length + payload.
     **Primary optimization target** — every inference token serializes/deserializes here.
   - `cake-core/src/cake/sharding/proto/mod.rs` — RawTensor: dtype-aware tensor ↔ bytes
     conversion. Shape encoding, data extraction from candle tensors.

   ### Client/Worker I/O
   - `cake-core/src/cake/sharding/client.rs` — Client (master-side): connects to worker,
     sends forward requests, receives results. Buffer reuse patterns (write_buf, read_buf).
     Async I/O with tokio.
   - `cake-core/src/cake/sharding/worker.rs` — Worker: accepts connections, processes
     forward requests, sends results. Main I/O loop with message dispatch.

   ### Model Data Transfer
   - Uses zstd compression (level 1) + CRC32 checksums for model weight chunks.
     `ModelDataChunk` messages carry safetensors data to workers.

   ### Authentication
   - `cake-core/src/cake/sharding/auth.rs` — HMAC-based authentication. Challenge-response
     handshake: master sends nonce, worker computes HMAC, master verifies. Constant-time
     comparison for security.

   ### Discovery
   - `cake-core/src/cake/sharding/discovery.rs` — UDP broadcast-based cluster discovery.
     Workers announce GPU capabilities (VRAM, TFLOPS). Cluster hash for namespace isolation.
     GPU info detection (CUDA, Metal, Vulkan).

   ### Topology
   - `cake-core/src/cake/sharding/topology.rs` — Layer-to-worker mapping. Auto-assignment
     proportional to TFLOPS. Layer ownership queries for weight loading.
   - `cake-core/src/cake/sharding/default.rs` — DefaultStrategy for TFLOPS-proportional
     layer distribution.

   ### Benchmarks
   - `cake-core/benches/bench_protocol.rs` — Message encode/decode, pipe roundtrips,
     buffer reuse, batch encoding, zstd compression, CRC32 checksums.
   - `cake-core/benches/bench_serialization.rs` — RawTensor from_tensor/to_tensor roundtrips.
   - `cake-core/benches/bench_discovery.rs` — Cluster hash, VRAM calculations, packet codec.
   - `cake-core/benches/bench_topology.rs` — Layer lookup, auto-assignment, TFLOPS estimation.
   - `cake-core/benches/bench_auth.rs` — HMAC computation, constant-time eq, full handshake.

3. **Run prepare.sh**: `bash autoresearch/network/protocol/prepare.sh`
4. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/cake/sharding/proto/message.rs` — Message serialization. Wire format,
  dtype encoding, payload layout, buffer management.
- `cake-core/src/cake/sharding/proto/mod.rs` — RawTensor conversion. Shape encoding,
  data copy patterns, dtype dispatch.
- `cake-core/src/cake/sharding/client.rs` — Client I/O. Buffer allocation strategy,
  message framing, async read/write patterns.
- `cake-core/src/cake/sharding/worker.rs` — Worker I/O. Message dispatch, buffer reuse,
  response construction.
- `cake-core/src/cake/sharding/auth.rs` — Authentication. HMAC implementation, handshake
  sequence, key derivation.
- `cake-core/src/cake/sharding/discovery.rs` — Discovery protocol. Packet encoding,
  cluster hash, GPU detection, VRAM/TFLOPS calculation.
- `cake-core/src/cake/sharding/topology.rs` — Topology management. Layer lookup data
  structure, auto-assignment algorithm, ownership queries.
- `cake-core/src/cake/sharding/default.rs` — Sharding strategy implementation.

## Files You Must NOT Modify

- `autoresearch/network/protocol/benchmark.sh`
- `autoresearch/network/protocol/prepare.sh`
- `cake-core/benches/bench_protocol.rs`
- `cake-core/benches/bench_serialization.rs`
- `cake-core/benches/bench_discovery.rs`
- `cake-core/benches/bench_topology.rs`
- `cake-core/benches/bench_auth.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files (`cake-core/tests/`)

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/cake/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol`
5. **Benchmark**: `bash autoresearch/network/protocol/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Protocol Architecture Constraints

### Wire Format
Messages use a 4-byte magic header + 4-byte length prefix + `speedy` BigEndian payload.
The magic bytes provide framing — do not change the magic value or framing structure as
this would break backward compatibility with running workers.

### Async I/O
Client and worker use tokio async I/O. All reads/writes go through `AsyncReadExt` /
`AsyncWriteExt`. Buffer reuse is critical — the `to_writer_buf` / `from_reader_buf`
variants use caller-provided buffers to avoid allocation on the hot path.

### Tensor Wire Format
`RawTensor` encodes tensors as: dtype tag + shape dimensions + raw byte data.
The dtype tag must round-trip exactly — any mismatch corrupts inference results.
Shape is encoded as a fixed-size array for fast parsing.

### Authentication
HMAC uses SHA-256 via the `hmac` and `sha2` crates. The constant-time comparison
(`constant_time_eq`) is security-critical — do not replace with a non-constant-time
implementation even if faster.

### Compression
Model data chunks use zstd level 1 (fast compression, moderate ratio). The compression
level is a tradeoff between CPU time and network bandwidth. Higher levels may reduce
transfer time on slow networks but increase CPU overhead.

## Promising Areas to Explore

1. **Pre-allocated message buffers** — `to_bytes()` allocates a new Vec each call. Using
   a reusable buffer (like `to_writer_buf`) for the in-memory path could eliminate allocation.

2. **Zero-copy tensor extraction** — `RawTensor::to_tensor()` copies data into a new candle
   Tensor. If the message buffer lifetime can be extended, the tensor could reference the
   original bytes directly.

3. **Batch message encoding** — The batch message encodes layer names as separate strings.
   Using string interning or index-based layer references could reduce per-message size.

4. **Topology lookup optimization** — `get_node_for_layer` iterates all nodes. A pre-built
   HashMap from layer name → node would be O(1) instead of O(nodes × layers).

5. **Auto-assignment algorithm** — TFLOPS-proportional assignment iterates and divides.
   Could be simplified to a single-pass proportional distribution.

6. **Discovery packet compression** — GPU info payloads contain repeated strings.
   Compact binary encoding could reduce UDP packet sizes.

7. **HMAC key caching** — If the same key is used for many connections, the HMAC key
   expansion could be cached rather than recomputed per handshake.

8. **CRC32 acceleration** — The `crc32fast` crate already uses hardware acceleration
   where available, but the integration pattern (allocate compressed → compute CRC →
   build message) might allow streaming CRC computation.

9. **Speedy serialization alternatives** — Measure whether the `speedy` crate's BigEndian
   encoding is optimal, or if a custom binary encoding with fewer branches would be faster
   for the specific Message enum.

10. **Writer/reader buffer sizing** — The default buffer sizes in client/worker may not be
    optimal for typical tensor sizes. Profiling the size distribution could inform better defaults.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails
- If **3 consecutive discards**: move to a different area
- After every **keep**: update mental baseline

## Safety

- Do not change the wire format magic bytes (breaks running clusters)
- Do not change `constant_time_eq` to a non-constant-time implementation
- Do not change Message enum variant discriminants (breaks protocol compat)
- Do not add new dependencies to Cargo.toml
- Protocol tests (`cargo test --test protocol`) must pass — they verify wire compatibility

## NEVER STOP

Run autonomously until manually interrupted. Each experiment takes ~30-60 seconds.
