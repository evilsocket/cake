# Model Transfer Autoresearch

Autonomous optimization of model weight distribution from master to workers.

## Objective

**Minimize model transfer time** when distributing safetensors weights to workers over
the network. This is a one-time cost per session but directly impacts time-to-first-token.
For large models (7B+), transfer can take minutes. Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/network/model-transfer`
2. **Read the in-scope files** for full context:

   ### Model data transfer
   - `cake-core/src/cake/sharding/proto/message.rs` — `ModelDataChunk` message: carries
     filename, offset, total_size, compressed flag, CRC32 checksum, and data payload.
     `ModelDataResume` for resumable transfers. `ModelDataDone` signals completion.
   - `cake-core/src/cake/sharding/master.rs` — `send_model_data()`: reads safetensors files,
     splits into 128MB chunks, optionally compresses with zstd level 1, computes CRC32,
     sends via `ModelDataChunk` messages. Handles resume requests.
   - `cake-core/src/cake/sharding/worker.rs` — `receive_model_data()`: receives chunks,
     verifies CRC32, decompresses if needed, writes to disk. Detects partial files for
     resume. Loads model weights from saved files.
   - `cake-core/src/cake/sharding/mod.rs` — `MODEL_DATA_CHUNK_SIZE` (128MB), compression
     constants, model hash computation.

   ### Benchmarks
   - `cake-core/benches/bench_protocol.rs` — `zstd_compress_chunk` (1KB, 64KB, 1MB),
     `zstd_decompress_chunk`, `crc32_checksum`, `model_data_chunk_roundtrip_compressed`,
     `model_data_chunk_roundtrip_uncompressed`.

3. **Run prepare.sh**: `bash autoresearch/network/model-transfer/prepare.sh`
4. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/cake/sharding/proto/message.rs` — ModelDataChunk serialization,
  payload layout.
- `cake-core/src/cake/sharding/master.rs` — Chunk size, compression strategy,
  file reading patterns, parallel transfer logic.
- `cake-core/src/cake/sharding/worker.rs` — Chunk reception, decompression,
  file writing, resume logic.
- `cake-core/src/cake/sharding/mod.rs` — MODEL_DATA_CHUNK_SIZE constant,
  compression level, model hash.

## Files You Must NOT Modify

- `autoresearch/network/model-transfer/benchmark.sh`
- `autoresearch/network/model-transfer/prepare.sh`
- `cake-core/benches/bench_protocol.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/cake/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol`
5. **Benchmark**: `bash autoresearch/network/model-transfer/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Transfer Architecture

### Current Flow
```
Master                                    Worker
  │                                         │
  ├─ Read safetensors file (128MB chunk)    │
  ├─ zstd compress (level 1)               │
  ├─ CRC32 checksum                        │
  ├─ ModelDataChunk message ────────────►   │
  │                                         ├─ Verify CRC32
  │                                         ├─ zstd decompress
  │                                         ├─ Write to disk
  │  (repeat until all chunks sent)         │
  ├─ ModelDataDone ─────────────────────►   │
  │                                         ├─ Load model from disk
```

### Constants
- `MODEL_DATA_CHUNK_SIZE`: 128 MB (per chunk)
- zstd compression level: 1 (fastest)
- CRC32: computed on compressed data

## Promising Areas to Explore

1. **Streaming compression** — Currently the entire chunk is compressed into a Vec, then
   checksummed, then serialized. Streaming zstd → CRC32 → wire would reduce memory usage
   and potentially overlap CPU and I/O.

2. **Chunk size tuning** — 128MB is arbitrary. Smaller chunks enable better progress
   reporting and resume granularity. Larger chunks reduce per-chunk overhead. Profile
   the optimal size for typical network conditions.

3. **Parallel chunk transfer** — If a worker has multiple assigned layers from different
   safetensors files, chunks from different files could be sent in parallel.

4. **Adaptive compression** — Model weights are pseudo-random (poor compression ratio).
   For F16/BF16 weights, compression overhead may exceed savings. Skip compression
   when the data is incompressible.

5. **CRC32 streaming** — Compute CRC32 during compression instead of after, eliminating
   one full pass over the data.

6. **Memory-mapped file reading** — Use mmap instead of read() for the safetensors files
   to let the OS handle page caching and reduce memory copies.

7. **Decompression buffer reuse** — Worker-side decompression allocates a new Vec per
   chunk. Reusing a buffer would reduce allocation pressure.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## Safety

- Do not change ModelDataChunk message structure (wire compat with workers)
- CRC32 verification must remain — it prevents corrupt weight loading
- Resume logic must be preserved — workers may restart mid-transfer

## NEVER STOP

Run autonomously until manually interrupted. Each experiment takes ~30-60 seconds.
