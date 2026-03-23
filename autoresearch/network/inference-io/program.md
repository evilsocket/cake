# Inference I/O Autoresearch

Autonomous optimization of the per-token client↔worker data path during distributed inference.

## Objective

**Minimize per-token network round-trip latency** between master and workers. Every generated
token requires: serialize tensor → send → worker forward → serialize result → receive. This
loop runs once per layer per token. For a 24-layer model sharded across 3 workers, that's
~8 roundtrips per token. Any overhead here directly impacts tok/s. Lower is better.

## Setup

1. **Create the branch**: `git checkout -b autoresearch/network/inference-io`
2. **Read the in-scope files** for full context:

   ### Client (master-side per-token path)
   - `cake-core/src/cake/sharding/client.rs` — Client struct connects to a worker via TCP.
     **Hot path**: `forward_request()` serializes the tensor, sends via `to_writer_buf()` with
     a reusable 64KB buffer, receives the response via `from_reader_buf()` with another 64KB
     buffer. Measures send_elapsed and recv_elapsed separately. `forward_mut()` wraps this for
     the `Forwarder` trait (single-op path). `forward_batch()` sends multiple layers in one
     message to reduce roundtrips.

   ### Worker (worker-side per-token path)
   - `cake-core/src/cake/sharding/worker.rs` — Worker's inference loop (the main hot path):
     1. `from_reader_buf()` — deserialize incoming message (reusable 64KB buffer)
     2. `RawTensor::to_tensor()` — load tensor to device (zero-copy wrapping)
     3. Per-layer: bind CUDA context (multi-GPU) → `block.forward()` → GPU sync
     4. `RawTensor::from_tensor()` — serialize result tensor
     5. `to_writer_buf()` — send response (reusable 64KB buffer)
     Reports timing breakdown: read/load/fwd/ser/write every 5 ops.

   ### Protocol messages
   - `cake-core/src/cake/sharding/proto/message.rs` — Message::SingleOp and Message::Batch
     are the per-token message types. `to_writer_buf` / `from_reader_buf` use caller-provided
     buffers. `to_bytes()` / `from_bytes()` allocate (used in benchmarks).
   - `cake-core/src/cake/sharding/proto/mod.rs` — RawTensor: tensor↔bytes conversion.

   ### Master orchestration
   - `cake-core/src/cake/sharding/master.rs` — Coordinates workers. `next_token()` iterates
     layers, dispatching each to the appropriate worker or running locally.
   - `cake-core/src/cake/mod.rs` — `Forwarder` trait, `Context` struct, generation loop.

   ### Benchmarks
   - `cake-core/benches/bench_protocol.rs` — Tensor encode/decode, pipe roundtrips,
     batch encoding, buffer reuse comparison.
   - `cake-core/benches/bench_serialization.rs` — RawTensor from/to tensor roundtrips.

3. **Run prepare.sh**: `bash autoresearch/network/inference-io/prepare.sh`
4. **Confirm baseline** and start experimenting.

## Files You May Modify

- `cake-core/src/cake/sharding/client.rs` — Client buffer management, message construction,
  connection pooling, TCP options, batch vs single-op decision logic.
- `cake-core/src/cake/sharding/worker.rs` — Worker inference loop, device transfer, response
  construction, multi-GPU context binding, synchronization strategy.
- `cake-core/src/cake/sharding/proto/message.rs` — Message serialization, wire format,
  buffer management in to_writer_buf/from_reader_buf.
- `cake-core/src/cake/sharding/proto/mod.rs` — RawTensor encoding, dtype dispatch,
  shape serialization.
- `cake-core/src/cake/sharding/master.rs` — Layer dispatch logic, worker coordination.
- `cake-core/src/cake/mod.rs` — Forwarder trait default implementations, Context.

## Files You Must NOT Modify

- `autoresearch/network/inference-io/benchmark.sh`
- `autoresearch/network/inference-io/prepare.sh`
- `cake-core/benches/bench_protocol.rs`
- `cake-core/benches/bench_serialization.rs`
- `cake-core/benches/bench_helpers.rs`
- Any test files

## The Experiment Loop

**LOOP FOREVER:**

1. **Propose** a single, focused optimization hypothesis.
2. **Implement** the change in the allowed source files.
3. **Commit**: `git add cake-core/src/cake/ && git commit -m "<description>"`
4. **Quality gate**: `cargo clippy -p cake-core --lib --tests -- -D warnings && cargo test -p cake-core --lib && cargo test -p cake-core --test unit && cargo test -p cake-core --test protocol`
5. **Benchmark**: `bash autoresearch/network/inference-io/benchmark.sh`
6. **Parse** BENCH_RESULT, **decide**, **record**, **repeat**.

## Hot Path Architecture

### Per-Token Data Flow
```
Master                          Worker
  │                               │
  ├─ RawTensor::from_tensor()     │         ← serialize (0.1-0.3 µs)
  ├─ Message::to_writer_buf()  ──►│         ← network send
  │                               ├─ from_reader_buf()    ← deserialize
  │                               ├─ RawTensor::to_tensor() ← device load
  │                               ├─ block.forward()      ← compute (1-100 ms)
  │                               ├─ RawTensor::from_tensor() ← serialize
  │◄── to_writer_buf()  ─────────┤         ← network send
  ├─ from_reader_buf()            │         ← deserialize
  ├─ RawTensor::to_tensor()       │         ← device load
  │                               │
```

### Buffer Reuse
Both client and worker pre-allocate 64KB read/write buffers (`Vec<u8>`) that persist across
all tokens in a session. The `truncate(0)` + capacity preservation pattern avoids heap
allocation on every message. **Do not break this pattern.**

### Tensor Sizes (Qwen3.5-0.8B)
- Hidden state: `(1, seq_len, 1024)` → 1024 F16 elements = 2048 bytes per token
- KV cache grows with context: `(1, heads, ctx_len, head_dim)`
- Batch mode: same tensor, multiple `(layer_name, idx, block_idx)` tuples

## Promising Areas to Explore

1. **Eliminate RawTensor allocation on hot path** — `from_tensor()` does `x.data().into_owned()`
   which copies the tensor data to a new Vec. If we could serialize the tensor directly into
   the write buffer without the intermediate RawTensor, we'd save one allocation + memcpy.

2. **Batch layer dispatch** — Currently the client sends one SingleOp per layer. The Batch
   message type exists but may not be used optimally. Batching all layers for a worker into
   one message would reduce roundtrips.

3. **TCP_NODELAY** — Check if the TCP connections use TCP_NODELAY. Without it, Nagle's
   algorithm adds up to 40ms delay on small packets.

4. **Response pre-allocation** — The worker creates a new RawTensor for every response.
   Pre-allocating a response buffer sized to the expected output would avoid allocation.

5. **Device transfer optimization** — `RawTensor::to_tensor()` calls
   `Tensor::from_raw_buffer()` which may copy data to GPU. On CPU, this should be zero-copy.
   Verify the copy behavior on each device type.

6. **Reduce message framing overhead** — The 8-byte header (magic + length) is read in two
   separate `read_u32()` calls. Reading 8 bytes at once would save one syscall.

7. **Worker-side read buffer sizing** — The 64KB default may be suboptimal. For small tensors
   (2KB), the buffer is oversized. For large batches, it may need to grow. Profile the
   actual size distribution.

8. **Async pipeline overlap** — Can the worker start deserializing the next request while
   the previous response is still being written? (Requires careful ordering.)

9. **Zero-copy write path** — Instead of serializing into a buffer then writing, write
   the header directly followed by streaming serialization into the TCP socket.

## Recording Results

Append to `experiments.tsv` (tab-separated):
```
id	timestamp	commit	score_ns	tests	clippy	status	notes
```

## Decision Rules

- **Keep** if: `score <= previous_best * 1.005` AND tests PASS AND clippy PASS
- **Discard** if: `score > previous_best * 1.005` OR any gate fails

## Safety

- Do not change the wire format magic bytes
- Do not break buffer reuse patterns (this would regress per-token allocation)
- Protocol tests must pass (they verify client↔worker communication)
- Do not change the Forwarder trait signature

## NEVER STOP

Run autonomously until manually interrupted. Each experiment takes ~30-60 seconds.
