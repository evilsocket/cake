use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use safetensors::View;
use speedy::{BigEndian, Readable, Writable};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Map a candle DType to a compact u8 tag for wire encoding.
fn dtype_to_u8(dtype: DType) -> u8 {
    match dtype {
        DType::U8 => 0,
        DType::U32 => 1,
        DType::I64 => 2,
        DType::BF16 => 3,
        DType::F16 => 4,
        DType::F32 => 5,
        DType::F64 => 6,
        DType::F8E4M3 => 7,
        // Catch-all for newer candle dtypes we don't use on the wire yet.
        _ => 255,
    }
}

/// Map a u8 wire tag back to a candle DType.
fn u8_to_dtype(tag: u8) -> Result<DType> {
    match tag {
        0 => Ok(DType::U8),
        1 => Ok(DType::U32),
        2 => Ok(DType::I64),
        3 => Ok(DType::BF16),
        4 => Ok(DType::F16),
        5 => Ok(DType::F32),
        6 => Ok(DType::F64),
        7 => Ok(DType::F8E4M3),
        _ => Err(anyhow!("unknown dtype tag: {tag}")),
    }
}

/// Represents a tensor in Cake protocol.
#[derive(Readable, Writable)]
pub struct RawTensor {
    /// Tensor data.
    pub data: Vec<u8>,
    /// The data type as a compact u8 tag (see dtype_to_u8 / u8_to_dtype).
    pub dtype: u8,
    /// The tensor shape.
    pub shape: Vec<usize>,
}

impl std::fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl RawTensor {
    /// Convert x into a RawTensor.
    pub fn from_tensor(x: &Tensor) -> Self {
        let data: Vec<u8> = x.data().into_owned();
        let dtype = dtype_to_u8(x.dtype());
        let shape = x.shape().dims().to_vec();
        Self { data, dtype, shape }
    }

    /// Convert the raw tensor in a Tensor allocated on the given device.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let dtype = u8_to_dtype(self.dtype)?;
        Ok(Tensor::from_raw_buffer(
            &self.data,
            dtype,
            &self.shape,
            device,
        )?)
    }
}

/// Diagnostic information about a worker.
#[derive(Debug, Default, Readable, Writable)]
pub struct WorkerInfo {
    /// Protocol version.
    pub version: String,
    /// Tensors data type.
    pub dtype: String,
    /// Operating system.
    pub os: String,
    /// Architecture.
    pub arch: String,
    /// Device.
    pub device: String,
    /// Device index for multi GPU environments.
    pub device_idx: usize,
    /// Latency in millisenconds.
    pub latency: u128,
}

/// A Cake protocol message.
#[derive(Debug, Readable, Writable)]
pub enum Message {
    /// First message sent.
    Hello,
    /// Message that the worker sends when a master connects with runtime information.
    WorkerInfo(WorkerInfo),
    /// Single inference operation for a given layer.
    SingleOp {
        layer_name: String,
        x: RawTensor,
        index_pos: usize,
        block_idx: usize,
    },
    /// Batched inference operations over a Tensor.
    Batch {
        x: RawTensor,
        batch: Vec<(String, usize, usize)>,
    },
    /// A message to transmit tensors.
    Tensor(RawTensor),
    /// Last message sent.
    Goodbye,

    // ── Zero-config setup messages ──────────────────────────────

    /// Master tells worker which layers to serve.
    LayerAssignment {
        layers: Vec<String>,
        /// Short hash of model config for cache keying.
        model_hash: String,
    },
    /// Worker tells master whether it needs model data.
    LayerAssignmentAck { needs_data: bool },
    /// Chunk of model file data from master to worker.
    ModelDataChunk {
        filename: String,
        offset: u64,
        total_size: u64,
        /// Whether `data` is zstd-compressed.
        compressed: bool,
        /// CRC32 checksum of `data` (after compression if compressed).
        checksum: u32,
        data: Vec<u8>,
    },
    /// All model files have been sent.
    ModelDataDone,
    /// Worker requests resumption from a partial transfer.
    ModelDataResume {
        filename: String,
        /// Byte offset to resume from (worker's current file size).
        offset: u64,
    },
    /// Worker has loaded all assigned layers and is ready for inference.
    WorkerReady,
    /// Worker encountered an error during inference.
    WorkerError { message: String },
}

impl Message {
    /// Create a Message::SingleOp message.
    pub fn single_op(layer_name: &str, x: &Tensor, index_pos: usize, block_idx: usize) -> Self {
        let layer_name = layer_name.to_owned();
        let x = RawTensor::from_tensor(x);
        Self::SingleOp {
            layer_name,
            x,
            index_pos,
            block_idx,
        }
    }

    /// Create a Message::Tensor message.
    pub fn from_tensor(x: &Tensor) -> Self {
        Self::Tensor(RawTensor::from_tensor(x))
    }

    /// Create a Message::Batch message.
    pub fn from_batch(x: &Tensor, batch: Vec<(String, usize, usize)>) -> Self {
        Self::Batch {
            x: RawTensor::from_tensor(x),
            batch,
        }
    }

    // Yes, I could use GRPC, but this is simpler and faster.
    // Check speedy benchmarks ;)

    /// Serializes the message to raw bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.write_to_vec_with_ctx(BigEndian::default())?)
    }

    /// Deserializes a Message from raw bytes.
    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        Ok(Self::read_from_buffer_with_ctx(BigEndian::default(), raw)?)
    }

    /// Read a Message with the provided reader.
    pub async fn from_reader<R>(reader: &mut R) -> Result<(usize, Self)>
    where
        R: AsyncReadExt + Unpin,
    {
        let mut buf = Vec::new();
        Self::from_reader_buf(reader, &mut buf).await
    }

    /// Read a Message, reusing `buf` to avoid per-message heap allocation.
    pub async fn from_reader_buf<R>(reader: &mut R, buf: &mut Vec<u8>) -> Result<(usize, Self)>
    where
        R: AsyncReadExt + Unpin,
    {
        // read_u32() reads 4 bytes as big-endian and returns the native value.
        let magic = reader.read_u32().await?;
        if magic != super::PROTO_MAGIC {
            return Err(anyhow!("invalid magic value: {magic}"));
        }

        let req_size = reader.read_u32().await?;
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }

        let req_size = req_size as usize;
        buf.resize(req_size, 0);
        reader.read_exact(&mut buf[..req_size]).await?;

        Ok((req_size, Self::from_bytes(&buf[..req_size])?))
    }

    /// Write a Message with the provided writer.
    pub async fn to_writer<W>(&self, writer: &mut W) -> Result<usize>
    where
        W: AsyncWriteExt + Unpin,
    {
        let mut buf = Vec::new();
        self.to_writer_buf(writer, &mut buf).await
    }

    /// Write a Message, reusing `buf` to avoid per-message heap allocation.
    pub async fn to_writer_buf<W>(&self, writer: &mut W, buf: &mut Vec<u8>) -> Result<usize>
    where
        W: AsyncWriteExt + Unpin,
    {
        // Reserve 8 bytes for the header (magic + length), then serialize the
        // message directly into `buf` via speedy's stream writer.  This avoids
        // the intermediate `Vec<u8>` that `write_to_vec_with_ctx` would create.
        buf.truncate(0); // preserve capacity across calls
        buf.resize(8, 0); // placeholder for header (no-op if capacity >= 8)
        self.write_to_stream_with_ctx(BigEndian::default(), &mut *buf)?;

        let payload_size = (buf.len() - 8) as u32;
        if payload_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {payload_size} > MESSAGE_MAX_SIZE"));
        }

        // Fill in the header in-place.
        buf[0..4].copy_from_slice(&super::PROTO_MAGIC.to_be_bytes());
        buf[4..8].copy_from_slice(&payload_size.to_be_bytes());

        writer.write_all(buf).await?;
        Ok(buf.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_f32_tensor(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn make_f16_tensor(shape: &[usize]) -> Tensor {
        make_f32_tensor(shape).to_dtype(DType::F16).unwrap()
    }

    // ── RawTensor round-trips ──────────────────────────────────

    #[test]
    fn test_raw_tensor_roundtrip_f32() {
        let original = make_f32_tensor(&[2, 64]);
        let raw = RawTensor::from_tensor(&original);
        assert_eq!(raw.dtype, dtype_to_u8(DType::F32));
        assert_eq!(raw.shape, vec![2, 64]);
        assert_eq!(raw.data.len(), 2 * 64 * 4);

        let recovered = raw.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(recovered.dtype(), DType::F32);
        assert_eq!(recovered.shape().dims(), &[2, 64]);
    }

    #[test]
    fn test_raw_tensor_roundtrip_f16() {
        let original = make_f16_tensor(&[1, 128]);
        let raw = RawTensor::from_tensor(&original);
        assert_eq!(raw.dtype, dtype_to_u8(DType::F16));
        assert_eq!(raw.shape, vec![1, 128]);
        assert_eq!(raw.data.len(), 128 * 2);

        let recovered = raw.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(recovered.dtype(), DType::F16);
        assert_eq!(recovered.shape().dims(), &[1, 128]);
    }

    #[test]
    fn test_raw_tensor_data_integrity() {
        let original = make_f16_tensor(&[1, 256]);
        let orig_bytes: Vec<u8> = original.data().to_vec();
        let raw = RawTensor::from_tensor(&original);
        let recovered = raw.to_tensor(&Device::Cpu).unwrap();
        let recv_bytes: Vec<u8> = recovered.data().to_vec();
        assert_eq!(orig_bytes, recv_bytes);
    }

    // ── Message serialization (to_bytes / from_bytes) ──────────

    #[test]
    fn test_message_hello_roundtrip() {
        let bytes = Message::Hello.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        assert!(matches!(decoded, Message::Hello));
    }

    #[test]
    fn test_message_goodbye_roundtrip() {
        let bytes = Message::Goodbye.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        assert!(matches!(decoded, Message::Goodbye));
    }

    #[test]
    fn test_message_worker_info_roundtrip() {
        let info = WorkerInfo {
            version: "0.1.0".into(),
            dtype: "F16".into(),
            os: "linux".into(),
            arch: "x86_64".into(),
            device: "cuda".into(),
            device_idx: 2,
            latency: 42,
        };
        let bytes = Message::WorkerInfo(info).to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::WorkerInfo(wi) => {
                assert_eq!(wi.version, "0.1.0");
                assert_eq!(wi.dtype, "F16");
                assert_eq!(wi.os, "linux");
                assert_eq!(wi.arch, "x86_64");
                assert_eq!(wi.device, "cuda");
                assert_eq!(wi.device_idx, 2);
                assert_eq!(wi.latency, 42);
            }
            other => panic!("expected WorkerInfo, got {:?}", other),
        }
    }

    #[test]
    fn test_message_tensor_roundtrip() {
        let tensor = make_f16_tensor(&[1, 64]);
        let bytes = Message::from_tensor(&tensor).to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::Tensor(raw) => {
                let t = raw.to_tensor(&Device::Cpu).unwrap();
                assert_eq!(t.dtype(), DType::F16);
                assert_eq!(t.shape().dims(), &[1, 64]);
            }
            other => panic!("expected Tensor, got {:?}", other),
        }
    }

    #[test]
    fn test_message_single_op_roundtrip() {
        let tensor = make_f16_tensor(&[1, 64]);
        let msg = Message::single_op("model.layers.5", &tensor, 42, 7);
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::SingleOp {
                layer_name,
                x,
                index_pos,
                block_idx,
            } => {
                assert_eq!(layer_name, "model.layers.5");
                assert_eq!(index_pos, 42);
                assert_eq!(block_idx, 7);
                let t = x.to_tensor(&Device::Cpu).unwrap();
                assert_eq!(t.shape().dims(), &[1, 64]);
            }
            other => panic!("expected SingleOp, got {:?}", other),
        }
    }

    #[test]
    fn test_message_batch_roundtrip() {
        let tensor = make_f16_tensor(&[1, 128]);
        let batch = vec![
            ("model.layers.0".into(), 0usize, 0usize),
            ("model.layers.1".into(), 1, 1),
            ("model.layers.2".into(), 2, 2),
        ];
        let msg = Message::from_batch(&tensor, batch);
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::Batch { x, batch } => {
                assert_eq!(batch.len(), 3);
                assert_eq!(batch[0].0, "model.layers.0");
                assert_eq!(batch[1].1, 1);
                assert_eq!(batch[2].2, 2);
                let t = x.to_tensor(&Device::Cpu).unwrap();
                assert_eq!(t.shape().dims(), &[1, 128]);
            }
            other => panic!("expected Batch, got {:?}", other),
        }
    }

    #[test]
    fn test_message_worker_error_roundtrip() {
        let msg = Message::WorkerError {
            message: "layer not found".into(),
        };
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::WorkerError { message } => {
                assert_eq!(message, "layer not found");
            }
            other => panic!("expected WorkerError, got {:?}", other),
        }
    }

    // ── Wire format (to_writer / from_reader) ──────────────────

    #[tokio::test]
    async fn test_wire_hello() {
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        let written = Message::Hello.to_writer(&mut writer).await.unwrap();
        drop(writer);

        let (payload_size, decoded) = Message::from_reader(&mut reader).await.unwrap();
        assert!(matches!(decoded, Message::Hello));
        assert_eq!(written, 8 + payload_size);
    }

    #[tokio::test]
    async fn test_wire_tensor() {
        let (mut writer, mut reader) = tokio::io::duplex(64 * 1024);
        let tensor = make_f16_tensor(&[1, 128]);
        let orig_bytes: Vec<u8> = tensor.data().to_vec();

        Message::from_tensor(&tensor)
            .to_writer(&mut writer)
            .await
            .unwrap();
        drop(writer);

        let (_, decoded) = Message::from_reader(&mut reader).await.unwrap();
        match decoded {
            Message::Tensor(raw) => {
                assert_eq!(raw.dtype, dtype_to_u8(DType::F16));
                assert_eq!(raw.shape, vec![1, 128]);
                let t = raw.to_tensor(&Device::Cpu).unwrap();
                assert_eq!(t.data().to_vec(), orig_bytes);
            }
            other => panic!("expected Tensor, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_wire_invalid_magic() {
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        writer.write_u32(0xDEADBEEF_u32).await.unwrap();
        writer.write_u32(4_u32).await.unwrap();
        writer.write_all(&[0, 0, 0, 0]).await.unwrap();
        drop(writer);

        let result = Message::from_reader(&mut reader).await;
        assert!(result.is_err());
        assert!(
            result.unwrap_err().to_string().contains("invalid magic"),
            "should report invalid magic"
        );
    }

    #[tokio::test]
    async fn test_wire_oversized_message() {
        let (mut writer, mut reader) = tokio::io::duplex(1024);
        // Write valid magic, then size > MESSAGE_MAX_SIZE
        // write_u32 already writes in big-endian, so pass native values directly.
        writer
            .write_u32(crate::cake::sharding::proto::PROTO_MAGIC)
            .await
            .unwrap();
        writer
            .write_u32(crate::cake::sharding::proto::MESSAGE_MAX_SIZE + 1)
            .await
            .unwrap();
        drop(writer);

        let result = Message::from_reader(&mut reader).await;
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("MESSAGE_MAX_SIZE"),
            "should report oversized message"
        );
    }

    // ── dtype round-trips ─────────────────────────────────────────

    #[test]
    fn test_dtype_roundtrip_all() {
        let dtypes = [
            DType::U8,
            DType::U32,
            DType::I64,
            DType::BF16,
            DType::F16,
            DType::F32,
            DType::F64,
            DType::F8E4M3,
        ];
        for dt in dtypes {
            let tag = dtype_to_u8(dt);
            let recovered = u8_to_dtype(tag).unwrap();
            assert_eq!(recovered, dt, "round-trip failed for {:?} (tag={})", dt, tag);
        }
    }

    #[test]
    fn test_dtype_tags_unique() {
        let tags: Vec<u8> = [
            DType::U8, DType::U32, DType::I64, DType::BF16,
            DType::F16, DType::F32, DType::F64, DType::F8E4M3,
        ].iter().map(|d| dtype_to_u8(*d)).collect();
        let mut deduped = tags.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(tags.len(), deduped.len(), "dtype tags must be unique");
    }

    #[test]
    fn test_dtype_unknown_tag() {
        assert!(u8_to_dtype(255).is_err());
        assert!(u8_to_dtype(100).is_err());
    }

    // ── RawTensor with BF16 ─────────────────────────────────────

    #[test]
    fn test_raw_tensor_roundtrip_bf16() {
        let original = make_f32_tensor(&[2, 32]).to_dtype(DType::BF16).unwrap();
        let raw = RawTensor::from_tensor(&original);
        assert_eq!(raw.dtype, dtype_to_u8(DType::BF16));
        assert_eq!(raw.shape, vec![2, 32]);
        assert_eq!(raw.data.len(), 2 * 32 * 2); // BF16 = 2 bytes

        let recovered = raw.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(recovered.dtype(), DType::BF16);
        assert_eq!(recovered.shape().dims(), &[2, 32]);
        // Verify data integrity
        assert_eq!(original.data().to_vec(), recovered.data().to_vec());
    }

    // ── RawTensor with empty data ───────────────────────────────

    #[test]
    fn test_raw_tensor_empty() {
        let original = Tensor::zeros((0, 64), DType::F32, &Device::Cpu).unwrap();
        let raw = RawTensor::from_tensor(&original);
        assert_eq!(raw.dtype, dtype_to_u8(DType::F32));
        assert_eq!(raw.shape, vec![0, 64]);
        assert_eq!(raw.data.len(), 0);

        let recovered = raw.to_tensor(&Device::Cpu).unwrap();
        assert_eq!(recovered.shape().dims(), &[0, 64]);
    }

    // ── Message::LayerAssignment round-trip ──────────────────────

    #[test]
    fn test_message_layer_assignment_roundtrip() {
        let msg = Message::LayerAssignment {
            layers: vec![
                "model.layers.0".into(),
                "model.layers.1".into(),
                "model.layers.2".into(),
            ],
            model_hash: "abc12345".into(),
        };
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::LayerAssignment { layers, model_hash } => {
                assert_eq!(layers.len(), 3);
                assert_eq!(layers[0], "model.layers.0");
                assert_eq!(layers[2], "model.layers.2");
                assert_eq!(model_hash, "abc12345");
            }
            other => panic!("expected LayerAssignment, got {:?}", other),
        }
    }

    #[test]
    fn test_message_layer_assignment_empty_layers() {
        let msg = Message::LayerAssignment {
            layers: vec![],
            model_hash: "".into(),
        };
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::LayerAssignment { layers, model_hash } => {
                assert!(layers.is_empty());
                assert!(model_hash.is_empty());
            }
            other => panic!("expected LayerAssignment, got {:?}", other),
        }
    }

    // ── Message::ModelDataChunk round-trip ───────────────────────

    #[test]
    fn test_message_model_data_chunk_roundtrip() {
        let data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0x42];
        let checksum = crc32fast::hash(&data);
        let msg = Message::ModelDataChunk {
            filename: "model-00001-of-00003.safetensors".into(),
            offset: 1024,
            total_size: 4_000_000_000,
            compressed: false,
            checksum,
            data: data.clone(),
        };
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::ModelDataChunk { filename, offset, total_size, compressed, checksum: cs, data: d } => {
                assert_eq!(filename, "model-00001-of-00003.safetensors");
                assert_eq!(offset, 1024);
                assert_eq!(total_size, 4_000_000_000);
                assert!(!compressed);
                assert_eq!(cs, checksum);
                assert_eq!(d, data);
            }
            other => panic!("expected ModelDataChunk, got {:?}", other),
        }
    }

    #[test]
    fn test_message_model_data_chunk_compressed_roundtrip() {
        let data = vec![0x42; 1000]; // compressible data
        let compressed_data = zstd::encode_all(data.as_slice(), 1).unwrap();
        let checksum = crc32fast::hash(&compressed_data);
        let msg = Message::ModelDataChunk {
            filename: "shard.safetensors".into(),
            offset: 0,
            total_size: 5000,
            compressed: true,
            checksum,
            data: compressed_data.clone(),
        };
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::ModelDataChunk { compressed, checksum: cs, data: d, .. } => {
                assert!(compressed);
                assert_eq!(cs, checksum);
                // Verify decompression works
                let decompressed = zstd::decode_all(d.as_slice()).unwrap();
                assert_eq!(decompressed, data);
            }
            other => panic!("expected ModelDataChunk, got {:?}", other),
        }
    }

    #[test]
    fn test_message_model_data_resume_roundtrip() {
        let msg = Message::ModelDataResume {
            filename: "model-00002-of-00003.safetensors".into(),
            offset: 256 * 1024 * 1024,
        };
        let bytes = msg.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        match decoded {
            Message::ModelDataResume { filename, offset } => {
                assert_eq!(filename, "model-00002-of-00003.safetensors");
                assert_eq!(offset, 256 * 1024 * 1024);
            }
            other => panic!("expected ModelDataResume, got {:?}", other),
        }
    }

    #[test]
    fn test_crc32_detects_corruption() {
        let data = vec![1, 2, 3, 4, 5];
        let checksum = crc32fast::hash(&data);
        let mut corrupted = data.clone();
        corrupted[2] = 99;
        assert_ne!(crc32fast::hash(&corrupted), checksum);
    }

    // ── Message::WorkerReady round-trip ──────────────────────────

    #[test]
    fn test_message_worker_ready_roundtrip() {
        let bytes = Message::WorkerReady.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        assert!(matches!(decoded, Message::WorkerReady));
    }

    // ── Message::ModelDataDone round-trip ────────────────────────

    #[test]
    fn test_message_model_data_done_roundtrip() {
        let bytes = Message::ModelDataDone.to_bytes().unwrap();
        let decoded = Message::from_bytes(&bytes).unwrap();
        assert!(matches!(decoded, Message::ModelDataDone));
    }

    // ── Message::LayerAssignmentAck round-trip ───────────────────

    #[test]
    fn test_message_layer_assignment_ack_roundtrip() {
        for needs in [true, false] {
            let msg = Message::LayerAssignmentAck { needs_data: needs };
            let bytes = msg.to_bytes().unwrap();
            let decoded = Message::from_bytes(&bytes).unwrap();
            match decoded {
                Message::LayerAssignmentAck { needs_data } => {
                    assert_eq!(needs_data, needs);
                }
                other => panic!("expected LayerAssignmentAck, got {:?}", other),
            }
        }
    }

    #[tokio::test]
    async fn test_wire_multiple_messages() {
        let (mut writer, mut reader) = tokio::io::duplex(64 * 1024);
        let tensor = make_f16_tensor(&[1, 32]);

        Message::Hello.to_writer(&mut writer).await.unwrap();
        Message::single_op("model.layers.0", &tensor, 0, 0)
            .to_writer(&mut writer)
            .await
            .unwrap();
        Message::from_tensor(&tensor)
            .to_writer(&mut writer)
            .await
            .unwrap();
        Message::Goodbye.to_writer(&mut writer).await.unwrap();
        drop(writer);

        let (_, m1) = Message::from_reader(&mut reader).await.unwrap();
        assert!(matches!(m1, Message::Hello));

        let (_, m2) = Message::from_reader(&mut reader).await.unwrap();
        assert!(matches!(m2, Message::SingleOp { .. }));

        let (_, m3) = Message::from_reader(&mut reader).await.unwrap();
        assert!(matches!(m3, Message::Tensor(_)));

        let (_, m4) = Message::from_reader(&mut reader).await.unwrap();
        assert!(matches!(m4, Message::Goodbye));
    }

    // ── to_writer_buf (hot path with buffer reuse) ──────────────

    #[tokio::test]
    async fn test_writer_buf_roundtrip() {
        let (mut writer, mut reader) = tokio::io::duplex(64 * 1024);
        let mut write_buf = Vec::new();
        let mut read_buf = Vec::new();
        let tensor = make_f16_tensor(&[1, 1024]);

        Message::from_tensor(&tensor)
            .to_writer_buf(&mut writer, &mut write_buf)
            .await
            .unwrap();
        drop(writer);

        let (_, decoded) = Message::from_reader_buf(&mut reader, &mut read_buf).await.unwrap();
        assert!(matches!(decoded, Message::Tensor(_)));
    }

    #[tokio::test]
    async fn test_writer_buf_preserves_capacity() {
        let (mut writer, _reader) = tokio::io::duplex(64 * 1024);
        let mut buf = Vec::new();
        let tensor = make_f16_tensor(&[1, 512]);

        // First write allocates
        Message::from_tensor(&tensor)
            .to_writer_buf(&mut writer, &mut buf)
            .await
            .unwrap();
        let cap_after_first = buf.capacity();

        // Second write should reuse the same allocation
        Message::from_tensor(&tensor)
            .to_writer_buf(&mut writer, &mut buf)
            .await
            .unwrap();
        let cap_after_second = buf.capacity();

        assert_eq!(cap_after_first, cap_after_second, "buffer should reuse capacity");
    }

    #[tokio::test]
    async fn test_writer_buf_large_tensor() {
        let (mut writer, mut reader) = tokio::io::duplex(512 * 1024);
        let mut write_buf = Vec::new();
        let mut read_buf = Vec::new();
        // 1 × 16384 F16 = 32 KB tensor (realistic inference size)
        let tensor = make_f16_tensor(&[1, 16384]);
        let orig_bytes: Vec<u8> = tensor.data().to_vec();

        Message::from_tensor(&tensor)
            .to_writer_buf(&mut writer, &mut write_buf)
            .await
            .unwrap();
        drop(writer);

        let (_, decoded) = Message::from_reader_buf(&mut reader, &mut read_buf).await.unwrap();
        match decoded {
            Message::Tensor(raw) => {
                assert_eq!(raw.data.len(), 16384 * 2);
                let t = raw.to_tensor(&Device::Cpu).unwrap();
                assert_eq!(t.data().to_vec(), orig_bytes);
            }
            other => panic!("expected Tensor, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_writer_buf_batch_many_layers() {
        let (mut writer, mut reader) = tokio::io::duplex(64 * 1024);
        let mut write_buf = Vec::new();
        let mut read_buf = Vec::new();
        let tensor = make_f16_tensor(&[1, 1024]);
        let batch: Vec<(String, usize, usize)> = (0..12)
            .map(|i| (format!("model.layers.{i}"), i, i))
            .collect();
        let msg = Message::from_batch(&tensor, batch);

        msg.to_writer_buf(&mut writer, &mut write_buf).await.unwrap();
        drop(writer);

        let (_, decoded) = Message::from_reader_buf(&mut reader, &mut read_buf).await.unwrap();
        match decoded {
            Message::Batch { x, batch } => {
                assert_eq!(batch.len(), 12);
                assert_eq!(batch[11].0, "model.layers.11");
                let t = x.to_tensor(&Device::Cpu).unwrap();
                assert_eq!(t.shape().dims(), &[1, 1024]);
            }
            other => panic!("expected Batch, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_writer_buf_single_op_roundtrip() {
        let (mut writer, mut reader) = tokio::io::duplex(64 * 1024);
        let mut write_buf = Vec::new();
        let mut read_buf = Vec::new();
        let tensor = make_f16_tensor(&[1, 1024]);

        Message::single_op("model.layers.5", &tensor, 42, 7)
            .to_writer_buf(&mut writer, &mut write_buf)
            .await
            .unwrap();
        drop(writer);

        let (_, decoded) = Message::from_reader_buf(&mut reader, &mut read_buf).await.unwrap();
        match decoded {
            Message::SingleOp { layer_name, index_pos, block_idx, .. } => {
                assert_eq!(layer_name, "model.layers.5");
                assert_eq!(index_pos, 42);
                assert_eq!(block_idx, 7);
            }
            other => panic!("expected SingleOp, got {:?}", other),
        }
    }

}
