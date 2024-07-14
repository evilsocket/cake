use std::str::FromStr;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use safetensors::View;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Represents a tensor in Cake protocol.
#[derive(Serialize, Debug, Deserialize)]
pub struct RawTensor {
    /// Tensor data.
    pub data: Vec<u8>,
    /// The data type as string.
    pub dtype: String,
    /// The tensor shape.
    pub shape: Vec<usize>,
}

impl RawTensor {
    /// Convert x into a RawTensor.
    pub fn from_tensor(x: &Tensor) -> Self {
        let data: Vec<u8> = x.data().to_vec();
        let dtype = x.dtype().as_str().to_string();
        let shape = x.shape().clone().into_dims();
        Self { data, dtype, shape }
    }

    /// Convert the raw tensor in a Tensor allocated on the given device.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let dtype = DType::from_str(&self.dtype)?;
        Tensor::from_raw_buffer(&self.data, dtype, &self.shape, device).map_err(|e| anyhow!(e))
    }
}

/// Diagnostic information about a worker.
#[derive(Serialize, Debug, Default, Deserialize)]
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
#[derive(Serialize, Debug, Deserialize)]
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
    // Check bitcode benchmarks ;)

    /// Serializes the message to raw bytes.
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bitcode::serialize(self).map_err(|e| anyhow!(e))
    }

    /// Deserializes a Message from raw bytes.
    fn from_bytes(raw: &[u8]) -> Result<Self> {
        bitcode::deserialize(raw).map_err(|e| anyhow!(e))
    }

    /// Read a Message with the provided reader.
    pub async fn from_reader<R>(reader: &mut R) -> Result<(usize, Self)>
    where
        R: AsyncReadExt + Unpin,
    {
        let magic = reader.read_u32().await?;
        if magic != super::PROTO_MAGIC {
            return Err(anyhow!("invalid magic value: {magic}"));
        }

        let req_size = reader.read_u32().await?;
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }

        let mut req = vec![0_u8; req_size as usize];

        reader.read_exact(&mut req).await?;

        Ok((req.len(), Self::from_bytes(&req)?))
    }

    /// Write a Message with the provided writer.
    pub async fn to_writer<W>(&self, writer: &mut W) -> Result<usize>
    where
        W: AsyncWriteExt + Unpin,
    {
        let req = self.to_bytes()?;
        let req_size = req.len() as u32;
        if req_size > super::MESSAGE_MAX_SIZE {
            return Err(anyhow!("request size {req_size} > MESSAGE_MAX_SIZE"));
        }

        writer.write_u32(super::PROTO_MAGIC).await?;
        writer.write_u32(req_size).await?;
        writer.write_all(&req).await?;

        Ok(8 + req.len())
    }
}
