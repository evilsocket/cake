use std::str::FromStr;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use safetensors::View;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Serialize, Debug, Deserialize)]
pub struct RawTensor {
    pub data: Vec<u8>,
    pub dtype: String,
    pub shape: Vec<usize>,
}

impl RawTensor {
    pub fn from_tensor(x: &Tensor) -> Self {
        let data: Vec<u8> = x.data().to_vec();
        let dtype = x.dtype().as_str().to_string();
        let shape = x.shape().clone().into_dims();
        Self { data, dtype, shape }
    }

    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let dtype = DType::from_str(&self.dtype)?;
        Tensor::from_raw_buffer(&self.data, dtype, &self.shape, device).map_err(|e| anyhow!(e))
    }
}

#[derive(Serialize, Debug, Default, Deserialize)]
pub struct WorkerInfo {
    pub version: String,
    pub dtype: String,
    pub os: String,
    pub arch: String,
    pub device: String,
    pub device_idx: usize,
    pub latency: u128,
}

#[derive(Serialize, Debug, Deserialize)]
pub enum Message {
    Hello,
    WorkerInfo(WorkerInfo),
    SingleOp {
        layer_name: String,
        x: RawTensor,
        index_pos: usize,
        block_idx: usize,
    },
    Batch {
        x: RawTensor,
        batch: Vec<(String, usize, usize)>,
    },
    Tensor(RawTensor),
}

impl Message {
    pub fn transformer_op(
        layer_name: &str,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
    ) -> Self {
        let layer_name = layer_name.to_owned();
        let x = RawTensor::from_tensor(x);
        Self::SingleOp {
            layer_name,
            x,
            index_pos,
            block_idx,
        }
    }

    pub fn from_tensor(x: &Tensor) -> Self {
        Self::Tensor(RawTensor::from_tensor(x))
    }

    pub fn from_batch(x: &Tensor, batch: Vec<(String, usize, usize)>) -> Self {
        Self::Batch {
            x: RawTensor::from_tensor(x),
            batch,
        }
    }

    // Yes, I could use GRPC, but this is simpler and faster.
    // Check bitcode benchmarks ;)
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bitcode::serialize(self).map_err(|e| anyhow!(e))
    }

    fn from_bytes(raw: &[u8]) -> Result<Self> {
        bitcode::deserialize(raw).map_err(|e| anyhow!(e))
    }

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
