use anyhow::Result;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use tokio::net::TcpStream;

use crate::model::Cache;

use super::{Message, WorkerInfo};

#[derive(Debug)]
pub struct Client {
    device: Device,
    address: String,
    layer_name: String,
    stream: TcpStream,
    info: WorkerInfo,
}

impl Client {
    pub async fn new(device: Device, address: &str, layer_name: &str) -> Result<Self> {
        let address = address.to_string();
        let layer_name = layer_name.to_string();
        let stream = TcpStream::connect(&address)
            .await
            .map_err(|e| anyhow!("can't connect to {address}: {e}"))?;
        let worker_info = WorkerInfo::default();

        let mut client = Self {
            address,
            device,
            stream,
            layer_name,
            info: worker_info,
        };

        let resp = client.request(Message::Hello).await?;
        client.info = if let Message::WorkerInfo(info) = resp {
            info
        } else {
            return Err(anyhow!("unexpected worker info message: {:?}", &resp));
        };

        Ok(client)
    }

    async fn request(&mut self, req: Message) -> Result<Message> {
        req.to_writer(&mut self.stream)
            .await
            .map_err(|e| anyhow!("error sending message {:?}: {}", req, e))?;

        let (_, msg) = super::Message::from_reader(&mut self.stream)
            .await
            .map_err(|e| anyhow!("error receiving response for {:?}: {}", req, e))?;
        Ok(msg)
    }

    async fn forward_request(&mut self, req: Message) -> Result<Tensor> {
        let resp = self.request(req).await?;
        match resp {
            Message::Tensor(raw) => Ok(raw.to_tensor(&self.device)?),
            _ => Err(anyhow!("unexpected response {:?}", &resp)),
        }
    }
}

impl std::fmt::Display for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}@{} [{}<{}> {}-{} latency={}ms]",
            &self.layer_name,
            &self.address,
            &self.info.device,
            &self.info.device_idx,
            &self.info.os,
            &self.info.arch,
            self.info.latency
        )
    }
}

#[async_trait]
impl super::Forwarder for Client {
    fn load(_: String, _: candle_nn::VarBuilder, _: &crate::model::Config) -> Result<Box<Self>> {
        Err(anyhow!("load should never be called on cake::Client"))
    }

    async fn forward(&self, _: &Tensor, _: usize, _: usize, _: &mut Cache) -> Result<Tensor> {
        Err(anyhow!(
            "immutable forward should never be called on cake::Client"
        ))
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        _: &mut Cache,
    ) -> Result<Tensor> {
        self.forward_request(super::Message::transformer_op(
            &self.layer_name,
            x,
            index_pos,
            block_idx,
        ))
        .await
    }

    async fn forward_batch(
        &mut self,
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        _: &mut Cache,
    ) -> Result<Tensor> {
        self.forward_request(super::Message::from_batch(x, batch))
            .await
    }

    fn ident(&self) -> &str {
        &self.address
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }
}
