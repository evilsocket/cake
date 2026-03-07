use std::collections::HashMap;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use tokio::net::TcpStream;

use super::{Context, Message, WorkerInfo};

/// TCP connect timeout.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Maximum number of connection attempts before giving up.
const MAX_CONNECT_RETRIES: u32 = 3;
/// Base delay between retries (doubles each attempt).
const RETRY_BASE_DELAY: Duration = Duration::from_secs(1);

/// Lightweight stub for non-primary remote layer slots.
///
/// When multiple layers map to the same worker, only the first gets a real
/// `Client` (TCP connection). The rest get a `RemoteRef` that returns the
/// same `ident()` so the batching logic groups them correctly, but holds
/// no connection. Its `forward_*` methods are never called directly.
#[derive(Debug)]
pub struct RemoteRef {
    address: String,
    layer_name: String,
}

impl RemoteRef {
    pub fn new(address: &str, layer_name: &str) -> Self {
        Self {
            address: address.to_string(),
            layer_name: layer_name.to_string(),
        }
    }
}

impl std::fmt::Display for RemoteRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}@{} [ref]", &self.layer_name, &self.address)
    }
}

#[async_trait]
impl super::Forwarder for RemoteRef {
    fn load(_: String, _: &Context) -> Result<Box<Self>> {
        Err(anyhow!("load should never be called on RemoteRef"))
    }

    async fn forward(&self, _: &Tensor, _: usize, _: usize, _: &mut Context) -> Result<Tensor> {
        Err(anyhow!("forward should never be called on RemoteRef (batching uses the primary Client)"))
    }

    async fn forward_mut(&mut self, _: &Tensor, _: usize, _: usize, _: &mut Context) -> Result<Tensor> {
        Err(anyhow!("forward_mut should never be called on RemoteRef (batching uses the primary Client)"))
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn ident(&self) -> &str {
        &self.address
    }
}

/// Connect to remote workers, deduplicating by host address.
///
/// Returns a map of layer_index → Box<dyn Forwarder>. The first layer for
/// each worker gets a real `Client`; subsequent layers get a `RemoteRef`.
pub async fn connect_remote_layers(
    remote_layers: &[(usize, String, String)], // (index, layer_name, host)
    device: &Device,
    cluster_key: Option<&str>,
) -> Result<HashMap<usize, Box<dyn super::Forwarder>>> {
    let mut result: HashMap<usize, Box<dyn super::Forwarder>> = HashMap::new();
    let mut connected_hosts: HashMap<String, usize> = HashMap::new(); // host → first layer index

    for (idx, layer_name, host) in remote_layers {
        if connected_hosts.contains_key(host) {
            log::info!("  {} → {} [shared connection]", layer_name, host);
            result.insert(*idx, Box::new(RemoteRef::new(host, layer_name)));
        } else {
            log::info!("connecting {} to {} ...", layer_name, host);
            let client = Client::new(device.clone(), host, layer_name, cluster_key).await?;
            connected_hosts.insert(host.clone(), *idx);
            result.insert(*idx, Box::new(client));
        }
    }

    Ok(result)
}

/// A client object used by the master to connect and orchestrate the workers.
/// From the Cake perspective, each worker is a server and the master uses
/// multiple Client instances to connect to them.
#[derive(Debug)]
pub struct Client {
    device: Device,
    address: String,
    layer_name: String,
    stream: TcpStream,
    info: WorkerInfo,
    read_buf: Vec<u8>,
    write_buf: Vec<u8>,
}

impl Client {
    /// Connects to the given worker address.
    /// NOTE: device and layer_name here are only passed for std::fmt::Display.
    /// If `cluster_key` is provided, mutual PSK authentication is performed
    /// before any protocol messages.
    pub async fn new(
        device: Device,
        address: &str,
        layer_name: &str,
        cluster_key: Option<&str>,
    ) -> Result<Self> {
        let address = address.to_string();
        let layer_name = layer_name.to_string();

        let mut last_err = None;
        let mut stream_opt = None;
        for attempt in 0..MAX_CONNECT_RETRIES {
            match tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(&address)).await {
                Ok(Ok(s)) => {
                    stream_opt = Some(s);
                    break;
                }
                Ok(Err(e)) => {
                    last_err = Some(format!("{e}"));
                    if attempt + 1 < MAX_CONNECT_RETRIES {
                        let delay = RETRY_BASE_DELAY * 2u32.pow(attempt);
                        log::warn!(
                            "connection to {} failed (attempt {}/{}): {} — retrying in {:?}",
                            &address, attempt + 1, MAX_CONNECT_RETRIES, e, delay
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
                Err(_) => {
                    last_err = Some("connection timed out".to_string());
                    if attempt + 1 < MAX_CONNECT_RETRIES {
                        let delay = RETRY_BASE_DELAY * 2u32.pow(attempt);
                        log::warn!(
                            "connection to {} timed out (attempt {}/{}) — retrying in {:?}",
                            &address, attempt + 1, MAX_CONNECT_RETRIES, delay
                        );
                        tokio::time::sleep(delay).await;
                    }
                }
            }
        }
        let stream = stream_opt.ok_or_else(|| {
            anyhow!(
                "can't connect to {} after {} attempts: {}",
                &address,
                MAX_CONNECT_RETRIES,
                last_err.unwrap_or_default()
            )
        })?;
        stream.set_nodelay(true)?;
        let worker_info = WorkerInfo::default();

        let mut client = Self {
            address,
            device,
            stream,
            layer_name,
            info: worker_info,
            read_buf: Vec::new(),
            write_buf: Vec::new(),
        };

        // Authenticate if cluster key is set
        if let Some(key) = cluster_key {
            super::auth::authenticate_as_master(&mut client.stream, key).await?;
        }

        let resp = client.request(Message::Hello).await?;
        client.info = if let Message::WorkerInfo(info) = resp {
            info
        } else {
            return Err(anyhow!("unexpected worker info message: {:?}", &resp));
        };

        Ok(client)
    }

    /// Send a Message to the worker and return a response.
    async fn request(&mut self, req: Message) -> Result<Message> {
        req.to_writer(&mut self.stream)
            .await
            .map_err(|e| anyhow!("error sending message {:?}: {}", req, e))?;

        let (_, msg) = super::Message::from_reader_buf(&mut self.stream, &mut self.read_buf)
            .await
            .map_err(|e| anyhow!("error receiving response for {:?}: {}", req, e))?;
        Ok(msg)
    }

    async fn forward_request(&mut self, req: Message) -> Result<Tensor> {
        let send_start = std::time::Instant::now();
        req.to_writer_buf(&mut self.stream, &mut self.write_buf)
            .await
            .map_err(|e| anyhow!("error sending message {:?}: {}", req, e))?;
        let send_elapsed = send_start.elapsed();

        let recv_start = std::time::Instant::now();
        let (resp_size, msg) = super::Message::from_reader_buf(&mut self.stream, &mut self.read_buf)
            .await
            .map_err(|e| anyhow!("error receiving response for {:?}: {}", req, e))?;
        let recv_elapsed = recv_start.elapsed();

        log::debug!(
            "    {} send={:.1}ms recv={:.1}ms ({})",
            &self.address,
            send_elapsed.as_secs_f64() * 1000.0,
            recv_elapsed.as_secs_f64() * 1000.0,
            human_bytes::human_bytes(resp_size as f64),
        );

        match msg {
            Message::Tensor(raw) => Ok(raw.to_tensor(&self.device)?),
            Message::WorkerError { message } => Err(anyhow!(
                "worker {} reported error: {}",
                &self.address,
                message
            )),
            _ => Err(anyhow!("unexpected response {:?}", &msg)),
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
    fn load(_: String, _: &Context) -> Result<Box<Self>> {
        Err(anyhow!("load should never be called on cake::Client"))
    }

    async fn forward(&self, _: &Tensor, _: usize, _: usize, _: &mut Context) -> Result<Tensor> {
        Err(anyhow!(
            "immutable forward should never be called on cake::Client"
        ))
    }

    /// Executes the worker's pipeline for this tensor.
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        _: &mut Context,
    ) -> Result<Tensor> {
        log::debug!("forwarding single op");
        self.forward_request(super::Message::single_op(
            &self.layer_name,
            x,
            index_pos,
            block_idx,
        ))
        .await
    }

    /// Executes the worker's pipeline with multiple batched steps for this tensor.
    async fn forward_batch(
        &mut self,
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        _: &mut Context,
    ) -> Result<Tensor> {
        log::debug!("forwarding batch of {} elements", batch.len());
        self.forward_request(super::Message::from_batch(x, batch))
            .await
    }

    async fn goodbye(&mut self) -> Result<()> {
        self.request(Message::Goodbye).await?;
        Ok(())
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn ident(&self) -> &str {
        &self.address
    }
}
