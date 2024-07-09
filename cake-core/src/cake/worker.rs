use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::Instant};

use super::{Context, Message, WorkerInfo};
use crate::model::{Block, Cache};

use anyhow::Result;
use candle_core::{DType, Device};
use tokio::net::{TcpListener, TcpStream};

struct WorkerContext {
    device: Device,
    device_idx: usize,
    dtype: DType,
    blocks: Arc<HashMap<String, Block>>,
    cache: Cache,
}

impl WorkerContext {
    fn to_info(&self, latency: u128) -> WorkerInfo {
        WorkerInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            device: if self.device.is_cuda() {
                "cuda".to_string()
            } else if self.device.is_metal() {
                "metal".to_string()
            } else {
                "cpu".to_string()
            },
            device_idx: self.device_idx,
            latency,
            dtype: format!("{:?}", self.dtype),
        }
    }
}

pub struct Worker {
    listener: TcpListener,

    cache: Cache,
    blocks: Arc<HashMap<String, Block>>,
    device: Device,
    device_idx: usize,
    dtype: DType,
}

impl Worker {
    pub async fn new(ctx: Context) -> Result<Self> {
        let worker_name = if let Some(name) = &ctx.args.name {
            name.to_string()
        } else {
            return Err(anyhow!("no --name provided for worker"));
        };

        let worker_topology = if let Some(node) = ctx.topology.get(&worker_name) {
            node
        } else {
            return Err(anyhow!("could not find topology for {worker_name}"));
        };

        let mut blocks = HashMap::new();

        for block_layer_name in &worker_topology.layers {
            log::info!("loading {} ...", &block_layer_name);

            let block = Block::load(
                block_layer_name,
                ctx.var_builder.pp(block_layer_name),
                &ctx.config,
            )?;

            blocks.insert(block_layer_name.to_string(), block);
        }

        let blocks = Arc::new(blocks);

        let listener = TcpListener::bind(&ctx.args.address).await?;

        log::info!(
            "listening on {} (mem:{}) ...",
            &ctx.args.address,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let cache = ctx.cache;
        let device = ctx.device;
        let dtype = ctx.dtype;
        let device_idx = ctx.args.device;

        Ok(Self {
            dtype,
            listener,
            cache,
            blocks,
            device,
            device_idx,
        })
    }

    async fn handle_client(
        mut socket: TcpStream,
        client: SocketAddr,
        mut context: WorkerContext,
    ) -> Result<()> {
        // read and validate Hello
        let start = Instant::now();
        let hello = Message::from_reader(&mut socket).await;
        let latency = start.elapsed();
        let hello = if let Ok((_, hello)) = hello {
            hello
        } else {
            return Err(anyhow!("[{}] could not read Hello: {:?}", &client, hello));
        };
        if !matches!(hello, Message::Hello) {
            return Err(anyhow!(
                "[{}] unpexpected message instead of hello: {:?}",
                &client,
                hello
            ));
        }

        // send info
        let info = Message::WorkerInfo(context.to_info(latency.as_millis()));

        if let Err(e) = info.to_writer(&mut socket).await {
            return Err(anyhow!("[{}] could not send worker info: {:?}", &client, e));
        }

        let mut msg_idx = 0;
        let mut avg_ops = 0;
        let mut avg_write = 0;
        let mut avg_read = 0;

        loop {
            // read next message
            let start_read = Instant::now();
            let ret = Message::from_reader(&mut socket).await;
            let elaps_read = start_read.elapsed();

            if let Ok((msg_size, msg)) = ret {
                let (x, ops) = match msg {
                    // single block operation
                    Message::TransformerOp {
                        layer_name,
                        x,
                        index_pos,
                        block_idx,
                    } => (x, vec![(layer_name, index_pos, block_idx)]),
                    Message::Batch { x, batch } => (x, batch),
                    _ => {
                        return Err(anyhow!(
                            "[{}] unhandled message in loop: {:?}",
                            &client,
                            msg
                        ));
                    }
                };

                // load raw tensor to device
                let mut x = x.to_tensor(&context.device).unwrap();

                // log::info!("{}", if ops.len() == 1 { "RUN" } else { "BATCH" });

                let num_ops = ops.len();
                let start_ops = Instant::now();

                // for each element in the ops batch
                for (layer_name, index_pos, block_idx) in ops {
                    // get layer block by name
                    if let Some(block) = context.blocks.get(&layer_name) {
                        // log::info!("  x = {}.forward(x, {index_pos}, {block_idx})", &layer_name);
                        // run forward pass
                        x = block
                            .forward_imm(&x, index_pos, block_idx, &mut context.cache)
                            .await
                            .unwrap()
                    } else {
                        return Err(anyhow!("could not find layer {}", &layer_name));
                    }
                }

                let elaps_ops = start_ops.elapsed();
                let start_write = Instant::now();

                // send response tensor
                match Message::from_tensor(&x).to_writer(&mut socket).await {
                    Err(e) => {
                        return Err(anyhow!(
                            "[{}] could not send response tensor: {:?}",
                            &client,
                            e
                        ));
                    }
                    Ok(n) => {
                        let elaps_write = start_write.elapsed();

                        let ops_per_sec = (num_ops as f64 / elaps_ops.as_secs_f64()) as usize;
                        let write_bytes_per_sec = (n as f64 / elaps_write.as_secs_f64()) as usize;
                        let read_bytes_per_sec =
                            (msg_size as f64 / elaps_read.as_secs_f64()) as usize;

                        avg_ops += ops_per_sec;
                        avg_write += write_bytes_per_sec;
                        avg_read += read_bytes_per_sec;
                    }
                }

                if msg_idx % 3 == 0 {
                    log::info!(
                        "ops={}/s read={}/s write={}/s",
                        avg_ops / 3,
                        human_bytes::human_bytes(avg_read as f64 / 3.0),
                        human_bytes::human_bytes(avg_write as f64 / 3.0)
                    );
                    avg_ops = 0;
                    avg_write = 0;
                    avg_read = 0;
                }

                msg_idx += 1;
            } else {
                log::error!("[{}] {:?}", &client, ret);
                break;
            }
        }

        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        while let Ok((socket, client)) = self.listener.accept().await {
            log::info!("{} connected", &client);

            let context = WorkerContext {
                device: self.device.clone(),
                device_idx: self.device_idx,
                dtype: self.dtype,
                blocks: self.blocks.clone(),
                cache: self.cache.as_new(), // each client loop gets a new cache
            };

            tokio::spawn(async move {
                if let Err(e) = Self::handle_client(socket, client, context).await {
                    log::error!("{}", e);
                }
            });
        }

        Ok(())
    }
}
