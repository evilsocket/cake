use std::{collections::HashMap, fs, net::SocketAddr, sync::Arc};

use super::{Context, Message, Topology, WorkerInfo};
use crate::model::{Block, Cache};

use anyhow::Result;
use candle_core::Device;
use tokio::{
    net::{TcpListener, TcpStream},
    sync::Mutex,
};

pub struct Worker {
    listener: TcpListener,

    cache: Arc<Mutex<Cache>>,
    blocks: Arc<HashMap<String, Block>>,
    device: Device,
}

impl Worker {
    pub async fn new(ctx: Context) -> Result<Self> {
        log::info!("loading worker topology from {}", &ctx.args.topology_path);

        let topology: Topology =
            serde_json::from_str(&fs::read_to_string(&ctx.args.topology_path)?)?;

        let mut blocks = HashMap::new();

        for block_layer_name in topology.keys() {
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

        let cache = Arc::new(Mutex::new(ctx.cache));
        let device = ctx.device;

        Ok(Self {
            listener,
            cache,
            blocks,
            device,
        })
    }

    async fn handle_client(
        mut socket: TcpStream,
        client: SocketAddr,
        blocks: Arc<HashMap<String, Block>>,
        device: Device,
        cache: Arc<Mutex<Cache>>,
    ) -> Result<()> {
        // read and validate Hello
        let hello = Message::from_reader(&mut socket).await;
        let hello = if let Ok(hello) = hello {
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
        let info = Message::WorkerInfo(WorkerInfo {
            device: format!("{:?}", &device),
        });
        if let Err(e) = info.to_writer(&mut socket).await {
            return Err(anyhow!("[{}] could not send worker info: {:?}", &client, e));
        }

        // read next message
        while let Ok(msg) = Message::from_reader(&mut socket).await {
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
            let mut x = x.to_tensor(&device).unwrap();

            // log::info!("{}", if ops.len() == 1 { "RUN" } else { "BATCH" });

            // for each element in the ops batch
            for (layer_name, index_pos, block_idx) in ops {
                // get layer block by name
                if let Some(block) = blocks.get(&layer_name) {
                    // log::info!("  x = {}.forward(x, {index_pos}, {block_idx})", &layer_name);
                    // run forward pass
                    x = {
                        let mut guard = cache.lock().await;
                        block
                            .forward_imm(&x, index_pos, block_idx, &mut guard)
                            .await
                            .unwrap()
                    };
                } else {
                    return Err(anyhow!("could not find layer {}", &layer_name));
                }
            }

            // send response tensor
            if let Err(e) = Message::from_tensor(&x).to_writer(&mut socket).await {
                return Err(anyhow!(
                    "[{}] could not send response tensor: {:?}",
                    &client,
                    e
                ));
            }
        }

        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        while let Ok((socket, client)) = self.listener.accept().await {
            log::info!("{} connected", &client);

            let blocks = self.blocks.clone();
            let device = self.device.clone();
            let cache = self.cache.clone();

            tokio::spawn(async move {
                if let Err(e) = Self::handle_client(socket, client, blocks, device, cache).await {
                    log::error!("{}", e);
                }
            });
        }

        Ok(())
    }
}
