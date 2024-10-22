use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{Context, Forwarder, Message, WorkerInfo};
use crate::{models::Generator, ModelType};

use anyhow::Result;
use candle_core::{DType, Device};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
};

/// Determines how often worker statistics are calculated and printed.
const NUM_OPS_TO_STATS: usize = 5;

/// A single worker state.
#[derive(Clone)]
struct WorkerContext<F> {
    device: Device,
    device_idx: usize,
    dtype: DType,
    blocks: Arc<HashMap<String, Box<F>>>,
    context: Context,
}

impl<F: Forwarder> WorkerContext<F> {
    /// Create a WorkerInfo structure to be sent to the master.
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

    /// Create a copy of self with new kv-cache.
    fn get_client_context(&self) -> Self {

        let cache = match &self.context.cache {
            None => None,
            Some(cache) => {
                Some(cache.as_new())
            }
        };

        let mut cloned_context = self.context.clone();
        cloned_context.cache = cache;

        WorkerContext {
            device: self.device.clone(),
            device_idx: self.device_idx,
            dtype: self.dtype,
            blocks: self.blocks.clone(),
            // each client loop gets a new cache
            context: cloned_context
        }
    }
}

/// Cake worker node.
pub struct Worker<G: Generator> {
    listener: TcpListener,
    context: WorkerContext<G::Shardable>,
}

impl<G: Generator + 'static> Worker<G> {
    /// Create a new Worker from the context.
    pub async fn new(ctx: &mut Context) -> Result<Self> {
        let worker_name = if let Some(name) = &ctx.args.name {
            name.to_string()
        } else {
            return Err(anyhow!("no --name provided for worker"));
        };

        let worker_topology = if let Some(node) = ctx.topology.get(&worker_name) {
            node
        } else if !ctx.topology.is_empty() {
            let first = ctx.topology.keys().next().unwrap();
            log::warn!(
                "topology for worker name '{}' not found, using '{}'",
                &worker_name,
                first
            );
            ctx.topology.get(first).unwrap()
        } else {
            return Err(anyhow!(
                "could not find topology for {worker_name} and topology file is empty"
            ));
        };

        let mut blocks = HashMap::new();

        let vb = ctx.var_builder.clone();

        for block_layer_name in &worker_topology.layers {
            log::info!("loading {} ...", &block_layer_name);

            if ctx.args.model_type == ModelType::TextModel {
                ctx.var_builder = Some(vb.clone().expect("Error retrieving var_builder").pp(block_layer_name));
            }

            let block = G::Shardable::load(
                block_layer_name.to_string(),
                &ctx,
            )?;

            blocks.insert(block_layer_name.to_string(), block);
        }

        ctx.var_builder = vb;

        let blocks = Arc::new(blocks);

        let listener = TcpListener::bind(&ctx.args.address).await?;

        log::info!(
            "listening on {} (mem:{}) ...",
            &ctx.args.address,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let device = ctx.device.clone();
        let dtype = ctx.dtype;
        let device_idx = ctx.args.device;

        let context = WorkerContext {
            device,
            device_idx,
            dtype,
            blocks,
            context: ctx.clone()
        };

        Ok(Self { listener, context })
    }

    /// Read a message from the socket and return elapsed time, message size and message.
    async fn read_message_timed<R>(mut socket: R) -> Result<(Duration, usize, Message)>
    where
        R: AsyncReadExt + Unpin,
    {
        let start = Instant::now();
        let (size, message) = Message::from_reader(&mut socket).await?;
        let latency = start.elapsed();

        Ok((latency, size, message))
    }

    /// Write a message to the socket and return the elapsed time with written size.
    async fn write_message_timed<W>(mut socket: W, message: Message) -> Result<(Duration, usize)>
    where
        W: AsyncWriteExt + Unpin,
    {
        let start = Instant::now();
        let size = message.to_writer(&mut socket).await?;
        let latency = start.elapsed();

        Ok((latency, size))
    }

    /// Main loop handling communication with the master.
    async fn handle_master_client(
        mut socket: TcpStream,
        client: SocketAddr,
        mut context: WorkerContext<G::Shardable>,
    ) -> Result<()> {
        // read and validate Hello
        let (latency, _size, hello) = Self::read_message_timed(&mut socket).await?;
        if !matches!(hello, Message::Hello) {
            return Err(anyhow!(
                "[{}] unpexpected message instead of hello: {:?}",
                &client,
                hello
            ));
        }

        // send info
        if let Err(e) = Self::write_message_timed(
            &mut socket,
            Message::WorkerInfo(context.to_info(latency.as_millis())),
        )
        .await
        {
            return Err(anyhow!("[{}] could not send worker info: {:?}", &client, e));
        }

        let mut msg_idx = 0;
        let mut avg_ops = 0;
        let mut avg_write = 0;
        let mut avg_read = 0;

        // keep reading messages
        while let Ok((read_time, read_size, op_message)) =
            Self::read_message_timed(&mut socket).await
        {
            if matches!(op_message, Message::Goodbye) {
                log::info!("[{}] goodbye", &client);
                context
                    .context
                    .cache
                    .as_mut()
                    .expect("No cache specified")
                    .clear();

                // send info
                if let Err(e) = Self::write_message_timed(
                    &mut socket,
                    Message::WorkerInfo(context.to_info(read_time.as_millis())),
                )
                .await
                {
                    return Err(anyhow!("[{}] could not send worker info: {:?}", &client, e));
                }

                continue;
            }

            let (x, ops) = match op_message {
                // single block operation
                Message::SingleOp {
                    layer_name,
                    x,
                    index_pos,
                    block_idx,
                } => (x, vec![(layer_name, index_pos, block_idx)]),
                // batched
                Message::Batch { x, batch } => (x, batch),
                _ => {
                    return Err(anyhow!(
                        "[{}] unhandled message in loop: {:?}",
                        &client,
                        op_message
                    ));
                }
            };

            // load raw tensor to device
            let mut x = x.to_tensor(&context.device).unwrap();
            let num_ops = ops.len();
            let start_ops = Instant::now();

            // for each element in the ops batch
            for (layer_name, index_pos, block_idx) in ops {
                // get layer block by name
                if let Some(block) = context.blocks.get(&layer_name) {
                    // run forward pass
                    x = block
                        .forward(&x, index_pos, block_idx, &mut context.context)
                        .await
                        .unwrap();
                } else {
                    return Err(anyhow!("could not find layer {}", &layer_name));
                }
            }

            let elaps_ops = start_ops.elapsed();

            // send response tensor
            match Self::write_message_timed(&mut socket, Message::from_tensor(&x)).await {
                Ok((elaps_write, written)) => {
                    let ops_per_sec = (num_ops as f64 / elaps_ops.as_secs_f64()) as usize;
                    let write_bytes_per_sec = (written as f64 / elaps_write.as_secs_f64()) as usize;
                    let read_bytes_per_sec = (read_size as f64 / read_time.as_secs_f64()) as usize;

                    avg_ops += ops_per_sec;
                    avg_write += write_bytes_per_sec;
                    avg_read += read_bytes_per_sec;
                }
                Err(e) => {
                    return Err(anyhow!(
                        "[{}] could not send response tensor: {:?}",
                        &client,
                        e
                    ));
                }
            }

            // compute and print stats every NUM_OPS_TO_STATS operations to avoid spamming stdout
            if msg_idx % NUM_OPS_TO_STATS == 0 {
                log::info!(
                    "ops={}/s read={}/s write={}/s",
                    avg_ops / NUM_OPS_TO_STATS,
                    human_bytes::human_bytes(avg_read as f64 / NUM_OPS_TO_STATS as f64),
                    human_bytes::human_bytes(avg_write as f64 / NUM_OPS_TO_STATS as f64)
                );
                avg_ops = 0;
                avg_write = 0;
                avg_read = 0;
            }
            msg_idx += 1;
        }

        Ok(())
    }

    /// Run the worker server accept loop.
    pub async fn run(&mut self) -> Result<()> {
        while let Ok((socket, client)) = self.listener.accept().await {
            log::info!("{} connected", &client);

            let context = self.context.get_client_context();
            tokio::spawn(async move {
                if let Err(e) = Self::handle_master_client(socket, client, context).await {
                    log::error!("{}", e);
                }
            });
        }

        Ok(())
    }
}
