use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{Context, Forwarder, Message, WorkerInfo};
use crate::models::Generator;

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
    /// Maps each layer name to the device it was loaded on.
    layer_devices: Arc<HashMap<String, Device>>,
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
        let cache = self.context.cache.as_ref().map(|cache| cache.as_new());

        let mut cloned_context = self.context.clone();
        cloned_context.cache = cache;

        WorkerContext {
            device: self.device.clone(),
            device_idx: self.device_idx,
            dtype: self.dtype,
            blocks: self.blocks.clone(),
            layer_devices: self.layer_devices.clone(),
            // each client loop gets a new cache
            context: cloned_context,
        }
    }
}

/// Cake worker node.
pub struct Worker<G: Generator> {
    listener: TcpListener,
    context: WorkerContext<G::Shardable>,
}

impl<G: Generator + 'static> Worker<G> {
    /// Detect how many CUDA devices are available.
    fn detect_cuda_device_count() -> usize {
        #[cfg(feature = "cuda")]
        {
            // Try creating devices until one fails
            let mut count = 0;
            while Device::new_cuda(count).is_ok() {
                count += 1;
            }
            count
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

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

        // Detect available GPUs for multi-GPU support
        let num_gpus = if ctx.device.is_cuda() {
            Self::detect_cuda_device_count().max(1)
        } else {
            1
        };

        let use_multi_gpu = num_gpus > 1 && worker_topology.layers.len() > 1;

        if use_multi_gpu {
            log::info!(
                "detected {} CUDA devices, splitting {} layers across GPUs",
                num_gpus,
                worker_topology.layers.len()
            );
        }

        // Pre-create devices and var_builders for each GPU
        let mut gpu_devices: Vec<Device> = Vec::new();
        let mut gpu_var_builders: Vec<candle_nn::VarBuilder<'static>> = Vec::new();

        if use_multi_gpu {
            let model_index = ctx.data_path.join("model.safetensors.index.json");
            for ordinal in 0..num_gpus {
                let dev = Device::new_cuda(ordinal)?;

                // Disable cudarc event tracking for multi-GPU: cudarc's CudaStream::wait()
                // rejects events from a different CudaContext (returns CUDA_ERROR_INVALID_CONTEXT).
                // When cross-device transfers call device_ptr(), the source events belong to
                // a different context, poisoning the destination context's error state.
                // Disabling event tracking prevents events from being created on CudaSlices,
                // which is safe since we use a single stream per device.
                #[cfg(feature = "cuda")]
                if let Device::Cuda(cuda_dev) = &dev {
                    unsafe { cuda_dev.disable_event_tracking(); }
                }

                let vb =
                    crate::utils::load_var_builder_from_index(model_index.clone(), ctx.dtype, dev.clone(), ctx.fp8)?;
                log::info!("  GPU {} ready", ordinal);
                gpu_devices.push(dev);
                gpu_var_builders.push(vb);
            }
        }

        let mut blocks = HashMap::new();
        let mut layer_devices: HashMap<String, Device> = HashMap::new();

        for (i, block_layer_name) in worker_topology.layers.iter().enumerate() {
            if use_multi_gpu {
                // Assign layers to GPUs: split evenly
                let gpu_idx = i * num_gpus / worker_topology.layers.len();
                let dev = &gpu_devices[gpu_idx];

                log::info!("loading {} on cuda:{} ...", &block_layer_name, gpu_idx);

                // Temporarily swap device and var_builder in context
                let orig_device = ctx.device.clone();
                let orig_vb = ctx.var_builder.take();

                ctx.device = dev.clone();
                ctx.var_builder = Some(gpu_var_builders[gpu_idx].clone());

                let block = G::Shardable::load(block_layer_name.to_string(), ctx)?;

                // Restore original context
                ctx.device = orig_device;
                ctx.var_builder = orig_vb;

                layer_devices.insert(block_layer_name.to_string(), dev.clone());
                blocks.insert(block_layer_name.to_string(), block);
            } else {
                log::info!("loading {} ...", &block_layer_name);

                // NOTE: Do NOT prefix ctx.var_builder here — Transformer::load
                // already calls vb.pp(&name) internally, so passing the root
                // var_builder is correct.
                let block = G::Shardable::load(block_layer_name.to_string(), ctx)?;
                layer_devices.insert(block_layer_name.to_string(), ctx.device.clone());
                blocks.insert(block_layer_name.to_string(), block);
            }
        }

        let blocks = Arc::new(blocks);
        let layer_devices = Arc::new(layer_devices);

        let listener = {
            let taken = ctx.listener_override.lock().unwrap().take();
            if let Some(existing) = taken {
                existing
            } else {
                TcpListener::bind(&ctx.args.address).await?
            }
        };

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
            layer_devices,
            context: ctx.clone(),
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
        // Authenticate if cluster key is set
        if let Some(ref cluster_key) = context.context.args.cluster_key {
            super::auth::authenticate_as_worker(&mut socket, cluster_key)
                .await
                .map_err(|e| anyhow!("[{}] authentication failed: {}", &client, e))?;
            log::debug!("[{}] authenticated", &client);
        }

        // read first message: expect Hello, but handle LayerAssignment for master restarts
        let (latency, _size, first_msg) = Self::read_message_timed(&mut socket).await?;
        match first_msg {
            Message::Hello => { /* normal inference handshake, continue below */ }
            Message::LayerAssignment { ref layers, .. } => {
                // Master restarted and is re-running setup against an already-running worker.
                // Ack the assignment (we already have cached data) and signal ready,
                // then close this connection so the master can reconnect for inference.
                log::info!(
                    "[{}] master re-setup: accepting {} layer assignment(s)",
                    &client,
                    layers.len()
                );
                let ack = Message::LayerAssignmentAck { needs_data: false };
                ack.to_writer(&mut socket).await?;
                Message::WorkerReady.to_writer(&mut socket).await?;
                log::info!("[{}] re-setup complete, closing setup connection", &client);
                return Ok(());
            }
            other => {
                return Err(anyhow!(
                    "[{}] unexpected first message (expected Hello): {:?}",
                    &client,
                    other
                ));
            }
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
                log::debug!("[{}] goodbye", &client);
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

            // load raw tensor to the first block's device
            let first_device = ops
                .first()
                .and_then(|(name, _, _)| context.layer_devices.get(name))
                .unwrap_or(&context.device);

            // Ensure the CUDA context for the target device is active on this thread.
            #[cfg(feature = "cuda")]
            if let Device::Cuda(cuda_dev) = first_device {
                if let Err(e) = cuda_dev.cuda_stream().context().bind_to_thread() {
                    log::error!("[{client}] failed to bind CUDA context: {:?}", e);
                }
            }

            let mut x = match x.to_tensor(first_device) {
                Ok(t) => t,
                Err(e) => {
                    let msg = format!("failed to load tensor to device: {e}");
                    log::error!("[{}] {}", &client, &msg);
                    let _ = Self::write_message_timed(
                        &mut socket,
                        Message::WorkerError { message: msg },
                    )
                    .await;
                    continue;
                }
            };

            let num_ops = ops.len();
            let start_ops = Instant::now();

            let mut batch_error = false;

            // for each element in the ops batch
            for (layer_name, index_pos, block_idx) in ops {
                // move tensor to the block's device if needed (multi-GPU)
                if let Some(block_device) = context.layer_devices.get(&layer_name) {
                    // Bind CUDA context before cross-device transfer
                    #[cfg(feature = "cuda")]
                    if let Device::Cuda(cuda_dev) = block_device {
                        if let Err(e) = cuda_dev.cuda_stream().context().bind_to_thread() {
                            log::error!("[{client}] failed to bind CUDA context for {}: {:?}", &layer_name, e);
                        }
                    }

                    x = match x.to_device(block_device) {
                        Ok(t) => t,
                        Err(e) => {
                            let msg = format!(
                                "failed to move tensor to device for layer {}: {e}",
                                &layer_name
                            );
                            log::error!("[{}] {}", &client, &msg);
                            let _ = Self::write_message_timed(
                                &mut socket,
                                Message::WorkerError { message: msg },
                            )
                            .await;
                            batch_error = true;
                            break;
                        }
                    };
                }

                // get layer block by name
                if let Some(block) = context.blocks.get(&layer_name) {
                    // run forward pass
                    x = match block
                        .forward(&x, index_pos, block_idx, &mut context.context)
                        .await
                    {
                        Ok(t) => t,
                        Err(e) => {
                            let msg = format!(
                                "forward pass failed for layer {} (block_idx={}): {e}",
                                &layer_name, block_idx
                            );
                            log::error!("[{}] {}", &client, &msg);
                            let _ = Self::write_message_timed(
                                &mut socket,
                                Message::WorkerError { message: msg },
                            )
                            .await;
                            batch_error = true;
                            break;
                        }
                    };
                } else {
                    let msg = format!("could not find layer {}", &layer_name);
                    log::error!("[{}] {}", &client, &msg);
                    let _ = Self::write_message_timed(
                        &mut socket,
                        Message::WorkerError { message: msg },
                    )
                    .await;
                    batch_error = true;
                    break;
                }
            }

            if batch_error {
                continue;
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
            let _ = socket.set_nodelay(true);
            log::debug!("{} connected", &client);

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
