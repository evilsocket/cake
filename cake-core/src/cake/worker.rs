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

/// Return a human-readable device type string for a candle Device.
pub(crate) fn device_type_str(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda"
    } else if device.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

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
            device: device_type_str(&self.device).to_string(),
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

        let mut blocks = HashMap::new();
        let mut layer_devices: HashMap<String, Device> = HashMap::new();

        if use_multi_gpu {
            let model_index = ctx.data_path.join("model.safetensors.index.json");

            // Group layers by GPU assignment
            let mut gpu_layer_groups: Vec<Vec<String>> = vec![vec![]; num_gpus];
            for (i, name) in worker_topology.layers.iter().enumerate() {
                let gpu_idx = i * num_gpus / worker_topology.layers.len();
                gpu_layer_groups[gpu_idx].push(name.clone());
            }

            // Create per-GPU devices and VarBuilders (filtered to each GPU's layers)
            let mut gpu_devices: Vec<Device> = Vec::new();
            let mut gpu_var_builders: Vec<candle_nn::VarBuilder<'static>> = Vec::new();

            for (ordinal, _group) in gpu_layer_groups.iter().enumerate().take(num_gpus) {
                let dev = Device::new_cuda(ordinal)?;

                #[cfg(feature = "cuda")]
                if let Device::Cuda(cuda_dev) = &dev {
                    unsafe {
                        cuda_dev.disable_event_tracking();
                    }
                }

                let vb = crate::utils::load_var_builder_for_specific_layers(
                    model_index.clone(),
                    ctx.dtype,
                    dev.clone(),
                    &gpu_layer_groups[ordinal],
                    ctx.fp8,
                    None,
                )?;
                log::info!("  GPU {} ready", ordinal);
                gpu_devices.push(dev);
                gpu_var_builders.push(vb);
            }

            // Load layers in parallel across GPUs
            let mut handles = Vec::new();
            for gpu_idx in 0..num_gpus {
                let dev = gpu_devices[gpu_idx].clone();
                let vb = gpu_var_builders[gpu_idx].clone();
                let layers = std::mem::take(&mut gpu_layer_groups[gpu_idx]);
                let mut thread_ctx = ctx.clone();
                thread_ctx.device = dev.clone();
                thread_ctx.var_builder = Some(vb);

                handles.push(std::thread::spawn(
                    #[allow(clippy::type_complexity)]
                    move || -> Result<Vec<(String, Device, Box<G::Shardable>)>> {
                        #[cfg(feature = "cuda")]
                        if let Device::Cuda(ref cuda_dev) = dev {
                            cuda_dev
                                .cuda_stream()
                                .context()
                                .bind_to_thread()
                                .map_err(|e| {
                                    anyhow!(
                                        "failed to bind CUDA context for GPU {gpu_idx}: {e:?}"
                                    )
                                })?;
                        }

                        let mut results = Vec::new();
                        for layer_name in layers {
                            log::info!("loading {} on cuda:{} ...", &layer_name, gpu_idx);
                            let block =
                                G::Shardable::load(layer_name.clone(), &thread_ctx)?;
                            results.push((layer_name, dev.clone(), block));
                        }
                        Ok(results)
                    },
                ));
            }

            // Collect results from all GPU threads
            for handle in handles {
                let results = handle
                    .join()
                    .map_err(|_| anyhow!("GPU loading thread panicked"))??;
                for (name, dev, block) in results {
                    layer_devices.insert(name.clone(), dev);
                    blocks.insert(name, block);
                }
            }
        } else {
            for block_layer_name in worker_topology.layers.iter() {
                log::info!("loading {} ...", &block_layer_name);

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
            human_bytes::human_bytes(memory_stats::memory_stats().map(|m| m.physical_mem).unwrap_or(0) as f64)
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
        let mut read_buf = Vec::new();
        let mut write_buf = Vec::new();

        // keep reading messages
        while let Ok((read_time, read_size, op_message)) = {
            let start = Instant::now();
            Message::from_reader_buf(&mut socket, &mut read_buf)
                .await
                .map(|(size, msg)| (start.elapsed(), size, msg))
        } {
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
            let load_start = Instant::now();
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

            let load_elapsed = load_start.elapsed();

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
                            log::error!(
                                "[{client}] failed to bind CUDA context for {}: {:?}",
                                &layer_name,
                                e
                            );
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
                        Ok(t) => {
                            // Metal requires per-layer sync to prevent command buffer
                            // accumulation which causes catastrophic performance degradation.
                            if t.device().is_metal() {
                                let _ = t.device().synchronize();
                            }
                            t
                        }
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

            // serialize response tensor (includes GPU sync for data readback)
            let ser_start = Instant::now();
            let resp_msg = Message::from_tensor(&x);
            let ser_elapsed = ser_start.elapsed();

            // send response tensor (reuse write buffer)
            let write_start = Instant::now();
            match resp_msg.to_writer_buf(&mut socket, &mut write_buf).await {
                Ok(written) => {
                    let elaps_write = write_start.elapsed();
                    log::debug!(
                        "[{}] read={:.1}ms load={:.1}ms fwd={:.1}ms ser={:.1}ms write={:.1}ms ({} ops)",
                        &client,
                        read_time.as_secs_f64() * 1000.0,
                        load_elapsed.as_secs_f64() * 1000.0,
                        elaps_ops.as_secs_f64() * 1000.0,
                        ser_elapsed.as_secs_f64() * 1000.0,
                        elaps_write.as_secs_f64() * 1000.0,
                        num_ops,
                    );

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

#[cfg(test)]
mod tests {
    use super::*;

    // --- device_type_str ---

    #[test]
    fn device_type_str_cpu() {
        assert_eq!(device_type_str(&Device::Cpu), "cpu");
    }

    // CUDA/Metal variants can only be tested when those features are enabled and
    // hardware is available. The CPU path exercises the fallback branch.

    #[test]
    fn device_type_str_is_one_of_known() {
        let result = device_type_str(&Device::Cpu);
        assert!(["cpu", "cuda", "metal"].contains(&result));
    }

    // --- detect_cuda_device_count (CPU-only fallback) ---

    // When built without the `cuda` feature, detect_cuda_device_count returns 0.
    // When built with `cuda` but no GPU present, it also returns 0.
    // We can't assert an exact count, but we can verify it doesn't panic.
    #[test]
    fn detect_cuda_device_count_does_not_panic() {
        // We need a concrete Generator type. Use SD since it's always available.
        use crate::models::sd::SD;
        let count = <crate::cake::Worker<SD>>::detect_cuda_device_count();
        // Without cuda feature, this is always 0.
        #[cfg(not(feature = "cuda"))]
        assert_eq!(count, 0);
        // With cuda feature, it's >= 0 (might be 0 if no GPU).
        #[cfg(feature = "cuda")]
        let _ = count;
    }
}
