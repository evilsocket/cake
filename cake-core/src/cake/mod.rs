use std::{
    fmt::{Debug, Display},
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::net::TcpListener;

use crate::{
    models::common::{detect_text_model_arch, Cache, Config},
    utils, Args, ModelType, TextModelArch,
};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

#[cfg(feature = "master")]
pub mod api;
#[cfg(feature = "master")]
pub mod master;

pub mod auth;
pub mod client;
pub mod discovery;
mod proto;
pub mod setup;
mod topology;
mod worker;

#[cfg(feature = "master")]
pub use master::*;

pub use client::*;
pub use proto::*;
pub use topology::*;
pub use worker::*;

/// Determines if we run in master or worker mode.
#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum Mode {
    #[default]
    Master,
    Worker,
}

/// Main contect object used as a shared state.
#[derive(Clone)]
pub struct Context {
    pub args: Args,
    pub dtype: DType,
    pub topology: Topology,
    pub data_path: PathBuf,
    pub device: Device,
    pub config: Option<Config>,
    pub cache: Option<Cache>,
    pub var_builder: Option<VarBuilder<'static>>,
    /// Resolved text model architecture.
    pub text_model_arch: TextModelArch,
    /// True if the model uses FP8 block-wise quantization.
    pub fp8: bool,
    /// Pre-bound TCP listener from setup phase (taken once by Worker::new).
    pub listener_override: Arc<Mutex<Option<TcpListener>>>,
}

impl Context {
    /// Create the context from the parsed command line arguments.
    pub fn from_args(mut args: Args) -> Result<Self> {
        let dtype: DType = match args.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => bail!("unsupported dtype {dtype}"),
            None => DType::F16,
        };

        let device = utils::get_inference_device(args.cpu, args.device)
            .map_err(|e| anyhow!("can't attach to device: {:?}", e))?;

        // Disable cudarc event tracking for CUDA devices: cudarc's CudaStream::wait()
        // rejects events from a different CudaContext, which breaks cross-device tensor
        // transfers in multi-GPU setups. Safe since we use a single stream per device.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda_dev) = &device {
            unsafe { cuda_dev.disable_event_tracking(); }
        }

        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(memory_stats::memory_stats().map(|m| m.physical_mem).unwrap_or(0) as f64)
        );

        let data_path = PathBuf::from(&args.model);
        let data_path = if !data_path.exists() {
            if utils::hf::looks_like_hf_repo(&args.model) {
                // Image models (LTX-2, Flux, etc.) use diffusers format without a root
                // config.json — their forwarders handle HF resolution internally.
                // Only download via the generic path for text models.
                if args.model_type == ModelType::TextModel {
                    utils::hf::ensure_model_downloaded(&args.model)?
                } else {
                    // Pass the repo ID through; forwarders resolve it themselves
                    data_path
                }
            } else {
                bail!("model path does not exist: {}", data_path.display());
            }
        } else {
            data_path
        };

        let topology = if let Some(topo) = args.topology_override.take() {
            // Zero-config setup already built the topology
            topo
        } else if let Some(path) = &args.topology {
            Topology::from_path(path, &args.model_type)?
        } else {
            log::warn!("no topology file specified, the entire model will be loaded");
            Topology::new()
        };

        let mut config: Option<Config> = None;
        let mut cache: Option<Cache> = None;
        let mut var_builder: Option<VarBuilder> = None;
        let mut text_model_arch = args.text_model_arch;
        let mut fp8 = false;

        if args.model_type == ModelType::TextModel {
            let config_filename = data_path.join("config.json");

            // Auto-detect architecture if needed
            if text_model_arch == TextModelArch::Auto {
                let arch_str = detect_text_model_arch(&config_filename).unwrap_or_default();
                text_model_arch = match arch_str.as_str() {
                    #[cfg(feature = "qwen2")]
                    "Qwen2ForCausalLM" => TextModelArch::Qwen2,
                    #[cfg(feature = "qwen3_5")]
                    "Qwen3_5ForConditionalGeneration" => TextModelArch::Qwen3_5,
                    #[cfg(feature = "llava")]
                    "LlavaForConditionalGeneration" | "LlavaLlamaForCausalLM" => {
                        TextModelArch::Llava
                    }
                    #[cfg(feature = "mixtral")]
                    "MixtralForCausalLM" => TextModelArch::Mixtral,
                    _ => TextModelArch::Llama,
                };
            }

            log::info!("text model architecture: {:?}", text_model_arch);

            let config_internal = match text_model_arch {
                #[cfg(feature = "qwen2")]
                TextModelArch::Qwen2 => {
                    crate::models::qwen2::QwenConfig::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "qwen3_5")]
                TextModelArch::Qwen3_5 => {
                    crate::models::qwen3_5::Qwen3_5Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "llava")]
                TextModelArch::Llava => {
                    crate::models::llava::LlavaConfig::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "mixtral")]
                TextModelArch::Mixtral => {
                    crate::models::mixtral::MixtralConfig::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "llama")]
                TextModelArch::Llama => {
                    crate::models::llama3::LlamaConfig::from_path(&config_filename)?.into_config()
                }
                _ => {
                    // Fallback: use a generic config parser approach
                    // Parse the raw JSON and construct Config directly
                    bail!("no text model feature enabled for architecture {:?}", text_model_arch)
                }
            };

            // Check for GGUF file first, then fall back to safetensors
            let gguf_file = utils::gguf::detect_gguf_file(&data_path);

            if let Some(ref gguf_path) = gguf_file {
                log::info!("detected GGUF model: {}", gguf_path.display());
                var_builder = Some(utils::gguf::load_var_builder_from_gguf(
                    gguf_path,
                    dtype,
                    device.clone(),
                    &config_internal.model_prefix,
                )?);
            } else {
                let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
                fp8 = utils::fp8::is_fp8_quantized(&config_filename);
                if fp8 {
                    log::info!("model uses FP8 quantization — weights will be dequantized at load time");
                }
                let is_master = matches!(args.mode, Mode::Master);
                let my_layers: Vec<String> = if !is_master {
                    topology.all_worker_layers().into_iter().collect()
                } else {
                    vec![]
                };

                var_builder = Some(if is_master {
                    // Master: exclude shards that only contain remote-worker tensors
                    let worker_layers = topology.all_worker_layers();
                    if worker_layers.is_empty() {
                        utils::load_var_builder_from_index(
                            model_tensors_index,
                            dtype,
                            device.clone(),
                            fp8,
                        )?
                    } else {
                        utils::load_var_builder_for_local_layers(
                            model_tensors_index,
                            dtype,
                            device.clone(),
                            &worker_layers,
                            fp8,
                        )?
                    }
                } else if !my_layers.is_empty() {
                    // Worker with known layers: only load shards containing our layers
                    utils::load_var_builder_for_specific_layers(
                        model_tensors_index,
                        dtype,
                        device.clone(),
                        &my_layers,
                        fp8,
                    )?
                } else {
                    // Worker without known layers: load everything
                    utils::load_var_builder_from_index(
                        model_tensors_index,
                        dtype,
                        device.clone(),
                        fp8,
                    )?
                });
            }
            cache = Some(Cache::new(true, dtype, &config_internal, &device)?);
            config = Some(config_internal);
        }

        Ok(Context {
            args,
            dtype,
            topology,
            data_path,
            device,
            config,
            cache,
            var_builder,
            text_model_arch,
            fp8,
            listener_override: Arc::new(Mutex::new(None)),
        })
    }
}

/// This is the trait that a shardable object must implement.
#[async_trait]
pub trait Forwarder: Debug + Send + Sync + Display {
    /// Create an instance of this object loading the specified layer(s) from a VarBuilder.
    fn load(name: String, ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized;

    /// Applies a forward operation to the input tensor, does not require mutability.
    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor>;

    /// Applies a forward operation to the input tensor, requires mutability.
    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor>;

    /// Applies a batch of forward operations to the input tensor.
    async fn forward_batch(
        &mut self,
        _x: &Tensor,
        _batch: Vec<(String, usize, usize)>,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    async fn goodbye(&mut self) -> Result<()> {
        Ok(())
    }

    /// Return the layer name.
    fn layer_name(&self) -> &str;

    /// Return the unique identity or local.
    fn ident(&self) -> &str {
        "local"
    }
}
