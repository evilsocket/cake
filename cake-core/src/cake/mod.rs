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
mod master;

pub mod auth;
mod client;
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
                utils::hf::ensure_model_downloaded(&args.model)?
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
                    #[cfg(feature = "qwen3")]
                    "Qwen3ForCausalLM" => TextModelArch::Qwen3,
                    #[cfg(feature = "qwen3_moe")]
                    "Qwen3MoeForCausalLM" => TextModelArch::Qwen3Moe,
                    #[cfg(feature = "qwen3_5_moe")]
                    "Qwen3_5MoeForConditionalGeneration" => TextModelArch::Qwen3_5Moe,
                    #[cfg(feature = "phi4")]
                    "Phi3ForCausalLM" | "Phi4ForCausalLM" => TextModelArch::Phi4,
                    #[cfg(feature = "mistral")]
                    "MistralForCausalLM" => TextModelArch::Mistral,
                    #[cfg(feature = "gemma3")]
                    "Gemma3ForCausalLM" => TextModelArch::Gemma3,
                    #[cfg(feature = "falcon3")]
                    "FalconForCausalLM" => TextModelArch::Falcon3,
                    #[cfg(feature = "olmo2")]
                    "OLMo2ForCausalLM" | "Olmo2ForCausalLM" => TextModelArch::OLMo2,
                    #[cfg(feature = "exaone4")]
                    "ExaoneForCausalLM" => TextModelArch::EXAONE4,
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
                #[cfg(feature = "qwen3")]
                TextModelArch::Qwen3 => {
                    crate::models::qwen3::Qwen3Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "qwen3_moe")]
                TextModelArch::Qwen3Moe => {
                    crate::models::qwen3_moe::Qwen3MoeConfig::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "qwen3_5_moe")]
                TextModelArch::Qwen3_5Moe => {
                    crate::models::qwen3_5_moe::Qwen3_5MoeConfig::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "phi4")]
                TextModelArch::Phi4 => {
                    crate::models::phi4::Phi4Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "mistral")]
                TextModelArch::Mistral => {
                    crate::models::mistral::MistralConfig::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "gemma3")]
                TextModelArch::Gemma3 => {
                    crate::models::gemma3::Gemma3Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "falcon3")]
                TextModelArch::Falcon3 => {
                    crate::models::falcon3::Falcon3Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "olmo2")]
                TextModelArch::OLMo2 => {
                    crate::models::olmo2::OLMo2Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "exaone4")]
                TextModelArch::EXAONE4 => {
                    crate::models::exaone4::EXAONE4Config::from_path(&config_filename)?.into_config()
                }
                #[cfg(feature = "llama")]
                TextModelArch::Llama => {
                    crate::models::llama3::LlamaConfig::from_path(&config_filename)?.into_config()
                }
                _ => {
                    bail!("no text model feature enabled for architecture {:?}", text_model_arch)
                }
            };

            let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
            fp8 = utils::fp8::is_fp8_quantized(&config_filename);
            if fp8 {
                log::info!("model uses FP8 quantization — weights will be dequantized at load time");
            }
            let gptq_group_size = if utils::gptq::is_gptq_quantized(&config_filename) {
                let gs = utils::gptq::gptq_group_size(&config_filename);
                log::info!("model uses GPTQ quantization (group_size={gs}) — weights will be dequantized at load time");
                Some(gs)
            } else {
                None
            };
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
                        gptq_group_size,
                    )?
                } else {
                    utils::load_var_builder_for_local_layers(
                        model_tensors_index,
                        dtype,
                        device.clone(),
                        &worker_layers,
                        fp8,
                        gptq_group_size,
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
                    gptq_group_size,
                )?
            } else {
                // Worker without known layers: load everything
                utils::load_var_builder_from_index(
                    model_tensors_index,
                    dtype,
                    device.clone(),
                    fp8,
                    gptq_group_size,
                )?
            });
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
