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
    /// Quantization strategy for weight loading.
    pub quant: Arc<dyn utils::Quantization>,
    /// Pre-bound TCP listener from setup phase (taken once by Worker::new).
    pub listener_override: Arc<Mutex<Option<TcpListener>>>,
}

/// Parse a dtype string ("f16", "bf16", "f32") into a candle DType.
/// Returns F16 when the input is `None`.
pub(crate) fn parse_dtype_str(s: Option<&str>) -> Result<DType> {
    match s {
        Some("f16") => Ok(DType::F16),
        Some("bf16") => Ok(DType::BF16),
        Some("f32") => Ok(DType::F32),
        Some(other) => bail!("unsupported dtype {other}"),
        None => Ok(DType::F16),
    }
}

/// Map a config.json `architectures` string to our `TextModelArch` enum.
/// Feature-gated variants are only matched when the corresponding feature is enabled;
/// unrecognised strings fall back to `Llama`.
pub(crate) fn arch_str_to_text_model_arch(arch: &str) -> TextModelArch {
    match arch {
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
    }
}

impl Context {
    /// Create the context from the parsed command line arguments.
    pub fn from_args(mut args: Args) -> Result<Self> {
        let dtype = parse_dtype_str(args.dtype.as_deref())?;

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
        let data_path = if args.model_type == ModelType::ImageModel {
            // Image models (SD, FLUX) download components on-demand via their own
            // ModelFile::get() methods. Just use the path or repo ID as-is.
            data_path
        } else if !data_path.exists() {
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
        let mut quant: Arc<dyn utils::Quantization> = Arc::new(utils::NoQuantization);

        if args.model_type == ModelType::TextModel {
            // Check if the model path is a GGUF file
            let is_gguf = utils::gguf::is_gguf_file(&data_path);

            if is_gguf {
                // GGUF path: extract config and architecture from GGUF metadata
                let arch_str = utils::gguf::arch_from_gguf(&data_path)?;
                text_model_arch = arch_str_to_text_model_arch(&arch_str);
                log::info!("GGUF model: architecture={:?} (from metadata)", text_model_arch);

                let config_internal = utils::gguf::config_from_gguf(&data_path)?;
                log::info!(
                    "GGUF config: hidden={}, layers={}, heads={}/{}, vocab={}",
                    config_internal.hidden_size,
                    config_internal.num_hidden_layers,
                    config_internal.num_attention_heads,
                    config_internal.num_key_value_heads,
                    config_internal.vocab_size,
                );

                var_builder = Some(utils::gguf::load_gguf_var_builder(
                    &data_path,
                    dtype,
                    &device,
                    &config_internal.model_prefix,
                )?);
                cache = Some(Cache::new(true, dtype, &config_internal, &device)?);
                config = Some(config_internal);
            } else {
            // Safetensors path (existing flow)
            let config_filename = data_path.join("config.json");

            // Auto-detect architecture if needed
            if text_model_arch == TextModelArch::Auto {
                let arch_str = detect_text_model_arch(&config_filename).unwrap_or_default();
                text_model_arch = arch_str_to_text_model_arch(&arch_str);
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
            quant = Arc::from(utils::detect_quantization(&config_filename));
            let is_master = matches!(args.mode, Mode::Master);
            let my_layers: Vec<String> = if !is_master {
                topology.all_worker_layers().into_iter().collect()
            } else {
                vec![]
            };

            var_builder = Some(if is_master {
                let worker_layers = topology.all_worker_layers();
                if worker_layers.is_empty() {
                    utils::load_var_builder_from_index(
                        model_tensors_index, dtype, device.clone(), &*quant,
                    )?
                } else {
                    utils::load_var_builder_for_local_layers(
                        model_tensors_index, dtype, device.clone(), &worker_layers, &*quant,
                    )?
                }
            } else if !my_layers.is_empty() {
                utils::load_var_builder_for_specific_layers(
                    model_tensors_index, dtype, device.clone(), &my_layers, &*quant,
                )?
            } else {
                utils::load_var_builder_from_index(
                    model_tensors_index, dtype, device.clone(), &*quant,
                )?
            });
            cache = Some(Cache::new(true, dtype, &config_internal, &device)?);
            config = Some(config_internal);
            } // end else (safetensors path)
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
            quant,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    // --- parse_dtype_str ---

    #[test]
    fn parse_dtype_f16() {
        assert_eq!(parse_dtype_str(Some("f16")).unwrap(), DType::F16);
    }

    #[test]
    fn parse_dtype_bf16() {
        assert_eq!(parse_dtype_str(Some("bf16")).unwrap(), DType::BF16);
    }

    #[test]
    fn parse_dtype_f32() {
        assert_eq!(parse_dtype_str(Some("f32")).unwrap(), DType::F32);
    }

    #[test]
    fn parse_dtype_none_defaults_to_f16() {
        assert_eq!(parse_dtype_str(None).unwrap(), DType::F16);
    }

    #[test]
    fn parse_dtype_invalid_returns_err() {
        assert!(parse_dtype_str(Some("int8")).is_err());
        assert!(parse_dtype_str(Some("")).is_err());
    }

    // --- arch_str_to_text_model_arch ---

    #[test]
    fn arch_str_unknown_falls_back_to_llama() {
        assert_eq!(arch_str_to_text_model_arch("UnknownArchXYZ"), TextModelArch::Llama);
        assert_eq!(arch_str_to_text_model_arch(""), TextModelArch::Llama);
    }

    #[test]
    #[cfg(feature = "qwen2")]
    fn arch_str_qwen2() {
        assert_eq!(arch_str_to_text_model_arch("Qwen2ForCausalLM"), TextModelArch::Qwen2);
    }

    #[test]
    #[cfg(feature = "qwen3_5")]
    fn arch_str_qwen3_5() {
        assert_eq!(
            arch_str_to_text_model_arch("Qwen3_5ForConditionalGeneration"),
            TextModelArch::Qwen3_5
        );
    }

    #[test]
    #[cfg(feature = "qwen3")]
    fn arch_str_qwen3() {
        assert_eq!(arch_str_to_text_model_arch("Qwen3ForCausalLM"), TextModelArch::Qwen3);
    }

    #[test]
    #[cfg(feature = "qwen3_moe")]
    fn arch_str_qwen3_moe() {
        assert_eq!(arch_str_to_text_model_arch("Qwen3MoeForCausalLM"), TextModelArch::Qwen3Moe);
    }

    #[test]
    #[cfg(feature = "qwen3_5_moe")]
    fn arch_str_qwen3_5_moe() {
        assert_eq!(
            arch_str_to_text_model_arch("Qwen3_5MoeForConditionalGeneration"),
            TextModelArch::Qwen3_5Moe
        );
    }

    #[test]
    #[cfg(feature = "phi4")]
    fn arch_str_phi3_and_phi4() {
        assert_eq!(arch_str_to_text_model_arch("Phi3ForCausalLM"), TextModelArch::Phi4);
        assert_eq!(arch_str_to_text_model_arch("Phi4ForCausalLM"), TextModelArch::Phi4);
    }

    #[test]
    #[cfg(feature = "mistral")]
    fn arch_str_mistral() {
        assert_eq!(arch_str_to_text_model_arch("MistralForCausalLM"), TextModelArch::Mistral);
    }

    #[test]
    #[cfg(feature = "gemma3")]
    fn arch_str_gemma3() {
        assert_eq!(arch_str_to_text_model_arch("Gemma3ForCausalLM"), TextModelArch::Gemma3);
    }

    #[test]
    #[cfg(feature = "falcon3")]
    fn arch_str_falcon3() {
        assert_eq!(arch_str_to_text_model_arch("FalconForCausalLM"), TextModelArch::Falcon3);
    }

    #[test]
    #[cfg(feature = "olmo2")]
    fn arch_str_olmo2_both_spellings() {
        assert_eq!(arch_str_to_text_model_arch("OLMo2ForCausalLM"), TextModelArch::OLMo2);
        assert_eq!(arch_str_to_text_model_arch("Olmo2ForCausalLM"), TextModelArch::OLMo2);
    }

    #[test]
    #[cfg(feature = "exaone4")]
    fn arch_str_exaone4() {
        assert_eq!(arch_str_to_text_model_arch("ExaoneForCausalLM"), TextModelArch::EXAONE4);
    }
}
