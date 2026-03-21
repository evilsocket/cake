use std::{
    fmt::{Debug, Display},
    path::PathBuf,
    sync::{Arc, Mutex},
};
use tokio::net::TcpListener;

use crate::{
    backends::{self, ComputeBackend},
    models::common::{detect_text_model_arch, Cache, Config},
    utils, Args, ModelType, TextModelArch,
};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

pub mod sharding;

#[cfg(feature = "master")]
pub use sharding::master::*;

pub use sharding::{Strategy, WorkerCapacity, DefaultStrategy};
pub use sharding::topology::*;
pub use sharding::client::*;
pub use sharding::proto::*;
pub use sharding::worker::*;
pub use sharding::discovery;
pub use sharding::auth;

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
    /// Compute backend for fused operations (CPU, CUDA, Metal, Vulkan).
    pub backend: Arc<dyn ComputeBackend>,
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
        #[cfg(feature = "luxtts")]
        "LuxTTSForTextToSpeech" => TextModelArch::LuxTTS,
        _ => TextModelArch::Llama,
    }
}

impl Context {
    /// Create the context from the parsed command line arguments.
    pub fn from_args(mut args: Args) -> Result<Self> {
        #[allow(unused_mut)] // mutated only with vibevoice feature
        let mut dtype = parse_dtype_str(args.dtype.as_deref())?;

        let device = utils::get_inference_device(args.cpu, args.device)
            .map_err(|e| anyhow!("can't attach to device: {:?}", e))?;

        // Disable cudarc event tracking for CUDA devices: cudarc's CudaStream::wait()
        // rejects events from a different CudaContext, which breaks cross-device tensor
        // transfers in multi-GPU setups. Safe since we use a single stream per device.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda_dev) = &device {
            unsafe {
                cuda_dev.disable_event_tracking();
            }
        }

        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(
                memory_stats::memory_stats()
                    .map(|m| m.physical_mem)
                    .unwrap_or(0) as f64
            )
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

        let mut topology = if let Some(topo) = args.topology_override.take() {
            // Zero-config setup already built the topology
            topo
        } else if let Some(path) = &args.topology {
            Topology::from_path(path, &args.model_type)?
        } else {
            log::warn!("no topology file specified, the entire model will be loaded");
            Topology::new()
        };

        // If the topology has nodes with no layer assignments, automatically
        // distribute layers using the TFLOPS-proportional sharding algorithm.
        // This lets users specify worker addresses without manual layer ranges.
        if topology.needs_auto_sharding()
            && (args.model_type == ModelType::TextModel || args.model_type == ModelType::AudioModel)
        {
            let config_path = data_path.join("config.json");
            if config_path.exists() {
                let config_data = std::fs::read_to_string(&config_path)?;
                let config_json: serde_json::Value = serde_json::from_str(&config_data)?;

                let num_layers = config_json
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .or_else(|| {
                        config_json
                            .get("text_config")
                            .and_then(|tc| tc.get("num_hidden_layers"))
                            .and_then(|v| v.as_u64())
                    })
                    .unwrap_or(0) as usize;

                if num_layers > 0 {
                    let layer_prefix = sharding::default::layer_prefix_for_config(&config_json);
                    let layer_size =
                        sharding::default::estimate_layer_size(&data_path, num_layers, &layer_prefix);

                    // Detect master GPU for TFLOPS estimate
                    let master_gpus = discovery::detect_gpus();
                    let master_tflops: f64 = master_gpus.iter().map(|g| g.tflops as f64).sum();

                    log::info!(
                        "topology has {} worker(s) with no layer assignments — auto-sharding {} layers",
                        topology.len(),
                        num_layers,
                    );

                    topology.auto_assign_layers(
                        num_layers,
                        master_tflops,
                        layer_size,
                        usize::MAX, // no master VRAM cap for topology mode
                        &layer_prefix,
                    );

                    // Log final assignments
                    for (name, node) in topology.iter() {
                        if !node.layers.is_empty() {
                            log::info!(
                                "  {} → {} layers ({} — {})",
                                name,
                                node.layers.len(),
                                node.layers.first().unwrap(),
                                node.layers.last().unwrap(),
                            );
                        }
                    }
                }
            }
        }

        let mut config: Option<Config> = None;
        let mut cache: Option<Cache> = None;
        let mut var_builder: Option<VarBuilder> = None;
        let mut text_model_arch = args.text_model_arch;
        let mut quant: Arc<dyn utils::Quantization> = Arc::new(utils::NoQuantization);

        if args.model_type == ModelType::TextModel {
            // Check if the model path is a GGUF file
            if utils::gguf::is_gguf_file(&data_path) {
                // GGUF path: extract config and architecture from GGUF metadata
                let arch_str = utils::gguf::arch_from_gguf(&data_path)?;
                text_model_arch = arch_str_to_text_model_arch(&arch_str);
                log::info!(
                    "GGUF model: architecture={:?} (from metadata)",
                    text_model_arch
                );

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
                        crate::models::qwen3_5::Qwen3_5Config::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "qwen3")]
                    TextModelArch::Qwen3 => {
                        crate::models::qwen3::Qwen3Config::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "qwen3_moe")]
                    TextModelArch::Qwen3Moe => {
                        crate::models::qwen3_moe::Qwen3MoeConfig::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "qwen3_5_moe")]
                    TextModelArch::Qwen3_5Moe => {
                        crate::models::qwen3_5_moe::Qwen3_5MoeConfig::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "phi4")]
                    TextModelArch::Phi4 => {
                        crate::models::phi4::Phi4Config::from_path(&config_filename)?.into_config()
                    }
                    #[cfg(feature = "mistral")]
                    TextModelArch::Mistral => {
                        crate::models::mistral::MistralConfig::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "gemma3")]
                    TextModelArch::Gemma3 => {
                        crate::models::gemma3::Gemma3Config::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "falcon3")]
                    TextModelArch::Falcon3 => {
                        crate::models::falcon3::Falcon3Config::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "olmo2")]
                    TextModelArch::OLMo2 => {
                        crate::models::olmo2::OLMo2Config::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "exaone4")]
                    TextModelArch::EXAONE4 => {
                        crate::models::exaone4::EXAONE4Config::from_path(&config_filename)?
                            .into_config()
                    }
                    #[cfg(feature = "luxtts")]
                    TextModelArch::LuxTTS => {
                        let luxtts_cfg =
                            crate::models::luxtts::LuxTTSConfig::from_path(&config_filename)?;
                        // Create a minimal common Config for the framework
                        Config {
                            hidden_size: luxtts_cfg.model.fm_decoder_dim,
                            intermediate_size: luxtts_cfg.model.fm_decoder_feedforward_dim,
                            vocab_size: luxtts_cfg.model.vocab_size,
                            num_hidden_layers: luxtts_cfg.total_fm_layers(),
                            num_attention_heads: luxtts_cfg.model.fm_decoder_num_heads,
                            num_key_value_heads: luxtts_cfg.model.fm_decoder_num_heads,
                            rms_norm_eps: 1e-5,
                            rope_theta: 10000.0,
                            bos_token_id: None,
                            eos_token_id: None,
                            rope_scaling: None,
                            tie_word_embeddings: false,
                            max_seq_len: 4096,
                            use_qkv_bias: false,
                            model_prefix: "fm_decoder".to_string(),
                            head_dim: None,
                            partial_rotary_factor: 1.0,
                            linear_attn: None,
                            residual_rms_norm: false,
                            use_qk_norm: false,
                            pre_reshape_qk_norm: false,
                            sliding_window: None,
                            fused_qkv_proj: false,
                            fused_gate_up_proj: false,
                            global_layers: vec![],
                            use_gelu_mlp: false,
                            embed_scale: None,
                            moe_intermediate_size: None,
                            num_experts: 0,
                            num_experts_per_tok: 0,
                            norm_topk_prob: false,
                            shared_expert_intermediate_size: None,
                            attn_output_gate: false,
                        }
                    }
                    #[cfg(feature = "llama")]
                    TextModelArch::Llama => {
                        crate::models::llama3::LlamaConfig::from_path(&config_filename)?
                            .into_config()
                    }
                    _ => {
                        bail!(
                            "no text model feature enabled for architecture {:?}",
                            text_model_arch
                        )
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
                            model_tensors_index,
                            dtype,
                            device.clone(),
                            &*quant,
                        )?
                    } else {
                        utils::load_var_builder_for_local_layers(
                            model_tensors_index,
                            dtype,
                            device.clone(),
                            &worker_layers,
                            &*quant,
                        )?
                    }
                } else if !my_layers.is_empty() {
                    utils::load_var_builder_for_specific_layers(
                        model_tensors_index,
                        dtype,
                        device.clone(),
                        &my_layers,
                        &*quant,
                    )?
                } else {
                    utils::load_var_builder_from_index(
                        model_tensors_index,
                        dtype,
                        device.clone(),
                        &*quant,
                    )?
                });
                cache = Some(Cache::new(true, dtype, &config_internal, &device)?);
                config = Some(config_internal);
            } // end else (safetensors path)
        }

        // AudioModel workers need a VarBuilder and config to load transformer layers.
        // Parse config.json for the LM backbone and create VarBuilder from safetensors.
        #[cfg(feature = "vibevoice")]
        if args.model_type == ModelType::AudioModel
            && matches!(args.mode, Mode::Worker)
            && var_builder.is_none()
        {
            let config_filename = data_path.join("config.json");
            if config_filename.exists() {
                // VibeVoice models: parse decoder_config for the LM backbone
                text_model_arch = TextModelArch::Qwen2;
                let config_internal =
                    crate::models::vibevoice::config_1_5b::VibeVoice1_5BConfig::from_path(
                        &config_filename,
                    )
                    .map(|c| c.into_config())
                    .or_else(|_| {
                        crate::models::vibevoice::config::VibeVoiceConfig::from_path(
                            &config_filename,
                        )
                        .map(|c| c.into_config())
                    })
                    .ok();
                if let Some(ref cfg) = config_internal {
                    // VibeVoice-1.5B uses BF16 — detect from decoder_config.torch_dtype
                    // to match the master's dtype (avoids BF16 vs F16 mismatch).
                    if let Ok(raw) = std::fs::read_to_string(&config_filename) {
                        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&raw) {
                            let torch_dtype = json
                                .get("decoder_config")
                                .and_then(|d| d.get("torch_dtype"))
                                .and_then(|v| v.as_str());
                            if torch_dtype == Some("bfloat16") {
                                log::info!("AudioModel worker: using BF16 dtype (from decoder_config.torch_dtype)");
                                dtype = candle_core::DType::BF16;
                            }
                        }
                    }

                    let model_tensors_index = data_path.join("model.safetensors.index.json");
                    quant = Arc::from(utils::detect_quantization(&config_filename));
                    let my_layers: Vec<String> = topology.all_worker_layers().into_iter().collect();
                    var_builder = Some(if !my_layers.is_empty() {
                        utils::load_var_builder_for_specific_layers(
                            model_tensors_index,
                            dtype,
                            device.clone(),
                            &my_layers,
                            &*quant,
                        )?
                    } else {
                        utils::load_var_builder_from_index(
                            model_tensors_index,
                            dtype,
                            device.clone(),
                            &*quant,
                        )?
                    });
                    cache = Some(Cache::new(true, dtype, cfg, &device)?);
                    config = Some(cfg.clone());
                    log::info!("AudioModel worker: loaded Qwen2 config + VarBuilder for LM layers");
                }
            }
        }

        let backend = backends::create_backend(&device);

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
            backend,
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
        assert_eq!(
            arch_str_to_text_model_arch("UnknownArchXYZ"),
            TextModelArch::Llama
        );
        assert_eq!(arch_str_to_text_model_arch(""), TextModelArch::Llama);
    }

    #[test]
    #[cfg(feature = "qwen2")]
    fn arch_str_qwen2() {
        assert_eq!(
            arch_str_to_text_model_arch("Qwen2ForCausalLM"),
            TextModelArch::Qwen2
        );
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
        assert_eq!(
            arch_str_to_text_model_arch("Qwen3ForCausalLM"),
            TextModelArch::Qwen3
        );
    }

    #[test]
    #[cfg(feature = "qwen3_moe")]
    fn arch_str_qwen3_moe() {
        assert_eq!(
            arch_str_to_text_model_arch("Qwen3MoeForCausalLM"),
            TextModelArch::Qwen3Moe
        );
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
        assert_eq!(
            arch_str_to_text_model_arch("Phi3ForCausalLM"),
            TextModelArch::Phi4
        );
        assert_eq!(
            arch_str_to_text_model_arch("Phi4ForCausalLM"),
            TextModelArch::Phi4
        );
    }

    #[test]
    #[cfg(feature = "mistral")]
    fn arch_str_mistral() {
        assert_eq!(
            arch_str_to_text_model_arch("MistralForCausalLM"),
            TextModelArch::Mistral
        );
    }

    #[test]
    #[cfg(feature = "gemma3")]
    fn arch_str_gemma3() {
        assert_eq!(
            arch_str_to_text_model_arch("Gemma3ForCausalLM"),
            TextModelArch::Gemma3
        );
    }

    #[test]
    #[cfg(feature = "falcon3")]
    fn arch_str_falcon3() {
        assert_eq!(
            arch_str_to_text_model_arch("FalconForCausalLM"),
            TextModelArch::Falcon3
        );
    }

    #[test]
    #[cfg(feature = "olmo2")]
    fn arch_str_olmo2_both_spellings() {
        assert_eq!(
            arch_str_to_text_model_arch("OLMo2ForCausalLM"),
            TextModelArch::OLMo2
        );
        assert_eq!(
            arch_str_to_text_model_arch("Olmo2ForCausalLM"),
            TextModelArch::OLMo2
        );
    }

    #[test]
    #[cfg(feature = "exaone4")]
    fn arch_str_exaone4() {
        assert_eq!(
            arch_str_to_text_model_arch("ExaoneForCausalLM"),
            TextModelArch::EXAONE4
        );
    }

    #[test]
    #[cfg(feature = "luxtts")]
    fn arch_str_luxtts() {
        assert_eq!(
            arch_str_to_text_model_arch("LuxTTSForTextToSpeech"),
            TextModelArch::LuxTTS
        );
    }
}
