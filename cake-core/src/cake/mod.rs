use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

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

mod client;
mod proto;
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
}

impl Context {
    /// Create the context from the parsed command line arguments.
    pub fn from_args(args: Args) -> Result<Self> {
        let dtype: DType = match args.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => bail!("unsupported dtype {dtype}"),
            None => DType::F16,
        };

        let device = utils::get_inference_device(args.cpu, args.device)
            .map_err(|e| anyhow!("can't attach to device: {:?}", e))?;

        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let data_path = PathBuf::from(&args.model);
        if !data_path.exists() {
            bail!("model path does not exist: {}", data_path.display());
        }

        let topology = if let Some(path) = &args.topology {
            Topology::from_path(path, &args.model_type)?
        } else {
            log::warn!("no topology file specified, the entire model will be loaded");
            Topology::new()
        };

        let mut config: Option<Config> = None;
        let mut cache: Option<Cache> = None;
        let mut var_builder: Option<VarBuilder> = None;
        let mut text_model_arch = args.text_model_arch;

        if args.model_type == ModelType::TextModel {
            let config_filename = data_path.join("config.json");

            // Auto-detect architecture if needed
            if text_model_arch == TextModelArch::Auto {
                let arch_str = detect_text_model_arch(&config_filename).unwrap_or_default();
                text_model_arch = match arch_str.as_str() {
                    #[cfg(feature = "qwen2")]
                    "Qwen2ForCausalLM" => TextModelArch::Qwen2,
                    _ => TextModelArch::Llama,
                };
            }

            log::info!("text model architecture: {:?}", text_model_arch);

            let config_internal = match text_model_arch {
                #[cfg(feature = "qwen2")]
                TextModelArch::Qwen2 => {
                    crate::models::qwen2::QwenConfig::from_path(&config_filename)?.into_config()
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

            let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
            var_builder = Some(utils::load_var_builder_from_index(
                model_tensors_index,
                dtype,
                device.clone(),
            )?);
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
