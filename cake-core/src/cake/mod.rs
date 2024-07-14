use std::{
    fmt::{Debug, Display},
    path::PathBuf,
};

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::{
    model::{Cache, Config, LlamaConfig},
    utils, Args,
};

mod client;
mod master;
mod proto;
mod topology;
mod worker;

pub use client::*;
pub use master::*;
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
    pub config: Config,
    pub cache: Cache,
    pub var_builder: VarBuilder<'static>,
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

        log::info!("loading topology from {}", &args.topology);

        let data_path = PathBuf::from(&args.model);

        let config_filename = data_path.join("config.json");
        let config = LlamaConfig::from_path(&config_filename)?.into_config();

        let topology = Topology::from_path(&args.topology)?;
        let cache = Cache::new(true, dtype, &config, &device)?;

        let model_tensors_index: PathBuf = data_path.join("model.safetensors.index.json");
        let var_builder =
            utils::load_var_builder_from_index(model_tensors_index, dtype, device.clone())?;

        Ok(Context {
            args,
            dtype,
            topology,
            data_path,
            device,
            config,
            cache,
            var_builder,
        })
    }
}

#[async_trait]
pub trait Forwarder: Debug + Send + Sync + Display {
    fn load(name: String, vb: VarBuilder, cfg: &Config) -> Result<Box<Self>>
    where
        Self: Sized;

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor>;

    async fn forward_batch(
        &mut self,
        _x: &Tensor,
        _batch: Vec<(String, usize, usize)>,
        _cache: &mut Cache,
    ) -> Result<Tensor> {
        unimplemented!()
    }

    fn layer_name(&self) -> &str;

    fn ident(&self) -> &str {
        "local"
    }
}
