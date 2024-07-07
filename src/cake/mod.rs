use std::{
    collections::HashMap,
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
mod worker;

pub use client::*;
pub use master::*;
pub use proto::*;
pub use worker::*;

#[derive(clap::ValueEnum, Clone, Debug, Default)]
pub enum Mode {
    #[default]
    Master,
    Worker,
}

pub type Topology = HashMap<String, String>;

pub struct Context {
    pub args: Args,
    pub data_path: PathBuf,
    pub device: Device,
    pub config: Config,
    pub cache: Cache,
    pub var_builder: VarBuilder<'static>,
}

impl Context {
    pub fn from_args(args: Args) -> Result<Self> {
        let dtype = match args.dtype.as_deref() {
            Some("f16") => DType::F16,
            Some("bf16") => DType::BF16,
            Some("f32") => DType::F32,
            Some(dtype) => bail!("unsupported dtype {dtype}"),
            None => DType::F16,
        };

        let device = utils::get_inference_device(args.cpu, args.device)?;

        log::info!(
            "[{:?}] dtype={:?} device={:?} mem={}",
            args.mode,
            &dtype,
            &device,
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let data_path = PathBuf::from(&args.model);
        let config_filename = data_path.join("config.json");
        let model_tensors_index = data_path.join("model.safetensors.index.json");

        log::info!("loading configuration from {}", config_filename.display());

        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config();

        let cache = Cache::new(true, dtype, &config, &device)?;

        log::info!("loading tensors from {} ...", model_tensors_index.display());

        let filenames: Vec<std::path::PathBuf> =
            utils::load_safetensors_from_index(model_tensors_index)?;

        let var_builder =
            unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

        Ok(Context {
            args,
            data_path,
            device,
            config,
            cache,
            var_builder,
        })
    }
}

#[async_trait]
pub(crate) trait Forwarder: Debug + Send + Display {
    async fn forward(
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
