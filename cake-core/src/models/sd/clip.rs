use std::fmt::{Debug, Display, Formatter};
use async_trait::async_trait;
use candle_core::{Device, DType, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use crate::cake::{Context, Forwarder};
use crate::models::llama3::Cache;
use crate::models::sd::sd::ModelFile;
use crate::models::sd::util::{get_device, get_sd_config};
use crate::StableDiffusionVersion;

pub struct Clip {
    clip_model: ClipTextTransformer,
    layer_name: &'static str
}

impl Debug for Clip {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

impl Display for Clip {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[async_trait]
impl Forwarder for Clip {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        let model_file;
        let model_filename;
        let sd_config = get_sd_config(ctx)?;
        let clip_config;

        match name.as_str() {
            "clip" => {
                model_file = ModelFile::Clip;
                model_filename = ctx.args.sd_args.clip.clone();
                clip_config = sd_config.clip;
            },
            "clip2" => {
                model_file = ModelFile::Clip2;
                model_filename = ctx.args.sd_args.clip2.clone();
                clip_config = sd_config.clip2.unwrap();
            }
            _ => {
                anyhow::bail!("name not recognized");
            }
        };

        let dtype = if ctx.args.sd_args.use_f16 { DType::F16 } else { DType::F32 };
        let device = get_device(ctx.args.cpu)?;

        Self::load_model(
            model_file,
            model_filename,
            ctx.args.sd_args.sd_version,
            ctx.args.sd_args.use_f16,
            &device,
            dtype,
            ctx.args.model.clone(),
            &clip_config
        )
    }

    async fn forward(&self, x: &Tensor, _index_pos: usize, _block_idx: usize, _cache: Option<&mut Cache>) -> anyhow::Result<Tensor> {
        Ok(self.clip_model.forward(x).expect("Error running Clip forward"))
    }

    async fn forward_mut(&mut self, x: &Tensor, index_pos: usize, block_idx: usize, cache: Option<&mut Cache>) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, cache).await
    }

    fn layer_name(&self) -> &str {
        self.layer_name
    }
}

impl Clip {
    pub fn load_model(
        model_file: ModelFile,
        name: Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
        device: &Device,
        dtype: DType,
        cache_dir: String,
        config: &stable_diffusion::clip::Config) -> anyhow::Result<Box<Self>>
    where
        Self: Sized
    {
        let clip_weights = model_file.get(name, version, use_f16, cache_dir)?;
        let clip_model = stable_diffusion::build_clip_transformer(config, clip_weights, device, dtype)?;
        let layer_name = model_file.name();
        Ok(Box::new(Self {
            clip_model,
            layer_name,
        }))
    }
}
