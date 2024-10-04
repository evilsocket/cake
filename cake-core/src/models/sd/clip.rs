use crate::cake::{Context, Forwarder};
use crate::models::sd::sd::ModelFile;
use crate::models::sd::util::get_sd_config;
use crate::StableDiffusionVersion;
use async_trait::async_trait;
use candle_core::{DType, Device, Module, Tensor};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use log::info;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct Clip {
    clip_model: ClipTextTransformer,
    layer_name: &'static str,
}

impl Display for Clip {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for Clip {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
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
            }
            "clip2" => {
                model_file = ModelFile::Clip2;
                model_filename = ctx.args.sd_args.clip2.clone();
                clip_config = sd_config.clip2.unwrap();
            }
            _ => {
                anyhow::bail!("name not recognized");
            }
        };

        Self::load_model(
            model_file,
            model_filename,
            ctx.args.sd_args.sd_version,
            ctx.args.sd_args.use_f16,
            &ctx.device,
            ctx.dtype,
            ctx.args.model.clone(),
            &clip_config,
        )
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("Clip model forwarding");
        Ok(self
            .clip_model
            .forward(x)
            .expect("Error running Clip forward"))
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
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
        config: &stable_diffusion::clip::Config,
    ) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let clip_weights = model_file.get(name, version, use_f16, cache_dir)?;
        let clip_model =
            stable_diffusion::build_clip_transformer(config, clip_weights, device, dtype)?;
        let layer_name = model_file.name();

        info!("Loading Clip model: {layer_name}");

        Ok(Box::new(Self {
            clip_model,
            layer_name,
        }))
    }
}
