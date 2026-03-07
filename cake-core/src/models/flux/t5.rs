use crate::cake::{Context, Forwarder};
use async_trait::async_trait;
use candle_core::Tensor;
use candle_transformers::models::t5::{self, T5EncoderModel};
use log::info;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct FluxT5 {
    model: T5EncoderModel,
}

impl Display for FluxT5 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "flux-t5 (local)")
    }
}

#[async_trait]
impl Forwarder for FluxT5 {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(ctx)
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        anyhow::bail!("T5 encoder requires forward_mut (has KV cache)")
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("T5 encoder forwarding...");
        Ok(self.model.forward(x)?)
    }

    fn layer_name(&self) -> &str {
        "flux-t5"
    }
}

impl FluxT5 {
    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let variant = ctx.args.flux_args.flux_variant;

        // Load T5 config
        let config_path = super::flux::FluxModelFile::T5Config.get(
            ctx.args.flux_args.flux_t5_config.clone(),
            variant,
            &ctx.args.model,
        )?;

        info!("Loading T5 config from {:?}...", config_path);
        let config: t5::Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

        // Load T5 weights (potentially sharded)
        let weight_files = super::flux::get_t5_weight_files(
            ctx.args.flux_args.flux_t5.clone(),
            variant,
            &ctx.args.model,
        )?;

        info!("Loading T5 encoder from {:?}...", weight_files);

        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, ctx.dtype, &ctx.device)?
        };
        let model = T5EncoderModel::load(vb, &config)?;

        info!("T5 encoder loaded!");

        Ok(Box::new(Self { model }))
    }

    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        tokens: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        forwarder.forward_mut(&tokens, 0, 0, ctx).await
    }
}
