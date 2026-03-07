use crate::cake::{Context, Forwarder};
use async_trait::async_trait;
use candle_core::{IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use candle_transformers::models::stable_diffusion::clip::ClipTextTransformer;
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use log::info;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct FluxClip {
    clip_model: ClipTextTransformer,
}

impl Display for FluxClip {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "flux-clip (local)")
    }
}

#[async_trait]
impl Forwarder for FluxClip {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(ctx)
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("Flux CLIP forwarding...");

        let output = self.clip_model.forward(x)?;

        // Extract pooled output: embedding at the EOS token position.
        // CLIP's EOS token has the highest ID in the vocabulary, so argmax
        // over the token IDs gives us the EOS position.
        let eos_pos = x.argmax(D::Minus1)?.to_scalar::<u32>()? as usize;
        let pooled = output.i((.., eos_pos, ..))?.contiguous()?;

        Ok(pooled)
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
        "flux-clip"
    }
}

impl FluxClip {
    pub fn load_model(ctx: &Context) -> anyhow::Result<Box<Self>> {
        let variant = ctx.args.flux_args.flux_variant;

        let weights_path = super::flux::FluxModelFile::ClipWeights.get(
            ctx.args.flux_args.flux_clip.clone(),
            variant,
            &ctx.args.model,
        )?;

        info!("Loading Flux CLIP from {:?}...", weights_path);

        // Use SDXL's CLIP-L config — same architecture as Flux's CLIP encoder
        let sdxl_config = StableDiffusionConfig::sdxl(None, None, None);
        let clip_config = sdxl_config.clip;

        let clip_model = stable_diffusion::build_clip_transformer(
            &clip_config,
            weights_path,
            &ctx.device,
            ctx.dtype,
        )?;

        info!("Flux CLIP loaded!");

        Ok(Box::new(Self { clip_model }))
    }

    pub async fn encode(
        forwarder: &mut Box<dyn Forwarder>,
        tokens: Tensor,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        forwarder.forward_mut(&tokens, 0, 0, ctx).await
    }
}
