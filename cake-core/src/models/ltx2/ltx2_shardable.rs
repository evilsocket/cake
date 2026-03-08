use crate::cake::{Context, Forwarder};
use super::gemma::Ltx2Gemma;
use super::transformer::Ltx2Transformer;
use super::vae_forwarder::Ltx2Vae;
use super::vocoder::Ltx2Vocoder;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

/// Dispatches layer names to the appropriate LTX-2 component:
/// - `"ltx2-transformer"` → Dual-stream DiT (14B video + 5B audio)
/// - `"ltx2-gemma"` → Gemma-3 12B text encoder
/// - `"ltx2-vae"` → Video VAE decoder
/// - `"ltx2-vocoder"` → Audio vocoder
#[derive(Debug)]
pub struct Ltx2Shardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for Ltx2Shardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for Ltx2Shardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder> = if name == "ltx2-transformer"
            || name.starts_with("ltx2-transformer.")
        {
            Ltx2Transformer::load(name.clone(), ctx)?
        } else {
            match name.as_str() {
                "ltx2-gemma" => Ltx2Gemma::load(name.clone(), ctx)?,
                "ltx2-vae" => Ltx2Vae::load(name.clone(), ctx)?,
                "ltx2-vocoder" => Ltx2Vocoder::load(name.clone(), ctx)?,
                _ => anyhow::bail!("LTX-2 component name not recognized: {}", name),
            }
        };

        Ok(Box::new(Self {
            forwarder: model,
            layer_name: name,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder.forward(x, index_pos, block_idx, ctx).await
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder
            .forward_mut(x, index_pos, block_idx, ctx)
            .await
    }

    async fn forward_batch(
        &mut self,
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder.forward_batch(x, batch, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn ident(&self) -> &str {
        &self.layer_name
    }
}
