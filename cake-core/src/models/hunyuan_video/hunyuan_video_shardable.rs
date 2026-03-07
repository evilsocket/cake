use crate::cake::{Context, Forwarder};
use super::clip::HunyuanClip;
use super::t5::HunyuanT5;
use super::transformer::HunyuanTransformer;
use super::vae_forwarder::HunyuanVae;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

/// Dispatches layer names to the appropriate HunyuanVideo component:
/// - `"hunyuan-transformer"` → DiT transformer
/// - `"hunyuan-t5"` → T5-XXL text encoder
/// - `"hunyuan-clip"` → CLIP-L text encoder
/// - `"hunyuan-vae"` → 3D VAE decoder
#[derive(Debug)]
pub struct HunyuanVideoShardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for HunyuanVideoShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for HunyuanVideoShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder> = match name.as_str() {
            "hunyuan-transformer" => HunyuanTransformer::load(name.clone(), ctx)?,
            "hunyuan-t5" => HunyuanT5::load(name.clone(), ctx)?,
            "hunyuan-clip" => HunyuanClip::load(name.clone(), ctx)?,
            "hunyuan-vae" => HunyuanVae::load(name.clone(), ctx)?,
            _ => anyhow::bail!("HunyuanVideo component name not recognized: {}", name),
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
