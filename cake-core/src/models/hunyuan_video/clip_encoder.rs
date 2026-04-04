// HunyuanVideo CLIP text encoder forwarder.
//
// Layer name: "hunyuan-clip"
//
// HunyuanVideo uses a CLIP text encoder as a secondary encoder alongside
// the LLaMA-based primary encoder. CLIP provides pooled text embeddings
// that condition the DiT via adaptive layer norm.
//
// Input: token IDs [B, L]
// Output: pooled embedding [B, 768]
//
// TODO: Load and run the CLIP encoder weights.

use crate::cake::{Context, Forwarder};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

pub struct HunyuanClipEncoder {
    name: String,
    // TODO: CLIP text encoder weights
}

impl Debug for HunyuanClipEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanClipEncoder[{}]", self.name)
    }
}

impl Display for HunyuanClipEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanClipEncoder[{}] (local, stub)", self.name)
    }
}

#[async_trait]
impl Forwarder for HunyuanClipEncoder {
    fn load(name: String, _ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        log::info!("HunyuanVideo CLIP encoder: stub loaded (weights not yet vendored)");
        Ok(Box::new(Self { name }))
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        // TODO: implement CLIP text encoding
        anyhow::bail!("HunyuanVideo CLIP encoder forward not yet implemented")
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}
