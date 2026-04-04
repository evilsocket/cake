// HunyuanVideo LLaMA-based text encoder forwarder.
//
// Layer name: "hunyuan-text"
//
// HunyuanVideo uses a custom LLaMA-based text encoder (not T5-XXL like LTX/Wan).
// Input: token IDs [B, L]
// Output: hidden states [B, L, 4096]
//
// TODO: Load and run the LLaMA-based text encoder weights.

use crate::cake::{Context, Forwarder};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

pub struct HunyuanTextEncoder {
    name: String,
    // TODO: LLaMA-based text encoder weights
}

impl Debug for HunyuanTextEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanTextEncoder[{}]", self.name)
    }
}

impl Display for HunyuanTextEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanTextEncoder[{}] (local, stub)", self.name)
    }
}

#[async_trait]
impl Forwarder for HunyuanTextEncoder {
    fn load(name: String, _ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        log::info!("HunyuanVideo text encoder: stub loaded (weights not yet vendored)");
        Ok(Box::new(Self { name }))
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        // TODO: implement LLaMA-based text encoding
        anyhow::bail!("HunyuanVideo text encoder forward not yet implemented")
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
