// UMT5-XXL text encoder forwarder for Wan2.2.
//
// Layer name: "wan-t5"
// Input: token IDs [B, L]
// Output: hidden states [B, L, 4096]

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::pack_tensors;
use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct WanT5Encoder {
    // TODO: implement UMT5-XXL model
    // For now this is a placeholder
}

impl Display for WanT5Encoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "WanT5Encoder (local)")
    }
}

#[async_trait]
impl Forwarder for WanT5Encoder {
    fn load(_name: String, _ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        anyhow::bail!("WanT5Encoder::load not yet implemented")
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        anyhow::bail!("WanT5Encoder::forward not yet implemented")
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
        "wan-t5"
    }
}
