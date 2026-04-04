// HunyuanVideo dual-stream DiT transformer forwarder.
//
// Layer name: "hunyuan-transformer"
//
// TODO: Implement actual dual-stream (MMDiT) forward pass:
//   - 20 double-stream blocks (joint image+text attention)
//   - 40 single-stream blocks (merged sequence)
//   - 3D RoPE for temporal/height/width
//   - Patch embedding and unpatchify

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::{pack_tensors, unpack_tensors};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

pub struct HunyuanTransformer {
    name: String,
    // TODO: actual transformer weights (dual-stream DiT blocks)
}

impl Debug for HunyuanTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanTransformer[{}]", self.name)
    }
}

impl Display for HunyuanTransformer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanTransformer[{}] (local, stub)", self.name)
    }
}

#[async_trait]
impl Forwarder for HunyuanTransformer {
    fn load(name: String, _ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        log::info!("HunyuanVideo transformer: stub loaded (weights not yet vendored)");
        Ok(Box::new(Self { name }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        // Protocol: unpack [latents, timestep, context, num_frames, height, width]
        // TODO: implement actual dual-stream DiT forward pass
        // For now, return latents unchanged (identity)
        let unpacked = unpack_tensors(x)?;
        let latents = &unpacked[0];
        log::warn!("HunyuanVideo transformer forward: STUB - returning latents unchanged");
        Ok(latents.clone())
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

impl HunyuanTransformer {
    /// Pack inputs for RPC transfer, matching the Wan pattern.
    pub async fn forward_packed(
        transformer: &mut Box<dyn Forwarder>,
        latents: Tensor,
        timestep: Tensor,
        context: Tensor,
        num_frames: usize,
        height: usize,
        width: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let device = &ctx.device;
        let tensors = vec![
            latents,
            timestep,
            context,
            Tensor::from_slice(&[num_frames as f32], 1, device)?,
            Tensor::from_slice(&[height as f32], 1, device)?,
            Tensor::from_slice(&[width as f32], 1, device)?,
        ];
        let combined = pack_tensors(tensors, device)?;
        transformer.forward_mut(&combined, 0, 0, ctx).await
    }
}
