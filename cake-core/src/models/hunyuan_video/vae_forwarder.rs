// HunyuanVideo 3D causal VAE forwarder.
//
// Layer name: "hunyuan-vae"
// Direction flag: 0.0 = decode (same convention as Wan/LTX)
//
// The HunyuanVideo VAE is a 3D causal video autoencoder that compresses
// video frames into a latent space with 16 channels.
// Spatial compression: 8x, temporal compression: 4x.
//
// TODO: Load and run the 3D VAE decoder weights.

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::unpack_tensors;
use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

pub struct HunyuanVae {
    name: String,
    // TODO: 3D causal VAE decoder weights
}

impl Debug for HunyuanVae {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanVae[{}]", self.name)
    }
}

impl Display for HunyuanVae {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "HunyuanVae[{}] (local, stub)", self.name)
    }
}

#[async_trait]
impl Forwarder for HunyuanVae {
    fn load(name: String, _ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        log::info!("HunyuanVideo VAE: stub loaded (weights not yet vendored)");
        Ok(Box::new(Self { name }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        let unpacked = unpack_tensors(x)?;
        let direction = unpacked[0].to_vec1::<f32>()?[0];
        let _input = &unpacked[1];

        if direction == 1.0 {
            anyhow::bail!("HunyuanVideo VAE encode not implemented (decode-only for T2V)")
        } else {
            // TODO: implement actual 3D causal VAE decode
            anyhow::bail!("HunyuanVideo VAE decode not yet implemented")
        }
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
