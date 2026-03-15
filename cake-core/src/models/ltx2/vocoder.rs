use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};

/// LTX-2 audio vocoder Forwarder.
///
/// Layer name: `"ltx2-vocoder"`
///
/// Converts latent audio representations to waveform audio,
/// synchronized with the generated video.
#[derive(Debug)]
pub struct Ltx2Vocoder {
    name: String,
}

impl std::fmt::Display for Ltx2Vocoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

impl Ltx2Vocoder {
    pub fn load_model(_ctx: &Context) -> Result<Box<dyn Forwarder>> {
        log::warn!("LTX-2 vocoder: vendored model code not yet implemented");
        Ok(Box::new(Self {
            name: "ltx2-vocoder".to_string(),
        }))
    }
}

#[async_trait]
impl Forwarder for Ltx2Vocoder {
    fn load(name: String, _ctx: &Context) -> Result<Box<Self>> {
        Ok(Box::new(Self { name }))
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        anyhow::bail!("LTX-2 vocoder forward not yet implemented")
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
