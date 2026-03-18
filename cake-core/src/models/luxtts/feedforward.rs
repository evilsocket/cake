//! FeedforwardModule -- Linear -> SwooshR -> Linear.
//!
//! Weight names: `in_proj.{weight,bias}`, `out_proj.{weight,bias}`.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

use super::activations::swoosh_l;

#[derive(Debug, Clone)]
pub struct FeedforwardModule {
    in_proj: Linear,
    out_proj: Linear,
}

impl FeedforwardModule {
    /// Load a feed-forward module.
    /// `dim` is the model dimension, `ff_dim` is the intermediate dimension.
    pub fn load(dim: usize, ff_dim: usize, vb: VarBuilder) -> Result<Self> {
        let in_proj = candle_nn::linear(dim, ff_dim, vb.pp("in_proj"))?;
        let out_proj = candle_nn::linear(ff_dim, dim, vb.pp("out_proj"))?;
        Ok(Self { in_proj, out_proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.in_proj.forward(x)?;
        // Python uses: in_proj -> Balancer (no-op at inference) -> SwooshL -> Linear
        let x = swoosh_l(&x)?;
        let x = self.out_proj.forward(&x)?;
        Ok(x)
    }
}
