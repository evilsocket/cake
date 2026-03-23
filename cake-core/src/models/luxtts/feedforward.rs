//! FeedforwardModule -- Linear -> SwooshR -> Linear.
//!
//! Weight names: `in_proj.{weight,bias}`, `out_proj.{weight,bias}`.

use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

use super::activations::swoosh_l;

#[derive(Debug, Clone)]
pub struct FeedforwardModule {
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    out_proj_weight: Tensor,
    out_proj_bias: Option<Tensor>,
    backend: Arc<dyn ComputeBackend>,
}

impl FeedforwardModule {
    /// Load a feed-forward module.
    /// `dim` is the model dimension, `ff_dim` is the intermediate dimension.
    pub fn load(dim: usize, ff_dim: usize, vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let in_proj_weight = vb.pp("in_proj").get((ff_dim, dim), "weight")?;
        let in_proj_bias = Some(vb.pp("in_proj").get(ff_dim, "bias")?);
        let out_proj_weight = vb.pp("out_proj").get((dim, ff_dim), "weight")?;
        let out_proj_bias = Some(vb.pp("out_proj").get(dim, "bias")?);
        Ok(Self { in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias, backend })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.backend.linear_forward(x, &self.in_proj_weight, self.in_proj_bias.as_ref())?;
        // Python uses: in_proj -> Balancer (no-op at inference) -> SwooshL -> Linear
        let x = swoosh_l(&x)?;
        let x = self.backend.linear_forward(&x, &self.out_proj_weight, self.out_proj_bias.as_ref())?;
        Ok(x)
    }
}
