use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

/// Wan FFN: Linear(dim -> ffn_dim) + GELU(tanh) + Linear(ffn_dim -> dim).
/// Weight names: ffn.0.{weight,bias}, ffn.2.{weight,bias} (Sequential indexing).
#[derive(Debug, Clone)]
pub struct WanFeedForward {
    linear_in: Linear,
    linear_out: Linear,
}

impl WanFeedForward {
    pub fn load(vb: VarBuilder, dim: usize, ffn_dim: usize) -> Result<Self> {
        let linear_in = candle_nn::linear(dim, ffn_dim, vb.pp("0"))?;
        let linear_out = candle_nn::linear(ffn_dim, dim, vb.pp("2"))?;
        Ok(Self { linear_in, linear_out })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_in.forward(x)?;
        let x = x.gelu_erf()?; // GELU with tanh approximation
        let x = self.linear_out.forward(&x)?;
        Ok(x)
    }
}
