//! FeedForward (GEGLU-style) for LTX-2 transformer blocks.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_approx(x: &Tensor) -> Result<Tensor> {
    x.gelu()
}

/// GELU + Linear projection (GELUApprox in Python).
#[derive(Debug)]
struct GeluProjection {
    linear: Linear,
}

impl GeluProjection {
    fn new(dim_in: usize, dim_out: usize, vb: VarBuilder) -> Result<Self> {
        let linear = candle_nn::linear(dim_in, dim_out, vb.pp("proj"))?;
        Ok(Self { linear })
    }
}

impl Module for GeluProjection {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.linear.forward(xs)?;
        gelu_approx(&x)
    }
}

/// FeedForward: GELUApprox(dim -> inner_dim) -> Linear(inner_dim -> dim_out)
#[derive(Debug)]
pub struct FeedForward {
    gelu_proj: GeluProjection,
    out_proj: Linear,
}

impl FeedForward {
    pub fn new(dim: usize, dim_out: usize, mult: usize, vb: VarBuilder) -> Result<Self> {
        let inner_dim = dim * mult;
        let gelu_proj = GeluProjection::new(dim, inner_dim, vb.pp("net.0"))?;
        let out_proj = candle_nn::linear(inner_dim, dim_out, vb.pp("net.2"))?;
        Ok(Self {
            gelu_proj,
            out_proj,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.gelu_proj.forward(xs)?;
        self.out_proj.forward(&x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_feed_forward_shape() {
        let device = Device::Cpu;
        let dim = 16;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let ff = FeedForward::new(dim, dim, 4, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, 8, dim), &device).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 8, dim]);
    }

    #[test]
    fn test_feed_forward_different_out_dim() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let ff = FeedForward::new(16, 32, 4, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (2, 4, 16), &device).unwrap();
        let out = ff.forward(&x).unwrap();
        assert_eq!(out.dims(), &[2, 4, 32]);
    }
}

