//! ConvolutionModule -- pointwise (GLU) -> depthwise conv -> SwooshR -> pointwise.
//!
//! Weight names: `in_proj.{weight,bias}`, `depthwise_conv.{weight,bias}`, `out_proj.{weight,bias}`.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Linear, Module, VarBuilder};

#[derive(Debug, Clone)]
pub struct ConvolutionModule {
    in_proj: Linear,
    depthwise_weight: Tensor,
    depthwise_bias: Tensor,
    out_proj: Linear,
    kernel_size: usize,
    dim: usize,
}

impl ConvolutionModule {
    pub fn load(dim: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        // in_proj projects to 2*dim for GLU gating
        let in_proj = candle_nn::linear(dim, 2 * dim, vb.pp("in_proj"))?;
        let depthwise_weight = vb.get((dim, 1, kernel_size), "depthwise_conv.weight")?;
        let depthwise_bias = vb.get(dim, "depthwise_conv.bias")?;
        let out_proj = candle_nn::linear(dim, dim, vb.pp("out_proj"))?;
        Ok(Self {
            in_proj,
            depthwise_weight,
            depthwise_bias,
            out_proj,
            kernel_size,
            dim,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq, dim]
        let x = self.in_proj.forward(x)?;

        // GLU gate: split into two halves along last dim
        let half = self.dim;
        let a = x.narrow(candle_core::D::Minus1, 0, half)?;
        let b = x.narrow(candle_core::D::Minus1, half, half)?;
        let gate = candle_nn::ops::sigmoid(&b)?;
        let x = (a * gate)?;

        // Depthwise conv1d: transpose to [batch, dim, seq], apply, transpose back
        let x = x.transpose(1, 2)?; // [batch, dim, seq]
        let x = self.depthwise_conv1d(&x)?;
        let x = x.transpose(1, 2)?; // [batch, seq, dim]

        // SwooshR activation before out_proj (out_proj is ActivationDropoutAndLinear with SwooshR)
        let x = super::activations::swoosh_r(&x)?;
        let x = self.out_proj.forward(&x)?;
        Ok(x)
    }

    /// Manual depthwise conv1d using broadcast_mul + sum pattern.
    fn depthwise_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let (_batch, channels, seq_len) = x.dims3()?;
        let pad = self.kernel_size / 2;

        let x = if pad > 0 {
            x.pad_with_zeros(2, pad, pad)?
        } else {
            x.clone()
        };

        let w = self.depthwise_weight.squeeze(1)?; // [channels, kernel_size]

        let mut outputs = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let slice = x.narrow(2, i, self.kernel_size)?;
            let prod = slice.broadcast_mul(&w)?;
            let summed = prod.sum(candle_core::D::Minus1)?;
            outputs.push(summed);
        }
        let result = Tensor::stack(&outputs, 2)?;
        let bias = self.depthwise_bias.reshape((1, channels, 1))?;
        Ok(result.broadcast_add(&bias)?)
    }
}
