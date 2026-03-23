//! ConvolutionModule -- pointwise (GLU) -> depthwise conv -> SwooshR -> pointwise.
//!
//! Weight names: `in_proj.{weight,bias}`, `depthwise_conv.{weight,bias}`, `out_proj.{weight,bias}`.

use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

#[derive(Debug, Clone)]
pub struct ConvolutionModule {
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    depthwise_weight: Tensor,
    depthwise_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Option<Tensor>,
    kernel_size: usize,
    dim: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl ConvolutionModule {
    pub fn load(dim: usize, kernel_size: usize, vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        // in_proj projects to 2*dim for GLU gating
        let in_proj_weight = vb.pp("in_proj").get((2 * dim, dim), "weight")?;
        let in_proj_bias = Some(vb.pp("in_proj").get(2 * dim, "bias")?);
        let depthwise_weight = vb.get((dim, 1, kernel_size), "depthwise_conv.weight")?;
        let depthwise_bias = vb.get(dim, "depthwise_conv.bias")?;
        let out_proj_weight = vb.pp("out_proj").get((dim, dim), "weight")?;
        let out_proj_bias = Some(vb.pp("out_proj").get(dim, "bias")?);
        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            depthwise_weight,
            depthwise_bias,
            out_proj_weight,
            out_proj_bias,
            kernel_size,
            dim,
            backend,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch, seq, dim]
        let x = self.backend.linear_forward(x, &self.in_proj_weight, self.in_proj_bias.as_ref())?;

        // GLU gate: split into two halves along last dim
        let half = self.dim;
        let a = x.narrow(candle_core::D::Minus1, 0, half)?;
        let b = x.narrow(candle_core::D::Minus1, half, half)?;
        let gate = self.backend.sigmoid(&b)?;
        let x = (a * gate)?;

        // Depthwise conv1d: transpose to [batch, dim, seq], apply, transpose back
        let x = x.transpose(1, 2)?; // [batch, dim, seq]
        let x = self.depthwise_conv1d(&x)?;
        let x = x.transpose(1, 2)?; // [batch, seq, dim]

        // SwooshR activation before out_proj (out_proj is ActivationDropoutAndLinear with SwooshR)
        let x = super::activations::swoosh_r(&x)?;
        let x = self.backend.linear_forward(&x, &self.out_proj_weight, self.out_proj_bias.as_ref())?;
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
