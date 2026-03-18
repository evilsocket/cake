//! BiasNorm — the normalization layer used throughout Zipformer.
//!
//! `output = (x + bias) * exp(log_scale) / sqrt(mean(x^2) + eps)`

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct BiasNorm {
    bias: Tensor,
    log_scale: Tensor,
    eps: f64,
}

impl BiasNorm {
    pub fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let bias = vb.get(dim, "bias")?;
        let log_scale = vb.get(1, "log_scale")?;
        Ok(Self {
            bias,
            log_scale,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Python: scales = mean((x - bias)^2, dim=-1)^(-0.5) * exp(log_scale)
        //         output = x * scales
        let x_centered = x.broadcast_sub(&self.bias)?;
        let mean_sq = x_centered.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let inv_rms = (mean_sq + self.eps)?.sqrt()?.recip()?;
        let scale = self.log_scale.exp()?;
        let scales = inv_rms.broadcast_mul(&scale)?;
        Ok(x.broadcast_mul(&scales)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    fn make_bias_norm(dim: usize) -> BiasNorm {
        let device = Device::Cpu;
        let dtype = DType::F32;
        BiasNorm {
            bias: Tensor::zeros(dim, dtype, &device).unwrap(),
            log_scale: Tensor::zeros(1, dtype, &device).unwrap(),
            eps: 1e-5,
        }
    }

    #[test]
    fn test_bias_norm_shape() {
        let bn = make_bias_norm(64);
        let x = Tensor::randn(0f32, 1.0, (2, 10, 64), &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap();
        assert_eq!(y.shape().dims(), &[2, 10, 64]);
    }

    #[test]
    fn test_bias_norm_zero_params() {
        // With zero bias and log_scale=0 (scale=1):
        // scales = mean((x - 0)^2)^(-0.5) * exp(0) = 1/rms(x)
        // output = x * scales = x / rms(x)
        let bn = make_bias_norm(4);
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();
        let y = bn.forward(&x).unwrap().to_vec2::<f32>().unwrap();
        let rms = (7.5f32).sqrt();
        for (i, &val) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
            assert!((y[0][i] - val / rms).abs() < 1e-4);
        }
    }
}
