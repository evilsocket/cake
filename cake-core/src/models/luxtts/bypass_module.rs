//! BypassModule — learned residual interpolation.
//!
//! `output = x + scale * (module_output - x)` where scale is learned and clamped [0, 1].
//! At inference: just a lerp between input and module output.

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct BypassModule {
    /// Learned bypass scale, clamped to [0, 1] at inference.
    scale: Tensor,
}

impl BypassModule {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        Self::load_dim(1, vb)
    }

    /// Load with explicit dimension (per-channel bypass scale).
    pub fn load_dim(dim: usize, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(dim, "bypass_scale")?;
        Ok(Self { scale })
    }

    /// Create a bypass module with a fixed scale value (for testing/defaults).
    pub fn with_scale(scale: f32, device: &candle_core::Device) -> Result<Self> {
        let scale = Tensor::new(&[scale], device)?;
        Ok(Self { scale })
    }

    /// Apply bypass: output = x + scale * (module_out - x)
    /// bypass_scale is the weight on the non-residual (processed) path
    pub fn forward(&self, x: &Tensor, module_out: &Tensor) -> Result<Tensor> {
        // At inference, no clamping — use raw bypass_scale
        let diff = (module_out - x)?;
        let scaled = diff.broadcast_mul(&self.scale)?;
        Ok((x + scaled)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_bypass_zero_scale() {
        // scale=0: output = x (no bypass)
        let bypass = BypassModule::with_scale(0.0, &Device::Cpu).unwrap();
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let module_out = Tensor::new(&[[10.0f32, 20.0, 30.0]], &Device::Cpu).unwrap();
        let out = bypass.forward(&x, &module_out).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![1.0, 2.0, 3.0]]);
    }

    #[test]
    fn test_bypass_one_scale() {
        // scale=1: output = module_out
        let bypass = BypassModule::with_scale(1.0, &Device::Cpu).unwrap();
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let module_out = Tensor::new(&[[10.0f32, 20.0, 30.0]], &Device::Cpu).unwrap();
        let out = bypass.forward(&x, &module_out).unwrap().to_vec2::<f32>().unwrap();
        assert_eq!(out, vec![vec![10.0, 20.0, 30.0]]);
    }

    #[test]
    fn test_bypass_half_scale() {
        // scale=0.5: output = midpoint
        let bypass = BypassModule::with_scale(0.5, &Device::Cpu).unwrap();
        let x = Tensor::new(&[[0.0f32, 0.0]], &Device::Cpu).unwrap();
        let module_out = Tensor::new(&[[10.0f32, 20.0]], &Device::Cpu).unwrap();
        let out = bypass.forward(&x, &module_out).unwrap().to_vec2::<f32>().unwrap();
        assert!((out[0][0] - 5.0).abs() < 1e-5);
        assert!((out[0][1] - 10.0).abs() < 1e-5);
    }
}
