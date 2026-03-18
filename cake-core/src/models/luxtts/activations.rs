//! Custom activations used by Zipformer: SwooshR and SwooshL.

use anyhow::Result;
use candle_core::Tensor;

/// SwooshR(x) = log(1 + exp(x - 1)) - 0.08*x - 0.313261687
///
/// Matches Python's SwooshRForward with offset=1.
pub fn swoosh_r(x: &Tensor) -> Result<Tensor> {
    let x_offset = (x - 1.0)?;
    // Numerically stable softplus: max(z,0) + log(1 + exp(-|z|))
    let abs_xo = x_offset.abs()?;
    let relu_xo = x_offset.relu()?;
    let log_sum = (relu_xo + (abs_xo.neg()?.exp()? + 1.0)?.log()?)?;
    let linear = (x * 0.08)?;
    let result = ((log_sum - linear)? - 0.313261687)?;
    Ok(result)
}

/// SwooshL(x) = log(1 + exp(x - 4)) - 0.08*x - 0.035
///
/// Matches Python's SwooshLForward with offset=4.
pub fn swoosh_l(x: &Tensor) -> Result<Tensor> {
    let x_offset = (x - 4.0)?;
    // Numerically stable softplus: max(z,0) + log(1 + exp(-|z|))
    let abs_xo = x_offset.abs()?;
    let relu_xo = x_offset.relu()?;
    let log_sum = (relu_xo + (abs_xo.neg()?.exp()? + 1.0)?.log()?)?;
    let linear = (x * 0.08)?;
    let result = ((log_sum - linear)? - 0.035)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType};

    #[test]
    fn test_swoosh_r_values() {
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &Device::Cpu).unwrap();
        let y = swoosh_r(&x).unwrap().to_vec1::<f32>().unwrap();
        // swoosh_r(0) = log(1+exp(-1)) - 0 - 0.3133 ≈ 0.3133 - 0.3133 ≈ 0
        assert!(y[0].abs() < 0.01, "swoosh_r(0) = {}", y[0]);
        // swoosh_r(1) = log(1+exp(0)) - 0.08 - 0.3133 = 0.6931 - 0.3933 ≈ 0.300
        assert!((y[1] - 0.300).abs() < 0.01, "swoosh_r(1) = {}", y[1]);
    }

    #[test]
    fn test_swoosh_l_values() {
        let x = Tensor::new(&[0.0f32, 4.0, -4.0], &Device::Cpu).unwrap();
        let y = swoosh_l(&x).unwrap().to_vec1::<f32>().unwrap();
        // swoosh_l(0) = log(1+exp(-4)) - 0 - 0.035 ≈ 0.0183 - 0.035 ≈ -0.017
        assert!(y[0].abs() < 0.05, "swoosh_l(0) = {}", y[0]);
        // swoosh_l(4) = log(1+exp(0)) - 0.32 - 0.035 = 0.6931 - 0.355 ≈ 0.338
        assert!((y[1] - 0.338).abs() < 0.01, "swoosh_l(4) = {}", y[1]);
    }

    #[test]
    fn test_swoosh_r_shape() {
        let x = Tensor::randn(0f32, 1.0, (2, 3, 4), &Device::Cpu).unwrap();
        let y = swoosh_r(&x).unwrap();
        assert_eq!(y.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_swoosh_l_shape() {
        let x = Tensor::randn(0f32, 1.0, (2, 3, 4), &Device::Cpu).unwrap();
        let y = swoosh_l(&x).unwrap();
        assert_eq!(y.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_swoosh_r_f16() {
        let x = Tensor::new(&[0.0f32, 1.0], &Device::Cpu).unwrap().to_dtype(DType::F16).unwrap();
        let y = swoosh_r(&x).unwrap();
        assert_eq!(y.dtype(), DType::F16);
    }
}
