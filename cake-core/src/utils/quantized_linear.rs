//! Quantized linear layer support for fused 4-bit Metal matmul.
//!
//! [`QuantizedWeight`] stores packed 4-bit weights (U32), per-group F16 scales,
//! and per-group F16 biases on a Metal device without dequantizing.
//! [`LinearWeight`] is an enum dispatching between standard dense (`Tensor`)
//! weights and quantized weights, used by MLP and Attention layers.

use candle_core::{Result, Tensor};

use crate::backends::ComputeBackend;

/// Packed 4-bit weight data kept on-device (Metal) without dequantization.
///
/// Memory layout matches the Phase 1 `q4_matmul_f16` kernel:
/// - `packed`: (out_features, in_features/8) U32 -- 8 nibbles per U32, LSB-first
/// - `scales`: (out_features, num_groups) F16
/// - `biases`: (out_features, num_groups) F16
#[derive(Debug, Clone)]
pub struct QuantizedWeight {
    /// Packed 4-bit weight tensor, U32, on Metal device.
    pub packed: Tensor,
    /// Per-group scale factors, F16, on Metal device.
    pub scales: Tensor,
    /// Per-group bias offsets, F16, on Metal device.
    pub biases: Tensor,
    /// Number of elements per quantization group.
    pub group_size: usize,
}

/// A linear layer weight that is either dense (standard F16 Tensor) or
/// quantized (packed 4-bit with scales/biases). Used as a drop-in replacement
/// for raw `Tensor` weight fields in MLP and Attention.
#[derive(Debug, Clone)]
pub enum LinearWeight {
    /// Standard dense weight tensor (pre-transposed for Metal backend).
    Dense(Tensor),
    /// Quantized 4-bit weight kept packed on Metal.
    Quantized(QuantizedWeight),
}

impl LinearWeight {
    /// Perform `x @ weight^T + bias`, dispatching to the fused q4 kernel
    /// for quantized weights on Metal.
    pub fn forward(
        &self,
        x: &Tensor,
        bias: Option<&Tensor>,
        backend: &dyn ComputeBackend,
    ) -> Result<Tensor> {
        match self {
            LinearWeight::Dense(weight) => backend.linear_forward(x, weight, bias),
            LinearWeight::Quantized(qw) => {
                let out = backend.q4_linear_forward(
                    &qw.packed,
                    &qw.scales,
                    &qw.biases,
                    x,
                    qw.group_size,
                )?;
                match bias {
                    Some(b) => out.broadcast_add(b),
                    None => Ok(out),
                }
            }
        }
    }

    /// Wrap a dense weight tensor.
    pub fn dense(weight: Tensor) -> Self {
        LinearWeight::Dense(weight)
    }

    /// Wrap quantized weight components.
    pub fn quantized(packed: Tensor, scales: Tensor, biases: Tensor, group_size: usize) -> Self {
        LinearWeight::Quantized(QuantizedWeight {
            packed,
            scales,
            biases,
            group_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_linear_weight_dense_forward() {
        let backend = crate::backends::create_backend(&Device::Cpu);
        // Simple 3x2 weight, input 1x2
        let w = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), &Device::Cpu)
            .unwrap();
        let w = backend.preprocess_linear_weight(&w).unwrap();
        let lw = LinearWeight::Dense(w);
        let x = Tensor::from_vec(vec![1.0f32, 1.0], (1, 2), &Device::Cpu).unwrap();
        let out = lw.forward(&x, None, &*backend).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // weight rows: [1,2], [3,4], [5,6] -> x@w^T = [3, 7, 11]
        assert!((vals[0] - 3.0).abs() < 1e-5);
        assert!((vals[1] - 7.0).abs() < 1e-5);
        assert!((vals[2] - 11.0).abs() < 1e-5);
    }

    #[test]
    fn test_linear_weight_dense_with_bias() {
        let backend = crate::backends::create_backend(&Device::Cpu);
        let w = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), &Device::Cpu).unwrap();
        let w = backend.preprocess_linear_weight(&w).unwrap();
        let lw = LinearWeight::Dense(w);
        let x = Tensor::from_vec(vec![3.0f32, 5.0], (1, 2), &Device::Cpu).unwrap();
        let bias = Tensor::from_vec(vec![10.0f32, 20.0], 2, &Device::Cpu).unwrap();
        let out = lw.forward(&x, Some(&bias), &*backend).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // identity weight + bias: [3+10, 5+20] = [13, 25]
        assert!((vals[0] - 13.0).abs() < 1e-5);
        assert!((vals[1] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn test_quantized_weight_cpu_fallback() {
        let backend = crate::backends::create_backend(&Device::Cpu);
        // in_features=8, out_features=2, group_size=8, 1 group
        // All nibbles=1, scale=1.0, bias=0.0 -> all weights = 1.0
        let packed =
            Tensor::from_vec(vec![0x11111111u32; 2], (2, 1), &Device::Cpu).unwrap();
        let scales = Tensor::from_vec(vec![1.0f32; 2], (2, 1), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let biases = Tensor::from_vec(vec![0.0f32; 2], (2, 1), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let qw = LinearWeight::quantized(packed, scales, biases, 8);

        let x = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let out = qw.forward(&x, None, &*backend).unwrap();
        let vals: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Each weight=1.0, x=[1;8] -> dot = 8.0
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                (v - 8.0).abs() < 0.5,
                "output[{i}] = {v}, expected ~8.0"
            );
        }
    }
}
