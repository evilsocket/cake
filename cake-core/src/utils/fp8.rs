//! FP8 (float8_e4m3fn) dequantization support.
//!
//! Models like Qwen3.5-27B-FP8 store most weight tensors in F8_E4M3 format with
//! per-block scale factors (`weight_scale_inv`). This module provides a custom
//! VarBuilder backend that transparently dequantizes FP8 weights at load time,
//! allowing cake to run FP8-quantized models on any backend (CUDA, Metal, CPU).
//!
//! Dequantization formula (block size 128×128):
//!   bf16_weight[i*128..(i+1)*128, j*128..(j+1)*128]
//!     = cast(fp8_weight[...same...]) * scale_inv[i, j]

use std::path::Path;

use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};

const FP8_BLOCK_SIZE: usize = 128;

/// Check whether a model uses FP8 block-wise quantization by looking at its config.
pub fn is_fp8_quantized(config_path: &Path) -> bool {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return false;
    };
    // Check top-level and nested text_config for quantization_config
    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        let is_fp8 = root
            .get("quantization_config")
            .and_then(|qc| qc.get("quant_method"))
            .and_then(|qm| qm.as_str())
            .map(|s| s == "fp8")
            .unwrap_or(false);
        if is_fp8 {
            return true;
        }
    }
    false
}

/// Dequantize a 2-D FP8 weight tensor using its per-block scale factor.
pub fn dequantize_fp8_blockwise(weight: &Tensor, scale_inv: &Tensor) -> candle_core::Result<Tensor> {
    let (m, n) = weight.dims2()?;
    let bm = FP8_BLOCK_SIZE;
    let bn = FP8_BLOCK_SIZE;
    let blocks_m = m.div_ceil(bm);
    let blocks_n = n.div_ceil(bn);

    // Cast FP8 → F32 on CPU (candle supports this on CPU)
    let weight_f32 = weight.to_dtype(DType::F32)?;
    let scale_f32 = scale_inv.to_dtype(DType::F32)?;

    // Reshape for block-wise broadcast multiply:
    //   weight: [M, N]         → [blocks_m, bm, blocks_n, bn]
    //   scale:  [blocks_m, blocks_n] → [blocks_m, 1, blocks_n, 1]
    let weight_blocked = weight_f32.reshape((blocks_m, bm, blocks_n, bn))?;
    let scale_blocked = scale_f32.reshape((blocks_m, 1usize, blocks_n, 1usize))?;

    let dequantized = weight_blocked.broadcast_mul(&scale_blocked)?;
    dequantized.reshape((m, n))
}

/// Custom VarBuilder backend that wraps MmapedSafetensors and transparently
/// dequantizes FP8-quantized weight tensors on CPU before moving to the target device.
struct Fp8Backend {
    inner: MmapedSafetensors,
}

impl SimpleBackend for Fp8Backend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_tensor(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_tensor(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.get(name).is_ok()
    }
}

impl Fp8Backend {
    fn load_tensor(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let scale_name = format!("{name}_scale_inv");

        if self.inner.get(&scale_name).is_ok() {
            // FP8 quantized tensor — dequantize on CPU then move to device
            let weight = self.inner.load(name, &Device::Cpu)?;
            let scale = self.inner.load(&scale_name, &Device::Cpu)?;

            let dequantized = dequantize_fp8_blockwise(&weight, &scale)?;
            dequantized.to_dtype(dtype)?.to_device(dev)
        } else {
            // Non-quantized tensor — check if the on-file dtype needs CPU-side handling
            let view = self.inner.get(name)?;
            let file_dtype: DType = view.dtype().try_into()?;

            if file_dtype == DType::F8E4M3 {
                // FP8 without scale (shouldn't happen, but handle gracefully)
                let tensor = self.inner.load(name, &Device::Cpu)?;
                tensor.to_dtype(dtype)?.to_device(dev)
            } else {
                // Normal path — load directly on target device
                self.inner.load(name, dev)?.to_dtype(dtype)
            }
        }
    }
}

/// Create a VarBuilder that transparently dequantizes FP8 weights.
///
/// # Safety
///
/// Inherits the mmap safety requirements from `MmapedSafetensors`.
pub unsafe fn load_fp8_var_builder<'a>(
    filenames: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
) -> anyhow::Result<VarBuilder<'a>> {
    let inner = MmapedSafetensors::multi(filenames)?;

    let fp8_count = inner
        .tensors()
        .iter()
        .filter(|(_, v)| v.dtype() == safetensors::tensor::Dtype::F8_E4M3)
        .count();
    log::info!(
        "FP8 model detected: {} tensors will be dequantized at load time",
        fp8_count
    );

    let backend: Box<dyn SimpleBackend> = Box::new(Fp8Backend { inner });
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

// ─── Fp8Linear: reusable FP8-aware Linear layer ─────────────────────────────

/// Linear layer that stores weights in their native dtype (possibly F8E4M3)
/// and dequantizes to F16 on first forward call, caching the result.
///
/// Used by FLUX.1-dev and any future FP8 model that needs lazy per-layer dequant.
#[derive(Debug)]
pub struct Fp8Linear {
    f8_weight: Option<candle_core::Tensor>,
    weight: std::sync::RwLock<Option<candle_core::Tensor>>,
    bias: Option<candle_core::Tensor>,
}

impl Clone for Fp8Linear {
    fn clone(&self) -> Self {
        Self {
            f8_weight: self.f8_weight.clone(),
            weight: std::sync::RwLock::new(self.weight.read().unwrap().clone()),
            bias: self.bias.clone(),
        }
    }
}

impl Fp8Linear {
    /// Create an Fp8Linear from a weight tensor and optional bias.
    /// If the weight is F8E4M3, it's stored as-is and dequantized lazily.
    pub fn new(weight: candle_core::Tensor, bias: Option<candle_core::Tensor>) -> Self {
        if weight.dtype() == candle_core::DType::F8E4M3 {
            Self {
                f8_weight: Some(weight),
                weight: std::sync::RwLock::new(None),
                bias,
            }
        } else {
            // Pre-transpose and make contiguous for efficient matmul
            let w_t = weight.t().unwrap().contiguous().unwrap();
            Self {
                f8_weight: None,
                weight: std::sync::RwLock::new(Some(w_t)),
                bias,
            }
        }
    }

    fn get_weight(&self) -> candle_core::Result<candle_core::Tensor> {
        // Fast path: weight already dequantized and cached
        if let Some(w) = self.weight.read().unwrap().as_ref() {
            return Ok(w.clone());
        }
        // Slow path: dequantize F8→F32→F16, pre-transpose, and cache
        let f8_w = self
            .f8_weight
            .as_ref()
            .expect("no F8 weight and no cached weight");
        let w = f8_w.to_dtype(candle_core::DType::F32)?.to_dtype(candle_core::DType::F16)?;
        // Pre-transpose and make contiguous so matmul doesn't have to
        let w_t = w.t()?.contiguous()?;
        let mut cache = self.weight.write().unwrap();
        *cache = Some(w_t.clone());
        Ok(w_t)
    }

    /// Pre-dequantize F8→F16 and cache. Call once before inference loop.
    pub fn warmup(&mut self) -> candle_core::Result<()> {
        let _ = self.get_weight()?;
        self.f8_weight = None;
        Ok(())
    }

    /// Forward pass: dequantizes weight if needed, computes matmul in weight dtype.
    pub fn forward(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        let in_dtype = x.dtype();
        // get_weight returns pre-transposed weight: shape (in_features, out_features)
        let w_t = self.get_weight()?;
        let compute = w_t.dtype();
        let x = x.to_dtype(compute)?;

        // For 2D input, skip reshaping — matmul handles it directly
        let y = if x.rank() == 2 {
            x.matmul(&w_t)?
        } else {
            let dims = x.dims();
            let last = dims[dims.len() - 1];
            let batch: usize = dims[..dims.len() - 1].iter().product();
            let y_2d = x.reshape((batch, last))?.matmul(&w_t)?;
            let mut out_dims = dims[..dims.len() - 1].to_vec();
            out_dims.push(w_t.dim(1)?);
            y_2d.reshape(out_dims)?
        };

        let y = match &self.bias {
            Some(b) => y.broadcast_add(&b.to_dtype(compute)?)?,
            None => y,
        };
        y.to_dtype(in_dtype)
    }
}

/// Create an Fp8Linear from a VarBuilder (no bias).
pub fn fp8_linear(
    in_features: usize,
    out_features: usize,
    vb: candle_nn::VarBuilder,
) -> candle_core::Result<Fp8Linear> {
    let weight = vb.get_unchecked_dtype("weight", candle_core::DType::F32)?;
    if weight.dims().len() == 2 && weight.dim(0)? == out_features && weight.dim(1)? == in_features {
        Ok(Fp8Linear::new(weight, None))
    } else if weight.dims().len() == 2
        && weight.dim(0)? == in_features
        && weight.dim(1)? == out_features
    {
        Ok(Fp8Linear::new(weight.t()?, None))
    } else {
        Ok(Fp8Linear::new(weight, None))
    }
}

/// Create an Fp8Linear from a VarBuilder (with bias).
pub fn fp8_linear_b(
    in_features: usize,
    out_features: usize,
    vb: candle_nn::VarBuilder,
) -> candle_core::Result<Fp8Linear> {
    let weight = vb.get_unchecked_dtype("weight", candle_core::DType::F32)?;
    let bias = vb
        .get_unchecked_dtype("bias", candle_core::DType::F32)
        .ok();
    if weight.dims().len() == 2 && weight.dim(0)? == out_features && weight.dim(1)? == in_features {
        Ok(Fp8Linear::new(weight, bias))
    } else if weight.dims().len() == 2
        && weight.dim(0)? == in_features
        && weight.dim(1)? == out_features
    {
        Ok(Fp8Linear::new(weight.t()?, bias))
    } else {
        Ok(Fp8Linear::new(weight, bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_fp8_blockwise_identity_scale() {
        // Create F8E4M3 tensor via candle's DType::F8E4M3
        // Start from F32 0.5, cast to F8E4M3, then dequantize with scale=1.0
        let f32_weight =
            Tensor::from_vec(vec![0.5f32; 128 * 128], (128, 128), &Device::Cpu).unwrap();
        let weight = f32_weight.to_dtype(DType::F8E4M3).unwrap();
        let scale_inv = Tensor::from_vec(vec![1.0f32], (1, 1), &Device::Cpu).unwrap();
        let result = dequantize_fp8_blockwise(&weight, &scale_inv).unwrap();
        assert_eq!(result.dims(), &[128, 128]);
        let first: f32 = result.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(
            (first - 0.5).abs() < 0.05,
            "identity scale should preserve value, got {first}"
        );
    }

    #[test]
    fn test_dequantize_fp8_blockwise_scaling() {
        let f32_weight =
            Tensor::from_vec(vec![1.0f32; 128 * 128], (128, 128), &Device::Cpu).unwrap();
        let weight = f32_weight.to_dtype(DType::F8E4M3).unwrap();
        let scale_inv = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap();
        let result = dequantize_fp8_blockwise(&weight, &scale_inv).unwrap();
        let first: f32 = result.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!(
            (first - 2.0).abs() < 0.1,
            "scale=2.0 should double value, got {first}"
        );
    }

    #[test]
    fn test_is_fp8_quantized() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"quantization_config": {"quant_method": "fp8"}}"#,
        )
        .unwrap();
        assert!(is_fp8_quantized(&config_path));

        let config_path2 = dir.path().join("config2.json");
        std::fs::write(&config_path2, r#"{"hidden_size": 4096}"#).unwrap();
        assert!(!is_fp8_quantized(&config_path2));
    }

    #[test]
    fn test_fp8_linear_new_f32_weight_stored_in_weight() {
        let w = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let lin = Fp8Linear::new(w, None);
        // F32 weight should be in `weight`, not f8_weight
        assert!(lin.f8_weight.is_none());
        assert!(lin.weight.read().unwrap().is_some());
    }

    #[test]
    fn test_fp8_linear_forward_preserves_dtype_f32() {
        let w = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let lin = Fp8Linear::new(w, None);
        let x = Tensor::new(&[[2.0f32, 3.0]], &Device::Cpu).unwrap();
        let out = lin.forward(&x).unwrap();
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.dims(), &[1, 2]);
    }

    #[test]
    fn test_fp8_linear_forward_with_bias() {
        // Weight = identity, bias = [10, 20]
        let w = Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0f32, 20.0], &Device::Cpu).unwrap();
        let lin = Fp8Linear::new(w, Some(b));
        let x = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
        let out = lin.forward(&x).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // [1*1+2*0+10, 1*0+2*1+20] = [11, 22]
        assert!((vals[0] - 11.0).abs() < 1e-5);
        assert!((vals[1] - 22.0).abs() < 1e-5);
    }

    #[test]
    fn test_fp8_linear_warmup_caches_weight() {
        let w = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let mut lin = Fp8Linear::new(w, None);
        lin.warmup().unwrap();
        // After warmup, f8_weight should be cleared (it was already None for F32)
        assert!(lin.f8_weight.is_none());
        // weight should still be available
        assert!(lin.weight.read().unwrap().is_some());
    }

    #[test]
    fn test_fp8_linear_helper_loads_from_vb() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "weight".to_string(),
            Tensor::new(&[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]], &Device::Cpu).unwrap(),
        );
        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        let lin = fp8_linear(3, 2, vb).unwrap();
        let x = Tensor::zeros((1, 3), DType::F32, &Device::Cpu).unwrap();
        let out = lin.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 2]);
    }

    #[test]
    fn test_fp8_linear_helper_transposes_if_needed() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        // Store weight as (in, out) instead of (out, in)
        map.insert(
            "weight".to_string(),
            Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]], &Device::Cpu).unwrap(),
        );
        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        // in=3, out=2; weight is (3,2) so needs transpose
        let lin = fp8_linear(3, 2, vb).unwrap();
        let x = Tensor::ones((1, 3), DType::F32, &Device::Cpu).unwrap();
        let out = lin.forward(&x).unwrap();
        assert_eq!(out.dims(), &[1, 2]);
    }

    #[test]
    fn test_fp8_linear_b_loads_weight_and_bias() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "weight".to_string(),
            Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap(),
        );
        map.insert(
            "bias".to_string(),
            Tensor::new(&[5.0f32, 10.0], &Device::Cpu).unwrap(),
        );
        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        let lin = fp8_linear_b(2, 2, vb).unwrap();
        let x = Tensor::new(&[[1.0f32, 2.0]], &Device::Cpu).unwrap();
        let out = lin.forward(&x).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 6.0).abs() < 1e-5);
        assert!((vals[1] - 12.0).abs() < 1e-5);
    }

    #[test]
    fn test_fp8_linear_b_no_bias_in_vb() {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(
            "weight".to_string(),
            Tensor::new(&[[1.0f32, 0.0], [0.0, 1.0]], &Device::Cpu).unwrap(),
        );
        // No bias tensor
        let vb = candle_nn::VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
        let lin = fp8_linear_b(2, 2, vb).unwrap();
        assert!(lin.bias.is_none());
    }
}
