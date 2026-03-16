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
}
