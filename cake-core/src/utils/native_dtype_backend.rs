//! Native-dtype VarBuilder backend for mixed-precision safetensors.
//!
//! Unlike the standard VarBuilder which casts all tensors to a single dtype,
//! this backend loads each tensor in its **stored dtype** (F8E4M3, F16, F32, etc.).
//! This is critical for FLUX.1-dev FP8 where we need transformer weights to remain
//! in F8E4M3 on GPU (~12GB) rather than expanding to BF16 (~24GB).

use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};

/// Backend that loads tensors in their native stored dtype.
/// F8E4M3 tensors are kept in F8E4M3 on GPU (~1 byte/param).
/// During forward, Fp8Linear converts to BF16 per-layer on GPU.
struct NativeDtypeBackend {
    inner: MmapedSafetensors,
}

impl NativeDtypeBackend {
    fn load_tensor(
        &self,
        name: &str,
        _dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let view = self.inner.get(name)?;
        let file_dtype: DType = view.dtype().try_into()?;

        if file_dtype == DType::F8E4M3 {
            let tensor = self.inner.load(name, &Device::Cpu)?;
            let ndim = tensor.dims().len();
            if ndim <= 1 {
                // Small tensors (norms, scales, biases): dequantize to F32 on CPU
                // using our software dequant (works on all GPU architectures).
                crate::backends::f8_dequant::f8e4m3_to_f32(&tensor)?.to_device(dev)
            } else {
                // Large tensors (weights, 2D+): keep as F8E4M3 on GPU (~1 byte/param).
                // Fp8Linear casts to F32 per-forward-call (~60MB temporary per layer).
                // candle has cast_f8_e4m3_f32 kernel for the F8→F32 conversion.
                tensor.to_device(dev)
            }
        } else {
            // Non-F8 tensors: load to device, cast to F32 for consistency
            let tensor = self.inner.load(name, dev)?;
            if tensor.dtype() != DType::F32 {
                tensor.to_dtype(DType::F32)
            } else {
                Ok(tensor)
            }
        }
    }
}

impl SimpleBackend for NativeDtypeBackend {
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

/// Create a VarBuilder that loads tensors in their native stored dtype.
/// F8E4M3 tensors are kept in F8E4M3 on the target device.
///
/// # Safety
///
/// Inherits the mmap safety requirements from `MmapedSafetensors`.
/// Create a VarBuilder that dequantizes F8E4M3 tensors on CPU per-tensor.
///
/// Unlike the standard FP8 backend which loads everything at once, this backend
/// converts each F8E4M3 tensor individually (CPU F8→target_dtype, then GPU transfer),
/// keeping peak memory low. Non-F8 tensors are loaded directly to the device.
///
/// # Safety
///
/// Inherits the mmap safety requirements from `MmapedSafetensors`.
pub unsafe fn load_native_dtype_var_builder<'a>(
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
    let total = inner.tensors().len();
    log::info!(
        "native-dtype backend: {} total tensors, {} F8E4M3 (kept as F8 on GPU)",
        total,
        fp8_count,
    );

    let backend: Box<dyn SimpleBackend> = Box::new(NativeDtypeBackend { inner });
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use std::collections::HashMap;

    /// Helper: write tensors to a temp .safetensors file, return path.
    fn write_safetensors(
        tensors: &HashMap<String, Tensor>,
        dir: &std::path::Path,
    ) -> std::path::PathBuf {
        let path = dir.join("test.safetensors");
        candle_core::safetensors::save(tensors, &path).unwrap();
        path
    }

    #[test]
    fn test_non_f8_tensor_cast_to_f32() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        // Store as F32 (non-F8)
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        tensors.insert("my_weight".to_string(), t);
        let path = write_safetensors(&tensors, dir.path());

        let vb = unsafe {
            load_native_dtype_var_builder(
                &[path],
                DType::F32,
                &Device::Cpu,
            )
            .unwrap()
        };
        let loaded = vb.get_unchecked_dtype("my_weight", DType::F32).unwrap();
        assert_eq!(loaded.dtype(), DType::F32);
        let vals: Vec<f32> = loaded.to_vec1().unwrap();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bf16_tensor_cast_to_f32() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        // Store as BF16
        let t = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        tensors.insert("bf16_weight".to_string(), t);
        let path = write_safetensors(&tensors, dir.path());

        let vb = unsafe {
            load_native_dtype_var_builder(
                &[path],
                DType::F32,
                &Device::Cpu,
            )
            .unwrap()
        };
        let loaded = vb
            .get_unchecked_dtype("bf16_weight", DType::F32)
            .unwrap();
        // Non-F8 tensors are cast to F32
        assert_eq!(loaded.dtype(), DType::F32);
        let vals: Vec<f32> = loaded.to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_contains_tensor_true() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "exists".to_string(),
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        );
        let path = write_safetensors(&tensors, dir.path());

        let vb = unsafe {
            load_native_dtype_var_builder(&[path], DType::F32, &Device::Cpu).unwrap()
        };
        assert!(vb.contains_tensor("exists"));
    }

    #[test]
    fn test_contains_tensor_false() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "exists".to_string(),
            Tensor::zeros(4, DType::F32, &Device::Cpu).unwrap(),
        );
        let path = write_safetensors(&tensors, dir.path());

        let vb = unsafe {
            load_native_dtype_var_builder(&[path], DType::F32, &Device::Cpu).unwrap()
        };
        assert!(!vb.contains_tensor("does_not_exist"));
    }

    #[test]
    fn test_get_shape_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::zeros((3, 4), DType::F32, &Device::Cpu).unwrap(),
        );
        let path = write_safetensors(&tensors, dir.path());

        let vb = unsafe {
            load_native_dtype_var_builder(&[path], DType::F32, &Device::Cpu).unwrap()
        };
        // Request wrong shape
        let result = vb.get((5, 4), "w");
        assert!(result.is_err(), "expected shape mismatch error");
    }

    #[test]
    fn test_get_correct_shape() {
        let dir = tempfile::tempdir().unwrap();
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            Tensor::ones((3, 4), DType::F32, &Device::Cpu).unwrap(),
        );
        let path = write_safetensors(&tensors, dir.path());

        let vb = unsafe {
            load_native_dtype_var_builder(&[path], DType::F32, &Device::Cpu).unwrap()
        };
        let loaded = vb.get((3, 4), "w").unwrap();
        assert_eq!(loaded.dims(), &[3, 4]);
    }
}
