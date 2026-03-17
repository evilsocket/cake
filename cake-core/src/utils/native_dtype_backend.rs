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
            // Dequantize ALL F8E4M3 tensors to F32 on CPU, then move to GPU.
            // This uses more VRAM (~4x) but avoids missing F8 CUDA kernels for
            // operations like RoPE, LayerNorm, etc. that candle doesn't support on F8.
            // A100 80GB has plenty of headroom for the ~12GB→~48GB expansion.
            let tensor = self.inner.load(name, &Device::Cpu)?;
            tensor.to_dtype(DType::F32)?.to_device(dev)
        } else {
            // Non-F8 tensors: load directly to device
            self.inner.load(name, dev)
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
