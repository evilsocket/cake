//! F8E4M3 dequantization to F32, F16, and BF16.

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/fused_ops_ptx.rs"));
}
#[cfg(feature = "cuda")]
const FUSED_OPS_PTX: &str = ptx::FUSED_OPS;

/// Dequantize F8E4M3 tensor to F32.
/// Uses our custom CUDA kernel on GPU (works on SM80/A100 where candle lacks native F8 support).
/// Falls back to CPU dequantization on non-CUDA devices.
pub fn f8e4m3_to_f32(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F8E4M3 {
        return x.to_dtype(candle_core::DType::F32);
    }
    // Try candle's native cast first (works on SM89+)
    if let Ok(t) = x.to_dtype(candle_core::DType::F32) {
        return Ok(t);
    }
    // Fall back to our software kernel
    x.apply_op1_no_bwd(&F8E4M3ToF32)
}

/// Dequantize F8E4M3 tensor to BF16.
/// Direct F8→BF16 avoids F16 intermediate when using BF16 compute.
/// On Metal: dequant via CPU then transfer.
pub fn f8e4m3_to_bf16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F8E4M3 {
        return x.to_dtype(candle_core::DType::BF16);
    }
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        let dev = x.device().clone();
        return x.to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?
            .to_dtype(candle_core::DType::BF16)?
            .to_device(&dev);
    }
    x.apply_op1_no_bwd(&F8E4M3ToBF16)
}

/// Dequantize F8E4M3 tensor to F16.
/// Uses our custom CUDA kernel. F16 matmul is 2x faster than F32 on A100.
/// On Metal: dequant via CPU then transfer (Metal has no native F8 compute).
pub fn f8e4m3_to_f16(x: &Tensor) -> Result<Tensor> {
    if x.dtype() != candle_core::DType::F8E4M3 {
        return x.to_dtype(candle_core::DType::F16);
    }
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        let dev = x.device().clone();
        return x.to_device(&candle_core::Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?  // F8→F32 on CPU
            .to_dtype(candle_core::DType::F16)?   // F32→F16
            .to_device(&dev);                     // back to Metal
    }
    x.apply_op1_no_bwd(&F8E4M3ToF16)
}

struct F8E4M3ToBF16;

impl candle_core::CustomOp1 for F8E4M3ToBF16 {
    fn name(&self) -> &'static str {
        "f8e4m3_to_bf16"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let (f32_storage, shape) = F8E4M3ToF32.cpu_fwd(s, l)?;
        match f32_storage {
            CpuStorage::F32(data) => {
                let bf16_data: Vec<half::bf16> =
                    data.iter().map(|&v| half::bf16::from_f32(v)).collect();
                Ok((CpuStorage::BF16(bf16_data), shape))
            }
            _ => candle_core::bail!("expected F32 from F8E4M3ToF32"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device();
        let src = match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F8E4M3(s) => s,
            _ => candle_core::bail!("f8e4m3_to_bf16: expected F8E4M3 storage"),
        };
        let src = match l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => candle_core::bail!("f8e4m3_to_bf16: input must be contiguous"),
        };
        let el = l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let func =
            dev.get_or_load_custom_func("f8e4m3_to_bf16", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<half::bf16>(el)? };
        let mut builder = func.builder();
        builder.arg(&el);
        builder.arg(&src);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::BF16(out),
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }
}

struct F8E4M3ToF16;

impl candle_core::CustomOp1 for F8E4M3ToF16 {
    fn name(&self) -> &'static str {
        "f8e4m3_to_f16"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        // CPU: decode F8 bytes to f32, then quantize to f16
        let (f32_storage, shape) = F8E4M3ToF32.cpu_fwd(s, l)?;
        match f32_storage {
            CpuStorage::F32(data) => {
                let f16_data: Vec<half::f16> = data.iter().map(|&v| half::f16::from_f32(v)).collect();
                Ok((CpuStorage::F16(f16_data), shape))
            }
            _ => candle_core::bail!("expected F32 from F8E4M3ToF32"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device();
        let src = match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F8E4M3(s) => s,
            _ => candle_core::bail!("f8e4m3_to_f16: expected F8E4M3 storage"),
        };
        let src = match l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => candle_core::bail!("f8e4m3_to_f16: input must be contiguous"),
        };
        let el = l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let func =
            dev.get_or_load_custom_func("f8e4m3_to_f16", "cake_fused_ops", FUSED_OPS_PTX)?;
        // half::f16 is used by CudaStorageSlice::F16
        let out = unsafe { dev.alloc::<half::f16>(el)? };
        let mut builder = func.builder();
        builder.arg(&el);
        builder.arg(&src);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::F16(out),
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }
}

struct F8E4M3ToF32;

impl candle_core::CustomOp1 for F8E4M3ToF32 {
    fn name(&self) -> &'static str {
        "f8e4m3_to_f32"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let data = match s {
            CpuStorage::F8E4M3(data) => data,
            _ => candle_core::bail!("f8e4m3_to_f32: expected F8E4M3, got {:?}", s.dtype()),
        };
        let data = match l.contiguous_offsets() {
            Some((o1, o2)) => &data[o1..o2],
            None => candle_core::bail!("f8e4m3_to_f32: input must be contiguous"),
        };
        let dst: Vec<f32> = data
            .iter()
            .map(|&b| {
                let bits = b.to_bits();
                let sign = (bits >> 7) & 1;
                let exp = (bits >> 3) & 0xF;
                let mant = bits & 0x7;
                let result = if exp == 0 && mant == 0 {
                    0.0f32
                } else if exp == 0 {
                    f32::from(mant) / 8.0 * 2.0f32.powi(-6)
                } else if exp == 0xF && mant == 0x7 {
                    f32::NAN
                } else {
                    (1.0 + f32::from(mant) / 8.0) * 2.0f32.powi(i32::from(exp) - 7)
                };
                if sign == 1 { -result } else { result }
            })
            .collect();
        let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, l.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device();
        let src = match &s.slice {
            candle_core::cuda_backend::CudaStorageSlice::F8E4M3(s) => s,
            _ => candle_core::bail!("f8e4m3_to_f32: expected F8E4M3 storage"),
        };
        let src = match l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => candle_core::bail!("f8e4m3_to_f32: input must be contiguous"),
        };
        let el = l.shape().elem_count();
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let func =
            dev.get_or_load_custom_func("f8e4m3_to_f32", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<f32>(el)? };
        let mut builder = func.builder();
        builder.arg(&el);
        builder.arg(&src);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage {
                slice: candle_core::cuda_backend::CudaStorageSlice::F32(out),
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn test_f8e4m3_to_f32_cpu() {
        let f32_orig = Tensor::new(&[0.5f32, 1.0, -0.25, 2.0], &Device::Cpu).unwrap();
        let f8 = f32_orig.to_dtype(candle_core::DType::F8E4M3).unwrap();
        let result = f8e4m3_to_f32(&f8).unwrap();
        assert_eq!(result.dtype(), candle_core::DType::F32);
        assert_eq!(result.dims(), &[4]);
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_f8e4m3_to_f16_cpu() {
        let f32_orig = Tensor::new(&[0.5f32, 1.0, -0.25], &Device::Cpu).unwrap();
        let f8 = f32_orig.to_dtype(candle_core::DType::F8E4M3).unwrap();
        let result = f8e4m3_to_f16(&f8).unwrap();
        assert_eq!(result.dtype(), candle_core::DType::F16);
        assert_eq!(result.dims(), &[3]);
    }

    #[test]
    fn test_f8e4m3_to_bf16_cpu() {
        let f32_orig = Tensor::new(&[0.5f32, 1.0, -0.25], &Device::Cpu).unwrap();
        let f8 = f32_orig.to_dtype(candle_core::DType::F8E4M3).unwrap();
        let result = f8e4m3_to_bf16(&f8).unwrap();
        assert_eq!(result.dtype(), candle_core::DType::BF16);
        assert_eq!(result.dims(), &[3]);
    }

    #[test]
    fn test_f8e4m3_to_f16_passthrough_non_f8() {
        // Non-F8 input should just convert dtype
        let f32_tensor = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let result = f8e4m3_to_f16(&f32_tensor).unwrap();
        assert_eq!(result.dtype(), candle_core::DType::F16);
    }

    #[test]
    fn test_f8e4m3_to_bf16_passthrough_non_f8() {
        let f32_tensor = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let result = f8e4m3_to_bf16(&f32_tensor).unwrap();
        assert_eq!(result.dtype(), candle_core::DType::BF16);
    }
}
