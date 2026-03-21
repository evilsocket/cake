//! Fused activation kernels: silu_mul and stable_softplus.

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

// ─── silu_mul: silu(gate) * up ──────────────────────────────────────

struct SiluMul;

impl candle_core::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "silu_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        use rayon::prelude::*;

        fn inner<T: candle_core::WithDType + num_traits::Float>(
            gate: &[T],
            l1: &Layout,
            up: &[T],
            l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let gate = match l1.contiguous_offsets() {
                Some((o1, o2)) => &gate[o1..o2],
                None => candle_core::bail!("silu_mul: gate must be contiguous"),
            };
            let up = match l2.contiguous_offsets() {
                Some((o1, o2)) => &up[o1..o2],
                None => candle_core::bail!("silu_mul: up must be contiguous"),
            };
            let n = gate.len();
            let mut dst = vec![T::zero(); n];
            const CHUNK: usize = 8192;
            dst.par_chunks_mut(CHUNK)
                .enumerate()
                .for_each(|(chunk_idx, dst_chunk)| {
                    let start = chunk_idx * CHUNK;
                    for (i, d) in dst_chunk.iter_mut().enumerate() {
                        let x = gate[start + i];
                        let y = up[start + i];
                        // silu(x) * y = x * sigmoid(x) * y
                        *d = x / (T::one() + (-x).exp()) * y;
                    }
                });
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l1.shape().clone()))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(a), C::BF16(b)) => inner(a, l1, b, l2),
            (C::F16(a), C::F16(b)) => inner(a, l1, b, l2),
            (C::F32(a), C::F32(b)) => inner(a, l1, b, l2),
            (C::F64(a), C::F64(b)) => inner(a, l1, b, l2),
            _ => candle_core::bail!("silu_mul: unsupported dtype {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map2, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        struct S;
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                gate: &CudaSlice<T>,
                l1: &Layout,
                up: &CudaSlice<T>,
                l2: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let gate = match l1.contiguous_offsets() {
                    Some((o1, o2)) => gate.slice(o1..o2),
                    None => candle_core::bail!("silu_mul: gate must be contiguous"),
                };
                let up = match l2.contiguous_offsets() {
                    Some((o1, o2)) => up.slice(o1..o2),
                    None => candle_core::bail!("silu_mul: up must be contiguous"),
                };
                let el = l1.shape().elem_count();
                let cfg = LaunchConfig::for_num_elems(el as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("silu_mul"),
                    "cake_fused_ops",
                    super::FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&gate);
                builder.arg(&up);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s1.device();
        let slice = S.map(&s1.slice, l1, &s2.slice, l2, dev)?;
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            l1.shape().clone(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle_core::MetalStorage,
        l1: &Layout,
        s2: &candle_core::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::DType;
        let device = s1.device();
        let el = l1.shape().elem_count();
        let kernel_name = match s1.dtype() {
            DType::F32 => "silu_mul_f32",
            DType::F16 => "silu_mul_f16",
            dt => candle_core::bail!("silu_mul metal: unsupported dtype {dt:?}"),
        };
        let lib = device.new_library_with_source(super::FUSED_OPS_MSL, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal shader compile: {e}")))?;
        let func = lib.get_function(kernel_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal get_function: {e}")))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("metal pipeline: {e}")))?;
        let output = device.new_buffer(el, s1.dtype(), "silu_mul")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let off1 = l1.start_offset() * s1.dtype().size_in_bytes();
        let off2 = l2.start_offset() * s2.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s1.buffer(), off1));
        candle_metal_kernels::utils::set_param(&encoder, 1, (s2.buffer(), off2));
        candle_metal_kernels::utils::set_param(&encoder, 2, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 3, el as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s1.dtype()), l1.shape().clone()))
    }
}

/// Fused silu(gate) * up — single kernel on CUDA and Metal.
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    gate.apply_op2_no_bwd(up, &SiluMul)
}

// ─── stable_softplus: ln(1+exp(clamp(x,−∞,88))) with max(x,·) ──────

struct StableSoftplus;

impl candle_core::CustomOp1 for StableSoftplus {
    fn name(&self) -> &'static str {
        "stable_softplus"
    }

    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                Some((o1, o2)) => &src[o1..o2],
                None => candle_core::bail!("stable_softplus: input must be contiguous"),
            };
            let t88 = T::from(88.0).unwrap();
            let one = T::one();
            let dst: Vec<T> = src
                .iter()
                .map(|&x| {
                    let clamped = if x < t88 { x } else { t88 };
                    let sp = (clamped.exp() + one).ln();
                    if x > sp { x } else { sp }
                })
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, layout.shape().clone()))
        }

        use CpuStorage as C;
        match s {
            C::BF16(s) => inner(s, l),
            C::F16(s) => inner(s, l),
            C::F32(s) => inner(s, l),
            C::F64(s) => inner(s, l),
            _ => candle_core::bail!("stable_softplus: unsupported dtype {:?}", s.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map1, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        struct S;
        impl Map1 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                src: &CudaSlice<T>,
                dev: &CudaDevice,
                layout: &Layout,
            ) -> Result<CudaSlice<T>> {
                let src = match layout.contiguous_offsets() {
                    Some((o1, o2)) => src.slice(o1..o2),
                    None => candle_core::bail!("stable_softplus: input must be contiguous"),
                };
                let el = layout.shape().elem_count();
                let cfg = LaunchConfig::for_num_elems(el as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("stable_softplus"),
                    "cake_fused_ops",
                    super::FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&src);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s.device();
        let slice = S.map(&s.slice, dev, l)?;
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            l.shape().clone(),
        ))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s: &candle_core::MetalStorage,
        l: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        use candle_core::DType;
        let device = s.device();
        let el = l.shape().elem_count();
        let kernel_name = match s.dtype() {
            DType::F32 => "stable_softplus_f32",
            DType::F16 => "stable_softplus_f16",
            dt => candle_core::bail!("stable_softplus metal: unsupported dtype {dt:?}"),
        };
        let lib = device.new_library_with_source(super::FUSED_OPS_MSL, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal shader compile: {e}")))?;
        let func = lib.get_function(kernel_name, None)
            .map_err(|e| candle_core::Error::Msg(format!("metal get_function: {e}")))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| candle_core::Error::Msg(format!("metal pipeline: {e}")))?;
        let output = device.new_buffer(el, s.dtype(), "stable_softplus")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let offset = l.start_offset() * s.dtype().size_in_bytes();
        candle_metal_kernels::utils::set_param(&encoder, 0, (s.buffer(), offset));
        candle_metal_kernels::utils::set_param(&encoder, 1, (&*output, 0usize));
        candle_metal_kernels::utils::set_param(&encoder, 2, el as u32);
        let grid = objc2_metal::MTLSize { width: el, height: 1, depth: 1 };
        let group = candle_metal_kernels::utils::get_block_dims(el, 1, 1);
        encoder.dispatch_threads(grid, group);
        Ok((candle_core::MetalStorage::new(output, device.clone(), el, s.dtype()), l.shape().clone()))
    }
}

/// Fused stable softplus: ln(1 + exp(clamp(x, -inf, 88))) with max(x, result).
///
/// On CUDA: single fused kernel (1 dispatch).
/// On Metal: single fused kernel via MSL shader (1 dispatch).
/// On CPU: scalar implementation via CustomOp1.
pub fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&StableSoftplus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_silu_mul_correctness() {
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.0, -2.0], &Device::Cpu).unwrap();
        let up = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &Device::Cpu).unwrap();

        let fused: Vec<f32> = silu_mul(&gate, &up).unwrap().to_vec1().unwrap();

        // Reference: silu(gate) * up
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate).unwrap() * &up)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-6),
            "silu_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[test]
    fn test_silu_mul_2d() {
        let gate = Tensor::new(&[[1.0f32, 2.0], [-1.0, 0.5]], &Device::Cpu).unwrap();
        let up = Tensor::new(&[[0.5f32, -1.0], [2.0, 3.0]], &Device::Cpu).unwrap();

        let fused = silu_mul(&gate, &up).unwrap();
        assert_eq!(fused.dims(), &[2, 2]);

        let fused: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate).unwrap() * &up)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(approx_eq(&fused, &reference, 1e-6));
    }

    #[test]
    fn test_stable_softplus_correctness() {
        let x = Tensor::new(&[-10.0f32, -1.0, 0.0, 1.0, 10.0, 100.0], &Device::Cpu).unwrap();
        let result: Vec<f32> = stable_softplus(&x).unwrap().to_vec1().unwrap();

        // Reference values
        let expected: Vec<f32> = vec![
            ((-10.0f32).exp() + 1.0).ln(),          // ~4.5e-5
            ((-1.0f32).exp() + 1.0).ln(),            // ~0.3133
            (0.0f32.exp() + 1.0).ln(),               // ~0.6931
            (1.0f32.exp() + 1.0).ln(),               // ~1.3133
            (10.0f32.exp() + 1.0).ln(),              // ~10.0
            100.0,                                     // clamped: max(100, softplus(88))
        ];
        assert!(
            approx_eq(&result, &expected, 1e-4),
            "softplus mismatch: result={result:?} expected={expected:?}"
        );
    }

    #[test]
    fn test_stable_softplus_matches_original() {
        let x = Tensor::new(&[-5.0f32, -1.0, 0.0, 1.0, 5.0, 50.0], &Device::Cpu).unwrap();

        let fused: Vec<f32> = stable_softplus(&x).unwrap().to_vec1().unwrap();

        // Original implementation from linear_attention.rs
        let sp = (x.minimum(88f64).unwrap().exp().unwrap() + 1.0)
            .unwrap()
            .log()
            .unwrap();
        let reference: Vec<f32> = x.maximum(&sp).unwrap().to_vec1().unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "softplus vs original: fused={fused:?} ref={reference:?}"
        );
    }

    #[test]
    fn test_silu_mul_f16() {
        let gate = Tensor::new(&[1.0f32, -1.0, 2.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let up = Tensor::new(&[2.0f32, 3.0, 0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let result = silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dtype(), DType::F16);
        assert_eq!(result.dims(), &[3]);
    }

    // ── CUDA tests ───────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_silu_mul_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return, // skip if no GPU
        };
        let gate = Tensor::new(&[0.0f32, 1.0, -1.0, 2.0, -2.0], &dev).unwrap();
        let up = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &dev).unwrap();

        let fused: Vec<f32> = silu_mul(&gate, &up).unwrap().to_vec1().unwrap();

        // Reference on CPU
        let gate_cpu = gate.to_device(&Device::Cpu).unwrap();
        let up_cpu = up.to_device(&Device::Cpu).unwrap();
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate_cpu).unwrap() * &up_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA silu_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_silu_mul_cuda_f16() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let gate = Tensor::new(&[1.0f32, -1.0, 2.0, -0.5], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&dev)
            .unwrap();
        let up = Tensor::new(&[2.0f32, 3.0, 0.5, -1.0], &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
            .to_device(&dev)
            .unwrap();

        let result = silu_mul(&gate, &up).unwrap();
        assert_eq!(result.dtype(), DType::F16);

        let fused: Vec<f32> = result
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference
        let gate_cpu = gate.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
        let up_cpu = up.to_device(&Device::Cpu).unwrap().to_dtype(DType::F32).unwrap();
        let reference: Vec<f32> = (candle_nn::ops::silu(&gate_cpu).unwrap() * &up_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-2), // F16 has less precision
            "CUDA F16 silu_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_stable_softplus_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(&[-10.0f32, -1.0, 0.0, 1.0, 10.0, 100.0], &dev).unwrap();
        let result: Vec<f32> = stable_softplus(&x).unwrap().to_vec1().unwrap();

        let expected: Vec<f32> = vec![
            ((-10.0f32).exp() + 1.0).ln(),
            ((-1.0f32).exp() + 1.0).ln(),
            (0.0f32.exp() + 1.0).ln(),
            (1.0f32.exp() + 1.0).ln(),
            (10.0f32.exp() + 1.0).ln(),
            100.0,
        ];
        assert!(
            approx_eq(&result, &expected, 1e-4),
            "CUDA softplus mismatch: result={result:?} expected={expected:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_silu_mul_cuda_large() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Test with realistic hidden size
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let n = 1024;
        let gate_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();
        let up_data: Vec<f32> = (0..n).map(|_| rng.gen_range(-2.0..2.0)).collect();

        let gate_cpu = Tensor::new(gate_data.as_slice(), &Device::Cpu).unwrap();
        let up_cpu = Tensor::new(up_data.as_slice(), &Device::Cpu).unwrap();
        let gate_gpu = gate_cpu.to_device(&dev).unwrap();
        let up_gpu = up_cpu.to_device(&dev).unwrap();

        let gpu_result: Vec<f32> = silu_mul(&gate_gpu, &up_gpu).unwrap().to_vec1().unwrap();
        let cpu_result: Vec<f32> = silu_mul(&gate_cpu, &up_cpu).unwrap().to_vec1().unwrap();

        assert!(
            approx_eq(&gpu_result, &cpu_result, 1e-5),
            "CUDA vs CPU silu_mul mismatch on 1024 elements"
        );
    }
}
