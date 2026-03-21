//! Fused elementwise kernels: add3, exp_mul, sub_mul, add_scaled.

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

// ─── add3: a + b + c ────────────────────────────────────────────────

struct Add3;

impl candle_core::CustomOp3 for Add3 {
    fn name(&self) -> &'static str {
        "add3"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_c: &CpuStorage,
        l_c: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            c: &[T],
            l_c: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("add3: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("add3: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => &c[o1..o2],
                None => candle_core::bail!("add3: c must be contiguous"),
            };
            let dst: Vec<T> = a
                .iter()
                .zip(b)
                .zip(c)
                .map(|((&av, &bv), &cv)| av + bv + cv)
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_a.shape().clone()))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_c) {
            (C::BF16(a), C::BF16(b), C::BF16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F16(a), C::F16(b), C::F16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F32(a), C::F32(b), C::F32(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F64(a), C::F64(b), C::F64(c)) => inner(a, l_a, b, l_b, c, l_c),
            _ => candle_core::bail!("add3: unsupported dtype {:?}", s_a.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_c: &candle_core::CudaStorage,
        l_c: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>,
            l_a: &Layout,
            b: &CudaSlice<T>,
            l_b: &Layout,
            c: &CudaSlice<T>,
            l_c: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("add3: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("add3: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => c.slice(o1..o2),
                None => candle_core::bail!("add3: c must be contiguous"),
            };
            let el = l_a.shape().elem_count();
            let cfg = LaunchConfig::for_num_elems(el as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("add3"),
                "cake_fused_ops",
                super::FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&el);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as SS;
        let dev = s_a.device();

        let slice = match (&s_a.slice, &s_b.slice, &s_c.slice) {
            (SS::BF16(a), SS::BF16(b), SS::BF16(c)) => {
                SS::BF16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F16(a), SS::F16(b), SS::F16(c)) => {
                SS::F16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F32(a), SS::F32(b), SS::F32(c)) => {
                SS::F32(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F64(a), SS::F64(b), SS::F64(c)) => {
                SS::F64(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            _ => candle_core::bail!("add3: unsupported dtype"),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            l_a.shape().clone(),
        ))
    }
}

/// Fused a + b + c — replaces 2 kernel launches (add + add) with 1.
pub fn add3(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.apply_op3_no_bwd(b, c, &Add3)
}

// ─── exp_mul: x * exp(y) ────────────────────────────────────────────

struct ExpMul;

impl candle_core::CustomOp2 for ExpMul {
    fn name(&self) -> &'static str {
        "exp_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            x: &[T],
            l1: &Layout,
            y: &[T],
            l2: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l1.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("exp_mul: x must be contiguous"),
            };
            let y = match l2.contiguous_offsets() {
                Some((o1, o2)) => &y[o1..o2],
                None => candle_core::bail!("exp_mul: y must be contiguous"),
            };
            let dst: Vec<T> = x.iter().zip(y).map(|(&a, &b)| a * b.exp()).collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l1.shape().clone()))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(a), C::BF16(b)) => inner(a, l1, b, l2),
            (C::F16(a), C::F16(b)) => inner(a, l1, b, l2),
            (C::F32(a), C::F32(b)) => inner(a, l1, b, l2),
            (C::F64(a), C::F64(b)) => inner(a, l1, b, l2),
            _ => candle_core::bail!("exp_mul: unsupported dtype {:?}", s1.dtype()),
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
                x: &CudaSlice<T>,
                l1: &Layout,
                y: &CudaSlice<T>,
                l2: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let x = match l1.contiguous_offsets() {
                    Some((o1, o2)) => x.slice(o1..o2),
                    None => candle_core::bail!("exp_mul: x must be contiguous"),
                };
                let y = match l2.contiguous_offsets() {
                    Some((o1, o2)) => y.slice(o1..o2),
                    None => candle_core::bail!("exp_mul: y must be contiguous"),
                };
                let el = l1.shape().elem_count();
                let cfg = LaunchConfig::for_num_elems(el as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("exp_mul"),
                    "cake_fused_ops",
                    super::FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(el)? };
                let mut builder = func.builder();
                builder.arg(&el);
                builder.arg(&x);
                builder.arg(&y);
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
}

/// Fused x * exp(y) — replaces 2 kernel launches (exp + mul) with 1.
pub fn exp_mul(x: &Tensor, y: &Tensor) -> Result<Tensor> {
    x.apply_op2_no_bwd(y, &ExpMul)
}

// ─── sub_mul: (a - b) * c ───────────────────────────────────────────

struct SubMul;

impl candle_core::CustomOp3 for SubMul {
    fn name(&self) -> &'static str {
        "sub_mul"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_c: &CpuStorage,
        l_c: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            c: &[T],
            l_c: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("sub_mul: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("sub_mul: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => &c[o1..o2],
                None => candle_core::bail!("sub_mul: c must be contiguous"),
            };
            let dst: Vec<T> = a
                .iter()
                .zip(b)
                .zip(c)
                .map(|((&av, &bv), &cv)| (av - bv) * cv)
                .collect();
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_a.shape().clone()))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_c) {
            (C::BF16(a), C::BF16(b), C::BF16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F16(a), C::F16(b), C::F16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F32(a), C::F32(b), C::F32(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F64(a), C::F64(b), C::F64(c)) => inner(a, l_a, b, l_b, c, l_c),
            _ => candle_core::bail!("sub_mul: unsupported dtype {:?}", s_a.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_c: &candle_core::CudaStorage,
        l_c: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>,
            l_a: &Layout,
            b: &CudaSlice<T>,
            l_b: &Layout,
            c: &CudaSlice<T>,
            l_c: &Layout,
            dev: &CudaDevice,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("sub_mul: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("sub_mul: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => c.slice(o1..o2),
                None => candle_core::bail!("sub_mul: c must be contiguous"),
            };
            let el = l_a.shape().elem_count();
            let cfg = LaunchConfig::for_num_elems(el as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("sub_mul"),
                "cake_fused_ops",
                super::FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&el);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
            builder.arg(&out);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as SS;
        let dev = s_a.device();

        let slice = match (&s_a.slice, &s_b.slice, &s_c.slice) {
            (SS::BF16(a), SS::BF16(b), SS::BF16(c)) => {
                SS::BF16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F16(a), SS::F16(b), SS::F16(c)) => {
                SS::F16(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F32(a), SS::F32(b), SS::F32(c)) => {
                SS::F32(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            (SS::F64(a), SS::F64(b), SS::F64(c)) => {
                SS::F64(launch(a, l_a, b, l_b, c, l_c, dev)?)
            }
            _ => candle_core::bail!("sub_mul: unsupported dtype"),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            l_a.shape().clone(),
        ))
    }
}

/// Fused (a - b) * c — replaces 2 kernel launches (sub + mul) with 1.
pub fn sub_mul(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.apply_op3_no_bwd(b, c, &SubMul)
}

// ─── add_scaled: a + b * c with broadcast on c ─────────────────────

struct AddScaled;

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for AddScaled {
    fn name(&self) -> &'static str {
        "add_scaled"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_c: &CpuStorage,
        l_c: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            c: &[T],
            l_c: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("add_scaled: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("add_scaled: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => &c[o1..o2],
                None => candle_core::bail!("add_scaled: c must be contiguous"),
            };
            let dims = l_a.shape().dims();
            let (channels, time_len) = (dims[1], dims[2]);
            let numel = l_a.shape().elem_count();
            let mut dst = vec![T::zero(); numel];
            for (i, d) in dst.iter_mut().enumerate() {
                let chan = (i / time_len) % channels;
                *d = a[i] + b[i] * c[chan];
            }
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_a.shape().clone()))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_c) {
            (C::BF16(a), C::BF16(b), C::BF16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F16(a), C::F16(b), C::F16(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F32(a), C::F32(b), C::F32(c)) => inner(a, l_a, b, l_b, c, l_c),
            (C::F64(a), C::F64(b), C::F64(c)) => inner(a, l_a, b, l_b, c, l_c),
            _ => candle_core::bail!("add_scaled: unsupported dtype"),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_c: &candle_core::CudaStorage,
        l_c: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let dims = l_a.shape().dims();
        let (channels, time_len) = (dims[1], dims[2]);
        let numel = l_a.shape().elem_count();

        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>, l_a: &Layout,
            b: &CudaSlice<T>, l_b: &Layout,
            c: &CudaSlice<T>, l_c: &Layout,
            dev: &CudaDevice,
            channels: i32, time_len: i32, numel: usize,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("add_scaled: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("add_scaled: b must be contiguous"),
            };
            let c = match l_c.contiguous_offsets() {
                Some((o1, o2)) => c.slice(o1..o2),
                None => candle_core::bail!("add_scaled: c must be contiguous"),
            };
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("add_scaled"),
                "cake_fused_ops",
                super::FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(numel)? };
            let mut builder = func.builder();
            builder.arg(&numel);
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
            builder.arg(&out);
            candle_core::builder_arg!(builder, channels, time_len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_a.device();
        let ch = channels as i32;
        let tl = time_len as i32;

        let slice = match (&s_a.slice, &s_b.slice, &s_c.slice) {
            (S::BF16(a), S::BF16(b), S::BF16(c)) => S::BF16(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            (S::F16(a), S::F16(b), S::F16(c)) => S::F16(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            (S::F32(a), S::F32(b), S::F32(c)) => S::F32(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            (S::F64(a), S::F64(b), S::F64(c)) => S::F64(launch(a, l_a, b, l_b, c, l_c, dev, ch, tl, numel)?),
            _ => candle_core::bail!("add_scaled: unsupported dtype"),
        };

        Ok((
            candle_core::CudaStorage { slice, device: dev.clone() },
            l_a.shape().clone(),
        ))
    }
}

/// Fused a + b * c where c is (channels,) broadcast over (batch, channels, time).
/// Replaces broadcast_mul + add (2 kernels) with 1.
pub fn add_scaled(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    let a = a.contiguous()?;
    let b = b.contiguous()?;
    let c = c.contiguous()?;
    a.apply_op3_no_bwd(&b, &c, &AddScaled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── add3 tests ──────────────────────────────────────────────────

    #[test]
    fn test_add3_correctness() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[10.0f32, 20.0, 30.0, 40.0], &Device::Cpu).unwrap();
        let c = Tensor::new(&[100.0f32, 200.0, 300.0, 400.0], &Device::Cpu).unwrap();
        let result: Vec<f32> = add3(&a, &b, &c).unwrap().to_vec1().unwrap();
        assert!(approx_eq(&result, &[111.0, 222.0, 333.0, 444.0], 1e-5));
    }

    #[test]
    fn test_add3_2d_shape() {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &Device::Cpu).unwrap();
        let c = Tensor::new(&[[0.01f32, 0.02], [0.03, 0.04]], &Device::Cpu).unwrap();
        let result = add3(&a, &b, &c).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(approx_eq(&vals, &[1.11, 2.22, 3.33, 4.44], 1e-4));
    }

    // ── exp_mul tests ───────────────────────────────────────────────

    #[test]
    fn test_exp_mul_zero_exponent() {
        // exp(0) = 1, so x * exp(0) = x
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu).unwrap();
        let y = Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap();
        let result: Vec<f32> = exp_mul(&x, &y).unwrap().to_vec1().unwrap();
        assert!(approx_eq(&result, &[1.0, 2.0, 3.0], 1e-6));
    }

    #[test]
    fn test_exp_mul_correctness() {
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5], &Device::Cpu).unwrap();
        let y = Tensor::new(&[0.0f32, 1.0, -1.0, 0.5], &Device::Cpu).unwrap();

        let fused: Vec<f32> = exp_mul(&x, &y).unwrap().to_vec1().unwrap();
        let reference: Vec<f32> = (&x * y.exp().unwrap())
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-6),
            "exp_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    // ── sub_mul tests ───────────────────────────────────────────────

    #[test]
    fn test_sub_mul_zero_diff() {
        let a = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let b = Tensor::new(&[1.0f32, 2.0], &Device::Cpu).unwrap();
        let c = Tensor::new(&[999.0f32, 999.0], &Device::Cpu).unwrap();
        let result: Vec<f32> = sub_mul(&a, &b, &c).unwrap().to_vec1().unwrap();
        assert!(approx_eq(&result, &[0.0, 0.0], 1e-6));
    }

    #[test]
    fn test_sub_mul_correctness() {
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5], &Device::Cpu).unwrap();
        let b = Tensor::new(&[0.5f32, 1.0, 4.0, -0.5], &Device::Cpu).unwrap();
        let c = Tensor::new(&[2.0f32, 0.5, -1.0, 3.0], &Device::Cpu).unwrap();

        let fused: Vec<f32> = sub_mul(&a, &b, &c).unwrap().to_vec1().unwrap();
        let reference: Vec<f32> = ((&a - &b).unwrap() * &c)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-6),
            "sub_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    // ── add_scaled tests ────────────────────────────────────────────

    #[test]
    fn test_add_scaled_cpu_correctness() {
        // a: (1, 2, 3), b: (1, 2, 3), c: (2,)
        let a = Tensor::new(&[1f32, 2., 3., 4., 5., 6.], &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let b = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let c = Tensor::new(&[10.0f32, 20.0], &Device::Cpu).unwrap();

        let out = add_scaled(&a, &b, &c).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // chan0 (scale=10): [1+0.1*10, 2+0.2*10, 3+0.3*10] = [2, 4, 6]
        // chan1 (scale=20): [4+0.4*20, 5+0.5*20, 6+0.6*20] = [12, 15, 18]
        assert!(approx_eq(&vals, &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0], 1e-5));
    }

    // ── CUDA tests ───────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_exp_mul_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &dev).unwrap();
        let y = Tensor::new(&[0.0f32, 1.0, -1.0, 0.5, -2.0], &dev).unwrap();

        let fused: Vec<f32> = exp_mul(&x, &y).unwrap().to_vec1().unwrap();

        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let y_cpu = y.to_device(&Device::Cpu).unwrap();
        let reference: Vec<f32> = (&x_cpu * y_cpu.exp().unwrap())
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA exp_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_sub_mul_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let a = Tensor::new(&[1.0f32, 2.0, 3.0, 0.5, -1.0], &dev).unwrap();
        let b = Tensor::new(&[0.5f32, 1.0, 4.0, -0.5, 2.0], &dev).unwrap();
        let c = Tensor::new(&[2.0f32, 0.5, -1.0, 3.0, 0.1], &dev).unwrap();

        let fused: Vec<f32> = sub_mul(&a, &b, &c).unwrap().to_vec1().unwrap();

        let a_cpu = a.to_device(&Device::Cpu).unwrap();
        let b_cpu = b.to_device(&Device::Cpu).unwrap();
        let c_cpu = c.to_device(&Device::Cpu).unwrap();
        let reference: Vec<f32> = ((&a_cpu - &b_cpu).unwrap() * &c_cpu)
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA sub_mul mismatch: fused={fused:?} ref={reference:?}"
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_add_scaled_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // (1, 4, 3) tensor
        let a = Tensor::new(&[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.], &dev)
            .unwrap()
            .reshape((1, 4, 3))
            .unwrap();
        let b = Tensor::new(&[0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], &dev)
            .unwrap()
            .reshape((1, 4, 3))
            .unwrap();
        let c = Tensor::new(&[2.0f32, 3.0, 4.0, 5.0], &dev).unwrap();
        let out: Vec<f32> = add_scaled(&a, &b, &c)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // Expected: a[i] + b[i] * c[chan]
        // chan 0 (scale=2): [1+0.1*2, 2+0.2*2, 3+0.3*2] = [1.2, 2.4, 3.6]
        assert!((out[0] - 1.2).abs() < 1e-5, "wrong: {}", out[0]);
        assert!((out[1] - 2.4).abs() < 1e-5, "wrong: {}", out[1]);
    }
}
