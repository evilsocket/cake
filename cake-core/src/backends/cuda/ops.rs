//! CUDA CustomOp implementations for all 13 fused operations.
//!
//! This module is gated by `#[cfg(feature = "cuda")]` at the parent level,
//! so individual methods do NOT carry cfg guards.

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/fused_ops_ptx.rs"));
}
pub(super) const FUSED_OPS_PTX: &str = ptx::FUSED_OPS;

// ─── SiluMul: silu(gate) * up ──────────────────────────────────────

pub(super) struct SiluMul;

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
                    FUSED_OPS_PTX,
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
}

// ─── StableSoftplus: ln(1+exp(clamp(x,-inf,88))) with max(x,.) ─────

pub(super) struct StableSoftplus;

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
                    FUSED_OPS_PTX,
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
}

// ─── Add3: a + b + c ────────────────────────────────────────────────

pub(super) struct Add3;

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
                FUSED_OPS_PTX,
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

// ─── ExpMul: x * exp(y) ─────────────────────────────────────────────

pub(super) struct ExpMul;

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
                    FUSED_OPS_PTX,
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

// ─── SubMul: (a - b) * c ────────────────────────────────────────────

pub(super) struct SubMul;

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
                FUSED_OPS_PTX,
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

// ─── AddScaled: a + b * c with broadcast on c ──────────────────────

pub(super) struct AddScaled;

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
                FUSED_OPS_PTX,
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

// ─── RmsNormGated: rms_norm(x, weight) * silu(z) ────────────────────

pub(super) struct RmsNormGated {
    pub eps: f32,
}

impl candle_core::CustomOp3 for RmsNormGated {
    fn name(&self) -> &'static str {
        "rms_norm_gated"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_z: &CpuStorage,
        l_z: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use rayon::prelude::*;

        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            x: &[T],
            l_x: &Layout,
            z: &[T],
            l_z: &Layout,
            w: &[T],
            l_w: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("rms_norm_gated: x must be contiguous"),
            };
            let z = match l_z.contiguous_offsets() {
                Some((o1, o2)) => &z[o1..o2],
                None => candle_core::bail!("rms_norm_gated: z must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("rms_norm_gated: weight must be contiguous"),
            };
            let dims = l_x.shape().dims();
            let n_cols = dims[dims.len() - 1];
            let el = l_x.shape().elem_count();
            let mut dst = vec![T::zero(); el];

            dst.par_chunks_mut(n_cols)
                .enumerate()
                .for_each(|(row, dst_row)| {
                    let x_row = &x[row * n_cols..(row + 1) * n_cols];
                    let z_row = &z[row * n_cols..(row + 1) * n_cols];
                    let sum2: f32 = x_row.iter().map(|v| { let f: f32 = v.as_(); f * f }).sum();
                    let inv_rms = 1.0 / (sum2 / n_cols as f32 + eps).sqrt();
                    for i in 0..n_cols {
                        let xv: f32 = x_row[i].as_();
                        let wv: f32 = w[i].as_();
                        let zv: f32 = z_row[i].as_();
                        let silu_z = zv / (1.0 + (-zv).exp());
                        dst_row[i] = T::from_f32(xv * inv_rms * wv * silu_z)
                            .unwrap_or_else(T::nan);
                    }
                });

            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s_x, s_z, s_w) {
            (C::BF16(x), C::BF16(z), C::BF16(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            (C::F16(x), C::F16(z), C::F16(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            (C::F32(x), C::F32(z), C::F32(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            (C::F64(x), C::F64(z), C::F64(w)) => inner(x, l_x, z, l_z, w, l_w, self.eps),
            _ => candle_core::bail!("rms_norm_gated: unsupported dtype {:?}", s_x.dtype()),
        }
    }

    fn cuda_fwd(
        &self,
        s_x: &candle_core::CudaStorage,
        l_x: &Layout,
        s_z: &candle_core::CudaStorage,
        l_z: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            x: &CudaSlice<T>,
            l_x: &Layout,
            z: &CudaSlice<T>,
            l_z: &Layout,
            w: &CudaSlice<T>,
            l_w: &Layout,
            dev: &CudaDevice,
            eps: f32,
        ) -> Result<CudaSlice<T>> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => x.slice(o1..o2),
                None => candle_core::bail!("rms_norm_gated: x must be contiguous"),
            };
            let z = match l_z.contiguous_offsets() {
                Some((o1, o2)) => z.slice(o1..o2),
                None => candle_core::bail!("rms_norm_gated: z must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => w.slice(o1..o2),
                None => candle_core::bail!("rms_norm_gated: weight must be contiguous"),
            };

            let dims = l_x.shape().dims();
            let n_cols = dims[dims.len() - 1];
            let el = l_x.shape().elem_count();
            let n_rows = el / n_cols;

            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };

            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("rms_norm_gated"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&z);
            builder.arg(&w);
            builder.arg(&out);
            candle_core::builder_arg!(builder, n_cols as i32, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_x.device();

        let slice = match (&s_x.slice, &s_z.slice, &s_w.slice) {
            (S::BF16(x), S::BF16(z), S::BF16(w)) => {
                S::BF16(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            (S::F16(x), S::F16(z), S::F16(w)) => {
                S::F16(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            (S::F32(x), S::F32(z), S::F32(w)) => {
                S::F32(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            (S::F64(x), S::F64(z), S::F64(w)) => {
                S::F64(launch(x, l_x, z, l_z, w, l_w, dev, self.eps)?)
            }
            _ => candle_core::bail!("rms_norm_gated: unsupported dtype"),
        };

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            l_x.shape().clone(),
        ))
    }
}

// ─── AddRmsNorm: rms_norm(a + b, weight, eps) with residual ─────────

pub(super) struct AddRmsNorm {
    pub eps: f32,
    pub n_cols: usize,
}

impl candle_core::CustomOp3 for AddRmsNorm {
    fn name(&self) -> &'static str {
        "add_rms_norm"
    }

    fn cpu_fwd(
        &self,
        s_a: &CpuStorage,
        l_a: &Layout,
        s_b: &CpuStorage,
        l_b: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use rayon::prelude::*;

        #[allow(clippy::too_many_arguments)]
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            a: &[T],
            l_a: &Layout,
            b: &[T],
            l_b: &Layout,
            w: &[T],
            l_w: &Layout,
            eps: f32,
            n_cols: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => &a[o1..o2],
                None => candle_core::bail!("add_rms_norm: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => &b[o1..o2],
                None => candle_core::bail!("add_rms_norm: b must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("add_rms_norm: weight must be contiguous"),
            };
            let dims = l_a.shape().dims();
            let el = l_a.shape().elem_count();
            let n_rows = el / n_cols;
            // Output: [sum, normed] concatenated on last dim
            let out_cols = n_cols * 2;
            let mut dst = vec![T::zero(); n_rows * out_cols];

            dst.par_chunks_mut(out_cols)
                .enumerate()
                .for_each(|(row, dst_row)| {
                    let a_row = &a[row * n_cols..(row + 1) * n_cols];
                    let b_row = &b[row * n_cols..(row + 1) * n_cols];
                    // Compute sum and sum of squares
                    let mut sum2: f32 = 0.0;
                    for i in 0..n_cols {
                        let s: f32 = a_row[i].as_() + b_row[i].as_();
                        dst_row[i] = T::from_f32(s).unwrap_or_else(T::nan);
                        sum2 += s * s;
                    }
                    let inv_rms = 1.0 / (sum2 / n_cols as f32 + eps).sqrt();
                    for i in 0..n_cols {
                        let s: f32 = dst_row[i].as_();
                        let wv: f32 = w[i].as_();
                        dst_row[n_cols + i] =
                            T::from_f32(s * inv_rms * wv).unwrap_or_else(T::nan);
                    }
                });

            let mut out_dims = dims.to_vec();
            *out_dims.last_mut().unwrap() = out_cols;
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(&out_dims)))
        }

        use CpuStorage as C;
        match (s_a, s_b, s_w) {
            (C::BF16(a), C::BF16(b), C::BF16(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            (C::F16(a), C::F16(b), C::F16(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            (C::F32(a), C::F32(b), C::F32(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            (C::F64(a), C::F64(b), C::F64(w)) => {
                inner(a, l_a, b, l_b, w, l_w, self.eps, self.n_cols)
            }
            _ => candle_core::bail!("add_rms_norm: unsupported dtype {:?}", s_a.dtype()),
        }
    }

    fn cuda_fwd(
        &self,
        s_a: &candle_core::CudaStorage,
        l_a: &Layout,
        s_b: &candle_core::CudaStorage,
        l_b: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, CudaStorage, WithDType};

        let n_cols = self.n_cols;
        let eps = self.eps;
        let el = l_a.shape().elem_count();
        let n_rows = el / n_cols;
        let out_cols = n_cols * 2;

        #[allow(clippy::too_many_arguments)]
        fn launch<T: DeviceRepr + WithDType>(
            a: &CudaSlice<T>,
            l_a: &Layout,
            b: &CudaSlice<T>,
            l_b: &Layout,
            w: &CudaSlice<T>,
            l_w: &Layout,
            dev: &CudaDevice,
            n_cols: usize,
            n_rows: usize,
            eps: f32,
        ) -> Result<CudaSlice<T>> {
            let a = match l_a.contiguous_offsets() {
                Some((o1, o2)) => a.slice(o1..o2),
                None => candle_core::bail!("add_rms_norm: a must be contiguous"),
            };
            let b = match l_b.contiguous_offsets() {
                Some((o1, o2)) => b.slice(o1..o2),
                None => candle_core::bail!("add_rms_norm: b must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => w.slice(o1..o2),
                None => candle_core::bail!("add_rms_norm: weight must be contiguous"),
            };
            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("add_rms_norm"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out_el = n_rows * n_cols * 2;
            let out = unsafe { dev.alloc::<T>(out_el)? };
            let mut builder = func.builder();
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&w);
            builder.arg(&out);
            candle_core::builder_arg!(builder, n_cols as i32, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as SS;
        let dev = s_a.device();

        let slice = match (&s_a.slice, &s_b.slice, &s_w.slice) {
            (SS::BF16(a), SS::BF16(b), SS::BF16(w)) => {
                SS::BF16(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            (SS::F16(a), SS::F16(b), SS::F16(w)) => {
                SS::F16(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            (SS::F32(a), SS::F32(b), SS::F32(w)) => {
                SS::F32(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            (SS::F64(a), SS::F64(b), SS::F64(w)) => {
                SS::F64(launch(a, l_a, b, l_b, w, l_w, dev, n_cols, n_rows, eps)?)
            }
            _ => candle_core::bail!("add_rms_norm: unsupported dtype"),
        };

        let mut out_dims = l_a.shape().dims().to_vec();
        *out_dims.last_mut().unwrap() = out_cols;

        Ok((
            CudaStorage {
                slice,
                device: dev.clone(),
            },
            Shape::from_dims(&out_dims),
        ))
    }
}

// ─── RmsNormChannel: RMS-normalize over channel dim of (b,c,t) ──────

pub(super) struct RmsNormChannel {
    pub eps: f32,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp2 for RmsNormChannel {
    fn name(&self) -> &'static str {
        "rms_norm_channel"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            x: &[T],
            l_x: &Layout,
            w: &[T],
            l_w: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("rms_norm_channel: x must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("rms_norm_channel: weight must be contiguous"),
            };
            let dims = l_x.shape().dims();
            let (batch, channels, time_len) = (dims[0], dims[1], dims[2]);
            let mut dst = vec![T::zero(); batch * channels * time_len];
            for b in 0..batch {
                for t in 0..time_len {
                    let mut sum2 = 0f32;
                    for c in 0..channels {
                        let v: f32 = x[b * channels * time_len + c * time_len + t].as_();
                        sum2 += v * v;
                    }
                    let inv_rms = 1.0f32 / (sum2 / channels as f32 + eps).sqrt();
                    for (c, wv_t) in w.iter().enumerate().take(channels) {
                        let off = b * channels * time_len + c * time_len + t;
                        let xv: f32 = x[off].as_();
                        let wv: f32 = (*wv_t).as_();
                        dst[off] = T::from_f32(xv * inv_rms * wv).unwrap_or(T::zero());
                    }
                }
            }
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_x.shape().clone()))
        }

        use CpuStorage as C;
        match (s_x, s_w) {
            (C::BF16(x), C::BF16(w)) => inner(x, l_x, w, l_w, self.eps),
            (C::F16(x), C::F16(w)) => inner(x, l_x, w, l_w, self.eps),
            (C::F32(x), C::F32(w)) => inner(x, l_x, w, l_w, self.eps),
            (C::F64(x), C::F64(w)) => inner(x, l_x, w, l_w, self.eps),
            _ => candle_core::bail!("rms_norm_channel: unsupported dtype"),
        }
    }

    fn cuda_fwd(
        &self,
        s_x: &candle_core::CudaStorage,
        l_x: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let dims = l_x.shape().dims();
        let (batch, channels, time_len) = (dims[0], dims[1], dims[2]);
        let n_rows = batch * time_len; // one block per (batch, time) position
        let el = l_x.shape().elem_count();
        let eps = self.eps;

        fn launch<T: DeviceRepr + WithDType>(
            x: &CudaSlice<T>,
            l_x: &Layout,
            w: &CudaSlice<T>,
            l_w: &Layout,
            dev: &CudaDevice,
            channels: i32,
            time_len: i32,
            n_rows: usize,
            el: usize,
            eps: f32,
        ) -> Result<CudaSlice<T>> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => x.slice(o1..o2),
                None => candle_core::bail!("rms_norm_channel: x must be contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => w.slice(o1..o2),
                None => candle_core::bail!("rms_norm_channel: weight must be contiguous"),
            };
            let block_size: u32 = if channels < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (n_rows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("rms_norm_channel"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&x);
            builder.arg(&w);
            builder.arg(&out);
            candle_core::builder_arg!(builder, channels, time_len, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_x.device();
        let ch = channels as i32;
        let tl = time_len as i32;

        let slice = match (&s_x.slice, &s_w.slice) {
            (S::BF16(x), S::BF16(w)) => S::BF16(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            (S::F16(x), S::F16(w)) => S::F16(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            (S::F32(x), S::F32(w)) => S::F32(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            (S::F64(x), S::F64(w)) => S::F64(launch(x, l_x, w, l_w, dev, ch, tl, n_rows, el, eps)?),
            _ => candle_core::bail!("rms_norm_channel: unsupported dtype"),
        };

        Ok((
            candle_core::CudaStorage { slice, device: dev.clone() },
            l_x.shape().clone(),
        ))
    }
}

// ─── AdaLnModulate: rms_norm(x,w,eps) * (1+scale) + shift ──────────

pub(super) struct AdaLnModulate {
    pub eps: f32,
    pub shift: Tensor,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for AdaLnModulate {
    fn name(&self) -> &'static str {
        "adaln_modulate"
    }

    fn cpu_fwd(
        &self,
        s_x: &CpuStorage,
        l_x: &Layout,
        s_w: &CpuStorage,
        l_w: &Layout,
        s_sc: &CpuStorage,
        l_sc: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<
            T: candle_core::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            x: &[T], l_x: &Layout,
            w: &[T], l_w: &Layout,
            scale: &[T], l_sc: &Layout,
            shift_t: &Tensor, eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let x = match l_x.contiguous_offsets() {
                Some((o1, o2)) => &x[o1..o2],
                None => candle_core::bail!("adaln: x contiguous"),
            };
            let w = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &w[o1..o2],
                None => candle_core::bail!("adaln: w contiguous"),
            };
            let scale = match l_sc.contiguous_offsets() {
                Some((o1, o2)) => &scale[o1..o2],
                None => candle_core::bail!("adaln: scale contiguous"),
            };
            let shift_v: Vec<f32> = shift_t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
            let dims = l_x.shape().dims();
            let n_cols = dims[dims.len() - 1];
            let el = l_x.shape().elem_count();
            let n_rows = el / n_cols;
            let mut dst = vec![T::zero(); el];
            for r in 0..n_rows {
                let off = r * n_cols;
                let mut sum2 = 0f32;
                for c in 0..n_cols {
                    let v: f32 = x[off + c].as_();
                    sum2 += v * v;
                }
                let inv_rms = 1.0f32 / (sum2 / n_cols as f32 + eps).sqrt();
                for c in 0..n_cols {
                    let xv: f32 = x[off + c].as_() * inv_rms;
                    let wv: f32 = w[c].as_();
                    let sv: f32 = scale[off + c].as_();
                    let shv = shift_v[off + c];
                    dst[off + c] = T::from_f32(xv * wv * (1.0 + sv) + shv).unwrap_or(T::zero());
                }
            }
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, l_x.shape().clone()))
        }

        use CpuStorage as C;
        match (s_x, s_w, s_sc) {
            (C::BF16(x), C::BF16(w), C::BF16(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            (C::F16(x), C::F16(w), C::F16(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            (C::F32(x), C::F32(w), C::F32(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            (C::F64(x), C::F64(w), C::F64(s)) => inner(x, l_x, w, l_w, s, l_sc, &self.shift, self.eps),
            _ => candle_core::bail!("adaln: unsupported dtype"),
        }
    }

    fn cuda_fwd(
        &self,
        s_x: &candle_core::CudaStorage,
        l_x: &Layout,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
        s_sc: &candle_core::CudaStorage,
        l_sc: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let dims = l_x.shape().dims();
        let n_cols = dims[dims.len() - 1];
        let el = l_x.shape().elem_count();
        let n_rows = el / n_cols;
        let eps = self.eps;

        let shift_sl = self.shift.storage_and_layout();
        let (shift_storage, shift_layout) = (&*shift_sl.0, &shift_sl.1);

        fn launch<T: DeviceRepr + WithDType>(
            x: &CudaSlice<T>, l_x: &Layout,
            w: &CudaSlice<T>, l_w: &Layout,
            sc: &CudaSlice<T>, l_sc: &Layout,
            sh: &CudaSlice<T>, l_sh: &Layout,
            dev: &CudaDevice,
            n_cols: i32, n_rows: usize, el: usize, eps: f32,
        ) -> Result<CudaSlice<T>> {
            let x = match l_x.contiguous_offsets() { Some((a,b)) => x.slice(a..b), None => candle_core::bail!("adaln: x") };
            let w = match l_w.contiguous_offsets() { Some((a,b)) => w.slice(a..b), None => candle_core::bail!("adaln: w") };
            let sc = match l_sc.contiguous_offsets() { Some((a,b)) => sc.slice(a..b), None => candle_core::bail!("adaln: sc") };
            let sh = match l_sh.contiguous_offsets() { Some((a,b)) => sh.slice(a..b), None => candle_core::bail!("adaln: sh") };
            let block_size: u32 = if n_cols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig { grid_dim: (n_rows as u32, 1, 1), block_dim: (block_size, 1, 1), shared_mem_bytes: 0 };
            let func = dev.get_or_load_custom_func(&kernel_name::<T>("adaln_modulate"), "cake_fused_ops", FUSED_OPS_PTX)?;
            let out = unsafe { dev.alloc::<T>(el)? };
            let mut builder = func.builder();
            builder.arg(&x); builder.arg(&w); builder.arg(&sc); builder.arg(&sh); builder.arg(&out);
            candle_core::builder_arg!(builder, n_cols, block_size as i32, eps);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_x.device();
        let nc = n_cols as i32;

        let sh_s = match shift_storage {
            candle_core::Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("adaln: shift must be on CUDA"),
        };

        let slice = match (&s_x.slice, &s_w.slice, &s_sc.slice, &sh_s.slice) {
            (S::BF16(x), S::BF16(w), S::BF16(sc), S::BF16(sh)) => S::BF16(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            (S::F16(x), S::F16(w), S::F16(sc), S::F16(sh)) => S::F16(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            (S::F32(x), S::F32(w), S::F32(sc), S::F32(sh)) => S::F32(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            (S::F64(x), S::F64(w), S::F64(sc), S::F64(sh)) => S::F64(launch(x, l_x, w, l_w, sc, l_sc, sh, shift_layout, dev, nc, n_rows, el, eps)?),
            _ => candle_core::bail!("adaln: unsupported dtype"),
        };

        Ok((candle_core::CudaStorage { slice, device: dev.clone() }, l_x.shape().clone()))
    }
}

// ─── DepthwiseConv1dSilu: dot(window, weight) per channel + silu ────

pub(super) struct DepthwiseConv1dSilu {
    pub kernel_size: usize,
    pub channels: usize,
}

impl candle_core::CustomOp2 for DepthwiseConv1dSilu {
    fn name(&self) -> &'static str {
        "depthwise_conv1d_silu"
    }

    fn cpu_fwd(
        &self,
        s_w: &CpuStorage,
        l_w: &Layout,
        s_wt: &CpuStorage,
        l_wt: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            window: &[T],
            l_w: &Layout,
            weight: &[T],
            l_wt: &Layout,
            kernel_size: usize,
            channels: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let window = match l_w.contiguous_offsets() {
                Some((o1, o2)) => &window[o1..o2],
                None => candle_core::bail!("conv1d_silu: window must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => &weight[o1..o2],
                None => candle_core::bail!("conv1d_silu: weight must be contiguous"),
            };
            let dims = l_w.shape().dims();
            let batch = dims[0];
            let numel = batch * channels;
            let mut dst = vec![T::zero(); numel];
            for b in 0..batch {
                for c in 0..channels {
                    let mut acc = T::zero();
                    let w_off = b * channels * kernel_size + c * kernel_size;
                    let wt_off = c * kernel_size;
                    for k in 0..kernel_size {
                        acc += window[w_off + k] * weight[wt_off + k];
                    }
                    // silu(acc) = acc * sigmoid(acc)
                    let sig = T::one() / (T::one() + (-acc).exp());
                    dst[b * channels + c] = acc * sig;
                }
            }
            let out_shape = Shape::from_dims(&[batch, channels]);
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, out_shape))
        }

        use CpuStorage as C;
        match (s_w, s_wt) {
            (C::BF16(w), C::BF16(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            (C::F16(w), C::F16(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            (C::F32(w), C::F32(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            (C::F64(w), C::F64(wt)) => inner(w, l_w, wt, l_wt, self.kernel_size, self.channels),
            _ => candle_core::bail!("conv1d_silu: unsupported dtype {:?}", s_w.dtype()),
        }
    }

    fn cuda_fwd(
        &self,
        s_w: &candle_core::CudaStorage,
        l_w: &Layout,
        s_wt: &candle_core::CudaStorage,
        l_wt: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, Map2, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let kernel_size = self.kernel_size;
        let channels = self.channels;
        let batch = l_w.shape().dims()[0];

        struct S {
            kernel_size: i32,
            channels: i32,
            numel: usize,
        }
        impl Map2 for S {
            fn f<T: DeviceRepr + WithDType>(
                &self,
                window: &CudaSlice<T>,
                l_w: &Layout,
                weight: &CudaSlice<T>,
                l_wt: &Layout,
                dev: &CudaDevice,
            ) -> Result<CudaSlice<T>> {
                let window = match l_w.contiguous_offsets() {
                    Some((o1, o2)) => window.slice(o1..o2),
                    None => candle_core::bail!("conv1d_silu: window must be contiguous"),
                };
                let weight = match l_wt.contiguous_offsets() {
                    Some((o1, o2)) => weight.slice(o1..o2),
                    None => candle_core::bail!("conv1d_silu: weight must be contiguous"),
                };
                let cfg = LaunchConfig::for_num_elems(self.numel as u32);
                let func = dev.get_or_load_custom_func(
                    &kernel_name::<T>("depthwise_conv1d_silu"),
                    "cake_fused_ops",
                    FUSED_OPS_PTX,
                )?;
                let out = unsafe { dev.alloc::<T>(self.numel)? };
                let mut builder = func.builder();
                builder.arg(&self.numel);
                builder.arg(&window);
                builder.arg(&weight);
                builder.arg(&out);
                candle_core::builder_arg!(builder, self.kernel_size, self.channels);
                unsafe { builder.launch(cfg) }.w()?;
                Ok(out)
            }
        }

        use candle_core::backend::BackendStorage;
        let dev = s_w.device();
        let numel = batch * channels;
        let slice = S {
            kernel_size: kernel_size as i32,
            channels: channels as i32,
            numel,
        }
        .map(&s_w.slice, l_w, &s_wt.slice, l_wt, dev)?;
        let out_shape = Shape::from_dims(&[batch, channels]);
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            out_shape,
        ))
    }
}

// ─── DepthwiseConv1dBias: full depthwise conv1d + bias (no activation) ──

pub(super) struct DepthwiseConv1dBias {
    pub kernel_size: usize,
    pub channels: usize,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for DepthwiseConv1dBias {
    fn name(&self) -> &'static str {
        "depthwise_conv1d_bias"
    }

    fn cpu_fwd(
        &self,
        s_in: &CpuStorage,
        l_in: &Layout,
        s_wt: &CpuStorage,
        l_wt: &Layout,
        s_bi: &CpuStorage,
        l_bi: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float>(
            input: &[T],
            l_in: &Layout,
            weight: &[T],
            l_wt: &Layout,
            bias: &[T],
            l_bi: &Layout,
            kernel_size: usize,
            channels: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => &input[o1..o2],
                None => candle_core::bail!("conv1d_bias: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => &weight[o1..o2],
                None => candle_core::bail!("conv1d_bias: weight must be contiguous"),
            };
            let bias = match l_bi.contiguous_offsets() {
                Some((o1, o2)) => &bias[o1..o2],
                None => candle_core::bail!("conv1d_bias: bias must be contiguous"),
            };
            let dims = l_in.shape().dims();
            let batch = dims[0];
            let input_len = dims[2];
            let out_len = input_len - kernel_size + 1;
            let numel = batch * channels * out_len;
            let mut dst = vec![T::zero(); numel];
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..out_len {
                        let mut acc = T::zero();
                        let in_off = (b * channels + c) * input_len + t;
                        let wt_off = c * kernel_size;
                        for k in 0..kernel_size {
                            acc += input[in_off + k] * weight[wt_off + k];
                        }
                        acc += bias[c];
                        dst[(b * channels + c) * out_len + t] = acc;
                    }
                }
            }
            let out_shape = Shape::from_dims(&[batch, channels, out_len]);
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, out_shape))
        }

        use CpuStorage as C;
        match (s_in, s_wt, s_bi) {
            (C::BF16(i), C::BF16(w), C::BF16(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            (C::F16(i), C::F16(w), C::F16(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            (C::F32(i), C::F32(w), C::F32(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            (C::F64(i), C::F64(w), C::F64(b)) => {
                inner(i, l_in, w, l_wt, b, l_bi, self.kernel_size, self.channels)
            }
            _ => candle_core::bail!("conv1d_bias: unsupported dtype {:?}", s_in.dtype()),
        }
    }

    fn cuda_fwd(
        &self,
        s_in: &candle_core::CudaStorage,
        l_in: &Layout,
        s_wt: &candle_core::CudaStorage,
        l_wt: &Layout,
        s_bi: &candle_core::CudaStorage,
        l_bi: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let kernel_size = self.kernel_size;
        let channels = self.channels;
        let dims = l_in.shape().dims();
        let batch = dims[0];
        let input_len = dims[2];
        let out_len = input_len - kernel_size + 1;
        let numel = batch * channels * out_len;

        fn launch<T: DeviceRepr + WithDType>(
            input: &CudaSlice<T>,
            l_in: &Layout,
            weight: &CudaSlice<T>,
            l_wt: &Layout,
            bias: &CudaSlice<T>,
            l_bi: &Layout,
            dev: &CudaDevice,
            kernel_size: i32,
            channels: i32,
            input_len: i32,
            numel: usize,
        ) -> Result<CudaSlice<T>> {
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => input.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => weight.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias: weight must be contiguous"),
            };
            let bias = match l_bi.contiguous_offsets() {
                Some((o1, o2)) => bias.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias: bias must be contiguous"),
            };
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("depthwise_conv1d_bias"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(numel)? };
            let mut builder = func.builder();
            builder.arg(&numel);
            builder.arg(&input);
            builder.arg(&weight);
            builder.arg(&bias);
            builder.arg(&out);
            candle_core::builder_arg!(builder, kernel_size, channels, input_len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_in.device();
        let ks = kernel_size as i32;
        let ch = channels as i32;
        let il = input_len as i32;

        let slice = match (&s_in.slice, &s_wt.slice, &s_bi.slice) {
            (S::BF16(i), S::BF16(w), S::BF16(b)) => {
                S::BF16(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            (S::F16(i), S::F16(w), S::F16(b)) => {
                S::F16(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            (S::F32(i), S::F32(w), S::F32(b)) => {
                S::F32(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            (S::F64(i), S::F64(w), S::F64(b)) => {
                S::F64(launch(i, l_in, w, l_wt, b, l_bi, dev, ks, ch, il, numel)?)
            }
            _ => candle_core::bail!("conv1d_bias: unsupported dtype"),
        };

        let out_shape = Shape::from_dims(&[batch, channels, out_len]);
        Ok((
            candle_core::CudaStorage {
                slice,
                device: dev.clone(),
            },
            out_shape,
        ))
    }
}

// ─── DepthwiseConv1dBiasCtx: conv with separate context + input ─────

pub(super) struct DepthwiseConv1dBiasCtx {
    pub kernel_size: usize,
    pub channels: usize,
    pub bias: Tensor,
}

#[allow(clippy::too_many_arguments)]
impl candle_core::CustomOp3 for DepthwiseConv1dBiasCtx {
    fn name(&self) -> &'static str {
        "depthwise_conv1d_bias_ctx"
    }

    fn cpu_fwd(
        &self,
        s_ctx: &CpuStorage,
        l_ctx: &Layout,
        s_in: &CpuStorage,
        l_in: &Layout,
        s_wt: &CpuStorage,
        l_wt: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle_core::WithDType + num_traits::Float + num_traits::AsPrimitive<f32> + num_traits::FromPrimitive>(
            ctx: &[T], l_ctx: &Layout,
            input: &[T], l_in: &Layout,
            weight: &[T], l_wt: &Layout,
            bias_t: &Tensor,
            kernel_size: usize, channels: usize,
        ) -> Result<(CpuStorage, Shape)> {
            let ctx = match l_ctx.contiguous_offsets() {
                Some((o1, o2)) => &ctx[o1..o2],
                None => candle_core::bail!("conv1d_bias_ctx: ctx must be contiguous"),
            };
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => &input[o1..o2],
                None => candle_core::bail!("conv1d_bias_ctx: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => &weight[o1..o2],
                None => candle_core::bail!("conv1d_bias_ctx: weight must be contiguous"),
            };
            let bias_v: Vec<f32> = bias_t.to_dtype(candle_core::DType::F32)?.flatten_all()?.to_vec1()?;
            let dims = l_in.shape().dims();
            let (batch, time_len) = (dims[0], dims[2]);
            let ctx_len = kernel_size - 1;
            let numel = batch * channels * time_len;
            let mut dst = vec![T::zero(); numel];
            for b in 0..batch {
                for c in 0..channels {
                    for t in 0..time_len {
                        let mut acc = 0f32;
                        for k in 0..kernel_size {
                            let pos = t + k;
                            let v: f32 = if pos < ctx_len {
                                num_traits::AsPrimitive::as_(ctx[(b * channels + c) * ctx_len + pos])
                            } else {
                                num_traits::AsPrimitive::as_(input[(b * channels + c) * time_len + (pos - ctx_len)])
                            };
                            let w: f32 = num_traits::AsPrimitive::as_(weight[c * kernel_size + k]);
                            acc += v * w;
                        }
                        acc += bias_v[c];
                        dst[(b * channels + c) * time_len + t] =
                            num_traits::FromPrimitive::from_f32(acc).unwrap_or(T::zero());
                    }
                }
            }
            let out_shape = Shape::from_dims(&[batch, channels, time_len]);
            let storage = candle_core::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, out_shape))
        }

        use CpuStorage as C;
        match (s_ctx, s_in, s_wt) {
            (C::BF16(c), C::BF16(i), C::BF16(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            (C::F16(c), C::F16(i), C::F16(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            (C::F32(c), C::F32(i), C::F32(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            (C::F64(c), C::F64(i), C::F64(w)) => inner(c, l_ctx, i, l_in, w, l_wt, &self.bias, self.kernel_size, self.channels),
            _ => candle_core::bail!("conv1d_bias_ctx: unsupported dtype"),
        }
    }

    fn cuda_fwd(
        &self,
        s_ctx: &candle_core::CudaStorage,
        l_ctx: &Layout,
        s_in: &candle_core::CudaStorage,
        l_in: &Layout,
        s_wt: &candle_core::CudaStorage,
        l_wt: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{
            CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg,
        };
        use candle_core::cuda_backend::{kernel_name, WrapErr};
        use candle_core::{CudaDevice, WithDType};

        let kernel_size = self.kernel_size;
        let channels = self.channels;
        let dims = l_in.shape().dims();
        let (batch, time_len) = (dims[0], dims[2]);
        let ctx_len = kernel_size - 1;
        let numel = batch * channels * time_len;

        // Get bias CUDA storage
        let bias_cuda = self.bias.storage_and_layout();
        let (bias_storage, bias_layout) = (&*bias_cuda.0, &bias_cuda.1);

        fn launch<T: DeviceRepr + WithDType>(
            ctx: &CudaSlice<T>, l_ctx: &Layout,
            input: &CudaSlice<T>, l_in: &Layout,
            weight: &CudaSlice<T>, l_wt: &Layout,
            bias: &CudaSlice<T>, l_bi: &Layout,
            dev: &CudaDevice,
            kernel_size: i32, channels: i32, ctx_len: i32, time_len: i32,
            numel: usize,
        ) -> Result<CudaSlice<T>> {
            let ctx = match l_ctx.contiguous_offsets() {
                Some((o1, o2)) => ctx.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: ctx must be contiguous"),
            };
            let input = match l_in.contiguous_offsets() {
                Some((o1, o2)) => input.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: input must be contiguous"),
            };
            let weight = match l_wt.contiguous_offsets() {
                Some((o1, o2)) => weight.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: weight must be contiguous"),
            };
            let bias = match l_bi.contiguous_offsets() {
                Some((o1, o2)) => bias.slice(o1..o2),
                None => candle_core::bail!("conv1d_bias_ctx: bias must be contiguous"),
            };
            let cfg = LaunchConfig::for_num_elems(numel as u32);
            let func = dev.get_or_load_custom_func(
                &kernel_name::<T>("depthwise_conv1d_bias_ctx"),
                "cake_fused_ops",
                FUSED_OPS_PTX,
            )?;
            let out = unsafe { dev.alloc::<T>(numel)? };
            let mut builder = func.builder();
            builder.arg(&numel);
            builder.arg(&ctx);
            builder.arg(&input);
            builder.arg(&weight);
            builder.arg(&bias);
            builder.arg(&out);
            candle_core::builder_arg!(builder, kernel_size, channels, ctx_len, time_len);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(out)
        }

        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::CudaStorageSlice as S;
        let dev = s_in.device();
        let ks = kernel_size as i32;
        let ch = channels as i32;
        let cl = ctx_len as i32;
        let tl = time_len as i32;

        let bias_s = match bias_storage {
            candle_core::Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("conv1d_bias_ctx: bias must be on CUDA"),
        };

        let slice = match (&s_ctx.slice, &s_in.slice, &s_wt.slice, &bias_s.slice) {
            (S::BF16(c), S::BF16(i), S::BF16(w), S::BF16(b)) => S::BF16(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            (S::F16(c), S::F16(i), S::F16(w), S::F16(b)) => S::F16(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            (S::F32(c), S::F32(i), S::F32(w), S::F32(b)) => S::F32(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            (S::F64(c), S::F64(i), S::F64(w), S::F64(b)) => S::F64(launch(c, l_ctx, i, l_in, w, l_wt, b, bias_layout, dev, ks, ch, cl, tl, numel)?),
            _ => candle_core::bail!("conv1d_bias_ctx: unsupported dtype"),
        };

        let out_shape = Shape::from_dims(&[batch, channels, time_len]);
        Ok((
            candle_core::CudaStorage { slice, device: dev.clone() },
            out_shape,
        ))
    }
}
