//! CUDA CustomOp implementations for all 13 fused operations.
//!
//! This module is gated by `#[cfg(feature = "cuda")]` at the parent level,
//! so individual methods do NOT carry cfg guards.

use candle_core::{CpuStorage, Layout, Result, Shape, Tensor};

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/fused_ops_ptx.rs"));
}
pub(super) const FUSED_OPS_PTX: &str = ptx::OPS;

// ─── SiluMul: silu(gate) * up ──────────────────────────────────────

pub(super) struct SiluMul;

impl candle_core::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "silu_mul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
            candle_core::builder_arg!(builder, n_cols as i32, block_size as i32, eps, n_rows as i32);
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

        // Output layout: (2*n_rows, n_cols) — residual rows then normed rows
        let mut out_dims = l_a.shape().dims().to_vec();
        out_dims[0] *= 2;

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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
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

// ─── F8E4M3 dequantization ──────────────────────────────────────────

pub(super) struct F8E4M3ToF32;

impl candle_core::CustomOp1 for F8E4M3ToF32 {
    fn name(&self) -> &'static str { "f8e4m3_to_f32" }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
    }

    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
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
        let func = dev.get_or_load_custom_func("f8e4m3_to_f32", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<f32>(el)? };
        let mut builder = func.builder();
        builder.arg(&el); builder.arg(&src); builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage { slice: candle_core::cuda_backend::CudaStorageSlice::F32(out), device: dev.clone() }, l.shape().clone()))
    }
}

pub(super) struct F8E4M3ToF16;

impl candle_core::CustomOp1 for F8E4M3ToF16 {
    fn name(&self) -> &'static str { "f8e4m3_to_f16" }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
    }

    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
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
        let func = dev.get_or_load_custom_func("f8e4m3_to_f16", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<half::f16>(el)? };
        let mut builder = func.builder();
        builder.arg(&el); builder.arg(&src); builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage { slice: candle_core::cuda_backend::CudaStorageSlice::F16(out), device: dev.clone() }, l.shape().clone()))
    }
}

pub(super) struct F8E4M3ToBF16;

impl candle_core::CustomOp1 for F8E4M3ToBF16 {
    fn name(&self) -> &'static str { "f8e4m3_to_bf16" }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("{}: expected CUDA device", self.name())
    }

    fn cuda_fwd(&self, s: &candle_core::CudaStorage, l: &Layout) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
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
        let func = dev.get_or_load_custom_func("f8e4m3_to_bf16", "cake_fused_ops", FUSED_OPS_PTX)?;
        let out = unsafe { dev.alloc::<half::bf16>(el)? };
        let mut builder = func.builder();
        builder.arg(&el); builder.arg(&src); builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok((candle_core::CudaStorage { slice: candle_core::cuda_backend::CudaStorageSlice::BF16(out), device: dev.clone() }, l.shape().clone()))
    }
}
