//! Fused CUDA/CPU kernels for inference-critical operations.
//!
//! Each fused op replaces 2-5 separate kernel launches with a single launch,
//! saving ~4-7µs per launch on modern GPUs.
//!
//! Fused operations:
//! - `silu_mul(gate, up)` → silu(gate) * up (MLP activation, saves 1 launch)
//! - `stable_softplus(x)` → ln(1+exp(clamp(x))) clamped (GDN gates, saves 4 launches)
//! - `rms_norm_gated(x, z, weight, eps)` → rms_norm(x,w) * silu(z) (GDN norm, saves 2 launches)

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

// ─── PTX module (compiled from src/cuda/fused_ops.cu) ───────────────
#[cfg(feature = "cuda")]
mod ptx {
    include!(concat!(env!("OUT_DIR"), "/fused_ops_ptx.rs"));
}

#[cfg(feature = "cuda")]
const FUSED_OPS_PTX: &str = ptx::FUSED_OPS;

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

/// Fused silu(gate) * up — replaces 2 kernel launches with 1.
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

/// Fused stable softplus — replaces 5 kernel launches with 1.
pub fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    x.apply_op1_no_bwd(&StableSoftplus)
}

// ─── rms_norm_gated: rms_norm(x, weight) * silu(z) ─────────────────

struct RmsNormGated {
    eps: f32,
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

    #[cfg(feature = "cuda")]
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

/// Fused rms_norm(x, weight) * silu(z) — replaces 3 kernel launches with 1.
pub fn rms_norm_gated(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    x.apply_op3_no_bwd(z, weight, &RmsNormGated { eps })
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

/// Fused (a - b) * c — replaces 2 kernel launches (sub + mul) with 1.
pub fn sub_mul(a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
    a.apply_op3_no_bwd(b, c, &SubMul)
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
    fn test_rms_norm_gated_correctness() {
        // Small test: 2 rows, 4 cols
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [0.5, -1.0, 0.5, -1.0]], &Device::Cpu)
            .unwrap();
        let z = Tensor::new(
            &[[0.1f32, 0.2, 0.3, 0.4], [-0.5, 1.0, -0.5, 1.0]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap();
        let eps = 1e-6f32;

        let fused: Vec<f32> = rms_norm_gated(&x, &z, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference: rms_norm(x, weight, eps) * silu(z)
        let x_normed: Vec<f32> = candle_nn::ops::rms_norm(&x, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let silu_z: Vec<f32> = candle_nn::ops::silu(&z)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let reference: Vec<f32> = x_normed.iter().zip(&silu_z).map(|(a, b)| a * b).collect();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "rms_norm_gated mismatch:\nfused={fused:?}\nref  ={reference:?}"
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
    fn test_rms_norm_gated_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [0.5, -1.0, 0.5, -1.0]], &dev).unwrap();
        let z = Tensor::new(
            &[[0.1f32, 0.2, 0.3, 0.4], [-0.5, 1.0, -0.5, 1.0]],
            &dev,
        )
        .unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &dev).unwrap();
        let eps = 1e-6f32;

        let fused: Vec<f32> = rms_norm_gated(&x, &z, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Reference on CPU
        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let z_cpu = z.to_device(&Device::Cpu).unwrap();
        let w_cpu = weight.to_device(&Device::Cpu).unwrap();
        let x_normed: Vec<f32> = candle_nn::ops::rms_norm(&x_cpu, &w_cpu, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let silu_z: Vec<f32> = candle_nn::ops::silu(&z_cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let reference: Vec<f32> = x_normed.iter().zip(&silu_z).map(|(a, b)| a * b).collect();

        assert!(
            approx_eq(&fused, &reference, 1e-5),
            "CUDA rms_norm_gated mismatch:\nfused={fused:?}\nref  ={reference:?}"
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

    // ── exp_mul tests ────────────────────────────────────────────────

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

    // ── sub_mul tests ────────────────────────────────────────────────

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
}
