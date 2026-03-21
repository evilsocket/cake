//! Fused normalization kernels: rms_norm_gated, add_rms_norm, rms_norm_channel, adaln_modulate.

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

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
                super::FUSED_OPS_PTX,
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

/// Fused rms_norm(x, weight) * silu(z) — replaces 3 kernel launches with 1 on CUDA.
/// On Metal, uses candle's built-in ops (rms_norm + silu + mul).
pub fn rms_norm_gated(x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if x.device().is_metal() {
        let normed = candle_nn::ops::rms_norm(x, weight, eps)?;
        let gate = candle_nn::ops::silu(z)?;
        return normed.mul(&gate);
    }
    x.apply_op3_no_bwd(z, weight, &RmsNormGated { eps })
}

// ─── add_rms_norm: rms_norm(a + b, weight, eps) with residual ────────

struct AddRmsNorm {
    eps: f32,
    n_cols: usize,
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

    #[cfg(feature = "cuda")]
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
                super::FUSED_OPS_PTX,
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

/// Fused residual add + RmsNorm — replaces 2 kernel launches with 1.
/// Returns (residual_sum, rms_normed) as contiguous tensors.
/// `a` and `b` are added, then the sum is RMS-normalized with `weight` and `eps`.
pub fn add_rms_norm(
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let n_cols = *a.dims().last().unwrap();
    let combined = a.apply_op3_no_bwd(b, weight, &AddRmsNorm { eps, n_cols })?;
    let residual = combined.narrow(candle_core::D::Minus1, 0, n_cols)?.contiguous()?;
    let normed = combined.narrow(candle_core::D::Minus1, n_cols, n_cols)?.contiguous()?;
    Ok((residual, normed))
}

/// Fused residual add + RmsNorm — returns ONLY the normed result.
/// Use when caller will reconstruct residual via add3 at the end of the block.
pub fn add_rms_norm_normed(
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    // Fuse add+norm into 1 kernel, return only the normed half
    let n_cols = *a.dims().last().unwrap();
    let combined = a.apply_op3_no_bwd(b, weight, &AddRmsNorm { eps, n_cols })?;
    combined.narrow(candle_core::D::Minus1, n_cols, n_cols)?.contiguous()
}

// ─── rms_norm_channel: RMS-normalize over channel dim of (b,c,t) ────

struct RmsNormChannel {
    eps: f32,
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

    #[cfg(feature = "cuda")]
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
                super::FUSED_OPS_PTX,
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

/// RMS-normalize over the channel dimension of a (batch, channels, time) tensor.
/// Replaces transpose + rms_norm + transpose (3 ops including copy) with 1 kernel.
pub fn rms_norm_channel(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    let w = weight.contiguous()?;
    x.apply_op2_no_bwd(&w, &RmsNormChannel { eps })
}

// ─── adaln_modulate: rms_norm(x,w,eps) * (1+scale) + shift ──────────

struct AdaLnModulate {
    eps: f32,
    shift: Tensor,
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

    #[cfg(feature = "cuda")]
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
            let func = dev.get_or_load_custom_func(&kernel_name::<T>("adaln_modulate"), "cake_fused_ops", super::FUSED_OPS_PTX)?;
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

/// Fused AdaLN modulation: rms_norm(x, weight, eps) * (1 + scale) + shift.
/// Replaces 4 kernel launches (rms_norm + add_1 + mul + add_shift) with 1.
pub fn adaln_modulate(
    x: &Tensor,
    norm_weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let x = x.contiguous()?;
    let w = norm_weight.contiguous()?;
    let sc = scale.contiguous()?;
    let sh = shift.contiguous()?;
    x.apply_op3_no_bwd(
        &w,
        &sc,
        &AdaLnModulate { eps, shift: sh },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
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

    // ── add_rms_norm shape and residual tests ──────────────────────

    #[test]
    fn test_add_rms_norm_shape_preservation_2d() {
        let a = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[0.1f32, 0.2], [0.3, 0.4]], &Device::Cpu).unwrap();
        let w = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
        let (residual, normed) = add_rms_norm(&a, &b, &w, 1e-6).unwrap();
        assert_eq!(residual.dims(), &[2, 2]);
        assert_eq!(normed.dims(), &[2, 2]);
    }

    #[test]
    fn test_add_rms_norm_residual_is_sum() {
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[10.0f32, 20.0, 30.0]], &Device::Cpu).unwrap();
        let w = Tensor::ones(3, DType::F32, &Device::Cpu).unwrap();
        let (residual, _) = add_rms_norm(&a, &b, &w, 1e-6).unwrap();
        let vals: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
        assert!(approx_eq(&vals, &[11.0, 22.0, 33.0], 1e-5));
    }

    #[test]
    fn test_add_rms_norm_correctness() {
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0]], &Device::Cpu).unwrap();
        let b = Tensor::new(&[[0.5f32, -1.0, 0.5, -1.0]], &Device::Cpu).unwrap();
        let weight = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0], &Device::Cpu).unwrap();
        let eps = 1e-6f32;

        let (residual, normed) = add_rms_norm(&a, &b, &weight, eps).unwrap();

        // Check residual = a + b
        let expected_sum: Vec<f32> = (&a + &b).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let actual_sum: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            approx_eq(&actual_sum, &expected_sum, 1e-5),
            "residual mismatch: got={actual_sum:?} expected={expected_sum:?}"
        );

        // Check normed = rms_norm(a + b, weight, eps)
        let sum_tensor = (&a + &b).unwrap();
        let expected_norm: Vec<f32> = candle_nn::ops::rms_norm(&sum_tensor, &weight, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let actual_norm: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();
        assert!(
            approx_eq(&actual_norm, &expected_norm, 1e-5),
            "normed mismatch: got={actual_norm:?} expected={expected_norm:?}"
        );
    }

    // ── rms_norm_channel tests ───────────────────────────────────

    #[test]
    fn test_rms_norm_channel_cpu() {
        // x: (1, 4, 3) — batch=1, channels=4, time=3
        let x = Tensor::new(
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &Device::Cpu,
        )
        .unwrap()
        .reshape((1, 4, 3))
        .unwrap();
        let w = Tensor::ones(4, DType::F32, &Device::Cpu).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        let out_v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(!out_v.iter().any(|v| v.is_nan()), "NaN in rms_norm_channel: {out_v:?}");
        assert_eq!(out.dims(), &[1, 4, 3]);
        // For time=0: values [1,4,7,10], rms = sqrt((1+16+49+100)/4) = sqrt(41.5) ≈ 6.44
        // normalized: [1/6.44, 4/6.44, 7/6.44, 10/6.44] ≈ [0.155, 0.621, 1.087, 1.553]
        assert!((out_v[0] - 0.1553).abs() < 0.01, "wrong norm val: {}", out_v[0]);
    }

    #[test]
    fn test_rms_norm_channel_with_nonunit_weight() {
        // (1, 2, 3) with weight=[2, 0.5]
        let x = Tensor::new(&[1f32, 1., 1., 2., 2., 2.], &Device::Cpu)
            .unwrap()
            .reshape((1, 2, 3))
            .unwrap();
        let w = Tensor::new(&[2.0f32, 0.5], &Device::Cpu).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        assert_eq!(out.dims(), &[1, 2, 3]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()), "all values should be finite");
        assert!(vals.iter().any(|v| *v != 0.0), "output should be non-zero");
    }

    // ── CUDA tests ───────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
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
    fn test_add_rms_norm_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let a = Tensor::new(&[[1.0f32, 2.0, 3.0, 4.0], [0.5, -1.0, 0.5, -1.0]], &dev).unwrap();
        let b = Tensor::new(&[[0.1f32, 0.2, 0.3, 0.4], [-0.5, 1.0, -0.5, 1.0]], &dev).unwrap();
        let weight = Tensor::new(&[1.0f32, 0.5, 2.0, 1.5], &dev).unwrap();
        let eps = 1e-6f32;

        let (residual, normed) = add_rms_norm(&a, &b, &weight, eps).unwrap();
        let res_vals: Vec<f32> = residual.flatten_all().unwrap().to_vec1().unwrap();
        let norm_vals: Vec<f32> = normed.flatten_all().unwrap().to_vec1().unwrap();

        // CPU reference
        let a_cpu = a.to_device(&Device::Cpu).unwrap();
        let b_cpu = b.to_device(&Device::Cpu).unwrap();
        let w_cpu = weight.to_device(&Device::Cpu).unwrap();
        let sum_cpu = (&a_cpu + &b_cpu).unwrap();
        let expected_sum: Vec<f32> = sum_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let expected_norm: Vec<f32> = candle_nn::ops::rms_norm(&sum_cpu, &w_cpu, eps)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(approx_eq(&res_vals, &expected_sum, 1e-5), "CUDA residual mismatch");
        assert!(approx_eq(&norm_vals, &expected_norm, 1e-4), "CUDA normed mismatch");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_bf16_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Larger test: (1, 32, 100) in BF16
        let data: Vec<f32> = (0..3200).map(|i| (i as f32 * 0.01) - 16.0).collect();
        let x = Tensor::new(data.as_slice(), &dev)
            .unwrap()
            .reshape((1, 32, 100))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::ones(32, DType::BF16, &dev).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        let out_v: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let nan_count = out_v.iter().filter(|v| v.is_nan()).count();
        assert!(
            nan_count == 0,
            "NaN in bf16 rms_norm_channel: {nan_count}/{} values",
            out_v.len()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_large_channels_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Large channels like encoder stage 6: (1, 2048, 1) in BF16
        let n = 2048;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01) - 10.0).collect();
        let x = Tensor::new(data.as_slice(), &dev)
            .unwrap()
            .reshape((1, 2048, 1))
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();
        let w = Tensor::ones(2048, DType::BF16, &dev).unwrap();
        let out = rms_norm_channel(&x, &w, 1e-5).unwrap();
        let out_v: Vec<f32> = out
            .to_dtype(DType::F32)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let nan_count = out_v.iter().filter(|v| v.is_nan()).count();
        assert!(
            nan_count == 0,
            "NaN in 2048-channel rms_norm_channel: {nan_count}/{n} values"
        );
        // Also check correctness vs CPU
        let x_cpu = x.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
        let w_cpu = w.to_dtype(DType::F32).unwrap().to_device(&Device::Cpu).unwrap();
        let ref_v: Vec<f32> = rms_norm_channel(&x_cpu, &w_cpu, 1e-5)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        // BF16 precision: ~0.4% relative error, use wider tolerance
        assert!(
            approx_eq(&out_v, &ref_v, 0.5),
            "2048-channel CUDA/CPU mismatch: cuda[0..5]={:?} cpu[0..5]={:?}",
            &out_v[..5],
            &ref_v[..5]
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_vs_candle_cuda() {
        use candle_core::Module;
        // Compare our rms_norm_channel against candle's transpose+rms_norm+transpose
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        // Test multiple channel counts that the VAE encoder uses
        for (channels, time_len) in [(32, 1000), (64, 500), (128, 200), (2048, 10)] {
            let n = channels * time_len;
            let data: Vec<f32> = (0..n)
                .map(|i| ((i as f32 * 7.13) % 5.0) - 2.5)
                .collect();
            let x = Tensor::new(data.as_slice(), &dev)
                .unwrap()
                .reshape((1, channels, time_len))
                .unwrap()
                .to_dtype(DType::BF16)
                .unwrap();
            let w_data: Vec<f32> = (0..channels)
                .map(|i| 0.5 + (i as f32) * 0.01)
                .collect();
            let w = Tensor::new(w_data.as_slice(), &dev)
            .unwrap()
            .to_dtype(DType::BF16)
            .unwrap();

            // Our kernel
            let ours = rms_norm_channel(&x, &w, 1e-5).unwrap();
            let ours_v: Vec<f32> = ours
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            // candle's approach: transpose -> rms_norm -> transpose
            let norm = candle_nn::RmsNorm::new(w.clone(), 1e-5_f64);
            let ref_out = norm
                .forward(&x.transpose(1, 2).unwrap())
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let ref_v: Vec<f32> = ref_out
                .to_dtype(DType::F32)
                .unwrap()
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();

            let nan_ours = ours_v.iter().filter(|v| v.is_nan()).count();
            let nan_ref = ref_v.iter().filter(|v| v.is_nan()).count();
            assert!(
                nan_ours == 0,
                "ch={channels} t={time_len}: {nan_ours} NaN in rms_norm_channel"
            );
            assert!(
                nan_ref == 0,
                "ch={channels} t={time_len}: {nan_ref} NaN in candle rms_norm"
            );

            // Check max absolute difference
            let max_diff = ours_v
                .iter()
                .zip(ref_v.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(
                max_diff < 0.5,
                "ch={channels} t={time_len}: max_diff={max_diff} (first 5: ours={:?} ref={:?})",
                &ours_v[..5],
                &ref_v[..5]
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_rms_norm_channel_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let x = Tensor::new(
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            &dev,
        )
        .unwrap()
        .reshape((1, 4, 3))
        .unwrap();
        let w = Tensor::ones(4, DType::F32, &dev).unwrap();
        let out_cuda: Vec<f32> = rms_norm_channel(&x, &w, 1e-5)
            .unwrap()
            .to_device(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // Compare with CPU
        let x_cpu = x.to_device(&Device::Cpu).unwrap();
        let w_cpu = w.to_device(&Device::Cpu).unwrap();
        let out_cpu: Vec<f32> = rms_norm_channel(&x_cpu, &w_cpu, 1e-5)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&out_cuda, &out_cpu, 1e-3),
            "CUDA rms_norm_channel mismatch:\n  cuda={out_cuda:?}\n  cpu ={out_cpu:?}"
        );
    }
}
