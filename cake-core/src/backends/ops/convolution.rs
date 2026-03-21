//! Fused depthwise convolution kernels: depthwise_conv1d_silu, depthwise_conv1d_bias, depthwise_conv1d_bias_ctx.

use candle_core::{backend::BackendStorage as _, CpuStorage, Layout, Result, Shape, Tensor};

// ─── depthwise_conv1d_silu: dot(window, weight) per channel + silu ──

struct DepthwiseConv1dSilu {
    kernel_size: usize,
    channels: usize,
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

    #[cfg(feature = "cuda")]
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
                    super::FUSED_OPS_PTX,
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

/// Fused depthwise conv1d + SiLU on a single token window.
/// On CUDA: single fused kernel. On Metal: candle built-in ops (3 dispatches).
pub fn depthwise_conv1d_silu(
    window: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
    channels: usize,
) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if window.device().is_metal() {
        let w = weight.unsqueeze(0)?;
        let dot = window.broadcast_mul(&w)?.sum(candle_core::D::Minus1)?;
        return candle_nn::ops::silu(&dot);
    }
    window.apply_op2_no_bwd(
        weight,
        &DepthwiseConv1dSilu {
            kernel_size,
            channels,
        },
    )
}

// ─── depthwise_conv1d_bias: full depthwise conv1d + bias (no activation) ──

struct DepthwiseConv1dBias {
    kernel_size: usize,
    channels: usize,
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

    #[cfg(feature = "cuda")]
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
                super::FUSED_OPS_PTX,
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

/// Fused depthwise conv1d + bias — replaces 14 kernel launches with 1.
/// padded_input: (batch, channels, input_len) — already causal-padded
/// weight: (channels, kernel_size), bias: (channels,)
/// Returns: (batch, channels, out_len) where out_len = input_len - kernel_size + 1
pub fn depthwise_conv1d_bias(
    padded_input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    kernel_size: usize,
    channels: usize,
) -> Result<Tensor> {
    // Ensure all inputs are contiguous (weight may come from squeeze)
    let inp = padded_input.contiguous()?;
    let wt = weight.contiguous()?;
    let bi = bias.contiguous()?;
    inp.apply_op3_no_bwd(
        &wt,
        &bi,
        &DepthwiseConv1dBias {
            kernel_size,
            channels,
        },
    )
}

// ─── depthwise_conv1d_bias_ctx: conv with separate context + input ──

struct DepthwiseConv1dBiasCtx {
    kernel_size: usize,
    channels: usize,
    bias: Tensor,
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

    #[cfg(feature = "cuda")]
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
                super::FUSED_OPS_PTX,
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

/// Fused depthwise conv1d with separate context + input (no cat needed).
/// Replaces Tensor::zeros + Tensor::cat + depthwise_conv1d_bias (3 kernels -> 1).
/// ctx: (batch, channels, kernel_size-1), input: (batch, channels, time_len)
/// weight: (channels, kernel_size), bias: (channels,)
/// Returns: (batch, channels, time_len)
pub fn depthwise_conv1d_bias_ctx(
    ctx: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    kernel_size: usize,
    channels: usize,
) -> Result<Tensor> {
    let ctx = ctx.contiguous()?;
    let inp = input.contiguous()?;
    let wt = weight.contiguous()?;
    ctx.apply_op3_no_bwd(
        &inp,
        &wt,
        &DepthwiseConv1dBiasCtx {
            kernel_size,
            channels,
            bias: bias.contiguous()?,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── depthwise_conv1d_silu tests ──────────────────────────────────

    #[test]
    fn test_depthwise_conv1d_silu_correctness() {
        // batch=1, channels=4, kernel_size=3
        let window = Tensor::new(
            &[[[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.1, 0.0, 0.1], [1.0, -1.0, 0.5]]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[1.0f32, 0.5, 0.25], [0.1, 0.2, 0.3], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]],
            &Device::Cpu,
        )
        .unwrap();

        let fused = depthwise_conv1d_silu(&window, &weight, 3, 4).unwrap();
        assert_eq!(fused.dims(), &[1, 4]);

        // Reference: broadcast_mul + sum + silu
        let ref_y = window
            .broadcast_mul(&weight.unsqueeze(0).unwrap())
            .unwrap()
            .sum(candle_core::D::Minus1)
            .unwrap();
        let ref_y: Vec<f32> = candle_nn::ops::silu(&ref_y)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let fused_vals: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();

        assert!(
            approx_eq(&fused_vals, &ref_y, 1e-5),
            "conv1d_silu mismatch: fused={fused_vals:?} ref={ref_y:?}"
        );
    }

    #[test]
    fn test_depthwise_conv1d_silu_batch2() {
        // batch=2, channels=2, kernel_size=2
        let window = Tensor::new(
            &[
                [[1.0f32, 2.0], [3.0, 4.0]],
                [[0.5, -0.5], [-1.0, 1.0]],
            ],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(&[[1.0f32, 1.0], [0.5, 0.5]], &Device::Cpu).unwrap();
        let result = depthwise_conv1d_silu(&window, &weight, 2, 2).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    // ── depthwise_conv1d_bias tests ─────────────────────────────────

    #[test]
    fn test_depthwise_conv1d_bias_cpu() {
        // batch=1, channels=2, input_len=4, kernel_size=3 -> out_len=2
        let input = Tensor::new(
            &[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
            &Device::Cpu,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[1.0f32, 0.0, 0.0], [0.0, 0.0, 1.0]],
            &Device::Cpu,
        )
        .unwrap();
        let bias = Tensor::new(&[10.0f32, 20.0], &Device::Cpu).unwrap();

        let out = depthwise_conv1d_bias(&input, &weight, &bias, 3, 2).unwrap();
        assert_eq!(out.dims(), &[1, 2, 2]);
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        // chan0, w=[1,0,0]: pos0=1*1+0+0+10=11, pos1=1*2+0+0+10=12
        // chan1, w=[0,0,1]: pos0=0+0+7*1+20=27, pos1=0+0+8*1+20=28
        assert!(approx_eq(&vals, &[11.0, 12.0, 27.0, 28.0], 1e-5));
    }

    #[test]
    fn test_depthwise_conv1d_bias_shape() {
        // batch=2, channels=3, input_len=6, kernel_size=2 -> out_len=5
        let input = Tensor::zeros((2, 3, 6), DType::F32, &Device::Cpu).unwrap();
        let weight = Tensor::ones((3, 2), DType::F32, &Device::Cpu).unwrap();
        let bias = Tensor::zeros(3, DType::F32, &Device::Cpu).unwrap();
        let out = depthwise_conv1d_bias(&input, &weight, &bias, 2, 3).unwrap();
        assert_eq!(out.dims(), &[2, 3, 5]);
    }

    // ── CUDA tests ───────────────────────────────────────────────────

    #[cfg(feature = "cuda")]
    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_depthwise_conv1d_silu_cuda() {
        let dev = match cuda_device() {
            Some(d) => d,
            None => return,
        };
        let window = Tensor::new(
            &[[[0.1f32, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [-0.1, 0.0, 0.1, 0.2]]],
            &dev,
        )
        .unwrap();
        let weight = Tensor::new(
            &[[1.0f32, 0.5, 0.25, 0.1], [0.1, 0.2, 0.3, 0.4], [1.0, 1.0, 1.0, 1.0]],
            &dev,
        )
        .unwrap();

        let fused_vals: Vec<f32> = depthwise_conv1d_silu(&window, &weight, 4, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        // CPU reference
        let window_cpu = window.to_device(&Device::Cpu).unwrap();
        let weight_cpu = weight.to_device(&Device::Cpu).unwrap();
        let ref_vals: Vec<f32> = depthwise_conv1d_silu(&window_cpu, &weight_cpu, 4, 3)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        assert!(
            approx_eq(&fused_vals, &ref_vals, 1e-5),
            "CUDA conv1d_silu mismatch: fused={fused_vals:?} ref={ref_vals:?}"
        );
    }
}
