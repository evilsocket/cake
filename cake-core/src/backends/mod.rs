//! Compute backend abstraction.
//!
//! The [`ComputeBackend`] trait defines all backend-specific operations (fused kernels,
//! attention, device control). Implementations exist for CPU (default), CUDA, Metal,
//! and Vulkan (via wgpu).
//!
//! Models call `ctx.backend().method()` instead of backend-specific code paths,
//! making it trivial to add new GPU backends.

use candle_core::{DType, Device, Result, Tensor};
use std::sync::Arc;

mod cpu;
pub use cpu::CpuBackend;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

#[cfg(feature = "metal")]
mod metal;
#[cfg(feature = "metal")]
pub use self::metal::MetalBackend;

#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "vulkan")]
pub use vulkan::VulkanBackend;

#[cfg(feature = "rocm")]
mod rocm;
#[cfg(feature = "rocm")]
pub use rocm::RocmBackend;

/// Abstraction over compute backends (CPU, CUDA, Metal, Vulkan).
///
/// Each method has a CPU-based default via [`CpuBackend`]. GPU backends override
/// specific methods for acceleration while falling back to CPU for unimplemented ops.
pub trait ComputeBackend: Send + Sync + std::fmt::Debug {
    /// Human-readable backend name for logging.
    fn name(&self) -> &str;

    /// The candle device this backend operates on.
    fn device(&self) -> &Device;

    // ── Attention ────────────────────────────────────────────────────

    /// Scaled dot-product attention.
    ///
    /// Backends may use flash-attn (CUDA), fused SDPA (Metal), wgpu matmul (Vulkan),
    /// or manual matmul + softmax (CPU).
    ///
    /// Input layout: `(batch, heads, seq_len, head_dim)`.
    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor>;

    /// Fused scaled dot-product attention (Metal SDPA, Flash-Attn style).
    /// Returns `softmax(Q @ K^T * scale + mask) @ V`.
    /// Default: delegates to `candle_nn::ops::sdpa`.
    fn sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        causal: bool,
        scale: f32,
    ) -> Result<Tensor> {
        candle_nn::ops::sdpa(q, k, v, mask, causal, scale, 1.0)
    }

    // ── Fused activations ────────────────────────────────────────────

    /// `silu(gate) * up` — MLP activation gate.
    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor>;

    /// `ln(1 + exp(clamp(x, -inf, 88)))` with `max(x, result)` — GDN gate.
    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor>;

    // ── Fused normalization ──────────────────────────────────────────

    /// `rms_norm(x, weight, eps) * silu(z)` — GDN output gating.
    fn rms_norm_gated(
        &self,
        x: &Tensor,
        z: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor>;

    /// `rms_norm(a + b, weight, eps)` — residual + norm fusion.
    /// Returns `(residual, normed)` where `residual = a + b`.
    fn add_rms_norm(
        &self,
        a: &Tensor,
        b: &Tensor,
        weight: &Tensor,
        eps: f32,
    ) -> Result<(Tensor, Tensor)>;

    /// Channel-wise RMS normalization for (batch, channels, time) layout.
    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    // ── Fused convolutions (GDN / VibeVoice) ──────────────────────────

    /// Depthwise conv1d + SiLU on a single token window.
    fn depthwise_conv1d_silu(
        &self,
        window: &Tensor,
        weight: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor>;

    /// Depthwise conv1d + bias (no activation).
    fn depthwise_conv1d_bias(
        &self,
        padded_input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor>;

    /// Depthwise conv1d + bias with separate context tensor.
    /// Virtually concatenates `[ctx, input]` without allocating the merged tensor.
    fn depthwise_conv1d_bias_ctx(
        &self,
        ctx: &Tensor,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        kernel_size: usize,
        channels: usize,
    ) -> Result<Tensor>;

    // ── Elementwise fusions ──────────────────────────────────────────

    /// `a + b + c` — three-way element-wise add.
    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor>;

    /// `x * exp(y)`.
    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor>;

    /// `(a - b) * c`.
    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor>;

    /// `a + b * c` — scaled addition.
    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor>;

    // ── Adaptive LayerNorm (DiT) ──────────────────────────────────────

    /// `rms_norm(x, norm_weight, eps) * (1 + scale) + shift` — AdaLN modulation.
    fn adaln_modulate(
        &self,
        x: &Tensor,
        norm_weight: &Tensor,
        scale: &Tensor,
        shift: &Tensor,
        eps: f32,
    ) -> Result<Tensor>;

    // ── F8 dequantization ────────────────────────────────────────────

    /// Dequantize F8E4M3 tensor to F32.
    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor>;

    /// Dequantize F8E4M3 tensor to F16.
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor>;

    /// Dequantize F8E4M3 tensor to BF16.
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor>;

    // ── Linear algebra ────────────────────────────────────────────────

    /// Matrix multiplication. GPU backends override to use accelerated matmul.
    /// Default: delegates to candle's CPU/CUDA/Metal matmul.
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        a.matmul(b)
    }

    /// Pre-process a linear weight for optimal matmul performance.
    /// Called once at model load time. Backends can pre-convert dtype,
    /// transpose, or upload to GPU. Returns the optimized weight.
    /// Default: returns the weight as-is.
    fn preprocess_linear_weight(&self, weight: &Tensor) -> Result<Tensor> {
        Ok(weight.clone())
    }

    // ── Inference primitives ──────────────────────────────────────────

    /// Linear layer forward: `x @ weight^T + bias`.
    ///
    /// Matches candle_nn::Linear::forward() semantics exactly:
    /// - For contiguous 3D/4D inputs: reshape to 2D → matmul → reshape back
    ///   (avoids slow broadcast_matmul on CUDA/CPU)
    /// - For non-contiguous 3D+: uses broadcast_left on weight
    /// - No dtype conversion (caller is responsible)
    fn linear_forward(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
    ) -> Result<Tensor> {
        let out = match x.dims() {
            [b1, b2, m, k] => {
                if x.is_contiguous() {
                    let w = weight.t()?;
                    x.reshape((b1 * b2 * m, *k))?
                        .matmul(&w)?
                        .reshape((*b1, *b2, *m, ()))?
                } else {
                    let w = weight.broadcast_left((*b1, *b2))?.t()?;
                    x.matmul(&w)?
                }
            }
            [bsize, m, k] => {
                if x.is_contiguous() {
                    let w = weight.t()?;
                    x.reshape((bsize * m, *k))?
                        .matmul(&w)?
                        .reshape((*bsize, *m, ()))?
                } else {
                    let w = weight.broadcast_left(*bsize)?.t()?;
                    x.matmul(&w)?
                }
            }
            _ => x.matmul(&weight.t()?)?,
        };
        match bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }

    /// RMS normalization: `x * weight / sqrt(mean(x^2) + eps)`.
    fn rms_norm(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, weight, eps)
    }

    /// Layer normalization: `(x - mean) / sqrt(var + eps) * weight + bias`.
    /// Matches candle_nn::LayerNorm::forward() — uses fused kernel when contiguous + has bias,
    /// otherwise falls back to manual F32 computation with dtype promotion for F16/BF16.
    fn layer_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        use candle_core::{DType, D};
        // Fast path: contiguous F32 — raw computation avoids tensor op overhead
        if x.dtype() == DType::F32 && x.is_contiguous() {
            let shape = x.dims();
            let hidden = *shape.last().unwrap_or(&0);
            let x_data = x.flatten_all()?.to_vec1::<f32>()?;
            let w_data = weight.to_vec1::<f32>()?;
            let b_data = match bias {
                Some(b) => Some(b.to_vec1::<f32>()?),
                None => None,
            };
            let rows = x_data.len() / hidden;
            let mut out = vec![0f32; x_data.len()];
            let eps64 = eps as f64;
            for r in 0..rows {
                let off = r * hidden;
                let row = &x_data[off..off + hidden];
                let mut sum = 0f64;
                let mut sum_sq = 0f64;
                for &v in row {
                    let v64 = v as f64;
                    sum += v64;
                    sum_sq += v64 * v64;
                }
                let mean = sum / hidden as f64;
                let var = sum_sq / hidden as f64 - mean * mean;
                let rstd = 1.0 / (var + eps64).sqrt();
                match &b_data {
                    Some(bd) => {
                        for i in 0..hidden {
                            out[off + i] =
                                (((row[i] as f64 - mean) * rstd) * w_data[i] as f64
                                    + bd[i] as f64) as f32;
                        }
                    }
                    None => {
                        for i in 0..hidden {
                            out[off + i] =
                                (((row[i] as f64 - mean) * rstd) * w_data[i] as f64) as f32;
                        }
                    }
                }
            }
            return Tensor::from_vec(out, shape, x.device());
        }
        // Fused kernel for contiguous non-F32 with bias
        if x.is_contiguous() {
            if let Some(b) = bias {
                return candle_nn::ops::layer_norm(x, weight, b, eps);
            }
        }
        // Generic fallback with F32 promotion
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(weight)?;
        match bias {
            Some(b) => x.broadcast_add(b),
            None => Ok(x),
        }
    }

    /// Group normalization: `(x - mean) / sqrt(var + eps) * weight + bias` per group.
    /// Matches candle_nn::GroupNorm::forward() — F32 promotion for F16/BF16.
    /// Input: `(batch, channels, ...)`, weight/bias: `(channels,)`.
    fn group_norm(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor> {
        use candle_core::DType;
        let x_shape = x.dims();
        let x_dtype = x.dtype();
        // Fast path: F32 CPU data — single-pass raw computation
        if x_dtype == DType::F32 {
            let (b_sz, n_channels) = (x_shape[0], x_shape[1]);
            let spatial: usize = x_shape[2..].iter().product();
            let channels_per_group = n_channels / num_groups;
            let group_size = channels_per_group * spatial;
            let x_data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let w_data = weight.to_vec1::<f32>()?;
            let b_data = bias.to_vec1::<f32>()?;
            let mut out = vec![0f32; x_data.len()];
            for batch in 0..b_sz {
                let batch_off = batch * n_channels * spatial;
                for g in 0..num_groups {
                    let group_off = batch_off + g * group_size;
                    let mut sum = 0f64;
                    let mut sum_sq = 0f64;
                    for i in 0..group_size {
                        let v = x_data[group_off + i] as f64;
                        sum += v;
                        sum_sq += v * v;
                    }
                    let mean = sum / group_size as f64;
                    let var = sum_sq / group_size as f64 - mean * mean;
                    let rstd = 1.0 / (var + eps as f64).sqrt();
                    for c in 0..channels_per_group {
                        let ch = g * channels_per_group + c;
                        let w = w_data[ch] as f64;
                        let b = b_data[ch] as f64;
                        let ch_off = group_off + c * spatial;
                        for s in 0..spatial {
                            let v = x_data[ch_off + s] as f64;
                            out[ch_off + s] = ((v - mean) * rstd * w + b) as f32;
                        }
                    }
                }
            }
            return Tensor::from_vec(out, x_shape, x.device());
        }
        // Fallback: tensor ops with dtype promotion
        let (b_sz, n_channels) = (x_shape[0], x_shape[1]);
        let hidden_size = x_shape[2..].iter().product::<usize>() * n_channels / num_groups;
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x = x.reshape((b_sz, num_groups, hidden_size))?;
        let x = x.to_dtype(internal_dtype)?;
        let mean_x = (x.sum_keepdim(2)? / hidden_size as f64)?;
        let x = x.broadcast_sub(&mean_x)?;
        let norm_x = (x.sqr()?.sum_keepdim(2)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
        let mut w_dims = vec![1; x_shape.len()];
        w_dims[1] = n_channels;
        let weight = weight.reshape(w_dims.clone())?;
        let bias = bias.reshape(w_dims)?;
        x_normed
            .to_dtype(x_dtype)?
            .reshape(x_shape)?
            .broadcast_mul(&weight)?
            .broadcast_add(&bias)
    }

    /// Softmax over the given dimension.
    /// Uses raw f32 computation for last-dim F32 (avoids CustomOp dispatch),
    /// fused kernel for other dtypes, generic path for non-last dim.
    fn softmax(&self, x: &Tensor, dim: usize) -> Result<Tensor> {
        if dim == x.rank() - 1 {
            // Fast path: F32 last-dim softmax on raw data
            if x.dtype() == DType::F32 {
                let shape = x.dims();
                let last = *shape.last().unwrap_or(&0);
                let data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
                let rows = data.len() / last;
                let mut out = vec![0f32; data.len()];
                for r in 0..rows {
                    let off = r * last;
                    let row = &data[off..off + last];
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for i in 0..last {
                        let e = (row[i] - max).exp();
                        out[off + i] = e;
                        sum += e;
                    }
                    let inv_sum = 1.0 / sum;
                    for i in 0..last {
                        out[off + i] *= inv_sum;
                    }
                }
                return Tensor::from_vec(out, shape, x.device());
            }
            candle_nn::ops::softmax_last_dim(x)
        } else {
            let max = x.max_keepdim(dim)?;
            let exp = x.broadcast_sub(&max)?.exp()?;
            let sum = exp.sum_keepdim(dim)?;
            exp.broadcast_div(&sum)
        }
    }

    /// Rotary position embedding: apply cos/sin rotation to tensor.
    /// `cos` and `sin` have shape `(seq_len, head_dim/2)` or compatible broadcast shape.
    fn rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        // Fast path: F32 CPU data — process pairs directly
        if x.dtype() == DType::F32 && cos.dtype() == DType::F32 {
            let x_shape = x.dims();
            let head_dim = *x_shape.last().unwrap_or(&0);
            let half = head_dim / 2;
            let x_data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let cos_data = cos.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let sin_data = sin.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let total_vecs = x_data.len() / head_dim;
            let cos_stride = cos_data.len() / half; // number of sequence positions in cos
            let mut out = vec![0f32; x_data.len()];
            for v in 0..total_vecs {
                let x_off = v * head_dim;
                // Determine which cos/sin row to use (seq position)
                let seq_idx = if cos_stride > 1 {
                    // x is (batch, heads, seq, dim), cos is (seq, half)
                    let seq_len = if x_shape.len() >= 3 {
                        x_shape[x_shape.len() - 2]
                    } else {
                        1
                    };
                    (v % seq_len) * half
                } else {
                    0
                };
                for i in 0..half {
                    let c = cos_data[seq_idx + i];
                    let s = sin_data[seq_idx + i];
                    let x1 = x_data[x_off + i];
                    let x2 = x_data[x_off + half + i];
                    out[x_off + i] = x1 * c - x2 * s;
                    out[x_off + half + i] = x2 * c + x1 * s;
                }
            }
            return Tensor::from_vec(out, x_shape, x.device());
        }
        candle_nn::rotary_emb::rope(x, cos, sin)
    }

    /// SiLU (Swish) activation: `x * sigmoid(x)`.
    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::silu(x)
    }

    /// GELU activation function.
    fn gelu(&self, x: &Tensor) -> Result<Tensor> {
        // Fast path: raw f32 GELU approximation (tanh-based, matches PyTorch)
        if x.dtype() == DType::F32 {
            let data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
            let shape = x.dims();
            let mut out = data;
            let sqrt_2_over_pi: f32 = 0.797_884_6; // sqrt(2/pi)
            for v in out.iter_mut() {
                let x = *v;
                let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                *v = 0.5 * x * (1.0 + inner.tanh());
            }
            return Tensor::from_vec(out, shape, x.device());
        }
        x.gelu()
    }

    /// Sigmoid activation: `1 / (1 + exp(-x))`.
    fn sigmoid(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::sigmoid(x)
    }

    /// Embedding lookup: select rows from weight matrix by token IDs.
    fn embedding(&self, ids: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let hidden_size = weight.dim(1)?;
        let dims = ids.dims();
        // Fast path: 1D input — no reshape needed
        if dims.len() == 1 {
            let selected = weight.index_select(ids, 0)?;
            return Ok(selected);
        }
        // General path: flatten, select, reshape
        let elem_count: usize = dims.iter().product();
        let flat_ids = ids.reshape(elem_count)?;
        let selected = weight.index_select(&flat_ids, 0)?;
        let mut out_shape = dims.to_vec();
        out_shape.push(hidden_size);
        selected.reshape(out_shape.as_slice())
    }

    /// Create a causal attention mask. Returns a U8 tensor of shape `(seq_len, kv_len)`
    /// where 1 = masked (future position), 0 = attend.
    /// Callers use `masked_fill` or `where_cond` to apply the mask.
    fn causal_mask(
        &self,
        seq_len: usize,
        kv_len: usize,
        device: &Device,
    ) -> Result<Tensor> {
        if seq_len == 1 {
            return Tensor::zeros((1, kv_len), DType::U8, device);
        }
        // Build full (seq_len, kv_len) mask in one allocation — no Tensor::cat.
        let prefix = kv_len.saturating_sub(seq_len);
        let mut mask = vec![0u8; seq_len * kv_len];
        for i in 0..seq_len {
            let row_start = i * kv_len + prefix + i + 1;
            let row_end = (i + 1) * kv_len;
            if row_start < row_end {
                mask[row_start..row_end].fill(1);
            }
        }
        Tensor::from_vec(mask, (seq_len, kv_len), device)
    }

    /// Top-K selection: returns `(values, indices)` for the largest K elements
    /// along the last dimension. Uses partial sort O(N + K log K) for large N.
    fn topk(&self, x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
        let x = x.contiguous()?;
        let shape = x.shape();
        let last = shape.dims().last().copied().unwrap_or(0);
        if last <= 32 || k * 2 >= last {
            let last_dim = x.rank() - 1;
            let (sorted, indices) = x.sort_last_dim(false)?;
            let top_vals = sorted.narrow(last_dim, 0, k)?;
            let top_idx = indices.narrow(last_dim, 0, k)?;
            return Ok((top_vals, top_idx));
        }
        let rank = x.rank();
        let batch: usize = shape.dims()[..rank - 1].iter().product();
        let flat = x.reshape((batch, last))?;
        let data = flat.to_vec2::<f32>()?;
        let mut all_vals = Vec::with_capacity(batch * k);
        let mut all_idxs = Vec::with_capacity(batch * k);
        for row in &data {
            let mut indices: Vec<u32> = (0..last as u32).collect();
            indices.select_nth_unstable_by(k - 1, |&a, &b| {
                row[b as usize]
                    .partial_cmp(&row[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let top_slice = &mut indices[..k];
            top_slice.sort_unstable_by(|&a, &b| {
                row[b as usize]
                    .partial_cmp(&row[a as usize])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for &idx in top_slice.iter() {
                all_vals.push(row[idx as usize]);
                all_idxs.push(idx);
            }
        }
        let mut out_shape: Vec<usize> = shape.dims()[..rank - 1].to_vec();
        out_shape.push(k);
        let vals = Tensor::from_vec(all_vals, out_shape.as_slice(), x.device())?;
        let idxs = Tensor::from_vec(all_idxs, out_shape.as_slice(), x.device())?;
        Ok((vals, idxs))
    }

    // ── Convolutions ──────────────────────────────────────────────────

    /// 1D convolution: `conv1d(x, weight) + bias`.
    /// Matches candle_nn::Conv1d::forward() semantics.
    /// Input: `(batch, in_channels, length)`, weight: `(out_channels, in_channels/groups, kernel_size)`.
    #[allow(clippy::too_many_arguments)]
    fn conv1d(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let out = x.conv1d(weight, padding, stride, dilation, groups)?;
        match bias {
            Some(b) => {
                let b = b.reshape((1, b.dim(0)?, 1))?;
                out.broadcast_add(&b)
            }
            None => Ok(out),
        }
    }

    /// Transposed 1D convolution: `conv_transpose1d(x, weight) + bias`.
    /// Matches candle_nn::ConvTranspose1d::forward() semantics.
    #[allow(clippy::too_many_arguments)]
    fn conv_transpose1d(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        padding: usize,
        output_padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let out = x.conv_transpose1d(weight, padding, output_padding, stride, dilation, groups)?;
        match bias {
            Some(b) => {
                let b = b.reshape((1, b.dim(0)?, 1))?;
                out.broadcast_add(&b)
            }
            None => Ok(out),
        }
    }

    /// 2D convolution: `conv2d(x, weight) + bias`.
    /// Matches candle_nn::Conv2d::forward() semantics.
    /// Input: `(batch, in_channels, height, width)`, weight: `(out_channels, in_channels/groups, kH, kW)`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        x: &Tensor,
        weight: &Tensor,
        bias: Option<&Tensor>,
        padding: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor> {
        let out = x.conv2d(weight, padding, stride, dilation, groups)?;
        match bias {
            Some(b) => {
                let b = b.reshape((1, b.dim(0)?, 1, 1))?;
                out.broadcast_add(&b)
            }
            None => Ok(out),
        }
    }

    // ── Device control ───────────────────────────────────────────────

    /// Flush GPU command buffer. No-op on CPU/CUDA, required on Metal
    /// to prevent command buffer accumulation (>25 commands = 50x slowdown).
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}

/// Create the appropriate backend for the given device.
pub fn create_backend(device: &Device) -> Arc<dyn ComputeBackend> {
    match device {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => Arc::new(CudaBackend::new(device.clone())),
        #[cfg(feature = "metal")]
        Device::Metal(_) => Arc::new(MetalBackend::new(device.clone())),
        _ => {
            // No GPU device — try ROCm first (native AMD), then Vulkan (wgpu)
            #[cfg(feature = "rocm")]
            match RocmBackend::new() {
                Ok(r) => {
                    log::info!("using ROCm backend (rocBLAS GEMM)");
                    return Arc::new(r);
                }
                Err(e) => log::warn!("ROCm init failed ({e}), trying next backend"),
            }
            #[cfg(feature = "vulkan")]
            match VulkanBackend::new() {
                Ok(vk) => {
                    log::info!("using Vulkan backend for GPU-accelerated ops");
                    return Arc::new(vk);
                }
                Err(e) => log::warn!("Vulkan init failed ({e}), falling back to CPU"),
            }
            Arc::new(CpuBackend::new())
        }
    }
}
