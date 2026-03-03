//! Gated DeltaNet linear attention for Qwen3.5.
//!
//! Implements recurrent linear attention with a delta rule update:
//!   1. Project input to Q, K, V, A, B, Z via fused in_proj
//!   2. Apply causal depthwise conv1d with SiLU
//!   3. Compute gates: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)
//!   4. Recurrent update: S = S * exp(g) + outer(k, beta*(v - S^T k))
//!   5. Output: o = S^T q, gated by sigmoid(z) via RMSNormGated

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::models::common::{Cache, Config};

/// RMSNorm with multiplicative SiLU gating: rms_norm(x) * silu(z).
/// Uses candle's fused rms_norm kernel (1 kernel) instead of 7 separate ops.
#[derive(Debug, Clone)]
struct RmsNormGated {
    weight: Tensor,
    eps: f32,
}

impl RmsNormGated {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        // Store weight as F32 to match the recurrent step's F32 output.
        let weight = vb.get(size, "weight")?.to_dtype(DType::F32)?;
        Ok(Self { weight, eps: eps as f32 })
    }

    /// Apply gated RMS normalization: weight * rms_norm(x) * silu(z).
    /// Uses fused rms_norm (1 kernel) + silu + mul = 4 kernels instead of 10.
    fn forward(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        // Fused RMS norm on F32 input (x is F32 from recurrent step)
        let x_normed = candle_nn::ops::rms_norm(&x.contiguous()?, &self.weight, self.eps)?;
        // Gate: silu(z) in F32, then multiply
        let gate = candle_nn::ops::silu(&z.to_dtype(x.dtype())?)?;
        x_normed * gate
    }
}

/// Numerically stable softplus: ln(1 + exp(x)).
/// Uses max(x, 0) + ln(1 + exp(-|x|)) to avoid exp overflow for large x.
fn stable_softplus(x: &Tensor) -> Result<Tensor> {
    let abs_x = x.abs()?;
    let pos_part = x.maximum(0f64)?;
    let log_part = (abs_x.neg()?.exp()? + 1.0)?.log()?;
    &pos_part + &log_part
}

/// Gated DeltaNet linear attention block with fused input projections.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet {
    /// Fused projection: QKV + A + B + Z in a single matmul
    in_proj: Linear,
    conv1d_weight: Tensor,
    norm: RmsNormGated,
    out_proj: Linear,

    // Precomputed constants (F32, avoids per-call recomputation)
    neg_a_exp_f32: Tensor,
    dt_bias_f32: Tensor,
    conv1d_weight_3d: Tensor,
    q_scale: f64,

    num_heads: usize,
    num_key_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,

    // Split offsets for the fused projection output
    conv_dim: usize,
}

impl GatedDeltaNet {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let la = cfg.linear_attn.as_ref().expect("no linear_attn config");

        let num_heads = la.num_value_heads;
        let num_key_heads = la.num_key_heads;
        let key_head_dim = la.key_head_dim;
        let value_head_dim = la.value_head_dim;
        let key_dim = num_key_heads * key_head_dim;
        let value_dim = num_heads * value_head_dim;
        let conv_dim = key_dim * 2 + value_dim; // Q + K + V
        let h_size = cfg.hidden_size;

        // Fuse all 4 input projections into a single Linear:
        // in_proj_qkv (conv_dim) + in_proj_a (num_heads) + in_proj_b (num_heads) + in_proj_z (value_dim)
        let qkv_w = vb.pp("in_proj_qkv").get((conv_dim, h_size), "weight")?;
        let a_w = vb.pp("in_proj_a").get((num_heads, h_size), "weight")?;
        let b_w = vb.pp("in_proj_b").get((num_heads, h_size), "weight")?;
        let z_w = vb.pp("in_proj_z").get((value_dim, h_size), "weight")?;
        let fused_w = Tensor::cat(&[&qkv_w, &a_w, &b_w, &z_w], 0)?;
        let in_proj = Linear::new(fused_w, None);

        let out_w = vb.pp("out_proj").get((h_size, value_dim), "weight")?;
        let out_proj = Linear::new(out_w, None);

        // Conv1d weight: stored as (conv_dim, 1, kernel_size)
        let conv1d_weight_3d = vb.get((conv_dim, 1, la.conv_kernel_dim), "conv1d.weight")?;
        // Squeezed view for causal_conv1d_step: (conv_dim, kernel_size)
        let conv1d_weight = conv1d_weight_3d.squeeze(1)?;

        let a_log = vb.get(num_heads, "A_log")?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        // Precompute constants in F32 to avoid per-call recomputation
        let neg_a_exp_f32 = a_log.to_dtype(DType::F32)?.exp()?.neg()?;
        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let q_scale = 1.0 / (key_head_dim as f64).sqrt();

        let norm = RmsNormGated::load(value_head_dim, cfg.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            norm,
            out_proj,
            neg_a_exp_f32,
            dt_bias_f32,
            conv1d_weight_3d,
            q_scale,
            num_heads,
            num_key_heads,
            key_head_dim,
            value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: la.conv_kernel_dim,
            conv_dim,
        })
    }

    /// Apply causal depthwise conv1d + SiLU on a single new token, updating conv_state in-place.
    /// x: (batch, 1, channels), conv_state: (batch, channels, kernel_size-1)
    fn causal_conv1d_step(&self, x: &Tensor, conv_state: &Tensor) -> Result<(Tensor, Tensor)> {
        // x is (batch, 1, channels) -> (batch, channels, 1)
        let x_t = x.transpose(1, 2)?;

        // Build full window: cat(state, new_input) -> (batch, channels, kernel_size)
        let full_window = Tensor::cat(&[conv_state, &x_t], 2)?;

        // Depthwise conv: for each channel, dot product of window with kernel
        // conv1d_weight: (channels, kernel_size)
        // full_window: (batch, channels, kernel_size)
        // Result: sum along kernel dimension -> (batch, channels)
        let weight = self.conv1d_weight.to_dtype(full_window.dtype())?;
        let y = full_window.broadcast_mul(&weight.unsqueeze(0)?)?.sum(D::Minus1)?;

        // SiLU activation, reshape to (batch, 1, channels)
        let y = candle_nn::ops::silu(&y)?.unsqueeze(1)?;

        // New state: last (kernel_size-1) elements of full_window
        let new_state = full_window.narrow(2, 1, self.conv_kernel_size - 1)?;

        Ok((y, new_state))
    }

    /// Apply causal depthwise conv1d + SiLU on a full sequence.
    /// Uses native Tensor::conv1d with groups=channels (depthwise) instead of a per-timestep loop.
    /// x: (batch, seq_len, channels)
    fn causal_conv1d_seq(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, channels) = x.dims3()?;
        let pad = self.conv_kernel_size - 1;

        // Transpose to (batch, channels, seq_len) for conv1d API
        let x_t = x.transpose(1, 2)?;

        // Left-pad with zeros for causal conv: (batch, channels, pad + seq_len)
        let padded = x_t.pad_with_zeros(2, pad, 0)?;

        // Depthwise conv1d: groups=channels, kernel (channels, 1, kernel_size)
        let w = self.conv1d_weight_3d.to_dtype(padded.dtype())?;
        let y = padded.conv1d(&w, 0, 1, 1, channels)?;

        // Transpose back to (batch, seq_len, channels) and apply SiLU
        let y = y.transpose(1, 2)?;
        let y = candle_nn::ops::silu(&y)?;

        // Save conv state: last (kernel_size-1) timesteps of input
        let conv_state = if seq_len >= pad {
            x_t.narrow(2, seq_len - pad, pad)?.contiguous()?
        } else {
            let extra = Tensor::zeros(
                (batch, channels, pad - seq_len),
                x_t.dtype(),
                x_t.device(),
            )?;
            Tensor::cat(&[&extra, &x_t.contiguous()?], 2)?
        };

        Ok((y, conv_state))
    }

    /// Repeat key heads to match value head count (GQA-like expansion).
    /// Input: (batch, seq, num_key_heads, head_dim)
    /// Output: (batch, seq, num_key_heads * n_rep, head_dim)
    fn repeat_key_heads(x: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (b, s, h, d) = x.dims4()?;
        // Unsqueeze to (b, s, h, 1, d), expand to (b, s, h, n_rep, d), flatten
        x.unsqueeze(3)?
            .expand((b, s, h, n_rep, d))?
            .contiguous()?
            .reshape((b, s, h * n_rep, d))
    }

    /// L2-normalize along the last dimension, returning F32.
    /// Stays in F32 to avoid round-tripping when the result feeds into F32 recurrent computation.
    fn l2_normalize_f32(x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.sqr()?.sum_keepdim(D::Minus1)?;
        let inv_norm = (sum_sq + 1e-6)?.sqrt()?.recip()?;
        x_f32.broadcast_mul(&inv_norm)
    }

    /// Recurrent forward for a single timestep (token-by-token generation).
    /// All inputs and state must be F32 to avoid per-step dtype conversions.
    fn recurrent_step(
        q: &Tensor,    // (batch, num_heads, key_head_dim) — F32
        k: &Tensor,    // (batch, num_heads, key_head_dim) — F32
        v: &Tensor,    // (batch, num_heads, value_head_dim) — F32
        g: &Tensor,    // (batch, num_heads) — F32
        beta: &Tensor, // (batch, num_heads) — F32
        state: &Tensor, // (batch, num_heads, key_head_dim, value_head_dim) — F32
    ) -> Result<(Tensor, Tensor)> {
        // 1. Decay state: S = S * exp(g)
        let decay = g.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.exp()?;
        let state = state.broadcast_mul(&decay)?;

        // 2. Retrieve: retrieved = S^T @ k  (batched matrix-vector)
        let k_4d = k.unsqueeze(D::Minus1)?;
        let retrieved = state.transpose(2, 3)?.matmul(&k_4d)?.squeeze(D::Minus1)?;

        // 3. Delta rule: delta = beta * (v - retrieved)
        let beta_3d = beta.unsqueeze(D::Minus1)?;
        let delta = (v - &retrieved)?.broadcast_mul(&beta_3d)?;

        // 4. Update state: S = S + k @ delta^T  (rank-1 outer product)
        let update = k_4d.matmul(&delta.unsqueeze(2)?)?;
        let state = (state + update)?;

        // 5. Query: output = S^T @ q  (batched matrix-vector)
        let q_4d = q.unsqueeze(D::Minus1)?;
        let output = state.transpose(2, 3)?.matmul(&q_4d)?.squeeze(D::Minus1)?;

        Ok((output, state))
    }

    /// Full forward pass through the Gated DeltaNet layer.
    pub fn forward(
        &self,
        x: &Tensor,
        block_idx: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3().map_err(|e| anyhow!("dims3: {e}"))?;
        let model_dtype = x.dtype();

        // Single fused projection: QKV + A + B + Z
        let proj = self.in_proj.forward(x)
            .map_err(|e| anyhow!("in_proj: {e}"))?;

        // Split fused output
        let mixed_qkv = proj.narrow(2, 0, self.conv_dim)
            .map_err(|e| anyhow!("split qkv: {e}"))?;
        let a = proj.narrow(2, self.conv_dim, self.num_heads)
            .map_err(|e| anyhow!("split a: {e}"))?;
        let b = proj.narrow(2, self.conv_dim + self.num_heads, self.num_heads)
            .map_err(|e| anyhow!("split b: {e}"))?;
        let z = proj.narrow(2, self.conv_dim + self.num_heads * 2, self.value_dim)
            .map_err(|e| anyhow!("split z: {e}"))?;

        // Apply causal conv1d + SiLU
        let (mixed_qkv, new_conv_state) = if seq_len == 1 {
            if let Some(conv_state) = cache.get_conv_state(block_idx) {
                let (y, s) = self.causal_conv1d_step(&mixed_qkv, conv_state)
                    .map_err(|e| anyhow!("conv1d_step: {e}"))?;
                (y, s)
            } else {
                // First token without prior state — create zero state
                let zero_state = Tensor::zeros(
                    (batch, self.conv_dim, self.conv_kernel_size - 1),
                    mixed_qkv.dtype(),
                    mixed_qkv.device(),
                ).map_err(|e| anyhow!("zero conv state: {e}"))?;
                let (y, s) = self.causal_conv1d_step(&mixed_qkv, &zero_state)
                    .map_err(|e| anyhow!("conv1d_step: {e}"))?;
                (y, s)
            }
        } else {
            self.causal_conv1d_seq(&mixed_qkv)
                .map_err(|e| anyhow!("conv1d_seq: {e}"))?
        };
        cache.set_conv_state(block_idx, new_conv_state);

        // Split into Q, K, V: (batch, seq_len, dim)
        let q = mixed_qkv.narrow(2, 0, self.key_dim)
            .map_err(|e| anyhow!("split q: {e}"))?;
        let k = mixed_qkv.narrow(2, self.key_dim, self.key_dim)
            .map_err(|e| anyhow!("split k: {e}"))?;
        let v = mixed_qkv.narrow(2, self.key_dim * 2, self.value_dim)
            .map_err(|e| anyhow!("split v: {e}"))?;

        // Reshape Q/K to (batch, seq_len, num_key_heads, key_head_dim).
        // When num_key_heads < num_heads (GQA-like), repeat key heads to match
        // the number of recurrent heads (num_heads = num_value_heads).
        let q = q.reshape((batch, seq_len, self.num_key_heads, self.key_head_dim))
            .map_err(|e| anyhow!("q reshape: {e}"))?;
        let k = k.reshape((batch, seq_len, self.num_key_heads, self.key_head_dim))
            .map_err(|e| anyhow!("k reshape: {e}"))?;
        let (q, k) = if self.num_key_heads < self.num_heads {
            let repeats = self.num_heads / self.num_key_heads;
            let q = Self::repeat_key_heads(&q, repeats)
                .map_err(|e| anyhow!("q repeat: {e}"))?;
            let k = Self::repeat_key_heads(&k, repeats)
                .map_err(|e| anyhow!("k repeat: {e}"))?;
            (q, k)
        } else {
            (q, k)
        };
        let v = v.reshape((batch, seq_len, self.num_heads, self.value_head_dim))
            .map_err(|e| anyhow!("v reshape: {e}"))?;

        // L2-normalize Q and K, staying in F32 to avoid round-tripping.
        // The recurrent computation runs entirely in F32.
        let q = Self::l2_normalize_f32(&q).map_err(|e| anyhow!("l2norm q: {e}"))?;
        let k = Self::l2_normalize_f32(&k).map_err(|e| anyhow!("l2norm k: {e}"))?;
        let q = (q * self.q_scale).map_err(|e| anyhow!("q scale: {e}"))?;

        // Convert v to F32 once before the recurrent loop
        let v = v.to_dtype(DType::F32).map_err(|e| anyhow!("v to f32: {e}"))?;

        // Compute gating parameters in F32 using precomputed constants
        // g = neg_a_exp * softplus(a + dt_bias), per head per timestep
        let a_f32 = a.to_dtype(DType::F32).map_err(|e| anyhow!("a dtype: {e}"))?;
        let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32).map_err(|e| anyhow!("a+dt: {e}"))?;
        let softplus_a = stable_softplus(&a_plus_dt)?;
        let g = softplus_a.broadcast_mul(&self.neg_a_exp_f32)
            .map_err(|e| anyhow!("g compute: {e}"))?;

        // beta = sigmoid(b) — computed in F32
        let beta = candle_nn::ops::sigmoid(&b.to_dtype(DType::F32)
            .map_err(|e| anyhow!("b dtype: {e}"))?)
            .map_err(|e| anyhow!("sigmoid: {e}"))?;

        // Get or initialize recurrent state (always F32)
        let dev = x.device();
        let mut state = if let Some(s) = cache.get_recurrent_state(block_idx) {
            s.clone()
        } else {
            Tensor::zeros(
                (batch, self.num_heads, self.key_head_dim, self.value_head_dim),
                DType::F32,
                dev,
            ).map_err(|e| anyhow!("init state: {e}"))?
        };

        // Recurrent processing: all inputs are F32, no per-step dtype conversions
        let output = if seq_len == 1 {
            // Single token: skip loop and stack overhead
            let q_t = q.squeeze(1)?;
            let k_t = k.squeeze(1)?;
            let v_t = v.squeeze(1)?;
            let g_t = g.squeeze(1)?;
            let beta_t = beta.squeeze(1)?;
            let (o_t, new_state) = Self::recurrent_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &state)
                .map_err(|e| anyhow!("recurrent step: {e}"))?;
            state = new_state;
            o_t.unsqueeze(1)?
        } else {
            let mut outputs = Vec::with_capacity(seq_len);
            for t in 0..seq_len {
                let q_t = q.narrow(1, t, 1)?.squeeze(1)?;
                let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
                let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
                let g_t = g.narrow(1, t, 1)?.squeeze(1)?;
                let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

                let (o_t, new_state) = Self::recurrent_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &state)
                    .map_err(|e| anyhow!("recurrent step t={t}: {e}"))?;
                state = new_state;
                outputs.push(o_t);
            }
            Tensor::stack(&outputs, 1)
                .map_err(|e| anyhow!("stack outputs: {e}"))?
        };

        // Save updated state
        cache.set_recurrent_state(block_idx, state);

        // Apply gated RMS norm: norm(output_f32, z_model_dtype)
        // z: (batch, seq_len, value_dim) -> (batch, seq_len, heads, val_head_dim)
        let z = z.reshape((batch, seq_len, self.num_heads, self.value_head_dim))
            .map_err(|e| anyhow!("z reshape: {e}"))?;
        let output = self.norm.forward(&output, &z)
            .map_err(|e| anyhow!("norm: {e}"))?;

        // Flatten heads and convert back to model dtype for output projection
        let output = output.reshape((batch, seq_len, self.value_dim))?
            .to_dtype(model_dtype)
            .map_err(|e| anyhow!("output reshape: {e}"))?;

        // Project output
        let output = self.out_proj.forward(&output)
            .map_err(|e| anyhow!("out_proj: {e}"))?;

        Ok(output)
    }
}
