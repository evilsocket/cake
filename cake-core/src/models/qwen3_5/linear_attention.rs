//! Gated DeltaNet linear attention for Qwen3.5.
//!
//! Implements recurrent linear attention with a delta rule update:
//!   1. Project input to Q, K, V via fused in_proj_qkv
//!   2. Apply causal depthwise conv1d with SiLU
//!   3. Compute gates: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)
//!   4. Recurrent update: S = S * exp(g) + outer(k, beta*(v - S^T k))
//!   5. Output: o = S^T q, gated by sigmoid(z) via RMSNormGated

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::models::common::{Cache, Config};

/// RMSNorm with multiplicative SiLU gating: rms_norm(x) * silu(z).
#[derive(Debug, Clone)]
struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    /// Apply gated RMS normalization: (1 + weight) * rms_norm(x) * silu(z).
    fn forward(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let z = z.to_dtype(DType::F32)?;

        // RMS norm
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;

        // weight * norm(x)
        let weight_f32 = self.weight.to_dtype(DType::F32)?;
        let x_normed = x_normed.broadcast_mul(&weight_f32)?;

        // Gating: silu(z)
        let gate = candle_nn::ops::silu(&z)?;
        let result = (x_normed * gate)?;

        result.to_dtype(in_dtype)
    }
}

/// Gated DeltaNet linear attention block.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_a: Linear,
    in_proj_b: Linear,
    in_proj_z: Linear,
    conv1d_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm: RmsNormGated,
    out_proj: Linear,

    num_heads: usize,
    num_key_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
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

        let in_proj_qkv = candle_nn::linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_a = candle_nn::linear_no_bias(cfg.hidden_size, num_heads, vb.pp("in_proj_a"))?;
        let in_proj_b = candle_nn::linear_no_bias(cfg.hidden_size, num_heads, vb.pp("in_proj_b"))?;
        let in_proj_z = candle_nn::linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let out_proj = candle_nn::linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))?;

        // Conv1d weight: stored as (conv_dim, 1, kernel_size), we want (conv_dim, kernel_size)
        let conv1d_weight = vb.get((conv_dim, 1, la.conv_kernel_dim), "conv1d.weight")?
            .squeeze(1)?;

        let a_log = vb.get(num_heads, "A_log")?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;

        let norm = RmsNormGated::load(value_head_dim, cfg.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_a,
            in_proj_b,
            in_proj_z,
            conv1d_weight,
            a_log,
            dt_bias,
            norm,
            out_proj,
            num_heads,
            num_key_heads,
            key_head_dim,
            value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: la.conv_kernel_dim,
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
    /// x: (batch, seq_len, channels)
    fn causal_conv1d_seq(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, channels) = x.dims3()?;
        let kernel_size = self.conv_kernel_size;
        let pad = kernel_size - 1;

        // Transpose to (batch, channels, seq_len) for conv
        let x_t = x.transpose(1, 2)?;

        // Left-pad with zeros for causal conv
        let padding = Tensor::zeros((batch, channels, pad), x_t.dtype(), x_t.device())?;
        let padded = Tensor::cat(&[&padding, &x_t], 2)?;

        // Depthwise conv: for each position, multiply-accumulate over kernel
        // padded: (batch, channels, seq_len + pad)
        // weight: (channels, kernel_size)
        let weight = self.conv1d_weight.to_dtype(padded.dtype())?;
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let window = padded.narrow(2, t, kernel_size)?;
            let y = (window.broadcast_mul(&weight.unsqueeze(0)?)?).sum(D::Minus1)?;
            outputs.push(y);
        }
        let y = Tensor::stack(&outputs, 1)?; // (batch, seq_len, channels)

        // SiLU activation
        let y = candle_nn::ops::silu(&y)?;

        // Save last (kernel_size-1) timesteps as conv state
        let conv_state = x_t.narrow(2, seq_len.saturating_sub(pad), seq_len.min(pad))?;
        let conv_state = if conv_state.dims()[2] < pad {
            let extra_pad = Tensor::zeros(
                (batch, channels, pad - conv_state.dims()[2]),
                conv_state.dtype(),
                conv_state.device(),
            )?;
            Tensor::cat(&[&extra_pad, &conv_state], 2)?
        } else {
            conv_state.contiguous()?
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

    /// L2-normalize along the last dimension: x / sqrt(sum(x^2) + eps).
    fn l2_normalize(x: &Tensor) -> Result<Tensor> {
        let x_f32 = x.to_dtype(DType::F32)?;
        let sum_sq = x_f32.sqr()?.sum_keepdim(D::Minus1)?;
        // rsqrt(sum_sq + eps) to match Python: torch.rsqrt(x*x.sum(-1,keepdim=True) + 1e-6)
        let inv_norm = (sum_sq + 1e-6)?.sqrt()?.recip()?;
        let result = x_f32.broadcast_mul(&inv_norm)?;
        result.to_dtype(x.dtype())
    }

    /// Recurrent forward for a single timestep (token-by-token generation).
    fn recurrent_step(
        &self,
        q: &Tensor,    // (batch, num_heads, key_head_dim)
        k: &Tensor,    // (batch, num_heads, key_head_dim)
        v: &Tensor,    // (batch, num_heads, value_head_dim)
        g: &Tensor,    // (batch, num_heads) - log-decay
        beta: &Tensor, // (batch, num_heads) - input gate
        state: &Tensor, // (batch, num_heads, key_head_dim, value_head_dim)
    ) -> Result<(Tensor, Tensor)> {
        let in_dtype = state.dtype();

        // Work in f32 for numerical stability
        let state = state.to_dtype(DType::F32)?;
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let g = g.to_dtype(DType::F32)?;
        let beta = beta.to_dtype(DType::F32)?;

        // 1. Decay state: S = S * exp(g)
        // g: (batch, num_heads) -> (batch, num_heads, 1, 1)
        let decay = g.unsqueeze(D::Minus1)?.unsqueeze(D::Minus1)?.exp()?;
        let state = state.broadcast_mul(&decay)?;

        // 2. Retrieve from state: retrieved = einsum('bhkv,bhk->bhv' state, k)
        // k: (batch, heads, key_dim) -> (batch, heads, key_dim, 1)
        // state: (batch, heads, key_dim, val_dim)
        // result: (batch, heads, 1, val_dim) -> (batch, heads, val_dim)
        let k_4d = k.unsqueeze(D::Minus1)?;
        let retrieved = state.broadcast_mul(&k_4d)?.sum(2)?; // (batch, heads, val_dim)

        // 3. Delta rule: delta = beta * (v - retrieved)
        let beta_3d = beta.unsqueeze(D::Minus1)?; // (batch, heads, 1)
        let diff = v.broadcast_sub(&retrieved)?;
        let delta = diff.broadcast_mul(&beta_3d)?; // (batch, heads, val_dim)

        // 4. Update state: S = S + outer(k, delta) = S + k[:,:,:,None] * delta[:,:,None,:]
        let update = k_4d.broadcast_mul(&delta.unsqueeze(2)?)?;
        let state = state.broadcast_add(&update)?;

        // 5. Query: o = einsum('bhkv,bhk->bhv' state, q)
        let q_4d = q.unsqueeze(D::Minus1)?;
        let output = state.broadcast_mul(&q_4d)?.sum(2)?; // (batch, heads, val_dim)

        Ok((output.to_dtype(in_dtype)?, state.to_dtype(in_dtype)?))
    }

    /// Full forward pass through the Gated DeltaNet layer.
    pub fn forward(
        &self,
        x: &Tensor,
        block_idx: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3().map_err(|e| anyhow!("dims3: {e}"))?;

        // Project QKV
        let mixed_qkv = self.in_proj_qkv.forward(x)
            .map_err(|e| anyhow!("in_proj_qkv: {e}"))?;

        // Project gates
        let z = self.in_proj_z.forward(x)
            .map_err(|e| anyhow!("in_proj_z: {e}"))?;
        let a = self.in_proj_a.forward(x)
            .map_err(|e| anyhow!("in_proj_a: {e}"))?;
        let b = self.in_proj_b.forward(x)
            .map_err(|e| anyhow!("in_proj_b: {e}"))?;

        // Apply causal conv1d + SiLU
        let (mixed_qkv, new_conv_state) = if seq_len == 1 {
            if let Some(conv_state) = cache.get_conv_state(block_idx) {
                let (y, s) = self.causal_conv1d_step(&mixed_qkv, conv_state)
                    .map_err(|e| anyhow!("conv1d_step: {e}"))?;
                (y, s)
            } else {
                // First token without prior state — create zero state
                let conv_dim = self.key_dim * 2 + self.value_dim;
                let zero_state = Tensor::zeros(
                    (batch, conv_dim, self.conv_kernel_size - 1),
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

        // L2-normalize Q and K, then scale Q by 1/sqrt(key_head_dim)
        let q = Self::l2_normalize(&q).map_err(|e| anyhow!("l2norm q: {e}"))?;
        let k = Self::l2_normalize(&k).map_err(|e| anyhow!("l2norm k: {e}"))?;
        let scale = 1.0 / (self.key_head_dim as f64).sqrt();
        let q = (q * scale).map_err(|e| anyhow!("q scale: {e}"))?;

        // Compute gating parameters
        // g = -exp(A_log) * softplus(a + dt_bias), per head per timestep
        let a_log_f32 = self.a_log.to_dtype(DType::F32).map_err(|e| anyhow!("a_log dtype: {e}"))?;
        let dt_bias_f32 = self.dt_bias.to_dtype(DType::F32).map_err(|e| anyhow!("dt_bias dtype: {e}"))?;
        let a_f32 = a.to_dtype(DType::F32).map_err(|e| anyhow!("a dtype: {e}"))?;
        let a_plus_dt = a_f32.broadcast_add(&dt_bias_f32).map_err(|e| anyhow!("a+dt: {e}"))?;
        // softplus(x) = ln(1 + exp(x))
        let softplus_a = (a_plus_dt.exp()? + 1.0)?.log()?;
        let neg_a_exp = a_log_f32.exp()?.neg()?;
        let g = softplus_a.broadcast_mul(&neg_a_exp)
            .map_err(|e| anyhow!("g compute: {e}"))?;
        // g: (batch, seq_len, num_heads)

        // beta = sigmoid(b)
        let beta = candle_nn::ops::sigmoid(&b.to_dtype(DType::F32)
            .map_err(|e| anyhow!("b dtype: {e}"))?)
            .map_err(|e| anyhow!("sigmoid: {e}"))?;
        // beta: (batch, seq_len, num_heads)

        // Get or initialize recurrent state
        let dev = x.device();
        let dtype = x.dtype();
        let mut state = if let Some(s) = cache.get_recurrent_state(block_idx) {
            s.clone()
        } else {
            Tensor::zeros(
                (batch, self.num_heads, self.key_head_dim, self.value_head_dim),
                dtype,
                dev,
            ).map_err(|e| anyhow!("init state: {e}"))?
        };

        // Recurrent processing: iterate over timesteps
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let q_t = q.narrow(1, t, 1)?.squeeze(1)?; // (batch, heads, key_dim)
            let k_t = k.narrow(1, t, 1)?.squeeze(1)?;
            let v_t = v.narrow(1, t, 1)?.squeeze(1)?;
            let g_t = g.narrow(1, t, 1)?.squeeze(1)?; // (batch, heads)
            let beta_t = beta.narrow(1, t, 1)?.squeeze(1)?;

            let (o_t, new_state) = self.recurrent_step(&q_t, &k_t, &v_t, &g_t, &beta_t, &state)
                .map_err(|e| anyhow!("recurrent step t={t}: {e}"))?;
            state = new_state;
            outputs.push(o_t); // (batch, heads, val_dim)
        }

        // Save updated state
        cache.set_recurrent_state(block_idx, state);

        // Stack outputs: (batch, seq_len, heads, val_dim)
        let output = Tensor::stack(&outputs, 1)
            .map_err(|e| anyhow!("stack outputs: {e}"))?;

        // Apply gated RMS norm: norm(output, z)
        // z: (batch, seq_len, value_dim) -> (batch, seq_len, heads, val_head_dim)
        let z = z.reshape((batch, seq_len, self.num_heads, self.value_head_dim))
            .map_err(|e| anyhow!("z reshape: {e}"))?;
        let output = self.norm.forward(&output, &z)
            .map_err(|e| anyhow!("norm: {e}"))?;

        // Flatten heads: (batch, seq_len, value_dim)
        let output = output.reshape((batch, seq_len, self.value_dim))
            .map_err(|e| anyhow!("output reshape: {e}"))?;

        // Project output
        let output = self.out_proj.forward(&output)
            .map_err(|e| anyhow!("out_proj: {e}"))?;

        Ok(output)
    }
}
