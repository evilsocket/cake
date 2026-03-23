//! Gated DeltaNet linear attention for Qwen3.5.
//!
//! Implements recurrent linear attention with a delta rule update:
//!   1. Project input to Q, K, V, A, B, Z via fused in_proj
//!   2. Apply causal depthwise conv1d with SiLU
//!   3. Compute gates: g = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b)
//!   4. Recurrent update: S = S * exp(g) + outer(k, beta*(v - S^T k))
//!   5. Output: o = S^T q, gated by sigmoid(z) via RMSNormGated

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};

use crate::backends::ComputeBackend;
use crate::models::common::{Cache, Config};

/// RMSNorm with multiplicative SiLU gating: rms_norm(x) * silu(z).
/// Uses candle's fused rms_norm kernel (1 kernel) instead of 7 separate ops.
#[derive(Debug, Clone)]
struct RmsNormGated {
    weight: Tensor,
    eps: f32,
    backend: Arc<dyn ComputeBackend>,
}

impl RmsNormGated {
    fn load(size: usize, eps: f64, vb: VarBuilder, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        // Store weight as F32 to match the recurrent step's F32 output.
        let weight = vb.get(size, "weight")?.to_dtype(DType::F32)?;
        Ok(Self { weight, eps: eps as f32, backend })
    }

    /// Apply gated RMS normalization: weight * rms_norm(x) * silu(z).
    /// Uses fused kernel (1 launch) instead of rms_norm + silu + mul (3 launches).
    fn forward(&self, x: &Tensor, z: &Tensor) -> Result<Tensor> {
        let z = z.to_dtype(x.dtype())?;
        self.backend.rms_norm_gated(
            &x.contiguous()?,
            &z.contiguous()?,
            &self.weight,
            self.eps,
        )
    }
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

    // Fused L2-norm alpha vectors: rms_norm(x, alpha, eps/N) = L2_normalize(x) * scale
    // Q alpha includes q_scale: 1/(key_head_dim) = 1/sqrt(N) * 1/sqrt(N)
    // K alpha is plain: 1/sqrt(key_head_dim)
    l2_alpha_q: Tensor,
    l2_alpha_k: Tensor,
    l2_norm_eps: f32,

    num_heads: usize,
    num_key_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,

    // Split offsets for the fused projection output
    conv_dim: usize,

    backend: Arc<dyn ComputeBackend>,
}

impl GatedDeltaNet {
    pub fn load(vb: VarBuilder, cfg: &Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
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

        // Absorb dt_bias into the projection bias so `a` output already includes it.
        // Bias layout: [zeros for QKV | dt_bias for A | zeros for B | zeros for Z]
        let dt_bias = vb.get(num_heads, "dt_bias")?;
        let total_out = conv_dim + num_heads + num_heads + value_dim;
        let dt_bias_vec = dt_bias.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut bias_data = vec![0.0f32; total_out];
        bias_data[conv_dim..(num_heads + conv_dim)].copy_from_slice(&dt_bias_vec[..num_heads]);
        let dev_for_bias = fused_w.device().clone();
        let fused_bias = Tensor::from_slice(&bias_data, total_out, &dev_for_bias)?
            .to_dtype(fused_w.dtype())?;
        let in_proj = Linear::new(fused_w, Some(fused_bias));

        let out_w = vb.pp("out_proj").get((h_size, value_dim), "weight")?;
        let out_proj = Linear::new(out_w, None);

        // Conv1d weight: stored as F32 (matches post-projection F32 data path).
        let conv1d_weight = vb.get((conv_dim, 1, la.conv_kernel_dim), "conv1d.weight")?
            .to_dtype(DType::F32)?
            .squeeze(1)?;

        let a_log = vb.get(num_heads, "A_log")?;

        // Precompute constants in F32 to avoid per-call recomputation
        let neg_a_exp_f32 = a_log.to_dtype(DType::F32)?.exp()?.neg()?;

        // Precompute alpha vectors for fused L2 normalization via rms_norm.
        // rms_norm(x, alpha, eps/N) = x / sqrt(mean(x²) + eps/N) * alpha
        //                            = x / sqrt((sum(x²)+eps)/N) * alpha
        //                            = x * sqrt(N) / sqrt(sum(x²)+eps) * alpha
        // Setting alpha = 1/sqrt(N): gives L2_normalize(x)
        // Setting alpha = 1/N:       gives L2_normalize(x) * 1/sqrt(N) = L2_normalize(x) * q_scale
        let dev = neg_a_exp_f32.device().clone();
        let inv_sqrt_dim = 1.0f32 / (key_head_dim as f32).sqrt();
        let inv_dim = inv_sqrt_dim * inv_sqrt_dim; // 1/N
        let l2_alpha_q = Tensor::from_slice(
            &vec![inv_dim; key_head_dim], key_head_dim, &dev,
        )?;
        let l2_alpha_k = Tensor::from_slice(
            &vec![inv_sqrt_dim; key_head_dim], key_head_dim, &dev,
        )?;
        let l2_norm_eps = 1e-6f32 / key_head_dim as f32;

        let norm = RmsNormGated::load(value_head_dim, cfg.rms_norm_eps, vb.pp("norm"), backend.clone())?;

        Ok(Self {
            in_proj,
            conv1d_weight,
            norm,
            out_proj,
            neg_a_exp_f32,
            l2_alpha_q,
            l2_alpha_k,
            l2_norm_eps,
            num_heads,
            num_key_heads,
            key_head_dim,
            value_head_dim,
            key_dim,
            value_dim,
            conv_kernel_size: la.conv_kernel_dim,
            conv_dim,
            backend,
        })
    }

    /// Apply causal depthwise conv1d + SiLU on a single new token, updating conv_state in-place.
    /// x: (batch, 1, channels), conv_state: (batch, channels, kernel_size-1)
    fn causal_conv1d_step(&self, x: &Tensor, conv_state: &Tensor) -> Result<(Tensor, Tensor)> {
        // x is (batch, 1, channels) -> (batch, channels, 1)
        // squeeze+unsqueeze preserves contiguity (transpose would trigger a copy kernel)
        let x_t = x.squeeze(1)?.unsqueeze(2)?;

        // Build full window: cat(state, new_input) -> (batch, channels, kernel_size)
        let full_window = Tensor::cat(&[conv_state, &x_t], 2)?;

        // Fused depthwise conv + silu: 1 kernel instead of 3 (broadcast_mul + sum + silu)
        // conv1d_weight: (channels, kernel_size)
        // full_window: (batch, channels, kernel_size)
        // Result: (batch, channels) -> unsqueeze to (batch, 1, channels)
        let y = self.backend.depthwise_conv1d_silu(
            &full_window.contiguous()?,
            &self.conv1d_weight,
            self.conv_kernel_size,
            self.conv_dim,
        )?
        .unsqueeze(1)?;

        // New state: last (kernel_size-1) elements of full_window
        let new_state = full_window.narrow(2, 1, self.conv_kernel_size - 1)?;

        Ok((y, new_state))
    }

    /// Apply causal depthwise conv1d + SiLU on a full sequence.
    /// Uses vectorized unfold to process all timesteps in ~5 kernel calls
    /// instead of 3*N calls from a per-timestep loop.
    /// x: (batch, seq_len, channels)
    fn causal_conv1d_seq(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, channels) = x.dims3()?;
        let kernel_size = self.conv_kernel_size;
        let pad = kernel_size - 1;

        // Transpose to (batch, channels, seq_len) for windowed access
        let x_t = x.transpose(1, 2)?;

        // Left-pad with zeros for causal conv
        let padded = if pad > 0 {
            let padding = Tensor::zeros((batch, channels, pad), x_t.dtype(), x_t.device())?;
            Tensor::cat(&[&padding, &x_t.contiguous()?], 2)?
        } else {
            x_t.contiguous()?
        };

        // Vectorized depthwise conv1d via unfold:
        // unfold creates all sliding windows as a strided view (zero-copy),
        // then a single broadcast_mul+sum processes all timesteps at once.
        // (batch, channels, seq_len+pad) -> (batch, channels, seq_len, kernel_size)
        let unfolded = padded.unfold(2, kernel_size, 1)?.contiguous()?;
        // weight: (channels, kernel_size) -> (1, channels, 1, kernel_size)
        let weight = self.conv1d_weight.unsqueeze(0)?.unsqueeze(2)?;
        let y = unfolded.broadcast_mul(&weight)?.sum(D::Minus1)?;
        // (batch, channels, seq_len) -> (batch, seq_len, channels)
        let y = y.transpose(1, 2)?.contiguous()?;

        // SiLU activation
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

        // Periodic GPU command buffer flushes prevent catastrophic slowdown from
        // command accumulation on Metal. Each GDN layer dispatches ~40 commands;
        // without intermediate syncs, performance degrades by 50x+. Four sync points
        // (after in_proj, after conv1d, after recurrent, after out_proj) keep each
        // section under ~25 commands. No-op on CPU/CUDA.

        // Single fused projection: QKV + A + B + Z (with dt_bias absorbed into A bias)
        let proj = self.in_proj.forward(x)
            .map_err(|e| anyhow!("in_proj: {e}"))?;

        // Bulk F32 conversion: one kernel instead of 5 individual to_dtype calls.
        // Everything downstream (conv1d, L2 norm, gates, recurrent step) operates in F32.
        let proj = proj.to_dtype(DType::F32)
            .map_err(|e| anyhow!("proj to_f32: {e}"))?;

        // Split fused output (all views on F32 tensor — zero cost)
        // Note: sync deferred to after conv1d to reduce sync overhead
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

        // Flush GPU commands after conv1d — needed for prefill (many ops in conv_seq),
        // skip for generation (seq_len=1) since conv_step is just ~5 commands
        if seq_len > 1 {
            let _ = self.backend.synchronize();
        }

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

        // L2-normalize Q and K using fused rms_norm kernel (1 kernel each).
        // Q and K are already F32 from bulk conversion — no individual to_dtype needed.
        // rms_norm(x, alpha=1/N, eps/N) = L2_normalize(x) * q_scale (for Q)
        // rms_norm(x, alpha=1/√N, eps/N) = L2_normalize(x) (for K)
        let q = candle_nn::ops::rms_norm(&q.contiguous()?, &self.l2_alpha_q, self.l2_norm_eps)
            .map_err(|e| anyhow!("q l2norm: {e}"))?;
        let k = candle_nn::ops::rms_norm(&k.contiguous()?, &self.l2_alpha_k, self.l2_norm_eps)
            .map_err(|e| anyhow!("k l2norm: {e}"))?;
        // v is already F32 from bulk conversion

        // Compute gating parameters — a already includes dt_bias (absorbed into projection bias),
        // already F32 from bulk conversion. Saves 2 kernels (to_dtype + broadcast_add).
        let softplus_a = self.backend.stable_softplus(&a.contiguous()?)?;
        let g = softplus_a.broadcast_mul(&self.neg_a_exp_f32)
            .map_err(|e| anyhow!("g compute: {e}"))?;

        // beta = sigmoid(b) — already F32 from bulk conversion
        let beta = candle_nn::ops::sigmoid(&b)
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

        // Save updated state (sync deferred to after out_proj)
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

        // Flush Metal commands after the conv+recurrent+norm+out_proj section
        let _ = self.backend.synchronize();

        Ok(output)
    }
}
