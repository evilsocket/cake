//! Full (softmax) attention for Qwen3.5 with partial RoPE, Q/K norms, and output gating.

use candle_core::{Result, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, RmsNorm, VarBuilder};

use crate::models::common::{Cache, Config};

#[inline]
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

/// Full attention with Q/K RMS norms, partial RoPE, and output gating via Q projection.
#[derive(Debug, Clone)]
pub struct Qwen3_5FullAttention {
    q_proj: Linear, // projects to num_heads * head_dim * 2 (query + gate)
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
}

impl Qwen3_5FullAttention {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize;

        // Q projects to double size for output gating
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim * 2,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )?;

        let q_norm = crate::models::common::load_rms_norm(
            head_dim, cfg.rms_norm_eps, cfg.residual_rms_norm, vb.pp("q_norm"),
        )?;
        let k_norm = crate::models::common::load_rms_norm(
            head_dim, cfg.rms_norm_eps, cfg.residual_rms_norm, vb.pp("k_norm"),
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            rotary_dim,
        })
    }

    /// Apply partial RoPE: rotate only the first rotary_dim dimensions.
    fn apply_partial_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &Cache,
    ) -> Result<Tensor> {
        let (_batch, _heads, seq_len, _dim) = x.dims4()?;
        let cos = cache.cosine(index_pos, seq_len, x.device())?;
        let sin = cache.sine(index_pos, seq_len, x.device())?;

        if self.rotary_dim == self.head_dim {
            // Full rotation
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        } else {
            // Partial: split at rotary_dim, rotate first part, concat back
            let x_rot = x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let x_pass = x.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?.contiguous()?;

            let x_rot = candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&x_rot, &x_pass], D::Minus1)
        }
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }

    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> anyhow::Result<Tensor> {
        let (b_sz, seq_len, _hidden) = x.dims3().map_err(|e| anyhow!("dims3: {e}"))?;
        let hidden_size = self.num_attention_heads * self.head_dim;

        // Q projection: output is doubled for gating
        let q_out = self.q_proj.forward(x)
            .map_err(|e| anyhow!("q_proj: {e}"))?;
        // Reshape to (batch, seq, num_heads, head_dim * 2), split into query and gate
        let q_out = q_out.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim * 2))
            .map_err(|e| anyhow!("q reshape: {e}"))?;
        let query = q_out.narrow(D::Minus1, 0, self.head_dim)
            .map_err(|e| anyhow!("q narrow: {e}"))?;
        let gate = q_out.narrow(D::Minus1, self.head_dim, self.head_dim)
            .map_err(|e| anyhow!("gate narrow: {e}"))?;
        // Flatten gate for later: (batch, seq, num_heads * head_dim)
        let gate = gate.reshape((b_sz, seq_len, hidden_size))
            .map_err(|e| anyhow!("gate reshape: {e}"))?;

        // K, V projections
        let k = self.k_proj.forward(x).map_err(|e| anyhow!("k_proj: {e}"))?;
        let v = self.v_proj.forward(x).map_err(|e| anyhow!("v_proj: {e}"))?;

        // Reshape K, V
        let k = k.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(|e| anyhow!("k reshape: {e}"))?;
        let v = v.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(|e| anyhow!("v reshape: {e}"))?;

        // Apply Q/K RMS norms (per-head)
        let q = self.q_norm.forward(&query).map_err(|e| anyhow!("q_norm: {e}"))?;
        let k = self.k_norm.forward(&k).map_err(|e| anyhow!("k_norm: {e}"))?;

        // Transpose to (batch, heads, seq, head_dim) for attention
        let q = q.transpose(1, 2)?.contiguous().map_err(|e| anyhow!("q transpose: {e}"))?;
        let k = k.transpose(1, 2)?.contiguous().map_err(|e| anyhow!("k transpose: {e}"))?;
        let v = v.transpose(1, 2).map_err(|e| anyhow!("v transpose: {e}"))?;

        // Apply partial RoPE
        let q = self.apply_partial_rotary_emb(&q, index_pos, cache)
            .map_err(|e| anyhow!("q rope: {e}"))?;
        let k = self.apply_partial_rotary_emb(&k, index_pos, cache)
            .map_err(|e| anyhow!("k rope: {e}"))?;

        // KV cache
        let (k, v) = cache.process_kv(block_idx, k, v)
            .map_err(|e| anyhow!("process_kv: {e}"))?;

        // Attention
        #[allow(unused_labels)]
        let y = 'attn: {
            // Fused SDPA on Metal — single kernel, native GQA (no repeat_kv needed)
            #[cfg(feature = "metal")]
            if matches!(q.device(), candle_core::Device::Metal(_)) {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                break 'attn candle_nn::ops::sdpa(&q, &k, &v, None, seq_len > 1, scale, 1.0)
                    .map_err(|e| anyhow!("sdpa: {e}"))?;
            }

            // Manual attention with GQA head expansion (CUDA, CPU)
            let k = self.repeat_kv(k).map_err(|e| anyhow!("repeat_kv k: {e}"))?;
            let v = self.repeat_kv(v).map_err(|e| anyhow!("repeat_kv v: {e}"))?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                let mask = cache.mask(seq_len, att.device())
                    .map_err(|e| anyhow!("mask: {e}"))?
                    .broadcast_as(att.shape())
                    .map_err(|e| anyhow!("mask broadcast: {e}"))?;
                masked_fill(&att, &mask, f32::NEG_INFINITY)
                    .map_err(|e| anyhow!("masked_fill: {e}"))?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        // Reshape: (batch, heads, seq, head_dim) -> (batch, seq, hidden_size)
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])
            .map_err(|e| anyhow!("y reshape: {e}"))?;

        // Output gating: y * sigmoid(gate)
        let gate = candle_nn::ops::sigmoid(&gate).map_err(|e| anyhow!("sigmoid gate: {e}"))?;
        let y = (y * gate).map_err(|e| anyhow!("gating: {e}"))?;

        // Final projection
        let y = self.o_proj.forward(&y).map_err(|e| anyhow!("o_proj: {e}"))?;
        Ok(y)
    }
}
