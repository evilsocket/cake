//! Full (softmax) attention for Qwen3.5 with fused QKV projection,
//! partial RoPE, Q/K norms, and output gating.

use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::backends::ComputeBackend;
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

/// Full attention with fused QKV, Q/K RMS norms, partial RoPE, and output gating.
#[derive(Debug, Clone)]
pub struct Qwen3_5FullAttention {
    qkv_proj: Linear, // fused: Q (num_heads * head_dim * 2) + K + V
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rotary_dim: usize,

    // Split offsets for the fused projection
    q_size: usize,  // num_heads * head_dim * 2 (includes gate)
    kv_size: usize,  // num_kv_heads * head_dim

    backend: Arc<dyn ComputeBackend>,
}

impl Qwen3_5FullAttention {
    pub fn load(vb: VarBuilder, cfg: &Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize;
        let h_size = cfg.hidden_size;

        let q_size = cfg.num_attention_heads * head_dim * 2;
        let kv_size = cfg.num_key_value_heads * head_dim;

        // Fuse Q, K, V projections into a single Linear
        let q_w = vb.pp("q_proj").get((q_size, h_size), "weight")?;
        let k_w = vb.pp("k_proj").get((kv_size, h_size), "weight")?;
        let v_w = vb.pp("v_proj").get((kv_size, h_size), "weight")?;
        let fused_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
        let qkv_proj = Linear::new(fused_w, None);

        let o_size = cfg.num_attention_heads * head_dim;
        let o_w = vb.pp("o_proj").get((h_size, o_size), "weight")?;
        let o_proj = Linear::new(o_w, None);

        let q_norm = crate::models::common::load_rms_norm(
            head_dim, cfg.rms_norm_eps, cfg.residual_rms_norm, vb.pp("q_norm"),
        )?;
        let k_norm = crate::models::common::load_rms_norm(
            head_dim, cfg.rms_norm_eps, cfg.residual_rms_norm, vb.pp("k_norm"),
        )?;

        Ok(Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            rotary_dim,
            q_size,
            kv_size,
            backend,
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

        // Periodic GPU command buffer flushes (see linear_attention.rs).

        // Single fused QKV projection
        let qkv = self.qkv_proj.forward(x)
            .map_err(|e| anyhow!("qkv_proj: {e}"))?;

        // Flush GPU commands after QKV matmul (always needed — full attention
        // accumulates ~24 commands between syncs, can't afford more)
        let _ = self.backend.synchronize();

        // Split: Q (doubled for gating), K, V
        let q_out = qkv.narrow(D::Minus1, 0, self.q_size)
            .map_err(|e| anyhow!("q split: {e}"))?;
        let k = qkv.narrow(D::Minus1, self.q_size, self.kv_size)
            .map_err(|e| anyhow!("k split: {e}"))?;
        let v = qkv.narrow(D::Minus1, self.q_size + self.kv_size, self.kv_size)
            .map_err(|e| anyhow!("v split: {e}"))?;

        // Reshape Q to (batch, seq, num_heads, head_dim * 2), split into query and gate
        let q_out = q_out.reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim * 2))
            .map_err(|e| anyhow!("q reshape: {e}"))?;
        let query = q_out.narrow(D::Minus1, 0, self.head_dim)
            .map_err(|e| anyhow!("q narrow: {e}"))?;
        let gate = q_out.narrow(D::Minus1, self.head_dim, self.head_dim)
            .map_err(|e| anyhow!("gate narrow: {e}"))?;
        // Flatten gate for later: (batch, seq, num_heads * head_dim)
        let gate = gate.reshape((b_sz, seq_len, hidden_size))
            .map_err(|e| anyhow!("gate reshape: {e}"))?;

        // Reshape K, V
        let k = k.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(|e| anyhow!("k reshape: {e}"))?;
        let v = v.reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))
            .map_err(|e| anyhow!("v reshape: {e}"))?;

        // Apply Q/K RMS norms (per-head)
        let q = self.q_norm.forward(&query).map_err(|e| anyhow!("q_norm: {e}"))?;
        let k = self.k_norm.forward(&k).map_err(|e| anyhow!("k_norm: {e}"))?;

        // Transpose to (batch, heads, seq, head_dim) for attention.
        // For seq_len=1, squeeze+unsqueeze is zero-copy (avoids contiguous copy kernel)
        // because removing a size-1 dim preserves contiguity.
        let (q, k, v) = if seq_len == 1 {
            (q.squeeze(1)?.unsqueeze(2)?, k.squeeze(1)?.unsqueeze(2)?, v.squeeze(1)?.unsqueeze(2)?)
        } else {
            (q.transpose(1, 2)?.contiguous()?, k.transpose(1, 2)?.contiguous()?, v.transpose(1, 2)?)
        };

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
            // Flash Attention on CUDA — fused kernel, native GQA (no repeat_kv needed)
            #[cfg(feature = "flash-attn")]
            if matches!(q.device(), candle_core::Device::Cuda(_)) {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                break 'attn crate::utils::flash_attn::flash_attention(
                    &q, &k, &v, scale, seq_len > 1,
                ).map_err(|e| anyhow!("flash_attn: {e}"))?;
            }

            // Metal: mixed-precision attention — F16 matmuls with F32 softmax.
            // F16 SDPA causes garbage (precision loss in softmax), F32 SDPA exceeds
            // threadgroup memory. This hybrid keeps F16 matmul speed + F32 softmax precision.
            #[cfg(feature = "metal")]
            if matches!(q.device(), candle_core::Device::Metal(_)) {
                let k = self.repeat_kv(k).map_err(|e| anyhow!("repeat_kv k: {e}"))?;
                let v = self.repeat_kv(v).map_err(|e| anyhow!("repeat_kv v: {e}"))?;
                // QK^T in native dtype (F16) for fast matmul
                let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                // Softmax in F32 for precision (the critical part)
                let att = att.to_dtype(candle_core::DType::F32)?;
                let att = if seq_len == 1 {
                    att
                } else {
                    let tril = Tensor::tril2(seq_len, candle_core::DType::F32, att.device())
                        .map_err(|e| anyhow!("tril: {e}"))?;
                    let mask = ((tril - 1.0)? * 1e9)?;
                    let mask = mask.broadcast_as(att.shape())
                        .map_err(|e| anyhow!("mask broadcast: {e}"))?;
                    (att + mask).map_err(|e| anyhow!("mask add: {e}"))?
                };
                let att = candle_nn::ops::softmax_last_dim(&att)?;
                // Att @ V: convert att back to input dtype for F16 matmul
                let att = att.to_dtype(v.dtype())?;
                break 'attn att.matmul(&v.contiguous()?)
                    .map_err(|e| anyhow!("att matmul v: {e}"))?;
            }

            // Manual attention with GQA head expansion (CPU fallback)
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
        // Convert back to model dtype for gating and output projection
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])
            .map_err(|e| anyhow!("y reshape: {e}"))?
            .to_dtype(x.dtype())
            .map_err(|e| anyhow!("y to_dtype: {e}"))?;

        // Output gating: y * sigmoid(gate)
        let gate = candle_nn::ops::sigmoid(&gate).map_err(|e| anyhow!("sigmoid gate: {e}"))?;
        let y = (y * gate).map_err(|e| anyhow!("gating: {e}"))?;

        // Final projection
        let y = self.o_proj.forward(&y).map_err(|e| anyhow!("o_proj: {e}"))?;

        // Flush GPU commands — needed for prefill, skip for generation
        if seq_len > 1 {
            let _ = self.backend.synchronize();
        }

        Ok(y)
    }
}
