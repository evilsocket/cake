//! Full (softmax) attention for Qwen3.5 with fused QKV projection,
//! partial RoPE, Q/K norms, and output gating.

use std::sync::Arc;

use candle_core::{Result, Tensor, D};
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;
use crate::models::common::{load_rms_norm_weight, Cache, Config};

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
    qkv_proj_weight: Tensor, // fused: Q (num_heads * head_dim * 2) + K + V
    o_proj_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    qk_norm_eps: f32,

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

        // Load QKV projections: either fused (qkv_proj) or split (q_proj + k_proj + v_proj).
        // The 0.8B model may have a fused qkv_proj; the 35B model has separate projections.
        let total_qkv = q_size + kv_size + kv_size;
        let qkv_proj_weight = if vb.pp("qkv_proj").contains_tensor("weight") {
            vb.pp("qkv_proj").get((total_qkv, h_size), "weight")?
        } else {
            let q_w = vb.pp("q_proj").get((q_size, h_size), "weight")?;
            let k_w = vb.pp("k_proj").get((kv_size, h_size), "weight")?;
            let v_w = vb.pp("v_proj").get((kv_size, h_size), "weight")?;
            Tensor::cat(&[&q_w, &k_w, &v_w], 0)?
        };
        let qkv_proj_weight = backend.preprocess_linear_weight(&qkv_proj_weight)?;

        let o_size = cfg.num_attention_heads * head_dim;
        let o_proj_weight = backend.preprocess_linear_weight(
            &vb.pp("o_proj").get((h_size, o_size), "weight")?,
        )?;

        let q_norm_weight = load_rms_norm_weight(
            head_dim, cfg.residual_rms_norm, vb.pp("q_norm"),
        )?;
        let k_norm_weight = load_rms_norm_weight(
            head_dim, cfg.residual_rms_norm, vb.pp("k_norm"),
        )?;
        let qk_norm_eps = cfg.rms_norm_eps as f32;

        Ok(Self {
            qkv_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
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
            self.backend.rope(x, &cos, &sin)
        } else {
            // Partial: split at rotary_dim, rotate first part, concat back
            let x_rot = x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let x_pass = x.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?.contiguous()?;

            let x_rot = self.backend.rope(&x_rot, &cos, &sin)?;
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
        let qkv = self.backend.linear_forward(x, &self.qkv_proj_weight, None)
            .map_err(|e| anyhow!("qkv_proj: {e}"))?;

        // Flush GPU commands after QKV matmul — needed for prefill where many
        // operations follow. Generation (seq_len=1) uses fused SDPA with few commands.
        if seq_len > 1 {
            let _ = self.backend.synchronize();
        }

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
        let q = self.backend.rms_norm(&query.contiguous()
            .map_err(|e| anyhow!("q contiguous: {e}"))?, &self.q_norm_weight, self.qk_norm_eps)
            .map_err(|e| anyhow!("q_norm: {e}"))?;
        let k = self.backend.rms_norm(&k.contiguous()
            .map_err(|e| anyhow!("k contiguous: {e}"))?, &self.k_norm_weight, self.qk_norm_eps)
            .map_err(|e| anyhow!("k_norm: {e}"))?;

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

            // Metal path: fused SDPA for generation, mixed-precision for prefill.
            #[cfg(feature = "metal")]
            if matches!(q.device(), candle_core::Device::Metal(_)) {
                // Generation (seq_len=1): fused kernel — single dispatch with native
                // GQA (no repeat_kv), online softmax, no attention matrix materialization.
                // Replaces 4+ separate dispatches (repeat_kv + 2 matmuls + softmax + dtype casts).
                if seq_len == 1 {
                    let scale = 1.0 / (self.head_dim as f32).sqrt();
                    break 'attn self.backend.sdpa(&q, &k, &v, None, false, scale)
                        .map_err(|e| anyhow!("sdpa: {e}"))?;
                }

                // Prefill (seq_len > 1): F16 matmuls + F32 softmax (F16 SDPA causes
                // garbage, F32 SDPA exceeds threadgroup memory).
                let k = self.repeat_kv(k).map_err(|e| anyhow!("repeat_kv k: {e}"))?;
                let v = self.repeat_kv(v).map_err(|e| anyhow!("repeat_kv v: {e}"))?;
                let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
                let att = att.to_dtype(candle_core::DType::F32)?;
                let tril = Tensor::tril2(seq_len, candle_core::DType::F32, att.device())
                    .map_err(|e| anyhow!("tril: {e}"))?;
                let mask = ((tril - 1.0)? * 1e9)?;
                let mask = mask.broadcast_as(att.shape())
                    .map_err(|e| anyhow!("mask broadcast: {e}"))?;
                let att = (att + mask).map_err(|e| anyhow!("mask add: {e}"))?;
                let att = self.backend.softmax(&att, att.rank() - 1)?;
                let att = att.to_dtype(v.dtype())?;
                break 'attn att.matmul(&v.contiguous()?)
                    .map_err(|e| anyhow!("att matmul v: {e}"))?;
            }

            // CPU: manual attention with GQA head expansion
            let k = self.repeat_kv(k).map_err(|e| anyhow!("repeat_kv k: {e}"))?;
            let v = self.repeat_kv(v).map_err(|e| anyhow!("repeat_kv v: {e}"))?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = cache.mask(seq_len, att.device())
                .map_err(|e| anyhow!("mask: {e}"))?
                .broadcast_as(att.shape())
                .map_err(|e| anyhow!("mask broadcast: {e}"))?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)
                .map_err(|e| anyhow!("masked_fill: {e}"))?;
            let att = self.backend.softmax(&att, att.rank() - 1)?;
            att.matmul(&v.contiguous()?)?
        };

        // Reshape: (batch, heads, seq, head_dim) -> (batch, seq, hidden_size)
        // Convert back to model dtype for gating and output projection
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])
            .map_err(|e| anyhow!("y reshape: {e}"))?
            .to_dtype(x.dtype())
            .map_err(|e| anyhow!("y to_dtype: {e}"))?;

        // Output gating: y * sigmoid(gate)
        let gate = self.backend.sigmoid(&gate).map_err(|e| anyhow!("sigmoid gate: {e}"))?;
        let y = (y * gate).map_err(|e| anyhow!("gating: {e}"))?;

        // Final projection
        let y = self.backend.linear_forward(&y, &self.o_proj_weight, None)
            .map_err(|e| anyhow!("o_proj: {e}"))?;

        // Flush GPU commands — needed for prefill, skip for generation
        if seq_len > 1 {
            let _ = self.backend.synchronize();
        }

        Ok(y)
    }
}
