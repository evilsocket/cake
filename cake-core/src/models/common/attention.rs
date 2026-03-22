//! Causal self attention implementation with fused QKV projection.
use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder};

use crate::backends::ComputeBackend;
use super::config::load_rms_norm;

#[derive(Debug, Clone)]
pub struct CausalSelfAttention {
    qkv_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    /// Number of head dimensions that receive RoPE. Equal to head_dim unless partial_rotary_factor < 1.
    rotary_dim: usize,
    size_q: usize,
    size_kv: usize,
    /// Optional RmsNorm applied to Q (after reshape unless pre_reshape_qk_norm).
    q_norm: Option<RmsNorm>,
    /// Optional RmsNorm applied to K (after reshape unless pre_reshape_qk_norm).
    k_norm: Option<RmsNorm>,
    /// When true, QK-norm is applied before the head reshape (OLMo2-style, weight size = size_q/size_kv).
    /// When false (default), applied after reshape per head (weight size = head_dim).
    pre_reshape_qk_norm: bool,
    /// Sliding window size for local attention (None = full context, e.g. Mistral, Gemma3 local).
    sliding_window: Option<usize>,
    /// Whether to apply Rotary Position Embeddings. False for Gemma3 local layers.
    use_rope: bool,
    /// Compute backend for routing matmuls through GPU-accelerated paths.
    backend: Arc<dyn ComputeBackend>,
}

#[inline]
fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?
        .to_dtype(on_false.dtype())?
        .broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl CausalSelfAttention {
    /// Linear forward: x @ w^T + bias via backend matmul.
    /// Handles dtype promotion when weights are pre-converted to F32.
    fn linear_forward(&self, x: &Tensor, linear: &Linear) -> Result<Tensor> {
        let w = linear.weight().t()?;
        let x_matched = x.to_dtype(w.dtype())?;
        let x_dims = x.dims();
        let out = if x_dims.len() > 2 {
            let leading: usize = x_dims[..x_dims.len() - 1].iter().product();
            let inner = *x_dims.last().unwrap();
            let x_2d = x_matched.reshape((leading, inner))?;
            let out_2d = self.backend.matmul(&x_2d, &w)?;
            let out_dim = out_2d.dim(1)?;
            let mut out_shape = x_dims[..x_dims.len() - 1].to_vec();
            out_shape.push(out_dim);
            out_2d.reshape(out_shape.as_slice())?
        } else {
            self.backend.matmul(&x_matched, &w)?
        };
        let out = out.to_dtype(x.dtype())?;
        match linear.bias() {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }

    fn apply_rotary_emb(
        &self,
        x: &Tensor,
        index_pos: usize,
        cache: &super::Cache,
    ) -> Result<Tensor> {
        let (_batch_size, _, seq_len, _hidden_size) = x.dims4()?;
        let cos = cache.cosine(index_pos, seq_len, x.device())?;
        let sin = cache.sine(index_pos, seq_len, x.device())?;
        if self.rotary_dim == self.head_dim {
            candle_nn::rotary_emb::rope(x, &cos, &sin)
        } else {
            // Partial RoPE: apply only to first rotary_dim channels, pass the rest through.
            let x_rot = x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?;
            let x_pass = x.narrow(D::Minus1, self.rotary_dim, self.head_dim - self.rotary_dim)?.contiguous()?;
            let x_rot = candle_nn::rotary_emb::rope(&x_rot, &cos, &sin)?;
            Tensor::cat(&[&x_rot, &x_pass], D::Minus1)
        }
    }

    /// Standard load — derives all flags from `cfg`.
    pub fn load(vb: VarBuilder, cfg: &super::Config, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        Self::load_custom(vb, cfg, cfg.use_qk_norm, cfg.sliding_window, true, backend)
    }

    /// Custom load with explicit per-layer options (used by Gemma3 for interleaved local/global).
    pub fn load_custom(
        vb: VarBuilder,
        cfg: &super::Config,
        use_qk_norm: bool,
        sliding_window: Option<usize>,
        use_rope: bool,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        let size_in = cfg.hidden_size;
        let head_dim = cfg.head_dim.unwrap_or(cfg.hidden_size / cfg.num_attention_heads);
        let rotary_dim = (head_dim as f32 * cfg.partial_rotary_factor) as usize;
        let size_q = head_dim * cfg.num_attention_heads;
        let size_kv = head_dim * cfg.num_key_value_heads;

        let qkv_proj = if cfg.fused_qkv_proj {
            // Phi-3/4 style: weights already fused as a single 'qkv_proj' tensor.
            let w = vb.pp("qkv_proj").get((size_q + 2 * size_kv, size_in), "weight")?;
            let w = backend.preprocess_linear_weight(&w)?;
            Linear::new(w, None)
        } else if cfg.use_qkv_bias {
            let q_w = vb.pp("q_proj").get((size_q, size_in), "weight")?;
            let k_w = vb.pp("k_proj").get((size_kv, size_in), "weight")?;
            let v_w = vb.pp("v_proj").get((size_kv, size_in), "weight")?;
            let fused_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
            let fused_w = backend.preprocess_linear_weight(&fused_w)?;

            let q_b = vb.pp("q_proj").get(size_q, "bias")?;
            let k_b = vb.pp("k_proj").get(size_kv, "bias")?;
            let v_b = vb.pp("v_proj").get(size_kv, "bias")?;
            let fused_b = Tensor::cat(&[&q_b, &k_b, &v_b], 0)?;

            Linear::new(fused_w, Some(fused_b))
        } else {
            let q_w = vb.pp("q_proj").get((size_q, size_in), "weight")?;
            let k_w = vb.pp("k_proj").get((size_kv, size_in), "weight")?;
            let v_w = vb.pp("v_proj").get((size_kv, size_in), "weight")?;
            let fused_w = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;
            let fused_w = backend.preprocess_linear_weight(&fused_w)?;
            Linear::new(fused_w, None)
        };

        let o_w = vb.pp("o_proj").get((size_in, size_q), "weight")?;
        let o_w = backend.preprocess_linear_weight(&o_w)?;
        let o_proj = Linear::new(o_w, None);

        let (q_norm, k_norm) = if use_qk_norm {
            let eps = cfg.rms_norm_eps;
            let norm_dim = if cfg.pre_reshape_qk_norm { size_q } else { head_dim };
            let norm_kv_dim = if cfg.pre_reshape_qk_norm { size_kv } else { head_dim };
            // Gemma3 QK-norms use GemmaRMSNorm: weight stored as delta, apply (1+weight).
            let residual = cfg.residual_rms_norm;
            let qn = load_rms_norm(norm_dim, eps, residual, vb.pp("q_norm"))?;
            let kn = load_rms_norm(norm_kv_dim, eps, residual, vb.pp("k_norm"))?;
            (Some(qn), Some(kn))
        } else {
            (None, None)
        };

        Ok(Self {
            qkv_proj,
            o_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_key_value_heads,
            head_dim,
            rotary_dim,
            size_q,
            size_kv,
            q_norm,
            k_norm,
            pre_reshape_qk_norm: cfg.pre_reshape_qk_norm,
            sliding_window,
            use_rope,
            backend,
        })
    }

    /// Process the input tensor using the given state indexes and cache.
    pub fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut super::Cache,
    ) -> anyhow::Result<Tensor> {
        let (b_sz, seq_len, _hidden_size) = x.dims3().map_err(|e| anyhow!("x.dims3 -> {e}"))?;

        // Single fused QKV projection (routed through backend for GPU acceleration)
        let qkv = self
            .linear_forward(x, &self.qkv_proj)
            .map_err(|e| anyhow!("qkv.forward -> {e}"))?;

        let q = qkv
            .narrow(D::Minus1, 0, self.size_q)
            .map_err(|e| anyhow!("q split -> {e}"))?;
        let k = qkv
            .narrow(D::Minus1, self.size_q, self.size_kv)
            .map_err(|e| anyhow!("k split -> {e}"))?;
        let v = qkv
            .narrow(D::Minus1, self.size_q + self.size_kv, self.size_kv)
            .map_err(|e| anyhow!("v split -> {e}"))?;

        // OLMo2-style: apply QK-norm BEFORE head reshape (norm dim = size_q/size_kv).
        let (q, k) = if self.pre_reshape_qk_norm {
            let q = if let Some(norm) = &self.q_norm {
                norm.forward(&q).map_err(|e| anyhow!("pre_reshape q_norm -> {e}"))?
            } else { q };
            let k = if let Some(norm) = &self.k_norm {
                norm.forward(&k).map_err(|e| anyhow!("pre_reshape k_norm -> {e}"))?
            } else { k };
            (q, k)
        } else {
            (q, k)
        };

        // Reshape: (b, seq, heads, head_dim)
        let q = q
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?;
        let k = k
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?;
        let v = v
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?;

        // Standard QK-norm: applied after reshape (on head_dim, last dim) before transpose.
        let q = if !self.pre_reshape_qk_norm {
            if let Some(norm) = &self.q_norm {
                norm.forward(&q).map_err(|e| anyhow!("q_norm -> {e}"))?
            } else { q }
        } else { q };
        let k = if !self.pre_reshape_qk_norm {
            if let Some(norm) = &self.k_norm {
                norm.forward(&k).map_err(|e| anyhow!("k_norm -> {e}"))?
            } else { k }
        } else { k };

        // Transpose to (b, heads, seq, head_dim).
        // For generation (seq_len=1), squeeze+unsqueeze avoids the contiguous
        // copy that transpose triggers — the memory layout is already correct
        // when the swapped dimension has size 1.
        let (q, k, v) = if seq_len == 1 {
            (
                q.squeeze(1)?.unsqueeze(2)
                    .map_err(|e| anyhow!("q.squeeze/unsqueeze -> {e}"))?,
                k.squeeze(1)?.unsqueeze(2)
                    .map_err(|e| anyhow!("k.squeeze/unsqueeze -> {e}"))?,
                v.squeeze(1)?.unsqueeze(2)
                    .map_err(|e| anyhow!("v.squeeze/unsqueeze -> {e}"))?,
            )
        } else {
            (
                q.transpose(1, 2)?.contiguous()
                    .map_err(|e| anyhow!("q.transpose -> {e}"))?,
                k.transpose(1, 2)?.contiguous()
                    .map_err(|e| anyhow!("k.transpose -> {e}"))?,
                v.transpose(1, 2)
                    .map_err(|e| anyhow!("v.transpose -> {e}"))?,
            )
        };

        // Apply RoPE (optional — Gemma3 local layers skip this)
        let q = if self.use_rope {
            self.apply_rotary_emb(&q, index_pos, cache)
                .map_err(|e| anyhow!("q.apply_rotary_emb -> {e}"))?
        } else {
            q
        };
        let k = if self.use_rope {
            self.apply_rotary_emb(&k, index_pos, cache)
                .map_err(|e| anyhow!("k.apply_rotary_emb -> {e}"))?
        } else {
            k
        };

        let (k, v) = match self.sliding_window {
            Some(w) => cache
                .process_kv_windowed(block_idx, k, v, w)
                .map_err(|e| anyhow!("cache.process_kv_windowed(block={block_idx}) -> {e}"))?,
            None => cache
                .process_kv(block_idx, k, v)
                .map_err(|e| anyhow!("cache.process_kv(block={block_idx}) -> {e}"))?,
        };

        let in_dtype = q.dtype();

        #[allow(unused_labels)]
        let y = 'attn: {
            // Flash Attention on CUDA — fused kernel, native GQA (no repeat_kv needed)
            // Only for F16/BF16 (flash-attn doesn't support F32)
            #[cfg(feature = "flash-attn")]
            if matches!(q.device(), candle_core::Device::Cuda(_))
                && matches!(q.dtype(), DType::F16 | DType::BF16)
            {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                break 'attn crate::utils::flash_attn::flash_attention(
                    &q, &k, &v, scale, seq_len > 1,
                ).map_err(|e| anyhow!("flash_attn: {e}"))?;
            }

            // Compute attention in F32 for numerical stability (Metal, CPU).
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;

            // The actual kv seq_len (may differ from query seq_len with sliding window)
            let kv_seq_len = k.dims()[2];

            // Fused SDPA on Metal — single kernel, native GQA (no repeat_kv needed)
            #[cfg(feature = "metal")]
            if matches!(q.device(), candle_core::Device::Metal(_)) {
                let scale = 1.0 / (self.head_dim as f32).sqrt();
                break 'attn candle_nn::ops::sdpa(&q, &k, &v, None, seq_len > 1, scale, 1.0)
                    .map_err(|e| anyhow!("sdpa: {e}"))?;
            }

            // Manual attention with GQA head expansion (CPU fallback)
            let k = self
                .repeat_kv(k)
                .map_err(|e| anyhow!("repeat_kv(k) -> {e}"))?;
            let v = self
                .repeat_kv(v)
                .map_err(|e| anyhow!("repeat_kv(v) -> {e}"))?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = if seq_len == 1 {
                att
            } else {
                // Causal mask — for sliding window the kv cache is already trimmed,
                // so we only need to mask within the query's own positions.
                let mask = cache
                    .mask(seq_len, q.device())
                    .map_err(|e| anyhow!("cache.mask({seq_len}) -> {e}"))?;

                // If kv_seq_len > seq_len (prefill into existing cache), extend mask
                let mask = if kv_seq_len > seq_len {
                    let pad = Tensor::zeros(
                        (seq_len, kv_seq_len - seq_len),
                        mask.dtype(),
                        mask.device(),
                    )?;
                    Tensor::cat(&[&pad, &mask], 1)?
                } else {
                    mask
                };

                let mask = mask
                    .broadcast_as(att.shape())
                    .map_err(|e| anyhow!("mask.broadcast_as({:?}) -> {e}", att.shape()))?;

                masked_fill(&att, &mask, f32::NEG_INFINITY)
                    .map_err(|e| anyhow!("masked_fill -> {e}"))?
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        let y = y.to_dtype(in_dtype)?;
        // For generation (seq_len=1), squeeze+reshape avoids the contiguous
        // copy that transpose(1,2) would trigger before reshape.
        let y = if seq_len == 1 {
            y.squeeze(2)?.reshape(&[b_sz, 1, self.size_q])?
        } else {
            y.transpose(1, 2)?.reshape(&[b_sz, seq_len, self.size_q])?
        };
        let y = self.linear_forward(&y, &self.o_proj)?;

        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        candle_transformers::utils::repeat_kv(
            x,
            self.num_attention_heads / self.num_key_value_heads,
        )
    }
}
