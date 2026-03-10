//! Quantized Gemma-3 model for all-hidden-states extraction.
//!
//! Adapted from `candle_transformers::models::quantized_gemma3` to:
//! 1. Return hidden states from ALL layers (not just final logits)
//! 2. Support padding masks (needed for left-padded text encoding)
//! 3. Fix the sliding window pattern bug
//!
//! Used by `Gemma3TextEncoder::load_gguf()` as an alternative to
//! the full-precision safetensors path. GGUF Q4_K_M of Gemma-3-12B
//! is ~7.4 GB and fits on a 24 GB GPU alongside the LTX-2 connector + VAE.

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::Embedding;

const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;

/// Max sequence length for RoPE precomputation.
/// We only use this for encoding 1024-token prompts, not for generation,
/// so we cap at 1024 instead of the model's full 131072 context window.
/// This saves ~6.4 GB of GPU memory (48 layers × 134 MB per RoPE table).
const ENCODER_MAX_SEQ_LEN: usize = 1024;

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        Ok(Self { inner })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul,
    feed_forward_up: QMatMul,
    feed_forward_down: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, max_seq_len: usize, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_q_norm: RmsNorm,
    attention_k_norm: RmsNorm,
    attention_norm: RmsNorm,
    post_attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
    post_ffn_norm: RmsNorm,
    mlp: Mlp,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    q_dim: usize,
    sliding_window_size: Option<usize>,
    rotary_embedding: std::sync::Arc<RotaryEmbedding>,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl LayerWeights {
    /// Build attention mask combining causal mask with optional padding mask.
    ///
    /// Returns a binary mask (1=attend, 0=block) of shape `[B, 1, seq_len, seq_len+index_pos]`.
    fn mask(
        &self,
        b_sz: usize,
        seq_len: usize,
        index_pos: usize,
        padding_mask: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        // Causal mask (with optional sliding window)
        let causal: Vec<u32> = if let Some(sw) = self.sliding_window_size {
            (0..seq_len)
                .flat_map(|i| {
                    (0..seq_len).map(move |j| {
                        if i < j || j + sw < i { 0u32 } else { 1u32 }
                    })
                })
                .collect()
        } else {
            (0..seq_len)
                .flat_map(|i| (0..seq_len).map(move |j| if i < j { 0u32 } else { 1u32 }))
                .collect()
        };
        let causal = Tensor::from_slice(&causal, (seq_len, seq_len), device)?;
        let causal = if index_pos > 0 {
            let zeros = Tensor::zeros((seq_len, index_pos), DType::U32, device)?;
            Tensor::cat(&[&zeros, &causal], D::Minus1)?
        } else {
            causal
        };
        // [B, 1, seq_len, total_len]
        let mut mask = causal.expand((b_sz, 1, seq_len, seq_len + index_pos))?;

        // Combine with padding mask if provided
        if let Some(pm) = padding_mask {
            // pm: [B, seq_len] with 1=valid, 0=padding
            // Expand to [B, 1, 1, seq_len] — keys that are padding should not be attended to
            let pm_u32 = pm.to_dtype(DType::U32)?
                .unsqueeze(1)?   // [B, 1, seq_len]
                .unsqueeze(1)?;  // [B, 1, 1, seq_len]
            mask = mask.broadcast_mul(&pm_u32)?;
        }

        Ok(mask)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.attention_q_norm.forward(&q.contiguous()?)?;
        let k = self.attention_k_norm.forward(&k.contiguous()?)?;

        let (q, k) = self.rotary_embedding.apply_rotary_emb_qkv(&q, &k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA
        let k = candle_transformers::utils::repeat_kv(k, self.n_head / self.n_kv_head)?;
        let v = candle_transformers::utils::repeat_kv(v, self.n_head / self.n_kv_head)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = mask {
            let mask = mask.broadcast_as(attn_weights.shape())?;
            let neg_inf = self.neg_inf.broadcast_as(attn_weights.dims())?;
            attn_weights = mask.eq(0u32)?.where_cond(&neg_inf, &attn_weights)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.q_dim))?
            .apply(&|t: &Tensor| self.attention_wo.forward(t))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Quantized Gemma-3 model that returns hidden states from all layers.
///
/// This is the quantized equivalent of `Gemma3AllHidden` in `gemma_encoder.rs`.
/// Loads from GGUF format and runs on GPU with quantized weights (~7.4 GB for Q4_K_M).
#[derive(Debug, Clone)]
pub(crate) struct Gemma3QuantizedAllHidden {
    tok_embeddings: Embedding,
    embedding_length: usize,
    layers: Vec<LayerWeights>,
}

impl Gemma3QuantizedAllHidden {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Detect architecture prefix
        let prefix = ["gemma3", "gemma2", "gemma", "gemma-embedding"]
            .iter()
            .find(|p| {
                ct.metadata
                    .contains_key(&format!("{}.attention.head_count", p))
            })
            .copied()
            .unwrap_or("gemma3");

        let md_get = |s: &str| {
            let key = format!("{prefix}.{s}");
            match ct.metadata.get(&key) {
                None => candle_core::bail!("cannot find {key} in metadata"),
                Some(v) => Ok(v),
            }
        };

        let head_count = md_get("attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("block_count")?.to_u32()? as usize;
        let embedding_length = md_get("embedding_length")?.to_u32()? as usize;
        let key_length = md_get("attention.key_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        let q_dim = head_count * key_length;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings (dequantized to F16 to save 2 GB vs F32)
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?.to_dtype(DType::F16)?;

        // Pre-compute shared RoPE tables (only 2 distinct frequencies)
        let rope_global = std::sync::Arc::new(
            RotaryEmbedding::new(key_length, rope_freq_base, ENCODER_MAX_SEQ_LEN, device)?
        );
        let rope_sliding = std::sync::Arc::new(
            RotaryEmbedding::new(key_length, rope_freq_base_sliding, ENCODER_MAX_SEQ_LEN, device)?
        );

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let pfx = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{pfx}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{pfx}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{pfx}.attn_v.weight"), device)?;
            let attention_wo = ct.tensor(reader, &format!("{pfx}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{pfx}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{pfx}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{pfx}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{pfx}.post_attention_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{pfx}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;
            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{pfx}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{pfx}.ffn_gate.weight"), device)?,
                )?,
                feed_forward_up: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{pfx}.ffn_up.weight"), device)?,
                )?,
                feed_forward_down: QMatMul::from_qtensor(
                    ct.tensor(reader, &format!("{pfx}.ffn_down.weight"), device)?,
                )?,
            };

            // Fixed sliding window pattern: layer_idx % N != 0 means sliding window
            // (upstream candle has a bug using (layer_idx + 1) % N > 0)
            let is_sliding = layer_idx % sliding_window_type != 0;
            let sw = is_sliding.then_some(sliding_window_size);
            let rotary_embedding = if is_sliding {
                rope_sliding.clone()
            } else {
                rope_global.clone()
            };

            layers.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                sliding_window_size: sw,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                kv_cache: None,
            });
        }

        log::info!(
            "Quantized Gemma-3 loaded: {} layers, {}d, {} heads ({}kv), head_dim={}",
            block_count, embedding_length, head_count, head_count_kv, key_length,
        );

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
        })
    }

    /// Forward pass returning hidden states from ALL layers.
    ///
    /// Returns `num_layers + 1` tensors: [embedding, layer_0, ..., layer_N].
    /// Each tensor is `[B, seq_len, hidden_size]`.
    pub fn forward_all_hidden(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        padding_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let (b_sz, seq_len) = x.dims2()?;

        let mut layer_in = self.tok_embeddings.forward(x)?.to_dtype(DType::F32)?;
        layer_in = (layer_in * (self.embedding_length as f64).sqrt())?;

        let mut all_hidden = Vec::with_capacity(self.layers.len() + 1);
        all_hidden.push(layer_in.clone());

        for layer in self.layers.iter_mut() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                Some(layer.mask(b_sz, seq_len, index_pos, padding_mask, x.device())?)
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(&x, attention_mask.as_ref(), index_pos)?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;

            all_hidden.push(x.clone());
            layer_in = x;
        }

        Ok(all_hidden)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}
