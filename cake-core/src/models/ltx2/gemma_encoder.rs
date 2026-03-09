//! Gemma-3 text encoder for LTX-2.
//!
//! This wraps the candle-transformers Gemma-3 model to extract hidden states
//! from ALL layers (embedding + 48 transformer layers = 49 total), normalize
//! them, and pack into the format expected by the LTX-2 text connector:
//! `[B, seq_len, hidden_dim * num_layers]` = `[B, 1024, 188160]`.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma3;
use log::info;
use tokenizers::Tokenizer;

/// Gemma-3 config for the 12B model used by LTX-2.
pub fn gemma3_12b_config() -> gemma3::Config {
    gemma3::Config {
        attention_bias: false,
        head_dim: 256,
        hidden_activation: candle_nn::Activation::GeluPytorchTanh,
        hidden_size: 3840,
        intermediate_size: 15360,
        num_attention_heads: 16,
        num_hidden_layers: 48,
        num_key_value_heads: 8,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.0,
        rope_local_base_freq: 10_000.0,
        vocab_size: 262_208,
        final_logit_softcapping: None,
        attn_logit_softcapping: None,
        query_pre_attn_scalar: 256,
        sliding_window: 1024,
        sliding_window_pattern: 6, // 5 local : 1 global
        max_position_embeddings: 131_072,
    }
}

/// Maximum sequence length for text encoding.
/// Matches the default `max_sequence_length=256` in the Python LTX-2 pipeline.
/// Using 1024 causes OOM on 32GB GPUs during the 48-layer forward pass.
pub const MAX_SEQ_LEN: usize = 256;

/// Scale factor for normalization (matches Python pipeline).
pub const PACK_SCALE_FACTOR: f32 = 8.0;

/// Gemma-3 text encoder that extracts all hidden states.
///
/// Unlike the standard `gemma3::Model` which only returns logits,
/// this version collects hidden states from all 49 layers
/// (1 embedding + 48 transformer layers) for the LTX-2 connector.
pub struct Gemma3TextEncoder {
    model: Gemma3AllHidden,
    #[allow(dead_code)]
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
}

impl Gemma3TextEncoder {
    /// Load Gemma-3 model and tokenizer from safetensors files.
    pub fn load(
        model_paths: &[std::path::PathBuf],
        tokenizer_path: &std::path::Path,
        config: &gemma3::Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        info!("Loading Gemma-3 tokenizer from {:?}...", tokenizer_path);
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        info!(
            "Loading Gemma-3 model ({} layers, {}d) from {} file(s)...",
            config.num_hidden_layers,
            config.hidden_size,
            model_paths.len()
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(model_paths, dtype, device)?
        };

        let model = Gemma3AllHidden::new(false, config, vb)?;

        info!("Gemma-3 model loaded!");

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            dtype,
        })
    }

    /// Encode a text prompt into packed hidden states for LTX-2 connector.
    ///
    /// Returns `(packed_embeds, attention_mask)`:
    /// - `packed_embeds`: `[B, seq_len, hidden_dim * num_layers]` = `[1, L, 188160]`
    /// - `attention_mask`: `[B, seq_len]` binary mask (1=valid, 0=padding)
    #[allow(dead_code)]
    pub fn encode(&mut self, prompt: &str) -> Result<(Tensor, Tensor)> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();
        let seq_len = tokens.len().min(MAX_SEQ_LEN);

        // Left-pad to MAX_SEQ_LEN (Gemma uses left padding)
        let pad_len = MAX_SEQ_LEN.saturating_sub(seq_len);
        let mut padded_ids = vec![0u32; pad_len];
        padded_ids.extend_from_slice(&tokens[..seq_len]);

        // Attention mask: 0 for padding, 1 for valid
        let mut mask_vals = vec![0.0f32; pad_len];
        mask_vals.extend(vec![1.0f32; seq_len]);

        let input_ids = Tensor::new(padded_ids.as_slice(), &self.device)?
            .unsqueeze(0)?; // [1, MAX_SEQ_LEN]
        let attention_mask = Tensor::new(mask_vals.as_slice(), &self.device)?
            .unsqueeze(0)?; // [1, MAX_SEQ_LEN]

        // Run Gemma-3 forward pass, collecting all hidden states
        self.model.clear_kv_cache();
        let all_hidden = self.model.forward_all_hidden(&input_ids, 0, Some(&attention_mask))?;
        // all_hidden: Vec of 49 tensors, each [1, MAX_SEQ_LEN, 3840]

        // Debug: check raw Gemma hidden state statistics
        {
            // Check embedding output (layer 0) and last layer
            let emb_flat = all_hidden[0].flatten_all()?.to_dtype(DType::F32)?;
            let last_flat = all_hidden[all_hidden.len()-1].flatten_all()?.to_dtype(DType::F32)?;
            let emb_std: f32 = emb_flat.var(0)?.to_scalar::<f32>()?.sqrt();
            let last_std: f32 = last_flat.var(0)?.to_scalar::<f32>()?.sqrt();
            let emb_min: f32 = emb_flat.min(0)?.to_scalar()?;
            let emb_max: f32 = emb_flat.max(0)?.to_scalar()?;
            let last_min: f32 = last_flat.min(0)?.to_scalar()?;
            let last_max: f32 = last_flat.max(0)?.to_scalar()?;
            log::info!(
                "Gemma raw hidden: embed std={:.4} [{:.2},{:.2}], layer48 std={:.4} [{:.2},{:.2}], {} layers, seq_len={}",
                emb_std, emb_min, emb_max,
                last_std, last_min, last_max,
                all_hidden.len(), seq_len,
            );
        }

        // Stack to [B, seq_len, hidden_dim, num_layers]
        let stacked = Tensor::stack(&all_hidden, D::Minus1)?;

        // Compute sequence lengths for normalization
        let sequence_lengths = Tensor::new(&[seq_len as f32], &self.device)?;

        // Pack and normalize
        let packed = pack_text_embeds(
            &stacked,
            &sequence_lengths,
            "left",
            PACK_SCALE_FACTOR,
        )?
        .to_dtype(self.dtype)?;

        Ok((packed, attention_mask.to_dtype(DType::F32)?))
    }

    /// Encode from pre-tokenized input tensors (for worker-side encoding).
    ///
    /// `input_ids`: `[B, L]` u32 token IDs (left-padded to MAX_SEQ_LEN)
    /// `attention_mask`: `[B, L]` float mask (1=valid, 0=padding)
    ///
    /// Returns `(packed_embeds, attention_mask)` same as `encode()`.
    pub fn encode_from_tokens(
        &mut self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let _seq_len = input_ids.dim(1)?;

        // Move tensors to encoder's device if needed
        let input_ids = input_ids.to_device(&self.device)?;
        let attention_mask_f = attention_mask.to_dtype(DType::F32)?.to_device(&self.device)?;

        // Run Gemma-3 forward pass
        self.model.clear_kv_cache();
        let all_hidden = self.model.forward_all_hidden(&input_ids, 0, Some(&attention_mask_f))?;

        // Stack to [B, seq_len, hidden_dim, num_layers]
        let stacked = Tensor::stack(&all_hidden, D::Minus1)?;

        // Compute sequence lengths from mask (sum of valid tokens per batch)
        let sequence_lengths = attention_mask_f.sum(1)?; // [B]

        let packed = pack_text_embeds(
            &stacked,
            &sequence_lengths,
            "left",
            PACK_SCALE_FACTOR,
        )?
        .to_dtype(self.dtype)?;

        Ok((packed, attention_mask_f))
    }
}

/// Pack and normalize text encoder hidden states.
///
/// Matches the Python `_pack_text_embeds` function in the LTX-2 pipeline.
///
/// Input: `[B, seq_len, hidden_dim, num_layers]`
/// Output: `[B, seq_len, hidden_dim * num_layers]`
///
/// Normalization per batch, per layer:
/// 1. Compute masked mean over non-padding positions
/// 2. Compute masked min/max over non-padding positions
/// 3. Normalize: `(x - mean) / (max - min + eps) * scale_factor`
/// 4. Flatten last two dims and zero out padding positions
pub fn pack_text_embeds(
    text_hidden_states: &Tensor,
    sequence_lengths: &Tensor,
    padding_side: &str,
    scale_factor: f32,
) -> candle_core::Result<Tensor> {
    let eps = 1e-6f64;
    let (batch_size, seq_len, hidden_dim, num_layers) = text_hidden_states.dims4()?;
    let device = text_hidden_states.device();

    // Create padding mask [B, seq_len, 1, 1]
    let token_indices = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(0)?; // [1, seq_len]

    let mask = match padding_side {
        "left" => {
            // Valid tokens are from (seq_len - sequence_length) to end
            let start_indices = Tensor::full(seq_len as f32, (batch_size, 1), device)?
                .broadcast_sub(&sequence_lengths.unsqueeze(1)?)?; // [B, 1]
            token_indices.broadcast_ge(&start_indices)? // [B, seq_len]
        }
        "right" => {
            // Valid tokens are from 0 to sequence_length - 1
            token_indices.broadcast_lt(&sequence_lengths.unsqueeze(1)?)? // [B, seq_len]
        }
        _ => candle_core::bail!("padding_side must be 'left' or 'right'"),
    };
    // mask: [B, seq_len] -> [B, seq_len, 1, 1]
    let mask_f = mask.to_dtype(DType::F32)?.unsqueeze(2)?.unsqueeze(3)?;

    // Work in F32 for numerical stability
    let x = text_hidden_states.to_dtype(DType::F32)?;

    // Masked hidden states (zero out padding)
    let masked_x = x.broadcast_mul(&mask_f)?;

    // Compute masked mean: sum over (seq_len, hidden_dim) / num_valid_positions
    // num_valid_positions = sequence_lengths * hidden_dim
    let num_valid = sequence_lengths
        .to_dtype(DType::F32)?
        .affine(hidden_dim as f64, 0.0)?
        .reshape((batch_size, 1, 1, 1))?;
    let sum_x = masked_x.sum(1)?.sum(1)?; // [B, num_layers]
    let sum_x = sum_x.unsqueeze(1)?.unsqueeze(1)?; // [B, 1, 1, num_layers]
    let num_valid_eps = (num_valid + eps)?;
    let masked_mean = sum_x.broadcast_div(&num_valid_eps)?;

    // Compute masked min/max
    // For min: fill padding with +inf, then amin
    // For max: fill padding with -inf, then amax
    let inv_mask = mask_f.affine(-1.0, 1.0)?; // 1 where padding, 0 where valid
    let inf_fill = inv_mask.affine(f32::MAX as f64, 0.0)?;
    let neg_inf_fill = inv_mask.affine(f32::MIN as f64, 0.0)?;

    let x_for_min = x.broadcast_add(&inf_fill)?;
    let x_for_max = x.broadcast_add(&neg_inf_fill)?;

    // amin/amax over dims 1 and 2 (seq_len, hidden_dim), keeping [B, 1, 1, num_layers]
    let x_min = x_for_min.flatten(1, 2)?.min(1)?.unsqueeze(1)?.unsqueeze(1)?;
    let x_max = x_for_max.flatten(1, 2)?.max(1)?.unsqueeze(1)?.unsqueeze(1)?;

    // Normalize: (x - mean) / (max - min + eps) * scale_factor
    let range = (x_max.broadcast_sub(&x_min)? + eps)?;
    let normalized = x
        .broadcast_sub(&masked_mean)?
        .broadcast_div(&range)?
        .affine(scale_factor as f64, 0.0)?;

    // Flatten last two dims: [B, seq_len, hidden_dim, num_layers] -> [B, seq_len, hidden_dim * num_layers]
    let packed = normalized.flatten(2, 3)?;

    // Zero out padding positions
    let mask_flat = mask
        .to_dtype(DType::F32)?
        .unsqueeze(2)? // [B, seq_len, 1]
        .broadcast_as((batch_size, seq_len, hidden_dim * num_layers))?
        .contiguous()?;

    packed.broadcast_mul(&mask_flat)
}

// ---------------------------------------------------------------------------
// Modified Gemma-3 model that returns all hidden states
// ---------------------------------------------------------------------------

/// Gemma-3 model modified to return hidden states from all layers.
///
/// Based on `candle_transformers::models::gemma3::Model` but the forward
/// pass collects and returns all intermediate hidden states instead of
/// just the final logits.
struct Gemma3AllHidden {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Gemma3DecoderLayer>,
    hidden_size: usize,
    sliding_window: usize,
    dtype: DType,
    device: Device,
}

impl Gemma3AllHidden {
    fn new(use_flash_attn: bool, cfg: &gemma3::Config, vb: VarBuilder) -> candle_core::Result<Self> {
        // google/gemma-3-12b-pt uses "language_model.model." prefix
        let vb_m = if vb.contains_tensor("language_model.model.embed_tokens.weight") {
            vb.pp("language_model").pp("model")
        } else {
            vb.pp("model")
        };
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let sliding_window = layer_idx % cfg.sliding_window_pattern != 0;
            let layer = Gemma3DecoderLayer::new(
                use_flash_attn,
                cfg,
                vb_l.pp(layer_idx),
                sliding_window.then_some(cfg.sliding_window),
            )?;
            layers.push(layer);
        }
        Ok(Self {
            embed_tokens,
            layers,
            hidden_size: cfg.hidden_size,
            sliding_window: cfg.sliding_window,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    /// Forward pass that returns hidden states from ALL layers.
    ///
    /// Returns a Vec of `num_hidden_layers + 1` tensors:
    /// - `[0]`: embedding output (before any transformer layer)
    /// - `[1..=N]`: output of each transformer layer
    ///
    /// Each tensor has shape `[B, seq_len, hidden_size]`.
    fn forward_all_hidden(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        padding_mask: Option<&Tensor>,
    ) -> candle_core::Result<Vec<Tensor>> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        xs = (xs * (self.hidden_size as f64).sqrt())?;

        let mut all_hidden = Vec::with_capacity(self.layers.len() + 1);
        all_hidden.push(xs.clone());

        // Convert padding mask [B, L] (1=valid, 0=pad) to additive form [B, 1, 1, L]
        // where padding positions get -inf (added to attention weights before softmax)
        let padding_attn_mask = if let Some(pm) = padding_mask {
            // (mask - 1) gives -1 for padding, 0 for valid
            // Multiply by large value to get -inf-like for padding
            let additive = pm
                .to_dtype(DType::F32)?
                .affine(1.0, -1.0)? // 1→0, 0→-1
                .affine(1e9, 0.0)?  // 0→0, -1→-1e9
                .unsqueeze(1)?      // [B, 1, L]
                .unsqueeze(1)?;     // [B, 1, 1, L]
            Some(additive.to_dtype(self.dtype)?)
        } else {
            None
        };

        // Create causal attention masks
        let (attention_mask, sliding_attention_mask) = if seq_len <= 1 {
            (None, None)
        } else {
            let causal = prepare_decoder_attention_mask(
                b_size,
                seq_len,
                seqlen_offset,
                None,
                self.dtype,
                &self.device,
            )?;
            let sliding_causal = prepare_decoder_attention_mask(
                b_size,
                seq_len,
                seqlen_offset,
                Some(self.sliding_window),
                self.dtype,
                &self.device,
            )?;
            // Combine causal masks with padding mask
            let mask = match &padding_attn_mask {
                Some(pm) => causal.broadcast_add(pm)?,
                None => causal,
            };
            let sliding_mask = match &padding_attn_mask {
                Some(pm) => sliding_causal.broadcast_add(pm)?,
                None => sliding_causal,
            };
            (Some(mask), Some(sliding_mask))
        };

        let num_layers = self.layers.len();
        for i in 0..num_layers {
            let layer = &mut self.layers[i];
            let mask = if layer.sliding_window.is_some() {
                &sliding_attention_mask
            } else {
                &attention_mask
            };
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offset)?;

            // Debug: log every 12th layer and last layer
            if i % 12 == 0 || i == num_layers - 1 {
                let flat = xs.flatten_all()?.to_dtype(DType::F32)?;
                let std_val: f32 = flat.var(0)?.to_scalar::<f32>()?.sqrt();
                let max_val: f32 = flat.max(0)?.to_scalar()?;
                log::info!("Gemma layer {} hidden: std={:.2}, max={:.2}", i, std_val, max_val);
            }

            all_hidden.push(xs.clone());
        }

        Ok(all_hidden)
    }

    fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

// ---------------------------------------------------------------------------
// Internal Gemma-3 components (duplicated from candle-transformers because
// the upstream types are not public and we need mutable access for KV cache)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct GemmaRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl GemmaRmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for GemmaRmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&(&self.weight + 1.0)?)
    }
}

#[derive(Debug, Clone)]
struct GemmaRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl GemmaRotaryEmbedding {
    fn new(dtype: DType, cfg: &gemma3::Config, dev: &Device, sliding_window: Option<usize>) -> candle_core::Result<Self> {
        let dim = cfg.head_dim;
        let max_seq_len = cfg.max_position_embeddings;
        let rope_freq = if sliding_window.is_some() {
            cfg.rope_local_base_freq
        } else {
            cfg.rope_theta
        };
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_freq.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct GemmaMLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    act_fn: candle_nn::Activation,
}

impl GemmaMLP {
    fn new(cfg: &gemma3::Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = candle_nn::linear_b(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_b(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_b(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_activation,
        })
    }
}

impl Module for GemmaMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
enum GemmaKvCache {
    Normal(candle_nn::kv_cache::KvCache),
    Rotating(candle_nn::kv_cache::RotatingKvCache),
}

#[derive(Debug, Clone)]
struct GemmaAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    q_norm: GemmaRmsNorm,
    k_norm: GemmaRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    attn_logit_softcapping: Option<f64>,
    rotary_emb: std::sync::Arc<GemmaRotaryEmbedding>,
    kv_cache: GemmaKvCache,
}

impl GemmaAttention {
    fn new(
        rotary_emb: std::sync::Arc<GemmaRotaryEmbedding>,
        cfg: &gemma3::Config,
        sliding_window: Option<usize>,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = cfg.head_dim;
        let bias = cfg.attention_bias;
        let q_proj = candle_nn::linear_b(hidden_sz, num_heads * head_dim, bias, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear_b(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear_b(hidden_sz, num_kv_heads * head_dim, bias, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_b(num_heads * head_dim, hidden_sz, bias, vb.pp("o_proj"))?;
        let q_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = GemmaRmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        let kv_cache = if let Some(sw) = sliding_window {
            GemmaKvCache::Rotating(candle_nn::kv_cache::RotatingKvCache::new(2, sw))
        } else {
            GemmaKvCache::Normal(candle_nn::kv_cache::KvCache::new(2, cfg.max_position_embeddings))
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            attn_logit_softcapping: cfg.attn_logit_softcapping,
            rotary_emb,
            kv_cache,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let query_states = self.q_norm.forward(&query_states)?;
        let key_states = self.k_norm.forward(&key_states)?;

        let (query_states, key_states) =
            self.rotary_emb.apply_rotary_emb_qkv(&query_states, &key_states, seqlen_offset)?;

        // KV cache's slice_set requires contiguous tensors
        let key_states = key_states.contiguous()?;
        let value_states = value_states.contiguous()?;
        let (key_states, value_states) = match &mut self.kv_cache {
            GemmaKvCache::Normal(cache) => cache.append(&key_states, &value_states)?,
            GemmaKvCache::Rotating(cache) => cache.append(&key_states, &value_states)?,
        };

        let key_states = candle_transformers::utils::repeat_kv(key_states, self.num_kv_groups)?
            .contiguous()?;
        let value_states = candle_transformers::utils::repeat_kv(value_states, self.num_kv_groups)?
            .contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;

        let attn_weights = match self.attn_logit_softcapping {
            None => attn_weights,
            Some(sc) => ((attn_weights / sc)?.tanh()? * sc)?,
        };

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, ()))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.kv_cache {
            GemmaKvCache::Normal(c) => c.reset(),
            GemmaKvCache::Rotating(c) => c.reset(),
        }
    }
}

struct Gemma3DecoderLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMLP,
    input_layernorm: GemmaRmsNorm,
    pre_feedforward_layernorm: GemmaRmsNorm,
    post_feedforward_layernorm: GemmaRmsNorm,
    post_attention_layernorm: GemmaRmsNorm,
    sliding_window: Option<usize>,
}

impl Gemma3DecoderLayer {
    fn new(
        use_flash_attn: bool,
        cfg: &gemma3::Config,
        vb: VarBuilder,
        sliding_window: Option<usize>,
    ) -> candle_core::Result<Self> {
        let _ = use_flash_attn; // Not used in encoder mode (full sequence, no causal needed for hidden state extraction)
        let rotary_emb = std::sync::Arc::new(GemmaRotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb.device(),
            sliding_window,
        )?);
        let self_attn = GemmaAttention::new(rotary_emb, cfg, sliding_window, vb.pp("self_attn"))?;
        let mlp = GemmaMLP::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            GemmaRmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let pre_feedforward_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("pre_feedforward_layernorm"),
        )?;
        let post_feedforward_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_feedforward_layernorm"),
        )?;
        let post_attention_layernorm = GemmaRmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            post_attention_layernorm,
            sliding_window,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = xs.apply(&self.post_attention_layernorm)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.pre_feedforward_layernorm)?;
        let xs = xs.apply(&self.mlp)?;
        let xs = xs.apply(&self.post_feedforward_layernorm)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Prepare decoder attention mask (causal + optional sliding window).
fn prepare_decoder_attention_mask(
    b_size: usize,
    tgt_len: usize,
    seqlen_offset: usize,
    sliding_window: Option<usize>,
    dtype: DType,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mask: Vec<_> = if let Some(sliding_window) = sliding_window {
        (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect()
    } else {
        (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0f32 }))
            .collect()
    };
    let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
    let mask = if seqlen_offset > 0 {
        let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
        Tensor::cat(&[&mask0, &mask], D::Minus1)?
    } else {
        mask
    };
    mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
        .to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_pack_text_embeds_shape() {
        let device = Device::Cpu;
        let b = 2;
        let seq_len = 16;
        let hidden_dim = 8;
        let num_layers = 4;

        let hidden = Tensor::randn(
            0f32, 1f32,
            (b, seq_len, hidden_dim, num_layers),
            &device,
        ).unwrap();
        let seq_lengths = Tensor::new(&[12.0f32, 16.0], &device).unwrap();

        let packed = pack_text_embeds(&hidden, &seq_lengths, "left", 8.0).unwrap();
        assert_eq!(packed.dims(), &[b, seq_len, hidden_dim * num_layers]);
    }

    #[test]
    fn test_pack_text_embeds_padding_zeroed() {
        let device = Device::Cpu;
        let b = 1;
        let seq_len = 8;
        let hidden_dim = 4;
        let num_layers = 2;

        let hidden = Tensor::ones(
            (b, seq_len, hidden_dim, num_layers),
            DType::F32,
            &device,
        ).unwrap();
        // Only last 4 tokens are valid (left padding)
        let seq_lengths = Tensor::new(&[4.0f32], &device).unwrap();

        let packed = pack_text_embeds(&hidden, &seq_lengths, "left", 8.0).unwrap();
        let vals: Vec<f32> = packed.flatten_all().unwrap().to_vec1().unwrap();

        // First 4 positions (padding) should be zero
        for i in 0..(4 * hidden_dim * num_layers) {
            assert_eq!(vals[i], 0.0, "Padding position {} should be zero", i);
        }
    }

    #[test]
    fn test_pack_text_embeds_right_padding() {
        let device = Device::Cpu;
        let hidden = Tensor::ones((1, 8, 4, 2), DType::F32, &device).unwrap();
        // First 6 tokens valid, last 2 padding
        let seq_lengths = Tensor::new(&[6.0f32], &device).unwrap();

        let packed = pack_text_embeds(&hidden, &seq_lengths, "right", 8.0).unwrap();
        let vals: Vec<f32> = packed.flatten_all().unwrap().to_vec1().unwrap();

        // Last 2 positions (padding) should be zero
        let packed_dim = 4 * 2;
        for i in (6 * packed_dim)..(8 * packed_dim) {
            assert_eq!(vals[i], 0.0, "Padding position {} should be zero", i);
        }
    }

    #[test]
    fn test_gemma3_12b_config() {
        let cfg = gemma3_12b_config();
        assert_eq!(cfg.hidden_size, 3840);
        assert_eq!(cfg.num_hidden_layers, 48);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 256);
        assert_eq!(cfg.intermediate_size, 15360);
        assert_eq!(cfg.vocab_size, 262_208);
        assert_eq!(cfg.sliding_window, 1024);
        assert_eq!(cfg.sliding_window_pattern, 6); // 5 local : 1 global
    }
}
