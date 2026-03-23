//! Qwen3 text encoder for FLUX.2-klein.
//!
//! Loads the Qwen3 model in encoder mode: full bidirectional attention,
//! no KV cache, returns hidden states (not logits).

use crate::backends::ComputeBackend;
use crate::cake::{Context, Forwarder};
use crate::models::sd::util::pack_tensors;
use async_trait::async_trait;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::VarBuilder;
use log::info;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

use super::config::FluxModelFile;

/// Qwen3 transformer block for encoder mode (bidirectional attention, no KV cache).
#[derive(Debug, Clone)]
struct EncoderBlock {
    rms_1_weight: Tensor,
    rms_2_weight: Tensor,
    rms_eps: f32,
    // Attention
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    qk_norm_eps: f32,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // MLP
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    backend: Arc<dyn ComputeBackend>,
}

impl EncoderBlock {
    fn load(vb: VarBuilder, cfg: &EncoderConfig, backend: Arc<dyn ComputeBackend>) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let size_q = cfg.num_heads * cfg.head_dim;
        let size_kv = cfg.num_kv_heads * cfg.head_dim;

        let attn = vb.pp("self_attn");
        let q_proj_weight = attn.pp("q_proj").get((size_q, h), "weight")?;
        let k_proj_weight = attn.pp("k_proj").get((size_kv, h), "weight")?;
        let v_proj_weight = attn.pp("v_proj").get((size_kv, h), "weight")?;
        let o_proj_weight = attn.pp("o_proj").get((h, size_q), "weight")?;
        let q_norm_weight = attn.pp("q_norm").get(cfg.head_dim, "weight")?;
        let k_norm_weight = attn.pp("k_norm").get(cfg.head_dim, "weight")?;
        let qk_norm_eps = cfg.rms_norm_eps as f32;

        let mlp = vb.pp("mlp");
        let gate_proj_weight = mlp.pp("gate_proj").get((i, h), "weight")?;
        let up_proj_weight = mlp.pp("up_proj").get((i, h), "weight")?;
        let down_proj_weight = mlp.pp("down_proj").get((h, i), "weight")?;

        let rms_1_weight = vb.pp("input_layernorm").get(h, "weight")?;
        let rms_2_weight = vb.pp("post_attention_layernorm").get(h, "weight")?;
        let rms_eps = cfg.rms_norm_eps as f32;

        Ok(Self {
            rms_1_weight,
            rms_2_weight,
            rms_eps,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
            qk_norm_eps,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            backend,
        })
    }

    /// attn_mask: optional (1, seq) tensor with 1 for real tokens, 0 for padding
    fn forward_with_mask(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _h) = x.dims3()?;

        // Pre-norm + attention
        let residual = x;
        let x = self.backend.rms_norm(x, &self.rms_1_weight, self.rms_eps)?;

        // QKV projections
        let q = self.backend.linear_forward(&x, &self.q_proj_weight, None)?;
        let k = self.backend.linear_forward(&x, &self.k_proj_weight, None)?;
        let v = self.backend.linear_forward(&x, &self.v_proj_weight, None)?;

        // Reshape to (b, seq, heads, head_dim)
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq, self.num_kv_heads, self.head_dim))?;

        // QK-norm (per-head)
        let q = self.backend.rms_norm(&q, &self.q_norm_weight, self.qk_norm_eps)?;
        let k = self.backend.rms_norm(&k, &self.k_norm_weight, self.qk_norm_eps)?;

        // Transpose to (b, heads, seq, head_dim)
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?;

        // Apply RoPE (Qwen3 uses rope_theta=1000000)
        let (q, k) = {
            let dev = q.device();
            let theta = 1_000_000f64;
            let half_dim = self.head_dim / 2;
            let inv_freq: Vec<f32> = (0..half_dim)
                .map(|i| 1.0 / theta.powf(i as f64 / half_dim as f64) as f32)
                .collect();
            let inv_freq = Tensor::new(inv_freq.as_slice(), dev)?
                .reshape((1, 1, 1, half_dim))?
                .to_dtype(q.dtype())?;
            let positions = Tensor::arange(0u32, seq as u32, dev)?
                .to_dtype(q.dtype())?
                .reshape((1, 1, seq, 1))?;
            let freqs = positions.broadcast_mul(&inv_freq)?;
            let cos = freqs.cos()?;
            let sin = freqs.sin()?;

            let apply = |x: &Tensor| -> Result<Tensor> {
                let x1 = x.narrow(D::Minus1, 0, half_dim)?;
                let x2 = x.narrow(D::Minus1, half_dim, half_dim)?;
                let rotated = Tensor::cat(
                    &[
                        &(x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?,
                        &(x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?,
                    ],
                    D::Minus1,
                )?;
                Ok(rotated)
            };
            (apply(&q)?.contiguous()?, apply(&k)?.contiguous()?)
        };

        // GQA expansion
        let repeat = self.num_heads / self.num_kv_heads;
        let k = if repeat > 1 {
            candle_transformers::utils::repeat_kv(k, repeat)?
        } else {
            k
        };
        let v = if repeat > 1 {
            candle_transformers::utils::repeat_kv(v, repeat)?
        } else {
            v
        };

        // Causal attention with optional padding mask
        let scale = (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? / scale)?;
        let att = if seq > 1 {
            // Causal mask: mask[i][j] = true if j > i (future positions)
            let rows = Tensor::arange(0u32, seq as u32, att.device())?
                .reshape((seq, 1))?;
            let cols = Tensor::arange(0u32, seq as u32, att.device())?
                .reshape((1, seq))?;
            let causal_mask = cols.broadcast_gt(&rows)?; // true = masked

            // Combine with attention mask (padding mask) if provided
            let full_mask = if let Some(am) = attn_mask {
                // am is (1, seq) with 1=real, 0=padding. We want to mask where am=0.
                let pad_mask = am.eq(0.0)?.reshape((1, 1, 1, seq))?
                    .broadcast_as(att.shape())?;
                let causal_mask = causal_mask.broadcast_as(att.shape())?;
                // Mask where causal OR padding
                causal_mask.add(&pad_mask)?.gt(0.0)?
            } else {
                causal_mask.broadcast_as(att.shape())?
            };

            let on_true = Tensor::new(f32::NEG_INFINITY, att.device())?
                .to_dtype(att.dtype())?
                .broadcast_as(att.shape())?;
            full_mask.where_cond(&on_true, &att)?
        } else {
            att
        };
        let last_dim = att.rank() - 1;
        let att = self.backend.softmax(&att, last_dim)?;
        let y = att.matmul(&v.contiguous()?)?;

        // Reshape back
        let y = y.transpose(1, 2)?;
        let y = y.reshape((b, seq, self.num_heads * self.head_dim))?;
        let x = (self.backend.linear_forward(&y, &self.o_proj_weight, None)? + residual)?;

        // Pre-norm + MLP
        let residual = &x;
        let h = self.backend.rms_norm(&x, &self.rms_2_weight, self.rms_eps)?;
        let gate_out = self.backend.linear_forward(&h, &self.gate_proj_weight, None)?;
        let gate = self.backend.silu(&gate_out)?.to_dtype(gate_out.dtype())?;
        let up = self.backend.linear_forward(&h, &self.up_proj_weight, None)?;
        let x = self.backend.linear_forward(&(gate * up)?, &self.down_proj_weight, None)?;
        x + residual
    }
}

/// Configuration for the Qwen3 text encoder within FLUX.2-klein.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
}

impl EncoderConfig {
    /// FLUX.2-klein-4B text encoder config (from text_encoder/config.json).
    pub fn flux2_klein() -> Self {
        Self {
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            vocab_size: 151936,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Qwen3 text encoder for FLUX — extracts hidden states, not logits.
#[derive(Debug)]
pub struct FluxTextEncoder {
    embeddings_weight: Tensor,
    blocks: Vec<EncoderBlock>,
    #[allow(dead_code)]
    final_norm_weight: Tensor,
    #[allow(dead_code)]
    final_norm_eps: f32,
    #[allow(dead_code)]
    cfg: EncoderConfig,
    #[allow(dead_code)]
    backend: Arc<dyn ComputeBackend>,
}

impl Display for FluxTextEncoder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FluxTextEncoder (local)")
    }
}

#[async_trait]
impl Forwarder for FluxTextEncoder {
    fn load(_name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        Self::load_model(&ctx.device, ctx.dtype, &ctx.args.model, ctx.backend.clone())
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,  // used as real_len for attention mask
        _block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        info!("FluxTextEncoder forwarding...");

        // x is token IDs: (batch, seq_len) — padded to max_length
        // index_pos carries the real (unpadded) token count for the attention mask
        let seq_len = x.dim(1)?;
        let attn_mask = if index_pos > 0 && index_pos < seq_len {
            // Build attention mask: 1 for real tokens, 0 for padding
            let mut mask_data = vec![1.0f32; index_pos];
            mask_data.resize(seq_len, 0.0);
            Some(Tensor::new(mask_data.as_slice(), x.device())?.unsqueeze(0)?)
        } else {
            None
        };

        let hidden = self.encode(x, attn_mask.as_ref())?;

        let tensors = vec![hidden];
        let packed = pack_tensors(tensors, &ctx.device)?;
        Ok(packed)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        "flux_text_encoder"
    }
}

impl FluxTextEncoder {
    pub fn load_model(
        device: &Device,
        dtype: DType,
        model_repo: &str,
        backend: Arc<dyn ComputeBackend>,
    ) -> anyhow::Result<Box<Self>> {
        let cfg = EncoderConfig::flux2_klein();

        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .to_string_lossy()
            .to_string();

        let all_paths = FluxModelFile::TextEncoder.get_all(model_repo, &cache_dir)?;
        // Filter to only safetensor files (exclude config.json, index.json)
        let weight_paths: Vec<_> = all_paths
            .into_iter()
            .filter(|p| {
                p.extension()
                    .map(|e| e == "safetensors")
                    .unwrap_or(false)
            })
            .collect();
        info!(
            "loading FLUX text encoder from {} shard(s)",
            weight_paths.len()
        );

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_paths, dtype, device)?
        };

        // Weights are prefixed with "model." in the text_encoder safetensors
        let vb_model = vb.pp("model");

        let embeddings_weight = vb_model.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let block = EncoderBlock::load(vb_layers.pp(i), &cfg, backend.clone())?;
            blocks.push(block);
        }

        let final_norm_weight = vb_model.pp("norm").get(cfg.hidden_size, "weight")?;
        let final_norm_eps = cfg.rms_norm_eps as f32;

        info!("FLUX text encoder loaded ({} layers)", cfg.num_hidden_layers);

        Ok(Box::new(Self {
            embeddings_weight,
            blocks,
            final_norm_weight,
            final_norm_eps,
            cfg,
            backend,
        }))
    }

    /// Run the full encoder forward pass, returning hidden states.
    ///
    /// FLUX.2-klein extracts hidden states from layers [9, 18, 27] (0-indexed)
    /// and concatenates them: 3 × 2560 = 7680 = joint_attention_dim.
    /// Encode with attention mask. attn_mask: (1, seq) with 1=real, 0=padding.
    pub fn encode(&self, token_ids: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        const OUTPUT_LAYERS: [usize; 3] = [8, 17, 26];

        let mut x = self.backend.embedding(token_ids, &self.embeddings_weight)?;
        let mut layer_outputs: Vec<Tensor> = Vec::new();

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward_with_mask(&x, attn_mask)?;
            if OUTPUT_LAYERS.contains(&i) {
                layer_outputs.push(x.clone());
            }
        }

        // Concatenate along feature dimension: (b, seq, 2560*3) = (b, seq, 7680)
        Tensor::cat(&layer_outputs, candle_core::D::Minus1)
    }
}
