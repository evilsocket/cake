//! Qwen3 text encoder for FLUX.2-klein.
//!
//! Loads the Qwen3 model in encoder mode: full bidirectional attention,
//! no KV cache, returns hidden states (not logits).

use crate::cake::{Context, Forwarder};
use crate::models::sd::util::pack_tensors;
use async_trait::async_trait;
use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{linear_no_bias, Embedding, Linear, RmsNorm, VarBuilder};
use log::info;
use std::fmt::{Debug, Display, Formatter};

use super::config::FluxModelFile;

/// Qwen3 transformer block for encoder mode (bidirectional attention, no KV cache).
#[derive(Debug, Clone)]
struct EncoderBlock {
    rms_1: RmsNorm,
    rms_2: RmsNorm,
    // Attention
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    // MLP
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl EncoderBlock {
    fn load(vb: VarBuilder, cfg: &EncoderConfig) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let size_q = cfg.num_heads * cfg.head_dim;
        let size_kv = cfg.num_kv_heads * cfg.head_dim;

        let attn = vb.pp("self_attn");
        let q_proj = linear_no_bias(h, size_q, attn.pp("q_proj"))?;
        let k_proj = linear_no_bias(h, size_kv, attn.pp("k_proj"))?;
        let v_proj = linear_no_bias(h, size_kv, attn.pp("v_proj"))?;
        let o_proj = linear_no_bias(size_q, h, attn.pp("o_proj"))?;
        let q_norm = candle_nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, attn.pp("q_norm"))?;
        let k_norm = candle_nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, attn.pp("k_norm"))?;

        let mlp = vb.pp("mlp");
        let gate_proj = linear_no_bias(h, i, mlp.pp("gate_proj"))?;
        let up_proj = linear_no_bias(h, i, mlp.pp("up_proj"))?;
        let down_proj = linear_no_bias(i, h, mlp.pp("down_proj"))?;

        let rms_1 = candle_nn::rms_norm(h, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(h, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?;

        Ok(Self {
            rms_1,
            rms_2,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_heads,
            num_kv_heads: cfg.num_kv_heads,
            head_dim: cfg.head_dim,
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    /// attn_mask: optional (1, seq) tensor with 1 for real tokens, 0 for padding
    fn forward_with_mask(&self, x: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq, _h) = x.dims3()?;

        // Pre-norm + attention
        let residual = x;
        let x = self.rms_1.forward(x)?;

        // QKV projections
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;

        // Reshape to (b, seq, heads, head_dim)
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq, self.num_kv_heads, self.head_dim))?;

        // QK-norm (per-head)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

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
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let y = att.matmul(&v.contiguous()?)?;

        // Reshape back
        let y = y.transpose(1, 2)?;
        let y = y.reshape((b, seq, self.num_heads * self.head_dim))?;
        let x = (self.o_proj.forward(&y)? + residual)?;

        // Pre-norm + MLP
        let residual = &x;
        let h = self.rms_2.forward(&x)?;
        let gate_out = self.gate_proj.forward(&h)?;
        let gate = candle_nn::ops::silu(&gate_out)?.to_dtype(gate_out.dtype())?;
        let up = self.up_proj.forward(&h)?;
        let x = self.down_proj.forward(&(gate * up)?)?;
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
    embeddings: Embedding,
    blocks: Vec<EncoderBlock>,
    #[allow(dead_code)]
    final_norm: RmsNorm,
    #[allow(dead_code)]
    cfg: EncoderConfig,
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
        Self::load_model(&ctx.device, ctx.dtype, &ctx.args.model)
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

        let embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb_model.pp("embed_tokens"),
        )?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb_model.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            let block = EncoderBlock::load(vb_layers.pp(i), &cfg)?;
            blocks.push(block);
        }

        let final_norm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb_model.pp("norm"),
        )?;

        info!("FLUX text encoder loaded ({} layers)", cfg.num_hidden_layers);

        Ok(Box::new(Self {
            embeddings,
            blocks,
            final_norm,
            cfg,
        }))
    }

    /// Run the full encoder forward pass, returning hidden states.
    ///
    /// FLUX.2-klein extracts hidden states from layers [9, 18, 27] (0-indexed)
    /// and concatenates them: 3 × 2560 = 7680 = joint_attention_dim.
    /// Encode with attention mask. attn_mask: (1, seq) with 1=real, 0=padding.
    pub fn encode(&self, token_ids: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        const OUTPUT_LAYERS: [usize; 3] = [8, 17, 26];

        let mut x = self.embeddings.forward(token_ids)?;
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
