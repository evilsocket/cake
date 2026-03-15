//! LTX-2 Text Connectors — self-attention transformer with learnable registers.
//!
//! Matches HF diffusers `LTX2TextConnectors` + `LTX2ConnectorTransformer1d`.
//!
//! Architecture:
//! 1. Project packed Gemma tokens (3840 * 49 = 188160 → 3840) via linear (no bias)
//! 2. Replace padding tokens with learnable registers
//! 3. Apply 1D RoPE self-attention transformer (2 layers)
//! 4. norm_out (RMSNorm, no learnable weights)
//!
//! Key difference from perceiver: registers replace padding tokens in the SAME
//! sequence (not separate queries). The transformer does pure self-attention.

use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::attention::{rms_norm, Attention};
use super::config::Ltx2ConnectorConfig;
use super::feed_forward::FeedForward;
use super::rope::precompute_freqs_cis;

/// A single 1D transformer block (self-attention + FFN, no cross-attention).
///
/// Matches `LTX2TransformerBlock1d`.
#[derive(Debug)]
struct ConnectorBlock {
    attn1: Attention,
    ff: FeedForward,
    norm_eps: f64,
}

impl ConnectorBlock {
    fn new(
        dim: usize,
        heads: usize,
        d_head: usize,
        norm_eps: f64,
        gated: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn1 = Attention::new(dim, None, heads, d_head, norm_eps, gated, vb.pp("attn1"))?;
        let ff = FeedForward::new(dim, dim, 4, vb.pp("ff"))?;
        Ok(Self {
            attn1,
            ff,
            norm_eps,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
        pe: Option<&(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // Self-attention
        let norm_h = rms_norm(hidden_states, self.norm_eps)?;
        let attn_out = self.attn1.forward(&norm_h, None, pe, None, mask)?;
        let h = hidden_states.broadcast_add(&attn_out)?;

        // FFN
        let norm_h = rms_norm(&h, self.norm_eps)?;
        let ff_out = self.ff.forward(&norm_h)?;
        h.broadcast_add(&ff_out)
    }
}

/// 1D connector transformer (matches `LTX2ConnectorTransformer1d`).
///
/// Self-attention transformer with learnable registers that replace padding tokens.
#[derive(Debug)]
struct ConnectorTransformer1d {
    learnable_registers: Tensor, // [num_registers, inner_dim]
    num_registers: usize,
    blocks: Vec<ConnectorBlock>,
    norm_eps: f64,
    // 1D RoPE parameters
    inner_dim: usize,
    num_heads: usize,
    rope_theta: f32,
    base_seq_len: usize,
}

impl ConnectorTransformer1d {
    fn new(
        num_layers: usize,
        num_registers: usize,
        heads: usize,
        d_head: usize,
        norm_eps: f64,
        rope_theta: f32,
        base_seq_len: usize,
        gated: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let inner_dim = heads * d_head;

        let learnable_registers = vb.get((num_registers, inner_dim), "learnable_registers")?;

        let mut blocks = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            blocks.push(ConnectorBlock::new(
                inner_dim,
                heads,
                d_head,
                norm_eps,
                gated,
                vb.pp(format!("transformer_blocks.{i}")),
            )?);
        }

        Ok(Self {
            learnable_registers,
            num_registers,
            blocks,
            norm_eps,
            inner_dim,
            num_heads: heads,
            rope_theta,
            base_seq_len,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Replace padding with learned registers
        let (mut h, new_mask) = self.replace_padding_with_registers(
            hidden_states,
            attention_mask,
            seq_len,
            batch_size,
        )?;

        // 1D RoPE: build position grid [B, 1, seq_len] with arange(seq_len)
        let positions_1d: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_t = Tensor::new(positions_1d, h.device())?;
        let pos_grid = pos_t
            .unsqueeze(0)? // [1, seq_len]
            .unsqueeze(0)? // [1, 1, seq_len]
            .broadcast_as((batch_size, 1, seq_len))?
            .contiguous()?;
        let pe = precompute_freqs_cis(
            &pos_grid,
            self.inner_dim,
            self.rope_theta,
            &[self.base_seq_len],
            self.num_heads,
            h.dtype(),
        )?;

        // Run transformer blocks
        for block in &self.blocks {
            h = block.forward(&h, None, Some(&pe))?;
        }

        // norm_out (no learnable weights)
        let h = rms_norm(&h, self.norm_eps)?;

        Ok((h, new_mask))
    }

    /// Replace padding tokens with learned registers.
    ///
    /// For each batch element:
    /// 1. Extract non-padding tokens (where mask >= threshold)
    /// 2. Pad to seq_len with zeros
    /// 3. Tile registers to fill sequence
    /// 4. Use flipped mask to blend: mask * padded_text + (1-mask) * registers
    fn replace_padding_with_registers(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seq_len: usize,
        batch_size: usize,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mask = match attention_mask {
            Some(m) => m,
            None => return Ok((hidden_states.clone(), None)),
        };

        // Binarize mask: >= -9000 means valid token
        let threshold = -9000.0f32;
        let binary_mask = mask.ge(threshold)?.to_dtype(DType::F32)?;
        // binary_mask: [B, L] or [B, 1, 1, L]
        let binary_mask = if binary_mask.rank() == 4 {
            binary_mask.squeeze(1)?.squeeze(1)?
        } else {
            binary_mask
        };

        // Tile registers to fill sequence
        if seq_len % self.num_registers != 0 {
            candle_core::bail!(
                "seq_len ({}) must be divisible by num_learnable_registers ({})",
                seq_len,
                self.num_registers
            );
        }
        let num_repeats = seq_len / self.num_registers;
        let inner_dim = self.learnable_registers.dim(1)?;

        // [num_registers, dim] -> tile -> [seq_len, dim]
        let registers = if num_repeats > 1 {
            let mut parts = Vec::with_capacity(num_repeats);
            for _ in 0..num_repeats {
                parts.push(self.learnable_registers.clone());
            }
            Tensor::cat(&parts, 0)?
        } else {
            self.learnable_registers.clone()
        };
        let registers = registers
            .to_dtype(hidden_states.dtype())?;

        // For each batch: extract non-padded tokens, re-pack, blend with registers
        let mut batch_results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let h_i = hidden_states.get(i)?; // [L, D]
            let m_i = binary_mask.get(i)?; // [L]

            // Count valid tokens
            let m_vals: Vec<f32> = m_i.to_vec1()?;
            let valid_count: usize = m_vals.iter().filter(|&&v| v > 0.5).count();

            // Extract valid tokens
            let mut valid_indices = Vec::with_capacity(valid_count);
            for (j, &v) in m_vals.iter().enumerate() {
                if v > 0.5 {
                    valid_indices.push(j as u32);
                }
            }

            let padded = if valid_count > 0 && valid_count < seq_len {
                let idx = Tensor::from_vec(valid_indices, (valid_count,), h_i.device())?;
                let valid_tokens = h_i.index_select(&idx, 0)?; // [valid_count, D]
                // Pad with zeros to seq_len
                let pad = Tensor::zeros((seq_len - valid_count, inner_dim), hidden_states.dtype(), h_i.device())?;
                Tensor::cat(&[valid_tokens, pad], 0)?
            } else {
                h_i.clone()
            };

            // Flip mask and use as blend factor
            // flipped_mask[j] = mask[L-1-j]
            let flip_indices: Vec<u32> = (0..seq_len).rev().map(|j| j as u32).collect();
            let flip_idx = Tensor::from_vec(flip_indices, (seq_len,), m_i.device())?;
            let flipped_mask = binary_mask.get(i)?.index_select(&flip_idx, 0)?; // [L]
            let flipped_mask = flipped_mask.unsqueeze(1)?; // [L, 1]

            // blend: flipped_mask * padded + (1 - flipped_mask) * registers
            let one_minus = flipped_mask.affine(-1.0, 1.0)?;
            let blended = padded
                .to_dtype(hidden_states.dtype())?
                .broadcast_mul(&flipped_mask.to_dtype(hidden_states.dtype())?)?
                .broadcast_add(
                    &registers.broadcast_mul(&one_minus.to_dtype(hidden_states.dtype())?)?
                )?;

            batch_results.push(blended.unsqueeze(0)?);
        }

        let result = Tensor::cat(&batch_results, 0)?;

        // With registers, attention mask becomes all-zeros (all tokens attend)
        let new_mask = Tensor::zeros_like(mask)?;

        Ok((result, Some(new_mask)))
    }
}

/// Full LTX-2 text connectors module.
///
/// Matches HF `LTX2TextConnectors`:
/// - text_proj_in: Linear(caption_channels * text_proj_in_factor, caption_channels, bias=False)
/// - video_connector: ConnectorTransformer1d
/// - audio_connector: ConnectorTransformer1d
#[derive(Debug)]
pub struct Ltx2TextConnectors {
    /// Input projection (LTX-2: text_proj_in, LTX-2.3: feature_extractor.video_aggregate_embed)
    text_proj_in: Linear,
    video_connector: ConnectorTransformer1d,
    #[allow(dead_code)]
    audio_connector: Option<ConnectorTransformer1d>,
}

impl Ltx2TextConnectors {
    pub fn new(config: &Ltx2ConnectorConfig, has_audio: bool, vb: VarBuilder) -> Result<Self> {
        let text_dim = config.caption_channels; // 3840
        let proj_in_dim = text_dim * config.text_proj_in_factor; // 3840 * 49 = 188160
        let gated = config.gated_attention;

        // Input projection: packed Gemma tokens → output dim
        // LTX-2: text_proj_in (3840*49 → 3840, no bias)
        // LTX-2.3: feature_extractor.video_aggregate_embed (3840*49 → 4096, with bias)
        let text_proj_in = if config.has_feature_extractor {
            let out_dim = if config.feature_extractor_out_dim > 0 {
                config.feature_extractor_out_dim // LTX-2.3: 4096
            } else {
                config.video_inner_dim() // fallback: 3840
            };
            candle_nn::linear(proj_in_dim, out_dim, vb.pp("feature_extractor.video_aggregate_embed"))?
        } else {
            candle_nn::linear_no_bias(proj_in_dim, text_dim, vb.pp("text_proj_in"))?
        };

        let video_connector = ConnectorTransformer1d::new(
            config.video_connector_num_layers,
            config.video_connector_num_learnable_registers,
            config.video_connector_num_attention_heads,
            config.video_connector_attention_head_dim,
            1e-6,
            config.rope_theta,
            config.connector_rope_base_seq_len,
            gated,
            vb.pp("video_connector"),
        )?;

        let audio_connector = if has_audio {
            Some(ConnectorTransformer1d::new(
                config.audio_connector_num_layers,
                config.audio_connector_num_learnable_registers,
                config.audio_connector_num_attention_heads,
                config.audio_connector_attention_head_dim,
                1e-6,
                config.rope_theta,
                config.connector_rope_base_seq_len,
                gated,
                vb.pp("audio_connector"),
            )?)
        } else {
            None
        };

        Ok(Self {
            text_proj_in,
            video_connector,
            audio_connector,
        })
    }

    /// Process packed Gemma embeddings into video context tokens.
    ///
    /// `text_embeds`: `[B, L, caption_channels * text_proj_in_factor]` — packed Gemma output
    /// `attention_mask`: `[B, L]` — binary mask (1=valid, 0=padding)
    ///
    /// Returns `(video_embeddings, attention_mask)`:
    /// - `video_embeddings`: `[B, L, caption_channels]`
    /// - `attention_mask`: `[B, L]`
    pub fn forward_video(
        &self,
        text_embeds: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Convert binary mask to additive format: (mask - 1) * finfo.max
        let additive_mask = attention_mask.map(|m| {
            let text_dtype = text_embeds.dtype();
            // (mask - 1) gives -1 for padding, 0 for valid
            let shifted = m.affine(1.0, -1.0); // 0 → -1, 1 → 0
            let max_val = match text_dtype {
                DType::F32 => f32::MAX as f64,
                DType::F16 => 65504.0,
                DType::BF16 => 3.39e38,
                _ => f32::MAX as f64,
            };
            shifted.and_then(|s| {
                let shaped = s.reshape((s.dim(0)?, 1, 1, s.dim(1)?))?;
                shaped.affine(max_val, 0.0)
            })
        }).transpose()?;

        // Project text embeddings
        let projected = self.text_proj_in.forward(text_embeds)?;

        // Run video connector
        let (video_emb, new_mask) = self.video_connector.forward(&projected, additive_mask.as_ref())?;

        // Apply output mask: zero out padded positions
        let (video_emb, out_mask) = if let Some(ref nm) = new_mask {
            // (new_mask < 1e-6) gives 1 for ~zero positions (valid after register replacement)
            let attn_mask = nm.lt(1e-6f32)?.to_dtype(DType::F32)?;
            let attn_mask = if attn_mask.rank() == 4 {
                attn_mask.squeeze(1)?.squeeze(1)?
            } else {
                attn_mask
            };
            let mask_3d = attn_mask.unsqueeze(2)?; // [B, L, 1]
            let masked_emb = video_emb.broadcast_mul(&mask_3d.to_dtype(video_emb.dtype())?)?;
            (masked_emb, Some(attn_mask))
        } else {
            (video_emb, None)
        };

        Ok((video_emb, out_mask))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_connector_transformer_1d_shapes() {
        let device = Device::Cpu;
        let b = 2;
        let seq_len = 128;
        let heads = 4;
        let d_head = 16;
        let inner_dim = heads * d_head;
        let num_registers = 64;

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let ct = ConnectorTransformer1d::new(
            2,             // num_layers
            num_registers,
            heads,
            d_head,
            1e-6,
            10000.0,       // rope_theta
            4096,          // base_seq_len
            false,         // gated
            vb,
        )
        .unwrap();

        let hidden = Tensor::randn(0f32, 1f32, (b, seq_len, inner_dim), &device).unwrap();
        // All-zeros additive mask = no masking
        let mask = Tensor::zeros((b, 1, 1, seq_len), DType::F32, &device).unwrap();
        let (out, new_mask) = ct.forward(&hidden, Some(&mask)).unwrap();

        assert_eq!(out.dims(), &[b, seq_len, inner_dim]);
        assert!(new_mask.is_some());
    }

    #[test]
    fn test_connector_transformer_1d_no_mask() {
        let device = Device::Cpu;
        let b = 1;
        let seq_len = 64;
        let heads = 2;
        let d_head = 8;
        let inner_dim = heads * d_head;

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let ct = ConnectorTransformer1d::new(1, 32, heads, d_head, 1e-6, 10000.0, 4096, false, vb)
            .unwrap();

        let hidden = Tensor::randn(0f32, 1f32, (b, seq_len, inner_dim), &device).unwrap();
        let (out, new_mask) = ct.forward(&hidden, None).unwrap();

        assert_eq!(out.dims(), &[b, seq_len, inner_dim]);
        assert!(new_mask.is_none());
    }

    #[test]
    fn test_connector_block_shapes() {
        let device = Device::Cpu;
        let dim = 32;
        let heads = 2;
        let d_head = 16;

        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let block = ConnectorBlock::new(dim, heads, d_head, 1e-6, false, vb).unwrap();

        let x = Tensor::randn(0f32, 1f32, (1, 8, dim), &device).unwrap();
        let out = block.forward(&x, None, None).unwrap();
        assert_eq!(out.dims(), x.dims());
    }
}
