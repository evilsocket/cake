//! TTSZipformer text encoder -- always runs on master, not sharded.
//!
//! Weights:
//! - `embed.weight` [vocab_size, dim]
//! - `text_encoder.in_proj` [dim, dim]
//! - `text_encoder.layers.{i}.{component}` -- Zipformer layers
//! - `text_encoder.out_proj` [feat_dim, dim]

use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use super::config::LuxTTSConfig;
use super::zipformer_layer::ZipformerEncoderLayer;

#[derive(Debug, Clone)]
pub struct TextEncoder {
    embed_weight: Tensor,
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    layers: Vec<ZipformerEncoderLayer>,
    out_proj_weight: Tensor,
    out_proj_bias: Option<Tensor>,
    #[allow(dead_code)]
    dim: usize,
    pos_dim: usize,
    backend: Arc<dyn crate::backends::ComputeBackend>,
}

impl TextEncoder {
    pub fn load(config: &LuxTTSConfig, embed_vb: VarBuilder, enc_vb: VarBuilder, backend: Arc<dyn crate::backends::ComputeBackend>) -> Result<Self> {
        let m = &config.model;
        let dim = m.text_encoder_dim;

        // embed.weight is at top level
        let embed_weight = embed_vb.pp("embed").get((m.vocab_size, dim), "weight")?;

        // text_encoder.in_proj
        let in_proj_weight = enc_vb.pp("in_proj").get((dim, dim), "weight")?;
        let in_proj_bias = Some(enc_vb.pp("in_proj").get(dim, "bias")?);

        // text_encoder.layers
        let mut layers = Vec::new();
        for i in 0..m.text_encoder_num_layers {
            let layer = ZipformerEncoderLayer::load(
                dim,
                m.text_encoder_feedforward_dim,
                m.text_encoder_num_heads,
                m.query_head_dim,
                m.value_head_dim,
                m.pos_dim,
                m.pos_head_dim,
                m.text_encoder_cnn_module_kernel,
                enc_vb.pp(format!("layers.{i}")),
                backend.clone(),
            )?;
            layers.push(layer);
        }

        // text_encoder.out_proj
        let out_proj_weight = enc_vb.pp("out_proj").get((m.feat_dim, dim), "weight")?;
        let out_proj_bias = Some(enc_vb.pp("out_proj").get(m.feat_dim, "bias")?);

        Ok(Self {
            embed_weight,
            in_proj_weight,
            in_proj_bias,
            layers,
            out_proj_weight,
            out_proj_bias,
            dim,
            pos_dim: m.pos_dim,
            backend,
        })
    }

    /// Forward pass: token_ids [batch, seq] -> [batch, seq, feat_dim].
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        let x = self.backend.embedding(token_ids, &self.embed_weight)?;
        let x = self.backend.linear_forward(&x, &self.in_proj_weight, self.in_proj_bias.as_ref())?;

        let seq_len = x.dim(1)?;
        let pos_emb = self.make_pos_emb(seq_len, x.device(), x.dtype())?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &pos_emb, None)?;
        }

        let x = self.backend.linear_forward(&x, &self.out_proj_weight, self.out_proj_bias.as_ref())?;
        Ok(x)
    }

    /// Create CompactRelPositionalEncoding [1, 2*seq-1, pos_dim].
    fn make_pos_emb(&self, seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor> {
        let pos_len = 2 * seq_len - 1;
        let half_dim = self.pos_dim / 2;
        let compression_length = (self.pos_dim as f32).sqrt();
        let length_scale = 1.0 * self.pos_dim as f32 / (2.0 * std::f32::consts::PI);

        let mut pos_data = vec![0.0f32; pos_len * self.pos_dim];
        for pos in 0..pos_len {
            let t = pos as f32 - (seq_len as f32 - 1.0);
            let x_compressed = compression_length
                * t.signum()
                * ((t.abs() + compression_length).ln() - compression_length.ln());
            let x_atan = (x_compressed / length_scale).atan();

            for i in 0..half_dim {
                let freq = (i + 1) as f32;
                pos_data[pos * self.pos_dim + 2 * i] = (x_atan * freq).cos();
                pos_data[pos * self.pos_dim + 2 * i + 1] = (x_atan * freq).sin();
            }
            pos_data[pos * self.pos_dim + self.pos_dim - 1] = 1.0;
        }

        let pos_emb = Tensor::from_vec(pos_data, (1, pos_len, self.pos_dim), device)?;
        Ok(pos_emb.to_dtype(dtype)?)
    }
}
