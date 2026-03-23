//! ZipformerBlock -- implements `Forwarder` for FM decoder layer sharding.
//!
//! Each block wraps a single Zipformer encoder layer from the FM decoder.
//! Time embedding is packed into the tensor as the first frame, extracted
//! before the layer forward, and re-packed into the output.

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Tensor};

use crate::cake::{Context, Forwarder};
use super::config::LuxTTSConfig;
use super::zipformer_layer::ZipformerEncoderLayer;

/// A single FM decoder layer that can be distributed across workers.
#[derive(Debug, Clone)]
pub struct ZipformerBlock {
    name: String,
    layer: ZipformerEncoderLayer,
    /// Which stack this layer belongs to (0..num_stacks).
    #[allow(dead_code)]
    stack_idx: usize,
    /// Flat index across all stacks.
    #[allow(dead_code)]
    flat_idx: usize,
    /// Model dimension.
    #[allow(dead_code)]
    dim: usize,
    /// Positional encoding dimension.
    pos_dim: usize,
}

impl std::fmt::Display for ZipformerBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (local, stack={})",
            self.name, self.stack_idx
        )
    }
}

impl ZipformerBlock {
    /// Extract the flat layer index from a name like "fm_decoder.layers.3".
    fn layer_index(name: &str) -> usize {
        name.rsplit('.')
            .next()
            .and_then(|s| s.parse().ok())
            .expect("invalid layer name -- no trailing index")
    }

    fn load_luxtts_config(ctx: &Context) -> Result<LuxTTSConfig> {
        let config_path = ctx.data_path.join("config.json");
        LuxTTSConfig::from_path(&config_path)
    }
}

#[async_trait]
impl Forwarder for ZipformerBlock {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let luxtts_config = Self::load_luxtts_config(ctx)?;
        let flat_idx = Self::layer_index(&name);
        let (stack_idx, _layer_in_stack) = luxtts_config.flat_to_stack(flat_idx);

        let m = &luxtts_config.model;
        let dim = m.fm_decoder_dim;
        let ff_dim = m.fm_decoder_feedforward_dim;
        let num_heads = m.fm_decoder_num_heads;
        let cnn_kernel = m.fm_decoder_cnn_module_kernel[stack_idx];

        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);

        let layer = ZipformerEncoderLayer::load(
            dim,
            ff_dim,
            num_heads,
            m.query_head_dim,
            m.value_head_dim,
            m.pos_dim,
            m.pos_head_dim,
            cnn_kernel,
            vb,
            ctx.backend.clone(),
        )?;

        Ok(Box::new(Self {
            name,
            layer,
            stack_idx,
            flat_idx,
            dim,
            pos_dim: m.pos_dim,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        // The input tensor packs time_emb as the first frame:
        // [batch, 1 + seq_len, dim] where x[:, 0:1, :] is time_emb
        let (batch, total_seq, _) = x.dims3()?;
        let seq_len = total_seq - 1;

        // Extract time_emb (first frame) and data (rest)
        let time_emb = x.narrow(1, 0, 1)?; // [batch, 1, dim]
        let data = x.narrow(1, 1, seq_len)?; // [batch, seq_len, dim]

        // Generate relative position embeddings
        let pos_emb = self.make_pos_emb(seq_len, x.device(), x.dtype())?;

        // Apply the Zipformer layer with time embedding
        let out = self.layer.forward(&data, &pos_emb, Some(&time_emb))?;

        // Re-pack time_emb as first frame for next layer
        let result = Tensor::cat(&[&time_emb, &out], 1)?; // [batch, 1 + seq_len, dim]
        let _ = batch; // suppress unused warning
        Ok(result)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}

impl ZipformerBlock {
    /// Create CompactRelPositionalEncoding [1, 2*seq-1, pos_dim].
    /// Matches the Python implementation: log compression -> atan -> Fourier.
    fn make_pos_emb(
        &self,
        seq_len: usize,
        device: &candle_core::Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let pos_len = 2 * seq_len - 1;
        let half_dim = self.pos_dim / 2;
        let compression_length = (self.pos_dim as f32).sqrt();
        let length_scale = 1.0 * self.pos_dim as f32 / (2.0 * std::f32::consts::PI);

        let mut pos_data = vec![0.0f32; pos_len * self.pos_dim];
        for pos in 0..pos_len {
            let t = pos as f32 - (seq_len as f32 - 1.0); // -(seq-1) to +(seq-1)

            // Log compression
            let x_compressed = compression_length
                * t.signum()
                * ((t.abs() + compression_length).ln() - compression_length.ln());

            // Map to finite range via atan
            let x_atan = (x_compressed / length_scale).atan();

            for i in 0..half_dim {
                let freq = (i + 1) as f32; // freqs = 1, 2, 3, ...
                pos_data[pos * self.pos_dim + 2 * i] = (x_atan * freq).cos();
                pos_data[pos * self.pos_dim + 2 * i + 1] = (x_atan * freq).sin();
            }
            // Last element is 1.0 (bias term)
            pos_data[pos * self.pos_dim + self.pos_dim - 1] = 1.0;
        }

        let pos_emb = Tensor::from_vec(pos_data, (1, pos_len, self.pos_dim), device)?;
        Ok(pos_emb.to_dtype(dtype)?)
    }
}
