//! ZipformerEncoderLayer -- single Zipformer encoder layer.
//!
//! Components per layer (weight prefix: `{layer_name}.{component}`):
//! - `norm` -- BiasNorm
//! - `feed_forward1/2/3` -- FeedforwardModule (ff_dim * 3/4, ff_dim, ff_dim * 5/4)
//! - `self_attn_weights` -- RelPositionMultiheadAttentionWeights
//! - `self_attn1`, `self_attn2` -- SelfAttention (value projection)
//! - `nonlin_attention` -- NonlinAttention
//! - `conv_module1`, `conv_module2` -- ConvolutionModule
//! - `bypass`, `bypass_mid` -- BypassModule

use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

use super::bias_norm::BiasNorm;
use super::bypass_module::BypassModule;
use super::convolution_module::ConvolutionModule;
use super::feedforward::FeedforwardModule;
use super::nonlin_attention::NonlinAttention;
use super::rel_pos_attention::{RelPositionMultiheadAttentionWeights, SelfAttention};

#[derive(Debug, Clone)]
pub struct ZipformerEncoderLayer {
    norm: BiasNorm,
    feed_forward1: FeedforwardModule,
    feed_forward2: FeedforwardModule,
    feed_forward3: FeedforwardModule,
    self_attn_weights: RelPositionMultiheadAttentionWeights,
    self_attn1: SelfAttention,
    self_attn2: SelfAttention,
    nonlin_attention: NonlinAttention,
    conv_module1: ConvolutionModule,
    conv_module2: ConvolutionModule,
    bypass: BypassModule,
    bypass_mid: BypassModule,
}

impl ZipformerEncoderLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        dim: usize,
        ff_dim: usize,
        num_heads: usize,
        query_head_dim: usize,
        value_head_dim: usize,
        pos_dim: usize,
        pos_head_dim: usize,
        cnn_kernel: usize,
        vb: VarBuilder,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        let norm = BiasNorm::load(dim, vb.pp("norm"))?;

        // Three feed-forward modules with different intermediate sizes
        let ff1_dim = ff_dim * 3 / 4;
        let ff2_dim = ff_dim;
        let ff3_dim = ff_dim * 5 / 4;
        let feed_forward1 = FeedforwardModule::load(dim, ff1_dim, vb.pp("feed_forward1"), backend.clone())?;
        let feed_forward2 = FeedforwardModule::load(dim, ff2_dim, vb.pp("feed_forward2"), backend.clone())?;
        let feed_forward3 = FeedforwardModule::load(dim, ff3_dim, vb.pp("feed_forward3"), backend.clone())?;

        // Attention weights (computes Q, K, pos -> attention matrix)
        let self_attn_weights = RelPositionMultiheadAttentionWeights::load(
            dim,
            num_heads,
            query_head_dim,
            pos_head_dim,
            pos_dim,
            vb.pp("self_attn_weights"),
            backend.clone(),
        )?;

        // Two self-attention modules (apply weights to values)
        let self_attn1 = SelfAttention::load(dim, num_heads, value_head_dim, vb.pp("self_attn1"), backend.clone())?;
        let self_attn2 = SelfAttention::load(dim, num_heads, value_head_dim, vb.pp("self_attn2"), backend.clone())?;

        // Nonlinear attention
        let nonlin_attention = NonlinAttention::load(dim, num_heads, vb.pp("nonlin_attention"), backend.clone())?;

        // Two convolution modules
        let conv_module1 = ConvolutionModule::load(dim, cnn_kernel, vb.pp("conv_module1"), backend.clone())?;
        let conv_module2 = ConvolutionModule::load(dim, cnn_kernel, vb.pp("conv_module2"), backend)?;

        // Bypass modules (per-channel scale)
        let bypass = BypassModule::load_dim(dim, vb.pp("bypass"))?;
        let bypass_mid = BypassModule::load_dim(dim, vb.pp("bypass_mid"))?;

        Ok(Self {
            norm,
            feed_forward1,
            feed_forward2,
            feed_forward3,
            self_attn_weights,
            self_attn1,
            self_attn2,
            nonlin_attention,
            conv_module1,
            conv_module2,
            bypass,
            bypass_mid,
        })
    }

    /// Forward pass through the Zipformer encoder layer.
    ///
    /// `pos_emb`: relative position embeddings [1, 2*seq-1, pos_dim]
    /// `time_emb`: optional time embedding [1, 1, dim] to add at specific points
    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor, time_emb: Option<&Tensor>) -> Result<Tensor> {
        let src_orig = x.clone();

        // Compute attention weights once (shared between self_attn1, self_attn2, nonlin_attn)
        let attn_weights = self.self_attn_weights.forward(x, pos_emb)?;

        // Add time embedding
        let mut src = if let Some(te) = time_emb {
            x.broadcast_add(te)?
        } else {
            x.clone()
        };

        // feed_forward1
        let ff1_out = self.feed_forward1.forward(&src)?;
        src = (&src + &ff1_out)?;

        // nonlin_attention (uses first head of attention weights)
        let na_weights = attn_weights.narrow(1, 0, 1)?; // [batch, 1, seq, seq]
        let na_out = self.nonlin_attention.forward(&src, &na_weights)?;
        src = (&src + &na_out)?;

        // self_attn1
        let sa1_out = self.self_attn1.forward(&src, &attn_weights)?;
        src = (&src + &sa1_out)?;

        // Add time embedding again before conv1
        if let Some(te) = time_emb {
            src = src.broadcast_add(te)?;
        }

        // conv_module1
        let conv1_out = self.conv_module1.forward(&src)?;
        src = (&src + &conv1_out)?;

        // feed_forward2
        let ff2_out = self.feed_forward2.forward(&src)?;
        src = (&src + &ff2_out)?;

        // bypass_mid (mid-layer bypass)
        src = self.bypass_mid.forward(&src_orig, &src)?;

        // self_attn2
        let sa2_out = self.self_attn2.forward(&src, &attn_weights)?;
        src = (&src + &sa2_out)?;

        // Add time embedding again before conv2
        if let Some(te) = time_emb {
            src = src.broadcast_add(te)?;
        }

        // conv_module2
        let conv2_out = self.conv_module2.forward(&src)?;
        src = (&src + &conv2_out)?;

        // feed_forward3
        let ff3_out = self.feed_forward3.forward(&src)?;
        src = (&src + &ff3_out)?;

        // BiasNorm at end
        src = self.norm.forward(&src)?;

        // Final bypass
        src = self.bypass.forward(&src_orig, &src)?;

        Ok(src)
    }

}
