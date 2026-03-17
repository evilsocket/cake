//! CLIP-L text encoder for FLUX.1-dev.
//!
//! Wraps candle's ClipTextTransformer to produce a pooled 768-dim embedding
//! from the input prompt. This embedding is used as the `vec` (y) input to
//! the FLUX transformer for global conditioning.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip::text_model::{
    Activation, ClipTextConfig, ClipTextTransformer,
};
use log::info;

/// CLIP-L configuration matching openai/clip-vit-large-patch14.
fn clip_l_config() -> ClipTextConfig {
    ClipTextConfig {
        vocab_size: 49408,
        embed_dim: 768,
        activation: Activation::QuickGelu,
        intermediate_size: 3072,
        max_position_embeddings: 77,
        pad_with: None,
        num_hidden_layers: 12,
        num_attention_heads: 12,
        projection_dim: 768,
    }
}

/// Load and run the CLIP-L encoder from a safetensors file.
/// Returns the pooled output (batch_size, 768).
pub fn encode_clip(
    checkpoint_path: &std::path::Path,
    prefix: &str,
    input_ids: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let cfg = clip_l_config();

    info!("loading CLIP-L text encoder...");
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[checkpoint_path.to_path_buf()], DType::F32, device)?
    };
    let vb = vb.pp(prefix).pp("text_model");

    let model = ClipTextTransformer::new(vb, &cfg)?;
    info!("CLIP-L loaded, encoding...");

    // CLIP forward returns pooled output at END token position
    let output = model.forward(input_ids)?;
    Ok(output)
}
