//! T5-XXL text encoder for FLUX.1-dev.
//!
//! Wraps candle's T5EncoderModel to produce token-level embeddings (seq_len × 4096)
//! from the input prompt. These are used as the `txt` context input to the
//! FLUX transformer for cross-attention.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::t5::{self, T5EncoderModel};
use log::info;

/// T5-XXL v1.1 configuration.
fn t5_xxl_config() -> t5::Config {
    t5::Config {
        vocab_size: 32128,
        d_model: 4096,
        d_kv: 64,
        d_ff: 10240,
        num_layers: 24,
        num_decoder_layers: None,
        num_heads: 64,
        relative_attention_num_buckets: 32,
        relative_attention_max_distance: 128,
        dropout_rate: 0.0,
        layer_norm_epsilon: 1e-6,
        initializer_factor: 1.0,
        feed_forward_proj: t5::ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::NewGelu,
        },
        tie_word_embeddings: false,
        is_decoder: false,
        is_encoder_decoder: true,
        use_cache: false,
        pad_token_id: 0,
        eos_token_id: 1,
        decoder_start_token_id: None,
    }
}

/// Load and run the T5-XXL encoder from a safetensors file.
/// Returns hidden states (batch_size, seq_len, 4096).
///
/// FP8 weights are dequantized to BF16 at load time (not F32) to stay within
/// ~10GB RAM instead of ~19GB for F32.
pub fn encode_t5(
    checkpoint_path: &std::path::Path,
    prefix: &str,
    input_ids: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let cfg = t5_xxl_config();

    info!("loading T5-XXL text encoder (BF16 on CPU, ~10GB RAM)...");
    // Dequantize F8E4M3 → BF16 on CPU (~10GB vs ~19GB for F32)
    let vb = unsafe {
        let filenames = vec![checkpoint_path.to_path_buf()];
        crate::utils::fp8::load_fp8_var_builder(&filenames, DType::BF16, device)?
    };
    let vb = vb.pp(prefix);

    let mut model = T5EncoderModel::load(vb, &cfg)?;
    info!("T5-XXL loaded, encoding...");

    // Forward in BF16, then cast output to F32 for downstream use
    let output = model.forward_dt(input_ids, Some(DType::F32))?;
    Ok(output)
}
