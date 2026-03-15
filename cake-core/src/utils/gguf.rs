//! GGUF model loading support.
//!
//! Loads quantized GGUF files, dequantizes tensors to the target dtype,
//! and remaps GGUF tensor names to HuggingFace-style names so that
//! existing model code (LLaMA, Qwen2, etc.) works unchanged.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

/// Remap a GGUF tensor name to HuggingFace-style name.
///
/// GGUF (llama.cpp) uses names like `blk.0.attn_q.weight`,
/// HuggingFace uses `model.layers.0.self_attn.q_proj.weight`.
fn remap_gguf_name(name: &str, prefix: &str) -> String {
    // Non-layer tensors
    if name == "token_embd.weight" {
        return format!("{prefix}.embed_tokens.weight");
    }
    if name == "output_norm.weight" {
        return format!("{prefix}.norm.weight");
    }
    if name == "output.weight" {
        return "lm_head.weight".to_string();
    }

    // Block-level tensors: blk.{i}.{component}.weight
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_idx = &rest[..dot_pos];
            let component = &rest[dot_pos + 1..];

            let hf_component = match component {
                // Attention
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                // MLP
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                // Norms
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                // Qwen-specific (QKV bias)
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.bias" => "self_attn.v_proj.bias",
                // Pass through unknown components
                other => return format!("{prefix}.layers.{layer_idx}.{other}"),
            };

            return format!("{prefix}.layers.{layer_idx}.{hf_component}");
        }
    }

    // Unknown: pass through unchanged
    name.to_string()
}

/// Load a GGUF file and return a standard VarBuilder with dequantized tensors.
///
/// All quantized tensors are dequantized to `dtype` and placed on `device`.
/// Tensor names are remapped from GGUF conventions to HuggingFace conventions
/// using the given `model_prefix` (e.g., "model" for LLaMA, "model.language_model"
/// for Qwen3.5).
pub fn load_var_builder_from_gguf<'a>(
    gguf_path: &Path,
    dtype: DType,
    device: Device,
    model_prefix: &str,
) -> Result<VarBuilder<'a>> {
    log::info!("loading GGUF model from {} ...", gguf_path.display());

    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| anyhow!("can't open GGUF file {}: {e}", gguf_path.display()))?;

    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("can't parse GGUF file {}: {e}", gguf_path.display()))?;

    log::info!(
        "GGUF: {} tensors, {} metadata entries",
        content.tensor_infos.len(),
        content.metadata.len(),
    );

    // Log useful metadata
    for key in ["general.architecture", "general.name", "general.quantization_version"] {
        if let Some(val) = content.metadata.get(key) {
            log::info!("  {}: {:?}", key, val);
        }
    }

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let start = std::time::Instant::now();

    for tensor_name in content.tensor_infos.keys() {
        let qtensor = content
            .tensor(&mut file, tensor_name, &device)
            .map_err(|e| anyhow!("can't load GGUF tensor '{}': {e}", tensor_name))?;

        // Dequantize to target dtype
        let tensor = if dtype == DType::F16 {
            qtensor
                .dequantize_f16(&device)
                .map_err(|e| anyhow!("can't dequantize_f16 '{}': {e}", tensor_name))?
        } else {
            qtensor
                .dequantize(&device)
                .map_err(|e| anyhow!("can't dequantize '{}': {e}", tensor_name))?
                .to_dtype(dtype)
                .map_err(|e| anyhow!("can't cast '{}' to {:?}: {e}", tensor_name, dtype))?
        };

        let hf_name = remap_gguf_name(tensor_name, model_prefix);
        log::debug!("  {} → {} {:?}", tensor_name, hf_name, tensor.shape());
        tensors.insert(hf_name, tensor);
    }

    log::info!(
        "GGUF: loaded and dequantized {} tensors in {:.1}s",
        tensors.len(),
        start.elapsed().as_secs_f64(),
    );

    Ok(VarBuilder::from_tensors(tensors, dtype, &device))
}

/// Detect GGUF file(s) in a model directory.
/// Returns the path to the first `.gguf` file found, or None.
pub fn detect_gguf_file(model_dir: &Path) -> Option<std::path::PathBuf> {
    if model_dir.is_file() && model_dir.extension().map_or(false, |ext| ext == "gguf") {
        return Some(model_dir.to_path_buf());
    }

    if model_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().map_or(false, |ext| ext == "gguf") {
                    return Some(path);
                }
            }
        }
    }

    None
}

/// Extract the model architecture string from GGUF metadata.
/// Returns e.g. "llama", "qwen2", etc.
pub fn detect_architecture_from_gguf(gguf_path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(gguf_path)
        .map_err(|e| anyhow!("can't open GGUF file: {e}"))?;

    let content = candle_core::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow!("can't parse GGUF file: {e}"))?;

    if let Some(val) = content.metadata.get("general.architecture") {
        Ok(format!("{:?}", val).trim_matches('"').to_string())
    } else {
        bail!("GGUF file missing general.architecture metadata")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_gguf_names_llama() {
        let prefix = "model";

        assert_eq!(
            remap_gguf_name("token_embd.weight", prefix),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            remap_gguf_name("output_norm.weight", prefix),
            "model.norm.weight"
        );
        assert_eq!(remap_gguf_name("output.weight", prefix), "lm_head.weight");

        assert_eq!(
            remap_gguf_name("blk.0.attn_q.weight", prefix),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            remap_gguf_name("blk.15.attn_output.weight", prefix),
            "model.layers.15.self_attn.o_proj.weight"
        );
        assert_eq!(
            remap_gguf_name("blk.3.ffn_gate.weight", prefix),
            "model.layers.3.mlp.gate_proj.weight"
        );
        assert_eq!(
            remap_gguf_name("blk.7.attn_norm.weight", prefix),
            "model.layers.7.input_layernorm.weight"
        );
        assert_eq!(
            remap_gguf_name("blk.7.ffn_norm.weight", prefix),
            "model.layers.7.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_remap_gguf_names_qwen3_5() {
        let prefix = "model.language_model";

        assert_eq!(
            remap_gguf_name("token_embd.weight", prefix),
            "model.language_model.embed_tokens.weight"
        );
        assert_eq!(
            remap_gguf_name("blk.0.attn_q.weight", prefix),
            "model.language_model.layers.0.self_attn.q_proj.weight"
        );
    }

    #[test]
    fn test_remap_unknown_passthrough() {
        assert_eq!(
            remap_gguf_name("some.unknown.tensor", "model"),
            "some.unknown.tensor"
        );
    }
}
