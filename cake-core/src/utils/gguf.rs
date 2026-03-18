//! GGUF model loading support.
//!
//! Loads GGUF quantized models (Q4_K_M, Q8_0, etc.) by dequantizing weights
//! at load time, following the same pattern as the GPTQ and FP8 backends.
//! Supports LLaMA and Qwen2 architectures (standard separate Q/K/V projections).
//!
//! GGUF tensor names follow the llama.cpp convention (`blk.{N}.attn_q.weight`)
//! and are mapped to HuggingFace names (`model.layers.{N}.self_attn.q_proj.weight`)
//! transparently by the backend.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};

/// Check whether a path points to a GGUF file.
pub fn is_gguf_file(path: &Path) -> bool {
    path.extension().is_some_and(|ext| ext == "gguf")
}

/// Map a GGUF tensor name to the HuggingFace convention.
///
/// GGUF uses `blk.{N}.attn_q.weight` style; HF uses
/// `model.layers.{N}.self_attn.q_proj.weight` style.
fn gguf_name_to_hf(name: &str, model_prefix: &str) -> String {
    // Embedding / head / final norm
    if name == "token_embd.weight" {
        return format!("{model_prefix}.embed_tokens.weight");
    }
    if name == "output.weight" {
        return "lm_head.weight".to_string();
    }
    if name == "output_norm.weight" {
        return format!("{model_prefix}.norm.weight");
    }

    // Block tensors: blk.{N}.{suffix}
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_num = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let prefix = format!("{model_prefix}.layers.{layer_num}");

            // Attention projections
            let mapped = match suffix {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.bias" => "self_attn.o_proj.bias",
                // QK norm
                "attn_q_norm.weight" => "self_attn.q_norm.weight",
                "attn_k_norm.weight" => "self_attn.k_norm.weight",
                // Norms
                "attn_norm.weight" => "input_layernorm.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "post_attention_norm.weight" => "post_attention_layernorm.weight",
                // MLP
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                // If no mapping found, keep original suffix
                other => return format!("{prefix}.{other}"),
            };
            return format!("{prefix}.{mapped}");
        }
    }

    // No mapping — return as-is
    name.to_string()
}

/// Extract a `Config` from GGUF metadata.
pub fn config_from_gguf(
    path: &Path,
) -> anyhow::Result<crate::models::common::Config> {
    use candle_core::quantized::gguf_file;

    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF: {e}"))?;

    let arch = get_str(&content, "general.architecture")
        .unwrap_or_else(|| "llama".to_string());

    let hidden_size = get_u32(&content, &format!("{arch}.embedding_length")).unwrap_or(4096) as usize;
    let intermediate_size =
        get_u32(&content, &format!("{arch}.feed_forward_length")).unwrap_or(11008) as usize;
    let num_hidden_layers =
        get_u32(&content, &format!("{arch}.block_count")).unwrap_or(32) as usize;
    let num_attention_heads =
        get_u32(&content, &format!("{arch}.attention.head_count")).unwrap_or(32) as usize;
    let num_key_value_heads =
        get_u32(&content, &format!("{arch}.attention.head_count_kv")).unwrap_or(num_attention_heads as u32) as usize;
    let rms_norm_eps = get_f32(
        &content,
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
    )
    .unwrap_or(1e-5) as f64;
    let rope_theta =
        get_f32(&content, &format!("{arch}.rope.freq_base")).unwrap_or(10000.0);
    let max_seq_len =
        get_u32(&content, &format!("{arch}.context_length")).unwrap_or(4096) as usize;
    let vocab_size =
        get_u32(&content, &format!("{arch}.vocab_size"))
            .or_else(|| {
                // Fallback: count from tokenizer tokens
                content.tensor_infos.get("token_embd.weight")
                    .map(|ti| ti.shape.dims()[0] as u32)
            })
            .unwrap_or(32000) as usize;

    // Detect QKV bias from tensor presence
    let has_qkv_bias = content.tensor_infos.contains_key("blk.0.attn_q.bias");

    Ok(crate::models::common::Config {
        hidden_size,
        intermediate_size,
        vocab_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        rms_norm_eps,
        rope_theta,
        bos_token_id: None,
        eos_token_id: None,
        rope_scaling: None,
        tie_word_embeddings: !content.tensor_infos.contains_key("output.weight"),
        max_seq_len,
        use_qkv_bias: has_qkv_bias,
        model_prefix: "model".into(),
        head_dim: None,
        partial_rotary_factor: 1.0,
        linear_attn: None,
        residual_rms_norm: false,
        use_qk_norm: content.tensor_infos.contains_key("blk.0.attn_q_norm.weight"),
        pre_reshape_qk_norm: false,
        sliding_window: None,
        fused_qkv_proj: false,
        fused_gate_up_proj: false,
        global_layers: vec![],
        use_gelu_mlp: false,
        embed_scale: None,
        moe_intermediate_size: None,
        num_experts: 0,
        num_experts_per_tok: 0,
        norm_topk_prob: false,
        shared_expert_intermediate_size: None,
        attn_output_gate: false,
    })
}

/// Detect the architecture string from GGUF metadata and map to TextModelArch.
pub fn arch_from_gguf(path: &Path) -> anyhow::Result<String> {
    use candle_core::quantized::gguf_file;

    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF: {e}"))?;

    let arch = get_str(&content, "general.architecture")
        .unwrap_or_else(|| "llama".to_string());

    // Map GGUF arch names to HF architecture strings
    let hf_arch = match arch.as_str() {
        "llama" => "LlamaForCausalLM",
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" => "Qwen3ForCausalLM",
        "mistral" => "MistralForCausalLM",
        "gemma" | "gemma2" | "gemma3" => "Gemma3ForCausalLM",
        "phi3" | "phi4" => "Phi4ForCausalLM",
        "falcon" => "FalconForCausalLM",
        other => {
            log::warn!("unknown GGUF architecture '{other}', defaulting to LlamaForCausalLM");
            "LlamaForCausalLM"
        }
    };
    Ok(hf_arch.to_string())
}

/// Custom VarBuilder backend that loads and dequantizes GGUF tensors.
///
/// All GGUF tensors are dequantized at load time and stored in a HashMap
/// keyed by HuggingFace-style tensor names.
struct GgufBackend {
    tensors: HashMap<String, Tensor>,
}

impl SimpleBackend for GgufBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self
            .tensors
            .get(name)
            .ok_or_else(|| {
                candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .to_dtype(dtype)?
            .to_device(dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        self.tensors
            .get(name)
            .ok_or_else(|| {
                candle_core::Error::CannotFindTensor {
                    path: name.to_string(),
                }
                .bt()
            })?
            .to_dtype(dtype)?
            .to_device(dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// Load a GGUF file and create a VarBuilder that serves dequantized tensors.
///
/// All quantized tensors are dequantized to F32 on CPU at load time, then
/// cast to the requested dtype and moved to the device on demand.
pub fn load_gguf_var_builder<'a>(
    path: &Path,
    dtype: DType,
    device: &Device,
    model_prefix: &str,
) -> anyhow::Result<VarBuilder<'a>> {
    use candle_core::quantized::gguf_file;

    let mut file = std::fs::File::open(path)?;
    let content = gguf_file::Content::read(&mut file)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF: {e}"))?;

    let n_tensors = content.tensor_infos.len();
    let mut tensors = HashMap::with_capacity(n_tensors);
    let mut q_count = 0;

    for name in content.tensor_infos.keys() {
        let qtensor = content
            .tensor(&mut file, name, &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("failed to read tensor '{name}': {e}"))?;

        let is_quantized = !matches!(
            qtensor.dtype(),
            candle_core::quantized::GgmlDType::F32
                | candle_core::quantized::GgmlDType::F16
        );
        if is_quantized {
            q_count += 1;
        }

        // Dequantize to F32 on CPU
        let tensor = qtensor
            .dequantize(&Device::Cpu)
            .map_err(|e| anyhow::anyhow!("failed to dequantize '{name}': {e}"))?;

        let hf_name = gguf_name_to_hf(name, model_prefix);
        tensors.insert(hf_name, tensor);
    }

    log::info!(
        "GGUF loaded: {} tensors ({} quantized), dequantized to F32",
        n_tensors,
        q_count,
    );

    let backend: Box<dyn SimpleBackend> = Box::new(GgufBackend { tensors });
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

// ── Metadata helpers ─────────────────────────────────────────────────

fn get_str(
    content: &candle_core::quantized::gguf_file::Content,
    key: &str,
) -> Option<String> {
    use candle_core::quantized::gguf_file::Value;
    match content.metadata.get(key)? {
        Value::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn get_u32(
    content: &candle_core::quantized::gguf_file::Content,
    key: &str,
) -> Option<u32> {
    use candle_core::quantized::gguf_file::Value;
    match content.metadata.get(key)? {
        Value::U32(v) => Some(*v),
        Value::U64(v) => Some(*v as u32),
        Value::I32(v) => Some(*v as u32),
        _ => None,
    }
}

fn get_f32(
    content: &candle_core::quantized::gguf_file::Content,
    key: &str,
) -> Option<f32> {
    use candle_core::quantized::gguf_file::Value;
    match content.metadata.get(key)? {
        Value::F32(v) => Some(*v),
        Value::F64(v) => Some(*v as f32),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_name_mapping_llama() {
        assert_eq!(
            gguf_name_to_hf("token_embd.weight", "model"),
            "model.embed_tokens.weight"
        );
        assert_eq!(gguf_name_to_hf("output.weight", "model"), "lm_head.weight");
        assert_eq!(
            gguf_name_to_hf("output_norm.weight", "model"),
            "model.norm.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_q.weight", "model"),
            "model.layers.0.self_attn.q_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.15.attn_output.weight", "model"),
            "model.layers.15.self_attn.o_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.3.ffn_gate.weight", "model"),
            "model.layers.3.mlp.gate_proj.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_norm.weight", "model"),
            "model.layers.0.input_layernorm.weight"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.ffn_norm.weight", "model"),
            "model.layers.0.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn test_gguf_name_mapping_qwen2_bias() {
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_q.bias", "model"),
            "model.layers.0.self_attn.q_proj.bias"
        );
        assert_eq!(
            gguf_name_to_hf("blk.0.attn_k.bias", "model"),
            "model.layers.0.self_attn.k_proj.bias"
        );
    }

    #[test]
    fn test_is_gguf_file() {
        assert!(is_gguf_file(Path::new("/tmp/model.gguf")));
        assert!(!is_gguf_file(Path::new("/tmp/model.safetensors")));
        assert!(!is_gguf_file(Path::new("/tmp/model")));
    }
}
