/// Tests for config.json → common Config parsing for each model architecture.

#[test]
fn test_llama3_config_parse() {
    let json = r#"{
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "tie_word_embeddings": true
    }"#;
    let cfg: cake_core::models::llama3::config::LlamaConfig = serde_json::from_str(json).unwrap();
    let common = cfg.into_config();
    assert_eq!(common.hidden_size, 2048);
    assert_eq!(common.num_attention_heads, 32);
    assert_eq!(common.num_key_value_heads, 8);
    assert!(common.tie_word_embeddings);
    assert!(!common.use_qk_norm);
}

#[test]
fn test_qwen2_config_parse() {
    let json = r#"{
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "num_hidden_layers": 28,
        "num_attention_heads": 12,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "tie_word_embeddings": true,
        "use_sliding_window": false,
        "sliding_window": 32768
    }"#;
    let cfg: cake_core::models::qwen2::config::QwenConfig = serde_json::from_str(json).unwrap();
    let common = cfg.into_config();
    assert_eq!(common.hidden_size, 1536);
    assert!(common.use_qkv_bias);
    assert_eq!(common.num_key_value_heads, 2);
}

#[test]
fn test_qwen3_config_parse() {
    let json = r#"{
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "vocab_size": 151936,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 40960,
        "head_dim": 128
    }"#;
    let cfg: cake_core::models::qwen3::config::Qwen3Config = serde_json::from_str(json).unwrap();
    let common = cfg.into_config();
    assert_eq!(common.hidden_size, 2048);
    assert!(common.use_qk_norm);
    assert_eq!(common.head_dim, Some(128));
}

#[test]
fn test_phi4_config_parse() {
    let json = r#"{
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "vocab_size": 100352,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 250000.0,
        "max_position_embeddings": 16384,
        "original_max_position_embeddings": 4096,
        "partial_rotary_factor": 0.75
    }"#;
    let cfg: cake_core::models::phi4::config::Phi4Config = serde_json::from_str(json).unwrap();
    let common = cfg.into_config();
    assert_eq!(common.hidden_size, 3072);
    assert!(common.fused_qkv_proj);
    assert!(common.fused_gate_up_proj);
    assert!((common.partial_rotary_factor - 0.75).abs() < 1e-6);
}

#[test]
fn test_mistral_config_parse() {
    let json = r#"{
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "vocab_size": 32768,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 1000000.0,
        "max_position_embeddings": 32768,
        "sliding_window": 4096
    }"#;
    let cfg: cake_core::models::mistral::config::MistralConfig =
        serde_json::from_str(json).unwrap();
    let common = cfg.into_config();
    assert_eq!(common.hidden_size, 4096);
    assert_eq!(common.sliding_window, Some(4096));
}

#[test]
fn test_detect_text_model_arch() {
    use cake_core::models::common::detect_text_model_arch;
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"architectures": ["LlamaForCausalLM"]}"#,
    )
    .unwrap();
    let arch = detect_text_model_arch(&config_path).unwrap();
    assert_eq!(arch, "LlamaForCausalLM");
}
