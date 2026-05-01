//! MLX 4-bit quantization detection.
//!
//! MLX-community models store linear-layer weights as three tensors:
//!   - `*.weight` — uint32, shape `(rows, cols / 8)`: 8 x 4-bit values packed per int32
//!   - `*.scales` — f16,   shape `(rows, groups)`: one scale per group
//!   - `*.biases` — f16,   shape `(rows, groups)`: one bias per group
//!
//! The config.json uses a `"quantization"` key (not `"quantization_config"`):
//!   `{"quantization": {"group_size": 64, "bits": 4}}`
//!
//! Dequantization formula: `w_dequant[i, j] = w4(i, j) * scale(i, group(j)) + bias(i, group(j))`
//!
//! This is identical to the affine 4-bit path in `gptq::dequantize_packed_4bit`,
//! so we reuse the GPTQ backend for actual dequantization.

use std::path::Path;

/// Check whether a model uses MLX-style packed quantization.
///
/// Detects the `"quantization"` key (MLX convention) with `bits: 4` and no
/// `quant_method` field (which would indicate GPTQ/FP8 instead).
/// Also catches `"quantization_config"` entries that lack `quant_method` and `mode`.
pub fn is_mlx_quantized(config_path: &Path) -> bool {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return false;
    };

    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        // Primary MLX key: "quantization" (not "quantization_config")
        if let Some(q) = root.get("quantization") {
            if is_mlx_quant_block(q) {
                return true;
            }
        }
        // Some MLX models also populate "quantization_config" with the same data.
        // Catch those that slipped past GPTQ detection (no quant_method, no mode).
        if let Some(qc) = root.get("quantization_config") {
            if is_mlx_quant_block(qc) {
                return true;
            }
        }
    }
    false
}

/// Returns true if the JSON block looks like the implemented MLX quantization:
/// has `bits: 4`, has `group_size`, and does NOT have `quant_method`.
fn is_mlx_quant_block(qc: &serde_json::Value) -> bool {
    // Must not have quant_method — that's GPTQ or FP8
    if qc.get("quant_method").is_some() {
        return false;
    }
    // Only detect bit widths we actually implement dequant for.
    // dequantize_packed_4bit assumes 8 × 4-bit values per uint32.
    let has_bits = qc
        .get("bits")
        .and_then(|b| b.as_u64())
        .map(|b| b == 4)
        .unwrap_or(false);
    let has_group_size = qc.get("group_size").is_some();
    has_bits && has_group_size
}

/// Read the group_size from MLX quantization config (defaults to 64).
pub fn mlx_group_size(config_path: &Path) -> usize {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return 64;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return 64;
    };
    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        for key in ["quantization", "quantization_config"] {
            if let Some(gs) = root
                .get(key)
                .and_then(|q| q.get("group_size"))
                .and_then(|v| v.as_u64())
            {
                return gs as usize;
            }
        }
    }
    64
}

/// Read the bits from MLX quantization config (defaults to 4).
pub fn mlx_bits(config_path: &Path) -> usize {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return 4;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return 4;
    };
    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        for key in ["quantization", "quantization_config"] {
            if let Some(bits) = root
                .get(key)
                .and_then(|q| q.get("bits"))
                .and_then(|v| v.as_u64())
            {
                return bits as usize;
            }
        }
    }
    4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detects_mlx_quantization_key() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(&cfg, r#"{"quantization": {"group_size": 64, "bits": 4}}"#).unwrap();
        assert!(is_mlx_quantized(&cfg));
    }

    #[test]
    fn test_detects_mlx_quantization_config_key() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"quantization_config": {"group_size": 64, "bits": 4}}"#,
        )
        .unwrap();
        assert!(is_mlx_quantized(&cfg));
    }

    #[test]
    fn test_rejects_gptq_quant_method() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}}"#,
        )
        .unwrap();
        assert!(!is_mlx_quantized(&cfg));
    }

    #[test]
    fn test_rejects_fp8_quant_method() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"quantization_config": {"quant_method": "fp8", "bits": 8, "group_size": 128}}"#,
        )
        .unwrap();
        assert!(!is_mlx_quantized(&cfg));
    }

    #[test]
    fn test_rejects_no_quantization() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(&cfg, r#"{"hidden_size": 4096}"#).unwrap();
        assert!(!is_mlx_quantized(&cfg));
    }

    #[test]
    fn test_rejects_3bit_mlx() {
        // 3-bit packing is not implemented — dequant assumes 8 × 4-bit per uint32
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(&cfg, r#"{"quantization": {"group_size": 64, "bits": 3}}"#).unwrap();
        assert!(!is_mlx_quantized(&cfg));
    }

    #[test]
    fn test_group_size_from_quantization_key() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(&cfg, r#"{"quantization": {"group_size": 32, "bits": 4}}"#).unwrap();
        assert_eq!(mlx_group_size(&cfg), 32);
    }

    #[test]
    fn test_group_size_default() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(&cfg, r#"{"hidden_size": 4096}"#).unwrap();
        assert_eq!(mlx_group_size(&cfg), 64);
    }

    #[test]
    fn test_bits_from_config() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(&cfg, r#"{"quantization": {"group_size": 64, "bits": 3}}"#).unwrap();
        assert_eq!(mlx_bits(&cfg), 3);
    }

    #[test]
    fn test_nested_text_config() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"text_config": {"quantization": {"group_size": 64, "bits": 4}}}"#,
        )
        .unwrap();
        assert!(is_mlx_quantized(&cfg));
        assert_eq!(mlx_group_size(&cfg), 64);
    }

    #[test]
    fn test_with_affine_mode_still_detected() {
        // MLX models sometimes have mode: "affine" — should still be detected
        // since they don't have quant_method
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.json");
        std::fs::write(
            &cfg,
            r#"{"quantization": {"group_size": 64, "bits": 4, "mode": "affine"}}"#,
        )
        .unwrap();
        assert!(is_mlx_quantized(&cfg));
    }
}
