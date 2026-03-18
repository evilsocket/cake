//! LuxTTS configuration.

use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

/// Top-level LuxTTS config.
#[derive(Debug, Clone, Deserialize)]
pub struct LuxTTSConfig {
    pub model: ModelConfig,
    pub feature: FeatureConfig,
}

/// Model architecture parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    /// Downsampling factors per FM decoder stack, e.g. [1,2,4,2,1].
    pub fm_decoder_downsampling_factor: Vec<usize>,
    /// Number of layers per FM decoder stack, e.g. [2,2,4,4,4].
    pub fm_decoder_num_layers: Vec<usize>,
    /// CNN module kernel sizes per stack, e.g. [31,15,7,15,31].
    pub fm_decoder_cnn_module_kernel: Vec<usize>,
    /// Feed-forward hidden dimension in FM decoder.
    #[serde(default = "default_ff_dim")]
    pub fm_decoder_feedforward_dim: usize,
    /// Number of attention heads in FM decoder.
    #[serde(default = "default_num_heads")]
    pub fm_decoder_num_heads: usize,
    /// Model dimension of FM decoder.
    #[serde(default = "default_fm_dim")]
    pub fm_decoder_dim: usize,
    /// Number of text encoder layers.
    #[serde(default = "default_text_enc_layers")]
    pub text_encoder_num_layers: usize,
    /// Text encoder feed-forward dimension.
    #[serde(default = "default_text_enc_ff")]
    pub text_encoder_feedforward_dim: usize,
    /// Text encoder CNN kernel size.
    #[serde(default = "default_text_enc_kernel")]
    pub text_encoder_cnn_module_kernel: usize,
    /// Text encoder attention heads.
    #[serde(default = "default_num_heads")]
    pub text_encoder_num_heads: usize,
    /// Text encoder model dimension.
    #[serde(default = "default_text_enc_dim")]
    pub text_encoder_dim: usize,
    /// Query head dimension.
    #[serde(default = "default_query_head_dim")]
    pub query_head_dim: usize,
    /// Value head dimension.
    #[serde(default = "default_value_head_dim")]
    pub value_head_dim: usize,
    /// Positional head dimension.
    #[serde(default = "default_pos_head_dim")]
    pub pos_head_dim: usize,
    /// Positional encoding dimension.
    #[serde(default = "default_pos_dim")]
    pub pos_dim: usize,
    /// Time embedding dimension.
    #[serde(default = "default_time_embed_dim")]
    pub time_embed_dim: usize,
    /// Text embedding dimension.
    #[serde(default = "default_text_embed_dim")]
    pub text_embed_dim: usize,
    /// Number of mel frequency bins.
    #[serde(default = "default_feat_dim")]
    pub feat_dim: usize,
    /// Vocabulary size for phoneme tokens.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

/// Audio feature extraction parameters.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureConfig {
    /// Sample rate in Hz.
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
    /// FFT window size.
    #[serde(default = "default_n_fft")]
    pub n_fft: usize,
    /// Hop length for STFT.
    #[serde(default = "default_hop_length")]
    pub hop_length: usize,
    /// Number of mel bins.
    #[serde(default = "default_n_mels")]
    pub n_mels: usize,
}

impl LuxTTSConfig {
    pub fn from_path(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Total number of FM decoder layers across all stacks.
    pub fn total_fm_layers(&self) -> usize {
        self.model.fm_decoder_num_layers.iter().sum()
    }

    /// Map a flat layer index to (stack_idx, layer_in_stack).
    pub fn flat_to_stack(&self, flat_idx: usize) -> (usize, usize) {
        let mut remaining = flat_idx;
        for (stack_idx, &count) in self.model.fm_decoder_num_layers.iter().enumerate() {
            if remaining < count {
                return (stack_idx, remaining);
            }
            remaining -= count;
        }
        panic!("flat_idx {} out of range", flat_idx);
    }

    /// Check if a flat layer index is the first layer of its stack.
    pub fn is_stack_first(&self, flat_idx: usize) -> bool {
        let (_, layer_in_stack) = self.flat_to_stack(flat_idx);
        layer_in_stack == 0
    }

    /// Check if a flat layer index is the last layer of its stack.
    pub fn is_stack_last(&self, flat_idx: usize) -> bool {
        let (stack_idx, layer_in_stack) = self.flat_to_stack(flat_idx);
        layer_in_stack == self.model.fm_decoder_num_layers[stack_idx] - 1
    }
}

fn default_ff_dim() -> usize { 1536 }
fn default_num_heads() -> usize { 4 }
fn default_fm_dim() -> usize { 512 }
fn default_text_enc_layers() -> usize { 4 }
fn default_text_enc_ff() -> usize { 512 }
fn default_text_enc_kernel() -> usize { 9 }
fn default_text_enc_dim() -> usize { 192 }
fn default_query_head_dim() -> usize { 32 }
fn default_value_head_dim() -> usize { 12 }
fn default_pos_head_dim() -> usize { 4 }
fn default_pos_dim() -> usize { 48 }
fn default_time_embed_dim() -> usize { 192 }
fn default_text_embed_dim() -> usize { 192 }
fn default_feat_dim() -> usize { 100 }
fn default_vocab_size() -> usize { 360 }
fn default_sample_rate() -> usize { 24000 }
fn default_n_fft() -> usize { 1024 }
fn default_hop_length() -> usize { 256 }
fn default_n_mels() -> usize { 100 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_config() {
        let json = r#"{
            "model": {
                "fm_decoder_downsampling_factor": [1, 2, 4, 2, 1],
                "fm_decoder_num_layers": [2, 2, 4, 4, 4],
                "fm_decoder_cnn_module_kernel": [31, 15, 7, 15, 31],
                "fm_decoder_feedforward_dim": 1536,
                "fm_decoder_num_heads": 4,
                "fm_decoder_dim": 512,
                "text_encoder_num_layers": 4,
                "text_encoder_feedforward_dim": 512,
                "text_encoder_cnn_module_kernel": 9,
                "text_encoder_num_heads": 4,
                "text_encoder_dim": 192,
                "query_head_dim": 32,
                "value_head_dim": 12,
                "pos_head_dim": 4,
                "pos_dim": 48,
                "time_embed_dim": 192,
                "text_embed_dim": 192,
                "feat_dim": 100,
                "vocab_size": 360
            },
            "feature": {
                "sample_rate": 24000,
                "n_fft": 1024,
                "hop_length": 256,
                "n_mels": 100
            }
        }"#;

        let config: LuxTTSConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.model.fm_decoder_dim, 512);
        assert_eq!(config.model.fm_decoder_num_layers, vec![2, 2, 4, 4, 4]);
        assert_eq!(config.model.fm_decoder_downsampling_factor, vec![1, 2, 4, 2, 1]);
        assert_eq!(config.model.text_encoder_dim, 192);
        assert_eq!(config.model.vocab_size, 360);
        assert_eq!(config.feature.sample_rate, 24000);
        assert_eq!(config.total_fm_layers(), 16);
    }

    #[test]
    fn test_flat_to_stack() {
        let json = r#"{
            "model": {
                "fm_decoder_downsampling_factor": [1, 2, 4, 2, 1],
                "fm_decoder_num_layers": [2, 2, 4, 4, 4],
                "fm_decoder_cnn_module_kernel": [31, 15, 7, 15, 31]
            },
            "feature": {}
        }"#;
        let config: LuxTTSConfig = serde_json::from_str(json).unwrap();

        assert_eq!(config.flat_to_stack(0), (0, 0));
        assert_eq!(config.flat_to_stack(1), (0, 1));
        assert_eq!(config.flat_to_stack(2), (1, 0));
        assert_eq!(config.flat_to_stack(3), (1, 1));
        assert_eq!(config.flat_to_stack(4), (2, 0));
        assert_eq!(config.flat_to_stack(7), (2, 3));
        assert_eq!(config.flat_to_stack(8), (3, 0));
        assert_eq!(config.flat_to_stack(15), (4, 3));

        assert!(config.is_stack_first(0));
        assert!(config.is_stack_first(2));
        assert!(!config.is_stack_first(1));
        assert!(config.is_stack_last(1));
        assert!(config.is_stack_last(3));
        assert!(!config.is_stack_last(0));
    }
}
