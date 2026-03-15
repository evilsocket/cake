//! Weight format detection and key remapping for LTX-Video models.
//!
//! Supports two formats:
//! - Diffusers: separate files in transformer/, vae/, text_encoder/ directories
//! - Official: single unified safetensors file (e.g., ltx-video-2b-v0.9.5.safetensors)
//!
//! Key mapping based on diffusers/scripts/convert_ltx_to_diffusers.py

use regex::Regex;
use std::path::Path;

/// Weight format detection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFormat {
    /// Diffusers format: separate files in subdirectories
    Diffusers,
    /// Official LTX-Video format: single unified safetensors file
    Official,
}

/// Detect weight format from path
pub fn detect_format(path: &Path) -> WeightFormat {
    if path.is_file() {
        WeightFormat::Official
    } else {
        // Both directory and non-existent paths default to Diffusers format
        WeightFormat::Diffusers
    }
}

/// Key remapping from Official (native LTX-Video) format to Diffusers format.
/// Based on diffusers/scripts/convert_ltx_to_diffusers.py VAE_095_RENAME_DICT
#[derive(Debug, Clone)]
pub struct KeyRemapper {
    encoder_block_re: Regex,
    decoder_block_re: Regex,
}

impl Default for KeyRemapper {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyRemapper {
    pub fn new() -> Self {
        Self {
            encoder_block_re: Regex::new(r"encoder\.down_blocks\.(\d+)").unwrap(),
            decoder_block_re: Regex::new(r"decoder\.up_blocks\.(\d+)").unwrap(),
        }
    }

    /// Remap a key from Official (native) format to Diffusers format
    /// Uses VAE_095_RENAME_DICT mapping from convert_ltx_to_diffusers.py
    pub fn remap_key(&self, key: &str) -> String {
        let mut result = key.to_string();

        // 1. Transformer mappings (simple replacements)
        result = result.replace("patchify_proj", "proj_in");
        result = result.replace("adaln_single", "time_embed");
        result = result.replace("q_norm", "norm_q");
        result = result.replace("k_norm", "norm_k");

        // 2. VAE: Replace res_blocks -> resnets
        result = result.replace("res_blocks", "resnets");

        // 3. VAE: Remap encoder block indices (0.9.5+ format)
        result = self.remap_encoder_blocks_095(&result);

        // 4. VAE: Remap decoder block indices (0.9.5+ format)
        result = self.remap_decoder_blocks_095(&result);

        // 5. Other VAE mappings from VAE_095_RENAME_DICT
        result = result.replace("last_time_embedder", "time_embedder");
        result = result.replace("last_scale_shift_table", "scale_shift_table");
        result = result.replace("norm3.norm", "norm3");
        result = result.replace("per_channel_statistics.mean-of-means", "latents_mean");
        result = result.replace("per_channel_statistics.std-of-means", "latents_std");

        result
    }

    /// Remap encoder block indices from native flat format to Diffusers hierarchical format
    /// Based on VAE_095_RENAME_DICT from convert_ltx_to_diffusers.py:
    /// Native 0 -> Diffusers down_blocks.0
    /// Native 1 -> Diffusers down_blocks.0.downsamplers.0
    /// Native 2 -> Diffusers down_blocks.1
    /// Native 3 -> Diffusers down_blocks.1.downsamplers.0
    /// Native 4 -> Diffusers down_blocks.2
    /// Native 5 -> Diffusers down_blocks.2.downsamplers.0
    /// Native 6 -> Diffusers down_blocks.3
    /// Native 7 -> Diffusers down_blocks.3.downsamplers.0
    /// Native 8 -> Diffusers mid_block
    fn remap_encoder_blocks_095(&self, key: &str) -> String {
        self.encoder_block_re
            .replace_all(key, |caps: &regex::Captures| {
                let native_idx: usize = caps[1].parse().unwrap_or(0);
                match native_idx {
                    0 => "encoder.down_blocks.0".to_string(),
                    1 => "encoder.down_blocks.0.downsamplers.0".to_string(),
                    2 => "encoder.down_blocks.1".to_string(),
                    3 => "encoder.down_blocks.1.downsamplers.0".to_string(),
                    4 => "encoder.down_blocks.2".to_string(),
                    5 => "encoder.down_blocks.2.downsamplers.0".to_string(),
                    6 => "encoder.down_blocks.3".to_string(),
                    7 => "encoder.down_blocks.3.downsamplers.0".to_string(),
                    8 => "encoder.mid_block".to_string(),
                    _ => format!("encoder.down_blocks.{}", native_idx),
                }
            })
            .to_string()
    }

    /// Remap decoder block indices from native flat format to Diffusers hierarchical format
    /// Based on VAE_095_RENAME_DICT from convert_ltx_to_diffusers.py:
    /// Native 0 -> Diffusers mid_block
    /// Native 1 -> Diffusers up_blocks.0.upsamplers.0
    /// Native 2 -> Diffusers up_blocks.0
    /// Native 3 -> Diffusers up_blocks.1.upsamplers.0
    /// Native 4 -> Diffusers up_blocks.1
    /// Native 5 -> Diffusers up_blocks.2.upsamplers.0
    /// Native 6 -> Diffusers up_blocks.2
    /// Native 7 -> Diffusers up_blocks.3.upsamplers.0
    /// Native 8 -> Diffusers up_blocks.3
    fn remap_decoder_blocks_095(&self, key: &str) -> String {
        self.decoder_block_re
            .replace_all(key, |caps: &regex::Captures| {
                let native_idx: usize = caps[1].parse().unwrap_or(0);
                match native_idx {
                    0 => "decoder.mid_block".to_string(),
                    1 => "decoder.up_blocks.0.upsamplers.0".to_string(),
                    2 => "decoder.up_blocks.0".to_string(),
                    3 => "decoder.up_blocks.1.upsamplers.0".to_string(),
                    4 => "decoder.up_blocks.1".to_string(),
                    5 => "decoder.up_blocks.2.upsamplers.0".to_string(),
                    6 => "decoder.up_blocks.2".to_string(),
                    7 => "decoder.up_blocks.3.upsamplers.0".to_string(),
                    8 => "decoder.up_blocks.3".to_string(),
                    _ => format!("decoder.up_blocks.{}", native_idx),
                }
            })
            .to_string()
    }

    /// Check if a key belongs to the transformer
    pub fn is_transformer_key(key: &str) -> bool {
        key.starts_with("transformer.")
            || key.starts_with("model.diffusion_model.")  // Native format prefix
            || key.contains("transformer_blocks")
            || key.contains("patchify_proj")
            || key.contains("proj_in")
            || key.contains("adaln_single")
            || key.contains("time_embed")
    }

    /// Check if a key belongs to the VAE
    pub fn is_vae_key(key: &str) -> bool {
        key.starts_with("vae.")
            || key.starts_with("encoder.")
            || key.starts_with("decoder.")
            || key.contains("per_channel_statistics")
            || key.contains("latents_mean")
            || key.contains("latents_std")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remap_transformer_key() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("transformer.patchify_proj.weight"),
            "transformer.proj_in.weight"
        );
        assert_eq!(
            remapper.remap_key("transformer.adaln_single.linear.weight"),
            "transformer.time_embed.linear.weight"
        );
    }

    #[test]
    fn test_remap_encoder_blocks_095() {
        let remapper = KeyRemapper::new();

        // Native block 0 -> Diffusers block 0
        assert_eq!(
            remapper.remap_key("encoder.down_blocks.0.res_blocks.0.conv1.weight"),
            "encoder.down_blocks.0.resnets.0.conv1.weight"
        );

        // Native block 1 -> Diffusers downsamplers
        assert_eq!(
            remapper.remap_key("encoder.down_blocks.1.conv.weight"),
            "encoder.down_blocks.0.downsamplers.0.conv.weight"
        );

        // Native block 2 -> Diffusers block 1 (NOT conv_out for 0.9.5+)
        assert_eq!(
            remapper.remap_key("encoder.down_blocks.2.res_blocks.0.conv1.weight"),
            "encoder.down_blocks.1.resnets.0.conv1.weight"
        );

        // Native block 6 -> Diffusers block 3
        assert_eq!(
            remapper.remap_key("encoder.down_blocks.6.res_blocks.0.weight"),
            "encoder.down_blocks.3.resnets.0.weight"
        );

        // Native block 8 -> mid_block
        assert_eq!(
            remapper.remap_key("encoder.down_blocks.8.res_blocks.0.weight"),
            "encoder.mid_block.resnets.0.weight"
        );
    }

    #[test]
    fn test_remap_decoder_blocks_095() {
        let remapper = KeyRemapper::new();

        // Native block 0 -> mid_block
        assert_eq!(
            remapper.remap_key("decoder.up_blocks.0.res_blocks.0.weight"),
            "decoder.mid_block.resnets.0.weight"
        );

        // Native block 1 -> upsamplers
        assert_eq!(
            remapper.remap_key("decoder.up_blocks.1.conv.weight"),
            "decoder.up_blocks.0.upsamplers.0.conv.weight"
        );

        // Native block 2 -> Diffusers block 0
        assert_eq!(
            remapper.remap_key("decoder.up_blocks.2.res_blocks.0.weight"),
            "decoder.up_blocks.0.resnets.0.weight"
        );

        // Native block 8 -> Diffusers block 3
        assert_eq!(
            remapper.remap_key("decoder.up_blocks.8.res_blocks.0.weight"),
            "decoder.up_blocks.3.resnets.0.weight"
        );
    }

    #[test]
    fn test_remap_time_embedder() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("decoder.last_time_embedder.weight"),
            "decoder.time_embedder.weight"
        );
    }

    #[test]
    fn test_remap_latents_stats() {
        let remapper = KeyRemapper::new();
        assert_eq!(
            remapper.remap_key("per_channel_statistics.mean-of-means"),
            "latents_mean"
        );
        assert_eq!(
            remapper.remap_key("per_channel_statistics.std-of-means"),
            "latents_std"
        );
    }
}
