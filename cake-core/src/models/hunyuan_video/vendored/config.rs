use serde::Deserialize;

/// HunyuanVideo transformer configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanTransformerConfig {
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_text_embed_dim")]
    pub text_embed_dim: usize,
}

fn default_hidden_size() -> usize {
    3072
}
fn default_num_heads() -> usize {
    24
}
fn default_num_layers() -> usize {
    40
}
fn default_patch_size() -> usize {
    2
}
fn default_in_channels() -> usize {
    16
}
fn default_text_embed_dim() -> usize {
    4096
}

impl Default for HunyuanTransformerConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_attention_heads: default_num_heads(),
            num_layers: default_num_layers(),
            patch_size: default_patch_size(),
            in_channels: default_in_channels(),
            text_embed_dim: default_text_embed_dim(),
        }
    }
}

/// HunyuanVideo 3D VAE configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanVaeConfig {
    #[serde(default = "default_latent_channels")]
    pub latent_channels: usize,
    #[serde(default = "default_temporal_compression")]
    pub temporal_compression_ratio: usize,
    #[serde(default = "default_spatial_compression")]
    pub spatial_compression_ratio: usize,
}

fn default_latent_channels() -> usize {
    16
}
fn default_temporal_compression() -> usize {
    4
}
fn default_spatial_compression() -> usize {
    8
}

impl Default for HunyuanVaeConfig {
    fn default() -> Self {
        Self {
            latent_channels: default_latent_channels(),
            temporal_compression_ratio: default_temporal_compression(),
            spatial_compression_ratio: default_spatial_compression(),
        }
    }
}
