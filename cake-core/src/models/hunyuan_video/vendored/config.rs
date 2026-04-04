use serde::Deserialize;

/// HunyuanVideo transformer configuration.
///
/// 13B parameter dual-stream DiT (MMDiT) architecture:
/// - 20 double-stream blocks + 40 single-stream blocks = 60 total
/// - 3D RoPE with separate temporal/height/width dimensions
/// - Dual text encoding: LLaMA-based encoder + CLIP
#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanVideoConfig {
    /// Hidden dimension of the transformer (default 3072).
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    /// Number of attention heads (default 24).
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    /// Total number of transformer layers: 20 double-stream + 40 single-stream (default 60).
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,
    /// Number of double-stream (MMDiT) blocks (default 20).
    #[serde(default = "default_num_double_layers")]
    pub num_double_layers: usize,
    /// Number of single-stream blocks (default 40).
    #[serde(default = "default_num_single_layers")]
    pub num_single_layers: usize,
    /// Patch size for video tokenization [temporal, height, width] (default [1, 2, 2]).
    #[serde(default = "default_patch_size")]
    pub patch_size: Vec<usize>,
    /// Input latent channels from VAE (default 16).
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    /// Output latent channels (default 16).
    #[serde(default = "default_out_channels")]
    pub out_channels: usize,
    /// Text embedding dimension from LLaMA encoder (default 4096).
    #[serde(default = "default_text_embed_dim")]
    pub text_embed_dim: usize,
    /// Maximum text sequence length (default 256).
    #[serde(default = "default_text_len")]
    pub text_len: usize,
    /// CLIP text embedding dimension (default 768).
    #[serde(default = "default_clip_embed_dim")]
    pub clip_embed_dim: usize,
    /// RoPE dimension list for [temporal, height, width] (default [16, 56, 56]).
    #[serde(default = "default_rope_dim_list")]
    pub rope_dim_list: Vec<usize>,
}

fn default_hidden_size() -> usize { 3072 }
fn default_num_attention_heads() -> usize { 24 }
fn default_num_layers() -> usize { 60 }
fn default_num_double_layers() -> usize { 20 }
fn default_num_single_layers() -> usize { 40 }
fn default_patch_size() -> Vec<usize> { vec![1, 2, 2] }
fn default_in_channels() -> usize { 16 }
fn default_out_channels() -> usize { 16 }
fn default_text_embed_dim() -> usize { 4096 }
fn default_text_len() -> usize { 256 }
fn default_clip_embed_dim() -> usize { 768 }
fn default_rope_dim_list() -> Vec<usize> { vec![16, 56, 56] }

impl Default for HunyuanVideoConfig {
    fn default() -> Self {
        Self {
            hidden_size: default_hidden_size(),
            num_attention_heads: default_num_attention_heads(),
            num_layers: default_num_layers(),
            num_double_layers: default_num_double_layers(),
            num_single_layers: default_num_single_layers(),
            patch_size: default_patch_size(),
            in_channels: default_in_channels(),
            out_channels: default_out_channels(),
            text_embed_dim: default_text_embed_dim(),
            text_len: default_text_len(),
            clip_embed_dim: default_clip_embed_dim(),
            rope_dim_list: default_rope_dim_list(),
        }
    }
}

impl HunyuanVideoConfig {
    /// Head dimension = hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
