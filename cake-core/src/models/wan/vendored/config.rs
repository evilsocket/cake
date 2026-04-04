use serde::Deserialize;
use std::path::Path;

/// Raw serde struct matching both Wan original and diffusers config.json formats.
#[derive(Debug, Clone, Deserialize)]
struct RawWanTransformerConfig {
    /// Original format: "dim" = hidden_size directly.
    #[serde(alias = "dim")]
    hidden_size: Option<usize>,
    /// Diffusers format: attention_head_dim × num_attention_heads = hidden_size.
    attention_head_dim: Option<usize>,
    /// Number of attention heads.
    #[serde(alias = "num_heads")]
    num_attention_heads: usize,
    /// FFN intermediate dimension.
    ffn_dim: usize,
    /// Timestep embedding dimension.
    freq_dim: usize,
    /// Input latent channels.
    #[serde(alias = "in_dim")]
    in_channels: usize,
    /// Output latent channels.
    #[serde(alias = "out_dim")]
    out_channels: usize,
    /// Number of transformer blocks.
    num_layers: usize,
    /// Text encoder hidden dimension.
    #[serde(default = "default_text_dim")]
    text_dim: usize,
    /// Max text sequence length.
    #[serde(default = "default_text_len")]
    text_len: usize,
}

/// Wan transformer config (resolved from either format).
#[derive(Debug, Clone)]
pub struct WanTransformerConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub ffn_dim: usize,
    pub freq_dim: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub num_layers: usize,
    pub text_dim: usize,
    pub text_len: usize,
}

fn default_text_dim() -> usize { 4096 }
fn default_text_len() -> usize { 512 }

impl WanTransformerConfig {
    /// Load from a config.json file (supports both original and diffusers formats).
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let data = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("can't read {}: {:?}", path.display(), e))?;
        let raw: RawWanTransformerConfig = serde_json::from_slice(&data)
            .map_err(|e| anyhow::anyhow!("can't parse {}: {:?}", path.display(), e))?;

        // Resolve hidden_size: either explicit "dim"/"hidden_size" or from attention_head_dim * num_heads
        let hidden_size = raw.hidden_size.unwrap_or_else(|| {
            raw.attention_head_dim.unwrap_or(128) * raw.num_attention_heads
        });

        Ok(Self {
            hidden_size,
            num_attention_heads: raw.num_attention_heads,
            ffn_dim: raw.ffn_dim,
            freq_dim: raw.freq_dim,
            in_channels: raw.in_channels,
            out_channels: raw.out_channels,
            num_layers: raw.num_layers,
            text_dim: raw.text_dim,
            text_len: raw.text_len,
        })
    }

    /// Head dimension = hidden_size / num_attention_heads.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Patch size for converting latents to token sequence.
    /// Wan uses (1, 2, 2): no temporal patching, 2x2 spatial.
    pub fn patch_size(&self) -> (usize, usize, usize) {
        (1, 2, 2)
    }

    /// 3D RoPE dimension split: (temporal, height, width).
    /// Total = head_dim = 128. Split: t=44, h=42, w=42.
    pub fn rope_dims(&self) -> (usize, usize, usize) {
        let d = self.head_dim();
        let hw = 2 * (d / 6); // 42
        let t = d - 2 * hw;   // 44
        (t, hw, hw)
    }

    /// Default config for Wan2.2-T2V-A14B.
    pub fn wan22_14b() -> Self {
        Self {
            hidden_size: 5120,
            num_attention_heads: 40,
            ffn_dim: 13824,
            freq_dim: 256,
            in_channels: 16,
            out_channels: 16,
            num_layers: 40,
            text_dim: 4096,
            text_len: 512,
        }
    }
}

/// Wan VAE config.
#[derive(Debug, Clone)]
pub struct WanVaeConfig {
    /// Base channel dimension (default 96).
    pub base_dim: usize,
    /// Latent channels (default 16).
    pub z_dim: usize,
    /// Channel multipliers per stage.
    pub dim_mult: Vec<usize>,
    /// ResBlocks per stage.
    pub num_res_blocks: usize,
    /// Which stages do temporal downsampling.
    pub temporal_downsample: Vec<bool>,
    /// Temporal compression ratio.
    pub temporal_compression: usize,
    /// Spatial compression ratio.
    pub spatial_compression: usize,
}

impl Default for WanVaeConfig {
    fn default() -> Self {
        Self {
            base_dim: 96,
            z_dim: 16,
            dim_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            temporal_downsample: vec![false, true, true],
            temporal_compression: 4,
            spatial_compression: 8,
        }
    }
}

/// Latent normalization constants for Wan2.2.
pub const LATENTS_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

pub const LATENTS_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wan22_14b_defaults() {
        let cfg = WanTransformerConfig::wan22_14b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_attention_heads, 40);
        assert_eq!(cfg.num_layers, 40);
        assert_eq!(cfg.in_channels, 16);
        assert_eq!(cfg.out_channels, 16);
    }

    #[test]
    fn test_head_dim() {
        let cfg = WanTransformerConfig::wan22_14b();
        assert_eq!(cfg.head_dim(), 128); // 5120 / 40
    }

    #[test]
    fn test_rope_dims_sum_to_head_dim() {
        let cfg = WanTransformerConfig::wan22_14b();
        let (t, h, w) = cfg.rope_dims();
        assert_eq!(t + h + w, cfg.head_dim(),
            "RoPE dims ({t}+{h}+{w}={}) should sum to head_dim={}",
            t + h + w, cfg.head_dim());
    }

    #[test]
    fn test_rope_dims_values() {
        let cfg = WanTransformerConfig::wan22_14b();
        let (t, h, w) = cfg.rope_dims();
        assert_eq!(t, 44);
        assert_eq!(h, 42);
        assert_eq!(w, 42);
    }

    #[test]
    fn test_patch_size() {
        let cfg = WanTransformerConfig::wan22_14b();
        assert_eq!(cfg.patch_size(), (1, 2, 2));
    }

    #[test]
    fn test_vae_config_defaults() {
        let cfg = WanVaeConfig::default();
        assert_eq!(cfg.z_dim, 16);
        assert_eq!(cfg.temporal_compression, 4);
        assert_eq!(cfg.spatial_compression, 8);
        assert_eq!(cfg.dim_mult, vec![1, 2, 4, 4]);
    }

    #[test]
    fn test_latents_constants_length() {
        assert_eq!(LATENTS_MEAN.len(), 16);
        assert_eq!(LATENTS_STD.len(), 16);
    }

    #[test]
    fn test_latents_std_positive() {
        for (i, &s) in LATENTS_STD.iter().enumerate() {
            assert!(s > 0.0, "LATENTS_STD[{i}]={s} should be positive");
        }
    }

    #[test]
    fn test_config_deserialize_original_format() {
        let dir = std::env::temp_dir().join("wan_test_config_orig");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{
            "dim": 5120,
            "num_heads": 40,
            "ffn_dim": 13824,
            "freq_dim": 256,
            "in_dim": 16,
            "out_dim": 16,
            "num_layers": 40
        }"#).unwrap();
        let cfg = WanTransformerConfig::from_path(&path).unwrap();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_attention_heads, 40);
        assert_eq!(cfg.text_dim, 4096); // default
        assert_eq!(cfg.text_len, 512);  // default
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_config_deserialize_diffusers_format() {
        let dir = std::env::temp_dir().join("wan_test_config_diff");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.json");
        std::fs::write(&path, r#"{
            "_class_name": "WanTransformer3DModel",
            "attention_head_dim": 128,
            "num_attention_heads": 12,
            "ffn_dim": 8960,
            "freq_dim": 256,
            "in_channels": 16,
            "out_channels": 16,
            "num_layers": 30,
            "text_dim": 4096
        }"#).unwrap();
        let cfg = WanTransformerConfig::from_path(&path).unwrap();
        assert_eq!(cfg.hidden_size, 1536); // 128 * 12
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.num_layers, 30);
        assert_eq!(cfg.ffn_dim, 8960);
        std::fs::remove_dir_all(&dir).ok();
    }
}
