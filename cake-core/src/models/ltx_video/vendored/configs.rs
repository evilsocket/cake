//! Official LTX-Video configurations and presets.
//! Based on official configs from tp/LTX-Video/configs/
//! Supports versions 0.9.5+

use super::ltx_transformer::LtxVideoTransformer3DModelConfig;
use super::scheduler::FlowMatchEulerDiscreteSchedulerConfig;
use super::vae::AutoencoderKLLtxVideoConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXVInferenceConfig {
    pub guidance_scale: f32,
    pub num_inference_steps: usize,
    pub stg_scale: f32,
    pub rescaling_scale: f32,
    pub stochastic_sampling: bool,
    pub skip_block_list: Vec<usize>,
    pub timesteps: Option<Vec<f32>>,
    pub decode_timestep: Option<Vec<f32>>,
    pub decode_noise_scale: Option<Vec<f32>>,
}

impl Default for LTXVInferenceConfig {
    fn default() -> Self {
        Self {
            guidance_scale: 3.0,
            num_inference_steps: 40,
            stg_scale: 1.0,
            rescaling_scale: 0.7,
            stochastic_sampling: false,
            skip_block_list: vec![],
            timesteps: None,
            decode_timestep: None,
            decode_noise_scale: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXVFullConfig {
    pub inference: LTXVInferenceConfig,
    pub transformer: LtxVideoTransformer3DModelConfig,
    pub vae: AutoencoderKLLtxVideoConfig,
    pub scheduler: FlowMatchEulerDiscreteSchedulerConfig,
}

/// Returns the full configuration for a given version string.
/// Supports 0.9.5+ only.
pub fn get_config_by_version(version: &str) -> LTXVFullConfig {
    match version {
        // 0.9.5
        "0.9.5" | "0.9.5-2b" => presets::v0_9_5_2b(),

        // 0.9.6
        "0.9.6-dev" | "0.9.6-2b-dev" => presets::v0_9_6_dev_2b(),
        "0.9.6-distilled" | "0.9.6-2b-distilled" => presets::v0_9_6_distilled_2b(),

        // 0.9.8 2B
        "0.9.8-2b-distilled" | "0.9.8-distilled" => presets::v0_9_8_distilled_2b(),

        // 0.9.8 13B
        "0.9.8-13b-dev" => presets::v0_9_8_dev_13b(),
        "0.9.8-13b-distilled" | "0.9.8-13b" => presets::v0_9_8_distilled_13b(),

        // Default to 0.9.5
        _ => presets::v0_9_5_2b(),
    }
}

use super::scheduler::TimeShiftType;

pub mod presets {
    use super::*;

    /// Common VAE config for 0.9.5+
    /// Based on OURS_VAE_CONFIG from diffusers_config_mapping.py:
    /// - dims: 3
    /// - latent_channels: 128
    /// - blocks: [res_x(4), compress_all, res_x_y, res_x(3), compress_all, res_x_y, res_x(3), compress_all, res_x(3), res_x(4)]
    /// - norm_layer: "pixel_norm"
    /// - patch_size: 4
    /// - latent_log_var: "uniform"
    /// - causal_decoder: false
    fn common_vae_config() -> AutoencoderKLLtxVideoConfig {
        AutoencoderKLLtxVideoConfig {
            block_out_channels: vec![128, 256, 512, 1024, 2048],
            layers_per_block: vec![4, 6, 6, 2, 2],
            latent_channels: 128,
            patch_size: 4,
            timestep_conditioning: true,
            ..Default::default()
        }
    }

    /// Common scheduler config for 0.9.5+
    /// Based on OURS_SCHEDULER_CONFIG:
    /// - num_train_timesteps: 1000
    /// - shifting: "SD3"
    /// - target_shift_terminal: 0.1
    fn common_scheduler_config() -> FlowMatchEulerDiscreteSchedulerConfig {
        // Official LTX-Video uses SD3 resolution-dependent shifting with target_shift_terminal=0.1
        // For now we use a fixed shift approximation based on typical latent sizes
        // TODO: Implement proper SD3 resolution-dependent shifting
        FlowMatchEulerDiscreteSchedulerConfig {
            num_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: false, // LTX uses manual mu
            base_shift: Some(0.95),
            max_shift: Some(2.05),
            base_image_seq_len: Some(1024),
            max_image_seq_len: Some(4096),
            invert_sigmas: false,
            shift_terminal: Some(0.1),
            use_karras_sigmas: false,
            use_exponential_sigmas: false,
            use_beta_sigmas: false,
            time_shift_type: TimeShiftType::Exponential,
            stochastic_sampling: false,
        }
    }

    /// 2B transformer config (28 layers)
    /// Based on OURS_TRANSFORMER_CONFIG from diffusers_config_mapping.py:
    /// - num_layers: 28
    /// - num_attention_heads: 32
    /// - attention_head_dim: 64
    /// - cross_attention_dim: 2048
    /// - caption_channels: 4096
    /// - in_channels/out_channels: 128
    /// - qk_norm: "rms_norm"
    /// - positional_embedding_type: "rope"
    /// - positional_embedding_theta: 10000.0
    /// - positional_embedding_max_pos: [20, 2048, 2048]
    /// - timestep_scale_multiplier: 1000
    fn transformer_2b_config() -> LtxVideoTransformer3DModelConfig {
        LtxVideoTransformer3DModelConfig {
            num_layers: 28,
            num_attention_heads: 32,
            attention_head_dim: 64,
            cross_attention_dim: 2048,
            caption_channels: 4096,
            ..Default::default()
        }
    }

    /// 13B transformer config (48 layers)
    /// Larger model with:
    /// - num_layers: 48
    /// - attention_head_dim: 128
    /// - cross_attention_dim: 4096
    fn transformer_13b_config() -> LtxVideoTransformer3DModelConfig {
        LtxVideoTransformer3DModelConfig {
            num_layers: 48,
            num_attention_heads: 32,
            attention_head_dim: 128,
            cross_attention_dim: 4096,
            caption_channels: 4096,
            ..Default::default()
        }
    }

    /// ltxv-2b-0.9.5.yaml
    pub fn v0_9_5_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![19],
                timesteps: None,
                decode_timestep: None,
                decode_noise_scale: None,
            },
            transformer: transformer_2b_config(),
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    /// ltxv-2b-0.9.6-dev.yaml
    pub fn v0_9_6_dev_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 3.0,
                num_inference_steps: 40,
                stg_scale: 1.0,
                rescaling_scale: 0.7,
                stochastic_sampling: false,
                skip_block_list: vec![19],
                timesteps: None,
                decode_timestep: None,
                decode_noise_scale: None,
            },
            transformer: transformer_2b_config(),
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    /// ltxv-2b-0.9.6-distilled.yaml
    pub fn v0_9_6_distilled_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 8,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: true,
                skip_block_list: vec![],
                timesteps: None,
                decode_timestep: None,
                decode_noise_scale: None,
            },
            transformer: transformer_2b_config(),
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    /// ltxv-2b-0.9.8-distilled.yaml (first_pass config)
    pub fn v0_9_8_distilled_2b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 7,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: false,
                skip_block_list: vec![],
                timesteps: Some(vec![1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]),
                decode_timestep: Some(vec![0.05]),
                decode_noise_scale: Some(vec![0.025]),
            },
            transformer: transformer_2b_config(),
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    /// ltxv-13b-0.9.8-dev.yaml (first_pass config)
    pub fn v0_9_8_dev_13b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                // Uses dynamic guidance, we use peak value
                guidance_scale: 8.0,
                num_inference_steps: 30,
                stg_scale: 4.0,
                rescaling_scale: 0.5,
                stochastic_sampling: false,
                // First skip_block_list from guidance schedule
                skip_block_list: vec![11, 25, 35, 39],
                timesteps: None,
                decode_timestep: None,
                decode_noise_scale: None,
            },
            transformer: transformer_13b_config(),
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }

    /// ltxv-13b-0.9.8-distilled.yaml (first_pass config)
    pub fn v0_9_8_distilled_13b() -> LTXVFullConfig {
        LTXVFullConfig {
            inference: LTXVInferenceConfig {
                guidance_scale: 1.0,
                num_inference_steps: 7,
                stg_scale: 0.0,
                rescaling_scale: 1.0,
                stochastic_sampling: false,
                skip_block_list: vec![42],
                timesteps: Some(vec![1.0000, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]),
                decode_timestep: Some(vec![0.05]),
                decode_noise_scale: Some(vec![0.025]),
            },
            transformer: transformer_13b_config(),
            vae: common_vae_config(),
            scheduler: common_scheduler_config(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v0_9_5_2b_config() {
        let config = get_config_by_version("0.9.5");
        assert_eq!(config.transformer.num_layers, 28);
        assert_eq!(config.inference.guidance_scale, 3.0);
        assert_eq!(config.inference.num_inference_steps, 40);
        assert_eq!(config.inference.skip_block_list, vec![19]);
    }

    #[test]
    fn test_v0_9_8_distilled_2b_config() {
        let config = get_config_by_version("0.9.8-2b-distilled");
        assert_eq!(config.transformer.num_layers, 28);
        assert_eq!(config.inference.guidance_scale, 1.0);
        assert_eq!(config.inference.stg_scale, 0.0);
    }

    #[test]
    fn test_v0_9_8_13b_distilled_config() {
        let config = get_config_by_version("0.9.8-13b-distilled");
        assert_eq!(config.transformer.num_layers, 48);
        assert_eq!(config.transformer.attention_head_dim, 128);
        assert_eq!(config.transformer.cross_attention_dim, 4096);
        assert_eq!(config.inference.skip_block_list, vec![42]);
    }

    #[test]
    fn test_vae_config_5_blocks() {
        let config = get_config_by_version("0.9.5");
        assert_eq!(config.vae.block_out_channels.len(), 5);
        assert_eq!(
            config.vae.block_out_channels,
            vec![128, 256, 512, 1024, 2048]
        );
        assert_eq!(config.vae.layers_per_block, vec![4, 6, 6, 2, 2]);
    }
}
