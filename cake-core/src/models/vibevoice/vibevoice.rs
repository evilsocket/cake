//! VibeVoice-Realtime-0.5B TTS pipeline.
//!
//! Generates 24kHz mono audio from text using:
//! 1. Qwen2.5-0.5B LLM backbone for text understanding
//! 2. DDPM diffusion head for acoustic latent generation
//! 3. σ-VAE decoder for waveform synthesis

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use log::info;

use super::acoustic_connector::AcousticConnector;
use super::config::VibeVoiceConfig;
use super::ddpm::DdpmScheduler;
use super::eos_classifier::EosClassifier;
use super::prediction_head::PredictionHead;
use super::vae_decoder::AcousticVaeDecoder;

/// VibeVoice TTS model — all components loaded and ready for inference.
pub struct VibeVoiceTTS {
    /// LLM embedding layer (shared with tts_language_model).
    embed_tokens: candle_nn::Embedding,
    /// LLM final norm.
    lm_norm: candle_nn::RmsNorm,
    /// LLM layers — loaded as a simple list of transformer blocks.
    lm_layers: Vec<crate::models::common::Transformer>,
    /// Diffusion prediction head.
    prediction_head: PredictionHead,
    /// DDPM noise scheduler.
    scheduler: DdpmScheduler,
    /// Acoustic VAE decoder.
    vae_decoder: AcousticVaeDecoder,
    /// Acoustic connector (VAE latent → LLM hidden).
    connector: AcousticConnector,
    /// End-of-speech classifier.
    eos_classifier: EosClassifier,
    /// Speech scaling/bias factors for VAE denormalization.
    speech_scaling_factor: Tensor,
    speech_bias_factor: Tensor,
    /// Config.
    config: VibeVoiceConfig,
    /// Device.
    device: Device,
}

impl VibeVoiceTTS {
    /// Load the full VibeVoice model from a safetensors file.
    pub fn load(
        config_path: &std::path::Path,
        weights_path: &std::path::Path,
        device: &Device,
    ) -> Result<Self> {
        let config = VibeVoiceConfig::from_path(config_path)?;
        let common_cfg = config.into_config();

        info!("Loading VibeVoice-Realtime-0.5B...");

        // Use F16 on CUDA for Flash Attention compatibility, F32 on CPU
        let load_dtype = if matches!(device, Device::Cuda(_)) {
            DType::F16
        } else {
            DType::F32
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path.to_path_buf()],
                load_dtype,
                device,
            )?
        };

        // LLM backbone: model.tts_language_model.*
        let lm_vb = vb.pp("model").pp("tts_language_model");
        info!("  Loading LLM embedding...");
        let embed_tokens = candle_nn::embedding(
            common_cfg.vocab_size,
            common_cfg.hidden_size,
            lm_vb.pp("embed_tokens"),
        )?;
        info!("  Loading LLM norm...");
        let lm_norm = candle_nn::rms_norm(
            common_cfg.hidden_size,
            common_cfg.rms_norm_eps,
            lm_vb.pp("norm"),
        )?;

        // LLM layers — use tts_backbone_num_hidden_layers (may be fewer than decoder's full count)
        let num_lm_layers = config.tts_backbone_num_hidden_layers;
        info!("  Loading {} LLM layers...", num_lm_layers);
        let mut lm_layers = Vec::with_capacity(num_lm_layers);
        for i in 0..num_lm_layers {
            let block = crate::models::common::Transformer::load_for_vibevoice(
                lm_vb.pp("layers").pp(i),
                &common_cfg,
            )?;
            lm_layers.push(block);
        }

        // Prediction head: model.prediction_head.*
        info!("  Loading prediction head...");
        let prediction_head = PredictionHead::load(
            vb.pp("model").pp("prediction_head"),
            &config.diffusion_head_config,
        )?;

        // VAE decoder: model.acoustic_tokenizer.decoder.*
        info!("  Loading acoustic VAE decoder...");
        let vae_decoder = AcousticVaeDecoder::load(
            vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
            &config.acoustic_tokenizer_config,
        )?;

        // Acoustic connector: model.acoustic_connector.*
        info!("  Loading acoustic connector...");
        let connector = AcousticConnector::load(
            vb.pp("model").pp("acoustic_connector"),
            config.acoustic_vae_dim,
            config.decoder_config.hidden_size,
            config.decoder_config.rms_norm_eps,
        )?;

        // EOS classifier: tts_eos_classifier.*
        info!("  Loading EOS classifier...");
        let eos_classifier = EosClassifier::load(vb.pp("tts_eos_classifier"))?;

        // Speech scaling factors
        let speech_scaling_factor = vb.pp("model").get((), "speech_scaling_factor")?;
        let speech_bias_factor = vb.pp("model").get((), "speech_bias_factor")?;

        // DDPM scheduler
        let scheduler = DdpmScheduler::new_cosine(
            config.diffusion_head_config.ddpm_num_steps,
            config.diffusion_head_config.ddpm_num_inference_steps,
        );

        info!("VibeVoice model loaded!");

        Ok(Self {
            embed_tokens,
            lm_norm,
            lm_layers,
            prediction_head,
            scheduler,
            vae_decoder,
            connector,
            eos_classifier,
            speech_scaling_factor,
            speech_bias_factor,
            config,
            device: device.clone(),
        })
    }

    /// Generate audio from text.
    /// Returns PCM f32 samples at 24kHz.
    pub fn generate(&self, token_ids: &[u32], max_frames: usize) -> Result<Vec<f32>> {
        let dev = &self.device;

        // 1. Embed text tokens
        let input_ids = Tensor::new(token_ids, dev)?.unsqueeze(0)?;
        let mut hidden = self.embed_tokens.forward(&input_ids)?;

        // 2. Run through LLM layers with proper RoPE cache
        let common_cfg = self.config.into_config();
        let mut cache = crate::models::common::Cache::new(
            false, // no KV cache for encoding
            hidden.dtype(),
            &common_cfg,
            dev,
        )?;
        for (i, layer) in self.lm_layers.iter().enumerate() {
            hidden = layer.forward_with_cache(&hidden, 0, i, &mut cache)?;
        }
        hidden = self.lm_norm.forward(&hidden)?;

        // 3. Generate speech frames via diffusion
        let mut audio_latents = Vec::new();
        let last_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?.squeeze(1)?; // (1, hidden)

        let mut condition = last_hidden;

        for frame in 0..max_frames {
            // Sample speech token via DDPM
            let latent = self.sample_speech_latent(&condition)?;

            // Denormalize for VAE
            let scale = self.speech_scaling_factor.to_dtype(latent.dtype())?;
            let bias = self.speech_bias_factor.to_dtype(latent.dtype())?;
            let scaled_latent = ((&latent / &scale)? - &bias)?;

            audio_latents.push(scaled_latent);

            // Check EOS
            if self.eos_classifier.should_stop(&condition, 0.5)? {
                info!("  EOS detected at frame {}", frame + 1);
                break;
            }

            // Feed acoustic latent back through connector for next frame
            let acoustic_embed = self.connector.forward(&latent)?;
            // Update condition for next frame (simplified — full model uses autoregressive LLM)
            condition = acoustic_embed;

            if (frame + 1) % 10 == 0 {
                info!("  Generated {} frames...", frame + 1);
            }
        }

        info!("Generated {} speech frames, decoding audio...", audio_latents.len());

        if audio_latents.is_empty() {
            return Ok(vec![]);
        }

        // 4. Stack latents and decode through VAE
        let latents = Tensor::stack(&audio_latents, 1)?; // (1, frames, vae_dim)
        let latents = latents.transpose(1, 2)?; // (1, vae_dim, frames)
        let audio = self.vae_decoder.decode(&latents)?; // (1, 1, samples)

        // 5. Extract PCM samples
        let audio = audio.squeeze(0)?.squeeze(0)?; // (samples,)
        let samples: Vec<f32> = audio.to_dtype(DType::F32)?.to_vec1()?;

        Ok(samples)
    }

    /// Sample a single speech latent via DDPM reverse diffusion.
    fn sample_speech_latent(&self, condition: &Tensor) -> Result<Tensor> {
        let vae_dim = self.config.acoustic_vae_dim;

        // Start from random noise
        let mut sample = Tensor::randn(0f32, 1., (1, vae_dim), &self.device)?;

        // Reverse diffusion loop
        for &t in self.scheduler.timesteps() {
            let t_tensor = Tensor::new(&[t as f32 / 1000.0], &self.device)?;
            let v_pred = self.prediction_head.forward(&sample, &t_tensor, condition)?;
            sample = self.scheduler.step(&v_pred, t, &sample)?;
        }

        Ok(sample)
    }
}

/// Write PCM samples to a WAV file (24kHz, mono, 16-bit).
pub fn save_wav(samples: &[f32], path: &std::path::Path, sample_rate: u32) -> Result<()> {
    use std::io::Write;

    let num_samples = samples.len();
    let data_size = (num_samples * 2) as u32; // 16-bit = 2 bytes per sample
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    // WAV header
    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;

    // fmt chunk
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?; // chunk size
    file.write_all(&1u16.to_le_bytes())?; // PCM format
    file.write_all(&1u16.to_le_bytes())?; // mono
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?; // byte rate
    file.write_all(&2u16.to_le_bytes())?; // block align
    file.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        file.write_all(&i16_val.to_le_bytes())?;
    }

    Ok(())
}
