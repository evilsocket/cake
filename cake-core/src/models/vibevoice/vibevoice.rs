//! VibeVoice-Realtime-0.5B TTS pipeline.
//!
//! Generates 24kHz mono audio from text using:
//! 1. Qwen2.5-0.5B LLM backbone for text understanding
//! 2. DDPM diffusion head for acoustic latent generation with CFG
//! 3. σ-VAE decoder for waveform synthesis
//!
//! Generation follows the interleaved text-speech windowed approach:
//! [text_window_1 (5 tokens)] → [speech_frames (6)] → [text_window_2] → ...

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

const TTS_TEXT_WINDOW_SIZE: usize = 5;
const TTS_SPEECH_WINDOW_SIZE: usize = 6;

/// VibeVoice TTS model — all components loaded and ready for inference.
pub struct VibeVoiceTTS {
    /// TTS language model embedding.
    embed_tokens: candle_nn::Embedding,
    /// TTS LM final norm.
    lm_norm: candle_nn::RmsNorm,
    /// TTS LM layers.
    lm_layers: Vec<crate::models::common::Transformer>,
    /// Input type embedding: 0 = speech token, 1 = text token.
    tts_input_types: candle_nn::Embedding,
    /// Diffusion prediction head (4-layer DiT).
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
    /// RoPE cache for LLM.
    cache: crate::models::common::Cache,
    /// Config.
    config: VibeVoiceConfig,
    /// Device.
    device: Device,
    /// Model dtype.
    dtype: DType,
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

        let dtype = if matches!(device, Device::Cuda(_)) {
            DType::F16
        } else {
            DType::F32
        };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[weights_path.to_path_buf()],
                dtype,
                device,
            )?
        };

        // TTS language model: model.tts_language_model.*
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

        // Input type embedding: model.tts_input_types (2 types: speech=0, text=1)
        let tts_input_types = candle_nn::embedding(
            2,
            common_cfg.hidden_size,
            vb.pp("model").pp("tts_input_types"),
        )?;

        // Prediction head
        info!("  Loading prediction head...");
        let prediction_head = PredictionHead::load(
            vb.pp("model").pp("prediction_head"),
            &config.diffusion_head_config,
        )?;

        // VAE decoder
        info!("  Loading acoustic VAE decoder...");
        let vae_decoder = AcousticVaeDecoder::load(
            vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
            &config.acoustic_tokenizer_config,
        )?;

        // Acoustic connector
        info!("  Loading acoustic connector...");
        let connector = AcousticConnector::load(
            vb.pp("model").pp("acoustic_connector"),
            config.acoustic_vae_dim,
            config.decoder_config.hidden_size,
            config.decoder_config.rms_norm_eps,
        )?;

        // EOS classifier
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

        // RoPE + KV cache for autoregressive generation
        let cache = crate::models::common::Cache::new(true, dtype, &common_cfg, device)?;

        info!("VibeVoice model loaded!");

        Ok(Self {
            embed_tokens,
            lm_norm,
            lm_layers,
            tts_input_types,
            prediction_head,
            scheduler,
            vae_decoder,
            connector,
            eos_classifier,
            speech_scaling_factor,
            speech_bias_factor,
            cache,
            config,
            device: device.clone(),
            dtype,
        })
    }

    /// Run TTS LM forward on a sequence of embeddings.
    fn forward_tts_lm(&mut self, embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let mut h = embeds.clone();
        for (i, layer) in self.lm_layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, index_pos, i, &mut self.cache)?;
        }
        self.lm_norm.forward(&h).map_err(|e| anyhow::anyhow!("lm_norm: {e}"))
    }

    /// Sample a single speech latent via DDPM with optional CFG.
    fn sample_speech_latent(
        &self,
        condition: &Tensor,
        cfg_scale: f32,
    ) -> Result<Tensor> {
        let vae_dim = self.config.acoustic_vae_dim;
        let timesteps = self.scheduler.timesteps().to_vec();

        let mut sample = Tensor::randn(0f32, 1., (1, vae_dim), &self.device)?
            .to_dtype(self.dtype)?;

        for t in &timesteps {
            let t_tensor = Tensor::new(&[*t as f32 / 1000.0], &self.device)?
                .to_dtype(self.dtype)?;

            let v_pred = self.prediction_head.forward(&sample, &t_tensor, condition)?;

            let v_pred = if cfg_scale != 1.0 {
                (v_pred * cfg_scale as f64)?
            } else {
                v_pred
            };

            sample = self.scheduler.step(&v_pred, *t, &sample)?;
        }

        Ok(sample)
    }

    /// Generate audio from text token IDs.
    /// Returns PCM f32 samples at 24kHz.
    pub fn generate(
        &mut self,
        token_ids: &[u32],
        max_frames: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        let dev = self.device.clone();
        let hidden_size = self.config.decoder_config.hidden_size;

        // Reset KV cache for fresh generation
        self.cache.clear();

        let mut audio_latents: Vec<Tensor> = Vec::new();
        let mut text_pos = 0;
        let mut seq_pos = 0;
        let mut finished = false;

        // Type embeddings: 0 = speech, 1 = text
        let text_type_id = Tensor::new(&[1u32], &dev)?;
        let speech_type_id = Tensor::new(&[0u32], &dev)?;
        let text_type_embed = self.tts_input_types.forward(&text_type_id)?; // (1, hidden)
        let speech_type_embed = self.tts_input_types.forward(&speech_type_id)?;

        while !finished && audio_latents.len() < max_frames {
            // ── Text window ─────────────────────────────────────────
            let window_end = (text_pos + TTS_TEXT_WINDOW_SIZE).min(token_ids.len());
            let window_size = window_end - text_pos;

            if window_size > 0 {
                let window_ids = Tensor::new(
                    &token_ids[text_pos..window_end],
                    &dev,
                )?.unsqueeze(0)?;

                // Embed text tokens + add type embedding
                let text_embeds = self.embed_tokens.forward(&window_ids)?;
                let type_embed = text_type_embed.broadcast_as(text_embeds.shape())?;
                let text_embeds = (text_embeds + type_embed)?;

                // Forward through TTS LM
                let _hidden = self.forward_tts_lm(&text_embeds, seq_pos)?;
                seq_pos += window_size;
                text_pos = window_end;
            }

            // ── Speech window ───────────────────────────────────────
            for _ in 0..TTS_SPEECH_WINDOW_SIZE {
                if audio_latents.len() >= max_frames {
                    break;
                }

                // Get last hidden state as diffusion condition
                // We need to run one more token through LM to get the condition
                // Use a dummy speech token embedding
                let speech_embed = speech_type_embed.unsqueeze(0)?; // (1, 1, hidden)
                // Initialize with zeros + type embed for speech position
                let dummy_embed = Tensor::zeros((1, 1, hidden_size), self.dtype, &dev)?;
                let speech_input = (dummy_embed + speech_embed)?;

                let hidden_out = self.forward_tts_lm(&speech_input, seq_pos)?;
                seq_pos += 1;

                // Extract condition: last hidden state
                let condition = hidden_out.squeeze(1)?; // (1, hidden)

                // Sample speech latent via DDPM
                let latent = self.sample_speech_latent(&condition, cfg_scale)?;

                // Denormalize for VAE
                let scale = self.speech_scaling_factor
                    .to_dtype(self.dtype)?
                    .broadcast_as(latent.shape())?;
                let bias = self.speech_bias_factor
                    .to_dtype(self.dtype)?
                    .broadcast_as(latent.shape())?;
                let scaled_latent = ((&latent / &scale)? - &bias)?;
                audio_latents.push(scaled_latent);

                // Check EOS
                if self.eos_classifier.should_stop(&condition, 0.5)? {
                    info!("  EOS at frame {}", audio_latents.len());
                    finished = true;
                    break;
                }

                // Feed latent back through acoustic connector for next step
                let acoustic_embed = self.connector.forward(&latent)?;
                let acoustic_input = (acoustic_embed.unsqueeze(1)? + speech_type_embed.unsqueeze(0)?)?;
                let _hidden = self.forward_tts_lm(&acoustic_input, seq_pos)?;
                seq_pos += 1;

                #[allow(clippy::manual_is_multiple_of)]
                if audio_latents.len() % 10 == 0 {
                    info!("  {} frames generated...", audio_latents.len());
                }
            }

            // If no more text, keep generating speech until EOS or max
            if text_pos >= token_ids.len() && !finished {
                // Continue generating without new text
                continue;
            }
        }

        info!("Generated {} speech frames, decoding audio...", audio_latents.len());

        if audio_latents.is_empty() {
            return Ok(vec![]);
        }

        // Stack and decode through VAE
        let latents = Tensor::stack(&audio_latents, 1)?; // (1, frames, vae_dim)
        let latents = latents.transpose(1, 2)?; // (1, vae_dim, frames)
        let audio = self.vae_decoder.decode(&latents)?; // (1, 1, samples)

        let audio = audio.squeeze(0)?.squeeze(0)?;
        let samples: Vec<f32> = audio.to_dtype(DType::F32)?.to_vec1()?;

        Ok(samples)
    }
}

/// Write PCM samples to a WAV file (24kHz, mono, 16-bit).
pub fn save_wav(samples: &[f32], path: &std::path::Path, sample_rate: u32) -> Result<()> {
    use std::io::Write;

    let num_samples = samples.len();
    let data_size = (num_samples * 2) as u32;
    let file_size = 36 + data_size;

    let mut file = std::fs::File::create(path)?;

    file.write_all(b"RIFF")?;
    file.write_all(&file_size.to_le_bytes())?;
    file.write_all(b"WAVE")?;
    file.write_all(b"fmt ")?;
    file.write_all(&16u32.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&1u16.to_le_bytes())?;
    file.write_all(&sample_rate.to_le_bytes())?;
    file.write_all(&(sample_rate * 2).to_le_bytes())?;
    file.write_all(&2u16.to_le_bytes())?;
    file.write_all(&16u16.to_le_bytes())?;
    file.write_all(b"data")?;
    file.write_all(&data_size.to_le_bytes())?;

    for &s in samples {
        let i16_val = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        file.write_all(&i16_val.to_le_bytes())?;
    }

    Ok(())
}
