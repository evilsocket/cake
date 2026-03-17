//! VibeVoice-Realtime-0.5B TTS pipeline.
//!
//! Two-LM architecture:
//! 1. Base LM (4 layers) encodes text → hidden states
//! 2. TTS LM (20 layers) takes text embeds + base LM hidden states → conditions
//! 3. Diffusion head (4 layers, DDPM) generates acoustic latents per frame
//! 4. σ-VAE decoder converts latents to 24kHz audio

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

/// Text tokens processed per window before generating speech frames.
#[allow(dead_code)]
const TTS_TEXT_WINDOW_SIZE: usize = 5;
/// Speech frames generated per text window.
#[allow(dead_code)]
const TTS_SPEECH_WINDOW_SIZE: usize = 6;

/// VibeVoice TTS model.
pub struct VibeVoiceTTS {
    // Base language model (4 layers — text encoder)
    base_embed: candle_nn::Embedding,
    base_layers: Vec<crate::models::common::Transformer>,
    base_cache: crate::models::common::Cache,

    // TTS language model (20 layers — speech generation)
    tts_embed: candle_nn::Embedding,
    tts_norm: candle_nn::RmsNorm,
    tts_layers: Vec<crate::models::common::Transformer>,
    tts_cache: crate::models::common::Cache,

    // Input type embedding: 0 = speech, 1 = text
    tts_input_types: candle_nn::Embedding,

    // Diffusion + decoding
    prediction_head: PredictionHead,
    scheduler: DdpmScheduler,
    vae_decoder: AcousticVaeDecoder,
    connector: AcousticConnector,
    eos_classifier: EosClassifier,

    speech_scaling_factor: Tensor,
    speech_bias_factor: Tensor,

    config: VibeVoiceConfig,
    device: Device,
    dtype: DType,
}

impl VibeVoiceTTS {
    pub fn load(
        config_path: &std::path::Path,
        weights_path: &std::path::Path,
        device: &Device,
        diffusion_steps: Option<usize>,
    ) -> Result<Self> {
        let config = VibeVoiceConfig::from_path(config_path)?;
        let common_cfg = config.into_config();

        info!("Loading VibeVoice-Realtime-0.5B...");

        let dtype = if matches!(device, Device::Cuda(_)) { DType::F16 } else { DType::F32 };

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], dtype, device)?
        };

        // Base LM (4 layers for text encoding)
        info!("  Loading base LM (4 layers)...");
        let base_vb = vb.pp("model").pp("language_model");
        let base_embed = candle_nn::embedding(
            common_cfg.vocab_size, common_cfg.hidden_size, base_vb.pp("embed_tokens"),
        )?;
        let mut base_layers = Vec::new();
        for i in 0..4 {
            base_layers.push(crate::models::common::Transformer::load_for_vibevoice(
                base_vb.pp("layers").pp(i), &common_cfg,
            )?);
        }
        let base_cache = crate::models::common::Cache::new(false, dtype, &common_cfg, device)?;

        // TTS LM (20 layers for speech generation)
        let num_tts_layers = config.tts_backbone_num_hidden_layers;
        info!("  Loading TTS LM ({} layers)...", num_tts_layers);
        let tts_vb = vb.pp("model").pp("tts_language_model");
        let tts_embed = candle_nn::embedding(
            common_cfg.vocab_size, common_cfg.hidden_size, tts_vb.pp("embed_tokens"),
        )?;
        let tts_norm = candle_nn::rms_norm(
            common_cfg.hidden_size, common_cfg.rms_norm_eps, tts_vb.pp("norm"),
        )?;
        let mut tts_layers = Vec::new();
        for i in 0..num_tts_layers {
            tts_layers.push(crate::models::common::Transformer::load_for_vibevoice(
                tts_vb.pp("layers").pp(i), &common_cfg,
            )?);
        }
        let tts_cache = crate::models::common::Cache::new(true, dtype, &common_cfg, device)?;

        // Type embedding
        let tts_input_types = candle_nn::embedding(
            2, common_cfg.hidden_size, vb.pp("model").pp("tts_input_types"),
        )?;

        // Diffusion
        info!("  Loading prediction head...");
        let prediction_head = PredictionHead::load(
            vb.pp("model").pp("prediction_head"), &config.diffusion_head_config,
        )?;

        // VAE decoder
        info!("  Loading acoustic VAE decoder...");
        let vae_decoder = AcousticVaeDecoder::load(
            vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
            &config.acoustic_tokenizer_config,
        )?;

        // Connector + EOS
        info!("  Loading connector + EOS...");
        let connector = AcousticConnector::load(
            vb.pp("model").pp("acoustic_connector"),
            config.acoustic_vae_dim, config.decoder_config.hidden_size,
            config.decoder_config.rms_norm_eps,
        )?;
        let eos_classifier = EosClassifier::load(vb.pp("tts_eos_classifier"))?;

        let speech_scaling_factor = vb.pp("model").get((), "speech_scaling_factor")?;
        let speech_bias_factor = vb.pp("model").get((), "speech_bias_factor")?;

        let inference_steps = diffusion_steps
            .unwrap_or(config.diffusion_head_config.ddpm_num_inference_steps);
        info!("  DDPM: {} inference steps", inference_steps);
        let scheduler = DdpmScheduler::new_cosine(
            config.diffusion_head_config.ddpm_num_steps, inference_steps,
        );

        info!("VibeVoice model loaded!");

        Ok(Self {
            base_embed, base_layers, base_cache,
            tts_embed, tts_norm, tts_layers, tts_cache,
            tts_input_types,
            prediction_head, scheduler, vae_decoder, connector, eos_classifier,
            speech_scaling_factor, speech_bias_factor,
            config, device: device.clone(), dtype,
        })
    }

    /// Forward through base LM (text encoder, 4 layers).
    fn forward_base_lm(&mut self, embeds: &Tensor) -> Result<Tensor> {
        let mut h = embeds.clone();
        for (i, layer) in self.base_layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, 0, i, &mut self.base_cache)?;
        }
        Ok(h)
    }

    /// Forward through TTS LM (20 layers) with KV caching.
    fn forward_tts_lm(&mut self, embeds: &Tensor, index_pos: usize) -> Result<Tensor> {
        let mut h = embeds.clone();
        for (i, layer) in self.tts_layers.iter().enumerate() {
            h = layer.forward_with_cache(&h, index_pos, i, &mut self.tts_cache)?;
        }
        self.tts_norm.forward(&h).map_err(|e| anyhow::anyhow!("tts_norm: {e}"))
    }

    /// Sample speech latent via DDPM.
    fn sample_speech_latent(&self, condition: &Tensor, cfg_scale: f32) -> Result<Tensor> {
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

    /// Generate audio from text.
    pub fn generate(
        &mut self,
        token_ids: &[u32],
        max_frames: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        let dev = self.device.clone();
        let hidden_size = self.config.decoder_config.hidden_size;

        // Reset caches
        self.base_cache.clear();
        self.tts_cache.clear();

        // Type embeddings
        let text_type = self.tts_input_types.forward(
            &Tensor::new(&[1u32], &dev)?,
        )?;
        let speech_type = self.tts_input_types.forward(
            &Tensor::new(&[0u32], &dev)?,
        )?;

        // Step 1: Encode ALL text through base LM
        let input_ids = Tensor::new(token_ids, &dev)?.unsqueeze(0)?;
        let text_embeds = self.base_embed.forward(&input_ids)?;
        let base_hidden = self.forward_base_lm(&text_embeds)?;
        // base_hidden: (1, seq_len, hidden_size) — text context

        // Step 2: Feed text through TTS LM
        // TTS LM input = TTS embeddings + base LM hidden states + text type embedding
        let tts_text_embeds = self.tts_embed.forward(&input_ids)?;
        // Splice: replace TTS embeddings with base LM hidden states (reference: inputs_embeds[:, start_idx:, :] = lm_last_hidden_state)
        let tts_input = (base_hidden + text_type.broadcast_as(tts_text_embeds.shape())?)?;

        let tts_hidden = self.forward_tts_lm(&tts_input, 0)?;
        let mut seq_pos = token_ids.len();

        // Step 3: Generate speech frames
        let mut audio_latents: Vec<Tensor> = Vec::new();

        for _frame in 0..max_frames {
            // Get condition from last TTS LM hidden state
            let condition = tts_hidden.narrow(1, tts_hidden.dim(1)? - 1, 1)?.squeeze(1)?;

            // Sample speech latent
            let latent = self.sample_speech_latent(&condition, cfg_scale)?;

            // Denormalize for VAE
            let scale = self.speech_scaling_factor.to_dtype(self.dtype)?
                .broadcast_as(latent.shape())?;
            let bias = self.speech_bias_factor.to_dtype(self.dtype)?
                .broadcast_as(latent.shape())?;
            audio_latents.push(((&latent / &scale)? - &bias)?);

            // Check EOS
            if self.eos_classifier.should_stop(&condition, 0.5)? {
                info!("  EOS at frame {}", audio_latents.len());
                break;
            }

            // Feed acoustic latent back through connector → TTS LM for next frame
            let acoustic_embed = self.connector.forward(&latent)?;
            let speech_input = (acoustic_embed.unsqueeze(1)?
                + speech_type.unsqueeze(0)?.broadcast_as((1, 1, hidden_size))?)?;
            let _tts_hidden = self.forward_tts_lm(&speech_input, seq_pos)?;
            seq_pos += 1;

            #[allow(clippy::manual_is_multiple_of)]
            if audio_latents.len() % 10 == 0 {
                info!("  {} frames...", audio_latents.len());
            }
        }

        info!("Generated {} frames, decoding...", audio_latents.len());
        if audio_latents.is_empty() {
            return Ok(vec![]);
        }

        let latents = Tensor::stack(&audio_latents, 1)?.transpose(1, 2)?;
        let audio = self.vae_decoder.decode(&latents)?;
        let samples: Vec<f32> = audio.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
        Ok(samples)
    }
}

/// Write PCM samples to a WAV file.
pub fn save_wav(samples: &[f32], path: &std::path::Path, sample_rate: u32) -> Result<()> {
    use std::io::Write;
    let data_size = (samples.len() * 2) as u32;
    let mut f = std::fs::File::create(path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_size).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        f.write_all(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())?;
    }
    Ok(())
}
