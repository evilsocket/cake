//! VibeVoice-Realtime-0.5B TTS pipeline with voice cloning.
//!
//! Requires a voice prompt (pre-computed KV caches from a reference voice).
//! Uses classifier-free guidance with 4 parallel model streams:
//! positive (base LM + TTS LM) and negative (unconditional).

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;
use log::info;

use super::acoustic_connector::AcousticConnector;
use super::config::VibeVoiceConfig;
use super::ddpm::DpmSolverPP;
use super::eos_classifier::EosClassifier;
use super::prediction_head::PredictionHead;
use super::vae_decoder::AcousticVaeDecoder;
use super::voice_prompt::{self, VoicePrompt};

/// Text tokens processed per window before generating speech frames.
#[allow(dead_code)]
const TTS_TEXT_WINDOW_SIZE: usize = 5;
/// Speech frames generated per text window.
#[allow(dead_code)]
const TTS_SPEECH_WINDOW_SIZE: usize = 6;

/// VibeVoice TTS model with voice cloning support.
pub struct VibeVoiceTTS {
    // Base LM (4 layers) — each local Transformer or remote Client
    base_embed: candle_nn::Embedding,
    base_norm: candle_nn::RmsNorm,
    base_layers: Vec<Box<dyn crate::cake::Forwarder>>,

    // TTS LM (20 layers) — each local Transformer or remote Client
    #[allow(dead_code)]
    tts_embed: candle_nn::Embedding,
    tts_norm: candle_nn::RmsNorm,
    tts_layers: Vec<Box<dyn crate::cake::Forwarder>>,

    // Type embedding: 0=speech, 1=text
    tts_input_types: candle_nn::Embedding,

    // Diffusion + decoding
    prediction_head: PredictionHead,
    scheduler: DpmSolverPP,
    vae_decoder: AcousticVaeDecoder,
    connector: AcousticConnector,
    eos_classifier: EosClassifier,

    speech_scaling_factor: Tensor,
    speech_bias_factor: Tensor,

    config: VibeVoiceConfig,
    common_cfg: crate::models::common::Config,
    device: Device,
    dtype: DType,
    fwd_ctx: crate::cake::Context,
}

impl VibeVoiceTTS {
    pub async fn load(
        config_path: &std::path::Path,
        weights_path: &std::path::Path,
        device: &Device,
        diffusion_steps: Option<usize>,
        topology: &crate::cake::Topology,
        cluster_key: Option<&str>,
    ) -> Result<Self> {
        let config = VibeVoiceConfig::from_path(config_path)?;
        let common_cfg = config.into_config();

        info!("Loading VibeVoice-Realtime-0.5B...");
        // Use F32 for exact numerical match with reference implementation.
        // This disables Flash Attention but ensures correct conditioning.
        let dtype = DType::F32;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], dtype, device)?
        };

        // Base LM (4 layers)
        info!("  Loading base LM (4 layers)...");
        let base_vb = vb.pp("model").pp("language_model");
        let base_embed = candle_nn::embedding(common_cfg.vocab_size, common_cfg.hidden_size, base_vb.pp("embed_tokens"))?;
        // Base LM norm: not in checkpoint (initialized to ones, matching Qwen2Model default)
        let base_norm_w = Tensor::ones(common_cfg.hidden_size, dtype, device)?;
        let base_norm = candle_nn::RmsNorm::new(base_norm_w, common_cfg.rms_norm_eps);
        let mut base_layers: Vec<Box<dyn crate::cake::Forwarder>> = Vec::new();
        for i in 0..4 {
            let layer_name = format!("model.language_model.layers.{i}");
            if let Some((_node_name, node)) = topology.get_node_for_layer(&layer_name) {
                info!("  {layer_name} → remote worker at {}", node.host);
                base_layers.push(Box::new(crate::cake::Client::new(
                    device.clone(), &node.host, &layer_name, cluster_key,
                ).await?));
            } else {
                base_layers.push(Box::new(
                    crate::models::common::Transformer::load_for_vibevoice(base_vb.pp("layers").pp(i), &common_cfg)?,
                ));
            }
        }

        // TTS LM (20 layers)
        let num_tts = config.tts_backbone_num_hidden_layers;
        info!("  Loading TTS LM ({num_tts} layers)...");
        let tts_vb = vb.pp("model").pp("tts_language_model");
        let tts_embed = candle_nn::embedding(common_cfg.vocab_size, common_cfg.hidden_size, tts_vb.pp("embed_tokens"))?;
        let tts_norm = candle_nn::rms_norm(common_cfg.hidden_size, common_cfg.rms_norm_eps, tts_vb.pp("norm"))?;
        let mut tts_layers: Vec<Box<dyn crate::cake::Forwarder>> = Vec::new();
        for i in 0..num_tts {
            let layer_name = format!("model.tts_language_model.layers.{i}");
            if let Some((_node_name, node)) = topology.get_node_for_layer(&layer_name) {
                info!("  {layer_name} → remote worker at {}", node.host);
                tts_layers.push(Box::new(crate::cake::Client::new(
                    device.clone(), &node.host, &layer_name, cluster_key,
                ).await?));
            } else {
                tts_layers.push(Box::new(
                    crate::models::common::Transformer::load_for_vibevoice(tts_vb.pp("layers").pp(i), &common_cfg)?,
                ));
            }
        }

        let tts_input_types = candle_nn::embedding(2, common_cfg.hidden_size, vb.pp("model").pp("tts_input_types"))?;

        info!("  Loading prediction head + VAE + connector...");
        let steps = diffusion_steps.unwrap_or(config.diffusion_head_config.ddpm_num_inference_steps);
        info!("  DPM-Solver++: {steps} steps");
        let scheduler = DpmSolverPP::new_cosine(config.diffusion_head_config.ddpm_num_steps, steps);
        let prediction_head = PredictionHead::load(vb.pp("model").pp("prediction_head"), &config.diffusion_head_config, scheduler.timesteps())?;
        let vae_decoder = AcousticVaeDecoder::load(vb.pp("model").pp("acoustic_tokenizer").pp("decoder"), &config.acoustic_tokenizer_config)?;
        let connector = AcousticConnector::load(vb.pp("model").pp("acoustic_connector"), config.acoustic_vae_dim, config.decoder_config.hidden_size, config.decoder_config.rms_norm_eps)?;
        let eos_classifier = EosClassifier::load(vb.pp("tts_eos_classifier"))?;

        let speech_scaling_factor = vb.pp("model").get((), "speech_scaling_factor")?;
        let speech_bias_factor = vb.pp("model").get((), "speech_bias_factor")?;

        info!("VibeVoice loaded!");

        Ok(Self {
            base_embed, base_norm, base_layers,
            tts_embed, tts_norm, tts_layers,
            tts_input_types,
            prediction_head, scheduler, vae_decoder, connector, eos_classifier,
            speech_scaling_factor, speech_bias_factor,
            config, common_cfg: common_cfg.clone(), device: device.clone(), dtype,
            fwd_ctx: crate::cake::Context {
                args: Default::default(),
                dtype,
                topology: topology.clone(),
                data_path: std::path::PathBuf::new(),
                device: device.clone(),
                config: Some(common_cfg),
                cache: None,
                var_builder: None,
                text_model_arch: crate::TextModelArch::Auto,
                quant: std::sync::Arc::new(crate::utils::NoQuantization),
                listener_override: std::sync::Arc::new(std::sync::Mutex::new(None)),
            },
        })
    }

    /// Forward through a set of transformer layers with a cache.
    async fn forward_layers(
        layers: &mut [Box<dyn crate::cake::Forwarder>],
        norm: Option<&candle_nn::RmsNorm>,
        embeds: &Tensor,
        index_pos: usize,
        cache: &mut crate::models::common::Cache,
        fwd_ctx: &mut crate::cake::Context,
    ) -> Result<Tensor> {
        fwd_ctx.cache = Some(cache.clone());
        let mut h = embeds.clone();
        for (i, layer) in layers.iter_mut().enumerate() {
            h = layer.forward_mut(&h, index_pos, i, fwd_ctx).await?;
        }
        if let Some(updated) = fwd_ctx.cache.take() {
            *cache = updated;
        }
        if let Some(n) = norm {
            h = n.forward(&h).map_err(|e| anyhow::anyhow!("norm: {e}"))?;
        }
        Ok(h)
    }

    /// Sample speech latent with CFG using DPM-Solver++.
    fn sample_speech_latent(&self, pos_cond: &Tensor, neg_cond: &Tensor, cfg_scale: f32) -> Result<Tensor> {
        let vae_dim = self.config.acoustic_vae_dim;
        let num_steps = self.scheduler.timesteps().len();

        // Stack positive+negative for batched prediction
        let condition = Tensor::cat(&[pos_cond, neg_cond], 0)?; // (2, hidden)

        let mut sample = Tensor::randn(0f32, 1., (1, vae_dim), &self.device)?.to_dtype(self.dtype)?;
        let mut x0_buffer: Vec<Tensor> = Vec::new();
        let mut ts_buffer: Vec<usize> = Vec::new();

        for step_idx in 0..num_steps {
            let t = self.scheduler.timesteps()[step_idx];
            let t_tensor = Tensor::new(&[t as f32], &self.device)?.to_dtype(self.dtype)?;

            // Duplicate sample for both conditions
            let doubled = Tensor::cat(&[&sample, &sample], 0)?; // (2, vae_dim)
            let t_doubled = Tensor::cat(&[&t_tensor, &t_tensor], 0)?; // (2,)

            let v_pred = self.prediction_head.forward(&doubled, &t_doubled, &condition)?;

            // CFG: uncond + scale * (cond - uncond)
            let cond_pred = v_pred.narrow(0, 0, 1)?;
            let uncond_pred = v_pred.narrow(0, 1, 1)?;
            let guided = (&uncond_pred + (&cond_pred - &uncond_pred)? * cfg_scale as f64)?;

            sample = self.scheduler.step(&guided, step_idx, &sample, &mut x0_buffer, &mut ts_buffer)?;
        }
        Ok(sample)
    }

    /// Generate audio from text with a voice prompt.
    pub async fn generate(
        &mut self,
        token_ids: &[u32],
        voice_prompt: &VoicePrompt,
        max_frames: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        let dev = self.device.clone();
        let hidden_size = self.config.decoder_config.hidden_size;

        // Create 4 caches (positive + negative × base + TTS)
        let mut pos_base_cache = crate::models::common::Cache::new(true, self.dtype, &self.common_cfg, &dev)?;
        let mut pos_tts_cache = crate::models::common::Cache::new(true, self.dtype, &self.common_cfg, &dev)?;
        let mut neg_base_cache = crate::models::common::Cache::new(true, self.dtype, &self.common_cfg, &dev)?;
        let mut neg_tts_cache = crate::models::common::Cache::new(true, self.dtype, &self.common_cfg, &dev)?;

        // Inject voice prompt KV caches
        voice_prompt::inject_kv_cache(&mut pos_base_cache, &voice_prompt.lm);
        voice_prompt::inject_kv_cache(&mut pos_tts_cache, &voice_prompt.tts_lm);
        voice_prompt::inject_kv_cache(&mut neg_base_cache, &voice_prompt.neg_lm);
        voice_prompt::inject_kv_cache(&mut neg_tts_cache, &voice_prompt.neg_tts_lm);

        let pos_base_seq = voice_prompt.lm.seq_len;
        let pos_tts_seq = voice_prompt.tts_lm.seq_len;
        #[allow(unused)] let neg_base_seq = voice_prompt.neg_lm.seq_len;
        let neg_tts_seq = voice_prompt.neg_tts_lm.seq_len;

        // Type embeddings
        let text_type = self.tts_input_types.forward(&Tensor::new(&[1u32], &dev)?)?;
        let speech_type = self.tts_input_types.forward(&Tensor::new(&[0u32], &dev)?)?;

        let mut pos_base_pos = pos_base_seq;
        let mut pos_tts_pos = pos_tts_seq;
        let mut neg_tts_pos = neg_tts_seq;

        // Get initial negative condition from cached hidden state
        let neg_last_hidden = &voice_prompt.neg_tts_lm.last_hidden_state;
        let neg_seq = neg_last_hidden.dim(1)?;
        let mut neg_cond = neg_last_hidden.narrow(1, neg_seq - 1, 1)?.squeeze(1)?;

        let mut audio_latents: Vec<Tensor> = Vec::new();
        let mut text_cursor = 0;
        let mut pos_last: Option<Tensor> = None;

        // Debug: allow overriding conditions with reference values
        let ref_conds = if let Ok(ref_path) = std::env::var("REF_CONDITIONS") {
            let vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[std::path::PathBuf::from(ref_path)], DType::F32, &dev,
                )?
            };
            Some((
                vb.get_unchecked("pos")?.to_dtype(self.dtype)?,
                vb.get_unchecked("neg")?.to_dtype(self.dtype)?,
            ))
        } else {
            None
        };

        // Interleaved generation: text window → speech window → text window → ...
        // Matching reference: TTS_TEXT_WINDOW_SIZE=5 text tokens, TTS_SPEECH_WINDOW_SIZE=6 speech frames
        while audio_latents.len() < max_frames {
            // ── Text window: process up to 5 text tokens ──
            let window_end = (text_cursor + TTS_TEXT_WINDOW_SIZE).min(token_ids.len());
            let window_tokens = &token_ids[text_cursor..window_end];
            let window_size = window_tokens.len();

            if window_size > 0 {
                let win_ids = Tensor::new(window_tokens, &dev)?.unsqueeze(0)?;
                let win_embeds = self.base_embed.forward(&win_ids)?;

                // Forward through base LM
                let base_hidden = Self::forward_layers(
                    &mut self.base_layers, Some(&self.base_norm), &win_embeds,
                    pos_base_pos, &mut pos_base_cache,
                    &mut self.fwd_ctx,
                ).await?;
                pos_base_pos += window_size;

                // Forward through TTS LM (base hidden + text type)
                let tts_input = (base_hidden + text_type.broadcast_as(win_embeds.shape())?)?;
                pos_last = Some(Self::forward_layers(
                    &mut self.tts_layers, Some(&self.tts_norm), &tts_input,
                    pos_tts_pos, &mut pos_tts_cache, &mut self.fwd_ctx,
                ).await?);
                pos_tts_pos += window_size;

                text_cursor = window_end;
            }

            // Need at least one text window before generating speech
            let pos_hidden = match pos_last.clone() {
                Some(h) => h,
                None => break,
            };

            // ── Speech window: generate up to 6 speech frames ──
            for _speech_idx in 0..TTS_SPEECH_WINDOW_SIZE {
                if audio_latents.len() >= max_frames { break; }

                let seq_len = pos_hidden.dim(1)?;
                let pos_cond = if let Some((ref_pos, _)) = &ref_conds {
                    ref_pos.clone()
                } else {
                    pos_hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?
                };
                let neg_cond_use = if let Some((_, ref_neg)) = &ref_conds {
                    ref_neg.clone()
                } else {
                    neg_cond.clone()
                };

                // Sample with CFG
                let latent = self.sample_speech_latent(&pos_cond, &neg_cond_use, cfg_scale)?;

                // Denormalize for VAE
                let scale = self.speech_scaling_factor.to_dtype(self.dtype)?.broadcast_as(latent.shape())?;
                let bias = self.speech_bias_factor.to_dtype(self.dtype)?.broadcast_as(latent.shape())?;
                audio_latents.push(((&latent / &scale)? - &bias)?);

                // Check EOS
                if self.eos_classifier.should_stop(&pos_cond, 0.9)? {
                    info!("  EOS at frame {}", audio_latents.len());
                    break;
                }

                // Feed acoustic latent back through both TTS LMs
                let acoustic_embed = self.connector.forward(&latent)?;
                let speech_embed = (acoustic_embed.unsqueeze(1)?
                    + speech_type.unsqueeze(0)?.broadcast_as((1, 1, hidden_size))?)?;

                pos_last = Some(Self::forward_layers(
                    &mut self.tts_layers, Some(&self.tts_norm), &speech_embed,
                    pos_tts_pos, &mut pos_tts_cache, &mut self.fwd_ctx,
                ).await?);
                pos_tts_pos += 1;

                let neg_out = Self::forward_layers(
                    &mut self.tts_layers, Some(&self.tts_norm), &speech_embed,
                    neg_tts_pos, &mut neg_tts_cache,
                    &mut self.fwd_ctx,
                ).await?;
                neg_cond = neg_out.narrow(1, neg_out.dim(1)? - 1, 1)?.squeeze(1)?;
                neg_tts_pos += 1;

                #[allow(clippy::manual_is_multiple_of)]
                if audio_latents.len() % 10 == 0 {
                    info!("  {} frames...", audio_latents.len());
                }
            }

            // If all text consumed and no more to process, keep generating speech
            if text_cursor >= token_ids.len() {
                // Continue generating speech-only windows until max or EOS
                while audio_latents.len() < max_frames {
                    let pos_hidden = match &pos_last {
                        Some(h) => h,
                        None => break,
                    };
                    let seq_len = pos_hidden.dim(1)?;
                    let pos_cond = pos_hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

                    let latent = self.sample_speech_latent(&pos_cond, &neg_cond, cfg_scale)?;
                    let scale = self.speech_scaling_factor.to_dtype(self.dtype)?.broadcast_as(latent.shape())?;
                    let bias = self.speech_bias_factor.to_dtype(self.dtype)?.broadcast_as(latent.shape())?;
                    audio_latents.push(((&latent / &scale)? - &bias)?);

                    if self.eos_classifier.should_stop(&pos_cond, 0.9)? {
                        info!("  EOS at frame {}", audio_latents.len());
                        break;
                    }

                    let acoustic_embed = self.connector.forward(&latent)?;
                    let speech_embed = (acoustic_embed.unsqueeze(1)?
                        + speech_type.unsqueeze(0)?.broadcast_as((1, 1, hidden_size))?)?;

                    pos_last = Some(Self::forward_layers(
                        &mut self.tts_layers, Some(&self.tts_norm), &speech_embed,
                        pos_tts_pos, &mut pos_tts_cache, &mut self.fwd_ctx,
                    ).await?);
                    pos_tts_pos += 1;

                    let neg_out = Self::forward_layers(
                        &mut self.tts_layers, Some(&self.tts_norm), &speech_embed,
                        neg_tts_pos, &mut neg_tts_cache,
                        &mut self.fwd_ctx,
                ).await?;
                    neg_cond = neg_out.narrow(1, neg_out.dim(1)? - 1, 1)?.squeeze(1)?;
                    neg_tts_pos += 1;

                    #[allow(clippy::manual_is_multiple_of)]
                    if audio_latents.len() % 10 == 0 {
                        info!("  {} frames...", audio_latents.len());
                    }
                }
                break;
            }
        }

        info!("Generated {} frames, decoding...", audio_latents.len());
        if audio_latents.is_empty() { return Ok(vec![]); }

        // Allow overriding latents with reference data for VAE isolation testing
        let latents = if let Ok(ref_path) = std::env::var("REF_LATENTS") {
            info!("  [debug] Loading reference latents from {}", ref_path);
            let ref_vb = unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[std::path::PathBuf::from(ref_path)],
                    DType::F32,
                    &dev,
                )?
            };
            let denorm = ref_vb.get_unchecked("denorm")?.to_dtype(self.dtype)?; // (18, 64)
            denorm.unsqueeze(0)?.transpose(1, 2)? // (1, 64, 18)
        } else {
            Tensor::stack(&audio_latents, 1)?.transpose(1, 2)?
        };
        let audio = self.vae_decoder.decode(&latents)?;
        let audio_f32 = audio.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let peak = audio_f32.abs()?.max(0)?.to_scalar::<f32>()?.max(1e-6);
        let normalized = (audio_f32 / peak as f64)?;
        Ok(normalized.to_vec1()?)
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
