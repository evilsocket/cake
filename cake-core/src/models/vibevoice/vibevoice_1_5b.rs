//! VibeVoice-1.5B TTS pipeline (non-streaming, multi-speaker).
//!
//! Single 28-layer Qwen2 LM with acoustic+semantic tokenizers.
//! Autoregressive next-token prediction with diffusion-based speech generation.
//! Supports up to 4 distinct speakers with voice cloning from .wav files.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::VarBuilder;
use log::info;

use super::acoustic_connector::AcousticConnector;
use super::config_1_5b::*;
use super::ddpm::DpmSolverPP;
use super::prediction_head::PredictionHead;
use super::vae_decoder::{AcousticVaeDecoder, StreamingConvCache};
use super::vae_encoder::TokenizerEncoder;

/// VibeVoice-1.5B TTS model.
///
/// The LM backbone (28 Qwen2 layers) can be sharded across workers using topology.
/// Each layer either runs locally or is forwarded to a remote worker via `Client`.
/// Non-LM components (prediction head, VAE, encoders) always stay on master.
pub struct VibeVoice1_5B {
    // Single Qwen2 LM (28 layers) — each is either local Transformer or remote Client
    embed_tokens: candle_nn::Embedding,
    lm_norm: candle_nn::RmsNorm,
    layers: Vec<Box<dyn crate::cake::Forwarder>>,
    lm_head_weight: Tensor, // tied to embed_tokens

    // Tokenizers
    acoustic_encoder: TokenizerEncoder,
    acoustic_decoder: AcousticVaeDecoder,
    semantic_encoder: TokenizerEncoder,

    // Connectors (fc1 → RmsNorm → fc2)
    acoustic_connector: AcousticConnector,
    semantic_connector: AcousticConnector,

    // Diffusion
    prediction_head: PredictionHead,
    scheduler: DpmSolverPP,

    // Scaling factors
    speech_scaling_factor: Tensor,
    speech_bias_factor: Tensor,

    // Pre-computed speech_diffusion embedding (reused every diffusion frame)
    speech_diff_embed: Tensor,

    config: VibeVoice1_5BConfig,
    common_cfg: crate::models::common::Config,
    device: Device,
    dtype: DType,
    /// Lightweight context for forwarding through Forwarder trait (holds mutable cache).
    fwd_ctx: crate::cake::Context,
}

impl VibeVoice1_5B {
    pub async fn load(
        config_path: &std::path::Path,
        weight_paths: &[std::path::PathBuf],
        device: &Device,
        diffusion_steps: Option<usize>,
        topology: &crate::cake::Topology,
        cluster_key: Option<&str>,
    ) -> Result<Self> {
        let config = VibeVoice1_5BConfig::from_path(config_path)?;
        let common_cfg = config.into_config();

        info!("Loading VibeVoice-1.5B...");
        let dtype = DType::BF16;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(weight_paths, dtype, device)?
        };

        // Single LM (28 layers)
        let num_layers = config.decoder_config.num_hidden_layers;
        info!("  Loading LM ({num_layers} layers, hidden={})...", config.decoder_config.hidden_size);
        let lm_vb = vb.pp("model").pp("language_model");
        let embed_tokens = candle_nn::embedding(
            common_cfg.vocab_size,
            common_cfg.hidden_size,
            lm_vb.pp("embed_tokens"),
        )?;
        let lm_norm = candle_nn::rms_norm(
            common_cfg.hidden_size,
            common_cfg.rms_norm_eps,
            lm_vb.pp("norm"),
        )?;

        let mut layers: Vec<Box<dyn crate::cake::Forwarder>> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_name = format!("{}.layers.{i}", common_cfg.model_prefix);
            if let Some((_node_name, node)) = topology.get_node_for_layer(&layer_name) {
                info!("  {layer_name} → remote worker at {}", node.host);
                layers.push(Box::new(
                    crate::cake::Client::new(
                        device.clone(),
                        &node.host,
                        &layer_name,
                        cluster_key,
                    )
                    .await?,
                ));
            } else {
                layers.push(Box::new(
                    crate::models::common::Transformer::load_for_vibevoice(
                        lm_vb.pp("layers").pp(i),
                        &common_cfg,
                        crate::backends::create_backend(device),
                    )?,
                ));
            }
        }

        // lm_head weight is tied to embed_tokens
        let lm_head_weight = embed_tokens.embeddings().clone();

        // Acoustic tokenizer (encoder + decoder)
        info!("  Loading acoustic tokenizer...");
        let backend = crate::backends::create_backend(device);
        let acoustic_encoder = TokenizerEncoder::load(
            vb.pp("model").pp("acoustic_tokenizer").pp("encoder"),
            &config.acoustic_tokenizer_config,
            backend.clone(),
        )?;
        let acoustic_decoder = AcousticVaeDecoder::load(
            vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
            &config.acoustic_tokenizer_config,
            backend.clone(),
        )?;

        // Semantic tokenizer (encoder only)
        info!("  Loading semantic tokenizer...");
        // The semantic tokenizer uses a compatible config structure
        let semantic_tok_cfg = super::config::AcousticTokenizerConfig {
            vae_dim: config.semantic_tokenizer_config.vae_dim,
            encoder_n_filters: config.semantic_tokenizer_config.encoder_n_filters,
            decoder_n_filters: None,
            encoder_ratios: config.semantic_tokenizer_config.encoder_ratios.clone(),
            decoder_ratios: None,
            encoder_depths: config.semantic_tokenizer_config.encoder_depths.clone(),
            decoder_depths: None,
            layernorm: config.semantic_tokenizer_config.layernorm.clone(),
            layernorm_eps: config.semantic_tokenizer_config.layernorm_eps,
            causal: config.semantic_tokenizer_config.causal,
        };
        let semantic_encoder = TokenizerEncoder::load(
            vb.pp("model").pp("semantic_tokenizer").pp("encoder"),
            &semantic_tok_cfg,
            backend.clone(),
        )?;

        // Connectors
        info!("  Loading connectors + prediction head...");
        let hidden = config.decoder_config.hidden_size;
        let rms_eps = config.decoder_config.rms_norm_eps;
        let acoustic_connector = AcousticConnector::load(
            vb.pp("model").pp("acoustic_connector"),
            config.acoustic_vae_dim,
            hidden,
            rms_eps,
        )?;
        let semantic_connector = AcousticConnector::load(
            vb.pp("model").pp("semantic_connector"),
            config.semantic_vae_dim,
            hidden,
            rms_eps,
        )?;

        // Prediction head + scheduler
        let steps = diffusion_steps.unwrap_or(config.diffusion_head_config.ddpm_num_inference_steps);
        let scheduler =
            DpmSolverPP::new_cosine(config.diffusion_head_config.ddpm_num_steps, steps);
        let prediction_head = PredictionHead::load(
            vb.pp("model").pp("prediction_head"),
            &config.diffusion_head_config,
            scheduler.timesteps(),
            backend,
        )?;

        let speech_scaling_factor = vb.pp("model").get((), "speech_scaling_factor")?;
        let speech_bias_factor = vb.pp("model").get((), "speech_bias_factor")?;

        info!("  DPM-Solver++: {steps} steps");

        // Pre-compute speech_diffusion embedding (reused every diffusion frame)
        let diff_token = Tensor::new(&[SPEECH_DIFFUSION_ID], device)?.unsqueeze(0)?;
        let speech_diff_embed = embed_tokens.forward(&diff_token)?;

        info!("VibeVoice-1.5B loaded!");

        Ok(Self {
            embed_tokens,
            lm_norm,
            layers,
            lm_head_weight,
            acoustic_encoder,
            acoustic_decoder,
            semantic_encoder,
            acoustic_connector,
            semantic_connector,
            prediction_head,
            scheduler,
            speech_scaling_factor,
            speech_bias_factor,
            speech_diff_embed,
            config,
            common_cfg: common_cfg.clone(),
            device: device.clone(),
            dtype,
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
                backend: crate::backends::create_backend(device),
            },
        })
    }

    /// Forward through LM layers with cache.
    /// Sets the cache in fwd_ctx before each call. Local layers use it for KV caching;
    /// remote layers (Client) ignore it and manage their own cache on the worker.
    async fn forward_lm(
        &mut self,
        embeds: &Tensor,
        index_pos: usize,
        cache: &mut crate::models::common::Cache,
    ) -> Result<Tensor> {
        // Swap the cache into fwd_ctx for local Transformer layers
        self.fwd_ctx.cache = Some(cache.clone());
        let mut h = embeds.clone();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            h = layer.forward_mut(&h, index_pos, i, &mut self.fwd_ctx).await?;
        }
        // Swap the updated cache back
        if let Some(updated) = self.fwd_ctx.cache.take() {
            *cache = updated;
        }
        h = self.lm_norm.forward(&h).map_err(|e| anyhow::anyhow!("lm_norm: {e}"))?;
        Ok(h)
    }

    /// Compute logits from hidden states. Handles both 2D and 3D inputs.
    fn lm_head(&self, hidden: &Tensor) -> Result<Tensor> {
        // embed_tokens weight is (vocab, hidden), we need hidden @ weight^T = logits
        let w = self.lm_head_weight.t()?; // (hidden, vocab)
        let logits = hidden.broadcast_matmul(&w)?;
        Ok(logits)
    }

    /// Encode voice reference from f32 PCM samples (24kHz mono).
    /// Returns: (acoustic_features, connected_embeds)
    pub fn encode_voice_from_samples(&self, samples: &[f32]) -> Result<(Tensor, Tensor)> {
        let audio = Tensor::from_vec(samples.to_vec(), (1, samples.len()), &self.device)?
            .to_dtype(self.dtype)?;
        self.encode_voice_reference(&audio)
    }

    /// Encode voice reference audio tensor to scaled acoustic features.
    /// Returns: (acoustic_features, connected_embeds)
    fn encode_voice_reference(&self, audio: &Tensor) -> Result<(Tensor, Tensor)> {
        // Encode through acoustic encoder: (batch, samples) → (batch, frames, vae_dim)
        let latents = self.acoustic_encoder.encode(audio)?;

        // Apply scaling: features = (latents + bias) * scale
        let scale = self.speech_scaling_factor.to_dtype(self.dtype)?;
        let bias = self.speech_bias_factor.to_dtype(self.dtype)?;
        let features = (latents.broadcast_add(&bias)?.broadcast_mul(&scale))?;

        // Connect to LM hidden space
        let connected = self.acoustic_connector.forward(&features)?;

        Ok((features, connected))
    }

    /// Sample speech latent with CFG using DPM-Solver++.
    fn sample_speech_latent(
        &self,
        pos_cond: &Tensor,
        neg_cond: &Tensor,
        cfg_scale: f32,
    ) -> Result<Tensor> {
        let vae_dim = self.config.acoustic_vae_dim;
        let num_steps = self.scheduler.timesteps().len();

        // Pre-compute condition projection once (constant across all steps)
        let condition = Tensor::cat(&[pos_cond, neg_cond], 0)?;
        let cond_proj = self.prediction_head.project_condition(&condition)?;

        let mut sample =
            Tensor::randn(0f32, 1., (1, vae_dim), &self.device)?.to_dtype(self.dtype)?;
        let mut x0_buffer: Vec<Tensor> = Vec::new();
        let mut ts_buffer: Vec<usize> = Vec::new();

        for step_idx in 0..num_steps {
            let doubled = Tensor::cat(&[&sample, &sample], 0)?;

            // Use forward_fast: pre-computed t_emb + cached silu(cond) across blocks
            let v_pred = self
                .prediction_head
                .forward_fast(&doubled, step_idx, &cond_proj)?;

            let cond_pred = v_pred.narrow(0, 0, 1)?;
            let uncond_pred = v_pred.narrow(0, 1, 1)?;
            let guided =
                (&uncond_pred + (&cond_pred - &uncond_pred)? * cfg_scale as f64)?;

            sample = self
                .scheduler
                .step(&guided, step_idx, &sample, &mut x0_buffer, &mut ts_buffer)?;
        }
        Ok(sample)
    }

    /// Generate audio from text with voice cloning.
    ///
    /// # Arguments
    /// * `token_ids` - Full tokenized input sequence (system prompt + voice section + text + speech_start)
    /// * `speech_input_mask` - Boolean positions where speech embeddings should be spliced in
    /// * `speech_embeds` - Pre-computed speech embeddings from voice reference (connected)
    /// * `max_tokens` - Maximum generation steps
    /// * `cfg_scale` - Classifier-free guidance scale
    pub async fn generate(
        &mut self,
        token_ids: &[u32],
        speech_input_mask: &[bool],
        speech_embeds: &Tensor,
        max_tokens: usize,
        cfg_scale: f32,
    ) -> Result<Vec<f32>> {
        let dev = self.device.clone();
        let _hidden_size = self.config.decoder_config.hidden_size;

        // Create positive and negative KV caches
        let mut pos_cache = crate::models::common::Cache::new(
            true,
            self.dtype,
            &self.common_cfg,
            &dev,
        )?;
        let mut neg_cache = crate::models::common::Cache::new(
            true,
            self.dtype,
            &self.common_cfg,
            &dev,
        )?;

        // === PREFILL POSITIVE ===
        // Build input embeddings with speech spliced in
        let input_ids_t = Tensor::new(token_ids, &dev)?.unsqueeze(0)?;
        let mut embeds = self.embed_tokens.forward(&input_ids_t)?;

        // Splice speech embeddings at marked positions
        let speech_positions: Vec<usize> = speech_input_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &m)| if m { Some(i) } else { None })
            .collect();
        if !speech_positions.is_empty() {
            let num_speech = speech_positions.len();
            let speech_seq_len = speech_embeds.dim(1)?;
            let actual_len = num_speech.min(speech_seq_len);
            for (j, &pos) in speech_positions.iter().take(actual_len).enumerate() {
                let emb = speech_embeds.i((0, j))?.unsqueeze(0)?.unsqueeze(0)?;
                // embeds[:, pos, :] = emb
                let before = if pos > 0 {
                    Some(embeds.narrow(1, 0, pos)?)
                } else {
                    None
                };
                let after = if pos + 1 < token_ids.len() {
                    Some(embeds.narrow(1, pos + 1, token_ids.len() - pos - 1)?)
                } else {
                    None
                };
                let parts: Vec<&Tensor> = [before.as_ref(), Some(&emb), after.as_ref()]
                    .iter()
                    .filter_map(|x| *x)
                    .collect();
                embeds = Tensor::cat(&parts, 1)?;
            }
        }

        // Forward through LM
        let hidden = self.forward_lm(&embeds, 0, &mut pos_cache).await?;
        let mut pos_pos = token_ids.len();

        // === PREFILL NEGATIVE ===
        // Negative prompt is just [speech_start_id]
        let neg_ids = Tensor::new(&[SPEECH_START_ID], &dev)?.unsqueeze(0)?;
        let neg_embeds = self.embed_tokens.forward(&neg_ids)?;
        let _neg_hidden = self.forward_lm(&neg_embeds, 0, &mut neg_cache).await?;
        let mut neg_pos = 1usize;

        // === AUTOREGRESSIVE GENERATION ===
        let mut audio_chunks: Vec<Tensor> = Vec::new();
        let mut last_hidden = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;

        // Pre-compute scaling factors once (avoid per-frame dtype conversion)
        let scale = self.speech_scaling_factor.to_dtype(self.dtype)?;
        let bias = self.speech_bias_factor.to_dtype(self.dtype)?;

        // Valid token IDs for generation (constrained logits)
        let valid_tokens = [SPEECH_START_ID, SPEECH_END_ID, SPEECH_DIFFUSION_ID, EOS_ID];

        // Streaming VAE caches for correct inter-frame context
        let mut acoustic_cache = StreamingConvCache::new(64);
        let mut semantic_cache = StreamingConvCache::new(64);

        for step in 0..max_tokens {
            // Get logits and select next token
            let logits = self.lm_head(&last_hidden)?; // (1, 1, vocab)
            let logits = logits.squeeze(1)?; // (1, vocab)

            // Mask logits to valid tokens only
            let logits_f32 = logits.to_dtype(DType::F32)?;
            let mut scores = vec![f32::NEG_INFINITY; self.common_cfg.vocab_size];
            let logits_vec: Vec<f32> = logits_f32.squeeze(0)?.to_vec1()?;
            for &tid in &valid_tokens {
                if (tid as usize) < scores.len() {
                    scores[tid as usize] = logits_vec[tid as usize];
                }
            }
            let next_token = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(EOS_ID);

            if next_token == EOS_ID {
                info!("  EOS at step {step}");
                break;
            }

            if next_token == SPEECH_END_ID {
                info!("  Speech end at step {step}");
                // Clear streaming caches for next speech segment
                acoustic_cache.clear();
                semantic_cache.clear();
            }

            if next_token == SPEECH_START_ID {
                // Reset negative cache (matches Python refresh_negative=True)
                neg_cache = crate::models::common::Cache::new(
                    true,
                    self.dtype,
                    &self.common_cfg,
                    &dev,
                )?;
                let neg_ids = Tensor::new(&[SPEECH_START_ID], &dev)?.unsqueeze(0)?;
                let neg_embeds = self.embed_tokens.forward(&neg_ids)?;
                let _neg_h = self.forward_lm(&neg_embeds, 0, &mut neg_cache).await?;
                neg_pos = 1;
                // Clear streaming caches
                acoustic_cache.clear();
                semantic_cache.clear();
            }

            let next_embed;

            if next_token == SPEECH_DIFFUSION_ID {
                let frame_start = std::time::Instant::now();

                // Forward negative model with pre-computed speech_diffusion embedding
                let neg_hidden =
                    self.forward_lm(&self.speech_diff_embed.clone(), neg_pos, &mut neg_cache).await?;
                neg_pos += 1;
                let t_neg_lm = frame_start.elapsed();

                let pos_cond = last_hidden.squeeze(1)?; // (1, hidden)
                let neg_cond = neg_hidden.squeeze(1)?; // (1, hidden)

                // Diffusion sampling
                let t0 = std::time::Instant::now();
                let latent = self.sample_speech_latent(&pos_cond, &neg_cond, cfg_scale)?;
                let t_diffusion = t0.elapsed();

                // Decode to audio with streaming cache
                let t0 = std::time::Instant::now();
                let scaled_latent = latent.broadcast_div(&scale)?.broadcast_sub(&bias)?;

                let audio_chunk = self.acoustic_decoder.decode_streaming(
                    &scaled_latent.unsqueeze(0)?.transpose(1, 2)?,
                    &mut acoustic_cache,
                )?;
                let t_vae_decode = t0.elapsed();
                audio_chunks.push(audio_chunk.clone());

                // Encode feedback through semantic encoder with streaming cache
                let t0 = std::time::Instant::now();
                let semantic_features =
                    self.semantic_encoder.encode_streaming(&audio_chunk, &mut semantic_cache)?;
                let t_sem_encode = t0.elapsed();

                // Combine acoustic + semantic for next input
                let t0 = std::time::Instant::now();
                let acoustic_embed = self.acoustic_connector.forward(&latent)?; // (1, hidden)
                let semantic_embed = self.semantic_connector.forward(&semantic_features)?; // (1, frames, hidden)
                let semantic_mean = semantic_embed.mean(1)?; // (1, hidden)
                next_embed = (acoustic_embed + semantic_mean)?.unsqueeze(1)?; // (1, 1, hidden)
                let t_connectors = t0.elapsed();

                let t_total = frame_start.elapsed();

                #[allow(unknown_lints, clippy::manual_is_multiple_of)]
                if audio_chunks.len() % 10 == 0 || audio_chunks.len() <= 3 {
                    info!(
                        "  frame {} ({:.0}ms): neg_lm={:.1}ms diff={:.1}ms vae_dec={:.1}ms sem_enc={:.1}ms conn={:.1}ms",
                        audio_chunks.len(),
                        t_total.as_secs_f64() * 1000.0,
                        t_neg_lm.as_secs_f64() * 1000.0,
                        t_diffusion.as_secs_f64() * 1000.0,
                        t_vae_decode.as_secs_f64() * 1000.0,
                        t_sem_encode.as_secs_f64() * 1000.0,
                        t_connectors.as_secs_f64() * 1000.0,
                    );
                }
            } else {
                // Non-diffusion token: embed normally, skip neg model (Python refresh_negative=True)
                let tid = Tensor::new(&[next_token], &dev)?.unsqueeze(0)?;
                next_embed = self.embed_tokens.forward(&tid)?;
            }

            // Forward through positive LM
            last_hidden = self.forward_lm(&next_embed, pos_pos, &mut pos_cache).await?;
            pos_pos += 1;
        }

        info!("Generated {} speech frames, decoding...", audio_chunks.len());
        if audio_chunks.is_empty() {
            return Ok(vec![]);
        }

        // Concatenate all audio chunks
        let audio = Tensor::cat(
            &audio_chunks.iter().collect::<Vec<_>>(),
            D::Minus1,
        )?;
        let audio_f32 = audio.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
        let peak = audio_f32.abs()?.max(0)?.to_scalar::<f32>()?.max(1e-6);
        let normalized = (audio_f32 / peak as f64)?;
        Ok(normalized.to_vec1()?)
    }
}
