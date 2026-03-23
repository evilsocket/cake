//! LuxTTS main model -- implements `Generator` and `TextGenerator` for the sharding machinery.
//!
//! Weight layout:
//! - `embed.weight` [vocab_size, text_dim] -- phoneme embedding (top-level)
//! - `text_encoder.*` -- text encoder
//! - `fm_decoder.in_proj` [fm_dim, feat_dim*3] -- input projection
//! - `fm_decoder.out_proj` [feat_dim, fm_dim] -- output projection
//! - `fm_decoder.time_embed.0` [384, 192] -- time MLP layer 1
//! - `fm_decoder.time_embed.2` [192, 384] -- time MLP layer 2
//! - `fm_decoder.stack_time_emb.{stack}.1` [fm_dim, 192] -- per-stack time projection
//! - `fm_decoder.layers.{i}.*` -- FM decoder layers (shardable)
//! - `backbone.*`, `head.*` -- Vocos vocoder

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;

use crate::cake::{Context, Forwarder};
use crate::models::chat::Message;
use crate::models::{Generator, TextGenerator, Token};

use super::block::ZipformerBlock;
use super::config::LuxTTSConfig;
use super::euler_solver::EulerSolver;
use super::mel;
use super::text_encoder::TextEncoder;
use super::tokenizer::Phonemizer;
use super::vocos::{self, Vocos};

/// BypassModule: combines original (pre-downsample) with processed (upsampled) output.
/// `output = bypass_scale * src_orig + (1 - bypass_scale) * processed`
struct BypassModule {
    bypass_scale: Tensor, // [dim]
}

impl BypassModule {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let bypass_scale = vb.get(dim, "bypass_scale")?;
        Ok(Self { bypass_scale })
    }

    fn forward(&self, src_orig: &Tensor, processed: &Tensor) -> Result<Tensor> {
        // Python: output = src_orig + (processed - src_orig) * bypass_scale
        // bypass_scale is the weight on the non-residual (processed) path
        let scale = self.bypass_scale.unsqueeze(0)?.unsqueeze(0)?;
        let diff = (processed - src_orig)?;
        let out = (src_orig + diff.broadcast_mul(&scale)?)?;
        Ok(out)
    }
}

/// SimpleDownsample: weighted average over groups of `ds` frames using learned softmax weights.
fn simple_downsample(src: &Tensor, ds: usize, bias: &Tensor) -> Result<Tensor> {
    // src: [batch, seq_len, dim], bias: [ds]
    let (batch, seq_len, dim) = src.dims3()?;
    let d_seq_len = seq_len.div_ceil(ds);
    let padded_len = d_seq_len * ds;

    // Pad by repeating last frame if needed
    let src = if padded_len > seq_len {
        let last_frame = src.narrow(1, seq_len - 1, 1)?; // [batch, 1, dim]
        let pad_count = padded_len - seq_len;
        let padding = last_frame.expand((batch, pad_count, dim))?;
        Tensor::cat(&[src, &padding], 1)?
    } else {
        src.clone()
    };

    // Reshape to [batch, d_seq_len, ds, dim]
    let src = src.reshape((batch, d_seq_len, ds, dim))?;

    // Softmax weights: bias [ds] -> softmax -> [1, 1, ds, 1]
    let b = bias.unsqueeze(0)?;
    let max = b.max_keepdim(1)?;
    let exp = b.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(1)?;
    let weights = exp.broadcast_div(&sum)?; // [1, ds]
    let weights = weights.reshape((1, 1, ds, 1))?;

    // Weighted sum over ds dimension
    let weighted = src.broadcast_mul(&weights)?; // [batch, d_seq_len, ds, dim]
    let result = weighted.sum(2)?; // [batch, d_seq_len, dim]

    Ok(result)
}

/// SimpleUpsample: repeat each frame `ds` times.
fn simple_upsample(src: &Tensor, ds: usize) -> Result<Tensor> {
    // src: [batch, seq_len, dim]
    let (batch, seq_len, dim) = src.dims3()?;
    // [batch, seq_len, 1, dim] -> expand to [batch, seq_len, ds, dim] -> reshape
    let expanded = src.unsqueeze(2)?.expand((batch, seq_len, ds, dim))?;
    let result = expanded.reshape((batch, seq_len * ds, dim))?;
    Ok(result)
}

/// LuxTTS text-to-speech model.
pub struct LuxTTS {
    ctx: Context,
    // Non-shardable components (always on master)
    phonemizer: Phonemizer,
    text_encoder: TextEncoder,
    vocos: Vocos,
    // FM decoder projections
    fm_in_proj_weight: Tensor,
    fm_in_proj_bias: Option<Tensor>,
    fm_out_proj_weight: Tensor,
    fm_out_proj_bias: Option<Tensor>,
    // Time embedding MLP: Linear -> SiLU -> Linear
    time_embed_0_weight: Tensor,
    time_embed_0_bias: Option<Tensor>,
    time_embed_2_weight: Tensor,
    time_embed_2_bias: Option<Tensor>,
    // Per-stack time embedding projections
    stack_time_emb_weights: Vec<Tensor>,
    stack_time_emb_biases: Vec<Option<Tensor>>,
    // Per-stack downsample bias (None for ds=1 stacks)
    downsample_biases: Vec<Option<Tensor>>,
    // Per-stack bypass combiner (None for ds=1 stacks)
    out_combiners: Vec<Option<BypassModule>>,

    // Shardable FM decoder blocks
    fm_blocks: Vec<Box<dyn Forwarder>>,

    // Config
    config: LuxTTSConfig,

    // State for TextGenerator interface
    prompt_text: String,
    generated: bool,
}

/// Sinusoidal time embedding: cat([cos(t*freqs), sin(t*freqs)]).
fn sinusoidal_time_embedding(t: f32, dim: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let half = dim / 2;
    let mut data = vec![0.0f32; dim];
    let log_10000 = 10000.0f32.ln();
    for i in 0..half {
        let freq = (-log_10000 / (half as f32 - 1.0) * i as f32).exp();
        let arg = t * freq;
        data[i] = arg.cos();
        data[half + i] = arg.sin();
    }
    Ok(Tensor::from_vec(data, (1, dim), device)?.to_dtype(dtype)?)
}

#[async_trait]
impl Generator for LuxTTS {
    type Shardable = ZipformerBlock;
    const MODEL_NAME: &'static str = "luxtts";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let config = LuxTTSConfig::from_path(&ctx.data_path.join("config.json"))?;
        let m = &config.model;

        log::info!(
            "[LuxTTS] Loading model ({} FM decoder layers)...",
            config.total_fm_layers()
        );

        // Load VarBuilder from safetensors
        let model_weights = ctx.data_path.join("model.safetensors");

        let vb = if let Some(ref vb) = ctx.var_builder {
            vb.clone()
        } else {
            let storage = crate::utils::tensor_storage::SafetensorsStorage::from_file(&model_weights)?;
            let vb_data = storage.load_all(ctx.dtype, &ctx.device)?;
            VarBuilder::from_tensors(vb_data, ctx.dtype, &ctx.device)
        };

        // Text encoder (embed is at top level, encoder under text_encoder)
        log::info!(
            "[LuxTTS] Loading text encoder ({} layers, dim={})...",
            m.text_encoder_num_layers,
            m.text_encoder_dim
        );
        let text_encoder =
            TextEncoder::load(&config, vb.clone(), vb.pp("text_encoder"), ctx.backend.clone())?;

        // FM decoder projections
        // in_proj: [fm_dim, feat_dim*3] where 3 = noise + text_cond + speech_cond
        let fm_in_proj_weight = vb.pp("fm_decoder.in_proj").get((m.fm_decoder_dim, m.feat_dim * 3), "weight")?;
        let fm_in_proj_bias = Some(vb.pp("fm_decoder.in_proj").get(m.fm_decoder_dim, "bias")?);
        let fm_out_proj_weight = vb.pp("fm_decoder.out_proj").get((m.feat_dim, m.fm_decoder_dim), "weight")?;
        let fm_out_proj_bias = Some(vb.pp("fm_decoder.out_proj").get(m.feat_dim, "bias")?);

        // Time embedding MLP: time_embed.0 and time_embed.2 (Sequential indices)
        let time_embed_0_weight = vb.pp("fm_decoder.time_embed.0").get((m.time_embed_dim * 2, m.time_embed_dim), "weight")?;
        let time_embed_0_bias = Some(vb.pp("fm_decoder.time_embed.0").get(m.time_embed_dim * 2, "bias")?);
        let time_embed_2_weight = vb.pp("fm_decoder.time_embed.2").get((m.time_embed_dim, m.time_embed_dim * 2), "weight")?;
        let time_embed_2_bias = Some(vb.pp("fm_decoder.time_embed.2").get(m.time_embed_dim, "bias")?);

        // Per-stack time embedding projections
        let num_stacks = m.fm_decoder_num_layers.len();
        let mut stack_time_emb_weights = Vec::with_capacity(num_stacks);
        let mut stack_time_emb_biases = Vec::with_capacity(num_stacks);
        for s in 0..num_stacks {
            let w = vb.pp(format!("fm_decoder.stack_time_emb.{s}.1")).get((m.fm_decoder_dim, m.time_embed_dim), "weight")?;
            let b = Some(vb.pp(format!("fm_decoder.stack_time_emb.{s}.1")).get(m.fm_decoder_dim, "bias")?);
            stack_time_emb_weights.push(w);
            stack_time_emb_biases.push(b);
        }

        // Per-stack downsample biases and bypass combiners
        let mut downsample_biases = Vec::with_capacity(num_stacks);
        let mut out_combiners = Vec::with_capacity(num_stacks);
        for s in 0..num_stacks {
            let ds = m.fm_decoder_downsampling_factor[s];
            if ds > 1 {
                let bias = vb
                    .pp(format!("fm_decoder.downsample.{s}"))
                    .get(ds, "bias")?;
                downsample_biases.push(Some(bias));
                let combiner =
                    BypassModule::load(m.fm_decoder_dim, vb.pp(format!("fm_decoder.out_combiner.{s}")))?;
                out_combiners.push(Some(combiner));
            } else {
                downsample_biases.push(None);
                out_combiners.push(None);
            }
        }

        // FM decoder blocks (shardable)
        log::info!(
            "[LuxTTS] Loading {} FM decoder blocks...",
            config.total_fm_layers()
        );
        let total_layers = config.total_fm_layers();

        let mut fm_blocks: Vec<Option<Box<dyn Forwarder>>> =
            (0..total_layers).map(|_| None).collect();

        // Pass 1: local layers
        for (i, block) in fm_blocks.iter_mut().enumerate() {
            let block_name = format!("fm_decoder.layers.{i}");
            if ctx.topology.get_node_for_layer(&block_name).is_none() {
                log::info!("  loading {} ...", &block_name);
                *block = Some(ZipformerBlock::load(block_name, ctx)?);
            }
        }

        // Pass 2: remote layers
        for (i, block) in fm_blocks.iter_mut().enumerate() {
            let block_name = format!("fm_decoder.layers.{i}");
            if let Some((_node_name, node)) = ctx.topology.get_node_for_layer(&block_name) {
                log::info!("  connecting {} to {} ...", &block_name, &node.host);
                *block = Some(Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        &block_name,
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                ));
            }
        }

        let fm_blocks: Vec<Box<dyn Forwarder>> =
            fm_blocks.into_iter().map(|b| b.unwrap()).collect();
        for block in &fm_blocks {
            log::info!("  {}", block);
        }

        // Vocos vocoder — loaded from separate vocos.safetensors
        log::info!("[LuxTTS] Loading Vocos vocoder...");
        let vocos_path = ctx.data_path.join("vocos.safetensors");
        let vocos_vb = if vocos_path.exists() {
            let storage = crate::utils::tensor_storage::SafetensorsStorage::from_file(&vocos_path)?;
            let vb_data = storage.load_all(ctx.dtype, &ctx.device)?;
            VarBuilder::from_tensors(vb_data, ctx.dtype, &ctx.device)
        } else {
            // Fall back to main weights (vocoder might be embedded)
            vb.clone()
        };
        let vocos = Vocos::load(
            m.feat_dim,
            m.fm_decoder_dim,
            config.feature.n_fft,
            config.feature.hop_length,
            vocos_vb,
            ctx.backend.clone(),
        )?;

        // Phonemizer
        let tokens_path = ctx.data_path.join("tokens.txt");
        let dict_path = ctx.data_path.join("cmudict-0.7b-ipa.txt");
        let dict_path_opt = if dict_path.exists() {
            Some(dict_path.as_path())
        } else {
            None
        };
        let phonemizer = Phonemizer::load(&tokens_path, dict_path_opt)?;

        log::info!(
            "[LuxTTS] Model loaded - mem={}",
            human_bytes::human_bytes(
                memory_stats::memory_stats()
                    .map(|m| m.physical_mem)
                    .unwrap_or(0) as f64
            )
        );

        Ok(Some(Box::new(Self {
            ctx: ctx.clone(),
            phonemizer,
            text_encoder,
            vocos,
            fm_in_proj_weight,
            fm_in_proj_bias,
            fm_out_proj_weight,
            fm_out_proj_bias,
            time_embed_0_weight,
            time_embed_0_bias,
            time_embed_2_weight,
            time_embed_2_bias,
            stack_time_emb_weights,
            stack_time_emb_biases,
            downsample_biases,
            out_combiners,
            fm_blocks,
            config,
            prompt_text: String::new(),
            generated: false,
        })))
    }
}

#[async_trait]
impl TextGenerator for LuxTTS {
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.prompt_text = message.content;
        self.generated = false;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.prompt_text.clear();
        self.generated = false;
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        for block in &mut self.fm_blocks {
            block.goodbye().await?;
        }
        Ok(())
    }

    async fn next_token(&mut self, _index: usize) -> Result<Token> {
        Ok(Token {
            id: 0,
            text: None,
            is_end_of_stream: true,
        })
    }

    fn generated_tokens(&self) -> usize {
        if self.generated {
            1
        } else {
            0
        }
    }
}

impl LuxTTS {
    /// Run the full TTS pipeline: text -> phonemes -> mel -> waveform.
    pub async fn generate_speech(
        &mut self,
        text: &str,
        reference_audio: Option<&[f32]>,
        t_shift: f32,
        _cfg_scale: f32,
        num_steps: usize,
        speed: f32,
    ) -> Result<Vec<f32>> {
        let device = self.ctx.device.clone();
        let dtype = self.ctx.dtype;
        let m = &self.config.model;

        // 1. Phonemize text (or use pre-computed token IDs)
        let token_ids = if let Some(ref ids_path) = self.ctx.args.tts_token_ids {
            log::info!("[LuxTTS] Loading pre-computed token IDs from {}", ids_path);
            let content = std::fs::read_to_string(ids_path)?;
            content.split_whitespace()
                .map(|s| s.parse::<u32>().map_err(|e| anyhow!("invalid token id: {e}")))
                .collect::<Result<Vec<u32>>>()?
        } else {
            log::info!("[LuxTTS] Phonemizing: \"{}\"", text);
            self.phonemizer.tokenize(text)?
        };
        log::info!("[LuxTTS] {} phoneme tokens", token_ids.len());

        let token_tensor = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;

        // 2. Text encoder -> [1, seq, feat_dim]
        log::info!("[LuxTTS] Running text encoder...");
        let text_cond = self.text_encoder.forward(&token_tensor)?;

        // 3. Determine target frames and prepare conditions
        let text_seq_len = text_cond.dim(1)?;
        let target_frames = ((text_seq_len as f32 / speed) as usize).max(1);

        // Expand text_cond to target_frames by repeating/interpolating
        // For now: repeat each frame to fill target length
        let text_cond_expanded = if text_seq_len != target_frames {
            let indices: Vec<u32> = (0..target_frames)
                .map(|i| ((i * text_seq_len) / target_frames) as u32)
                .collect();
            let idx = Tensor::new(indices.as_slice(), &device)?;
            text_cond.index_select(&idx, 1)?
        } else {
            text_cond.clone()
        };

        // Speech condition from reference audio
        let speech_cond = if let Some(ref_audio) = reference_audio {
            let mel_spec = mel::mel_spectrogram(
                ref_audio,
                self.config.feature.n_fft,
                self.config.feature.hop_length,
                self.config.feature.n_mels,
                self.config.feature.sample_rate,
                &device,
                dtype,
            )?;
            // mel: [1, n_mels, time] -> [1, time, n_mels]
            let mel_t = mel_spec.transpose(1, 2)?;
            // Expand/truncate to target_frames
            let mel_frames = mel_t.dim(1)?;
            if mel_frames != target_frames {
                let indices: Vec<u32> = (0..target_frames)
                    .map(|i| (i * mel_frames / target_frames).min(mel_frames - 1) as u32)
                    .collect();
                let idx = Tensor::new(indices.as_slice(), &device)?;
                mel_t.index_select(&idx, 1)?
            } else {
                mel_t
            }
        } else {
            Tensor::zeros((1, target_frames, m.feat_dim), dtype, &device)?
        };

        // 4. Flow matching with Euler solver
        log::info!("[LuxTTS] Running flow matching ({} steps, {} frames)...", num_steps, target_frames);
        let solver = EulerSolver::new(num_steps, t_shift);
        let times = solver.time_schedule();

        // Initialize with random noise [1, target_frames, feat_dim]
        let mut x = Tensor::randn(0f32, 1.0, (1, target_frames, m.feat_dim), &device)?
            .to_dtype(dtype)?;

        for step in 0..num_steps {
            let t_cur = times[step];
            let t_next = times[step + 1];
            let is_last = step == num_steps - 1;

            // Time embedding: sinusoidal -> MLP
            let time_emb = sinusoidal_time_embedding(t_cur, m.time_embed_dim, &device, dtype)?;
            // [1, time_embed_dim]
            let time_emb = self.ctx.backend.linear_forward(&time_emb, &self.time_embed_0_weight, self.time_embed_0_bias.as_ref())?; // [1, time_embed_dim*2]
            let time_emb = super::activations::swoosh_r(&time_emb)?;
            let time_emb = self.ctx.backend.linear_forward(&time_emb, &self.time_embed_2_weight, self.time_embed_2_bias.as_ref())?; // [1, time_embed_dim]

            // Concat [noise, text_cond, speech_cond] along feature dim -> [1, target_frames, feat_dim*3]
            let input = Tensor::cat(&[&x, &text_cond_expanded, &speech_cond], 2)?;

            // fm_decoder.in_proj -> [1, target_frames, fm_dim]
            let mut hidden = self.ctx.backend.linear_forward(&input, &self.fm_in_proj_weight, self.fm_in_proj_bias.as_ref())?;

            // Forward through FM decoder blocks, stack by stack
            let mut flat_idx = 0;
            let num_stacks = self.config.model.fm_decoder_num_layers.len();
            for stack_idx in 0..num_stacks {
                let num_layers = self.config.model.fm_decoder_num_layers[stack_idx];
                let ds = self.config.model.fm_decoder_downsampling_factor[stack_idx];

                // Save original for bypass combine
                let src_orig = if ds > 1 {
                    Some(hidden.clone())
                } else {
                    None
                };

                // Downsample if needed
                if ds > 1 {
                    let bias = self.downsample_biases[stack_idx]
                        .as_ref()
                        .expect("downsample bias missing for ds>1 stack");
                    hidden = simple_downsample(&hidden, ds, bias)?;
                }

                let downsampled_frames = hidden.dim(1)?;

                // Per-stack time embedding: SwooshR -> Linear(192, 512)
                let te_swooshed = super::activations::swoosh_r(&time_emb)?;
                let stack_te = self.ctx.backend.linear_forward(&te_swooshed, &self.stack_time_emb_weights[stack_idx], self.stack_time_emb_biases[stack_idx].as_ref())?; // [1, fm_dim]
                let stack_te = stack_te.unsqueeze(1)?; // [1, 1, fm_dim]

                // Process layers in this stack
                for layer_in_stack in 0..num_layers {
                    let block = &self.fm_blocks[flat_idx];

                    // Pack time_emb as first frame for the Forwarder interface
                    let packed = Tensor::cat(&[&stack_te, &hidden], 1)?; // [1, 1+frames, fm_dim]
                    let out = block
                        .forward(&packed, step, flat_idx, &mut self.ctx)
                        .await?;
                    // Unpack: skip first frame (time_emb)
                    hidden = out.narrow(1, 1, downsampled_frames)?;

                    let _ = layer_in_stack; // suppress unused warning
                    flat_idx += 1;
                }

                // Upsample and bypass combine if needed
                if ds > 1 {
                    hidden = simple_upsample(&hidden, ds)?;
                    // Trim to original length (upsample may overshoot due to padding in downsample)
                    let orig_len = src_orig.as_ref().unwrap().dim(1)?;
                    if hidden.dim(1)? > orig_len {
                        hidden = hidden.narrow(1, 0, orig_len)?;
                    }
                    hidden = self.out_combiners[stack_idx]
                        .as_ref()
                        .expect("out_combiner missing for ds>1 stack")
                        .forward(src_orig.as_ref().unwrap(), &hidden)?;
                }
            }

            // fm_decoder.out_proj -> [1, target_frames, feat_dim] (velocity prediction)
            let v = self.ctx.backend.linear_forward(&hidden, &self.fm_out_proj_weight, self.fm_out_proj_bias.as_ref())?;

            // Euler step
            x = EulerSolver::step(&x, &v, t_cur, t_next, is_last)?;
        }

        // 5. Vocos vocoder: feature -> waveform
        log::info!("[LuxTTS] Running Vocos vocoder...");
        // Scale features for vocoder: divide by feat_scale (0.1)
        // Python: pred_features = pred_features.permute(0, 2, 1) / feat_scale
        let feat_scale = 0.1f64;
        let mel_out = x.transpose(1, 2)?.affine(1.0 / feat_scale, 0.0)?;
        let samples_24k = self.vocos.forward(&mel_out)?;

        // 6. Upsample 24kHz -> 48kHz
        let samples_48k = vocos::upsample(&samples_24k, 24000, 48000);

        log::info!(
            "[LuxTTS] Generated {:.1}s audio ({} samples @ 48kHz)",
            samples_48k.len() as f64 / 48000.0,
            samples_48k.len()
        );

        self.generated = true;
        Ok(samples_48k)
    }
}
