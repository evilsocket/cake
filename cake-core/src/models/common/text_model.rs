use std::collections::HashSet;

use anyhow::Result;
#[cfg(feature = "cuda")]
use candle_core::Device;
use candle_core::{IndexOp, Tensor};
use candle_nn::{linear_no_bias as linear, Embedding, Linear, Module, RmsNorm};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

use super::EosTokenId;
use crate::{
    cake::{Context, Forwarder},
    models::Token,
};

/// Load the tokenizer and resolve EOS token ID(s).
/// `default_eos_token` is the model-specific fallback (e.g. "<|eot_id|>" for LLaMA,
/// "<|endoftext|>" for Qwen2).
pub fn load_tokenizer(
    ctx: &Context,
    default_eos_token: &str,
) -> Result<(Tokenizer, Option<EosTokenId>)> {
    // For GGUF files, look for tokenizer.json in the same directory as the .gguf file
    let tokenizer_filename = if ctx.data_path.is_file() {
        ctx.data_path
            .parent()
            .unwrap_or(&ctx.data_path)
            .join("tokenizer.json")
    } else {
        ctx.data_path.join("tokenizer.json")
    };

    log::info!("loading tokenizer from {}", tokenizer_filename.display());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;

    let config = ctx.config.as_ref().expect("No config specified");

    let eos_token_id = if config.eos_token_id.is_some() {
        config.eos_token_id.clone()
    } else {
        // Fallback: try to resolve from tokenizer vocabulary
        tokenizer
            .token_to_id(default_eos_token)
            .map(EosTokenId::Single)
    };

    Ok((tokenizer, eos_token_id))
}

/// Apply repeat penalty entirely on GPU to avoid costly GPU↔CPU round-trips.
///
/// The upstream `candle_transformers::utils::apply_repeat_penalty` copies the entire
/// logits tensor (vocab_size × 4 bytes ≈ 600 KB) to CPU, modifies a handful of elements,
/// then copies everything back.  This forces a full GPU synchronisation and two large PCIe
/// transfers per token.
///
/// This implementation stays on-device: it selects only the penalty positions, computes
/// sign-aware multipliers, and scatters the deltas back with `index_add`.
fn apply_repeat_penalty_gpu(
    logits: &Tensor,
    penalty: f32,
    context: &[u32],
) -> Result<Tensor> {
    // Deduplicate tokens (same semantics as the upstream version).
    let mut seen = HashSet::new();
    let unique: Vec<u32> = context
        .iter()
        .filter(|t| seen.insert(**t))
        .copied()
        .collect();

    if unique.is_empty() {
        return Ok(logits.clone());
    }

    let device = logits.device();
    let dtype = logits.dtype();
    let indices = Tensor::new(unique.as_slice(), device)?;

    // Gather logits at penalty positions  (N elements, tiny).
    let selected = logits.index_select(&indices, 0)?;

    // Sign-aware multiplier:  1/penalty for logits ≥ 0,  penalty for logits < 0.
    let is_non_negative = selected.ge(0f32)?;
    let recip = Tensor::new(1.0f32 / penalty, device)?
        .to_dtype(dtype)?
        .broadcast_as(selected.shape())?;
    let pen = Tensor::new(penalty, device)?
        .to_dtype(dtype)?
        .broadcast_as(selected.shape())?;
    let mult = is_non_negative.where_cond(&recip, &pen)?;

    // delta = selected * mult - selected  (what to add to the original logits).
    let penalized = (&selected * &mult)?;
    let delta = (&penalized - &selected)?;

    Ok(logits.index_add(&indices, &delta, 0)?)
}

/// Create the logit sampling logic from the context.
pub fn create_logits_processor(ctx: &Context) -> LogitsProcessor {
    let temperature = ctx.args.temperature;
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (ctx.args.top_k, ctx.args.top_p) {
            // Gumbel-Softmax keeps everything on GPU: generates random noise,
            // adds to logits/temperature, and takes argmax — only 4 bytes
            // transferred instead of the full 600 KB vocabulary vector.
            (None, None) => Sampling::GumbelSoftmax { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(ctx.args.seed, sampling)
}

/// Shared base for decoder-only text models (LLaMA, Qwen2, Qwen3.5, etc.).
///
/// Contains all the state and logic that is identical across model architectures:
/// embedding, transformer blocks, final norm, lm_head, tokenizer, sampling, and
/// the forward/generation loop.
pub struct TextModelBase {
    pub ctx: Context,

    pub tokenizer: Tokenizer,
    pub embedding: Embedding,
    pub eos_token_id: Option<EosTokenId>,
    pub index_pos: usize,
    pub generated: usize,
    pub prompt_len: usize,

    pub blocks: Vec<Box<dyn Forwarder>>,

    pub ln_f: RmsNorm,
    pub lm_head: Linear,

    pub logits_processor: LogitsProcessor,

    pub tokens: Vec<u32>,
}

impl TextModelBase {
    /// Load the shared model structure from the context.
    /// `default_eos_token` is the model-specific fallback EOS string.
    /// The type parameter `B` determines which block type to use for local layers.
    pub async fn load<B: Forwarder + 'static>(
        ctx: &mut Context,
        default_eos_token: &str,
    ) -> Result<Self> {
        let config = ctx.config.as_ref().expect("No config specified");
        let var_builder = ctx.var_builder.as_ref().expect("No var_builder specified");
        let prefix = &config.model_prefix;

        log::info!("loading embeddings (prefix={}) ...", prefix);
        let embedding: Embedding = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            var_builder.pp(format!("{prefix}.embed_tokens")),
        )?;

        log::info!("loading lm_head ...");
        let lm_head = if config.tie_word_embeddings {
            log::info!("  using tied word embeddings (lm_head = embed_tokens)");
            Linear::new(embedding.embeddings().clone(), None)
        } else {
            // Try root-level lm_head first (LLaMA/Qwen2), then prefixed (Qwen3.5)
            match linear(
                config.hidden_size,
                config.vocab_size,
                var_builder.pp("lm_head"),
            ) {
                Ok(l) => l,
                Err(_) => linear(
                    config.hidden_size,
                    config.vocab_size,
                    var_builder.pp(format!("{prefix}.lm_head")),
                )?,
            }
        };

        log::info!("loading {prefix}.norm ...");
        let ln_f = crate::models::common::load_rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            config.residual_rms_norm,
            var_builder.pp(format!("{prefix}.norm")),
        )?;

        log::info!("loading {} blocks ...", config.num_hidden_layers);

        // Two-pass loading: local layers first (no network wait), then remote
        // layers (may block until workers finish loading). This overlaps
        // master's local layer loading with worker startup time.
        let mut blocks: Vec<Option<Box<dyn Forwarder>>> =
            (0..config.num_hidden_layers).map(|_| None).collect();

        // Pass 1: load local layers
        for (i, block) in blocks.iter_mut().enumerate().take(config.num_hidden_layers) {
            let block_layer_name = format!("{prefix}.layers.{i}");
            if ctx.topology.get_node_for_layer(&block_layer_name).is_none() {
                log::info!("loading {} ...", &block_layer_name);
                *block = Some(B::load(block_layer_name, ctx)?);
            }
        }

        // Pass 2: connect to remote layers
        for (i, block) in blocks.iter_mut().enumerate().take(config.num_hidden_layers) {
            let block_layer_name = format!("{prefix}.layers.{i}");
            if let Some((_node_name, node)) = ctx.topology.get_node_for_layer(&block_layer_name) {
                log::info!("connecting {} to {} ...", &block_layer_name, &node.host);
                *block = Some(Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        &block_layer_name,
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                ));
            }
        }

        let blocks: Vec<Box<dyn Forwarder>> = blocks.into_iter().map(|b| b.unwrap()).collect();

        for block in &blocks {
            log::info!("  {}", block)
        }

        let (tokenizer, eos_token_id) = load_tokenizer(ctx, default_eos_token)?;
        let tokens = vec![];

        let logits_processor = create_logits_processor(ctx);
        let index_pos = 0;

        log::info!(
            "model loaded - mem={}",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let generated = 0;

        Ok(Self {
            tokenizer,
            tokens,
            generated,
            eos_token_id,
            index_pos,
            prompt_len: 0,
            ctx: ctx.clone(),
            embedding,
            blocks,
            ln_f,
            lm_head,
            logits_processor,
        })
    }

    /// Forward pass through all blocks.
    pub async fn forward(&mut self, x: &Tensor, idx: usize) -> Result<Tensor> {
        let forward_start = std::time::Instant::now();
        let (_batch_size, seq_len) = x.dims2()?;

        let emb_start = std::time::Instant::now();
        let mut x = self.embedding.forward(x)?;
        // Apply embedding scale if configured (Gemma scales by sqrt(hidden_size)).
        if let Some(scale) = self.ctx.config.as_ref().and_then(|c| c.embed_scale) {
            x = (x * scale as f64)?;
        }
        let emb_elapsed = emb_start.elapsed();

        let num_blocks = self.blocks.len();
        let mut block_idx = 0;
        let mut local_elapsed = std::time::Duration::ZERO;
        let mut local_count: usize = 0;

        while block_idx < num_blocks {
            let curr_block_id = self.blocks[block_idx].ident().to_owned();
            if curr_block_id == "local" {
                let local_start = std::time::Instant::now();
                x = self.blocks[block_idx]
                    .forward_mut(&x, idx, block_idx, &mut self.ctx)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward operation of local block {block_idx}: {e}")
                    })?;
                local_elapsed += local_start.elapsed();
                local_count += 1;

                block_idx += 1;
            } else {
                // collect all contiguous layers running on the same worker
                let mut batch = vec![];
                let first = block_idx;
                while block_idx < num_blocks && self.blocks[block_idx].ident() == curr_block_id {
                    batch.push((
                        self.blocks[block_idx].layer_name().to_string(),
                        idx,
                        block_idx,
                    ));
                    block_idx += 1;
                }

                let num_layers = batch.len();
                let batch_start = std::time::Instant::now();
                x = self.blocks[first]
                    .forward_batch(&x, batch, &mut self.ctx)
                    .await
                    .map_err(|e| {
                        anyhow!(
                            "error in forward batch for blocks {first}..{block_idx} on {}: {e}",
                            &curr_block_id
                        )
                    })?;
                let batch_elapsed = batch_start.elapsed();
                log::debug!(
                    "  worker {} layers {}-{} ({} layers): {:.1}ms",
                    &curr_block_id,
                    first,
                    block_idx - 1,
                    num_layers,
                    batch_elapsed.as_secs_f64() * 1000.0
                );
            }
        }

        let head_start = std::time::Instant::now();
        let x = self
            .ln_f
            .forward(&x)
            .map_err(|e| anyhow!("error in ln_f.forward: {e}"))?;

        let x = x
            .i((.., seq_len - 1, ..))
            .map_err(|e| anyhow!("error in x.i: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("error in x.i.contiguous: {e}"))?;

        let logits = self
            .lm_head
            .forward(&x)
            .map_err(|e| anyhow!("error in lm_head.forward: {e}"))?;
        // Note: no explicit sync needed here — the CPU-side logits sampling
        // (to_vec1 in LogitsProcessor) implicitly synchronizes the Metal command buffer.
        let head_elapsed = head_start.elapsed();

        let total_elapsed = forward_start.elapsed();
        log::debug!(
            "  forward total={:.1}ms emb={:.1}ms local={:.1}ms ({} blocks) head={:.1}ms",
            total_elapsed.as_secs_f64() * 1000.0,
            emb_elapsed.as_secs_f64() * 1000.0,
            local_elapsed.as_secs_f64() * 1000.0,
            local_count,
            head_elapsed.as_secs_f64() * 1000.0,
        );

        Ok(logits)
    }

    /// Tokenize a prompt string and set up token state for generation.
    pub fn prepare_prompt(&mut self, dialog: &str) -> Result<()> {
        // make sure we start clean
        self.tokens.clear();
        self.ctx.cache.as_mut().expect("No cache specified").clear();
        self.index_pos = 0;

        log::debug!("dialog={}", dialog);

        // tokenize raw
        self.tokens = self
            .tokenizer
            .encode(dialog, false) // do not add special tokens as we already added them
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        log::debug!("encoded={:?}", &self.tokens);
        log::debug!("history tokens: {}", self.tokens.len());

        // Track prompt length for repeat penalty scoping
        self.prompt_len = self.tokens.len();

        Ok(())
    }

    /// Generate the next token. Assumes `prepare_prompt()` has been called for the first token.
    pub async fn next_token(&mut self, index: usize) -> Result<Token> {
        log::trace!("model.next_token({index})");

        let num_tokens = self.tokens.len();
        let (context_size, context_index) = if self
            .ctx
            .cache
            .as_ref()
            .expect("No cache specified")
            .with_kv_cache()
            && index > 0
        {
            (1, self.index_pos)
        } else {
            (num_tokens, 0)
        };

        let context_offset = num_tokens.saturating_sub(context_size);
        let context_tokens = &self.tokens[context_offset..];
        let num_context_tokens = context_tokens.len();

        let input = Tensor::new(context_tokens, &self.ctx.device)?
            .unsqueeze(0)
            .map_err(|e| anyhow!("error squeezing context tokens: {e}"))?;

        let logits = self
            .forward(&input, context_index)
            .await
            .map_err(|e| anyhow!("error in model.forward: {e}"))?;

        let post_start = std::time::Instant::now();

        let logits = logits
            .squeeze(0)
            .map_err(|e| anyhow!("error squeezing logits: {e}"))?;

        // Apply repeat penalty only to generated tokens (not prompt tokens)
        let penalty_start = std::time::Instant::now();
        let logits = if self.ctx.args.repeat_penalty == 1. {
            logits
        } else {
            let generated_start = self.prompt_len;
            let penalty_tokens = &self.tokens[generated_start..];
            if penalty_tokens.is_empty() {
                logits
            } else {
                let start_at = penalty_tokens
                    .len()
                    .saturating_sub(self.ctx.args.repeat_last_n);
                apply_repeat_penalty_gpu(
                    &logits,
                    self.ctx.args.repeat_penalty,
                    &penalty_tokens[start_at..],
                )?
            }
        };
        let penalty_elapsed = penalty_start.elapsed();
        self.index_pos += num_context_tokens;

        let sample_start = std::time::Instant::now();
        let next_token = self
            .logits_processor
            .sample(&logits)
            .map_err(|e| anyhow!("error sampling logits {logits}: {e}"))?;
        let sample_elapsed = sample_start.elapsed();

        self.generated += 1;
        self.tokens.push(next_token);

        let is_end_of_stream = self
            .eos_token_id
            .as_ref()
            .is_some_and(|eos| eos.is_eos(next_token));

        let decode_start = std::time::Instant::now();
        let text = match self.tokenizer.decode(&[next_token], false) {
            Ok(s) => Some(s),
            Err(e) => {
                log::error!("could not decode token {next_token}: {e}");
                None
            }
        };
        let decode_elapsed = decode_start.elapsed();
        let post_elapsed = post_start.elapsed();

        log::debug!(
            "  post-forward: total={:.1}ms penalty={:.1}ms sample={:.1}ms decode={:.1}ms",
            post_elapsed.as_secs_f64() * 1000.0,
            penalty_elapsed.as_secs_f64() * 1000.0,
            sample_elapsed.as_secs_f64() * 1000.0,
            decode_elapsed.as_secs_f64() * 1000.0,
        );

        Ok(Token {
            id: next_token,
            text,
            is_end_of_stream,
        })
    }

    /// Reset all generation state.
    pub fn reset(&mut self) {
        self.tokens.clear();
        self.ctx.cache.as_mut().expect("No cache specified").clear();
        self.index_pos = 0;
        self.generated = 0;
        self.prompt_len = 0;

        // Clear any stale CUDA error state left by tensor cleanup (CudaSlice drops).
        // cudarc's error_state is an atomic that gets poisoned by internal operations
        // (e.g. SyncOnDrop event recording, async memory frees) and causes the NEXT
        // inference request to fail via check_err(). Clearing it here prevents the
        // alternating success/failure pattern.
        #[cfg(feature = "cuda")]
        if let Device::Cuda(cuda_dev) = &self.ctx.device {
            let _ = cuda_dev.cuda_stream().context().bind_to_thread();
        }
    }

    /// Notify all remote blocks of session end (clears their KV caches).
    pub async fn goodbye(&mut self) -> Result<()> {
        let num_blocks = self.blocks.len();
        let mut block_idx = 0;
        while block_idx < num_blocks {
            self.blocks[block_idx]
                .goodbye()
                .await
                .map_err(|e| anyhow!("error in goodbye operation for block {block_idx}: {e}"))?;
            block_idx += 1;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Helper: create a 1-D F32 tensor from a slice.
    fn f32_tensor(data: &[f32]) -> Tensor {
        Tensor::from_slice(data, (data.len(),), &Device::Cpu).unwrap()
    }

    // ── apply_repeat_penalty_gpu ─────────────────────────────────

    #[test]
    fn test_repeat_penalty_positive_logits_divided() {
        // Positive logits should be divided by penalty (multiplied by 1/penalty).
        let logits = f32_tensor(&[0.0, 10.0, 20.0, 5.0]);
        let penalty = 2.0;
        let context = vec![1u32, 2]; // penalize indices 1 and 2

        let result = apply_repeat_penalty_gpu(&logits, penalty, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-6, "index 0 unchanged");
        assert!((vals[1] - 5.0).abs() < 1e-6, "10.0 / 2.0 = 5.0, got {}", vals[1]);
        assert!((vals[2] - 10.0).abs() < 1e-6, "20.0 / 2.0 = 10.0, got {}", vals[2]);
        assert!((vals[3] - 5.0).abs() < 1e-6, "index 3 unchanged");
    }

    #[test]
    fn test_repeat_penalty_negative_logits_multiplied() {
        // Negative logits should be multiplied by penalty (made more negative).
        let logits = f32_tensor(&[0.0, -10.0, -4.0, 5.0]);
        let penalty = 2.0;
        let context = vec![1u32, 2]; // penalize indices 1 and 2

        let result = apply_repeat_penalty_gpu(&logits, penalty, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 0.0).abs() < 1e-6, "index 0 unchanged");
        assert!((vals[1] - (-20.0)).abs() < 1e-6, "-10.0 * 2.0 = -20.0, got {}", vals[1]);
        assert!((vals[2] - (-8.0)).abs() < 1e-6, "-4.0 * 2.0 = -8.0, got {}", vals[2]);
        assert!((vals[3] - 5.0).abs() < 1e-6, "index 3 unchanged");
    }

    #[test]
    fn test_repeat_penalty_one_returns_unchanged() {
        // penalty = 1.0: logits >=0 divided by 1, logits <0 multiplied by 1 — no change.
        let logits = f32_tensor(&[-5.0, 0.0, 3.0, 10.0]);
        let context = vec![0u32, 1, 2, 3];

        let result = apply_repeat_penalty_gpu(&logits, 1.0, &context).unwrap();
        let orig: Vec<f32> = logits.to_vec1().unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        for (i, (o, v)) in orig.iter().zip(vals.iter()).enumerate() {
            assert!((o - v).abs() < 1e-6, "index {} changed: {} -> {}", i, o, v);
        }
    }

    #[test]
    fn test_repeat_penalty_empty_context_returns_unchanged() {
        let logits = f32_tensor(&[1.0, -2.0, 3.0]);
        let context: Vec<u32> = vec![];

        let result = apply_repeat_penalty_gpu(&logits, 2.0, &context).unwrap();
        let orig: Vec<f32> = logits.to_vec1().unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert_eq!(orig, vals);
    }

    #[test]
    fn test_repeat_penalty_deduplicates_context() {
        // Duplicate tokens in context should be deduplicated (applied once, not twice).
        let logits = f32_tensor(&[0.0, 10.0, 0.0]);
        let context = vec![1u32, 1, 1]; // index 1 repeated

        let result = apply_repeat_penalty_gpu(&logits, 2.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        // 10.0 / 2.0 = 5.0 (applied once, not three times)
        assert!((vals[1] - 5.0).abs() < 1e-6, "got {}", vals[1]);
    }

    #[test]
    fn test_repeat_penalty_zero_logit() {
        // Zero logits: ge(0) is true, so divided by penalty. 0/penalty = 0.
        let logits = f32_tensor(&[0.0, 0.0]);
        let context = vec![0u32, 1];

        let result = apply_repeat_penalty_gpu(&logits, 3.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0]).abs() < 1e-6);
        assert!((vals[1]).abs() < 1e-6);
    }

    #[test]
    fn test_repeat_penalty_mixed_signs() {
        // Mix of positive and negative at penalized positions
        let logits = f32_tensor(&[6.0, -3.0, 0.0, 9.0, -1.0]);
        let penalty = 3.0;
        let context = vec![0u32, 1, 2, 3, 4];

        let result = apply_repeat_penalty_gpu(&logits, penalty, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();

        assert!((vals[0] - 2.0).abs() < 1e-6, "6/3=2, got {}", vals[0]);
        assert!((vals[1] - (-9.0)).abs() < 1e-6, "-3*3=-9, got {}", vals[1]);
        assert!((vals[2] - 0.0).abs() < 1e-6, "0/3=0, got {}", vals[2]);
        assert!((vals[3] - 3.0).abs() < 1e-6, "9/3=3, got {}", vals[3]);
        assert!((vals[4] - (-3.0)).abs() < 1e-6, "-1*3=-3, got {}", vals[4]);
    }

    // ── apply_repeat_penalty_gpu: additional edge cases ─────────

    #[test]
    fn test_repeat_penalty_single_element_logits() {
        let logits = f32_tensor(&[7.0]);
        let context = vec![0u32];
        let result = apply_repeat_penalty_gpu(&logits, 7.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6, "7/7=1, got {}", vals[0]);
    }

    #[test]
    fn test_repeat_penalty_large_vocab_sparse_context() {
        // Large-ish vocab, only a few tokens penalized
        let mut data = vec![1.0f32; 1000];
        data[500] = 10.0;
        data[999] = -4.0;
        let logits = f32_tensor(&data);
        let context = vec![500u32, 999];
        let result = apply_repeat_penalty_gpu(&logits, 2.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        // Unpenalized positions unchanged
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[499] - 1.0).abs() < 1e-6);
        // Penalized positions
        assert!((vals[500] - 5.0).abs() < 1e-6, "10/2=5, got {}", vals[500]);
        assert!((vals[999] - (-8.0)).abs() < 1e-6, "-4*2=-8, got {}", vals[999]);
    }

    #[test]
    fn test_repeat_penalty_fractional_penalty() {
        // Penalty < 1.0: positive logits get multiplied (boosted), negative get divided
        let logits = f32_tensor(&[4.0, -4.0]);
        let context = vec![0u32, 1];
        let result = apply_repeat_penalty_gpu(&logits, 0.5, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        // 4.0 >= 0 -> 4.0 / 0.5 = 8.0 (boosted by reciprocal)
        assert!((vals[0] - 8.0).abs() < 1e-6, "4/0.5=8, got {}", vals[0]);
        // -4.0 < 0 -> -4.0 * 0.5 = -2.0 (reduced magnitude)
        assert!((vals[1] - (-2.0)).abs() < 1e-6, "-4*0.5=-2, got {}", vals[1]);
    }

    // ── create_logits_processor: sampling strategy selection ─────

    /// Helper to build a minimal Context for testing create_logits_processor.
    fn make_test_context(temperature: f64, top_k: Option<usize>, top_p: Option<f64>, seed: u64) -> crate::cake::Context {
        use crate::cake::{Context, Topology};
        use crate::Args;
        use candle_core::DType;
        use std::path::PathBuf;
        use std::sync::{Arc, Mutex};

        let args = Args {
            temperature,
            top_k,
            top_p,
            seed,
            ..Args::default()
        };

        Context {
            args,
            dtype: DType::F32,
            topology: Topology::new(),
            data_path: PathBuf::from("/tmp"),
            device: Device::Cpu,
            config: None,
            cache: None,
            var_builder: None,
            text_model_arch: crate::TextModelArch::Llama,
            quant: Arc::new(crate::utils::NoQuantization),
            listener_override: Arc::new(Mutex::new(None)),
            tensor_storage: None,
            backend: Arc::new(crate::backends::CpuBackend::new()),
        }
    }

    #[test]
    fn test_create_logits_processor_argmax_at_zero_temp() {
        // temperature <= 0 should select ArgMax (greedy decoding)
        let ctx = make_test_context(0.0, None, None, 42);
        let mut lp = create_logits_processor(&ctx);
        // ArgMax should always pick the highest logit
        let logits = f32_tensor(&[-10.0, 5.0, 3.0, 1.0]);
        let token = lp.sample(&logits).unwrap();
        assert_eq!(token, 1, "argmax should pick index 1 (value 5.0)");
    }

    #[test]
    fn test_create_logits_processor_argmax_negative_temp() {
        let ctx = make_test_context(-1.0, Some(10), Some(0.9), 42);
        let mut lp = create_logits_processor(&ctx);
        // Even with top_k/top_p set, negative temp should force ArgMax
        let logits = f32_tensor(&[0.0, 0.0, 100.0, 0.0]);
        let token = lp.sample(&logits).unwrap();
        assert_eq!(token, 2, "argmax should pick index 2 (value 100.0)");
    }

    #[test]
    fn test_create_logits_processor_gumbel_softmax_default() {
        // temperature > 0 with no top_k/top_p should use GumbelSoftmax
        let ctx = make_test_context(0.7, None, None, 42);
        let lp = create_logits_processor(&ctx);
        // We can't inspect the sampling variant directly, but we can verify it
        // produces valid output (i.e., doesn't panic).
        let logits = f32_tensor(&[1.0, 2.0, 3.0]);
        let mut lp = lp;
        let token = lp.sample(&logits).unwrap();
        assert!(token < 3, "token should be a valid index");
    }

    #[test]
    fn test_create_logits_processor_top_k_only() {
        let ctx = make_test_context(0.5, Some(2), None, 42);
        let mut lp = create_logits_processor(&ctx);
        // With top_k=2, only the top 2 logits should be candidates
        // Make one logit overwhelmingly large to deterministically test
        let logits = f32_tensor(&[0.0, 0.0, 1000.0, 0.0]);
        let token = lp.sample(&logits).unwrap();
        // The token with value 1000.0 should almost always be picked
        assert_eq!(token, 2);
    }

    #[test]
    fn test_create_logits_processor_top_p_only() {
        let ctx = make_test_context(0.5, None, Some(0.1), 42);
        let mut lp = create_logits_processor(&ctx);
        let logits = f32_tensor(&[0.0, 0.0, 1000.0, 0.0]);
        let token = lp.sample(&logits).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_create_logits_processor_top_k_then_top_p() {
        let ctx = make_test_context(0.5, Some(3), Some(0.5), 42);
        let mut lp = create_logits_processor(&ctx);
        let logits = f32_tensor(&[0.0, 0.0, 1000.0, 0.0]);
        let token = lp.sample(&logits).unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn test_create_logits_processor_deterministic_with_same_seed() {
        // Same seed with TopK should produce identical token sequences
        let logits = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let ctx1 = make_test_context(1.0, Some(3), None, 12345);
        let mut lp1 = create_logits_processor(&ctx1);
        let tokens1: Vec<u32> = (0..10).map(|_| lp1.sample(&logits).unwrap()).collect();

        let ctx2 = make_test_context(1.0, Some(3), None, 12345);
        let mut lp2 = create_logits_processor(&ctx2);
        let tokens2: Vec<u32> = (0..10).map(|_| lp2.sample(&logits).unwrap()).collect();

        assert_eq!(tokens1, tokens2, "same seed should produce identical sequences");
    }

    #[test]
    fn test_create_logits_processor_different_seeds_differ() {
        // Different seeds should (very likely) produce different sequences with TopK sampling
        let logits = f32_tensor(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

        let ctx1 = make_test_context(1.0, Some(10), None, 1);
        let mut lp1 = create_logits_processor(&ctx1);
        let tokens1: Vec<u32> = (0..20).map(|_| lp1.sample(&logits).unwrap()).collect();

        let ctx2 = make_test_context(1.0, Some(10), None, 999);
        let mut lp2 = create_logits_processor(&ctx2);
        let tokens2: Vec<u32> = (0..20).map(|_| lp2.sample(&logits).unwrap()).collect();

        assert_ne!(tokens1, tokens2, "different seeds should produce different sequences");
    }

    // ── create_logits_processor: high temperature spreads distribution ──

    #[test]
    fn test_create_logits_processor_high_temperature_spreads() {
        // With very high temperature, even small logit differences should produce varied tokens.
        // Use top_k to force TopK sampling (not GumbelSoftmax) for predictability.
        let logits = f32_tensor(&[10.0, 9.5, 9.0, 8.5, 8.0]);
        let ctx = make_test_context(100.0, Some(5), None, 42);
        let mut lp = create_logits_processor(&ctx);
        let tokens: Vec<u32> = (0..50).map(|_| lp.sample(&logits).unwrap()).collect();
        let unique: std::collections::HashSet<u32> = tokens.into_iter().collect();
        // High temperature should cause at least 3 distinct tokens to be sampled
        assert!(unique.len() >= 3, "high temp should produce variety, got {:?}", unique);
    }

    #[test]
    fn test_create_logits_processor_low_temperature_concentrates() {
        // Low (but positive) temperature should heavily favor the top logit.
        let logits = f32_tensor(&[10.0, 5.0, 0.0, -5.0, -10.0]);
        let ctx = make_test_context(0.01, Some(5), None, 42);
        let mut lp = create_logits_processor(&ctx);
        let tokens: Vec<u32> = (0..20).map(|_| lp.sample(&logits).unwrap()).collect();
        // With temp=0.01, nearly all tokens should be index 0 (logit 10.0)
        let count_zero = tokens.iter().filter(|&&t| t == 0).count();
        assert!(count_zero >= 18, "low temp should concentrate on max logit, got {} out of 20", count_zero);
    }

    // ── apply_repeat_penalty_gpu: high penalty ──

    #[test]
    fn test_repeat_penalty_very_high_penalty() {
        // Very high penalty should crush positive logits near zero
        let logits = f32_tensor(&[100.0, -0.5]);
        let context = vec![0u32, 1];
        let result = apply_repeat_penalty_gpu(&logits, 100.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-4, "100/100=1, got {}", vals[0]);
        assert!((vals[1] - (-50.0)).abs() < 1e-4, "-0.5*100=-50, got {}", vals[1]);
    }

    // ── apply_repeat_penalty_gpu: partial context ──

    #[test]
    fn test_repeat_penalty_partial_overlap() {
        // Only some positions are penalized; others should be untouched.
        let logits = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let context = vec![1u32, 3]; // only indices 1 and 3
        let result = apply_repeat_penalty_gpu(&logits, 2.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6, "unchanged");
        assert!((vals[1] - 1.0).abs() < 1e-6, "2/2=1, got {}", vals[1]);
        assert!((vals[2] - 3.0).abs() < 1e-6, "unchanged");
        assert!((vals[3] - 2.0).abs() < 1e-6, "4/2=2, got {}", vals[3]);
        assert!((vals[4] - 5.0).abs() < 1e-6, "unchanged");
    }

    // ── create_logits_processor: argmax is deterministic across calls ──

    #[test]
    fn test_argmax_is_fully_deterministic() {
        let logits = f32_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]);
        let ctx = make_test_context(0.0, None, None, 0);
        let mut lp = create_logits_processor(&ctx);
        for _ in 0..10 {
            let token = lp.sample(&logits).unwrap();
            assert_eq!(token, 5, "argmax should always pick index 5 (value 9.0)");
        }
    }

    // ── create_logits_processor: top_p with extreme values ──

    #[test]
    fn test_top_p_very_small_picks_max() {
        // top_p near 0 should only allow the single highest-probability token
        let logits = f32_tensor(&[0.0, 0.0, 1000.0, 0.0, 0.0]);
        let ctx = make_test_context(0.5, None, Some(0.01), 42);
        let mut lp = create_logits_processor(&ctx);
        let token = lp.sample(&logits).unwrap();
        assert_eq!(token, 2, "tiny top_p should pick the dominant logit");
    }

    // ── create_logits_processor: single-element vocab ──

    #[test]
    fn test_logits_processor_single_token_vocab() {
        // A vocab of size 1 should always return token 0
        let logits = f32_tensor(&[42.0]);
        let ctx = make_test_context(1.0, Some(1), None, 42);
        let mut lp = create_logits_processor(&ctx);
        let token = lp.sample(&logits).unwrap();
        assert_eq!(token, 0);
    }

    // ── create_logits_processor: two-element vocab argmax ──

    #[test]
    fn test_logits_processor_two_tokens_argmax() {
        let logits = f32_tensor(&[0.1, 99.9]);
        let ctx = make_test_context(0.0, None, None, 0);
        let mut lp = create_logits_processor(&ctx);
        assert_eq!(lp.sample(&logits).unwrap(), 1);
    }

    // ── repeat penalty with index at boundary ──

    #[test]
    fn test_repeat_penalty_last_index() {
        // Penalize only the last index of the logits vector
        let logits = f32_tensor(&[1.0, 2.0, 3.0, 8.0]);
        let context = vec![3u32];
        let result = apply_repeat_penalty_gpu(&logits, 4.0, &context).unwrap();
        let vals: Vec<f32> = result.to_vec1().unwrap();
        assert!((vals[0] - 1.0).abs() < 1e-6);
        assert!((vals[1] - 2.0).abs() < 1e-6);
        assert!((vals[2] - 3.0).abs() < 1e-6);
        assert!((vals[3] - 2.0).abs() < 1e-6, "8/4=2, got {}", vals[3]);
    }

    // ── repeat penalty preserves tensor shape ──

    #[test]
    fn test_repeat_penalty_preserves_shape() {
        let logits = f32_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let context = vec![0u32, 2, 4];
        let result = apply_repeat_penalty_gpu(&logits, 2.0, &context).unwrap();
        assert_eq!(result.shape().dims(), logits.shape().dims());
    }
}
