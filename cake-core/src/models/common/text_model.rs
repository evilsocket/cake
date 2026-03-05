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
    let tokenizer_filename = ctx.data_path.join("tokenizer.json");

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
        for i in 0..config.num_hidden_layers {
            let block_layer_name = format!("{prefix}.layers.{i}");
            if ctx.topology.get_node_for_layer(&block_layer_name).is_none() {
                log::info!("loading {} ...", &block_layer_name);
                blocks[i] = Some(B::load(block_layer_name, ctx)?);
            }
        }

        // Pass 2: connect to remote layers
        for i in 0..config.num_hidden_layers {
            let block_layer_name = format!("{prefix}.layers.{i}");
            if let Some((_node_name, node)) = ctx.topology.get_node_for_layer(&block_layer_name) {
                log::info!("connecting {} to {} ...", &block_layer_name, &node.host);
                blocks[i] = Some(Box::new(
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
            .map_or(false, |eos| eos.is_eos(next_token));

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
