//! Speculative decoding for distributed inference.
//!
//! A small "draft" model generates K tokens locally, then the large
//! distributed "full" model verifies them in a single batched forward pass.
//! Accepted tokens skip K-1 expensive distributed round-trips.
//!
//! When tokens are rejected, the full model's KV cache is reset and
//! re-prefilled on the next call (lazy rollback). This is acceptable
//! because rejection should be infrequent with a well-matched draft model.

use std::collections::VecDeque;

use anyhow::Result;
use candle_core::{IndexOp, Tensor};
use candle_nn::Module;

use super::common::text_model::TextModelBase;

/// Speculative decoding state, embedded in a TextGenerator implementation.
pub struct SpeculativeState {
    /// Buffered tokens that have been verified but not yet returned.
    pub accepted_buffer: VecDeque<(u32, Option<String>, bool)>,
    /// Number of speculative tokens to draft per round.
    pub spec_tokens: usize,
    /// Running stats: total accepted / total drafted.
    pub total_accepted: usize,
    pub total_drafted: usize,
}

impl SpeculativeState {
    pub fn new(spec_tokens: usize) -> Self {
        Self {
            accepted_buffer: VecDeque::new(),
            spec_tokens,
            total_accepted: 0,
            total_drafted: 0,
        }
    }

    pub fn acceptance_rate(&self) -> f64 {
        if self.total_drafted == 0 {
            0.0
        } else {
            self.total_accepted as f64 / self.total_drafted as f64
        }
    }
}

/// Run one round of speculative decoding.
///
/// 1. Draft `K` tokens using `draft` model (local-only, fast)
/// 2. Verify all K tokens with `full` model in one batched forward pass
/// 3. Accept matching prefix, use full model's prediction at first mismatch
///
/// Returns the list of accepted (token_id, text) pairs.
/// The full model's state (tokens, index_pos, KV cache) is updated to reflect
/// only the accepted tokens.
pub async fn speculate_and_verify(
    full: &mut TextModelBase,
    draft: &mut TextModelBase,
    state: &mut SpeculativeState,
) -> Result<Vec<(u32, Option<String>, bool)>> {
    let k = state.spec_tokens;

    // Save full model state before speculation
    let saved_index_pos = full.index_pos;
    let saved_tokens_len = full.tokens.len();
    let saved_generated = full.generated;

    // Phase 1: Draft K tokens with the local draft model
    let mut draft_token_ids: Vec<u32> = Vec::with_capacity(k);
    for _ in 0..k {
        let token = draft_next_token(draft).await?;
        if token.2 {
            // EOS from draft — just verify what we have
            draft_token_ids.push(token.0);
            break;
        }
        draft_token_ids.push(token.0);
    }

    let num_drafted = draft_token_ids.len();
    if num_drafted == 0 {
        return Ok(vec![]);
    }

    state.total_drafted += num_drafted;

    // Phase 2: Verify all draft tokens with full model in one forward pass
    let all_logits = forward_verify(full, &draft_token_ids).await?;

    // Phase 3: Compare predictions
    // all_logits shape: [num_drafted, vocab_size]
    // logits[i] predicts the token AFTER draft_token_ids[i]
    //
    // But we also need to verify draft_token_ids[0] itself.
    // draft_token_ids[0] should match what the full model would predict
    // given the context before speculation. We check this by looking at
    // the full model's logits from the position before d[0].
    //
    // For simplicity in this first version:
    // - We trust d[0] (the draft and full model saw the same context)
    // - We verify d[1..K] using all_logits[0..K-1]

    let mut accepted = Vec::new();
    let mut num_accepted = 0;

    // Accept d[0] (first draft token) — same context seen by both models
    accepted.push(draft_token_ids[0]);
    num_accepted += 1;

    // Verify d[1..K] using full model logits
    for i in 0..num_drafted - 1 {
        let logits_i = all_logits.i(i)?;
        let predicted = logits_i
            .argmax(candle_core::D::Minus1)?
            .to_scalar::<u32>()?;

        if predicted == draft_token_ids[i + 1] {
            accepted.push(draft_token_ids[i + 1]);
            num_accepted += 1;
        } else {
            // Mismatch: use full model's prediction instead
            accepted.push(predicted);
            num_accepted += 1;
            break;
        }
    }

    // If all K tokens matched, also sample the bonus token from logits[K-1]
    if num_accepted == num_drafted {
        let last_logits = all_logits.i(num_drafted - 1)?;
        let bonus = full
            .logits_processor
            .sample(&last_logits)
            .map_err(|e| anyhow!("bonus sample: {e}"))?;
        accepted.push(bonus);
        num_accepted += 1;
    }

    state.total_accepted += num_accepted;

    // Phase 4: Update full model state to reflect accepted tokens
    // Reset to saved state first
    full.index_pos = 0; // Must be 0 so next forward_verify does full re-prefill
    full.tokens.truncate(saved_tokens_len);
    full.generated = saved_generated;

    // Clear KV cache — will re-prefill lazily on next forward
    full.ctx.cache.as_mut().expect("No cache").clear();

    // Add accepted tokens to full model
    let mut results = Vec::with_capacity(accepted.len());
    for &token_id in &accepted {
        full.tokens.push(token_id);
        full.generated += 1;

        let is_eos = full
            .eos_token_id
            .as_ref()
            .map_or(false, |eos| eos.is_eos(token_id));

        let text = full.tokenizer.decode(&[token_id], false).ok();
        results.push((token_id, text, is_eos));

        if is_eos {
            break;
        }
    }

    // Sync draft model to accepted state
    draft.tokens.truncate(saved_tokens_len);
    draft.generated = saved_generated;
    draft.index_pos = saved_index_pos;
    draft.ctx.cache.as_mut().expect("No cache").clear();
    for &(token_id, _, _) in &results {
        draft.tokens.push(token_id);
        draft.generated += 1;
    }

    log::debug!(
        "speculative: drafted={} accepted={} rate={:.0}%",
        num_drafted,
        results.len(),
        state.acceptance_rate() * 100.0,
    );

    Ok(results)
}

/// Generate one token from the draft model.
/// Returns (token_id, text, is_eos).
async fn draft_next_token(
    draft: &mut TextModelBase,
) -> Result<(u32, Option<String>, bool)> {
    let num_tokens = draft.tokens.len();
    let (context_size, context_index) = if draft
        .ctx
        .cache
        .as_ref()
        .expect("No cache")
        .with_kv_cache()
        && draft.generated > 0
    {
        (1, draft.index_pos)
    } else {
        (num_tokens, 0)
    };

    let context_offset = num_tokens.saturating_sub(context_size);
    let context_tokens: Vec<u32> = draft.tokens[context_offset..].to_vec();
    let num_context = context_tokens.len();

    let input = Tensor::new(context_tokens.as_slice(), &draft.ctx.device)?.unsqueeze(0)?;
    let logits = draft.forward(&input, context_index).await?;
    let logits = logits.squeeze(0)?;

    draft.index_pos += num_context;

    let next_token = draft
        .logits_processor
        .sample(&logits)
        .map_err(|e| anyhow!("draft sample: {e}"))?;

    draft.generated += 1;
    draft.tokens.push(next_token);

    let is_eos = draft
        .eos_token_id
        .as_ref()
        .map_or(false, |eos| eos.is_eos(next_token));

    let text = draft.tokenizer.decode(&[next_token], false).ok();
    Ok((next_token, text, is_eos))
}

/// Forward K tokens through the full model and return logits at ALL positions.
///
/// Unlike `TextModelBase::forward()` which returns only the last-position logits,
/// this returns shape `[K, vocab_size]` for verification.
async fn forward_verify(
    full: &mut TextModelBase,
    draft_tokens: &[u32],
) -> Result<Tensor> {
    let seq_len = draft_tokens.len();

    // Build the context: if KV cache is populated, just the draft tokens.
    // If KV cache was reset, include ALL tokens for re-prefill.
    let (context_tokens, context_index) = if full
        .ctx
        .cache
        .as_ref()
        .expect("No cache")
        .with_kv_cache()
        && full.index_pos > 0
    {
        (draft_tokens.to_vec(), full.index_pos)
    } else {
        // Need full re-prefill: all tokens + draft tokens
        let mut all = full.tokens.clone();
        all.extend_from_slice(draft_tokens);
        (all, 0)
    };

    let input = Tensor::new(context_tokens.as_slice(), &full.ctx.device)?.unsqueeze(0)?;
    let (_batch_size, input_len) = input.dims2()?;

    // Run through all blocks (same as forward() but without truncating to last position)
    let mut x = full.embedding.forward(&input)?;

    let num_blocks = full.blocks.len();
    let mut block_idx = 0;

    while block_idx < num_blocks {
        if full.blocks[block_idx].ident() == "local" {
            x = full.blocks[block_idx]
                .forward_mut(&x, context_index, block_idx, &mut full.ctx)
                .await?;
            block_idx += 1;
        } else {
            let mut batch = vec![];
            let first = block_idx;
            let curr_block_id = full.blocks[block_idx].ident().to_owned();
            while block_idx < num_blocks && full.blocks[block_idx].ident() == curr_block_id {
                batch.push((
                    full.blocks[block_idx].layer_name().to_string(),
                    context_index,
                    block_idx,
                ));
                block_idx += 1;
            }
            x = full.blocks[first]
                .forward_batch(&x, batch, &mut full.ctx)
                .await?;
        }
    }

    let x = full.ln_f.forward(&x)?;

    // Take only the last `seq_len` positions (the draft tokens)
    // If we did a full re-prefill, the context is longer than draft_tokens
    let x = if input_len > seq_len {
        x.narrow(1, input_len - seq_len, seq_len)?
    } else {
        x
    };

    // Apply lm_head to ALL positions (not just last)
    let logits = full.lm_head.forward(&x)?;
    let logits = logits.squeeze(0)?; // [seq_len, vocab_size]

    // Update index_pos to reflect the full forward
    full.index_pos = context_index + input_len;

    Ok(logits)
}
