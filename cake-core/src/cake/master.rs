use super::Context;
use crate::{
    model::{Llama, EOS_TOKEN},
    utils::{self, TokenOutputStream},
};

use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

pub struct Master {
    ctx: Context,
    model: Llama,
    tokenizer: TokenOutputStream,
    tokens: Vec<u32>,
    eos_token_id: Option<u32>,
    logits_processor: LogitsProcessor,
}

impl Master {
    fn create_tokenizer(ctx: &Context) -> Result<(TokenOutputStream, Vec<u32>, Option<u32>)> {
        let tokenizer_filename = ctx.data_path.join("tokenizer.json");

        log::info!("loading tokenizer from {}", tokenizer_filename.display());

        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
        let eos_token_id = ctx
            .config
            .eos_token_id
            .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

        let tokens = tokenizer
            .encode(ctx.args.prompt.clone(), true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        let tokenizer = utils::TokenOutputStream::new(tokenizer);

        log::debug!("prompt tokens: {:?}", &tokens);

        Ok((tokenizer, tokens, eos_token_id))
    }

    fn create_logits_processor(ctx: &Context) -> LogitsProcessor {
        let temperature = ctx.args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (ctx.args.top_k, ctx.args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(ctx.args.seed, sampling)
    }

    pub async fn new(ctx: Context) -> Result<Self> {
        log::info!("loading master topology from {}", &ctx.args.topology);

        let model = Llama::load(&ctx.var_builder, &ctx.config, &ctx.device, &ctx.topology).await?;

        log::info!(
            "model loaded - mem={}",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let (tokenizer, tokens, eos_token_id) = Self::create_tokenizer(&ctx)?;
        let logits_processor = Self::create_logits_processor(&ctx);

        Ok(Self {
            ctx,
            model,
            tokenizer,
            tokens,
            eos_token_id,
            logits_processor,
        })
    }

    pub async fn generate<S>(&mut self, stream: S) -> Result<()>
    where
        S: Fn(&str),
    {
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        stream(&self.ctx.args.prompt);

        let mut start_gen = std::time::Instant::now();
        let mut index_pos = 0;

        for index in 0..self.ctx.args.sample_len {
            let (context_size, context_index) = if self.ctx.cache.use_kv_cache && index > 0 {
                (1, index_pos)
            } else {
                (self.tokens.len(), 0)
            };

            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }

            let context_tokens = &self.tokens[self.tokens.len().saturating_sub(context_size)..];
            let input = Tensor::new(context_tokens, &self.ctx.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, context_index, &mut self.ctx.cache)
                .await?;
            let logits = logits.squeeze(0)?;
            let logits = if self.ctx.args.repeat_penalty == 1. {
                logits
            } else {
                let start_at = self
                    .tokens
                    .len()
                    .saturating_sub(self.ctx.args.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.ctx.args.repeat_penalty,
                    &self.tokens[start_at..],
                )?
            };
            index_pos += context_tokens.len();

            let next_token = self.logits_processor.sample(&logits)?;
            self.tokens.push(next_token);

            if Some(next_token) == self.eos_token_id {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                stream(&t);
            }
        }
        if let Some(rest) = self.tokenizer.decode_rest().map_err(anyhow::Error::msg)? {
            stream(&rest);
        }

        // signal end of stream
        stream("");

        let dt = start_gen.elapsed();
        let generated = self.tokens.len();

        log::info!(
            "{} tokens generated ({} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }
}
