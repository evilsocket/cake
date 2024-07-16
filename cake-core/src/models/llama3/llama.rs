use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{linear_no_bias as linear, Embedding, Linear, Module, RmsNorm};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use tokenizers::Tokenizer;

use crate::{
    cake::{Context, Forwarder},
    models::{chat::Message, Generator, Token},
    utils::{self, TokenOutputStream},
};

use super::transformer::Transformer;

/// End of stream token.
const EOS_TOKEN: &str = "</s>";

/// Load the tokenizer and return the first tokens from the prompt in context.
fn load_tokenizer(ctx: &Context) -> Result<(TokenOutputStream, Option<u32>)> {
    let tokenizer_filename = ctx.data_path.join("tokenizer.json");

    log::info!("loading tokenizer from {}", tokenizer_filename.display());

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(anyhow::Error::msg)?;
    let eos_token_id = ctx
        .config
        .eos_token_id
        .or_else(|| tokenizer.token_to_id(EOS_TOKEN));

    let tokenizer = utils::TokenOutputStream::new(tokenizer);

    Ok((tokenizer, eos_token_id))
}

/// Create the logit sampling logic from the context.
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

/// LLama main class.
pub struct LLama {
    ctx: Context,

    tokenizer: TokenOutputStream,
    embedding: Embedding,
    eos_token_id: Option<u32>,
    index_pos: usize,

    blocks: Vec<Box<dyn Forwarder>>,

    ln_f: RmsNorm,
    lm_head: Linear,

    logits_processor: LogitsProcessor,

    history: Vec<Message>,
    tokens: Vec<u32>,
}

impl LLama {
    async fn forward(&mut self, x: &Tensor, idx: usize) -> Result<Tensor> {
        let (_batch_size, seq_len) = x.dims2()?;
        let mut x = self.embedding.forward(x)?;

        let num_blocks = self.blocks.len();
        let mut block_idx = 0;

        // log::info!("X = {}", &x);

        while block_idx < num_blocks {
            let curr_block_id = self.blocks[block_idx].ident().to_owned();
            if curr_block_id == "local" {
                // do not batch local inferences
                x = self.blocks[block_idx]
                    .forward_mut(&x, idx, block_idx, &mut self.ctx.cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward operation of local block {block_idx}: {e}")
                    })?;

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

                x = self.blocks[first]
                    .forward_batch(&x, batch, &mut self.ctx.cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward batch operation for block {block_idx}: {e}")
                    })?;
            }

            // log::info!("{}.forward(X) -> {}", &curr_block_id, &x);
        }

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

        logits
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("error converting logits: {e}"))
    }

    fn raw_message(&self, message: &Message) -> String {
        format!(
            "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>\n",
            message.role, &message.content
        )
    }
}

#[async_trait]
impl Generator for LLama {
    type Shardable = Transformer;

    const MODEL_NAME: &'static str = "llama3";

    /// Load this model from the context.
    async fn load(ctx: Context) -> Result<Box<Self>> {
        log::info!("loading embeddings ...");
        let embedding: Embedding = candle_nn::embedding(
            ctx.config.vocab_size,
            ctx.config.hidden_size,
            ctx.var_builder.pp("model.embed_tokens"),
        )?;

        log::info!("loading lm_head ...");
        let lm_head = linear(
            ctx.config.hidden_size,
            ctx.config.vocab_size,
            ctx.var_builder.pp("lm_head"),
        )?;

        log::info!("loading model.norm ...");
        let ln_f = candle_nn::rms_norm(
            ctx.config.hidden_size,
            ctx.config.rms_norm_eps,
            ctx.var_builder.pp("model.norm"),
        )?;

        log::info!("loading {} blocks ...", ctx.config.num_hidden_layers);

        let mut blocks: Vec<Box<dyn Forwarder>> = vec![];

        for i in 0..ctx.config.num_hidden_layers {
            let block_layer_name = format!("model.layers.{i}");
            if let Some((node_name, node)) = ctx.topology.get_node_for_layer(&block_layer_name) {
                log::debug!("node {node_name} will serve {}", &block_layer_name);
                blocks.push(Box::new(
                    crate::cake::Client::new(ctx.device.clone(), &node.host, &block_layer_name)
                        .await?,
                ));
            } else {
                log::debug!("{} will be served locally", &block_layer_name);
                blocks.push(Transformer::load(
                    block_layer_name.clone(),
                    ctx.var_builder.pp(&block_layer_name),
                    &ctx.config,
                )?);
            }
        }

        for block in &blocks {
            log::info!("  {}", block)
        }

        let (tokenizer, eos_token_id) = load_tokenizer(&ctx)?;
        let tokens = vec![];
        let history = vec![];

        let logits_processor = create_logits_processor(&ctx);
        let index_pos = 0;

        log::info!(
            "model loaded - mem={}",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(Box::new(Self {
            tokenizer,
            tokens,
            history,
            eos_token_id,
            index_pos,
            ctx,
            embedding,
            blocks,
            ln_f,
            lm_head,
            logits_processor,
        }))
    }

    /// Add a message to the chat history.
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);

        // generate raw from history
        let mut raw = "<|begin_of_text|>".to_string();

        for message in &self.history {
            raw += &self.raw_message(message);
        }

        raw += "<|start_header_id|>assistant<|end_header_id|>";

        log::debug!("{}", &raw);

        // tokenize raw
        self.tokens = self
            .tokenizer
            .tokenizer()
            .encode(raw, true)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();

        Ok(())
    }

    fn clear_history(&mut self) -> Result<()> {
        self.tokens.clear();
        self.history.clear();
        Ok(())
    }

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token> {
        log::trace!("model.next_token({index})");

        let num_tokens = self.generated_tokens();

        let (context_size, context_index) = if self.ctx.cache.with_kv_cache() && index > 0 {
            (1, self.index_pos)
        } else {
            (num_tokens, 0)
        };

        let context_offset = num_tokens.saturating_sub(context_size);
        let context_tokens = self.tokens[context_offset..].to_vec();
        let num_context_tokens = context_tokens.len();
        let input = Tensor::new(context_tokens, &self.ctx.device)?
            .unsqueeze(0)
            .map_err(|e| anyhow!("error squeezing context tokens: {e}"))?;

        let logits = self
            .forward(&input, context_index)
            .await
            .map_err(|e| anyhow!("error in model.forward: {e}"))?;

        let logits = logits
            .squeeze(0)
            .map_err(|e| anyhow!("error squeezing logits: {e}"))?;

        let logits = if self.ctx.args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = num_tokens.saturating_sub(self.ctx.args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.ctx.args.repeat_penalty,
                &self.tokens[start_at..],
            )?
        };
        self.index_pos += num_context_tokens;

        let next_token = self
            .logits_processor
            .sample(&logits)
            .map_err(|e| anyhow!("error sampling logits {logits}: {e}"))?;
        self.tokens.push(next_token);

        Ok(Token {
            id: next_token,
            text: self.tokenizer.next_token(next_token)?,
            is_end_of_stream: Some(next_token) == self.eos_token_id,
        })
    }

    /// Return any resitual token if necessary or None if not.
    async fn last(&mut self) -> Result<Option<String>> {
        self.tokenizer.decode_rest().map_err(anyhow::Error::msg)
    }

    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize {
        self.tokens.len()
    }
}
