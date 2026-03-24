use anyhow::Result;
use async_trait::async_trait;

use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::EosTokenId;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::Qwen3_5Block;

/// Default end of stream token if not found in configuration.
const DEFAULT_EOS_TOKEN: &str = "<|endoftext|>";

/// Qwen3.5 main class.
pub struct Qwen3_5 {
    base: TextModelBase,
    history: ChatMLHistory,
}

#[async_trait]
impl Generator for Qwen3_5 {
    type Shardable = Qwen3_5Block;
    const MODEL_NAME: &'static str = "qwen3_5";

    /// Load this model from the context.
    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let mut base = TextModelBase::load::<Qwen3_5Block>(ctx, DEFAULT_EOS_TOKEN).await?;
        // Qwen3.5 config sets eos_token_id=<|endoftext|> but ChatML uses <|im_end|> as turn terminator.
        // Add <|im_end|> to the EOS set so generation stops at turn boundaries.
        if let Some(im_end_id) = base.tokenizer.token_to_id("<|im_end|>") {
            base.eos_token_id = Some(match base.eos_token_id.take() {
                Some(EosTokenId::Single(id)) => EosTokenId::Multiple(vec![id, im_end_id]),
                Some(EosTokenId::Multiple(mut ids)) => {
                    ids.push(im_end_id);
                    EosTokenId::Multiple(ids)
                }
                None => EosTokenId::Single(im_end_id),
            });
        }
        let history = ChatMLHistory::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for Qwen3_5 {
    /// Add a message to the chat history.
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    /// Reset the chat pipeline state.
    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.base.reset();
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        self.base.goodbye().await
    }

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token> {
        // Prefill tokens with chat history the first time.
        // Qwen3.5's chat template appends an empty <think> block to skip
        // reasoning mode and generate the response directly in the user's language.
        if self.base.generated == 0 {
            let dialog = self.history.encode_dialog_to_prompt() + "<think>\n\n</think>\n\n";
            self.base.prepare_prompt(&dialog)?;
        }
        self.base.next_token(index).await
    }

    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize {
        self.base.generated
    }
}
