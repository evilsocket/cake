use anyhow::Result;
use async_trait::async_trait;

use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::Qwen3_5Block;

/// Default end of stream token if not found in configuration.
const DEFAULT_EOS_TOKEN: &str = "<|im_end|>";

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

        if let Some(ref draft_model) = ctx.args.draft_model.clone() {
            // Draft model uses standard Transformer blocks (dense model)
            base.load_draft::<crate::models::common::Transformer>(draft_model, DEFAULT_EOS_TOKEN).await?;
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
        if self.base.generated == 0 {
            let dialog = self.history.encode_dialog_to_prompt();
            self.base.prepare_prompt(&dialog)?;
        }
        self.base.next_token(index).await
    }

    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize {
        self.base.generated
    }
}
