use anyhow::Result;
use async_trait::async_trait;

use super::QwenHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::common::Transformer;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

/// Default end of stream token if not found in configuration.
const DEFAULT_EOS_TOKEN: &str = "<|endoftext|>";

/// Qwen2/Qwen2.5 main class.
pub struct Qwen2 {
    base: TextModelBase,
    history: QwenHistory,
}

#[async_trait]
impl Generator for Qwen2 {
    type Shardable = Transformer;
    const MODEL_NAME: &'static str = "qwen2";

    /// Load this model from the context.
    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let mut base = TextModelBase::load::<Transformer>(ctx, DEFAULT_EOS_TOKEN).await?;

        if let Some(ref draft_model) = ctx.args.draft_model.clone() {
            base.load_draft::<Transformer>(draft_model, DEFAULT_EOS_TOKEN).await?;
        }

        let history = QwenHistory::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for Qwen2 {
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
