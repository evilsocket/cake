use anyhow::Result;
use async_trait::async_trait;

use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::OLMo2Block;

/// Default EOS token for OLMo 2 models.
const DEFAULT_EOS_TOKEN: &str = "<|endoftext|>";

/// OLMo 2 model (OLMo2ForCausalLM).
/// Dense GQA with QK-norm and post-norm architecture.
pub struct OLMo2 {
    base: TextModelBase,
    history: ChatMLHistory,
}

#[async_trait]
impl Generator for OLMo2 {
    type Shardable = OLMo2Block;
    const MODEL_NAME: &'static str = "olmo2";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let base = TextModelBase::load::<OLMo2Block>(ctx, DEFAULT_EOS_TOKEN).await?;
        let history = ChatMLHistory::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for OLMo2 {
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.base.reset();
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        self.base.goodbye().await
    }

    async fn next_token(&mut self, index: usize) -> Result<Token> {
        if self.base.generated == 0 {
            let dialog = self.history.encode_dialog_to_prompt();
            self.base.prepare_prompt(&dialog)?;
        }
        self.base.next_token(index).await
    }

    fn generated_tokens(&self) -> usize {
        self.base.generated
    }
}
