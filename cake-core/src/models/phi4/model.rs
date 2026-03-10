use anyhow::Result;
use async_trait::async_trait;

use super::history::Phi4History;
use crate::models::common::EosTokenId;
use crate::models::common::text_model::TextModelBase;
use crate::models::common::Transformer;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

/// Default EOS token for Phi-3/4 chat models.
const DEFAULT_EOS_TOKEN: &str = "<|endoftext|>";

/// Phi-4-mini / Phi-4 model.
pub struct Phi4 {
    base: TextModelBase,
    history: Phi4History,
}

#[async_trait]
impl Generator for Phi4 {
    type Shardable = Transformer;
    const MODEL_NAME: &'static str = "phi4";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let mut base = TextModelBase::load::<Transformer>(ctx, DEFAULT_EOS_TOKEN).await?;

        // Phi-4 also stops at <|end|> (end-of-turn marker) in addition to <|endoftext|>.
        if let Some(end_id) = base.tokenizer.token_to_id("<|end|>") {
            base.eos_token_id = Some(match base.eos_token_id.take() {
                Some(EosTokenId::Single(id)) => EosTokenId::Multiple(vec![id, end_id]),
                Some(EosTokenId::Multiple(mut ids)) => {
                    ids.push(end_id);
                    EosTokenId::Multiple(ids)
                }
                None => EosTokenId::Single(end_id),
            });
        }

        let history = Phi4History::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for Phi4 {
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
