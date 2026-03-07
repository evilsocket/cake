use anyhow::Result;
use async_trait::async_trait;

use super::mixtral_shardable::MixtralShardable;
use super::moe_block::MoeBlock;
use crate::cake::Context;
use crate::models::chat::Message;
use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::{Generator, TextGenerator, Token};

const DEFAULT_EOS_TOKEN: &str = "</s>";

/// Mixtral MoE main model.
///
/// Uses MoeBlock (attention + sparse expert MLP) for transformer layers,
/// with the rest handled by TextModelBase (embedding, ln_f, lm_head).
pub struct Mixtral {
    base: TextModelBase,
    history: ChatMLHistory,
}

#[async_trait]
impl Generator for Mixtral {
    type Shardable = MixtralShardable;
    const MODEL_NAME: &'static str = "mixtral";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let base = TextModelBase::load::<MoeBlock>(ctx, DEFAULT_EOS_TOKEN).await?;
        let history = ChatMLHistory::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for Mixtral {
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
