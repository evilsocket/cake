use anyhow::Result;
use async_trait::async_trait;

use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::MixtralBlock;

/// Default EOS token for Mixtral Instruct models (Mistral family convention).
const DEFAULT_EOS_TOKEN: &str = "</s>";

/// Mixtral MoE model (`MixtralForCausalLM`).
///
/// Covers Mixtral-8x7B-Instruct and Mixtral-8x22B-Instruct.
/// Uses the same tokenizer and chat template as dense Mistral;
/// only the FFN is replaced with a sparse Mixture-of-Experts block.
pub struct MixtralMoe {
    base: TextModelBase,
    history: ChatMLHistory,
}

#[async_trait]
impl Generator for MixtralMoe {
    type Shardable = MixtralBlock;
    const MODEL_NAME: &'static str = "mixtral_moe";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let base = TextModelBase::load::<MixtralBlock>(ctx, DEFAULT_EOS_TOKEN).await?;
        let history = ChatMLHistory::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for MixtralMoe {
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
