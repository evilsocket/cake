use anyhow::Result;
use async_trait::async_trait;

use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::Qwen3MoeBlock;

/// Default EOS token — same as Qwen3 dense (ChatML).
const DEFAULT_EOS_TOKEN: &str = "<|im_end|>";

/// Qwen3 MoE model (`Qwen3MoeForCausalLM`).
///
/// Covers Qwen3-30B-A3B, Qwen3-235B-A22B, and Qwen3-Coder MoE variants.
/// Identical tokenizer, chat template, and attention to dense Qwen3;
/// only the FFN is replaced with a sparse Mixture-of-Experts block.
pub struct Qwen3Moe {
    base: TextModelBase,
    history: ChatMLHistory,
}

#[async_trait]
impl Generator for Qwen3Moe {
    type Shardable = Qwen3MoeBlock;
    const MODEL_NAME: &'static str = "qwen3_moe";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let base = TextModelBase::load::<Qwen3MoeBlock>(ctx, DEFAULT_EOS_TOKEN).await?;
        let history = ChatMLHistory::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for Qwen3Moe {
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
