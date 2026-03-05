use anyhow::Result;
use async_trait::async_trait;

use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::Qwen3_5MoeBlock;

/// EOS token for Qwen3.5 MoE — same ChatML template as dense Qwen3.5.
const DEFAULT_EOS_TOKEN: &str = "<|endoftext|>";

/// Qwen3.5 MoE model (`Qwen3_5MoeForConditionalGeneration`).
///
/// Covers Qwen3.5-35B-A3B and GPTQ variants. Reuses Qwen3.5's hybrid
/// linear/full attention; only the FFN is replaced with a sparse
/// MoE block (256 experts, top-8 per token) plus a shared expert.
pub struct Qwen3_5Moe {
    base: TextModelBase,
    history: ChatMLHistory,
}

#[async_trait]
impl Generator for Qwen3_5Moe {
    type Shardable = Qwen3_5MoeBlock;
    const MODEL_NAME: &'static str = "qwen3_5_moe";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let mut base =
            TextModelBase::load::<Qwen3_5MoeBlock>(ctx, DEFAULT_EOS_TOKEN).await?;
        // Also register <|im_end|> as EOS (ChatML turn terminator).
        use crate::models::common::EosTokenId;
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
impl TextGenerator for Qwen3_5Moe {
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
