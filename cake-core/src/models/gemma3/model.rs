use anyhow::Result;
use async_trait::async_trait;

use super::history::Gemma3History;
use crate::models::common::EosTokenId;
use crate::models::common::text_model::TextModelBase;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

use super::block::Gemma3Block;

/// Default EOS token for Gemma 3 Instruct models.
const DEFAULT_EOS_TOKEN: &str = "<eos>";

/// Gemma 3 model (Gemma3ForCausalLM).
///
/// Supports 1B / 4B / 12B / 27B variants. Uses interleaved local (sliding-window,
/// no RoPE) and global (full RoPE) attention layers with QK-norm on both.
pub struct Gemma3 {
    base: TextModelBase,
    history: Gemma3History,
}

#[async_trait]
impl Generator for Gemma3 {
    type Shardable = Gemma3Block;
    const MODEL_NAME: &'static str = "gemma3";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let mut base = TextModelBase::load::<Gemma3Block>(ctx, DEFAULT_EOS_TOKEN).await?;

        // Gemma 3 Instruct also stops at <end_of_turn> (token 106).
        if let Some(eot_id) = base.tokenizer.token_to_id("<end_of_turn>") {
            base.eos_token_id = Some(match base.eos_token_id.take() {
                Some(EosTokenId::Single(id)) => EosTokenId::Multiple(vec![id, eot_id]),
                Some(EosTokenId::Multiple(mut ids)) => {
                    ids.push(eot_id);
                    EosTokenId::Multiple(ids)
                }
                None => EosTokenId::Single(eot_id),
            });
        }

        let history = Gemma3History::new();
        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for Gemma3 {
    fn add_message(&mut self, message: Message) -> Result<()> {
        // Gemma 3 IT has no dedicated system role — the system prompt is injected into
        // the first user turn. Small models (1B) degrade badly when a system prompt is
        // present. Warn users so they know to pass --system-prompt "".
        if message.role == crate::models::chat::MessageRole::System && !message.content.trim().is_empty() {
            log::warn!(
                "gemma3: system prompt is injected into the first user turn. \
                 The 1B model may generate incoherent output — pass --system-prompt \"\" \
                 for reliable results."
            );
        }
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
