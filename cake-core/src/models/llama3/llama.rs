use anyhow::Result;
use async_trait::async_trait;

use super::History;
use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::common::Transformer;
use crate::models::TextGenerator;
use crate::{
    cake::Context,
    models::{chat::Message, Generator, Token},
};

/// Default end of stream token if not found in configuration.
const DEFAULT_EOS_TOKEN: &str = "<|eot_id|>";

/// Chat history format: LLaMA-3 native or ChatML fallback (e.g. for SmolLM2).
enum ChatHistory {
    Llama(History),
    ChatML(ChatMLHistory),
}

impl ChatHistory {
    fn push(&mut self, msg: Message) {
        match self {
            Self::Llama(h) => h.push(msg),
            Self::ChatML(h) => h.push(msg),
        }
    }
    fn encode_dialog_to_prompt(&self) -> String {
        match self {
            Self::Llama(h) => h.encode_dialog_to_prompt(),
            Self::ChatML(h) => h.encode_dialog_to_prompt(),
        }
    }
    fn clear(&mut self) {
        match self {
            Self::Llama(h) => h.clear(),
            Self::ChatML(h) => h.clear(),
        }
    }
}

/// LLama main class.
pub struct LLama {
    base: TextModelBase,
    history: ChatHistory,
}

#[async_trait]
impl Generator for LLama {
    type Shardable = Transformer;
    const MODEL_NAME: &'static str = "llama3";

    /// Load this model from the context.
    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let base = TextModelBase::load::<Transformer>(ctx, DEFAULT_EOS_TOKEN).await?;

        // Auto-detect chat template: LLaMA-3 uses <|begin_of_text|>;
        // other LlamaForCausalLM models (e.g. SmolLM2) may use ChatML (<|im_start|>).
        let history = if base.tokenizer.token_to_id("<|begin_of_text|>").is_some() {
            log::debug!("llama: using LLaMA-3 chat format");
            ChatHistory::Llama(History::new())
        } else {
            log::info!("llama: tokenizer has no <|begin_of_text|>, falling back to ChatML format");
            ChatHistory::ChatML(ChatMLHistory::new())
        };

        Ok(Some(Box::new(Self { base, history })))
    }
}

#[async_trait]
impl TextGenerator for LLama {
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
