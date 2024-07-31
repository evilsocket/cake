pub mod chat;
pub mod llama3;
pub mod sd;

use crate::cake::{Context, Forwarder};

use anyhow::Result;
use async_trait::async_trait;
use chat::Message;
use crate::models::sd::ImageGenerationArgs;

/// A token.
pub struct Token {
    /// Numerical identifier.
    pub id: u32,
    /// Resolved text token or None if not present in the tokenizer.
    pub text: Option<String>,
    /// Set to true if the stream of tokens is over.
    pub is_end_of_stream: bool,
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            if let Some(text) = &self.text {
                text.clone()
            } else {
                format!("<token {}>", self.id)
            }
        )
    }
}

/// A model must implement this trait in order to be usable by the Cake framework.
#[async_trait]
pub trait Generator {

    /// This associated type determines which part of the model can be sharded.
    type Shardable: Forwarder;

    /// The model name.
    const MODEL_NAME: &'static str;

    /// Load the model from the context.
    async fn load(context: Context) -> Result<Option<Box<Self>>>;
}

pub trait TextGenerator: Generator {

    /// Add a message to the chat.
    fn add_message(&mut self, message: Message) -> Result<()>;
    /// Clear chat history.
    fn reset(&mut self) -> Result<()>;

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token>;
    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize;
}

pub trait ImageGenerator: Generator {
    async fn generate_image(&mut self, args: ImageGenerationArgs) -> Result<String>;
}
