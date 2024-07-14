//! This module contains model and inference specific code.
mod attention;
mod cache;
mod config;
mod llama;
mod mlp;
mod transformer;

pub use attention::*;
pub use cache::*;
pub use config::*;
pub use llama::*;
pub use mlp::*;
pub use transformer::*;

use crate::cake::{Context, Forwarder};

use anyhow::Result;
use async_trait::async_trait;

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
    type Shardable : Forwarder;
    /// Load the model from the context.
    async fn load(context: Context) -> Result<Box<Self>>;
    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token>;
    /// Return the last residual data if any.
    async fn last(&mut self) -> Result<Option<String>>;
    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize;
}
