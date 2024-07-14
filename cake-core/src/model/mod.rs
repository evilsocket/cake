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

pub struct Token {
    pub id: u32,
    pub text: Option<String>,
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

#[async_trait]
pub trait Generator {
    async fn load(context: Context) -> Result<Box<Self>>;
    async fn next_token(&mut self, index: usize) -> Result<Token>;
    async fn last(&mut self) -> Result<Option<String>>;
    fn generated_tokens(&self) -> usize;
}
