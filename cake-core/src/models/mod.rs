use anyhow::Result;
use async_trait::async_trait;
use image::{ImageBuffer, Rgb};

use chat::Message;

use crate::cake::{Context, Forwarder};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;

pub mod chat;
pub mod common;
#[cfg(feature = "llama")]
pub mod llama3;
#[cfg(feature = "qwen2")]
pub mod qwen2;
#[cfg(feature = "qwen3_5")]
pub mod qwen3_5;
pub mod flux;
#[cfg(feature = "llava")]
pub mod llava;
pub mod ltx_video;
pub mod ltx2;
#[cfg(feature = "mixtral")]
pub mod mixtral;
pub mod sd;
pub mod speculative;
pub mod hunyuan_video;

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
    async fn load(context: &mut Context) -> Result<Option<Box<Self>>>;
}

#[async_trait]
pub trait TextGenerator: Generator {
    /// Add a message to the chat.
    fn add_message(&mut self, message: Message) -> Result<()>;
    /// Clear chat history.
    fn reset(&mut self) -> Result<()>;
    /// clear worker kv cache
    async fn goodbye(&mut self) -> Result<()>;

    /// Return the next token.
    async fn next_token(&mut self, index: usize) -> Result<Token>;
    /// Return the number of generated tokens so far.
    fn generated_tokens(&self) -> usize;
}

#[async_trait]
pub trait ImageGenerator: Generator {
    async fn generate_image<F>(
        &mut self,
        args: &ImageGenerationArgs,
        mut callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static;
}

/// A model that generates video (sequence of frames with temporal metadata).
#[async_trait]
pub trait VideoGenerator: Generator {
    /// Generate a video from the given arguments.
    /// Returns a `VideoOutput` containing all frames, fps, and dimensions.
    async fn generate_video(
        &mut self,
        args: &ImageGenerationArgs,
    ) -> Result<VideoOutput>;
}

/// A vision-language model that extends text generation with image understanding.
#[async_trait]
pub trait VisionLanguageGenerator: TextGenerator {
    /// Process an image tensor and return visual embeddings.
    async fn encode_image(&mut self, image: &candle_core::Tensor) -> Result<candle_core::Tensor>;
    /// Add pre-encoded image embeddings to the conversation context.
    /// These will be merged with text embeddings on the next forward pass.
    fn add_image(&mut self, image_embeddings: candle_core::Tensor) -> Result<()>;
}
