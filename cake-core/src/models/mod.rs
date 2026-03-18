use anyhow::{bail, Result};
use async_trait::async_trait;
use candle_core::Tensor;
use image::{ImageBuffer, Rgb};

use chat::Message;

use crate::cake::{Context, Forwarder};
use crate::ImageGenerationArgs;

pub mod chat;
pub mod common;
#[cfg(feature = "exaone4")]
pub mod exaone4;
#[cfg(feature = "falcon3")]
pub mod falcon3;
#[cfg(feature = "gemma3")]
pub mod gemma3;
#[cfg(feature = "llama")]
pub mod llama3;
#[cfg(feature = "mistral")]
pub mod mistral;
#[cfg(feature = "olmo2")]
pub mod olmo2;
#[cfg(feature = "phi4")]
pub mod phi4;
#[cfg(feature = "qwen2")]
pub mod qwen2;
#[cfg(feature = "qwen3")]
pub mod qwen3;
#[cfg(feature = "qwen3_moe")]
pub mod qwen3_moe;
#[cfg(feature = "qwen3_5")]
pub mod qwen3_5;
#[cfg(feature = "qwen3_5_moe")]
pub mod qwen3_5_moe;
#[cfg(feature = "flux")]
pub mod flux;
pub mod sd;
#[cfg(feature = "luxtts")]
pub mod luxtts;
#[cfg(feature = "vibevoice")]
pub mod vibevoice;

/// The input modality a model accepts.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum InputModality {
    /// The model accepts text input.
    Text,
}

/// The output modality a model produces.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum OutputModality {
    /// The model produces text tokens.
    Text,
    /// The model produces images.
    Image,
    /// The model produces audio.
    Audio,
}

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

/// Arguments for audio/TTS generation.
pub struct AudioGenerationArgs {
    /// Text to synthesize.
    pub input: String,
    /// Raw WAV bytes for voice cloning (from API base64 upload).
    pub voice_data: Option<Vec<u8>>,
    /// Path to voice prompt file (.safetensors or .wav).
    pub voice_path: Option<String>,
    /// Classifier-free guidance scale.
    pub cfg_scale: f32,
    /// Maximum speech frames to generate.
    pub max_frames: usize,
    /// Number of diffusion steps per frame.
    pub diffusion_steps: usize,
}

/// Output from audio generation.
pub struct AudioOutput {
    /// PCM f32 samples.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl AudioOutput {
    /// Encode as WAV bytes (16-bit PCM).
    pub fn to_wav_bytes(&self) -> Vec<u8> {
        crate::utils::wav::encode_wav_bytes(&self.samples, self.sample_rate)
    }

    /// Return raw PCM f32 bytes (little-endian).
    pub fn to_pcm_bytes(&self) -> Vec<u8> {
        self.samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect()
    }
}

/// Encode PCM f32 samples as a WAV file (16-bit PCM, mono).
///
/// Re-exported from [`crate::utils::wav::encode_wav_bytes`].
pub use crate::utils::wav::encode_wav_bytes;

#[async_trait]
pub trait AudioGenerator: Generator {
    async fn generate_audio(&mut self, args: &AudioGenerationArgs) -> Result<AudioOutput>;
}

/// Unified model trait — every model used by Master implements this.
/// Each model provides real implementations for its modality and
/// default "not supported" stubs for the others.
#[async_trait]
pub trait Model: Generator + Send + Sync + 'static {
    /// The input modality this model accepts (always Text for now).
    fn input_modality(&self) -> InputModality {
        InputModality::Text
    }

    /// The output modality this model produces.
    fn output_modality(&self) -> OutputModality;

    // ── Text generation (default: not supported) ──

    fn add_message(&mut self, _message: Message) -> Result<()> {
        bail!("text generation not supported by this model")
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        Ok(())
    }

    async fn next_token(&mut self, _index: usize) -> Result<Token> {
        bail!("text generation not supported")
    }

    fn generated_tokens(&self) -> usize {
        0
    }

    // ── Image generation (default: not supported) ──

    async fn generate_image(
        &mut self,
        _args: &ImageGenerationArgs,
        _callback: Box<dyn FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send>,
    ) -> Result<()> {
        bail!("image generation not supported")
    }

    // ── Audio generation (default: not supported) ──

    async fn generate_audio(&mut self, _args: &AudioGenerationArgs) -> Result<AudioOutput> {
        bail!("audio generation not supported")
    }
}

/// Implement `Model` for a `TextGenerator` — delegates text methods, stubs the rest.
#[macro_export]
macro_rules! impl_model_for_text {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl $crate::models::Model for $ty {
            fn output_modality(&self) -> $crate::models::OutputModality {
                $crate::models::OutputModality::Text
            }
            fn add_message(
                &mut self,
                message: $crate::models::chat::Message,
            ) -> anyhow::Result<()> {
                $crate::models::TextGenerator::add_message(self, message)
            }
            fn reset(&mut self) -> anyhow::Result<()> {
                $crate::models::TextGenerator::reset(self)
            }
            async fn goodbye(&mut self) -> anyhow::Result<()> {
                $crate::models::TextGenerator::goodbye(self).await
            }
            async fn next_token(&mut self, index: usize) -> anyhow::Result<$crate::models::Token> {
                $crate::models::TextGenerator::next_token(self, index).await
            }
            fn generated_tokens(&self) -> usize {
                $crate::models::TextGenerator::generated_tokens(self)
            }
        }
    };
}

/// Implement `Model` for an `ImageGenerator` — delegates image method, stubs the rest.
#[macro_export]
macro_rules! impl_model_for_image {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl $crate::models::Model for $ty {
            fn output_modality(&self) -> $crate::models::OutputModality {
                $crate::models::OutputModality::Image
            }
            async fn generate_image(
                &mut self,
                args: &$crate::ImageGenerationArgs,
                callback: Box<
                    dyn FnMut(Vec<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>>) + Send,
                >,
            ) -> anyhow::Result<()> {
                $crate::models::ImageGenerator::generate_image(self, args, callback).await
            }
        }
    };
}

/// Implement `Model` for an `AudioGenerator` — delegates audio method, stubs the rest.
#[macro_export]
macro_rules! impl_model_for_audio {
    ($ty:ty) => {
        #[async_trait::async_trait]
        impl $crate::models::Model for $ty {
            fn output_modality(&self) -> $crate::models::OutputModality {
                $crate::models::OutputModality::Audio
            }
            async fn generate_audio(
                &mut self,
                args: &$crate::models::AudioGenerationArgs,
            ) -> anyhow::Result<$crate::models::AudioOutput> {
                $crate::models::AudioGenerator::generate_audio(self, args).await
            }
        }
    };
}

// ── NoAudio stub (always compiled, never loaded) ──

/// Stub audio generator used when no audio model is active.
pub struct NoAudio;

/// Stub block for NoAudio.
pub struct NoAudioBlock;

impl std::fmt::Debug for NoAudio {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoAudio")
    }
}

impl std::fmt::Debug for NoAudioBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoAudioBlock")
    }
}

impl std::fmt::Display for NoAudioBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoAudioBlock")
    }
}

#[async_trait]
impl Generator for NoAudio {
    type Shardable = NoAudioBlock;
    const MODEL_NAME: &'static str = "none";

    async fn load(_context: &mut Context) -> Result<Option<Box<Self>>> {
        Ok(None)
    }
}

#[async_trait]
impl AudioGenerator for NoAudio {
    async fn generate_audio(&mut self, _args: &AudioGenerationArgs) -> Result<AudioOutput> {
        anyhow::bail!("No audio model loaded")
    }
}

#[async_trait]
impl Forwarder for NoAudioBlock {
    fn load(_name: String, _ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        anyhow::bail!("NoAudioBlock cannot be loaded")
    }

    async fn forward(
        &self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        anyhow::bail!("NoAudioBlock cannot forward")
    }

    async fn forward_mut(
        &mut self,
        _x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        anyhow::bail!("NoAudioBlock cannot forward")
    }

    fn layer_name(&self) -> &str {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_output_to_wav_bytes() {
        let output = AudioOutput {
            samples: vec![0.0; 10],
            sample_rate: 24000,
        };
        let wav = output.to_wav_bytes();
        assert_eq!(&wav[0..4], b"RIFF");
    }

    #[test]
    fn test_audio_output_to_pcm_bytes() {
        let output = AudioOutput {
            samples: vec![1.0, -1.0],
            sample_rate: 24000,
        };
        let pcm = output.to_pcm_bytes();
        assert_eq!(pcm.len(), 8); // 2 floats * 4 bytes
        let f0 = f32::from_le_bytes([pcm[0], pcm[1], pcm[2], pcm[3]]);
        assert!((f0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_no_audio_debug() {
        let na = NoAudio;
        assert_eq!(format!("{:?}", na), "NoAudio");
    }

    #[test]
    fn test_no_audio_block_debug_display() {
        let b = NoAudioBlock;
        assert_eq!(format!("{:?}", b), "NoAudioBlock");
        assert_eq!(format!("{}", b), "NoAudioBlock");
    }

    #[test]
    fn test_audio_generation_args_construction() {
        let args = AudioGenerationArgs {
            input: "hello".into(),
            voice_data: None,
            voice_path: Some("/tmp/voice.wav".into()),
            cfg_scale: 1.5,
            max_frames: 150,
            diffusion_steps: 10,
        };
        assert_eq!(args.input, "hello");
        assert_eq!(args.cfg_scale, 1.5);
    }
}
