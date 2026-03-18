//! Mock models and test app builders for API integration tests.
//!
//! All mocks run offline — no GPU, no model files, no network.

use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use std::sync::Arc;
use tokio::sync::RwLock;

use actix_web::{web, App};

use crate::cake::api;
use crate::cake::{Context, Forwarder, Master};
use crate::models::chat::Message;
use crate::models::{
    AudioGenerationArgs, AudioGenerator, AudioOutput, Generator, ImageGenerator, NoAudio,
    TextGenerator, Token,
};
use crate::ImageGenerationArgs;
use image::{ImageBuffer, Rgb};

// ── MockBlock (shared by all mock generators) ──

pub struct MockBlock;

impl std::fmt::Debug for MockBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockBlock")
    }
}

impl std::fmt::Display for MockBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MockBlock")
    }
}

#[async_trait]
impl Forwarder for MockBlock {
    fn load(_name: String, _ctx: &Context) -> Result<Box<Self>> {
        Ok(Box::new(Self))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        Ok(x.clone())
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        Ok(x.clone())
    }

    fn layer_name(&self) -> &str {
        "mock"
    }
}

// ── MockTextGenerator ──

pub struct MockTextGenerator {
    pub tokens: Vec<String>,
    pub cursor: usize,
    pub messages: Vec<Message>,
    pub generated: usize,
}

#[async_trait]
impl Generator for MockTextGenerator {
    type Shardable = MockBlock;
    const MODEL_NAME: &'static str = "mock-text";

    async fn load(_ctx: &mut Context) -> Result<Option<Box<Self>>> {
        Ok(Some(Box::new(Self {
            tokens: vec!["Hello".into(), " world".into()],
            cursor: 0,
            messages: Vec::new(),
            generated: 0,
        })))
    }
}

#[async_trait]
impl TextGenerator for MockTextGenerator {
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.messages.push(message);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.messages.clear();
        self.cursor = 0;
        self.generated = 0;
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        Ok(())
    }

    async fn next_token(&mut self, _index: usize) -> Result<Token> {
        if self.cursor >= self.tokens.len() {
            return Ok(Token {
                id: 0,
                text: None,
                is_end_of_stream: true,
            });
        }
        let text = self.tokens[self.cursor].clone();
        self.cursor += 1;
        self.generated += 1;
        Ok(Token {
            id: self.cursor as u32,
            text: Some(text),
            is_end_of_stream: false,
        })
    }

    fn generated_tokens(&self) -> usize {
        self.generated
    }
}

// ── MockImageGenerator ──

pub struct MockImageGenerator;

#[async_trait]
impl Generator for MockImageGenerator {
    type Shardable = MockBlock;
    const MODEL_NAME: &'static str = "mock-image";

    async fn load(_ctx: &mut Context) -> Result<Option<Box<Self>>> {
        Ok(Some(Box::new(Self)))
    }
}

#[async_trait]
impl ImageGenerator for MockImageGenerator {
    async fn generate_image<F>(
        &mut self,
        _args: &ImageGenerationArgs,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static,
    {
        let img = ImageBuffer::from_pixel(1, 1, Rgb([255u8, 0, 0]));
        callback(vec![img]);
        Ok(())
    }
}

// ── MockAudioGenerator ──

pub struct MockAudioGenerator;

#[async_trait]
impl Generator for MockAudioGenerator {
    type Shardable = MockBlock;
    const MODEL_NAME: &'static str = "mock-audio";

    async fn load(_ctx: &mut Context) -> Result<Option<Box<Self>>> {
        Ok(Some(Box::new(Self)))
    }
}

#[async_trait]
impl AudioGenerator for MockAudioGenerator {
    async fn generate_audio(&mut self, _args: &AudioGenerationArgs) -> Result<AudioOutput> {
        let samples: Vec<f32> = (0..2400)
            .map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 24000.0).sin())
            .collect();
        Ok(AudioOutput {
            samples,
            sample_rate: 24000,
        })
    }
}

// ── Helper: create Master with specific model combos ──

fn dummy_context() -> Context {
    let args = crate::Args {
        sample_len: 2048, // Args::default() gives 0, clap gives 2048
        ..Default::default()
    };
    // We need a minimal context — no files, no GPU
    Context {
        args,
        device: candle_core::Device::Cpu,
        dtype: candle_core::DType::F32,
        config: None,
        var_builder: None,
        data_path: std::path::PathBuf::from("/nonexistent"),
        topology: crate::cake::Topology::new(),
        cache: None,
        text_model_arch: crate::TextModelArch::Auto,
        quant: std::sync::Arc::new(crate::utils::NoQuantization),
        listener_override: std::sync::Mutex::new(None).into(),
    }
}

/// Master with text model loaded (image/audio return 404).
pub fn mock_master_text() -> Master<MockTextGenerator, MockImageGenerator, NoAudio> {
    Master {
        ctx: dummy_context(),
        llm_model: Some(Box::new(MockTextGenerator {
            tokens: vec!["Hello".into(), " world".into()],
            cursor: 0,
            messages: Vec::new(),
            generated: 0,
        })),
        sd_model: None,
        audio_model: None,
    }
}

/// Master with image model loaded.
pub fn mock_master_image() -> Master<MockTextGenerator, MockImageGenerator, NoAudio> {
    Master {
        ctx: dummy_context(),
        llm_model: None,
        sd_model: Some(Box::new(MockImageGenerator)),
        audio_model: None,
    }
}

/// Master with audio model loaded.
pub fn mock_master_audio() -> Master<MockTextGenerator, MockImageGenerator, MockAudioGenerator> {
    Master {
        ctx: dummy_context(),
        llm_model: None,
        sd_model: None,
        audio_model: Some(Box::new(MockAudioGenerator)),
    }
}

/// Master with no models loaded (all 404).
pub fn mock_master_none() -> Master<MockTextGenerator, MockImageGenerator, NoAudio> {
    Master {
        ctx: dummy_context(),
        llm_model: None,
        sd_model: None,
        audio_model: None,
    }
}

// ── Helper: create actix App for testing ──

pub fn test_app_text(
) -> App<
    impl actix_web::dev::ServiceFactory<
        actix_web::dev::ServiceRequest,
        Config = (),
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    type TG = MockTextGenerator;
    type IG = MockImageGenerator;
    type AG = NoAudio;

    let state = Arc::new(RwLock::new(mock_master_text()));
    App::new()
        .app_data(web::Data::new(state))
        .route(
            "/v1/chat/completions",
            web::post().to(api::text::generate_text::<TG, IG, AG>),
        )
        .route(
            "/api/v1/chat/completions",
            web::post().to(api::text::generate_text::<TG, IG, AG>),
        )
        .route(
            "/v1/models",
            web::get().to(api::list_models::<TG, IG, AG>),
        )
        .route(
            "/v1/audio/speech",
            web::post().to(api::audio::generate_speech::<TG, IG, AG>),
        )
        .route(
            "/v1/images/generations",
            web::post().to(api::image::generate_image_openai::<TG, IG, AG>),
        )
        .route(
            "/api/v1/image",
            web::post().to(api::image::generate_image::<TG, IG, AG>),
        )
}

pub fn test_app_audio(
) -> App<
    impl actix_web::dev::ServiceFactory<
        actix_web::dev::ServiceRequest,
        Config = (),
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    type TG = MockTextGenerator;
    type IG = MockImageGenerator;
    type AG = MockAudioGenerator;

    let state = Arc::new(RwLock::new(mock_master_audio()));
    App::new()
        .app_data(web::Data::new(state))
        .route(
            "/v1/chat/completions",
            web::post().to(api::text::generate_text::<TG, IG, AG>),
        )
        .route(
            "/v1/models",
            web::get().to(api::list_models::<TG, IG, AG>),
        )
        .route(
            "/v1/audio/speech",
            web::post().to(api::audio::generate_speech::<TG, IG, AG>),
        )
        .route(
            "/v1/images/generations",
            web::post().to(api::image::generate_image_openai::<TG, IG, AG>),
        )
        .route(
            "/api/v1/image",
            web::post().to(api::image::generate_image::<TG, IG, AG>),
        )
}

pub fn test_app_image(
) -> App<
    impl actix_web::dev::ServiceFactory<
        actix_web::dev::ServiceRequest,
        Config = (),
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    type TG = MockTextGenerator;
    type IG = MockImageGenerator;
    type AG = NoAudio;

    let state = Arc::new(RwLock::new(mock_master_image()));
    App::new()
        .app_data(web::Data::new(state))
        .route(
            "/v1/chat/completions",
            web::post().to(api::text::generate_text::<TG, IG, AG>),
        )
        .route(
            "/v1/models",
            web::get().to(api::list_models::<TG, IG, AG>),
        )
        .route(
            "/v1/audio/speech",
            web::post().to(api::audio::generate_speech::<TG, IG, AG>),
        )
        .route(
            "/v1/images/generations",
            web::post().to(api::image::generate_image_openai::<TG, IG, AG>),
        )
        .route(
            "/api/v1/image",
            web::post().to(api::image::generate_image::<TG, IG, AG>),
        )
}

pub fn test_app_none(
) -> App<
    impl actix_web::dev::ServiceFactory<
        actix_web::dev::ServiceRequest,
        Config = (),
        Response = actix_web::dev::ServiceResponse,
        Error = actix_web::Error,
        InitError = (),
    >,
> {
    type TG = MockTextGenerator;
    type IG = MockImageGenerator;
    type AG = NoAudio;

    let state = Arc::new(RwLock::new(mock_master_none()));
    App::new()
        .app_data(web::Data::new(state))
        .route(
            "/v1/chat/completions",
            web::post().to(api::text::generate_text::<TG, IG, AG>),
        )
        .route(
            "/v1/models",
            web::get().to(api::list_models::<TG, IG, AG>),
        )
        .route(
            "/v1/audio/speech",
            web::post().to(api::audio::generate_speech::<TG, IG, AG>),
        )
        .route(
            "/v1/images/generations",
            web::post().to(api::image::generate_image_openai::<TG, IG, AG>),
        )
        .route(
            "/api/v1/image",
            web::post().to(api::image::generate_image::<TG, IG, AG>),
        )
}
