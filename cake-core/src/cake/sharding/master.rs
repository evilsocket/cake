use std::io::Write;

use crate::models::chat::Message;
use crate::models::{AudioGenerationArgs, AudioOutput, Model};

use super::api;
use crate::cake::Context;

use crate::{ImageGenerationArgs, ModelType};
use anyhow::Result;
use image::{ImageBuffer, Rgb};

/// A master connects to, communicates with and orchestrates the workers.
pub struct Master<M: Model> {
    pub ctx: Context,
    pub model: Option<Box<M>>,
}

impl<M: Model> Master<M> {
    /// Create a new instance.
    pub async fn new(mut ctx: Context) -> Result<Self> {
        let model = M::load(&mut ctx).await?;
        Ok(Self { ctx, model })
    }

    pub async fn run(mut self) -> Result<()> {
        if self.ctx.args.api.is_some() {
            // run as REST api
            api::start(self).await?;
        } else {
            match self.ctx.args.model_type {
                ModelType::TextModel => {
                    let model = self.model.as_mut().expect("model not found");
                    model.add_message(Message::system(self.ctx.args.system_prompt.clone()))?;
                    model.add_message(Message::user(self.ctx.args.prompt.clone()))?;

                    self.generate_text(None, |data| {
                        if data.is_empty() {
                            println!();
                        } else {
                            print!("{data}")
                        }
                        std::io::stdout().flush().unwrap();
                    })
                    .await?;
                }
                ModelType::ImageModel => {
                    let image_output = self.ctx.args.image_output.clone();
                    self.generate_image(self.ctx.args.sd_img_gen_args.clone(), move |images| {
                        if let Some(image) = images.into_iter().next() {
                            if let Some(parent) = std::path::Path::new(&image_output).parent() {
                                if !parent.as_os_str().is_empty() {
                                    std::fs::create_dir_all(parent).ok();
                                }
                            }
                            image
                                .save(&image_output)
                                .expect("Error saving image to disk");
                        }
                    })
                    .await?;
                }
                ModelType::AudioModel => {
                    let args = AudioGenerationArgs {
                        input: self.ctx.args.prompt.clone(),
                        voice_data: None,
                        voice_path: self.ctx.args.voice_prompt.clone(),
                        cfg_scale: self.ctx.args.tts_cfg_scale,
                        max_frames: self.ctx.args.max_audio_frames,
                        diffusion_steps: self.ctx.args.tts_diffusion_steps,
                    };
                    let output = self.generate_audio(&args).await?;
                    let wav_bytes = output.to_wav_bytes();
                    let output_path = &self.ctx.args.audio_output;
                    std::fs::write(output_path, &wav_bytes)?;
                    log::info!(
                        "Audio saved to {} ({:.1}s, {} samples)",
                        output_path,
                        output.samples.len() as f64 / output.sample_rate as f64,
                        output.samples.len()
                    );
                }
            }
        }

        Ok(())
    }

    /// Reset the master state for a new inference.
    pub fn reset(&mut self) -> Result<()> {
        match self.model.as_mut() {
            Some(m) => m.reset(),
            None => Ok(()),
        }
    }

    /// clear worker kv cache
    pub async fn goodbye(&mut self) -> Result<()> {
        match self.model.as_mut() {
            Some(m) => m.goodbye().await,
            None => Ok(()),
        }
    }

    /// Start the generation loop and call the stream function for every token.
    /// `max_tokens` overrides the default sample length if provided.
    pub async fn generate_text<S>(
        &mut self,
        max_tokens: Option<usize>,
        mut stream: S,
    ) -> Result<()>
    where
        S: FnMut(&str),
    {
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let sample_len = max_tokens.unwrap_or(self.ctx.args.sample_len);
        log::debug!("  sample_len = {}", sample_len);

        let mut start_gen = std::time::Instant::now();
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No text model loaded"))?;

        for index in 0..sample_len {
            if index == 1 {
                start_gen = std::time::Instant::now()
            }

            let token_start = std::time::Instant::now();
            let token = model.next_token(index).await?;
            let token_elapsed = token_start.elapsed();

            log::debug!(
                "token {} generated in {:.1}ms ({:.1} tok/s)",
                index,
                token_elapsed.as_secs_f64() * 1000.0,
                1.0 / token_elapsed.as_secs_f64(),
            );

            if token.is_end_of_stream {
                break;
            } else {
                stream(&token.to_string());
                // Yield to the runtime so the SSE stream task can flush
                // this token to the client before we start the next one.
                tokio::task::yield_now().await;
            }
        }

        // signal end of stream
        stream("");

        let dt = start_gen.elapsed();
        let generated = model.generated_tokens();

        log::info!(
            "{} tokens generated ({:.2} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }

    pub async fn generate_image<F>(&mut self, args: ImageGenerationArgs, callback: F) -> Result<()>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static,
    {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No image model loaded"))?;
        model.generate_image(&args, Box::new(callback)).await
    }

    pub async fn generate_audio(&mut self, args: &AudioGenerationArgs) -> Result<AudioOutput> {
        let model = self
            .model
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("No audio model loaded"))?;
        model.generate_audio(args).await
    }
}
