use std::io::Write;

use crate::models::{chat::Message, ImageGenerator, TextGenerator, VideoGenerator};
use crate::video::VideoOutput;

use super::{api, Context};

use crate::{ImageGenerationArgs, ModelType};
use anyhow::Result;
use image::{ImageBuffer, Rgb};

/// A master connects to, communicates with and orchestrates the workers.
pub struct Master<TG, IG> {
    pub ctx: Context,
    pub llm_model: Option<Box<TG>>,
    pub sd_model: Option<Box<IG>>,
}

impl<TG: TextGenerator + Send + Sync + 'static, IG: ImageGenerator + Send + Sync + 'static>
    Master<TG, IG>
{
    /// Create a new instance.
    pub async fn new(mut ctx: Context) -> Result<Self> {
        match ctx.args.model_type {
            ModelType::ImageModel => {
                let sd_model = IG::load(&mut ctx).await?;
                Ok(Self {
                    ctx,
                    sd_model,
                    llm_model: None,
                })
            }
            ModelType::TextModel => {
                let llm_model = TG::load(&mut ctx).await?;
                Ok(Self {
                    ctx,
                    llm_model,
                    sd_model: None,
                })
            }
        }
    }

    pub async fn run(mut self) -> Result<()> {
        if self.ctx.args.api.is_some() {
            // run as REST api
            api::start(self).await?;
        } else {
            // if running in cli mode, pre add system and user prompts
            if self.ctx.args.model_type == ModelType::TextModel {
                let llm_model = self.llm_model.as_mut().expect("LLM model not found");
                llm_model.add_message(Message::system(self.ctx.args.system_prompt.clone()))?;
                llm_model.add_message(Message::user(self.ctx.args.prompt.clone()))?;

                // just run one generation to stdout
                self.generate_text(None, |data| {
                    if data.is_empty() {
                        println!();
                    } else {
                        print!("{data}")
                    }
                    std::io::stdout().flush().unwrap();
                })
                .await?;
            } else {
                let mut step_num = 0;

                self.generate_image(self.ctx.args.sd_img_gen_args.clone(), move |images| {
                    let mut batched_num = 0;
                    for image in images {
                        image
                            .save(format!("images/image_{}_{}.png", batched_num, step_num))
                            .expect("Error saving image to disk");
                        batched_num += 1;
                    }
                    step_num += 1;
                })
                .await?;
            }
        }

        Ok(())
    }

    /// Reset the master state for a new inference.
    pub fn reset(&mut self) -> Result<()> {
        self.llm_model
            .as_mut()
            .expect("LLM model not found")
            .reset()
    }

    /// clear worker kv cache
    pub async fn goodbye(&mut self) -> Result<()> {
        self.llm_model
            .as_mut()
            .expect("LLM model not found")
            .goodbye()
            .await
    }

    /// Start the generation loop and call the stream function for every token.
    /// `max_tokens` overrides the default sample length if provided.
    pub async fn generate_text<S>(&mut self, max_tokens: Option<usize>, mut stream: S) -> Result<()>
    where
        S: FnMut(&str),
    {
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        let sample_len = max_tokens.unwrap_or(self.ctx.args.sample_len);
        log::debug!("  sample_len = {}", sample_len);

        // stream(&self.ctx.args.prompt);

        let mut start_gen = std::time::Instant::now();
        let llm_model = self.llm_model.as_mut().expect("LLM model not found");

        let mut index = 0;
        loop {
            if llm_model.generated_tokens() >= sample_len {
                break;
            }

            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }

            let token_start = std::time::Instant::now();
            let token = llm_model.next_token(index).await?;
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
            }

            index += 1;
        }

        // signal end of stream
        stream("");

        let dt = start_gen.elapsed();
        let generated = llm_model.generated_tokens();

        log::info!(
            "{} tokens generated ({:.2} token/s) - mem={}",
            generated,
            (generated.saturating_sub(1)) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }

    pub async fn generate_image<F>(&mut self, args: ImageGenerationArgs, callback: F) -> Result<()>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static,
    {
        let sd_model = self.sd_model.as_mut().expect("SD model not found");
        sd_model.generate_image(&args, callback).await
    }
}

/// Video-capable master variant. Separate impl block because VideoGenerator
/// is a different trait from ImageGenerator.
pub struct VideoMaster<TG, VG> {
    pub ctx: Context,
    pub llm_model: Option<Box<TG>>,
    pub video_model: Option<Box<VG>>,
}

impl<TG: TextGenerator + Send + Sync + 'static, VG: VideoGenerator + Send + Sync + 'static>
    VideoMaster<TG, VG>
{
    pub async fn new(mut ctx: Context) -> Result<Self> {
        match ctx.args.model_type {
            ModelType::ImageModel => {
                let video_model = VG::load(&mut ctx).await?;
                Ok(Self {
                    ctx,
                    video_model,
                    llm_model: None,
                })
            }
            ModelType::TextModel => {
                anyhow::bail!("VideoMaster cannot be used for text models");
            }
        }
    }

    pub async fn run(mut self) -> Result<()> {
        if self.ctx.args.api.is_some() {
            api::start_video(self).await?;
        } else {
            std::fs::create_dir_all("videos")?;
            let video = self.generate_video(self.ctx.args.sd_img_gen_args.clone()).await?;

            // Save as AVI
            let avi_path = std::path::PathBuf::from("videos/output.avi");
            video.save_avi(&avi_path)?;
            log::info!(
                "Saved video: {} frames, {:.1}s @ {} fps -> {}",
                video.num_frames(),
                video.duration_secs(),
                video.fps,
                avi_path.display()
            );

            // Also save individual frames for convenience
            video.save_frames(std::path::Path::new("videos/frames"), "frame")?;
            log::info!("Saved {} individual frames to videos/frames/", video.num_frames());
        }

        Ok(())
    }

    pub async fn generate_video(&mut self, args: ImageGenerationArgs) -> Result<VideoOutput> {
        let video_model = self.video_model.as_mut().expect("Video model not found");
        video_model.generate_video(&args).await
    }
}
