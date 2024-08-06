use std::io::Write;

use crate::models::{chat::Message, ImageGenerator, TextGenerator};

use super::{api, Context};

use anyhow::Result;
use image::ImageBuffer;
use crate::{ImageGenerationArgs, ModelType};

/// A master connects to, communicates with and orchestrates the workers.
pub struct Master<TG, IG> {
    pub ctx: Context,
    pub llm_model: Option<Box<TG>>,
    pub sd_model: Option<Box<IG>>
}

impl<TG: TextGenerator + Send + Sync + 'static, IG: ImageGenerator + Send + Sync + 'static> Master<TG, IG> {
    /// Create a new instance.
    pub async fn new(ctx: Context) -> Result<Self> {

        let llm_model = TG::load(ctx.clone()).await?;
        let sd_model = IG::load(ctx.clone()).await?;
        Ok(Self { ctx, llm_model, sd_model })
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
                self.generate_text(|data| {
                    if data.is_empty() {
                        println!();
                    } else {
                        print!("{data}")
                    }
                    std::io::stdout().flush().unwrap();
                })
                .await?;
            } else {
                let images = self.generate_image(self.ctx.args.sd_img_gen_args.clone()).await?;

                let mut num = 0;
                for image in images {
                    image.save(format!("image_{}.png", num)).expect("Error saving image to disk");
                    num+=1;
                }
            }
        }

        Ok(())
    }

    /// Reset the master state for a new inference.
    pub fn reset(&mut self) -> Result<()> {
        self.llm_model.as_mut().expect("LLM model not found").reset()
    }

    /// Start the generation loop and call the stream function for every token.
    pub async fn generate_text<S>(&mut self, mut stream: S) -> Result<()>
    where
        S: FnMut(&str),
    {
        log::info!(
            "starting the inference loop (mem={})\n\n",
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        log::debug!("  ctx.args.sample_len = {}", self.ctx.args.sample_len);

        stream(&self.ctx.args.prompt);

        let mut start_gen = std::time::Instant::now();
        let llm_model = self.llm_model.as_mut().expect("LLM model not found");

        for index in 0..self.ctx.args.sample_len {
            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }

            let token = llm_model.next_token(index).await?;
            if token.is_end_of_stream {
                break;
            } else {
                stream(&token.to_string());
            }
        }

        // signal end of stream
        stream("");

        let dt = start_gen.elapsed();
        let generated = llm_model.generated_tokens();

        log::info!(
            "{} tokens generated ({} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }

    pub async fn generate_image(&mut self, args: ImageGenerationArgs) -> Result<Vec<ImageBuffer<image::Rgb<u8>, Vec<u8>>>> {
        let sd_model = self.sd_model.as_mut().expect("SD model not found");
        Ok(sd_model.generate_image(&args).await?)
    }
}
