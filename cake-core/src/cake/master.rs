use std::io::{Cursor, Write};

use crate::models::{chat::Message, Generator, ImageGenerator, TextGenerator};

use super::{api, Context};

use anyhow::Result;
use image::{DynamicImage, ImageReader};

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
            if self.ctx.args.model_type == "text" {
                self.llm_model
                .add_message(Message::system(self.ctx.args.system_prompt.clone()))?;
            self.llm_model
                .add_message(Message::user(self.ctx.args.prompt.clone()))?;

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
                self.generate_image().await?;
            }
        }

        Ok(())
    }

    /// Reset the master state for a new inference.
    pub fn reset(&mut self) -> Result<()> {
        self.llm_model.reset()
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

        for index in 0..self.ctx.args.sample_len {
            if index == 1 {
                // record start time again since the first token is the warmup
                start_gen = std::time::Instant::now()
            }

            let token = self.llm_model.next_token(index).await?;
            if token.is_end_of_stream {
                break;
            } else {
                stream(&token.to_string());
            }
        }

        // signal end of stream
        stream("");

        let dt = start_gen.elapsed();
        let generated = self.llm_model.generated_tokens();

        log::info!(
            "{} tokens generated ({} token/s) - mem={}",
            generated,
            (generated - 1) as f64 / dt.as_secs_f64(),
            human_bytes::human_bytes(memory_stats::memory_stats().unwrap().physical_mem as f64)
        );

        Ok(())
    }

    pub async fn generate_image(&mut self) -> Result<(DynamicImage)> {
        let bytes: Vec<u8> = Vec::new();
        Ok(ImageReader::new(Cursor::new(bytes)).with_guessed_format()?.decode()?)
    }
}
