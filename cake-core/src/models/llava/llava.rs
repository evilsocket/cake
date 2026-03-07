use anyhow::Result;
use async_trait::async_trait;
use candle_core::{IndexOp, Tensor};
use candle_nn::Module;
use candle_transformers::models::llava::config::LLaVAConfig as CandleLLaVAConfig;

use super::config::LlavaConfig;
use super::llava_shardable::LlavaShardable;
use super::vision::LlavaVision;
use crate::cake::{Context, Forwarder};
use crate::models::chat::Message;
use crate::models::common::text_model::TextModelBase;
use crate::models::common::Transformer;
use crate::models::{Generator, TextGenerator, Token, VisionLanguageGenerator};

const DEFAULT_EOS_TOKEN: &str = "<|eot_id|>";

/// LLaVA main model.
///
/// The LLM layers are handled by TextModelBase.
/// The vision tower is either local (LlavaVision) or remote (Client).
#[allow(dead_code)]
pub struct LLava {
    base: TextModelBase,
    history: Vec<Message>,

    /// Vision encoder (local or remote).
    vision_encoder: Box<dyn Forwarder>,
    /// Candle LLaVA config for image processing helpers.
    candle_config: CandleLLaVAConfig,
    /// Pending image embeddings to merge on next forward pass.
    pending_image_embeddings: Option<Tensor>,
    /// Image newline tensor (for spatial_unpad merge).
    image_newline: Option<Tensor>,
}

#[async_trait]
impl Generator for LLava {
    type Shardable = LlavaShardable;
    const MODEL_NAME: &'static str = "llava";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let config_path = ctx.data_path.join("config.json");
        let llava_config = LlavaConfig::from_path(&config_path)?;
        let candle_config = llava_config.to_candle_llava_config();

        // Load vision encoder
        log::info!("loading vision encoder ...");
        let vision_encoder: Box<dyn Forwarder> =
            if let Some((_node_name, node)) = ctx.topology.get_node_for_layer("llava-vision") {
                log::info!("vision encoder will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        ctx.device.clone(),
                        &node.host,
                        "llava-vision",
                        ctx.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                log::info!("vision encoder will be served locally");
                LlavaVision::load_model(ctx)?
            };
        log::info!("vision encoder ready");

        // Load image_newline tensor if available
        let vb = ctx.var_builder.as_ref().expect("No var_builder specified");
        let hidden_size = llava_config.effective_hidden_size();
        let image_newline = if candle_config.hf {
            vb.get(&[hidden_size], "image_newline").ok()
        } else {
            vb.get(&[hidden_size], "model.image_newline").ok()
        };

        // Load LLM layers via TextModelBase
        let base = TextModelBase::load::<Transformer>(ctx, DEFAULT_EOS_TOKEN).await?;

        Ok(Some(Box::new(Self {
            base,
            history: Vec::new(),
            vision_encoder,
            candle_config,
            pending_image_embeddings: None,
            image_newline,
        })))
    }
}

impl LLava {
    /// Encode the dialog to LLaMA-style prompt format.
    fn encode_dialog_to_prompt(&self) -> String {
        let mut encoded = "<|begin_of_text|>".to_string();
        for message in &self.history {
            encoded += &format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                message.role,
                message.content.trim()
            );
        }
        encoded += "<|start_header_id|>assistant<|end_header_id|>\n\n";
        encoded
    }

    /// Merge visual embeddings with text embeddings at <image> token positions.
    fn merge_visual_embeddings(
        &self,
        text_embeddings: &Tensor,
        image_embeddings: &Tensor,
        input_ids: &[u32],
    ) -> Result<Tensor> {
        let image_token_index = self.candle_config.image_token_index as i64;

        // Find image token positions
        let image_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id as i64 == image_token_index)
            .map(|(i, _)| i)
            .collect();

        if image_positions.is_empty() {
            return Ok(text_embeddings.clone());
        }

        // Build the merged embedding sequence
        let mut segments: Vec<Tensor> = Vec::new();
        let mut prev_pos = 0;

        for &img_pos in &image_positions {
            // Text tokens before this image token
            if img_pos > prev_pos {
                segments.push(text_embeddings.i((0, prev_pos..img_pos, ..))?.squeeze(0)?);
            }
            // Image embeddings replace the image token
            let img_emb = if image_embeddings.dims().len() == 3 {
                image_embeddings.i(0)?.clone()
            } else {
                image_embeddings.clone()
            };
            segments.push(img_emb);
            prev_pos = img_pos + 1;
        }

        // Remaining text tokens after last image token
        let seq_len = text_embeddings.dim(1)?;
        if prev_pos < seq_len {
            segments.push(text_embeddings.i((0, prev_pos..seq_len, ..))?.squeeze(0)?);
        }

        let merged = Tensor::cat(&segments, 0)?.unsqueeze(0)?;
        Ok(merged)
    }

    /// Forward pass that handles visual token merging when image embeddings are pending.
    async fn forward_with_images(
        &mut self,
        input: &Tensor,
        index_pos: usize,
    ) -> Result<Tensor> {
        let input_ids: Vec<u32> = input.squeeze(0)?.to_vec1()?;

        // Embed text tokens
        let text_embeddings = self.base.embedding.forward(input)?;

        // Merge image embeddings if pending
        let input_embeds = if let Some(ref image_embeddings) = self.pending_image_embeddings {
            self.merge_visual_embeddings(&text_embeddings, image_embeddings, &input_ids)?
        } else {
            text_embeddings
        };

        // Clear pending images after merging
        self.pending_image_embeddings = None;

        // Forward through transformer blocks (skip embedding in base.forward)
        let forward_start = std::time::Instant::now();
        let (_batch_size, seq_len) = input_embeds.dims2().unwrap_or((1, input_embeds.dim(1)?));

        let mut x = input_embeds;
        let num_blocks = self.base.blocks.len();
        let mut block_idx = 0;

        while block_idx < num_blocks {
            let curr_block_id = self.base.blocks[block_idx].ident().to_owned();
            if curr_block_id == "local" {
                x = self.base.blocks[block_idx]
                    .forward_mut(&x, index_pos, block_idx, &mut self.base.ctx)
                    .await?;
                block_idx += 1;
            } else {
                let mut batch = vec![];
                let first = block_idx;
                while block_idx < num_blocks
                    && self.base.blocks[block_idx].ident() == curr_block_id
                {
                    batch.push((
                        self.base.blocks[block_idx].layer_name().to_string(),
                        index_pos,
                        block_idx,
                    ));
                    block_idx += 1;
                }
                x = self.base.blocks[first]
                    .forward_batch(&x, batch, &mut self.base.ctx)
                    .await?;
            }
        }

        let x = self.base.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.base.lm_head.forward(&x)?;

        let total_elapsed = forward_start.elapsed();
        log::debug!(
            "  llava forward total={:.1}ms",
            total_elapsed.as_secs_f64() * 1000.0,
        );

        Ok(logits)
    }
}

#[async_trait]
impl TextGenerator for LLava {
    fn add_message(&mut self, message: Message) -> Result<()> {
        self.history.push(message);
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.history.clear();
        self.base.reset();
        self.pending_image_embeddings = None;
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        self.base.goodbye().await
    }

    async fn next_token(&mut self, index: usize) -> Result<Token> {
        if self.base.generated == 0 {
            let dialog = self.encode_dialog_to_prompt();
            self.base.prepare_prompt(&dialog)?;
        }

        // If there are pending image embeddings on the first token, use the image-aware forward
        if index == 0 && self.pending_image_embeddings.is_some() {
            let num_tokens = self.base.tokens.len();
            let context_tokens = &self.base.tokens[..];
            let input = Tensor::new(context_tokens, &self.base.ctx.device)?.unsqueeze(0)?;

            let logits = self.forward_with_images(&input, 0).await?;
            let logits = logits.squeeze(0)?;

            self.base.index_pos += num_tokens;
            let next_token = self.base.logits_processor.sample(&logits)?;
            self.base.generated += 1;
            self.base.tokens.push(next_token);

            let is_end_of_stream = self
                .base
                .eos_token_id
                .as_ref()
                .map_or(false, |eos| eos.is_eos(next_token));

            let text = match self.base.tokenizer.decode(&[next_token], false) {
                Ok(s) => Some(s),
                Err(e) => {
                    log::error!("could not decode token {next_token}: {e}");
                    None
                }
            };

            return Ok(Token {
                id: next_token,
                text,
                is_end_of_stream,
            });
        }

        // Normal text-only generation (after first token or no images)
        self.base.next_token(index).await
    }

    fn generated_tokens(&self) -> usize {
        self.base.generated
    }
}

#[async_trait]
impl VisionLanguageGenerator for LLava {
    async fn encode_image(&mut self, image: &Tensor) -> Result<Tensor> {
        self.vision_encoder
            .forward_mut(image, 0, 0, &mut self.base.ctx)
            .await
    }

    fn add_image(&mut self, image_embeddings: Tensor) -> Result<()> {
        self.pending_image_embeddings = Some(image_embeddings);
        Ok(())
    }
}
