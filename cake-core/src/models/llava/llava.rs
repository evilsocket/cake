//! LLaVA (Large Language and Vision Assistant) main model.
//!
//! Implements a vision-language model that combines a CLIP vision tower
//! with a Llama-based language model, connected via a multi-modal projector.

use anyhow::Result;
use async_trait::async_trait;
use candle_core::{IndexOp, Module, Tensor};
use candle_nn::{linear, Activation, Linear, VarBuilder};

use super::config::LlavaConfig;
use super::llava_shardable::LlavaShardable;
use super::vision::LlavaVision;
use crate::cake::{Context, Forwarder};
use crate::models::chat::Message;
use crate::models::common::chatml_history::ChatMLHistory;
use crate::models::common::text_model::TextModelBase;
use crate::models::{Generator, TextGenerator, Token};

/// Default end of stream token for LLaVA (Llama-based).
const DEFAULT_EOS_TOKEN: &str = "<|eot_id|>";

/// MM projector: two-layer MLP with GELU activation.
/// Projects vision features from CLIP hidden size to LLM hidden size.
#[derive(Debug)]
struct MMProjector {
    linear_1: Linear,
    linear_2: Linear,
}

impl MMProjector {
    fn load(vb: VarBuilder, mm_hidden_size: usize, llm_hidden_size: usize, hf_format: bool) -> Result<Self> {
        // HF format: multi_modal_projector.linear_1 / .linear_2
        // Original format: model.mm_projector.0 / .2
        let (prefix_1, prefix_2) = if hf_format {
            ("linear_1", "linear_2")
        } else {
            ("0", "2")
        };
        let linear_1 = linear(mm_hidden_size, llm_hidden_size, vb.pp(prefix_1))?;
        let linear_2 = linear(llm_hidden_size, llm_hidden_size, vb.pp(prefix_2))?;
        Ok(Self { linear_1, linear_2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(x)?;
        let x = Activation::Gelu.forward(&x)?;
        let x = self.linear_2.forward(&x)?;
        Ok(x)
    }
}

/// LLaVA main class.
pub struct LLava {
    base: TextModelBase,
    history: ChatMLHistory,
    // Vision components
    vision_tower: Box<dyn Forwarder>,
    mm_projector: MMProjector,
    image_newline: Option<Tensor>,
    // Pending image data: pixel values tensors (batch, 3, H, W)
    pending_images: Vec<Tensor>,
    // LLaVA-specific config
    image_token_index: u32,
    mm_vision_select_feature: String,
}

#[async_trait]
impl Generator for LLava {
    type Shardable = LlavaShardable;
    const MODEL_NAME: &'static str = "llava";

    async fn load(ctx: &mut Context) -> Result<Option<Box<Self>>> {
        let config_path = ctx.data_path.join("config.json");
        let llava_config = LlavaConfig::from_path(&config_path)?;

        // Determine vision hidden size and LLM hidden size
        let mm_hidden_size = if let Some(ref vc) = llava_config.vision_config {
            vc.hidden_size
        } else {
            llava_config.mm_hidden_size.unwrap_or(1024)
        };

        let llm_hidden_size = if let Some(ref tc) = llava_config.text_config {
            tc.hidden_size
        } else {
            llava_config.hidden_size.unwrap_or(4096)
        };

        let image_token_index = llava_config.image_token_index;
        let mm_vision_select_feature = llava_config.mm_vision_select_feature.clone();
        let is_hf_format = llava_config.text_config.is_some();

        // ctx.config and ctx.cache are already set by Context::from_args()

        // Load vision tower: check topology for remote "llava-vision", else load locally
        log::info!("loading LLaVA vision tower ...");
        let vision_tower: Box<dyn Forwarder> =
            if let Some((_name, node)) = ctx.topology.get_node_for_layer("llava-vision") {
                // Remote vision tower
                let remote_layers = vec![(0usize, "llava-vision".to_string(), node.host.clone())];
                let connected = crate::cake::client::connect_remote_layers(
                    &remote_layers,
                    &ctx.device,
                    ctx.args.cluster_key.as_deref(),
                )
                .await?;
                connected.into_iter().next().unwrap().1
            } else {
                // Local vision tower
                LlavaVision::load("llava-vision".to_string(), ctx)?
            };

        // Load MM projector
        log::info!("loading LLaVA multi-modal projector ...");
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified");

        // Try HF format prefix first, then original format
        let proj_vb = if is_hf_format {
            vb.pp("multi_modal_projector")
        } else {
            vb.pp("model.mm_projector")
        };
        let mm_projector = MMProjector::load(proj_vb, mm_hidden_size, llm_hidden_size, is_hf_format)?;

        // Load image_newline tensor (optional, present in some LLaVA versions)
        // HF format: "image_newline", original format: "model.image_newline"
        let image_newline = if is_hf_format {
            vb.get(llm_hidden_size, "image_newline").ok()
        } else {
            vb.get(llm_hidden_size, "model.image_newline").ok()
        };
        if image_newline.is_some() {
            log::info!("loaded image_newline embedding");
        }

        // Load LLM backbone
        let base = TextModelBase::load::<crate::models::common::Transformer>(ctx, DEFAULT_EOS_TOKEN)
            .await?;

        // Auto-detect chat template: LLaVA-1.5 uses ChatML (<|im_start|>),
        // LLaVA with Llama-3 uses Llama-3 format. Default to ChatML.
        let history = ChatMLHistory::new();

        Ok(Some(Box::new(Self {
            base,
            history,
            vision_tower,
            mm_projector,
            image_newline,
            pending_images: vec![],
            image_token_index,
            mm_vision_select_feature,
        })))
    }
}

impl LLava {
    /// Add an image to be processed with the next generation call.
    /// `pixel_values` should be a tensor of shape (1, 3, H, W) with normalized pixel values.
    pub fn add_image(&mut self, pixel_values: Tensor) {
        self.pending_images.push(pixel_values);
    }

    /// Process pending images through the vision tower and MM projector,
    /// then merge visual embeddings with text embeddings at `<image>` token positions.
    async fn prepare_multimodal_input(
        &mut self,
        text_tokens: &[u32],
    ) -> Result<Tensor> {
        // Get text embeddings from the embedding table
        let device = &self.base.ctx.device;
        let token_tensor = Tensor::new(text_tokens, device)?.unsqueeze(0)?;
        let text_embeds = self.base.embedding.forward(&token_tensor)?;
        // Apply embedding scale if configured
        let text_embeds = if let Some(scale) = self.base.ctx.config.as_ref().and_then(|c| c.embed_scale) {
            (text_embeds * scale as f64)?
        } else {
            text_embeds
        };

        if self.pending_images.is_empty() {
            return Ok(text_embeds);
        }

        // Process each image through vision tower + MM projector
        let images: Vec<Tensor> = self.pending_images.drain(..).collect();
        let mut image_features = Vec::new();
        for pixel_values in &images {
            let vision_output = self
                .vision_tower
                .forward(pixel_values, 0, 0, &mut self.base.ctx)
                .await?;

            // Project vision features to LLM space
            let projected = self.mm_projector.forward(&vision_output)?;
            image_features.push(projected);
        }

        // Find <image> token positions and replace with visual embeddings
        let mut image_idx = 0;
        let mut segments: Vec<Tensor> = Vec::new();
        let mut current_start = 0;

        for (i, &token_id) in text_tokens.iter().enumerate() {
            if token_id == self.image_token_index {
                // Add text segment before this image token
                if i > current_start {
                    let text_seg = text_embeds.narrow(1, current_start, i - current_start)?;
                    segments.push(text_seg);
                }

                // Add image features
                if image_idx < image_features.len() {
                    // image_features[image_idx] has shape (1, num_patches, hidden_size)
                    let img_feat = &image_features[image_idx];

                    // Optionally append image_newline
                    if let Some(ref newline) = self.image_newline {
                        let num_patches = img_feat.dims()[1];
                        let hidden_size = img_feat.dims()[2];
                        let newline = newline
                            .reshape((1, 1, hidden_size))?
                            .broadcast_as((1, 1, hidden_size))?;
                        let combined = Tensor::cat(&[img_feat, &newline], 1)?;
                        segments.push(combined);
                    } else {
                        segments.push(img_feat.clone());
                    }
                    image_idx += 1;
                }

                current_start = i + 1;
            }
        }

        // Add remaining text after last image token
        if current_start < text_tokens.len() {
            let text_seg =
                text_embeds.narrow(1, current_start, text_tokens.len() - current_start)?;
            segments.push(text_seg);
        }

        if segments.is_empty() {
            // No image tokens found, just return text embeddings
            return Ok(text_embeds);
        }

        // Concatenate all segments along the sequence dimension
        let combined = Tensor::cat(&segments, 1)?;
        Ok(combined)
    }

    /// Forward pass with pre-computed input embeddings (bypasses embedding lookup).
    async fn forward_with_embeds(&mut self, input_embeds: &Tensor, idx: usize) -> Result<Tensor> {
        self.base.forward_input_embed(input_embeds.clone(), idx).await
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
        self.pending_images.clear();
        self.base.reset();
        Ok(())
    }

    async fn goodbye(&mut self) -> Result<()> {
        self.base.goodbye().await
    }

    async fn next_token(&mut self, index: usize) -> Result<Token> {
        // On first token, encode dialog and handle multimodal input
        if self.base.generated == 0 {
            let dialog = self.history.encode_dialog_to_prompt();
            self.base.prepare_prompt(&dialog)?;

            // If we have pending images, do the multimodal forward
            if !self.pending_images.is_empty() {
                let tokens = self.base.tokens.clone();
                let input_embeds = self.prepare_multimodal_input(&tokens).await?;

                let logits = self.forward_with_embeds(&input_embeds, 0).await?;

                // Sample from logits
                let logits = logits.squeeze(0)?;
                let next_token = self.base.logits_processor.sample(&logits)?;

                // Update state: the combined sequence length is different from token count
                let combined_seq_len = input_embeds.dims()[1];
                self.base.index_pos = combined_seq_len;
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
        }

        // Normal text-only forward through TextModelBase
        self.base.next_token(index).await
    }

    fn generated_tokens(&self) -> usize {
        self.base.generated
    }
}

#[cfg(test)]
mod tests {
    use candle_core::{Device, Tensor};

    /// Test the image token replacement algorithm in isolation.
    /// This mirrors the logic in `prepare_multimodal_input` but operates on
    /// raw tensors without needing model weights.
    fn merge_text_and_image_embeds(
        text_embeds: &Tensor,      // (1, seq_len, hidden)
        text_tokens: &[u32],
        image_features: &[Tensor], // each (1, num_patches, hidden)
        image_token_id: u32,
    ) -> candle_core::Result<Tensor> {
        let mut image_idx = 0;
        let mut segments: Vec<Tensor> = Vec::new();
        let mut current_start = 0;

        for (i, &token_id) in text_tokens.iter().enumerate() {
            if token_id == image_token_id {
                if i > current_start {
                    let text_seg = text_embeds.narrow(1, current_start, i - current_start)?;
                    segments.push(text_seg);
                }
                if image_idx < image_features.len() {
                    segments.push(image_features[image_idx].clone());
                    image_idx += 1;
                }
                current_start = i + 1;
            }
        }

        if current_start < text_tokens.len() {
            let text_seg =
                text_embeds.narrow(1, current_start, text_tokens.len() - current_start)?;
            segments.push(text_seg);
        }

        if segments.is_empty() {
            return Ok(text_embeds.clone());
        }

        Tensor::cat(&segments, 1)
    }

    #[test]
    fn test_merge_no_images() {
        let dev = &Device::Cpu;
        let text_embeds = Tensor::zeros((1, 5, 8), candle_core::DType::F32, dev).unwrap();
        let tokens = vec![1, 2, 3, 4, 5];
        let result = merge_text_and_image_embeds(&text_embeds, &tokens, &[], 32000).unwrap();
        // No image tokens → output == input
        assert_eq!(result.dims(), &[1, 5, 8]);
    }

    #[test]
    fn test_merge_single_image_middle() {
        let dev = &Device::Cpu;
        // 5 text tokens with image placeholder at position 2
        let text_embeds = Tensor::ones((1, 5, 8), candle_core::DType::F32, dev).unwrap();
        let tokens = vec![1, 2, 32000, 4, 5]; // image token at idx 2
        let img_feat = Tensor::zeros((1, 3, 8), candle_core::DType::F32, dev).unwrap(); // 3 patches

        let result =
            merge_text_and_image_embeds(&text_embeds, &tokens, &[img_feat], 32000).unwrap();
        // 2 text tokens before + 3 image patches + 2 text tokens after = 7
        assert_eq!(result.dims(), &[1, 7, 8]);
    }

    #[test]
    fn test_merge_image_at_start() {
        let dev = &Device::Cpu;
        let text_embeds = Tensor::ones((1, 3, 4), candle_core::DType::F32, dev).unwrap();
        let tokens = vec![32000, 2, 3]; // image at start
        let img_feat = Tensor::zeros((1, 5, 4), candle_core::DType::F32, dev).unwrap();

        let result =
            merge_text_and_image_embeds(&text_embeds, &tokens, &[img_feat], 32000).unwrap();
        // 0 text before + 5 patches + 2 text after = 7
        assert_eq!(result.dims(), &[1, 7, 4]);
    }

    #[test]
    fn test_merge_image_at_end() {
        let dev = &Device::Cpu;
        let text_embeds = Tensor::ones((1, 3, 4), candle_core::DType::F32, dev).unwrap();
        let tokens = vec![1, 2, 32000]; // image at end
        let img_feat = Tensor::zeros((1, 5, 4), candle_core::DType::F32, dev).unwrap();

        let result =
            merge_text_and_image_embeds(&text_embeds, &tokens, &[img_feat], 32000).unwrap();
        // 2 text before + 5 patches + 0 text after = 7
        assert_eq!(result.dims(), &[1, 7, 4]);
    }

    #[test]
    fn test_merge_two_images() {
        let dev = &Device::Cpu;
        let text_embeds = Tensor::ones((1, 7, 4), candle_core::DType::F32, dev).unwrap();
        let tokens = vec![1, 32000, 3, 4, 32000, 6, 7]; // two images
        let img1 = Tensor::zeros((1, 2, 4), candle_core::DType::F32, dev).unwrap();
        let img2 = Tensor::zeros((1, 3, 4), candle_core::DType::F32, dev).unwrap();

        let result =
            merge_text_and_image_embeds(&text_embeds, &tokens, &[img1, img2], 32000).unwrap();
        // 1 text + 2 patches + 2 text + 3 patches + 2 text = 10
        assert_eq!(result.dims(), &[1, 10, 4]);
    }
}
