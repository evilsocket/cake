//! CLIP Vision Tower wrapper as a distributable Forwarder component.
//!
//! Loads a CLIP ViT model from safetensors and runs it as a Forwarder,
//! selecting the hidden state from a specific layer and stripping the CLS token.

use crate::cake::{Context, Forwarder};
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, linear, Conv2d, Conv2dConfig, LayerNorm, Linear, VarBuilder};

/// CLIP Vision Transformer encoder layer.
#[derive(Debug, Clone)]
struct ClipEncoderLayer {
    self_attn_q_proj: Linear,
    self_attn_k_proj: Linear,
    self_attn_v_proj: Linear,
    self_attn_out_proj: Linear,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    mlp_fc1: Linear,
    mlp_fc2: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl ClipEncoderLayer {
    fn load(vb: VarBuilder, hidden_size: usize, intermediate_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;
        let attn = vb.pp("self_attn");
        let self_attn_q_proj = linear(hidden_size, hidden_size, attn.pp("q_proj"))?;
        let self_attn_k_proj = linear(hidden_size, hidden_size, attn.pp("k_proj"))?;
        let self_attn_v_proj = linear(hidden_size, hidden_size, attn.pp("v_proj"))?;
        let self_attn_out_proj = linear(hidden_size, hidden_size, attn.pp("out_proj"))?;
        let layer_norm1 = layer_norm(hidden_size, 1e-5, vb.pp("layer_norm1"))?;
        let layer_norm2 = layer_norm(hidden_size, 1e-5, vb.pp("layer_norm2"))?;
        let mlp = vb.pp("mlp");
        let mlp_fc1 = linear(hidden_size, intermediate_size, mlp.pp("fc1"))?;
        let mlp_fc2 = linear(intermediate_size, hidden_size, mlp.pp("fc2"))?;
        Ok(Self {
            self_attn_q_proj,
            self_attn_k_proj,
            self_attn_v_proj,
            self_attn_out_proj,
            layer_norm1,
            layer_norm2,
            mlp_fc1,
            mlp_fc2,
            num_heads,
            head_dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.layer_norm1.forward(x)?;

        // Self attention
        let (b, seq_len, _) = x.dims3()?;
        let q = self.self_attn_q_proj.forward(&x)?;
        let k = self.self_attn_k_proj.forward(&x)?;
        let v = self.self_attn_v_proj.forward(&x)?;

        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, seq_len, self.num_heads, self.head_dim))?.transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b, seq_len, ()))?;
        let attn_output = self.self_attn_out_proj.forward(&attn_output)?;

        let x = (attn_output + residual)?;
        let residual = &x;
        let x = self.layer_norm2.forward(&x)?;
        // MLP with quick_gelu activation: x * sigmoid(1.702 * x)
        let x = self.mlp_fc1.forward(&x)?;
        let x = (&x * candle_nn::ops::sigmoid(&(&x * 1.702f64)?)?)?;
        let x = self.mlp_fc2.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

/// CLIP Vision Transformer for LLaVA.
#[derive(Debug)]
pub struct LlavaVision {
    name: String,
    embeddings_patch_embedding: Conv2d,
    embeddings_position_embedding: Tensor,
    embeddings_class_embedding: Tensor,
    pre_layrnorm: LayerNorm,
    encoder_layers: Vec<ClipEncoderLayer>,
    select_layer: isize,
    hidden_size: usize,
}

impl std::fmt::Display for LlavaVision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

#[async_trait]
impl Forwarder for LlavaVision {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>>
    where
        Self: Sized,
    {
        // Read vision config from the LlavaConfig (re-parse config.json)
        let config_path = ctx.data_path.join("config.json");
        let data = std::fs::read(&config_path)
            .map_err(|e| anyhow!("can't read {}: {:?}", config_path.display(), e))?;
        let raw: serde_json::Value = serde_json::from_slice(&data)?;

        // Determine VarBuilder prefix based on format:
        // HF format:       vision_tower.vision_model
        // Original format: model.vision_tower.vision_tower.vision_model
        let is_hf_format = raw.get("text_config").is_some();
        let vb_prefix = if is_hf_format {
            "vision_tower.vision_model"
        } else {
            "model.vision_tower.vision_tower.vision_model"
        };
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(vb_prefix);

        // Try HF nested format first, then fall back to defaults
        let (hidden_size, intermediate_size, num_heads, num_layers, image_size, patch_size) =
            if let Some(vc) = raw.get("vision_config") {
                (
                    vc["hidden_size"].as_u64().unwrap_or(1024) as usize,
                    vc["intermediate_size"].as_u64().unwrap_or(4096) as usize,
                    vc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
                    vc["num_hidden_layers"].as_u64().unwrap_or(24) as usize,
                    vc["image_size"].as_u64().unwrap_or(336) as usize,
                    vc["patch_size"].as_u64().unwrap_or(14) as usize,
                )
            } else {
                // Original LLaVA: vision config comes from CLIP model itself
                // Use sensible defaults for CLIP-ViT-L/14-336
                (1024, 4096, 16, 24, 336, 14)
            };

        let select_layer = raw
            .get("mm_vision_select_layer")
            .and_then(|v| v.as_i64())
            .unwrap_or(-2) as isize;

        log::info!(
            "loading CLIP vision tower: hidden={hidden_size}, heads={num_heads}, \
             layers={num_layers}, image={image_size}, patch={patch_size}, select_layer={select_layer}"
        );

        let emb_vb = vb.pp("embeddings");

        // Patch embedding conv2d
        let conv_cfg = Conv2dConfig {
            stride: patch_size,
            ..Default::default()
        };
        let embeddings_patch_embedding = candle_nn::conv2d(
            3,
            hidden_size,
            patch_size,
            conv_cfg,
            emb_vb.pp("patch_embedding"),
        )?;

        // Position and class embeddings
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let embeddings_position_embedding =
            emb_vb.get((num_patches + 1, hidden_size), "position_embedding")?;
        let embeddings_class_embedding = emb_vb.get(hidden_size, "class_embedding")?;

        let pre_layrnorm = layer_norm(hidden_size, 1e-5, vb.pp("pre_layrnorm"))?;

        // Load encoder layers
        let enc_vb = vb.pp("encoder.layers");
        let mut encoder_layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            encoder_layers.push(ClipEncoderLayer::load(
                enc_vb.pp(i.to_string()),
                hidden_size,
                intermediate_size,
                num_heads,
            )?);
        }

        Ok(Box::new(Self {
            name,
            embeddings_patch_embedding,
            embeddings_position_embedding,
            embeddings_class_embedding,
            pre_layrnorm,
            encoder_layers,
            select_layer,
            hidden_size,
        }))
    }

    async fn forward(
        &self,
        pixel_values: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        // pixel_values: (batch, 3, H, W)
        let batch_size = pixel_values.dims()[0];

        // Patch embedding: (batch, hidden, grid_h, grid_w) -> (batch, num_patches, hidden)
        let patch_embeds = self.embeddings_patch_embedding.forward(pixel_values)?;
        let (_, _, gh, gw) = patch_embeds.dims4()?;
        let num_patches = gh * gw;
        let patch_embeds = patch_embeds.flatten_from(2)?.transpose(1, 2)?;

        // Prepend CLS token
        let cls = self
            .embeddings_class_embedding
            .reshape((1, 1, self.hidden_size))?
            .broadcast_as((batch_size, 1, self.hidden_size))?;
        let embeddings = Tensor::cat(&[&cls, &patch_embeds], 1)?;

        // Add position embeddings (truncate if needed)
        let pos_embed = self
            .embeddings_position_embedding
            .narrow(0, 0, num_patches + 1)?
            .unsqueeze(0)?;
        let x = (embeddings + pos_embed)?;

        let x = self.pre_layrnorm.forward(&x)?;

        // Resolve which layer to extract from
        let total_layers = self.encoder_layers.len();
        let target_layer = if self.select_layer < 0 {
            (total_layers as isize + self.select_layer) as usize
        } else {
            self.select_layer as usize
        };

        // Run encoder layers, returning hidden state from selected layer
        let mut hidden = x;
        for (i, layer) in self.encoder_layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            if i == target_layer {
                // Return patches only (strip CLS token at position 0)
                return Ok(hidden.narrow(1, 1, num_patches)?);
            }
        }

        // Fallback: return last layer output (strip CLS)
        Ok(hidden.narrow(1, 1, num_patches)?)
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        self.forward(x, index_pos, block_idx, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}
