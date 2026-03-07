use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;
use candle_transformers::models::clip::vision_model::ClipVisionConfig;
use candle_transformers::models::llava::{ClipVisionTower, MMProjector};

use crate::cake::{Context, Forwarder};
use super::config::LlavaConfig;

/// Forwarder wrapping the CLIP vision tower + MM projector.
///
/// Layer name: `"llava-vision"`
///
/// Input tensor: pixel values `[B, C, H, W]`
/// Output tensor: projected visual embeddings `[B, N, D]`
pub struct LlavaVision {
    name: String,
    clip_vision_tower: ClipVisionTower,
    mm_projector: MMProjector,
}

// Safety: LlavaVision contains ClipVisionTower and MMProjector which internally hold
// Linear layers (Tensor + Option<Tensor>). Tensors are Send+Sync. The `dyn Module`
// in Sequential doesn't have Send+Sync bounds, but the concrete types stored are
// Linear and Activation which are both Send+Sync. We only access this from one
// inference thread at a time.
unsafe impl Send for LlavaVision {}
unsafe impl Sync for LlavaVision {}

impl std::fmt::Debug for LlavaVision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlavaVision")
            .field("name", &self.name)
            .finish()
    }
}

impl std::fmt::Display for LlavaVision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

fn load_vision_components(
    ctx: &Context,
) -> Result<(ClipVisionTower, MMProjector)> {
    let config_path = ctx.data_path.join("config.json");
    let llava_config = LlavaConfig::from_path(&config_path)?;
    let candle_config = llava_config.to_candle_llava_config();

    let vb = ctx
        .var_builder
        .as_ref()
        .expect("No var_builder specified");

    let clip_vision_config = if let Some(ref vc) = llava_config.vision_config {
        Some(ClipVisionConfig {
            embed_dim: vc.hidden_size,
            activation: candle_transformers::models::clip::text_model::Activation::QuickGelu,
            intermediate_size: vc.intermediate_size,
            num_hidden_layers: vc.num_hidden_layers,
            num_attention_heads: vc.num_attention_heads,
            projection_dim: vc.projection_dim.unwrap_or(768),
            num_channels: 3,
            image_size: vc.image_size,
            patch_size: vc.patch_size,
        })
    } else {
        None
    };

    let vb_vision = if candle_config.hf {
        vb.pp("vision_tower.vision_model")
    } else {
        vb.pp("model.vision_tower.vision_tower.vision_model")
    };

    let clip_vision_tower = ClipVisionTower::new(
        vb_vision,
        candle_config.mm_vision_select_layer,
        &candle_config.mm_vision_select_feature,
        &clip_vision_config,
    )?;

    let mm_projector = MMProjector::load(vb, &candle_config)?;

    Ok((clip_vision_tower, mm_projector))
}

impl LlavaVision {
    pub fn load_model(ctx: &Context) -> Result<Box<dyn Forwarder>> {
        let (clip_vision_tower, mm_projector) = load_vision_components(ctx)?;
        Ok(Box::new(Self {
            name: "llava-vision".to_string(),
            clip_vision_tower,
            mm_projector,
        }))
    }

    /// Encode images: CLIP vision tower + MM projector.
    pub fn encode_images(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let image_features = self.clip_vision_tower.forward(pixel_values)?;
        let projected = self.mm_projector.forward(&image_features)?;
        Ok(projected)
    }
}

#[async_trait]
impl Forwarder for LlavaVision {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let (clip_vision_tower, mm_projector) = load_vision_components(ctx)?;
        Ok(Box::new(Self {
            name,
            clip_vision_tower,
            mm_projector,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        Ok(self.encode_images(x)?)
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
