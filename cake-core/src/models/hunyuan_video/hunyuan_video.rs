use anyhow::Result;
use async_trait::async_trait;

use super::clip::HunyuanClip;
use super::hunyuan_video_shardable::HunyuanVideoShardable;
use super::t5::HunyuanT5;
use super::transformer::HunyuanTransformer;
use super::vae_forwarder::HunyuanVae;
use crate::cake::{Context, Forwarder};
use crate::models::{Generator, VideoGenerator};
use crate::video::VideoOutput;
use crate::ImageGenerationArgs;

/// HunyuanVideo model.
///
/// Follows the same component distribution pattern as LTX-Video:
/// each component (transformer, T5, CLIP, VAE) can be local or remote.
#[allow(dead_code)]
pub struct HunyuanVideo {
    t5_encoder: Box<dyn Forwarder>,
    clip_encoder: Box<dyn Forwarder>,
    transformer: Box<dyn Forwarder>,
    vae: Box<dyn Forwarder>,
    context: Context,
}

#[async_trait]
impl Generator for HunyuanVideo {
    type Shardable = HunyuanVideoShardable;
    const MODEL_NAME: &'static str = "hunyuan-video";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        log::info!("Loading HunyuanVideo components...");

        // T5 encoder
        let t5_encoder: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("hunyuan-t5") {
                log::info!("hunyuan-t5 will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "hunyuan-t5",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                HunyuanT5::load_model(context)?
            };

        // CLIP encoder
        let clip_encoder: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("hunyuan-clip") {
                log::info!("hunyuan-clip will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "hunyuan-clip",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                HunyuanClip::load_model(context)?
            };

        // Transformer
        let transformer: Box<dyn Forwarder> = if let Some((_name, node)) =
            context.topology.get_node_for_layer("hunyuan-transformer")
        {
            log::info!("hunyuan-transformer will be served by {}", &node.host);
            Box::new(
                crate::cake::Client::new(
                    context.device.clone(),
                    &node.host,
                    "hunyuan-transformer",
                    context.args.cluster_key.as_deref(),
                )
                .await?,
            )
        } else {
            HunyuanTransformer::load_model(context)?
        };

        // VAE
        let vae: Box<dyn Forwarder> =
            if let Some((_name, node)) = context.topology.get_node_for_layer("hunyuan-vae") {
                log::info!("hunyuan-vae will be served by {}", &node.host);
                Box::new(
                    crate::cake::Client::new(
                        context.device.clone(),
                        &node.host,
                        "hunyuan-vae",
                        context.args.cluster_key.as_deref(),
                    )
                    .await?,
                )
            } else {
                HunyuanVae::load_model(context)?
            };

        log::info!("HunyuanVideo components loaded");

        Ok(Some(Box::new(Self {
            t5_encoder,
            clip_encoder,
            transformer,
            vae,
            context: context.clone(),
        })))
    }
}

#[async_trait]
impl VideoGenerator for HunyuanVideo {
    async fn generate_video(&mut self, _args: &ImageGenerationArgs) -> Result<VideoOutput> {
        anyhow::bail!(
            "HunyuanVideo generation not yet implemented — vendored transformer/VAE code required. \
             The component distribution infrastructure is ready; implement the vendored model code \
             in cake-core/src/models/hunyuan_video/vendored/ to enable generation."
        )
    }
}
