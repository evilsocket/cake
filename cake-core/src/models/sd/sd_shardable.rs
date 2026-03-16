use crate::cake::{Context, Forwarder};
use crate::models::sd::clip::Clip;
use crate::models::sd::unet::UNet;
use crate::models::sd::vae::VAE;
use async_trait::async_trait;
use candle_core::Tensor;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct SDShardable {
    forwarder: Box<dyn Forwarder>,
    layer_name: String,
}

impl Display for SDShardable {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.layer_name)
    }
}

#[async_trait]
impl Forwarder for SDShardable {
    fn load(name: String, ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        let model: Box<dyn Forwarder> = match name.as_str() {
            "vae" => VAE::load(name.clone(), ctx)?,
            "clip" => Clip::load(name.clone(), ctx)?,
            "clip2" => Clip::load(name.clone(), ctx)?,
            "unet" => UNet::load(name.clone(), ctx)?,
            _ => {
                anyhow::bail!("Model name not recognized");
            }
        };

        Ok(Box::new(Self {
            forwarder: model,
            layer_name: name,
        }))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        self.forwarder.forward(x, index_pos, block_idx, ctx).await
    }

    async fn forward_mut(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        log::debug!("forwarding single op");
        self.forwarder
            .forward_mut(x, index_pos, block_idx, ctx)
            .await
    }

    async fn forward_batch(
        &mut self,
        x: &Tensor,
        batch: Vec<(String, usize, usize)>,
        ctx: &mut Context,
    ) -> anyhow::Result<Tensor> {
        log::debug!("forwarding batch of {} elements", batch.len());
        self.forwarder.forward_batch(x, batch, ctx).await
    }

    fn layer_name(&self) -> &str {
        &self.layer_name
    }

    fn ident(&self) -> &str {
        &self.layer_name
    }
}

#[cfg(test)]
const KNOWN_LAYER_NAMES: &[&str] = &["vae", "clip", "clip2", "unet"];

#[cfg(test)]
fn is_valid_sd_layer_name(name: &str) -> bool {
    KNOWN_LAYER_NAMES.contains(&name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_layer_names() {
        assert!(is_valid_sd_layer_name("vae"));
        assert!(is_valid_sd_layer_name("clip"));
        assert!(is_valid_sd_layer_name("clip2"));
        assert!(is_valid_sd_layer_name("unet"));
    }

    #[test]
    fn invalid_layer_names() {
        assert!(!is_valid_sd_layer_name("tokenizer"));
        assert!(!is_valid_sd_layer_name("tokenizer_2"));
        assert!(!is_valid_sd_layer_name(""));
        assert!(!is_valid_sd_layer_name("VAE"));
        assert!(!is_valid_sd_layer_name("unknown"));
    }

    #[test]
    fn known_layer_names_count() {
        assert_eq!(KNOWN_LAYER_NAMES.len(), 4);
    }
}
