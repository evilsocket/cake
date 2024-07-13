mod attention;
mod cache;
mod config;
mod mlp;
mod transformer;

use std::fmt::Debug;

pub use attention::*;
pub use cache::*;
pub use config::*;
pub use mlp::*;

pub use transformer::*;

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{linear_no_bias as linear, Embedding, Linear, Module, RmsNorm, VarBuilder};

use crate::cake::{Forwarder, Topology};

pub const EOS_TOKEN: &str = "</s>";

#[derive(Debug)]
pub struct Llama {
    embedding: Embedding,
    blocks: Vec<Box<dyn Forwarder>>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Llama {
    pub async fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        let (_batch_size, seq_len) = x.dims2()?;
        let mut x = self.embedding.forward(x)?;

        let num_blocks = self.blocks.len();
        let mut block_idx = 0;

        // log::info!("X = {}", &x);

        while block_idx < num_blocks {
            let curr_block_id = self.blocks[block_idx].ident().to_owned();
            if curr_block_id == "local" {
                // do not batch local inferences
                x = self.blocks[block_idx]
                    .forward(&x, index_pos, block_idx, cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward operation of local block {block_idx}: {e}")
                    })?;

                block_idx += 1;
            } else {
                // collect all contiguous layers running on the same worker
                let mut batch = vec![];
                let first = block_idx;
                while block_idx < num_blocks && self.blocks[block_idx].ident() == curr_block_id {
                    batch.push((
                        self.blocks[block_idx].layer_name().to_string(),
                        index_pos,
                        block_idx,
                    ));
                    block_idx += 1;
                }

                x = self.blocks[first]
                    .forward_batch(&x, batch, cache)
                    .await
                    .map_err(|e| {
                        anyhow!("error in forward batch operation for block {block_idx}: {e}")
                    })?;
            }

            // log::info!("{}.forward(X) -> {}", &curr_block_id, &x);
        }

        let x = self
            .ln_f
            .forward(&x)
            .map_err(|e| anyhow!("error in ln_f.forward: {e}"))?;

        let x = x
            .i((.., seq_len - 1, ..))
            .map_err(|e| anyhow!("error in x.i: {e}"))?
            .contiguous()
            .map_err(|e| anyhow!("error in x.i.contiguous: {e}"))?;

        let logits = self
            .lm_head
            .forward(&x)
            .map_err(|e| anyhow!("error in lm_head.forward: {e}"))?;

        logits
            .to_dtype(DType::F32)
            .map_err(|e| anyhow!("error converting logits: {e}"))
    }

    pub async fn load(
        vb: &VarBuilder<'static>,
        cfg: &Config,
        device: &Device,
        topology: &Topology,
    ) -> Result<Self> {
        log::info!("loading embeddings ...");
        let embedding =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        log::info!("loading lm_head ...");
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        log::info!("loading model.norm ...");
        let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        log::info!("loading {} blocks ...", cfg.num_hidden_layers);

        let mut blocks: Vec<Box<dyn Forwarder>> = vec![];

        for i in 0..cfg.num_hidden_layers {
            let block_layer_name = format!("model.layers.{i}");

            if let Some((node_name, node)) = topology.get_node_for_layer(&block_layer_name) {
                log::debug!("node {node_name} will serve {}", &block_layer_name);

                let client =
                    crate::cake::Client::new(device.clone(), &node.host, &block_layer_name).await?;

                blocks.push(Box::new(client));
            } else {
                log::debug!("{} will be served locally", &block_layer_name);

                let block = Block::load(&block_layer_name, vb.pp(&block_layer_name), cfg)?;

                blocks.push(Box::new(block));
            }
        }

        for block in &blocks {
            log::info!("  {}", block)
        }

        Ok(Self {
            embedding,
            blocks,
            ln_f,
            lm_head,
        })
    }
}
