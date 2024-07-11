use anyhow::Result;
use candle_core::Tensor;
use candle_nn::{Module, RmsNorm, VarBuilder};

use async_trait::async_trait;

use super::{Cache, CausalSelfAttention, Forwarder, MLP};

#[derive(Debug, Clone)]
pub struct Block {
    name: String,
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: MLP,
}

impl Block {
    pub fn load(name: &str, vb: VarBuilder, cfg: &super::Config) -> Result<Self> {
        let name = name.to_string();
        let attn = super::CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let mlp = super::MLP::load(vb.pp("mlp"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            name,
            rms_1,
            attn,
            rms_2,
            mlp,
        })
    }

    pub async fn forward_imm(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        // log::info!("block forward[{index_pos}, {block_idx}]");
        let residual = x;

        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx, cache)? + residual)?;
        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        let x = (self.mlp.forward(&x)? + residual)?;
        Ok(x)
    }
}

impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (local)", &self.name)
    }
}

#[async_trait]
impl Forwarder for Block {
    async fn forward(
        &mut self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.forward_imm(x, index_pos, block_idx, cache).await
    }

    fn layer_name(&self) -> &str {
        &self.name
    }
}
