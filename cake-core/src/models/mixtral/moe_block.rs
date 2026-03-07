use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Module, Tensor};
use candle_nn::{Activation, VarBuilder};

use crate::cake::{Context, Forwarder};
use crate::models::common::CausalSelfAttention;

/// A single expert MLP (gate_proj + up_proj + down_proj with SiLU activation).
#[derive(Debug, Clone)]
pub struct ExpertMLP {
    w1: candle_nn::Linear,
    w2: candle_nn::Linear,
    w3: candle_nn::Linear,
    act_fn: Activation,
}

impl ExpertMLP {
    pub fn load(vb: VarBuilder, hidden_size: usize, intermediate_size: usize) -> Result<Self> {
        let w1 = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("w1"))?;
        let w2 = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("w2"))?;
        let w3 = candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("w3"))?;
        Ok(Self {
            w1,
            w2,
            w3,
            act_fn: Activation::Silu,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.w1.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.w3.forward(xs)?;
        Ok(self.w2.forward(&(lhs * rhs)?)?)
    }
}

/// MoE-aware transformer block.
///
/// Attention runs locally. The MLP is replaced by a sparse mixture of experts
/// with a routing gate. Experts can be local or dispatched to remote workers
/// via expert group forwarders.
#[derive(Debug)]
#[allow(dead_code)]
pub struct MoeBlock {
    name: String,
    rms_1: candle_nn::RmsNorm,
    attn: CausalSelfAttention,
    rms_2: candle_nn::RmsNorm,
    gate: candle_nn::Linear,
    experts: Vec<ExpertMLP>,
    num_experts_per_tok: usize,
    /// Remote expert group forwarders (keyed by expert group name).
    remote_expert_groups: Vec<Box<dyn Forwarder>>,
    /// Which expert indices are remote (mapped to remote_expert_groups index).
    remote_expert_mapping: Vec<(usize, usize)>, // (expert_idx, group_idx)
}

impl std::fmt::Display for MoeBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (local, {} experts, {} remote groups)",
            &self.name,
            self.experts.len(),
            self.remote_expert_groups.len()
        )
    }
}

impl MoeBlock {
    pub fn load(name: String, ctx: &Context) -> Result<Self> {
        let cfg = ctx.config.as_ref().expect("No config specified");
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified")
            .pp(&name);

        let attn = CausalSelfAttention::load(vb.pp("self_attn"), cfg)?;
        let rms_1 =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let rms_2 = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        // Load MoE components
        let moe_vb = vb.pp("block_sparse_moe");

        // Extract MoE parameters from the model config JSON
        let config_path = ctx.data_path.join("config.json");
        let moe_config: super::config::MixtralConfig =
            super::config::MixtralConfig::from_path(&config_path)?;

        let num_experts = moe_config.num_local_experts;
        let num_experts_per_tok = moe_config.num_experts_per_tok;

        let gate = candle_nn::linear_no_bias(
            cfg.hidden_size,
            num_experts,
            moe_vb.pp("gate"),
        )?;

        // Load all local experts
        let experts_vb = moe_vb.pp("experts");
        let mut experts = Vec::with_capacity(num_experts);
        for i in 0..num_experts {
            let expert =
                ExpertMLP::load(experts_vb.pp(i), cfg.hidden_size, cfg.intermediate_size)?;
            experts.push(expert);
        }

        Ok(Self {
            name,
            rms_1,
            attn,
            rms_2,
            gate,
            experts,
            num_experts_per_tok,
            remote_expert_groups: Vec::new(),
            remote_expert_mapping: Vec::new(),
        })
    }

    /// Forward pass for the MoE block.
    fn moe_forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b_size, seq_len, hidden_dim) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let router_logits = self.gate.forward(&xs_flat)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Extract routing weights to CPU for topk selection
        let routing_weights_vec = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

        let mut top_x = vec![vec![]; self.experts.len()];
        let mut selected_rws = vec![vec![]; self.experts.len()];

        for (row_idx, rw) in routing_weights_vec.iter().enumerate() {
            let mut dst: Vec<u32> = (0..rw.len() as u32).collect();
            dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));

            let mut sum_routing_weights = 0f32;
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                sum_routing_weights += rw[expert_idx as usize];
            }
            for &expert_idx in dst.iter().take(self.num_experts_per_tok) {
                let expert_idx = expert_idx as usize;
                let routing_weight = rw[expert_idx];
                top_x[expert_idx].push(row_idx as u32);
                selected_rws[expert_idx].push(routing_weight / sum_routing_weights);
            }
        }

        let mut ys = xs_flat.zeros_like()?;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            let top_x_expert = &top_x[expert_idx];
            if top_x_expert.is_empty() {
                continue;
            }
            let top_x_tensor = Tensor::new(top_x_expert.as_slice(), xs.device())?;
            let selected_rws_tensor = Tensor::new(
                selected_rws[expert_idx].as_slice(),
                xs.device(),
            )?
            .reshape(((), 1))?;

            let current_state =
                xs_flat.index_select(&top_x_tensor, 0)?.reshape(((), hidden_dim))?;
            let current_hidden_states = expert.forward(&current_state)?;
            let current_hidden_states =
                current_hidden_states.broadcast_mul(&selected_rws_tensor)?;
            ys = ys.index_add(&top_x_tensor, &current_hidden_states, 0)?;
        }

        Ok(ys.reshape((b_size, seq_len, hidden_dim))?)
    }
}

#[async_trait]
impl Forwarder for MoeBlock {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        Ok(Box::new(Self::load(name, ctx)?))
    }

    async fn forward(
        &self,
        x: &Tensor,
        index_pos: usize,
        block_idx: usize,
        ctx: &mut Context,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self
            .rms_1
            .forward(x)
            .map_err(|e| anyhow!("moe rms_1: {e}"))?;
        let x = (self
            .attn
            .forward(
                &x,
                index_pos,
                block_idx,
                ctx.cache.as_mut().expect("No cache specified"),
            )
            .map_err(|e| anyhow!("moe attention: {e}"))?
            + residual)
            .map_err(|e| anyhow!("moe attn residual: {e}"))?;

        let residual = &x;
        let x = self
            .rms_2
            .forward(&x)
            .map_err(|e| anyhow!("moe rms_2: {e}"))?;
        let x = (self.moe_forward(&x).map_err(|e| anyhow!("moe forward: {e}"))? + residual)
            .map_err(|e| anyhow!("moe mlp residual: {e}"))?;

        Ok(x)
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
