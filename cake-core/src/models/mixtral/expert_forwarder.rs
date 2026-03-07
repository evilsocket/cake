use anyhow::Result;
use async_trait::async_trait;
use candle_core::Tensor;

use crate::cake::{Context, Forwarder};
use super::moe_block::ExpertMLP;

/// Forwarder that serves a group of expert MLPs for all layers.
///
/// Layer name pattern: `"experts-group-{N}"`
///
/// This loads expert weights for a specified range of expert indices
/// across all MoE layers. When it receives a forward request, the
/// input tensor is treated as pre-gated tokens that need to be
/// processed by the appropriate expert(s).
///
/// For now, this serves as a local forwarder for worker-side expert
/// serving. The worker dispatches to this based on layer name matching.
#[derive(Debug)]
pub struct ExpertGroupForwarder {
    name: String,
    /// experts[layer_idx][expert_local_idx] = ExpertMLP
    experts: Vec<Vec<ExpertMLP>>,
    /// Which global expert indices this group covers.
    expert_range_start: usize,
    expert_range_end: usize,
    num_layers: usize,
}

impl std::fmt::Display for ExpertGroupForwarder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} (experts {}-{}, {} layers, local)",
            &self.name,
            self.expert_range_start,
            self.expert_range_end - 1,
            self.num_layers,
        )
    }
}

#[async_trait]
impl Forwarder for ExpertGroupForwarder {
    fn load(name: String, ctx: &Context) -> Result<Box<Self>> {
        let cfg = ctx.config.as_ref().expect("No config specified");
        let vb = ctx
            .var_builder
            .as_ref()
            .expect("No var_builder specified");

        // Parse expert group index from name: "experts-group-0", "experts-group-1", etc.
        let group_idx: usize = name
            .strip_prefix("experts-group-")
            .ok_or_else(|| anyhow!("invalid expert group name: {}", &name))?
            .parse()
            .map_err(|e| anyhow!("invalid expert group index in {}: {}", &name, e))?;

        let config_path = ctx.data_path.join("config.json");
        let moe_config = super::config::MixtralConfig::from_path(&config_path)?;
        let num_experts = moe_config.num_local_experts;
        let num_layers = cfg.num_hidden_layers;

        // Determine expert range for this group
        // Simple split: divide experts evenly across 2 groups
        let experts_per_group = num_experts / 2;
        let start = group_idx * experts_per_group;
        let end = if group_idx == 1 {
            num_experts
        } else {
            start + experts_per_group
        };

        log::info!(
            "loading expert group {} (experts {}-{}) for {} layers",
            group_idx,
            start,
            end - 1,
            num_layers,
        );

        let prefix = &cfg.model_prefix;
        let mut all_layer_experts = Vec::with_capacity(num_layers);

        for layer_idx in 0..num_layers {
            let layer_vb = vb.pp(format!(
                "{prefix}.layers.{layer_idx}.block_sparse_moe.experts"
            ));
            let mut layer_experts = Vec::with_capacity(end - start);
            for expert_idx in start..end {
                let expert = ExpertMLP::load(
                    layer_vb.pp(expert_idx),
                    cfg.hidden_size,
                    cfg.intermediate_size,
                )?;
                layer_experts.push(expert);
            }
            all_layer_experts.push(layer_experts);
        }

        Ok(Box::new(Self {
            name,
            experts: all_layer_experts,
            expert_range_start: start,
            expert_range_end: end,
            num_layers,
        }))
    }

    /// Forward pass for expert group.
    ///
    /// The input tensor `x` contains the hidden states for tokens routed to experts
    /// in this group. `block_idx` indicates which layer's experts to use.
    async fn forward(
        &self,
        x: &Tensor,
        _index_pos: usize,
        block_idx: usize,
        _ctx: &mut Context,
    ) -> Result<Tensor> {
        if block_idx >= self.num_layers {
            anyhow::bail!(
                "block_idx {} out of range (num_layers={})",
                block_idx,
                self.num_layers
            );
        }

        // For now, apply the first expert in the group.
        // In a full implementation, the routing information would be
        // packed into the tensor or sent as a separate message.
        let layer_experts = &self.experts[block_idx];
        if layer_experts.is_empty() {
            return Ok(x.clone());
        }
        layer_experts[0].forward(x)
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
