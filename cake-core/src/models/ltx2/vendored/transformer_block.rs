//! BasicAVTransformerBlock — the core dual-stream block for LTX-2.
//!
//! Each block has:
//! - Video: self-attn → text cross-attn → FFN (with AdaLN modulation)
//! - Audio: self-attn → text cross-attn → FFN (with AdaLN modulation)
//! - Audio↔Video bidirectional cross-attention
//!
//! In video-only mode, audio components are None.

use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

use super::attention::{rms_norm, Attention};
use super::config::Ltx2TransformerConfig;
use super::feed_forward::FeedForward;

/// Per-stream config for one block.
#[allow(dead_code)]
struct StreamConfig {
    dim: usize,
    heads: usize,
    d_head: usize,
    context_dim: usize,
}

/// A single dual-stream transformer block.
#[derive(Debug)]
#[allow(dead_code)]
pub struct BasicAVTransformerBlock {
    // Video stream
    attn1: Option<Attention>,        // video self-attention
    attn2: Option<Attention>,        // video text cross-attention
    ff: Option<FeedForward>,         // video feedforward
    scale_shift_table: Option<Tensor>, // [adaln_params, video_dim]

    // LTX-2.3: prompt-specific AdaLN modulation
    prompt_scale_shift_table: Option<Tensor>, // [3, video_dim]

    // Audio stream (None in video-only mode)
    audio_attn1: Option<Attention>,
    audio_attn2: Option<Attention>,
    audio_ff: Option<FeedForward>,
    audio_scale_shift_table: Option<Tensor>,

    // Audio↔Video cross-attention (None in unimodal mode)
    audio_to_video_attn: Option<Attention>,
    video_to_audio_attn: Option<Attention>,
    scale_shift_table_a2v_ca_audio: Option<Tensor>,
    scale_shift_table_a2v_ca_video: Option<Tensor>,

    norm_eps: f64,
    adaln_params: usize,
}

/// Inputs/outputs for one modality stream through a block.
pub struct StreamArgs {
    pub x: Tensor,
    pub timesteps: Tensor,          // [B, adaln_params, dim] pre-split modulation
    pub pe: Option<(Tensor, Tensor)>, // RoPE (cos, sin)
    pub context: Tensor,            // text embeddings
    pub context_mask: Option<Tensor>,
    pub self_attention_mask: Option<Tensor>,
    pub cross_pe: Option<(Tensor, Tensor)>, // cross-modal RoPE
    pub enabled: bool,
}

impl BasicAVTransformerBlock {
    pub fn new(
        _idx: usize,
        config: &Ltx2TransformerConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm_eps = config.norm_eps;
        let adaln_params = config.adaln_params();

        let video_dim = config.video_inner_dim();
        let audio_dim = config.audio_inner_dim();
        let has_video = config.model_type.is_video_enabled();
        let has_audio = config.model_type.is_audio_enabled();
        let gated = config.gated_attention;

        // Video components
        let (attn1, attn2, ff, scale_shift_table) = if has_video {
            let attn1 = Attention::new(
                video_dim,
                None,
                config.num_attention_heads,
                config.attention_head_dim,
                norm_eps,
                gated,
                vb.pp("attn1"),
            )?;
            let attn2 = Attention::new(
                video_dim,
                Some(config.cross_attention_dim),
                config.num_attention_heads,
                config.attention_head_dim,
                norm_eps,
                gated,
                vb.pp("attn2"),
            )?;
            let ff = FeedForward::new(video_dim, video_dim, 4, vb.pp("ff"))?;
            let sst = vb.get((adaln_params, video_dim), "scale_shift_table")?;
            (Some(attn1), Some(attn2), Some(ff), Some(sst))
        } else {
            (None, None, None, None)
        };

        // LTX-2.3: prompt modulation table (shift + scale, no gate)
        let prompt_scale_shift_table = if has_video && config.prompt_modulation {
            Some(vb.get((2, video_dim), "prompt_scale_shift_table")?)
        } else {
            None
        };

        // Audio components
        let (audio_attn1, audio_attn2, audio_ff, audio_sst) = if has_audio {
            let a1 = Attention::new(
                audio_dim,
                None,
                config.audio_num_attention_heads,
                config.audio_attention_head_dim,
                norm_eps,
                gated,
                vb.pp("audio_attn1"),
            )?;
            let a2 = Attention::new(
                audio_dim,
                Some(config.audio_cross_attention_dim),
                config.audio_num_attention_heads,
                config.audio_attention_head_dim,
                norm_eps,
                gated,
                vb.pp("audio_attn2"),
            )?;
            let ff = FeedForward::new(audio_dim, audio_dim, 4, vb.pp("audio_ff"))?;
            let sst = vb.get((adaln_params, audio_dim), "audio_scale_shift_table")?;
            (Some(a1), Some(a2), Some(ff), Some(sst))
        } else {
            (None, None, None, None)
        };

        // Cross-modal attention
        let (a2v, v2a, sst_a2v_audio, sst_a2v_video) = if has_video && has_audio {
            let a2v = Attention::new(
                video_dim,
                Some(audio_dim),
                config.audio_num_attention_heads,
                config.audio_attention_head_dim,
                norm_eps,
                gated,
                vb.pp("audio_to_video_attn"),
            )?;
            let v2a = Attention::new(
                audio_dim,
                Some(video_dim),
                config.audio_num_attention_heads,
                config.audio_attention_head_dim,
                norm_eps,
                gated,
                vb.pp("video_to_audio_attn"),
            )?;
            let sst_audio = vb.get((5, audio_dim), "audio_a2v_cross_attn_scale_shift_table")?;
            let sst_video = vb.get((5, video_dim), "video_a2v_cross_attn_scale_shift_table")?;
            (Some(a2v), Some(v2a), Some(sst_audio), Some(sst_video))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            attn1,
            attn2,
            ff,
            scale_shift_table,
            prompt_scale_shift_table,
            audio_attn1,
            audio_attn2,
            audio_ff,
            audio_scale_shift_table: audio_sst,
            audio_to_video_attn: a2v,
            video_to_audio_attn: v2a,
            scale_shift_table_a2v_ca_audio: sst_a2v_audio,
            scale_shift_table_a2v_ca_video: sst_a2v_video,
            norm_eps,
            adaln_params,
        })
    }

    /// Extract AdaLN modulation values from scale_shift_table + timestep.
    ///
    /// `sst`: `[N, dim]`
    /// `timestep`: `[B, 1, N, dim]` (pre-reshaped)
    /// `indices`: range of params to extract
    ///
    /// Returns tuple of tensors, each `[B, 1, dim]`.
    fn get_ada_values(
        sst: &Tensor,
        timestep: &Tensor,
        start: usize,
        end: usize,
    ) -> Result<Vec<Tensor>> {
        let count = end - start;
        // sst[start..end]: [count, dim] -> [1, 1, count, dim]
        let sst_slice = sst.narrow(0, start, count)?.unsqueeze(0)?.unsqueeze(0)?;

        // timestep[:, :, start..end, :]: [B, 1, count, dim]
        let ts_slice = timestep.narrow(2, start, count)?;

        // Add: [B, 1, count, dim]
        let combined = sst_slice
            .to_dtype(ts_slice.dtype())?
            .broadcast_add(&ts_slice)?;

        // Unbind along dim 2 -> count tensors of [B, 1, dim]
        let mut result = Vec::with_capacity(count);
        for i in 0..count {
            result.push(combined.narrow(2, i, 1)?.squeeze(2)?);
        }
        Ok(result)
    }

    /// Forward pass for video-only mode.
    ///
    /// `video`: current video hidden states
    /// `timesteps`: pre-computed AdaLN modulation, `[B, 1, adaln_params, dim]`
    /// `pe`: RoPE (cos, sin)
    /// `context`: text embeddings
    /// `context_mask`: attention mask for text
    /// `prompt_temb`: prompt timestep embedding for prompt modulation (LTX-2.3), `[B, 1, 3, dim]`
    /// `skip_self_attn`: if true, bypass self-attention Q/K (STG perturbation)
    pub fn forward_video_only(
        &self,
        video: &Tensor,
        timesteps: &Tensor,
        pe: Option<&(Tensor, Tensor)>,
        context: &Tensor,
        context_mask: Option<&Tensor>,
        prompt_temb: Option<&Tensor>,
        skip_self_attn: bool,
    ) -> Result<Tensor> {
        let sst = self
            .scale_shift_table
            .as_ref()
            .expect("video scale_shift_table required");
        let attn1 = self.attn1.as_ref().unwrap();
        let attn2 = self.attn2.as_ref().unwrap();
        let ff = self.ff.as_ref().unwrap();

        // Get modulation params: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
        let ada_msa = Self::get_ada_values(sst, timesteps, 0, 3)?;
        let (shift_msa, scale_msa, gate_msa) = (&ada_msa[0], &ada_msa[1], &ada_msa[2]);

        // Self-attention with AdaLN
        let norm_x = rms_norm(video, self.norm_eps)?;
        let norm_x = norm_x
            .broadcast_mul(&scale_msa.broadcast_add(&Tensor::ones_like(scale_msa)?)?)?
            .broadcast_add(shift_msa)?;

        // STG: skip Q/K attention, pass V through directly
        let attn_out = if skip_self_attn {
            attn1.forward_skip_attn(&norm_x, None)?
        } else {
            attn1.forward(&norm_x, None, pe, None, None)?
        };
        let vx = video.broadcast_add(&attn_out.broadcast_mul(gate_msa)?)?;

        // Text cross-attention with AdaLN
        let norm_vx = rms_norm(&vx, self.norm_eps)?;

        // Cross-attention AdaLN: modulate query input (LTX-2.3)
        // Guard on actual temb tensor dim (not stored field) to handle config mismatches
        let has_ca_adaln = self.adaln_params > 6 && timesteps.dim(2)? > 6;
        let (norm_vx, gate_ca) = if has_ca_adaln {
            let ada_ca = Self::get_ada_values(sst, timesteps, 6, 9)?;
            let (shift_ca, scale_ca, gate) = (&ada_ca[0], &ada_ca[1], ada_ca[2].clone());
            let modulated = norm_vx
                .broadcast_mul(&scale_ca.broadcast_add(&Tensor::ones_like(scale_ca)?)?)?
                .broadcast_add(shift_ca)?;
            (modulated, Some(gate))
        } else {
            (norm_vx, None)
        };

        // LTX-2.3: prompt modulation — modulate CONTEXT (key/value) for cross-attention
        // Python: encoder_hidden_states = context * (1 + scale_kv) + shift_kv
        let ca_context = if let (Some(psst), Some(pt)) = (&self.prompt_scale_shift_table, prompt_temb) {
            let prompt_ada = Self::get_ada_values(psst, pt, 0, 2)?;
            let (p_shift, p_scale) = (&prompt_ada[0], &prompt_ada[1]);
            context
                .broadcast_mul(&p_scale.broadcast_add(&Tensor::ones_like(p_scale)?)?)?
                .broadcast_add(p_shift)?
        } else {
            context.clone()
        };

        // Expand context_mask from [B, L] to [B, T_q, L] for cross-attention
        let t_q = norm_vx.dim(1)?;
        let expanded_mask = context_mask.map(|m| {
            m.unsqueeze(1)
                .and_then(|m| m.broadcast_as((m.dim(0)?, t_q, m.dim(2)?)))
                .and_then(|m| m.contiguous())
        }).transpose()?;
        let ca_out = attn2.forward(&norm_vx, Some(&ca_context), None, None, expanded_mask.as_ref())?;

        // Apply cross-attention gate (LTX-2.3)
        let ca_out = if let Some(ref gate) = gate_ca {
            ca_out.broadcast_mul(gate)?
        } else {
            ca_out
        };
        let vx = vx.broadcast_add(&ca_out)?;

        // FFN with AdaLN
        let ada_mlp = Self::get_ada_values(sst, timesteps, 3, 6)?;
        let (shift_mlp, scale_mlp, gate_mlp) = (&ada_mlp[0], &ada_mlp[1], &ada_mlp[2]);

        let norm_vx = rms_norm(&vx, self.norm_eps)?;
        let norm_vx = norm_vx
            .broadcast_mul(&scale_mlp.broadcast_add(&Tensor::ones_like(scale_mlp)?)?)?
            .broadcast_add(shift_mlp)?;

        let ff_out = ff.forward(&norm_vx)?;

        let vx = vx.broadcast_add(&ff_out.broadcast_mul(gate_mlp)?)?;

        Ok(vx)
    }

}
