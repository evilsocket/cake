//! Relative-position multi-head attention for Zipformer.
//!
//! Split into two structs matching the actual weight layout:
//! - `RelPositionMultiheadAttentionWeights`: computes attention weight matrices
//! - `SelfAttention`: applies those weights to value projections

use std::sync::Arc;

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use crate::backends::ComputeBackend;

/// Computes attention weight matrices from input + positional embeddings.
///
/// Weights:
/// - `in_proj` [4*(query_head_dim + key_head_dim + pos_head_dim), dim]
///   = [272, 512] for the FM decoder (4*(32+32+4) = 272)
/// - `linear_pos` [num_heads*pos_head_dim, pos_dim] = [16, 48]
#[derive(Debug, Clone)]
pub struct RelPositionMultiheadAttentionWeights {
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    linear_pos: Tensor, // no bias
    num_heads: usize,
    query_head_dim: usize,
    pos_head_dim: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl RelPositionMultiheadAttentionWeights {
    pub fn load(
        dim: usize,
        num_heads: usize,
        query_head_dim: usize,
        pos_head_dim: usize,
        pos_dim: usize,
        vb: VarBuilder,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        // in_proj output size = num_heads * (query_head_dim + key_head_dim + pos_head_dim)
        // where key_head_dim == query_head_dim
        let proj_dim = num_heads * (query_head_dim + query_head_dim + pos_head_dim);
        let in_proj_weight = vb.pp("in_proj").get((proj_dim, dim), "weight")?;
        let in_proj_bias = Some(vb.pp("in_proj").get(proj_dim, "bias")?);
        let linear_pos = vb.get((num_heads * pos_head_dim, pos_dim), "linear_pos.weight")?;

        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            linear_pos,
            num_heads,
            backend,
            query_head_dim,
            pos_head_dim,
        })
    }

    /// Compute attention weights from input and positional embeddings.
    ///
    /// Returns attention weights tensor of shape [batch, num_heads, seq, seq].
    pub fn forward(&self, x: &Tensor, pos_emb: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let nh = self.num_heads;
        let qhd = self.query_head_dim;
        let phd = self.pos_head_dim;

        // Project to Q, K, pos
        let projected = self.backend.linear_forward(x, &self.in_proj_weight, self.in_proj_bias.as_ref())?; // [batch, seq, proj_dim]
        // Split: Q [nh*qhd], K [nh*qhd], pos [nh*phd]
        let q = projected.narrow(candle_core::D::Minus1, 0, nh * qhd)?;
        let k = projected.narrow(candle_core::D::Minus1, nh * qhd, nh * qhd)?;
        let pos_part = projected.narrow(candle_core::D::Minus1, 2 * nh * qhd, nh * phd)?;

        // Reshape to multi-head: [batch, seq, nh, hd] -> [batch, nh, seq, hd]
        let q = q
            .reshape((batch, seq_len, nh, qhd))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, nh, qhd))?
            .transpose(1, 2)?;

        // Content attention: Q @ K^T (no scaling — Python doesn't scale)
        let content_attn = q.matmul(&k.transpose(2, 3)?)?;

        // Positional attention
        // pos_part: [batch, seq, nh*phd]
        let pos_part = pos_part
            .reshape((batch, seq_len, nh, phd))?
            .transpose(1, 2)?; // [batch, nh, seq, phd]

        // linear_pos: [nh*phd, pos_dim] applied to pos_emb [1, 2*seq-1, pos_dim]
        // -> [1, 2*seq-1, nh*phd] -> [1, nh, 2*seq-1, phd]
        let pos_len = pos_emb.dim(1)?;
        let pos_proj = pos_emb.broadcast_matmul(&self.linear_pos.t()?)?; // [1, pos_len, nh*phd]
        let pos_proj = pos_proj
            .reshape((1, pos_len, nh, phd))?
            .transpose(1, 2)?; // [1, nh, pos_len, phd]

        // pos_part @ pos_proj^T: [batch, nh, seq, phd] @ [1, nh, phd, pos_len]
        let pos_attn = pos_part.matmul(&pos_proj.transpose(2, 3)?)?; // [batch, nh, seq, 2*seq-1]

        // Relative position shift
        let pos_attn = self.rel_shift(&pos_attn, seq_len)?;

        // Combined attention weights
        let attn = (content_attn + pos_attn)?;
        let attn = self.backend.softmax(&attn, attn.rank() - 1)?;
        Ok(attn)
    }

    /// Relative position shift: convert from (batch/heads, ?, seq, 2*seq-1) to (batch/heads, ?, seq, seq).
    /// For row i, extract columns [seq-1-i .. 2*seq-1-i], i.e. diagonal extraction.
    fn rel_shift(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let dims = x.dims4()?;
        let (d0, d1, rows, cols) = dims;
        if cols == seq_len {
            return Ok(x.clone());
        }
        // For each row i, we need columns starting at (seq_len - 1 - i)
        // Build index tensor for gather
        let mut indices = vec![0u32; rows * seq_len];
        for i in 0..rows {
            for j in 0..seq_len {
                indices[i * seq_len + j] = (seq_len - 1 - i + j) as u32;
            }
        }
        let idx = candle_core::Tensor::new(&indices[..], x.device())?
            .reshape((1, 1, rows, seq_len))?
            .broadcast_as((d0, d1, rows, seq_len))?
            .contiguous()?;
        let x_contig = x.contiguous()?;
        Ok(x_contig.gather(&idx, 3)?)
    }
}

/// Self-attention that applies precomputed attention weights to value projections.
///
/// Weights:
/// - `in_proj` [num_heads * value_head_dim, dim] = [48, 512]
/// - `out_proj` [dim, num_heads * value_head_dim] = [512, 48]
#[derive(Debug, Clone)]
pub struct SelfAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    out_proj_weight: Tensor,
    out_proj_bias: Option<Tensor>,
    num_heads: usize,
    value_head_dim: usize,
    backend: Arc<dyn ComputeBackend>,
}

impl SelfAttention {
    pub fn load(
        dim: usize,
        num_heads: usize,
        value_head_dim: usize,
        vb: VarBuilder,
        backend: Arc<dyn ComputeBackend>,
    ) -> Result<Self> {
        let v_dim = num_heads * value_head_dim;
        let in_proj_weight = vb.pp("in_proj").get((v_dim, dim), "weight")?;
        let in_proj_bias = Some(vb.pp("in_proj").get(v_dim, "bias")?);
        let out_proj_weight = vb.pp("out_proj").get((dim, v_dim), "weight")?;
        let out_proj_bias = Some(vb.pp("out_proj").get(dim, "bias")?);
        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj_weight,
            out_proj_bias,
            num_heads,
            value_head_dim,
            backend,
        })
    }

    /// Apply attention using precomputed weights.
    /// `attn_weights`: [batch, num_heads, seq, seq]
    /// `x`: [batch, seq, dim]
    pub fn forward(&self, x: &Tensor, attn_weights: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;

        // Value projection
        let v = self.backend.linear_forward(x, &self.in_proj_weight, self.in_proj_bias.as_ref())?; // [batch, seq, nh*vhd]
        let v = v
            .reshape((batch, seq_len, self.num_heads, self.value_head_dim))?
            .transpose(1, 2)?; // [batch, nh, seq, vhd]

        // Apply attention weights to values
        let out = attn_weights.matmul(&v)?; // [batch, nh, seq, vhd]
        let out = out
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.value_head_dim))?;

        let out = self.backend.linear_forward(&out, &self.out_proj_weight, self.out_proj_bias.as_ref())?;
        Ok(out)
    }
}
