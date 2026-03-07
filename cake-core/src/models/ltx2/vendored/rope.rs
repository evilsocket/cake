//! Rotary Position Embeddings for LTX-2.
//!
//! Matches HF `LTX2AudioVideoRotaryPosEmbed` and `apply_split_rotary_emb`.
//!
//! Split RoPE: frequencies are reshaped per-head `[B, H, T, D_per_head//2]`.
//! Each head's embedding is independently rotated in halves.

use candle_core::{DType, Device, Result, Tensor};

/// Precompute split RoPE (cos, sin) for the given positions grid.
///
/// `indices_grid`: `[B, n_pos_dims, T]` — positional coordinates per token.
///   For video: n_pos_dims=3 (time, height, width).
/// `dim`: total head dimension = heads * d_head.
/// `theta`: base frequency (default 10000).
/// `max_pos`: max position per dimension (for fractional scaling).
/// `num_heads`: number of attention heads for per-head reshape.
///
/// Returns `(cos, sin)` each of shape `[B, H, T, D_per_head//2]`.
pub fn precompute_freqs_cis(
    indices_grid: &Tensor,
    dim: usize,
    theta: f32,
    max_pos: &[usize],
    num_heads: usize,
    out_dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let device = indices_grid.device();
    let (_b, n_pos_dims, _t) = indices_grid.dims3()?;

    // num_rope_elems = n_pos_dims * 2
    let num_rope_elems = n_pos_dims * 2;
    let dim_per_pos = dim / num_rope_elems;

    let freqs = generate_freq_grid(theta, dim_per_pos, device)?;

    // [B, n_pos_dims, T] -> [B, T, n_pos_dims]
    let grid = indices_grid.transpose(1, 2)?;

    // Fractional positions: divide by max_pos, then map to [-1, 1]
    let max_pos_t = Tensor::new(
        max_pos.iter().map(|&m| m as f32).collect::<Vec<_>>(),
        device,
    )?;
    let frac_pos = grid.broadcast_div(&max_pos_t)?;
    let scaled = frac_pos.affine(2.0, -1.0)?;

    // Outer product: [B, T, n_pos_dims, dim_per_pos]
    let scaled_unsq = scaled.unsqueeze(3)?;
    let freqs_unsq = freqs.unsqueeze(0)?.unsqueeze(0)?.unsqueeze(0)?;
    let freqs_out = scaled_unsq.broadcast_mul(&freqs_unsq)?;

    // transpose(-1, -2): [B, T, dim_per_pos, n_pos_dims]
    // flatten(2): [B, T, dim_per_pos * n_pos_dims] = [B, T, dim//2 - maybe]
    let freqs_out = freqs_out.transpose(2, 3)?.flatten_from(2)?;

    let cos_raw = freqs_out.cos()?;
    let sin_raw = freqs_out.sin()?;

    // Pad to expected_freqs = dim // 2 (PREPEND padding)
    let expected_freqs = dim / 2;
    let current_freqs = cos_raw.dim(2)?;
    let pad_size = expected_freqs - current_freqs;

    let (cos, sin) = if pad_size > 0 {
        let b_size = cos_raw.dim(0)?;
        let t_size = cos_raw.dim(1)?;
        let cos_pad = Tensor::ones((b_size, t_size, pad_size), DType::F32, device)?;
        let sin_pad = Tensor::zeros((b_size, t_size, pad_size), DType::F32, device)?;
        // PREPEND: [pad, raw] (Python does concatenate([padding, freq], axis=-1))
        (
            Tensor::cat(&[cos_pad, cos_raw], 2)?,
            Tensor::cat(&[sin_pad, sin_raw], 2)?,
        )
    } else {
        (cos_raw, sin_raw)
    };

    // Reshape per-head: [B, T, dim//2] -> [B, T, H, D_per_head//2] -> [B, H, T, D_per_head//2]
    let d_per_head_half = expected_freqs / num_heads;
    let (b, t, _) = cos.dims3()?;
    let cos = cos
        .reshape((b, t, num_heads, d_per_head_half))?
        .transpose(1, 2)?
        .contiguous()?;
    let sin = sin
        .reshape((b, t, num_heads, d_per_head_half))?
        .transpose(1, 2)?
        .contiguous()?;

    Ok((cos.to_dtype(out_dtype)?, sin.to_dtype(out_dtype)?))
}

/// Apply split RoPE to input tensor.
///
/// Matches HF `apply_split_rotary_emb`:
/// - cos/sin: `[B, H, T, r]` where r = D_per_head // 2
/// - x: `[B, T, D]` (flat) — reshaped to `[B, H, T, D_per_head]` internally
/// - Per-head: split D_per_head into [2, r], rotate halves independently.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let needs_reshape = x.rank() == 3 && cos.rank() == 4;

    let (b, h, t, d_per_head) = if needs_reshape {
        let b = cos.dim(0)?;
        let h = cos.dim(1)?;
        let t = cos.dim(2)?;
        let d_per_head = x.dim(2)? / h;
        (b, h, t, d_per_head)
    } else {
        // Already 4D: [B, H, T, D_per_head]
        (x.dim(0)?, x.dim(1)?, x.dim(2)?, x.dim(3)?)
    };

    let r = d_per_head / 2;

    // Get x in [B, H, T, D_per_head] shape
    let x4d = if needs_reshape {
        x.reshape((b, t, h, d_per_head))?.transpose(1, 2)?
    } else {
        x.clone()
    };

    // Split: [B, H, T, D_per_head] -> [B, H, T, 2, r]
    let split_x = x4d.reshape((b, h, t, 2, r))?.to_dtype(DType::F32)?;
    let first_x = split_x.narrow(3, 0, 1)?; // [B, H, T, 1, r]
    let second_x = split_x.narrow(3, 1, 1)?; // [B, H, T, 1, r]

    // cos/sin: [B, H, T, r] -> [B, H, T, 1, r]
    let cos_f = cos.to_dtype(DType::F32)?.unsqueeze(3)?;
    let sin_f = sin.to_dtype(DType::F32)?.unsqueeze(3)?;

    // out = split_x * cos (element-wise broadcast)
    let out = split_x.broadcast_mul(&cos_f)?; // [B, H, T, 2, r]

    // first_out = first_x * cos - second_x * sin
    let first_out = out.narrow(3, 0, 1)?;
    let first_out = first_out.broadcast_sub(&sin_f.broadcast_mul(&second_x)?)?;

    // second_out = second_x * cos + first_x * sin
    let second_out = out.narrow(3, 1, 1)?;
    let second_out = second_out.broadcast_add(&sin_f.broadcast_mul(&first_x)?)?;

    // Concat: [B, H, T, 2, r]
    let out = Tensor::cat(&[first_out, second_out], 3)?;

    // Reshape: [B, H, T, 2, r] -> [B, H, T, D_per_head]
    let out = out.reshape((b, h, t, d_per_head))?;

    // If we reshaped, convert back: [B, H, T, D] -> [B, T, H, D] -> [B, T, H*D]
    let out = if needs_reshape {
        out.transpose(1, 2)?.reshape((b, t, h * d_per_head))?
    } else {
        out
    };

    out.to_dtype(x.dtype())
}

/// Generate log-spaced frequency grid: pow(theta, linspace(0, 1, steps)) * pi/2.
fn generate_freq_grid(theta: f32, dim_per_pos: usize, device: &Device) -> Result<Tensor> {
    let end = theta as f64;

    let indices: Vec<f32> = (0..dim_per_pos)
        .map(|i| {
            let t = if dim_per_pos > 1 {
                i as f64 / (dim_per_pos - 1) as f64
            } else {
                0.0
            };
            let val = end.powf(t) * std::f64::consts::FRAC_PI_2;
            val as f32
        })
        .collect();

    Tensor::from_vec(indices, (dim_per_pos,), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    #[test]
    fn test_precompute_freqs_shape() {
        let device = Device::Cpu;
        let batch = 2;
        let seq = 12;
        let heads = 32;
        let d_head = 128;
        let dim = heads * d_head;
        let n_pos = 3;

        let grid = Tensor::randn(0f32, 1f32, (batch, n_pos, seq), &device).unwrap();
        let max_pos = vec![20, 2048, 2048];

        let (cos, sin) =
            precompute_freqs_cis(&grid, dim, 10000.0, &max_pos, heads, DType::F32).unwrap();

        // [B, H, T, D_per_head//2]
        assert_eq!(cos.dims(), &[batch, heads, seq, d_head / 2]);
        assert_eq!(sin.dims(), &[batch, heads, seq, d_head / 2]);
    }

    #[test]
    fn test_apply_rotary_emb_identity() {
        // cos=1, sin=0 should be identity
        let device = Device::Cpu;
        let b = 1;
        let t = 4;
        let h = 2;
        let d_head = 8;
        let dim = h * d_head;
        let r = d_head / 2;

        let x = Tensor::randn(0f32, 1f32, (b, t, dim), &device).unwrap();
        let cos = Tensor::ones((b, h, t, r), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((b, h, t, r), DType::F32, &device).unwrap();

        let out = apply_rotary_emb(&x, &cos, &sin).unwrap();
        assert_eq!(out.dims(), &[b, t, dim]);

        let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let o_vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in x_vals.iter().zip(o_vals.iter()) {
            assert!((a - b).abs() < 1e-5, "Identity RoPE failed: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_apply_rotary_emb_rotation() {
        let device = Device::Cpu;
        // 1 head, d_head=4, r=2
        // x = [1, 0, 0, 0] (first_half=[1,0], second_half=[0,0])
        // cos=0, sin=1:
        //   first_out = first*cos - sin*second = [0,0] - [0,0] = [0,0]
        //   second_out = second*cos + sin*first = [0,0] + [1,0] = [1,0]
        //   result = [0, 0, 1, 0]
        let x = Tensor::new(&[1.0f32, 0.0, 0.0, 0.0], &device)
            .unwrap()
            .reshape((1, 1, 4))
            .unwrap();
        let cos = Tensor::new(&[0.0f32, 0.0], &device)
            .unwrap()
            .reshape((1, 1, 1, 2))
            .unwrap();
        let sin = Tensor::new(&[1.0f32, 1.0], &device)
            .unwrap()
            .reshape((1, 1, 1, 2))
            .unwrap();

        let out = apply_rotary_emb(&x, &cos, &sin).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 0.0).abs() < 1e-6);
        assert!((vals[1] - 0.0).abs() < 1e-6);
        assert!((vals[2] - 1.0).abs() < 1e-6);
        assert!((vals[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_apply_rotary_emb_4d() {
        // Already 4D input: [B, H, T, D_per_head]
        let device = Device::Cpu;
        let b = 1;
        let h = 2;
        let t = 3;
        let d_head = 8;
        let r = d_head / 2;

        let x = Tensor::randn(0f32, 1f32, (b, h, t, d_head), &device).unwrap();
        let cos = Tensor::ones((b, h, t, r), DType::F32, &device).unwrap();
        let sin = Tensor::zeros((b, h, t, r), DType::F32, &device).unwrap();

        let out = apply_rotary_emb(&x, &cos, &sin).unwrap();
        assert_eq!(out.dims(), &[b, h, t, d_head]);

        // Identity check
        let x_vals: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let o_vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in x_vals.iter().zip(o_vals.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}
