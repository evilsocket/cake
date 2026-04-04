use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Precompute 3D RoPE frequencies for Wan's (temporal, height, width) split.
///
/// Wan splits head_dim=128 into: t_dim=44, h_dim=42, w_dim=42.
/// Each axis gets its own set of sinusoidal frequencies.
pub fn precompute_wan_rope_3d(
    num_frames: usize,
    height: usize,
    width: usize,
    t_dim: usize,
    h_dim: usize,
    w_dim: usize,
    theta: f64,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Compute 1D frequencies for each axis
    let freqs_t = compute_1d_freqs(t_dim / 2, theta, num_frames, device)?;
    let freqs_h = compute_1d_freqs(h_dim / 2, theta, height, device)?;
    let freqs_w = compute_1d_freqs(w_dim / 2, theta, width, device)?;

    // freqs_t: [F, t_dim/2], freqs_h: [H, h_dim/2], freqs_w: [W, w_dim/2]
    // Expand to full sequence: [F*H*W, head_dim/2]
    let seq_len = num_frames * height * width;

    // Build position indices for the 3D grid
    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();

    // Python uses repeat_interleave(2) on each axis's cos/sin, doubling the dim.
    // E.g. t_dim/2 frequencies → t_dim values via [c0, c0, c1, c1, ...].
    // This matches the interleaved RoPE application in apply_wan_rope.

    // Temporal component: each frame position repeated H*W times
    let cos_t = repeat_interleave_2(&freqs_t.cos()?)?; // [F, t_dim]
    let sin_t = repeat_interleave_2(&freqs_t.sin()?)?;
    let cos_t = cos_t
        .unsqueeze(1)?.unsqueeze(1)?
        .broadcast_as((num_frames, height, width, t_dim))?
        .reshape((seq_len, t_dim))?;
    let sin_t = sin_t
        .unsqueeze(1)?.unsqueeze(1)?
        .broadcast_as((num_frames, height, width, t_dim))?
        .reshape((seq_len, t_dim))?;
    cos_parts.push(cos_t);
    sin_parts.push(sin_t);

    // Height component
    let cos_h = repeat_interleave_2(&freqs_h.cos()?)?; // [H, h_dim]
    let sin_h = repeat_interleave_2(&freqs_h.sin()?)?;
    let cos_h = cos_h
        .unsqueeze(0)?.unsqueeze(2)?
        .broadcast_as((num_frames, height, width, h_dim))?
        .reshape((seq_len, h_dim))?;
    let sin_h = sin_h
        .unsqueeze(0)?.unsqueeze(2)?
        .broadcast_as((num_frames, height, width, h_dim))?
        .reshape((seq_len, h_dim))?;
    cos_parts.push(cos_h);
    sin_parts.push(sin_h);

    // Width component
    let cos_w = repeat_interleave_2(&freqs_w.cos()?)?; // [W, w_dim]
    let sin_w = repeat_interleave_2(&freqs_w.sin()?)?;
    let cos_w = cos_w
        .unsqueeze(0)?.unsqueeze(0)?
        .broadcast_as((num_frames, height, width, w_dim))?
        .reshape((seq_len, w_dim))?;
    let sin_w = sin_w
        .unsqueeze(0)?.unsqueeze(0)?
        .broadcast_as((num_frames, height, width, w_dim))?
        .reshape((seq_len, w_dim))?;
    cos_parts.push(cos_w);
    sin_parts.push(sin_w);

    // Concatenate along last dim: [seq_len, head_dim] (full dim, with repeat_interleave)
    let cos = Tensor::cat(&cos_parts.iter().collect::<Vec<_>>(), 1)?;
    let sin = Tensor::cat(&sin_parts.iter().collect::<Vec<_>>(), 1)?;

    // Reshape to [1, seq_len, 1, head_dim] for broadcasting with [B, S, H, D]
    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    Ok((cos, sin))
}

/// Repeat-interleave along last dimension: [a, b, c] -> [a, a, b, b, c, c].
/// Matches PyTorch's `tensor.repeat_interleave(2, dim=-1)`.
fn repeat_interleave_2(x: &Tensor) -> Result<Tensor> {
    let shape = x.shape().dims();
    let last = shape.len() - 1;
    // Stack with itself on a new last dim, then flatten the last two dims
    let stacked = Tensor::stack(&[x, x], last + 1)?; // [..., D, 2]
    let mut new_shape: Vec<usize> = shape[..last].to_vec();
    new_shape.push(shape[last] * 2);
    Ok(stacked.reshape(new_shape)?)
}

/// Compute 1D sinusoidal frequency embeddings.
/// Returns [max_pos, dim] tensor of angle values (before cos/sin).
fn compute_1d_freqs(
    dim: usize,
    theta: f64,
    max_pos: usize,
    device: &Device,
) -> Result<Tensor> {
    let inv_freq: Vec<f32> = (0..dim)
        .map(|i| 1.0 / theta.powf(i as f64 / dim as f64) as f32)
        .collect();
    let inv_freq = Tensor::new(inv_freq.as_slice(), device)?; // [dim]

    let positions: Vec<f32> = (0..max_pos).map(|p| p as f32).collect();
    let positions = Tensor::new(positions.as_slice(), device)?; // [max_pos]

    // Outer product: [max_pos, dim]
    let freqs = positions
        .unsqueeze(1)?
        .broadcast_mul(&inv_freq.unsqueeze(0)?)?;

    Ok(freqs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_precompute_wan_rope_3d_shape() {
        let dev = &Device::Cpu;
        let (cos, sin) = precompute_wan_rope_3d(
            4, 3, 3,  // F=4, H=3, W=3
            44, 42, 42,
            10000.0,
            dev,
        ).unwrap();
        // Expected shape: [1, F*H*W, 1, head_dim] = [1, 36, 1, 128]
        // (full head_dim with repeat_interleave(2))
        let seq_len = 4 * 3 * 3;
        let head_dim = 44 + 42 + 42;
        assert_eq!(cos.dims(), &[1, seq_len, 1, head_dim]);
        assert_eq!(sin.dims(), &[1, seq_len, 1, head_dim]);
    }

    #[test]
    fn test_apply_wan_rope_shape() {
        let dev = &Device::Cpu;
        let (b, s, h, d) = (1, 12, 4, 128);
        let x = Tensor::randn(0f32, 1.0, (b, s, h, d), dev).unwrap();
        let cos = Tensor::ones((1, s, 1, d), DType::F32, dev).unwrap();
        let sin = Tensor::zeros((1, s, 1, d), DType::F32, dev).unwrap();

        let result = apply_wan_rope(&x, &cos, &sin).unwrap();
        assert_eq!(result.dims(), &[b, s, h, d]);
    }

    #[test]
    fn test_apply_wan_rope_identity_with_zero_sin() {
        // When sin=0 and cos=1, RoPE should be identity
        let dev = &Device::Cpu;
        let (b, s, h, d) = (1, 4, 2, 8);
        let x = Tensor::randn(0f32, 1.0, (b, s, h, d), dev).unwrap();
        let cos = Tensor::ones((1, s, 1, d), DType::F32, dev).unwrap();
        let sin = Tensor::zeros((1, s, 1, d), DType::F32, dev).unwrap();

        let result = apply_wan_rope(&x, &cos, &sin).unwrap();
        let diff = (result - &x).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(candle_core::D::Minus1).unwrap()
            .max(candle_core::D::Minus1).unwrap()
            .max(candle_core::D::Minus1).unwrap()
            .max(candle_core::D::Minus1).unwrap()
            .to_scalar::<f32>().unwrap();
        assert!(max_diff < 1e-6, "identity rotation should preserve input, max_diff={max_diff}");
    }

    #[test]
    fn test_compute_1d_freqs_shape() {
        let dev = &Device::Cpu;
        let freqs = compute_1d_freqs(21, 10000.0, 10, dev).unwrap();
        assert_eq!(freqs.dims(), &[10, 21]);
    }

    #[test]
    fn test_compute_1d_freqs_first_position_is_zero() {
        let dev = &Device::Cpu;
        let freqs = compute_1d_freqs(4, 10000.0, 5, dev).unwrap();
        // Position 0 should have all-zero frequencies
        let row0 = freqs.get(0).unwrap().to_vec1::<f32>().unwrap();
        for (i, &v) in row0.iter().enumerate() {
            assert!(v.abs() < 1e-7, "freqs[0][{i}] should be 0, got {v}");
        }
    }
}

/// Apply RoPE to Q or K tensor (matches diffusers apply_rotary_emb with use_real=True).
/// Input: [B, S, H, D] where D = head_dim.
/// cos, sin: [1, S, 1, D] (full head_dim, with repeat_interleave(2) applied).
///
/// For each consecutive pair (x_real, x_imag) at positions (2i, 2i+1):
///   out_real = x_real * cos - x_imag * sin
///   out_imag = x_imag * cos + x_real * sin
pub fn apply_wan_rope(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<Tensor> {
    let (_b, _s, _h, d) = x.dims4()?;
    let half_d = d / 2;

    let cos = cos.to_dtype(x.dtype())?;
    let sin = sin.to_dtype(x.dtype())?;

    // Split x into real/imag pairs: reshape to [..., D/2, 2], unbind last dim
    let x_pairs = x.reshape((_b, _s, _h, half_d, 2))?;
    let x_real = x_pairs.narrow(4, 0, 1)?.squeeze(4)?; // [B, S, H, D/2]
    let x_imag = x_pairs.narrow(4, 1, 1)?.squeeze(4)?;

    // Build rotated version: [-x_imag, x_real] interleaved
    let neg_x_imag = x_imag.neg()?;
    let x_rotated = Tensor::stack(&[&neg_x_imag, &x_real], 4)?.reshape((_b, _s, _h, d))?;

    // out = x * cos + x_rotated * sin
    let out = (x.broadcast_mul(&cos)?.to_dtype(candle_core::DType::F32)?
        + x_rotated.broadcast_mul(&sin)?.to_dtype(candle_core::DType::F32)?)?
        .to_dtype(x.dtype())?;

    Ok(out)
}
