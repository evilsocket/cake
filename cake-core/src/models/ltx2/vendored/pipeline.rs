//! LTX-2 video generation pipeline.
//!
//! Orchestrates: text encoding → noise init → denoising loop → VAE decode.

use candle_core::{Device, Result, Tensor};

/// Pack latents from `[B, C, F, H, W]` to `[B, S, C]` (patchified tokens).
///
/// LTX-2 uses patch_size=1 so this is just a reshape/flatten.
pub fn pack_latents(latents: &Tensor) -> Result<Tensor> {
    let (b, c, f, h, w) = latents.dims5()?;
    // [B, C, F, H, W] -> [B, C, F*H*W] -> [B, F*H*W, C]
    let latents = latents.reshape((b, c, f * h * w))?;
    latents.transpose(1, 2)
}

/// Unpack latents from `[B, S, C]` back to `[B, C, F, H, W]`.
pub fn unpack_latents(
    latents: &Tensor,
    num_frames: usize,
    height: usize,
    width: usize,
) -> Result<Tensor> {
    let (b, _s, c) = latents.dims3()?;
    // [B, S, C] -> [B, C, S] -> [B, C, F, H, W]
    let latents = latents.transpose(1, 2)?;
    latents.reshape((b, c, num_frames, height, width))
}

/// Build 3D positional coordinate grid for video tokens.
///
/// Returns `[B, 3, F*H*W]` where 3 = (time, height, width).
pub fn build_video_positions(
    batch_size: usize,
    num_frames: usize,
    height: usize,
    width: usize,
    temporal_compression: usize,
    spatial_compression: usize,
    frame_rate: usize,
    device: &Device,
) -> Result<Tensor> {
    let total = num_frames * height * width;

    // Build coordinate grids
    let mut t_coords = Vec::with_capacity(total);
    let mut h_coords = Vec::with_capacity(total);
    let mut w_coords = Vec::with_capacity(total);

    let tc = temporal_compression as f32;
    let sc = spatial_compression as f32;
    let fps = frame_rate as f32;
    // causal_offset=1 matches Python's default
    let causal_offset = 1.0f32;

    for f in 0..num_frames {
        for h in 0..height {
            for w in 0..width {
                // Temporal: patch boundary [start, end) in pixel space, then midpoint.
                // Python: pixel = (latent * tc + causal_offset - tc).clamp(min=0)
                // patch_size_t=1, so latent_start=f, latent_end=f+1
                let t_start = (f as f32 * tc + causal_offset - tc).max(0.0);
                let t_end = ((f as f32 + 1.0) * tc + causal_offset - tc).max(0.0);
                t_coords.push((t_start + t_end) / (2.0 * fps));

                // Spatial: patch boundary midpoint.
                // patch_size=1, so latent_start=h, latent_end=h+1
                // pixel = latent * sc, midpoint = (h*sc + (h+1)*sc) / 2
                h_coords.push((h as f32 + 0.5) * sc);
                w_coords.push((w as f32 + 0.5) * sc);
            }
        }
    }

    let t = Tensor::new(t_coords, device)?;
    let h = Tensor::new(h_coords, device)?;
    let w = Tensor::new(w_coords, device)?;

    // Stack to [3, total] then expand to [B, 3, total]
    let grid = Tensor::stack(&[t, h, w], 0)?; // [3, total]
    let grid = grid.unsqueeze(0)?; // [1, 3, total]
    grid.broadcast_as((batch_size, 3, total))?.contiguous()
}

/// Normalize latents using per-channel mean/std.
pub fn normalize_latents(
    latents: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    scaling_factor: f32,
) -> Result<Tensor> {
    let c = latents.dim(1)?;
    let mean = mean
        .reshape((1, c, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let std = std
        .reshape((1, c, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let x = latents.broadcast_sub(&mean)?;
    x.affine(scaling_factor as f64, 0.0)?.broadcast_div(&std)
}

/// Denormalize latents (inverse of normalize_latents).
pub fn denormalize_latents(
    latents: &Tensor,
    mean: &Tensor,
    std: &Tensor,
    scaling_factor: f32,
) -> Result<Tensor> {
    let c = latents.dim(1)?;
    let mean = mean
        .reshape((1, c, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let std = std
        .reshape((1, c, 1, 1, 1))?
        .to_device(latents.device())?
        .to_dtype(latents.dtype())?;
    let x = latents.broadcast_mul(&std)?;
    x.affine((1.0 / scaling_factor) as f64, 0.0)?
        .broadcast_add(&mean)
}

/// Postprocess video tensor from VAE: [-1,1] → [0,255] uint8.
pub fn postprocess_video(video: &Tensor) -> Result<Tensor> {
    let v = video.affine(0.5, 0.5)?; // [-1,1] -> [0,1]
    let v = v.clamp(0.0f32, 1.0f32)?;
    v.affine(255.0, 0.0) // [0,1] -> [0,255]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, IndexOp, Tensor};

    #[test]
    fn test_pack_unpack_roundtrip() {
        let device = Device::Cpu;
        let b = 1;
        let c = 4;
        let f = 2;
        let h = 3;
        let w = 3;

        let latents = Tensor::randn(0f32, 1f32, (b, c, f, h, w), &device).unwrap();
        let packed = pack_latents(&latents).unwrap();

        // packed should be [B, F*H*W, C]
        assert_eq!(packed.dims(), &[b, f * h * w, c]);

        let unpacked = unpack_latents(&packed, f, h, w).unwrap();
        assert_eq!(unpacked.dims(), &[b, c, f, h, w]);

        // Values should roundtrip
        let orig: Vec<f32> = latents.flatten_all().unwrap().to_vec1().unwrap();
        let rt: Vec<f32> = unpacked.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in orig.iter().zip(rt.iter()) {
            assert!((a - b).abs() < 1e-6, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_build_video_positions_shape() {
        let device = Device::Cpu;
        let pos = build_video_positions(2, 3, 4, 5, 8, 32, 25, &device).unwrap();
        // [B, 3, F*H*W]
        assert_eq!(pos.dims(), &[2, 3, 3 * 4 * 5]);
    }

    #[test]
    fn test_build_video_positions_first_frame_midpoint_time() {
        let device = Device::Cpu;
        let pos = build_video_positions(1, 2, 1, 1, 8, 32, 25, &device).unwrap();
        // First frame: t_start = max(0, 0*8+1-8) = 0, t_end = max(0, 1*8+1-8) = 1
        // midpoint = (0 + 1) / (2 * 25) = 0.02
        let t_coords: Vec<f32> = pos.i((0, 0, ..)).unwrap().to_vec1().unwrap();
        assert!((t_coords[0] - 0.02).abs() < 1e-6);
        // Second frame: t_start = max(0, 1*8+1-8) = 1, t_end = max(0, 2*8+1-8) = 9
        // midpoint = (1 + 9) / (2 * 25) = 0.2
        assert!((t_coords[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_build_video_positions_spatial_midpoints() {
        let device = Device::Cpu;
        let pos = build_video_positions(1, 1, 2, 3, 8, 32, 25, &device).unwrap();
        let h_coords: Vec<f32> = pos.i((0, 1, ..)).unwrap().to_vec1().unwrap();
        let w_coords: Vec<f32> = pos.i((0, 2, ..)).unwrap().to_vec1().unwrap();
        // h=0: midpoint = 0.5 * 32 = 16.0, h=1: midpoint = 1.5 * 32 = 48.0
        assert!((h_coords[0] - 16.0).abs() < 1e-4);
        assert!((h_coords[3] - 48.0).abs() < 1e-4);
        // w=0: 16.0, w=1: 48.0, w=2: 80.0
        assert!((w_coords[0] - 16.0).abs() < 1e-4);
        assert!((w_coords[1] - 48.0).abs() < 1e-4);
        assert!((w_coords[2] - 80.0).abs() < 1e-4);
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let device = Device::Cpu;
        let c = 4;
        let latents = Tensor::randn(0f32, 1f32, (1, c, 2, 3, 3), &device).unwrap();
        let mean = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &device).unwrap();
        let std = Tensor::new(vec![1.0f32, 1.5, 0.8, 1.2], &device).unwrap();
        let sf = 1.0;

        let normalized = normalize_latents(&latents, &mean, &std, sf).unwrap();
        let recovered = denormalize_latents(&normalized, &mean, &std, sf).unwrap();

        let orig: Vec<f32> = latents.flatten_all().unwrap().to_vec1().unwrap();
        let rec: Vec<f32> = recovered.flatten_all().unwrap().to_vec1().unwrap();
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-4, "Mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_postprocess_video_range() {
        let device = Device::Cpu;
        // Values in [-1, 1]
        let video = Tensor::new(&[-1.0f32, 0.0, 0.5, 1.0], &device)
            .unwrap()
            .reshape((1, 1, 1, 2, 2))
            .unwrap();
        let result = postprocess_video(&video).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert!((vals[0] - 0.0).abs() < 1e-4);    // -1 -> 0
        assert!((vals[1] - 127.5).abs() < 1e-4);   // 0 -> 127.5
        assert!((vals[2] - 191.25).abs() < 1e-4);   // 0.5 -> 191.25
        assert!((vals[3] - 255.0).abs() < 1e-4);    // 1 -> 255
    }
}
