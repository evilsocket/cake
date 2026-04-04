use anyhow::Result;
use candle_core::Tensor;

use super::config::{LATENTS_MEAN, LATENTS_STD};

/// Denormalize latents before VAE decode.
/// Wan uses: latents = latents / latents_std + latents_mean
/// where latents_std is stored as 1/std (so division reverses the normalization).
pub fn denormalize_latents(latents: &Tensor) -> Result<Tensor> {
    let device = latents.device();
    let dtype = latents.dtype();

    let mean = Tensor::new(&LATENTS_MEAN, device)?.to_dtype(dtype)?;
    let std = Tensor::new(&LATENTS_STD, device)?.to_dtype(dtype)?;

    // latents: [B, C, F, H, W], mean/std: [C]
    // Reshape to [1, C, 1, 1, 1] for broadcasting
    let mean = mean.reshape((1, 16, 1, 1, 1))?;
    let std = std.reshape((1, 16, 1, 1, 1))?;

    // denorm = latents / std + mean
    // (std here is actually the std, not 1/std — multiply by std to undo normalization)
    Ok(latents.broadcast_mul(&std)?.broadcast_add(&mean)?)
}

/// Compute the number of latent frames from video frames.
/// temporal_compression = 4: latent_frames = (frames - 1) / 4 + 1
pub fn num_latent_frames(video_frames: usize, temporal_compression: usize) -> usize {
    (video_frames - 1) / temporal_compression + 1
}

/// Compute latent spatial dimensions from pixel dimensions.
/// spatial_compression = 8: latent_h = h / 8, latent_w = w / 8
pub fn latent_spatial(height: usize, width: usize, spatial_compression: usize) -> (usize, usize) {
    (height / spatial_compression, width / spatial_compression)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, DType, Tensor};

    #[test]
    fn test_num_latent_frames() {
        // temporal_compression = 4: (frames - 1) / 4 + 1
        assert_eq!(num_latent_frames(81, 4), 21); // (81-1)/4+1 = 21
        assert_eq!(num_latent_frames(1, 4), 1);   // single frame
        assert_eq!(num_latent_frames(5, 4), 2);   // (5-1)/4+1 = 2
        assert_eq!(num_latent_frames(4, 4), 1);   // (4-1)/4+1 = 1
    }

    #[test]
    fn test_latent_spatial() {
        assert_eq!(latent_spatial(480, 832, 8), (60, 104));
        assert_eq!(latent_spatial(512, 704, 8), (64, 88));
        assert_eq!(latent_spatial(720, 1280, 8), (90, 160));
    }

    #[test]
    fn test_denormalize_latents_shape() {
        let dev = &Device::Cpu;
        // (B, C=16, F, H, W)
        let latents = Tensor::zeros((1, 16, 5, 8, 8), DType::F32, dev).unwrap();
        let result = denormalize_latents(&latents).unwrap();
        assert_eq!(result.dims(), &[1, 16, 5, 8, 8]);
    }

    #[test]
    fn test_denormalize_latents_applies_mean_std() {
        let dev = &Device::Cpu;
        // Zero latents: denorm = 0 * std + mean = mean
        let latents = Tensor::zeros((1, 16, 1, 1, 1), DType::F32, dev).unwrap();
        let result = denormalize_latents(&latents).unwrap();
        // Check channel 0: should equal LATENTS_MEAN[0] = -0.7571
        let val: f32 = result.flatten_all().unwrap().to_vec1::<f32>().unwrap()[0];
        assert!((val - LATENTS_MEAN[0]).abs() < 1e-5,
            "channel 0 should be mean={}, got {}", LATENTS_MEAN[0], val);
    }
}
