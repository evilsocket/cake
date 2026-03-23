//! Mel spectrogram extraction using rustfft.
//!
//! Computes STFT with Hann window, then applies a mel filterbank.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use rustfft::{num_complex::Complex, FftPlanner};

/// Extract log-mel spectrogram from raw audio samples.
///
/// Returns tensor of shape `[1, n_mels, time_frames]`.
#[allow(clippy::too_many_arguments)]
pub fn mel_spectrogram(
    samples: &[f32],
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    sample_rate: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let stft = compute_stft(samples, n_fft, hop_length);
    let n_freq = n_fft / 2 + 1;
    let n_frames = stft.len() / n_freq;

    // Build mel filterbank
    let mel_filters = mel_filterbank(n_mels, n_freq, sample_rate, n_fft);

    // Apply mel filterbank: [n_mels, n_freq] @ [n_freq, n_frames] -> [n_mels, n_frames]
    let mut mel = vec![0.0f32; n_mels * n_frames];
    for m in 0..n_mels {
        for t in 0..n_frames {
            let mut sum = 0.0f32;
            for f in 0..n_freq {
                sum += mel_filters[m * n_freq + f] * stft[t * n_freq + f];
            }
            mel[m * n_frames + t] = sum;
        }
    }

    // Log mel (with floor to avoid log(0))
    for v in &mut mel {
        *v = (*v).max(1e-10).ln();
    }

    // Convert to tensor [1, n_mels, n_frames]
    let mel_tensor = Tensor::from_vec(mel, (1, n_mels, n_frames), device)?.to_dtype(dtype)?;
    Ok(mel_tensor)
}

/// Compute STFT magnitude squared, returned as flat Vec [n_frames * n_freq].
fn compute_stft(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let window = hann_window(n_fft);

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);

    let n_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        0
    };

    // Pre-allocate result and reusable FFT buffer
    let mut result = vec![0.0f32; n_frames * n_freq];
    let mut buffer = vec![Complex::new(0.0f32, 0.0); n_fft];

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_length;

        // Fill buffer with windowed samples (reuse allocation)
        for i in 0..n_fft {
            let sample = if start + i < samples.len() {
                samples[start + i]
            } else {
                0.0
            };
            buffer[i] = Complex::new(sample * window[i], 0.0);
        }

        fft.process(&mut buffer);

        // Magnitude squared for first n_freq bins
        let out_offset = frame_idx * n_freq;
        for (j, item) in buffer.iter().take(n_freq).enumerate() {
            result[out_offset + j] = item.norm_sqr();
        }
    }

    result
}

/// Generate a Hann window of the given size.
fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / size as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Build mel filterbank matrix [n_mels, n_freq].
fn mel_filterbank(n_mels: usize, n_freq: usize, sample_rate: usize, n_fft: usize) -> Vec<f32> {
    let fmin = 0.0f32;
    let fmax = sample_rate as f32 / 2.0;

    // Hz to mel
    let hz_to_mel = |f: f32| -> f32 { 2595.0 * (1.0 + f / 700.0).log10() };
    let mel_to_hz = |m: f32| -> f32 { 700.0 * (10.0f32.powf(m / 2595.0) - 1.0) };

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 equally spaced points in mel space
    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert Hz to FFT bin indices
    let bin_points: Vec<f32> = hz_points
        .iter()
        .map(|&f| f * n_fft as f32 / sample_rate as f32)
        .collect();

    let mut filters = vec![0.0f32; n_mels * n_freq];

    for m in 0..n_mels {
        let f_left = bin_points[m];
        let f_center = bin_points[m + 1];
        let f_right = bin_points[m + 2];

        for k in 0..n_freq {
            let freq = k as f32;
            let weight = if freq >= f_left && freq <= f_center {
                (freq - f_left) / (f_center - f_left + 1e-10)
            } else if freq > f_center && freq <= f_right {
                (f_right - freq) / (f_right - f_center + 1e-10)
            } else {
                0.0
            };
            filters[m * n_freq + k] = weight.max(0.0);
        }
    }

    filters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let w = hann_window(4);
        assert_eq!(w.len(), 4);
        assert!((w[0]).abs() < 1e-6); // Start at 0
        // Periodic Hann(4) = [0, 0.5, 1.0, 0.5]
        assert!((w[1] - 0.5).abs() < 0.01);
        assert!((w[2] - 1.0).abs() < 0.01);
        assert!((w[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = mel_filterbank(100, 513, 24000, 1024);
        assert_eq!(fb.len(), 100 * 513);
        // Each filter should have non-zero values
        for m in 0..100 {
            let row = &fb[m * 513..(m + 1) * 513];
            let sum: f32 = row.iter().sum();
            assert!(sum > 0.0, "mel filter {} has zero sum", m);
        }
    }

    #[test]
    fn test_stft_shape() {
        // 1 second of silence at 24kHz
        let samples = vec![0.0f32; 24000];
        let stft = compute_stft(&samples, 1024, 256);
        let n_freq = 513;
        assert_eq!(stft.len() % n_freq, 0);
        let n_frames = stft.len() / n_freq;
        // Expected: (24000 - 1024) / 256 + 1 = 90
        assert!(n_frames > 0);
    }

    #[test]
    fn test_mel_spectrogram_shape() {
        let samples = vec![0.0f32; 24000];
        let mel = mel_spectrogram(&samples, 1024, 256, 100, 24000, &Device::Cpu, DType::F32).unwrap();
        let dims = mel.shape().dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 100);
        assert!(dims[2] > 0);
    }

    #[test]
    fn test_mel_spectrogram_sine_wave() {
        // 440Hz sine wave, should show energy in the corresponding mel bin
        let sample_rate = 24000;
        let duration = 0.5;
        let n_samples = (sample_rate as f32 * duration) as usize;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let mel = mel_spectrogram(&samples, 1024, 256, 100, sample_rate, &Device::Cpu, DType::F32).unwrap();
        let mel_data = mel.to_vec3::<f32>().unwrap();
        // Should have non-uniform energy distribution across mel bins
        let max_val: f32 = mel_data[0].iter().flat_map(|row| row.iter()).cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val: f32 = mel_data[0].iter().flat_map(|row| row.iter()).cloned().fold(f32::INFINITY, f32::min);
        assert!(max_val > min_val, "mel spectrogram should show energy variation");
    }
}
