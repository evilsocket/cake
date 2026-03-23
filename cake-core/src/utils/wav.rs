//! WAV file encoding, decoding, and I/O utilities.
//!
//! All functions operate on PCM f32 samples in the range \[-1.0, 1.0\].
//! WAV output is always 16-bit PCM mono.

use std::path::Path;

use anyhow::{bail, Result};

/// Encode PCM f32 samples as WAV bytes (16-bit PCM, mono).
pub fn encode_wav_bytes(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono = 2 bytes per sample
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    let mut buf = Vec::with_capacity(file_size as usize + 8);
    // RIFF header
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_size.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    // fmt chunk
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    buf.extend_from_slice(&1u16.to_le_bytes()); // mono
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&2u16.to_le_bytes()); // block align
    buf.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    // data chunk
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_size.to_le_bytes());
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        let i = (clamped * 32767.0) as i16;
        buf.extend_from_slice(&i.to_le_bytes());
    }
    buf
}

/// Save PCM f32 samples as a WAV file (16-bit PCM, mono).
pub fn save_wav(samples: &[f32], path: &Path, sample_rate: u32) -> Result<()> {
    use std::io::{BufWriter, Write};
    let data_size = (samples.len() * 2) as u32;
    let mut f = BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_size).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&1u16.to_le_bytes())?; // Mono
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&(sample_rate * 2).to_le_bytes())?;
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_size.to_le_bytes())?;
    for &s in samples {
        f.write_all(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())?;
    }
    Ok(())
}

/// Load a WAV file and return mono PCM f32 samples at the given target sample rate.
///
/// Supports 16-bit integer and 32-bit float WAV files, mono or stereo.
/// Stereo is mixed down to mono. If the file's sample rate differs from
/// `target_sample_rate`, linear interpolation resampling is applied.
pub fn load_wav_mono(path: &Path, target_sample_rate: u32) -> Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    decode_wav_mono(&buf, target_sample_rate)
}

/// Decode WAV bytes and return mono PCM f32 samples at the given target sample rate.
///
/// Same processing as [`load_wav_mono`] but operates on in-memory bytes.
pub fn decode_wav_mono(buf: &[u8], target_sample_rate: u32) -> Result<Vec<f32>> {
    if buf.len() < 44 || &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        bail!("Not a valid WAV file");
    }

    // Parse chunks to find fmt and data
    let mut pos = 12;
    let mut data_start = 0;
    let mut data_size = 0u32;
    let mut channels = 1u16;
    let mut sample_rate = target_sample_rate;
    let mut bits_per_sample = 16u16;

    while pos + 8 <= buf.len() {
        let chunk_id = &buf[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([buf[pos + 4], buf[pos + 5], buf[pos + 6], buf[pos + 7]]);
        if chunk_id == b"fmt " {
            if pos + 24 > buf.len() {
                bail!("fmt chunk too short");
            }
            channels = u16::from_le_bytes([buf[pos + 10], buf[pos + 11]]);
            sample_rate =
                u32::from_le_bytes([buf[pos + 12], buf[pos + 13], buf[pos + 14], buf[pos + 15]]);
            bits_per_sample = u16::from_le_bytes([buf[pos + 22], buf[pos + 23]]);
        } else if chunk_id == b"data" {
            data_start = pos + 8;
            data_size = chunk_size;
            break;
        }
        pos += 8 + chunk_size as usize;
        if pos % 2 != 0 {
            pos += 1; // WAV chunks are word-aligned
        }
    }

    if data_start == 0 {
        bail!("No data chunk in WAV file");
    }

    let end = (data_start + data_size as usize).min(buf.len());
    let data = &buf[data_start..end];

    // Convert to f32 samples
    let mut samples = match bits_per_sample {
        16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect::<Vec<f32>>(),
        32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>(),
        other => bail!("Unsupported bits_per_sample: {}", other),
    };

    // Mix to mono if stereo
    if channels == 2 {
        samples = samples
            .chunks(2)
            .map(|c| (c[0] + c.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect();
    }

    // Resample to target rate if needed (linear interpolation)
    if sample_rate != target_sample_rate {
        let ratio = target_sample_rate as f64 / sample_rate as f64;
        let new_len = (samples.len() as f64 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);
        for i in 0..new_len {
            let src_pos = i as f64 / ratio;
            let idx = src_pos as usize;
            let frac = (src_pos - idx as f64) as f32;
            let s0 = samples.get(idx).copied().unwrap_or(0.0);
            let s1 = samples.get(idx + 1).copied().unwrap_or(s0);
            resampled.push(s0 + frac * (s1 - s0));
        }
        samples = resampled;
    }

    Ok(samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── encode_wav_bytes ────────────────────────────────────────

    #[test]
    fn test_encode_wav_valid_riff() {
        let wav = encode_wav_bytes(&[0.0; 100], 24000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(&wav[12..16], b"fmt ");
        assert_eq!(&wav[36..40], b"data");
    }

    #[test]
    fn test_encode_wav_correct_sizes() {
        let wav = encode_wav_bytes(&[0.5; 50], 24000);
        let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_size, 100); // 50 samples * 2 bytes
        let file_size = u32::from_le_bytes([wav[4], wav[5], wav[6], wav[7]]);
        assert_eq!(file_size, 136); // 36 + 100
        assert_eq!(wav.len(), 144); // 8 + 136
    }

    #[test]
    fn test_encode_wav_sample_rate() {
        let wav = encode_wav_bytes(&[0.0], 48000);
        let sr = u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]);
        assert_eq!(sr, 48000);
    }

    #[test]
    fn test_encode_wav_clamps() {
        let wav = encode_wav_bytes(&[2.0, -2.0], 24000);
        let s0 = i16::from_le_bytes([wav[44], wav[45]]);
        let s1 = i16::from_le_bytes([wav[46], wav[47]]);
        assert_eq!(s0, 32767);
        assert_eq!(s1, -32767);
    }

    #[test]
    fn test_encode_wav_empty() {
        let wav = encode_wav_bytes(&[], 24000);
        assert_eq!(wav.len(), 44); // header only
        let data_size = u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]);
        assert_eq!(data_size, 0);
    }

    #[test]
    fn test_encode_wav_silence() {
        let wav = encode_wav_bytes(&[0.0; 10], 24000);
        // All sample bytes should be zero
        for i in (44..wav.len()).step_by(2) {
            let s = i16::from_le_bytes([wav[i], wav[i + 1]]);
            assert_eq!(s, 0);
        }
    }

    #[test]
    fn test_encode_wav_byte_rate() {
        let wav = encode_wav_bytes(&[0.0], 44100);
        let byte_rate = u32::from_le_bytes([wav[28], wav[29], wav[30], wav[31]]);
        assert_eq!(byte_rate, 44100 * 2); // 16-bit mono
    }

    // ── save_wav ────────────────────────────────────────────────

    #[test]
    fn test_save_wav_creates_valid_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");
        save_wav(&[0.5, -0.5, 0.0], &path, 24000).unwrap();

        let data = std::fs::read(&path).unwrap();
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");
        let sr = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        assert_eq!(sr, 24000);
        let data_size = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);
        assert_eq!(data_size, 6); // 3 samples * 2 bytes
    }

    #[test]
    fn test_save_wav_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.wav");
        save_wav(&[], &path, 48000).unwrap();
        let data = std::fs::read(&path).unwrap();
        assert_eq!(data.len(), 44);
    }

    #[test]
    fn test_save_wav_bad_path() {
        let result = save_wav(&[0.0], Path::new("/nonexistent/dir/out.wav"), 24000);
        assert!(result.is_err());
    }

    // ── decode_wav_mono ─────────────────────────────────────────

    #[test]
    fn test_decode_roundtrip_16bit() {
        // Encode then decode — samples should survive the i16 roundtrip
        let original = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav_bytes(&original, 24000);
        let decoded = decode_wav_mono(&wav, 24000).unwrap();
        assert_eq!(decoded.len(), original.len());
        for (o, d) in original.iter().zip(decoded.iter()) {
            // 16-bit quantization error: max ~1/32768 ≈ 3e-5
            assert!(
                (o - d).abs() < 0.001,
                "expected ~{}, got {}",
                o,
                d
            );
        }
    }

    #[test]
    fn test_decode_invalid_header() {
        assert!(decode_wav_mono(b"NOT_A_WAV_FILE_AT_ALL!!!!!!!!!!!!!!!!!!!!!!!", 24000).is_err());
    }

    #[test]
    fn test_decode_too_short() {
        assert!(decode_wav_mono(b"RIFF", 24000).is_err());
    }

    #[test]
    fn test_decode_missing_data_chunk() {
        // Valid RIFF+WAVE but no data chunk
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RIFF");
        buf.extend_from_slice(&36u32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // mono
        buf.extend_from_slice(&24000u32.to_le_bytes());
        buf.extend_from_slice(&48000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        assert!(decode_wav_mono(&buf, 24000).is_err());
    }

    #[test]
    fn test_decode_resamples() {
        // 48kHz file loaded at 24kHz target should halve the sample count
        let samples: Vec<f32> = (0..4800).map(|i| (i as f32 / 4800.0).sin()).collect();
        let wav = encode_wav_bytes(&samples, 48000);
        let decoded = decode_wav_mono(&wav, 24000).unwrap();
        assert_eq!(decoded.len(), 2400);
    }

    #[test]
    fn test_decode_upsample() {
        // 24kHz file loaded at 48kHz should double the sample count
        let samples: Vec<f32> = (0..2400).map(|i| (i as f32 / 2400.0).sin()).collect();
        let wav = encode_wav_bytes(&samples, 24000);
        let decoded = decode_wav_mono(&wav, 48000).unwrap();
        assert_eq!(decoded.len(), 4800);
    }

    #[test]
    fn test_decode_same_rate_no_resample() {
        let samples = vec![0.1, 0.2, 0.3];
        let wav = encode_wav_bytes(&samples, 24000);
        let decoded = decode_wav_mono(&wav, 24000).unwrap();
        assert_eq!(decoded.len(), 3);
    }

    // ── load_wav_mono ───────────────────────────────────────────

    #[test]
    fn test_load_wav_mono_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.wav");
        let original = vec![0.0, 0.25, -0.25, 0.5, -0.5];
        save_wav(&original, &path, 24000).unwrap();
        let loaded = load_wav_mono(&path, 24000).unwrap();
        assert_eq!(loaded.len(), original.len());
        for (o, l) in original.iter().zip(loaded.iter()) {
            assert!((o - l).abs() < 0.001);
        }
    }

    #[test]
    fn test_load_wav_mono_missing_file() {
        assert!(load_wav_mono(Path::new("/nonexistent/audio.wav"), 24000).is_err());
    }

    #[test]
    fn test_load_wav_mono_resample_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("48k.wav");
        let samples: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.01).sin()).collect();
        save_wav(&samples, &path, 48000).unwrap();
        let loaded = load_wav_mono(&path, 24000).unwrap();
        // Should be ~half the samples
        assert_eq!(loaded.len(), 2400);
    }
}
