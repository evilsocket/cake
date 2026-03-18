//! Integration test: verify the LuxTTS audio pipeline produces audible output.
//!
//! Tests mel spectrogram extraction and WAV saving with known signals.

#[cfg(feature = "luxtts")]
mod luxtts_audio {
    use candle_core::{DType, Device};

    /// Generate a sine wave at the given frequency.
    fn sine_wave(freq: f32, sample_rate: usize, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.8
            })
            .collect()
    }

    /// Generate a multi-note melody for a more interesting test.
    #[allow(clippy::needless_range_loop)]
    fn melody(sample_rate: usize, duration_secs: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * duration_secs) as usize;
        let notes = [
            (261.63f32, 0.0f32, 0.5f32),  // C4
            (293.66, 0.5, 1.0),             // D4
            (329.63, 1.0, 1.5),             // E4
            (349.23, 1.5, 2.0),             // F4
            (392.00, 2.0, 2.5),             // G4
            (440.00, 2.5, 3.0),             // A4
            (493.88, 3.0, 3.5),             // B4
            (523.25, 3.5, 4.0),             // C5
        ];

        let mut samples = vec![0.0f32; n];
        for &(freq, start, end) in &notes {
            let s = (start * sample_rate as f32) as usize;
            let e = (end * sample_rate as f32).min(n as f32) as usize;
            let len = e - s;
            for i in s..e {
                let t = (i - s) as f32 / sample_rate as f32;
                let phase = 2.0 * std::f32::consts::PI * freq * t;
                // Envelope: quick attack, sustain, release
                let pos = (i - s) as f32 / len as f32;
                let env = if pos < 0.05 { pos / 0.05 }
                    else if pos > 0.85 { (1.0 - pos) / 0.15 }
                    else { 1.0 };
                samples[i] = (0.6 * phase.sin() + 0.2 * (2.0 * phase).sin()) * env * 0.5;
            }
        }
        samples
    }

    #[test]
    fn test_mel_spectrogram_roundtrip() {
        // Generate a 440Hz sine wave at 24kHz
        let samples = sine_wave(440.0, 24000, 1.0);
        assert_eq!(samples.len(), 24000);

        // Extract mel spectrogram
        let mel = cake_core::models::luxtts::mel::mel_spectrogram(
            &samples, 1024, 256, 100, 24000, &Device::Cpu, DType::F32,
        ).unwrap();

        let dims = mel.shape().dims();
        assert_eq!(dims[0], 1);   // batch
        assert_eq!(dims[1], 100); // mel bins
        assert!(dims[2] > 0);    // time frames

        // Verify mel values are finite
        let mel_data = mel.to_vec3::<f32>().unwrap();
        for frame in &mel_data[0] {
            for &val in frame {
                assert!(val.is_finite(), "mel value should be finite");
            }
        }
    }

    #[test]
    fn test_save_wav_produces_valid_file() {
        let samples = sine_wave(440.0, 48000, 0.5);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wav");

        cake_core::models::luxtts::vocos::save_wav(&samples, &path, 48000).unwrap();

        // Verify file exists and has correct WAV header
        let data = std::fs::read(&path).unwrap();
        assert!(data.len() > 44, "WAV file should be larger than header");
        assert_eq!(&data[0..4], b"RIFF");
        assert_eq!(&data[8..12], b"WAVE");
        assert_eq!(&data[12..16], b"fmt ");

        // Check sample rate in header (bytes 24-27)
        let sr = u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
        assert_eq!(sr, 48000);

        // Check data size matches
        let data_size = u32::from_le_bytes([data[40], data[41], data[42], data[43]]);
        assert_eq!(data_size as usize, samples.len() * 2);
    }

    #[test]
    fn test_upsample() {
        let samples = vec![0.0f32, 1.0, 0.0, -1.0]; // 4 samples at 1Hz
        let up = cake_core::models::luxtts::vocos::upsample(&samples, 1, 2);
        assert_eq!(up.len(), 8); // doubled
        // Check interpolation: midpoint between 0 and 1 should be ~0.5
        assert!((up[1] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_euler_solver_flow_matching() {
        // Test the flow matching Euler solver.
        // With constant velocity v=1, starting from x=0:
        // After 4 steps from t=0 to t=1, x should reach ~1.0
        use cake_core::models::luxtts::euler_solver::EulerSolver;

        let solver = EulerSolver::new(4, 1.0);
        let times = solver.time_schedule();

        let mut x = candle_core::Tensor::new(&[0.0f32], &Device::Cpu).unwrap();
        for step in 0..4 {
            let t_cur = times[step];
            let t_next = times[step + 1];
            let is_last = step == 3;
            // Constant velocity v = 1.0
            let v = candle_core::Tensor::new(&[1.0f32], &Device::Cpu).unwrap();
            x = EulerSolver::step(&x, &v, t_cur, t_next, is_last).unwrap();
        }

        let final_val = x.to_vec1::<f32>().unwrap()[0];
        // With constant v=1 and flow matching:
        // last step uses x_1_pred = x + (1-t)*v, which should give 1.0
        assert!(
            (final_val - 1.0).abs() < 0.1,
            "Flow matching with constant v should converge to 1.0: got {}",
            final_val
        );
    }

    #[test]
    fn test_generate_melody_wav() {
        // Generate a 4-second melody and save it
        let samples = melody(48000, 4.0);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("melody.wav");

        cake_core::models::luxtts::vocos::save_wav(&samples, &path, 48000).unwrap();

        let data = std::fs::read(&path).unwrap();
        assert!(data.len() > 48000 * 2, "melody WAV should contain substantial audio");
        assert_eq!(&data[0..4], b"RIFF");
    }

    #[test]
    fn test_phonemizer_basic() {
        // Test the rule-based fallback phonemizer (no dictionary needed)
        let dir = tempfile::tempdir().unwrap();
        let tokens_path = dir.path().join("tokens.txt");

        // Write a minimal tokens file
        std::fs::write(&tokens_path, "h\t1\n\u{025B}\t2\nl\t3\nk\t4\n\u{00E6}\t5\nt\t6\n \t7\n").unwrap();

        let p = cake_core::models::luxtts::tokenizer::Phonemizer::load(
            &tokens_path, None
        ).unwrap();

        // Tokenize with rule-based fallback
        let tokens = p.tokenize("cat").unwrap();
        assert!(!tokens.is_empty(), "should produce tokens for 'cat'");
    }
}
