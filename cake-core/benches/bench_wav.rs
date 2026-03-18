use cake_core::utils::wav;

fn sine_samples(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * 0.1).sin() * 0.8)
        .collect()
}

#[divan::bench(args = [1024, 24000, 240000])]
fn encode_wav_bytes(bencher: divan::Bencher, num_samples: usize) {
    let samples = sine_samples(num_samples);
    bencher
        .counter(divan::counter::BytesCount::new(num_samples * 2))
        .bench_local(|| wav::encode_wav_bytes(&samples, 24000));
}

#[divan::bench(args = [1024, 24000, 240000])]
fn decode_wav_mono_same_rate(bencher: divan::Bencher, num_samples: usize) {
    let samples = sine_samples(num_samples);
    let encoded = wav::encode_wav_bytes(&samples, 24000);
    bencher
        .counter(divan::counter::BytesCount::new(encoded.len()))
        .bench_local(|| wav::decode_wav_mono(&encoded, 24000).unwrap());
}

#[divan::bench(args = [1024, 24000, 240000])]
fn decode_wav_mono_resample(bencher: divan::Bencher, num_samples: usize) {
    let samples = sine_samples(num_samples);
    let encoded = wav::encode_wav_bytes(&samples, 48000);
    bencher
        .counter(divan::counter::BytesCount::new(encoded.len()))
        .bench_local(|| wav::decode_wav_mono(&encoded, 24000).unwrap());
}

#[divan::bench(args = [1024, 24000, 240000])]
fn save_wav(bencher: divan::Bencher, num_samples: usize) {
    let samples = sine_samples(num_samples);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.wav");
    bencher
        .counter(divan::counter::BytesCount::new(num_samples * 2))
        .bench_local(|| wav::save_wav(&samples, &path, 24000).unwrap());
}

#[divan::bench(args = [1024, 24000, 240000])]
fn load_wav_mono(bencher: divan::Bencher, num_samples: usize) {
    let samples = sine_samples(num_samples);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bench.wav");
    wav::save_wav(&samples, &path, 24000).unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(num_samples * 2))
        .bench_local(|| wav::load_wav_mono(&path, 24000).unwrap());
}
