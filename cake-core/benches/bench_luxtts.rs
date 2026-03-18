//! Benchmarks for LuxTTS components.

#[cfg(feature = "luxtts")]
mod luxtts_benches {
    use candle_core::{DType, Device, Tensor};
    use divan::Bencher;

    // ---- Activations ----

    #[divan::bench(args = [64, 256, 512])]
    fn swoosh_r(bencher: Bencher, dim: usize) {
        let x = Tensor::randn(0f32, 1.0, (1, 32, dim), &Device::Cpu).unwrap();
        bencher.bench(|| {
            cake_core::models::luxtts::activations::swoosh_r(&x).unwrap()
        });
    }

    #[divan::bench(args = [64, 256, 512])]
    fn swoosh_l(bencher: Bencher, dim: usize) {
        let x = Tensor::randn(0f32, 1.0, (1, 32, dim), &Device::Cpu).unwrap();
        bencher.bench(|| {
            cake_core::models::luxtts::activations::swoosh_l(&x).unwrap()
        });
    }

    // ---- BiasNorm ----

    #[divan::bench(args = [64, 192, 512])]
    fn bias_norm(bencher: Bencher, dim: usize) {
        let bn = cake_core::models::luxtts::bias_norm::BiasNorm::load(
            dim,
            candle_nn::VarBuilder::from_tensors(
                [
                    ("bias".to_string(), Tensor::zeros(dim, DType::F32, &Device::Cpu).unwrap()),
                    ("log_scale".to_string(), Tensor::zeros(1, DType::F32, &Device::Cpu).unwrap()),
                ].into_iter().collect(),
                DType::F32,
                &Device::Cpu,
            ),
        ).unwrap();
        let x = Tensor::randn(0f32, 1.0, (1, 32, dim), &Device::Cpu).unwrap();
        bencher.bench(|| bn.forward(&x).unwrap());
    }

    // ---- Mel spectrogram ----

    #[divan::bench]
    fn mel_spectrogram_1s(bencher: Bencher) {
        // 1 second of audio at 24kHz
        let samples = vec![0.0f32; 24000];
        bencher.bench(|| {
            cake_core::models::luxtts::mel::mel_spectrogram(
                &samples, 1024, 256, 100, 24000,
                &Device::Cpu, DType::F32,
            ).unwrap()
        });
    }

    #[divan::bench]
    fn mel_spectrogram_5s(bencher: Bencher) {
        let samples = vec![0.0f32; 120000];
        bencher.bench(|| {
            cake_core::models::luxtts::mel::mel_spectrogram(
                &samples, 1024, 256, 100, 24000,
                &Device::Cpu, DType::F32,
            ).unwrap()
        });
    }

    // ---- Euler solver ----

    #[divan::bench]
    fn euler_step(bencher: Bencher) {
        let x = Tensor::randn(0f32, 1.0, (1, 100, 512), &Device::Cpu).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, 100, 512), &Device::Cpu).unwrap();
        bencher.bench(|| {
            cake_core::models::luxtts::euler_solver::EulerSolver::step(&x, &v, 0.25).unwrap()
        });
    }
}
