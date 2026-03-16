use candle_core::{DType, Device, Tensor};

fn gpu_device() -> Device {
    if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).unwrap()
    } else {
        Device::Cpu
    }
}

fn make_gpu_tensor(shape: &[usize], seed: u64) -> Tensor {
    super::bench_helpers::make_tensor(shape, seed)
        .to_device(&gpu_device())
        .unwrap()
}

#[divan::bench(args = [256, 1024, 4096])]
fn rms_norm_forward(bencher: divan::Bencher, hidden_size: usize) {
    let x = super::bench_helpers::make_tensor(&[1, 1, hidden_size], 100);
    let weight = Tensor::ones(hidden_size, DType::F32, &Device::Cpu).unwrap();
    let eps = 1e-6f32;
    bencher.bench_local(|| candle_nn::ops::rms_norm(&x, &weight, eps).unwrap());
}

#[divan::bench(args = [256, 1024, 4096])]
fn silu_forward(bencher: divan::Bencher, size: usize) {
    let x = super::bench_helpers::make_tensor(&[1, 1, size], 110);
    bencher.bench_local(|| candle_nn::ops::silu(&x).unwrap());
}

#[divan::bench(args = [256, 1024, 4096])]
fn softmax_last_dim(bencher: divan::Bencher, size: usize) {
    let x = super::bench_helpers::make_tensor(&[1, size], 120);
    bencher.bench_local(|| candle_nn::ops::softmax_last_dim(&x).unwrap());
}

#[divan::bench(args = [64, 256, 1024])]
fn matmul_square(bencher: divan::Bencher, n: usize) {
    let a = super::bench_helpers::make_tensor(&[n, n], 130);
    let b = super::bench_helpers::make_tensor(&[n, n], 131);
    bencher
        .counter(divan::counter::BytesCount::new(n * n * 4 * 2))
        .bench_local(|| a.matmul(&b).unwrap());
}

#[divan::bench(args = [64, 256, 1024])]
fn tensor_cat(bencher: divan::Bencher, size: usize) {
    let a = super::bench_helpers::make_tensor(&[1, size], 140);
    let b = super::bench_helpers::make_tensor(&[1, size], 141);
    bencher.bench_local(|| Tensor::cat(&[&a, &b], 1).unwrap());
}

// ── Fused ops benchmarks ─────────────────────────────────────────────

// ── CPU fused ops ────────────────────────────────────────────────────

#[divan::bench(args = [256, 1024, 4096])]
fn fused_silu_mul_cpu(bencher: divan::Bencher, size: usize) {
    let gate = super::bench_helpers::make_tensor(&[1, 1, size], 150);
    let up = super::bench_helpers::make_tensor(&[1, 1, size], 151);
    bencher.bench_local(|| cake_core::utils::fused_ops::silu_mul(&gate, &up).unwrap());
}

#[divan::bench(args = [256, 1024, 4096])]
fn fused_stable_softplus_cpu(bencher: divan::Bencher, size: usize) {
    let x = super::bench_helpers::make_tensor(&[1, 1, size], 160);
    bencher.bench_local(|| cake_core::utils::fused_ops::stable_softplus(&x).unwrap());
}

#[divan::bench(args = [256, 1024, 4096])]
fn fused_rms_norm_gated_cpu(bencher: divan::Bencher, size: usize) {
    let x = super::bench_helpers::make_tensor(&[1, 1, size], 170);
    let z = super::bench_helpers::make_tensor(&[1, 1, size], 171);
    let weight = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| {
        cake_core::utils::fused_ops::rms_norm_gated(&x, &z, &weight, 1e-6).unwrap()
    });
}

// ── GPU fused ops (CUDA when available, CPU fallback) ────────────────

#[divan::bench(args = [1024, 4096])]
fn fused_silu_mul_gpu(bencher: divan::Bencher, size: usize) {
    let gate = make_gpu_tensor(&[1, 1, size], 250);
    let up = make_gpu_tensor(&[1, 1, size], 251);
    bencher.bench_local(|| cake_core::utils::fused_ops::silu_mul(&gate, &up).unwrap());
}

#[divan::bench(args = [1024, 4096])]
fn fused_stable_softplus_gpu(bencher: divan::Bencher, size: usize) {
    let x = make_gpu_tensor(&[1, 1, size], 260);
    bencher.bench_local(|| cake_core::utils::fused_ops::stable_softplus(&x).unwrap());
}

#[divan::bench(args = [1024, 4096])]
fn fused_rms_norm_gated_gpu(bencher: divan::Bencher, size: usize) {
    let dev = gpu_device();
    let x = make_gpu_tensor(&[1, 1, size], 270);
    let z = make_gpu_tensor(&[1, 1, size], 271);
    let weight = Tensor::ones(size, DType::F32, &dev).unwrap();
    bencher.bench_local(|| {
        cake_core::utils::fused_ops::rms_norm_gated(&x, &z, &weight, 1e-6).unwrap()
    });
}

/// Baseline: separate silu + mul on GPU (to compare against fused)
#[divan::bench(args = [1024, 4096])]
fn unfused_silu_mul_gpu(bencher: divan::Bencher, size: usize) {
    let gate = make_gpu_tensor(&[1, 1, size], 280);
    let up = make_gpu_tensor(&[1, 1, size], 281);
    bencher.bench_local(|| (candle_nn::ops::silu(&gate).unwrap() * &up).unwrap());
}

/// Baseline: separate rms_norm + silu + mul on GPU
#[divan::bench(args = [1024, 4096])]
fn unfused_rms_norm_gated_gpu(bencher: divan::Bencher, size: usize) {
    let dev = gpu_device();
    let x = make_gpu_tensor(&[1, 1, size], 290);
    let z = make_gpu_tensor(&[1, 1, size], 291);
    let weight = Tensor::ones(size, DType::F32, &dev).unwrap();
    let eps = 1e-6f32;
    bencher.bench_local(|| {
        let normed = candle_nn::ops::rms_norm(&x, &weight, eps).unwrap();
        let gate = candle_nn::ops::silu(&z).unwrap();
        (normed * gate).unwrap()
    });
}

// ── exp_mul / sub_mul GPU benchmarks ─────────────────────────────────

#[divan::bench(args = [1024, 4096])]
fn fused_exp_mul_gpu(bencher: divan::Bencher, size: usize) {
    let x = make_gpu_tensor(&[1, 1, size], 300);
    let y = make_gpu_tensor(&[1, 1, size], 301);
    bencher.bench_local(|| cake_core::utils::fused_ops::exp_mul(&x, &y).unwrap());
}

#[divan::bench(args = [1024, 4096])]
fn unfused_exp_mul_gpu(bencher: divan::Bencher, size: usize) {
    let x = make_gpu_tensor(&[1, 1, size], 310);
    let y = make_gpu_tensor(&[1, 1, size], 311);
    bencher.bench_local(|| (&x * y.exp().unwrap()).unwrap());
}

#[divan::bench(args = [1024, 4096])]
fn fused_sub_mul_gpu(bencher: divan::Bencher, size: usize) {
    let a = make_gpu_tensor(&[1, 1, size], 320);
    let b = make_gpu_tensor(&[1, 1, size], 321);
    let c = make_gpu_tensor(&[1, 1, size], 322);
    bencher.bench_local(|| cake_core::utils::fused_ops::sub_mul(&a, &b, &c).unwrap());
}

#[divan::bench(args = [1024, 4096])]
fn unfused_sub_mul_gpu(bencher: divan::Bencher, size: usize) {
    let a = make_gpu_tensor(&[1, 1, size], 330);
    let b = make_gpu_tensor(&[1, 1, size], 331);
    let c = make_gpu_tensor(&[1, 1, size], 332);
    bencher.bench_local(|| ((&a - &b).unwrap() * &c).unwrap());
}
