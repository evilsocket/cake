use candle_core::{DType, Device, Tensor};

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
