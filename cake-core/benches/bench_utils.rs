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

// ── Additional fused ops (CPU) ───────────────────────────────────────

#[divan::bench(args = [256, 1024, 4096])]
fn fused_add3_cpu(bencher: divan::Bencher, size: usize) {
    let a = super::bench_helpers::make_tensor(&[1, 1, size], 400);
    let b = super::bench_helpers::make_tensor(&[1, 1, size], 401);
    let c = super::bench_helpers::make_tensor(&[1, 1, size], 402);
    bencher.bench_local(|| cake_core::utils::fused_ops::add3(&a, &b, &c).unwrap());
}

#[divan::bench(args = [256, 1024, 4096])]
fn fused_add_rms_norm_cpu(bencher: divan::Bencher, size: usize) {
    let a = super::bench_helpers::make_tensor(&[1, 1, size], 410);
    let b = super::bench_helpers::make_tensor(&[1, 1, size], 411);
    let w = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| cake_core::utils::fused_ops::add_rms_norm(&a, &b, &w, 1e-6).unwrap());
}

#[divan::bench(args = [256, 1024, 4096])]
fn fused_add_scaled_cpu(bencher: divan::Bencher, size: usize) {
    let a = super::bench_helpers::make_tensor(&[1, 1, size], 420);
    let b = super::bench_helpers::make_tensor(&[1, 1, size], 421);
    let c = super::bench_helpers::make_tensor(&[1, 1, size], 422);
    bencher.bench_local(|| cake_core::utils::fused_ops::add_scaled(&a, &b, &c).unwrap());
}

#[divan::bench(args = [256, 1024])]
fn fused_rms_norm_channel_cpu(bencher: divan::Bencher, channels: usize) {
    // (batch=1, channels, seq_len=8)
    let x = super::bench_helpers::make_tensor(&[1, channels, 8], 430);
    let w = Tensor::ones(channels, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| cake_core::utils::fused_ops::rms_norm_channel(&x, &w, 1e-6).unwrap());
}

#[divan::bench(args = [256, 1024])]
fn fused_depthwise_conv1d_silu_cpu(bencher: divan::Bencher, channels: usize) {
    let kernel_size = 4;
    let window = super::bench_helpers::make_tensor(&[1, channels, kernel_size], 440);
    let weight = super::bench_helpers::make_tensor(&[channels, kernel_size], 441);
    bencher.bench_local(|| {
        cake_core::utils::fused_ops::depthwise_conv1d_silu(&window, &weight, kernel_size, channels)
            .unwrap()
    });
}

#[divan::bench(args = [256, 1024])]
fn fused_depthwise_conv1d_bias_cpu(bencher: divan::Bencher, channels: usize) {
    let kernel_size = 4usize;
    let seq_len = 8usize;
    let padded = super::bench_helpers::make_tensor(&[1, channels, seq_len + kernel_size - 1], 450);
    let weight = super::bench_helpers::make_tensor(&[channels, 1, kernel_size], 451);
    let bias = super::bench_helpers::make_tensor(&[channels], 452);
    bencher.bench_local(|| {
        cake_core::utils::fused_ops::depthwise_conv1d_bias(&padded, &weight, &bias, kernel_size, channels)
            .unwrap()
    });
}

// ── GPU variants for new fused ops ───────────────────────────────────

#[divan::bench(args = [1024, 4096])]
fn fused_add3_gpu(bencher: divan::Bencher, size: usize) {
    let a = make_gpu_tensor(&[1, 1, size], 500);
    let b = make_gpu_tensor(&[1, 1, size], 501);
    let c = make_gpu_tensor(&[1, 1, size], 502);
    bencher.bench_local(|| cake_core::utils::fused_ops::add3(&a, &b, &c).unwrap());
}

#[divan::bench(args = [1024, 4096])]
fn fused_add_scaled_gpu(bencher: divan::Bencher, size: usize) {
    let a = make_gpu_tensor(&[1, 1, size], 510);
    let b = make_gpu_tensor(&[1, 1, size], 511);
    let c = make_gpu_tensor(&[1, 1, size], 512);
    bencher.bench_local(|| cake_core::utils::fused_ops::add_scaled(&a, &b, &c).unwrap());
}

// ── Fp8Linear forward benchmark ──────────────────────────────────────

#[divan::bench(args = [64, 256, 1024])]
fn fp8_linear_forward_cpu(bencher: divan::Bencher, size: usize) {
    let weight = super::bench_helpers::make_tensor(&[size, size], 600);
    let linear = cake_core::utils::fp8::Fp8Linear::new(weight, None);
    let x = super::bench_helpers::make_tensor(&[1, size], 601);
    bencher
        .counter(divan::counter::BytesCount::new(size * size * 4usize))
        .bench_local(|| linear.forward(&x).unwrap());
}

#[divan::bench(args = [64, 256])]
fn fp8_linear_forward_with_bias_cpu(bencher: divan::Bencher, size: usize) {
    let weight = super::bench_helpers::make_tensor(&[size, size], 610);
    let bias = super::bench_helpers::make_tensor(&[size], 611);
    let linear = cake_core::utils::fp8::Fp8Linear::new(weight, Some(bias));
    let x = super::bench_helpers::make_tensor(&[1, size], 612);
    bencher
        .counter(divan::counter::BytesCount::new(size * size * 4usize))
        .bench_local(|| linear.forward(&x).unwrap());
}

// ── GPU variants for new fused ops (CUDA when available, CPU fallback) ──

#[divan::bench(args = [1024, 4096])]
fn fused_add_rms_norm_gpu(bencher: divan::Bencher, size: usize) {
    let dev = gpu_device();
    let a = make_gpu_tensor(&[1, 1, size], 700);
    let b = make_gpu_tensor(&[1, 1, size], 701);
    let w = Tensor::ones(size, DType::F32, &dev).unwrap();
    bencher.bench_local(|| cake_core::utils::fused_ops::add_rms_norm(&a, &b, &w, 1e-6).unwrap());
}

#[divan::bench(args = [256, 1024])]
fn fused_depthwise_conv1d_silu_gpu(bencher: divan::Bencher, channels: usize) {
    let kernel_size = 4;
    let window = make_gpu_tensor(&[1, channels, kernel_size], 710);
    let weight = make_gpu_tensor(&[channels, kernel_size], 711);
    bencher.bench_local(|| {
        cake_core::utils::fused_ops::depthwise_conv1d_silu(&window, &weight, kernel_size, channels)
            .unwrap()
    });
}

#[divan::bench(args = [256, 1024])]
fn fused_rms_norm_channel_gpu(bencher: divan::Bencher, channels: usize) {
    let dev = gpu_device();
    let x = make_gpu_tensor(&[1, channels, 8], 720);
    let w = Tensor::ones(channels, DType::F32, &dev).unwrap();
    bencher.bench_local(|| cake_core::utils::fused_ops::rms_norm_channel(&x, &w, 1e-6).unwrap());
}

#[divan::bench(args = [256, 1024])]
fn fused_depthwise_conv1d_bias_gpu(bencher: divan::Bencher, channels: usize) {
    let kernel_size = 4usize;
    let seq_len = 8usize;
    let padded = make_gpu_tensor(&[1, channels, seq_len + kernel_size - 1], 730);
    let weight = make_gpu_tensor(&[channels, 1, kernel_size], 731);
    let bias = make_gpu_tensor(&[channels], 732);
    bencher.bench_local(|| {
        cake_core::utils::fused_ops::depthwise_conv1d_bias(&padded, &weight, &bias, kernel_size, channels)
            .unwrap()
    });
}

// ── Transformer block forward benchmark ──────────────────────────────

#[divan::bench(args = [1, 8, 64])]
fn transformer_block_forward(bencher: divan::Bencher, seq_len: usize) {
    use cake_core::models::common::Transformer;
    let cfg = super::bench_helpers::test_config();
    let vb = super::bench_helpers::make_vb_transformer_block(&cfg);
    let block = Transformer::load_for_vibevoice(vb, &cfg, std::sync::Arc::new(cake_core::backends::CpuBackend::new())).unwrap();
    let mut cache = super::bench_helpers::make_cache(&cfg);
    let x = super::bench_helpers::make_tensor(&[1, seq_len, cfg.hidden_size], 800);
    bencher.bench_local(|| block.forward_with_cache(&x, 0, 0, &mut cache).unwrap());
}

// ── Native dtype backend load benchmark ──────────────────────────────

#[divan::bench]
fn native_dtype_backend_load(bencher: divan::Bencher) {
    use candle_core::safetensors::save;
    use std::collections::HashMap;
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("model.safetensors");
    let t = super::bench_helpers::make_tensor(&[256, 256], 900);
    let mut tensors = HashMap::new();
    tensors.insert("weight".to_string(), t);
    save(&tensors, &path).unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(256usize * 256 * 4))
        .bench_local(|| {
            unsafe {
                cake_core::utils::native_dtype_backend::load_native_dtype_var_builder(
                    std::slice::from_ref(&path),
                    DType::F32,
                    &Device::Cpu,
                )
                .unwrap()
            }
        });
}
