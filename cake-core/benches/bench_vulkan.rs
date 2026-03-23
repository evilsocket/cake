/// Vulkan backend benchmarks — GPU matmul, elementwise ops, and full MLP.
///
/// All "vulkan_*" benchmarks exercise the GPU compute path.
/// All "cpu_*" benchmarks use CPU-only candle ops for comparison.
///
/// Run on Steam Deck: `cargo bench -p cake-core --features vulkan -- vulkan`

use cake_core::backends::{ComputeBackend, VulkanBackend};
use candle_core::{DType, Device, Tensor};

fn vk() -> VulkanBackend {
    VulkanBackend::new().expect("Vulkan backend required for these benchmarks")
}

fn cpu_tensor(shape: &[usize], seed: u64) -> Tensor {
    super::bench_helpers::make_tensor(shape, seed)
}

// ── Dispatch overhead ────────────────────────────────────────────────

#[divan::bench]
fn vulkan_dispatch_overhead(bencher: divan::Bencher) {
    let backend = vk();
    // Tiny tensor to measure fixed dispatch cost (not data transfer)
    let a = cpu_tensor(&[1, 1, 16], 1000);
    let b = cpu_tensor(&[1, 1, 16], 1001);
    bencher.bench_local(|| backend.silu_mul(&a, &b).unwrap());
}

// ── GPU matmul at model sizes ────────────────────────────────────────
// Qwen3-0.6B: hidden=1024, intermediate=3072, head_dim=128
// M=2 is the smallest prefill batch that goes through GPU (M>1).

#[divan::bench(args = [2, 8, 32, 64])]
fn vulkan_gemm_Mx1024x4096(bencher: divan::Bencher, m: usize) {
    let backend = vk();
    let a = cpu_tensor(&[m, 1024], 1300);
    let b = cpu_tensor(&[1024, 4096], 1301);
    bencher.bench_local(|| backend.matmul(&a, &b).unwrap());
}

#[divan::bench(args = [2, 8, 32, 64])]
fn cpu_gemm_Mx1024x4096(bencher: divan::Bencher, m: usize) {
    let a = cpu_tensor(&[m, 1024], 1300);
    let b = cpu_tensor(&[1024, 4096], 1301);
    bencher.bench_local(|| a.matmul(&b).unwrap());
}

// ── GPU matmul at other model shapes ─────────────────────────────────
// gate_up: Mx1024x6144, down: Mx3072x1024

#[divan::bench(args = [2, 8, 32])]
fn vulkan_gemm_Mx1024x6144(bencher: divan::Bencher, m: usize) {
    let backend = vk();
    let a = cpu_tensor(&[m, 1024], 1100);
    let b = cpu_tensor(&[1024, 6144], 1101);
    bencher.bench_local(|| backend.matmul(&a, &b).unwrap());
}

#[divan::bench(args = [2, 8, 32])]
fn vulkan_gemm_Mx3072x1024(bencher: divan::Bencher, m: usize) {
    let backend = vk();
    let a = cpu_tensor(&[m, 3072], 1200);
    let b = cpu_tensor(&[3072, 1024], 1201);
    bencher.bench_local(|| backend.matmul(&a, &b).unwrap());
}

// ── CPU generation (M=1) baseline ────────────────────────────────────
// M=1 uses CPU fallback (dispatch overhead > compute gain).

#[divan::bench(args = [1024, 4096, 6144])]
fn cpu_gemv_1024xN(bencher: divan::Bencher, n: usize) {
    let a = cpu_tensor(&[1, 1024], 1100);
    let b = cpu_tensor(&[1024, n], 1101);
    bencher.bench_local(|| a.matmul(&b).unwrap());
}

#[divan::bench]
fn cpu_gemv_3072x1024(bencher: divan::Bencher) {
    let a = cpu_tensor(&[1, 3072], 1200);
    let b = cpu_tensor(&[3072, 1024], 1201);
    bencher.bench_local(|| a.matmul(&b).unwrap());
}

// ── Elementwise ops — GPU path (large tensors) ──────────────────────
// Above 8192 element threshold to ensure GPU dispatch.

#[divan::bench(args = [16384, 32768])]
fn vulkan_silu_mul(bencher: divan::Bencher, size: usize) {
    let backend = vk();
    let gate = cpu_tensor(&[1, 1, size], 1400);
    let up = cpu_tensor(&[1, 1, size], 1401);
    bencher.bench_local(|| backend.silu_mul(&gate, &up).unwrap());
}

#[divan::bench(args = [16384, 32768])]
fn cpu_silu_mul(bencher: divan::Bencher, size: usize) {
    let gate = cpu_tensor(&[1, 1, size], 1400);
    let up = cpu_tensor(&[1, 1, size], 1401);
    bencher.bench_local(|| {
        (candle_nn::ops::silu(&gate).unwrap() * &up).unwrap()
    });
}

#[divan::bench(args = [16384, 32768])]
fn vulkan_add3(bencher: divan::Bencher, size: usize) {
    let backend = vk();
    let a = cpu_tensor(&[1, 1, size], 1500);
    let b = cpu_tensor(&[1, 1, size], 1501);
    let c = cpu_tensor(&[1, 1, size], 1502);
    bencher.bench_local(|| backend.add3(&a, &b, &c).unwrap());
}

// ── RMS norm (CPU fallback in current backend) ──────────────────────

#[divan::bench(args = [1024, 3072])]
fn vulkan_rms_norm_gated(bencher: divan::Bencher, size: usize) {
    let backend = vk();
    let x = cpu_tensor(&[1, 1, size], 1600);
    let z = cpu_tensor(&[1, 1, size], 1601);
    let w = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| backend.rms_norm_gated(&x, &z, &w, 1e-6).unwrap());
}

#[divan::bench(args = [1024, 3072])]
fn vulkan_add_rms_norm(bencher: divan::Bencher, size: usize) {
    let backend = vk();
    let a = cpu_tensor(&[1, 1, size], 1700);
    let b = cpu_tensor(&[1, 1, size], 1701);
    let w = Tensor::ones(size, DType::F32, &Device::Cpu).unwrap();
    bencher.bench_local(|| backend.add_rms_norm(&a, &b, &w, 1e-6).unwrap());
}

// ── Full MLP pass (gate_up + silu_mul + down) ────────────────────────
// Prefill MLP at M=8: all ops go through GPU.

#[divan::bench]
fn vulkan_mlp_full(bencher: divan::Bencher) {
    let backend = vk();
    let x = cpu_tensor(&[8, 1024], 1800);
    let gate_up_w = cpu_tensor(&[6144, 1024], 1801);
    let down_w = cpu_tensor(&[1024, 3072], 1802);
    bencher.bench_local(|| {
        let fused = backend.matmul(&x, &gate_up_w.t().unwrap()).unwrap();
        let gate = fused.narrow(1, 0, 3072).unwrap().contiguous().unwrap();
        let up = fused.narrow(1, 3072, 3072).unwrap().contiguous().unwrap();
        let act = backend.silu_mul(&gate, &up).unwrap();
        backend.matmul(&act, &down_w.t().unwrap()).unwrap()
    });
}

#[divan::bench]
fn cpu_mlp_full(bencher: divan::Bencher) {
    let x = cpu_tensor(&[8, 1024], 1800);
    let gate_up_w = cpu_tensor(&[6144, 1024], 1801);
    let down_w = cpu_tensor(&[1024, 3072], 1802);
    bencher.bench_local(|| {
        let fused = x.matmul(&gate_up_w.t().unwrap()).unwrap();
        let gate = fused.narrow(1, 0, 3072).unwrap();
        let up = fused.narrow(1, 3072, 3072).unwrap();
        let act = (candle_nn::ops::silu(&gate).unwrap() * &up).unwrap();
        act.matmul(&down_w.t().unwrap()).unwrap()
    });
}
