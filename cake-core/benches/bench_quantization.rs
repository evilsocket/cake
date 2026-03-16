use candle_core::{DType, Device, Tensor};

#[divan::bench(args = [16, 64, 256])]
fn dequantize_gptq_4bit(bencher: divan::Bencher, out_features: usize) {
    // in_features = packed_rows * 8, group_size = 8
    let packed_rows = 2usize;
    let in_features = packed_rows * 8;
    let groups = in_features / 8;
    let qweight = Tensor::from_vec(
        vec![0x21i32; packed_rows * out_features],
        (packed_rows, out_features),
        &Device::Cpu,
    )
    .unwrap();
    let scales = Tensor::from_vec(
        vec![1.0f32; groups * out_features],
        (groups, out_features),
        &Device::Cpu,
    )
    .unwrap();
    let qzeros = Tensor::from_vec(
        vec![0i32; groups * (out_features / 8).max(1)],
        (groups, (out_features / 8).max(1)),
        &Device::Cpu,
    )
    .unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(out_features * in_features * 4usize))
        .bench_local(|| {
            cake_core::utils::gptq::dequantize_gptq_4bit(&qweight, &scales, &qzeros, 8).unwrap()
        });
}

#[divan::bench]
fn dequantize_fp8_blockwise(bencher: divan::Bencher) {
    let f32_weight =
        Tensor::from_vec(vec![0.5f32; 128 * 128], (128, 128), &Device::Cpu).unwrap();
    let weight = f32_weight.to_dtype(DType::F8E4M3).unwrap();
    let scale_inv = Tensor::from_vec(vec![1.0f32], (1, 1), &Device::Cpu).unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(128usize * 128 * 4))
        .bench_local(|| {
            cake_core::utils::fp8::dequantize_fp8_blockwise(&weight, &scale_inv).unwrap()
        });
}
