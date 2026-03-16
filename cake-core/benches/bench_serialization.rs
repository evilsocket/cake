use cake_core::cake::RawTensor;
use candle_core::{DType, Device};

#[divan::bench(args = [1024, 4096, 65536])]
fn raw_tensor_from_tensor(bencher: divan::Bencher, size: usize) {
    let t = super::bench_helpers::make_tensor(&[1, size], 50)
        .to_dtype(DType::F16)
        .unwrap();
    bencher
        .counter(divan::counter::BytesCount::new(size * 2))
        .bench_local(|| RawTensor::from_tensor(&t));
}

#[divan::bench(args = [1024, 4096, 65536])]
fn raw_tensor_to_tensor(bencher: divan::Bencher, size: usize) {
    let t = super::bench_helpers::make_tensor(&[1, size], 51)
        .to_dtype(DType::F16)
        .unwrap();
    let raw = RawTensor::from_tensor(&t);
    let dev = Device::Cpu;
    bencher
        .counter(divan::counter::BytesCount::new(size * 2))
        .bench_local(|| raw.to_tensor(&dev).unwrap());
}

#[divan::bench(args = [1024, 4096, 65536])]
fn raw_tensor_roundtrip(bencher: divan::Bencher, size: usize) {
    let t = super::bench_helpers::make_tensor(&[1, size], 52)
        .to_dtype(DType::F16)
        .unwrap();
    let dev = Device::Cpu;
    bencher
        .counter(divan::counter::BytesCount::new(size * 2 * 2))
        .bench_local(|| {
            let raw = RawTensor::from_tensor(&t);
            raw.to_tensor(&dev).unwrap()
        });
}
