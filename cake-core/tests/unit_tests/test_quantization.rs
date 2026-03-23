//! Tests for FP8 dequantization paths.

use candle_core::{DType, Device, Tensor};

#[test]
fn test_fp8_to_f32() {
    let b = cake_core::backends::create_backend(&Device::Cpu);
    // Create a small F32 tensor
    let f32_vals = vec![0.5f32, 1.0, -0.25, 2.0];
    let t = Tensor::from_vec(f32_vals, (2, 2), &Device::Cpu).unwrap();
    // F8 roundtrip through to_dtype
    if let Ok(f8) = t.to_dtype(DType::F8E4M3) {
        let back = b.f8e4m3_to_f32(&f8).unwrap();
        assert_eq!(back.dtype(), DType::F32);
        assert_eq!(back.dims(), &[2, 2]);
    }
    // Skip if F8 not supported on this platform
}
