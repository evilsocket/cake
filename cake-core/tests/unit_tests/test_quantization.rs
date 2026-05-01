//! Tests for FP8 dequantization and fused 4-bit matmul paths.

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

// ─── Fused 4-bit matmul tests ──────────────────────────────────────────

/// CPU reference: dequantize packed 4-bit weights and multiply with x.
/// Returns (M, out_features) F32 result.
#[allow(clippy::too_many_arguments)]
fn cpu_q4_matmul_reference(
    packed: &[u32],
    scales: &[f32],
    biases: &[f32],
    x: &[f32],
    m: usize,
    in_features: usize,
    out_features: usize,
    group_size: usize,
    num_groups: usize,
) -> Vec<f32> {
    let packed_cols = in_features / 8;
    let mut output = vec![0f32; m * out_features];
    for row in 0..m {
        for col in 0..out_features {
            let mut acc = 0f32;
            for pc in 0..packed_cols {
                let packed_val = packed[col * packed_cols + pc];
                for bit in 0..8u32 {
                    let j = pc * 8 + bit as usize;
                    let w4 = ((packed_val >> (bit * 4)) & 0xF) as f32;
                    let g = j / group_size;
                    let scale = scales[col * num_groups + g];
                    let bias = biases[col * num_groups + g];
                    let w = w4 * scale + bias;
                    acc += w * x[row * in_features + j];
                }
            }
            output[row * out_features + col] = acc;
        }
    }
    output
}

#[test]
fn test_q4_matmul_cpu_reference_known_values() {
    // Tiny case: 1x16 activation, 2x16 weight (packed as 2x2 u32), group_size=8
    let in_features = 16;
    let out_features = 2;
    let group_size = 8;
    let num_groups = in_features / group_size; // 2

    // All nibbles = 1, scale=1.0, bias=0.0 → each weight = 1.0
    // x = [1.0; 16] → dot product = 16.0 for each output
    let packed = vec![0x11111111u32; out_features * (in_features / 8)]; // 2 * 2 = 4
    let scales = vec![1.0f32; out_features * num_groups]; // 2 * 2 = 4
    let biases = vec![0.0f32; out_features * num_groups]; // 2 * 2 = 4
    let x = vec![1.0f32; in_features];

    let result = cpu_q4_matmul_reference(
        &packed,
        &scales,
        &biases,
        &x,
        1,
        in_features,
        out_features,
        group_size,
        num_groups,
    );
    assert_eq!(result.len(), 2);
    assert!(
        (result[0] - 16.0).abs() < 1e-5,
        "expected 16.0, got {}",
        result[0]
    );
    assert!(
        (result[1] - 16.0).abs() < 1e-5,
        "expected 16.0, got {}",
        result[1]
    );
}

#[test]
fn test_q4_matmul_cpu_reference_varied_nibbles() {
    // 1x8 activation, 1x8 weight, group_size=8 (1 group)
    // packed[0] = 0x76543210 → nibbles [0,1,2,3,4,5,6,7]
    // scale=0.5, bias=-1.0 → weights = [0*0.5-1, 1*0.5-1, ..., 7*0.5-1]
    //                                 = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    // x = [1,1,1,1,1,1,1,1] → dot = sum(weights) = -1+(-0.5)+0+0.5+1+1.5+2+2.5 = 6.0
    let packed = vec![0x76543210u32];
    let scales = vec![0.5f32];
    let biases = vec![-1.0f32];
    let x = vec![1.0f32; 8];

    let result = cpu_q4_matmul_reference(&packed, &scales, &biases, &x, 1, 8, 1, 8, 1);
    assert!((result[0] - 6.0).abs() < 1e-5, "expected 6.0, got {}", result[0]);
}

#[cfg(feature = "metal")]
#[test]
fn test_q4_matmul_f16_metal_vs_cpu() {
    use candle_core::utils::metal_is_available;
    use half::f16;

    if !metal_is_available() {
        return; // Skip on non-Metal platforms
    }
    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };

    // Dimensions: M=2, in_features=64, out_features=4, group_size=32
    let m = 2usize;
    let in_features = 64usize;
    let out_features = 4usize;
    let group_size = 32usize;
    let num_groups = in_features / group_size; // 2
    let packed_cols = in_features / 8; // 8

    // Generate deterministic test data
    let mut packed_data = vec![0u32; out_features * packed_cols];
    for (i, val) in packed_data.iter_mut().enumerate() {
        // Varying nibble patterns
        let base = (i * 7 + 3) as u32;
        *val = 0;
        for bit in 0..8u32 {
            let nibble = (base + bit * 5) % 16;
            *val |= nibble << (bit * 4);
        }
    }

    let mut scales_f32 = vec![0f32; out_features * num_groups];
    for (i, s) in scales_f32.iter_mut().enumerate() {
        *s = 0.1 + (i as f32) * 0.05;
    }

    let mut biases_f32 = vec![0f32; out_features * num_groups];
    for (i, b) in biases_f32.iter_mut().enumerate() {
        *b = -0.5 + (i as f32) * 0.1;
    }

    let mut x_f32 = vec![0f32; m * in_features];
    for (i, v) in x_f32.iter_mut().enumerate() {
        *v = ((i % 17) as f32 - 8.0) / 16.0;
    }

    // CPU reference (F32 precision)
    let cpu_result = cpu_q4_matmul_reference(
        &packed_data,
        &scales_f32,
        &biases_f32,
        &x_f32,
        m,
        in_features,
        out_features,
        group_size,
        num_groups,
    );

    // Metal path: create tensors on Metal device
    let packed_tensor =
        Tensor::from_vec(packed_data, (out_features, packed_cols), &metal_device).unwrap();
    let scales_f16: Vec<f16> = scales_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let biases_f16: Vec<f16> = biases_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let x_f16: Vec<f16> = x_f32.iter().map(|&v| f16::from_f32(v)).collect();

    let scales_tensor =
        Tensor::from_vec(scales_f16, (out_features, num_groups), &metal_device).unwrap();
    let biases_tensor =
        Tensor::from_vec(biases_f16, (out_features, num_groups), &metal_device).unwrap();
    let x_tensor = Tensor::from_vec(x_f16, (m, in_features), &metal_device).unwrap();

    let metal_result = cake_core::backends::q4_matmul_f16(
        &packed_tensor,
        &scales_tensor,
        &biases_tensor,
        &x_tensor,
        group_size,
    )
    .unwrap();

    // Verify shape
    assert_eq!(metal_result.dims(), &[m, out_features]);
    assert_eq!(metal_result.dtype(), DType::F16);

    // Compare values with F16 tolerance
    let metal_f32: Vec<f32> = metal_result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut max_diff = 0f32;
    let mut max_idx = 0usize;
    for (i, (cpu, metal)) in cpu_result.iter().zip(metal_f32.iter()).enumerate() {
        let diff = (cpu - metal).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }

    // F16 accumulation tolerance: with 64 multiply-adds and F16 quantization of
    // scales/biases/activations, expect up to ~0.5 absolute error.
    let tolerance = 0.5f32;
    assert!(
        max_diff <= tolerance,
        "q4_matmul_f16: max diff {max_diff} at index {max_idx} (cpu={} metal={}) exceeds tolerance {tolerance}",
        cpu_result[max_idx],
        metal_f32[max_idx]
    );
}

#[cfg(feature = "metal")]
#[test]
fn test_q4_matmul_f16_metal_identity_weights() {
    // Test with weights that are effectively identity-like:
    // All nibbles=0, scale=0, bias=1 → all weights=1.0, so output = sum(x) per row.
    use candle_core::utils::metal_is_available;
    use half::f16;

    if !metal_is_available() {
        return;
    }
    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };

    let m = 1usize;
    let in_features = 32usize;
    let out_features = 2usize;
    let group_size = 32usize;
    let num_groups = 1usize;
    let packed_cols = in_features / 8; // 4

    // All nibbles=0, scale=0, bias=1.0 → all dequantized weights = 1.0
    let packed_data = vec![0u32; out_features * packed_cols];
    let scales_f16 = vec![f16::from_f32(0.0); out_features * num_groups];
    let biases_f16 = vec![f16::from_f32(1.0); out_features * num_groups];

    // x = [0.5; 32] → dot = 0.5 * 32 = 16.0
    let x_f16 = vec![f16::from_f32(0.5); m * in_features];

    let packed_tensor =
        Tensor::from_vec(packed_data, (out_features, packed_cols), &metal_device).unwrap();
    let scales_tensor =
        Tensor::from_vec(scales_f16, (out_features, num_groups), &metal_device).unwrap();
    let biases_tensor =
        Tensor::from_vec(biases_f16, (out_features, num_groups), &metal_device).unwrap();
    let x_tensor = Tensor::from_vec(x_f16, (m, in_features), &metal_device).unwrap();

    let result = cake_core::backends::q4_matmul_f16(
        &packed_tensor,
        &scales_tensor,
        &biases_tensor,
        &x_tensor,
        group_size,
    )
    .unwrap();

    let result_f32: Vec<f32> = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    assert_eq!(result.dims(), &[1, 2]);
    for (i, &v) in result_f32.iter().enumerate() {
        assert!(
            (v - 16.0).abs() < 0.1,
            "output[{i}] = {v}, expected 16.0"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_q4_matmul_f16_metal_batch() {
    // Test with M > 1 to verify batched dispatch works
    use candle_core::utils::metal_is_available;
    use half::f16;

    if !metal_is_available() {
        return;
    }
    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };

    let m = 4usize;
    let in_features = 16usize;
    let out_features = 3usize;
    let group_size = 8usize;
    let num_groups = 2usize;
    let packed_cols = 2usize;

    // Simple: all nibbles=2, scale=1.0, bias=0.0 → weights=2.0
    // x row i = [i+1; 16] → dot = (i+1)*2.0*16 = 32*(i+1)
    let packed_data = vec![0x22222222u32; out_features * packed_cols];
    let scales_f16 = vec![f16::from_f32(1.0); out_features * num_groups];
    let biases_f16 = vec![f16::from_f32(0.0); out_features * num_groups];

    let mut x_f16 = Vec::with_capacity(m * in_features);
    for row in 0..m {
        for _ in 0..in_features {
            x_f16.push(f16::from_f32((row + 1) as f32));
        }
    }

    let packed_tensor =
        Tensor::from_vec(packed_data, (out_features, packed_cols), &metal_device).unwrap();
    let scales_tensor =
        Tensor::from_vec(scales_f16, (out_features, num_groups), &metal_device).unwrap();
    let biases_tensor =
        Tensor::from_vec(biases_f16, (out_features, num_groups), &metal_device).unwrap();
    let x_tensor = Tensor::from_vec(x_f16, (m, in_features), &metal_device).unwrap();

    let result = cake_core::backends::q4_matmul_f16(
        &packed_tensor,
        &scales_tensor,
        &biases_tensor,
        &x_tensor,
        group_size,
    )
    .unwrap();

    assert_eq!(result.dims(), &[4, 3]);

    let result_f32: Vec<f32> = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for row in 0..m {
        let expected = 32.0 * (row + 1) as f32;
        for col in 0..out_features {
            let actual = result_f32[row * out_features + col];
            assert!(
                (actual - expected).abs() < 1.0,
                "row={row} col={col}: expected {expected}, got {actual}"
            );
        }
    }
}

// ─── LinearWeight + QuantizedLinear tests ──────────────────────────

#[test]
fn test_linear_weight_dense_matches_backend() {
    // Verify LinearWeight::Dense produces the same result as direct backend.linear_forward
    use cake_core::utils::quantized_linear::LinearWeight;

    let backend = cake_core::backends::create_backend(&Device::Cpu);
    let w = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (3, 2),
        &Device::Cpu,
    )
    .unwrap();
    let w_preprocessed = backend.preprocess_linear_weight(&w).unwrap();
    let lw = LinearWeight::Dense(w_preprocessed.clone());
    let x = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2), &Device::Cpu).unwrap();

    let direct = backend.linear_forward(&x, &w_preprocessed, None).unwrap();
    let via_lw = lw.forward(&x, None, &*backend).unwrap();

    let direct_vals: Vec<f32> = direct.flatten_all().unwrap().to_vec1().unwrap();
    let lw_vals: Vec<f32> = via_lw.flatten_all().unwrap().to_vec1().unwrap();
    assert_eq!(direct_vals.len(), lw_vals.len());
    for (d, l) in direct_vals.iter().zip(lw_vals.iter()) {
        assert!((d - l).abs() < 1e-6, "mismatch: direct={d}, lw={l}");
    }
}

#[test]
fn test_linear_weight_quantized_cpu_fallback_known_values() {
    // Quantized path with known values: all nibbles=2, scale=0.5, bias=-1.0
    // weight[j] = 2 * 0.5 + (-1.0) = 0.0 for all elements.
    // So output should be 0 for any input.
    use cake_core::utils::quantized_linear::LinearWeight;

    let backend = cake_core::backends::create_backend(&Device::Cpu);
    let in_features = 8;
    let out_features = 2;

    // All nibbles=2
    let packed = Tensor::from_vec(
        vec![0x22222222u32; out_features],
        (out_features, 1),
        &Device::Cpu,
    )
    .unwrap();
    let scales = Tensor::from_vec(vec![0.5f32; out_features], (out_features, 1), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let biases = Tensor::from_vec(
        vec![-1.0f32; out_features],
        (out_features, 1),
        &Device::Cpu,
    )
    .unwrap()
    .to_dtype(DType::F16)
    .unwrap();

    let lw = LinearWeight::quantized(packed, scales, biases, 8);
    let x = Tensor::from_vec(vec![1.0f32; in_features], (1, in_features), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let out = lw.forward(&x, None, &*backend).unwrap();
    let vals: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.abs() < 0.5,
            "output[{i}] = {v}, expected ~0.0 (all weights should be 0)"
        );
    }
}

#[test]
fn test_linear_weight_quantized_with_bias() {
    // Verify bias is correctly added after quantized matmul
    use cake_core::utils::quantized_linear::LinearWeight;

    let backend = cake_core::backends::create_backend(&Device::Cpu);
    // All nibbles=0, scale=0, bias=0 → zero weights → output from matmul = 0
    // Then add a linear bias of [10.0, 20.0] → final output should be [10, 20]
    let packed = Tensor::from_vec(vec![0u32; 2], (2, 1), &Device::Cpu).unwrap();
    let scales = Tensor::from_vec(vec![0f32; 2], (2, 1), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let biases = Tensor::from_vec(vec![0f32; 2], (2, 1), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let lw = LinearWeight::quantized(packed, scales, biases, 8);

    let x = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let linear_bias = Tensor::from_vec(vec![10.0f32, 20.0], 2, &Device::Cpu)
        .unwrap()
        .to_dtype(DType::F16)
        .unwrap();
    let out = lw.forward(&x, Some(&linear_bias), &*backend).unwrap();
    let vals: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert!(
        (vals[0] - 10.0).abs() < 0.5,
        "expected ~10.0, got {}",
        vals[0]
    );
    assert!(
        (vals[1] - 20.0).abs() < 0.5,
        "expected ~20.0, got {}",
        vals[1]
    );
}

#[cfg(feature = "metal")]
#[test]
fn test_linear_weight_quantized_metal() {
    // Test the full fused q4 path through LinearWeight on Metal
    use candle_core::utils::metal_is_available;
    use cake_core::utils::quantized_linear::LinearWeight;
    use half::f16;

    if !metal_is_available() {
        return;
    }
    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let backend = cake_core::backends::create_backend(&metal_device);

    let m = 2usize;
    let in_features = 32usize;
    let out_features = 4usize;
    let group_size = 32usize;
    let num_groups = 1usize;
    let packed_cols = in_features / 8; // 4

    // All nibbles=1, scale=1.0, bias=0.0 → all weights = 1.0
    // x = [1.0; 32] → dot = 32.0
    let packed_data = vec![0x11111111u32; out_features * packed_cols];
    let scales_f16 = vec![f16::from_f32(1.0); out_features * num_groups];
    let biases_f16 = vec![f16::from_f32(0.0); out_features * num_groups];

    let packed_tensor =
        Tensor::from_vec(packed_data, (out_features, packed_cols), &metal_device).unwrap();
    let scales_tensor =
        Tensor::from_vec(scales_f16, (out_features, num_groups), &metal_device).unwrap();
    let biases_tensor =
        Tensor::from_vec(biases_f16, (out_features, num_groups), &metal_device).unwrap();

    let lw = LinearWeight::quantized(packed_tensor, scales_tensor, biases_tensor, group_size);

    let x = Tensor::from_vec(vec![f16::from_f32(1.0); m * in_features], (m, in_features), &metal_device).unwrap();
    let result = lw.forward(&x, None, &*backend).unwrap();

    assert_eq!(result.dims(), &[m, out_features]);
    let result_f32: Vec<f32> = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for (i, &v) in result_f32.iter().enumerate() {
        assert!(
            (v - 32.0).abs() < 1.0,
            "output[{i}] = {v}, expected ~32.0"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_linear_weight_quantized_metal_3d_input() {
    // Test the Metal q4 path with 3D input (batch, seq, features) to verify
    // the batched reshape in q4_linear_forward works.
    use candle_core::utils::metal_is_available;
    use cake_core::utils::quantized_linear::LinearWeight;
    use half::f16;

    if !metal_is_available() {
        return;
    }
    let metal_device = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return,
    };
    let backend = cake_core::backends::create_backend(&metal_device);

    let batch = 1usize;
    let seq_len = 4usize;
    let in_features = 16usize;
    let out_features = 2usize;
    let group_size = 8usize;
    let num_groups = 2usize;
    let packed_cols = in_features / 8; // 2

    // All nibbles=1, scale=1.0, bias=0.0 → weights = 1.0
    // x = [1.0; 16] → each row dot = 16.0
    let packed_data = vec![0x11111111u32; out_features * packed_cols];
    let scales_f16 = vec![f16::from_f32(1.0); out_features * num_groups];
    let biases_f16 = vec![f16::from_f32(0.0); out_features * num_groups];

    let packed_tensor =
        Tensor::from_vec(packed_data, (out_features, packed_cols), &metal_device).unwrap();
    let scales_tensor =
        Tensor::from_vec(scales_f16, (out_features, num_groups), &metal_device).unwrap();
    let biases_tensor =
        Tensor::from_vec(biases_f16, (out_features, num_groups), &metal_device).unwrap();

    let lw = LinearWeight::quantized(packed_tensor, scales_tensor, biases_tensor, group_size);

    // 3D input: (1, 4, 16)
    let x = Tensor::from_vec(
        vec![f16::from_f32(1.0); batch * seq_len * in_features],
        (batch, seq_len, in_features),
        &metal_device,
    )
    .unwrap();
    let result = lw.forward(&x, None, &*backend).unwrap();

    assert_eq!(result.dims(), &[batch, seq_len, out_features]);
    let result_f32: Vec<f32> = result
        .to_device(&Device::Cpu)
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();

    for (i, &v) in result_f32.iter().enumerate() {
        assert!(
            (v - 16.0).abs() < 1.0,
            "output[{i}] = {v}, expected ~16.0"
        );
    }
}
