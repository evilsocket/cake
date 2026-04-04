//! Compare CausalConv3d against Python Conv3d reference.
//! Run with: cargo test --test wan_conv3d_compare -- --ignored --nocapture

use candle_core::{Device, Tensor, DType};

#[test]
#[ignore]
fn test_causal_conv3d_vs_python() {
    let path = "/tmp/conv3d_test.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {} not found", path);
        return;
    }

    let data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();

    let dev = Device::Cpu;

    // Load weight, bias, input, expected output
    let weight_data: Vec<f32> = data["weight"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let weight_shape: Vec<usize> = data["weight_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let bias_data: Vec<f32> = data["bias"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let input_data: Vec<f32> = data["input"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let input_shape: Vec<usize> = data["input_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let py_output: Vec<f32> = data["output"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let output_shape: Vec<usize> = data["output_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();

    let weight = Tensor::from_vec(weight_data, weight_shape.as_slice(), &dev).unwrap();
    let bias = Tensor::from_vec(bias_data, vec![weight_shape[0]], &dev).unwrap();
    let input = Tensor::from_vec(input_data, input_shape.as_slice(), &dev).unwrap();

    println!("Weight: {:?}", weight.shape());
    println!("Input: {:?}", input.shape());

    // Run our CausalConv3d manually (same logic as vendored/vae.rs)
    let (c_out, c_in, kt, kh, kw) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4]);
    let padding = (1usize, 1usize, 1usize); // matching Python's Conv3d padding=(0,1,1) + causal_pad=2
    let stride = (1usize, 1usize, 1usize);

    // Causal padding
    let first = input.narrow(2, 0, 1).unwrap();
    let pad_frames = first.repeat((1, 1, 2, 1, 1)).unwrap();
    let x = Tensor::cat(&[&pad_frames, &input], 2).unwrap();
    println!("After causal pad: {:?}", x.shape());

    let (b, c_in_actual, f, h, w) = x.dims5().unwrap();

    // Our 3D conv loop path (kt=3)
    let f_out = (f - kt) / stride.0 + 1;
    let w2d = weight.reshape((c_out, c_in * kt, kh, kw)).unwrap();

    let mut frames_out = Vec::with_capacity(f_out);
    for t in 0..f_out {
        let t_start = t * stride.0;
        let temporal_slice = x.narrow(2, t_start, kt).unwrap();
        let temporal_flat = temporal_slice.reshape((b, c_in * kt, h, w)).unwrap();
        let y_frame = temporal_flat.conv2d(&w2d, padding.1, stride.1, 1, 1).unwrap();
        frames_out.push(y_frame);
    }
    let y = Tensor::stack(&frames_out.iter().collect::<Vec<_>>(), 2).unwrap();

    // Add bias
    let bias_5d = bias.reshape((1, c_out, 1, 1, 1)).unwrap();
    let y = y.broadcast_add(&bias_5d).unwrap();

    let rust_output: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();

    println!("Rust output shape: {:?}", y.shape());
    println!("Python output shape: {:?}", output_shape);

    // Compare first 16 values
    println!("\nFirst 16 values comparison:");
    let mut max_diff: f32 = 0.0;
    for i in 0..16.min(rust_output.len()) {
        let diff = (rust_output[i] - py_output[i]).abs();
        max_diff = max_diff.max(diff);
        if diff > 0.001 {
            println!("  [{i}] rust={:.6} python={:.6} DIFF={:.6}", rust_output[i], py_output[i], diff);
        }
    }

    let total_max_diff: f32 = rust_output.iter().zip(py_output.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f64 = rust_output.iter().zip(py_output.iter())
        .map(|(r, p)| (r - p).abs() as f64)
        .sum::<f64>() / rust_output.len() as f64;

    println!("\nMax diff: {total_max_diff:.8}");
    println!("Mean diff: {mean_diff:.8}");

    if total_max_diff > 0.01 {
        // Print the 4x4 grid for visual comparison
        println!("\nRust [0,0,0,:4,:4]:");
        for h in 0..4 {
            let vals: Vec<f32> = (0..4).map(|w| rust_output[h * 8 + w]).collect();
            println!("  {:?}", vals);
        }
        println!("Python [0,0,0,:4,:4]:");
        for h in 0..4 {
            let vals: Vec<f32> = (0..4).map(|w| py_output[h * 8 + w]).collect();
            println!("  {:?}", vals);
        }
    }

    assert!(total_max_diff < 0.001, "CausalConv3d should match Python Conv3d, max_diff={total_max_diff}");
}

#[test]
#[ignore]
fn test_causal_conv3d_kt1_vs_python() {
    let path = "/tmp/conv3d_kt1_test.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping: {} not found", path);
        return;
    }

    let data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();

    let dev = candle_core::Device::Cpu;

    let weight_data: Vec<f32> = data["weight"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let weight_shape: Vec<usize> = data["weight_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let bias_data: Vec<f32> = data["bias"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let input_data: Vec<f32> = data["input"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let input_shape: Vec<usize> = data["input_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let py_output: Vec<f32> = data["output"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let weight = Tensor::from_vec(weight_data, weight_shape.as_slice(), &dev).unwrap();
    let bias = Tensor::from_vec(bias_data, vec![weight_shape[0]], &dev).unwrap();
    let input = Tensor::from_vec(input_data, input_shape.as_slice(), &dev).unwrap();

    let (c_out, c_in, kt, kh, kw) = (weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3], weight_shape[4]);
    let (b, _, f, h, w) = input.dims5().unwrap();

    assert_eq!(kt, 1, "This test is for kt=1 path");

    // Our kt=1 path (with the permute fix)
    let w2d = weight.squeeze(2).unwrap();
    // Fixed: permute before reshape
    let x_flat = input.permute((0, 2, 1, 3, 4)).unwrap().reshape((b * f, c_in, h, w)).unwrap();
    let y = x_flat.conv2d(&w2d, 1, 1, 1, 1).unwrap();
    let (_, c_o, h_o, w_o) = y.dims4().unwrap();
    let y = y.reshape((b, f, c_o, h_o, w_o)).unwrap()
        .permute((0, 2, 1, 3, 4)).unwrap()
        .contiguous().unwrap();
    let bias_5d = bias.reshape((1, c_out, 1, 1, 1)).unwrap();
    let y = y.broadcast_add(&bias_5d).unwrap();

    let rust_output: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();

    let total_max_diff: f32 = rust_output.iter().zip(py_output.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0f32, f32::max);

    println!("kt=1 max diff: {total_max_diff:.8}");
    println!("Rust [0,:2,0,0,0]: [{:.6}, {:.6}]", rust_output[0], rust_output[f*h_o*w_o]);
    println!("Rust [0,:2,1,0,0]: [{:.6}, {:.6}]", rust_output[h_o*w_o], rust_output[f*h_o*w_o + h_o*w_o]);

    assert!(total_max_diff < 0.001, "kt=1 path should match Python, max_diff={total_max_diff}");
}
