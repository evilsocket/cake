//! Compare Wan VAE decode against Python reference.
//! Run with: cargo test --test wan_vae_compare -- --ignored --nocapture

use candle_core::{Device, Tensor, DType};

#[test]
#[ignore]
fn test_wan_vae_decode_vs_python() {
    let ref_path = "/tmp/wan_vae_test.json";
    if !std::path::Path::new(ref_path).exists() {
        eprintln!("Skipping: {} not found", ref_path);
        return;
    }

    let vae_path = "/home/a/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b/vae/wan_vae_cake.safetensors";
    if !std::path::Path::new(vae_path).exists() {
        eprintln!("Skipping: {} not found", vae_path);
        return;
    }

    let data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(ref_path).unwrap()).unwrap();

    let dev = Device::Cpu;

    let input_data: Vec<f32> = data["input"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let input_shape: Vec<usize> = data["input_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let py_output: Vec<f32> = data["output"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let output_shape: Vec<usize> = data["output_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();

    let input = Tensor::from_vec(input_data, input_shape.as_slice(), &dev).unwrap();

    // Load VAE
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(
            &[std::path::PathBuf::from(vae_path)],
            DType::F32,
            &dev,
        ).unwrap()
    };

    let cfg = cake_core::models::wan::vendored::config::WanVaeConfig::default();
    println!("Loading VAE...");
    let mut decoder = cake_core::models::wan::vendored::vae::WanVaeDecoder::load(vb, &cfg).unwrap();

    println!("Decoding...");
    let output = decoder.decode(&input).unwrap();
    let rust_output: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    println!("Rust output shape: {:?}", output.shape());
    println!("Python output shape: {:?}", output_shape);
    println!("Rust first 8: {:?}", &rust_output[..8]);
    println!("Python first 8: {:?}", &py_output[..8]);

    let max_diff: f32 = rust_output.iter().zip(py_output.iter())
        .map(|(r, p)| (r - p).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f64 = rust_output.iter().zip(py_output.iter())
        .map(|(r, p)| (r - p).abs() as f64)
        .sum::<f64>() / rust_output.len().min(py_output.len()) as f64;

    println!("Max diff: {max_diff:.6}");
    println!("Mean diff: {mean_diff:.6}");

    if max_diff > 0.1 {
        println!("\nLARGE DIFF — checking element-by-element:");
        for i in 0..20.min(rust_output.len()) {
            let diff = (rust_output[i] - py_output[i]).abs();
            if diff > 0.01 {
                println!("  [{i}] rust={:.6} python={:.6} diff={:.6}", rust_output[i], py_output[i], diff);
            }
        }
    }
}
