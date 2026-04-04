//! Compare Wan transformer single forward pass against Python reference.
//! Run with: cargo test --test wan_forward_compare -- --ignored --nocapture

use candle_core::{DType, Device, IndexOp, Tensor};
use std::path::PathBuf;

#[test]
#[ignore] // requires model weights + reference JSON
fn test_wan_single_forward_vs_python() {
    let ref_path = "/tmp/wan_single_fwd.json";
    if !std::path::Path::new(ref_path).exists() {
        eprintln!("Skipping: {} not found (run Python reference first)", ref_path);
        return;
    }

    let wan_path = "/home/a/.cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b";
    let transformer_dir = format!("{}/transformer", wan_path);
    if !std::path::Path::new(&transformer_dir).exists() {
        eprintln!("Skipping: transformer weights not found");
        return;
    }

    // Load reference data
    let data: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(ref_path).unwrap()).unwrap();

    let latents_data: Vec<f32> = data["latents"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let latents_shape: Vec<usize> = data["latents_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let context_data: Vec<f32> = data["context"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let context_shape: Vec<usize> = data["context_shape"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let timestep_val: f32 = data["timestep"][0].as_f64().unwrap() as f32;
    let py_output: Vec<f32> = data["output"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap() as f32).collect();

    let dev = Device::Cpu;

    let latents = Tensor::from_vec(latents_data, latents_shape.as_slice(), &dev).unwrap();
    let context = Tensor::from_vec(context_data, context_shape.as_slice(), &dev).unwrap();
    let timestep = Tensor::from_slice(&[timestep_val], 1, &dev).unwrap();

    // Load Wan config
    let config_path = PathBuf::from(&transformer_dir).join("config.json");
    let cfg = cake_core::models::wan::vendored::config::WanTransformerConfig::from_path(&config_path).unwrap();

    println!("Config: hidden={}, heads={}, layers={}", cfg.hidden_size, cfg.num_attention_heads, cfg.num_layers);

    // Load transformer weights
    let weight_dir = PathBuf::from(&transformer_dir);
    let mut weight_files: Vec<PathBuf> = std::fs::read_dir(&weight_dir).unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |e| e == "safetensors"))
        .collect();
    weight_files.sort();

    println!("Loading {} weight files on CPU...", weight_files.len());
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, DType::F32, &dev).unwrap()
    };

    // Remap: our model requests internal keys → rename_f maps to diffusers keys in safetensors
    let vb = vb.rename_f(|key| {
        let mut k = key.to_string();
        // head -> proj_out
        k = k.replace("head.head.", "proj_out.");
        if k == "head.modulation" || k.starts_with("head.modulation.") {
            k = k.replace("head.modulation", "scale_shift_table");
        }
        if k.contains(".modulation") && k.starts_with("blocks.") {
            k = k.replace(".modulation", ".scale_shift_table");
        }
        k = k.replace(".norm3.", ".norm2.");
        k = k.replace(".ffn.0.", ".ffn.net.0.proj.");
        k = k.replace(".ffn.2.", ".ffn.net.2.");
        k = k.replace(".self_attn.", ".attn1.");
        k = k.replace(".cross_attn.", ".attn2.");
        if k.contains(".attn1.") || k.contains(".attn2.") {
            k = k.replace(".q.weight", ".to_q.weight");
            k = k.replace(".q.bias", ".to_q.bias");
            k = k.replace(".k.weight", ".to_k.weight");
            k = k.replace(".k.bias", ".to_k.bias");
            k = k.replace(".v.weight", ".to_v.weight");
            k = k.replace(".v.bias", ".to_v.bias");
            k = k.replace(".o.weight", ".to_out.0.weight");
            k = k.replace(".o.bias", ".to_out.0.bias");
        }
        k = k.replace("text_embedding.0.", "condition_embedder.text_embedder.linear_1.");
        k = k.replace("text_embedding.2.", "condition_embedder.text_embedder.linear_2.");
        k = k.replace("time_embedding.0.", "condition_embedder.time_embedder.linear_1.");
        k = k.replace("time_embedding.2.", "condition_embedder.time_embedder.linear_2.");
        k = k.replace("time_projection.1.", "condition_embedder.time_proj.");
        k
    });

    println!("Loading WanModel (all blocks)...");
    let model = cake_core::models::wan::vendored::model::WanModel::load(vb, &cfg).unwrap();

    // First, check RoPE values
    {
        use cake_core::models::wan::vendored::rope::precompute_wan_rope_3d;
        let (t_dim, h_dim, w_dim) = cfg.rope_dims();
        let (cos, sin) = precompute_wan_rope_3d(2, 8, 8, t_dim, h_dim, w_dim, 10000.0, &dev).unwrap();
        println!("RoPE cos shape: {:?}", cos.shape());
        // Position 0
        let cos_flat: Vec<f32> = cos.flatten_all().unwrap().to_vec1().unwrap();
        let head_dim = t_dim + h_dim + w_dim;
        println!("cos[pos=0, :10]: {:?}", &cos_flat[..10]);
        // Position 1 (w=1)
        let off1 = head_dim;
        println!("cos[pos=1, 86:90] (w-part): {:?}", &cos_flat[off1+86..off1+90]);
        // Position 8 (h=1, w=0)
        let off8 = 8 * head_dim;
        println!("cos[pos=8, 44:48] (h-part): {:?}", &cos_flat[off8+44..off8+48]);
    }

    println!("Running forward_setup...");
    let (_, _, f_lat, h_lat, w_lat) = (
        latents_shape[0], latents_shape[1], latents_shape[2], latents_shape[3], latents_shape[4]
    );
    let height = h_lat;
    let width = w_lat;

    // Run forward_setup to check patch embedding
    let (hidden, temb, timestep_proj, ctx_emb, rope_cos, rope_sin) =
        model.forward_setup(&latents, &timestep, &context, f_lat, height, width).unwrap();

    // Compare patch embedding with Python reference
    let pe0: Vec<f32> = hidden.get(0).unwrap().get(0).unwrap().narrow(0, 0, 5).unwrap().to_vec1().unwrap();
    let pe1: Vec<f32> = hidden.get(0).unwrap().get(1).unwrap().narrow(0, 0, 5).unwrap().to_vec1().unwrap();
    println!("Rust  patch_embed [0,0,:5]: {:?}", pe0);
    println!("Python patch_embed [0,0,:5]: [-0.13005, -0.23863, 0.10933, -0.34347, -0.59145]");
    println!("Rust  patch_embed [0,1,:5]: {:?}", pe1);
    println!("Python patch_embed [0,1,:5]: [-0.34251, 0.12603, 0.07332, 0.40211, -0.33028]");

    // Run just first block
    {
        // Access model's first block through forward_blocks with 1-block model
        // Actually, let's just call forward_blocks on the full model and check
    }
    let hidden = model.forward_blocks(hidden, &ctx_emb, &timestep_proj, &rope_cos, &rope_sin).unwrap();
    let b0_0: Vec<f32> = hidden.get(0).unwrap().get(0).unwrap().narrow(0, 0, 5).unwrap().to_vec1().unwrap();
    let b0_1: Vec<f32> = hidden.get(0).unwrap().get(1).unwrap().narrow(0, 0, 5).unwrap().to_vec1().unwrap();
    // Python block 0 output [0,0,:5]: [-0.044, 0.177, 0.166, -0.030, -0.552]
    println!("Rust  after all 30 blocks [0,0,:5]: {:?}", b0_0);
    println!("Rust  after all 30 blocks [0,1,:5]: {:?}", b0_1);
    // Element 1 is anomalously large (12.69) suggesting a weight loading issue

    let output = model.forward_finalize(&hidden, &temb, f_lat, height, width).unwrap();
    let output_flat: Vec<f32> = output.flatten_all().unwrap().to_vec1().unwrap();

    println!("Output shape: {:?}", output.shape());
    println!("Rust  first 10: {:?}", &output_flat[..10]);
    println!("Python first 10: {:?}", &py_output[..10]);

    // Compare
    let mut max_diff: f32 = 0.0;
    let mut sum_diff: f64 = 0.0;
    for (i, (r, p)) in output_flat.iter().zip(py_output.iter()).enumerate() {
        let diff = (r - p).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        sum_diff += diff as f64;
        if i < 20 && diff > 0.01 {
            println!("  DIFF[{i}]: rust={r:.6}, python={p:.6}, diff={diff:.6}");
        }
    }
    let mean_diff = sum_diff / output_flat.len() as f64;
    println!("Max diff: {max_diff:.6}");
    println!("Mean diff: {mean_diff:.6}");

    let rust_std: f32 = {
        let mean: f32 = output_flat.iter().sum::<f32>() / output_flat.len() as f32;
        let var: f32 = output_flat.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / output_flat.len() as f32;
        var.sqrt()
    };
    let py_std: f32 = {
        let mean: f32 = py_output.iter().sum::<f32>() / py_output.len() as f32;
        let var: f32 = py_output.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / py_output.len() as f32;
        var.sqrt()
    };
    println!("Rust std: {rust_std:.6}, Python std: {py_std:.6}, ratio: {:.4}", rust_std / py_std);

    assert!(max_diff < 0.5, "Single forward pass should match Python closely, max_diff={max_diff}");
}
