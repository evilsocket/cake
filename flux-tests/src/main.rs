//! Incremental FLUX.2 pipeline tests.
//! Compares Rust outputs against Python reference tensors saved in /tmp/flux_tests/.
//!
//! Usage: cargo run --release -- <test_number>
//!   test 1: Text encoder comparison

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

const TEST_DIR: &str = "/tmp/flux_tests";
const MODEL_REPO: &str = "black-forest-labs/FLUX.2-klein-4B";

fn load_f32(path: &str) -> Result<Vec<f32>> {
    let data = std::fs::read(path)?;
    Ok(data.chunks(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn compare(name: &str, rust: &[f32], python: &[f32]) {
    let n = rust.len().min(python.len());
    if n == 0 { println!("  {name}: EMPTY"); return; }

    let mut max_diff: f32 = 0.0;
    let mut max_idx = 0;
    let mut sum_sq_diff: f64 = 0.0;
    for i in 0..n {
        let d = (rust[i] - python[i]).abs();
        sum_sq_diff += (d as f64) * (d as f64);
        if d > max_diff { max_diff = d; max_idx = i; }
    }
    let rmse = (sum_sq_diff / n as f64).sqrt();
    let rust_norm: f64 = rust.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let py_norm: f64 = python.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();

    println!("  {name}:");
    println!("    Rust   norm={rust_norm:.4}, first5={:?}", &rust[..5.min(n)]);
    println!("    Python norm={py_norm:.4}, first5={:?}", &python[..5.min(n)]);
    println!("    max_diff={max_diff:.6} at idx {max_idx} (rust={:.6}, py={:.6})",
        rust[max_idx], python[max_idx]);
    println!("    RMSE={rmse:.6}, relative_norm_diff={:.6}", (rust_norm - py_norm).abs() / py_norm.max(1e-10));

    if max_diff < 0.5 {
        println!("    ✅ MATCH");
    } else if max_diff < 5.0 {
        println!("    ⚠️ CLOSE but not exact");
    } else {
        println!("    ❌ DIVERGED");
    }
}

fn test1_text_encoder() -> Result<()> {
    println!("=== Test 1: Text Encoder ===\n");

    let py_path = format!("{TEST_DIR}/txt_emb_python.bin");
    if !Path::new(&py_path).exists() {
        anyhow::bail!("Run Python reference first. Missing: {py_path}");
    }

    // Read metadata
    let meta = std::fs::read_to_string(format!("{TEST_DIR}/test1_meta.txt"))?;
    let mut real_len = 0usize;
    for line in meta.lines() {
        if let Some(v) = line.strip_prefix("real_len=") { real_len = v.parse()?; }
    }

    // Read Python token IDs
    let token_data = std::fs::read(format!("{TEST_DIR}/token_ids.bin"))?;
    let token_ids: Vec<u32> = token_data.chunks(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
        .collect();
    println!("Tokens: {} total, {} real, first5={:?}", token_ids.len(), real_len, &token_ids[..5]);

    // Load Rust text encoder on CPU (F32 for comparison)
    println!("Loading Rust text encoder on CPU...");
    let device = Device::Cpu;

    let encoder = cake_core::models::flux::text_encoder::FluxTextEncoder::load_model(
        &device, DType::F32, MODEL_REPO
    )?;

    // Create token tensor
    let token_tensor = Tensor::new(token_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Build attention mask: 1 for real tokens, 0 for padding
    let mut mask_data = vec![1.0f32; real_len];
    mask_data.resize(token_ids.len(), 0.0);
    let attn_mask = Tensor::new(mask_data.as_slice(), &device)?.unsqueeze(0)?;

    // Run encode
    println!("Running Rust text encoder...");
    let result = encoder.encode(&token_tensor, Some(&attn_mask))?;
    println!("Result shape: {:?}", result.shape());

    // Convert to f32 vec
    let rust_data: Vec<f32> = result.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;

    // Load Python F32 reference (fair comparison — same precision)
    let f32_path = format!("{TEST_DIR}/txt_emb_python_f32.bin");
    let py_path_used = if Path::new(&f32_path).exists() { &f32_path } else { &py_path };
    let python_data = load_f32(py_path_used)?;
    println!("Python ref: {} ({})", py_path_used, if py_path_used == &f32_path { "F32" } else { "BF16" });
    println!("Python data: {} floats, Rust data: {} floats\n", python_data.len(), rust_data.len());

    // Compare full output
    compare("full_output", &rust_data, &python_data);

    // Compare at specific positions
    let seq_len = 512;
    let dim = 7680;
    println!();
    compare("position [0,0,:5]", &rust_data[..5], &python_data[..5]);
    compare("position [0,100,:5]", &rust_data[100*dim..100*dim+5], &python_data[100*dim..100*dim+5]);
    compare("position [0,511,:5]", &rust_data[511*dim..511*dim+5], &python_data[511*dim..511*dim+5]);

    Ok(())
}

fn test3_transformer_internals() -> Result<()> {
    println!("=== Test 3: Transformer Internals ===\n");

    if !Path::new(&format!("{TEST_DIR}/test3_temb.bin")).exists() {
        anyhow::bail!("Run Python test3 script first");
    }

    let device = Device::cuda_if_available(0)?;

    // Load same inputs as test2
    let img_data = load_f32(&format!("{TEST_DIR}/test2_img.bin"))?;
    let txt_data = load_f32(&format!("{TEST_DIR}/test2_txt.bin"))?;
    let img = Tensor::from_vec(img_data, (1, 256, 128), &device)?;
    let txt = Tensor::from_vec(txt_data, (1, 512, 7680), &device)?;

    // Load transformer
    println!("Loading Rust transformer...");
    let transformer = cake_core::models::flux::transformer::FluxTransformerForwarder::load_model(
        &device, DType::BF16, MODEL_REPO
    )?;
    let model = &transformer.model;

    // Get weight dtype
    let w = model.x_embedder.weight().dtype();

    // 1. Embeddings
    let img_cast = img.to_dtype(w)?;
    let txt_cast = txt.to_dtype(w)?;
    use candle_core::Module as _;
    let img_emb = candle_core::Module::forward(&model.x_embedder, &img_cast)?;
    let txt_emb = candle_core::Module::forward(&model.context_embedder, &txt_cast)?;

    let r: Vec<f32> = img_emb.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_img_emb.bin"))?;
    compare("img_emb", &r, &p);

    let r: Vec<f32> = txt_emb.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_txt_emb.bin"))?;
    compare("txt_emb", &r, &p);

    // 2. Timestep
    let t_val = Tensor::new(&[1.0f32], &device)?;
    let t_emb = cake_core::models::flux::flux2_model::timestep_embedding(
        &t_val.to_dtype(DType::F32)?, 256, DType::F32
    )?.to_dtype(w)?;
    let vec = model.time_embedder.forward(&t_emb)?;

    let r: Vec<f32> = vec.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_temb.bin"))?;
    compare("temb", &r, &p);

    // 3. Modulation
    let vec_silu = vec.silu()?;
    let mod_img = candle_nn::Module::forward(&model.double_mod_img, &vec_silu)?;

    let r: Vec<f32> = mod_img.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_mod_img.bin"))?;
    compare("mod_img", &r, &p);

    // 4. PE
    let h_half = 16usize;
    let w_half = 16usize;
    let mut img_id_data = Vec::new();
    for h in 0..h_half { for w in 0..w_half {
        img_id_data.extend_from_slice(&[0.0f32, h as f32, w as f32, 0.0]);
    }}
    let img_ids_2d = Tensor::from_vec(img_id_data, (h_half * w_half, 4), &device)?;
    let mut txt_id_data = vec![0.0f32; 512 * 4];
    for i in 0..512 { txt_id_data[i * 4 + 3] = i as f32; }
    let txt_ids_2d = Tensor::from_vec(txt_id_data, (512, 4), &device)?;

    let (img_cos, img_sin) = model.pe_embedder.forward(&img_ids_2d)?;
    let (txt_cos, txt_sin) = model.pe_embedder.forward(&txt_ids_2d)?;
    let pe_cos = Tensor::cat(&[&txt_cos, &img_cos], 0)?.to_dtype(w)?;
    let pe_sin = Tensor::cat(&[&txt_sin, &img_sin], 0)?.to_dtype(w)?;

    let r: Vec<f32> = pe_cos.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_pe_cos.bin"))?;
    compare("pe_cos", &r, &p);

    // 5. Block 0
    let mod_txt = candle_nn::Module::forward(&model.double_mod_txt, &vec_silu)?;
    let img_mods = mod_img.chunk(6, candle_core::D::Minus1)?;
    let txt_mods = mod_txt.chunk(6, candle_core::D::Minus1)?;

    let (b0_img, b0_txt) = model.double_blocks[0].forward(
        &img_emb, &txt_emb, &img_mods, &txt_mods, &pe_cos, &pe_sin
    )?;

    let r: Vec<f32> = b0_img.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_block0_img.bin"))?;
    compare("block0_img", &r, &p);

    let r: Vec<f32> = b0_txt.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test3_block0_txt.bin"))?;
    compare("block0_txt", &r, &p);

    Ok(())
}

fn test4_block0_detail() -> Result<()> {
    println!("=== Test 4: Block 0 Detailed Intermediates ===\n");

    if !Path::new(&format!("{TEST_DIR}/test4_norm_img.bin")).exists() {
        anyhow::bail!("Run Python test4 first");
    }

    let device = Device::cuda_if_available(0)?;

    // Load inputs
    let img_data = load_f32(&format!("{TEST_DIR}/test2_img.bin"))?;
    let txt_data = load_f32(&format!("{TEST_DIR}/test2_txt.bin"))?;
    let img = Tensor::from_vec(img_data, (1, 256, 128), &device)?;
    let txt = Tensor::from_vec(txt_data, (1, 512, 7680), &device)?;

    // Load transformer
    println!("Loading transformer...");
    let transformer = cake_core::models::flux::transformer::FluxTransformerForwarder::load_model(
        &device, DType::BF16, MODEL_REPO
    )?;
    let model = &transformer.model;
    let w = model.x_embedder.weight().dtype();

    // Embeddings
    let img_cast = img.to_dtype(w)?;
    let txt_cast = txt.to_dtype(w)?;
    let img_emb = candle_core::Module::forward(&model.x_embedder, &img_cast)?;
    let txt_emb = candle_core::Module::forward(&model.context_embedder, &txt_cast)?;

    // Timestep + modulation
    let t_val = Tensor::new(&[1.0f32], &device)?;
    let t_emb = cake_core::models::flux::flux2_model::timestep_embedding(
        &t_val, 256, DType::F32
    )?.to_dtype(w)?;
    let vec = model.time_embedder.forward(&t_emb)?;
    let vec_silu = vec.silu()?;
    let mod_img = candle_nn::Module::forward(&model.double_mod_img, &vec_silu)?;
    let mod_txt = candle_nn::Module::forward(&model.double_mod_txt, &vec_silu)?;
    let img_mods = mod_img.chunk(6, candle_core::D::Minus1)?;
    let txt_mods = mod_txt.chunk(6, candle_core::D::Minus1)?;

    // PE
    let h_half = 16usize; let w_half = 16usize;
    let mut img_id_data = Vec::new();
    for h in 0..h_half { for ww in 0..w_half {
        img_id_data.extend_from_slice(&[0.0f32, h as f32, ww as f32, 0.0]);
    }}
    let img_ids_2d = Tensor::from_vec(img_id_data, (h_half * w_half, 4), &device)?;
    let mut txt_id_data = vec![0.0f32; 512 * 4];
    for i in 0..512 { txt_id_data[i * 4 + 3] = i as f32; }
    let txt_ids_2d = Tensor::from_vec(txt_id_data, (512, 4), &device)?;
    let (img_cos, img_sin) = model.pe_embedder.forward(&img_ids_2d)?;
    let (txt_cos, txt_sin) = model.pe_embedder.forward(&txt_ids_2d)?;
    let pe_cos = Tensor::cat(&[&txt_cos, &img_cos], 0)?.to_dtype(w)?;
    let pe_sin = Tensor::cat(&[&txt_sin, &img_sin], 0)?.to_dtype(w)?;

    // Now manually step through block 0 to compare intermediates
    println!("Running block 0 step by step...\n");

    // 1. Modulate (norm + scale + shift)
    // Our modulate: layer_norm(x) in F32, cast to BF16, then (1+scale)*norm + shift in BF16
    use cake_core::models::flux::flux2_model::modulate;
    let norm_img = modulate(&img_emb, &img_mods[0], &img_mods[1])?;
    let r: Vec<f32> = norm_img.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let p = load_f32(&format!("{TEST_DIR}/test4_norm_img.bin"))?;
    compare("norm_img (after modulate)", &r, &p);

    // 2. Attention output
    // Run full block to get the attention output
    let (b0_img, b0_txt) = model.double_blocks[0].forward(
        &img_emb, &txt_emb, &img_mods, &txt_mods, &pe_cos, &pe_sin
    )?;

    // Compare img_after_attn (which includes gated residual but NOT MLP)
    // Unfortunately our block does everything at once. Let's compare the final output instead.
    let p_img = load_f32(&format!("{TEST_DIR}/test3_block0_img.bin"))?;
    let r_img: Vec<f32> = b0_img.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    compare("block0_img (full output)", &r_img, &p_img);

    let p_txt = load_f32(&format!("{TEST_DIR}/test3_block0_txt.bin"))?;
    let r_txt: Vec<f32> = b0_txt.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    compare("block0_txt (full output)", &r_txt, &p_txt);

    // Also compare the intermediate: img_after_attn (before MLP)
    let p_aft = load_f32(&format!("{TEST_DIR}/test4_img_after_attn.bin"))?;
    // We don't have this from Rust since block does it internally.
    // But we can compare attn output
    let p_attn = load_f32(&format!("{TEST_DIR}/test4_attn_out_img.bin"))?;
    println!("\n  Python attn_out_img: norm={:.4}, first3={:?}",
        p_attn.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
        &p_attn[..3]);
    println!("  Python img_after_attn: norm={:.4}, first3={:?}",
        p_aft.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt(),
        &p_aft[..3]);

    Ok(())
}

fn test2_transformer() -> Result<()> {
    println!("=== Test 2: Transformer (single forward) ===\n");

    let meta_path = format!("{TEST_DIR}/test2_meta.txt");
    if !Path::new(&meta_path).exists() {
        anyhow::bail!("Run Python test2 script first. Missing: {meta_path}");
    }

    // Read metadata
    let meta = std::fs::read_to_string(&meta_path)?;
    let mut h_half = 16usize;
    let mut w_half = 16usize;
    for line in meta.lines() {
        if let Some(v) = line.strip_prefix("h_half=") { h_half = v.parse()?; }
        if let Some(v) = line.strip_prefix("w_half=") { w_half = v.parse()?; }
    }

    // Load inputs saved by Python
    let img_data = load_f32(&format!("{TEST_DIR}/test2_img.bin"))?;
    let txt_data = load_f32(&format!("{TEST_DIR}/test2_txt.bin"))?;
    println!("img: {} floats, txt: {} floats", img_data.len(), txt_data.len());

    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);

    // Create tensors on GPU
    let img = Tensor::from_vec(img_data, (1, h_half * w_half, 128), &device)?;
    let txt = Tensor::from_vec(txt_data, (1, 512, 7680), &device)?;
    let t_vec = Tensor::full(1.0f32, 1, &device)?;

    // Build IDs (matching Python's construction)
    let mut img_id_data = Vec::new();
    for h in 0..h_half {
        for w in 0..w_half {
            img_id_data.extend_from_slice(&[0.0f32, h as f32, w as f32, 0.0]);
        }
    }
    let img_ids = Tensor::from_vec(img_id_data, (1, h_half * w_half, 4), &device)?;

    let mut txt_id_data = vec![0.0f32; 512 * 4];
    for i in 0..512 { txt_id_data[i * 4 + 3] = i as f32; }
    let txt_ids = Tensor::from_vec(txt_id_data, (1, 512, 4), &device)?;

    // Load Rust transformer
    println!("Loading Rust transformer...");
    let transformer = cake_core::models::flux::transformer::FluxTransformerForwarder::load_model(
        &device, DType::BF16, MODEL_REPO
    )?;

    // Run forward
    println!("Running forward...");
    let pred = transformer.forward_direct(&img, &img_ids, &txt, &txt_ids, &t_vec)?;
    println!("pred shape: {:?}", pred.shape());

    // Convert to f32 for comparison
    let rust_data: Vec<f32> = pred.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let python_data = load_f32(&format!("{TEST_DIR}/test2_pred_python.bin"))?;

    println!();
    compare("transformer_pred", &rust_data, &python_data);

    // Also compare first and last 5 elements
    let n = rust_data.len();
    println!();
    compare("pred_first5", &rust_data[..5], &python_data[..5]);
    compare("pred_last5", &rust_data[n-5..], &python_data[n-5..]);

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_module_path(false)
        .format_target(false)
        .init();

    let test_num: usize = std::env::args().nth(1)
        .unwrap_or("1".to_string())
        .parse()
        .unwrap_or(1);

    match test_num {
        1 => test1_text_encoder()?,
        2 => test2_transformer()?,
        3 => test3_transformer_internals()?,
        4 => test4_block0_detail()?,
        _ => println!("Unknown test: {test_num}. Available: 1, 2, 3, 4"),
    }

    Ok(())
}
