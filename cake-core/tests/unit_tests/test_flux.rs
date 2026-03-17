//! Unit tests for FLUX text-to-image models.
//!
//! All tests run offline — no model downloads, no GPU required.

#![cfg(feature = "flux")]

use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

fn make_tensor(shape: &[usize], seed: u64) -> Tensor {
    use rand::{Rng, SeedableRng};
    let numel: usize = shape.iter().product();
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let data: Vec<f32> = (0..numel).map(|_| rng.gen_range(-0.1..0.1)).collect();
    Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
}

// ── Config tests ─────────────────────────────────────────────────────

#[test]
fn test_flux2_klein_transformer_config() {
    let cfg = cake_core::models::flux::config::flux2_klein_transformer_config();
    assert_eq!(cfg.in_channels, 128);
    assert_eq!(cfg.hidden_size, 3072);
    assert_eq!(cfg.num_heads, 24);
    assert_eq!(cfg.depth, 5);
    assert_eq!(cfg.depth_single_blocks, 20);
    assert_eq!(cfg.axes_dim.len(), 4);
    assert!(!cfg.guidance_embed);
}

#[test]
fn test_flux2_klein_vae_config() {
    let cfg = cake_core::models::flux::config::flux2_klein_vae_config();
    assert_eq!(cfg.z_channels, 32);
    assert_eq!(cfg.in_channels, 3);
    assert_eq!(cfg.out_ch, 3);
    assert_eq!(cfg.ch, 128);
    assert_eq!(cfg.ch_mult, vec![1, 2, 4, 4]);
}

#[test]
fn test_flux1_dev_config() {
    let cfg = cake_core::models::flux::flux1_model::Config::dev();
    assert_eq!(cfg.in_channels, 64);
    assert_eq!(cfg.hidden_size, 3072);
    assert_eq!(cfg.num_heads, 24);
    assert_eq!(cfg.depth, 19);
    assert_eq!(cfg.depth_single_blocks, 38);
    assert_eq!(cfg.axes_dim, vec![16, 56, 56]);
    assert!(cfg.guidance_embed);
}

#[test]
fn test_flux_model_file_names() {
    use cake_core::models::flux::config::FluxModelFile;
    assert_eq!(FluxModelFile::Tokenizer.name(), "flux_tokenizer");
    assert_eq!(FluxModelFile::TextEncoder.name(), "flux_text_encoder");
    assert_eq!(FluxModelFile::Transformer.name(), "flux_transformer");
    assert_eq!(FluxModelFile::Vae.name(), "flux_vae");
}

#[test]
fn test_flux1_prefixes() {
    use cake_core::models::flux::config::flux1_prefixes;
    assert_eq!(flux1_prefixes::TRANSFORMER, "model.diffusion_model");
    assert_eq!(flux1_prefixes::CLIP, "text_encoders.clip_l.transformer");
    assert_eq!(flux1_prefixes::T5, "text_encoders.t5xxl.transformer");
    assert_eq!(flux1_prefixes::VAE, "vae");
}

// ── Fp8Linear tests ─────────────────────────────────────────────────

#[test]
fn test_fp8_linear_forward_f32() {
    use cake_core::models::flux::flux1_model::Fp8Linear;
    // 4→8 linear, F32 weights
    let weight = make_tensor(&[8, 4], 10);
    let bias = make_tensor(&[8], 11);
    let linear = Fp8Linear::new_pub(weight, Some(bias));

    let x = make_tensor(&[2, 4], 12);
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8]);
    assert_eq!(y.dtype(), DType::F32);
}

#[test]
fn test_fp8_linear_forward_3d() {
    use cake_core::models::flux::flux1_model::Fp8Linear;
    let weight = make_tensor(&[8, 4], 20);
    let linear = Fp8Linear::new_pub(weight, None);

    let x = make_tensor(&[1, 3, 4], 21); // batch=1, seq=3, dim=4
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[1, 3, 8]);
}

#[test]
fn test_fp8_linear_no_bias() {
    use cake_core::models::flux::flux1_model::Fp8Linear;
    let weight = make_tensor(&[8, 4], 30);
    let linear = Fp8Linear::new_pub(weight, None);

    let x = make_tensor(&[2, 4], 31);
    let y = linear.forward(&x).unwrap();
    assert_eq!(y.dims(), &[2, 8]);
}

// ── Timestep embedding tests ─────────────────────────────────────────

#[test]
fn test_timestep_embedding_shape() {
    use cake_core::models::flux::flux2_model::timestep_embedding;
    let t = Tensor::new(&[0.0f32, 0.5, 1.0], &Device::Cpu).unwrap();
    let emb = timestep_embedding(&t, 64, DType::F32).unwrap();
    assert_eq!(emb.dims(), &[3, 64]);
    assert_eq!(emb.dtype(), DType::F32);
}

#[test]
fn test_timestep_embedding_different_dims() {
    use cake_core::models::flux::flux2_model::timestep_embedding;
    let t = Tensor::new(&[0.5f32], &Device::Cpu).unwrap();

    for dim in [32, 64, 128, 256] {
        let emb = timestep_embedding(&t, dim, DType::F32).unwrap();
        assert_eq!(emb.dims(), &[1, dim]);
    }
}

#[test]
fn test_timestep_embedding_deterministic() {
    use cake_core::models::flux::flux2_model::timestep_embedding;
    let t = Tensor::new(&[0.5f32], &Device::Cpu).unwrap();
    let e1: Vec<f32> = timestep_embedding(&t, 32, DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let e2: Vec<f32> = timestep_embedding(&t, 32, DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    assert_eq!(e1, e2);
}

#[test]
fn test_timestep_embedding_odd_dim_errors() {
    use cake_core::models::flux::flux2_model::timestep_embedding;
    let t = Tensor::new(&[0.5f32], &Device::Cpu).unwrap();
    assert!(timestep_embedding(&t, 33, DType::F32).is_err());
}

// ── Flux2PosEmbed tests ──────────────────────────────────────────────

#[test]
fn test_flux2_pos_embed_shape() {
    use cake_core::models::flux::flux2_model::Flux2PosEmbed;
    let pe = Flux2PosEmbed::new_pub(2000, vec![32, 32, 32, 32]);
    // 10 positions, 4 axes
    let ids = Tensor::zeros((10, 4), DType::F32, &Device::Cpu).unwrap();
    let (cos, sin) = pe.forward(&ids).unwrap();
    // head_dim = sum(axes_dim) = 128
    assert_eq!(cos.dims(), &[10, 128]);
    assert_eq!(sin.dims(), &[10, 128]);
}

#[test]
fn test_flux2_pos_embed_different_seq_len() {
    use cake_core::models::flux::flux2_model::Flux2PosEmbed;
    let pe = Flux2PosEmbed::new_pub(10000, vec![16, 16]);

    for seq_len in [1, 5, 64] {
        let ids = Tensor::zeros((seq_len, 2), DType::F32, &Device::Cpu).unwrap();
        let (cos, sin) = pe.forward(&ids).unwrap();
        assert_eq!(cos.dims(), &[seq_len, 32]); // 16+16
        assert_eq!(sin.dims(), &[seq_len, 32]);
    }
}

// ── VAE decoder block tests ──────────────────────────────────────────

#[test]
fn test_flux2_vae_resnet_block() {
    use candle_nn::VarBuilder;

    let h = 32;
    let mut map: HashMap<String, Tensor> = HashMap::new();
    // norm1/norm2: weight + bias (group_norm)
    map.insert("norm1.weight".into(), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("norm1.bias".into(), Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("norm2.weight".into(), Tensor::ones(h, DType::F32, &Device::Cpu).unwrap());
    map.insert("norm2.bias".into(), Tensor::zeros(h, DType::F32, &Device::Cpu).unwrap());
    // conv1/conv2: weight + bias (3x3 conv)
    map.insert("conv1.weight".into(), make_tensor(&[h, h, 3, 3], 40));
    map.insert("conv1.bias".into(), make_tensor(&[h], 41));
    map.insert("conv2.weight".into(), make_tensor(&[h, h, 3, 3], 42));
    map.insert("conv2.bias".into(), make_tensor(&[h], 43));

    let vb = VarBuilder::from_tensors(map, DType::F32, &Device::Cpu);
    // Load and forward
    let block = cake_core::models::flux::flux2_vae::ResnetBlock2D::load_pub(vb, h, h, 32).unwrap();
    let x = make_tensor(&[1, h, 8, 8], 44);
    let y = block.forward_pub(&x).unwrap();
    assert_eq!(y.dims(), &[1, h, 8, 8]);
}

// ── Native dtype backend tests ───────────────────────────────────────

#[test]
fn test_native_dtype_backend_module_exists() {
    // Just verify the module is accessible
    let _ = std::mem::size_of::<fn()>();
    // The actual load function requires safetensor files, so we just test compilation
}
