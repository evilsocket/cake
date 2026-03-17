//! FLUX.1-dev image generation pipeline.
//!
//! Uses a single bundled FP8 checkpoint from Comfy-Org/flux1-dev containing
//! transformer (F8E4M3), CLIP-L (F16), T5-XXL (F8E4M3), and VAE (F32).
//! Components are loaded sequentially to fit in 16GB VRAM.

use crate::cake::Context;
use crate::models::flux::clip_encoder;
use crate::models::flux::config::{flux1_prefixes, Flux1ModelFile};
use crate::models::flux::flux1_model::{Config, Flux1Transformer};
use crate::models::flux::t5_encoder;
use crate::models::{Generator, ImageGenerator};
use crate::ImageGenerationArgs;
use anyhow::Result;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::flux::autoencoder::{
    self as flux_ae, AutoEncoder,
};
use candle_transformers::models::flux::sampling;
use image::{ImageBuffer, Rgb};
use log::info;
use tokenizers::Tokenizer;

pub struct Flux1Gen {
    clip_tokenizer: Tokenizer,
    t5_tokenizer: Tokenizer,
    checkpoint_path: std::path::PathBuf,
    context: Context,
    height: usize,
    width: usize,
}

// Dummy shardable — FLUX.1 is local-only for now
#[derive(Debug)]
pub struct Flux1Shardable;

impl std::fmt::Display for Flux1Shardable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Flux1Shardable (local-only)")
    }
}

#[async_trait]
impl crate::cake::Forwarder for Flux1Shardable {
    fn load(_name: String, _ctx: &Context) -> anyhow::Result<Box<Self>>
    where
        Self: Sized,
    {
        anyhow::bail!("FLUX.1-dev does not support distributed inference yet")
    }

    async fn forward(
        &self,
        _x: &candle_core::Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<candle_core::Tensor> {
        anyhow::bail!("FLUX.1-dev does not support distributed inference yet")
    }

    async fn forward_mut(
        &mut self,
        _x: &candle_core::Tensor,
        _index_pos: usize,
        _block_idx: usize,
        _ctx: &mut Context,
    ) -> anyhow::Result<candle_core::Tensor> {
        anyhow::bail!("FLUX.1-dev does not support distributed inference yet")
    }

    fn layer_name(&self) -> &str {
        "flux1_shardable"
    }
}

#[async_trait]
impl Generator for Flux1Gen {
    type Shardable = Flux1Shardable;
    const MODEL_NAME: &'static str = "flux1";

    async fn load(context: &mut Context) -> Result<Option<Box<Self>>> {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(std::env::temp_dir)
            .to_string_lossy()
            .to_string();

        // Check if --model points to a direct file path
        let model_path = std::path::Path::new(&context.args.model);
        let checkpoint_path = if model_path.is_file() {
            info!("Using local checkpoint: {}", model_path.display());
            model_path.to_path_buf()
        } else {
            info!("Downloading FLUX.1-dev checkpoint...");
            Flux1ModelFile::Checkpoint.get(&cache_dir)?
        };

        // Download tokenizers
        info!("Downloading tokenizers...");
        let clip_tok_path = Flux1ModelFile::ClipTokenizer.get(&cache_dir)?;
        let t5_tok_path = Flux1ModelFile::T5Tokenizer.get(&cache_dir)?;
        info!("All FLUX.1-dev files ready");

        // Load tokenizers
        let clip_tokenizer = Tokenizer::from_file(&clip_tok_path)
            .map_err(|e| anyhow!("failed to load CLIP tokenizer: {e}"))?;

        // T5 uses SentencePiece — try loading as tokenizers JSON first,
        // fall back to building from spiece.model
        let t5_tokenizer = load_t5_tokenizer(&t5_tok_path)?;

        let height = context.args.flux_args.height;
        let width = context.args.flux_args.width;

        Ok(Some(Box::new(Self {
            clip_tokenizer,
            t5_tokenizer,
            checkpoint_path,
            context: context.clone(),
            height,
            width,
        })))
    }
}

#[async_trait]
impl ImageGenerator for Flux1Gen {
    async fn generate_image<F>(
        &mut self,
        args: &ImageGenerationArgs,
        mut callback: F,
    ) -> Result<(), anyhow::Error>
    where
        F: FnMut(Vec<ImageBuffer<Rgb<u8>, Vec<u8>>>) + Send + 'static,
    {
        let ImageGenerationArgs {
            image_prompt,
            image_seed,
            ..
        } = args;

        let num_steps = self.context.args.flux_args.num_steps;
        let guidance_scale = self.context.args.flux_args.guidance_scale;
        let dev = self.context.device.clone();

        if let Some(seed) = image_seed {
            self.context.device.set_seed(*seed)?;
        }

        // ── 1. Tokenize ─────────────────────────────────────────────────────
        info!("Tokenizing prompt: \"{}\"", image_prompt);

        // CLIP tokenization (max 77 tokens)
        let clip_tokens = self
            .clip_tokenizer
            .encode(image_prompt.as_str(), true)
            .map_err(|e| anyhow!("CLIP tokenizer: {e}"))?;
        let clip_ids = Tensor::new(clip_tokens.get_ids(), &Device::Cpu)?.unsqueeze(0)?;
        info!("CLIP: {} tokens", clip_tokens.get_ids().len());

        // T5 tokenization (max 512 tokens)
        let t5_tokens = self
            .t5_tokenizer
            .encode(image_prompt.as_str(), true)
            .map_err(|e| anyhow!("T5 tokenizer: {e}"))?;
        let mut t5_ids = t5_tokens.get_ids().to_vec();
        // Pad to max 512
        const T5_MAX_LEN: usize = 512;
        if t5_ids.len() > T5_MAX_LEN {
            t5_ids.truncate(T5_MAX_LEN);
        }
        let t5_id_tensor = Tensor::new(t5_ids.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
        info!("T5: {} tokens", t5_ids.len());

        // ── 2. CLIP encode on CPU ───────────────────────────────────────────
        info!("Running CLIP-L text encoder (CPU)...");
        let clip_embed = clip_encoder::encode_clip(
            &self.checkpoint_path,
            flux1_prefixes::CLIP,
            &clip_ids,
            &Device::Cpu,
        )?;
        info!(
            "CLIP embedding: {:?}, dtype={:?}",
            clip_embed.shape(),
            clip_embed.dtype()
        );

        // ── 3. T5 encode (GPU when available, CPU fallback) ─────────────────
        info!("Running T5-XXL text encoder...");
        let t5_embed = t5_encoder::encode_t5(
            &self.checkpoint_path,
            flux1_prefixes::T5,
            &t5_id_tensor,
            &dev,
        )?;
        info!(
            "T5 embedding: {:?}, dtype={:?}",
            t5_embed.shape(),
            t5_embed.dtype()
        );

        // ── 4. Prepare sampling state ──────────────────────────────────────
        info!("Preparing noise and sampling state...");

        // Generate noise latents
        let noise = sampling::get_noise(1, self.height, self.width, &dev)?;
        let noise = noise.to_dtype(DType::F32)?;

        // Build state (packs noise into patches, creates position IDs)
        let t5_embed = t5_embed.to_dtype(DType::F32)?;
        let clip_embed = clip_embed.to_dtype(DType::F32)?;
        let state = sampling::State::new(&t5_embed, &clip_embed, &noise)?;

        // Move all state tensors to GPU
        let img = state.img.to_device(&dev)?;
        let img_ids = state.img_ids.to_device(&dev)?;
        let txt = state.txt.to_device(&dev)?;
        let txt_ids = state.txt_ids.to_device(&dev)?;
        let vec = state.vec.to_device(&dev)?;

        // Schedule with time shift
        let image_seq_len = img.dim(1)?;
        let schedule = sampling::get_schedule(
            num_steps,
            Some((image_seq_len, 0.5, 1.15)),
        );

        info!(
            "State ready: img={:?}, txt={:?}, vec={:?}, {} steps",
            img.shape(),
            txt.shape(),
            vec.shape(),
            num_steps,
        );

        // Drop CPU embeddings to free RAM before loading transformer
        drop(t5_embed);
        drop(clip_embed);
        drop(state);

        // ── 5. Load transformer on GPU ─────────────────────────────────────
        info!("Loading FLUX.1 transformer (FP8 on GPU, ~12GB)...");
        let cfg = Config::dev();

        // Load with native dtypes: F8E4M3 tensors stay in F8 on GPU (~12GB).
        // During forward, Fp8Linear casts weights to BF16 per-layer (temporary ~60MB).
        // Total VRAM: ~12GB (F8 weights) + ~2GB (activations) ≈ 14GB.
        let vb = unsafe {
            crate::utils::native_dtype_backend::load_native_dtype_var_builder(
                &[self.checkpoint_path.clone()],
                DType::F32,
                &dev,
            )?
        };
        let vb = vb.pp(flux1_prefixes::TRANSFORMER);
        let transformer = Flux1Transformer::new(&cfg, vb)?;
        info!(
            "Transformer loaded ({} double + {} single blocks)",
            cfg.depth, cfg.depth_single_blocks
        );

        // First forward call triggers F8→F16 dequant and caching for all Fp8Linear layers.
        // After this, F8 weights can be garbage-collected (only F16 cache is kept).
        // The F8 originals will be dropped when Fp8Linear's RwLock cache is populated.

        // ── 6. Denoise ─────────────────────────────────────────────────────
        info!("Starting denoising ({num_steps} steps, guidance={guidance_scale})...");
        let b_sz = img.dim(0)?;
        let guidance_tensor = Tensor::full(guidance_scale as f32, b_sz, &dev)?
            .to_dtype(DType::F16)?;

        // Cast state tensors to F16 for 2x faster matmul on A100
        let img_ids = img_ids.to_dtype(DType::F32)?; // IDs stay F32 for position encoding
        let txt = txt.to_dtype(DType::F16)?;
        let txt_ids = txt_ids.to_dtype(DType::F32)?;
        let vec = vec.to_dtype(DType::F16)?;

        let mut img = img.to_dtype(DType::F16)?;
        for window in schedule.windows(2) {
            let (t_curr, t_prev) = match window {
                [a, b] => (a, b),
                _ => continue,
            };
            let t_vec = Tensor::full(*t_curr as f32, b_sz, &dev)?
                .to_dtype(DType::F16)?;
            let pred = transformer.forward(
                &img,
                &img_ids,
                &txt,
                &txt_ids,
                &t_vec,
                &vec,
                Some(&guidance_tensor),
            )?;
            // Debug: check for NaN/zero in prediction
            let pred_abs_mean = pred.abs()?.mean_all()?.to_scalar::<f32>().unwrap_or(-1.0);
            img = (img + pred * (t_prev - t_curr))?;
            let img_abs_mean = img.abs()?.mean_all()?.to_scalar::<f32>().unwrap_or(-1.0);
            info!("  step: t={t_curr:.3}→{t_prev:.3} pred_mean={pred_abs_mean:.6} img_mean={img_abs_mean:.6}");
        }

        // ── 7. Unpack latents and move to CPU ────────────────────────────
        let img = sampling::unpack(&img, self.height, self.width)?;
        info!("Unpacked latents: {:?}", img.shape());

        // Move latents to CPU BEFORE dropping transformer.
        // This ensures the output is preserved while we free GPU memory.
        let img_cpu = img.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;

        // Now free ALL GPU memory — transformer (~12GB) + denoising state
        drop(img);
        drop(transformer);
        drop(txt);
        drop(txt_ids);
        drop(img_ids);
        drop(vec);
        drop(guidance_tensor);
        dev.synchronize()?;
        info!("Freed transformer VRAM");

        // ── 8. VAE decode ──────────────────────────────────────────────────
        // VAE is ~500MB in F32. After dropping transformer (~12GB F8) there
        // should be plenty of VRAM. Load VAE in BF16 to be conservative.
        info!("Loading VAE and decoding...");
        let vae_cfg = flux_ae::Config::dev();
        let vb_vae = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[self.checkpoint_path.clone()],
                DType::F32,
                &dev,
            )?
        };
        let vb_vae = vb_vae.pp(flux1_prefixes::VAE);
        let vae = AutoEncoder::new(&vae_cfg, vb_vae)?;
        let img = img_cpu.to_device(&dev)?;
        let img = vae.decode(&img)?;
        info!("VAE decoded: {:?}", img.shape());

        // ── 9. Convert to RGB ──────────────────────────────────────────────
        let images = ((img / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let images = (images.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?;

        let image_tensor = images.i(0)?;
        let (channel, height, width) = image_tensor.dims3()?;
        if channel != 3 {
            anyhow::bail!("expected 3 channels, got {channel}");
        }
        let image_tensor = image_tensor.permute((1, 2, 0))?.flatten_all()?;
        let pixels = image_tensor.to_vec1::<u8>()?;

        let image: ImageBuffer<Rgb<u8>, Vec<u8>> =
            ImageBuffer::from_raw(width as u32, height as u32, pixels)
                .ok_or_else(|| anyhow!("failed to create image buffer"))?;

        info!("Image generated ({}x{})", width, height);
        callback(vec![image]);

        Ok(())
    }
}

/// Load T5 tokenizer — downloads tokenizer.json from HuggingFace.
fn load_t5_tokenizer(path: &std::path::Path) -> Result<Tokenizer> {
    // Try loading as a tokenizers JSON file first
    if path.extension().map(|e| e == "json").unwrap_or(false) {
        return Tokenizer::from_file(path)
            .map_err(|e| anyhow!("failed to load T5 tokenizer: {e}"));
    }

    // spiece.model is SentencePiece binary format — try sibling tokenizer.json
    let parent = path.parent().unwrap_or(std::path::Path::new("."));
    let json_path = parent.join("tokenizer.json");
    if json_path.exists() {
        return Tokenizer::from_file(&json_path)
            .map_err(|e| anyhow!("failed to load T5 tokenizer.json: {e}"));
    }

    // Download tokenizer.json from HuggingFace
    // google/t5-v1_1-xxl only has spiece.model, use google-t5/t5-11b which has tokenizer.json
    info!("Downloading T5 tokenizer.json...");
    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .to_string_lossy()
        .to_string();
    let mut cache_path = std::path::PathBuf::from(&cache_dir);
    cache_path.push("hub");
    let cache = hf_hub::Cache::new(cache_path);
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache).build()?;

    // Try multiple repos that have the T5 tokenizer.json
    let repos = ["google-t5/t5-11b", "google-t5/t5-small", "google/flan-t5-xxl"];
    for repo_name in repos {
        let repo = api.model(repo_name.to_string());
        if let Ok(tok_path) = repo.get("tokenizer.json") {
            return Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("failed to load T5 tokenizer: {e}"));
        }
    }

    anyhow::bail!("could not find T5 tokenizer.json in any known repository")
}
