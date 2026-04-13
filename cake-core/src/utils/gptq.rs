//! GPTQ (4-bit post-training quantization) dequantization support.
//!
//! Models quantized with AutoGPTQ store linear-layer weights as three tensors
//! per weight matrix:
//!   - `*.qweight` — int32, shape (in_features // 8, out_features): 8 x 4-bit values
//!     packed into each int32 along the input dimension.
//!   - `*.scales`  — f16,  shape (groups, out_features): one scale per output neuron
//!     per group (group_size = in_features / groups).
//!   - `*.qzeros`  — int32, shape (groups, out_features // 8): 8 x 4-bit zero points
//!     packed into each int32 along the output dimension.
//!
//! This module provides a custom VarBuilder backend (`GptqBackend`) that
//! intercepts loads of `*.weight` tensors and transparently dequantizes them
//! on CPU, returning a plain F32→dtype tensor to the caller.
//!
//! Dequantization formula (for bits=4, symmetric or asymmetric):
//!   weight[i, j] = (q4(i,j) - zero4(group(i), j) - 1) * scale(group(i), j)
//!
//! The `-1` correction is the AutoGPTQ convention for zero-point encoding.
//!
//! Not all tensors are quantized. When no corresponding `qweight` exists the
//! backend falls through to a normal load, so non-quantized weights (attention
//! projections, embeddings, norms, etc.) are loaded as-is.

use std::path::Path;

use candle_core::{DType, Device, Shape, Tensor, safetensors::MmapedSafetensors};
use candle_nn::{Init, VarBuilder, var_builder::SimpleBackend};

/// Check whether a model uses 4-bit quantization by inspecting its config.json.
/// Detects both standard GPTQ (`quant_method: "gptq"`) and affine 4-bit
/// (`mode: "affine"`, `bits: 4`) used by some quantized models.
pub fn is_gptq_quantized(config_path: &Path) -> bool {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return false;
    };
    // Check top-level and nested text_config for quantization_config
    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        if let Some(qc) = root.get("quantization_config") {
            // Standard GPTQ: quant_method == "gptq"
            let is_gptq = qc
                .get("quant_method")
                .and_then(|qm| qm.as_str())
                .map(|s| s == "gptq")
                .unwrap_or(false);
            if is_gptq {
                return true;
            }
            // Affine 4-bit: mode == "affine" && bits == 4
            let is_affine_4bit = qc
                .get("mode")
                .and_then(|m| m.as_str())
                .map(|s| s == "affine")
                .unwrap_or(false)
                && qc
                    .get("bits")
                    .and_then(|b| b.as_u64())
                    .map(|b| b == 4)
                    .unwrap_or(false);
            if is_affine_4bit {
                return true;
            }
        }
    }
    false
}

/// Read the GPTQ group_size from config.json (defaults to 128).
pub fn gptq_group_size(config_path: &Path) -> usize {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return 128;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return 128;
    };
    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        if let Some(gs) = root
            .get("quantization_config")
            .and_then(|qc| qc.get("group_size"))
            .and_then(|v| v.as_u64())
        {
            return gs as usize;
        }
    }
    128
}

/// Dequantize a GPTQ 4-bit quantized weight matrix to F32.
///
/// Inputs (all on CPU):
/// - `qweight`: int32, shape `(packed_rows, out_features)` where `packed_rows = in_features / 8`
/// - `scales`:  f16,  shape `(groups, out_features)` where `groups = in_features / group_size`
/// - `qzeros`:  int32, shape `(groups, out_features / 8)`
///
/// Output: F32 tensor of shape `(out_features, in_features)`.
pub fn dequantize_gptq_4bit(
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    group_size: usize,
) -> candle_core::Result<Tensor> {
    let (packed_rows, out_features) = qweight.dims2()?;
    let in_features = packed_rows * 8;
    let (groups, _) = scales.dims2()?;
    debug_assert_eq!(in_features / group_size, groups);

    // Pull to CPU vecs — use flatten + to_vec1 to avoid intermediate Vec<Vec<T>>.
    let qw: Vec<i32> = qweight.flatten_all()?.to_vec1()?;
    let sc: Vec<f32> = scales.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    let qz: Vec<i32> = qzeros.flatten_all()?.to_vec1()?;
    let zero_cols = out_features / 8; // columns in qzeros

    let mut weight = vec![0f32; out_features * in_features];

    // Parallelize over output rows (j), writing directly to the output buffer.
    // Each output row is independent: row j reads qw[pr * out_features + j] for
    // all packed_rows, extracts nibbles, and writes to weight[j * in_features ..].
    // This eliminates the intermediate Vec<Vec<f32>> and sequential scatter pass.
    use rayon::prelude::*;
    weight
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(j, row)| {
            for pr in 0..packed_rows {
                let in_start = pr * 8;
                let g = in_start / group_size;
                let packed_w = qw[pr * out_features + j];
                let scale = sc[g * out_features + j];
                let zero_packed = qz[g * zero_cols + j / 8];
                let zero = ((zero_packed >> ((j % 8) * 4)) & 0xF) + 1; // AutoGPTQ +1 convention
                for bit in 0..8i32 {
                    let w4 = (packed_w >> (bit * 4)) & 0xF;
                    row[in_start + bit as usize] = (w4 - zero) as f32 * scale;
                }
            }
        });

    Tensor::from_vec(weight, (out_features, in_features), &Device::Cpu)
}

/// Dequantize a packed 4-bit weight tensor with per-group scales and biases.
///
/// Used for embeddings in some GPTQ models where the format is:
/// - `weight`: uint32, shape `(rows, packed_cols)` where `packed_cols = cols / 8`
/// - `scales`: f16/bf16, shape `(rows, groups)`
/// - `biases`: f16/bf16, shape `(rows, groups)`
///
/// Formula: `w_dequant[i, j] = w4(i, j) * scale(i, group(j)) + bias(i, group(j))`
///
/// Output: F32 tensor of shape `(rows, cols)`.
pub fn dequantize_packed_4bit(
    packed: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> candle_core::Result<Tensor> {
    // Handle 3D stacked tensors (e.g., [num_experts, rows, packed_cols])
    if packed.rank() == 3 {
        let n = packed.dim(0)?;
        let slices: Vec<Tensor> = (0..n)
            .map(|i| {
                let p = packed.get(i)?;
                let s = scales.get(i)?;
                let b = biases.get(i)?;
                dequantize_packed_4bit(&p, &s, &b, group_size)
            })
            .collect::<candle_core::Result<_>>()?;
        return Tensor::stack(&slices, 0);
    }
    let (rows, packed_cols) = packed.dims2()?;
    let cols = packed_cols * 8;
    let (_, groups) = scales.dims2()?;

    // Extract raw data — avoid Tensor intermediates for the hot path
    let pw: Vec<u32> = packed.flatten_all()?.to_vec1::<u32>()?;
    let sc: Vec<f32> = scales
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let bi: Vec<f32> = biases
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    use rayon::prelude::*;
    let mut weight = vec![0f32; rows * cols];
    weight
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, row)| {
            for pc in 0..packed_cols {
                let packed_val = pw[i * packed_cols + pc];
                for bit in 0..8u32 {
                    let j = pc * 8 + bit as usize;
                    let w4 = ((packed_val >> (bit * 4)) & 0xF) as f32;
                    let g = j / group_size;
                    let scale = sc[i * groups + g];
                    let bias = bi[i * groups + g];
                    row[j] = w4 * scale + bias;
                }
            }
        });

    Tensor::from_vec(weight, (rows, cols), &Device::Cpu)
}

/// Dequantize directly to F16, skipping the F32 intermediate.
///
/// This saves 50% peak memory vs `dequantize_packed_4bit` + `to_dtype(F16)`:
/// - Old path: 259 MiB F32 + 129.5 MiB F16 = 388.5 MiB peak per tensor
/// - New path: 129.5 MiB F16 only = 129.5 MiB peak per tensor
///
/// The arithmetic (w4 * scale + bias) is done in F32 then truncated to F16
/// per element, which matches the precision of the old path.
pub fn dequantize_packed_4bit_f16(
    packed: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: usize,
) -> candle_core::Result<Tensor> {
    if packed.rank() == 3 {
        let n = packed.dim(0)?;
        let slices: Vec<Tensor> = (0..n)
            .map(|i| {
                let p = packed.get(i)?;
                let s = scales.get(i)?;
                let b = biases.get(i)?;
                dequantize_packed_4bit_f16(&p, &s, &b, group_size)
            })
            .collect::<candle_core::Result<_>>()?;
        return Tensor::stack(&slices, 0);
    }
    let (rows, packed_cols) = packed.dims2()?;
    let cols = packed_cols * 8;
    let (_, groups) = scales.dims2()?;

    let pw: Vec<u32> = packed.flatten_all()?.to_vec1::<u32>()?;
    let sc: Vec<f32> = scales
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let bi: Vec<f32> = biases
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    use half::f16;
    use rayon::prelude::*;
    let mut weight = vec![f16::ZERO; rows * cols];
    weight
        .par_chunks_mut(cols)
        .enumerate()
        .for_each(|(i, row)| {
            for pc in 0..packed_cols {
                let packed_val = pw[i * packed_cols + pc];
                for bit in 0..8u32 {
                    let j = pc * 8 + bit as usize;
                    let w4 = ((packed_val >> (bit * 4)) & 0xF) as f32;
                    let g = j / group_size;
                    let scale = sc[i * groups + g];
                    let bias = bi[i * groups + g];
                    row[j] = f16::from_f32(w4 * scale + bias);
                }
            }
        });

    Tensor::from_vec(weight, (rows, cols), &Device::Cpu)
}

/// Custom VarBuilder backend that transparently dequantizes GPTQ weights.
///
/// When asked for `foo.weight`, checks if `foo.qweight` exists and, if so,
/// loads `foo.{qweight,scales,qzeros}` and returns the dequantized tensor.
/// Falls through to a normal load for non-quantized tensors.
struct GptqBackend {
    inner: MmapedSafetensors,
    group_size: usize,
}

impl GptqBackend {
    fn load_tensor(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        // Strip the ".weight" suffix to get the parameter prefix.
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        let qweight_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        if self.inner.get(&qweight_name).is_ok() {
            // Standard GPTQ: qweight + scales + qzeros
            let qweight = self.inner.load(&qweight_name, &Device::Cpu)?;
            let scales = self.inner.load(&scales_name, &Device::Cpu)?;
            let qzeros = self.inner.load(&qzeros_name, &Device::Cpu)?;
            let weight = dequantize_gptq_4bit(&qweight, &scales, &qzeros, self.group_size)?;
            weight.to_dtype(dtype)?.to_device(dev)
        } else if self.inner.get(&scales_name).is_ok() {
            // Affine 4-bit quantization: packed uint32 weight + scales + biases
            // Formula: w4 * scale + bias (no zero-point)
            let biases_name = format!("{prefix}.biases");
            let packed = self.inner.load(name, &Device::Cpu)?;
            let scales = self.inner.load(&scales_name, &Device::Cpu)?;
            let biases = self.inner.load(&biases_name, &Device::Cpu)?;
            // Use F16-native dequant when target is F16 to halve peak memory
            // (skips the 259 MiB F32 intermediate for large tensors).
            let weight = if dtype == DType::F16 {
                dequantize_packed_4bit_f16(&packed, &scales, &biases, self.group_size)?
            } else {
                let w = dequantize_packed_4bit(&packed, &scales, &biases, self.group_size)?;
                w.to_dtype(dtype)?
            };
            weight.to_device(dev)
        } else {
            // Non-quantized tensor — load directly.
            self.inner.load(name, dev)?.to_dtype(dtype)
        }
    }
}

impl SimpleBackend for GptqBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let tensor = self.load_tensor(name, dtype, dev)?;
        if tensor.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: tensor.shape().clone(),
            }
            .bt())?
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_tensor(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        // A tensor is "present" if either its direct name, GPTQ qweight, or scales exist.
        if self.inner.get(name).is_ok() {
            return true;
        }
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        self.inner.get(&format!("{prefix}.qweight")).is_ok()
            || self.inner.get(&format!("{prefix}.scales")).is_ok()
    }
}

/// Create a VarBuilder that transparently dequantizes GPTQ 4-bit weights.
///
/// # Safety
///
/// Inherits the mmap safety requirements from `MmapedSafetensors`.
pub unsafe fn load_gptq_var_builder<'a>(
    filenames: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
    group_size: usize,
) -> anyhow::Result<VarBuilder<'a>> {
    let inner = MmapedSafetensors::multi(filenames)?;

    let qweight_count = inner
        .tensors()
        .iter()
        .filter(|(name, _)| name.ends_with(".qweight"))
        .count();
    log::info!(
        "GPTQ model detected: {} weight matrices will be dequantized at load time (group_size={})",
        qweight_count,
        group_size,
    );

    let backend: Box<dyn SimpleBackend> = Box::new(GptqBackend { inner, group_size });
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

/// Create a VarBuilder for MLX-style packed 4-bit quantized models.
///
/// Reuses the `GptqBackend` since MLX and GPTQ-affine share the same
/// dequantization path (packed uint32 + scales + biases).
///
/// # Safety
///
/// Inherits the mmap safety requirements from `MmapedSafetensors`.
pub unsafe fn load_mlx_var_builder<'a>(
    filenames: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
    group_size: usize,
) -> anyhow::Result<VarBuilder<'a>> {
    let inner = MmapedSafetensors::multi(filenames)?;

    let scales_count = inner
        .tensors()
        .iter()
        .filter(|(name, _)| name.ends_with(".scales"))
        .count();
    log::info!(
        "MLX 4-bit model: {} quantized tensors will be dequantized at load time (group_size={})",
        scales_count,
        group_size,
    );

    let backend: Box<dyn SimpleBackend> = Box::new(GptqBackend { inner, group_size });
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

/// VarBuilder backend for Metal-native MLX 4-bit quantized models.
///
/// Unlike `GptqBackend`, this does NOT dequantize packed weights. Instead:
/// - `{prefix}.weight` → loaded as raw U32 packed tensor (no dequant)
/// - `{prefix}.scales` → loaded as F16
/// - `{prefix}.biases` → loaded as F16
/// - Non-quantized tensors → loaded normally with dtype conversion
///
/// This keeps weights at 0.5 bytes/element on Metal, enabling the fused
/// `q4_matmul_f16` kernel to dequantize on-the-fly during matmul.
struct MetalMlxBackend {
    inner: MmapedSafetensors,
    group_size: usize,
}

impl MetalMlxBackend {
    /// Dequantize a quantized tensor (fallback for non-linear layers like embeddings).
    fn load_dequantized(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        let scales_name = format!("{prefix}.scales");
        let biases_name = format!("{prefix}.biases");

        if name.ends_with(".weight") && self.inner.get(&scales_name).is_ok() {
            let packed = self.inner.load(name, &Device::Cpu)?;
            let scales = self.inner.load(&scales_name, &Device::Cpu)?;
            let biases = self.inner.load(&biases_name, &Device::Cpu)?;
            let weight = if dtype == DType::F16 {
                dequantize_packed_4bit_f16(&packed, &scales, &biases, self.group_size)?
            } else {
                let w = dequantize_packed_4bit(&packed, &scales, &biases, self.group_size)?;
                w.to_dtype(dtype)?
            };
            weight.to_device(dev)
        } else {
            self.inner.load(name, dev)?.to_dtype(dtype)
        }
    }

    /// Load a tensor in its raw format (packed U32 for quantized, native for others).
    /// Used by get_unchecked() for the fused q4 kernel path.
    fn load_tensor(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        let scales_name = format!("{prefix}.scales");

        // Check if this is a quantized tensor (has matching .scales)
        if name.ends_with(".weight") && self.inner.get(&scales_name).is_ok() {
            // Return packed U32 weight directly — no dequantization.
            self.inner.load(name, dev)
        } else if name.ends_with(".scales") || name.ends_with(".biases") {
            // Scales and biases: load as F16 (their native format).
            let t = self.inner.load(name, dev)?;
            t.to_dtype(DType::F16)
        } else {
            // Non-quantized tensor — standard load with dtype conversion.
            self.inner.load(name, dev)?.to_dtype(dtype)
        }
    }
}

impl SimpleBackend for MetalMlxBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Shape-checked path: try raw load first (works for non-quantized tensors
        // and quantized linear layers where caller expects packed shape).
        let tensor = self.load_tensor(name, dtype, dev)?;
        if tensor.shape() == &s {
            return Ok(tensor);
        }
        // Shape mismatch — this is a non-linear layer (embedding, norm) that has
        // quantized weights but needs full F16. Fall back to dequantization.
        let dequantized = self.load_dequantized(name, dtype, dev)?;
        if dequantized.shape() != &s {
            Err(candle_core::Error::UnexpectedShape {
                msg: format!("shape mismatch for {name}"),
                expected: s,
                got: dequantized.shape().clone(),
            }
            .bt())?
        }
        Ok(dequantized)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load_tensor(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        if self.inner.get(name).is_ok() {
            return true;
        }
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        self.inner.get(&format!("{prefix}.scales")).is_ok()
    }
}

/// Create a VarBuilder for Metal-native MLX 4-bit quantized models.
///
/// Unlike `load_mlx_var_builder`, this does NOT dequantize packed weights.
/// Quantized tensors are returned raw (U32 packed, F16 scales/biases) so
/// the fused `q4_matmul_f16` kernel can operate directly on packed data.
///
/// # Safety
///
/// Inherits the mmap safety requirements from `MmapedSafetensors`.
pub unsafe fn load_metal_mlx_var_builder<'a>(
    filenames: &[std::path::PathBuf],
    dtype: DType,
    device: &Device,
    group_size: usize,
) -> anyhow::Result<VarBuilder<'a>> {
    let inner = MmapedSafetensors::multi(filenames)?;

    let scales_count = inner
        .tensors()
        .iter()
        .filter(|(name, _)| name.ends_with(".scales"))
        .count();
    log::info!(
        "MLX 4-bit model on Metal: {} quantized tensors will use fused q4 kernel (group_size={}, 4x memory reduction)",
        scales_count,
        group_size,
    );

    let backend: Box<dyn SimpleBackend> = Box::new(MetalMlxBackend { inner, group_size });
    Ok(VarBuilder::from_backend(backend, dtype, device.clone()))
}

/// Returns the group_size stored in this backend (used to pass quantization
/// info to model loading code).
pub fn metal_mlx_group_size() -> Option<usize> {
    // This is a compile-time marker; actual group_size comes from config.json
    // and is passed through MlxQuantization.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dequantize_gptq_4bit_shape() {
        // 2 packed_rows × 4 out_features, group_size=8 → 1 group
        // in_features = 2*8 = 16, out_features = 4
        let _qweight_unused = Tensor::zeros((2, 4), DType::I64, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::I64)
            .unwrap();
        // Actually qweight is i32
        let qweight = Tensor::from_vec(vec![0i32; 8], (2, 4), &Device::Cpu).unwrap();
        let scales = Tensor::from_vec(vec![1.0f32; 4], (1, 4), &Device::Cpu).unwrap();
        let qzeros = Tensor::from_vec(vec![0i32; 1], (1, 1), &Device::Cpu).unwrap();
        // group_size=16, 1 group, packed_rows=2, out=4

        let result = dequantize_gptq_4bit(&qweight, &scales, &qzeros, 16).unwrap();
        assert_eq!(result.dims(), &[4, 16]); // (out_features, in_features)
    }

    #[test]
    fn test_dequantize_gptq_4bit_known_values() {
        // 1 packed_row × 1 out_feature, group_size=8
        // qweight[0,0] = 0x21 → bits: [1, 2, 0, 0, 0, 0, 0, 0] (4-bit nibbles)
        // scale = 2.0, qzeros = 0 → zero = (0 & 0xF) + 1 = 1
        // w[bit] = (nibble - 1) * 2.0
        let qweight = Tensor::from_vec(vec![0x21i32], (1, 1), &Device::Cpu).unwrap();
        let scales = Tensor::from_vec(vec![2.0f32], (1, 1), &Device::Cpu).unwrap();
        let qzeros = Tensor::from_vec(vec![0i32], (1, 1), &Device::Cpu).unwrap();

        let result = dequantize_gptq_4bit(&qweight, &scales, &qzeros, 8).unwrap();
        let vals: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        // bit0: nibble=(0x21>>0)&0xF = 1, w=(1-1)*2.0 = 0.0
        // bit1: nibble=(0x21>>4)&0xF = 2, w=(2-1)*2.0 = 2.0
        assert!((vals[0] - 0.0).abs() < 1e-6, "bit0: got {}", vals[0]);
        assert!((vals[1] - 2.0).abs() < 1e-6, "bit1: got {}", vals[1]);
    }

    #[test]
    fn test_is_gptq_quantized() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128}}"#,
        )
        .unwrap();
        assert!(is_gptq_quantized(&config_path));

        let config_path2 = dir.path().join("config2.json");
        std::fs::write(&config_path2, r#"{"hidden_size": 4096}"#).unwrap();
        assert!(!is_gptq_quantized(&config_path2));
    }

    #[test]
    fn test_gptq_group_size() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            r#"{"quantization_config": {"quant_method": "gptq", "group_size": 64}}"#,
        )
        .unwrap();
        assert_eq!(gptq_group_size(&config_path), 64);
    }
}
