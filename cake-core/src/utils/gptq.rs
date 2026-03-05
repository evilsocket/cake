//! GPTQ (4-bit post-training quantization) dequantization support.
//!
//! Models quantized with AutoGPTQ store linear-layer weights as three tensors
//! per weight matrix:
//!   - `*.qweight` — int32, shape (in_features // 8, out_features): 8 × 4-bit values
//!                   packed into each int32 along the input dimension.
//!   - `*.scales`  — f16,  shape (groups, out_features): one scale per output neuron
//!                   per group (group_size = in_features / groups).
//!   - `*.qzeros`  — int32, shape (groups, out_features // 8): 8 × 4-bit zero points
//!                   packed into each int32 along the output dimension.
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

use candle_core::{safetensors::MmapedSafetensors, DType, Device, Shape, Tensor};
use candle_nn::{var_builder::SimpleBackend, Init, VarBuilder};

/// Check whether a model uses GPTQ quantization by inspecting its config.json.
pub fn is_gptq_quantized(config_path: &Path) -> bool {
    let Ok(data) = std::fs::read_to_string(config_path) else {
        return false;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&data) else {
        return false;
    };
    // Check top-level and nested text_config for quantization_config
    for root in [&json, json.get("text_config").unwrap_or(&json)] {
        let is_gptq = root
            .get("quantization_config")
            .and_then(|qc| qc.get("quant_method"))
            .and_then(|qm| qm.as_str())
            .map(|s| s == "gptq")
            .unwrap_or(false);
        if is_gptq {
            return true;
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
fn dequantize_gptq_4bit(
    qweight: &Tensor,
    scales: &Tensor,
    qzeros: &Tensor,
    group_size: usize,
) -> candle_core::Result<Tensor> {
    let (packed_rows, out_features) = qweight.dims2()?;
    let in_features = packed_rows * 8;
    let (groups, _) = scales.dims2()?;
    debug_assert_eq!(in_features / group_size, groups);

    // Pull to CPU vecs (these are tiny compared to the dequantized result).
    let qw: Vec<i32> = qweight.to_vec2::<i32>()?.into_iter().flatten().collect();
    let sc: Vec<f32> = scales
        .to_dtype(DType::F32)?
        .to_vec2::<f32>()?
        .into_iter()
        .flatten()
        .collect();
    let qz: Vec<i32> = qzeros.to_vec2::<i32>()?.into_iter().flatten().collect();
    let zero_cols = out_features / 8; // columns in qzeros

    let mut weight = vec![0f32; out_features * in_features];

    // Since group_size is always a multiple of 8 (standard GPTQ: 32/64/128),
    // all 8 input indices packed into one int32 belong to the same group.
    // We iterate over packed rows and output neurons together, then unpack
    // the 8 bits from each int32 in the inner loop.
    use rayon::prelude::*;
    let rows: Vec<Vec<f32>> = (0..packed_rows)
        .into_par_iter()
        .map(|pr| {
            let in_start = pr * 8;
            let g = in_start / group_size;
            let mut row = vec![0f32; out_features * 8];
            for j in 0..out_features {
                let packed_w = qw[pr * out_features + j];
                let scale = sc[g * out_features + j];
                let zero_packed = qz[g * zero_cols + j / 8];
                let zero = ((zero_packed >> ((j % 8) * 4)) & 0xF) + 1; // AutoGPTQ +1 convention
                for bit in 0..8i32 {
                    let w4 = (packed_w >> (bit * 4)) & 0xF;
                    // Output layout: (out_features, in_features) row-major
                    // → element [j, in_start + bit] = j * in_features + in_start + bit
                    row[j * 8 + bit as usize] = (w4 - zero) as f32 * scale;
                }
            }
            row
        })
        .collect();

    // Assemble: rows[pr][j*8 + bit] → weight[j * in_features + pr*8 + bit]
    for (pr, row) in rows.into_iter().enumerate() {
        let in_start = pr * 8;
        for j in 0..out_features {
            for bit in 0..8 {
                weight[j * in_features + in_start + bit] = row[j * 8 + bit];
            }
        }
    }

    Tensor::from_vec(weight, (out_features, in_features), &Device::Cpu)
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
    fn load_tensor(
        &self,
        name: &str,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        // Strip the ".weight" suffix to get the parameter prefix.
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        let qweight_name = format!("{prefix}.qweight");
        let scales_name = format!("{prefix}.scales");
        let qzeros_name = format!("{prefix}.qzeros");

        if self.inner.get(&qweight_name).is_ok() {
            // GPTQ quantized tensor — dequantize on CPU then cast + move to device.
            let qweight = self.inner.load(&qweight_name, &Device::Cpu)?;
            let scales = self.inner.load(&scales_name, &Device::Cpu)?;
            let qzeros = self.inner.load(&qzeros_name, &Device::Cpu)?;
            let weight = dequantize_gptq_4bit(&qweight, &scales, &qzeros, self.group_size)?;
            weight.to_dtype(dtype)?.to_device(dev)
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
        // A tensor is "present" if either its direct name or its GPTQ qweight exists.
        if self.inner.get(name).is_ok() {
            return true;
        }
        let prefix = name.strip_suffix(".weight").unwrap_or(name);
        self.inner.get(&format!("{prefix}.qweight")).is_ok()
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
