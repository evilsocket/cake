use candle_core::{DType, Tensor};

type Result<T> = anyhow::Result<T>;

/// FP32 Layer Norm without learnable parameters.
/// Computes: (x - mean) / sqrt(var + eps)
pub fn fp32_layer_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let in_dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    let x = x.broadcast_sub(&mean)?;
    let var = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
    let out = x.broadcast_div(&(var + eps)?.sqrt()?)?;
    Ok(out.to_dtype(in_dtype)?)
}

/// Apply AdaLN modulation: norm(x) * (1 + scale) + shift.
pub fn modulate(x: &Tensor, shift: &Tensor, scale: &Tensor, eps: f64) -> Result<Tensor> {
    let x_norm = fp32_layer_norm(x, eps)?;
    let ones = Tensor::ones_like(scale)?;
    Ok(x_norm
        .broadcast_mul(&(scale + ones)?)?
        .broadcast_add(shift)?)
}
