//! Metal Shading Language (MSL) kernels for fused ops.
//! Compiled once at first use and cached by the Metal pipeline.

pub(crate) const FUSED_OPS_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ─── stable_softplus: ln(1 + exp(clamp(x, -inf, 88))) with max(x, result) ───
kernel void stable_softplus_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float x = input[idx];
    float clamped = min(x, 88.0f);
    float sp = log(exp(clamped) + 1.0f);
    output[idx] = max(x, sp);
}

kernel void stable_softplus_f16(
    device const half* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float x = float(input[idx]);
    float clamped = min(x, 88.0f);
    float sp = log(exp(clamped) + 1.0f);
    output[idx] = half(max(x, sp));
}

// ─── silu_mul: silu(gate) * up ───────────────────────────────────────────────
kernel void silu_mul_f32(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float g = gate[idx];
    output[idx] = (g / (1.0f + exp(-g))) * up[idx];
}

kernel void silu_mul_f16(
    device const half* gate [[buffer(0)]],
    device const half* up [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint idx [[thread_position_in_grid]]
) {
    if (idx >= count) return;
    float g = float(gate[idx]);
    output[idx] = half((g / (1.0f + exp(-g))) * float(up[idx]));
}

// ─── depthwise_conv1d_silu: dot(window, weight) per channel + silu ───────────
// window: (batch, channels, kernel_size), weight: (channels, kernel_size)
// output: (batch, channels) with silu activation
kernel void depthwise_conv1d_silu_f32(
    device const float* window [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& channels [[buffer(3)]],
    constant uint& kernel_size [[buffer(4)]],
    uint2 idx [[thread_position_in_grid]]  // (channel, batch)
) {
    uint b = idx.y;
    uint c = idx.x;
    if (c >= channels) return;
    float sum = 0.0f;
    uint win_offset = b * channels * kernel_size + c * kernel_size;
    uint w_offset = c * kernel_size;
    for (uint k = 0; k < kernel_size; k++) {
        sum += window[win_offset + k] * weight[w_offset + k];
    }
    // silu activation
    output[b * channels + c] = sum / (1.0f + exp(-sum));
}
"#;
