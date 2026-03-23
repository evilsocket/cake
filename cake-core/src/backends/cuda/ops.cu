// Fused CUDA kernels for cake inference optimization.
//
// Each kernel fuses 2-3 element-wise operations into a single kernel launch,
// saving ~4-7µs per launch on modern GPUs.

#include <stdint.h>

// ─── helpers ────────────────────────────────────────────────────────
template<typename T> __device__ __forceinline__ T expg(T a);
template<> __device__ __forceinline__ float  expg<float>(float a)   { return expf(a); }
template<> __device__ __forceinline__ double expg<double>(double a) { return exp(a); }

#if __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
template<> __device__ __forceinline__ __half expg<__half>(__half a) {
    return hexp(a);
}
#endif

#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
template<> __device__ __forceinline__ __nv_bfloat16 expg<__nv_bfloat16>(__nv_bfloat16 a) {
    return hexp(a);
}
#endif

// ─── f8e4m3_to_f32: software FP8 dequantization for SM < 8.9 ────────
// On SM89+ (Ada/Hopper), candle uses native __nv_fp8_e4m3 hardware.
// On SM80 (A100) and below, the native FP8 type doesn't exist.
// F8E4M3 format: 1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
extern "C" __global__ void f8e4m3_to_f32(
    const size_t numel,
    const uint8_t *inp,
    float *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += blockDim.x * gridDim.x) {
        uint8_t bits = inp[i];
        uint32_t sign = (bits >> 7) & 1;
        uint32_t exp  = (bits >> 3) & 0xF;
        uint32_t mant = bits & 0x7;

        float result;
        if (exp == 0 && mant == 0) {
            result = 0.0f;
        } else if (exp == 0) {
            // Subnormal: 2^(-6) * (mant / 8)
            result = ldexpf((float)mant / 8.0f, -6);
        } else if (exp == 0xF && mant == 0x7) {
            result = __int_as_float(0x7FC00000); // NaN
        } else {
            // Normal: 2^(exp-7) * (1 + mant/8)
            result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
        }
        if (sign) result = -result;
        out[i] = result;
    }
}

// ─── f8e4m3_to_f16: software FP8→FP16 dequantization ────────────────
// Same algorithm as f8e4m3_to_f32 but outputs __half for faster matmul.
// A100 has 312 TFLOPS F16 vs 156 TFLOPS F32.
#if __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
extern "C" __global__ void f8e4m3_to_f16(
    const size_t numel,
    const uint8_t *inp,
    __half *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += blockDim.x * gridDim.x) {
        uint8_t bits = inp[i];
        uint32_t sign = (bits >> 7) & 1;
        uint32_t exp  = (bits >> 3) & 0xF;
        uint32_t mant = bits & 0x7;

        float result;
        if (exp == 0 && mant == 0) {
            result = 0.0f;
        } else if (exp == 0) {
            result = ldexpf((float)mant / 8.0f, -6);
        } else if (exp == 0xF && mant == 0x7) {
            result = __int_as_float(0x7FC00000);
        } else {
            result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
        }
        if (sign) result = -result;
        out[i] = __float2half(result);
    }
}
#endif

// ─── f8e4m3_to_bf16: software FP8→BF16 dequantization ───────────────
// Direct F8→BF16 avoids the F16 intermediate when using BF16 compute.
#if __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
extern "C" __global__ void f8e4m3_to_bf16(
    const size_t numel,
    const uint8_t *inp,
    __nv_bfloat16 *out
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < numel; i += blockDim.x * gridDim.x) {
        uint8_t bits = inp[i];
        uint32_t sign = (bits >> 7) & 1;
        uint32_t exp  = (bits >> 3) & 0xF;
        uint32_t mant = bits & 0x7;

        float result;
        if (exp == 0 && mant == 0) {
            result = 0.0f;
        } else if (exp == 0) {
            result = ldexpf((float)mant / 8.0f, -6);
        } else if (exp == 0xF && mant == 0x7) {
            result = __int_as_float(0x7FC00000);
        } else {
            result = ldexpf(1.0f + (float)mant / 8.0f, (int)exp - 7);
        }
        if (sign) result = -result;
        out[i] = __float2bfloat16(result);
    }
}
#endif

// ─── silu_mul: silu(x) * y = x * sigmoid(x) * y ────────────────────
// Fuses 2 kernels (silu + mul) into 1.
// Used in every MLP (gate activation * up projection).
// F32 path uses fast reciprocal to avoid slow fdiv.
template<typename T>
__device__ __forceinline__ T silu_mul_fwd(T x, T y) {
    T one = static_cast<T>(1);
    return x / (one + expg(-x)) * y;
}

// Fast f32 specialization: use __expf + __frcp_rn for fast sigmoid
template<>
__device__ __forceinline__ float silu_mul_fwd<float>(float x, float y) {
    float sig = __frcp_rn(1.0f + __expf(-x));
    return x * sig * y;
}

#define SILU_MUL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *x, \
    const TYPENAME *y, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        out[i] = silu_mul_fwd(x[i], y[i]); \
    } \
}

SILU_MUL_OP(float, silu_mul_f32)
SILU_MUL_OP(double, silu_mul_f64)
#if __CUDA_ARCH__ >= 530
SILU_MUL_OP(__half, silu_mul_f16)
#endif
#if __CUDA_ARCH__ >= 800
SILU_MUL_OP(__nv_bfloat16, silu_mul_bf16)
#endif

// ─── stable_softplus: ln(1 + exp(clamp(x, -inf, 88))) clamped to max(x, result) ──
// Fuses 5 kernels (minimum + exp + add + log + maximum) into 1.
// Used in GatedDeltaNet gate computation.
template<typename T>
__device__ __forceinline__ T stable_softplus_fwd(T x) {
    // Clamp to avoid exp overflow
    T clamped = (x < static_cast<T>(88)) ? x : static_cast<T>(88);
    T sp = log(expg(clamped) + static_cast<T>(1));
    // For large x, softplus(x) ≈ x
    return (x > sp) ? x : sp;
}

// Float specialization: fast-path branches + __expf + log1pf
template<>
__device__ __forceinline__ float stable_softplus_fwd<float>(float x) {
    if (x > 20.0f) return x;           // softplus(x) ≈ x for large x
    if (x < -10.0f) return __expf(x);   // softplus(x) ≈ exp(x) ≈ 0 for very negative x
    return log1pf(__expf(x));
}

// Specialise for half types using float intermediates
#if __CUDA_ARCH__ >= 530
template<>
__device__ __forceinline__ __half stable_softplus_fwd<__half>(__half x) {
    float fx = __half2float(x);
    if (fx > 20.0f) return x;
    if (fx < -10.0f) return __float2half(__expf(fx));
    return __float2half(log1pf(__expf(fx)));
}
#endif
#if __CUDA_ARCH__ >= 800
template<>
__device__ __forceinline__ __nv_bfloat16 stable_softplus_fwd<__nv_bfloat16>(__nv_bfloat16 x) {
    float fx = __bfloat162float(x);
    if (fx > 20.0f) return x;
    if (fx < -10.0f) return __float2bfloat16(__expf(fx));
    return __float2bfloat16(log1pf(__expf(fx)));
}
#endif

#define SOFTPLUS_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *x, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        out[i] = stable_softplus_fwd(x[i]); \
    } \
}

SOFTPLUS_OP(float, stable_softplus_f32)
SOFTPLUS_OP(double, stable_softplus_f64)
#if __CUDA_ARCH__ >= 530
SOFTPLUS_OP(__half, stable_softplus_f16)
#endif
#if __CUDA_ARCH__ >= 800
SOFTPLUS_OP(__nv_bfloat16, stable_softplus_bf16)
#endif

// ─── rms_norm_gated: rms_norm(x, weight) * silu(z) ─────────────────
// Fuses 3 kernels (rms_norm + silu + mul) into 1.
// Used in GatedDeltaNet output normalization.
// Each row of x is normalized, then multiplied by silu(corresponding row of z).
// weight is broadcast across rows (same for all rows).
// x, z: (n_rows, n_cols), weight: (n_cols,)
#define RMS_NORM_GATED_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *x, \
    const TYPENAME *z, \
    const TYPENAME *weight, \
    TYPENAME *out, \
    const int n_cols, \
    const int block_size, \
    const float eps \
) { \
    const int row = blockIdx.x; \
    const int offset = row * n_cols; \
    /* Compute sum of squares for this row */ \
    float sum2 = 0.0f; \
    for (int col = threadIdx.x; col < n_cols; col += block_size) { \
        float v = static_cast<float>(x[offset + col]); \
        sum2 += v * v; \
    } \
    /* Warp-level reduction */ \
    for (int mask = 16; mask > 0; mask >>= 1) { \
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
    } \
    /* Block-level reduction via shared memory */ \
    __shared__ float shared[32]; \
    int warp_id = threadIdx.x / 32; \
    int lane_id = threadIdx.x % 32; \
    if (lane_id == 0) shared[warp_id] = sum2; \
    __syncthreads(); \
    sum2 = (threadIdx.x < (block_size + 31) / 32) ? shared[lane_id] : 0.0f; \
    for (int mask = 16; mask > 0; mask >>= 1) { \
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
    } \
    float inv_rms = rsqrtf(sum2 / (float)n_cols + eps); \
    /* Apply: weight * rms_norm(x) * silu(z) */ \
    for (int col = threadIdx.x; col < n_cols; col += block_size) { \
        float xv = static_cast<float>(x[offset + col]); \
        float wv_scaled = static_cast<float>(weight[col]) * inv_rms; \
        float zv = static_cast<float>(z[offset + col]); \
        float silu_z = zv * __frcp_rn(1.0f + __expf(-zv)); \
        out[offset + col] = static_cast<TYPENAME>(xv * wv_scaled * silu_z); \
    } \
}

RMS_NORM_GATED_OP(float, rms_norm_gated_f32)
RMS_NORM_GATED_OP(double, rms_norm_gated_f64)
#if __CUDA_ARCH__ >= 530
RMS_NORM_GATED_OP(__half, rms_norm_gated_f16)
#endif
#if __CUDA_ARCH__ >= 800
RMS_NORM_GATED_OP(__nv_bfloat16, rms_norm_gated_bf16)
#endif

// ─── add_rms_norm: rms_norm(a + b, weight, eps) with residual output ──
// Fuses 2 kernels (add + rms_norm) into 1.
// Used in transformer blocks: residual add followed by layer norm.
// Output layout: [all residual rows, all normed rows] = (2*n_rows, n_cols).
// narrow(0, 0, n_rows) / narrow(0, n_rows, n_rows) yields contiguous views.
// a, b: (n_rows, n_cols), weight: (n_cols,)
#define ADD_RMS_NORM_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *a, \
    const TYPENAME *b, \
    const TYPENAME *weight, \
    TYPENAME *out, \
    const int n_cols, \
    const int block_size, \
    const float eps, \
    const int n_rows \
) { \
    const int row = blockIdx.x; \
    const int in_off = row * n_cols; \
    const int res_off = row * n_cols; \
    const int norm_off = n_rows * n_cols + row * n_cols; \
    /* Compute sum of squares in first pass (don't write yet) */ \
    float sum2 = 0.0f; \
    for (int col = threadIdx.x; col < n_cols; col += block_size) { \
        float av = static_cast<float>(a[in_off + col]); \
        float bv = static_cast<float>(b[in_off + col]); \
        float s = av + bv; \
        sum2 += s * s; \
    } \
    /* Warp reduction */ \
    for (int mask = 16; mask > 0; mask >>= 1) { \
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
    } \
    /* Block reduction via shared memory */ \
    __shared__ float shared[32]; \
    int warp_id = threadIdx.x / 32; \
    int lane_id = threadIdx.x % 32; \
    if (lane_id == 0) shared[warp_id] = sum2; \
    __syncthreads(); \
    sum2 = (threadIdx.x < (block_size + 31) / 32) ? shared[lane_id] : 0.0f; \
    for (int mask = 16; mask > 0; mask >>= 1) { \
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
    } \
    float inv_rms = rsqrtf(sum2 / (float)n_cols + eps); \
    /* Re-read a+b (L2-cached), residual rows first then normed rows */ \
    for (int col = threadIdx.x; col < n_cols; col += block_size) { \
        float s = static_cast<float>(a[in_off + col]) + static_cast<float>(b[in_off + col]); \
        out[res_off + col] = static_cast<TYPENAME>(s); \
        float wv_scaled = static_cast<float>(weight[col]) * inv_rms; \
        out[norm_off + col] = static_cast<TYPENAME>(s * wv_scaled); \
    } \
}

ADD_RMS_NORM_OP(float, add_rms_norm_f32)
ADD_RMS_NORM_OP(double, add_rms_norm_f64)
#if __CUDA_ARCH__ >= 530
ADD_RMS_NORM_OP(__half, add_rms_norm_f16)
#endif
#if __CUDA_ARCH__ >= 800
ADD_RMS_NORM_OP(__nv_bfloat16, add_rms_norm_bf16)
#endif

// ─── add3: out = a + b + c ───────────────────────────────────────────
// Fuses 2 kernels (add + add) into 1.
// Used in transformer blocks: residual + attn_out + mlp_out.
#define ADD3_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *a, \
    const TYPENAME *b, \
    const TYPENAME *c, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        out[i] = a[i] + b[i] + c[i]; \
    } \
}

ADD3_OP(float, add3_f32)
ADD3_OP(double, add3_f64)
#if __CUDA_ARCH__ >= 530
ADD3_OP(__half, add3_f16)
#endif
#if __CUDA_ARCH__ >= 800
ADD3_OP(__nv_bfloat16, add3_bf16)
#endif

// ─── exp_mul: out = x * exp(y) ──────────────────────────────────────
// Fuses 2 kernels (exp + broadcast_mul) into 1.
// Used in GatedDeltaNet recurrent decay: state = state * exp(gate).
// y is broadcast to match x (e.g. y has fewer dims).
#define EXP_MUL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *x, \
    const TYPENAME *y, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        out[i] = x[i] * expg(y[i]); \
    } \
}

EXP_MUL_OP(float, exp_mul_f32)
EXP_MUL_OP(double, exp_mul_f64)
#if __CUDA_ARCH__ >= 530
EXP_MUL_OP(__half, exp_mul_f16)
#endif
#if __CUDA_ARCH__ >= 800
EXP_MUL_OP(__nv_bfloat16, exp_mul_bf16)
#endif

// ─── sub_mul: out = (a - b) * c ─────────────────────────────────────
// Fuses 2 kernels (sub + broadcast_mul) into 1.
// Used in GatedDeltaNet delta rule: delta = beta * (v - retrieved).
// c is broadcast to match a,b (e.g. c has fewer dims).
#define SUB_MUL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *a, \
    const TYPENAME *b, \
    const TYPENAME *c, \
    TYPENAME *out \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        out[i] = (a[i] - b[i]) * c[i]; \
    } \
}

// ─── depthwise_conv1d_silu: dot(window, weight) per channel + silu ──
// Fuses 3 kernels (broadcast_mul + sum + silu) into 1.
// Used in GatedDeltaNet causal conv1d step.
// window: (batch, channels, kernel_size), weight: (channels, kernel_size)
// out: (batch, channels)
#define CONV1D_SILU_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *window, \
    const TYPENAME *weight, \
    TYPENAME *out, \
    const int kernel_size, \
    const int channels \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        int chan = i % channels; \
        int batch_idx = i / channels; \
        float acc = 0.0f; \
        int w_off = batch_idx * channels * kernel_size + chan * kernel_size; \
        int wt_off = chan * kernel_size; \
        _Pragma("unroll 8") \
        for (int k = 0; k < kernel_size; k++) { \
            acc = fmaf(static_cast<float>(window[w_off + k]), \
                       static_cast<float>(weight[wt_off + k]), acc); \
        } \
        /* silu(acc) = acc * sigmoid(acc) */ \
        float sig = __frcp_rn(1.0f + __expf(-acc)); \
        out[i] = static_cast<TYPENAME>(acc * sig); \
    } \
}

CONV1D_SILU_OP(float, depthwise_conv1d_silu_f32)
CONV1D_SILU_OP(double, depthwise_conv1d_silu_f64)
#if __CUDA_ARCH__ >= 530
CONV1D_SILU_OP(__half, depthwise_conv1d_silu_f16)
#endif
#if __CUDA_ARCH__ >= 800
CONV1D_SILU_OP(__nv_bfloat16, depthwise_conv1d_silu_bf16)
#endif

// ─── depthwise_conv1d_bias: full depthwise conv1d + bias ────────────
// Replaces 14 kernel launches (7 broadcast_mul + 6 add + 1 bias_add) with 1.
// Used in VAE encoder/decoder blocks.
// input: (batch, channels, input_len)  — already padded
// weight: (channels, kernel_size)
// bias: (channels,)
// out: (batch, channels, out_len)  where out_len = input_len - kernel_size + 1
#define CONV1D_BIAS_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *input, \
    const TYPENAME *weight, \
    const TYPENAME *bias, \
    TYPENAME *out, \
    const int kernel_size, \
    const int channels, \
    const int input_len \
) { \
    const int out_len = input_len - kernel_size + 1; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        int t = i % out_len; \
        int temp = i / out_len; \
        int c = temp % channels; \
        int b = temp / channels; \
        float acc = 0.0f; \
        int in_off = (b * channels + c) * input_len + t; \
        int wt_off = c * kernel_size; \
        _Pragma("unroll 8") \
        for (int k = 0; k < kernel_size; k++) { \
            acc = fmaf(static_cast<float>(input[in_off + k]), \
                       static_cast<float>(weight[wt_off + k]), acc); \
        } \
        acc += static_cast<float>(bias[c]); \
        out[i] = static_cast<TYPENAME>(acc); \
    } \
}

CONV1D_BIAS_OP(float, depthwise_conv1d_bias_f32)
CONV1D_BIAS_OP(double, depthwise_conv1d_bias_f64)
#if __CUDA_ARCH__ >= 530
CONV1D_BIAS_OP(__half, depthwise_conv1d_bias_f16)
#endif
#if __CUDA_ARCH__ >= 800
CONV1D_BIAS_OP(__nv_bfloat16, depthwise_conv1d_bias_bf16)
#endif

// ─── depthwise_conv1d_bias_ctx: conv with separate context + input ──
// Replaces Tensor::zeros + Tensor::cat + depthwise_conv1d_bias (3 kernels) with 1.
// Reads from virtual [ctx, input] concatenation without allocating the merged tensor.
// ctx: (batch, channels, kernel_size-1) — cached context from previous frame
// input: (batch, channels, time_len) — new data
// weight: (channels, kernel_size), bias: (channels,)
// out: (batch, channels, time_len) — same length as input
#define CONV1D_BIAS_CTX_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *ctx, \
    const TYPENAME *input, \
    const TYPENAME *weight, \
    const TYPENAME *bias, \
    TYPENAME *out, \
    const int kernel_size, \
    const int channels, \
    const int ctx_len, \
    const int time_len \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        int t = i % time_len; \
        int temp = i / time_len; \
        int c = temp % channels; \
        int b = temp / channels; \
        float acc = 0.0f; \
        int wt_off = c * kernel_size; \
        /* Virtual position in [ctx, input]: t + k, where ctx occupies [0, ctx_len) */ \
        _Pragma("unroll 8") \
        for (int k = 0; k < kernel_size; k++) { \
            int pos = t + k; /* position in virtual [ctx, input] */ \
            float v; \
            if (pos < ctx_len) { \
                v = static_cast<float>(ctx[(b * channels + c) * ctx_len + pos]); \
            } else { \
                v = static_cast<float>(input[(b * channels + c) * time_len + (pos - ctx_len)]); \
            } \
            acc = fmaf(v, static_cast<float>(weight[wt_off + k]), acc); \
        } \
        acc += static_cast<float>(bias[c]); \
        out[i] = static_cast<TYPENAME>(acc); \
    } \
}

CONV1D_BIAS_CTX_OP(float, depthwise_conv1d_bias_ctx_f32)
CONV1D_BIAS_CTX_OP(double, depthwise_conv1d_bias_ctx_f64)
#if __CUDA_ARCH__ >= 530
CONV1D_BIAS_CTX_OP(__half, depthwise_conv1d_bias_ctx_f16)
#endif
#if __CUDA_ARCH__ >= 800
CONV1D_BIAS_CTX_OP(__nv_bfloat16, depthwise_conv1d_bias_ctx_bf16)
#endif

// ─── rms_norm_channel: RMS-normalize over channel dim of (batch, channels, time) ──
// Replaces transpose + rms_norm + transpose (3 kernels including copy) with 1.
// Each (batch, time) position normalizes its c-dimensional channel vector.
// x: (batch, channels, time), weight: (channels,)
// out: (batch, channels, time)
// Grid: one block per (batch * time) element
#define RMS_NORM_CHANNEL_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *x, \
    const TYPENAME *weight, \
    TYPENAME *out, \
    const int channels, \
    const int time_len, \
    const int block_size, \
    const float eps \
) { \
    const int idx = blockIdx.x; \
    const int b = idx / time_len; \
    const int t = idx % time_len; \
    /* Compute sum of squares over channels for this (b, t) position */ \
    /* x layout: (batch, channels, time) → x[b*channels*time + c*time + t] */ \
    float sum2 = 0.0f; \
    for (int c = threadIdx.x; c < channels; c += block_size) { \
        float v = static_cast<float>(x[b * channels * time_len + c * time_len + t]); \
        sum2 += v * v; \
    } \
    /* Warp-level reduction */ \
    for (int mask = 16; mask > 0; mask >>= 1) { \
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
    } \
    /* Block-level reduction via shared memory */ \
    __shared__ float shared[32]; \
    int warp_id = threadIdx.x / 32; \
    int lane_id = threadIdx.x % 32; \
    if (lane_id == 0) shared[warp_id] = sum2; \
    __syncthreads(); \
    if (warp_id == 0) { \
        sum2 = (lane_id < (block_size + 31) / 32) ? shared[lane_id] : 0.0f; \
        for (int mask = 16; mask > 0; mask >>= 1) { \
            sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
        } \
        shared[0] = sum2; \
    } \
    __syncthreads(); \
    float inv_rms = rsqrtf(shared[0] / (float)channels + eps); \
    /* Apply normalization: out[b,c,t] = x[b,c,t] * inv_rms * weight[c] */ \
    for (int c = threadIdx.x; c < channels; c += block_size) { \
        int off = b * channels * time_len + c * time_len + t; \
        float xv = static_cast<float>(x[off]); \
        float wv_scaled = static_cast<float>(weight[c]) * inv_rms; \
        out[off] = static_cast<TYPENAME>(xv * wv_scaled); \
    } \
}

RMS_NORM_CHANNEL_OP(float, rms_norm_channel_f32)
RMS_NORM_CHANNEL_OP(double, rms_norm_channel_f64)
#if __CUDA_ARCH__ >= 530
RMS_NORM_CHANNEL_OP(__half, rms_norm_channel_f16)
#endif
#if __CUDA_ARCH__ >= 800
RMS_NORM_CHANNEL_OP(__nv_bfloat16, rms_norm_channel_bf16)
#endif

// ─── add_scaled: a + b * c with broadcast on c ─────────────────────
// Replaces broadcast_mul + add (2 kernels) with 1.
// Used for residual + h * gamma where gamma is (1, channels, 1).
// a, b: (batch, channels, time), c: (channels,)
// out: (batch, channels, time) = a + b * c[channel_idx]
#define ADD_SCALED_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *a, \
    const TYPENAME *b, \
    const TYPENAME *c, \
    TYPENAME *out, \
    const int channels, \
    const int time_len \
) { \
    const int ct = channels * time_len; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < numel; i += blockDim.x * gridDim.x) { \
        /* Use single modulo: position within one batch = i % (channels*time_len) */ \
        /* Then chan = pos / time_len */ \
        int pos = i % ct; \
        int chan = pos / time_len; \
        float av = static_cast<float>(a[i]); \
        float bv = static_cast<float>(b[i]); \
        float cv = static_cast<float>(c[chan]); \
        out[i] = static_cast<TYPENAME>(fmaf(bv, cv, av)); \
    } \
}

ADD_SCALED_OP(float, add_scaled_f32)
ADD_SCALED_OP(double, add_scaled_f64)
#if __CUDA_ARCH__ >= 530
ADD_SCALED_OP(__half, add_scaled_f16)
#endif
#if __CUDA_ARCH__ >= 800
ADD_SCALED_OP(__nv_bfloat16, add_scaled_bf16)
#endif

// ─── adaln_modulate: rms_norm(x, w, eps) * (1+scale) + shift ────────
// Fuses rms_norm + scale_add1 + mul + shift_add (4 kernels) into 1.
// Used in diffusion blocks' AdaLN modulation.
// x: (n_rows, n_cols), weight: (n_cols,), scale: (n_rows, n_cols), shift: (n_rows, n_cols)
// out: (n_rows, n_cols) = rms_norm(x) * (1 + scale) + shift
#define ADALN_MODULATE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *x, \
    const TYPENAME *weight, \
    const TYPENAME *scale, \
    const TYPENAME *shift, \
    TYPENAME *out, \
    const int n_cols, \
    const int block_size, \
    const float eps \
) { \
    const int row = blockIdx.x; \
    const int offset = row * n_cols; \
    float sum2 = 0.0f; \
    for (int col = threadIdx.x; col < n_cols; col += block_size) { \
        float v = static_cast<float>(x[offset + col]); \
        sum2 += v * v; \
    } \
    for (int mask = 16; mask > 0; mask >>= 1) { \
        sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
    } \
    __shared__ float shared[32]; \
    int warp_id = threadIdx.x / 32; \
    int lane_id = threadIdx.x % 32; \
    if (lane_id == 0) shared[warp_id] = sum2; \
    __syncthreads(); \
    if (warp_id == 0) { \
        sum2 = (lane_id < (block_size + 31) / 32) ? shared[lane_id] : 0.0f; \
        for (int mask = 16; mask > 0; mask >>= 1) { \
            sum2 += __shfl_xor_sync(0xffffffff, sum2, mask); \
        } \
        shared[0] = sum2; \
    } \
    __syncthreads(); \
    float inv_rms = rsqrtf(shared[0] / (float)n_cols + eps); \
    for (int col = threadIdx.x; col < n_cols; col += block_size) { \
        float xv = static_cast<float>(x[offset + col]) * inv_rms; \
        float wv = static_cast<float>(weight[col]); \
        float sv = static_cast<float>(scale[offset + col]); \
        float shv = static_cast<float>(shift[offset + col]); \
        out[offset + col] = static_cast<TYPENAME>(fmaf(xv * wv, 1.0f + sv, shv)); \
    } \
}

ADALN_MODULATE_OP(float, adaln_modulate_f32)
ADALN_MODULATE_OP(double, adaln_modulate_f64)
#if __CUDA_ARCH__ >= 530
ADALN_MODULATE_OP(__half, adaln_modulate_f16)
#endif
#if __CUDA_ARCH__ >= 800
ADALN_MODULATE_OP(__nv_bfloat16, adaln_modulate_bf16)
#endif

SUB_MUL_OP(float, sub_mul_f32)
SUB_MUL_OP(double, sub_mul_f64)
#if __CUDA_ARCH__ >= 530
SUB_MUL_OP(__half, sub_mul_f16)
#endif
#if __CUDA_ARCH__ >= 800
SUB_MUL_OP(__nv_bfloat16, sub_mul_bf16)
#endif
