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

// ─── silu_mul: silu(x) * y = x * sigmoid(x) * y ────────────────────
// Fuses 2 kernels (silu + mul) into 1.
// Used in every MLP (gate activation * up projection).
template<typename T>
__device__ __forceinline__ T silu_mul_fwd(T x, T y) {
    T one = static_cast<T>(1);
    return x / (one + expg(-x)) * y;
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

// Specialise for half types using float intermediates
#if __CUDA_ARCH__ >= 530
template<>
__device__ __forceinline__ __half stable_softplus_fwd<__half>(__half x) {
    float fx = __half2float(x);
    float clamped = fminf(fx, 88.0f);
    float sp = logf(expf(clamped) + 1.0f);
    float result = fmaxf(fx, sp);
    return __float2half(result);
}
#endif
#if __CUDA_ARCH__ >= 800
template<>
__device__ __forceinline__ __nv_bfloat16 stable_softplus_fwd<__nv_bfloat16>(__nv_bfloat16 x) {
    float fx = __bfloat162float(x);
    float clamped = fminf(fx, 88.0f);
    float sp = logf(expf(clamped) + 1.0f);
    float result = fmaxf(fx, sp);
    return __float2bfloat16(result);
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
        float wv = static_cast<float>(weight[col]); \
        float zv = static_cast<float>(z[offset + col]); \
        float silu_z = zv / (1.0f + expf(-zv)); \
        out[offset + col] = static_cast<TYPENAME>(xv * inv_rms * wv * silu_z); \
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

SUB_MUL_OP(float, sub_mul_f32)
SUB_MUL_OP(double, sub_mul_f64)
#if __CUDA_ARCH__ >= 530
SUB_MUL_OP(__half, sub_mul_f16)
#endif
#if __CUDA_ARCH__ >= 800
SUB_MUL_OP(__nv_bfloat16, sub_mul_bf16)
#endif
