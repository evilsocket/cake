//! ROCm compute backend via rocm-rs (HIP + rocBLAS).
//!
//! Native AMD GPU acceleration using ROCm's HIP runtime and rocBLAS GEMM.
//! Weight matrices are uploaded once to GPU memory and cached by `TensorId`.
//! rocBLAS provides hand-tuned GEMM kernels for RDNA 2 (Steam Deck gfx1033).
//!
//! **Requires**: `hip-runtime-amd` and `rocblas` packages installed.
//! **Steam Deck**: Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for gfx1033 → gfx1030 compat.

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Result, Tensor, TensorId, D};
use rocm_rs::hip::{self, DeviceMemory, Stream};
use rocm_rs::rocblas::{self, Handle as RocblasHandle, Operation};

use super::ComputeBackend;

/// Cached GPU buffer for a tensor.
struct GpuBuffer {
    mem: DeviceMemory<f32>,
    count: usize,
}

/// ROCm backend with rocBLAS GEMM and persistent GPU weight buffers.
pub struct RocmBackend {
    device: Device,
    stream: Stream,
    blas: RocblasHandle,
    /// Weight tensors cached on GPU by TensorId.
    cache: Mutex<HashMap<TensorId, GpuBuffer>>,
}

impl std::fmt::Debug for RocmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend").field("device", &"rocm").finish()
    }
}

impl RocmBackend {
    pub fn new() -> std::result::Result<Self, String> {
        let dev = hip::Device::new(0).map_err(|e| format!("HIP device: {e}"))?;
        dev.set_current().map_err(|e| format!("HIP set_current: {e}"))?;

        let info = hip::memory::memory_info().map_err(|e| format!("HIP meminfo: {e}"))?;
        log::info!(
            "ROCm backend: device 0, {:.1} GiB free / {:.1} GiB total",
            info.free as f64 / 1e9,
            info.total as f64 / 1e9,
        );

        let stream = Stream::new().map_err(|e| format!("HIP stream: {e}"))?;
        let blas = RocblasHandle::new().map_err(|e| format!("rocBLAS: {e}"))?;
        blas.set_stream(&stream).map_err(|e| format!("rocBLAS set_stream: {e}"))?;

        Ok(Self {
            device: Device::Cpu, // candle tensors still on CPU
            stream,
            blas,
            cache: Mutex::new(HashMap::new()),
        })
    }

    // ── GPU buffer management ───────────────────────────────────────

    /// Upload tensor data to GPU, caching by TensorId for weight reuse.
    fn get_or_upload(&self, tensor: &Tensor) -> Result<(*const f32, usize)> {
        let id = tensor.id();
        let mut cache = self.cache.lock().unwrap();

        if let Some(buf) = cache.get(&id) {
            return Ok((buf.mem.as_ptr() as *const f32, buf.count));
        }

        let data = Self::to_f32_vec(tensor)?;
        let count = data.len();
        let mut mem = DeviceMemory::<f32>::new(count)
            .map_err(|e| candle_core::Error::Msg(format!("HIP alloc: {e}")))?;
        mem.copy_from_host(&data)
            .map_err(|e| candle_core::Error::Msg(format!("HIP upload: {e}")))?;

        let ptr = mem.as_ptr() as *const f32;
        cache.insert(id, GpuBuffer { mem, count });
        Ok((ptr, count))
    }

    /// Allocate a temporary GPU output buffer.
    fn alloc_output(&self, count: usize) -> Result<DeviceMemory<f32>> {
        DeviceMemory::<f32>::new(count)
            .map_err(|e| candle_core::Error::Msg(format!("HIP alloc: {e}")))
    }

    /// Download GPU buffer to host Vec<f32>.
    fn download(mem: &DeviceMemory<f32>, count: usize) -> Result<Vec<f32>> {
        let mut host = vec![0f32; count];
        mem.copy_to_host(&mut host)
            .map_err(|e| candle_core::Error::Msg(format!("HIP download: {e}")))?;
        Ok(host)
    }

    fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()
    }

    // ── rocBLAS GEMM ────────────────────────────────────────────────

    /// C = alpha * A(m×k) × B(k×n) + beta * C
    /// A, B are already on GPU. Returns GPU output buffer.
    fn rocblas_gemm(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<DeviceMemory<f32>> {
        let mut c = self.alloc_output(m * n)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // rocBLAS uses column-major. To compute row-major C = A × B,
        // we compute C^T = B^T × A^T in column-major, which gives us
        // C in row-major layout. So we swap A↔B and M↔N.
        unsafe {
            rocblas::level3::gemm::<f32>(
                &self.blas,
                Operation::None,  // B^T → no transpose (B is already row-major = col-major transposed)
                Operation::None,  // A^T → no transpose
                n as i32,         // rows of op(B^T) = cols of B = N
                m as i32,         // cols of op(A^T) = rows of A = M
                k as i32,         // shared dim
                &alpha,
                b_ptr,            // "A" in col-major = B in row-major
                n as i32,         // lda = N (leading dim of B in col-major)
                a_ptr,            // "B" in col-major = A in row-major
                k as i32,         // ldb = K (leading dim of A in col-major)
                &beta,
                c.as_ptr() as *mut f32,
                n as i32,         // ldc = N
            )
            .map_err(|e| candle_core::Error::Msg(format!("rocblas_sgemm: {e}")))?;
        }

        // Synchronize to ensure GEMM completes before reading
        self.stream
            .synchronize()
            .map_err(|e| candle_core::Error::Msg(format!("HIP sync: {e}")))?;

        Ok(c)
    }

    /// Tensor matmul via rocBLAS. Handles batched dimensions.
    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = a.to_dtype(DType::F32)?.contiguous()?;
        let b = b.to_dtype(DType::F32)?.contiguous()?;

        let a_dims = a.dims();
        let b_dims = b.dims();
        let rank_a = a_dims.len();
        let rank_b = b_dims.len();

        let m = a_dims[rank_a - 2];
        let k = a_dims[rank_a - 1];
        let n = b_dims[rank_b - 1];

        let a_batch: usize = a_dims[..rank_a - 2].iter().product();
        let b_batch: usize = b_dims[..rank_b - 2].iter().product();
        let batch = a_batch.max(b_batch);

        if batch <= 1 {
            // Single batch: use cached buffers
            let (a_ptr, _) = self.get_or_upload(&a)?;
            let (b_ptr, _) = self.get_or_upload(&b)?;
            let c_mem = self.rocblas_gemm(a_ptr, b_ptr, m, k, n)?;
            let c_data = Self::download(&c_mem, m * n)?;

            let mut shape = a_dims[..rank_a - 2].to_vec();
            shape.push(m);
            shape.push(n);
            return Tensor::from_vec(c_data, shape.as_slice(), &Device::Cpu);
        }

        // Multi-batch: extract per-batch slices
        let a_data = Self::to_f32_vec(&a)?;
        let b_data = Self::to_f32_vec(&b)?;
        let mk = m * k;
        let kn = k * n;
        let mn = m * n;

        let mut out = Vec::with_capacity(batch * mn);
        for i in 0..batch {
            let a_off = if a_batch == 1 { 0 } else { i * mk };
            let b_off = if b_batch == 1 { 0 } else { i * kn };

            let mut a_mem = self.alloc_output(mk)?;
            let mut b_mem = self.alloc_output(kn)?;
            a_mem.copy_from_host(&a_data[a_off..a_off + mk])
                .map_err(|e| candle_core::Error::Msg(format!("HIP upload: {e}")))?;
            b_mem.copy_from_host(&b_data[b_off..b_off + kn])
                .map_err(|e| candle_core::Error::Msg(format!("HIP upload: {e}")))?;

            let c_mem = self.rocblas_gemm(
                a_mem.as_ptr() as *const f32,
                b_mem.as_ptr() as *const f32,
                m, k, n,
            )?;
            out.extend_from_slice(&Self::download(&c_mem, mn)?);
        }

        let mut shape = a_dims[..rank_a - 2].to_vec();
        shape.push(m);
        shape.push(n);
        Tensor::from_vec(out, shape.as_slice(), &Device::Cpu)
    }
}

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str { "rocm" }
    fn device(&self) -> &Device { &self.device }

    // ── rocBLAS-accelerated matmul ──────────────────────────────────

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let m = a.dims()[a.dims().len() - 2];
        // CPU fast path for tiny matmuls (generation M=1)
        if m <= 4 {
            return a.matmul(b);
        }
        let orig_dtype = a.dtype();
        self.tensor_matmul(a, b)?.to_dtype(orig_dtype)
    }

    // ── rocBLAS-accelerated attention ────────────────────────────────

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let orig_dtype = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Q @ K^T via rocBLAS
        let attn = self.tensor_matmul(&q, &k.t()?)?;
        let attn = (attn * scale as f64)?;

        let attn = if causal {
            let seq_len = q.dim(2)?;
            let kv_len = k.dim(2)?;
            let mut mask = vec![0u8; seq_len * kv_len];
            for i in 0..seq_len {
                let max_j = kv_len.saturating_sub(seq_len) + i;
                for j in 0..=max_j.min(kv_len - 1) {
                    mask[i * kv_len + j] = 1;
                }
            }
            let mask = Tensor::from_vec(mask, (1, 1, seq_len, kv_len), q.device())?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, attn.shape(), q.device())?;
            mask.broadcast_as(attn.shape())?.where_cond(&attn, &neg_inf)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // attn @ V via rocBLAS
        let out = self.tensor_matmul(&attn, &v)?;
        out.to_dtype(orig_dtype)
    }

    // ── CPU tensor ops for non-matmul operations ────────────────────
    // These are fast on CPU and don't benefit from GPU dispatch overhead.

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        (candle_nn::ops::silu(&gate.contiguous()?)? * up.contiguous()?)?.contiguous()
    }

    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        let t88 = Tensor::full(88.0f32, x.shape(), x.device())?.to_dtype(x.dtype())?;
        let clamped = x.minimum(&t88)?;
        let sp = (clamped.exp()? + 1.0)?.log()?;
        x.maximum(&sp)
    }

    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let n = candle_nn::ops::rms_norm(&x.contiguous()?, weight, eps)?;
        (n * candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?)?.contiguous()
    }

    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, weight: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        let res = (a + b)?;
        let normed = candle_nn::ops::rms_norm(&res.contiguous()?, weight, eps)?;
        Ok((res, normed))
    }

    fn rms_norm_channel(&self, x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        x.transpose(1, 2)?.contiguous()
            .and_then(|t| candle_nn::ops::rms_norm(&t, weight, eps))?
            .transpose(1, 2)?.contiguous()
    }

    fn depthwise_conv1d_silu(&self, window: &Tensor, weight: &Tensor, _ks: usize, _ch: usize) -> Result<Tensor> {
        candle_nn::ops::silu(&window.broadcast_mul(&weight.unsqueeze(0)?)?.sum(D::Minus1)?)
    }

    fn depthwise_conv1d_bias(&self, input: &Tensor, weight: &Tensor, bias: &Tensor, ks: usize, _ch: usize) -> Result<Tensor> {
        let out_t = input.dim(2)? - ks + 1;
        let mut slices = Vec::with_capacity(out_t);
        for t in 0..out_t {
            let w = input.narrow(2, t, ks)?.broadcast_mul(&weight.unsqueeze(0)?)?.sum(D::Minus1)?.broadcast_add(bias)?;
            slices.push(w.unsqueeze(2)?);
        }
        Tensor::cat(&slices, 2)
    }

    fn depthwise_conv1d_bias_ctx(&self, ctx: &Tensor, input: &Tensor, weight: &Tensor, bias: &Tensor, ks: usize, ch: usize) -> Result<Tensor> {
        self.depthwise_conv1d_bias(&Tensor::cat(&[ctx, input], 2)?, weight, bias, ks, ch)
    }

    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ((a + b)? + c)?.contiguous()
    }

    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> {
        (x * y.exp()?)?.contiguous()
    }

    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        ((a - b)? * c)?.contiguous()
    }

    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?.contiguous()
    }

    fn adaln_modulate(&self, x: &Tensor, nw: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
        (candle_nn::ops::rms_norm(&x.contiguous()?, nw, eps)?.broadcast_mul(&(scale + 1.0)?)? + shift)?.contiguous()
    }

    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32) }
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F16) }
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::BF16) }
}
