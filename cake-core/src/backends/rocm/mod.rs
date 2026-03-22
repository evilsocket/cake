//! ROCm compute backend via raw HIP + rocBLAS FFI.
//!
//! Native AMD GPU acceleration using HIP for memory management and rocBLAS
//! for GEMM. Links against `libamdhip64.so` and `librocblas.so` at runtime.
//!
//! **Requires**: `hip-runtime-amd` and `rocblas` packages installed.
//! **Steam Deck**: Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` for gfx1033 compat.

mod ffi;

use std::collections::HashMap;
use std::sync::Mutex;

use candle_core::{DType, Device, Result, Tensor, TensorId, D};

use super::ComputeBackend;

/// GPU buffer wrapper — frees on drop.
struct GpuBuf {
    ptr: *mut std::ffi::c_void,
    count: usize,
}

impl GpuBuf {
    fn new(count: usize) -> std::result::Result<Self, String> {
        let mut ptr = std::ptr::null_mut();
        let err = unsafe { ffi::hipMalloc(&mut ptr, count * 4) };
        if err != 0 {
            return Err(format!("hipMalloc failed: error {err}"));
        }
        Ok(Self { ptr, count })
    }

    fn upload(data: &[f32]) -> std::result::Result<Self, String> {
        let buf = Self::new(data.len())?;
        let err = unsafe {
            ffi::hipMemcpy(
                buf.ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len() * 4,
                ffi::HIP_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 {
            return Err(format!("hipMemcpy H2D failed: error {err}"));
        }
        Ok(buf)
    }

    fn download(&self) -> std::result::Result<Vec<f32>, String> {
        let mut host = vec![0f32; self.count];
        let err = unsafe {
            ffi::hipMemcpy(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                self.count * 4,
                ffi::HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if err != 0 {
            return Err(format!("hipMemcpy D2H failed: error {err}"));
        }
        Ok(host)
    }

    fn as_ptr(&self) -> *const f32 {
        self.ptr as *const f32
    }

    fn as_mut_ptr(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::hipFree(self.ptr) };
        }
    }
}

// GpuBuf is Send+Sync — GPU pointers are valid across threads after sync
unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

/// ROCm backend with rocBLAS GEMM and cached GPU weight buffers.
pub struct RocmBackend {
    device: Device,
    blas: *mut std::ffi::c_void, // rocblas_handle
    cache: Mutex<HashMap<TensorId, GpuBuf>>,
}

// rocblas_handle is thread-safe when not used concurrently (we hold Mutex)
unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl std::fmt::Debug for RocmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend").field("device", &"rocm").finish()
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        if !self.blas.is_null() {
            unsafe { ffi::rocblas_destroy_handle(self.blas) };
        }
    }
}

impl RocmBackend {
    pub fn new() -> std::result::Result<Self, String> {
        // Initialize HIP
        let err = unsafe { ffi::hipInit(0) };
        if err != 0 {
            return Err(format!("hipInit failed: error {err}"));
        }

        let mut count = 0i32;
        let err = unsafe { ffi::hipGetDeviceCount(&mut count) };
        if err != 0 || count == 0 {
            return Err("no HIP devices found".into());
        }

        let err = unsafe { ffi::hipSetDevice(0) };
        if err != 0 {
            return Err(format!("hipSetDevice(0) failed: error {err}"));
        }

        // Get device name
        let mut name = [0u8; 256];
        unsafe { ffi::hipDeviceGetName(name.as_mut_ptr() as *mut i8, 256, 0) };
        let name = std::ffi::CStr::from_bytes_until_nul(&name)
            .unwrap_or_default()
            .to_string_lossy();

        // Get memory info
        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { ffi::hipMemGetInfo(&mut free, &mut total) };

        log::info!(
            "ROCm backend: {} ({:.1} GiB free / {:.1} GiB total)",
            name,
            free as f64 / 1e9,
            total as f64 / 1e9,
        );

        // Create rocBLAS handle
        let mut handle = std::ptr::null_mut();
        let err = unsafe { ffi::rocblas_create_handle(&mut handle) };
        if err != 0 {
            return Err(format!("rocblas_create_handle failed: error {err}"));
        }

        Ok(Self {
            device: Device::Cpu,
            blas: handle,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()
    }

    fn get_or_upload(&self, tensor: &Tensor) -> Result<*const f32> {
        let id = tensor.id();
        let mut cache = self.cache.lock().unwrap();
        if let Some(buf) = cache.get(&id) {
            return Ok(buf.as_ptr());
        }
        let data = Self::to_f32_vec(tensor)?;
        let buf = GpuBuf::upload(&data)
            .map_err(|e| candle_core::Error::Msg(e))?;
        let ptr = buf.as_ptr();
        cache.insert(id, buf);
        Ok(ptr)
    }

    /// C = A(m×k) × B(k×n) using rocBLAS sgemm.
    /// Row-major: compute C^T = B^T × A^T in column-major.
    fn rocblas_gemm(&self, a: *const f32, b: *const f32, m: usize, k: usize, n: usize) -> Result<GpuBuf> {
        let c = GpuBuf::new(m * n)
            .map_err(|e| candle_core::Error::Msg(e))?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // Row-major trick: C = A×B  ⟺  C^T = B^T × A^T (column-major)
        let err = unsafe {
            ffi::rocblas_sgemm(
                self.blas,
                ffi::ROCBLAS_OPERATION_NONE, // B^T
                ffi::ROCBLAS_OPERATION_NONE, // A^T
                n as i32,  // rows of B^T = N
                m as i32,  // cols of A^T = M
                k as i32,
                &alpha,
                b, n as i32,  // B with ldb=N
                a, k as i32,  // A with lda=K
                &beta,
                c.as_mut_ptr(), n as i32,  // C with ldc=N
            )
        };
        if err != 0 {
            return Err(candle_core::Error::Msg(format!("rocblas_sgemm failed: {err}")));
        }

        // Sync
        let err = unsafe { ffi::hipDeviceSynchronize() };
        if err != 0 {
            return Err(candle_core::Error::Msg(format!("hipDeviceSynchronize: {err}")));
        }

        Ok(c)
    }

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
            let a_ptr = self.get_or_upload(&a)?;
            let b_ptr = self.get_or_upload(&b)?;
            let c = self.rocblas_gemm(a_ptr, b_ptr, m, k, n)?;
            let data = c.download().map_err(|e| candle_core::Error::Msg(e))?;
            let mut shape = a_dims[..rank_a - 2].to_vec();
            shape.push(m);
            shape.push(n);
            return Tensor::from_vec(data, shape.as_slice(), &Device::Cpu);
        }

        // Multi-batch
        let a_data = Self::to_f32_vec(&a)?;
        let b_data = Self::to_f32_vec(&b)?;
        let mk = m * k;
        let kn = k * n;
        let mn = m * n;
        let mut out = Vec::with_capacity(batch * mn);

        for i in 0..batch {
            let a_off = if a_batch == 1 { 0 } else { i * mk };
            let b_off = if b_batch == 1 { 0 } else { i * kn };
            let a_buf = GpuBuf::upload(&a_data[a_off..a_off + mk])
                .map_err(|e| candle_core::Error::Msg(e))?;
            let b_buf = GpuBuf::upload(&b_data[b_off..b_off + kn])
                .map_err(|e| candle_core::Error::Msg(e))?;
            let c = self.rocblas_gemm(a_buf.as_ptr(), b_buf.as_ptr(), m, k, n)?;
            out.extend_from_slice(&c.download().map_err(|e| candle_core::Error::Msg(e))?);
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

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let m = a.dims()[a.dims().len() - 2];
        if m <= 4 { return a.matmul(b); }
        let orig = a.dtype();
        self.tensor_matmul(a, b)?.to_dtype(orig)
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let orig = q.dtype();
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let attn = self.tensor_matmul(&q, &k.t()?)?;
        let attn = (attn * scale as f64)?;
        let attn = if causal {
            let sl = q.dim(2)?;
            let kl = k.dim(2)?;
            let mut mask = vec![0u8; sl * kl];
            for i in 0..sl {
                for j in 0..=kl.saturating_sub(sl) + i { if j < kl { mask[i * kl + j] = 1; } }
            }
            let mask = Tensor::from_vec(mask, (1, 1, sl, kl), q.device())?;
            let neg_inf = Tensor::full(f32::NEG_INFINITY, attn.shape(), q.device())?;
            mask.broadcast_as(attn.shape())?.where_cond(&attn, &neg_inf)?
        } else { attn };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        self.tensor_matmul(&attn, &v)?.to_dtype(orig)
    }

    // CPU tensor ops for non-matmul operations
    fn silu_mul(&self, g: &Tensor, u: &Tensor) -> Result<Tensor> {
        (candle_nn::ops::silu(&g.contiguous()?)? * u.contiguous()?)?.contiguous()
    }
    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> {
        let t = Tensor::full(88f32, x.shape(), x.device())?.to_dtype(x.dtype())?;
        x.maximum(&(x.minimum(&t)?.exp()? + 1.0)?.log()?)
    }
    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
        (candle_nn::ops::rms_norm(&x.contiguous()?, w, eps)? * candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?)?.contiguous()
    }
    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, w: &Tensor, eps: f32) -> Result<(Tensor, Tensor)> {
        let r = (a + b)?; Ok((r.clone(), candle_nn::ops::rms_norm(&r.contiguous()?, w, eps)?))
    }
    fn rms_norm_channel(&self, x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
        x.transpose(1,2)?.contiguous().and_then(|t| candle_nn::ops::rms_norm(&t, w, eps))?.transpose(1,2)?.contiguous()
    }
    fn depthwise_conv1d_silu(&self, win: &Tensor, w: &Tensor, _ks: usize, _ch: usize) -> Result<Tensor> {
        candle_nn::ops::silu(&win.broadcast_mul(&w.unsqueeze(0)?)?.sum(D::Minus1)?)
    }
    fn depthwise_conv1d_bias(&self, inp: &Tensor, w: &Tensor, b: &Tensor, ks: usize, _ch: usize) -> Result<Tensor> {
        let ot = inp.dim(2)? - ks + 1;
        let mut s = Vec::with_capacity(ot);
        for t in 0..ot { s.push(inp.narrow(2,t,ks)?.broadcast_mul(&w.unsqueeze(0)?)?.sum(D::Minus1)?.broadcast_add(b)?.unsqueeze(2)?); }
        Tensor::cat(&s, 2)
    }
    fn depthwise_conv1d_bias_ctx(&self, c: &Tensor, i: &Tensor, w: &Tensor, b: &Tensor, ks: usize, ch: usize) -> Result<Tensor> {
        self.depthwise_conv1d_bias(&Tensor::cat(&[c, i], 2)?, w, b, ks, ch)
    }
    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> { ((a+b)?+c)?.contiguous() }
    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> { (x * y.exp()?)?.contiguous() }
    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> { ((a-b)?*c)?.contiguous() }
    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> {
        (a + b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?.contiguous()
    }
    fn adaln_modulate(&self, x: &Tensor, nw: &Tensor, sc: &Tensor, sh: &Tensor, eps: f32) -> Result<Tensor> {
        (candle_nn::ops::rms_norm(&x.contiguous()?, nw, eps)?.broadcast_mul(&(sc + 1.0)?)? + sh)?.contiguous()
    }
    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32) }
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F16) }
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::BF16) }
}
