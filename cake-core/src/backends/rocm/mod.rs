//! ROCm compute backend via runtime-loaded HIP + rocBLAS.
//!
//! Loads libamdhip64 and librocblas via dlopen at runtime — no compile-time
//! ROCm dependency. Works with ROCm 5.4.2 libs on Steam Deck (set
//! LD_LIBRARY_PATH to the extracted libs directory).
//!
//! **Steam Deck setup**:
//! ```sh
//! export HSA_OVERRIDE_GFX_VERSION=10.3.0
//! export LD_LIBRARY_PATH=/path/to/rocm542/lib
//! export ROCBLAS_TENSILE_LIBPATH=/path/to/rocm542/lib/rocblas/library
//! ```

mod ffi;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;

/// Whether HIP runtime has been initialized (used for SIGABRT workaround).
static HIP_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// SIGABRT handler: HIP runtime crashes in Monitor::tryLock during process exit
/// because its atexit cleanup runs after threads are torn down. We catch the
/// resulting abort and force a clean exit since all useful work is already done.
///
/// Uses `exit_group` syscall directly for reliable process termination.
extern "C" fn hip_sigabrt_handler(_sig: libc::c_int) {
    if HIP_INITIALIZED.load(Ordering::Relaxed) {
        unsafe { libc::syscall(libc::SYS_exit_group, 0); }
    }
    // Not HIP-related — re-raise with default handler
    unsafe {
        libc::signal(libc::SIGABRT, libc::SIG_DFL);
        libc::raise(libc::SIGABRT);
    }
}

/// atexit handler: terminates process before HIP's broken atexit cleanup runs.
/// Registered after HIP init, so runs before HIP's handler (LIFO order).
extern "C" fn hip_atexit_handler() {
    if HIP_INITIALIZED.load(Ordering::Relaxed) {
        // Flush stdio then hard-exit to prevent HIP atexit crash
        unsafe {
            libc::fflush(std::ptr::null_mut());
            libc::syscall(libc::SYS_exit_group, 0);
        }
    }
}

use candle_core::{DType, Device, Result, Tensor, TensorId, D};

use super::ComputeBackend;
use ffi::{RocmFfi, HIP_MEMCPY_HOST_TO_DEVICE, HIP_MEMCPY_DEVICE_TO_HOST, ROCBLAS_OPERATION_NONE};

/// GPU buffer wrapper — frees via HIP on drop.
struct GpuBuf {
    ptr: *mut std::ffi::c_void,
    count: usize,
    ffi: *const RocmFfi, // non-owning ref, valid for backend lifetime
}

impl GpuBuf {
    #[inline] fn as_ptr(&self) -> *const f32 { self.ptr as *const f32 }
    #[inline] fn as_mut_ptr(&self) -> *mut f32 { self.ptr as *mut f32 }

    /// Async download using pinned host memory for faster DMA, then sync.
    fn download_on_stream(&self, stream: *mut std::ffi::c_void) -> std::result::Result<Vec<f32>, String> {
        let bytes = self.count * 4;
        // Allocate pinned (page-locked) host memory for faster DMA transfer
        let mut pinned_ptr = std::ptr::null_mut();
        let err = unsafe { ((*self.ffi).hip_host_malloc)(&mut pinned_ptr, bytes, 0) };
        if err != 0 {
            // Fallback to regular memory if pinned alloc fails
            let mut host = vec![0f32; self.count];
            let err = unsafe {
                ((*self.ffi).hip_memcpy_async)(
                    host.as_mut_ptr() as *mut std::ffi::c_void,
                    self.ptr, bytes, HIP_MEMCPY_DEVICE_TO_HOST, stream,
                )
            };
            if err != 0 { return Err(format!("hipMemcpyAsync D2H: error {err}")); }
            let err = unsafe { ((*self.ffi).hip_stream_synchronize)(stream) };
            if err != 0 { return Err(format!("hipStreamSync: error {err}")); }
            return Ok(host);
        }
        let err = unsafe {
            ((*self.ffi).hip_memcpy_async)(
                pinned_ptr, self.ptr, bytes, HIP_MEMCPY_DEVICE_TO_HOST, stream,
            )
        };
        if err != 0 {
            unsafe { ((*self.ffi).hip_host_free)(pinned_ptr) };
            return Err(format!("hipMemcpyAsync D2H pinned: error {err}"));
        }
        let err = unsafe { ((*self.ffi).hip_stream_synchronize)(stream) };
        if err != 0 {
            unsafe { ((*self.ffi).hip_host_free)(pinned_ptr) };
            return Err(format!("hipStreamSync: error {err}"));
        }
        // Copy from pinned to regular heap memory, then free pinned
        let mut host = vec![0f32; self.count];
        unsafe {
            std::ptr::copy_nonoverlapping(pinned_ptr as *const f32, host.as_mut_ptr(), self.count);
            ((*self.ffi).hip_host_free)(pinned_ptr);
        }
        Ok(host)
    }
}

impl Drop for GpuBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ((*self.ffi).hip_free)(self.ptr) };
        }
    }
}

unsafe impl Send for GpuBuf {}
unsafe impl Sync for GpuBuf {}

/// ROCm backend with rocBLAS GEMM and cached GPU weight buffers.
pub struct RocmBackend {
    device: Device,
    ffi: Box<RocmFfi>,
    blas_handle: *mut std::ffi::c_void,
    stream: *mut std::ffi::c_void,
    cache: RwLock<HashMap<TensorId, GpuBuf>>,
}

unsafe impl Send for RocmBackend {}
unsafe impl Sync for RocmBackend {}

impl std::fmt::Debug for RocmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RocmBackend").field("device", &"rocm").finish()
    }
}

impl Drop for RocmBackend {
    fn drop(&mut self) {
        if !self.blas_handle.is_null() {
            unsafe { (self.ffi.rocblas_destroy_handle)(self.blas_handle) };
        }
    }
}

impl RocmBackend {
    pub fn new() -> std::result::Result<Self, String> {
        let ffi = Box::new(RocmFfi::load()?);

        let err = unsafe { (ffi.hip_init)(0) };
        if err != 0 { return Err(format!("hipInit: error {err}")); }

        // Two-layer defense against HIP's broken atexit cleanup:
        // 1. SIGABRT handler catches abort() from HIP's assertion failures
        // 2. atexit handler terminates before HIP's cleanup runs (LIFO order)
        if !HIP_INITIALIZED.swap(true, Ordering::Relaxed) {
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = hip_sigabrt_handler as *const () as libc::sighandler_t;
                libc::sigemptyset(&mut sa.sa_mask);
                sa.sa_flags = libc::SA_NODEFER;
                libc::sigaction(libc::SIGABRT, &sa, std::ptr::null_mut());
                libc::atexit(hip_atexit_handler);
            }
        }

        let mut count = 0i32;
        unsafe { (ffi.hip_get_device_count)(&mut count) };
        if count == 0 { return Err("no HIP devices".into()); }

        unsafe { (ffi.hip_set_device)(0) };

        let mut name = [0u8; 256];
        unsafe { (ffi.hip_device_get_name)(name.as_mut_ptr() as *mut i8, 256, 0) };
        let name = std::ffi::CStr::from_bytes_until_nul(&name)
            .unwrap_or_default().to_string_lossy();

        let mut free = 0usize;
        let mut total = 0usize;
        unsafe { (ffi.hip_mem_get_info)(&mut free, &mut total) };

        log::info!("ROCm backend: {} ({:.1} GiB free / {:.1} GiB total)",
            name, free as f64 / 1e9, total as f64 / 1e9);

        let mut handle = std::ptr::null_mut();
        let err = unsafe { (ffi.rocblas_create_handle)(&mut handle) };
        if err != 0 { return Err(format!("rocblas_create_handle: error {err}")); }

        let mut stream = std::ptr::null_mut();
        let err = unsafe { (ffi.hip_stream_create)(&mut stream) };
        if err != 0 { return Err(format!("hipStreamCreate: error {err}")); }

        // Bind stream to rocBLAS for async GEMM
        let err = unsafe { (ffi.rocblas_set_stream)(handle, stream) };
        if err != 0 { return Err(format!("rocblas_set_stream: error {err}")); }

        Ok(Self {
            device: Device::Cpu,
            ffi,
            blas_handle: handle,
            stream,
            cache: RwLock::new(HashMap::with_capacity(64)),
        })
    }

    #[inline]
    fn gpu_alloc(&self, count: usize) -> Result<GpuBuf> {
        let mut ptr = std::ptr::null_mut();
        let err = unsafe { (self.ffi.hip_malloc)(&mut ptr, count * 4) };
        if err != 0 { return Err(candle_core::Error::Msg(format!("hipMalloc: {err}"))); }
        Ok(GpuBuf { ptr, count, ffi: &*self.ffi as *const RocmFfi })
    }

    #[inline]
    fn gpu_upload(&self, data: &[f32]) -> Result<GpuBuf> {
        let buf = self.gpu_alloc(data.len())?;
        let bytes = data.len() * 4;
        // Try pinned staging buffer for faster DMA (important on UMA like Steam Deck)
        let mut pinned = std::ptr::null_mut();
        if unsafe { (self.ffi.hip_host_malloc)(&mut pinned, bytes, 0) } == 0 {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, pinned as *mut u8, bytes);
            }
            let err = unsafe {
                (self.ffi.hip_memcpy_async)(buf.ptr, pinned, bytes, HIP_MEMCPY_HOST_TO_DEVICE, self.stream)
            };
            // Sync before freeing pinned buffer (memcpy must complete first)
            let _ = unsafe { (self.ffi.hip_stream_synchronize)(self.stream) };
            unsafe { (self.ffi.hip_host_free)(pinned) };
            if err != 0 { return Err(candle_core::Error::Msg(format!("hipMemcpyAsync H2D: {err}"))); }
        } else {
            // Fallback: regular async memcpy from unpinned memory
            let err = unsafe {
                (self.ffi.hip_memcpy_async)(buf.ptr, data.as_ptr() as *const _, bytes, HIP_MEMCPY_HOST_TO_DEVICE, self.stream)
            };
            if err != 0 { return Err(candle_core::Error::Msg(format!("hipMemcpyAsync H2D: {err}"))); }
        }
        Ok(buf)
    }

    #[inline]
    fn to_f32_vec(t: &Tensor) -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1()
    }

    #[inline]
    fn get_or_upload(&self, tensor: &Tensor) -> Result<*const f32> {
        let id = tensor.id();
        // Fast path: read lock for cache hit (no write contention)
        if let Some(buf) = self.cache.read().unwrap().get(&id) {
            return Ok(buf.as_ptr());
        }
        // Slow path: write lock for cache miss
        let data = Self::to_f32_vec(tensor)?;
        let buf = self.gpu_upload(&data)?;
        let ptr = buf.as_ptr();
        self.cache.write().unwrap().insert(id, buf);
        Ok(ptr)
    }

    #[inline]
    fn rocblas_gemm(&self, a: *const f32, b: *const f32, m: usize, k: usize, n: usize) -> Result<GpuBuf> {
        let c = self.gpu_alloc(m * n)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        let err = unsafe {
            (self.ffi.rocblas_sgemm)(
                self.blas_handle,
                ROCBLAS_OPERATION_NONE, ROCBLAS_OPERATION_NONE,
                n as i32, m as i32, k as i32,
                &alpha, b, n as i32, a, k as i32,
                &beta, c.as_mut_ptr(), n as i32,
            )
        };
        if err != 0 { return Err(candle_core::Error::Msg(format!("rocblas_sgemm: {err}"))); }
        // No sync here — caller does download_on_stream which syncs
        Ok(c)
    }

    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = if a.dtype() == DType::F32 && a.is_contiguous() { a.clone() }
            else { a.to_dtype(DType::F32)?.contiguous()? };
        let b = if b.dtype() == DType::F32 && b.is_contiguous() { b.clone() }
            else { b.to_dtype(DType::F32)?.contiguous()? };
        let ad = a.dims(); let bd = b.dims();
        let ra = ad.len(); let rb = bd.len();
        let m = ad[ra-2]; let k = ad[ra-1]; let n = bd[rb-1];
        let ab: usize = ad[..ra-2].iter().product();
        let bb: usize = bd[..rb-2].iter().product();
        let batch = ab.max(bb);

        if batch <= 1 {
            let ap = self.get_or_upload(&a)?;
            let bp = self.get_or_upload(&b)?;
            let c = self.rocblas_gemm(ap, bp, m, k, n)?;
            let data = c.download_on_stream(self.stream).map_err(candle_core::Error::Msg)?;
            let mut s = ad[..ra-2].to_vec(); s.push(m); s.push(n);
            return Tensor::from_vec(data, s.as_slice(), &Device::Cpu);
        }

        let ad_flat = Self::to_f32_vec(&a)?;
        let bd_flat = Self::to_f32_vec(&b)?;
        let (mk, kn, mn) = (m*k, k*n, m*n);
        // Upload entire A and B arrays at once — one H2D transfer per operand
        let a_gpu = self.gpu_upload(&ad_flat)?;
        let b_gpu = self.gpu_upload(&bd_flat)?;
        let c_gpu = self.gpu_alloc(batch * mn)?;
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // Use strided batched GEMM when both operands have uniform strides
        if ab == batch && bb == batch {
            let err = unsafe {
                (self.ffi.rocblas_sgemm_strided_batched)(
                    self.blas_handle,
                    ROCBLAS_OPERATION_NONE, ROCBLAS_OPERATION_NONE,
                    n as i32, m as i32, k as i32,
                    &alpha,
                    b_gpu.as_ptr(), n as i32, kn as i64,
                    a_gpu.as_ptr(), k as i32, mk as i64,
                    &beta,
                    c_gpu.as_mut_ptr(), n as i32, mn as i64,
                    batch as i32,
                )
            };
            if err != 0 { return Err(candle_core::Error::Msg(format!("rocblas_sgemm_strided_batched: {err}"))); }
        } else {
            // Fallback: loop for broadcast batches (one operand has batch=1)
            for i in 0..batch {
                let ao = if ab==1 {0} else {i*mk};
                let bo = if bb==1 {0} else {i*kn};
                let co = i * mn;
                let err = unsafe {
                    (self.ffi.rocblas_sgemm)(
                        self.blas_handle,
                        ROCBLAS_OPERATION_NONE, ROCBLAS_OPERATION_NONE,
                        n as i32, m as i32, k as i32,
                        &alpha,
                        b_gpu.as_ptr().add(bo), n as i32,
                        a_gpu.as_ptr().add(ao), k as i32,
                        &beta,
                        c_gpu.as_mut_ptr().add(co), n as i32,
                    )
                };
                if err != 0 { return Err(candle_core::Error::Msg(format!("rocblas_sgemm batch {i}: {err}"))); }
            }
        }
        let out = c_gpu.download_on_stream(self.stream).map_err(candle_core::Error::Msg)?;
        let mut s = ad[..ra-2].to_vec(); s.push(m); s.push(n);
        Tensor::from_vec(out, s.as_slice(), &Device::Cpu)
    }
}

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str { "rocm" }
    fn device(&self) -> &Device { &self.device }

    fn preprocess_linear_weight(&self, weight: &Tensor) -> Result<Tensor> {
        // Pre-ensure contiguous layout at load time to avoid runtime copies.
        // Don't change dtype — linear_forward expects matching dtypes with input.
        if weight.is_contiguous() { Ok(weight.clone()) } else { weight.contiguous() }
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let orig = a.dtype();
        let m = a.dims()[a.dims().len() - 2];
        if m <= 4 {
            // CPU fallback — ensure matching dtypes (weight may be pre-converted to F32)
            return if a.dtype() != b.dtype() {
                a.to_dtype(DType::F32)?.matmul(&b.to_dtype(DType::F32)?)?.to_dtype(orig)
            } else {
                a.matmul(b)
            };
        }
        let out = self.tensor_matmul(a, b)?;
        if orig == DType::F32 { Ok(out) } else { out.to_dtype(orig) }
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let orig = q.dtype();
        let (qf, kf, vf);
        let (q, k, v) = if orig == DType::F32 { (q, k, v) }
            else { qf = q.to_dtype(DType::F32)?; kf = k.to_dtype(DType::F32)?; vf = v.to_dtype(DType::F32)?; (&qf, &kf, &vf) };
        let kt = k.t()?;
        let attn = (self.tensor_matmul(q, &kt)? * scale as f64)?;
        let attn = if causal {
            let (sl, kl) = (q.dim(2)?, k.dim(2)?);
            let mut mask = vec![0u8; sl*kl];
            for i in 0..sl { for j in 0..=kl.saturating_sub(sl)+i { if j<kl { mask[i*kl+j]=1; } } }
            let m = Tensor::from_vec(mask, (1,1,sl,kl), q.device())?;
            let ni = Tensor::full(f32::NEG_INFINITY, attn.shape(), q.device())?;
            m.broadcast_as(attn.shape())?.where_cond(&attn, &ni)?
        } else { attn };
        self.tensor_matmul(&candle_nn::ops::softmax_last_dim(&attn)?, v)?.to_dtype(orig)
    }

    fn silu_mul(&self, g: &Tensor, u: &Tensor) -> Result<Tensor> { (candle_nn::ops::silu(&g.contiguous()?)? * u.contiguous()?)?.contiguous() }
    fn stable_softplus(&self, x: &Tensor) -> Result<Tensor> { let t=Tensor::full(88f32,x.shape(),x.device())?.to_dtype(x.dtype())?; x.maximum(&(x.minimum(&t)?.exp()?+1.0)?.log()?) }
    fn rms_norm_gated(&self, x: &Tensor, z: &Tensor, w: &Tensor, e: f32) -> Result<Tensor> { (candle_nn::ops::rms_norm(&x.contiguous()?,w,e)?*candle_nn::ops::silu(&z.contiguous()?.to_dtype(x.dtype())?)?)?.contiguous() }
    fn add_rms_norm(&self, a: &Tensor, b: &Tensor, w: &Tensor, e: f32) -> Result<(Tensor,Tensor)> { let r=(a+b)?; Ok((r.clone(),candle_nn::ops::rms_norm(&r.contiguous()?,w,e)?)) }
    fn rms_norm_channel(&self, x: &Tensor, w: &Tensor, e: f32) -> Result<Tensor> { x.transpose(1,2)?.contiguous().and_then(|t|candle_nn::ops::rms_norm(&t,w,e))?.transpose(1,2)?.contiguous() }
    fn depthwise_conv1d_silu(&self, win: &Tensor, w: &Tensor, _: usize, _: usize) -> Result<Tensor> { candle_nn::ops::silu(&win.broadcast_mul(&w.unsqueeze(0)?)?.sum(D::Minus1)?) }
    fn depthwise_conv1d_bias(&self, i: &Tensor, w: &Tensor, b: &Tensor, ks: usize, _: usize) -> Result<Tensor> { let ot=i.dim(2)?-ks+1; let mut s=Vec::with_capacity(ot); for t in 0..ot{s.push(i.narrow(2,t,ks)?.broadcast_mul(&w.unsqueeze(0)?)?.sum(D::Minus1)?.broadcast_add(b)?.unsqueeze(2)?);} Tensor::cat(&s,2) }
    fn depthwise_conv1d_bias_ctx(&self, c: &Tensor, i: &Tensor, w: &Tensor, b: &Tensor, ks: usize, ch: usize) -> Result<Tensor> { self.depthwise_conv1d_bias(&Tensor::cat(&[c,i],2)?,w,b,ks,ch) }
    fn add3(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> { ((a+b)?+c)?.contiguous() }
    fn exp_mul(&self, x: &Tensor, y: &Tensor) -> Result<Tensor> { (x*y.exp()?)?.contiguous() }
    fn sub_mul(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> { ((a-b)?*c)?.contiguous() }
    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> { let cb = if c.dims().len()==1 { c.unsqueeze(0)?.unsqueeze(2)? } else { c.clone() }; (a+b.broadcast_mul(&cb)?)?.contiguous() }
    fn adaln_modulate(&self, x: &Tensor, nw: &Tensor, sc: &Tensor, sh: &Tensor, e: f32) -> Result<Tensor> { (candle_nn::ops::rms_norm(&x.contiguous()?,nw,e)?.broadcast_mul(&(sc+1.0)?)?+sh)?.contiguous() }
    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32) }
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F16) }
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::BF16) }
}
