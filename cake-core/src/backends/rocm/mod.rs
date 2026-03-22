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
use std::sync::Mutex;

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
    fn as_ptr(&self) -> *const f32 { self.ptr as *const f32 }
    fn as_mut_ptr(&self) -> *mut f32 { self.ptr as *mut f32 }

    fn download(&self) -> std::result::Result<Vec<f32>, String> {
        let mut host = vec![0f32; self.count];
        let err = unsafe {
            ((*self.ffi).hip_memcpy)(
                host.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                self.count * 4,
                HIP_MEMCPY_DEVICE_TO_HOST,
            )
        };
        if err != 0 { return Err(format!("hipMemcpy D2H: error {err}")); }
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
    cache: Mutex<HashMap<TensorId, GpuBuf>>,
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

        Ok(Self {
            device: Device::Cpu,
            ffi,
            blas_handle: handle,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn gpu_alloc(&self, count: usize) -> Result<GpuBuf> {
        let mut ptr = std::ptr::null_mut();
        let err = unsafe { (self.ffi.hip_malloc)(&mut ptr, count * 4) };
        if err != 0 { return Err(candle_core::Error::Msg(format!("hipMalloc: {err}"))); }
        Ok(GpuBuf { ptr, count, ffi: &*self.ffi as *const RocmFfi })
    }

    fn gpu_upload(&self, data: &[f32]) -> Result<GpuBuf> {
        let buf = self.gpu_alloc(data.len())?;
        let err = unsafe {
            (self.ffi.hip_memcpy)(buf.ptr, data.as_ptr() as *const _, data.len() * 4, HIP_MEMCPY_HOST_TO_DEVICE)
        };
        if err != 0 { return Err(candle_core::Error::Msg(format!("hipMemcpy H2D: {err}"))); }
        Ok(buf)
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
        let buf = self.gpu_upload(&data)?;
        let ptr = buf.as_ptr();
        cache.insert(id, buf);
        Ok(ptr)
    }

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
        let err = unsafe { (self.ffi.hip_device_synchronize)() };
        if err != 0 { return Err(candle_core::Error::Msg(format!("hipSync: {err}"))); }
        Ok(c)
    }

    fn tensor_matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a = a.to_dtype(DType::F32)?.contiguous()?;
        let b = b.to_dtype(DType::F32)?.contiguous()?;
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
            let data = c.download().map_err(candle_core::Error::Msg)?;
            let mut s = ad[..ra-2].to_vec(); s.push(m); s.push(n);
            return Tensor::from_vec(data, s.as_slice(), &Device::Cpu);
        }

        let ad_flat = Self::to_f32_vec(&a)?;
        let bd_flat = Self::to_f32_vec(&b)?;
        let (mk, kn, mn) = (m*k, k*n, m*n);
        let mut out = Vec::with_capacity(batch * mn);
        for i in 0..batch {
            let ao = if ab==1 {0} else {i*mk};
            let bo = if bb==1 {0} else {i*kn};
            let ab = self.gpu_upload(&ad_flat[ao..ao+mk])?;
            let bb = self.gpu_upload(&bd_flat[bo..bo+kn])?;
            let c = self.rocblas_gemm(ab.as_ptr(), bb.as_ptr(), m, k, n)?;
            out.extend_from_slice(&c.download().map_err(candle_core::Error::Msg)?);
        }
        let mut s = ad[..ra-2].to_vec(); s.push(m); s.push(n);
        Tensor::from_vec(out, s.as_slice(), &Device::Cpu)
    }
}

impl ComputeBackend for RocmBackend {
    fn name(&self) -> &str { "rocm" }
    fn device(&self) -> &Device { &self.device }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let m = a.dims()[a.dims().len() - 2];
        if m <= 4 { return a.matmul(b); }
        self.tensor_matmul(a, b)?.to_dtype(a.dtype())
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
        let orig = q.dtype();
        let q = q.to_dtype(DType::F32)?; let k = k.to_dtype(DType::F32)?; let v = v.to_dtype(DType::F32)?;
        let attn = self.tensor_matmul(&q, &k.t()?)?;
        let attn = (attn * scale as f64)?;
        let attn = if causal {
            let (sl, kl) = (q.dim(2)?, k.dim(2)?);
            let mut mask = vec![0u8; sl*kl];
            for i in 0..sl { for j in 0..=kl.saturating_sub(sl)+i { if j<kl { mask[i*kl+j]=1; } } }
            let m = Tensor::from_vec(mask, (1,1,sl,kl), q.device())?;
            let ni = Tensor::full(f32::NEG_INFINITY, attn.shape(), q.device())?;
            m.broadcast_as(attn.shape())?.where_cond(&attn, &ni)?
        } else { attn };
        self.tensor_matmul(&candle_nn::ops::softmax_last_dim(&attn)?, &v)?.to_dtype(orig)
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
    fn add_scaled(&self, a: &Tensor, b: &Tensor, c: &Tensor) -> Result<Tensor> { (a+b.broadcast_mul(&c.unsqueeze(0)?.unsqueeze(2)?)?)?.contiguous() }
    fn adaln_modulate(&self, x: &Tensor, nw: &Tensor, sc: &Tensor, sh: &Tensor, e: f32) -> Result<Tensor> { (candle_nn::ops::rms_norm(&x.contiguous()?,nw,e)?.broadcast_mul(&(sc+1.0)?)?+sh)?.contiguous() }
    fn f8e4m3_to_f32(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F32) }
    fn f8e4m3_to_f16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::F16) }
    fn f8e4m3_to_bf16(&self, x: &Tensor) -> Result<Tensor> { x.to_dtype(DType::BF16) }
}
