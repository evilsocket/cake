//! Minimal raw FFI bindings for HIP + rocBLAS.
//!
//! Links against `libamdhip64.so` and `librocblas.so` at runtime.
//! Only the functions actually used by RocmBackend are declared here.

use std::ffi::c_void;

// hipMemcpyKind enum values
pub const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: i32 = 2;

// rocblas_operation enum values
pub const ROCBLAS_OPERATION_NONE: i32 = 111;

#[link(name = "amdhip64")]
extern "C" {
    pub fn hipInit(flags: u32) -> i32;
    pub fn hipGetDeviceCount(count: *mut i32) -> i32;
    pub fn hipSetDevice(device: i32) -> i32;
    pub fn hipDeviceGetName(name: *mut i8, len: i32, device: i32) -> i32;
    pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
    pub fn hipFree(ptr: *mut c_void) -> i32;
    pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
    pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    pub fn hipDeviceSynchronize() -> i32;
}

#[link(name = "rocblas")]
extern "C" {
    pub fn rocblas_create_handle(handle: *mut *mut c_void) -> i32;
    pub fn rocblas_destroy_handle(handle: *mut c_void) -> i32;
    pub fn rocblas_sgemm(
        handle: *mut c_void,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: *const f32,
        c: *mut f32,
        ldc: i32,
    ) -> i32;
}
