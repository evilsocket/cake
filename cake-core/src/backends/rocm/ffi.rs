//! Runtime-loaded FFI bindings for HIP + rocBLAS via dlopen.
//!
//! Loads libamdhip64.so and librocblas.so at runtime instead of link-time,
//! allowing use of specific ROCm versions (e.g., 5.4.2 on Steam Deck)
//! independent of what's in the system library path.

use std::ffi::c_void;
use std::os::raw::c_int;

/// Loaded HIP + rocBLAS function pointers.
#[allow(dead_code)]
pub struct RocmFfi {
    _hip_lib: libloading::Library,
    _blas_lib: libloading::Library,

    pub hip_init: unsafe extern "C" fn(u32) -> c_int,
    pub hip_get_device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
    pub hip_set_device: unsafe extern "C" fn(c_int) -> c_int,
    pub hip_device_get_name: unsafe extern "C" fn(*mut i8, c_int, c_int) -> c_int,
    pub hip_malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
    pub hip_free: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub hip_memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
    pub hip_mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> c_int,
    pub hip_device_synchronize: unsafe extern "C" fn() -> c_int,
    pub hip_stream_create: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    pub hip_stream_synchronize: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub hip_memcpy_async: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int, *mut c_void) -> c_int,

    pub rocblas_create_handle: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    pub rocblas_destroy_handle: unsafe extern "C" fn(*mut c_void) -> c_int,
    pub rocblas_set_stream: unsafe extern "C" fn(*mut c_void, *mut c_void) -> c_int,
    pub rocblas_sgemm: unsafe extern "C" fn(
        *mut c_void, c_int, c_int,
        c_int, c_int, c_int,
        *const f32, *const f32, c_int,
        *const f32, c_int,
        *const f32, *mut f32, c_int,
    ) -> c_int,
}

// dlopen'd libraries are safe to share across threads
unsafe impl Send for RocmFfi {}
unsafe impl Sync for RocmFfi {}

pub const HIP_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const HIP_MEMCPY_DEVICE_TO_HOST: c_int = 2;
pub const ROCBLAS_OPERATION_NONE: c_int = 111;

impl RocmFfi {
    /// Load HIP and rocBLAS from the library path.
    /// Searches LD_LIBRARY_PATH, then default system paths.
    #[allow(clippy::missing_transmute_annotations)]
    pub fn load() -> std::result::Result<Self, String> {
        unsafe {
            let hip_lib = libloading::Library::new("libamdhip64.so")
                .or_else(|_| libloading::Library::new("libamdhip64.so.5"))
                .map_err(|e| format!("cannot load libamdhip64: {e}"))?;

            let blas_lib = libloading::Library::new("librocblas.so")
                .or_else(|_| libloading::Library::new("librocblas.so.0"))
                .map_err(|e| format!("cannot load librocblas: {e}"))?;

            macro_rules! sym {
                ($lib:expr, $name:expr) => {{
                    let raw: libloading::Symbol<*const std::ffi::c_void> = $lib.get($name)
                        .map_err(|e| format!("symbol {}: {e}", String::from_utf8_lossy($name)))?;
                    std::mem::transmute(*raw)
                }};
            }

            Ok(Self {
                hip_init: sym!(hip_lib, b"hipInit\0"),
                hip_get_device_count: sym!(hip_lib, b"hipGetDeviceCount\0"),
                hip_set_device: sym!(hip_lib, b"hipSetDevice\0"),
                hip_device_get_name: sym!(hip_lib, b"hipDeviceGetName\0"),
                hip_malloc: sym!(hip_lib, b"hipMalloc\0"),
                hip_free: sym!(hip_lib, b"hipFree\0"),
                hip_memcpy: sym!(hip_lib, b"hipMemcpy\0"),
                hip_mem_get_info: sym!(hip_lib, b"hipMemGetInfo\0"),
                hip_device_synchronize: sym!(hip_lib, b"hipDeviceSynchronize\0"),
                hip_stream_create: sym!(hip_lib, b"hipStreamCreate\0"),
                hip_stream_synchronize: sym!(hip_lib, b"hipStreamSynchronize\0"),
                hip_memcpy_async: sym!(hip_lib, b"hipMemcpyAsync\0"),

                rocblas_create_handle: sym!(blas_lib, b"rocblas_create_handle\0"),
                rocblas_destroy_handle: sym!(blas_lib, b"rocblas_destroy_handle\0"),
                rocblas_set_stream: sym!(blas_lib, b"rocblas_set_stream\0"),
                rocblas_sgemm: sym!(blas_lib, b"rocblas_sgemm\0"),

                _hip_lib: hip_lib,
                _blas_lib: blas_lib,
            })
        }
    }
}
