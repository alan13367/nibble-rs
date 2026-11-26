//! Minimal, runtime-loaded CUDA Driver API.
//!
//! Only `cuInit`, `cuGetErrorString` and `cuGetProcAddress` are looked up in
//! the driver library itself. Every other entry point is resolved for CUDA
//! 12.0 or newer through `cuGetProcAddress`, so versioned symbols (the `_v2`
//! memory functions) have the intended ABI. The library is
//! unloaded only after the last object using its function table has dropped.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Arc;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlclose(handle: *mut c_void) -> c_int;
    fn dlerror() -> *const c_char;
}

#[cfg(target_os = "windows")]
#[link(name = "kernel32")]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn FreeLibrary(handle: *mut c_void) -> i32;
}

/// The CUDA version passed to `cuGetProcAddress`.
const CUDA_VERSION: c_int = 12_000;
const CU_GET_PROC_ADDRESS_DEFAULT: u64 = 0;
const CUDA_SUCCESS: c_int = 0;

pub(super) type RawDevice = c_int;
pub(super) type DevicePtr = u64;
pub(super) type RawContext = *mut c_void;
pub(super) type RawModule = *mut c_void;
pub(super) type RawFunction = *mut c_void;
pub(super) type RawStream = *mut c_void;
type ResultCode = c_int;

type Init = unsafe extern "system" fn(u32) -> ResultCode;
type GetErrorString = unsafe extern "system" fn(ResultCode, *mut *const c_char) -> ResultCode;
type GetProcAddress = unsafe extern "system" fn(
    *const c_char,
    *mut *mut c_void,
    c_int,
    u64,
    *mut c_int,
) -> ResultCode;
type DeviceGet = unsafe extern "system" fn(*mut RawDevice, c_int) -> ResultCode;
type DeviceGetName = unsafe extern "system" fn(*mut c_char, c_int, RawDevice) -> ResultCode;
type DeviceGetAttribute = unsafe extern "system" fn(*mut c_int, c_int, RawDevice) -> ResultCode;
type DevicePrimaryCtxRetain = unsafe extern "system" fn(*mut RawContext, RawDevice) -> ResultCode;
type DevicePrimaryCtxRelease = unsafe extern "system" fn(RawDevice) -> ResultCode;
type CtxSetCurrent = unsafe extern "system" fn(RawContext) -> ResultCode;
type CtxSynchronize = unsafe extern "system" fn() -> ResultCode;
type MemAlloc = unsafe extern "system" fn(*mut DevicePtr, usize) -> ResultCode;
type MemFree = unsafe extern "system" fn(DevicePtr) -> ResultCode;
type MemsetD8 = unsafe extern "system" fn(DevicePtr, u8, usize) -> ResultCode;
type MemcpyHtoD = unsafe extern "system" fn(DevicePtr, *const c_void, usize) -> ResultCode;
type MemcpyDtoH = unsafe extern "system" fn(*mut c_void, DevicePtr, usize) -> ResultCode;
type MemcpyHtoDAsync =
    unsafe extern "system" fn(DevicePtr, *const c_void, usize, RawStream) -> ResultCode;
type ModuleLoadDataEx = unsafe extern "system" fn(
    *mut RawModule,
    *const c_void,
    u32,
    *mut c_int,
    *mut *mut c_void,
) -> ResultCode;
type ModuleUnload = unsafe extern "system" fn(RawModule) -> ResultCode;
type ModuleGetFunction =
    unsafe extern "system" fn(*mut RawFunction, RawModule, *const c_char) -> ResultCode;
type LaunchKernel = unsafe extern "system" fn(
    RawFunction,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    u32,
    RawStream,
    *mut *mut c_void,
    *mut *mut c_void,
) -> ResultCode;
type StreamCreate = unsafe extern "system" fn(*mut RawStream, u32) -> ResultCode;
type StreamDestroy = unsafe extern "system" fn(RawStream) -> ResultCode;
type StreamSynchronize = unsafe extern "system" fn(RawStream) -> ResultCode;

/// A loaded CUDA driver and its immutable function table.
#[derive(Clone)]
pub(crate) struct Api {
    pub(super) inner: Arc<ApiInner>,
}

impl Api {
    /// Loads the system CUDA driver and initializes it.
    ///
    /// No CUDA library is linked into the executable. An error here is an
    /// ordinary backend-unavailable result, not a process failure.
    pub(crate) fn load() -> Result<Self, String> {
        let inner = ApiInner::load()?;
        // SAFETY: `cuInit` was resolved from this live driver; zero flags are
        // required and it has no pointer arguments.
        let code = unsafe { (inner.init)(0) };
        inner.check(code, "cuInit")?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
}

pub(super) struct ApiInner {
    /// Kept first conceptually, and in any case until this whole table drops.
    _library: Library,
    init: Init,
    get_error_string: Option<GetErrorString>,
    pub(super) device_get: DeviceGet,
    pub(super) device_get_name: DeviceGetName,
    pub(super) device_get_attribute: DeviceGetAttribute,
    pub(super) device_primary_ctx_retain: DevicePrimaryCtxRetain,
    pub(super) device_primary_ctx_release: DevicePrimaryCtxRelease,
    pub(super) ctx_set_current: CtxSetCurrent,
    pub(super) ctx_synchronize: CtxSynchronize,
    pub(super) mem_alloc: MemAlloc,
    pub(super) mem_free: MemFree,
    pub(super) memset_d8: MemsetD8,
    pub(super) memcpy_htod: MemcpyHtoD,
    pub(super) memcpy_dtoh: MemcpyDtoH,
    pub(super) memcpy_htod_async: MemcpyHtoDAsync,
    pub(super) module_load_data_ex: ModuleLoadDataEx,
    pub(super) module_unload: ModuleUnload,
    pub(super) module_get_function: ModuleGetFunction,
    pub(super) launch_kernel: LaunchKernel,
    pub(super) stream_create: StreamCreate,
    pub(super) stream_destroy: StreamDestroy,
    pub(super) stream_synchronize: StreamSynchronize,
}

// SAFETY: the library handle and resolved functions are immutable. CUDA
// initialization and driver entry points are process-wide and thread-safe;
// the library remains loaded until this value is dropped.
unsafe impl Send for ApiInner {}
// SAFETY: as above; sharing the table does not share a mutable CUDA object.
unsafe impl Sync for ApiInner {}

impl ApiInner {
    fn load() -> Result<Self, String> {
        let library = Library::open_cuda()?;
        // SAFETY: these three symbols have exactly the declared CUDA driver
        // prototypes. All remaining symbols go through `cuGetProcAddress`.
        let (init, get_error_string, get_proc_address) = unsafe {
            (
                library.function::<Init>(c"cuInit")?,
                library.function::<GetErrorString>(c"cuGetErrorString").ok(),
                library.function::<GetProcAddress>(c"cuGetProcAddress")?,
            )
        };
        // SAFETY: each requested symbol is cast only to its prototype in
        // cuda.h. `library` stays alive throughout and then moves into `api`.
        let api = unsafe {
            Self {
                _library: library,
                init,
                get_error_string,
                device_get: resolve(get_proc_address, get_error_string, c"cuDeviceGet")?,
                device_get_name: resolve(get_proc_address, get_error_string, c"cuDeviceGetName")?,
                device_get_attribute: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuDeviceGetAttribute",
                )?,
                device_primary_ctx_retain: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuDevicePrimaryCtxRetain",
                )?,
                device_primary_ctx_release: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuDevicePrimaryCtxRelease",
                )?,
                ctx_set_current: resolve(get_proc_address, get_error_string, c"cuCtxSetCurrent")?,
                ctx_synchronize: resolve(get_proc_address, get_error_string, c"cuCtxSynchronize")?,
                mem_alloc: resolve(get_proc_address, get_error_string, c"cuMemAlloc")?,
                mem_free: resolve(get_proc_address, get_error_string, c"cuMemFree")?,
                memset_d8: resolve(get_proc_address, get_error_string, c"cuMemsetD8")?,
                memcpy_htod: resolve(get_proc_address, get_error_string, c"cuMemcpyHtoD")?,
                memcpy_dtoh: resolve(get_proc_address, get_error_string, c"cuMemcpyDtoH")?,
                memcpy_htod_async: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuMemcpyHtoDAsync",
                )?,
                module_load_data_ex: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuModuleLoadDataEx",
                )?,
                module_unload: resolve(get_proc_address, get_error_string, c"cuModuleUnload")?,
                module_get_function: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuModuleGetFunction",
                )?,
                launch_kernel: resolve(get_proc_address, get_error_string, c"cuLaunchKernel")?,
                stream_create: resolve(get_proc_address, get_error_string, c"cuStreamCreate")?,
                stream_destroy: resolve(get_proc_address, get_error_string, c"cuStreamDestroy")?,
                stream_synchronize: resolve(
                    get_proc_address,
                    get_error_string,
                    c"cuStreamSynchronize",
                )?,
            }
        };
        Ok(api)
    }

    pub(super) fn check(&self, code: ResultCode, operation: &str) -> Result<(), String> {
        if code == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(self.error(code, operation))
        }
    }

    fn error(&self, code: ResultCode, operation: &str) -> String {
        driver_error(self.get_error_string, code, operation)
    }
}

/// Resolves `name` as `F` through the versioned driver entry point.
///
/// # Safety
///
/// `F` must be the exact function pointer type from the CUDA header, and both
/// direct functions must belong to a driver library that outlives the result.
unsafe fn resolve<F: Copy>(
    get_proc_address: GetProcAddress,
    get_error_string: Option<GetErrorString>,
    name: &CStr,
) -> Result<F, String> {
    if size_of::<F>() != size_of::<*mut c_void>() {
        return Err(format!(
            "internal CUDA prototype for {} is not a function pointer",
            name.to_string_lossy()
        ));
    }
    let mut ptr = std::ptr::null_mut();
    // SAFETY: `name` is NUL-terminated, `ptr` is a valid out-pointer, and the
    // optional status pointer may be null. The driver library is live.
    let code = unsafe {
        get_proc_address(
            name.as_ptr(),
            &mut ptr,
            CUDA_VERSION,
            CU_GET_PROC_ADDRESS_DEFAULT,
            std::ptr::null_mut(),
        )
    };
    if code != CUDA_SUCCESS {
        return Err(driver_error(
            get_error_string,
            code,
            &format!("cuGetProcAddress({})", name.to_string_lossy()),
        ));
    }
    if ptr.is_null() {
        return Err(format!(
            "CUDA driver returned no address for {}",
            name.to_string_lossy()
        ));
    }
    // SAFETY: the size was checked above, and the caller guarantees that `F`
    // is this symbol's prototype. The returned address was checked non-null.
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&ptr) })
}

fn driver_error(
    get_error_string: Option<GetErrorString>,
    code: ResultCode,
    operation: &str,
) -> String {
    let mut text = std::ptr::null();
    if let Some(get_error_string) = get_error_string {
        // SAFETY: the optional function came from the live driver and `text`
        // is a valid out-pointer.
        let result = unsafe { get_error_string(code, &mut text) };
        if result == CUDA_SUCCESS && !text.is_null() {
            // SAFETY: on success CUDA returns a static NUL-terminated
            // description, copied before any further driver call.
            let description = unsafe { CStr::from_ptr(text) }.to_string_lossy();
            return format!("{operation}: CUDA error {code}: {description}");
        }
    }
    format!("{operation}: CUDA error {code}")
}

struct Library {
    handle: *mut c_void,
}

impl Library {
    fn open_cuda() -> Result<Self, String> {
        #[cfg(target_os = "windows")]
        {
            Self::open(c"nvcuda.dll")
        }
        #[cfg(target_os = "linux")]
        {
            let first = match Self::open(c"libcuda.so.1") {
                Ok(library) => return Ok(library),
                Err(error) => error,
            };
            Self::open(c"libcuda.so")
                .map_err(|second| format!("cannot load the CUDA driver: {first}; {second}"))
        }
        #[cfg(not(any(target_os = "windows", target_os = "linux")))]
        {
            Err("the CUDA backend supports only Windows and Linux".to_owned())
        }
    }

    #[cfg(target_os = "windows")]
    fn open(path: &CStr) -> Result<Self, String> {
        use std::os::windows::ffi::OsStrExt;

        let wide: Vec<u16> = std::ffi::OsStr::new(path.to_string_lossy().as_ref())
            .encode_wide()
            .chain(Some(0))
            .collect();
        // SAFETY: `wide` is NUL-terminated and lives for the call. Windows
        // owns the returned module handle until `FreeLibrary`.
        let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
        if handle.is_null() {
            return Err(format!(
                "cannot load {}: {}",
                path.to_string_lossy(),
                std::io::Error::last_os_error()
            ));
        }
        Ok(Self { handle })
    }

    #[cfg(target_os = "linux")]
    fn open(path: &CStr) -> Result<Self, String> {
        const RTLD_NOW: c_int = 2;
        // SAFETY: `path` is NUL-terminated. Loading the system driver runs its
        // initializers and returns a handle owned by this wrapper.
        let handle = unsafe { dlopen(path.as_ptr(), RTLD_NOW) };
        if handle.is_null() {
            return Err(format!(
                "cannot load {}: {}",
                path.to_string_lossy(),
                last_dl_error()
            ));
        }
        Ok(Self { handle })
    }

    /// Looks up a direct symbol as its exact function pointer type.
    ///
    /// # Safety
    ///
    /// `F` must be the symbol's C prototype.
    unsafe fn function<F: Copy>(&self, name: &CStr) -> Result<F, String> {
        if size_of::<F>() != size_of::<*mut c_void>() {
            return Err(format!(
                "internal prototype for {} is not a function pointer",
                name.to_string_lossy()
            ));
        }
        #[cfg(target_os = "windows")]
        let ptr = {
            // SAFETY: the module handle is live and `name` is NUL-terminated.
            unsafe { GetProcAddress(self.handle, name.as_ptr()) }
        };
        #[cfg(target_os = "linux")]
        let ptr = {
            // SAFETY: the module handle is live and `name` is NUL-terminated.
            unsafe { dlsym(self.handle, name.as_ptr()) }
        };
        if ptr.is_null() {
            return Err(format!("CUDA driver is missing {}", name.to_string_lossy()));
        }
        // SAFETY: the size was checked and the caller guarantees `F` is the
        // exact prototype of this symbol.
        Ok(unsafe { std::mem::transmute_copy::<*mut c_void, F>(&ptr) })
    }
}

impl Drop for Library {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        {
            // SAFETY: this is the live handle returned by `LoadLibraryW`, and
            // no resolved function can still be called after the table drops.
            let _ = unsafe { FreeLibrary(self.handle) };
        }
        #[cfg(target_os = "linux")]
        {
            // SAFETY: this is the live handle returned by `dlopen`, released
            // exactly once after all users of its functions have dropped.
            let _ = unsafe { dlclose(self.handle) };
        }
    }
}

#[cfg(target_os = "linux")]
fn last_dl_error() -> String {
    // SAFETY: `dlerror` returns null or a thread-local NUL-terminated string;
    // it is copied before another dynamic-loader call on this thread.
    let error = unsafe { dlerror() };
    if error.is_null() {
        "unknown dynamic-loader error".to_owned()
    } else {
        // SAFETY: non-null and NUL-terminated as described above.
        unsafe { CStr::from_ptr(error) }
            .to_string_lossy()
            .into_owned()
    }
}
