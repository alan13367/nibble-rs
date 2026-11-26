//! RAII wrappers for a device's primary context and the CUDA objects a
//! backend needs.
//!
//! Every object keeps an `Arc` to the retained primary context, which in turn
//! keeps the driver function table and library alive. Operations make that
//! context current on the calling thread, so moving an object to a worker is
//! supported. Owned handles are `Send` but deliberately not `Sync`: work on a
//! stream, and destruction or replacement of its buffers, needs external
//! ordering. Synchronous buffer copies wait until they land but are not
//! ordered after work already queued on a stream; asynchronous copies place
//! their host-memory requirements on the caller.

use std::cell::Cell;
use std::ffi::{CString, c_void};
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;

use super::driver::{
    Api, ApiInner, DevicePtr, RawContext, RawDevice, RawFunction, RawModule, RawStream,
};

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;
const CU_STREAM_NON_BLOCKING: u32 = 1;

/// Makes wrappers movable without permitting shared access through `&T` on
/// several threads.
type NotSync = PhantomData<Cell<()>>;

struct Context {
    api: Arc<ApiInner>,
    device: RawDevice,
    raw: RawContext,
}

// SAFETY: a primary context may be retained, made current and released from
// any thread. All operations on this internal value are immutable driver
// calls; public wrappers serialize access to objects belonging to it.
unsafe impl Send for Context {}
// SAFETY: as above. The `Arc` is shared only to preserve the context lifetime
// and to make it current, both thread-safe driver operations.
unsafe impl Sync for Context {}

impl Context {
    fn activate(&self) -> Result<(), String> {
        // SAFETY: `raw` is this retained primary context, alive for `self`.
        let code = unsafe { (self.api.ctx_set_current)(self.raw) };
        self.api.check(code, "cuCtxSetCurrent")
    }

    /// Waits for all work in the context, on every stream.
    fn synchronize(&self) -> Result<(), String> {
        self.activate()?;
        // SAFETY: the context is current and the call takes no arguments.
        let code = unsafe { (self.api.ctx_synchronize)() };
        self.api.check(code, "cuCtxSynchronize")
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: this Arc owns exactly one primary-context retain on `device`.
        // Every CUDA object that needs the context also owns an Arc, so this
        // release runs only after those objects have been destroyed.
        let code = unsafe { (self.api.device_primary_ctx_release)(self.device) };
        let _ = code;
    }
}

/// A CUDA device and its retained primary context.
pub(crate) struct Device {
    context: Arc<Context>,
    name: String,
    compute_capability: (u32, u32),
    _not_sync: NotSync,
}

// SAFETY: a device handle and primary context can move between threads. Each
// operation makes the context current, and `Device` is not `Sync`.
unsafe impl Send for Device {}

impl Device {
    /// Loads CUDA and opens the device at `index`, retaining its primary
    /// context. The driver and context lifetimes are owned by the returned
    /// object graph, so callers do not need to manage an `Api` separately.
    pub(crate) fn new(index: usize) -> Result<Self, String> {
        let api = Api::load()?;
        Self::with_api(&api, index)
    }

    fn with_api(api: &Api, index: usize) -> Result<Self, String> {
        let index = i32::try_from(index)
            .map_err(|_| format!("CUDA device index {index} does not fit in a C int"))?;
        let driver = &api.inner;
        let mut raw_device = 0;
        // SAFETY: `raw_device` is a valid out-pointer and `index` is the
        // requested device; CUDA reports whether it exists.
        let code = unsafe { (driver.device_get)(&mut raw_device, index) };
        driver.check(code, "cuDeviceGet")?;

        let mut raw_context = ptr::null_mut();
        // SAFETY: `raw_context` is a valid out-pointer and `raw_device` came
        // from `cuDeviceGet`. Success transfers one retain to `Context`.
        let code = unsafe { (driver.device_primary_ctx_retain)(&mut raw_context, raw_device) };
        driver.check(code, "cuDevicePrimaryCtxRetain")?;
        if raw_context.is_null() {
            // The retain succeeded, so balance it before reporting the bad
            // external result.
            // SAFETY: this balances the successful retain above.
            let _ = unsafe { (driver.device_primary_ctx_release)(raw_device) };
            return Err(format!(
                "CUDA returned a null primary context for device {index}"
            ));
        }
        let context = Arc::new(Context {
            api: Arc::clone(driver),
            device: raw_device,
            raw: raw_context,
        });
        context.activate()?;

        let mut name = [0_u8; 256];
        // SAFETY: `name` provides 256 writable bytes and `raw_device` is live.
        let code = unsafe {
            (driver.device_get_name)(name.as_mut_ptr().cast(), name.len() as i32, raw_device)
        };
        driver.check(code, "cuDeviceGetName")?;
        let name_len = name
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(name.len());
        let name = String::from_utf8_lossy(&name[..name_len]).into_owned();
        let major = device_attribute(&context, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = device_attribute(&context, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        Ok(Self {
            context,
            name,
            compute_capability: (major, minor),
            _not_sync: PhantomData,
        })
    }

    /// The device's display name.
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// `(major, minor)` compute capability, for example `(8, 9)`.
    pub(crate) fn compute_capability(&self) -> (u32, u32) {
        self.compute_capability
    }

    /// Allocates `len` zeroed bytes of device memory. A zero-length logical
    /// buffer still owns a one-byte CUDA allocation so its handle is valid.
    pub(crate) fn buffer(&self, len: usize) -> Result<Buffer, String> {
        let mut buffer = Buffer::new(&self.context, len)?;
        buffer.zero()?;
        Ok(buffer)
    }

    /// Allocates and synchronously uploads a device buffer.
    pub(crate) fn buffer_with(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let mut buffer = self.buffer(bytes.len())?;
        buffer.upload(0, bytes)?;
        Ok(buffer)
    }

    /// Loads an embedded fatbin, cubin or PTX image into this context.
    pub(crate) fn module(&self, image: &[u8]) -> Result<Module, String> {
        Module::new(&self.context, image)
    }

    /// Creates a non-blocking stream.
    pub(crate) fn stream(&self) -> Result<Stream, String> {
        Stream::new(&self.context)
    }
}

fn device_attribute(context: &Context, attribute: i32) -> Result<u32, String> {
    let mut value = 0;
    // SAFETY: `value` is a valid out-pointer, `attribute` is one of the two
    // compute-capability attributes, and the device handle is live.
    let code = unsafe { (context.api.device_get_attribute)(&mut value, attribute, context.device) };
    context.api.check(code, "cuDeviceGetAttribute")?;
    u32::try_from(value)
        .map_err(|_| format!("CUDA device attribute {attribute} is negative: {value}"))
}

/// An owned CUDA device allocation.
pub(crate) struct Buffer {
    context: Arc<Context>,
    raw: DevicePtr,
    len: usize,
    _not_sync: NotSync,
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("ptr", &format_args!("{:#x}", self.raw))
            .field("len", &self.len)
            .finish()
    }
}

// SAFETY: ownership of the allocation moves with the wrapper. CUDA memory
// functions are thread-safe and the context is made current for each call.
unsafe impl Send for Buffer {}

impl Buffer {
    fn new(context: &Arc<Context>, len: usize) -> Result<Self, String> {
        context.activate()?;
        let allocation_len = len.max(1);
        let mut raw = 0;
        // SAFETY: `raw` is a valid out-pointer, the context is current, and
        // the allocation size is at least one. Success transfers ownership.
        let code = unsafe { (context.api.mem_alloc)(&mut raw, allocation_len) };
        context.api.check(code, "cuMemAlloc")?;
        if raw == 0 {
            return Err(format!(
                "cuMemAlloc returned a null pointer for a {len}-byte buffer"
            ));
        }
        Ok(Self {
            context: Arc::clone(context),
            raw,
            len,
            _not_sync: PhantomData,
        })
    }

    /// The base CUDA device address, suitable for a kernel argument.
    pub(crate) fn ptr(&self) -> DevicePtr {
        self.raw
    }

    /// The device address `offset` bytes into this allocation.
    ///
    /// An offset equal to `len` is allowed as an end pointer. Arithmetic and
    /// bounds are checked rather than delegated to the driver.
    pub(crate) fn offset(&self, offset: usize) -> Result<DevicePtr, String> {
        self.checked_range(offset, 0)
    }

    fn checked_range(&self, offset: usize, len: usize) -> Result<DevicePtr, String> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| "CUDA buffer range overflows usize".to_owned())?;
        if end > self.len {
            return Err(format!(
                "CUDA buffer range {offset}..{end} exceeds its {} bytes",
                self.len
            ));
        }
        let offset = u64::try_from(offset)
            .map_err(|_| "CUDA buffer offset does not fit in CUdeviceptr".to_owned())?;
        self.raw
            .checked_add(offset)
            .ok_or_else(|| "CUDA device address arithmetic overflows u64".to_owned())
    }

    /// Sets every byte in the allocation to zero and waits until it is done.
    /// The caller must order this after earlier work using the buffer.
    pub(crate) fn zero(&mut self) -> Result<(), String> {
        if self.len == 0 {
            return Ok(());
        }
        self.context.activate()?;
        // SAFETY: `raw` is a live allocation of at least `len` bytes and the
        // context is current. The caller supplies stream ordering.
        let code = unsafe { (self.context.api.memset_d8)(self.raw, 0, self.len) };
        self.context.api.check(code, "cuMemsetD8")?;
        // The memset runs on the legacy default stream, which a non-blocking
        // stream does not wait for, and may still be running on return.
        self.context.synchronize()
    }

    /// Copies `bytes` to `offset` in the allocation and waits until it has
    /// landed. The caller must order this after earlier work using the buffer.
    pub(crate) fn upload(&mut self, offset: usize, bytes: &[u8]) -> Result<(), String> {
        let dst = self.checked_range(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        self.context.activate()?;
        // SAFETY: the destination range was checked, `bytes` is readable for
        // its exact length, and the context is current.
        let code = unsafe {
            (self.context.api.memcpy_htod)(dst, bytes.as_ptr().cast::<c_void>(), bytes.len())
        };
        self.context.api.check(code, "cuMemcpyHtoD")?;
        // From pageable memory the copy may return once the bytes are staged,
        // before the DMA reaches the device, and a non-blocking stream does
        // not wait for it.
        self.context.synchronize()
    }

    /// Copies exactly `out.len()` bytes from `offset`, synchronously with the
    /// host. The caller must order this against streams using the buffer.
    pub(crate) fn download(&self, offset: usize, out: &mut [u8]) -> Result<(), String> {
        let src = self.checked_range(offset, out.len())?;
        if out.is_empty() {
            return Ok(());
        }
        self.context.activate()?;
        // SAFETY: the source range was checked, `out` is writable for its
        // exact length, and the context is current.
        let code = unsafe {
            (self.context.api.memcpy_dtoh)(out.as_mut_ptr().cast::<c_void>(), src, out.len())
        };
        self.context.api.check(code, "cuMemcpyDtoH")
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.context.activate();
        // SAFETY: `raw` was allocated once by this wrapper in the retained
        // context, and this is its only `cuMemFree`. Drop cannot report a
        // driver error. The owner must have synchronized work using it.
        let code = unsafe { (self.context.api.mem_free)(self.raw) };
        let _ = code;
    }
}

/// A loaded CUDA module.
pub(crate) struct Module {
    inner: Arc<ModuleInner>,
    _not_sync: NotSync,
}

struct ModuleInner {
    context: Arc<Context>,
    raw: RawModule,
}

// SAFETY: module handles may move and are unloaded from any thread with the
// owning context current. The public `Function` contract requires its module
// to remain alive independently.
unsafe impl Send for ModuleInner {}
// SAFETY: sharing the internal lifetime token is safe; public module and
// function wrappers themselves are not `Sync`.
unsafe impl Sync for ModuleInner {}

// SAFETY: see `ModuleInner`; the public wrapper is movable but not shareable.
unsafe impl Send for Module {}

impl Module {
    fn new(context: &Arc<Context>, image: &[u8]) -> Result<Self, String> {
        if image.is_empty() {
            return Err("cannot load an empty CUDA module image".to_owned());
        }
        context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `image` is readable for the call, `raw` is a valid
        // out-pointer, zero JIT options are represented by null pointers, and
        // the context is current. CUDA consumes the image before returning.
        let code = unsafe {
            (context.api.module_load_data_ex)(
                &mut raw,
                image.as_ptr().cast::<c_void>(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
            )
        };
        context.api.check(code, "cuModuleLoadDataEx")?;
        if raw.is_null() {
            return Err("cuModuleLoadDataEx returned a null module".to_owned());
        }
        Ok(Self {
            inner: Arc::new(ModuleInner {
                context: Arc::clone(context),
                raw,
            }),
            _not_sync: PhantomData,
        })
    }

    /// Looks up a kernel by its unmangled CUDA name.
    pub(crate) fn function(&self, name: &str) -> Result<Function, String> {
        let name = CString::new(name)
            .map_err(|_| "CUDA function name contains an interior NUL byte".to_owned())?;
        self.inner.context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: the module is live, `name` is NUL-terminated and `raw` is a
        // valid out-pointer; the module's context is current.
        let code = unsafe {
            (self.inner.context.api.module_get_function)(&mut raw, self.inner.raw, name.as_ptr())
        };
        self.inner.context.api.check(code, "cuModuleGetFunction")?;
        if raw.is_null() {
            return Err("cuModuleGetFunction returned a null function".to_owned());
        }
        Ok(Function {
            context: self.inner.context.raw,
            raw,
            _not_sync: PhantomData,
        })
    }
}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        let _ = self.context.activate();
        // SAFETY: this inner object owns the module handle and unloads it once.
        // The launch safety contract requires copied `Function` handles not
        // to outlive it.
        let code = unsafe { (self.context.api.module_unload)(self.raw) };
        let _ = code;
    }
}

/// A copyable kernel handle. Its [`Module`] must remain alive until all
/// launches using it have completed.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Function {
    context: RawContext,
    raw: RawFunction,
    _not_sync: NotSync,
}

// SAFETY: a function handle may be launched from another thread with its
// module and context alive. Launch arguments are checked by `Stream::launch`'s
// caller contract; the wrapper is not `Sync`.
unsafe impl Send for Function {}

/// A non-blocking CUDA stream.
pub(crate) struct Stream {
    context: Arc<Context>,
    raw: RawStream,
    _not_sync: NotSync,
}

// SAFETY: non-blocking streams can be used from a thread other than their
// creator. The wrapper is not `Sync`; lifetime ordering remains the backend's duty.
unsafe impl Send for Stream {}

impl Stream {
    fn new(context: &Arc<Context>) -> Result<Self, String> {
        context.activate()?;
        let mut raw = ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer, the context is current, and
        // non-blocking is a valid stream creation flag.
        let code = unsafe { (context.api.stream_create)(&mut raw, CU_STREAM_NON_BLOCKING) };
        context.api.check(code, "cuStreamCreate")?;
        if raw.is_null() {
            return Err("cuStreamCreate returned a null stream".to_owned());
        }
        Ok(Self {
            context: Arc::clone(context),
            raw,
            _not_sync: PhantomData,
        })
    }

    /// Enqueues a kernel on this stream.
    ///
    /// `params` is CUDA's `kernelParams` array: each element points directly
    /// to one argument value, not to an array of argument values. The array is
    /// passed to the driver without allocation.
    ///
    /// # Safety
    ///
    /// Every element of `params` must point to a fully initialized value of
    /// the exact type and size expected by the kernel and remain valid for the
    /// call. Device addresses must name live allocations compatible with this
    /// context. The function's module and all host and device resources the
    /// kernel accesses must remain alive and appropriately ordered until its
    /// execution completes.
    pub(crate) unsafe fn launch(
        &self,
        function: &Function,
        grid: [u32; 3],
        block: [u32; 3],
        shared_bytes: u32,
        params: &mut [*mut c_void],
    ) -> Result<(), String> {
        if grid.contains(&0) || block.contains(&0) {
            return Err("CUDA launch grid and block dimensions must be nonzero".to_owned());
        }
        if function.context != self.context.raw {
            return Err("CUDA function and stream belong to different contexts".to_owned());
        }
        self.context.activate()?;
        let params_ptr = if params.is_empty() {
            ptr::null_mut()
        } else {
            params.as_mut_ptr()
        };
        // SAFETY: the function and stream are live in the same current
        // context, dimensions are nonzero, `shared_bytes` already has the ABI
        // type, and `params_ptr` is null or points to an array of argument
        // pointers. Their pointees satisfy the caller's safety contract;
        // `extra` is null because `kernelParams` is used.
        let code = unsafe {
            (self.context.api.launch_kernel)(
                function.raw,
                grid[0],
                grid[1],
                grid[2],
                block[0],
                block[1],
                block[2],
                shared_bytes,
                self.raw,
                params_ptr,
                ptr::null_mut(),
            )
        };
        self.context.api.check(code, "cuLaunchKernel")
    }

    /// Asynchronously copies host bytes into a device allocation on this
    /// stream.
    ///
    /// # Safety
    ///
    /// `src` must remain readable and unmodified until the copy completes
    /// (normally by synchronizing this stream). `dst` must remain alive, and
    /// accesses to it must be ordered, until then. Ordinary pageable slices
    /// must not be assumed to satisfy this merely because CUDA sometimes
    /// stages them.
    pub(crate) unsafe fn memcpy_htod_async(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &[u8],
    ) -> Result<(), String> {
        let dst_ptr = dst.checked_range(dst_offset, src.len())?;
        if src.is_empty() {
            return Ok(());
        }
        ensure_same_context(&self.context, &dst.context)?;
        self.context.activate()?;
        // SAFETY: the device range was checked, the stream and buffer are live
        // in the same current context, and the caller guarantees the host
        // source and device destination remain valid until completion.
        let code = unsafe {
            (self.context.api.memcpy_htod_async)(
                dst_ptr,
                src.as_ptr().cast::<c_void>(),
                src.len(),
                self.raw,
            )
        };
        self.context.api.check(code, "cuMemcpyHtoDAsync")
    }

    /// Waits until all previously enqueued work on this stream completes.
    pub(crate) fn synchronize(&self) -> Result<(), String> {
        self.context.activate()?;
        // SAFETY: the stream is live.
        let code = unsafe { (self.context.api.stream_synchronize)(self.raw) };
        self.context.api.check(code, "cuStreamSynchronize")
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _ = self.context.activate();
        // SAFETY: this wrapper owns the stream and destroys it exactly once.
        // The owner must synchronize first if in-flight work uses other
        // objects that are being dropped. Drop cannot report driver errors.
        let code = unsafe { (self.context.api.stream_destroy)(self.raw) };
        let _ = code;
    }
}

fn ensure_same_context(a: &Arc<Context>, b: &Arc<Context>) -> Result<(), String> {
    if Arc::ptr_eq(a, b) {
        Ok(())
    } else {
        Err("CUDA objects belong to different primary-context retains".to_owned())
    }
}
