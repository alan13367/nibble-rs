//! Safe wrappers over the few Metal objects the backend uses: the device and
//! its queue, compute pipelines compiled from MSL source, shared buffers,
//! and command buffers with one compute encoder each.

use std::ffi::c_void;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

use super::objc::{self, AutoreleasePool, Encode, Id, Object, sel, send};

/// `MTLResourceStorageModeShared`: CPU and GPU see the same memory.
const STORAGE_MODE_SHARED: usize = 0;
/// `MTLDispatchTypeConcurrent`: dispatches may overlap unless separated by a
/// barrier.
const DISPATCH_TYPE_CONCURRENT: usize = 1;
/// `MTLBarrierScopeBuffers`.
const BARRIER_SCOPE_BUFFERS: usize = 1;
/// `MTLCommandBufferStatusCompleted` and `…Error`.
const STATUS_COMPLETED: usize = 4;
const STATUS_ERROR: usize = 5;

/// How long [`Commands::wait`] polls before it blocks.
const SPIN: Duration = Duration::from_millis(20);

/// `MTLSize`.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct Size {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) depth: usize,
}

impl Size {
    pub(crate) const fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }
}

// SAFETY: `repr(C)` struct of three `NSUInteger`s, as `MTLSize` is declared.
unsafe impl Encode for Size {}

/// A Metal device with one command queue.
#[derive(Debug)]
pub(crate) struct Device {
    device: Object,
    queue: Object,
    name: String,
    max_buffer_length: usize,
    max_threadgroup_memory: usize,
}

impl Device {
    /// The system's default GPU.
    pub(crate) fn system_default() -> Result<Self, String> {
        objc::init()?;
        let _pool = AutoreleasePool::new();
        // SAFETY: `MTLCreateSystemDefaultDevice` returns +1 or nil.
        let device = unsafe { Object::from_owned(objc::create_system_default_device()) }
            .ok_or("no Metal device")?;
        // SAFETY: `newCommandQueue` returns +1 or nil.
        let queue = unsafe { Object::from_owned(send(device.id(), sel!("newCommandQueue"), ())) }
            .ok_or("cannot create a Metal command queue")?;
        // SAFETY: `name` returns an NSString.
        let name = unsafe { objc::string(send(device.id(), sel!("name"), ())) };
        // SAFETY: `maxBufferLength` returns NSUInteger.
        let max_buffer_length: usize = unsafe { send(device.id(), sel!("maxBufferLength"), ()) };
        // SAFETY: `maxThreadgroupMemoryLength` returns NSUInteger.
        let max_threadgroup_memory: usize =
            unsafe { send(device.id(), sel!("maxThreadgroupMemoryLength"), ()) };
        Ok(Self {
            device,
            queue,
            name,
            max_buffer_length,
            max_threadgroup_memory,
        })
    }

    /// The GPU's name, e.g. "Apple M4 Max".
    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    /// The longest buffer the device allows, in bytes.
    pub(crate) fn max_buffer_length(&self) -> usize {
        self.max_buffer_length
    }

    /// Bytes of threadgroup memory a threadgroup may use.
    pub(crate) fn max_threadgroup_memory(&self) -> usize {
        self.max_threadgroup_memory
    }

    /// Compiles MSL `source` into a library.
    pub(crate) fn library(&self, source: &str) -> Result<Library, String> {
        let _pool = AutoreleasePool::new();
        let source = objc::ns_string(source)?;
        let mut error = Id::NIL;
        // SAFETY: `newLibraryWithSource:options:error:` takes an NSString,
        // nil options (the defaults, as llama.cpp uses) and an out-pointer
        // for an autoreleased NSError; it returns +1 or nil.
        let library = unsafe {
            Object::from_owned(send(
                self.device.id(),
                sel!("newLibraryWithSource:options:error:"),
                (source.id(), Id::NIL, &raw mut error),
            ))
        };
        // SAFETY: `error` is nil or the NSError the call produced, alive
        // until the pool drops.
        library
            .map(|library| Library { library })
            .ok_or_else(|| unsafe { objc::error_description(error, "shader compilation failed") })
    }

    /// The compute pipeline for function `name` of `library`.
    pub(crate) fn pipeline(&self, library: &Library, name: &str) -> Result<Pipeline, String> {
        let _pool = AutoreleasePool::new();
        let ns_name = objc::ns_string(name)?;
        // SAFETY: `newFunctionWithName:` takes an NSString and returns +1 or nil.
        let function = unsafe {
            Object::from_owned(send(
                library.library.id(),
                sel!("newFunctionWithName:"),
                (ns_name.id(),),
            ))
        }
        .ok_or_else(|| format!("no Metal function `{name}`"))?;
        let mut error = Id::NIL;
        // SAFETY: `newComputePipelineStateWithFunction:error:` takes a
        // function and an out-pointer for an autoreleased NSError; it
        // returns +1 or nil.
        let pipeline = unsafe {
            Object::from_owned(send(
                self.device.id(),
                sel!("newComputePipelineStateWithFunction:error:"),
                (function.id(), &raw mut error),
            ))
        }
        // SAFETY: `error` is nil or a live NSError (see above).
        .ok_or_else(|| unsafe { objc::error_description(error, "pipeline creation failed") })
        .map_err(|e| format!("{name}: {e}"))?;
        // SAFETY: returns NSUInteger.
        let max_threads: usize =
            unsafe { send(pipeline.id(), sel!("maxTotalThreadsPerThreadgroup"), ()) };
        Ok(Pipeline {
            pipeline,
            max_threads,
        })
    }

    /// A zero-filled shared buffer of `len` bytes (at least 1).
    pub(crate) fn buffer(&self, len: usize) -> Result<Buffer, String> {
        let len = len.max(1);
        if len > self.max_buffer_length {
            return Err(format!(
                "a {len}-byte buffer exceeds the device's limit of {} bytes",
                self.max_buffer_length
            ));
        }
        // SAFETY: `newBufferWithLength:options:` returns +1 or nil. Shared
        // buffers are zero-filled.
        let buffer = unsafe {
            Object::from_owned(send(
                self.device.id(),
                sel!("newBufferWithLength:options:"),
                (len, STORAGE_MODE_SHARED),
            ))
        }
        .ok_or_else(|| format!("cannot allocate a {len}-byte Metal buffer"))?;
        Buffer::new(buffer, len)
    }

    /// A shared buffer holding a copy of `bytes`.
    pub(crate) fn buffer_with(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let buffer = self.buffer(bytes.len())?;
        // SAFETY: the buffer is new, so nothing else accesses its contents,
        // and it holds at least `bytes.len()` bytes.
        unsafe {
            buffer
                .contents()
                .copy_from_nonoverlapping(bytes.as_ptr(), bytes.len())
        };
        Ok(buffer)
    }

    /// Wraps the memory of `bytes` in a shared buffer without copying it,
    /// for the GPU to read only.
    ///
    /// # Safety
    ///
    /// `bytes` must start on a page boundary, and the pages it touches must
    /// all be readable up to the end of the last one (true for a memory
    /// map of a file). The memory must not be freed or unmapped while the
    /// buffer is alive or any command buffer that reads it is running, and
    /// the GPU must not write to the buffer.
    pub(crate) unsafe fn wrap<'a>(&self, bytes: &'a [u8]) -> Result<Wrapped<'a>, String> {
        let page = page_size();
        let len = bytes.len().max(1).next_multiple_of(page);
        if !bytes.as_ptr().addr().is_multiple_of(page) {
            return Err("the weights are not page-aligned".to_owned());
        }
        if len > self.max_buffer_length {
            return Err(format!(
                "the {len}-byte model exceeds the device's buffer limit of {} bytes",
                self.max_buffer_length
            ));
        }
        // SAFETY: page-aligned pointer and page-multiple length, readable
        // as the caller guarantees; a nil deallocator leaves the memory to
        // its owner. Returns +1 or nil.
        let buffer = unsafe {
            Object::from_owned(send(
                self.device.id(),
                sel!("newBufferWithBytesNoCopy:length:options:deallocator:"),
                (
                    bytes.as_ptr().cast::<c_void>().cast_mut(),
                    len,
                    STORAGE_MODE_SHARED,
                    Id::NIL,
                ),
            ))
        }
        .ok_or("cannot wrap the model in a Metal buffer")?;
        Ok(Wrapped {
            buffer: Buffer::new(buffer, len)?,
            _bytes: PhantomData,
        })
    }

    /// A new command buffer with one compute encoder.
    pub(crate) fn commands(&self) -> Result<Commands, String> {
        let _pool = AutoreleasePool::new();
        // SAFETY: returns an autoreleased command buffer or nil; we retain it.
        // Unretained references (as llama.cpp uses): every buffer the
        // commands use is owned by the backend, which outlives them.
        let command_buffer = unsafe {
            Object::retain(send(
                self.queue.id(),
                sel!("commandBufferWithUnretainedReferences"),
                (),
            ))
        }
        .ok_or("cannot create a Metal command buffer")?;
        // SAFETY: returns an autoreleased encoder or nil; we retain it.
        let encoder = unsafe {
            Object::retain(send(
                command_buffer.id(),
                sel!("computeCommandEncoderWithDispatchType:"),
                (DISPATCH_TYPE_CONCURRENT,),
            ))
        }
        .ok_or("cannot create a Metal compute encoder")?;
        Ok(Commands {
            command_buffer,
            encoder: Some(encoder),
        })
    }
}

/// The CPU's memory page size.
pub(crate) fn page_size() -> usize {
    unsafe extern "C" {
        fn getpagesize() -> std::ffi::c_int;
    }
    // SAFETY: no preconditions.
    let page = unsafe { getpagesize() };
    usize::try_from(page).unwrap_or(16384).max(1)
}

/// A compiled MSL library.
#[derive(Debug)]
pub(crate) struct Library {
    library: Object,
}

/// A compute pipeline state.
#[derive(Debug)]
pub(crate) struct Pipeline {
    pipeline: Object,
    max_threads: usize,
}

impl Pipeline {
    /// The most threads a threadgroup of this pipeline can have.
    pub(crate) fn max_threads(&self) -> usize {
        self.max_threads
    }
}

/// A shared buffer: CPU and GPU address the same memory.
#[derive(Debug)]
pub(crate) struct Buffer {
    buffer: Object,
    contents: *mut u8,
    len: usize,
}

// SAFETY: see `Object`; `contents` is only dereferenced through `&mut self`
// or by the caller of an `unsafe` accessor.
unsafe impl Send for Buffer {}

impl Buffer {
    fn new(buffer: Object, len: usize) -> Result<Self, String> {
        // SAFETY: `contents` returns the CPU address of a shared buffer.
        let contents: *mut c_void = unsafe { send(buffer.id(), sel!("contents"), ()) };
        if contents.is_null() {
            return Err("Metal buffer has no CPU address".to_owned());
        }
        Ok(Self {
            buffer,
            contents: contents.cast(),
            len,
        })
    }

    /// The CPU address of the contents.
    pub(crate) fn contents(&self) -> *mut u8 {
        self.contents
    }

    /// The contents as bytes.
    ///
    /// # Safety
    ///
    /// No command buffer that writes this buffer may be running.
    pub(crate) unsafe fn bytes(&self) -> &[u8] {
        // SAFETY: `contents` points to `len` bytes that live as long as the
        // buffer; the caller guarantees the GPU is not writing them.
        unsafe { std::slice::from_raw_parts(self.contents, self.len) }
    }

    /// The contents as mutable bytes.
    ///
    /// # Safety
    ///
    /// No command buffer that reads or writes this buffer may be running.
    pub(crate) unsafe fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as `bytes`, and `&mut self` excludes other CPU access.
        unsafe { std::slice::from_raw_parts_mut(self.contents, self.len) }
    }
}

/// A buffer over memory borrowed for `'a` (see [`Device::wrap`]).
#[derive(Debug)]
pub(crate) struct Wrapped<'a> {
    buffer: Buffer,
    _bytes: PhantomData<&'a [u8]>,
}

impl Wrapped<'_> {
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// A command buffer being encoded, or committed.
#[derive(Debug)]
pub(crate) struct Commands {
    command_buffer: Object,
    /// `None` once encoding has ended.
    encoder: Option<Object>,
}

impl Commands {
    fn encoder(&self) -> Id {
        self.encoder
            .as_ref()
            .expect("commands are still being encoded")
            .id()
    }

    /// Selects the pipeline for the next dispatches.
    pub(crate) fn set_pipeline(&mut self, pipeline: &Pipeline) {
        // SAFETY: `setComputePipelineState:` takes a pipeline state.
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("setComputePipelineState:"),
                (pipeline.pipeline.id(),),
            );
        };
    }

    /// Binds `buffer` from byte `offset` at index `index`.
    ///
    /// # Panics
    ///
    /// If `offset` is past the end of the buffer.
    pub(crate) fn set_buffer(&mut self, index: usize, buffer: &Buffer, offset: usize) {
        assert!(offset <= buffer.len, "buffer offset out of range");
        // SAFETY: `setBuffer:offset:atIndex:` takes a buffer and two
        // NSUIntegers; the offset is within the buffer.
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("setBuffer:offset:atIndex:"),
                (buffer.buffer.id(), offset, index),
            );
        };
    }

    /// Copies `value` into the argument table at `index`.
    pub(crate) fn set_bytes<T: KernelArgs>(&mut self, index: usize, value: &T) {
        // SAFETY: `setBytes:length:atIndex:` copies `size_of::<T>()` bytes
        // from the pointer now. `T: KernelArgs` has no padding, so every
        // byte is initialized.
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("setBytes:length:atIndex:"),
                (
                    std::ptr::from_ref(value).cast::<c_void>(),
                    size_of::<T>(),
                    index,
                ),
            );
        };
    }

    /// Copies `slice` into the argument table at `index` (at most 4096 bytes).
    ///
    /// # Panics
    ///
    /// If `slice` exceeds Metal's 4096-byte `setBytes` limit.
    pub(crate) fn set_slice<T: KernelArgs>(&mut self, index: usize, slice: &[T]) {
        let len = std::mem::size_of_val(slice);
        assert!(len <= 4096, "setBytes slice ({len} bytes) exceeds 4096");
        if len == 0 {
            return;
        }
        // SAFETY: `setBytes:length:atIndex:` copies `len` initialized bytes
        // from `slice.as_ptr()` now (`T: KernelArgs` has no padding).
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("setBytes:length:atIndex:"),
                (slice.as_ptr().cast::<c_void>(), len, index),
            );
        };
    }
    /// Reserves `len` bytes of threadgroup memory at `index`.
    pub(crate) fn set_threadgroup_memory(&mut self, index: usize, len: usize) {
        // SAFETY: `setThreadgroupMemoryLength:atIndex:` takes two NSUIntegers.
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("setThreadgroupMemoryLength:atIndex:"),
                (len.next_multiple_of(16), index),
            );
        };
    }

    /// Dispatches `groups` threadgroups of `threads` threads.
    pub(crate) fn dispatch(&mut self, groups: Size, threads: Size) {
        // SAFETY: `dispatchThreadgroups:threadsPerThreadgroup:` takes two
        // MTLSizes by value.
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("dispatchThreadgroups:threadsPerThreadgroup:"),
                (groups, threads),
            );
        };
    }

    /// Makes later dispatches wait for the buffer writes of earlier ones.
    pub(crate) fn barrier(&mut self) {
        // SAFETY: `memoryBarrierWithScope:` takes an MTLBarrierScope.
        unsafe {
            send::<()>(
                self.encoder(),
                sel!("memoryBarrierWithScope:"),
                (BARRIER_SCOPE_BUFFERS,),
            );
        };
    }

    /// Ends encoding and submits the commands.
    pub(crate) fn commit(&mut self) {
        if let Some(encoder) = self.encoder.take() {
            // SAFETY: `endEncoding` takes no arguments.
            unsafe { send::<()>(encoder.id(), sel!("endEncoding"), ()) };
        }
        // SAFETY: `commit` takes no arguments; committing twice is not
        // possible since the encoder is gone after the first.
        unsafe { send::<()>(self.command_buffer.id(), sel!("commit"), ()) };
    }

    /// Whether committed commands have finished, without waiting; an error
    /// if they failed.
    pub(crate) fn status(&self) -> Result<bool, String> {
        // SAFETY: `status` takes no arguments.
        let status: usize = unsafe { send(self.command_buffer.id(), sel!("status"), ()) };
        match status {
            STATUS_COMPLETED => Ok(true),
            STATUS_ERROR => {
                let _pool = AutoreleasePool::new();
                // SAFETY: `error` returns an NSError or nil, alive until the
                // pool drops.
                let error: Id = unsafe { send(self.command_buffer.id(), sel!("error"), ()) };
                // SAFETY: as above.
                Err(unsafe { objc::error_description(error, "Metal command buffer failed") })
            }
            _ => Ok(false),
        }
    }

    /// Waits for committed commands to finish.
    ///
    /// Polls for up to [`SPIN`] first: waking from `waitUntilCompleted`
    /// takes long enough to matter once per token, and a CPU kept busy also
    /// keeps the GPU's clocks up.
    pub(crate) fn wait(&self) -> Result<(), String> {
        let _pool = AutoreleasePool::new();
        let start = Instant::now();
        // SAFETY: `status` takes no arguments.
        while unsafe { send::<usize>(self.command_buffer.id(), sel!("status"), ()) }
            < STATUS_COMPLETED
            && start.elapsed() < SPIN
        {
            std::hint::spin_loop();
        }
        // SAFETY: `waitUntilCompleted` and `status` take no arguments.
        let status: usize = unsafe {
            send::<()>(self.command_buffer.id(), sel!("waitUntilCompleted"), ());
            send(self.command_buffer.id(), sel!("status"), ())
        };
        match status {
            STATUS_COMPLETED => Ok(()),
            STATUS_ERROR => {
                // SAFETY: `error` returns an NSError or nil.
                let error: Id = unsafe { send(self.command_buffer.id(), sel!("error"), ()) };
                // SAFETY: as above.
                Err(unsafe { objc::error_description(error, "Metal command buffer failed") })
            }
            other => Err(format!("Metal command buffer ended with status {other}")),
        }
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        // An encoder must end before its command buffer is released.
        if let Some(encoder) = self.encoder.take() {
            // SAFETY: `endEncoding` takes no arguments.
            unsafe { send::<()>(encoder.id(), sel!("endEncoding"), ()) };
        }
    }
}

/// Kernel argument structs passed with [`Commands::set_bytes`].
///
/// # Safety
///
/// The type must be `repr(C)` with no padding bytes (only 4-byte fields,
/// for instance), and match the MSL struct the kernel declares.
pub(crate) unsafe trait KernelArgs: Copy {}

// SAFETY: `u32` is 4-byte aligned with no padding, matching MSL `uint`.
unsafe impl KernelArgs for u32 {}
