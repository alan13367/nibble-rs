//! File bytes for the parser: memory-mapped where the OS supports it, read
//! into an aligned buffer elsewhere (e.g. `wasm32`).
//!
//! The mapping is implemented directly on the OS APIs (`mmap` on Unix,
//! `MapViewOfFile` on Windows) to keep the crate dependency-free.

use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::ops::Deref;
use std::path::Path;
use std::ptr::NonNull;

/// Model bytes, either memory-mapped or owned.
pub enum Storage {
    /// A read-only memory map of the file.
    Mapped(Mmap),
    /// The file's bytes copied into memory.
    Owned(AlignedBytes),
}

impl Storage {
    /// Memory-maps `path`, or reads it into memory on platforms without `mmap`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Mmap::map`]: the file must not be modified or
    /// truncated, by this or any other process, while the returned value is alive.
    ///
    /// # Errors
    ///
    /// Returns any error from opening, mapping or reading the file.
    pub unsafe fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let path = path.as_ref();
        if !Mmap::SUPPORTED {
            return Self::read(path);
        }
        let file = File::open(path)?;
        // SAFETY: the caller upholds `Mmap::map`'s contract for this file.
        let map = unsafe { Mmap::map(&file)? };
        Ok(Self::Mapped(map))
    }

    /// Reads `path` into a 64-byte-aligned buffer. Safe, but costs a copy of the whole file.
    ///
    /// # Errors
    ///
    /// Returns any error from opening or reading the file.
    pub fn read(path: impl AsRef<Path>) -> io::Result<Self> {
        let mut file = File::open(path)?;
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "file is larger than the address space",
            )
        })?;
        let mut bytes = AlignedBytes::zeroed(len);
        file.read_exact(&mut bytes)?;
        Ok(Self::Owned(bytes))
    }

    /// Copies `bytes` into an aligned buffer.
    #[must_use]
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self::Owned(AlignedBytes::from_slice(bytes))
    }

    /// Whether the bytes are memory-mapped rather than copied.
    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        matches!(self, Self::Mapped(_))
    }
}

impl Deref for Storage {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        match self {
            Self::Mapped(map) => map,
            Self::Owned(bytes) => bytes,
        }
    }
}

impl AsRef<[u8]> for Storage {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = if self.is_mapped() { "Mapped" } else { "Owned" };
        f.debug_struct("Storage")
            .field("kind", &kind)
            .field("len", &self.len())
            .finish()
    }
}

/// One cache line; gives [`AlignedBytes`] its alignment.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct CacheLine([u8; 64]);

/// A heap buffer whose start is 64-byte aligned, so tensor data at aligned
/// file offsets can be viewed as `f32`, `u16` or SIMD-friendly blocks.
#[derive(Clone)]
pub struct AlignedBytes {
    lines: Vec<CacheLine>,
    len: usize,
}

impl AlignedBytes {
    /// A zero-filled buffer of `len` bytes.
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        Self {
            lines: vec![CacheLine([0; 64]); len.div_ceil(64)],
            len,
        }
    }

    /// A buffer holding a copy of `bytes`.
    #[must_use]
    pub fn from_slice(bytes: &[u8]) -> Self {
        let mut out = Self::zeroed(bytes.len());
        out.copy_from_slice(bytes);
        out
    }
}

impl Deref for AlignedBytes {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `CacheLine` is `repr(C)` over `[u8; 64]` with no padding, so
        // `lines` is `lines.len() * 64` initialized bytes, and
        // `len <= lines.len() * 64` by construction.
        unsafe { std::slice::from_raw_parts(self.lines.as_ptr().cast::<u8>(), self.len) }
    }
}

impl std::ops::DerefMut for AlignedBytes {
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `deref`; `&mut self` makes the borrow exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.lines.as_mut_ptr().cast::<u8>(), self.len) }
    }
}

impl fmt::Debug for AlignedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AlignedBytes")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// A read-only memory map of a whole file.
pub struct Mmap {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the mapping is read-only, owned exclusively by this value, and not
// tied to the thread that created it, so it can be sent and shared like `&[u8]`.
unsafe impl Send for Mmap {}
// SAFETY: see `Send`; no method mutates the mapped memory.
unsafe impl Sync for Mmap {}

impl Mmap {
    /// Whether this platform supports memory mapping.
    pub const SUPPORTED: bool = cfg!(any(unix, windows));

    /// Maps the whole of `file` read-only. The file handle may be closed afterwards.
    ///
    /// # Safety
    ///
    /// The file must not be modified or truncated, by this or any other
    /// process, while the map is alive. Changing it would change memory that
    /// Rust treats as immutable; truncating it can crash the process (SIGBUS).
    ///
    /// # Errors
    ///
    /// Returns the OS error if mapping fails, or `Unsupported` where
    /// [`Mmap::SUPPORTED`] is false.
    pub unsafe fn map(file: &File) -> io::Result<Self> {
        let len = usize::try_from(file.metadata()?.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "file is larger than the address space",
            )
        })?;
        if len == 0 {
            // Neither OS can map an empty file; an empty slice needs no mapping.
            return Ok(Self {
                ptr: NonNull::dangling(),
                len: 0,
            });
        }
        // SAFETY: `len` is the file's current length and the caller guarantees
        // the file stays unchanged while mapped.
        let ptr = unsafe { sys::map(file, len)? };
        Ok(Self { ptr, len })
    }
}

impl Deref for Mmap {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        // SAFETY: `ptr` is either dangling with `len == 0` (valid for an empty
        // slice) or the start of a live read-only mapping of `len` bytes.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for Mmap {
    fn drop(&mut self) {
        if self.len > 0 {
            // SAFETY: `ptr`/`len` describe a mapping created by `sys::map`,
            // unmapped exactly once, here.
            unsafe { sys::unmap(self.ptr, self.len) };
        }
    }
}

impl fmt::Debug for Mmap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mmap")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(unix)]
mod sys {
    use std::ffi::{c_int, c_long, c_void};
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::ptr::{self, NonNull};

    // Same values on Linux, Android, macOS, iOS and the BSDs.
    const PROT_READ: c_int = 1;
    const MAP_PRIVATE: c_int = 2;

    // `off_t` is `long` for the plain `mmap` symbol on every supported Unix.
    unsafe extern "C" {
        fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            offset: c_long,
        ) -> *mut c_void;
        fn munmap(addr: *mut c_void, len: usize) -> c_int;
    }

    /// # Safety
    ///
    /// `len` must be the non-zero length of `file`; see `Mmap::map`.
    pub(super) unsafe fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        // SAFETY: a null hint, a valid open descriptor and offset 0 are valid
        // arguments; the result is checked against MAP_FAILED below.
        let ptr = unsafe {
            mmap(
                ptr::null_mut(),
                len,
                PROT_READ,
                MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        // MAP_FAILED is (void*)-1.
        if ptr.addr() == usize::MAX {
            return Err(io::Error::last_os_error());
        }
        NonNull::new(ptr.cast::<u8>()).ok_or_else(|| io::Error::other("mmap returned null"))
    }

    /// # Safety
    ///
    /// `ptr`/`len` must come from a successful `map` that has not been unmapped.
    pub(super) unsafe fn unmap(ptr: NonNull<u8>, len: usize) {
        // SAFETY: guaranteed by the caller. A failure leaves the mapping in
        // place, which only leaks address space.
        unsafe { munmap(ptr.as_ptr().cast::<c_void>(), len) };
    }
}

#[cfg(windows)]
mod sys {
    use std::ffi::c_void;
    use std::fs::File;
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::ptr::{self, NonNull};

    type Handle = *mut c_void;

    const PAGE_READONLY: u32 = 0x02;
    const FILE_MAP_READ: u32 = 0x04;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateFileMappingW(
            file: Handle,
            attributes: *const c_void,
            protect: u32,
            max_size_high: u32,
            max_size_low: u32,
            name: *const u16,
        ) -> Handle;
        fn MapViewOfFile(
            mapping: Handle,
            access: u32,
            offset_high: u32,
            offset_low: u32,
            len: usize,
        ) -> *mut c_void;
        fn UnmapViewOfFile(base: *const c_void) -> i32;
        fn CloseHandle(handle: Handle) -> i32;
    }

    /// # Safety
    ///
    /// `len` must be the non-zero length of `file`; see `Mmap::map`.
    pub(super) unsafe fn map(file: &File, len: usize) -> io::Result<NonNull<u8>> {
        // SAFETY: a valid file handle with null security attributes and name;
        // sizes of 0 mean "the whole file". Failure returns null.
        let mapping = unsafe {
            CreateFileMappingW(
                file.as_raw_handle(),
                ptr::null(),
                PAGE_READONLY,
                0,
                0,
                ptr::null(),
            )
        };
        if mapping.is_null() {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: `mapping` is a valid read-only mapping handle of at least `len` bytes.
        let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, len) };
        let err = io::Error::last_os_error();
        // SAFETY: `mapping` is a valid handle we own. The view (if any) keeps
        // the mapping object alive after its handle is closed.
        unsafe { CloseHandle(mapping) };
        NonNull::new(view.cast::<u8>()).ok_or(err)
    }

    /// # Safety
    ///
    /// `ptr` must come from a successful `map` that has not been unmapped.
    pub(super) unsafe fn unmap(ptr: NonNull<u8>, _len: usize) {
        // SAFETY: guaranteed by the caller. A failure leaks address space only.
        unsafe { UnmapViewOfFile(ptr.as_ptr().cast_const().cast::<c_void>()) };
    }
}

#[cfg(not(any(unix, windows)))]
mod sys {
    use std::fs::File;
    use std::io;
    use std::ptr::NonNull;

    pub(super) unsafe fn map(_file: &File, _len: usize) -> io::Result<NonNull<u8>> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "memory mapping is not supported on this platform",
        ))
    }

    pub(super) unsafe fn unmap(_ptr: NonNull<u8>, _len: usize) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_bytes_are_aligned_and_copied() {
        for len in [0, 1, 63, 64, 65, 1000] {
            let data: Vec<u8> = (0..len).map(|i| (i * 7) as u8).collect();
            let bytes = AlignedBytes::from_slice(&data);
            assert_eq!(&*bytes, &data[..]);
            assert_eq!(bytes.as_ptr().addr() % 64, 0);
        }
    }
}
