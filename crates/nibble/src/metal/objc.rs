//! The Objective-C runtime, loaded with `dlopen` at run time.
//!
//! Nothing is linked: `libobjc` and the Foundation, CoreGraphics and Metal
//! frameworks are opened when a Metal backend is first created, so a binary
//! still starts (and runs on the CPU) where they are missing.
//!
//! Messages go through `objc_msgSend`, cast to the exact function type of
//! each call, as arm64 requires: [`send`] takes the receiver, a selector and a
//! tuple of arguments whose types (and the return type) implement [`Encode`].

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;
use std::sync::OnceLock;

unsafe extern "C" {
    // libSystem, which std already links.
    fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

const RTLD_LAZY: c_int = 0x1;
const RTLD_GLOBAL: c_int = 0x8;

/// An object pointer (`id`), possibly nil. Carries no ownership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub(crate) struct Id(pub(crate) *mut c_void);

impl Id {
    pub(crate) const NIL: Self = Self(std::ptr::null_mut());

    pub(crate) fn is_nil(self) -> bool {
        self.0.is_null()
    }
}

/// A selector (`SEL`).
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct Sel(*const c_void);

// SAFETY: a selector is an immutable, process-global interned value.
unsafe impl Send for Sel {}
// SAFETY: as above.
unsafe impl Sync for Sel {}

/// The entry points this module uses, resolved once per process.
struct Runtime {
    get_class: unsafe extern "C" fn(*const c_char) -> Id,
    register_name: unsafe extern "C" fn(*const c_char) -> Sel,
    /// `objc_msgSend`, to be cast to each call's type before it is called.
    msg_send: unsafe extern "C" fn(),
    pool_push: unsafe extern "C" fn() -> *mut c_void,
    pool_pop: unsafe extern "C" fn(*mut c_void),
    create_system_default_device: unsafe extern "C" fn() -> Id,
}

/// Opens `path` with `dlopen`.
fn open(path: &CStr) -> Result<*mut c_void, String> {
    // SAFETY: `path` is NUL-terminated. Loading a system framework runs only
    // its own initializers.
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_LAZY | RTLD_GLOBAL) };
    if handle.is_null() {
        return Err(format!(
            "cannot load {}: {}",
            path.to_string_lossy(),
            last_dl_error()
        ));
    }
    Ok(handle)
}

/// The address of `name` in `handle`.
fn symbol(handle: *mut c_void, name: &CStr) -> Result<NonNull<c_void>, String> {
    // SAFETY: `handle` came from a successful `dlopen` and `name` is
    // NUL-terminated.
    let ptr = unsafe { dlsym(handle, name.as_ptr()) };
    NonNull::new(ptr).ok_or_else(|| {
        format!(
            "missing symbol {}: {}",
            name.to_string_lossy(),
            last_dl_error()
        )
    })
}

/// The function `name` in `handle`, as the function pointer type `F`.
///
/// # Safety
///
/// `F` must be the `unsafe extern "C" fn` type of the symbol's C prototype.
unsafe fn function<F: Copy>(handle: *mut c_void, name: &CStr) -> Result<F, String> {
    let ptr = symbol(handle, name)?;
    assert_eq!(
        size_of::<F>(),
        size_of::<NonNull<c_void>>(),
        "F is a function pointer"
    );
    // SAFETY: `F` is a function pointer type (same size as the address) for
    // the symbol's own prototype, as the caller guarantees.
    Ok(unsafe { std::mem::transmute_copy::<NonNull<c_void>, F>(&ptr) })
}

fn last_dl_error() -> String {
    // SAFETY: `dlerror` returns null or a NUL-terminated string that stays
    // valid until the next dl call on this thread; it is copied at once.
    let msg = unsafe { dlerror() };
    if msg.is_null() {
        return "unknown error".to_owned();
    }
    // SAFETY: non-null, NUL-terminated (see above).
    unsafe { CStr::from_ptr(msg) }
        .to_string_lossy()
        .into_owned()
}

impl Runtime {
    fn load() -> Result<Self, String> {
        let objc = open(c"/usr/lib/libobjc.A.dylib")?;
        // Loading Foundation registers NSString and friends with the
        // runtime. Without CoreGraphics, `MTLCreateSystemDefaultDevice`
        // returns nil in command-line programs (Apple's documentation says
        // to link it).
        open(c"/System/Library/Frameworks/Foundation.framework/Foundation")?;
        open(c"/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")?;
        let metal = open(c"/System/Library/Frameworks/Metal.framework/Metal")?;
        // SAFETY: each symbol is the C function of that name,
        // whose prototype in the system headers is the function type it is
        // read as. `objc_msgSend` is kept untyped and cast per call in `send`.
        unsafe {
            Ok(Self {
                get_class: function(objc, c"objc_getClass")?,
                register_name: function(objc, c"sel_registerName")?,
                msg_send: function(objc, c"objc_msgSend")?,
                pool_push: function(objc, c"objc_autoreleasePoolPush")?,
                pool_pop: function(objc, c"objc_autoreleasePoolPop")?,
                create_system_default_device: function(metal, c"MTLCreateSystemDefaultDevice")?,
            })
        }
    }
}

fn runtime() -> Result<&'static Runtime, String> {
    static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();
    RUNTIME
        .get_or_init(Runtime::load)
        .as_ref()
        .map_err(Clone::clone)
}

/// Loads the runtime and the frameworks. Every other function here may only
/// be called after this succeeded.
pub(crate) fn init() -> Result<(), String> {
    runtime().map(|_| ())
}

fn loaded() -> &'static Runtime {
    runtime().expect("objc::init must succeed before the runtime is used")
}

/// The class named `name`, or nil.
pub(crate) fn class(name: &CStr) -> Id {
    // SAFETY: `name` is NUL-terminated; `objc_getClass` returns nil for
    // unknown names.
    unsafe { (loaded().get_class)(name.as_ptr()) }
}

/// The selector named `name`.
pub(crate) fn register_selector(name: &CStr) -> Sel {
    // SAFETY: `name` is NUL-terminated; registering is idempotent.
    unsafe { (loaded().register_name)(name.as_ptr()) }
}

/// A selector resolved once per call site.
macro_rules! sel {
    ($name:literal) => {{
        static SEL: std::sync::OnceLock<$crate::metal::objc::Sel> = std::sync::OnceLock::new();
        *SEL.get_or_init(|| {
            $crate::metal::objc::register_selector(
                std::ffi::CStr::from_bytes_with_nul(concat!($name, "\0").as_bytes())
                    .expect("selector names have no interior NUL"),
            )
        })
    }};
}
pub(crate) use sel;

/// `MTLCreateSystemDefaultDevice()`: a +1 reference, or nil.
pub(crate) fn create_system_default_device() -> Id {
    // SAFETY: the function takes no arguments and returns a retained device
    // or nil.
    unsafe { (loaded().create_system_default_device)() }
}

/// Types that are passed to and returned from `objc_msgSend` exactly as the
/// C ABI passes the corresponding Objective-C type.
///
/// # Safety
///
/// The type must have the size, alignment and calling convention of the C
/// type it stands for (`id`, `SEL`, `NSUInteger`, `BOOL`, a `repr(C)` struct
/// of such fields, ...).
pub(crate) unsafe trait Encode: Copy {}

// SAFETY: each is the Rust type of the C scalar or pointer of the same name.
unsafe impl Encode for () {}
// SAFETY: as above.
unsafe impl Encode for Id {}
// SAFETY: as above.
unsafe impl Encode for Sel {}
// SAFETY: as above.
unsafe impl Encode for usize {}
// SAFETY: as above.
unsafe impl Encode for u64 {}
// SAFETY: as above.
unsafe impl Encode for u32 {}
// SAFETY: as above.
unsafe impl Encode for i64 {}
// SAFETY: as above.
unsafe impl Encode for bool {}
// SAFETY: as above.
unsafe impl Encode for i8 {}
// SAFETY: as above.
unsafe impl Encode for *const c_void {}
// SAFETY: as above.
unsafe impl Encode for *mut c_void {}
// SAFETY: as above.
unsafe impl Encode for *const c_char {}
// SAFETY: as above.
unsafe impl Encode for *mut Id {}

/// A tuple of message arguments.
pub(crate) trait Args {
    /// Sends `sel` to `receiver` with these arguments through `imp`
    /// (`objc_msgSend`).
    ///
    /// # Safety
    ///
    /// See [`send`].
    unsafe fn send<R: Encode>(self, imp: unsafe extern "C" fn(), receiver: Id, sel: Sel) -> R;
}

macro_rules! impl_args {
    ($($a:ident),*) => {
        impl<$($a: Encode),*> Args for ($($a,)*) {
            #[allow(non_snake_case, reason = "the type parameters name the values")]
            unsafe fn send<R: Encode>(self, imp: unsafe extern "C" fn(), receiver: Id, sel: Sel) -> R {
                // SAFETY: `objc_msgSend` must be called with the method's
                // own C type; the caller guarantees that the method `sel`
                // of `receiver` takes exactly these argument types and
                // returns `R`. All are `Encode`, so they are passed as C
                // would pass them.
                let f: unsafe extern "C" fn(Id, Sel $(, $a)*) -> R = unsafe { std::mem::transmute(imp) };
                let ($($a,)*) = self;
                // SAFETY: as above.
                unsafe { f(receiver, sel $(, $a)*) }
            }
        }
    };
}

impl_args!();
impl_args!(A);
impl_args!(A, B);
impl_args!(A, B, C);
impl_args!(A, B, C, D);
impl_args!(A, B, C, D, E);

/// Sends message `sel` to `receiver`.
///
/// # Safety
///
/// `receiver` must be a live object (or nil, in which case the result is
/// zero) that responds to `sel`, the method's parameters and return type
/// must be exactly `args` and `R`, and the arguments must satisfy the
/// method's own requirements (valid pointers, ranges, ...).
pub(crate) unsafe fn send<R: Encode>(receiver: Id, sel: Sel, args: impl Args) -> R {
    // SAFETY: guaranteed by the caller.
    unsafe { args.send(loaded().msg_send, receiver, sel) }
}

/// A strong (+1) reference, released on drop.
#[derive(Debug)]
#[repr(transparent)]
pub(crate) struct Object(NonNull<c_void>);

// SAFETY: retain and release are atomic, and the objects kept in an
// `Object` (Metal devices, queues, pipelines, buffers, command buffers and
// encoders) may be used from any thread as long as they are not used from
// two at once, which `&mut` access on the owning types ensures.
unsafe impl Send for Object {}

impl Object {
    /// Takes ownership of a +1 reference (from `new…`, `alloc`/`init…`,
    /// `copy`). Nil becomes `None`.
    ///
    /// # Safety
    ///
    /// `id` must be nil or an object the caller owns a reference to, which
    /// is handed over.
    pub(crate) unsafe fn from_owned(id: Id) -> Option<Self> {
        NonNull::new(id.0).map(Self)
    }

    /// Retains an object the caller does not own (an autoreleased result).
    ///
    /// # Safety
    ///
    /// `id` must be nil or a live object.
    pub(crate) unsafe fn retain(id: Id) -> Option<Self> {
        // SAFETY: `retain` takes no arguments and returns the receiver;
        // nil receivers are allowed.
        let id: Id = unsafe { send(id, sel!("retain"), ()) };
        // SAFETY: the retain above gave us a reference.
        unsafe { Self::from_owned(id) }
    }

    pub(crate) const fn id(&self) -> Id {
        Id(self.0.as_ptr())
    }
}

impl Drop for Object {
    fn drop(&mut self) {
        // SAFETY: we own one reference to a live object.
        unsafe { send::<()>(self.id(), sel!("release"), ()) };
    }
}

/// An autorelease pool for the current scope: objects autoreleased while it
/// lives are released when it drops.
pub(crate) struct AutoreleasePool(*mut c_void);

impl AutoreleasePool {
    pub(crate) fn new() -> Self {
        // SAFETY: pushing a pool has no preconditions.
        Self(unsafe { (loaded().pool_push)() })
    }
}

impl Drop for AutoreleasePool {
    fn drop(&mut self) {
        // SAFETY: the token came from the matching push on this thread
        // (the type is neither Send nor Sync), and pools are popped in
        // reverse order because they are scoped values.
        unsafe { (loaded().pool_pop)(self.0) };
    }
}

/// A new `NSString` holding `s`.
pub(crate) fn ns_string(s: &str) -> Result<Object, String> {
    let ns_string = class(c"NSString");
    // SAFETY: `alloc` returns an uninitialized +1 instance; nil receivers are
    // allowed.
    let alloc: Id = unsafe { send(ns_string, sel!("alloc"), ()) };
    // NSUTF8StringEncoding.
    const UTF8: usize = 4;
    // SAFETY: `initWithBytes:length:encoding:` copies `s.len()` bytes from
    // the pointer; the result is the +1 initialized string or nil.
    let id: Id = unsafe {
        send(
            alloc,
            sel!("initWithBytes:length:encoding:"),
            (s.as_ptr().cast::<c_void>(), s.len(), UTF8),
        )
    };
    // SAFETY: an `init…` result is owned.
    unsafe { Object::from_owned(id) }.ok_or_else(|| "cannot create an NSString".to_owned())
}

/// The contents of an `NSString` (or nil, as an empty string).
///
/// # Safety
///
/// `s` must be nil or a live `NSString`.
pub(crate) unsafe fn string(s: Id) -> String {
    // SAFETY: `UTF8String` returns null for nil or a NUL-terminated buffer
    // that lives at least as long as the autorelease pool; it is copied at once.
    let ptr: *const c_char = unsafe { send(s, sel!("UTF8String"), ()) };
    if ptr.is_null() {
        return String::new();
    }
    // SAFETY: non-null and NUL-terminated (see above).
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

/// `[error localizedDescription]`, or `fallback` when `error` is nil.
///
/// # Safety
///
/// `error` must be nil or a live `NSError`.
pub(crate) unsafe fn error_description(error: Id, fallback: &str) -> String {
    if error.is_nil() {
        return fallback.to_owned();
    }
    // SAFETY: `localizedDescription` returns an NSString.
    let description: Id = unsafe { send(error, sel!("localizedDescription"), ()) };
    // SAFETY: as above.
    unsafe { string(description) }
}
