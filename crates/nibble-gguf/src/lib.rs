//! Zero-copy, zero-dependency reader (and a small writer) for the
//! [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model format.
//!
//! [`Gguf::parse`] borrows everything from the input bytes: metadata strings,
//! arrays and tensor data are views into the buffer, never copies. The buffer
//! usually comes from [`Storage`], which memory-maps the file where the
//! platform supports it and reads it into an aligned buffer elsewhere.
//!
//! ```no_run
//! use nibble_gguf::{Gguf, Storage};
//!
//! // SAFETY: nothing modifies or truncates the model file while it is mapped.
//! let storage = unsafe { Storage::open("model.gguf")? };
//! let gguf = Gguf::parse(&storage)?;
//! println!("{:?} with {} tensors", gguf.architecture(), gguf.tensors().len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod error;
mod reader;
mod storage;
mod types;
mod value;
mod writer;

pub use error::{Error, Result};
pub use reader::{DEFAULT_ALIGNMENT, Gguf, MAX_DIMS, Shape, TensorInfo};
pub use storage::{AlignedBytes, Mmap, Storage};
pub use types::{BlockLayout, GgmlType, QK_K};
pub use value::{Array, ArrayIter, GgufStr, Value, ValueType};
pub use writer::{GgufWriter, OwnedValue};
