use std::fmt;

/// Result type used by the GGUF parser.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why a byte buffer could not be parsed as a GGUF file.
///
/// Offsets are byte positions from the start of the file, to help locate
/// corruption with a hex editor.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The file does not start with the `GGUF` magic bytes.
    BadMagic([u8; 4]),
    /// The GGUF version is not one this reader supports (2 or 3).
    UnsupportedVersion(u32),
    /// The file ended before a value was complete.
    UnexpectedEof {
        /// Where the incomplete value starts.
        offset: usize,
        /// How many bytes the value needed.
        needed: u64,
    },
    /// A metadata key or tensor name is not valid UTF-8.
    InvalidUtf8 {
        /// Where the string's bytes start.
        offset: usize,
    },
    /// A metadata value type id is not defined by the GGUF specification.
    InvalidValueType {
        /// Where the type id is stored.
        offset: usize,
        /// The type id that was found.
        raw: u32,
    },
    /// An array whose elements are arrays. Nothing writes these, so they are rejected.
    NestedArray {
        /// Where the array's element type is stored.
        offset: usize,
    },
    /// A boolean stored as something other than 0 or 1.
    InvalidBool {
        /// Where the byte is stored.
        offset: usize,
        /// The byte that was found.
        raw: u8,
    },
    /// The same metadata key appears more than once.
    DuplicateKey(String),
    /// `general.alignment` is not a non-zero power of two stored as `u32`.
    InvalidAlignment(String),
    /// The same tensor name appears more than once.
    DuplicateTensor(String),
    /// A tensor's shape cannot be stored with its type.
    InvalidShape {
        /// The tensor's name.
        tensor: String,
        /// What is wrong with the shape.
        reason: String,
    },
    /// A tensor's data offset is not a multiple of the file's alignment.
    MisalignedTensor {
        /// The tensor's name.
        tensor: String,
        /// The offset, relative to the start of the data section.
        offset: u64,
        /// The file's alignment.
        alignment: usize,
    },
    /// A tensor's data extends past the end of the file.
    TensorOutOfBounds {
        /// The tensor's name.
        tensor: String,
        /// The offset, relative to the start of the data section.
        offset: u64,
        /// The tensor's size in bytes.
        size: u64,
        /// The size of the data section.
        available: u64,
    },
    /// A count or length does not fit in this platform's address space.
    TooLarge {
        /// Where the count or length is stored.
        offset: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic(magic) => {
                write!(f, "not a GGUF file (starts with {magic:02x?})")
            }
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported GGUF version {version} (supported: 2, 3)")
            }
            Self::UnexpectedEof { offset, needed } => write!(
                f,
                "file ends early: a {needed}-byte value at offset {offset} is incomplete"
            ),
            Self::InvalidUtf8 { offset } => {
                write!(f, "string at offset {offset} is not valid UTF-8")
            }
            Self::InvalidValueType { offset, raw } => {
                write!(f, "unknown metadata value type {raw} at offset {offset}")
            }
            Self::NestedArray { offset } => {
                write!(f, "nested arrays are not supported (offset {offset})")
            }
            Self::InvalidBool { offset, raw } => {
                write!(
                    f,
                    "boolean at offset {offset} has value {raw}, expected 0 or 1"
                )
            }
            Self::DuplicateKey(key) => write!(f, "metadata key `{key}` appears more than once"),
            Self::InvalidAlignment(found) => write!(
                f,
                "general.alignment must be a non-zero power of two stored as u32, found {found}"
            ),
            Self::DuplicateTensor(name) => write!(f, "tensor `{name}` appears more than once"),
            Self::InvalidShape { tensor, reason } => {
                write!(f, "tensor `{tensor}` has an invalid shape: {reason}")
            }
            Self::MisalignedTensor {
                tensor,
                offset,
                alignment,
            } => write!(
                f,
                "tensor `{tensor}` starts at data offset {offset}, \
                 which is not a multiple of the alignment {alignment}"
            ),
            Self::TensorOutOfBounds {
                tensor,
                offset,
                size,
                available,
            } => write!(
                f,
                "tensor `{tensor}` ({size} bytes at data offset {offset}) extends past \
                 the end of the data section ({available} bytes)"
            ),
            Self::TooLarge { offset } => write!(
                f,
                "count or length at offset {offset} is too large for this platform"
            ),
        }
    }
}

impl std::error::Error for Error {}
