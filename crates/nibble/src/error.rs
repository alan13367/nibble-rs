use std::fmt;
use std::io;

use nibble_gguf::GgmlType;

use crate::model::Architecture;
use crate::quant;

/// Result type used throughout nibble.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Errors from loading models and running inference.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// Reading the model file failed.
    Io(io::Error),
    /// The model file is not valid GGUF.
    Gguf(nibble_gguf::Error),
    /// The model's `general.architecture` is not supported.
    UnsupportedArchitecture(String),
    /// A required metadata key is missing.
    MissingMetadata(String),
    /// A metadata value has the wrong type or an unusable value.
    InvalidMetadata {
        /// The metadata key.
        key: String,
        /// What is wrong with the value.
        reason: String,
    },
    /// A required tensor is missing.
    MissingTensor(String),
    /// A tensor uses a type the active backend cannot compute with yet.
    UnsupportedTensorType {
        /// The tensor's name.
        tensor: String,
        /// The tensor's type.
        ggml_type: GgmlType,
    },
    /// A tensor's shape does not match the model's hyperparameters.
    TensorShape {
        /// The tensor's name.
        tensor: String,
        /// The expected dimensions (ggml order).
        expected: Vec<u64>,
        /// The dimensions found in the file.
        found: Vec<u64>,
    },
    /// The tokenizer data is unusable.
    Tokenizer(String),
    /// Evaluating more tokens would exceed the session's context size.
    ContextFull {
        /// The session's context size in tokens.
        n_ctx: usize,
    },
    /// A token id outside the vocabulary.
    InvalidToken(u32),
    /// `Session::eval` was called with no tokens.
    EmptyInput,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Gguf(e) => write!(f, "invalid GGUF file: {e}"),
            Self::UnsupportedArchitecture(arch) => {
                let supported: Vec<&str> = Architecture::ALL.iter().map(|a| a.name()).collect();
                write!(
                    f,
                    "unsupported model architecture `{arch}` (supported: {})",
                    supported.join(", ")
                )
            }
            Self::MissingMetadata(key) => write!(f, "missing metadata key `{key}`"),
            Self::InvalidMetadata { key, reason } => write!(f, "metadata `{key}`: {reason}"),
            Self::MissingTensor(name) => write!(f, "missing tensor `{name}`"),
            Self::UnsupportedTensorType { tensor, ggml_type } => {
                let supported: Vec<&str> = quant::SUPPORTED.iter().map(|t| t.name()).collect();
                write!(
                    f,
                    "tensor `{tensor}` has type {ggml_type}, which is not supported yet (supported: {})",
                    supported.join(", ")
                )
            }
            Self::TensorShape {
                tensor,
                expected,
                found,
            } => write!(
                f,
                "tensor `{tensor}` has shape {found:?}, expected {expected:?}"
            ),
            Self::Tokenizer(msg) => write!(f, "tokenizer: {msg}"),
            Self::ContextFull { n_ctx } => {
                write!(
                    f,
                    "context is full ({n_ctx} tokens); start a new session or raise the context size"
                )
            }
            Self::InvalidToken(id) => write!(f, "token id {id} is outside the vocabulary"),
            Self::EmptyInput => write!(f, "no tokens to evaluate"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Gguf(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<nibble_gguf::Error> for Error {
    fn from(e: nibble_gguf::Error) -> Self {
        Self::Gguf(e)
    }
}
