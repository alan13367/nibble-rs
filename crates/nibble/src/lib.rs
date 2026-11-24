//! nibble: a tiny, dependency-free GGUF inference engine.
//!
//! ```no_run
//! use nibble::chat::{ChatOptions, Message};
//! use nibble::sampler::{Sampler, SamplerConfig};
//! use nibble::{Model, SessionOptions};
//!
//! // SAFETY: the model file is not modified while it is mapped.
//! let model = unsafe { Model::load_mmap("Qwen3-0.6B-Q8_0.gguf")? };
//! let template = model.chat_template().expect("known chat template");
//! let prompt = template.render(&[Message::user("Hello!")], ChatOptions::default())?;
//! let tokens = model.tokenizer().encode(&prompt, true);
//!
//! let mut session = model.session(SessionOptions::default());
//! let mut sampler = Sampler::new(SamplerConfig::default());
//! session.generate(&tokens, &mut sampler, 256, |token, _logits| {
//!     print!("{}", model.tokenizer().decode(&[token], false));
//!     true
//! })?;
//! # Ok::<(), nibble::Error>(())
//! ```
//!
//! Status: CPU engine for Qwen3 (`qwen3`) and Qwen3.5 (`qwen35`) on F32, F16,
//! BF16, Q8_0 and the K-quants Q3_K, Q4_K, Q5_K and Q6_K, with AVX2 and NEON
//! kernels. On x86 the K-quant dot products still use the scalar reference:
//! their AVX2 kernels are not written yet.

pub mod chat;
mod cpu;
mod error;
mod model;
mod quant;
pub mod sampler;
mod session;
pub mod tokenizer;

pub use error::{Error, Result};
pub use model::{Architecture, Model, ModelConfig};
pub use nibble_gguf as gguf;
pub use nibble_json as json;
pub use session::{CHECKPOINT_TAIL, MAX_CHECKPOINTS, Session, SessionOptions, StopReason};
