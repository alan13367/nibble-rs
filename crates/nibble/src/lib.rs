//! nibble: a tiny, dependency-free GGUF inference engine.
//!
//! Early development: this crate currently re-exports the GGUF reader.
//! The model, kernels, tokenizer and backends land here next; see `PLAN.md`.

pub use nibble_gguf as gguf;
