//! An OpenAI-compatible HTTP server for nibble, built on `std` only.
//!
//! Endpoints: `POST /v1/chat/completions` and `POST /v1/completions` (both
//! with server-sent-event streaming), `GET /v1/models` and `GET /health`.
//!
//! ```no_run
//! use std::net::TcpListener;
//! use nibble::Model;
//! use nibble_server::{ServerOptions, serve};
//!
//! // SAFETY: the model file is not modified while it is mapped.
//! let model = unsafe { Model::load_mmap("Qwen3-0.6B-Q8_0.gguf")? };
//! let listener = TcpListener::bind("127.0.0.1:8080")?;
//! serve(listener, model, ServerOptions::new("qwen3-0.6b"))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod http;

mod openai;
mod server;
mod text;
mod worker;

pub use server::{ServerOptions, serve};

#[cfg(test)]
mod tests;
