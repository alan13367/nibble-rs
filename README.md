# nibble-rs

A tiny, fast, cross-platform inference engine for GGUF models, written in Rust
with **zero external dependencies**.

> Early development. Today: a zero-copy GGUF reader and `nibble inspect`.
> Next: a CPU reference engine for Qwen3.5, an OpenAI-compatible server,
> then SIMD kernels and CUDA / Metal backends with CPU fallback.
> See [PLAN.md](PLAN.md).

## Build

Requires Rust 1.100 or newer.

```sh
cargo build --release
./target/release/nibble inspect path/to/model.gguf
```

`nibble inspect` prints the header, metadata and tensor table:

```
file     models/Qwen3.5-0.8B-Q8_0.gguf (795.0 MiB, memory-mapped)
format   GGUF v3, alignment 32, data at byte 10962464
model    Qwen3.5-0.8B [qwen35]
tensors  335 tensors, 772.85M parameters, 784.5 MiB
types    Q8_0 x195 (782.6 MiB), F32 x140 (1.9 MiB)
```

Options: `--all` (no truncation), `--no-metadata`, `--no-tensors`, `--read` (no mmap).

## Layout

| Crate | Purpose |
|---|---|
| `crates/nibble-gguf` | GGUF parser, memory mapping, writer |
| `crates/nibble` | The engine (library) |
| `crates/nibble-cli` | The `nibble` binary |

Local models go in `models/`, which is ignored by git.
