# Agent guidelines for nibble-rs

nibble-rs is a GGUF inference engine written in Rust with no external
dependencies. These rules apply to every change in this repository.

## Ground rules

- **Do not commit, push, or create branches unless you are told to.**
- **No external crates.** `[dependencies]`, `[dev-dependencies]` and
  `[build-dependencies]` may only name crates in this workspace. If something
  seems to need a crate, write the small part that is needed. Never add a
  dependency without asking first.
- **Builds and tests work offline** with only the Rust toolchain: no network
  access, no system libraries beyond what `std` already links, and no
  external tools in `build.rs`.
- **Never put model weights, llama.cpp binaries or other downloads in git.**
  Models live in `models/` and reference tools in `tools/`; both are gitignored.
- **Treat `crates/nibble/src/tokenizer/unicode_tables.rs` as data.** It holds
  the Unicode 15.0 general-category ranges the pre-tokenizers need. Do not
  edit it by hand; regenerate it from the Unicode character database only
  when moving to a newer Unicode version.
- **No helper or generator scripts in the repository.** Keep one-off scripts
  outside it and commit only their output.
- Ask before deleting files you did not create, installing software, or
  downloading more than about 1 GB.

## Repository layout

| Path | Contents |
|---|---|
| `crates/nibble-gguf` | GGUF parser, writer and memory mapping. Knows nothing about models. |
| `crates/nibble` | The engine: weight formats (`quant.rs`), CPU execution (`cpu/`: thread pool, kernel dispatch, SIMD kernels, scalar references), models (`model/`), tokenizer, chat templates and tool-call parsing (`chat/`), sampler, sessions. |
| `crates/nibble/src/model` | `mod.rs` is the architecture registry and the only place that matches on architectures. `layers.rs` holds the shared blocks (embedding/output head, GQA attention, KV cache, SwiGLU). One file per architecture. |
| `crates/nibble-json` | Strict JSON parser and writer, including the Python-style output of Jinja's `tojson`. Used by the chat templates and the server. |
| `crates/nibble-server` | OpenAI-compatible server: `http.rs` (HTTP/1.1, SSE), `openai.rs` (request/response mapping), `text.rs` (stop sequences, reasoning split), `worker.rs` (the generation queue), `server.rs` (connections, routing, chat output). Tested end to end over sockets on a tiny model. |
| `crates/nibble-cli` | The `nibble` binary. Argument parsing and output only; logic belongs in the library. |

## Before you report a change as done

Run what CI runs:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps
cargo check -p nibble-gguf -p nibble --target wasm32-unknown-unknown
```

If you changed inference behavior, also run the llama.cpp comparison
described below.

When you report back:

- **What changed**, and the exact commands you ran with their results.
- **What you could not run, and why.** Do not claim something works on a
  platform, backend or model you did not run it on.

## Code conventions

- **Rust version:** use only APIs available in the `rust-version` set in the
  workspace `Cargo.toml`, and no nightly features.
- **Model files are untrusted input.** Library code must not panic on any file
  contents: check bounds, use checked arithmetic on sizes and offsets, and
  return an error. Panics are only for broken internal invariants, and public
  functions that can panic document it in a `# Panics` section.
- **Errors** are the crate's own error enums, with messages that name the
  tensor, metadata key or byte offset involved. Libraries never return
  `Box<dyn Error>`; the CLI may.
- **Documentation:** every public item has a doc comment (`missing_docs` warns).
  Internal items are `pub(crate)`.
- **No allocation per token in the forward pass.** Buffers belong to the
  session state and are reused.
- **Comments** explain why, not what. No commented-out code, and no TODOs
  without an owner or a linked issue.
- **Formatting:** default rustfmt settings. Keep generated or table-like code
  readable with `#[rustfmt::skip]` rather than changing the configuration.

## `unsafe`

- **Allowed only for** OS and driver FFI (memory mapping, GPU APIs, the
  Objective-C runtime), SIMD intrinsics, reinterpreting byte buffers, and
  the CPU thread pool (`cpu/pool.rs`), which lends borrowed buffers to its
  persistent workers. Keep the pool's unsafe inside its safe API.
- **Every `unsafe` block** has a `// SAFETY:` comment saying why it is sound.
  Every `unsafe fn` has a `# Safety` section. Keep blocks as small as possible.
- **Unsafe contracts stay visible.** Memory-mapped loading is `unsafe` because
  the file must not change while it is mapped. Do not wrap it in a safe library
  function; callers acknowledge it at the call site, as the CLI does.
- **Every SIMD or GPU kernel needs a scalar reference** (in `quant.rs` or
  `cpu/scalar.rs`) and a test that compares the two on random inputs. Keep the
  scalar path even when a fast path exists.
- **SIMD kernels are reachable only through a CPU-feature token** (such as
  `cpu::avx2::Avx2`, created only by runtime detection) and are dispatched in
  `cpu/kernels.rs`. Never expose a safe function that runs SIMD code without
  that proof.

## Correctness: match llama.cpp

nibble must behave like llama.cpp on the same GGUF file. When llama.cpp and the
Hugging Face reference implementation disagree, match llama.cpp and note the
difference in a comment.

- **Read the llama.cpp source before implementing** an architecture, op or
  tokenizer: `src/models/<arch>.cpp`, `ggml/src/ggml-cpu/ops.cpp`,
  `src/llama-vocab.cpp`.
- **Check tensor names, shapes, types and metadata keys with
  `nibble inspect <file>`.** Do not rely on memory or on the Hugging Face config.
- **Reference binaries** are llama.cpp release builds in `tools/llama.cpp/<build>/`.
  Download them from the ggml-org/llama.cpp GitHub releases if they are missing.
  On Apple Silicon, put the macOS x64 build in `tools/llama.cpp/<build>-x64/`
  too: with Rosetta 2 and `ROSETTA_ADVERTISE_AVX=1`, it and
  `--target x86_64-apple-darwin` builds of nibble run the AVX2 kernels, so
  the x86 path can be compared locally (correctness only, not speed).

### Tokenizer changes

Token ids must match exactly. Pass text through files, not command-line
strings, so shell escaping cannot change the input:

```sh
nibble tokenize -m models/MODEL.gguf -f case.txt --ids
tools/llama.cpp/<build>/llama-tokenize -m models/MODEL.gguf -f case.txt --ids --log-disable
```

Cover whitespace runs, newlines, contractions, digits, non-Latin scripts,
combining marks, emoji sequences, and special tokens.

### Model and numerics changes

Compare greedy output with llama.cpp:

```sh
nibble run -m models/MODEL.gguf -f prompt.txt -n 64 --greedy -c 512
tools/llama.cpp/<build>/llama-completion -m models/MODEL.gguf -f prompt.txt -n 64 \
    --temp 0 -no-cnv --no-display-prompt -c 512 \
    -ngl 0 --device none -fa off --no-repack -b 512 -ub 1
```

- **Use `-fa off`.** nibble's attention matches llama.cpp's non-flash-attention path.
- **Use `--no-repack`.** By default llama.cpp repacks Q4_K, Q5_K, Q6_K and
  Q8_0 weights and multiplies them with other kernels than the ones nibble
  transcribes.
- **Use `-ub 1`.** nibble evaluates a prompt one token at a time; llama.cpp
  evaluates a batch with GEMM kernels that round differently. Keep `-b`
  large: llama-completion stops early when a chunk of `-b` prompt tokens
  ends with an end-of-generation token such as `<|im_end|>`, which every
  chat prompt contains.
- **Use `--device none`** so that no layer or op goes to the GPU.
- **llama.cpp's `-f` strips one trailing newline** from the prompt file. Make
  sure both tools tokenize the same prompt, e.g. with `--verbose-prompt`.
- **`llama-completion` appends `[end of text]`** when generation stops on an
  end-of-generation token. Strip it before comparing.
- **Diagnose divergences with `nibble run ... --probs 3`.** It prints the top
  candidates at each step. A divergence is acceptable only if:
  - the top two logits at that step are within roughly 0.1 of each other, and
  - llama.cpp itself flips at that token between `-fa on` and `-fa off`.

  Anything else is a bug: find the first layer that differs (a
  `cb_eval` callback in a program linked against the release's `libllama`
  prints llama.cpp's tensors by name). Running nibble with
  `NIBBLE_KERNELS=scalar` tells a SIMD kernel bug from a model bug.
- **Match ggml's numerics.** Activations are converted to the weight type's dot
  type before matrix products (BF16→BF16, Q8_0→Q8_0, K-quants→Q8_K); F32
  and F16 weights, and the F16 KV cache, meet the f32 activation as it is
  (ggml stopped rounding it to f16 in November 2026, before b12359). Keep
  this when adding formats or kernels.
- **SIMD kernels transcribe ggml's kernel for the same instruction set**,
  accumulator layout and reduction order included (they differ between
  `vec.cpp` and `quants.c`), so results match llama.cpp on the same CPU.
  Attention follows llama.cpp without flash attention: V is stored
  transposed, the queries and softmax probabilities stay f32, and the
  softmax runs over the row padded with masked (`-inf`) positions, as
  llama.cpp pads the cache.
- **Fuse where clang fuses.** llama.cpp's releases are built with
  `-ffp-contract=on`, so `a * b + c` within one C expression becomes a fused
  multiply-add (`sumf += d * isum`, `vmlaq_n_f32`, `x0*cos - x1*sin`). Rust
  never fuses: write `f32::mul_add` (or `vfmaq_*`) at exactly those places,
  in the scalar reference too. A last-bit difference is not harmless: it can
  flip a Q8_0 or bf16 rounding of the next activation and change the output.
- **Check bit for bit against ggml itself.** A small C program linked against
  the release's `libggml-cpu` can run `ggml_get_type_traits_cpu(type)->vec_dot`
  and `from_float` on the inputs a test builds; record its results in the
  test. Keep such programs outside the repository.

### Tests

- **`cargo test` must pass with no model files present.**
- Build tiny synthetic models with `nibble_gguf::GgufWriter` when a test needs a GGUF file.
- Tests that need real models read them from `models/`, are marked
  `#[ignore]`, and run with `cargo test --release -- --ignored`.
- When fixing a bug, add a test that fails without the fix.

## Extending the engine

- **New architecture:**
  - add an `Architecture` variant (and to `Architecture::ALL`), a module under
    `model/`, and the dispatch arms in `model/mod.rs`. Nothing outside
    `model/` should need to change.
  - build it from the blocks in `model/layers.rs`; add a block there when a
    second architecture needs it
  - read shared hyperparameters from `ModelConfig` and the
    architecture-specific ones with `config::Meta`, using llama.cpp's defaults;
    bind every tensor with the shape-checking helpers in `model/bind.rs`
  - if the state cannot be rewound (recurrent layers), return `false` from
    `truncate_state`, and the session replays
  - add a synthetic-model test in `crates/nibble/tests/` that checks the
    forward pass against an independent f64 implementation, and reference
    tests against llama.cpp
  - verify with the comparison above, on at least one unquantized (BF16/F16)
    file and one quantized file
- **New tokenizer family** (`tokenizer.ggml.model`, e.g. SentencePiece
  `llama`): add an `Encoder` variant in its own module under `tokenizer/`. The
  vocabulary, special tokens and decoding stay in `tokenizer/mod.rs`.
- **New weight type:** write the scalar dequantize, activation encoding and
  dot product in `quant.rs` first. Test them against values from llama.cpp's
  `gguf-py` (`gguf.quants`), then enable the type in `is_supported`.
- **New pre-tokenizer:** implement llama.cpp's regex for that
  `tokenizer.ggml.pre` exactly in `pretokenize.rs`, with unit tests for each
  alternation branch, and compare ids with `llama-tokenize`.
- **New chat template family:** detect it from `tokenizer.chat_template` and
  render it natively in `chat.rs`, with tests taken from the real Jinja
  template's output. Do not add a template engine.

## Performance work

- **Measure before and after** on a release or `--profile profiling` build.
  Report the model file, quantization, thread count, prompt and generated
  token counts, and the machine.
- **Compare against `llama-bench`** on the same file and thread count when
  claiming a speed result.
- **An optimization must not change results** beyond the tolerance its kernel
  test documents. Greedy output on the comparison prompts should stay the same.
- **Prefer clear code** unless a measurement shows the faster version matters.

## GPU backends

- **Runtime loading only.** GPU APIs (the CUDA driver, Metal) are loaded or
  linked from the system at runtime, never as crates. A machine without the GPU
  stack must still run nibble on the CPU.
- **CPU is the oracle.** GPU backends must produce the same greedy tokens as
  the CPU path on the comparison prompts, and fall back to CPU when the device
  or a required feature is missing.
- **CUDA kernels are compiled offline** with nvcc into an artifact that is
  committed. `cargo build` must not need the CUDA toolkit. Test CUDA changes on
  a local NVIDIA GPU.
- **Metal code cannot be tested on the Windows development machine.** Write it
  carefully, run `cargo check --target aarch64-apple-darwin` if that target is
  installed, and label the work as untested on real hardware when you report it.
