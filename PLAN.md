# nibble-rs plan

A tiny, fast, cross-platform GGUF inference engine in Rust with **zero external
dependencies**: CUDA on NVIDIA, Metal on Apple Silicon, a CPU fallback
everywhere, an OpenAI-compatible server, and a library + CLI.

This is a living document: update it when a decision changes.

## Status (2026-11-25)

| Area | State |
|---|---|
| GGUF reader, mmap, writer, `nibble inspect` | Done |
| CPU engine | `qwen3` and `qwen35` on F32/F16/BF16/Q8_0, Q3_K/Q4_K/Q5_K/Q6_K and IQ4_XS (NEON kernel; scalar on x86). AVX2 kernels on x86 (K-quant dots still scalar there; Rosetta can check new ones for correctness, speed needs a real x86 machine), NEON kernels on AArch64 (`dotprod` and `fp16` when the CPU has them, ggml's `vmull` fallback otherwise), a persistent thread pool per session, grouped and fused matvecs, parallel DeltaNet and attention heads. On Apple Silicon the default thread count is the performance cores (`hw.perflevel0.physicalcpu`) |
| Correctness | Tokenizer ids identical to llama.cpp on 7 tricky strings for both models. Greedy output identical to llama.cpp b12359 (release builds, CPU, `-fa off --no-repack -ub 1`) on 29 of 30 prompt/model pairs on an M4 Max (6 prompts × the 5 local files, 64 tokens); the other is a BF16 near-tie (logit gap 0.003) caused by nibble's faster bf16 dot. The x86 AVX2 path, run under Rosetta, is identical to the macOS x64 release on all 18 BF16/Q8_0 pairs. Kernels, RoPE and the activation encoders are checked bit for bit against ggml's own functions. The 5600X comparison (21/24 against b12675) predates the attention and fused-multiply-add fixes and has to be rerun |
| Speed (5600X, 6 threads, tg128) | Faster than llama.cpp b12675 on every local file: Qwen3-0.6B Q8_0 59.0 tok/s (llama.cpp 56.0), Qwen3.5-0.8B Q8_0 47.6 (41.0), Qwen3-0.6B BF16 34.6 (33.7), Qwen3.5-0.8B BF16 27.1 (25.2). Measured before the attention and fused-multiply-add fixes; to be rerun |
| Speed (M4 Max, 10 performance cores, tg128, 5 reps) | Against the b12359 release build on the CPU (`llama-bench -ngl 0 -dev none -fa 0`, repacking on as by default): Qwen3-0.6B BF16 172.4 tok/s (llama.cpp 28.9), Q8_0 240.4 (238.3 ± 7.8, a tie), Qwen3.5-0.8B Q8_0 192.1 (184.2), Q4_K_M 233.4 (212.5), Q3_K_M 250.6 (208.9). Making the numerics exactly ggml's left decode speed unchanged. Before NEON the scalar kernels were 23.5 / 63.7 / 54.2 on the three files that ran. 14 threads (P+E) is slower than 10 on both engines, so the default is the 10 performance cores |
| Metal backend | Done (Phase 6) and the default (`auto`): `qwen3` and `qwen35` on every supported weight type, zero-copy weights (split across buffers past the device limit), a few command buffers per token. Greedy output identical to llama.cpp b12359's Metal path (`-ngl 99 -fa off -ub 1`) on 29 of 30 short prompt/model pairs, all 7 local files after a 4,557-token prompt, and all 12 runs of the two 9B files on an M4 Max. The exception is a tie on Qwen3-0.6B BF16 (top two logits 0.001 apart) where llama.cpp's own CPU path takes nibble's side. Against nibble's CPU path the tokens differ only where llama.cpp's CPU and Metal paths differ too; see Phase 6. Tested on the M4 Max only |
| Speed (M4 Max GPU, 32 cores, tg128, 3 reps) | Faster than `llama-bench -ngl 99` (b12359) on every local file, flash attention off / on, at an empty context and at 4096 tokens (`nibble bench -d 4096`, llama-bench `-d 4096`). Empty: Qwen3-0.6B BF16 244.4 tok/s (202.5 / 215.0), Q8_0 373.7 (267.0 / 300.9), Qwen3.5-0.8B Q8_0 318.1 (244.6 / 241.6), Q4_K_M 392.9 (249.9 / 267.1), Q3_K_M 394.4 (265.2 / 256.0), Qwen3.5-9B UD-Q4_K_XL 59.4 (49.3 / 45.2), 9B Q8_0 33.2 (31.6 / 32.7). At 4096: 178.1 (126.0 / 167.5), 237.9 (154.3 / 228.0), 270.6 (217.2 / 224.0), 322.0 (245.7 / 255.8), 323.1 (236.2 / 244.1), 51.6 (49.0 / 43.9), 35.7 (31.1 / 32.6). The 9B Q4 file reads about 326 GB/s, 84% of the measured ~390 GB/s; the 0.6–0.8B files 46–71%, bound by the chain of about 180 dependent dispatches per token (4–6 µs each) |
| Qwen3.5-9B (UD-Q4_K_XL 5.6 GiB and Q8_0 8.9 GiB, M4 Max) | Both run on both backends (UD-Q4_K_XL mixes IQ4_XS into its K-quants; 32 value heads share 16 key heads). Greedy output identical to llama.cpp b12359 on all 6 comparison prompts for both files on the CPU (`--no-repack -ub 1`) and on Metal (`-ngl 99 -fa off -ub 1`), and after a 4,557-token prompt on Metal. Metal decode in the row above. CPU decode (UD-Q4_K_XL) 27.1 tok/s at 10 threads (llama.cpp 35.6 with repacking): the CPU gap at this size is not investigated yet. Prompt: 65.7 tok/s on Metal one token at a time (llama.cpp 685 batched) |
| CLI | `inspect`, `tokenize`, `run` (with `--probs N`), `chat`, `serve`, `bench`; `--backend auto\|cpu\|metal` (default auto: Metal where available, with the reason printed when it falls back to the CPU); `--draft N` (MTP drafting, with acceptance printed by `run`) |
| OpenAI server | Complete for Phase 3: tools, logprobs, recurrent-state checkpoints. Verified with curl, socket tests, and the official OpenAI Python client (3.58 on both models on the CPU; 3.59 on Metal with Qwen3-0.6B, Qwen3.5-0.8B and the 9B: models, multi-turn chat, reasoning, streaming with usage, tool calls and results, streamed tool calls, logprobs, stop, length, completions, seeded sampling, 6 concurrent async requests, context overflow). Where the CPU and Metal answers differ (a tool call on Qwen3.5-0.8B), llama-server's CPU and Metal paths give the same first-token log-probabilities as nibble's to 0.001. Open WebUI not tried |
| Tests | 232 pass in `cargo test --workspace` (no model files needed; the Metal tests skip where no Metal device exists and fail on a Mac whose GPU cannot start the backend); 202 as x86_64 under Rosetta before IQ4_XS (not rerun since); 10 reference tests on real models (9 pass here with `--ignored`; `qwen35_bf16` needs a file that is not on this machine) |
| Not started | AVX2 kernels for the K-quants (scalar on x86 until an x86 machine can test them), CUDA |
| In progress | Phase 7: batched evaluation and MTP speculative decoding (see its section). Done: one forward pass per batch on both backends (bit-identical to one token at a time), DeltaNet rollback within a short batch, `Session::eval_all`, the Qwen3.5 MTP head, speculative `generate` (`--draft N`; same tokens as without drafts), multi-input Q8_0 kernels on both backends. Qwen3.5-4B Q8_0 on the M4 Max, chat prompt, 128 tokens: Metal 73.1 tok/s plain, 126.7 with 2 drafts, 130.9 with 3 (llama.cpp b12722: 64.3, 114.5, 123.5); CPU 49.5, 80.7, 80.0 (llama.cpp 44.5, 54.8, 78.9). Multi-input kernels also for F16/BF16 and Q4_K/Q5_K/Q6_K (Q6_K and the floats on Metal). `nibble serve` works with coding harnesses (checked with Yawl). pp512 on Qwen3.5-0.8B Q8_0: CPU 298.4 → 924.3 tok/s, Metal 418.2 → 1276.1 (llama.cpp b12359: 1857.7 with its BLAS backend, 6924.2); tg128 CPU 221 → 240.9, Metal unchanged (315.4). Next: multi-input kernels for the other weight types, prompt GEMM |
| Git | Five commits; the Metal backend and IQ4_XS are committed |

## Decisions

| Topic | Decision |
|---|---|
| Name | `nibble-rs`. Crates `nibble-gguf` (format), `nibble-json` (JSON, shared by chat templates and the server), `nibble` (engine), `nibble-server` (HTTP API), `nibble-cli` (binary `nibble`) |
| Product | Library first. The CLI and the server are thin layers over the public API |
| Dependencies | None. Only path dependencies between workspace crates, including dev-dependencies |
| Backends | **CUDA** for NVIDIA, **Metal** for Apple Silicon, **CPU** fallback everywhere |
| Server | **OpenAI-compatible** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) with streaming |
| First models | **Qwen3-0.6B** (`qwen3`, a standard dense transformer) first, then **Qwen3.5-0.8B** (`qwen35`, hybrid DeltaNet). BF16 files are the reference, Q8_0 the first quantized path, Q4_K_M after that |
| Rust | Latest stable: edition 2024, MSRV **1.100** (checked in CI) |
| CI | GitHub Actions on pull requests and manual dispatch only, not on every push |
| Repository | No helper or generator scripts; only their output is committed |
| Delegation | Simple, well-specified tasks go to GLM-5.3 through opencode (`opencode-go/glm-5.3#max`) in an isolated directory; the result is reviewed before it is integrated. The Metal K-quant kernels (`metal/shaders/kquants.metal`) were written this way against a test harness, then checked end to end against llama.cpp |
| GPU correctness | A GPU backend matches llama.cpp's backend for the same API (Metal ↔ ggml-metal, CUDA ↔ ggml-cuda) token for token, with the same numerics (f32 activations into the weight-type matvecs). Against nibble's CPU path, which quantizes activations as ggml-cpu does, tokens may differ after near-ties, where llama.cpp's own CPU and GPU paths differ too |
| Correctness oracle | llama.cpp on the same GGUF file: greedy tokens, logits, and per-layer tensor dumps |

## Dependency policy

Everything a typical engine pulls from crates.io is written here instead, and kept small:

| Need | Usual crate | Our replacement |
|---|---|---|
| mmap | `memmap2` | `nibble-gguf::Mmap`: `mmap`/`MapViewOfFile` FFI (done) |
| f16/bf16 | `half` | Bit conversions, F16C intrinsics, and NEON `fp16` conversions when the CPU has them (done) |
| Threads | `rayon` | `cpu::ThreadPool` (done): persistent workers per session, dynamic task claiming, spin-then-park, panics propagated |
| SIMD | `wide`, `pulp` | `std::arch` intrinsics behind `is_*_feature_detected!` dispatch |
| Regex / Unicode | `regex`, `fancy-regex` | Hand-written pre-tokenizers plus committed `\p{L}`/`\p{M}`/`\p{N}` range tables (done) |
| Chat templates | `minijinja` | Built-in ChatML, Qwen3 and Qwen3.5 formatters; family detected from `tokenizer.chat_template` (done) |
| CLI args | `clap` | Hand-rolled parser (done) |
| CUDA | `cudarc` | Driver API bindings, library loaded at runtime |
| Metal | `metal`, `objc2` | Objective-C runtime FFI (`objc_msgSend`) against the system frameworks |
| HTTP / JSON | `hyper`, `axum`, `serde_json` | `std::net` HTTP/1.1, Server-Sent Events, own JSON parser and writer (done) |
| Property tests | `proptest` | Small xorshift/PCG generators and loops in tests (done) |

GPU code needs a compiler only at development time, never in `cargo build`:

- **CUDA kernels** are compiled offline with nvcc into a fatbin. The exact
  nvcc command is documented next to the kernels; there is no build script.
  The fatbin is committed and embedded with `include_bytes!`. Building nibble
  needs no CUDA toolkit; only editing kernels does.
- **Metal shaders** are embedded as MSL source and compiled by the OS at startup.
  No build tool at all.

## Backends

The forward pass is written once against an op-level `Backend` trait
(`backend.rs`: `matvecs`, `swiglu`, `add_rmsnorm`, `attention`, `delta_net`,
`logits`, ...), generic over the backend's vector and KV-cache types. Each
backend implements the ops on its own buffers. Ops are coarse where a GPU
can fuse them: `attention` covers the Q/K norms, RoPE, the cache append and
the gate, `delta_net` the convolution, the delta rule and the gated norm.
The CPU implementation of each op is exactly the sequence of CPU kernels it
replaced (checked: greedy output and the top-3 logits at every step
unchanged on 30 runs). The CPU backend is the fallback everywhere.

`BackendKind::Auto`, the default, picks in order (only Metal and the CPU
exist so far):

1. **Metal** on Apple Silicon.
2. **CUDA** if the NVIDIA driver library loads and a device of compute capability 7.5+ exists.
3. **CPU** otherwise. The session keeps why each GPU backend was skipped
   (`Session::fallback_reason`), and the CLI prints it.

`--backend cpu|cuda|metal` forces a choice. A binary built anywhere starts on
any machine: GPU libraries are opened at runtime, never linked.

### CUDA (Windows, Linux)

- **Driver API only**, loaded at runtime from `nvcuda.dll` / `libcuda.so.1`,
  which ship with the NVIDIA driver. Entry points are resolved through
  `cuGetProcAddress`, which avoids the `_v2` symbol-name traps.
- **Kernels in CUDA C++** (`crates/nibble/src/cuda/kernels/*.cu`), built into
  one fatbin. It contains SASS for sm_75, 80, 86, 89, 90, 100 and 120, plus
  PTX for compute_120 so future GPUs can JIT-compile it.
- **GPU coverage:** CUDA 13 supports Turing (RTX 20 / GTX 16) and newer.
  Pascal and older use the CPU backend.
- **No cuBLAS:** all kernels are hand-written.
  - `dp4a` int8 matvec for Q8_0/Q4_K/Q6_K, plus a BF16 matvec
  - fused norm/RoPE kernels and decode attention
  - conv1d + DeltaNet step, with the state kept in VRAM
  - logits and argmax/top-k on the GPU
  - tensor-core (`mma.sync`) GEMM for prompt processing
- **One CUDA graph per token:** capture the per-token sequence once and replay it.
  At 0.8B, about 24 layers × ~15 kernel launches would otherwise dominate the time.
- **Weights live in VRAM:** they are copied once at load (0.8B Q8_0 is 0.8 GB of the 3070 Ti's 8 GB).
  Partial offload for models larger than VRAM comes later.
- **Testing:** done locally on the RTX 3070 Ti (CUDA 13.4 toolkit, driver 621.24).
  GitHub-hosted runners have no GPU.

### Metal (macOS, iOS)

- **Objective-C runtime FFI:** `objc_getClass`, `sel_registerName`, and `objc_msgSend`
  called with the exact function type of each call (required on arm64). Links
  only the system Metal and Foundation frameworks.
- **MSL shaders compiled at startup** with `newLibraryWithSource`. Precompiled
  `.metallib` files only if startup time matters.
- **Zero-copy weights:** with unified memory, `newBufferWithBytesNoCopy` wraps
  the memory-mapped file directly. There is no upload and no second copy.
  A buffer holds at most `maxBufferLength` bytes (22.6 GB on the 36 GB M4
  Max), so a larger file is wrapped as several buffers that overlap by its
  largest tensor, as llama.cpp does; every tensor lies whole in one
  (`metal/weights.rs`). A model read into memory is copied the same way.
  `NIBBLE_METAL_MAX_BUFFER` lowers the limit to test the split: the 9B file
  in 1 GiB pieces gives the same tokens at the same speed.
- simdgroup reductions for matvec, `simdgroup_matrix` for prompt GEMM.
- **Runtime loading:** `libobjc`, Foundation, CoreGraphics (without it
  `MTLCreateSystemDefaultDevice` returns nil in command-line programs) and
  Metal are opened with `dlopen`; nothing is linked.
- **Kernels** (`metal/shaders/*.metal`, concatenated and compiled when a
  session starts) transcribe ggml-metal's: the matvecs keep its simdgroup
  layout, rows per threadgroup and reduction order, so greedy output matches
  llama.cpp's Metal path exactly. Each matvec kernel can also add its result
  to the output (the residual connection) or compute a gate and up row pair
  and write their SwiGLU, which changes no row's arithmetic. The other
  kernels are nibble's own, fused per op (see Backends). Attention: one
  threadgroup per key head and chunk of positions computes the norm, RoPE,
  cache store, scores, softmax and value sum for all the query heads that
  share the key head, so each cached key and value is read once per token
  (`half4` loads); chunks of 32–256 positions aim at about 512
  threadgroups, and past one chunk a reduce dispatch combines them. It is
  instantiated for up to 8 query heads per key head and heads of up to 512
  values (256 with 8 query heads). DeltaNet: the state update split over
  row groups, then the gated norm.
- **Encoding:** a concurrent compute encoder with a buffer barrier before
  each dependent dispatch; grouped matvecs share one barrier. A token's
  first command buffer is committed after 8 dispatches, so the GPU starts
  while the rest is encoded, then one every 64. Tokens whose logits are not
  needed are committed without waiting; a wait polls for up to 20 ms before
  it blocks (waking from `waitUntilCompleted` cost about 8% of decode
  speed). Every committed command buffer is kept until it is seen to have
  finished (at most 64, the queue's default limit), and the first failure
  is returned by the next op that returns a result. An op that cannot be
  encoded records an error that the end of the token returns; shapes the
  kernels do not cover are refused when the session starts, so `Auto` falls
  back to the CPU, and the CLI says why.
- **Testing:** on the M4 Max only (macOS 27.2); other Apple GPUs are
  untested. `metal/tests.rs` checks every matvec mode and embedding kernel
  against the CPU reference on random rows, also with the weights split
  across buffers, and the attention kernels against the CPU backend over
  300 positions for 5 head layouts and 3 chunkings; the synthetic
  `qwen3`/`qwen35` tests run on Metal too (skipped without a device, and
  failing on a Mac whose GPU cannot start the backend).
- **Where decode time goes** (Qwen3.5-0.8B Q4_K_M, 2.6 ms per token): the
  GPU is busy about 90% of it; the rest is the gap between tokens (wake,
  submit latency). On the GPU the logits take 0.55 ms at full bandwidth, and
  each layer about 70 µs against 35 µs of weight streaming: eight dependent
  steps of 4–6 µs each. Tried and measured slower or no faster, so not kept:
  each matvec threadgroup normalizing its input itself (the prologue delays
  every threadgroup's weight loads: 375 → 307 tok/s); the last threadgroup
  of a residual product computing the next norm with device-scope atomics
  and fences (bit-exact, but 375 → 343); four rows per simdgroup instead of
  two; other simdgroup counts; barriers on named buffers instead of scope
  barriers; a serial encoder.
- **Long contexts:** attention used to read the cache once per query head;
  sharing each key head's reads among its query heads took Qwen3-0.6B Q8_0
  from 119.5 to 238 tok/s at 4096 positions (its KV cache is 19 MB per
  layer there, about as much as its weights).
- **Next:** what is left is fewer dependent steps or cheaper ones. The norms
  (two per layer) and the DeltaNet gated norm are the candidates; a norm
  scale computed from per-threadgroup partial sums and applied as the next
  product loads its input would remove the norm dispatches at the cost of
  last-bit differences from llama.cpp. Bigger models are less bound by
  synchronization: their products are long enough to run at bandwidth.

### Later: Vulkan

Neither CUDA nor Metal covers AMD and Intel GPUs or Android. A Vulkan compute
backend (same `Backend` trait, loader opened at runtime) would cover them.
Not scheduled yet.

## Server: OpenAI-compatible API

`crates/nibble-server` is a library; `nibble serve -m model.gguf [--host 127.0.0.1] [--port 8080] [--api-key KEY]` runs it.

Implemented: everything below except items marked *later*. Also done beyond
the original list: `top_k`, `min_p` and `repeat_penalty` request fields, CORS
and `OPTIONS` preflight, SSE keep-alive comments while a request waits,
generation stopping as soon as the client disconnects, and 400 errors for
unsupported features (images, `n > 1`, forced tool calls).

- **Transport:** `std::net` HTTP/1.1 with keep-alive and Content-Length/chunked
  bodies. Streaming uses Server-Sent Events. Our own JSON parser and writer
  enforce size and depth limits.
- **Endpoints:**
  - `POST /v1/chat/completions`: messages, `stream`, `stream_options.include_usage`,
    `max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `stop`, `seed`,
    `presence_penalty`, `frequency_penalty`. Qwen's thinking text is returned as
    `reasoning_content` (the llama.cpp / DeepSeek convention), and
    `chat_template_kwargs.enable_thinking` toggles it.
  - `POST /v1/completions`: raw prompt completion.
  - `GET /v1/models` and `GET /health`.
  - Errors use OpenAI's `{"error": {...}}` shape.
- **Scheduling:** one generation at a time from a FIFO queue (done).
  *Later:* N parallel slots with batched decode. Each Qwen3.5 slot costs about
  19 MiB of DeltaNet state plus its KV cache.
- **Prompt cache:** the longest common prefix with the previous request is
  reused (done). DeltaNet state cannot be rewound, so the session keeps up
  to 4 checkpoints of it, taken 4 tokens before the end of each prompt as
  llama.cpp does; a new turn restores the latest one and evaluates only the
  tokens after it (done). Turn 2 of a 459-token Qwen3.5 chat went from
  13.9 s to 0.67 s. *Later:* checkpoints at the start of the last user
  message (llama.cpp's other rule), for edited messages.
- **Tool calling** (done): `tools`, `tool_choice` `auto`/`none`,
  `parallel_tool_calls`, assistant `tool_calls` and `tool` messages,
  streamed calls, `finish_reason: "tool_calls"`. The library renders tools
  and calls in each template's format (Qwen3 JSON, Qwen3.5 XML; tested
  against llama.cpp's renderings) and parses the model's calls back
  (`ChatTemplate::tool_call_parser`; Qwen3.5 arguments typed by the JSON
  Schema, as llama.cpp does). *Later:* `tool_choice: "required"` and named
  functions, which need constrained decoding.
- **Logprobs** (done): `logprobs`/`top_logprobs` (chat) and `logprobs: N`
  (completions), from the raw model distribution as llama.cpp reports them,
  up to 20 alternatives; reasoning tokens are included.
- **Safe defaults:** binds to localhost; `--api-key` enables bearer-token auth.

## Target model: Qwen3.5-0.8B

Verified with `nibble inspect` on `ggml-org/Qwen3.5-0.8B-GGUF` (Q8_0) and the
unsloth BF16 file, and checked against llama.cpp `src/models/qwen35.cpp` and
the CPU op `ggml_compute_forward_gated_delta_net`.

- **Hybrid 3:1 layout:** 24 trunk layers. Layers with `(i + 1) % 4 != 0` are
  **Gated DeltaNet** (linear attention, 18 layers); every 4th layer is **gated
  full attention** (6 layers).
- **MTP layer:** files converted with the head have `nextn_predict_layers = 1`
  and one more block (`block_count = 25`): layer 24 is the MTP (`nextn`)
  draft head, skipped in normal decoding. `convert_hf_to_gguf.py --mtp`
  instead writes the head alone to an `mtp-*.gguf` sidecar (with
  `token_embd`, `output_norm` and `output`). Of the local files only
  `Qwen3.5-4B-Q8_0.gguf` has one (`blk.32`, without `embed_tokens` or
  `shared_head_head`, so it uses the tied `token_embd` for both).
- **Dimensions:** `n_embd` 1024, FFN 3584 (SwiGLU), RMSNorm eps 1e-6.
- **Tied embeddings:** vocabulary 248,320. There is no `output.weight`, so
  `token_embd` is also the LM head.
- **Conversion already applied:** norm weights include Qwen's `1 + w` offset
  (values around 1.0), so plain RMSNorm is correct.
- **Context and RoPE:** context 262,144. RoPE base 1e7 over 64 of the 256 head dimensions (`rope.dimension_count`).

Per token, for one sequence:

```
x = token_embd[token]
for il in 0..24:
    h = rmsnorm(x) * attn_norm
    if il % 4 != 3:                                   # Gated DeltaNet
        qkv = attn_qkv · h        -> q[16×128] k[16×128] v[16×128]   (6144)
        z   = attn_gate · h       -> [16×128]
        b   = ssm_beta · h ; a = ssm_alpha · h        -> [16] each
        qkv = silu(causal_conv1d(conv_state ++ qkv, ssm_conv1d[4×6144]))
        q, k = l2_norm(q), l2_norm(k)                 per head, eps 1e-6
        beta = sigmoid(b) ; g = softplus(a + ssm_dt.bias) * ssm_a   # ssm_a = -exp(A_log)
        per head:  S *= exp(g)                        # S is 128×128 f32
                   d  = (v - Sᵀk) * beta
                   S += k ⊗ d
                   o  = Sᵀq / sqrt(128)
        o   = rmsnorm(o) * ssm_norm[128] * silu(z)    per head
        out = ssm_out · o
    else:                                             # gated full attention
        qg   = attn_q · h         -> per head [q(256) | gate(256)], 8 heads
        k, v = attn_k · h, attn_v · h                 2 KV heads (GQA 4:1)
        q, k = rmsnorm(q)*attn_q_norm, rmsnorm(k)*attn_k_norm   per head
        q, k = rope(q, k)         NeoX pairs on the first 64 dims, base 1e7
        o    = softmax(q·Kᵀ / 16) · V
        out  = attn_output · (o * sigmoid(gate))
    x = x + out
    n = rmsnorm(x) * post_attention_norm
    x = x + ffn_down · (silu(ffn_gate · n) * (ffn_up · n))
logits = token_embd · (rmsnorm(x) * output_norm)
```

What this means for the engine:

- **Recurrent state instead of a full KV cache.** Each DeltaNet layer keeps a
  16×128×128 f32 state plus a 3×6144 conv state: about 19 MiB in total, the
  same at any context length. Only 6 layers keep a KV cache: 2 heads × 256 ×
  (K+V) × 6 layers is 12 KiB per token in f16 (384 MiB at 32k context).
- **The LM head is a third of the weights** (254M of 773M parameters), so it
  is the largest single matmul per token, and the scan of 248k logits has to be fast.
- **The DeltaNet update touches 18 MiB of state per token.** That fits in the
  5600X's 32 MiB L3. Prompt processing needs the chunked (parallel) form later;
  decode is a sequential update.
- **RoPE:** the file uses interleaved M-RoPE (`rope.dimension_sections` = [11, 11, 10, 0]).
  For text-only input all position components are equal, so it reduces to
  ordinary NeoX RoPE on 64 dims (confirmed in ggml's `ggml_mrope_cache_init`
  and by matching llama.cpp's output).
- **Tokenizer:** GPT-2 byte-level BPE with 247,587 merges and
  `tokenizer.ggml.pre = "qwen35"`. `add_bos_token = false`, eos is 248046.
  The pre-tokenizer regex (from llama.cpp):
  `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
- **Decode ceilings:** decode reads every weight once per token, so speed is
  capped by memory bandwidth.
  - Dev machine: Ryzen 5 5600X (AVX2+FMA, no AVX-512) with dual-channel
    DDR4-3600 (about 40–45 GB/s usable), and an RTX 3070 Ti (608 GB/s).
  - Apple Silicon ranges from about 68 to 800 GB/s depending on the chip.

  | File | Bytes per token | CPU (5600X) | CUDA (3070 Ti) |
  |---|---|---|---|
  | BF16 | ~1.5 GB | ~28 tok/s | ~400 tok/s |
  | Q8_0 | ~0.80 GB | ~50–55 tok/s | ~750 tok/s |
  | Q4_K_M | ~0.53 GB | ~80 tok/s | ~1100 tok/s |

  On GPUs, kernel launches and synchronization will cap speed long before
  bandwidth does at this size. That is why CUDA graphs and one Metal command
  buffer per token are design requirements.

## Architecture

Current layout; `(planned)` entries do not exist yet.

```
crates/
  nibble-gguf/     GGUF parser (zero-copy), Storage (mmap / aligned buffer), writer
  nibble-json/     JSON parser and writer, plus Python-style output for Jinja's `tojson`
  nibble/          the engine
    quant.rs       weight formats: f16/bf16, dequantize, dot types and activation encoding, dot products (scalar reference)
    cpu/
      mod.rs       Cpu (pool + kernels): single, grouped and fused SwiGLU matvecs; norms, RoPE
      pool.rs      persistent thread pool: dynamic task claiming, spin-then-park, panic propagation
      kernels.rs   kernel dispatch (the one place that lists instruction sets)
      avx2.rs      AVX2/FMA/F16C kernels transcribed from ggml, behind a CPU-feature token. K-quant dots call the scalar reference until x86 kernels are tested
      neon.rs      NEON kernels (dotprod and fp16 when detected; vmull / scalar f16 widening otherwise)
      scalar.rs    scalar references of the vector kernels
      topology.rs  default thread count: performance cores on Apple Silicon, else one per physical core
    model/
      mod.rs       Model, the architecture registry and dispatch (the only place that matches on architectures)
      config.rs    shared hyperparameters + typed `{arch}.*` metadata access
      bind.rs      tensor binding with shape and type checks
      layers.rs    shared blocks: embedding/output head, GQA attention, KV cache, SwiGLU
      qwen3.rs     Qwen3
      qwen35.rs    Qwen3.5 (Gated DeltaNet + gated attention)
    tokenizer/     facade (vocab, specials, decoding) + bpe.rs, pretokenize.rs, unicode tables
    chat/          ChatML / Qwen3 / Qwen3.5 templates with tools (qwen.rs), tool-call parser (tool_calls.rs)
    sampler.rs     greedy, temperature, top-k/top-p/min-p, penalties, logprobs
    session.rs     Session: tokens, cache, recurrent-state checkpoints, generation loop
    backend.rs     the op-level Backend trait, BackendKind, the parameter arena
    cpu/neon.rs    NEON kernels. simd128 for WASM is still planned
    cuda/          (planned) driver API loader + bindings, memory, graphs; kernels/*.cu + nibble.fatbin
    cpu/backend.rs the CPU Backend (cpu/attention.rs: the f16 KV cache and attention)
    metal/         objc.rs (runtime loading, typed objc_msgSend), device.rs (device, buffers,
                   pipelines, command buffers), kernels.rs (argument structs, dispatch),
                   mod.rs (the Metal Backend), shaders/*.metal
  nibble-server/   http.rs, openai.rs, worker.rs, server.rs, text.rs (stop sequences, reasoning split)
  nibble-cli/      inspect, tokenize, run, chat, serve, bench
```

Design rules:

1. **No compute graph.** The forward pass is plain Rust calling backend ops on
   preallocated buffers: nothing is allocated per token, and there is no scheduler.
2. **Weights stay quantized.** On CPU and Metal they are views into the memory
   map, never copied. Block structs have alignment 1, so byte casts are always valid.
3. **Quantized dot products, the llama.cpp way.** Quantize the activation once
   (Q8_0/Q8_K), then integer dot products: `maddubs`/`madd` on AVX2, `sdot` on
   NEON, `dp4a` on CUDA, `simd_sum` reductions on Metal.
4. **`unsafe` is confined** to `nibble-gguf::storage`, `cpu/` SIMD and the
   thread pool, the default thread count's OS calls, and the `cuda/` and
   `metal/` FFI. Every block gets a `// SAFETY:` comment, and every kernel is
   tested against the scalar version.

## Roadmap

| Phase | Deliverable | Done when |
|---|---|---|
| **0. Scaffold** ✅ | Workspace, profiles, lints, CI (Windows/Linux x64+arm64/macOS, MSRV, wasm32, Miri) | All CI checks pass locally; the GitHub workflow has not run yet (it triggers on PRs and manual dispatch) |
| **1. GGUF + `inspect`** ✅ | Zero-copy parser, mmap, writer, `nibble inspect` | Every local GGUF (0.8B–40B, 12 quant types, unknown types) parses; 0.8B in ~70 ms |
| **2a. CPU reference: Qwen3** ✅ | F32/F16/BF16/Q8_0 weights with ggml's activation formats, F16 KV cache; byte-level BPE with `qwen2` and `qwen35` pre-tokenizers; Qwen3 ChatML; sampler; `nibble run`, `chat`, `tokenize` | Tokenizer ids match llama.cpp on every test string (Qwen3 and Qwen3.5); greedy output matches llama.cpp (`-fa off`) except at near-ties that also flip inside llama.cpp |
| **2b. CPU reference: Qwen3.5** ✅ | Gated DeltaNet layers (conv1d, recurrent state), gated attention with partial RoPE, Qwen3.5 chat template; architecture registry + shared layers so new models only touch `model/` | BF16 matches llama.cpp on 6/6 prompts, Q8_0 on 5/6 (the other is a tie that flips inside llama.cpp); synthetic model matches an independent f64 implementation over whole sequences |
| **3. OpenAI server** ✅ | `nibble-server` (std-only HTTP/1.1 + SSE, JSON in `nibble-json`), chat/completions/models/health, streaming with usage, `reasoning_content`, stop sequences, API key, CORS, FIFO worker with prefix reuse, tool calling, logprobs, recurrent-state checkpoints; `nibble serve` | Verified with curl, socket tests and the official OpenAI Python client on Qwen3 and Qwen3.5 (chat, streaming, reasoning, logprobs, tool round trips streamed and not, completions, errors). Open WebUI not tried |
| **4. CPU speed** (x86 AVX2 and Apple Silicon NEON done) | AVX2 kernels for F32/F16/BF16/Q8_0, NEON kernels for those plus Q3_K/Q4_K/Q5_K/Q6_K (and `Q8_K` activations), persistent thread pool, grouped and fused matvecs, parallel DeltaNet and attention heads, performance-core default on Apple Silicon; `nibble bench`. Still to do: AVX2 K-quant kernels, measured on x86 | Met on the 5600X (Q8_0 tg128 5–16% faster than llama.cpp, BF16 3–8% faster, before the numerics fixes) and on an M4 Max (NEON tg128 faster than llama.cpp b12359 on the CPU on four of the five local files and tied on Qwen3-0.6B Q8_0, at 10 performance cores; output identical to it on 29 of 30 comparison runs) |
| **5. CUDA backend** | Loader, bindings, memory, fatbin kernels, CUDA graphs, CPU fallback | Same greedy tokens as CPU; decode at least as fast as llama.cpp CUDA on the 3070 Ti |
| **6. Metal backend** ✅ | objc bindings, zero-copy weights (split across buffers past `maxBufferLength`), MSL kernels, a few command buffers per token, every one checked for errors; `qwen3`/`qwen35` on every supported type with the residual, SwiGLU, attention and DeltaNet fusions; attention that reads each cached key and value once per token for all query heads sharing it; `auto` as the default. Left for later: other Apple chips and older macOS versions (untested; `auto` falls back to the CPU and prints why when the backend does not start), fewer dependent steps per token on small models | Same greedy tokens as llama.cpp's Metal path (met: 29/30 short with one 0.001 tie, 7/7 after a long prompt, 12/12 on the 9B files), and as the CPU except at near-ties; decode at least as fast as llama.cpp Metal (met on every local file, flash attention off and on, at an empty context and at 4096 tokens) |
| **7. Batched evaluation and MTP** (in progress) | One forward pass for a batch of tokens on CPU and Metal (CUDA when Phase 5 exists), DeltaNet rollback within a batch, the Qwen3.5 MTP head and speculative decoding with it, prompt GEMM (CPU tiles, Metal simdgroup matrices), parallel server slots. Steps in the Phase 7 section | pp512 within 25% of llama.cpp on each backend; speculative decoding gives the same tokens as plain decoding and decodes faster on the MTP files |
| **8. Breadth** | Qwen3.5 2B/4B, `llama` and further dense archs, Vulkan for AMD/Intel, an AVX2 IQ4_XS kernel. Done: IQ4_XS (CPU and Metal) and the local Qwen3.5-9B UD-Q4_K_XL file, whose 32 value heads share 16 key heads (the grouping matches llama.cpp) | The local 4B/9B `qwen35` files run correctly on every backend |
| **9. Packaging** | Stable Rust API, C ABI (`cdylib` + header), WASM (simd128, in-memory weights), prebuilt release binaries | — |

## Phase 7: batched evaluation and MTP speculative decoding

Speculative decoding pays only when the target model checks a draft of n
tokens for about the cost of one: one forward pass over a batch, reading
each weight once. Prompt processing needs the same pass, so it comes first.

### What llama.cpp does (source at b12723, read 2026-11-25)

- **The MTP head** (`src/models/qwen35.cpp`, `graph_mtp`) is one block after
  the trunk, `blk.{n_layer}`: `nextn.eh_proj` `[2·n_embd, n_embd]`,
  `nextn.enorm`, `nextn.hnorm`, a gated-attention block with its own KV
  cache, a SwiGLU, and optional `nextn.embed_tokens` / `shared_head_head` /
  `shared_head_norm` (else `token_embd`, `output`, `output_norm`). At
  position p it reads the token at p and `h_nextn`, the trunk's final hidden
  state at p − 1 *after* `output_norm`:
  `x = eh_proj · [rmsnorm(e) * enorm ; rmsnorm(h) * hnorm]` (embedding
  first), then the block, the head norm and the LM head, predicting the
  token at p + 1. Its own normalized output is the `h` of the next draft step.
- **The loop** (`common/speculative.cpp` `draft-mtp`,
  `tools/server/server-context.cpp`): after every target batch (the prompt,
  or a verification), the head runs over the same tokens, each paired with
  the target's `h` of the position before (the first with the `h` carried
  over from the previous batch); that fills its KV cache, and no logits
  are computed. Drafting runs the head on the last sampled token, then on
  each draft, up to `--draft-max` (default 3), taking the top token and
  stopping when its probability (over the top 10) is below `--draft-p-min`
  (default 0). The target evaluates `[last, d1..dn]` as one batch with
  logits for every row; tokens are accepted while the target's own sample
  equals the draft (sample-and-match: exact for any sampler, since every
  kept token is sampled from the target). Rejected positions are dropped
  from both KV caches. Enabled with `--spec-type draft-mtp` (or `--mtp`).
- **Recurrent rollback** (`rs_replay`, `src/models/delta-net-base.cpp`): a
  batch of 2 to `n_rs_seq + 1` tokens leaves the committed DeltaNet state
  and convolution history as they were and stores the batch's inputs (conv
  rows, k, v, gate, beta) as pending. The next batch first replays the
  pending tokens that were kept, then runs its own. Rejecting draft tokens
  costs nothing, and no state is copied.
- **DeltaNet over a batch** (`ggml_compute_forward_gated_delta_net`) runs
  the tokens one after another per head, exactly as decoding does; ggml-cpu
  has no chunked form.
- **Batched products round differently from decoding:** ggml-cpu uses
  llamafile's tinyBLAS (F32, F16, Q8_0) or the repacked GEMMs, and
  llama-bench on macOS also lists the BLAS (Accelerate) backend; ggml-metal
  uses `mul_mv_ext` for 2–8 columns (4–8 for K-quants) and `mul_mm`
  (simdgroup matrices on f16 tiles) above that.

### Decisions

- **One forward pass for n tokens.** The `Backend` ops take a batch, set
  by `begin(pos, n)`; decoding is n = 1. Activation vectors hold up to
  `n_batch` rows. `Session::eval` splits its tokens into batches (and at the
  checkpoint position).
- **CPU batches are bit-identical to decoding.** Every output element comes
  from the same kernel call as when the tokens are evaluated one at a time:
  a product reuses each block of weight rows for every column of the batch,
  each token's attention runs over its own causal row with the same padding,
  and DeltaNet runs the tokens one after another. So prompt processing still
  compares with llama.cpp at `-ub 1`, and speculative decoding gives exactly
  the tokens of plain decoding, greedy or seeded (the sampler sees the same
  logits in the same order). That equality is the main test of the MTP loop.
- **Metal batches up to 8** use multi-column versions of the current matvec
  kernels, each column reduced as now (bit-identical to decoding, so the
  same test holds), not ggml's `mul_mv_ext`. **Prompt batches above 8** use
  a transcription of ggml-metal's `mul_mm`, compared with llama.cpp's Metal
  path at its default `-ub`.
- **DeltaNet rollback as llama.cpp's `rs_replay`**, not state snapshots:
  copying 19 MiB of state per verification would cost about what the drafts
  save on the 0.8B model.
- **MTP needs a file with the head:** in the main file
  (`nextn_predict_layers`) or as an `mtp-*.gguf` sidecar. Both layouts are
  supported; the draft runs on the same backend as the target.

### Steps

1. ✅ **Batched forward.** `Backend` ops over n tokens (`begin(pos, n)`),
   `Session::eval` in batches of `n_batch` (default 512, cut at the
   checkpoint), and a CPU product in which each task takes a range of
   output rows for every token, multiplying each 16 KiB block of weight
   rows with every token's input (`ThreadPool::for_each_segment_chunk`).
   Metal runs a batch one column at a time. Batched and per-token logits
   are bit-identical (unit tests on every weight type and kernel set, the
   synthetic models on both backends with batches of 2, 3, 5 and 64), and
   greedy output with the top-3 logits is unchanged on 69 runs (the 6
   comparison prompts on the 5 small files, CPU and Metal, and the 9B and
   4B files). pp512 above; Metal gained by committing a batch, not a
   token, at a time.
2. ✅ **DeltaNet rollback.** The convolution ring has `conv_kernel - 1 +
   MAX_PENDING` rows; a batch of 2 to 16 tokens runs from the committed
   state on a work copy, keeps its inputs in the ring and its beta/alpha
   projections in `pending_ba`, and the next batch first applies the kept
   ones with the same kernels (bit-identical). Checkpoints save the window
   and the kept pending tokens. `Session::eval_all` (up to 16 tokens, one
   pass, every token's logits, no checkpoint) is what drafts are verified
   with; rewinding into it replays nothing (tests on both backends).
3. ✅ **The MTP head** (main-file layout; the sidecar is not done). It
   runs on the trunk's backend with its own KV cache, its attention and
   feed-forward on the trunk's scratch buffers. After each trunk batch it
   catches up over the same tokens; a rewind into the last batch takes the
   hidden state of the new end from the rows it paired; checkpoints carry
   that hidden state. Checked against an f64 reference on the synthetic
   model (with drafts chained), bit for bit against a fresh evaluation
   after batches, rewinds and checkpoint restores, and on the 4B file by
   acceptance: with 3 drafts on the chat prompt nibble accepts 45/55 (82%)
   on Metal and 44/58 (76%) on the CPU, llama.cpp b12722's `draft-mtp`
   44/54 and 45/51 (same text). A tensor-level check with `cb_eval` is
   still to do.
4. ✅ **Speculative generation.** `SessionOptions::draft_max` (`--draft
   N`), greedy drafts, verification by sample-and-match,
   `Session::draft_stats` and `draft_logits`. Tokens, stop reason, final
   session and logits are the same with 0, 1, 3 and 15 drafts on the
   synthetic model (greedy and seeded sampling with a repeat penalty,
   every stop reason, both backends), and on the 4B file with 1, 3 and 6
   (reference test). The draft steps are chained on the device
   (`Backend::argmax` into a token slot, `embed_slot` from it) and read
   back once per pass. Still to do: `serve` reporting draft statistics,
   `bench` measuring drafting.
5. ✅ **Kernels that share each weight load across the inputs of a
   batch**, every one bit-identical to its one-input kernel (tests on
   every type and mode):
   - Metal: one dispatch per product for the whole batch. Q8_0 has 2- and
     4-input kernels (`matvec_q8_0_c2`/`_c4`, groups of inputs across the
     grid); the other types loop over the inputs inside each threadgroup,
     which reads its weights from memory once. Norms take one threadgroup
     per row; DeltaNet runs the replayed and the new tokens of a layer in
     one dispatch (`delta_net_batch`, the state in registers). Plain
     decoding keeps the one-token kernels.
   - CPU: a NEON Q8_0 tile of 2 rows by 2, 3 or 4 inputs, keeping each
     input's lanes, accumulators and scale order; the scales are widened
     with `fcvt` (`fp16`) instead of the software decode, which cost as
     much as a block's products and also made plain decoding 6% faster
     (47.2 → 50.1 tok/s on the 4B).
   - Verifying 1/2/3/4 tokens on the 4B: Metal 13.4/13.9/15.6/16.3 ms,
     CPU 19.7/21.5/23.1/26.5 ms (before: 15.4/23.9/–/43.7 and
     20.3/37.5/–/73.9). Measured and not kept: 4 rows per threadgroup
     for the Metal wide kernels (fewer threadgroups, slower), loading a
     block's weights first and the inputs one at a time (slower), two
     2-input groups instead of one 4-input group, a 2-row CPU tile for
     one input (no faster: one input is memory-bound at 10 threads).
   - Other types. Metal: F32/F16/BF16 and Q6_K have 2-input kernels
     (wider batches go in groups of 2 across the grid; 4-input versions
     measured slower), Q3_K/Q4_K/Q5_K/IQ4_XS keep the loop over inputs:
     their multi-input kernels either compiled to regrouped sums (fast
     math) or spilled registers and ran slower than the loop. Since
     bit-identity of these kernels depends on the shader compiler, the
     backend runs each wide kernel against the one-input kernel when it
     starts and stops using any that differ. CPU: NEON kernels for
     Q4_K/Q5_K/Q6_K (exact integer sums, the float update per super-block
     unchanged), F16 and BF16; Q3_K and IQ4_XS take one input at a time.
   - Verifying 1/2/3/4 tokens: Qwen3.5-0.8B Q4_K_M Metal 2.6/3.4/5.0/6.3 ms,
     CPU 4.0/5.3/6.7/9.5; Qwen3-0.6B BF16 Metal 4.1/4.5/4.9/5.3, CPU
     5.8/7.0/9.2/11.8; Qwen3.5-9B UD-Q4_K_XL Metal 16.5/22.8/–/–, CPU
     33.2/40.0/53.2/77.5. The K-quants on Metal and the CPU's 4-input
     batches are the weak spots.
   - Still to do: better K-quant multi-input kernels on Metal (llama.cpp's
     `mul_mv_ext` dequantizes to `float4x4` per lane, a layout that rounds
     differently from the one-input kernel), batched attention on Metal,
     AVX2 tiles.
6. **Prompt speed:** Metal `mul_mm` (simdgroup matrices; rounds
   differently, compared with llama.cpp's Metal path at its default
   `-ub`) and larger CPU tiles. Done when pp512 is within 25% of
   llama.cpp on each backend (on the CPU, compared also without
   llama.cpp's BLAS backend).
7. **Parallel server slots:** several sequences decoded in one batch.
8. **MTP sidecar files** (`mtp-*.gguf`) and `--draft-p-min`.

### Coding harnesses

Checked with [Yawl](https://github.com/alan13367/yawl) 0.2.1 through its
`openai-completions` provider, on Qwen3.5-4B Q8_0 with `-c 32768 --draft 3`
on Metal: reading files, editing one and running a script through its shell
tool all completed with no tool errors. Added for it: `max_model_len` and
`context_length` in `/v1/models`, `prompt_tokens_details.cached_tokens` in
`usage`, `reasoning_effort` mapped to thinking on or off (`none`/`minimal`
off), a log line per request in `nibble serve`, and a checkpoint where a
rewind that had to replay stops (a second session with the same system
prompt then reuses it instead of replaying 2,000 tokens again).

- **Prompt speed is the bottleneck:** Yawl's system prompt and tools are
  about 2,000 tokens, 8.8 s at 223 tok/s on the 4B on Metal; each tool
  result is evaluated at about 190 tok/s. Step 6 (prompt GEMM) is what
  makes harness use responsive.
- **Prefix reuse across sessions:** the first replay to a new shared
  prefix still costs a full re-evaluation; checkpoints at more points
  (llama.cpp also keeps one at the start of the last user message) would
  avoid it.

## Validation

- **Reference runs:** install a llama.cpp release build (CPU and CUDA) for logits,
  `llama-eval-callback` tensor dumps (compare per layer to find the first
  mismatch), `llama-perplexity`, `llama-bench` and `llama-server` (to compare API behavior).
- **Unit level:** each kernel against the scalar reference on random inputs; each
  quant format's dequantization against gguf-py reference outputs.
- **Tiny synthetic models in CI** (done): `GgufWriter` builds random tiny
  `qwen3` and `qwen35` models, checked against independent f64 implementations
  (whole sequences for `qwen35`, including the recurrence, partial RoPE and an
  MTP block to skip). CPU vs GPU comparisons will run locally on the 3070 Ti
  and on a Mac.
- **Comparing with llama.cpp:** use `llama-completion ... -ngl 0 --device none
  -fa off --no-repack -b 512 -ub 1`: nibble's attention matches the non-flash
  path, its kernels are the non-repacked ones, and its batches give the
  results of evaluating one token at a time. llama.cpp's `-f` strips one
  trailing newline from the
  prompt file. The commands are in `AGENTS.md`.
- **Bit-level checks:** small C programs outside the repository, linked
  against the release's `libggml-cpu` / `libllama`, run ggml's kernels on the
  tests' inputs and dump llama.cpp's tensors per layer; the tests record
  their results.
- **Server:** end-to-end tests over real sockets on a tiny model, and the
  official OpenAI Python client against the real models (done). Open WebUI
  has not been tried.
- **Real models locally:** 0.8B BF16 and Q8_0 in every phase; the 4B and 9B files from Phase 8.

## Known pitfalls

- GGUF dimensions are innermost first: `attn_qkv` is `[1024, 6144]`, i.e. 6144 rows of 1024.
- `attn_q` interleaves query and gate **per head** (`[q0 | g0 | q1 | g1 | ...]`), not as two halves.
- The conv kernel is `[4, 6144]`: the 4 taps of a channel are contiguous, oldest first.
- llama.cpp stores the DeltaNet state transposed (`M[j][i] = S[i][j]`) so rows are contiguous. Match that layout.
- The q/k head for value head `h` is `h % n_k_heads` in ggml. This is irrelevant
  at 0.8B (16/16) but matters for sizes with more V heads than K heads.
- `ssm_a` is already `-exp(A_log)`, and norm weights already include `+1`. Apply neither again.
- Skip the MTP layer (`nextn_predict_layers`) in the trunk loop.
- Qwen's tokenizer.json applies NFC normalization, but llama.cpp does not.
  Match llama.cpp first and note the difference.
- Qwen3's template thinks unless `enable_thinking` is false. Qwen3.5's adds
  `<think>\n` only when it is true, and an empty think block otherwise. Both
  are handled by `ChatOptions::enable_thinking` and `--no-think`.
- Qwen3.5 history drops the empty think block of earlier turns, so each chat
  turn diverges from the cached prompt right after `<|im_start|>assistant\n`.
  The checkpoint 4 tokens before the end of each prompt lands exactly there.
- llama.cpp passes tool-call arguments to templates as parsed objects, even
  to Qwen3's (which would accept a string), and normalizes tools to
  `{type, function: {name, description, parameters}}`; `tojson` is Python's
  `json.dumps` (`", "`, `": "`), and Qwen3.5 writes argument values with
  Python's `str` (`True`, `None`).
- ggml's AVX2 kernels reduce differently per file: `vec.cpp` uses four
  accumulators and `hadd`, `quants.c` one accumulator and `hsum_float_8`.
  Transcribe the exact one.
- llama.cpp without flash attention stores V transposed, rounds the softmax
  probabilities to f16 before the V product, and pads the cache to 256
  positions (masked to zero weight).
- Threads: on SMT CPUs use one thread per physical core. With a thread on
  every logical CPU, any other process preempts a worker mid-task and the
  op waits for it: 12 threads decode about a third slower than 6 on the
  5600X (llama.cpp loses about 20% too).
- BPE merge strings must be split at the first space by **byte** offset
  (llama.cpp's `find(' ', 1)`); splitting on chars dropped every merge with `Ġ`.
- **ggml switched F16 matmuls to f32 activations** (#31854, 8 Nov 2026,
  before b12359): F16 weights and the F16 KV cache meet unrounded f32
  inputs through `ggml_vec_dot_f16_f32`, so the attention queries and
  softmax probabilities are no longer rounded to f16.
- **clang fuses `a * b + c` in one expression** (`-ffp-contract=on`): K-quant
  block updates, `vmlaq_n_f32`, the Q4_K/Q5_K dequantizers, `ggml_ssm_conv`
  and the RoPE rotation are fused multiply-adds in llama.cpp. Each unfused
  rounding showed up as a different greedy token somewhere.
- **llama.cpp's defaults hide its vec_dot kernels:** weights are repacked
  (`--no-repack` to stop it), batched prompts use GEMM kernels (`-ub 1`),
  and llama-completion stops when a `-b` chunk of the prompt ends with an
  end-of-generation token (keep `-b` large). Its macOS arm64 release has
  `dotprod` but no fp16 arithmetic and no i8mm; Homebrew or `-mcpu=native`
  builds differ (f16 accumulation, i8mm kernels) and are not the reference.
- **CUDA:** a fatbin built by a newer toolkit than the user's driver supports
  cannot JIT its PTX. Ship SASS for every supported arch, and fall back to CPU with a clear message.
- **Metal:** `newBufferWithBytesNoCopy` needs a page-aligned pointer and a
  page-multiple length (16 KiB pages on Apple Silicon). Map the whole file and
  round the length up.
- **Metal barriers dominate small models:** a serial encoder is no faster
  than concurrent dispatch with barriers, nor are barriers on named
  buffers, nor device-scope atomics between threadgroups; only fewer
  dependent stages help. A kernel with a serial loop per thread (attention
  over every position) or few threadgroups (one per DeltaNet head) costs
  tens of µs per layer: split such work across threadgroups.
- **Compare speed at the same depth.** `nibble run` with a long prompt
  measures past its end, llama-bench `-d` at exactly that depth: use
  `nibble bench -d N`, which rewinds to the same `N` tokens before each
  repetition.
- **Benchmark in the real model.** Microbenchmarks of one layer on the GPU
  were bimodal between processes (the same kernels measured 81 or 99 µs),
  and skipping an op to time it overstates its cost. Alternate A/B runs of
  `nibble bench` were stable to about 1%.
- **GPU convolution history:** a step must not overwrite a history row that
  another threadgroup still reads (key heads are shared by several value
  heads in larger Qwen3.5 models). The history ring has one spare row per
  layer, `conv_kernel` rows in all, and each step writes the spare.
- **llama.cpp's `-f` strips one trailing newline** also for GPU runs: add one
  to prompt files that end with a newline before comparing.

## Open questions

1. **AMD/Intel GPUs:** whether to schedule the Vulkan backend.
2. **Open WebUI:** worth a manual check before calling the server done for
   UIs; it is a large install.
3. **AVX2 K-quant kernels.** On x86 Q3_K/Q4_K/Q5_K/Q6_K use the scalar dots;
   the reference tests pass that way under Rosetta, but the output is not
   bit-identical to ggml's AVX2 kernels. Rosetta can check new AVX2 kernels
   for correctness; speed needs a real x86 machine. NEON for those formats is
   done and checked bit for bit.
4. **BF16 on ARM.** ggml sums BF16 dots one product at a time in f64; nibble
   uses f32 vector accumulators, which is the one remaining difference from
   llama.cpp on the comparison prompts. An exact kernel (rows in f64 lanes,
   each summed in order) is possible but costs a large part of the speedup.
