# nibble-rs plan

A tiny, fast, cross-platform GGUF inference engine in Rust with **zero external
dependencies**: CUDA on NVIDIA, Metal on Apple Silicon, a CPU fallback
everywhere, an OpenAI-compatible server, and a library + CLI.

This is a living document: update it when a decision changes. Finished work
is summarized in a line; its details are in git history (PLAN.md at 8f000c7
has the full record of phases 0–6 and the first steps of phase 7).

## Status (2026-11-26)

| Area | State |
|---|---|
| Done | GGUF reader/writer and `inspect`; tokenizer (ids identical to llama.cpp); `qwen3`, `qwen35` and `qwen35moe` on every listed weight type; CPU engine (NEON for all types but Q2_K and the IQ1–IQ3 codebooks, AVX2 for F32/F16/BF16/Q8_0); Metal backend, the default (`auto`); OpenAI server with tools, logprobs and recurrent-state checkpoints; batched evaluation, DeltaNet rollback, the MTP head and `--draft N`; Metal prompt GEMM; llama.cpp's KV cache types on both backends |
| Correctness | CPU: greedy output identical to llama.cpp (`-fa off --no-repack -ub 1`) except one BF16 near-tie caused by nibble's faster bf16 dot; kernels checked bit for bit against `libggml-cpu`. Metal: identical to llama.cpp's Metal path except 0.001 ties on Qwen3-0.6B. KV cache types: 50/50 CPU runs identical, Metal identical where llama.cpp runs the same path. Tested on an M4 Max only; the x86 path only under Rosetta |
| Decode (M4 Max) | Faster than llama.cpp on every local file on Metal (empty context and 4096 tokens, `-fa` off and on) and on the CPU (tie on Qwen3-0.6B Q8_0). Qwen3.5-4B Q8_0 Metal 73 tok/s, 131 with `--draft 3` (llama.cpp 64, 124) |
| Prompt (M4 Max, pp512) | Metal level with llama.cpp on the 0.8B files, 1–3% behind on the 4B–27B, 8% behind on the 35B MoE, 11% on Qwen3-0.6B, about 45% behind with a quantized KV cache at depth (Phase 7 step 6). CPU far behind (no prompt tiles) |
| Tests | 247 pass in `cargo test --workspace` (Metal tests skip without a device); 13 reference tests on real models with `--ignored` (12 pass here; `qwen35_bf16` needs a file not on this machine) |
| Not started | CUDA; AVX2 kernels beyond F32/F16/BF16/Q8_0; NEON kernels for Q2_K and the IQ1–IQ3 codebooks; CPU prompt tiles; parallel server slots |
| To rerun | The 5600X comparisons (correctness and speed) predate the attention and fused-multiply-add fixes |
| Git | Five commits; the cache types and the prompt work since 8f000c7 are not committed |

## Decisions

| Topic | Decision |
|---|---|
| Name | `nibble-rs`. Crates `nibble-gguf` (format), `nibble-json` (JSON, shared by chat templates and the server), `nibble` (engine), `nibble-server` (HTTP API), `nibble-cli` (binary `nibble`) |
| Product | Library first. The CLI and the server are thin layers over the public API |
| Dependencies | None. Only path dependencies between workspace crates, including dev-dependencies |
| Backends | **CUDA** for NVIDIA, **Metal** for Apple Silicon, **CPU** fallback everywhere |
| Server | **OpenAI-compatible** (`/v1/chat/completions`, `/v1/completions`, `/v1/models`) with streaming |
| Rust | Latest stable: edition 2024, MSRV **1.100** (checked in CI) |
| CI | GitHub Actions on pull requests and manual dispatch only, not on every push |
| Repository | No helper or generator scripts; only their output is committed |
| Delegation | Simple, well-specified tasks go to GLM-5.3 through opencode (`opencode-go/glm-5.3#max`) in an isolated directory; the result is reviewed before it is integrated |
| GPU correctness | A GPU backend matches llama.cpp's backend for the same API (Metal ↔ ggml-metal, CUDA ↔ ggml-cuda) token for token, with the same numerics. Against nibble's CPU path, tokens may differ after near-ties, where llama.cpp's own CPU and GPU paths differ too |
| Correctness oracle | llama.cpp on the same GGUF file: greedy tokens, logits, and per-layer tensor dumps |

## Dependency policy

Everything a typical engine pulls from crates.io is written here instead, and
kept small: mmap (`nibble-gguf::Mmap`), f16/bf16 conversions, the thread pool
(`cpu::ThreadPool`), SIMD through `std::arch` behind feature detection,
hand-written pre-tokenizers with committed Unicode tables, built-in chat
templates, the CLI parser, `std::net` HTTP/1.1 with SSE and our own JSON.
Still to write: CUDA driver API bindings, loaded at runtime.

GPU code needs a compiler only at development time, never in `cargo build`:

- **CUDA kernels** are compiled offline with nvcc into a fatbin. The exact
  nvcc command is documented next to the kernels; there is no build script.
  The fatbin is committed and embedded with `include_bytes!`. Building nibble
  needs no CUDA toolkit; only editing kernels does.
- **Metal shaders** are embedded as MSL source and compiled by the OS at startup.

## Backends

The forward pass is written once against an op-level `Backend` trait
(`backend.rs`), generic over the backend's vector and KV-cache types. Ops
are coarse where a GPU can fuse them: `attention` covers the Q/K norms,
RoPE, the cache append and the gate, `delta_net` the convolution, the delta
rule and the gated norm. `BackendKind::Auto` picks Metal, then CUDA (once it
exists), then the CPU, and keeps why each GPU backend was skipped
(`Session::fallback_reason`). GPU libraries are opened at runtime, never linked.

### CUDA (Windows, Linux) — not started

- **Driver API only**, loaded at runtime from `nvcuda.dll` / `libcuda.so.1`.
  Entry points are resolved through `cuGetProcAddress`, which avoids the
  `_v2` symbol-name traps.
- **Kernels in CUDA C++** (`crates/nibble/src/cuda/kernels/*.cu`), built into
  one fatbin with SASS for sm_75, 80, 86, 89, 90, 100 and 120, plus PTX for
  compute_120. CUDA 13 supports Turing and newer; older GPUs use the CPU.
- **No cuBLAS:** `dp4a` int8 matvecs, a BF16 matvec, fused norm/RoPE,
  decode attention, conv1d + DeltaNet step with the state in VRAM, logits
  and argmax on the GPU, tensor-core (`mma.sync`) GEMM for prompts.
- **One CUDA graph per token**, captured once and replayed: at 0.8B,
  launches would otherwise dominate.
- **Weights in VRAM**, copied once at load; partial offload later.
- **Testing:** locally on the RTX 3070 Ti (CUDA 13.4 toolkit, driver 621.24).

### Metal (macOS, iOS) — done

Objective-C runtime FFI, shaders compiled at startup, zero-copy weights
(split across buffers past `maxBufferLength`, as llama.cpp does), kernels
that transcribe ggml-metal's for the matvecs and matrix products, a
concurrent encoder with barriers, a few command buffers per token. Tested on
the M4 Max (macOS 27.2) only.

- **Where decode time goes** on small models: about 180 dependent dispatches
  per token at 4–6 µs each, not bandwidth. Measured and not kept: matvec
  threadgroups normalizing their own input, the next norm computed by the
  last threadgroup with device atomics, four rows per simdgroup, other
  simdgroup counts, barriers on named buffers, a serial encoder.
- **Next:** fewer dependent steps. A norm scale computed from
  per-threadgroup partial sums and applied as the next product loads its
  input would remove the norm dispatches, at the cost of last-bit
  differences from llama.cpp.

### Later: Vulkan

Neither CUDA nor Metal covers AMD and Intel GPUs or Android. A Vulkan compute
backend (same `Backend` trait, loader opened at runtime) would cover them.
Not scheduled yet.

## Server: what is left

The server is complete for single-user use (see README). Still to do:

- **N parallel slots** with batched decode (Phase 7 step 7). Each Qwen3.5
  slot costs about 19 MiB of DeltaNet state plus its KV cache.
- **More checkpoints:** at the start of the last user message (llama.cpp's
  other rule), for edited messages and for a new session's first replay to
  a shared prefix, which still re-evaluates it.
- **`tool_choice: "required"`** and named functions, which need constrained
  decoding.
- Open WebUI has not been tried.

## Architecture

Current layout; `(planned)` entries do not exist yet.

```
crates/
  nibble-gguf/     GGUF parser (zero-copy), Storage (mmap / aligned buffer), writer
  nibble-json/     JSON parser and writer, plus Python-style output for Jinja's `tojson`
  nibble/          the engine
    quant.rs, quant/  weight formats: f16/bf16, dequantize, dot types and activation encoding, dot products (scalar reference)
    cpu/
      mod.rs       Cpu (pool + kernels): single, grouped and fused SwiGLU matvecs; norms, RoPE, Hadamard
      pool.rs      persistent thread pool: dynamic task claiming, spin-then-park, panic propagation
      kernels.rs   kernel dispatch (the one place that lists instruction sets)
      avx2.rs      AVX2/FMA/F16C kernels transcribed from ggml, behind a CPU-feature token
      neon.rs      NEON kernels (dotprod and fp16 when detected; vmull / scalar f16 widening otherwise)
      scalar.rs    scalar references of the vector kernels
      topology.rs  default thread count: performance cores on Apple Silicon, else one per physical core
      backend.rs   the CPU Backend
      attention.rs the KV cache (every cache type) and attention, with and without flash attention
    model/         registry (mod.rs), config, bind, shared layers, qwen3, qwen35 (also `qwen35moe`)
    tokenizer/     facade (vocab, specials, decoding) + bpe.rs, pretokenize.rs, unicode tables
    chat/          ChatML / Qwen3 / Qwen3.5 templates with tools, tool-call parser
    sampler.rs     greedy, temperature, top-k/top-p/min-p, penalties, logprobs
    session.rs     Session: tokens, cache, recurrent-state checkpoints, generation and drafting
    backend.rs     the op-level Backend trait, BackendKind, KV cache types, the parameter arena
    cuda/          (planned) driver API loader + bindings, memory, graphs; kernels/*.cu + nibble.fatbin
    metal/         objc.rs, device.rs, weights.rs, kernels.rs (argument structs, pipelines),
                   mod.rs (the Metal Backend), tests.rs, shaders/*.metal
  nibble-server/   http.rs, openai.rs, worker.rs, server.rs, text.rs
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
| **0–3** ✅ | Scaffold and CI, GGUF + `inspect`, CPU reference for Qwen3 and Qwen3.5, OpenAI server | — |
| **4. CPU speed** (AVX2 and NEON done) | Still to do: AVX2 K-quant kernels, measured on x86; NEON for Q2_K and the IQ1–IQ3 codebooks; the 9B's CPU decode gap (27.1 tok/s against llama.cpp's 35.6 with repacking, not investigated) | Decode at least as fast as llama.cpp on the CPU |
| **5. CUDA backend** | Loader, bindings, memory, fatbin kernels, CUDA graphs, CPU fallback | Same greedy tokens as CPU; decode at least as fast as llama.cpp CUDA on the 3070 Ti |
| **6. Metal backend** ✅ | Left for later: other Apple chips and older macOS versions (untested; `auto` falls back to the CPU and prints why), fewer dependent steps per token on small models | — |
| **7. Batched evaluation and MTP** (in progress) | Batches, rollback, MTP and drafting done; prompt GEMM on Metal done; CPU prompt tiles, parallel server slots and MTP sidecars to do | pp512 within 25% of llama.cpp on each backend (met on Metal); speculative decoding gives the same tokens as plain decoding (met) and decodes faster on the MTP files (met) |
| **8. Breadth** | Qwen3.5 2B, `llama` and further dense archs, Vulkan for AMD/Intel, an AVX2 IQ4_XS kernel | The local `qwen35` files run correctly on every backend |
| **9. Packaging** | Stable Rust API, C ABI (`cdylib` + header), WASM (simd128, in-memory weights), prebuilt release binaries | — |

## Phase 7: batched evaluation and MTP speculative decoding

### Decisions

- **One forward pass for n tokens.** The `Backend` ops take a batch, set by
  `begin(pos, n)`; decoding is n = 1.
- **CPU batches and Metal batches up to 16 are bit-identical to decoding**,
  so speculative decoding gives exactly the tokens of plain decoding, and
  CPU prompts still compare with llama.cpp at `-ub 1`.
- **Metal prompt batches above 16** (`MATMUL_MIN_BATCH`) use transcriptions
  of ggml-metal's `mul_mm`/`mul_mm_id` and nibble's own attention and
  chunked DeltaNet kernels, compared with llama.cpp's Metal path at its
  default `-ub`.
- **DeltaNet rollback as llama.cpp's `rs_replay`**, not state snapshots.
- **MTP needs a file with the head**, in the main file
  (`nextn_predict_layers`) or as an `mtp-*.gguf` sidecar (not done).

### Steps

1–4. ✅ Batched forward, DeltaNet rollback (`Session::eval_all`), the MTP
   head (main-file layout), speculative generation (`--draft N`). Left: a
   tensor-level check of the MTP head with `cb_eval`; `bench` measuring
   drafting.
5. ✅ **Multi-input kernels** sharing each weight load across a batch,
   bit-identical to their one-input kernels (Metal checks the wide kernels
   against the one-input ones when it starts and drops any that differ).
   Measured and not kept: 4 rows per threadgroup for the Metal wide
   kernels, loading a block's weights first, two 2-input groups instead of
   one 4-input group, a 2-row CPU tile for one input. Still to do: better
   K-quant multi-input kernels on Metal (the 27B is slower with drafts: no
   multi-input Q5_K/IQ4_XS kernels; llama.cpp's `mul_mv_ext` layout rounds
   differently from the one-input kernel), AVX2 tiles.
6. **Prompt speed:** Metal done, CPU tiles to do.
   - **Kernels:** `matmul_*` and `matmul_*_id` (ggml's `kernel_mul_mm`
     and `kernel_mul_mm_id`, 64 × 32 tiles); attention `attention_prep` then
     `attention_prompt_d{64,128,256}` (a simdgroup per query head and 8
     tokens, keys and values 32 positions at a time, online softmax), and
     `attention_prompt_staged_*` for quantized caches; DeltaNet
     `delta_prep`, then for heads of 128 the chunked delta rule
     (`delta_chunk_prep`, `delta_chunk_scan`, chunks of 32; relative error
     2e-6 after 100 tokens) and otherwise `delta_scan` with prefetch.
   - **pp512** on an M4 Max, alternating runs on a cool machine (llama.cpp
     b12722, best of `-fa 0` and `-fa 1`): 0.8B Q8_0 7101 (7044), 0.8B
     Q4_K_M 6709 (6684), 0.8B Q3_K_M 6700 (6709), 4B Q8_0 1282 (1305), 9B
     Q8_0 703 (725), 9B UD-Q4_K_XL 592 (601), 27B UD-Q4_K_XL 162 (166),
     35B-A3B 1232 (1337), Qwen3-0.6B Q8_0 8380 (9450, `-fa 1`); 4B at depth
     8192 951 (982). Long runs heat the machine: one that followed hours of
     benchmarks measured the 4B at 935 against llama.cpp's 1185 ± 104.
   - **Greedy output** after prompts of 3,500–4,200 tokens: identical to
     llama.cpp's Metal path on every local file except a 0.001 tie on
     Qwen3-0.6B Q8_0.
   - **Measured and not kept:** 64 × 64 matrix tiles (no faster; ggml's
     64 × 32 tile reaches about two thirds of the GPU's FP32 rate);
     64-position attention blocks; two tiles of 8 tokens per attention
     simdgroup; staging f16 keys and values in threadgroup memory (much
     slower at depth); a DeltaNet scan with one reduction per token; expert
     tiles of 16 slots (experts with 17–32 slots dequantize twice);
     batching the embedding dispatches.
   - **Where llama.cpp is still ahead:** attention-only models (Qwen3-0.6B,
     11%, its flash attention), the 35B's expert products (8%), the 4B–27B
     by 1–3%, and quantized caches at depth (the 4B at 8192 with q8_0: 498
     tok/s against 903; the staged kernel is the gap). Next: an attention
     kernel along the lines of ggml's tiled flash attention, better expert
     tiles.
   - **Not done:** matching llama.cpp's non-flash attention for a prompt
     bit for bit (it multiplies with `mul_mm` and a softmax over the padded
     row), CPU prompt tiles.
   - **KV cache types** (done): every type llama.cpp's `--cache-type-k/-v`
     takes but `f8_e4m3`, whose cache takes scales from the model file. As
     llama.cpp: each type's `from_float` on the CPU and ggml-metal's
     `quantize_*` on Metal; quantized caches rotated by a Walsh-Hadamard
     transform (`n_rot_k`/`n_rot_v`); quantized values take the
     flash-attention path; bf16 key dots follow `ggml_vec_dot_bf16`'s
     scalar f64 sum. Decode with q8_0 at depth 8192 on the 4B: 64.5 tok/s
     (llama.cpp 49.6).
7. **Parallel server slots:** several sequences decoded in one batch.
8. **MTP sidecar files** (`mtp-*.gguf`) and `--draft-p-min`.

## Validation

- **Reference runs:** llama.cpp release builds for logits,
  `llama-eval-callback` tensor dumps (compare per layer to find the first
  mismatch), `llama-perplexity`, `llama-bench` and `llama-server`.
- **Unit level:** each kernel against the scalar reference on random inputs;
  each quant format's dequantization against gguf-py reference outputs.
- **Tiny synthetic models in CI:** `GgufWriter` builds random tiny `qwen3`
  and `qwen35` models, checked against independent f64 implementations.
- **Comparing with llama.cpp:** the commands are in `AGENTS.md`.
- **Bit-level checks:** small C programs outside the repository, linked
  against the release's `libggml-cpu` / `libllama`; the tests record their
  results.
- **Server:** end-to-end tests over real sockets on a tiny model, and the
  official OpenAI Python client against the real models.

## Known pitfalls

- GGUF dimensions are innermost first: `attn_qkv` is `[1024, 6144]`, i.e. 6144 rows of 1024.
- `attn_q` interleaves query and gate **per head** (`[q0 | g0 | q1 | g1 | ...]`), not as two halves.
- The conv kernel is `[4, 6144]`: the 4 taps of a channel are contiguous, oldest first.
- llama.cpp stores the DeltaNet state transposed (`M[j][i] = S[i][j]`) so rows are contiguous. Match that layout.
- The q/k head for value head `h` is `h % n_k_heads` in ggml. This matters
  for sizes with more V heads than K heads.
- `ssm_a` is already `-exp(A_log)`, and norm weights already include `+1`. Apply neither again.
- Skip the MTP layer (`nextn_predict_layers`) in the trunk loop.
- Qwen3.5 uses interleaved M-RoPE (`rope.dimension_sections`); for text
  input it reduces to NeoX RoPE on the first `rope.dimension_count` dims.
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
- **Quantized KV caches in llama.cpp:** a quantized V cache needs flash
  attention; quantized caches are Hadamard-rotated; ggml-metal has no
  kernel for `iq4_nl` keys, so llama.cpp runs that attention on its CPU
  even with `-ngl 99`.
- **CUDA:** a fatbin built by a newer toolkit than the user's driver supports
  cannot JIT its PTX. Ship SASS for every supported arch, and fall back to CPU with a clear message.
- **Metal:** `newBufferWithBytesNoCopy` needs a page-aligned pointer and a
  page-multiple length (16 KiB pages on Apple Silicon). Map the whole file and
  round the length up.
- **Metal barriers dominate small models:** a serial encoder is no faster
  than concurrent dispatch with barriers, nor are barriers on named
  buffers, nor device-scope atomics between threadgroups; only fewer
  dependent stages help. A kernel with a serial loop per thread or few
  threadgroups costs tens of µs per layer: split such work across
  threadgroups.
- **Compare speed at the same depth.** `nibble run` with a long prompt
  measures past its end, llama-bench `-d` at exactly that depth: use
  `nibble bench -d N`, which rewinds to the same `N` tokens before each
  repetition.
- **Benchmark in the real model, alternating A/B runs on a cool machine.**
  Microbenchmarks of one layer on the GPU were bimodal between processes,
  skipping an op to time it overstates its cost, and hours of benchmarks
  heat the M4 Max enough to cost 10–25%.
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
   for correctness; speed needs a real x86 machine.
4. **BF16 on ARM.** ggml sums BF16 dots one product at a time in f64; nibble
   uses f32 vector accumulators, which is the one remaining difference from
   llama.cpp on the comparison prompts. An exact kernel (rows in f64 lanes,
   each summed in order) is possible but costs a large part of the speedup.
