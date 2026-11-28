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
| Done | GGUF reader/writer and `inspect`; tokenizer (ids identical to llama.cpp); `qwen3`, `qwen35` and `qwen35moe` on every listed weight type; `lfm2` (LFM2-350M F16 and Q8_0 equal to llama.cpp on CPU and CUDA) and `lfm2moe` (synthetic tests only: no file at hand); Ternary Bonsai (`qwen35` with `prism.hadamard` folded weights, Q2_0) on CPU and Metal; CPU engine (NEON for all types but Q2_K and the IQ1–IQ3 codebooks, AVX2 for F32/F16/BF16/Q8_0/Q2_0); Metal backend, the default (`auto`) on macOS; CUDA backend (Windows, Linux), the default where an NVIDIA GPU is present; OpenAI server with tools, logprobs and recurrent-state checkpoints; batched evaluation, DeltaNet rollback, the MTP head and `--draft N`; Metal prompt GEMM; CPU prompt batches (parallel batch ops and attention, NEON `i8mm` 8-input K-quant tiles); llama.cpp's KV cache types on both backends |
| Correctness | CPU: greedy output identical to llama.cpp (`-fa off --no-repack -ub 1`) except one BF16 near-tie caused by nibble's faster bf16 dot (rechecked after the CPU prompt work on Qwen3-0.6B Q8_0, Qwen3.5-0.8B Q4_K_M and 4B Q8_0); kernels checked bit for bit against `libggml-cpu`. Metal: identical to llama.cpp's Metal path except 0.001 ties on Qwen3-0.6B. KV cache types: 50/50 CPU runs identical, Metal identical where llama.cpp runs the same path. Tested on an M4 Max only; the x86 path only under Rosetta. CUDA: logits bit-identical to llama.cpp b12675's CUDA build (`-ngl 99 -fa off -ub 1`) on every local file (Qwen3-0.6B Q8_0/BF16; Qwen3.5-0.8B BF16, Q8_0, Q6_K, Q4_K, Q3_K, Q2_K, IQ2_M; 4B Q8_0; 9B Q3_K_M and UD-Q4_K_XL), at 8, 19, 300 and 4329 tokens, one token at a time or in batches of up to 8; greedy text identical on all of them. CUDA prompt batches (more than 8 tokens): greedy text identical to llama.cpp's default batching on Qwen3-0.6B Q8_0/BF16, 0.8B Q8_0/Q4_K_M and 9B UD-Q4_K_XL; the 4B Q8_0 differs after a 0.05 near-tie where llama.cpp's own `-ub 1` and default runs differ too (nibble takes the `-ub 1` token). RTX 3070 Ti (sm_86) only |
| Decode (M4 Max) | Faster than llama.cpp on every local file on Metal (empty context and 4096 tokens, `-fa` off and on) and on the CPU. Qwen3.5-4B Q8_0 Metal 74 tok/s, 131 with `--draft 3` (llama.cpp 64, 124); Qwen3-0.6B Q8_0 CPU 295 (261) |
| Prompt (M4 Max, pp512) | Metal level with llama.cpp on the 0.8B Q4_K_M, the 35B MoE and the 4B with a q8_0 cache at depth 4096; 5% behind on the 4B and on Qwen3-0.6B (its flash attention). CPU 11–23% behind llama.cpp, whose CPU prompt products run in Accelerate (Phase 7 step 6) |
| Tests | 261 pass in `cargo test --workspace` on the M4 Max (Metal tests skip without a device), 247 on Windows with the RTX 3070 Ti (including 4 CUDA kernel tests); 17 reference tests on real models with `--ignored` (the 3 CUDA ones pass on the 3070 Ti; `qwen35_bf16` needs a file not on the Mac) |
| Decode (RTX 3070 Ti, CUDA) | tg128, 2026-11-28: Qwen3.5-4B Q8_0 98 tok/s (llama.cpp 101), 9B UD-Q4_K_XL 84 (88), Qwen3-0.6B Q8_0 224–232 (402), 0.8B Q4_K_M 263 (408); at depth 4096 (2026-11-26): 0.6B 169 (291), 4B 88 (95). Small models lose to launch overhead (no CUDA graphs yet) |
| Prompt (RTX 3070 Ti, CUDA) | pp512, 2026-11-28, against `llama-bench -ngl 99 -fa 0`: Qwen3-0.6B Q8_0 23.7k tok/s (llama.cpp 19.1k), 0.8B Q4_K_M 16.1k (19.3k), 4B Q8_0 4.8k (5.9k), 9B UD-Q4_K_XL 1.7k (3.5k: its 10 IQ4_XS gate/up tensors have no GEMM and run per token) |
| Not started | PrismML's PTQ1_0 and PQ2_0 (their llama.cpp fork's types); AVX2 kernels beyond F32/F16/BF16/Q8_0/Q2_0; NEON kernels for Q2_K and the IQ1–IQ3 codebooks; AVX2 prompt tiles; parallel server slots |
| To rerun | The 5600X comparisons (correctness and speed) predate the attention and fused-multiply-add fixes |
| Git | Committed through `87b2dc0` (the bit-exact CUDA backend); the CUDA prompt batches (GEMM, flash attention, split DeltaNet) are not committed |

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
rule and the gated norm. `BackendKind::Auto` picks Metal, then CUDA, then the CPU, and keeps why each GPU backend was skipped
(`Session::fallback_reason`). GPU libraries are opened at runtime, never linked.

### CUDA (Windows, Linux) — first version done

Status: every op of `qwen3`, `qwen35` and `qwen35moe` runs on the GPU. For
`qwen3` and `qwen35` they round as ggml-cuda does for one token without
flash attention, so the logits equal llama.cpp's CUDA build bit for bit (see
Status). The mixture-of-experts ops (routing, expert products) are checked
only against the synthetic f64 reference: no MoE file was at hand to compare
with llama.cpp. Kernels are
built with `-use_fast_math`, as ggml-cuda is. What that took, for the next
kernels:

- **Transcribe ggml-cuda, including the launch geometry:** `mul_mat_vec_q`
  (4 warps on Ampere and newer, 2 for K-quants on Turing), `mul_mat_vec_f`
  for F32/F16/BF16 weights (block size from the row length, F16 accumulated
  in `half2`, activations not rounded), `rms_norm_f32` (`float4` geometry),
  `soft_max_f32`, `ssm_conv`, `gated_delta_net_cuda`, and `mul_mat_f`
  (tensor cores) for KQ once llama.cpp's padded n_kv passes 512.
- **Follow ggml-cuda's fusions:** RMS_NORM+MUL+ROPE is one kernel for NEOX
  RoPE (Qwen3) and separate kernels for multi-section RoPE (Qwen3.5), with
  different rounding (`QkNorm::multi_rope`).
- **Match nvcc's multiply-add fusion** where it decides the result (RoPE's
  second half): written with `fmaf` in the order nvcc chose for ggml.
- **Attention** follows the non-flash path: KQ, softmax over the row padded
  as llama.cpp pads its cache (256), KQV accumulated in `half2` (KQV has no
  precision request), V stored transposed. Quantized caches, which
  llama.cpp runs only with flash attention, keep an online softmax.
- **Debugging:** dump llama.cpp's tensors with a `cb_eval` program loaded
  against the release DLLs, and nibble's from an instrumented copy (both
  outside the repository). Observing an intermediate tensor splits
  llama.cpp's graph and disables its fusions for that token: observe only
  the ends of fused chains.

**Prompt batches** (more than 8 tokens, sm_80 and newer) take a separate
path that rounds differently from one token at a time, as llama.cpp's
batched kernels do (a decision of 2026-11-26; batches of up to 8 tokens
keep the exact kernels):

- **Int8 tensor-core GEMM** (`kernels/gemm.cu`) laid out as ggml-cuda's
  `mul_mat_q` on Ampere: 256 threads, 128 rows by 32/64/128 tokens, weights
  of 256 values per row unpacked to int8 in shared memory by whole warps,
  activations in a `block_q8_1_mmq`-like layout (`quantize_mmq`) copied in
  with `cp.async` and double-buffered, `ldmatrix` + `mma.sync`. All of Q8_0,
  Q4_0, Q4_1, Q5_0, Q5_1 and Q2_K–Q6_K. Stream-K (reduced in block order, so
  deterministic) where whole tiles would leave waves half empty, and a fused
  gate/up kernel that applies SwiGLU. MoE expert products still run per
  token.
- **Flash attention** for F16 caches (head sizes 64, 128, 256), K and V
  copied with `cp.async` while the other is in use.
- **DeltaNet** in two kernels: convolution, norms and gates for every token
  in parallel (`delta_prep`), then the sequential delta rule
  (`delta_seq`), with the exact path's arithmetic (bit-identical). A batch
  can leave its last tokens pending, so the session's checkpoint 4 tokens
  before the end costs no split batch (`Backend::set_delta_net_tail`).
- **Measured, not kept:** loader unrolls below full (load latency bounds
  the GEMM), the int-to-float trick with `0x4B400000` (more registers than
  `I2F`), a separate stream-K fixup kernel (its launches cost more than the
  in-kernel reduction), `delta_seq` at six blocks per SM (spills).
- **Where the time goes** (4B Q8_0, 512 tokens, ~103 ms of GPU time against
  llama.cpp's ~88): the GEMMs 73 ms (llama.cpp ~57), `delta_seq` 16 ms
  (llama.cpp's chunked tensor-core `gdn_single` 8 ms). Our GEMM uses all 255
  registers and spills; ggml's Q8_0 kernel uses 222 with no stack.

Next: GEMM loaders for the IQ types (IQ4_XS first: the 9B UD-Q4_K_XL's
gate/up products), the GEMM's register pressure, a chunked DeltaNet (ggml's
`gated-delta-net-mma.cu`), CUDA graphs per token (launch overhead on small
models), partial offload, and the plan below.

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
    model/         registry (mod.rs), config, bind, shared layers, qwen3, qwen35 (also `qwen35moe`),
                   lfm2 (also `lfm2moe`: shortconv blocks, sigmoid-gated experts),
                   hadamard (`prism.hadamard` folded weights), bonsai (qwen35 with folded weights)
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
| **7. Batched evaluation and MTP** (in progress) | Batches, rollback, MTP and drafting done; prompt GEMM on Metal and NEON prompt tiles done; AVX2 prompt tiles, parallel server slots and MTP sidecars to do | pp512 within 25% of llama.cpp on each backend (met on Metal and on the NEON CPU); speculative decoding gives the same tokens as plain decoding (met) and decodes faster on the MTP files (met) |
| **8. Breadth** | Qwen3.5 2B, `llama` and further dense archs, Vulkan for AMD/Intel, an AVX2 IQ4_XS kernel | The local `qwen35` files run correctly on every backend |
| **9. Packaging** | Stable Rust API, C ABI (`cdylib` + header), WASM (simd128, in-memory weights), prebuilt release binaries | — |

## Phase 7: batched evaluation and MTP speculative decoding

### Decisions

- **One forward pass for n tokens.** The `Backend` ops take a batch, set by
  `begin(pos, n)`; decoding is n = 1.
- **CPU batches and Metal batches up to 16 are bit-identical to decoding**,
  so speculative decoding gives exactly the tokens of plain decoding, and
  CPU prompts still compare with llama.cpp at `-ub 1`. This holds for every
  CPU tile, the 8-input `i8mm` ones included; a kernel that rounds
  differently (a Q8_0 `i8mm` GEMM) is not kept, however fast.
- **Metal decode folds RMSNorm into the residual product** before it: the
  product adds `x²` to a 64-bit fixed-point sum (`RMS_FP_SCALE` = 2^28;
  integer adds do not depend on the order of threadgroups) and the next
  product applies the scale. Its error is bounded against f64 in
  `metal/tests.rs`.
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
   one 4-input group, a 2-row CPU tile for one input, a Q8_0 `i8mm` tile
   (fast, but one integer per block instead of ggml's four lanes rounds
   differently and changed greedy output; a bit-identical one needs the
   same float work per lane as the dotprod kernel and gains little). Still
   to do: better
   K-quant multi-input kernels on Metal (the 27B is slower with drafts: no
   multi-input Q5_K/IQ4_XS kernels; llama.cpp's `mul_mv_ext` layout rounds
   differently from the one-input kernel), AVX2 tiles.
6. **Prompt speed:** Metal and NEON done, AVX2 tiles to do.
   - **CPU** (`fb05552`, `94f7d5b`): the batch's norms, residuals,
     activation encoding, DeltaNet convolution and gated norms run on the
     thread pool, attention runs once per batch over the heads instead of
     once per token, weight blocks are 32 KiB, and Q4_K/Q5_K/Q6_K take
     8-input `i8mm` tiles. pp512 on 10 threads (llama.cpp b12722
     `-ngl 0 --repack 0`): Qwen3-0.6B Q8_0 1110 (920 before; llama.cpp
     1370–1440), 0.8B Q4_K_M 895 (545; 1000–1070), 4B Q8_0 195 (153;
     232–236). llama.cpp's macOS release runs those products in its BLAS
     backend (Accelerate) rather than ggml-cpu (see Known pitfalls).
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
   - **After `61962e3`** (prompt attention, fused SwiGLU and expert
     products, batched embedding; alternating runs against `b15af98`):
     Qwen3-0.6B Q8_0 8840 (8420; llama.cpp 9250 with `-fa 1`, 8200 with
     `-fa 0`), 0.8B Q4_K_M 6865 (6670; 6786), 35B-A3B 1310 (1233; 1296),
     4B Q8_0 about 1250 (unchanged; 1311), and the 4B at depth 4096 with a
     q8_0 cache 1136 (793; 1136). Decode gained 4–5% on the dense files
     (deferred RMSNorm).
   - **Where llama.cpp is still ahead:** attention-only models (Qwen3-0.6B,
     5% behind its flash attention, 7% ahead of its non-flash path) and the
     4B–27B by a few percent. Next: an attention kernel along the lines of
     ggml's tiled flash attention.
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
- **llama.cpp's CPU prompt numbers on macOS are Accelerate's:** the
  release links a BLAS backend, and every matrix product of 32 or more
  tokens goes there (dequantized to f32) even with `-ngl 0 -dev none -nopo
  1`; `GGML_SCHED_DEBUG=2 llama-bench ... -v` shows the splits. Its CPU
  pp512 is not a ggml-cpu number. llama-bench spells the repack switch
  `--repack 0`, llama-completion `--no-repack`.
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
