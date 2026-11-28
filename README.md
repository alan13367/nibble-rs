# nibble-rs

A tiny, fast, cross-platform inference engine for GGUF models, written in Rust
with **zero external dependencies**.

> Early development. Today: Qwen3 (`qwen3`), Qwen3.5 (`qwen35`, hybrid
> Gated DeltaNet), Qwen3.5/3.6 mixture-of-experts models (`qwen35moe`),
> Liquid's LFM2 (`lfm2`, hybrid shortconv / attention, and `lfm2moe`) and
> PrismML's Ternary Bonsai (`qwen35` files with Hadamard-folded weights,
> `prism.hadamard`) on F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_0,
> Q2_K–Q6_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL,
> IQ4_XS and MXFP4 weights.
> On Apple Silicon it runs on the GPU with Metal by default, with the same
> greedy output as llama.cpp's Metal path; elsewhere (or with
> `--backend cpu`) on the CPU, with dot products that reproduce
> llama.cpp's ARM results bit for bit (NEON kernels for most types, the
> scalar reference for Q2_K and the IQ1–IQ3 codebooks). The exceptions are
> BF16 on ARM, where nibble is about six times faster instead, and x86,
> where only F32/F16/BF16/Q8_0/Q2_0 have AVX2 kernels and the other types use
> the scalar code (whose sums follow ggml's ARM kernels, so x86 results can
> differ from llama.cpp's in the last bits). Models with a
> multi-token-prediction head (Qwen3.5 files converted with it) can draft
> tokens and check them in one pass (`--draft N`), with the same output as
> without. It comes with an OpenAI-compatible server and `run`, `chat`,
> `serve`, `bench`, `tokenize` and `inspect` commands.

## Build

Requires Rust 1.100 or newer.

```sh
cargo build --release
./target/release/nibble run -m models/Qwen3-0.6B-Q8_0.gguf -p "The capital of France is" -n 32
./target/release/nibble chat -m models/Qwen3-0.6B-Q8_0.gguf --no-think
./target/release/nibble tokenize -m models/Qwen3-0.6B-Q8_0.gguf -p "Hello world"
./target/release/nibble inspect models/Qwen3-0.6B-Q8_0.gguf
```

`nibble <command> --help` lists each command's options: sampling, context
size, `--backend auto|cpu|metal` (by default Metal where it is available;
nibble says why when it falls back to the CPU), CPU threads (by default the
performance cores on Apple Silicon, else one per physical core), and
`--probs N` to show the top candidates at each step.

`nibble bench -m <model.gguf>` measures prompt and decode speed the way
`llama-bench` does (`pp512`, `tg128`), so the numbers compare directly.

`--draft N` (on `run`, `chat` and `serve`) drafts up to N tokens after each
token with the model's MTP head, if the file has one (`nibble inspect`
shows `nextn_predict_layers`), and checks them all in one forward pass.
The tokens generated are the same as without it. On an M4 Max, Qwen3.5-4B
Q8_0 decodes at about 131 tok/s with `--draft 3` on Metal (74 without) and
80 on the CPU (49); 2 or 3 drafts are best.

`-ctk T` / `-ctv T` (`--cache-type-k`, `--cache-type-v`) choose the type of
the cached attention keys and values, as llama.cpp's flags do: `f32`,
`f16` (the default), `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0` or
`q5_1`. `q8_0` halves the cache with almost no change in output; the 4-bit
types quarter it but can hurt (Qwen3-0.6B's answers fall apart with 4-bit
keys, in llama.cpp too). As in llama.cpp, quantized caches are rotated
with a Walsh-Hadamard transform, and quantized values use flash attention.
Output matches llama.cpp on the same types (`-fa on` when V is
quantized). `serve` prints the cache's size for the whole context.

## OpenAI-compatible server

```sh
./target/release/nibble serve -m models/Qwen3.5-0.8B-Q8_0.gguf --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Hello!"}],"stream":true}'
```

- `POST /v1/chat/completions`, `POST /v1/completions` (streaming with
  server-sent events, `stream_options.include_usage`), `GET /v1/models`, `GET /health`.
- Qwen reasoning is returned as `reasoning_content`. Turn it off with
  `"chat_template_kwargs": {"enable_thinking": false}`.
- Sampling: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
  `frequency_penalty`, `seed`, `stop`, `max_tokens` / `max_completion_tokens`.
- `--api-key KEY` (or `NIBBLE_API_KEY`) requires `Authorization: Bearer KEY`.
  The server listens on localhost unless `--host` says otherwise.
- Tool calling (`tools`, `tool_choice: "auto" | "none"`,
  `parallel_tool_calls`, `tool` messages), streamed and not, in each Qwen
  template's own format.
- `logprobs` / `top_logprobs` for chat, and `logprobs: N` for completions.
- Requests are served one at a time in arrival order; the cached prompt
  prefix is reused between requests. For Qwen3.5, whose recurrent state
  cannot be rewound, the session keeps checkpoints, so a new chat turn does
  not re-evaluate the conversation.
- `reasoning_effort` `none` or `minimal` turns Qwen's thinking off, the
  other levels on (an explicit `enable_thinking` wins).
- `usage.prompt_tokens_details.cached_tokens` reports the prompt tokens
  reused from the cache; `/v1/models` reports the context size as
  `max_model_len` and `context_length`.
- `nibble serve` prints a line per request: prompt tokens (cached and
  evaluated), generation speed, and drafts accepted.
- Rejected with a 400: images, `n > 1`, and forced tool calls
  (`tool_choice: "required"` or a named function), which need constrained
  decoding.
- Checked with the official OpenAI Python client.

### Coding harnesses

Agent harnesses that speak the OpenAI chat API can use the server as a
provider; give it a context large enough for the harness's system prompt
and tools, and drafting if the model has an MTP head:

```sh
./target/release/nibble serve -m models/Qwen3.5-4B-Q8_0.gguf -c 32768 --draft 3 \
    --port 8080 --model-id qwen3.5-4b
```

For example, in [Yawl](https://github.com/alan13367/yawl)'s
`~/.yawl/config.json`:

```json
{
  "model": "nibble:qwen3.5-4b",
  "providers": {
    "nibble": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "models": [{ "id": "qwen3.5-4b", "reasoning_efforts": ["minimal", "high"] }]
    }
  }
}
```

`yawl --reasoning minimal` then answers without thinking. The first request
evaluates the harness's whole system prompt (about 2,000 tokens for Yawl:
some 1.5 s on the 4B with Metal on an M4 Max, 11 s on its CPU); later
requests reuse it from the cache.

## Inspecting files

`nibble inspect` prints the header, metadata and tensor table:

```
file     models/Qwen3.5-0.8B-Q8_0.gguf (795.0 MiB, memory-mapped)
format   GGUF v3, alignment 32, data at byte 10962464
model    Qwen3.5-0.8B [qwen35]
tensors  335 tensors, 772.85M parameters, 784.5 MiB
types    Q8_0 x195 (782.6 MiB), F32 x140 (1.9 MiB)
```

## Layout

| Crate | Purpose |
|---|---|
| `crates/nibble-gguf` | GGUF parser, memory mapping, writer |
| `crates/nibble-json` | JSON parser and writer (for chat templates and the server) |
| `crates/nibble` | The engine (library) |
| `crates/nibble-server` | OpenAI-compatible HTTP server (std only) |
| `crates/nibble-cli` | The `nibble` binary |

Local models go in `models/`, which is ignored by git. Tests that compare
against llama.cpp on real models are ignored by default:

```sh
cargo test --release -p nibble --test reference -- --ignored
```

Guidelines for contributors and coding agents: [AGENTS.md](AGENTS.md).
