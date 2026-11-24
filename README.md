# nibble-rs

A tiny, fast, cross-platform inference engine for GGUF models, written in Rust
with **zero external dependencies**.

> Early development. Today: a CPU engine for Qwen3 (`qwen3`) and Qwen3.5
> (`qwen35`, hybrid Gated DeltaNet) on F32, F16, BF16 and Q8_0 weights,
> checked against llama.cpp and as fast as it on AVX2 CPUs, with an
> OpenAI-compatible server and `run`, `chat`, `bench`, `tokenize` and
> `inspect` commands.

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
size, threads (one per physical core by default), and `--probs N` to show
the top candidates at each step.

`nibble bench -m <model.gguf>` measures prompt and decode speed the way
`llama-bench` does (`pp512`, `tg128`), so the numbers compare directly.

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
- Rejected with a 400: images, `n > 1`, and forced tool calls
  (`tool_choice: "required"` or a named function), which need constrained
  decoding.
- Checked with the official OpenAI Python client.

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
