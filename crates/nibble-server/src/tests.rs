//! End-to-end server tests over real sockets, on a tiny random model.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};

use nibble::gguf::{GgmlType, GgufWriter, OwnedValue, Storage, ValueType};
use nibble::{Model, SessionOptions};

use crate::{ServerOptions, serve};
use nibble::json::{self, Json};

const N_EMBD: u64 = 16;

/// A one-layer Qwen3 with a byte-level vocabulary and a ChatML template.
fn tiny_model(chat_template: &str) -> Model {
    let mut seed = 0x2545_F491u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let mut tensor = |w: &mut GgufWriter, name: &str, dims: &[u64]| {
        let n: u64 = dims.iter().product();
        let data: Vec<u8> = (0..n).flat_map(|_| next().to_le_bytes()).collect();
        w.add_tensor(name, GgmlType::F32, dims, data);
    };

    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut vocab: Vec<String> = (0..256u32)
        .map(|b| {
            let cp = if printable(b) {
                b
            } else {
                256 + (0..b).filter(|&x| !printable(x)).count() as u32
            };
            char::from_u32(cp).unwrap().to_string()
        })
        .collect();
    vocab.extend(["<|im_start|>", "<|im_end|>"].map(str::to_owned));
    let types: Vec<i32> = (0..vocab.len())
        .map(|i| if i >= 256 { 3 } else { 1 })
        .collect();

    let mut w = GgufWriter::new();
    w.add("general.architecture", "qwen3")
        .add("qwen3.block_count", 1u32)
        .add("qwen3.context_length", 2048u32)
        .add("qwen3.embedding_length", N_EMBD as u32)
        .add("qwen3.feed_forward_length", 32u32)
        .add("qwen3.attention.head_count", 2u32)
        .add("qwen3.attention.head_count_kv", 1u32)
        .add("qwen3.attention.key_length", 8u32)
        .add("qwen3.attention.value_length", 8u32)
        .add("tokenizer.ggml.model", "gpt2")
        .add("tokenizer.ggml.pre", "qwen2")
        .add(
            "tokenizer.ggml.tokens",
            OwnedValue::array(ValueType::String, vocab.iter().map(String::as_str)),
        )
        .add(
            "tokenizer.ggml.token_type",
            OwnedValue::array(ValueType::I32, types),
        )
        .add(
            "tokenizer.ggml.merges",
            OwnedValue::array(ValueType::String, Vec::<String>::new()),
        )
        .add("tokenizer.ggml.eos_token_id", 257u32)
        .add("tokenizer.chat_template", chat_template);
    let e = N_EMBD;
    tensor(&mut w, "token_embd.weight", &[e, vocab.len() as u64]);
    tensor(&mut w, "output_norm.weight", &[e]);
    for (name, dims) in [
        ("attn_norm", vec![e]),
        ("attn_q", vec![e, 16]),
        ("attn_k", vec![e, 8]),
        ("attn_v", vec![e, 8]),
        ("attn_output", vec![16, e]),
        ("attn_q_norm", vec![8]),
        ("attn_k_norm", vec![8]),
        ("ffn_norm", vec![e]),
        ("ffn_gate", vec![e, 32]),
        ("ffn_up", vec![e, 32]),
        ("ffn_down", vec![32, e]),
    ] {
        tensor(&mut w, &format!("blk.0.{name}.weight"), &dims);
    }
    Model::from_storage(Storage::from_bytes(&w.to_bytes())).unwrap()
}

fn start(api_key: Option<&str>) -> SocketAddr {
    start_with(CHATML, api_key)
}

/// A plain ChatML template, which has no tool calls.
const CHATML: &str = "{{ '<|im_start|>' + message.role }}";

/// Enough of Qwen3's template for nibble to recognize it.
const QWEN3: &str = "{{ '<|im_start|>' }} {%- if '</think>' in content %}";

fn start_with(chat_template: &str, api_key: Option<&str>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut options = ServerOptions::new("tiny");
    options.api_key = api_key.map(str::to_owned);
    options.session = SessionOptions {
        n_ctx: 2048,
        threads: 1,
        backend: nibble::BackendKind::Cpu,
        n_batch: 0,
        draft_max: 0,
        ..Default::default()
    };
    let model = tiny_model(chat_template);
    std::thread::spawn(move || serve(listener, model, options));
    addr
}

struct Reply {
    status: u16,
    head: String,
    body: String,
}

/// Sends one request with `Connection: close` and reads the whole reply.
fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Reply {
    let mut stream = TcpStream::connect(addr).unwrap();
    let mut req = format!(
        "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    req.push_str(body);
    stream.write_all(req.as_bytes()).unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("a complete response head");
    let head = String::from_utf8(raw[..split].to_vec()).unwrap();
    let mut body = raw[split + 4..].to_vec();
    if head
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        body = dechunk(&body);
    }
    let status = head[9..12].parse().unwrap();
    Reply {
        status,
        head,
        body: String::from_utf8(body).unwrap(),
    }
}

fn dechunk(mut raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let eol = raw
            .windows(2)
            .position(|w| w == b"\r\n")
            .expect("a chunk size line");
        let size = usize::from_str_radix(std::str::from_utf8(&raw[..eol]).unwrap(), 16).unwrap();
        raw = &raw[eol + 2..];
        if size == 0 {
            return out;
        }
        out.extend_from_slice(&raw[..size]);
        assert_eq!(&raw[size..size + 2], b"\r\n");
        raw = &raw[size + 2..];
    }
}

fn parse(body: &str) -> Json {
    json::parse(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

const CHAT: &str =
    r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":5}"#;

#[test]
fn health_models_and_errors() {
    let addr = start(None);
    let r = request(addr, "GET", "/health", &[], "");
    assert_eq!((r.status, r.body.as_str()), (200, r#"{"status":"ok"}"#));
    assert!(r.head.contains("Access-Control-Allow-Origin: *"));

    let models = parse(&request(addr, "GET", "/v1/models", &[], "").body);
    assert_eq!(
        models
            .get("data")
            .and_then(|d| d.as_array())
            .map(|d| d[0].get("id").cloned()),
        Some(Some(Json::from("tiny")))
    );

    let r = request(addr, "POST", "/v1/unknown", &[], "{}");
    assert_eq!(r.status, 404);
    let r = request(
        addr,
        "POST",
        "/v1/chat/completions",
        &[],
        r#"{"messages":[]}"#,
    );
    assert_eq!(r.status, 400);
    assert_eq!(
        parse(&r.body)
            .get("error")
            .and_then(|e| e.get("param"))
            .and_then(Json::as_str),
        Some("messages")
    );
    assert_eq!(
        request(addr, "OPTIONS", "/v1/chat/completions", &[], "").status,
        204
    );
}

#[test]
fn chat_completion_has_the_openai_shape() {
    let addr = start(None);
    let r = request(addr, "POST", "/v1/chat/completions", &[], CHAT);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = parse(&r.body);
    assert_eq!(
        v.get("object").and_then(Json::as_str),
        Some("chat.completion")
    );
    let choice = &v.get("choices").and_then(Json::as_array).unwrap()[0];
    assert_eq!(
        choice
            .get("message")
            .and_then(|m| m.get("role"))
            .and_then(Json::as_str),
        Some("assistant")
    );
    assert!(matches!(
        choice.get("finish_reason").and_then(Json::as_str),
        Some("stop" | "length")
    ));
    let usage = v.get("usage").unwrap();
    let completion = usage
        .get("completion_tokens")
        .and_then(Json::as_u64)
        .unwrap();
    assert!(completion <= 5);
    let prompt = usage.get("prompt_tokens").and_then(Json::as_u64).unwrap();
    assert_eq!(
        usage.get("total_tokens").and_then(Json::as_u64),
        Some(prompt + completion)
    );

    // Greedy decoding gives the same answer again (and reuses the cached prompt).
    assert_eq!(
        parse(&request(addr, "POST", "/v1/chat/completions", &[], CHAT).body).get("choices"),
        v.get("choices")
    );
}

#[test]
fn streamed_chat_ends_with_usage_and_done() {
    let addr = start(None);
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":4,
                  "stream":true,"stream_options":{"include_usage":true}}"#;
    let r = request(addr, "POST", "/v1/chat/completions", &[], body);
    assert_eq!(r.status, 200);
    assert!(r.head.contains("Content-Type: text/event-stream"));
    let events: Vec<&str> = r
        .body
        .split("\n\n")
        .filter(|e| !e.is_empty())
        .map(|e| e.strip_prefix("data: ").unwrap())
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"));
    let chunks: Vec<Json> = events[..events.len() - 1]
        .iter()
        .map(|e| parse(e))
        .collect();
    let delta = |c: &Json| {
        c.get("choices")
            .and_then(Json::as_array)
            .and_then(|c| c.first().cloned())
    };
    assert_eq!(
        delta(&chunks[0]).and_then(|c| c.get("delta").and_then(|d| d.get("role")).cloned()),
        Some(Json::from("assistant"))
    );
    let finish = delta(&chunks[chunks.len() - 2]).and_then(|c| c.get("finish_reason").cloned());
    assert!(matches!(finish, Some(Json::String(_))));
    let usage = chunks.last().unwrap();
    assert_eq!(
        usage
            .get("choices")
            .and_then(Json::as_array)
            .map(<[Json]>::len),
        Some(0)
    );
    assert!(
        usage
            .get("usage")
            .and_then(|u| u.get("total_tokens"))
            .is_some()
    );
}

#[test]
fn text_completions_and_stop_sequences() {
    let addr = start(None);
    let r = request(
        addr,
        "POST",
        "/v1/completions",
        &[],
        r#"{"prompt":"abc","temperature":0,"max_tokens":6}"#,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v = parse(&r.body);
    assert_eq!(
        v.get("object").and_then(Json::as_str),
        Some("text_completion")
    );
    let text = v.get("choices").and_then(Json::as_array).unwrap()[0]
        .get("text")
        .and_then(Json::as_str)
        .unwrap()
        .to_owned();

    // Using the first generated character as a stop sequence cuts the text there.
    if let Some(first) = text.chars().next() {
        let body = format!(
            r#"{{"prompt":"abc","temperature":0,"max_tokens":6,"stop":[{}]}}"#,
            Json::from(first.to_string())
        );
        let v = parse(&request(addr, "POST", "/v1/completions", &[], &body).body);
        let choice = &v.get("choices").and_then(Json::as_array).unwrap()[0];
        assert_eq!(choice.get("text").and_then(Json::as_str), Some(""));
        assert_eq!(
            choice.get("finish_reason").and_then(Json::as_str),
            Some("stop")
        );
    }
}

#[test]
fn api_key_is_required_when_set() {
    let addr = start(Some("secret"));
    assert_eq!(
        request(addr, "POST", "/v1/chat/completions", &[], CHAT).status,
        401
    );
    let wrong = request(
        addr,
        "POST",
        "/v1/chat/completions",
        &[("Authorization", "Bearer nope")],
        CHAT,
    );
    assert_eq!(wrong.status, 401);
    let ok = request(
        addr,
        "POST",
        "/v1/chat/completions",
        &[("Authorization", "Bearer secret")],
        CHAT,
    );
    assert_eq!(ok.status, 200);
    // Health checks stay open for load balancers.
    assert_eq!(request(addr, "GET", "/health", &[], "").status, 200);
}

/// The first choice of a response body.
fn choice(v: &Json) -> Json {
    v.get("choices").and_then(Json::as_array).unwrap()[0].clone()
}

/// Checks one logprobs entry of a greedy generation: a valid log-probability
/// with `n_top` alternatives, the best of which is the chosen token.
fn check_greedy_entry(entry: &Json, n_top: usize) {
    let logprob = entry.get("logprob").and_then(Json::as_f64).unwrap();
    assert!(logprob <= 0.0, "{entry}");
    let bytes = entry.get("bytes").and_then(Json::as_array).unwrap();
    assert!(!bytes.is_empty(), "{entry}");
    let top = entry.get("top_logprobs").and_then(Json::as_array).unwrap();
    assert_eq!(top.len(), n_top, "{entry}");
    if let Some(best) = top.first() {
        assert_eq!(best.get("token"), entry.get("token"), "{entry}");
        assert_eq!(best.get("logprob").and_then(Json::as_f64), Some(logprob));
    }
}

#[test]
fn chat_logprobs_cover_every_generated_token() {
    let addr = start(None);
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"temperature":0,"max_tokens":6,
                  "logprobs":true,"top_logprobs":3}"#;
    let r = request(addr, "POST", "/v1/chat/completions", &[], body);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = parse(&r.body);
    let entries = choice(&v)
        .get("logprobs")
        .and_then(|l| l.get("content"))
        .and_then(Json::as_array)
        .unwrap()
        .to_vec();
    let generated = v
        .get("usage")
        .and_then(|u| u.get("completion_tokens"))
        .and_then(Json::as_u64)
        .unwrap();
    assert_eq!(entries.len() as u64, generated);
    for entry in &entries {
        check_greedy_entry(entry, 3);
    }

    // Streamed, the same entries arrive spread over the chunks.
    let stream = body.replace("\"logprobs\"", "\"stream\":true,\"logprobs\"");
    let r = request(addr, "POST", "/v1/chat/completions", &[], &stream);
    let streamed: Vec<Json> = r
        .body
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .filter(|e| *e != "[DONE]")
        .map(parse)
        .filter_map(|chunk| {
            choice(&chunk)
                .get("logprobs")
                .and_then(|l| l.get("content"))
                .and_then(Json::as_array)
                .map(<[Json]>::to_vec)
        })
        .flatten()
        .collect();
    assert_eq!(streamed, entries);
}

#[test]
fn completion_logprobs_use_the_legacy_shape() {
    let addr = start(None);
    let body = r#"{"prompt":"abc","temperature":0,"max_tokens":4,"logprobs":2}"#;
    let r = request(addr, "POST", "/v1/completions", &[], body);
    assert_eq!(r.status, 200, "{}", r.body);
    let logprobs = choice(&parse(&r.body)).get("logprobs").cloned().unwrap();
    let field = |k: &str| logprobs.get(k).and_then(Json::as_array).unwrap().to_vec();
    let (tokens, values, tops, offsets) = (
        field("tokens"),
        field("token_logprobs"),
        field("top_logprobs"),
        field("text_offset"),
    );
    assert!(!tokens.is_empty());
    assert!(values.len() == tokens.len() && tops.len() == tokens.len());
    assert_eq!(offsets[0], Json::from(0));
    for (top, value) in tops.iter().zip(&values) {
        let pairs = top.as_object().unwrap();
        assert!(!pairs.is_empty() && pairs.len() <= 2, "{top}");
        assert!(value.as_f64().unwrap() <= 0.0);
    }
}

#[test]
fn tools_need_a_template_with_tool_calls() {
    let tools =
        r#""tools":[{"type":"function","function":{"name":"now","parameters":{"type":"object"}}}]"#;
    let body =
        format!(r#"{{"messages":[{{"role":"user","content":"time?"}}],"max_tokens":3,{tools}}}"#);
    let r = request(start(None), "POST", "/v1/chat/completions", &[], &body);
    assert_eq!(r.status, 400, "{}", r.body);
    assert!(r.body.contains("no tool calls"), "{}", r.body);

    // With a Qwen3 template, tool definitions and a full tool round trip render.
    let addr = start_with(QWEN3, None);
    let r = request(addr, "POST", "/v1/chat/completions", &[], &body);
    assert_eq!(r.status, 200, "{}", r.body);
    let round_trip = format!(
        r#"{{"messages":[{{"role":"user","content":"time?"}},
            {{"role":"assistant","content":null,"tool_calls":[{{"id":"call_1","type":"function",
              "function":{{"name":"now","arguments":"{{}}"}}}}]}},
            {{"role":"tool","tool_call_id":"call_1","content":"12:00"}}],"max_tokens":3,{tools}}}"#
    );
    let r = request(addr, "POST", "/v1/chat/completions", &[], &round_trip);
    assert_eq!(r.status, 200, "{}", r.body);
}
