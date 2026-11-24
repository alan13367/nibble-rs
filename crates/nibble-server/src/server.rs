//! Connections and routing: one thread per connection, keep-alive, CORS,
//! optional bearer-token auth, and the OpenAI endpoints.

use std::io::{self, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use nibble::chat::ChatOptions;
use nibble::{Model, SessionOptions};

use crate::http::{self, Limits, RequestHead, Response, SseWriter};
use crate::json::Json;
use crate::openai::{self, ApiError, Generation, ResponseMeta};
use crate::text::ReasoningSplitter;
use crate::worker::{self, Event, Finish, Job};

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerOptions {
    /// The model name reported by `/v1/models` and in responses.
    pub model_id: String,
    /// When set, requests must send `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
    /// Context size and threads of the generation session.
    pub session: SessionOptions,
    /// Concurrent connections; further connections get a 503.
    pub max_connections: usize,
    /// Idle time after which a keep-alive connection is closed.
    pub idle_timeout: Duration,
}

impl ServerOptions {
    /// Defaults for serving `model_id`.
    #[must_use]
    pub fn new(model_id: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            api_key: None,
            session: SessionOptions::default(),
            max_connections: 64,
            idle_timeout: Duration::from_secs(60),
        }
    }
}

struct Shared {
    model: Arc<Model>,
    options: ServerOptions,
    jobs: Sender<Job>,
    active: AtomicUsize,
    next_id: AtomicU64,
    started: u64,
}

/// Serves the OpenAI API on `listener` until accepting fails.
///
/// Generation runs on one worker thread, one request at a time in arrival
/// order; connections are handled concurrently.
///
/// # Errors
///
/// Returns the error that stopped the accept loop.
pub fn serve(listener: TcpListener, model: Model, options: ServerOptions) -> io::Result<()> {
    let model = Arc::new(model);
    let (jobs, queue) = mpsc::channel::<Job>();
    let worker_model = Arc::clone(&model);
    let session = options.session;
    std::thread::Builder::new()
        .name("nibble-worker".into())
        .spawn(move || worker::run(&worker_model, session, &queue))?;

    let shared = Arc::new(Shared {
        model,
        options,
        jobs,
        active: AtomicUsize::new(0),
        next_id: AtomicU64::new(0),
        started: unix_time(),
    });
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(e) if is_transient(&e) => continue,
            Err(e) => return Err(e),
        };
        if shared.active.fetch_add(1, Ordering::AcqRel) >= shared.options.max_connections {
            shared.active.fetch_sub(1, Ordering::AcqRel);
            let busy = ApiError {
                status: 503,
                message: "too many connections".into(),
                kind: "server_error",
                param: None,
            };
            let _ = send_error(&mut stream, &busy, false);
            continue;
        }
        let conn = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("nibble-conn".into())
            .spawn(move || {
                let _ = handle_connection(stream, &conn);
                conn.active.fetch_sub(1, Ordering::AcqRel);
            });
        if spawned.is_err() {
            // The OS refused a thread, so the closure (and its decrement) never ran.
            shared.active.fetch_sub(1, Ordering::AcqRel);
        }
    }
    Ok(())
}

fn is_transient(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::ConnectionAborted
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::Interrupted
    )
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn handle_connection(stream: TcpStream, shared: &Shared) -> io::Result<()> {
    stream.set_read_timeout(Some(shared.options.idle_timeout))?;
    let _ = stream.set_nodelay(true);
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let limits = Limits::default();
    loop {
        let head = match http::read_head(&mut reader, &limits) {
            Ok(Some(head)) => head,
            Ok(None) => return Ok(()),
            Err(e) => {
                if let Some(status) = e.status() {
                    let err = ApiError {
                        status,
                        message: e.to_string(),
                        kind: "invalid_request_error",
                        param: None,
                    };
                    send_error(&mut writer, &err, false)?;
                }
                return Ok(());
            }
        };
        if head.expects_continue() {
            http::write_continue(&mut writer)?;
        }
        let body = match http::read_body(&mut reader, &head, &limits) {
            Ok(body) => body,
            Err(e) => {
                if let Some(status) = e.status() {
                    let err = ApiError {
                        status,
                        message: e.to_string(),
                        kind: "invalid_request_error",
                        param: None,
                    };
                    send_error(&mut writer, &err, false)?;
                }
                return Ok(());
            }
        };
        let keep_alive = head.keep_alive();
        let reusable = route(&head, &body, &mut writer, keep_alive, shared)?;
        if !keep_alive || !reusable {
            return Ok(());
        }
    }
}

const CORS: [(&str, &str); 3] = [
    ("Access-Control-Allow-Origin", "*"),
    ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    (
        "Access-Control-Allow-Headers",
        "Authorization, Content-Type",
    ),
];

fn cors_headers() -> Vec<(String, String)> {
    CORS.iter()
        .map(|&(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

fn with_cors(response: Response) -> Response {
    CORS.iter().fold(response, |r, &(k, v)| r.header(k, v))
}

/// How long a stream may go without output before a keep-alive comment is
/// sent, so proxies do not drop it while the request waits in the queue or
/// a long prompt is evaluated.
const PING_INTERVAL: Duration = Duration::from_secs(10);

/// Receives the next event, sending SSE comments while waiting.
fn next_event<W: Write>(
    rx: &mpsc::Receiver<Event>,
    sse: &mut SseWriter<W>,
) -> io::Result<Option<Event>> {
    loop {
        match rx.recv_timeout(PING_INTERVAL) {
            Ok(event) => return Ok(Some(event)),
            Err(mpsc::RecvTimeoutError::Timeout) => sse.comment("waiting")?,
            Err(mpsc::RecvTimeoutError::Disconnected) => return Ok(None),
        }
    }
}

fn send_json(w: &mut impl Write, status: u16, body: &Json, keep_alive: bool) -> io::Result<()> {
    http::write_response(
        w,
        &with_cors(Response::json(status, body.to_string())),
        keep_alive,
        false,
    )
}

fn send_error(w: &mut impl Write, err: &ApiError, keep_alive: bool) -> io::Result<()> {
    send_json(w, err.status, &err.to_json(), keep_alive)
}

/// Answers one request. Returns whether the connection can serve another.
fn route(
    head: &RequestHead,
    body: &[u8],
    w: &mut TcpStream,
    keep_alive: bool,
    shared: &Shared,
) -> io::Result<bool> {
    let method = head.method.as_str();
    let path = head.path.trim_end_matches('/');
    if method == "OPTIONS" {
        http::write_response(w, &with_cors(Response::new(204)), keep_alive, false)?;
        return Ok(true);
    }
    if path == "/health" {
        send_json(
            w,
            200,
            &Json::object([("status", Json::from("ok"))]),
            keep_alive,
        )?;
        return Ok(true);
    }
    if let Some(key) = &shared.options.api_key {
        let expected = format!("Bearer {key}");
        if head.header("Authorization") != Some(expected.as_str()) {
            send_error(w, &ApiError::unauthorized(), keep_alive)?;
            return Ok(true);
        }
    }
    let result = match (method, path) {
        ("GET" | "HEAD", "/v1/models") => {
            let body = openai::models(&shared.options.model_id, shared.started).to_string();
            let response = with_cors(Response::json(200, body));
            http::write_response(w, &response, keep_alive, method == "HEAD")?;
            return Ok(true);
        }
        ("POST", "/v1/chat/completions") => {
            openai::parse_chat(body).and_then(|r| chat(r, w, keep_alive, shared))
        }
        ("POST", "/v1/completions") => {
            openai::parse_completion(body).and_then(|r| complete(r, w, keep_alive, shared))
        }
        (_, "/v1/models" | "/v1/chat/completions" | "/v1/completions") => Err(ApiError {
            status: 405,
            message: format!("{method} is not allowed on {path}"),
            kind: "invalid_request_error",
            param: None,
        }),
        _ => Err(ApiError::not_found(format!(
            "unknown endpoint {method} {path}"
        ))),
    };
    match result {
        Ok(reusable) => Ok(reusable),
        Err(err) => {
            send_error(w, &err, keep_alive)?;
            Ok(true)
        }
    }
}

/// Tokenizes `text` (special tokens included) with BOS when the model wants it.
fn tokenize(model: &Model, text: &str) -> Vec<u32> {
    let tok = model.tokenizer();
    let mut tokens = Vec::new();
    if tok.add_bos()
        && let Some(bos) = tok.bos()
    {
        tokens.push(bos);
    }
    tokens.extend(tok.encode(text, true));
    tokens
}

/// Queues a job and returns the receiver of its events.
fn submit(
    shared: &Shared,
    prompt: Vec<u32>,
    g: &Generation,
) -> Result<mpsc::Receiver<Event>, ApiError> {
    let (events, rx) = mpsc::channel();
    let job = Job {
        prompt,
        sampler: g.sampler,
        max_tokens: g.max_tokens.unwrap_or(usize::MAX),
        stop: g.stop.clone(),
        events,
    };
    shared
        .jobs
        .send(job)
        .map_err(|_| ApiError::server("the generation worker has stopped"))?;
    Ok(rx)
}

fn new_id(shared: &Shared, prefix: &str) -> String {
    let n = shared.next_id.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}{n:04x}", unix_time())
}

/// Collected output of a non-streamed generation.
struct Collected {
    reasoning: String,
    content: String,
    finish: Finish,
    prompt_tokens: usize,
    completion_tokens: usize,
}

fn collect(
    rx: &mpsc::Receiver<Event>,
    mut splitter: Option<ReasoningSplitter>,
) -> Result<Collected, ApiError> {
    let (mut reasoning, mut content) = (String::new(), String::new());
    for event in rx {
        match event {
            Event::Text(text) => match &mut splitter {
                Some(s) => {
                    let part = s.push(&text);
                    reasoning.push_str(&part.reasoning);
                    content.push_str(&part.content);
                }
                None => content.push_str(&text),
            },
            Event::Done {
                finish,
                prompt_tokens,
                completion_tokens,
            } => {
                if let Some(s) = &mut splitter {
                    let part = s.finish();
                    reasoning.push_str(&part.reasoning);
                    content.push_str(&part.content);
                }
                return Ok(Collected {
                    reasoning,
                    content,
                    finish,
                    prompt_tokens,
                    completion_tokens,
                });
            }
            Event::Error(message) => return Err(ApiError::invalid(message, None)),
        }
    }
    Err(ApiError::server(
        "the generation worker stopped unexpectedly",
    ))
}

fn chat(
    req: openai::ChatRequest,
    w: &mut TcpStream,
    keep_alive: bool,
    shared: &Shared,
) -> Result<bool, ApiError> {
    let model = &shared.model;
    let template = model
        .chat_template()
        .ok_or_else(|| ApiError::server("the model's chat template is not supported"))?;
    let options = ChatOptions {
        add_generation_prompt: true,
        enable_thinking: req.enable_thinking.unwrap_or(true),
    };
    let prompt_text = template.render(&req.messages, options);
    let in_reasoning = prompt_text.ends_with("<think>\n");
    let rx = submit(shared, tokenize(model, &prompt_text), &req.generation)?;
    let id = new_id(shared, "chatcmpl");
    let meta = ResponseMeta {
        id: &id,
        created: unix_time(),
        model: &shared.options.model_id,
    };
    let splitter = ReasoningSplitter::new(in_reasoning);

    if !req.generation.stream {
        let out = collect(&rx, Some(splitter))?;
        let usage = openai::usage(out.prompt_tokens, out.completion_tokens);
        let body = openai::chat_completion(&meta, &out.content, &out.reasoning, out.finish, usage);
        return Ok(send_json(w, 200, &body, keep_alive).is_ok());
    }
    Ok(stream_chat(
        w,
        &rx,
        splitter,
        &meta,
        req.generation.include_usage,
        keep_alive,
    )
    .is_ok())
}

fn stream_chat(
    w: &mut TcpStream,
    rx: &mpsc::Receiver<Event>,
    mut splitter: ReasoningSplitter,
    meta: &ResponseMeta<'_>,
    include_usage: bool,
    keep_alive: bool,
) -> io::Result<()> {
    let mut sse = SseWriter::start(w, &cors_headers(), keep_alive)?;
    let first = vec![
        ("role", Json::from("assistant")),
        ("content", Json::from("")),
    ];
    sse.send(&openai::chat_chunk(meta, first, None).to_string())?;
    let send_split =
        |sse: &mut SseWriter<&mut TcpStream>, part: crate::text::Split| -> io::Result<()> {
            if !part.reasoning.is_empty() {
                let delta = vec![("reasoning_content", Json::from(part.reasoning))];
                sse.send(&openai::chat_chunk(meta, delta, None).to_string())?;
            }
            if !part.content.is_empty() {
                let delta = vec![("content", Json::from(part.content))];
                sse.send(&openai::chat_chunk(meta, delta, None).to_string())?;
            }
            Ok(())
        };
    while let Some(event) = next_event(rx, &mut sse)? {
        match event {
            Event::Text(text) => send_split(&mut sse, splitter.push(&text))?,
            Event::Done {
                finish,
                prompt_tokens,
                completion_tokens,
            } => {
                send_split(&mut sse, splitter.finish())?;
                sse.send(&openai::chat_chunk(meta, Vec::new(), Some(finish)).to_string())?;
                if include_usage {
                    let usage = openai::usage(prompt_tokens, completion_tokens);
                    sse.send(
                        &openai::usage_chunk(meta, "chat.completion.chunk", usage).to_string(),
                    )?;
                }
                sse.send("[DONE]")?;
                sse.finish()?;
                return Ok(());
            }
            Event::Error(message) => {
                sse.send(&ApiError::invalid(message, None).to_json().to_string())?;
                sse.finish()?;
                return Ok(());
            }
        }
    }
    sse.finish()?;
    Err(io::Error::other(
        "the generation worker stopped unexpectedly",
    ))
}

fn complete(
    req: openai::CompletionRequest,
    w: &mut TcpStream,
    keep_alive: bool,
    shared: &Shared,
) -> Result<bool, ApiError> {
    let rx = submit(
        shared,
        tokenize(&shared.model, &req.prompt),
        &req.generation,
    )?;
    let id = new_id(shared, "cmpl");
    let meta = ResponseMeta {
        id: &id,
        created: unix_time(),
        model: &shared.options.model_id,
    };

    if !req.generation.stream {
        let out = collect(&rx, None)?;
        let usage = openai::usage(out.prompt_tokens, out.completion_tokens);
        let body = openai::text_completion(&meta, &out.content, Some(out.finish), Some(usage));
        return Ok(send_json(w, 200, &body, keep_alive).is_ok());
    }
    let streamed = (|| -> io::Result<()> {
        let mut sse = SseWriter::start(&mut *w, &cors_headers(), keep_alive)?;
        while let Some(event) = next_event(&rx, &mut sse)? {
            match event {
                Event::Text(text) => {
                    sse.send(&openai::text_completion(&meta, &text, None, None).to_string())?
                }
                Event::Done {
                    finish,
                    prompt_tokens,
                    completion_tokens,
                } => {
                    sse.send(&openai::text_completion(&meta, "", Some(finish), None).to_string())?;
                    if req.generation.include_usage {
                        let usage = openai::usage(prompt_tokens, completion_tokens);
                        sse.send(
                            &openai::usage_chunk(&meta, "text_completion", usage).to_string(),
                        )?;
                    }
                    sse.send("[DONE]")?;
                    sse.finish()?;
                    return Ok(());
                }
                Event::Error(message) => {
                    sse.send(&ApiError::invalid(message, None).to_json().to_string())?;
                    sse.finish()?;
                    return Ok(());
                }
            }
        }
        sse.finish()?;
        Err(io::Error::other(
            "the generation worker stopped unexpectedly",
        ))
    })();
    Ok(streamed.is_ok())
}
