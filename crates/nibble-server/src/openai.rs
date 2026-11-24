//! OpenAI API request parsing and response bodies.
//!
//! Requests are validated strictly: an unsupported feature (images, `n > 1`,
//! forced tool calls) is a 400 with a clear message rather than being
//! silently ignored. Unknown fields are ignored, as OpenAI does.

use nibble::chat::{Message, Tool, ToolCall};
use nibble::json::{self, Json};
use nibble::sampler::SamplerConfig;

use crate::worker::{Finish, TokenLogprob};

/// An error answered in OpenAI's `{"error": {...}}` shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ApiError {
    pub(crate) status: u16,
    pub(crate) message: String,
    pub(crate) kind: &'static str,
    pub(crate) param: Option<&'static str>,
}

impl ApiError {
    pub(crate) fn invalid(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: 400,
            message: message.into(),
            kind: "invalid_request_error",
            param,
        }
    }

    pub(crate) fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: 404,
            message: message.into(),
            kind: "invalid_request_error",
            param: None,
        }
    }

    pub(crate) fn unauthorized() -> Self {
        Self {
            status: 401,
            message: "invalid or missing API key".into(),
            kind: "authentication_error",
            param: None,
        }
    }

    pub(crate) fn server(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: message.into(),
            kind: "server_error",
            param: None,
        }
    }

    pub(crate) fn to_json(&self) -> Json {
        Json::object([(
            "error",
            Json::object([
                ("message", Json::from(self.message.as_str())),
                ("type", Json::from(self.kind)),
                ("param", Json::from(self.param)),
                ("code", Json::Null),
            ]),
        )])
    }
}

/// The most alternatives per token a request may ask log-probabilities for.
const MAX_TOP_LOGPROBS: u64 = 20;

/// Sampling and length options shared by both completion endpoints.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Generation {
    pub(crate) sampler: SamplerConfig,
    /// `None` generates until the end of the turn or the context is full.
    pub(crate) max_tokens: Option<usize>,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    /// Report log-probabilities, with this many alternatives per token.
    pub(crate) logprobs: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatRequest {
    pub(crate) messages: Vec<Message>,
    /// `chat_template_kwargs.enable_thinking`, if the client set it.
    pub(crate) enable_thinking: Option<bool>,
    pub(crate) tools: Vec<Tool>,
    /// Whether tool calls in the output are returned as calls
    /// (`tool_choice` is `auto`) rather than left in the content (`none`).
    pub(crate) parse_tool_calls: bool,
    pub(crate) parallel_tool_calls: bool,
    pub(crate) generation: Generation,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompletionRequest {
    pub(crate) prompt: String,
    pub(crate) generation: Generation,
}

fn parse_body(body: &[u8]) -> Result<Json, ApiError> {
    let text = std::str::from_utf8(body)
        .map_err(|_| ApiError::invalid("the request body is not valid UTF-8", None))?;
    let value =
        json::parse(text).map_err(|e| ApiError::invalid(format!("invalid JSON: {e}"), None))?;
    if value.as_object().is_none() {
        return Err(ApiError::invalid(
            "the request body must be a JSON object",
            None,
        ));
    }
    Ok(value)
}

pub(crate) fn parse_chat(body: &[u8]) -> Result<ChatRequest, ApiError> {
    let v = parse_body(body)?;
    let list = v
        .get("messages")
        .and_then(Json::as_array)
        .ok_or_else(|| ApiError::invalid("`messages` must be an array", Some("messages")))?;
    if list.is_empty() {
        return Err(ApiError::invalid(
            "`messages` must not be empty",
            Some("messages"),
        ));
    }
    let messages = list
        .iter()
        .map(parse_message)
        .collect::<Result<Vec<_>, _>>()?;
    if let Some(format) = v.get("response_format")
        && format.get("type").and_then(Json::as_str) != Some("text")
    {
        return Err(ApiError::invalid(
            "only the `text` response format is supported",
            Some("response_format"),
        ));
    }
    let enable_thinking = match v
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
    {
        None | Some(Json::Null) => None,
        Some(Json::Bool(b)) => Some(*b),
        Some(_) => {
            return Err(ApiError::invalid(
                "`enable_thinking` must be a boolean",
                Some("chat_template_kwargs"),
            ));
        }
    };
    let tools = parse_tools(&v)?;
    let parse_tool_calls = parse_tool_choice(&v)? && !tools.is_empty();
    let parallel_tool_calls = match v.get("parallel_tool_calls") {
        None | Some(Json::Null) => true,
        Some(Json::Bool(b)) => *b,
        Some(_) => {
            return Err(ApiError::invalid(
                "`parallel_tool_calls` must be a boolean",
                Some("parallel_tool_calls"),
            ));
        }
    };
    let logprobs = match v.get("logprobs") {
        None | Some(Json::Null | Json::Bool(false)) => None,
        Some(Json::Bool(true)) => Some(top_logprobs(&v, "top_logprobs")?.unwrap_or(0)),
        Some(_) => {
            return Err(ApiError::invalid(
                "`logprobs` must be a boolean",
                Some("logprobs"),
            ));
        }
    };
    if logprobs.is_none() && top_logprobs(&v, "top_logprobs")?.is_some_and(|n| n > 0) {
        return Err(ApiError::invalid(
            "`top_logprobs` needs `logprobs: true`",
            Some("top_logprobs"),
        ));
    }
    Ok(ChatRequest {
        messages,
        enable_thinking,
        tools,
        parse_tool_calls,
        parallel_tool_calls,
        generation: parse_generation(&v, logprobs)?,
    })
}

pub(crate) fn parse_completion(body: &[u8]) -> Result<CompletionRequest, ApiError> {
    let v = parse_body(body)?;
    let prompt = match v.get("prompt") {
        Some(Json::String(s)) => s.clone(),
        Some(Json::Array(items)) if items.len() == 1 && items[0].as_str().is_some() => {
            items[0].as_str().unwrap_or_default().to_owned()
        }
        Some(Json::Array(_)) => {
            return Err(ApiError::invalid(
                "only a single prompt string is supported",
                Some("prompt"),
            ));
        }
        _ => {
            return Err(ApiError::invalid(
                "`prompt` must be a string",
                Some("prompt"),
            ));
        }
    };
    if v.get("echo").and_then(Json::as_bool) == Some(true) {
        return Err(ApiError::invalid("`echo` is not supported", Some("echo")));
    }
    if v.get("suffix").is_some_and(|s| !s.is_null()) {
        return Err(ApiError::invalid(
            "`suffix` is not supported",
            Some("suffix"),
        ));
    }
    // The legacy API asks for logprobs with the number of alternatives.
    let logprobs = top_logprobs(&v, "logprobs")?;
    Ok(CompletionRequest {
        prompt,
        generation: parse_generation(&v, logprobs)?,
    })
}

/// An optional count of alternatives per token, `0..=MAX_TOP_LOGPROBS`.
fn top_logprobs(v: &Json, key: &'static str) -> Result<Option<usize>, ApiError> {
    match v.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(n) => match n.as_u64() {
            Some(n) if n <= MAX_TOP_LOGPROBS => Ok(Some(n as usize)),
            _ => Err(ApiError::invalid(
                format!("`{key}` must be an integer from 0 to {MAX_TOP_LOGPROBS}"),
                Some(key),
            )),
        },
    }
}

/// `tools`: function definitions, normalized as llama.cpp does (a missing
/// description is empty, missing parameters are `{}`).
fn parse_tools(v: &Json) -> Result<Vec<Tool>, ApiError> {
    let invalid = |msg: String| ApiError::invalid(msg, Some("tools"));
    let list = match v.get("tools") {
        None | Some(Json::Null) => return Ok(Vec::new()),
        Some(Json::Array(list)) => list,
        Some(_) => return Err(invalid("`tools` must be an array".into())),
    };
    list.iter()
        .enumerate()
        .map(|(i, tool)| {
            if tool.get("type").and_then(Json::as_str) != Some("function") {
                return Err(invalid(format!(
                    "tools[{i}]: only tools of type `function` are supported"
                )));
            }
            let function = tool
                .get("function")
                .ok_or_else(|| invalid(format!("tools[{i}] has no `function`")))?;
            let name = function
                .get("name")
                .and_then(Json::as_str)
                .filter(|n| !n.is_empty())
                .ok_or_else(|| invalid(format!("tools[{i}].function needs a `name`")))?;
            let description = match function.get("description") {
                None | Some(Json::Null) => String::new(),
                Some(Json::String(d)) => d.clone(),
                Some(_) => {
                    return Err(invalid(format!(
                        "tools[{i}].function.description must be a string"
                    )));
                }
            };
            let parameters = match function.get("parameters") {
                None | Some(Json::Null) => Json::Object(Vec::new()),
                Some(p @ Json::Object(_)) => p.clone(),
                Some(_) => {
                    return Err(invalid(format!(
                        "tools[{i}].function.parameters must be a JSON Schema object"
                    )));
                }
            };
            Ok(Tool {
                name: name.to_owned(),
                description,
                parameters,
            })
        })
        .collect()
}

/// `tool_choice`: whether tool calls are parsed (`auto`) or not (`none`;
/// the tools are still shown to the model, as llama.cpp does).
fn parse_tool_choice(v: &Json) -> Result<bool, ApiError> {
    let unsupported = || {
        ApiError::invalid(
            "forcing a tool call needs constrained decoding, which nibble does not have yet; \
             use `tool_choice: \"auto\"`",
            Some("tool_choice"),
        )
    };
    match v.get("tool_choice") {
        None | Some(Json::Null) => Ok(true),
        Some(Json::String(s)) => match s.as_str() {
            "auto" => Ok(true),
            "none" => Ok(false),
            "required" => Err(unsupported()),
            other => Err(ApiError::invalid(
                format!("unknown `tool_choice` `{other}`"),
                Some("tool_choice"),
            )),
        },
        Some(Json::Object(_)) => Err(unsupported()),
        Some(_) => Err(ApiError::invalid(
            "`tool_choice` must be a string or an object",
            Some("tool_choice"),
        )),
    }
}

fn parse_message(m: &Json) -> Result<Message, ApiError> {
    let invalid = |msg: &str| ApiError::invalid(msg, Some("messages"));
    let role = m
        .get("role")
        .and_then(Json::as_str)
        .ok_or_else(|| invalid("every message needs a `role`"))?;
    let content = match m.get("content") {
        Some(Json::String(s)) => s.clone(),
        None | Some(Json::Null) if role == "assistant" => String::new(),
        Some(Json::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                match part.get("type").and_then(Json::as_str) {
                    Some("text") => {
                        text.push_str(part.get("text").and_then(Json::as_str).unwrap_or_default())
                    }
                    Some(other) => {
                        return Err(ApiError::invalid(
                            format!("content parts of type `{other}` are not supported"),
                            Some("messages"),
                        ));
                    }
                    None => return Err(invalid("every content part needs a `type`")),
                }
            }
            text
        }
        _ => {
            return Err(invalid(
                "message `content` must be a string or an array of text parts",
            ));
        }
    };
    let message = match role {
        "system" | "developer" => Message::system(content),
        "user" => Message::user(content),
        "tool" => Message::tool(content),
        "assistant" => {
            // A prior turn's reasoning, as returned by this server, goes
            // back where the templates look for it.
            let content = match m.get("reasoning_content").and_then(Json::as_str) {
                Some(reasoning) if !reasoning.is_empty() => {
                    format!("<think>\n{reasoning}\n</think>\n\n{content}")
                }
                _ => content,
            };
            Message::assistant(content).with_tool_calls(parse_tool_calls(m)?)
        }
        other => {
            return Err(ApiError::invalid(
                format!("unknown message role `{other}`"),
                Some("messages"),
            ));
        }
    };
    Ok(message)
}

/// The `tool_calls` of an assistant message. Arguments arrive as a JSON
/// string (or an object) and must hold an object.
fn parse_tool_calls(m: &Json) -> Result<Vec<ToolCall>, ApiError> {
    let invalid = |msg: String| ApiError::invalid(msg, Some("messages"));
    let calls = match m.get("tool_calls") {
        None | Some(Json::Null) => return Ok(Vec::new()),
        Some(Json::Array(calls)) => calls,
        Some(_) => return Err(invalid("`tool_calls` must be an array".into())),
    };
    calls
        .iter()
        .map(|call| {
            let function = call
                .get("function")
                .ok_or_else(|| invalid("every tool call needs a `function`".into()))?;
            let name = function
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| invalid("every tool call needs a function `name`".into()))?;
            let arguments = match function.get("arguments") {
                None | Some(Json::Null) => Json::Object(Vec::new()),
                Some(Json::String(text)) => json::parse(text).map_err(|e| {
                    invalid(format!(
                        "the arguments of tool call `{name}` are not valid JSON: {e}"
                    ))
                })?,
                Some(other) => other.clone(),
            };
            if arguments.as_object().is_none() {
                return Err(invalid(format!(
                    "the arguments of tool call `{name}` must be a JSON object"
                )));
            }
            Ok(ToolCall::new(name, arguments))
        })
        .collect()
}

fn parse_generation(v: &Json, logprobs: Option<usize>) -> Result<Generation, ApiError> {
    let mut sampler = SamplerConfig::default();
    if let Some(t) = number(v, "temperature", 0.0, 2.0)? {
        sampler.temperature = t as f32;
    }
    if let Some(p) = number(v, "top_p", 0.0, 1.0)? {
        sampler.top_p = p as f32;
    }
    if let Some(p) = number(v, "min_p", 0.0, 1.0)? {
        sampler.min_p = p as f32;
    }
    if let Some(p) = number(v, "presence_penalty", -2.0, 2.0)? {
        sampler.presence_penalty = p as f32;
    }
    if let Some(p) = number(v, "frequency_penalty", -2.0, 2.0)? {
        sampler.frequency_penalty = p as f32;
    }
    if let Some(p) = number(v, "repeat_penalty", 0.0, 10.0)? {
        sampler.repeat_penalty = p as f32;
    }
    if let Some(k) = integer(v, "top_k")? {
        sampler.top_k = k;
    }
    sampler.seed = match v.get("seed") {
        None | Some(Json::Null) => random_seed(),
        Some(s) => s
            .as_i64()
            .map(|s| s as u64)
            .or_else(|| s.as_u64())
            .ok_or_else(|| ApiError::invalid("`seed` must be an integer", Some("seed")))?,
    };

    let max_tokens = match integer(v, "max_completion_tokens")? {
        Some(n) => Some(n),
        None => integer(v, "max_tokens")?,
    };
    if max_tokens == Some(0) {
        return Err(ApiError::invalid(
            "`max_tokens` must be at least 1",
            Some("max_tokens"),
        ));
    }
    if let Some(n) = integer(v, "n")?
        && n != 1
    {
        return Err(ApiError::invalid("only `n = 1` is supported", Some("n")));
    }

    let stop = match v.get("stop") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::String(s)) => vec![s.clone()],
        Some(Json::Array(items)) => {
            let stops: Option<Vec<String>> = items
                .iter()
                .map(|s| s.as_str().map(str::to_owned))
                .collect();
            let stops =
                stops.ok_or_else(|| ApiError::invalid("`stop` must hold strings", Some("stop")))?;
            if stops.len() > 4 {
                return Err(ApiError::invalid(
                    "at most 4 stop sequences are allowed",
                    Some("stop"),
                ));
            }
            stops
        }
        Some(_) => {
            return Err(ApiError::invalid(
                "`stop` must be a string or an array of strings",
                Some("stop"),
            ));
        }
    };

    let stream = match v.get("stream") {
        None | Some(Json::Null) => false,
        Some(Json::Bool(b)) => *b,
        Some(_) => {
            return Err(ApiError::invalid(
                "`stream` must be a boolean",
                Some("stream"),
            ));
        }
    };
    let include_usage = v
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(Json::as_bool)
        .unwrap_or(false);
    Ok(Generation {
        sampler,
        max_tokens,
        stop,
        stream,
        include_usage,
        logprobs,
    })
}

/// An optional number field within `[min, max]`.
fn number(v: &Json, key: &'static str, min: f64, max: f64) -> Result<Option<f64>, ApiError> {
    match v.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(x) => match x.as_f64() {
            Some(n) if (min..=max).contains(&n) => Ok(Some(n)),
            _ => Err(ApiError::invalid(
                format!("`{key}` must be a number from {min} to {max}"),
                Some(key),
            )),
        },
    }
}

/// An optional non-negative integer field.
fn integer(v: &Json, key: &'static str) -> Result<Option<usize>, ApiError> {
    match v.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(x) => x
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .map(Some)
            .ok_or_else(|| {
                ApiError::invalid(format!("`{key}` must be a non-negative integer"), Some(key))
            }),
    }
}

fn random_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    nanos ^ u64::from(std::process::id()).rotate_left(32)
}

/// Identity of one response: id, creation time and model name.
pub(crate) struct ResponseMeta<'a> {
    pub(crate) id: &'a str,
    pub(crate) created: u64,
    pub(crate) model: &'a str,
}

pub(crate) fn usage(prompt_tokens: usize, completion_tokens: usize) -> Json {
    Json::object([
        ("prompt_tokens", Json::from(prompt_tokens)),
        ("completion_tokens", Json::from(completion_tokens)),
        (
            "total_tokens",
            Json::from(prompt_tokens + completion_tokens),
        ),
    ])
}

fn envelope(meta: &ResponseMeta<'_>, object: &str, choices: Json, usage: Option<Json>) -> Json {
    let mut fields = vec![
        ("id", Json::from(meta.id)),
        ("object", Json::from(object)),
        ("created", Json::from(meta.created)),
        ("model", Json::from(meta.model)),
        ("choices", choices),
    ];
    if let Some(usage) = usage {
        fields.push(("usage", usage));
    }
    Json::object(fields)
}

/// A tool call made by the model, with the id it is answered by.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CallMade {
    pub(crate) index: usize,
    pub(crate) id: String,
    pub(crate) call: ToolCall,
}

impl CallMade {
    /// The call as OpenAI writes it: arguments are a JSON string.
    fn to_json(&self, with_index: bool) -> Json {
        let mut fields = Vec::new();
        if with_index {
            fields.push(("index", Json::from(self.index)));
        }
        fields.extend([
            ("id", Json::from(self.id.as_str())),
            ("type", Json::from("function")),
            (
                "function",
                Json::object([
                    ("name", Json::from(self.call.name.as_str())),
                    ("arguments", Json::from(self.call.arguments.to_string())),
                ]),
            ),
        ]);
        Json::object(fields)
    }
}

/// The text and bytes of a token, as the logprobs entries show them.
fn token_text(bytes: &[u8]) -> (Json, Json) {
    (
        Json::from(String::from_utf8_lossy(bytes).into_owned()),
        Json::array(bytes.iter().map(|&b| Json::from(u32::from(b)))),
    )
}

/// The chat `logprobs` object: `{"content": [...], "refusal": null}`, one
/// entry per generated token (reasoning tokens included).
pub(crate) fn chat_logprobs(tokens: &[TokenLogprob]) -> Json {
    let entry = |bytes: &[u8], logprob: f32| {
        let (token, bytes) = token_text(bytes);
        vec![
            ("token", token),
            ("logprob", Json::from(logprob)),
            ("bytes", bytes),
        ]
    };
    let content = tokens.iter().map(|t| {
        let mut fields = entry(&t.bytes, t.logprob);
        let top = t
            .top
            .iter()
            .map(|(bytes, logprob)| Json::object(entry(bytes, *logprob)));
        fields.push(("top_logprobs", Json::array(top)));
        Json::object(fields)
    });
    Json::object([("content", Json::array(content)), ("refusal", Json::Null)])
}

/// The legacy completions `logprobs` object. `text_offset` counts the
/// characters of the tokens before each one, starting at `offset`.
pub(crate) fn completion_logprobs(tokens: &[TokenLogprob], mut offset: usize) -> Json {
    let mut texts = Vec::new();
    let mut logprobs = Vec::new();
    let mut tops = Vec::new();
    let mut offsets = Vec::new();
    for t in tokens {
        let text = String::from_utf8_lossy(&t.bytes).into_owned();
        offsets.push(Json::from(offset));
        offset += text.chars().count();
        texts.push(Json::from(text));
        logprobs.push(Json::from(t.logprob));
        tops.push(Json::object(t.top.iter().map(|(bytes, logprob)| {
            (
                String::from_utf8_lossy(bytes).into_owned(),
                Json::from(*logprob),
            )
        })));
    }
    Json::object([
        ("tokens", Json::Array(texts)),
        ("token_logprobs", Json::Array(logprobs)),
        ("top_logprobs", Json::Array(tops)),
        ("text_offset", Json::Array(offsets)),
    ])
}

/// The assistant message of a non-streamed chat completion.
pub(crate) struct Reply<'a> {
    pub(crate) content: &'a str,
    pub(crate) reasoning: &'a str,
    pub(crate) tool_calls: &'a [CallMade],
    /// `Some` when the request asked for logprobs.
    pub(crate) logprobs: Option<&'a [TokenLogprob]>,
}

/// The body of a non-streamed chat completion.
pub(crate) fn chat_completion(
    meta: &ResponseMeta<'_>,
    reply: &Reply<'_>,
    finish: Finish,
    usage: Json,
) -> Json {
    // A reply that only calls tools has no content.
    let content = if reply.content.is_empty() && !reply.tool_calls.is_empty() {
        Json::Null
    } else {
        Json::from(reply.content)
    };
    let mut message = vec![("role", Json::from("assistant")), ("content", content)];
    if !reply.reasoning.is_empty() {
        message.push(("reasoning_content", Json::from(reply.reasoning)));
    }
    if !reply.tool_calls.is_empty() {
        let calls = reply.tool_calls.iter().map(|c| c.to_json(false));
        message.push(("tool_calls", Json::array(calls)));
    }
    let choice = Json::object([
        ("index", Json::from(0)),
        ("message", Json::object(message)),
        ("logprobs", Json::from(reply.logprobs.map(chat_logprobs))),
        ("finish_reason", Json::from(finish.as_str())),
    ]);
    envelope(meta, "chat.completion", Json::array([choice]), Some(usage))
}

/// What one streamed chat chunk adds to the reply.
pub(crate) enum Delta<'a> {
    /// The first chunk: the role, and empty content.
    Start,
    Reasoning(&'a str),
    Content(&'a str),
    ToolCall(&'a CallMade),
    /// No new text; only logprobs or the finish reason.
    Empty,
}

/// One streamed chat chunk.
pub(crate) fn chat_chunk(
    meta: &ResponseMeta<'_>,
    delta: &Delta<'_>,
    logprobs: Option<&[TokenLogprob]>,
    finish: Option<Finish>,
) -> Json {
    let delta = match delta {
        Delta::Start => vec![
            ("role", Json::from("assistant")),
            ("content", Json::from("")),
        ],
        Delta::Reasoning(text) => vec![("reasoning_content", Json::from(*text))],
        Delta::Content(text) => vec![("content", Json::from(*text))],
        Delta::ToolCall(call) => vec![("tool_calls", Json::array([call.to_json(true)]))],
        Delta::Empty => Vec::new(),
    };
    let choice = Json::object([
        ("index", Json::from(0)),
        ("delta", Json::object(delta)),
        ("logprobs", Json::from(logprobs.map(chat_logprobs))),
        ("finish_reason", Json::from(finish.map(Finish::as_str))),
    ]);
    envelope(meta, "chat.completion.chunk", Json::array([choice]), None)
}

/// The final usage-only chunk sent when `stream_options.include_usage` is set.
pub(crate) fn usage_chunk(meta: &ResponseMeta<'_>, object: &str, usage: Json) -> Json {
    envelope(meta, object, Json::array([]), Some(usage))
}

/// A text completion body, or one streamed piece of it.
pub(crate) fn text_completion(
    meta: &ResponseMeta<'_>,
    text: &str,
    logprobs: Option<Json>,
    finish: Option<Finish>,
    usage: Option<Json>,
) -> Json {
    let choice = Json::object([
        ("text", Json::from(text)),
        ("index", Json::from(0)),
        ("logprobs", Json::from(logprobs)),
        ("finish_reason", Json::from(finish.map(Finish::as_str))),
    ]);
    envelope(meta, "text_completion", Json::array([choice]), usage)
}

pub(crate) fn models(id: &str, created: u64) -> Json {
    let model = Json::object([
        ("id", Json::from(id)),
        ("object", Json::from("model")),
        ("created", Json::from(created)),
        ("owned_by", Json::from("nibble")),
    ]);
    Json::object([
        ("object", Json::from("list")),
        ("data", Json::array([model])),
    ])
}

#[cfg(test)]
mod tests {
    use nibble::chat::Role;

    use super::*;

    #[test]
    fn parses_a_typical_chat_request() {
        let body = br#"{"model":"x","messages":[
            {"role":"system","content":"Be brief."},
            {"role":"user","content":[{"type":"text","text":"Hi "},{"type":"text","text":"there"}]},
            {"role":"assistant","content":"Hello","reasoning_content":"greet"},
            {"role":"user","content":"Bye"}],
            "temperature":0.2,"top_p":0.9,"max_tokens":32,"stop":["\n\n"],"seed":7,
            "stream":true,"stream_options":{"include_usage":true},
            "chat_template_kwargs":{"enable_thinking":false},"unknown_field":1}"#;
        let r = parse_chat(body).unwrap();
        assert_eq!(r.messages.len(), 4);
        assert_eq!(r.messages[1], Message::user("Hi there"));
        assert_eq!(
            r.messages[2],
            Message::assistant("<think>\ngreet\n</think>\n\nHello")
        );
        assert_eq!(r.enable_thinking, Some(false));
        assert!(r.tools.is_empty() && !r.parse_tool_calls);
        let g = r.generation;
        assert_eq!(
            (g.sampler.temperature, g.sampler.top_p, g.sampler.seed),
            (0.2, 0.9, 7)
        );
        assert_eq!(
            (g.max_tokens, g.stop, g.stream, g.include_usage, g.logprobs),
            (Some(32), vec!["\n\n".to_owned()], true, true, None)
        );
    }

    #[test]
    fn parses_tools_and_tool_messages() {
        let body = br#"{"messages":[
            {"role":"user","content":"Weather?"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function",
                "function":{"name":"get_weather","arguments":"{\"location\":\"Paris\"}"}}]},
            {"role":"tool","tool_call_id":"c1","content":"18 C"}],
            "tools":[{"type":"function","function":{"name":"get_weather","strict":true,
                "parameters":{"type":"object"}}},
                {"type":"function","function":{"name":"now"}}],
            "parallel_tool_calls":false}"#;
        let r = parse_chat(body).unwrap();
        assert_eq!(r.messages[1].role, Role::Assistant);
        assert_eq!(r.messages[1].content, "");
        assert_eq!(
            r.messages[1].tool_calls,
            [ToolCall::new(
                "get_weather",
                json::parse(r#"{"location":"Paris"}"#).unwrap()
            )]
        );
        assert_eq!(r.messages[2], Message::tool("18 C"));
        assert_eq!(r.tools.len(), 2);
        assert_eq!(
            (r.tools[1].name.as_str(), r.tools[1].description.as_str()),
            ("now", "")
        );
        assert_eq!(r.tools[1].parameters, Json::Object(Vec::new()));
        assert!(r.parse_tool_calls && !r.parallel_tool_calls);

        let none = br#"{"messages":[{"role":"user","content":"x"}],"tool_choice":"none",
            "tools":[{"type":"function","function":{"name":"f"}}]}"#;
        let r = parse_chat(none).unwrap();
        assert!(r.tools.len() == 1 && !r.parse_tool_calls);
    }

    #[test]
    fn parses_logprobs_options() {
        let chat = |extra: &str| {
            parse_chat(
                format!(r#"{{"messages":[{{"role":"user","content":"x"}}]{extra}}}"#).as_bytes(),
            )
        };
        assert_eq!(chat("").unwrap().generation.logprobs, None);
        assert_eq!(
            chat(r#","logprobs":true"#).unwrap().generation.logprobs,
            Some(0)
        );
        assert_eq!(
            chat(r#","logprobs":true,"top_logprobs":5"#)
                .unwrap()
                .generation
                .logprobs,
            Some(5)
        );
        assert_eq!(
            chat(r#","top_logprobs":2"#).unwrap_err().param,
            Some("top_logprobs")
        );
        assert_eq!(
            chat(r#","logprobs":true,"top_logprobs":21"#)
                .unwrap_err()
                .param,
            Some("top_logprobs")
        );
        let completion = parse_completion(br#"{"prompt":"x","logprobs":3}"#).unwrap();
        assert_eq!(completion.generation.logprobs, Some(3));
    }

    #[test]
    fn max_completion_tokens_wins_over_max_tokens() {
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"x"}],"max_tokens":5,"max_completion_tokens":9}"#);
        assert_eq!(r.unwrap().generation.max_tokens, Some(9));
    }

    #[test]
    fn rejects_unsupported_and_invalid_requests() {
        let cases: &[(&[u8], Option<&str>)] = &[
            (b"not json", None),
            (b"[1]", None),
            (br#"{"messages":[]}"#, Some("messages")),
            (br#"{"messages":[{"content":"x"}]}"#, Some("messages")),
            (
                br#"{"messages":[{"role":"tool"}]}"#,
                Some("messages"),
            ),
            (
                br#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#,
                Some("messages"),
            ),
            (
                br#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"{oops"}}]}]}"#,
                Some("messages"),
            ),
            (
                br#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"[1]"}}]}]}"#,
                Some("messages"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"temperature":3}"#,
                Some("temperature"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"n":2}"#,
                Some("n"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"max_tokens":0}"#,
                Some("max_tokens"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"stop":[1]}"#,
                Some("stop"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"retrieval"}]}"#,
                Some("tools"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{}}]}"#,
                Some("tools"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"tool_choice":"required"}"#,
                Some("tool_choice"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"tool_choice":{"type":"function","function":{"name":"f"}}}"#,
                Some("tool_choice"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"stream":"yes"}"#,
                Some("stream"),
            ),
        ];
        for (body, param) in cases {
            let err = parse_chat(body).unwrap_err();
            assert_eq!(err.status, 400, "{}", String::from_utf8_lossy(body));
            assert_eq!(err.param, *param, "{}", String::from_utf8_lossy(body));
        }
    }

    #[test]
    fn parses_completions() {
        assert_eq!(
            parse_completion(br#"{"prompt":"Once"}"#).unwrap().prompt,
            "Once"
        );
        assert_eq!(
            parse_completion(br#"{"prompt":["Once"]}"#).unwrap().prompt,
            "Once"
        );
        assert_eq!(
            parse_completion(br#"{"prompt":["a","b"]}"#)
                .unwrap_err()
                .param,
            Some("prompt")
        );
        assert_eq!(
            parse_completion(br#"{"prompt":"x","echo":true}"#)
                .unwrap_err()
                .param,
            Some("echo")
        );
    }

    #[test]
    fn response_bodies_have_the_openai_shape() {
        let meta = ResponseMeta {
            id: "chatcmpl-1",
            created: 5,
            model: "m",
        };
        let reply = Reply {
            content: "4",
            reasoning: "",
            tool_calls: &[],
            logprobs: None,
        };
        let body = chat_completion(&meta, &reply, Finish::Stop, usage(3, 1)).to_string();
        assert_eq!(
            body,
            r#"{"id":"chatcmpl-1","object":"chat.completion","created":5,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#
        );
        let chunk = chat_chunk(&meta, &Delta::Content("hi"), None, None).to_string();
        assert!(
            chunk.contains(r#""object":"chat.completion.chunk""#)
                && chunk
                    .contains(r#""delta":{"content":"hi"},"logprobs":null,"finish_reason":null"#)
        );
        let err = ApiError::invalid("bad", Some("n")).to_json().to_string();
        assert_eq!(
            err,
            r#"{"error":{"message":"bad","type":"invalid_request_error","param":"n","code":null}}"#
        );
    }

    #[test]
    fn tool_calls_and_logprobs_have_the_openai_shape() {
        let meta = ResponseMeta {
            id: "chatcmpl-1",
            created: 5,
            model: "m",
        };
        let call = CallMade {
            index: 0,
            id: "call_1".into(),
            call: ToolCall::new("f", json::parse(r#"{"a": 1}"#).unwrap()),
        };
        let logprobs = [TokenLogprob {
            bytes: b"Hi".to_vec(),
            logprob: -0.5,
            top: vec![(b"Hi".to_vec(), -0.5), (b" Hey".to_vec(), -1.5)],
        }];
        let reply = Reply {
            content: "",
            reasoning: "",
            tool_calls: std::slice::from_ref(&call),
            logprobs: Some(&logprobs),
        };
        let body = chat_completion(&meta, &reply, Finish::ToolCalls, usage(1, 1)).to_string();
        assert!(body.contains(r#""message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}"#), "{body}");
        assert!(body.contains(r#""logprobs":{"content":[{"token":"Hi","logprob":-0.5,"bytes":[72,105],"top_logprobs":[{"token":"Hi","logprob":-0.5,"bytes":[72,105]},{"token":" Hey","logprob":-1.5,"bytes":[32,72,101,121]}]}],"refusal":null}"#), "{body}");
        assert!(body.contains(r#""finish_reason":"tool_calls""#));

        let chunk = chat_chunk(&meta, &Delta::ToolCall(&call), None, None).to_string();
        assert!(chunk.contains(r#""delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]}"#), "{chunk}");

        let legacy = completion_logprobs(&logprobs, 3).to_string();
        assert_eq!(
            legacy,
            r#"{"tokens":["Hi"],"token_logprobs":[-0.5],"top_logprobs":[{"Hi":-0.5," Hey":-1.5}],"text_offset":[3]}"#
        );
    }
}
