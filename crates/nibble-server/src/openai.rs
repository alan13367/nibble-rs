//! OpenAI API request parsing and response bodies.
//!
//! Requests are validated strictly: an unsupported feature (tools, images,
//! `n > 1`, logprobs) is a 400 with a clear message rather than being
//! silently ignored. Unknown fields are ignored, as OpenAI does.

use nibble::chat::{Message, Role};
use nibble::sampler::SamplerConfig;

use crate::json::{self, Json};
use crate::worker::Finish;

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

/// Sampling and length options shared by both completion endpoints.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Generation {
    pub(crate) sampler: SamplerConfig,
    /// `None` generates until the end of the turn or the context is full.
    pub(crate) max_tokens: Option<usize>,
    pub(crate) stop: Vec<String>,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ChatRequest {
    pub(crate) messages: Vec<Message>,
    /// `chat_template_kwargs.enable_thinking`, if the client set it.
    pub(crate) enable_thinking: Option<bool>,
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
    let mut messages = Vec::with_capacity(list.len());
    for m in list {
        messages.push(parse_message(m)?);
    }
    if let Some(tools) = v.get("tools").and_then(Json::as_array)
        && !tools.is_empty()
    {
        return Err(ApiError::invalid(
            "tools are not supported yet",
            Some("tools"),
        ));
    }
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
    Ok(ChatRequest {
        messages,
        enable_thinking,
        generation: parse_generation(&v)?,
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
    Ok(CompletionRequest {
        prompt,
        generation: parse_generation(&v)?,
    })
}

fn parse_message(m: &Json) -> Result<Message, ApiError> {
    let invalid = |msg: &str| ApiError::invalid(msg, Some("messages"));
    let role = match m.get("role").and_then(Json::as_str) {
        Some("system" | "developer") => Role::System,
        Some("user") => Role::User,
        Some("assistant") => Role::Assistant,
        Some("tool" | "function") => return Err(invalid("tool messages are not supported yet")),
        Some(other) => {
            return Err(ApiError::invalid(
                format!("unknown message role `{other}`"),
                Some("messages"),
            ));
        }
        None => return Err(invalid("every message needs a `role`")),
    };
    if m.get("tool_calls")
        .and_then(Json::as_array)
        .is_some_and(|c| !c.is_empty())
    {
        return Err(invalid("tool calls are not supported yet"));
    }
    let mut content = match m.get("content") {
        Some(Json::String(s)) => s.clone(),
        None | Some(Json::Null) if role == Role::Assistant => String::new(),
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
    // A prior turn's reasoning, as returned by this server.
    if role == Role::Assistant
        && let Some(reasoning) = m.get("reasoning_content").and_then(Json::as_str)
        && !reasoning.is_empty()
    {
        content = format!("<think>\n{reasoning}\n</think>\n\n{content}");
    }
    Ok(Message { role, content })
}

fn parse_generation(v: &Json) -> Result<Generation, ApiError> {
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
    if v.get("logprobs").and_then(Json::as_bool) == Some(true)
        || v.get("logprobs")
            .and_then(Json::as_u64)
            .is_some_and(|n| n > 0)
    {
        return Err(ApiError::invalid(
            "`logprobs` are not supported yet",
            Some("logprobs"),
        ));
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

/// The body of a non-streamed chat completion.
pub(crate) fn chat_completion(
    meta: &ResponseMeta<'_>,
    content: &str,
    reasoning: &str,
    finish: Finish,
    usage: Json,
) -> Json {
    let mut message = vec![
        ("role", Json::from("assistant")),
        ("content", Json::from(content)),
    ];
    if !reasoning.is_empty() {
        message.push(("reasoning_content", Json::from(reasoning)));
    }
    let choice = Json::object([
        ("index", Json::from(0)),
        ("message", Json::object(message)),
        ("logprobs", Json::Null),
        ("finish_reason", Json::from(finish.as_str())),
    ]);
    envelope(meta, "chat.completion", Json::array([choice]), Some(usage))
}

/// One streamed chat chunk: `delta` fields, and the finish reason on the last one.
pub(crate) fn chat_chunk(
    meta: &ResponseMeta<'_>,
    delta: Vec<(&str, Json)>,
    finish: Option<Finish>,
) -> Json {
    let choice = Json::object([
        ("index", Json::from(0)),
        ("delta", Json::object(delta)),
        ("logprobs", Json::Null),
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
    finish: Option<Finish>,
    usage: Option<Json>,
) -> Json {
    let choice = Json::object([
        ("text", Json::from(text)),
        ("index", Json::from(0)),
        ("logprobs", Json::Null),
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
        let g = r.generation;
        assert_eq!(
            (g.sampler.temperature, g.sampler.top_p, g.sampler.seed),
            (0.2, 0.9, 7)
        );
        assert_eq!(
            (g.max_tokens, g.stop, g.stream, g.include_usage),
            (Some(32), vec!["\n\n".to_owned()], true, true)
        );
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
                br#"{"messages":[{"role":"tool","content":"x"}]}"#,
                Some("messages"),
            ),
            (
                br#"{"messages":[{"role":"user","content":[{"type":"image_url"}]}]}"#,
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
                br#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]}"#,
                Some("tools"),
            ),
            (
                br#"{"messages":[{"role":"user","content":"x"}],"logprobs":true}"#,
                Some("logprobs"),
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
        let body = chat_completion(&meta, "4", "", Finish::Stop, usage(3, 1)).to_string();
        assert_eq!(
            body,
            r#"{"id":"chatcmpl-1","object":"chat.completion","created":5,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"4"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}"#
        );
        let chunk = chat_chunk(&meta, vec![("content", Json::from("hi"))], None).to_string();
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
}
