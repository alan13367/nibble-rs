//! Tool calls in model output: a streaming parser that separates the reply's
//! content from its `<tool_call>` blocks.

use super::{Tool, ToolCall};
use crate::json::{self, Json};

/// How a template family's models write a tool call inside `<tool_call>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Format {
    /// Qwen3 (Hermes): `{"name": "f", "arguments": {...}}`.
    Json,
    /// Qwen3.5 (Qwen3-Coder): `<function=f>`, then one
    /// `<parameter=key>\nvalue\n</parameter>` block per argument, then
    /// `</function>`.
    Xml,
}

/// What a [`ToolCallParser`] found in the output.
#[derive(Clone, Debug, PartialEq)]
pub enum ToolEvent {
    /// Text of the reply's content.
    Content(String),
    /// A complete tool call.
    Call(ToolCall),
}

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";

/// Separates tool calls from content in model output as it is generated
/// (see [`super::ChatTemplate::tool_call_parser`]).
///
/// Content is passed on as soon as it cannot be the start of a call; the
/// newlines before a call, which the templates put between content and
/// calls, are dropped, as is whitespace between and after calls. A call is
/// reported when its block closes. A block that does not parse, or is still
/// open when the output ends, is passed on as content, so no text is lost.
#[derive(Debug)]
pub struct ToolCallParser {
    format: Format,
    tools: Vec<Tool>,
    pending: String,
    in_call: bool,
    after_call: bool,
    /// The newlines before the current block, given back if it is not a call.
    separator: String,
}

impl ToolCallParser {
    pub(super) fn new(format: Format, tools: Vec<Tool>) -> Self {
        Self {
            format,
            tools,
            pending: String::new(),
            in_call: false,
            after_call: false,
            separator: String::new(),
        }
    }

    /// Adds generated text and returns what it completes.
    pub fn push(&mut self, text: &str) -> Vec<ToolEvent> {
        self.pending.push_str(text);
        let mut events = Vec::new();
        loop {
            if self.in_call {
                let Some(end) = self.pending.find(CLOSE) else {
                    return events;
                };
                let block: String = self.pending.drain(..end + CLOSE.len()).collect();
                let block = &block[..end];
                self.in_call = false;
                self.after_call = true;
                match self.parse(block) {
                    Some(call) => {
                        self.separator.clear();
                        events.push(ToolEvent::Call(call));
                    }
                    None => {
                        let separator = std::mem::take(&mut self.separator);
                        push_content(&mut events, &format!("{separator}{OPEN}{block}{CLOSE}"));
                    }
                }
                continue;
            }
            if self.after_call {
                let ws = self.pending.len() - self.pending.trim_start().len();
                self.pending.drain(..ws);
                if self.pending.is_empty() {
                    return events;
                }
                self.after_call = false;
            }
            if let Some(start) = self.pending.find(OPEN) {
                let before = &self.pending[..start];
                let content = before.trim_end_matches('\n');
                push_content(&mut events, content);
                self.separator = before[content.len()..].to_owned();
                self.pending.drain(..start + OPEN.len());
                self.in_call = true;
                continue;
            }
            // Hold back what may still turn into `\n<tool_call>`.
            let hold = suffix_prefix_len(&self.pending, OPEN);
            let body = &self.pending[..self.pending.len() - hold];
            let cut = body.trim_end_matches('\n').len();
            push_content(&mut events, &self.pending[..cut]);
            self.pending.drain(..cut);
            return events;
        }
    }

    /// Returns what is left at the end of the output.
    pub fn finish(&mut self) -> Vec<ToolEvent> {
        let mut rest = std::mem::take(&mut self.pending);
        if std::mem::take(&mut self.in_call) {
            rest.insert_str(0, OPEN);
            rest.insert_str(0, &std::mem::take(&mut self.separator));
        }
        let mut events = Vec::new();
        if !(std::mem::take(&mut self.after_call) && rest.trim().is_empty()) {
            push_content(&mut events, &rest);
        }
        events
    }

    /// The call in the text between `<tool_call>` and `</tool_call>`.
    fn parse(&self, block: &str) -> Option<ToolCall> {
        match self.format {
            Format::Json => {
                let value = json::parse(block.trim()).ok()?;
                let name = value.get("name")?.as_str()?;
                let arguments = match value.get("arguments") {
                    None | Some(Json::Null) => Json::Object(Vec::new()),
                    // Some models write the arguments as a JSON string.
                    Some(Json::String(text)) => json::parse(text).ok()?,
                    Some(arguments) => arguments.clone(),
                };
                arguments
                    .as_object()
                    .map(|_| ToolCall::new(name, arguments.clone()))
            }
            Format::Xml => self.parse_xml(block.trim()),
        }
    }

    fn parse_xml(&self, block: &str) -> Option<ToolCall> {
        let rest = block.strip_prefix("<function=")?;
        let (name, mut body) = rest.split_once('>')?;
        body = body.strip_suffix("</function>")?;
        let tool = self.tools.iter().find(|t| t.name == name);
        let mut arguments = Vec::new();
        loop {
            body = body.trim_start();
            if body.is_empty() {
                break;
            }
            let rest = body.strip_prefix("<parameter=")?;
            let (key, rest) = rest.split_once('>')?;
            let (raw, rest) = rest.split_once("</parameter>")?;
            // The value sits on its own lines between the tags.
            let raw = raw.strip_prefix('\n').unwrap_or(raw);
            let raw = raw.strip_suffix('\n').unwrap_or(raw);
            let types = tool.map(|t| t.argument_types(key)).unwrap_or_default();
            arguments.push((key.to_owned(), typed_value(raw, &types)));
            body = rest;
        }
        Some(ToolCall::new(name, Json::Object(arguments)))
    }
}

/// An XML argument value as JSON, typed by its schema `types` as llama.cpp
/// does: a string when the schema allows only strings (or declares no type),
/// else the value parsed as JSON if it is one of the allowed types, falling
/// back to a string when the schema also allows strings.
fn typed_value(raw: &str, types: &[&str]) -> Json {
    let as_string = || Json::from(raw);
    if types.is_empty() || types == ["string"] {
        return as_string();
    }
    let allowed = |value: &Json| {
        let ty = match value {
            Json::Null => "null",
            Json::Bool(_) => "boolean",
            Json::Number(_) => {
                if value.as_i64().is_some() && types.contains(&"integer") {
                    "integer"
                } else {
                    "number"
                }
            }
            Json::String(_) => "string",
            Json::Array(_) => "array",
            Json::Object(_) => "object",
        };
        types.contains(&ty)
    };
    match json::parse(raw.trim()) {
        Ok(value) if allowed(&value) => value,
        // Without the string alternative the model broke the schema; keep
        // its text rather than invent a value.
        _ => as_string(),
    }
}

fn push_content(events: &mut Vec<ToolEvent>, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Some(ToolEvent::Content(last)) = events.last_mut() {
        last.push_str(text);
    } else {
        events.push(ToolEvent::Content(text.to_owned()));
    }
}

/// Length of the longest proper prefix of `marker` that `text` ends with.
fn suffix_prefix_len(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .find(|&k| text.ends_with(&marker[..k]))
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tools() -> Vec<Tool> {
        vec![Tool {
            name: "get_weather".into(),
            description: String::new(),
            parameters: json::parse(
                r#"{"type":"object","properties":{
                    "location":{"type":"string"},
                    "days":{"type":"integer"},
                    "units":{"type":["string","null"]},
                    "detailed":{"type":"boolean"},
                    "opts":{"type":"object"}}}"#,
            )
            .unwrap(),
        }]
    }

    /// Events of `output` fed in one piece and one char at a time; both
    /// must agree once adjacent content is merged.
    fn parse(format: Format, output: &str) -> Vec<ToolEvent> {
        let run = |pieces: Vec<&str>| {
            let mut parser = ToolCallParser::new(format, tools());
            let mut events = Vec::new();
            let all = pieces
                .into_iter()
                .flat_map(|p| parser.push(p))
                .collect::<Vec<_>>()
                .into_iter()
                .chain(parser.finish());
            for event in all {
                match event {
                    ToolEvent::Content(text) => push_content(&mut events, &text),
                    call => events.push(call),
                }
            }
            events
        };
        let at_once = run(vec![output]);
        let by_char = run(output
            .char_indices()
            .map(|(i, c)| &output[i..i + c.len_utf8()])
            .collect());
        assert_eq!(
            by_char, at_once,
            "streaming changed the result for {output:?}"
        );
        at_once
    }

    fn content(text: &str) -> ToolEvent {
        ToolEvent::Content(text.into())
    }

    fn call(name: &str, arguments: &str) -> ToolEvent {
        ToolEvent::Call(ToolCall::new(name, json::parse(arguments).unwrap()))
    }

    #[test]
    fn json_calls_are_separated_from_content() {
        let output = "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": \"{\\\"location\\\": \\\"Rome\\\"}\"}\n</tool_call>";
        assert_eq!(
            parse(Format::Json, output),
            [
                content("Let me check."),
                call("get_weather", r#"{"location": "Paris"}"#),
                call("get_weather", r#"{"location": "Rome"}"#),
            ]
        );
    }

    #[test]
    fn xml_arguments_are_typed_by_the_schema() {
        let output = "Checking.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nNew York\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=units>\nnull\n</parameter>\n<parameter=detailed>\ntrue\n</parameter>\n<parameter=opts>\n{\"a\": [1]}\n</parameter>\n<parameter=note>\nline one\nline two\n</parameter>\n</function>\n</tool_call>\n";
        assert_eq!(
            parse(Format::Xml, output),
            [
                content("Checking."),
                call(
                    "get_weather",
                    r#"{"location": "New York", "days": 3, "units": null, "detailed": true, "opts": {"a": [1]}, "note": "line one\nline two"}"#
                ),
            ]
        );
        // A value that breaks a non-string schema is kept as text.
        let output = "<tool_call>\n<function=get_weather>\n<parameter=days>\nthree\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(
            parse(Format::Xml, output),
            [call("get_weather", r#"{"days": "three"}"#)]
        );
    }

    #[test]
    fn text_without_calls_passes_through_unchanged() {
        let output = "Plain answer with <tags>, a < sign\nand trailing newlines\n\n";
        assert_eq!(parse(Format::Json, output), [content(output)]);
        assert_eq!(parse(Format::Xml, output), [content(output)]);
    }

    #[test]
    fn broken_or_unfinished_calls_become_content() {
        let broken = "<tool_call>\n{not json}\n</tool_call>";
        assert_eq!(parse(Format::Json, broken), [content(broken)]);
        let unfinished = "Hm.\n<tool_call>\n<function=get_weather>\n<parameter=location>\nPar";
        assert_eq!(parse(Format::Xml, unfinished), [content(unfinished)]);
        let no_function = "<tool_call>\n{\"name\": \"x\"}\n</tool_call>";
        assert_eq!(parse(Format::Xml, no_function), [content(no_function)]);
    }

    #[test]
    fn text_after_calls_is_content() {
        let output = "<tool_call>\n{\"name\": \"now\"}\n</tool_call>\n\nDone.";
        assert_eq!(
            parse(Format::Json, output),
            [call("now", "{}"), content("Done.")]
        );
    }
}
