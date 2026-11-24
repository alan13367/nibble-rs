//! Chat prompt formatting and tool calls.
//!
//! GGUF files carry a Jinja `tokenizer.chat_template`. Rather than embed a
//! Jinja engine, nibble recognizes template families and renders them
//! natively, byte for byte as the template does (the tests hold llama.cpp's
//! renderings of the real templates). A family also knows how its models
//! write tool calls: [`ChatTemplate::tool_call_parser`] turns generated text
//! back into [`ToolCall`]s.

mod qwen;
mod tool_calls;

pub use tool_calls::{ToolCallParser, ToolEvent};

use crate::error::{Error, Result};
use crate::json::Json;

/// Who wrote a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Instructions for the assistant.
    System,
    /// The user.
    User,
    /// The model.
    Assistant,
    /// The result of a tool call, sent back to the model.
    Tool,
}

impl Role {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

/// One chat message.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct Message {
    /// Who wrote it.
    pub role: Role,
    /// Its text. For assistant messages this may include `<think>...</think>`.
    pub content: String,
    /// The tools an assistant message calls, after its content.
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    fn new(role: Role, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
            tool_calls: Vec::new(),
        }
    }

    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }

    /// A tool's result, answering the assistant's call.
    pub fn tool(content: impl Into<String>) -> Self {
        Self::new(Role::Tool, content)
    }

    /// This message with `tool_calls` (for assistant messages).
    #[must_use]
    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }
}

/// A function the model can call.
#[derive(Clone, Debug, PartialEq)]
pub struct Tool {
    /// The function's name.
    pub name: String,
    /// What it does, for the model.
    pub description: String,
    /// A JSON Schema of its arguments, usually an object schema.
    pub parameters: Json,
}

impl Tool {
    /// The tool as the templates list it, in the shape llama.cpp normalizes
    /// every tool to: `{"type": "function", "function": {"name", "description", "parameters"}}`.
    fn to_json(&self) -> Json {
        Json::object([
            ("type", Json::from("function")),
            (
                "function",
                Json::object([
                    ("name", Json::from(self.name.as_str())),
                    ("description", Json::from(self.description.as_str())),
                    ("parameters", self.parameters.clone()),
                ]),
            ),
        ])
    }

    /// The JSON Schema `type`s allowed for argument `name`, if the schema
    /// declares any.
    fn argument_types(&self, name: &str) -> Vec<&str> {
        let schema = self.parameters.get("properties").and_then(|p| p.get(name));
        match schema.and_then(|s| s.get("type")) {
            Some(Json::String(t)) => vec![t.as_str()],
            Some(Json::Array(types)) => types.iter().filter_map(Json::as_str).collect(),
            _ => Vec::new(),
        }
    }
}

/// A call of a [`Tool`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    /// The tool's name.
    pub name: String,
    /// The arguments, a JSON object.
    pub arguments: Json,
}

impl ToolCall {
    /// A call of `name` with `arguments`.
    pub fn new(name: impl Into<String>, arguments: Json) -> Self {
        Self {
            name: name.into(),
            arguments,
        }
    }
}

/// Rendering options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChatOptions<'a> {
    /// End with the assistant header so the model writes the next reply.
    pub add_generation_prompt: bool,
    /// Let the model reason in `<think>` first. When false, an empty think
    /// block is inserted (Qwen3's `enable_thinking=false`).
    pub enable_thinking: bool,
    /// The tools the model may call; listed in the system prompt.
    pub tools: &'a [Tool],
}

impl Default for ChatOptions<'_> {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
            tools: &[],
        }
    }
}

/// A supported chat template family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatTemplate {
    /// Plain ChatML: `<|im_start|>role\ncontent<|im_end|>\n`. No tool support.
    ChatMl,
    /// Qwen3's ChatML with `<think>` reasoning and JSON tool calls.
    Qwen3,
    /// Qwen3.5's ChatML: trimmed content, an explicit empty think block when
    /// thinking is off, and XML tool calls.
    Qwen35,
}

impl ChatTemplate {
    /// Recognizes the family of a Jinja chat template.
    #[must_use]
    pub fn detect(template: &str) -> Option<Self> {
        if !template.contains("<|im_start|>") {
            return None;
        }
        Some(
            if template.contains("enable_thinking is defined and enable_thinking is true") {
                Self::Qwen35
            } else if template.contains("</think>") {
                Self::Qwen3
            } else {
                Self::ChatMl
            },
        )
    }

    /// Whether models with this template can call tools.
    #[must_use]
    pub const fn supports_tools(self) -> bool {
        !matches!(self, Self::ChatMl)
    }

    /// Renders `messages` into prompt text (special tokens included as text).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidChat`] if the conversation uses tools and the template
    /// has none, or (Qwen3.5) a tool call's arguments are not a JSON object.
    pub fn render(self, messages: &[Message], options: ChatOptions<'_>) -> Result<String> {
        let uses_tools = !options.tools.is_empty()
            || messages
                .iter()
                .any(|m| m.role == Role::Tool || !m.tool_calls.is_empty());
        if uses_tools && !self.supports_tools() {
            return Err(Error::InvalidChat(
                "the model's chat template has no tool calls".into(),
            ));
        }
        match self {
            Self::ChatMl => Ok(render_chatml(messages, options)),
            Self::Qwen3 => Ok(qwen::render_qwen3(messages, options)),
            Self::Qwen35 => qwen::render_qwen35(messages, options),
        }
    }

    /// A parser for the tool calls in this template's model output, which
    /// may call any of `tools`.
    ///
    /// Returns `None` for templates without tool calls.
    #[must_use]
    pub fn tool_call_parser(self, tools: &[Tool]) -> Option<ToolCallParser> {
        let format = match self {
            Self::ChatMl => return None,
            Self::Qwen3 => tool_calls::Format::Json,
            Self::Qwen35 => tool_calls::Format::Xml,
        };
        Some(ToolCallParser::new(format, tools.to_vec()))
    }
}

fn push_turn(out: &mut String, role: Role, content: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role.as_str());
    out.push('\n');
    out.push_str(content);
    out.push_str("<|im_end|>\n");
}

fn render_chatml(messages: &[Message], options: ChatOptions<'_>) -> String {
    let mut out = String::new();
    for m in messages {
        push_turn(&mut out, m.role, &m.content);
    }
    if options.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Splits an assistant message into `(reasoning, answer)` around `</think>`,
/// as the Qwen templates do: the answer is what follows the last
/// `</think>` (leading newlines dropped), the reasoning what precedes the
/// first one, after its last `<think>` (outer newlines dropped).
fn split_reasoning(content: &str) -> (&str, &str) {
    if !content.contains("</think>") {
        return ("", content);
    }
    let answer = content
        .rsplit("</think>")
        .next()
        .unwrap_or("")
        .trim_start_matches('\n');
    let before = content
        .split("</think>")
        .next()
        .unwrap_or("")
        .trim_end_matches('\n');
    let reasoning = before
        .rsplit("<think>")
        .next()
        .unwrap_or("")
        .trim_start_matches('\n');
    (reasoning, answer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_template_families() {
        assert_eq!(
            ChatTemplate::detect("{{'<|im_start|>' + role}}"),
            Some(ChatTemplate::ChatMl)
        );
        assert_eq!(
            ChatTemplate::detect("<|im_start|> ... </think>"),
            Some(ChatTemplate::Qwen3)
        );
        assert_eq!(
            ChatTemplate::detect(
                "<|im_start|> </think> {%- if enable_thinking is defined and enable_thinking is true %}"
            ),
            Some(ChatTemplate::Qwen35)
        );
        assert_eq!(ChatTemplate::detect("[INST]"), None);
    }

    #[test]
    fn chatml_rejects_tools() {
        let tool = Tool {
            name: "f".into(),
            description: String::new(),
            parameters: Json::object::<&str>([]),
        };
        let options = ChatOptions {
            tools: std::slice::from_ref(&tool),
            ..ChatOptions::default()
        };
        let err = ChatTemplate::ChatMl.render(&[Message::user("x")], options);
        assert!(matches!(err, Err(Error::InvalidChat(_))));
        assert!(ChatTemplate::ChatMl.tool_call_parser(&[tool]).is_none());
    }

    #[test]
    fn splits_reasoning_as_the_templates_do() {
        assert_eq!(split_reasoning("plain"), ("", "plain"));
        assert_eq!(
            split_reasoning("<think>\nwhy\n</think>\n\nanswer"),
            ("why", "answer")
        );
        assert_eq!(split_reasoning("r</think>a</think>\nb"), ("r", "b"));
    }
}
