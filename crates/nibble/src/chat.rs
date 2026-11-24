//! Chat prompt formatting.
//!
//! GGUF files carry a Jinja `tokenizer.chat_template`. Rather than embed a
//! Jinja engine, nibble recognizes template families and renders them natively.

/// Who wrote a message.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Instructions for the assistant.
    System,
    /// The user.
    User,
    /// The model.
    Assistant,
}

impl Role {
    const fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

/// One chat message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    /// Who wrote it.
    pub role: Role,
    /// Its text. For assistant messages this may include `<think>...</think>`.
    pub content: String,
}

impl Message {
    /// A system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// A user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// An assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// Rendering options.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChatOptions {
    /// End with the assistant header so the model writes the next reply.
    pub add_generation_prompt: bool,
    /// Let the model reason in `<think>` first. When false, an empty think
    /// block is inserted (Qwen3's `enable_thinking=false`).
    pub enable_thinking: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        Self {
            add_generation_prompt: true,
            enable_thinking: true,
        }
    }
}

/// A supported chat template family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatTemplate {
    /// Plain ChatML: `<|im_start|>role\ncontent<|im_end|>\n`.
    ChatMl,
    /// Qwen3's ChatML with `<think>` reasoning handling.
    Qwen3,
    /// Qwen3.5's ChatML: trimmed content, and an explicit empty think block
    /// when thinking is off.
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

    /// Renders `messages` into prompt text (special tokens included as text).
    #[must_use]
    pub fn render(self, messages: &[Message], options: ChatOptions) -> String {
        match self {
            Self::ChatMl => render_chatml(messages, options),
            Self::Qwen3 => render_qwen3(messages, options),
            Self::Qwen35 => render_qwen35(messages, options),
        }
    }
}

/// Follows Qwen3.5's Jinja template (text only, without tools): message
/// content is trimmed, reasoning after the last user message is kept in a
/// think block, and the generation prompt opens (thinking) or closes
/// (no thinking) the think block.
fn render_qwen35(messages: &[Message], options: ChatOptions) -> String {
    let mut out = String::new();
    if let Some(first) = messages.first()
        && first.role == Role::System
    {
        push_turn(&mut out, Role::System, first.content.trim());
    }
    let last_query = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len().saturating_sub(1));
    for (i, m) in messages.iter().enumerate() {
        let content = m.content.trim();
        match m.role {
            Role::System if i == 0 => {}
            Role::System | Role::User => push_turn(&mut out, m.role, content),
            Role::Assistant => {
                let (reasoning, answer) = split_reasoning(content);
                out.push_str("<|im_start|>assistant\n");
                if i > last_query {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.trim());
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(answer);
                out.push_str("<|im_end|>\n");
            }
        }
    }
    if options.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        out.push_str(if options.enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        });
    }
    out
}

fn push_turn(out: &mut String, role: Role, content: &str) {
    out.push_str("<|im_start|>");
    out.push_str(role.as_str());
    out.push('\n');
    out.push_str(content);
    out.push_str("<|im_end|>\n");
}

fn render_chatml(messages: &[Message], options: ChatOptions) -> String {
    let mut out = String::new();
    for m in messages {
        push_turn(&mut out, m.role, &m.content);
    }
    if options.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
    }
    out
}

/// Follows Qwen3's Jinja template (without tools): reasoning from earlier turns
/// is dropped, and reasoning after the last user message is kept.
fn render_qwen3(messages: &[Message], options: ChatOptions) -> String {
    let mut out = String::new();
    if let Some(first) = messages.first()
        && first.role == Role::System
    {
        push_turn(&mut out, Role::System, &first.content);
    }
    let last_query = messages
        .iter()
        .rposition(|m| m.role == Role::User)
        .unwrap_or(messages.len().saturating_sub(1));

    for (i, m) in messages.iter().enumerate() {
        match m.role {
            Role::System if i == 0 => {}
            Role::System | Role::User => push_turn(&mut out, m.role, &m.content),
            Role::Assistant => {
                let (reasoning, content) = split_reasoning(&m.content);
                out.push_str("<|im_start|>assistant\n");
                let is_last = i + 1 == messages.len();
                if i > last_query && (is_last || !reasoning.is_empty()) {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.trim_matches('\n'));
                    out.push_str("\n</think>\n\n");
                    out.push_str(content.trim_start_matches('\n'));
                } else {
                    out.push_str(content);
                }
                out.push_str("<|im_end|>\n");
            }
        }
    }
    if options.add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        if !options.enable_thinking {
            out.push_str("<think>\n\n</think>\n\n");
        }
    }
    out
}

/// Splits an assistant message into `(reasoning, answer)` around `</think>`,
/// as Qwen3's template does.
fn split_reasoning(content: &str) -> (&str, &str) {
    match content.rsplit_once("</think>") {
        None => ("", content),
        Some(_) => {
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
    }
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
    fn qwen35_trims_content_and_toggles_thinking() {
        let messages = [Message::system("  Be brief. "), Message::user(" Hi\n")];
        let thinking = ChatTemplate::Qwen35.render(&messages, ChatOptions::default());
        assert_eq!(
            thinking,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        let plain = ChatTemplate::Qwen35.render(
            &messages,
            ChatOptions {
                enable_thinking: false,
                ..ChatOptions::default()
            },
        );
        assert!(plain.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen35_history_drops_old_reasoning_and_keeps_new() {
        let messages = [
            Message::user("2+2?"),
            Message::assistant("<think>\nadd\n</think>\n\n4"),
            Message::user("3+3?"),
            Message::assistant("6"),
        ];
        let out = ChatTemplate::Qwen35.render(
            &messages,
            ChatOptions {
                add_generation_prompt: false,
                ..ChatOptions::default()
            },
        );
        assert_eq!(
            out,
            "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\n3+3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n6<|im_end|>\n"
        );
    }

    #[test]
    fn qwen3_single_turn() {
        let messages = [Message::system("Be brief."), Message::user("Hi")];
        let thinking = ChatTemplate::Qwen3.render(&messages, ChatOptions::default());
        assert_eq!(
            thinking,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        let no_think = ChatTemplate::Qwen3.render(
            &messages,
            ChatOptions {
                enable_thinking: false,
                ..ChatOptions::default()
            },
        );
        assert!(no_think.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn qwen3_drops_reasoning_from_earlier_turns() {
        let messages = [
            Message::user("2+2?"),
            Message::assistant("<think>\nadd them\n</think>\n\n4"),
            Message::user("3+3?"),
        ];
        let out = ChatTemplate::Qwen3.render(&messages, ChatOptions::default());
        assert_eq!(
            out,
            "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\n3+3?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen3_keeps_reasoning_after_the_last_query() {
        let messages = [Message::user("q"), Message::assistant("<think>r</think>a")];
        let out = ChatTemplate::Qwen3.render(
            &messages,
            ChatOptions {
                add_generation_prompt: false,
                ..ChatOptions::default()
            },
        );
        assert_eq!(
            out,
            "<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\nr\n</think>\n\na<|im_end|>\n"
        );
    }
}
