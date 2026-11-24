//! The Qwen3 and Qwen3.5 templates, transcribed from their Jinja source.
//!
//! The tests compare against llama.cpp's renderings of the real templates
//! (`/apply-template` of `llama-server` b12675). llama.cpp passes tools in
//! a normalized form ([`Tool::to_json`]) and tool-call arguments as
//! objects, so both templates write arguments with `tojson`, which is
//! Python's `json.dumps` ([`Json::to_python_json`]).

use super::{ChatOptions, Message, Role, Tool, push_turn, split_reasoning};
use crate::error::{Error, Result};
use crate::json::Json;

const QWEN3_TOOLS_INTRO: &str = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>";

const QWEN3_TOOLS_OUTRO: &str = "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

const QWEN35_TOOLS_INTRO: &str =
    "# Tools\n\nYou have access to the following functions:\n\n<tools>";

const QWEN35_TOOLS_OUTRO: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// `\n{tool}` for every tool, as both templates list them.
fn push_tools(out: &mut String, tools: &[Tool]) {
    for tool in tools {
        out.push('\n');
        out.push_str(&tool.to_json().to_python_json());
    }
}

/// The index of the last user message that is a query, not tool output;
/// the last message if there is none. `is_tool_output` is the template's test.
fn last_query(messages: &[Message], is_tool_output: impl Fn(&str) -> bool) -> usize {
    messages
        .iter()
        .rposition(|m| m.role == Role::User && !is_tool_output(&m.content))
        .unwrap_or(messages.len().saturating_sub(1))
}

/// Qwen3: tools in the system prompt, JSON tool calls, `<tool_response>`
/// blocks in user turns; reasoning is kept only after the last query.
pub(super) fn render_qwen3(messages: &[Message], options: ChatOptions<'_>) -> String {
    let mut out = String::new();
    let system = messages.first().filter(|m| m.role == Role::System);
    if !options.tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        if let Some(system) = system {
            out.push_str(&system.content);
            out.push_str("\n\n");
        }
        out.push_str(QWEN3_TOOLS_INTRO);
        push_tools(&mut out, options.tools);
        out.push_str(QWEN3_TOOLS_OUTRO);
    } else if let Some(system) = system {
        push_turn(&mut out, Role::System, &system.content);
    }
    let last_query = last_query(messages, |c| {
        c.contains("<tool_response>") && c.contains("</tool_response>")
    });

    for (i, m) in messages.iter().enumerate() {
        let is_last = i + 1 == messages.len();
        match m.role {
            Role::System if i == 0 => {}
            Role::System | Role::User => push_turn(&mut out, m.role, &m.content),
            Role::Assistant => {
                let (reasoning, content) = split_reasoning(&m.content);
                out.push_str("<|im_start|>assistant\n");
                if i > last_query && (is_last || !reasoning.is_empty()) {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.trim_matches('\n'));
                    out.push_str("\n</think>\n\n");
                    out.push_str(content.trim_start_matches('\n'));
                } else {
                    out.push_str(content);
                }
                for (j, call) in m.tool_calls.iter().enumerate() {
                    if j > 0 || !content.is_empty() {
                        out.push('\n');
                    }
                    out.push_str("<tool_call>\n{\"name\": \"");
                    out.push_str(&call.name);
                    out.push_str("\", \"arguments\": ");
                    out.push_str(&call.arguments.to_python_json());
                    out.push_str("}\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                if i == 0 || messages[i - 1].role != Role::Tool {
                    out.push_str("<|im_start|>user");
                }
                push_tool_response(&mut out, &m.content);
                if is_last || messages[i + 1].role != Role::Tool {
                    out.push_str("<|im_end|>\n");
                }
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

/// Qwen3.5: content is trimmed, tools come before the system prompt, tool
/// calls are XML with one block per argument, and the generation prompt
/// opens (thinking) or closes (no thinking) the think block.
pub(super) fn render_qwen35(messages: &[Message], options: ChatOptions<'_>) -> Result<String> {
    let mut out = String::new();
    let system = messages.first().filter(|m| m.role == Role::System);
    if !options.tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(QWEN35_TOOLS_INTRO);
        push_tools(&mut out, options.tools);
        out.push_str(QWEN35_TOOLS_OUTRO);
        if let Some(system) = system.map(|s| s.content.trim()).filter(|s| !s.is_empty()) {
            out.push_str("\n\n");
            out.push_str(system);
        }
        out.push_str("<|im_end|>\n");
    } else if let Some(system) = system {
        push_turn(&mut out, Role::System, system.content.trim());
    }
    let last_query = last_query(messages, |c| {
        let c = c.trim();
        c.starts_with("<tool_response>") && c.ends_with("</tool_response>")
    });

    for (i, m) in messages.iter().enumerate() {
        let is_last = i + 1 == messages.len();
        let content = m.content.trim();
        match m.role {
            Role::System if i == 0 => {}
            // The template only allows a leading system message; a later one
            // is kept as a turn rather than rejected.
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
                for (j, call) in m.tool_calls.iter().enumerate() {
                    out.push_str(match (j, answer.trim().is_empty()) {
                        (0, true) => "",
                        (0, false) => "\n\n",
                        _ => "\n",
                    });
                    out.push_str("<tool_call>\n<function=");
                    out.push_str(&call.name);
                    out.push_str(">\n");
                    let Some(arguments) = call.arguments.as_object() else {
                        return Err(Error::InvalidChat(format!(
                            "the arguments of tool call `{}` are not a JSON object",
                            call.name
                        )));
                    };
                    for (name, value) in arguments {
                        out.push_str("<parameter=");
                        out.push_str(name);
                        out.push_str(">\n");
                        out.push_str(&python_str(value));
                        out.push_str("\n</parameter>\n");
                    }
                    out.push_str("</function>\n</tool_call>");
                }
                out.push_str("<|im_end|>\n");
            }
            Role::Tool => {
                if i > 0 && messages[i - 1].role != Role::Tool {
                    out.push_str("<|im_start|>user");
                }
                push_tool_response(&mut out, content);
                if is_last || messages[i + 1].role != Role::Tool {
                    out.push_str("<|im_end|>\n");
                }
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
    Ok(out)
}

fn push_tool_response(out: &mut String, content: &str) {
    out.push_str("\n<tool_response>\n");
    out.push_str(content);
    out.push_str("\n</tool_response>");
}

/// An argument value as Qwen3.5's template writes it: objects and arrays
/// with `tojson`, everything else with Python's `str` (`True`, `None`, and
/// strings without quotes).
fn python_str(value: &Json) -> String {
    match value {
        Json::String(s) => s.clone(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Null => "None".into(),
        // Python's `str` of a number is its `repr`, as `json.dumps` writes it.
        Json::Number(_) | Json::Array(_) | Json::Object(_) => value.to_python_json(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ChatTemplate, ToolCall};
    use super::*;

    fn json(text: &str) -> Json {
        crate::json::parse(text).unwrap()
    }

    fn render(template: ChatTemplate, messages: &[Message], options: ChatOptions<'_>) -> String {
        template.render(messages, options).unwrap()
    }

    /// The tools of llama.cpp's renderings below. `now` has no description,
    /// which llama.cpp normalizes to "".
    fn tools() -> Vec<Tool> {
        vec![
            Tool {
                name: "get_weather".into(),
                description: "Get the current weather (°C)".into(),
                parameters: json(
                    r#"{"type":"object","properties":{"location":{"type":"string","description":"City"},"days":{"type":"integer","default":1},"precision":{"type":"number","default":0.5}},"required":["location"]}"#,
                ),
            },
            Tool {
                name: "now".into(),
                description: String::new(),
                parameters: json("{}"),
            },
        ]
    }

    fn weather(arguments: &str) -> ToolCall {
        ToolCall::new("get_weather", json(arguments))
    }

    /// A system prompt, tools, and a question.
    fn first_turn() -> Vec<Message> {
        vec![
            Message::system("You are helpful."),
            Message::user("What's the weather in Paris?"),
        ]
    }

    /// Parallel calls, consecutive tool results, an answer, then a second
    /// query whose call has content before it and non-string arguments.
    fn multi_step() -> Vec<Message> {
        vec![
            Message::user("Weather in Paris and Rome?"),
            Message::assistant("").with_tool_calls(vec![
                weather(r#"{"location": "Paris"}"#),
                weather(r#"{"location":"Rome","days":2}"#),
            ]),
            Message::tool("18 C, sunny"),
            Message::tool("25 C"),
            Message::assistant("Paris is 18 C and sunny; Rome is 25 C."),
            Message::user("Thanks! And Berlin?"),
            Message::assistant("Let me check.").with_tool_calls(vec![weather(
                r#"{"location": "Berlin", "flags": [true, null], "opts": {"unit": "C", "x": 1.5}, "n": 3, "ok": false}"#,
            )]),
            Message::tool("12 C"),
        ]
    }

    /// A call with reasoning after the last query (as the server passes
    /// `reasoning_content`), answered by a tool.
    fn reasoned_call() -> Vec<Message> {
        vec![
            Message::system("Be brief."),
            Message::user("Weather in Oslo?"),
            Message::assistant("<think>\nI should call the tool.\n</think>\n\n")
                .with_tool_calls(vec![weather(r#"{"location": "Oslo"}"#)]),
            Message::tool("-3 C"),
        ]
    }

    fn with_tools(tools: &[Tool], enable_thinking: bool) -> ChatOptions<'_> {
        ChatOptions {
            add_generation_prompt: true,
            enable_thinking,
            tools,
        }
    }

    /// The cases above, with whether thinking is on and the prompt that
    /// llama.cpp renders for them.
    fn cases(expected: [&'static str; 4]) -> [(Vec<Message>, bool, &'static str); 4] {
        let [a, b, c, d] = expected;
        [
            (first_turn(), true, a),
            (multi_step(), true, b),
            (reasoned_call(), false, c),
            (vec![Message::user("Hi")], true, d),
        ]
    }

    #[test]
    fn qwen3_with_tools_matches_llama_cpp() {
        let tools = tools();
        let expected = [
            "<|im_start|>system\n\
                You are helpful.\n\
                \n\
                # Tools\n\
                \n\
                You may call one or more functions to assist with the user query.\n\
                \n\
                You are provided with function signatures within <tools></tools> XML tags:\n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
                <tool_call>\n\
                {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                What's the weather in Paris?<|im_end|>\n\
                <|im_start|>assistant\n",
            "<|im_start|>system\n\
                # Tools\n\
                \n\
                You may call one or more functions to assist with the user query.\n\
                \n\
                You are provided with function signatures within <tools></tools> XML tags:\n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
                <tool_call>\n\
                {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                Weather in Paris and Rome?<|im_end|>\n\
                <|im_start|>assistant\n\
                <tool_call>\n\
                {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n\
                </tool_call>\n\
                <tool_call>\n\
                {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Rome\", \"days\": 2}}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                <tool_response>\n\
                18 C, sunny\n\
                </tool_response>\n\
                <tool_response>\n\
                25 C\n\
                </tool_response><|im_end|>\n\
                <|im_start|>assistant\n\
                Paris is 18 C and sunny; Rome is 25 C.<|im_end|>\n\
                <|im_start|>user\n\
                Thanks! And Berlin?<|im_end|>\n\
                <|im_start|>assistant\n\
                Let me check.\n\
                <tool_call>\n\
                {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Berlin\", \"flags\": [true, null], \"opts\": {\"unit\": \"C\", \"x\": 1.5}, \"n\": 3, \"ok\": false}}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                <tool_response>\n\
                12 C\n\
                </tool_response><|im_end|>\n\
                <|im_start|>assistant\n",
            "<|im_start|>system\n\
                Be brief.\n\
                \n\
                # Tools\n\
                \n\
                You may call one or more functions to assist with the user query.\n\
                \n\
                You are provided with function signatures within <tools></tools> XML tags:\n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
                <tool_call>\n\
                {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                Weather in Oslo?<|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n\
                I should call the tool.\n\
                </think>\n\
                \n\
                <tool_call>\n\
                {\"name\": \"get_weather\", \"arguments\": {\"location\": \"Oslo\"}}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                <tool_response>\n\
                -3 C\n\
                </tool_response><|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n\
                \n\
                </think>\n\
                \n",
            "<|im_start|>system\n\
                # Tools\n\
                \n\
                You may call one or more functions to assist with the user query.\n\
                \n\
                You are provided with function signatures within <tools></tools> XML tags:\n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n\
                <tool_call>\n\
                {\"name\": <function-name>, \"arguments\": <args-json-object>}\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                Hi<|im_end|>\n\
                <|im_start|>assistant\n",
        ];
        for (i, (messages, thinking, want)) in cases(expected).into_iter().enumerate() {
            let got = render(ChatTemplate::Qwen3, &messages, with_tools(&tools, thinking));
            assert_eq!(got, want, "case {i}");
        }
    }

    #[test]
    fn qwen35_with_tools_matches_llama_cpp() {
        let tools = tools();
        let expected = [
            "<|im_start|>system\n\
                # Tools\n\
                \n\
                You have access to the following functions:\n\
                \n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                If you choose to call a function ONLY reply in the following format with NO suffix:\n\
                \n\
                <tool_call>\n\
                <function=example_function_name>\n\
                <parameter=example_parameter_1>\n\
                value_1\n\
                </parameter>\n\
                <parameter=example_parameter_2>\n\
                This is the value for the second parameter\n\
                that can span\n\
                multiple lines\n\
                </parameter>\n\
                </function>\n\
                </tool_call>\n\
                \n\
                <IMPORTANT>\n\
                Reminder:\n\
                - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
                - Required parameters MUST be specified\n\
                - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
                - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
                </IMPORTANT>\n\
                \n\
                You are helpful.<|im_end|>\n\
                <|im_start|>user\n\
                What's the weather in Paris?<|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n",
            "<|im_start|>system\n\
                # Tools\n\
                \n\
                You have access to the following functions:\n\
                \n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                If you choose to call a function ONLY reply in the following format with NO suffix:\n\
                \n\
                <tool_call>\n\
                <function=example_function_name>\n\
                <parameter=example_parameter_1>\n\
                value_1\n\
                </parameter>\n\
                <parameter=example_parameter_2>\n\
                This is the value for the second parameter\n\
                that can span\n\
                multiple lines\n\
                </parameter>\n\
                </function>\n\
                </tool_call>\n\
                \n\
                <IMPORTANT>\n\
                Reminder:\n\
                - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
                - Required parameters MUST be specified\n\
                - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
                - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
                </IMPORTANT><|im_end|>\n\
                <|im_start|>user\n\
                Weather in Paris and Rome?<|im_end|>\n\
                <|im_start|>assistant\n\
                <tool_call>\n\
                <function=get_weather>\n\
                <parameter=location>\n\
                Paris\n\
                </parameter>\n\
                </function>\n\
                </tool_call>\n\
                <tool_call>\n\
                <function=get_weather>\n\
                <parameter=location>\n\
                Rome\n\
                </parameter>\n\
                <parameter=days>\n\
                2\n\
                </parameter>\n\
                </function>\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                <tool_response>\n\
                18 C, sunny\n\
                </tool_response>\n\
                <tool_response>\n\
                25 C\n\
                </tool_response><|im_end|>\n\
                <|im_start|>assistant\n\
                Paris is 18 C and sunny; Rome is 25 C.<|im_end|>\n\
                <|im_start|>user\n\
                Thanks! And Berlin?<|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n\
                \n\
                </think>\n\
                \n\
                Let me check.\n\
                \n\
                <tool_call>\n\
                <function=get_weather>\n\
                <parameter=location>\n\
                Berlin\n\
                </parameter>\n\
                <parameter=flags>\n\
                [true, null]\n\
                </parameter>\n\
                <parameter=opts>\n\
                {\"unit\": \"C\", \"x\": 1.5}\n\
                </parameter>\n\
                <parameter=n>\n\
                3\n\
                </parameter>\n\
                <parameter=ok>\n\
                False\n\
                </parameter>\n\
                </function>\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                <tool_response>\n\
                12 C\n\
                </tool_response><|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n",
            "<|im_start|>system\n\
                # Tools\n\
                \n\
                You have access to the following functions:\n\
                \n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                If you choose to call a function ONLY reply in the following format with NO suffix:\n\
                \n\
                <tool_call>\n\
                <function=example_function_name>\n\
                <parameter=example_parameter_1>\n\
                value_1\n\
                </parameter>\n\
                <parameter=example_parameter_2>\n\
                This is the value for the second parameter\n\
                that can span\n\
                multiple lines\n\
                </parameter>\n\
                </function>\n\
                </tool_call>\n\
                \n\
                <IMPORTANT>\n\
                Reminder:\n\
                - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
                - Required parameters MUST be specified\n\
                - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
                - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
                </IMPORTANT>\n\
                \n\
                Be brief.<|im_end|>\n\
                <|im_start|>user\n\
                Weather in Oslo?<|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n\
                I should call the tool.\n\
                </think>\n\
                \n\
                <tool_call>\n\
                <function=get_weather>\n\
                <parameter=location>\n\
                Oslo\n\
                </parameter>\n\
                </function>\n\
                </tool_call><|im_end|>\n\
                <|im_start|>user\n\
                <tool_response>\n\
                -3 C\n\
                </tool_response><|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n\
                \n\
                </think>\n\
                \n",
            "<|im_start|>system\n\
                # Tools\n\
                \n\
                You have access to the following functions:\n\
                \n\
                <tools>\n\
                {\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the current weather (°C)\", \"parameters\": {\"type\": \"object\", \"properties\": {\"location\": {\"type\": \"string\", \"description\": \"City\"}, \"days\": {\"type\": \"integer\", \"default\": 1}, \"precision\": {\"type\": \"number\", \"default\": 0.5}}, \"required\": [\"location\"]}}}\n\
                {\"type\": \"function\", \"function\": {\"name\": \"now\", \"description\": \"\", \"parameters\": {}}}\n\
                </tools>\n\
                \n\
                If you choose to call a function ONLY reply in the following format with NO suffix:\n\
                \n\
                <tool_call>\n\
                <function=example_function_name>\n\
                <parameter=example_parameter_1>\n\
                value_1\n\
                </parameter>\n\
                <parameter=example_parameter_2>\n\
                This is the value for the second parameter\n\
                that can span\n\
                multiple lines\n\
                </parameter>\n\
                </function>\n\
                </tool_call>\n\
                \n\
                <IMPORTANT>\n\
                Reminder:\n\
                - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
                - Required parameters MUST be specified\n\
                - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
                - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
                </IMPORTANT><|im_end|>\n\
                <|im_start|>user\n\
                Hi<|im_end|>\n\
                <|im_start|>assistant\n\
                <think>\n",
        ];
        for (i, (messages, thinking, want)) in cases(expected).into_iter().enumerate() {
            let got = render(
                ChatTemplate::Qwen35,
                &messages,
                with_tools(&tools, thinking),
            );
            assert_eq!(got, want, "case {i}");
        }
    }

    #[test]
    fn qwen35_rejects_arguments_that_are_not_an_object() {
        let messages = [
            Message::user("x"),
            Message::assistant("").with_tool_calls(vec![ToolCall::new("f", Json::from("text"))]),
        ];
        let err = ChatTemplate::Qwen35.render(&messages, ChatOptions::default());
        assert!(
            matches!(err, Err(Error::InvalidChat(ref m)) if m.contains("`f`")),
            "{err:?}"
        );
    }

    #[test]
    fn qwen35_trims_content_and_toggles_thinking() {
        let messages = [Message::system("  Be brief. "), Message::user(" Hi\n")];
        let thinking = render(ChatTemplate::Qwen35, &messages, ChatOptions::default());
        assert_eq!(
            thinking,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
        );
        let plain = render(
            ChatTemplate::Qwen35,
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
        let out = render(
            ChatTemplate::Qwen35,
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
        let thinking = render(ChatTemplate::Qwen3, &messages, ChatOptions::default());
        assert_eq!(
            thinking,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
        let no_think = render(
            ChatTemplate::Qwen3,
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
        let out = render(ChatTemplate::Qwen3, &messages, ChatOptions::default());
        assert_eq!(
            out,
            "<|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>\n\
             <|im_start|>user\n3+3?<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn qwen3_keeps_reasoning_after_the_last_query() {
        let messages = [Message::user("q"), Message::assistant("<think>r</think>a")];
        let out = render(
            ChatTemplate::Qwen3,
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
