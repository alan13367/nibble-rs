//! Streaming text filters: stop sequences and `<think>` reasoning blocks.
//!
//! Both work on text as it is generated, holding back only the bytes that
//! could still turn out to be part of a marker.

/// Ends generation at the first stop sequence, and never emits a partial one.
#[derive(Debug, Default)]
pub(crate) struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

impl StopMatcher {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            stops: stops.into_iter().filter(|s| !s.is_empty()).collect(),
            pending: String::new(),
        }
    }

    /// Adds generated text. Returns the text that is safe to emit, and whether
    /// a stop sequence was reached (the stop sequence itself is not emitted).
    pub(crate) fn push(&mut self, text: &str) -> (String, bool) {
        self.pending.push_str(text);
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
        {
            let out = self.pending[..at].to_owned();
            self.pending.clear();
            return (out, true);
        }
        let hold = self
            .stops
            .iter()
            .map(|s| suffix_prefix_len(&self.pending, s))
            .max()
            .unwrap_or(0);
        let cut = self.pending.len() - hold;
        let out = self.pending[..cut].to_owned();
        self.pending.drain(..cut);
        (out, false)
    }

    /// Returns the held-back text at the end of generation.
    pub(crate) fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

/// Length of the longest proper prefix of `marker` that `text` ends with.
/// Always a char boundary of both strings.
fn suffix_prefix_len(text: &str, marker: &str) -> usize {
    (1..marker.len())
        .rev()
        .filter(|&k| marker.is_char_boundary(k))
        .find(|&k| text.ends_with(&marker[..k]))
        .unwrap_or(0)
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Nothing emitted yet; the output may open a think block.
    Start,
    Reasoning,
    /// After `</think>`: leading whitespace is dropped.
    ContentStart,
    Content,
}

/// Splits model output into reasoning (`<think>...</think>`) and content.
#[derive(Debug)]
pub(crate) struct ReasoningSplitter {
    phase: Phase,
    pending: String,
}

/// Text split by [`ReasoningSplitter`].
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Split {
    pub(crate) reasoning: String,
    pub(crate) content: String,
}

impl ReasoningSplitter {
    /// `in_reasoning` is true when the prompt already opened the think block
    /// (Qwen3.5's thinking prompt ends with `<think>\n`).
    pub(crate) fn new(in_reasoning: bool) -> Self {
        Self {
            phase: if in_reasoning {
                Phase::Reasoning
            } else {
                Phase::Start
            },
            pending: String::new(),
        }
    }

    pub(crate) fn push(&mut self, text: &str) -> Split {
        self.pending.push_str(text);
        let mut out = Split::default();
        loop {
            match self.phase {
                Phase::Start => {
                    let trimmed = self.pending.trim_start();
                    if let Some(rest) = trimmed.strip_prefix(THINK_OPEN) {
                        self.pending = rest.trim_start_matches('\n').to_owned();
                        self.phase = Phase::Reasoning;
                    } else if THINK_OPEN.starts_with(trimmed) {
                        return out; // still possibly "<think>"
                    } else {
                        self.phase = Phase::Content;
                    }
                }
                Phase::Reasoning => {
                    if let Some(at) = self.pending.find(THINK_CLOSE) {
                        out.reasoning
                            .push_str(self.pending[..at].trim_end_matches('\n'));
                        self.pending.drain(..at + THINK_CLOSE.len());
                        self.phase = Phase::ContentStart;
                    } else {
                        let hold = suffix_prefix_len(&self.pending, THINK_CLOSE);
                        // Trailing newlines may sit just before `</think>`; hold them too.
                        let body = &self.pending[..self.pending.len() - hold];
                        let keep = body.len() - body.trim_end_matches('\n').len() + hold;
                        let cut = self.pending.len() - keep;
                        out.reasoning.push_str(&self.pending[..cut]);
                        self.pending.drain(..cut);
                        return out;
                    }
                }
                Phase::ContentStart => {
                    let trimmed = self.pending.trim_start();
                    if trimmed.is_empty() {
                        self.pending.clear();
                        return out;
                    }
                    self.pending = trimmed.to_owned();
                    self.phase = Phase::Content;
                }
                Phase::Content => {
                    out.content.push_str(&self.pending);
                    self.pending.clear();
                    return out;
                }
            }
        }
    }

    /// Flushes held-back text at the end of generation.
    pub(crate) fn finish(&mut self) -> Split {
        let pending = std::mem::take(&mut self.pending);
        match self.phase {
            Phase::Reasoning => Split {
                reasoning: pending,
                content: String::new(),
            },
            Phase::ContentStart => Split::default(),
            Phase::Start | Phase::Content => Split {
                reasoning: String::new(),
                content: pending,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `pieces` one at a time and concatenates everything emitted.
    fn run_stops(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let mut m = StopMatcher::new(stops.iter().map(|s| (*s).to_owned()).collect());
        let mut out = String::new();
        for p in pieces {
            let (text, stopped) = m.push(p);
            out.push_str(&text);
            if stopped {
                return (out, true);
            }
        }
        out.push_str(&m.finish());
        (out, false)
    }

    #[test]
    fn stops_at_the_first_stop_sequence_across_pieces() {
        assert_eq!(
            run_stops(&["END"], &["hello E", "N", "D world"]),
            ("hello ".into(), true)
        );
        assert_eq!(
            run_stops(&["\n\n", "###"], &["a\n", "b##", "#c"]),
            ("a\nb".into(), true)
        );
        assert_eq!(run_stops(&["xyz"], &["abc", "xy"]), ("abcxy".into(), false));
        assert_eq!(run_stops(&[], &["anything"]), ("anything".into(), false));
    }

    #[test]
    fn holds_back_only_possible_stop_prefixes() {
        let mut m = StopMatcher::new(vec!["STOP".into()]);
        assert_eq!(m.push("ab ST"), ("ab ".into(), false));
        assert_eq!(m.push("x"), ("STx".into(), false));
        let mut m = StopMatcher::new(vec!["é!".into()]);
        assert_eq!(m.push("café"), ("caf".into(), false));
    }

    fn run_split(in_reasoning: bool, pieces: &[&str]) -> Split {
        let mut s = ReasoningSplitter::new(in_reasoning);
        let mut out = Split::default();
        for p in pieces.iter().copied() {
            let part = s.push(p);
            out.reasoning.push_str(&part.reasoning);
            out.content.push_str(&part.content);
        }
        let part = s.finish();
        out.reasoning.push_str(&part.reasoning);
        out.content.push_str(&part.content);
        out
    }

    #[test]
    fn splits_think_blocks_streamed_in_pieces() {
        let s = run_split(
            false,
            &[
                "<th",
                "ink>\nLet me",
                " think.\n</th",
                "ink>\n\nThe answer",
                " is 4.",
            ],
        );
        assert_eq!(
            s,
            Split {
                reasoning: "Let me think.".into(),
                content: "The answer is 4.".into()
            }
        );
    }

    #[test]
    fn prompt_opened_think_block() {
        let s = run_split(true, &["reason\n", "\n</think>\n\nanswer"]);
        assert_eq!(
            s,
            Split {
                reasoning: "reason".into(),
                content: "answer".into()
            }
        );
    }

    #[test]
    fn output_without_reasoning_is_content() {
        assert_eq!(
            run_split(false, &["<", "b>hi"]),
            Split {
                reasoning: String::new(),
                content: "<b>hi".into()
            }
        );
        assert_eq!(
            run_split(false, &["Hello"]),
            Split {
                reasoning: String::new(),
                content: "Hello".into()
            }
        );
    }

    #[test]
    fn unterminated_reasoning_is_flushed_as_reasoning() {
        let s = run_split(false, &["<think>still thinking</th"]);
        assert_eq!(
            s,
            Split {
                reasoning: "still thinking</th".into(),
                content: String::new()
            }
        );
    }
}
