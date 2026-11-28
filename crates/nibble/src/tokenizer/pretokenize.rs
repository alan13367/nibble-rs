//! Pre-tokenizers: split text into pieces before BPE, exactly as the
//! tokenizer's regex would (leftmost-first alternation, greedy quantifiers).
//!
//! `Qwen2` implements the `qwen2` pattern from llama.cpp:
//!
//! ```text
//! (?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//! | [^\r\n\p{L}\p{N}]?\p{L}+
//! | \p{N}
//! |  ?[^\s\p{L}\p{N}]+[\r\n]*
//! | \s*[\r\n]+
//! | \s+(?!\S)
//! | \s+
//! ```
//!
//! `Qwen35` is the same with `\p{M}` (combining marks) joined to letters:
//! `[\p{L}\p{M}]+` in the second branch and `[^\s\p{L}\p{M}\p{N}]+` in the fourth.
//!
//! `Llama3` (`llama3`, and LFM2's `lfm2` alias) differs only in its third
//! branch: `\p{N}{1,3}`, numbers in groups of up to three digits, as
//! llama.cpp's `unicode_regex_split_custom_llama3` splits them.

use super::unicode::{is_letter, is_mark, is_number, is_space};

/// A pre-tokenizer, selected by `tokenizer.ggml.pre`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PreTokenizer {
    /// `qwen2` (Qwen2, Qwen2.5, Qwen3).
    Qwen2,
    /// `qwen35` (Qwen3.5 and later).
    Qwen35,
    /// `llama3`, and LFM2's `lfm2` alias of it (llama.cpp maps both to its
    /// llama3 pattern).
    Llama3,
}

impl PreTokenizer {
    /// The `tokenizer.ggml.pre` values this module implements.
    pub(crate) const NAMES: &[&str] = &["qwen2", "qwen35", "llama3", "lfm2"];

    /// Maps a `tokenizer.ggml.pre` value to a pre-tokenizer.
    pub(crate) fn from_name(name: &str) -> Option<Self> {
        match name {
            "qwen2" => Some(Self::Qwen2),
            "qwen35" => Some(Self::Qwen35),
            "llama3" | "lfm2" => Some(Self::Llama3),
            _ => None,
        }
    }

    /// Splits `text` into consecutive pieces that cover it exactly.
    pub(crate) fn split(self, text: &str) -> Vec<&str> {
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let byte_at = |i: usize| chars.get(i).map_or(text.len(), |&(b, _)| b);
        let mut pieces = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            let end = self.match_at(&chars, i);
            debug_assert!(end > i);
            pieces.push(&text[byte_at(i)..byte_at(end)]);
            i = end;
        }
        pieces
    }

    /// Letters for the second branch: `\p{L}`, plus `\p{M}` for Qwen35.
    fn is_word(self, c: char) -> bool {
        is_letter(c) || (self == Self::Qwen35 && is_mark(c))
    }

    /// The fourth branch's class: not whitespace, letter or number (or mark, for Qwen35).
    fn is_symbol(self, c: char) -> bool {
        !is_space(c) && !is_letter(c) && !is_number(c) && !(self == Self::Qwen35 && is_mark(c))
    }

    /// End (char index, exclusive) of the match starting at `i`.
    fn match_at(self, s: &[(usize, char)], i: usize) -> usize {
        let n = s.len();
        let at = |k: usize| s.get(k).map(|&(_, c)| c);
        let c = s[i].1;

        // 1. Contractions: 's 't 're 've 'm 'll 'd (ASCII case-insensitive).
        if c == '\'' {
            let c1 = at(i + 1).map(|c| c.to_ascii_lowercase());
            let c2 = at(i + 2).map(|c| c.to_ascii_lowercase());
            match (c1, c2) {
                (Some('s' | 't' | 'm' | 'd'), _) => return i + 2,
                (Some('r' | 'v'), Some('e')) | (Some('l'), Some('l')) => return i + 3,
                _ => {}
            }
        }

        // 2. [^\r\n\p{L}\p{N}]?\p{L}+
        let word_run = |mut j: usize| {
            while j < n && self.is_word(s[j].1) {
                j += 1;
            }
            j
        };
        let prefix_ok = c != '\r' && c != '\n' && !is_letter(c) && !is_number(c);
        if prefix_ok && at(i + 1).is_some_and(|c1| self.is_word(c1)) {
            return word_run(i + 2);
        }
        if self.is_word(c) {
            return word_run(i + 1);
        }

        // 3. \p{N} for Qwen, \p{N}{1,3} for Llama3: numbers in groups
        // of up to three digits.
        if is_number(c) {
            let max = if self == Self::Llama3 { 3 } else { 1 };
            let mut j = i + 1;
            while j < n && j - i < max && is_number(s[j].1) {
                j += 1;
            }
            return j;
        }

        // 4.  ?[^\s\p{L}\p{N}]+[\r\n]*
        let symbols_from = if c == ' ' && at(i + 1).is_some_and(|c1| self.is_symbol(c1)) {
            Some(i + 1)
        } else if self.is_symbol(c) {
            Some(i)
        } else {
            None
        };
        if let Some(start) = symbols_from {
            let mut j = start + 1;
            while j < n && self.is_symbol(s[j].1) {
                j += 1;
            }
            while j < n && matches!(s[j].1, '\r' | '\n') {
                j += 1;
            }
            return j;
        }

        // Only whitespace remains. Find the run [i, end).
        let mut end = i;
        while end < n && is_space(s[end].1) {
            end += 1;
        }
        // 5. \s*[\r\n]+ : greedy \s* backtracks to the last newline in the run.
        if let Some(last_newline) = (i..end).rev().find(|&k| matches!(s[k].1, '\r' | '\n')) {
            return last_newline + 1;
        }
        // 6. \s+(?!\S) : the whole run at the end of text, else all but its last char.
        if end == n {
            return end;
        }
        if end - i >= 2 {
            return end - 1;
        }
        // 7. \s+
        end.max(i + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::PreTokenizer::{Qwen2, Qwen35};

    #[track_caller]
    fn check(pre: super::PreTokenizer, text: &str, expected: &[&str]) {
        assert_eq!(pre.split(text), expected, "{text:?}");
    }

    #[test]
    fn words_spaces_and_punctuation() {
        check(Qwen2, "Hello world", &["Hello", " world"]);
        check(Qwen2, "Hello, world!", &["Hello", ",", " world", "!"]);
        check(Qwen2, "a  b", &["a", " ", " b"]);
        check(Qwen2, "a   ", &["a", "   "]);
        check(Qwen2, "\tindented", &["\tindented"]);
        check(Qwen2, "foo!!\nbar", &["foo", "!!\n", "bar"]);
        check(Qwen2, " !!", &[" !!"]);
        check(Qwen2, "x = y;", &["x", " =", " y", ";"]);
        check(Qwen2, "", &[]);
    }

    #[test]
    fn contractions() {
        check(Qwen2, "I'm here", &["I", "'m", " here"]);
        check(
            Qwen2,
            "they'RE we'll you'd",
            &["they", "'RE", " we", "'ll", " you", "'d"],
        );
        check(Qwen2, "'hello", &["'hello"]);
        check(Qwen2, "rock'n'roll", &["rock", "'n", "'roll"]);
    }

    #[test]
    fn numbers_are_single_digits() {
        check(Qwen2, "12345", &["1", "2", "3", "4", "5"]);
        check(Qwen2, "v2.0", &["v", "2", ".", "0"]);
        check(Qwen2, "a 1", &["a", " ", "1"]);
    }

    #[test]
    fn newlines() {
        check(Qwen2, "x\n\ny", &["x", "\n\n", "y"]);
        check(Qwen2, "x\n\n  y", &["x", "\n\n", " ", " y"]);
        check(Qwen2, "x \n y", &["x", " \n", " y"]);
        check(Qwen2, "\r\n", &["\r\n"]);
    }

    #[test]
    fn unicode() {
        check(Qwen2, "héllo wörld", &["héllo", " wörld"]);
        check(Qwen2, "日本語です", &["日本語です"]);
        check(Qwen2, "emoji 😀!", &["emoji", " 😀!"]);
        // A combining mark ends a Qwen2 word but joins a Qwen35 one.
        check(Qwen2, "e\u{301}x", &["e", "\u{301}x"]);
        check(Qwen35, "e\u{301}x", &["e\u{301}x"]);
    }

    #[test]
    fn llama3_groups_digits() {
        use super::PreTokenizer::Llama3;
        check(Llama3, "Hello world", &["Hello", " world"]);
        check(Llama3, "12345", &["123", "45"]);
        check(Llama3, "1234567", &["123", "456", "7"]);
        check(Llama3, "a 1,234.5", &["a", " ", "1", ",", "234", ".", "5"]);
        check(Llama3, "I'm 2x", &["I", "'m", " ", "2", "x"]);
        check(Llama3, "x\n\ny", &["x", "\n\n", "y"]);
        check(
            Llama3,
            "they'RE we'll you'd",
            &["they", "'RE", " we", "'ll", " you", "'d"],
        );
        // Qwen keeps single digits; Llama3 differs only there.
        check(Qwen2, "12345", &["1", "2", "3", "4", "5"]);
        assert_eq!(Llama3.split("x 1 y").concat(), "x 1 y");
    }

    #[test]
    fn pieces_cover_the_text() {
        let text = "  The quick\tbrown 🦊 jumps 3x over\r\n\r\nthe lazy dog's 💤!!  ";
        for pre in [Qwen2, Qwen35] {
            assert_eq!(pre.split(text).concat(), text);
        }
    }
}
