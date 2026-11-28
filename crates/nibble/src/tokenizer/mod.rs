//! Tokenizers loaded from GGUF metadata, matching llama.cpp.
//!
//! [`Tokenizer`] owns what every tokenizer family shares: the vocabulary,
//! token kinds, special-token matching, detokenization and end-of-generation
//! tokens. The family-specific encoding (`tokenizer.ggml.model`) is an
//! encoder; today that is byte-level BPE (`gpt2`).

mod bpe;
mod pretokenize;
mod unicode;

use nibble_gguf::Gguf;

use crate::error::{Error, Result};
use bpe::Bpe;

/// A token's role, from `tokenizer.ggml.token_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenKind {
    /// An ordinary vocabulary entry.
    Normal,
    /// The unknown token.
    Unknown,
    /// A control token such as `<|im_start|>`, hidden when decoding by default.
    Control,
    /// A user-defined token such as `<think>`, matched literally in input text.
    UserDefined,
    /// An unused slot.
    Unused,
    /// A raw byte token.
    Byte,
}

impl TokenKind {
    const fn from_raw(raw: i64) -> Self {
        match raw {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }

    /// Whether the token is matched as a unit in input text before encoding.
    const fn is_special(self) -> bool {
        matches!(self, Self::Control | Self::UserDefined | Self::Unknown)
    }
}

/// Supported `tokenizer.ggml.model` values.
const MODELS: &[&str] = &["gpt2"];

/// Text of tokens that end a generation turn, besides the declared EOS/EOT.
const END_OF_GENERATION: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end|>",
    "<end_of_turn>",
    "<|end_of_text|>",
    "<|return|>",
];

/// The family-specific part of a tokenizer.
enum Encoder {
    /// Byte-level BPE (`gpt2`).
    Bpe(Bpe),
}

/// A tokenizer: text to token ids and back.
pub struct Tokenizer {
    /// Token text as stored in the vocabulary.
    texts: Vec<String>,
    kinds: Vec<TokenKind>,
    /// Decoded bytes of every token, concatenated; token `i` is
    /// `piece_bytes[piece_offsets[i]..piece_offsets[i + 1]]`.
    piece_bytes: Vec<u8>,
    piece_offsets: Vec<u32>,
    /// Special tokens, longest text first.
    specials: Vec<u32>,
    encoder: Encoder,
    bos: Option<u32>,
    eos: Option<u32>,
    add_bos: bool,
    end_of_generation: Vec<u32>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let encoder = match &self.encoder {
            Encoder::Bpe(bpe) => bpe.describe(),
        };
        f.debug_struct("Tokenizer")
            .field("vocab", &self.texts.len())
            .field("encoder", &encoder)
            .field("bos", &self.bos)
            .field("eos", &self.eos)
            .finish_non_exhaustive()
    }
}

impl Tokenizer {
    /// Loads the tokenizer described by a GGUF file's `tokenizer.ggml.*` metadata.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Tokenizer`] for missing or unsupported tokenizer data.
    pub fn from_gguf(g: &Gguf<'_>) -> Result<Self> {
        let err = |msg: String| Error::Tokenizer(msg);
        let tokens = g
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| err("missing tokenizer.ggml.tokens".into()))?;
        let mut texts = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            let s = t
                .as_str()
                .ok_or_else(|| err(format!("token {i} is not a UTF-8 string")))?;
            texts.push(s.to_owned());
        }
        let n_vocab = texts.len();
        let id = |i: usize| u32::try_from(i).map_err(|_| err("vocabulary is too large".into()));
        id(n_vocab)?;

        let mut kinds = vec![TokenKind::Normal; n_vocab];
        if let Some(types) = g
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
        {
            for (kind, t) in kinds.iter_mut().zip(types.iter()) {
                *kind = TokenKind::from_raw(t.as_i64().unwrap_or(1));
            }
        }

        let model = g
            .get("tokenizer.ggml.model")
            .and_then(|v| v.as_str())
            .unwrap_or("(missing)");
        let encoder = match model {
            "gpt2" => Encoder::Bpe(Bpe::load(g, &texts)?),
            other => {
                return Err(err(format!(
                    "tokenizer model `{other}` is not supported (supported: {})",
                    MODELS.join(", ")
                )));
            }
        };

        let mut piece_bytes = Vec::new();
        let mut piece_offsets = Vec::with_capacity(n_vocab + 1);
        for (text, kind) in texts.iter().zip(&kinds) {
            piece_offsets.push(id(piece_bytes.len())?);
            match (kind, &encoder) {
                (TokenKind::Normal | TokenKind::Byte, Encoder::Bpe(_)) => {
                    Bpe::decode_text(text, &mut piece_bytes)
                }
                // Special tokens decode to their literal text.
                _ => piece_bytes.extend_from_slice(text.as_bytes()),
            }
        }
        piece_offsets.push(id(piece_bytes.len())?);

        let mut specials: Vec<u32> = (0..n_vocab)
            .filter(|&i| kinds[i].is_special() && !texts[i].is_empty())
            .map(|i| i as u32)
            .collect();
        specials.sort_by_key(|&i| std::cmp::Reverse(texts[i as usize].len()));

        let token_id = |key: &str| -> Option<u32> {
            g.get(key)
                .and_then(|v| v.as_u32())
                .filter(|&t| (t as usize) < n_vocab)
        };
        let bos = token_id("tokenizer.ggml.bos_token_id");
        let eos = token_id("tokenizer.ggml.eos_token_id");
        let eot = token_id("tokenizer.ggml.eot_token_id");
        let add_bos = g
            .get("tokenizer.ggml.add_bos_token")
            .and_then(|v| v.as_bool())
            .unwrap_or(match &encoder {
                Encoder::Bpe(b) => b.adds_bos(),
            });

        let mut end_of_generation: Vec<u32> = [eos, eot].into_iter().flatten().collect();
        for (i, text) in texts.iter().enumerate() {
            if kinds[i].is_special() && END_OF_GENERATION.contains(&text.as_str()) {
                end_of_generation.push(i as u32);
            }
        }
        end_of_generation.sort_unstable();
        end_of_generation.dedup();

        Ok(Self {
            texts,
            kinds,
            piece_bytes,
            piece_offsets,
            specials,
            encoder,
            bos,
            eos,
            add_bos,
            end_of_generation,
        })
    }

    /// Number of tokens in the vocabulary.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.texts.len()
    }

    /// Encodes `text`.
    ///
    /// User-defined tokens (such as `<think>`) are always matched literally;
    /// control tokens (such as `<|im_start|>`) only when `parse_special` is set.
    /// No BOS token is added; see [`Tokenizer::add_bos`].
    #[must_use]
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        for fragment in self.partition_special(text, parse_special) {
            match fragment {
                Fragment::Token(id) => out.push(id),
                Fragment::Text(span) => match &self.encoder {
                    Encoder::Bpe(bpe) => bpe.encode(span, &mut out),
                },
            }
        }
        out
    }

    /// Splits `text` around special tokens, longest token first (llama.cpp's order).
    fn partition_special<'t>(&self, text: &'t str, parse_special: bool) -> Vec<Fragment<'t>> {
        let mut fragments = vec![Fragment::Text(text)];
        for &id in &self.specials {
            let kind = self.kinds[id as usize];
            if !parse_special && kind != TokenKind::UserDefined {
                continue;
            }
            let pattern = self.texts[id as usize].as_str();
            if !fragments
                .iter()
                .any(|f| matches!(f, Fragment::Text(s) if s.contains(pattern)))
            {
                continue;
            }
            let mut next = Vec::with_capacity(fragments.len() + 2);
            for fragment in fragments {
                let Fragment::Text(mut rest) = fragment else {
                    next.push(fragment);
                    continue;
                };
                while let Some(at) = rest.find(pattern) {
                    if at > 0 {
                        next.push(Fragment::Text(&rest[..at]));
                    }
                    next.push(Fragment::Token(id));
                    rest = &rest[at + pattern.len()..];
                }
                if !rest.is_empty() {
                    next.push(Fragment::Text(rest));
                }
            }
            fragments = next;
        }
        fragments
    }

    /// The bytes a token decodes to. Control tokens decode to nothing unless
    /// `render_special` is set.
    #[must_use]
    pub fn token_bytes(&self, id: u32, render_special: bool) -> &[u8] {
        let i = id as usize;
        if i >= self.texts.len() || (!render_special && self.kinds[i] == TokenKind::Control) {
            return &[];
        }
        &self.piece_bytes[self.piece_offsets[i] as usize..self.piece_offsets[i + 1] as usize]
    }

    /// Decodes tokens to text, replacing invalid UTF-8 with U+FFFD.
    #[must_use]
    pub fn decode(&self, ids: &[u32], render_special: bool) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.token_bytes(id, render_special))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// The vocabulary text of a token, as stored in the file.
    #[must_use]
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.texts.get(id as usize).map(String::as_str)
    }

    /// The kind of a token.
    #[must_use]
    pub fn token_kind(&self, id: u32) -> Option<TokenKind> {
        self.kinds.get(id as usize).copied()
    }

    /// The id of the token whose vocabulary text is exactly `text`.
    #[must_use]
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.texts.iter().position(|t| t == text).map(|i| i as u32)
    }

    /// Special tokens (control, user-defined, unknown), longest first.
    #[must_use]
    pub fn special_tokens(&self) -> &[u32] {
        &self.specials
    }

    /// Whether `id` ends a generation (EOS, EOT, `<|im_end|>`, ...).
    #[must_use]
    pub fn is_end_of_generation(&self, id: u32) -> bool {
        self.end_of_generation.binary_search(&id).is_ok()
    }

    /// The beginning-of-sequence token, if the model declares one.
    #[must_use]
    pub const fn bos(&self) -> Option<u32> {
        self.bos
    }

    /// The end-of-sequence token, if the model declares one.
    #[must_use]
    pub const fn eos(&self) -> Option<u32> {
        self.eos
    }

    /// Whether prompts should start with the BOS token.
    #[must_use]
    pub const fn add_bos(&self) -> bool {
        self.add_bos
    }
}

enum Fragment<'t> {
    Text(&'t str),
    Token(u32),
}

/// Incremental detokenizer for streaming output: emits only complete UTF-8.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    /// An empty decoder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a token and returns the text that is now complete.
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32, render_special: bool) -> String {
        self.push_bytes(tokenizer.token_bytes(id, render_special))
    }

    /// Adds raw bytes and returns the text that is now complete.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    // `valid_up_to` marks the longest valid UTF-8 prefix.
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap_or_default());
                    match e.error_len() {
                        // An incomplete sequence at the end: wait for more bytes.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    /// Returns whatever is left, with incomplete sequences replaced by U+FFFD.
    pub fn finish(&mut self) -> String {
        let out = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_decoder_waits_for_complete_characters() {
        let mut d = StreamDecoder::new();
        let e_acute = "é".as_bytes();
        assert_eq!(d.push_bytes(&e_acute[..1]), "");
        assert_eq!(d.push_bytes(&e_acute[1..]), "é");
        assert_eq!(d.push_bytes(&[b'a', 0xFF, b'b']), "a\u{FFFD}b");
        assert_eq!(d.push_bytes(&e_acute[..1]), "");
        assert_eq!(d.finish(), "\u{FFFD}");
    }
}
