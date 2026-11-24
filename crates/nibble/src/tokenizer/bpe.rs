//! Byte-level BPE (`tokenizer.ggml.model = "gpt2"`), as in GPT-2, Llama 3 and Qwen.
//!
//! Each pre-tokenized piece is mapped byte by byte to GPT-2's printable
//! characters, then adjacent symbols are merged by merge rank (lowest first,
//! leftmost on ties) until no merge applies.

use std::collections::HashMap;

use nibble_gguf::Gguf;

use super::pretokenize::PreTokenizer;
use crate::error::{Error, Result};

pub(super) struct Bpe {
    /// `(left, right) -> (rank, merged token)`.
    merges: HashMap<(u32, u32), (u32, u32)>,
    /// Token for each single byte.
    byte_tokens: [u32; 256],
    pre: PreTokenizer,
}

impl Bpe {
    /// Reads the pre-tokenizer and merges for a vocabulary of `texts`.
    pub(super) fn load(g: &Gguf<'_>, texts: &[String]) -> Result<Self> {
        let err = |msg: String| Error::Tokenizer(msg);
        let pre_name = g
            .get("tokenizer.ggml.pre")
            .and_then(|v| v.as_str())
            .unwrap_or("(missing)");
        let pre = PreTokenizer::from_name(pre_name).ok_or_else(|| {
            err(format!(
                "pre-tokenizer `{pre_name}` is not supported (supported: {})",
                PreTokenizer::NAMES.join(", ")
            ))
        })?;

        let index: HashMap<&str, u32> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| (t.as_str(), i as u32))
            .collect();

        let mut byte_tokens = [0u32; 256];
        for (b, slot) in byte_tokens.iter_mut().enumerate() {
            let c = BYTE_TO_CHAR[b];
            *slot = *index
                .get(c.encode_utf8(&mut [0; 4]) as &str)
                .ok_or_else(|| err(format!("no token for byte 0x{b:02x}")))?;
        }

        let merge_list = g
            .get("tokenizer.ggml.merges")
            .and_then(|v| v.as_array())
            .ok_or_else(|| err("missing tokenizer.ggml.merges".into()))?;
        let mut merges = HashMap::with_capacity(merge_list.len());
        let mut joined = String::new();
        for (rank, m) in merge_list.iter().enumerate() {
            let Some(m) = m.as_str() else { continue };
            // Split at the first space after byte 0, as llama.cpp does (`find(' ', 1)`).
            let Some(split) = m.bytes().skip(1).position(|b| b == b' ').map(|p| p + 1) else {
                continue;
            };
            let (left, right) = (&m[..split], &m[split + 1..]);
            joined.clear();
            joined.push_str(left);
            joined.push_str(right);
            if let (Some(&l), Some(&r), Some(&merged)) = (
                index.get(left),
                index.get(right),
                index.get(joined.as_str()),
            ) {
                let rank = u32::try_from(rank).map_err(|_| err("too many merges".into()))?;
                merges.entry((l, r)).or_insert((rank, merged));
            }
        }
        Ok(Self {
            merges,
            byte_tokens,
            pre,
        })
    }

    /// Encodes plain text (no special tokens) into `out`.
    pub(super) fn encode(&self, text: &str, out: &mut Vec<u32>) {
        for piece in self.pre.split(text) {
            self.merge_piece(piece, out);
        }
    }

    /// BPE on one pre-tokenized piece.
    fn merge_piece(&self, piece: &str, out: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = piece
            .bytes()
            .map(|b| self.byte_tokens[usize::from(b)])
            .collect();
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    self.merges
                        .get(&(pair[0], pair[1]))
                        .map(|&(rank, merged)| (rank, i, merged))
                })
                .min_by_key(|&(rank, i, _)| (rank, i));
            let Some((_, i, merged)) = best else { break };
            symbols[i] = merged;
            symbols.remove(i + 1);
        }
        out.extend_from_slice(&symbols);
    }

    /// Appends the bytes a normal token's text stands for (reverses the byte
    /// mapping; characters outside the map are kept as UTF-8).
    pub(super) fn decode_text(text: &str, out: &mut Vec<u8>) {
        for c in text.chars() {
            match CHAR_TO_BYTE.get(c as usize).copied().flatten() {
                Some(b) => out.push(b),
                None => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
            }
        }
    }

    pub(super) fn describe(&self) -> String {
        format!(
            "byte-level BPE ({:?}, {} merges)",
            self.pre,
            self.merges.len()
        )
    }
}

/// GPT-2's map from bytes to printable characters: printable Latin-1 bytes
/// map to themselves, the other 68 to U+0100.. in byte order.
const BYTE_TO_CHAR: [char; 256] = {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    let mut b = 0usize;
    while b < 256 {
        let printable = matches!(b, 33..=126 | 161..=172 | 174..=255);
        let cp = if printable {
            b as u32
        } else {
            next += 1;
            next - 1
        };
        table[b] = match char::from_u32(cp) {
            Some(c) => c,
            None => '\0',
        };
        b += 1;
    }
    table
};

/// Inverse of [`BYTE_TO_CHAR`], indexed by code point (all mapped code points are below 324).
const CHAR_TO_BYTE: [Option<u8>; 324] = {
    let mut table = [None; 324];
    let mut b = 0usize;
    while b < 256 {
        table[BYTE_TO_CHAR[b] as usize] = Some(b as u8);
        b += 1;
    }
    table
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_round_trips() {
        assert_eq!(BYTE_TO_CHAR[b' ' as usize], 'Ġ');
        assert_eq!(BYTE_TO_CHAR[b'\n' as usize], 'Ċ');
        assert_eq!(BYTE_TO_CHAR[b'A' as usize], 'A');
        for b in 0..=255u8 {
            let mut out = Vec::new();
            Bpe::decode_text(BYTE_TO_CHAR[b as usize].encode_utf8(&mut [0; 4]), &mut out);
            assert_eq!(out, [b]);
        }
    }
}
