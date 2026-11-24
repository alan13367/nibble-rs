//! Helpers for the synthetic-model tests: a tiny tokenizer, deterministic
//! random weights, and f64 reference math.

// Each test crate uses a different subset of these helpers.
#![allow(dead_code)]

use nibble::gguf::{GgufWriter, OwnedValue, ValueType};

/// GPT-2's byte-to-character map (printable Latin-1 maps to itself).
pub(crate) fn byte_char(b: u8) -> char {
    let printable =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let b = u32::from(b);
    if printable(b) {
        return char::from_u32(b).unwrap();
    }
    let offset = (0..b).filter(|&x| !printable(x)).count() as u32;
    char::from_u32(256 + offset).unwrap()
}

/// 256 byte tokens, two merged tokens and two control tokens (260 in all).
pub(crate) fn vocab() -> Vec<String> {
    let mut vocab: Vec<String> = (0..=255u8).map(|b| byte_char(b).to_string()).collect();
    vocab.extend(["ab", "abc", "<|im_start|>", "<|im_end|>"].map(str::to_owned));
    vocab
}

pub(crate) fn token_id(vocab: &[String], text: &str) -> u32 {
    vocab.iter().position(|t| t == text).unwrap() as u32
}

/// Adds a byte-level BPE tokenizer over `vocab` (from [`vocab`]).
pub(crate) fn add_tokenizer(w: &mut GgufWriter, vocab: &[String], pre: &str) {
    let token_types: Vec<i32> = (0..vocab.len())
        .map(|i| if i >= 258 { 3 } else { 1 })
        .collect();
    w.add("tokenizer.ggml.model", "gpt2")
        .add("tokenizer.ggml.pre", pre)
        .add(
            "tokenizer.ggml.tokens",
            OwnedValue::array(ValueType::String, vocab.iter().map(String::as_str)),
        )
        .add(
            "tokenizer.ggml.token_type",
            OwnedValue::array(ValueType::I32, token_types),
        )
        .add(
            "tokenizer.ggml.merges",
            OwnedValue::array(ValueType::String, ["a b", "ab c"]),
        )
        .add("tokenizer.ggml.eos_token_id", token_id(vocab, "<|im_end|>"))
        .add("tokenizer.ggml.add_bos_token", false);
}

pub(crate) fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Deterministic values in [-0.5, 0.5).
pub(crate) struct Rng(pub(crate) u32);

impl Rng {
    pub(crate) fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / 16_777_216.0 - 0.5
    }

    pub(crate) fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }

    /// Weights for a `rows × cols` matrix, scaled to keep activations near unit size.
    pub(crate) fn matrix(&mut self, rows: usize, cols: usize) -> Vec<f32> {
        self.vec(rows * cols, 2.0 / (cols as f32).sqrt())
    }

    /// Norm weights near 1.
    pub(crate) fn norm(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 1.0 + 0.2 * self.next()).collect()
    }
}

/// `w · x` for a row-major `rows × x.len()` matrix.
pub(crate) fn matvec(w: &[f32], x: &[f64], rows: usize) -> Vec<f64> {
    let cols = x.len();
    (0..rows)
        .map(|r| (0..cols).map(|c| f64::from(w[r * cols + c]) * x[c]).sum())
        .collect()
}

pub(crate) fn rmsnorm(x: &[f64], w: &[f32], eps: f32) -> Vec<f64> {
    let scale =
        1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(eps)).sqrt();
    x.iter()
        .zip(w)
        .map(|(v, &w)| v * scale * f64::from(w))
        .collect()
}

pub(crate) fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub(crate) fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn add(x: &mut [f64], y: &[f64]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += b;
    }
}

/// Asserts `got` matches `expected` to within `rel` of the largest expected magnitude.
#[track_caller]
pub(crate) fn assert_close(got: &[f32], expected: &[f64], rel: f64) {
    assert_eq!(got.len(), expected.len());
    let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let max_err = got
        .iter()
        .zip(expected)
        .map(|(&a, b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max);
    assert!(
        max_err <= rel * scale,
        "max error {max_err} exceeds {rel} of scale {scale}"
    );
}
