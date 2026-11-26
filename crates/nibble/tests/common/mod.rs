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

/// A Metal session on `model`, or `None` (with a note) where Metal is not
/// available.
pub(crate) fn metal_session(model: &nibble::Model, n_ctx: usize) -> Option<nibble::Session<'_>> {
    let options = nibble::SessionOptions {
        n_ctx,
        backend: nibble::BackendKind::Metal,
        ..nibble::SessionOptions::default()
    };
    session_or_skip(model, options)
}

/// A session with `options`, or `None` (with a note) for a Metal session
/// where Metal is not available.
pub(crate) fn session_or_skip(
    model: &nibble::Model,
    options: nibble::SessionOptions,
) -> Option<nibble::Session<'_>> {
    match model.session(options) {
        Ok(s) => Some(s),
        // Only a missing device or framework skips: on a Mac with a GPU,
        // a backend that fails to start (a shader that does not compile,
        // say) is a bug.
        Err(nibble::Error::Backend(e))
            if !cfg!(target_os = "macos") || e.contains("no Metal device") =>
        {
            eprintln!("skipping: {e}");
            None
        }
        Err(e) => panic!("{e}"),
    }
}

/// Checks that evaluating `tokens` in batches of any size gives, bit for
/// bit, the logits of evaluating them one token at a time, on `backend`:
/// whole, and in two calls split at `split` (each call cuts a batch where
/// it takes its checkpoint, 4 tokens before its end). On Metal, batches of
/// more than [`nibble::MAX_EVAL_ALL`] tokens multiply matrices with f16
/// tiles, as llama.cpp does, and are only checked to be close.
pub(crate) fn assert_batches_match_steps(
    model: &nibble::Model,
    backend: nibble::BackendKind,
    n_ctx: usize,
    tokens: &[u32],
    split: usize,
) {
    let options = |n_batch: usize| nibble::SessionOptions {
        n_ctx,
        backend,
        threads: 2,
        n_batch,
        draft_max: 0,
        ..Default::default()
    };
    let Some(mut stepped) = session_or_skip(model, options(1)) else {
        return;
    };
    let steps: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| stepped.eval(&[t]).unwrap().to_vec())
        .collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for n_batch in [2, 3, 5, 64] {
        let exact = backend != nibble::BackendKind::Metal || n_batch <= nibble::MAX_EVAL_ALL;
        let check = |got: &[f32], want: &[f32], what: &str| {
            if exact {
                assert_eq!(
                    bits(got),
                    bits(want),
                    "{backend} batches of {n_batch}{what}"
                );
            } else {
                assert_logits_close(got, want, &format!("{backend} batches of {n_batch}{what}"));
            }
        };
        let mut s = session_or_skip(model, options(n_batch)).unwrap();
        assert_eq!(s.n_batch(), n_batch.min(n_ctx));
        let whole = s.eval(tokens).unwrap();
        check(whole, &steps[tokens.len() - 1], "");
        s.clear().unwrap();
        let first = s.eval(&tokens[..split]).unwrap();
        check(first, &steps[split - 1], ", first call");
        let second = s.eval(&tokens[split..]).unwrap();
        check(second, &steps[tokens.len() - 1], ", second call");
    }

    // `eval_all`: each row is the logits after its token.
    let n_vocab = model.config().n_vocab;
    let mut s = session_or_skip(model, options(0)).unwrap();
    let mut at = 0;
    for chunk in tokens.chunks(7) {
        let all = s.eval_all(chunk).unwrap();
        assert_eq!(all.len(), chunk.len() * n_vocab);
        for (i, row) in all.chunks(n_vocab).enumerate() {
            let t = at + i;
            assert_eq!(bits(row), bits(&steps[t]), "{backend} eval_all, token {t}");
        }
        assert_eq!(
            bits(s.logits().unwrap()),
            bits(&steps[at + chunk.len() - 1])
        );
        at += chunk.len();
    }
    s.clear().unwrap();
    let too_many = vec![tokens[0]; nibble::MAX_EVAL_ALL + 1];
    assert!(matches!(
        s.eval_all(&too_many),
        Err(nibble::Error::TooManyTokens { n, max })
            if n == too_many.len() && max == nibble::MAX_EVAL_ALL
    ));
}

/// Checks that `got` is within 1% of the largest magnitude of `want` at
/// every logit.
pub(crate) fn assert_logits_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 0.01 * scale,
            "{what}: logit {i} is {g}, not {w}"
        );
    }
}

/// Checks, on `backend`, that an evaluation of 2 to 16 tokens (a draft being
/// verified) can be taken back without replaying anything, and that
/// evaluation then continues bit for bit as from the kept prefix: after
/// one rewind or several, and from a checkpoint taken while the tokens of
/// such an evaluation were pending. `tokens` needs at least 30.
pub(crate) fn assert_rewinds_into_short_batches(
    model: &nibble::Model,
    backend: nibble::BackendKind,
    n_ctx: usize,
    tokens: &[u32],
) {
    let options = |n_batch: usize| nibble::SessionOptions {
        n_ctx,
        backend,
        threads: 2,
        n_batch,
        draft_max: 0,
        ..Default::default()
    };
    let Some(mut s) = session_or_skip(model, options(0)) else {
        return;
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let n_vocab = model.config().n_vocab;
    // The logits after `seq`, evaluated one token at a time.
    let stepped = |seq: &[u32]| {
        let mut r = session_or_skip(model, options(1)).unwrap();
        for &t in seq {
            r.eval(&[t]).unwrap();
        }
        bits(r.logits().unwrap())
    };
    let (prompt, draft, x, y) = (&tokens[..11], &tokens[11..17], tokens[17], tokens[18]);

    for kept in [0, 1, 4, 6] {
        s.clear().unwrap();
        s.eval(prompt).unwrap();
        s.eval_all(draft).unwrap();
        assert_eq!(
            s.truncate(prompt.len() + kept).unwrap(),
            0,
            "{backend} kept {kept}"
        );
        let got = bits(s.eval(&[x]).unwrap());
        let seq: Vec<u32> = prompt
            .iter()
            .chain(&draft[..kept])
            .chain([&x])
            .copied()
            .collect();
        assert_eq!(got, stepped(&seq), "{backend}: {kept} of the draft kept");
    }

    // Two rewinds into one evaluation, then into a second one.
    s.clear().unwrap();
    s.eval(prompt).unwrap();
    s.eval_all(draft).unwrap();
    assert_eq!(s.truncate(prompt.len() + 4).unwrap(), 0);
    assert_eq!(s.truncate(prompt.len() + 2).unwrap(), 0);
    let got = bits(s.eval_all(&[x, y]).unwrap()[n_vocab..].to_vec().as_slice());
    let mut seq: Vec<u32> = prompt.iter().chain(&draft[..2]).copied().collect();
    seq.extend([x, y]);
    assert_eq!(got, stepped(&seq), "{backend}: after two rewinds");
    assert_eq!(s.truncate(seq.len() - 1).unwrap(), 0);
    let got = bits(s.eval(&[tokens[19]]).unwrap());
    seq.pop();
    seq.push(tokens[19]);
    assert_eq!(
        got,
        stepped(&seq),
        "{backend}: after a rewind into the second evaluation"
    );

    // A checkpoint taken while a short batch is pending: the evaluation
    // below cuts a batch 4 tokens before its end and checkpoints there,
    // after its first 4 tokens, which are pending.
    s.clear().unwrap();
    s.eval(prompt).unwrap();
    s.eval_all(draft).unwrap();
    s.truncate(prompt.len() + 3).unwrap();
    let more = &tokens[20..28];
    s.eval(more).unwrap();
    s.eval(&[tokens[28]]).unwrap();
    let mut seq: Vec<u32> = prompt
        .iter()
        .chain(&draft[..3])
        .chain(more)
        .copied()
        .collect();
    seq.push(tokens[28]);
    let checkpoint = seq.len() - 1 - 4;
    // One token (not pending) was evaluated since, so this restores the
    // checkpoint and replays the token after it.
    assert_eq!(
        s.truncate(checkpoint + 1).unwrap(),
        1,
        "{backend}: replay from the checkpoint"
    );
    let got = bits(s.eval(&[tokens[29]]).unwrap());
    seq.truncate(checkpoint + 1);
    seq.push(tokens[29]);
    assert_eq!(
        got,
        stepped(&seq),
        "{backend}: from a checkpoint with pending tokens"
    );
}
