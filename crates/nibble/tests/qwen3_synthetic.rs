//! End-to-end tests on a tiny random Qwen3 model built in memory, so they run
//! everywhere without model files.

use nibble::gguf::{GgmlType, GgufWriter, OwnedValue, Storage, ValueType};
use nibble::sampler::{Sampler, SamplerConfig};
use nibble::{Error, Model, SessionOptions, StopReason};

mod common;
use common::{Rng, byte_char};

const N_EMBD: usize = 32;
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2;
const HEAD_DIM: usize = 8;
const N_FF: usize = 48;
const N_LAYER: usize = 2;
const N_CTX: usize = 64;
const EPS: f32 = 1e-6;

struct Layer {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    w_gate: Vec<f32>,
    w_up: Vec<f32>,
    w_down: Vec<f32>,
}

/// The model's weights, kept to compute reference outputs.
struct Weights {
    vocab: Vec<String>,
    embd: Vec<f32>,
    output_norm: Vec<f32>,
    layers: Vec<Layer>,
}

impl Weights {
    fn new() -> Self {
        let mut rng = Rng(0x1234_5678);
        let mut vocab: Vec<String> = (0..=255u8).map(|b| byte_char(b).to_string()).collect();
        vocab.extend(["ab", "abc", "<|im_start|>", "<|im_end|>"].map(str::to_owned));
        let q_dim = N_HEAD * HEAD_DIM;
        let kv_dim = N_HEAD_KV * HEAD_DIM;
        let s = |cols: usize| 2.0 / (cols as f32).sqrt();
        let layers = (0..N_LAYER)
            .map(|_| Layer {
                attn_norm: rng.norm(N_EMBD),
                wq: rng.vec(q_dim * N_EMBD, s(N_EMBD)),
                wk: rng.vec(kv_dim * N_EMBD, s(N_EMBD)),
                wv: rng.vec(kv_dim * N_EMBD, s(N_EMBD)),
                wo: rng.vec(N_EMBD * q_dim, s(q_dim)),
                q_norm: rng.norm(HEAD_DIM),
                k_norm: rng.norm(HEAD_DIM),
                ffn_norm: rng.norm(N_EMBD),
                w_gate: rng.vec(N_FF * N_EMBD, s(N_EMBD)),
                w_up: rng.vec(N_FF * N_EMBD, s(N_EMBD)),
                w_down: rng.vec(N_EMBD * N_FF, s(N_FF)),
            })
            .collect();
        let embd = rng.vec(vocab.len() * N_EMBD, 2.0);
        Self {
            embd,
            output_norm: rng.norm(N_EMBD),
            layers,
            vocab,
        }
    }

    fn n_vocab(&self) -> usize {
        self.vocab.len()
    }

    fn id(&self, text: &str) -> u32 {
        self.vocab.iter().position(|t| t == text).unwrap() as u32
    }

    /// A GGUF file holding these weights; `edit` can change it before encoding.
    fn gguf(&self, edit: impl FnOnce(&mut GgufWriter)) -> Vec<u8> {
        let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (n_embd, q_dim, kv_dim) = (
            N_EMBD as u64,
            (N_HEAD * HEAD_DIM) as u64,
            (N_HEAD_KV * HEAD_DIM) as u64,
        );
        let mut w = GgufWriter::new();
        let token_types: Vec<i32> = (0..self.n_vocab())
            .map(|i| if i >= 258 { 3 } else { 1 })
            .collect();
        w.add("general.architecture", "qwen3")
            .add("general.name", "synthetic")
            .add("qwen3.block_count", N_LAYER as u32)
            .add("qwen3.context_length", N_CTX as u32)
            .add("qwen3.embedding_length", N_EMBD as u32)
            .add("qwen3.feed_forward_length", N_FF as u32)
            .add("qwen3.attention.head_count", N_HEAD as u32)
            .add("qwen3.attention.head_count_kv", N_HEAD_KV as u32)
            .add("qwen3.attention.key_length", HEAD_DIM as u32)
            .add("qwen3.attention.value_length", HEAD_DIM as u32)
            .add("qwen3.rope.freq_base", 10_000.0f32)
            .add("qwen3.attention.layer_norm_rms_epsilon", EPS)
            .add("tokenizer.ggml.model", "gpt2")
            .add("tokenizer.ggml.pre", "qwen2")
            .add(
                "tokenizer.ggml.tokens",
                OwnedValue::array(ValueType::String, self.vocab.iter().map(String::as_str)),
            )
            .add(
                "tokenizer.ggml.token_type",
                OwnedValue::array(ValueType::I32, token_types),
            )
            .add(
                "tokenizer.ggml.merges",
                OwnedValue::array(ValueType::String, ["a b", "ab c"]),
            )
            .add("tokenizer.ggml.eos_token_id", self.id("<|im_end|>"))
            .add("tokenizer.ggml.add_bos_token", false);
        w.add_tensor(
            "token_embd.weight",
            GgmlType::F32,
            &[n_embd, self.n_vocab() as u64],
            f32s(&self.embd),
        )
        .add_tensor(
            "output_norm.weight",
            GgmlType::F32,
            &[n_embd],
            f32s(&self.output_norm),
        );
        for (i, l) in self.layers.iter().enumerate() {
            let name = |n: &str| format!("blk.{i}.{n}.weight");
            w.add_tensor(
                name("attn_norm"),
                GgmlType::F32,
                &[n_embd],
                f32s(&l.attn_norm),
            )
            .add_tensor(name("attn_q"), GgmlType::F32, &[n_embd, q_dim], f32s(&l.wq))
            .add_tensor(
                name("attn_k"),
                GgmlType::F32,
                &[n_embd, kv_dim],
                f32s(&l.wk),
            )
            .add_tensor(
                name("attn_v"),
                GgmlType::F32,
                &[n_embd, kv_dim],
                f32s(&l.wv),
            )
            .add_tensor(
                name("attn_output"),
                GgmlType::F32,
                &[q_dim, n_embd],
                f32s(&l.wo),
            )
            .add_tensor(
                name("attn_q_norm"),
                GgmlType::F32,
                &[HEAD_DIM as u64],
                f32s(&l.q_norm),
            )
            .add_tensor(
                name("attn_k_norm"),
                GgmlType::F32,
                &[HEAD_DIM as u64],
                f32s(&l.k_norm),
            )
            .add_tensor(
                name("ffn_norm"),
                GgmlType::F32,
                &[n_embd],
                f32s(&l.ffn_norm),
            )
            .add_tensor(
                name("ffn_gate"),
                GgmlType::F32,
                &[n_embd, N_FF as u64],
                f32s(&l.w_gate),
            )
            .add_tensor(
                name("ffn_up"),
                GgmlType::F32,
                &[n_embd, N_FF as u64],
                f32s(&l.w_up),
            )
            .add_tensor(
                name("ffn_down"),
                GgmlType::F32,
                &[N_FF as u64, n_embd],
                f32s(&l.w_down),
            );
        }
        edit(&mut w);
        w.to_bytes()
    }

    fn model(&self) -> Model {
        Model::from_storage(Storage::from_bytes(&self.gguf(|_| {}))).unwrap()
    }

    /// Logits for `token` at position 0, computed naively in f64. At position 0
    /// attention is exactly V (one key, softmax weight 1), so Q, K and RoPE drop out.
    fn reference_first_logits(&self, token: u32) -> Vec<f64> {
        let matvec = |w: &[f32], x: &[f64], rows: usize| -> Vec<f64> {
            let cols = x.len();
            (0..rows)
                .map(|r| (0..cols).map(|c| f64::from(w[r * cols + c]) * x[c]).sum())
                .collect()
        };
        let rmsnorm = |x: &[f64], w: &[f32]| -> Vec<f64> {
            let scale = 1.0
                / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + f64::from(EPS)).sqrt();
            x.iter()
                .zip(w)
                .map(|(v, &w)| v * scale * f64::from(w))
                .collect()
        };
        let t = token as usize;
        let mut x: Vec<f64> = self.embd[t * N_EMBD..(t + 1) * N_EMBD]
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        for l in &self.layers {
            let h = rmsnorm(&x, &l.attn_norm);
            let v = matvec(&l.wv, &h, N_HEAD_KV * HEAD_DIM);
            let group = N_HEAD / N_HEAD_KV;
            let att: Vec<f64> = (0..N_HEAD * HEAD_DIM)
                .map(|i| v[(i / HEAD_DIM / group) * HEAD_DIM + i % HEAD_DIM])
                .collect();
            for (xi, o) in x.iter_mut().zip(matvec(&l.wo, &att, N_EMBD)) {
                *xi += o;
            }
            let h = rmsnorm(&x, &l.ffn_norm);
            let gate = matvec(&l.w_gate, &h, N_FF);
            let up = matvec(&l.w_up, &h, N_FF);
            let act: Vec<f64> = gate
                .iter()
                .zip(&up)
                .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
                .collect();
            for (xi, o) in x.iter_mut().zip(matvec(&l.w_down, &act, N_EMBD)) {
                *xi += o;
            }
        }
        matvec(&self.embd, &rmsnorm(&x, &self.output_norm), self.n_vocab())
    }
}

#[test]
fn loads_config_and_tokenizer() {
    let w = Weights::new();
    let model = w.model();
    let c = model.config();
    assert_eq!(
        (
            c.n_vocab,
            c.n_embd,
            c.n_layer,
            c.n_head,
            c.n_head_kv,
            c.head_dim
        ),
        (260, 32, 2, 4, 2, 8)
    );
    assert!(c.tied_embeddings);
    assert_eq!(c.name.as_deref(), Some("synthetic"));

    let tok = model.tokenizer();
    let space = w.id(&byte_char(b' ').to_string());
    assert_eq!(
        tok.encode("abc ab", false),
        [w.id("abc"), space, w.id("ab")]
    );
    assert_eq!(tok.decode(&tok.encode("abc ab", false), false), "abc ab");
    assert_eq!(
        tok.encode("<|im_start|>x", true),
        [w.id("<|im_start|>"), w.id("x")]
    );
    assert_eq!(
        tok.encode("<|im_start|>", false).len(),
        "<|im_start|>".len()
    );
    assert!(tok.is_end_of_generation(w.id("<|im_end|>")));
    assert_eq!(tok.decode(&[w.id("<|im_end|>")], false), "");
}

#[test]
fn first_token_matches_a_naive_reference() {
    let w = Weights::new();
    let model = w.model();
    for token in [w.id("a"), w.id("abc"), 7] {
        let mut session = model.session(SessionOptions {
            n_ctx: 8,
            threads: 2,
        });
        let logits = session.eval(&[token]).unwrap();
        let expected = w.reference_first_logits(token);
        let scale = expected.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let max_err = logits
            .iter()
            .zip(&expected)
            .map(|(&a, b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        // The V cache is f16, so allow about 1e-3 relative error.
        assert!(
            max_err <= 2e-3 * scale,
            "token {token}: max error {max_err}, logit scale {scale}"
        );
    }
}

#[test]
fn incremental_and_batched_evaluation_agree() {
    let w = Weights::new();
    let model = w.model();
    let tokens = [5u32, 77, 200, 3, 258];
    let mut batched = model.session(SessionOptions {
        n_ctx: 16,
        threads: 3,
    });
    let batched_logits = batched.eval(&tokens).unwrap().to_vec();
    let mut stepped = model.session(SessionOptions {
        n_ctx: 16,
        threads: 1,
    });
    for &t in &tokens {
        stepped.eval(&[t]).unwrap();
    }
    assert_eq!(stepped.logits().unwrap(), &batched_logits[..]);
    assert_eq!(stepped.tokens(), tokens);
}

#[test]
fn truncate_restores_an_earlier_state() {
    let w = Weights::new();
    let model = w.model();
    let mut session = model.session(SessionOptions {
        n_ctx: 16,
        threads: 2,
    });
    session.eval(&[1, 2, 3, 4]).unwrap();
    session.truncate(2);
    assert!(session.logits().is_none());
    let after_truncate = session.eval(&[9]).unwrap().to_vec();

    let mut fresh = model.session(SessionOptions {
        n_ctx: 16,
        threads: 2,
    });
    assert_eq!(fresh.eval(&[1, 2, 9]).unwrap(), &after_truncate[..]);
}

#[test]
fn generation_is_deterministic_and_respects_limits() {
    let w = Weights::new();
    let model = w.model();
    let run = |config: SamplerConfig| {
        let mut session = model.session(SessionOptions {
            n_ctx: 32,
            threads: 2,
        });
        let mut sampler = Sampler::new(config);
        let mut out = Vec::new();
        let stop = session
            .generate(&[10, 20, 30], &mut sampler, 6, |t, logits| {
                assert_eq!(logits.len(), 260);
                out.push(t);
                true
            })
            .unwrap();
        assert_eq!(session.tokens().len(), 3 + out.len());
        (out, stop)
    };
    let (greedy, stop) = run(SamplerConfig::greedy());
    assert_eq!(greedy, run(SamplerConfig::greedy()).0);
    assert!(matches!(
        stop,
        StopReason::MaxTokens | StopReason::EndOfGeneration(_)
    ));
    let sampled = SamplerConfig {
        seed: 7,
        temperature: 1.5,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        ..SamplerConfig::default()
    };
    assert_eq!(run(sampled).0, run(sampled).0);
}

#[test]
fn evaluation_errors() {
    let w = Weights::new();
    let model = w.model();
    let mut session = model.session(SessionOptions {
        n_ctx: 4,
        threads: 1,
    });
    assert!(matches!(session.eval(&[]), Err(Error::EmptyInput)));
    assert!(matches!(
        session.eval(&[260]),
        Err(Error::InvalidToken(260))
    ));
    assert!(matches!(
        session.eval(&[1, 2, 3, 4, 5]),
        Err(Error::ContextFull { n_ctx: 4 })
    ));
    session.eval(&[1, 2, 3, 4]).unwrap();
    assert!(matches!(session.eval(&[5]), Err(Error::ContextFull { .. })));
    // Larger requests are capped at the training context.
    assert_eq!(
        model
            .session(SessionOptions {
                n_ctx: 1 << 20,
                threads: 1
            })
            .n_ctx(),
        N_CTX
    );
}

#[test]
fn load_errors_name_the_problem() {
    let w = Weights::new();
    let load = |bytes: Vec<u8>| Model::from_storage(Storage::from_bytes(&bytes)).unwrap_err();

    let err = load(w.gguf(|g| {
        g.add_tensor(
            "output.weight",
            GgmlType::Q4K,
            &[256, 260],
            vec![0; 260 * 144],
        );
    }));
    // output.weight has the wrong shape for this model.
    assert!(
        matches!(err, Error::TensorShape { ref tensor, .. } if tensor == "output.weight"),
        "{err}"
    );

    let err = load(w.gguf(|g| {
        g.add_tensor(
            "output.weight",
            GgmlType::I32,
            &[32, 260],
            vec![0; 32 * 260 * 4],
        );
    }));
    assert!(
        matches!(err, Error::UnsupportedTensorType { ref tensor, ggml_type: GgmlType::I32 } if tensor == "output.weight"),
        "{err}"
    );

    let mut bytes = Weights::new().gguf(|_| {});
    let at = bytes.windows(5).position(|w| w == b"qwen3").unwrap();
    bytes[at..at + 5].copy_from_slice(b"llamx");
    let err = load(bytes);
    assert!(
        matches!(err, Error::UnsupportedArchitecture(ref a) if a == "llamx"),
        "{err}"
    );

    // A zero size is reported by its key instead of breaking the forward pass.
    for name in [
        "qwen3.embedding_length",
        "qwen3.feed_forward_length",
        "qwen3.attention.key_length",
    ] {
        let mut bytes = Weights::new().gguf(|_| {});
        // The u32 value follows the key and its 4-byte type tag.
        let at = bytes
            .windows(name.len())
            .position(|w| w == name.as_bytes())
            .unwrap()
            + name.len()
            + 4;
        bytes[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
        let err = load(bytes);
        assert!(
            matches!(err, Error::InvalidMetadata { ref key, .. } if key == name),
            "{name}: {err}"
        );
    }
}
