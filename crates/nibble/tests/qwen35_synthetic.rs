//! End-to-end tests on a tiny random Qwen3.5 model built in memory: one
//! Gated DeltaNet layer, one gated attention layer and an MTP head, checked
//! against an independent f64 implementation over whole sequences.

use nibble::gguf::{GgmlType, GgufWriter, Storage};
use nibble::sampler::{Sampler, SamplerConfig};
use nibble::{Architecture, Model, SessionOptions, StopReason};

mod common;
use common::{
    Rng, add, add_tokenizer, assert_close, f32_bytes, matvec, rmsnorm, sigmoid, silu, vocab,
};

const N_EMBD: usize = 32;
const N_FF: usize = 48;
const N_CTX: usize = 64;
const EPS: f32 = 1e-6;
// Gated attention.
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2;
const HEAD_DIM: usize = 8;
const ROPE_DIMS: usize = 4;
const ROPE_BASE: f32 = 10_000.0;
// DeltaNet: more value heads than key heads, to exercise the head mapping.
const N_K: usize = 2;
const N_V: usize = 4;
const D: usize = 8;
const CONV_K: usize = 4;
const KEY_DIM: usize = N_K * D;
const VALUE_DIM: usize = N_V * D;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;

struct Ffn {
    norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
}

struct DeltaNet {
    attn_norm: Vec<f32>,
    qkv: Vec<f32>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    alpha: Vec<f32>,
    conv: Vec<f32>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
    ffn: Ffn,
}

struct Attention {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn: Ffn,
}

/// The MTP head: a gated attention block fed by the token's embedding and
/// a hidden state.
struct Mtp {
    eh_proj: Vec<f32>,
    enorm: Vec<f32>,
    hnorm: Vec<f32>,
    head_norm: Vec<f32>,
    block: Attention,
}

struct Weights {
    vocab: Vec<String>,
    embd: Vec<f32>,
    output_norm: Vec<f32>,
    dn: DeltaNet,
    attn: Attention,
    mtp: Mtp,
}

/// An attention layer's cached keys and values, by position.
#[derive(Default)]
struct Cache {
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
}

impl Attention {
    fn new(rng: &mut Rng) -> Self {
        Self {
            attn_norm: rng.norm(N_EMBD),
            wq: rng.matrix(2 * N_HEAD * HEAD_DIM, N_EMBD),
            wk: rng.matrix(N_HEAD_KV * HEAD_DIM, N_EMBD),
            wv: rng.matrix(N_HEAD_KV * HEAD_DIM, N_EMBD),
            wo: rng.matrix(N_EMBD, N_HEAD * HEAD_DIM),
            q_norm: rng.norm(HEAD_DIM),
            k_norm: rng.norm(HEAD_DIM),
            ffn: Ffn::new(rng),
        }
    }

    fn add_tensors(&self, w: &mut GgufWriter, layer: usize) {
        let (e, q2, kv) = (
            N_EMBD as u64,
            (2 * N_HEAD * HEAD_DIM) as u64,
            (N_HEAD_KV * HEAD_DIM) as u64,
        );
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        w.add_tensor(
            name("attn_norm"),
            GgmlType::F32,
            &[e],
            f32_bytes(&self.attn_norm),
        )
        .add_tensor(name("attn_q"), GgmlType::F32, &[e, q2], f32_bytes(&self.wq))
        .add_tensor(name("attn_k"), GgmlType::F32, &[e, kv], f32_bytes(&self.wk))
        .add_tensor(name("attn_v"), GgmlType::F32, &[e, kv], f32_bytes(&self.wv))
        .add_tensor(
            name("attn_output"),
            GgmlType::F32,
            &[q2 / 2, e],
            f32_bytes(&self.wo),
        )
        .add_tensor(
            name("attn_q_norm"),
            GgmlType::F32,
            &[HEAD_DIM as u64],
            f32_bytes(&self.q_norm),
        )
        .add_tensor(
            name("attn_k_norm"),
            GgmlType::F32,
            &[HEAD_DIM as u64],
            f32_bytes(&self.k_norm),
        );
        self.ffn.add_tensors(w, layer);
    }

    /// `x += gated attention(rmsnorm(x))` at `pos` (appending to `cache`),
    /// then `x += swiglu(rmsnorm(x))`.
    fn apply(&self, x: &mut [f64], pos: usize, cache: &mut Cache) {
        let at = self;
        let h = rmsnorm(x, &at.attn_norm, EPS);
        let qg = matvec(&at.wq, &h, 2 * N_HEAD * HEAD_DIM);
        let kk = matvec(&at.wk, &h, N_HEAD_KV * HEAD_DIM);
        let vv = matvec(&at.wv, &h, N_HEAD_KV * HEAD_DIM);
        let rope = |head: &mut [f64]| {
            let half = ROPE_DIMS / 2;
            for i in 0..half {
                let theta =
                    pos as f64 * f64::from(ROPE_BASE).powf(-2.0 * i as f64 / ROPE_DIMS as f64);
                let (a, b) = (head[i], head[i + half]);
                head[i] = a * theta.cos() - b * theta.sin();
                head[i + half] = a * theta.sin() + b * theta.cos();
            }
        };
        let mut key = Vec::new();
        for hk in 0..N_HEAD_KV {
            let mut head = rmsnorm(&kk[hk * HEAD_DIM..(hk + 1) * HEAD_DIM], &at.k_norm, EPS);
            rope(&mut head);
            key.extend(head);
        }
        cache.keys.push(key);
        cache.values.push(vv);
        let mut att = vec![0.0; N_HEAD * HEAD_DIM];
        for hq in 0..N_HEAD {
            let mut qh = rmsnorm(
                &qg[hq * 2 * HEAD_DIM..hq * 2 * HEAD_DIM + HEAD_DIM],
                &at.q_norm,
                EPS,
            );
            rope(&mut qh);
            let gate = &qg[hq * 2 * HEAD_DIM + HEAD_DIM..(hq + 1) * 2 * HEAD_DIM];
            let off = (hq / (N_HEAD / N_HEAD_KV)) * HEAD_DIM;
            let scores: Vec<f64> = cache
                .keys
                .iter()
                .map(|kt| {
                    (0..HEAD_DIM).map(|i| qh[i] * kt[off + i]).sum::<f64>()
                        / (HEAD_DIM as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for j in 0..HEAD_DIM {
                let a: f64 = weights
                    .iter()
                    .zip(&cache.values)
                    .map(|(w, vt)| w * vt[off + j])
                    .sum::<f64>()
                    / total;
                att[hq * HEAD_DIM + j] = a * sigmoid(gate[j]);
            }
        }
        add(x, &matvec(&at.wo, &att, N_EMBD));
        at.ffn.apply(x);
    }
}

impl Ffn {
    fn new(rng: &mut Rng) -> Self {
        Self {
            norm: rng.norm(N_EMBD),
            gate: rng.matrix(N_FF, N_EMBD),
            up: rng.matrix(N_FF, N_EMBD),
            down: rng.matrix(N_EMBD, N_FF),
        }
    }

    fn add_tensors(&self, w: &mut GgufWriter, layer: usize) {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        w.add_tensor(
            name("post_attention_norm"),
            GgmlType::F32,
            &[N_EMBD as u64],
            f32_bytes(&self.norm),
        )
        .add_tensor(
            name("ffn_gate"),
            GgmlType::F32,
            &[N_EMBD as u64, N_FF as u64],
            f32_bytes(&self.gate),
        )
        .add_tensor(
            name("ffn_up"),
            GgmlType::F32,
            &[N_EMBD as u64, N_FF as u64],
            f32_bytes(&self.up),
        )
        .add_tensor(
            name("ffn_down"),
            GgmlType::F32,
            &[N_FF as u64, N_EMBD as u64],
            f32_bytes(&self.down),
        );
    }

    /// `x += swiglu(rmsnorm(x))`.
    fn apply(&self, x: &mut [f64]) {
        let h = rmsnorm(x, &self.norm, EPS);
        let gate = matvec(&self.gate, &h, N_FF);
        let up = matvec(&self.up, &h, N_FF);
        let act: Vec<f64> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        add(x, &matvec(&self.down, &act, N_EMBD));
    }
}

impl Weights {
    fn new() -> Self {
        let mut rng = Rng(0x0BAD_5EED);
        let vocab = vocab();
        let dn = DeltaNet {
            attn_norm: rng.norm(N_EMBD),
            qkv: rng.matrix(CONV_DIM, N_EMBD),
            gate: rng.matrix(VALUE_DIM, N_EMBD),
            beta: rng.matrix(N_V, N_EMBD),
            alpha: rng.matrix(N_V, N_EMBD),
            conv: rng.vec(CONV_DIM * CONV_K, 1.5),
            dt_bias: rng.vec(N_V, 2.0),
            a: (0..N_V).map(|_| -(0.05 + rng.next().abs())).collect(),
            norm: rng.norm(D),
            out: rng.matrix(N_EMBD, VALUE_DIM),
            ffn: Ffn::new(&mut rng),
        };
        let attn = Attention::new(&mut rng);
        let embd = rng.vec(vocab.len() * N_EMBD, 2.0);
        let output_norm = rng.norm(N_EMBD);
        let mtp = Mtp {
            eh_proj: rng.matrix(N_EMBD, 2 * N_EMBD),
            enorm: rng.norm(N_EMBD),
            hnorm: rng.norm(N_EMBD),
            head_norm: rng.norm(N_EMBD),
            block: Attention::new(&mut rng),
        };
        Self {
            embd,
            output_norm,
            dn,
            attn,
            mtp,
            vocab,
        }
    }

    fn gguf(&self) -> Vec<u8> {
        self.gguf_with(true)
    }

    /// The file, with or without the MTP head's tensors (its block is
    /// declared either way).
    fn gguf_with(&self, mtp: bool) -> Vec<u8> {
        let e = N_EMBD as u64;
        let mut w = GgufWriter::new();
        w.add("general.architecture", "qwen35")
            // Three blocks, the last one an MTP head that is not part of the trunk.
            .add("qwen35.block_count", 3u32)
            .add("qwen35.nextn_predict_layers", 1u32)
            .add("qwen35.context_length", N_CTX as u32)
            .add("qwen35.embedding_length", N_EMBD as u32)
            .add("qwen35.feed_forward_length", N_FF as u32)
            .add("qwen35.attention.head_count", N_HEAD as u32)
            .add("qwen35.attention.head_count_kv", N_HEAD_KV as u32)
            .add("qwen35.attention.key_length", HEAD_DIM as u32)
            .add("qwen35.attention.value_length", HEAD_DIM as u32)
            .add("qwen35.attention.layer_norm_rms_epsilon", EPS)
            .add("qwen35.rope.freq_base", ROPE_BASE)
            .add("qwen35.rope.dimension_count", ROPE_DIMS as u32)
            .add("qwen35.ssm.conv_kernel", CONV_K as u32)
            .add("qwen35.ssm.state_size", D as u32)
            .add("qwen35.ssm.group_count", N_K as u32)
            .add("qwen35.ssm.time_step_rank", N_V as u32)
            .add("qwen35.ssm.inner_size", VALUE_DIM as u32)
            .add("qwen35.full_attention_interval", 2u32);
        add_tokenizer(&mut w, &self.vocab, "qwen35");
        w.add_tensor(
            "token_embd.weight",
            GgmlType::F32,
            &[e, self.vocab.len() as u64],
            f32_bytes(&self.embd),
        )
        .add_tensor(
            "output_norm.weight",
            GgmlType::F32,
            &[e],
            f32_bytes(&self.output_norm),
        );

        let dn = &self.dn;
        w.add_tensor(
            "blk.0.attn_norm.weight",
            GgmlType::F32,
            &[e],
            f32_bytes(&dn.attn_norm),
        )
        .add_tensor(
            "blk.0.attn_qkv.weight",
            GgmlType::F32,
            &[e, CONV_DIM as u64],
            f32_bytes(&dn.qkv),
        )
        .add_tensor(
            "blk.0.attn_gate.weight",
            GgmlType::F32,
            &[e, VALUE_DIM as u64],
            f32_bytes(&dn.gate),
        )
        .add_tensor(
            "blk.0.ssm_beta.weight",
            GgmlType::F32,
            &[e, N_V as u64],
            f32_bytes(&dn.beta),
        )
        .add_tensor(
            "blk.0.ssm_alpha.weight",
            GgmlType::F32,
            &[e, N_V as u64],
            f32_bytes(&dn.alpha),
        )
        .add_tensor(
            "blk.0.ssm_conv1d.weight",
            GgmlType::F32,
            &[CONV_K as u64, CONV_DIM as u64],
            f32_bytes(&dn.conv),
        )
        .add_tensor(
            "blk.0.ssm_dt.bias",
            GgmlType::F32,
            &[N_V as u64],
            f32_bytes(&dn.dt_bias),
        )
        .add_tensor(
            "blk.0.ssm_a",
            GgmlType::F32,
            &[N_V as u64],
            f32_bytes(&dn.a),
        )
        .add_tensor(
            "blk.0.ssm_norm.weight",
            GgmlType::F32,
            &[D as u64],
            f32_bytes(&dn.norm),
        )
        .add_tensor(
            "blk.0.ssm_out.weight",
            GgmlType::F32,
            &[VALUE_DIM as u64, e],
            f32_bytes(&dn.out),
        );
        dn.ffn.add_tensors(&mut w, 0);

        self.attn.add_tensors(&mut w, 1);
        if mtp {
            let m = &self.mtp;
            m.block.add_tensors(&mut w, 2);
            w.add_tensor(
                "blk.2.nextn.eh_proj.weight",
                GgmlType::F32,
                &[2 * e, e],
                f32_bytes(&m.eh_proj),
            )
            .add_tensor(
                "blk.2.nextn.enorm.weight",
                GgmlType::F32,
                &[e],
                f32_bytes(&m.enorm),
            )
            .add_tensor(
                "blk.2.nextn.hnorm.weight",
                GgmlType::F32,
                &[e],
                f32_bytes(&m.hnorm),
            )
            .add_tensor(
                "blk.2.nextn.shared_head_norm.weight",
                GgmlType::F32,
                &[e],
                f32_bytes(&m.head_norm),
            );
        }
        w.to_bytes()
    }

    fn model(&self) -> Model {
        Model::from_storage(Storage::from_bytes(&self.gguf())).unwrap()
    }

    /// Next-token logits after `tokens`, computed naively in f64 with explicit state.
    fn reference_logits(&self, tokens: &[u32]) -> Vec<f64> {
        self.reference_trunk(tokens).0
    }

    /// The MTP head's logits for the token after `chain`, drafted after
    /// `tokens`: the head first goes over `tokens` (each paired with the
    /// trunk's normalized hidden state at the position before, zero before
    /// the first), then over `chain`, the first paired with the trunk's
    /// last hidden state and each later one with the head's own output.
    fn reference_draft(&self, tokens: &[u32], chain: &[u32]) -> Vec<f64> {
        let m = &self.mtp;
        let (_, hs) = self.reference_trunk(tokens);
        let mut cache = Cache::default();
        let run = |t: u32, pos: usize, h: &[f64], cache: &mut Cache| {
            let t = t as usize;
            let e: Vec<f64> = self.embd[t * N_EMBD..(t + 1) * N_EMBD]
                .iter()
                .map(|&v| f64::from(v))
                .collect();
            let mut pair = rmsnorm(&e, &m.enorm, EPS);
            pair.extend(rmsnorm(h, &m.hnorm, EPS));
            let mut x = matvec(&m.eh_proj, &pair, N_EMBD);
            m.block.apply(&mut x, pos, cache);
            let out = rmsnorm(&x, &m.head_norm, EPS);
            (matvec(&self.embd, &out, self.vocab.len()), out)
        };
        let mut h = vec![0.0; N_EMBD];
        for (pos, &t) in tokens.iter().enumerate() {
            run(t, pos, &h, &mut cache);
            h.clone_from(&hs[pos]);
        }
        let mut logits = Vec::new();
        for (i, &t) in chain.iter().enumerate() {
            (logits, h) = run(t, tokens.len() + i, &h, &mut cache);
        }
        logits
    }

    /// The trunk's logits after `tokens`, and its final normalized hidden
    /// state at every position.
    fn reference_trunk(&self, tokens: &[u32]) -> (Vec<f64>, Vec<Vec<f64>>) {
        let dn = &self.dn;
        let mut cache = Cache::default();
        let mut hs = Vec::new();
        let mut conv_history = vec![[0.0f64; CONV_K - 1]; CONV_DIM];
        // state[h][i][j]: value head h, key index i, value index j.
        let mut state = vec![[[0.0f64; D]; D]; N_V];
        for (pos, &t) in tokens.iter().enumerate() {
            let t = t as usize;
            let mut x: Vec<f64> = self.embd[t * N_EMBD..(t + 1) * N_EMBD]
                .iter()
                .map(|&v| f64::from(v))
                .collect();

            // Layer 0: Gated DeltaNet.
            let h = rmsnorm(&x, &dn.attn_norm, EPS);
            let qkv = matvec(&dn.qkv, &h, CONV_DIM);
            let mut y = vec![0.0; CONV_DIM];
            for c in 0..CONV_DIM {
                let taps = &dn.conv[c * CONV_K..(c + 1) * CONV_K];
                let hist = &mut conv_history[c];
                let sum: f64 = (0..CONV_K - 1)
                    .map(|i| hist[i] * f64::from(taps[i]))
                    .sum::<f64>()
                    + qkv[c] * f64::from(taps[CONV_K - 1]);
                hist.rotate_left(1);
                hist[CONV_K - 2] = qkv[c];
                y[c] = silu(sum);
            }
            let l2 = |v: &[f64]| -> Vec<f64> {
                let n = (v.iter().map(|a| a * a).sum::<f64>() + f64::from(EPS)).sqrt();
                v.iter().map(|a| a / n).collect()
            };
            let q: Vec<Vec<f64>> = (0..N_K).map(|i| l2(&y[i * D..(i + 1) * D])).collect();
            let k: Vec<Vec<f64>> = (0..N_K)
                .map(|i| l2(&y[KEY_DIM + i * D..KEY_DIM + (i + 1) * D]))
                .collect();
            let v = &y[2 * KEY_DIM..];
            let z = matvec(&dn.gate, &h, VALUE_DIM);
            let beta = matvec(&dn.beta, &h, N_V);
            let alpha = matvec(&dn.alpha, &h, N_V);
            let mut o = vec![0.0; VALUE_DIM];
            for hv in 0..N_V {
                let (qh, kh) = (&q[hv % N_K], &k[hv % N_K]);
                let b = sigmoid(beta[hv]);
                let g = (1.0 + (alpha[hv] + f64::from(dn.dt_bias[hv])).exp()).ln()
                    * f64::from(dn.a[hv]);
                let s = &mut state[hv];
                for row in s.iter_mut() {
                    for e in row.iter_mut() {
                        *e *= g.exp();
                    }
                }
                for j in 0..D {
                    let kv: f64 = (0..D).map(|i| s[i][j] * kh[i]).sum();
                    let delta = (v[hv * D + j] - kv) * b;
                    for i in 0..D {
                        s[i][j] += kh[i] * delta;
                    }
                }
                for j in 0..D {
                    o[hv * D + j] =
                        (0..D).map(|i| s[i][j] * qh[i]).sum::<f64>() / (D as f64).sqrt();
                }
                let normed = rmsnorm(&o[hv * D..(hv + 1) * D], &dn.norm, EPS);
                for j in 0..D {
                    o[hv * D + j] = normed[j] * silu(z[hv * D + j]);
                }
            }
            add(&mut x, &matvec(&dn.out, &o, N_EMBD));
            dn.ffn.apply(&mut x);

            // Layer 1: gated attention with partial NeoX RoPE.
            self.attn.apply(&mut x, pos, &mut cache);
            hs.push(rmsnorm(&x, &self.output_norm, EPS));
        }
        let logits = matvec(&self.embd, hs.last().unwrap(), self.vocab.len());
        (logits, hs)
    }
}

fn session(model: &Model) -> nibble::Session<'_> {
    model
        .session(SessionOptions {
            n_ctx: 32,
            threads: 2,
            backend: nibble::BackendKind::Cpu,
            n_batch: 0,
            draft_max: 0,
        })
        .unwrap()
}

#[test]
fn loads_the_hybrid_layout() {
    let model = Weights::new().model();
    let c = model.config();
    assert_eq!(c.architecture, Architecture::Qwen35);
    assert_eq!(c.n_layer, 2, "the MTP block is not part of the trunk");
    assert_eq!((c.head_dim, c.rope_dims), (HEAD_DIM, ROPE_DIMS));
    assert!(c.tied_embeddings);
}

#[test]
fn matches_a_naive_reference_over_a_sequence() {
    let w = Weights::new();
    let model = w.model();
    let tokens = [17u32, 99, 258, 3, 200, 65, 66];
    for n in [1, 2, tokens.len()] {
        let mut s = session(&model);
        let logits = s.eval(&tokens[..n]).unwrap();
        // K/V are cached in f16 and queries meet keys in f16, as in llama.cpp.
        assert_close(logits, &w.reference_logits(&tokens[..n]), 3e-3);
    }
}

#[test]
fn incremental_and_batched_evaluation_agree() {
    let model = Weights::new().model();
    let tokens = [5u32, 77, 200, 3, 258];
    let mut batched = session(&model);
    let expected = batched.eval(&tokens).unwrap().to_vec();
    let mut stepped = session(&model);
    for &t in &tokens {
        stepped.eval(&[t]).unwrap();
    }
    assert_eq!(stepped.logits().unwrap(), &expected[..]);
}

#[test]
fn truncate_replays_recurrent_state() {
    let model = Weights::new().model();
    let mut s = session(&model);
    s.eval(&[1, 2, 3, 4]).unwrap();
    s.truncate(2).unwrap();
    assert_eq!(s.tokens(), [1, 2]);
    let after = s.eval(&[9]).unwrap().to_vec();
    let mut fresh = session(&model);
    assert_eq!(fresh.eval(&[1, 2, 9]).unwrap(), &after[..]);

    s.clear().unwrap();
    assert!(s.tokens().is_empty());
    assert_eq!(s.eval(&[1, 2, 9]).unwrap(), &after[..]);
}

/// A chat turn diverges from the cached prompt a few tokens before its end;
/// the checkpoint taken there spares the replay of everything before it.
#[test]
fn checkpoints_spare_most_of_the_replay() {
    use nibble::CHECKPOINT_TAIL;

    let model = Weights::new().model();
    let prompt = [17u32, 99, 258, 3, 200, 65, 66, 5, 9];
    let checkpoint = prompt.len() - CHECKPOINT_TAIL;
    let mut s = session(&model);
    s.eval(&prompt).unwrap();

    // The tokens after the checkpoint were the last batch, short enough to
    // stay pending: diverging inside it replays nothing.
    let keep = checkpoint + 2;
    assert_eq!(s.truncate(keep).unwrap(), 0);
    let after = s.eval(&[1, 2]).unwrap().to_vec();
    let mut fresh = session(&model);
    let expected: Vec<u32> = prompt[..keep].iter().copied().chain([1, 2]).collect();
    assert_eq!(fresh.eval(&expected).unwrap(), &after[..]);

    // Exactly at the checkpoint nothing is replayed.
    assert_eq!(s.truncate(checkpoint).unwrap(), 0);
    let at = s.eval(&[7]).unwrap().to_vec();
    let mut fresh = session(&model);
    let expected: Vec<u32> = prompt[..checkpoint].iter().copied().chain([7]).collect();
    assert_eq!(fresh.eval(&expected).unwrap(), &at[..]);

    // Before every checkpoint, the whole prefix is replayed.
    assert_eq!(s.truncate(3).unwrap(), 3);
    let early = s.eval(&[4]).unwrap().to_vec();
    assert_eq!(session(&model).eval(&[17, 99, 258, 4]).unwrap(), &early[..]);
}

#[test]
fn generation_is_deterministic() {
    let model = Weights::new().model();
    let run = || {
        let mut s = session(&model);
        let mut sampler = Sampler::new(SamplerConfig::greedy());
        let mut out = Vec::new();
        let stop = s
            .generate(&[10, 20], &mut sampler, 8, |t, _| {
                out.push(t);
                true
            })
            .unwrap();
        assert!(matches!(
            stop,
            StopReason::MaxTokens | StopReason::EndOfGeneration(_)
        ));
        out
    };
    assert_eq!(run(), run());
}

#[test]
fn metal_matches_a_naive_reference_and_rewinds() {
    let w = Weights::new();
    let model = w.model();
    let Some(mut s) = common::metal_session(&model, N_CTX) else {
        return;
    };
    let tokens = [17u32, 99, 258, 3, 200, 65, 66];
    let logits = s.eval(&tokens).unwrap();
    assert_close(logits, &w.reference_logits(&tokens), 3e-3);
    // Back to the checkpoint taken 4 tokens before the end, and on.
    s.truncate(4).unwrap();
    let logits = s.eval(&[1, 2]).unwrap();
    let mut expected = tokens[..4].to_vec();
    expected.extend([1, 2]);
    assert_close(logits, &w.reference_logits(&expected), 3e-3);
}

/// Batches give every token the logits it gets alone, on the CPU and on Metal.
#[test]
fn batch_size_does_not_change_the_logits() {
    let model = Weights::new().model();
    let n_vocab = model.config().n_vocab as u32;
    let tokens: Vec<u32> = (0..23u32).map(|i| (i * 37 + 11) % n_vocab).collect();
    for backend in [nibble::BackendKind::Cpu, nibble::BackendKind::Metal] {
        common::assert_batches_match_steps(&model, backend, N_CTX, &tokens, 9);
    }
}

/// Taking back a draft costs no replay, on the CPU and on Metal.
#[test]
fn short_evaluations_rewind_without_replay() {
    let model = Weights::new().model();
    let n_vocab = model.config().n_vocab as u32;
    let tokens: Vec<u32> = (0..30u32).map(|i| (i * 53 + 7) % n_vocab).collect();
    for backend in [nibble::BackendKind::Cpu, nibble::BackendKind::Metal] {
        common::assert_rewinds_into_short_batches(&model, backend, N_CTX, &tokens);
    }
}

fn drafting(
    model: &Model,
    backend: nibble::BackendKind,
    n_ctx: usize,
    n_batch: usize,
) -> Option<nibble::Session<'_>> {
    let options = SessionOptions {
        n_ctx,
        threads: 2,
        backend,
        n_batch,
        draft_max: 3,
    };
    common::session_or_skip(model, options)
}

const BACKENDS: [nibble::BackendKind; 2] = [nibble::BackendKind::Cpu, nibble::BackendKind::Metal];

#[test]
fn mtp_head_matches_a_naive_reference() {
    let w = Weights::new();
    let model = w.model();
    assert!(model.has_mtp());
    let tokens = [17u32, 99, 258, 3, 200, 65, 66];
    for backend in BACKENDS {
        let Some(mut s) = drafting(&model, backend, N_CTX, 0) else {
            continue;
        };
        s.eval(&tokens).unwrap();
        let next = [5u32, 77, 140];
        for n in 1..=next.len() {
            let got = s.draft_logits(&next[..n]).unwrap();
            assert_close(got, &w.reference_draft(&tokens, &next[..n]), 3e-3);
        }
        // Drafting leaves the session as it was.
        assert_eq!(s.tokens(), tokens);
        assert_close(s.logits().unwrap(), &w.reference_logits(&tokens), 3e-3);
    }
}

/// However the tokens got there (in small batches, a draft taken back, a
/// checkpoint restored), the head drafts exactly as after evaluating them
/// at once.
#[test]
fn mtp_state_follows_evaluation_and_rewinds() {
    let model = Weights::new().model();
    let n_vocab = model.config().n_vocab as u32;
    let t: Vec<u32> = (0..40u32).map(|i| (i * 29 + 3) % n_vocab).collect();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for backend in BACKENDS {
        let Some(mut s) = drafting(&model, backend, N_CTX, 3) else {
            continue;
        };
        // The draft logits after `seq`, evaluated at once.
        let want = |seq: &[u32], next: &[u32]| {
            let mut fresh = drafting(&model, backend, N_CTX, 0).unwrap();
            fresh.eval(seq).unwrap();
            bits(fresh.draft_logits(next).unwrap())
        };
        let check = |s: &mut nibble::Session<'_>, what: &str| {
            let seq = s.tokens().to_vec();
            let next = [t[39], t[38]];
            assert_eq!(
                bits(s.draft_logits(&next).unwrap()),
                want(&seq, &next),
                "{backend}: {what}"
            );
        };
        s.eval(&t[..10]).unwrap();
        check(&mut s, "batches of 3");
        s.eval_all(&t[10..16]).unwrap();
        check(&mut s, "after a verification");
        s.truncate(12).unwrap();
        check(&mut s, "after taking back a draft");
        s.truncate(10).unwrap();
        check(&mut s, "after taking back all of it");
        s.draft_logits(&t[20..23]).unwrap();
        s.eval(&t[23..33]).unwrap();
        check(&mut s, "after drafting and evaluating");
        s.eval(&[t[33]]).unwrap();
        // The checkpoint 4 tokens before the end of those 10 is at 16.
        assert_eq!(s.truncate(17).unwrap(), 1);
        check(&mut s, "after a checkpoint and a replay");
        s.clear().unwrap();
        s.eval(&t[..5]).unwrap();
        check(&mut s, "after clearing");
    }
}

#[test]
fn a_head_without_tensors_is_ignored() {
    let w = Weights::new();
    let model = Model::from_storage(Storage::from_bytes(&w.gguf_with(false))).unwrap();
    assert!(!model.has_mtp());
    let options = SessionOptions {
        backend: nibble::BackendKind::Cpu,
        draft_max: 2,
        ..SessionOptions::default()
    };
    assert!(matches!(
        model.session(options),
        Err(nibble::Error::NoDraftHead)
    ));
    let mut s = session(&model);
    s.eval(&[1, 2]).unwrap();
    assert!(matches!(
        s.draft_logits(&[3]),
        Err(nibble::Error::NoDraftHead)
    ));
}

/// Generation with drafts gives the tokens, stop reason, final session and
/// logits of generation without them, greedy and sampled, for each way of
/// stopping.
#[test]
fn drafting_does_not_change_generation() {
    let model = Weights::new().model();
    let prompt = [10u32, 20, 30, 40, 50];
    let configs = [
        SamplerConfig::greedy(),
        SamplerConfig {
            temperature: 0.9,
            top_k: 20,
            top_p: 0.9,
            repeat_penalty: 1.1,
            seed: 7,
            ..SamplerConfig::default()
        },
    ];
    for backend in BACKENDS {
        for config in &configs {
            // (n_ctx, max_tokens, stop after this many tokens)
            for (n_ctx, max_tokens, stop_at) in [
                (N_CTX, 30, None),
                (N_CTX, 1, None),
                (N_CTX, 30, Some(7)),
                (12, 30, None),
            ] {
                let run = |draft_max: usize| {
                    let options = SessionOptions {
                        n_ctx,
                        threads: 2,
                        backend,
                        n_batch: 0,
                        draft_max,
                    };
                    let mut s = common::session_or_skip(&model, options)?;
                    let mut sampler = Sampler::new(*config);
                    let mut out = Vec::new();
                    let stop = s
                        .generate(&prompt, &mut sampler, max_tokens, |t, logits| {
                            out.push((t, logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>()));
                            stop_at.is_none_or(|n| out.len() < n)
                        })
                        .unwrap();
                    let logits = s.logits().map(<[f32]>::to_vec);
                    Some((out, stop, s.tokens().to_vec(), logits, s.draft_stats()))
                };
                let Some(plain) = run(0) else { continue };
                for draft_max in [1, 3, 15] {
                    let drafted = run(draft_max).unwrap();
                    let what = format!(
                        "{backend}, {config:?}, n_ctx {n_ctx}, max {max_tokens}, stop {stop_at:?}, drafts {draft_max}"
                    );
                    assert!(drafted.0 == plain.0, "{what}: tokens");
                    assert_eq!(drafted.1, plain.1, "{what}: stop");
                    assert_eq!(drafted.2, plain.2, "{what}: session tokens");
                    assert!(drafted.3 == plain.3, "{what}: logits");
                    assert!(
                        max_tokens == 1 || drafted.4.drafted > 0,
                        "{what}: nothing drafted"
                    );
                }
            }
        }
    }
}
