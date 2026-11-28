//! End-to-end tests on tiny random LFM2 models built in memory: one
//! shortconv layer and one attention layer, repeated, checked against an
//! independent f64 implementation over whole sequences; and the same
//! model with sigmoid-gated mixture-of-experts feed-forwards (`lfm2moe`).

use nibble::gguf::{GgmlType, Gguf, GgufWriter, OwnedValue, Storage, ValueType};
use nibble::sampler::{Sampler, SamplerConfig};
use nibble::tokenizer::Tokenizer;
use nibble::{Architecture, Model, SessionOptions, StopReason};

mod common;
use common::{
    Rng, add, add_tokenizer, assert_close, f32_bytes, matvec, rmsnorm, sigmoid, silu, vocab,
};

const N_EMBD: usize = 32;
const N_FF: usize = 48;
const N_CTX: usize = 64;
const EPS: f32 = 1e-6;
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2;
const HEAD_DIM: usize = 8;
const ROPE_DIMS: usize = 4;
const ROPE_BASE: f32 = 1_000_000.0;
// The shortconv block: a 3-tap convolution, as LFM2's files carry.
const TAPS: usize = 3;
const ROWS: usize = TAPS - 1;
// Mixture of experts.
const N_EXPERT: usize = 6;
const N_USED: usize = 3;
const N_FF_EXP: usize = 16;

/// One block's feed-forward: dense, or LFM2's sigmoid-gated experts.
struct Ffn {
    norm: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    moe: Option<Moe>,
}

struct Moe {
    router: Vec<f32>,
    /// `N_EXPERT` matrices, one after another.
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    /// The selection bias, one value per expert.
    bias: Vec<f32>,
}

impl Moe {
    fn new(rng: &mut Rng) -> Self {
        Self {
            router: rng.matrix(N_EXPERT, N_EMBD),
            gate: rng.matrix(N_EXPERT * N_FF_EXP, N_EMBD),
            up: rng.matrix(N_EXPERT * N_FF_EXP, N_EMBD),
            down: rng.matrix(N_EXPERT * N_EMBD, N_FF_EXP),
            bias: rng.vec(N_EXPERT, 0.1),
        }
    }

    fn add_tensors(&self, w: &mut GgufWriter, layer: usize) {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let (e, x, f) = (N_EMBD as u64, N_EXPERT as u64, N_FF_EXP as u64);
        w.add_tensor(
            name("ffn_gate_inp"),
            GgmlType::F32,
            &[e, x],
            f32_bytes(&self.router),
        )
        .add_tensor(
            name("ffn_gate_exps"),
            GgmlType::F32,
            &[e, f, x],
            f32_bytes(&self.gate),
        )
        .add_tensor(
            name("ffn_up_exps"),
            GgmlType::F32,
            &[e, f, x],
            f32_bytes(&self.up),
        )
        .add_tensor(
            name("ffn_down_exps"),
            GgmlType::F32,
            &[f, e, x],
            f32_bytes(&self.down),
        );
        w.add_tensor(
            format!("blk.{layer}.ffn_exp_probs_b.bias"),
            GgmlType::F32,
            &[x],
            f32_bytes(&self.bias),
        );
    }

    /// The experts' weighted sum of `h`, as llama.cpp's `build_lfm_moe_ffn`
    /// runs it: each expert's sigmoid on its own, the top `N_USED` of those
    /// plus the selection bias chosen, weighted by the unbiased values
    /// divided by their sum.
    fn apply(&self, h: &[f64]) -> Vec<f64> {
        let logits = matvec(&self.router, h, N_EXPERT);
        let probs: Vec<f64> = logits.iter().map(|&v| sigmoid(v)).collect();
        let sel: Vec<f64> = probs
            .iter()
            .zip(&self.bias)
            .map(|(&p, &b)| p + f64::from(b))
            .collect();
        let mut order: Vec<usize> = (0..N_EXPERT).collect();
        order.sort_by(|&a, &b| sel[b].total_cmp(&sel[a]).then(a.cmp(&b)));
        order.truncate(N_USED);
        let weights: Vec<f64> = order.iter().map(|&e| probs[e]).collect();
        let total = weights.iter().sum::<f64>().max(6.103_515_625e-5);
        let mut out = vec![0.0; N_EMBD];
        for (j, &e) in order.iter().enumerate() {
            // Expert `e`'s matrices start `e` whole matrices in.
            let (ff, down) = (e * N_FF_EXP * N_EMBD, e * N_EMBD * N_FF_EXP);
            let gate = matvec(&self.gate[ff..], h, N_FF_EXP);
            let up = matvec(&self.up[ff..], h, N_FF_EXP);
            let hidden: Vec<f64> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
            let d = matvec(&self.down[down..], &hidden, N_EMBD);
            for (o, v) in out.iter_mut().zip(d) {
                *o += weights[j] / total * v;
            }
        }
        out
    }
}

impl Ffn {
    fn new(rng: &mut Rng) -> Self {
        Self {
            norm: rng.norm(N_EMBD),
            gate: rng.matrix(N_FF, N_EMBD),
            up: rng.matrix(N_FF, N_EMBD),
            down: rng.matrix(N_EMBD, N_FF),
            moe: None,
        }
    }

    fn add_tensors(&self, w: &mut GgufWriter, layer: usize) {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let (e, f) = (N_EMBD as u64, N_FF as u64);
        w.add_tensor(name("ffn_norm"), GgmlType::F32, &[e], f32_bytes(&self.norm));
        if let Some(moe) = &self.moe {
            moe.add_tensors(w, layer);
            return;
        }
        w.add_tensor(
            name("ffn_gate"),
            GgmlType::F32,
            &[e, f],
            f32_bytes(&self.gate),
        )
        .add_tensor(name("ffn_up"), GgmlType::F32, &[e, f], f32_bytes(&self.up))
        .add_tensor(
            name("ffn_down"),
            GgmlType::F32,
            &[f, e],
            f32_bytes(&self.down),
        );
    }

    /// `x += ffn(rmsnorm(x))` in f64.
    fn apply(&self, x: &mut [f64]) {
        let h = rmsnorm(x, &self.norm, EPS);
        if let Some(moe) = &self.moe {
            add(x, &moe.apply(&h));
            return;
        }
        let gate = matvec(&self.gate, &h, N_FF);
        let up = matvec(&self.up, &h, N_FF);
        let hidden: Vec<f64> = gate.iter().zip(&up).map(|(&g, &u)| silu(g) * u).collect();
        add(x, &matvec(&self.down, &hidden, N_EMBD));
    }
}

/// LFM2's shortconv block.
struct Conv {
    attn_norm: Vec<f32>,
    in_proj: Vec<f32>,
    /// The taps by channel: channel `ch`'s `TAPS` taps, oldest first,
    /// as the file stores them.
    taps: Vec<f32>,
    out_proj: Vec<f32>,
    ffn: Ffn,
}

impl Conv {
    fn new(rng: &mut Rng) -> Self {
        Self {
            attn_norm: rng.norm(N_EMBD),
            in_proj: rng.matrix(3 * N_EMBD, N_EMBD),
            taps: rng.matrix(N_EMBD, TAPS),
            out_proj: rng.matrix(N_EMBD, N_EMBD),
            ffn: Ffn::new(rng),
        }
    }

    fn add_tensors(&self, w: &mut GgufWriter, layer: usize) {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let e = N_EMBD as u64;
        w.add_tensor(
            name("attn_norm"),
            GgmlType::F32,
            &[e],
            f32_bytes(&self.attn_norm),
        )
        .add_tensor(
            name("shortconv.in_proj"),
            GgmlType::F32,
            &[e, 3 * e],
            f32_bytes(&self.in_proj),
        )
        .add_tensor(
            name("shortconv.conv"),
            GgmlType::F32,
            &[TAPS as u64, e],
            f32_bytes(&self.taps),
        )
        .add_tensor(
            name("shortconv.out_proj"),
            GgmlType::F32,
            &[e, e],
            f32_bytes(&self.out_proj),
        );
        self.ffn.add_tensors(w, layer);
    }

    /// `x += out_proj * (c * conv(b * x))` at one position, with the
    /// convolution over `history`'s rows and the token's own `b * x`;
    /// the history keeps the last `ROWS` values.
    fn apply(&self, x: &mut [f64], history: &mut Vec<Vec<f64>>) {
        let h = rmsnorm(x, &self.attn_norm, EPS);
        let bcx = matvec(&self.in_proj, &h, 3 * N_EMBD);
        let (b, rest) = bcx.split_at(N_EMBD);
        let (c, xi) = rest.split_at(N_EMBD);
        let bx: Vec<f64> = b.iter().zip(xi).map(|(&b, &x)| b * x).collect();
        let mut conv = vec![0.0; N_EMBD];
        for (ch, taps) in self.taps.as_chunks::<TAPS>().0.iter().enumerate() {
            let mut sum = 0.0;
            for (k, &tap) in taps.iter().enumerate() {
                let v = if k < ROWS {
                    history[history.len() - ROWS + k][ch]
                } else {
                    bx[ch]
                };
                sum += v * f64::from(tap);
            }
            conv[ch] = sum;
        }
        let y: Vec<f64> = c.iter().zip(&conv).map(|(&c, &v)| c * v).collect();
        add(x, &matvec(&self.out_proj, &y, N_EMBD));
        history.push(bx);
        if history.len() > ROWS {
            history.remove(0);
        }
    }
}

/// An attention layer, as LFM2's: ungated queries, per-head Q/K RMSNorm.
struct Attn {
    attn_norm: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    wo: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ffn: Ffn,
}

impl Attn {
    fn new(rng: &mut Rng) -> Self {
        Self {
            attn_norm: rng.norm(N_EMBD),
            wq: rng.matrix(N_HEAD * HEAD_DIM, N_EMBD),
            wk: rng.matrix(N_HEAD_KV * HEAD_DIM, N_EMBD),
            wv: rng.matrix(N_HEAD_KV * HEAD_DIM, N_EMBD),
            wo: rng.matrix(N_EMBD, N_HEAD * HEAD_DIM),
            q_norm: rng.norm(HEAD_DIM),
            k_norm: rng.norm(HEAD_DIM),
            ffn: Ffn::new(rng),
        }
    }

    fn add_tensors(&self, w: &mut GgufWriter, layer: usize) {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let e = N_EMBD as u64;
        let (q, kv) = ((N_HEAD * HEAD_DIM) as u64, (N_HEAD_KV * HEAD_DIM) as u64);
        w.add_tensor(
            name("attn_norm"),
            GgmlType::F32,
            &[e],
            f32_bytes(&self.attn_norm),
        )
        .add_tensor(name("attn_q"), GgmlType::F32, &[e, q], f32_bytes(&self.wq))
        .add_tensor(name("attn_k"), GgmlType::F32, &[e, kv], f32_bytes(&self.wk))
        .add_tensor(name("attn_v"), GgmlType::F32, &[e, kv], f32_bytes(&self.wv))
        .add_tensor(
            name("attn_output"),
            GgmlType::F32,
            &[q, e],
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

    /// Ungated grouped-query attention at one position, appended to `cache`,
    /// then the block's feed-forward.
    fn apply(&self, x: &mut [f64], pos: usize, cache: &mut Cache) {
        let h = rmsnorm(x, &self.attn_norm, EPS);
        let q = matvec(&self.wq, &h, N_HEAD * HEAD_DIM);
        let k = matvec(&self.wk, &h, N_HEAD_KV * HEAD_DIM);
        let v = matvec(&self.wv, &h, N_HEAD_KV * HEAD_DIM);
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
            let mut head = rmsnorm(&k[hk * HEAD_DIM..(hk + 1) * HEAD_DIM], &self.k_norm, EPS);
            rope(&mut head);
            key.extend(head);
        }
        cache.keys.push(key);
        cache.values.push(v);
        let mut att = vec![0.0; N_HEAD * HEAD_DIM];
        for hq in 0..N_HEAD {
            let mut qh = rmsnorm(&q[hq * HEAD_DIM..(hq + 1) * HEAD_DIM], &self.q_norm, EPS);
            rope(&mut qh);
            let off = (hq / (N_HEAD / N_HEAD_KV)) * HEAD_DIM;
            let scores: Vec<f64> = cache
                .keys
                .iter()
                .map(|key| {
                    qh.iter()
                        .zip(&key[off..off + HEAD_DIM])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        / (HEAD_DIM as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - max).exp()).collect();
            let total: f64 = exps.iter().sum();
            for (d, &e) in exps.iter().enumerate() {
                let weight = e / total;
                for (o, &v) in att[hq * HEAD_DIM..(hq + 1) * HEAD_DIM]
                    .iter_mut()
                    .zip(&cache.values[d][off..off + HEAD_DIM])
                {
                    *o += weight * v;
                }
            }
        }
        add(x, &matvec(&self.wo, &att, N_EMBD));
    }
}

/// An attention layer's cached keys and values, by position.
#[derive(Default)]
struct Cache {
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
}

enum Block {
    Conv(Conv),
    Attn(Attn),
}

struct Weights {
    vocab: Vec<String>,
    embd: Vec<f32>,
    output: Vec<f32>,
    token_embd_norm: Vec<f32>,
    blocks: Vec<Block>,
    moe: bool,
}

impl Weights {
    /// Four blocks: shortconv, attention, shortconv, attention.
    fn new() -> Self {
        let mut rng = Rng(0x0F12_A5E1);
        let vocab = vocab();
        let blocks = vec![
            Block::Conv(Conv::new(&mut rng)),
            Block::Attn(Attn::new(&mut rng)),
            Block::Conv(Conv::new(&mut rng)),
            Block::Attn(Attn::new(&mut rng)),
        ];
        Self {
            embd: rng.matrix(vocab.len(), N_EMBD),
            output: rng.matrix(vocab.len(), N_EMBD),
            token_embd_norm: rng.norm(N_EMBD),
            vocab,
            blocks,
            moe: false,
        }
    }

    /// The same model with sigmoid-gated mixture-of-experts feed-forwards,
    /// the first block dense-led (llama.cpp's `leading_dense_block_count`).
    fn moe() -> Self {
        let mut w = Self::new();
        w.moe = true;
        let mut rng = Rng(0x00E5_9E27);
        for (i, block) in w.blocks.iter_mut().enumerate() {
            let ffn = match block {
                Block::Conv(c) => &mut c.ffn,
                Block::Attn(a) => &mut a.ffn,
            };
            if i >= 1 {
                ffn.moe = Some(Moe::new(&mut rng));
            }
        }
        w
    }
}

impl Weights {
    fn gguf(&self) -> Vec<u8> {
        let e = N_EMBD as u64;
        let mut w = GgufWriter::new();
        let arch = if self.moe { "lfm2moe" } else { "lfm2" };
        let key = |k: &str| format!("{arch}.{k}");
        // Blocks 0 and 2 are shortconv ones, marked by a zero key/value
        // head count, as the converter writes them.
        let kv_heads: Vec<i32> = self
            .blocks
            .iter()
            .map(|b| match b {
                Block::Conv(_) => 0,
                Block::Attn(_) => N_HEAD_KV as i32,
            })
            .collect();
        w.add("general.architecture", arch)
            .add(key("block_count"), self.blocks.len() as u32)
            .add(key("context_length"), N_CTX as u32)
            .add(key("embedding_length"), N_EMBD as u32)
            .add(key("attention.head_count"), N_HEAD as u32)
            .add(
                key("attention.head_count_kv"),
                OwnedValue::array(ValueType::I32, kv_heads),
            )
            .add(key("attention.key_length"), HEAD_DIM as u32)
            .add(key("attention.value_length"), HEAD_DIM as u32)
            .add(key("attention.layer_norm_rms_epsilon"), EPS)
            .add(key("rope.freq_base"), ROPE_BASE)
            .add(key("rope.dimension_count"), ROPE_DIMS as u32)
            .add(key("shortconv.l_cache"), TAPS as u32)
            // The dense blocks' size, which `lfm2moe` files carry too.
            .add(key("feed_forward_length"), N_FF as u32);
        if self.moe {
            w.add(key("expert_count"), N_EXPERT as u32)
                .add(key("expert_used_count"), N_USED as u32)
                .add(key("expert_feed_forward_length"), N_FF_EXP as u32)
                .add(key("expert_gating_func"), "sigmoid")
                .add(key("leading_dense_block_count"), 1u32);
        }
        add_tokenizer(&mut w, &self.vocab, "lfm2");
        let v = self.vocab.len() as u64;
        w.add_tensor(
            "token_embd.weight",
            GgmlType::F32,
            &[e, v],
            f32_bytes(&self.embd),
        )
        .add_tensor(
            "output.weight",
            GgmlType::F32,
            &[e, v],
            f32_bytes(&self.output),
        )
        .add_tensor(
            "token_embd_norm.weight",
            GgmlType::F32,
            &[e],
            f32_bytes(&self.token_embd_norm),
        );
        for (i, block) in self.blocks.iter().enumerate() {
            match block {
                Block::Conv(c) => c.add_tensors(&mut w, i),
                Block::Attn(a) => a.add_tensors(&mut w, i),
            }
        }
        w.to_bytes()
    }

    fn model(&self) -> Model {
        Model::from_storage(Storage::from_bytes(&self.gguf())).unwrap()
    }

    /// Next-token logits after `tokens`, computed naively in f64 with
    /// explicit state.
    fn reference_logits(&self, tokens: &[u32]) -> Vec<f64> {
        // The conv blocks' histories and the attention blocks' caches, in
        // block order; the conv histories start at zero, as llama.cpp's do.
        let mut convs: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut caches: Vec<Cache> = Vec::new();
        for block in &self.blocks {
            match block {
                Block::Conv(_) => convs.push(vec![vec![0.0; N_EMBD]; ROWS]),
                Block::Attn(_) => caches.push(Cache::default()),
            }
        }
        let mut x = vec![0.0; N_EMBD];
        for (pos, &token) in tokens.iter().enumerate() {
            for (o, &v) in x.iter_mut().zip(&self.embd[token as usize * N_EMBD..]) {
                *o = f64::from(v);
            }
            let (mut ci, mut ai) = (0, 0);
            for block in &self.blocks {
                match block {
                    Block::Conv(c) => {
                        c.apply(&mut x, &mut convs[ci]);
                        c.ffn.apply(&mut x);
                        ci += 1;
                    }
                    Block::Attn(a) => {
                        a.apply(&mut x, pos, &mut caches[ai]);
                        a.ffn.apply(&mut x);
                        ai += 1;
                    }
                }
            }
        }
        let xn = rmsnorm(&x, &self.token_embd_norm, EPS);
        matvec(&self.output, &xn, self.vocab.len())
    }
}

fn session(model: &Model) -> nibble::Session<'_> {
    let options = SessionOptions {
        n_ctx: N_CTX,
        threads: 2,
        ..Default::default()
    };
    model.session(options).unwrap()
}

/// A tokenizer over [`vocab`] plus `xyz`, a token no merge reaches, with
/// the `pre` pre-tokenizer and no `tokenizer.ggml.add_bos_token`.
fn tokenizer(pre: &str) -> Tokenizer {
    let mut vocab = vocab();
    vocab.push("xyz".into());
    let mut w = GgufWriter::new();
    w.add("tokenizer.ggml.model", "gpt2")
        .add("tokenizer.ggml.pre", pre)
        .add(
            "tokenizer.ggml.tokens",
            OwnedValue::array(ValueType::String, vocab.iter().map(String::as_str)),
        )
        .add(
            "tokenizer.ggml.merges",
            OwnedValue::array(ValueType::String, ["a b", "ab c"]),
        )
        .add("tokenizer.ggml.bos_token_id", 258u32);
    let storage = Storage::from_bytes(&w.to_bytes());
    Tokenizer::from_gguf(&Gguf::parse(&storage).unwrap()).unwrap()
}

/// llama.cpp's `lfm2` (and `llama3`) pre-tokenizer starts prompts with BOS
/// when the file does not say, and looks a piece up whole before merging
/// (`ignore_merges`); `qwen2` does neither.
#[test]
fn lfm2_pre_tokenizer_adds_bos_and_ignores_merges() {
    let xyz = vocab().len() as u32;
    for (pre, bos, ids) in [
        ("lfm2", true, vec![xyz]),
        ("llama3", true, vec![xyz]),
        (
            "qwen2",
            false,
            b"xyz".iter().map(|&b| u32::from(b)).collect(),
        ),
    ] {
        let tok = tokenizer(pre);
        assert_eq!(tok.add_bos(), bos, "{pre}");
        // " abc" is no token: its bytes merge to the space and `abc`.
        let mut want = ids;
        want.extend([u32::from(b' '), 257]);
        assert_eq!(tok.encode("xyz abc", false), want, "{pre}");
    }
}

#[test]
fn loads_config_and_its_architecture() {
    let w = Weights::new();
    let model = w.model();
    assert_eq!(model.config().architecture, Architecture::Lfm2);
    assert_eq!(model.config().n_layer, 4);
    assert!(!model.has_mtp());
    let w = Weights::moe();
    assert_eq!(w.model().config().architecture, Architecture::Lfm2Moe);
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

/// A rewind into a state that cannot rewind (the conv blocks) replays from
/// the checkpoint; the logits still match a fresh session's.
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
fn gpu_matches_a_naive_reference_and_rewinds() {
    let w = Weights::new();
    let model = w.model();
    for backend in [nibble::BackendKind::Metal, nibble::BackendKind::Cuda] {
        let options = nibble::SessionOptions {
            n_ctx: N_CTX,
            backend,
            ..Default::default()
        };
        let Some(mut s) = common::session_or_skip(&model, options) else {
            continue;
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
}

/// Batches give every token the logits it gets alone, on the CPU and on
/// the GPUs.
#[test]
fn batch_size_does_not_change_the_logits() {
    let model = Weights::new().model();
    let n_vocab = model.config().n_vocab as u32;
    let tokens: Vec<u32> = (0..23u32).map(|i| (i * 37 + 11) % n_vocab).collect();
    for backend in [
        nibble::BackendKind::Cpu,
        nibble::BackendKind::Metal,
        nibble::BackendKind::Cuda,
    ] {
        common::assert_batches_match_steps(&model, backend, N_CTX, &tokens, 9);
    }
}

/// The mixture-of-experts model matches its reference, on the CPU and the
/// GPUs, one token at a time and batched.
#[test]
fn moe_matches_a_naive_reference() {
    let w = Weights::moe();
    let model = w.model();
    let tokens = [17u32, 99, 258, 3, 200];
    let mut s = session(&model);
    assert_close(s.eval(&tokens).unwrap(), &w.reference_logits(&tokens), 3e-3);
    for backend in [nibble::BackendKind::Metal, nibble::BackendKind::Cuda] {
        let options = nibble::SessionOptions {
            n_ctx: N_CTX,
            backend,
            ..Default::default()
        };
        let Some(mut s) = common::session_or_skip(&model, options) else {
            continue;
        };
        assert_close(s.eval(&tokens).unwrap(), &w.reference_logits(&tokens), 3e-3);
    }
}
