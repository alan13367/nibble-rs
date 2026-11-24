//! Building blocks shared by decoder architectures: the embedding/output
//! head, grouped-query attention with its KV cache, and the SwiGLU feed-forward.
//!
//! An architecture module composes these and adds only what is specific to it.

use nibble_gguf::Gguf;

use super::bind::{matrix, vector};
use super::{Ctx, ModelConfig};
use crate::cpu::{self, EncodedInput, Kernels, Weight};
use crate::error::Result;

/// Token embeddings, final norm and output projection.
pub(crate) struct Head {
    token_embd: Weight,
    output: Weight,
    output_norm: Vec<f32>,
}

impl Head {
    /// Binds `token_embd`, `output_norm` and `output` (or the token embeddings
    /// when the output projection is tied).
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig) -> Result<Self> {
        let token_embd = matrix(g, "token_embd.weight", c.n_embd, c.n_vocab)?;
        let output = if c.tied_embeddings {
            token_embd
        } else {
            matrix(g, "output.weight", c.n_embd, c.n_vocab)?
        };
        Ok(Self {
            token_embd,
            output,
            output_norm: vector(g, "output_norm.weight", c.n_embd)?,
        })
    }

    /// Writes the embedding of `token` to `x`.
    pub(crate) fn embed(&self, ctx: &Ctx<'_>, token: u32, x: &mut [f32]) {
        self.token_embd.dequantize_row(ctx.bytes, token as usize, x);
    }

    /// `logits = output · (rmsnorm(x) * output_norm)`.
    pub(crate) fn logits(&self, ctx: &Ctx<'_>, x: &[f32], s: &mut Scratch, logits: &mut [f32]) {
        cpu::rmsnorm(&mut s.xn, x, &self.output_norm, ctx.config.rms_eps);
        ctx.cpu
            .matvec(ctx.bytes, &self.output, &s.xn, logits, &mut s.input);
    }
}

/// Buffers reused by every block within one forward pass.
pub(crate) struct Scratch {
    /// A normalized copy of the hidden state.
    pub(crate) xn: Vec<f32>,
    /// A block's output, before it is added to the residual stream.
    pub(crate) out: Vec<f32>,
    /// The input of the current matrix products, encoded for their weights.
    pub(crate) input: EncodedInput,
    /// `(cos, sin)` per rotated pair for the current position.
    pub(crate) rope: Vec<(f32, f32)>,
}

impl Scratch {
    pub(crate) fn new(c: &ModelConfig) -> Self {
        Self {
            xn: vec![0.0; c.n_embd],
            out: vec![0.0; c.n_embd],
            input: EncodedInput::new(),
            rope: Vec::with_capacity(c.rope_dims / 2),
        }
    }
}

/// Positions are processed in blocks of this many: the step of ggml's f16
/// dot product, which llama.cpp's padding of the cache (to 256 positions)
/// always fills.
const KV_BLOCK: usize = 32;

/// Keys and values of one attention layer for up to `capacity` positions, in
/// f16 (llama.cpp's default cache type), laid out as llama.cpp lays them out
/// without flash attention: keys by position, values transposed. Both
/// attention products are then contiguous f16 dot products.
///
/// The buffers are allocated for the whole context up front, as llama.cpp
/// does, so the forward pass never reallocates them. The OS backs their
/// zeroed pages with memory only when they are written, though Windows
/// counts the whole allocation against its commit limit. Writing the first
/// position touches one page per value row.
pub(crate) struct KvCache {
    /// `capacity` rows of `kv_dim` keys.
    k: Vec<u8>,
    /// `kv_dim` rows of `stride` values: row `i` holds element `i` of every
    /// position's value. Entries past `len` are zero.
    vt: Vec<u8>,
    /// The value being written, as f16, before it is scattered into `vt`.
    staging: Vec<u8>,
    head_dim: usize,
    kv_dim: usize,
    capacity: usize,
    /// `capacity` rounded up to a whole block.
    stride: usize,
    len: usize,
}

impl KvCache {
    pub(crate) fn new(c: &ModelConfig, capacity: usize) -> Self {
        let kv_dim = c.n_head_kv * c.head_dim;
        let stride = capacity.next_multiple_of(KV_BLOCK);
        Self {
            k: vec![0; 2 * kv_dim * capacity],
            vt: vec![0; 2 * kv_dim * stride],
            staging: vec![0; 2 * kv_dim],
            head_dim: c.head_dim,
            kv_dim,
            capacity,
            stride,
            len: 0,
        }
    }

    /// Number of cached positions.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Appends one position.
    ///
    /// # Panics
    ///
    /// If the cache is full; sessions check the context size first.
    pub(crate) fn push(&mut self, kernels: Kernels, k: &[f32], v: &[f32]) {
        let t = self.len;
        assert!(
            t < self.capacity,
            "KV cache of {} positions is full",
            self.capacity
        );
        let row = 2 * self.kv_dim;
        kernels.to_f16(k, &mut self.k[t * row..(t + 1) * row]);
        kernels.to_f16(v, &mut self.staging);
        for (i, value) in self.staging.as_chunks::<2>().0.iter().enumerate() {
            let at = 2 * (i * self.stride + t);
            self.vt[at..at + 2].copy_from_slice(value);
        }
        self.len += 1;
    }

    /// Keeps the first `n_pos` positions.
    pub(crate) fn truncate(&mut self, n_pos: usize) {
        if n_pos >= self.len {
            return;
        }
        // The value product reads the dropped entries of the last block,
        // weighted by zero, so they must not hold an infinity.
        for row in self.vt.chunks_exact_mut(2 * self.stride) {
            row[2 * n_pos..2 * self.len].fill(0);
        }
        self.len = n_pos;
    }

    /// The keys of KV head `g` at position `t`.
    fn key(&self, t: usize, g: usize) -> &[u8] {
        let start = t * self.kv_dim + g * self.head_dim;
        &self.k[2 * start..2 * (start + self.head_dim)]
    }

    /// Element `i` of the value of every position, through the end of the
    /// block that holds the last position.
    fn values(&self, i: usize) -> &[u8] {
        let start = i * self.stride;
        &self.vt[2 * start..2 * (start + self.len.next_multiple_of(KV_BLOCK))]
    }
}

/// Per-head buffers for [`attend`].
struct HeadScratch {
    /// Attention scores, then probabilities, for every position.
    scores: Vec<f32>,
    /// The probabilities as f16, zero-padded to a whole block.
    probs: Vec<u8>,
}

/// Buffers for [`attend`].
pub(crate) struct AttentionScratch {
    /// The queries as f16.
    q16: Vec<u8>,
    heads: Vec<HeadScratch>,
}

impl AttentionScratch {
    pub(crate) fn new(c: &ModelConfig, n_ctx: usize) -> Self {
        let padded = n_ctx.next_multiple_of(KV_BLOCK);
        Self {
            q16: vec![0; 2 * c.n_head * c.head_dim],
            heads: (0..c.n_head)
                .map(|_| HeadScratch {
                    scores: vec![0.0; n_ctx],
                    probs: vec![0; 2 * padded],
                })
                .collect(),
        }
    }
}

/// Causal softmax attention of the query heads `q` over every cached position,
/// with grouped-query sharing (query head `h` reads KV head `h / (n_head / n_head_kv)`).
/// Writes `n_head * head_dim` values to `out`.
///
/// As llama.cpp without flash attention: queries meet the f16 keys in f16,
/// the softmax probabilities are rounded to f16, and each output value is
/// the f16 dot product of the probabilities with a row of transposed values.
/// Heads run in parallel.
pub(crate) fn attend(
    ctx: &Ctx<'_>,
    q: &[f32],
    cache: &KvCache,
    s: &mut AttentionScratch,
    out: &mut [f32],
) {
    let c = ctx.config;
    let hd = c.head_dim;
    let group = c.n_head / c.n_head_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let kernels = ctx.cpu.kernels();
    kernels.to_f16(q, &mut s.q16);
    let q16 = &s.q16;
    let n_pos = cache.len();
    let padded = n_pos.next_multiple_of(KV_BLOCK);
    ctx.cpu
        .pool()
        .for_each_chunk_pair(out, hd, &mut s.heads, 1, |h, out, head| {
            let head = &mut head[0];
            let kv_head = h / group;
            let q = &q16[2 * h * hd..2 * (h + 1) * hd];
            let scores = &mut head.scores[..n_pos];
            for (t, score) in scores.iter_mut().enumerate() {
                *score = kernels.dot_f16(q, cache.key(t, kv_head)) * scale;
            }
            kernels.softmax(scores);
            let probs = &mut head.probs[..2 * padded];
            let (p, padding) = probs.split_at_mut(2 * n_pos);
            kernels.to_f16(scores, p);
            padding.fill(0);
            for (i, o) in out.iter_mut().enumerate() {
                *o = kernels.dot_f16(probs, cache.values(kv_head * hd + i));
            }
        });
}

/// Standard grouped-query attention, as in Llama, Mistral and Qwen: Q/K/V and
/// output projections, optional per-head Q/K RMSNorm, NeoX RoPE.
pub(crate) struct Attention {
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
}

/// Buffers for [`Attention::forward`].
pub(crate) struct AttentionBuffers {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    att: Vec<f32>,
    core: AttentionScratch,
}

impl AttentionBuffers {
    pub(crate) fn new(c: &ModelConfig, n_ctx: usize) -> Self {
        let q_dim = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        Self {
            q: vec![0.0; q_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],
            att: vec![0.0; q_dim],
            core: AttentionScratch::new(c, n_ctx),
        }
    }
}

impl Attention {
    /// Binds `blk.{layer}.attn_{q,k,v,output}` and, when `qk_norm` is set,
    /// `attn_q_norm` / `attn_k_norm`.
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, layer: usize, qk_norm: bool) -> Result<Self> {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let q_dim = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        let norm = |n: &str| qk_norm.then(|| vector(g, &name(n), c.head_dim)).transpose();
        Ok(Self {
            wq: matrix(g, &name("attn_q"), c.n_embd, q_dim)?,
            wk: matrix(g, &name("attn_k"), c.n_embd, kv_dim)?,
            wv: matrix(g, &name("attn_v"), c.n_embd, kv_dim)?,
            wo: matrix(g, &name("attn_output"), q_dim, c.n_embd)?,
            q_norm: norm("attn_q_norm")?,
            k_norm: norm("attn_k_norm")?,
        })
    }

    /// Attention over `s.xn` (the normalized hidden state) at the cache's next
    /// position. Appends this position to `cache` and writes the projected
    /// output to `s.out`.
    pub(crate) fn forward(
        &self,
        ctx: &Ctx<'_>,
        cache: &mut KvCache,
        b: &mut AttentionBuffers,
        s: &mut Scratch,
    ) {
        let c = ctx.config;
        ctx.cpu.matvecs(
            ctx.bytes,
            &s.xn,
            &mut s.input,
            [
                (&self.wq, &mut b.q[..]),
                (&self.wk, &mut b.k[..]),
                (&self.wv, &mut b.v[..]),
            ],
        );
        if let Some(w) = &self.q_norm {
            cpu::rmsnorm_chunks(&mut b.q, w, c.rms_eps);
        }
        if let Some(w) = &self.k_norm {
            cpu::rmsnorm_chunks(&mut b.k, w, c.rms_eps);
        }
        cpu::rope_neox(&mut b.q, c.head_dim, &s.rope);
        cpu::rope_neox(&mut b.k, c.head_dim, &s.rope);
        cache.push(ctx.cpu.kernels(), &b.k, &b.v);
        attend(ctx, &b.q, cache, &mut b.core, &mut b.att);
        ctx.cpu
            .matvec(ctx.bytes, &self.wo, &b.att, &mut s.out, &mut s.input);
    }
}

/// SwiGLU feed-forward: `ffn_down · (silu(ffn_gate · x) * (ffn_up · x))`.
pub(crate) struct SwiGlu {
    gate: Weight,
    up: Weight,
    down: Weight,
}

/// Buffers for [`SwiGlu::forward`].
pub(crate) struct FfnBuffers {
    /// `silu(gate · x) * (up · x)`.
    hidden: Vec<f32>,
    up: Vec<f32>,
}

impl FfnBuffers {
    pub(crate) fn new(c: &ModelConfig) -> Self {
        Self {
            hidden: vec![0.0; c.n_ff],
            up: vec![0.0; c.n_ff],
        }
    }
}

impl SwiGlu {
    /// Binds `blk.{layer}.ffn_{gate,up,down}`.
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, layer: usize) -> Result<Self> {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        Ok(Self {
            gate: matrix(g, &name("ffn_gate"), c.n_embd, c.n_ff)?,
            up: matrix(g, &name("ffn_up"), c.n_embd, c.n_ff)?,
            down: matrix(g, &name("ffn_down"), c.n_ff, c.n_embd)?,
        })
    }

    /// Feed-forward of `s.xn`, written to `s.out`.
    pub(crate) fn forward(&self, ctx: &Ctx<'_>, b: &mut FfnBuffers, s: &mut Scratch) {
        ctx.cpu.swiglu(
            ctx.bytes,
            &self.gate,
            &self.up,
            &s.xn,
            &mut s.input,
            &mut b.hidden,
            &mut b.up,
        );
        ctx.cpu
            .matvec(ctx.bytes, &self.down, &b.hidden, &mut s.out, &mut s.input);
    }
}
