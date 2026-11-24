//! Building blocks shared by decoder architectures: the embedding/output
//! head, grouped-query attention with its KV cache, and the SwiGLU feed-forward.
//!
//! An architecture module composes these and adds only what is specific to it.

use nibble_gguf::Gguf;

use super::bind::{matrix, vector};
use super::{Ctx, ModelConfig};
use crate::cpu::{self, Weight, matvec};
use crate::error::Result;
use crate::quant::f32_to_f16;

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
        matvec(
            ctx.bytes,
            &self.output,
            &s.xn,
            logits,
            &mut s.xq,
            ctx.threads,
        );
    }
}

/// Buffers reused by every block within one forward pass.
pub(crate) struct Scratch {
    /// A normalized copy of the hidden state.
    pub(crate) xn: Vec<f32>,
    /// A block's output, before it is added to the residual stream.
    pub(crate) out: Vec<f32>,
    /// The activation encoded for the current matrix (see `quant::quantize_activation`).
    pub(crate) xq: Vec<u8>,
    /// `(cos, sin)` per rotated pair for the current position.
    pub(crate) rope: Vec<(f32, f32)>,
}

impl Scratch {
    pub(crate) fn new(c: &ModelConfig) -> Self {
        Self {
            xn: vec![0.0; c.n_embd],
            out: vec![0.0; c.n_embd],
            xq: Vec::new(),
            rope: Vec::with_capacity(c.rope_dims / 2),
        }
    }
}

/// Keys and values of one attention layer for every position so far, in f16
/// (llama.cpp's default cache type).
pub(crate) struct KvCache {
    k: Vec<u16>,
    v: Vec<u16>,
    kv_dim: usize,
}

impl KvCache {
    pub(crate) fn new(c: &ModelConfig) -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            kv_dim: c.n_head_kv * c.head_dim,
        }
    }

    /// Number of cached positions.
    pub(crate) fn len(&self) -> usize {
        self.k.len() / self.kv_dim
    }

    /// Appends one position.
    pub(crate) fn push(&mut self, k: &[f32], v: &[f32]) {
        self.k.extend(k.iter().map(|&x| f32_to_f16(x)));
        self.v.extend(v.iter().map(|&x| f32_to_f16(x)));
    }

    /// Keeps the first `n_pos` positions.
    pub(crate) fn truncate(&mut self, n_pos: usize) {
        self.k.truncate(n_pos * self.kv_dim);
        self.v.truncate(n_pos * self.kv_dim);
    }
}

/// Buffers for [`attend`].
pub(crate) struct AttentionScratch {
    q16: Vec<u16>,
    scores: Vec<f32>,
}

impl AttentionScratch {
    pub(crate) fn new(c: &ModelConfig, n_ctx: usize) -> Self {
        Self {
            q16: vec![0; c.n_head * c.head_dim],
            scores: vec![0.0; n_ctx],
        }
    }
}

/// Causal softmax attention of the query heads `q` over every cached position,
/// with grouped-query sharing (query head `h` reads KV head `h / (n_head / n_head_kv)`).
/// Writes `n_head * head_dim` values to `out`.
pub(crate) fn attend(
    c: &ModelConfig,
    q: &[f32],
    cache: &KvCache,
    s: &mut AttentionScratch,
    out: &mut [f32],
) {
    let hd = c.head_dim;
    let kv_dim = cache.kv_dim;
    let group = c.n_head / c.n_head_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let n_pos = cache.len();
    // Queries are compared with the f16 keys in f16, as ggml does.
    for (h, &x) in s.q16.iter_mut().zip(q) {
        *h = f32_to_f16(x);
    }
    for h in 0..c.n_head {
        let kv_off = (h / group) * hd;
        let q = &s.q16[h * hd..(h + 1) * hd];
        let scores = &mut s.scores[..n_pos];
        for (t, score) in scores.iter_mut().enumerate() {
            *score = cpu::dot_f16(q, &cache.k[t * kv_dim + kv_off..][..hd]) * scale;
        }
        cpu::softmax(scores);
        let out = &mut out[h * hd..(h + 1) * hd];
        out.fill(0.0);
        for (t, &p) in scores.iter().enumerate() {
            cpu::axpy_f16(out, p, &cache.v[t * kv_dim + kv_off..][..hd]);
        }
    }
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

    /// Attention over `xn` (the normalized hidden state) at the cache's next
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
        matvec(ctx.bytes, &self.wq, &s.xn, &mut b.q, &mut s.xq, ctx.threads);
        matvec(ctx.bytes, &self.wk, &s.xn, &mut b.k, &mut s.xq, ctx.threads);
        matvec(ctx.bytes, &self.wv, &s.xn, &mut b.v, &mut s.xq, ctx.threads);
        if let Some(w) = &self.q_norm {
            cpu::rmsnorm_chunks(&mut b.q, w, c.rms_eps);
        }
        if let Some(w) = &self.k_norm {
            cpu::rmsnorm_chunks(&mut b.k, w, c.rms_eps);
        }
        cpu::rope_neox(&mut b.q, c.head_dim, &s.rope);
        cpu::rope_neox(&mut b.k, c.head_dim, &s.rope);
        cache.push(&b.k, &b.v);
        attend(c, &b.q, cache, &mut b.core, &mut b.att);
        matvec(
            ctx.bytes,
            &self.wo,
            &b.att,
            &mut s.out,
            &mut s.xq,
            ctx.threads,
        );
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
    gate: Vec<f32>,
    up: Vec<f32>,
}

impl FfnBuffers {
    pub(crate) fn new(c: &ModelConfig) -> Self {
        Self {
            gate: vec![0.0; c.n_ff],
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
        matvec(
            ctx.bytes,
            &self.gate,
            &s.xn,
            &mut b.gate,
            &mut s.xq,
            ctx.threads,
        );
        matvec(
            ctx.bytes,
            &self.up,
            &s.xn,
            &mut b.up,
            &mut s.xq,
            ctx.threads,
        );
        for (g, &u) in b.gate.iter_mut().zip(&b.up) {
            *g = cpu::silu(*g) * u;
        }
        matvec(
            ctx.bytes,
            &self.down,
            &b.gate,
            &mut s.out,
            &mut s.xq,
            ctx.threads,
        );
    }
}
