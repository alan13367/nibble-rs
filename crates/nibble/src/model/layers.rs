//! Building blocks shared by decoder architectures: the embedding/output
//! head, grouped-query attention, and the SwiGLU feed-forward.
//!
//! An architecture module composes these and adds only what is specific to it.

use nibble_gguf::Gguf;

use super::bind::{matrix, vector};
use super::{Logits, ModelConfig};
use crate::backend::{Backend, Param, Params, QkNorm};
use crate::cpu::Weight;
use crate::error::Result;

/// Token embeddings, final norm and output projection.
pub(crate) struct Head {
    token_embd: Weight,
    output: Weight,
    pub(crate) output_norm: Param,
}

impl Head {
    /// Binds `token_embd`, `output_norm` and `output` (or the token embeddings
    /// when the output projection is tied).
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, p: &mut Params) -> Result<Self> {
        let token_embd = matrix(g, "token_embd.weight", c.n_embd, c.n_vocab)?;
        let output = if c.tied_embeddings {
            token_embd
        } else {
            matrix(g, "output.weight", c.n_embd, c.n_vocab)?
        };
        Ok(Self {
            token_embd,
            output,
            output_norm: vector(g, p, "output_norm.weight", c.n_embd)?,
        })
    }

    /// The token embeddings.
    pub(crate) const fn token_embd(&self) -> Weight {
        self.token_embd
    }

    /// The output projection (the token embeddings when tied).
    pub(crate) const fn output(&self) -> Weight {
        self.output
    }

    /// Writes the embedding of each token to its row of `x`.
    pub(crate) fn embed<B: Backend>(&self, b: &mut B, tokens: &[u32], x: &mut B::Vector) {
        b.embed(&self.token_embd, tokens, x);
    }

    /// Ends the batch with `logits = output · xn` for the rows asked for,
    /// where `xn` is the hidden state normalized with `output_norm`, or
    /// without logits.
    pub(crate) fn finish<B: Backend>(
        &self,
        b: &mut B,
        xn: &B::Vector,
        logits: Option<Logits<'_>>,
    ) -> Result<()> {
        match logits {
            Some(l) => b.logits(&self.output, xn, l.rows, l.out),
            None => b.end(),
        }
    }
}

/// Buffers reused by every block within one forward pass, one row per token
/// of a batch.
pub(crate) struct Scratch<B: Backend> {
    /// The residual stream.
    pub(crate) x: B::Vector,
    /// The hidden state normalized for the next block.
    pub(crate) xn: B::Vector,
}

/// The norm of the block after this one (or the output norm), which a
/// block applies to the hidden state as it adds its output.
#[derive(Clone, Copy)]
pub(crate) struct Next {
    pub(crate) norm: Param,
    pub(crate) eps: f32,
}

impl<B: Backend> Scratch<B> {
    /// `x += w · a`, then `xn = rmsnorm(x)` with the next block's norm.
    pub(crate) fn add(&mut self, b: &mut B, w: &Weight, a: &B::Vector, next: Next) {
        b.residual(w, a, &mut self.x, next.norm, next.eps, &mut self.xn);
    }
}

impl<B: Backend> Scratch<B> {
    pub(crate) fn new(b: &mut B, c: &ModelConfig, n_batch: usize) -> Result<Self> {
        Ok(Self {
            x: b.vector(n_batch * c.n_embd)?,
            xn: b.vector(n_batch * c.n_embd)?,
        })
    }
}

/// Standard grouped-query attention, as in Llama, Mistral and Qwen: Q/K/V and
/// output projections, optional per-head Q/K RMSNorm, NeoX RoPE.
pub(crate) struct Attention {
    wq: Weight,
    wk: Weight,
    wv: Weight,
    wo: Weight,
    q_norm: Option<Param>,
    k_norm: Option<Param>,
}

/// Buffers for [`Attention::forward`].
pub(crate) struct AttentionBuffers<B: Backend> {
    q: B::Vector,
    k: B::Vector,
    v: B::Vector,
    att: B::Vector,
}

impl<B: Backend> AttentionBuffers<B> {
    pub(crate) fn new(b: &mut B, c: &ModelConfig, n_batch: usize) -> Result<Self> {
        let q_dim = n_batch * c.n_head * c.head_dim;
        let kv_dim = n_batch * c.n_head_kv * c.head_dim;
        Ok(Self {
            q: b.vector(q_dim)?,
            k: b.vector(kv_dim)?,
            v: b.vector(kv_dim)?,
            att: b.vector(q_dim)?,
        })
    }
}

impl Attention {
    /// Binds `blk.{layer}.attn_{q,k,v,output}` and, when `qk_norm` is set,
    /// `attn_q_norm` / `attn_k_norm`.
    pub(crate) fn load(
        g: &Gguf<'_>,
        c: &ModelConfig,
        p: &mut Params,
        layer: usize,
        qk_norm: bool,
    ) -> Result<Self> {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let q_dim = c.n_head * c.head_dim;
        let kv_dim = c.n_head_kv * c.head_dim;
        let mut norm = |n: &str| {
            qk_norm
                .then(|| vector(g, p, &name(n), c.head_dim))
                .transpose()
        };
        let q_norm = norm("attn_q_norm")?;
        let k_norm = norm("attn_k_norm")?;
        Ok(Self {
            wq: matrix(g, &name("attn_q"), c.n_embd, q_dim)?,
            wk: matrix(g, &name("attn_k"), c.n_embd, kv_dim)?,
            wv: matrix(g, &name("attn_v"), c.n_embd, kv_dim)?,
            wo: matrix(g, &name("attn_output"), q_dim, c.n_embd)?,
            q_norm,
            k_norm,
        })
    }

    /// Attention over `s.xn` (the normalized hidden state) for the batch's
    /// tokens, at the cache's next positions. Appends them to `cache` and
    /// adds the projected output to the hidden state, normalized with
    /// `next` into `s.xn`.
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        cache: &mut B::KvCache,
        a: &mut AttentionBuffers<B>,
        s: &mut Scratch<B>,
        next: Next,
    ) {
        b.matvecs(
            &s.xn,
            [
                (&self.wq, &mut a.q),
                (&self.wk, &mut a.k),
                (&self.wv, &mut a.v),
            ],
        );
        let qk_norm = QkNorm {
            weights: self.q_norm.zip(self.k_norm),
            eps: c.rms_eps,
            gated: false,
        };
        b.attention(&mut a.q, &mut a.k, &a.v, qk_norm, cache, &mut a.att);
        s.add(b, &self.wo, &a.att, next);
    }
}

/// SwiGLU feed-forward: `ffn_down · (silu(ffn_gate · x) * (ffn_up · x))`.
pub(crate) struct SwiGlu {
    gate: Weight,
    up: Weight,
    down: Weight,
}

/// Buffers for [`SwiGlu::forward`].
pub(crate) struct FfnBuffers<B: Backend> {
    /// `silu(gate · x) * (up · x)`.
    hidden: B::Vector,
    up: B::Vector,
}

impl<B: Backend> FfnBuffers<B> {
    pub(crate) fn new(b: &mut B, c: &ModelConfig, n_batch: usize) -> Result<Self> {
        Ok(Self {
            hidden: b.vector(n_batch * c.n_ff)?,
            up: b.vector(n_batch * c.n_ff)?,
        })
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

    /// Feed-forward of `s.xn`, added to the hidden state, which is then
    /// normalized with `next` into `s.xn`.
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        f: &mut FfnBuffers<B>,
        s: &mut Scratch<B>,
        next: Next,
    ) {
        b.swiglu(&self.gate, &self.up, &s.xn, &mut f.hidden, &mut f.up);
        s.add(b, &self.down, &f.hidden, next);
    }
}
