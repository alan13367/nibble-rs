//! Building blocks shared by decoder architectures: the embedding/output
//! head, grouped-query attention, and the SwiGLU and mixture-of-experts
//! feed-forwards.
//!
//! An architecture module composes these and adds only what is specific to it.

use nibble_gguf::Gguf;

use super::bind::{experts, matrix, vector};
use super::{Logits, ModelConfig};
use crate::backend::{Backend, Gating, Param, Params, QkNorm};
use crate::cpu::{Experts, Weight};
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
        Self::load_with_norm(g, c, p, "output_norm.weight")
    }

    /// As [`Head::load`], with the final norm's tensor name (LFM2 stores it
    /// as `token_embd_norm`).
    pub(crate) fn load_with_norm(
        g: &Gguf<'_>,
        c: &ModelConfig,
        p: &mut Params,
        norm: &str,
    ) -> Result<Self> {
        let token_embd = matrix(g, "token_embd.weight", c.n_embd, c.n_vocab)?;
        let output = if c.tied_embeddings {
            token_embd
        } else {
            matrix(g, "output.weight", c.n_embd, c.n_vocab)?
        };
        Ok(Self {
            token_embd,
            output,
            output_norm: vector(g, p, norm, c.n_embd)?,
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
            multi_rope: false,
        };
        b.attention(&mut a.q, &mut a.k, &a.v, qk_norm, cache, &mut a.att);
        s.add(b, &self.wo, &a.att, next);
    }
}

/// SwiGLU feed-forward: `ffn_down · (silu(ffn_gate · x) * (ffn_up · x))`.
pub(crate) struct SwiGlu {
    pub(super) gate: Weight,
    pub(super) up: Weight,
    pub(super) down: Weight,
}

/// Buffers for [`SwiGlu::forward`].
pub(crate) struct FfnBuffers<B: Backend> {
    /// `silu(gate · x) * (up · x)`.
    pub(super) hidden: B::Vector,
    pub(super) up: B::Vector,
    /// A mixture of experts' buffers, for models that have one.
    pub(super) moe: Option<MoeBuffers<B>>,
}

impl<B: Backend> FfnBuffers<B> {
    /// Buffers for dense feed-forwards and, with `moe`, for mixtures of
    /// experts of those sizes.
    pub(crate) fn new(
        b: &mut B,
        c: &ModelConfig,
        moe: Option<MoeDims>,
        n_batch: usize,
    ) -> Result<Self> {
        if moe.is_some() {
            b.check_moe()?;
        }
        Ok(Self {
            hidden: b.vector(n_batch * c.n_ff)?,
            up: b.vector(n_batch * c.n_ff)?,
            moe: moe.map(|d| MoeBuffers::new(b, c, d, n_batch)).transpose()?,
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

/// A feed-forward block: dense SwiGLU, or a mixture of experts.
pub(crate) enum Ffn {
    Dense(SwiGlu),
    Moe(Box<Moe>),
}

impl Ffn {
    /// Feed-forward of `s.xn`, added to the hidden state, which is then
    /// normalized with `next` into `s.xn`.
    ///
    /// # Panics
    ///
    /// For a mixture of experts, if `f` has no buffers for one (an
    /// internal invariant: the state is built for the model).
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        f: &mut FfnBuffers<B>,
        s: &mut Scratch<B>,
        next: Next,
    ) {
        match self {
            Self::Dense(ffn) => ffn.forward(b, f, s, next),
            Self::Moe(m) => {
                let buffers = f
                    .moe
                    .as_mut()
                    .expect("a mixture of experts without its buffers");
                m.forward(b, buffers, s, next);
            }
        }
    }
}

/// Mixture-of-experts feed-forward with a gated shared expert, as
/// llama.cpp's `build_moe_ffn` with softmax gating and normalized weights,
/// plus Qwen's shared expert:
///
/// ```text
/// p      = softmax(ffn_gate_inp · x)
/// e_1..k = the k most probable experts,  w_j = p[e_j] / Σ p[e_j]
/// moe    = Σ_j w_j · down_e · (silu(gate_e · x) * (up_e · x))
/// shared = sigmoid(ffn_gate_inp_shexp · x) · swiglu_shexp(x)
/// out    = moe + shared
/// ```
pub(crate) struct Moe {
    router: Weight,
    gate: Experts,
    up: Experts,
    down: Experts,
    /// Experts per token.
    k: usize,
    shared: SwiGlu,
    /// `ffn_gate_inp_shexp`, one row.
    shared_gate: Weight,
}

/// The sizes of a mixture-of-experts block.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeDims {
    pub(crate) n_expert: usize,
    /// Experts per token.
    pub(crate) k: usize,
    /// Hidden size of each expert.
    pub(crate) n_ff: usize,
    /// Hidden size of the shared expert.
    pub(crate) n_ff_shared: usize,
}

/// Buffers for [`Moe::forward`].
pub(crate) struct MoeBuffers<B: Backend> {
    logits: B::Vector,
    route: B::Vector,
    hidden: B::Vector,
    up: B::Vector,
    /// Each expert's output, before the weighted sum.
    products: B::Vector,
    out: B::Vector,
    shared_hidden: B::Vector,
    shared_up: B::Vector,
    shared_out: B::Vector,
    shared_gate: B::Vector,
}

impl<B: Backend> MoeBuffers<B> {
    pub(crate) fn new(b: &mut B, c: &ModelConfig, d: MoeDims, n_batch: usize) -> Result<Self> {
        let rows = |n: usize| n_batch * n;
        Ok(Self {
            logits: b.vector(rows(d.n_expert))?,
            route: b.vector(rows(2 * d.k))?,
            hidden: b.vector(rows(d.k * d.n_ff))?,
            up: b.vector(rows(d.k * d.n_ff))?,
            products: b.vector(rows(d.k * c.n_embd))?,
            out: b.vector(rows(c.n_embd))?,
            shared_hidden: b.vector(rows(d.n_ff_shared))?,
            shared_up: b.vector(rows(d.n_ff_shared))?,
            shared_out: b.vector(rows(c.n_embd))?,
            shared_gate: b.vector(rows(1))?,
        })
    }
}

impl Moe {
    /// Binds `blk.{layer}.ffn_{gate_inp,gate_exps,up_exps,down_exps}` and the
    /// shared expert `ffn_{gate,up,down}_shexp` with its gate `ffn_gate_inp_shexp`.
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, d: MoeDims, layer: usize) -> Result<Self> {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        Ok(Self {
            router: matrix(g, &name("ffn_gate_inp"), c.n_embd, d.n_expert)?,
            gate: experts(g, &name("ffn_gate_exps"), c.n_embd, d.n_ff, d.n_expert)?,
            up: experts(g, &name("ffn_up_exps"), c.n_embd, d.n_ff, d.n_expert)?,
            down: experts(g, &name("ffn_down_exps"), d.n_ff, c.n_embd, d.n_expert)?,
            k: d.k,
            shared: SwiGlu {
                gate: matrix(g, &name("ffn_gate_shexp"), c.n_embd, d.n_ff_shared)?,
                up: matrix(g, &name("ffn_up_shexp"), c.n_embd, d.n_ff_shared)?,
                down: matrix(g, &name("ffn_down_shexp"), d.n_ff_shared, c.n_embd)?,
            },
            shared_gate: matrix(g, &name("ffn_gate_inp_shexp"), c.n_embd, 1)?,
        })
    }

    /// See [`Ffn::forward`].
    fn forward<B: Backend>(
        &self,
        b: &mut B,
        f: &mut MoeBuffers<B>,
        s: &mut Scratch<B>,
        next: Next,
    ) {
        let n_expert = self.router.rows;
        b.matvecs(
            &s.xn,
            [
                (&self.router, &mut f.logits),
                (&self.shared_gate, &mut f.shared_gate),
            ],
        );
        b.moe_route(&f.logits, Gating::Softmax, n_expert, self.k, &mut f.route);
        b.moe_swiglu(
            &self.gate,
            &self.up,
            &s.xn,
            &f.route,
            self.k,
            &mut f.hidden,
            &mut f.up,
        );
        b.moe_down(
            &self.down,
            &f.hidden,
            &f.route,
            self.k,
            &mut f.products,
            &mut f.out,
        );
        let sh = &self.shared;
        b.swiglu(
            &sh.gate,
            &sh.up,
            &s.xn,
            &mut f.shared_hidden,
            &mut f.shared_up,
        );
        b.matvecs(&f.shared_hidden, [(&sh.down, &mut f.shared_out)]);
        b.moe_residual(
            &f.out,
            &f.shared_out,
            &f.shared_gate,
            &mut s.x,
            next.norm,
            next.eps,
            &mut s.xn,
        );
    }
}
