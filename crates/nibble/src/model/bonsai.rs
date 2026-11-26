//! PrismML's Ternary Bonsai models: a Qwen3.5 hybrid decoder (`qwen35`)
//! whose weights have a Hadamard rotation folded in (`prism.hadamard.*`,
//! see [`super::hadamard`]), as llama.cpp runs them.
//!
//! The weights, the session state and its rewinds are [`Qwen35`]'s. The
//! forward pass is Qwen3.5's (see `qwen35.rs`) with each folded weight's
//! input rotated first, and the token embedding, stored rotated, turned
//! back after each lookup:
//!
//! ```text
//! x = D · H · token_embd[token]
//! a product with a folded weight W·D·H:   y = (W·D·H) · (H · D · P x)
//! ```
//!
//! llama.cpp folds the trunk's `attn_qkv`, `attn_gate`, `ssm_out` (whose
//! input is permuted, `P`), `attn_{q,k,v,output}`, `ffn_{gate,up,down}` and
//! the output head; never `ssm_beta`, `ssm_alpha` or the MTP head's
//! `eh_proj`. The MTP head reads the rotated embedding and, without its own
//! head, the folded output head.

use nibble_gguf::Gguf;

use super::hadamard::Folding;
use super::layers::{Ffn, Head, Next, Scratch};
use super::qwen35::{self, GatedAttention, LayerState, Mixer, Mtp, MtpState, Qwen35};
use super::{Logits, ModelConfig};
use crate::backend::{Backend, Param, Params, QkNorm, Rotation};
use crate::cpu::Weight;
use crate::error::Result;

/// Whether a file's weights are folded (and so a Bonsai model).
pub(crate) fn is_folded(g: &Gguf<'_>) -> bool {
    g.get("prism.hadamard.version").is_some()
}

pub(crate) struct Bonsai {
    base: Qwen35,
    /// The rotations of each trunk block's products, then the MTP head's.
    blocks: Vec<Rotations>,
    mtp: Rotations,
    /// Turns the token embedding's rows back.
    embd: Option<Rotation>,
    /// The rotation of the output head's input.
    output: Option<Rotation>,
    /// The MTP head's: its own embedding and head are never folded.
    mtp_embd: Option<Rotation>,
    mtp_head: Option<Rotation>,
    /// Every distinct rotation, for the backend to check.
    rotations: Vec<Rotation>,
}

/// The rotations of one block's products' inputs.
#[derive(Clone, Copy, Debug, Default)]
struct Rotations {
    /// DeltaNet `attn_qkv` and `attn_gate`, or attention `attn_{q,k,v}`.
    mixer_in: Option<Rotation>,
    /// `ssm_out` or `attn_output`.
    mixer_out: Option<Rotation>,
    ffn_in: Option<Rotation>,
    ffn_down: Option<Rotation>,
}

/// Per-session state: [`Qwen35`]'s, and the rotated input of a product.
pub(crate) struct State<B: Backend> {
    pub(crate) base: qwen35::State<B>,
    /// One row per token, as long as the longest rotation.
    rot: B::Vector,
}

/// Where the MTP head's tokens come from.
#[derive(Clone, Copy)]
enum Embed<'a> {
    Tokens(&'a [u32]),
    /// A token chosen on the device ([`Backend::argmax`]).
    Slot(usize),
}

/// Which hidden state the MTP head pairs with its tokens (as in `qwen35`).
#[derive(Clone, Copy)]
enum HeadInput {
    Batch,
    Last,
    Draft,
}

/// `x`, or `x` rotated into `scratch`.
fn rotated<'a, B: Backend>(
    b: &mut B,
    r: Option<Rotation>,
    x: &'a B::Vector,
    scratch: &'a mut B::Vector,
) -> &'a B::Vector {
    match r {
        Some(r) => {
            b.rotate(x, r, scratch);
            scratch
        }
        None => x,
    }
}

/// `s.x += w · a` (`a` rotated by `r` into `rot` first), then `s.xn =
/// rmsnorm(s.x)` with the next block's norm.
fn add<B: Backend>(
    b: &mut B,
    s: &mut Scratch<B>,
    rot: &mut B::Vector,
    w: &Weight,
    r: Option<Rotation>,
    a: &B::Vector,
    next: Next,
) {
    let a = rotated(b, r, a, rot);
    b.residual(w, a, &mut s.x, next.norm, next.eps, &mut s.xn);
}

impl Bonsai {
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, p: &mut Params) -> Result<Self> {
        // The registry sends only dense `qwen35` files here.
        let base = Qwen35::load(g, c, p)?;
        let f = Folding::read(g, p)?;
        let w = |layer: usize, n: &str| format!("blk.{layer}.{n}.weight");
        let attention = |layer: usize| -> Result<_> {
            let qkv = [w(layer, "attn_q"), w(layer, "attn_k"), w(layer, "attn_v")];
            Ok((
                f.input(&qkv.each_ref().map(String::as_str), c.n_embd)?,
                f.input(&[&w(layer, "attn_output")], c.n_head * c.head_dim)?,
            ))
        };
        let block = |layer: usize, mixer: Option<&Mixer>| -> Result<Rotations> {
            let (mixer_in, mixer_out) = match mixer {
                Some(Mixer::DeltaNet(dn)) => (
                    f.input(&[&w(layer, "attn_qkv"), &w(layer, "attn_gate")], c.n_embd)?,
                    f.delta_out(&w(layer, "ssm_out"), dn.dims)?,
                ),
                // The MTP head's block is a gated attention block.
                Some(Mixer::Attention(_)) | None => attention(layer)?,
            };
            Ok(Rotations {
                mixer_in,
                mixer_out,
                ffn_in: f.input(&[&w(layer, "ffn_gate"), &w(layer, "ffn_up")], c.n_embd)?,
                ffn_down: f.input(&[&w(layer, "ffn_down")], c.n_ff)?,
            })
        };
        let blocks = base
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| block(i, Some(&b.mixer)))
            .collect::<Result<_>>()?;
        let mtp = match base.mtp {
            Some(_) => block(c.n_layer, None)?,
            None => Rotations::default(),
        };
        let embd = f.lookup("token_embd.weight", c.n_embd)?;
        let output_name = if c.tied_embeddings {
            "token_embd.weight"
        } else {
            "output.weight"
        };
        let output = f.input(&[output_name], c.n_embd)?;
        let nextn = |n: &str| g.tensor(&w(c.n_layer, n)).is_some();
        let mtp_embd = if nextn("nextn.embed_tokens") {
            None
        } else {
            embd
        };
        let mtp_head = if nextn("nextn.shared_head_head") {
            None
        } else {
            output
        };
        Ok(Self {
            base,
            blocks,
            mtp,
            embd,
            output,
            mtp_embd,
            mtp_head,
            rotations: f.finish()?,
        })
    }

    pub(crate) const fn has_mtp(&self) -> bool {
        self.base.has_mtp()
    }

    pub(crate) const fn dims(&self) -> crate::backend::DeltaNetDims {
        self.base.dims()
    }

    pub(crate) fn new_state<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        n_ctx: usize,
        n_batch: usize,
        mtp: bool,
    ) -> Result<State<B>> {
        for &r in &self.rotations {
            b.check_rotation(r)?;
        }
        let len = self.rotations.iter().map(|r| r.len).max().unwrap_or(1);
        Ok(State {
            base: self.base.new_state(b, c, n_ctx, n_batch, mtp)?,
            rot: b.vector(n_batch * len)?,
        })
    }

    fn head(&self) -> &Head {
        &self.base.head
    }

    /// Runs `tokens` at positions from `pos` (the number of tokens seen by
    /// the state) on.
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        s: &mut State<B>,
        tokens: &[u32],
        pos: usize,
        logits: Option<Logits<'_>>,
    ) -> Result<()> {
        let head = self.head();
        let blocks = &self.base.blocks;
        let (st, rot) = (&mut s.base, &mut s.rot);
        let x = &mut st.scratch;
        b.begin(pos, tokens.len())?;
        b.embed(&head.token_embd(), tokens, &mut x.x);
        if let Some(r) = self.embd {
            b.unrotate(&mut x.x, r);
        }
        let norm = |p: Param| Next {
            norm: p,
            eps: c.rms_eps,
        };
        let first = blocks.first().map_or(head.output_norm, |b| b.attn_norm);
        b.rmsnorm(&x.x, first, c.rms_eps, &mut x.xn);
        let layers = blocks.iter().zip(&self.blocks).zip(&mut st.layers);
        for (i, ((block, r), layer)) in layers.enumerate() {
            let after_mixer = norm(block.post_attention_norm);
            match (&block.mixer, layer) {
                (Mixer::DeltaNet(dn), LayerState::DeltaNet(state)) => {
                    let f = &mut st.dn;
                    if let Some(ri) = r.mixer_in {
                        b.rotate(&x.xn, ri, rot);
                        b.matvecs(rot, [(&dn.qkv, &mut f.qkv), (&dn.gate, &mut f.z)]);
                        b.matvecs(&x.xn, [(&dn.beta, &mut f.beta), (&dn.alpha, &mut f.alpha)]);
                    } else {
                        b.matvecs(
                            &x.xn,
                            [
                                (&dn.qkv, &mut f.qkv),
                                (&dn.gate, &mut f.z),
                                (&dn.beta, &mut f.beta),
                                (&dn.alpha, &mut f.alpha),
                            ],
                        );
                    }
                    b.delta_net(
                        dn.dims,
                        dn.params,
                        c.rms_eps,
                        state,
                        &f.qkv,
                        &mut f.conv,
                        &mut f.z,
                        &mut f.beta,
                        &mut f.alpha,
                        &mut f.o,
                    );
                    add(b, x, rot, &dn.out, r.mixer_out, &f.o, after_mixer);
                }
                (Mixer::Attention(attn), LayerState::Attention(cache)) => {
                    debug_assert_eq!(b.kv_len(cache), pos);
                    attention(b, c, attn, r, cache, &mut st.attn, x, rot, after_mixer);
                }
                _ => unreachable!("layer state does not match its mixer"),
            }
            let next = blocks.get(i + 1).map_or(head.output_norm, |b| b.attn_norm);
            ffn(b, &block.ffn, r, &mut st.ffn, x, rot, norm(next));
        }
        match logits {
            Some(l) => {
                let xn = rotated(b, self.output, &x.xn, rot);
                b.logits(&head.output(), xn, l.rows, l.out)?;
            }
            None => b.end()?,
        }
        if let (Some(m), Some(_)) = (&self.base.mtp, &s.base.mtp) {
            self.catch_up(b, c, m, s, tokens, pos)?;
        }
        Ok(())
    }

    /// One draft step of the MTP head (see `Qwen35::draft`).
    ///
    /// # Panics
    ///
    /// If the state has no MTP head (an internal invariant).
    #[expect(clippy::too_many_arguments, reason = "one draft step")]
    pub(crate) fn draft<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        s: &mut State<B>,
        token: u32,
        pos: usize,
        first: bool,
        logits: &mut [f32],
    ) -> Result<()> {
        let st = &mut s.base;
        let (Some(head), Some(m)) = (&self.base.mtp, &mut st.mtp) else {
            panic!("drafting without an MTP head");
        };
        b.kv_truncate(&mut m.cache, pos)?;
        b.begin(pos, 1)?;
        let input = if first {
            HeadInput::Last
        } else {
            HeadInput::Draft
        };
        let embed = Embed::Tokens(std::slice::from_ref(&token));
        self.run(
            b,
            c,
            head,
            m,
            &mut st.attn,
            &mut st.ffn,
            &mut s.rot,
            embed,
            input,
        );
        b.copy_rows(&m.scratch.xn, 0, &mut m.draft_h, 0, 1, c.n_embd);
        let xn = rotated(b, self.mtp_head, &m.scratch.xn, &mut s.rot);
        b.logits(&head.head, xn, 0..1, logits)
    }

    /// `drafts.len()` greedy draft steps of the MTP head, chosen on the
    /// device (see `Qwen35::draft_chain`).
    ///
    /// # Panics
    ///
    /// If the state has no MTP head (an internal invariant).
    pub(crate) fn draft_chain<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        s: &mut State<B>,
        token: u32,
        pos: usize,
        drafts: &mut [u32],
    ) -> Result<()> {
        let st = &mut s.base;
        let (Some(head), Some(m)) = (&self.base.mtp, &mut st.mtp) else {
            panic!("drafting without an MTP head");
        };
        b.kv_truncate(&mut m.cache, pos)?;
        for i in 0..drafts.len() {
            b.begin(pos + i, 1)?;
            let (embed, input) = if i == 0 {
                (Embed::Tokens(std::slice::from_ref(&token)), HeadInput::Last)
            } else {
                (Embed::Slot(i - 1), HeadInput::Draft)
            };
            self.run(
                b,
                c,
                head,
                m,
                &mut st.attn,
                &mut st.ffn,
                &mut s.rot,
                embed,
                input,
            );
            b.copy_rows(&m.scratch.xn, 0, &mut m.draft_h, 0, 1, c.n_embd);
            let xn = rotated(b, self.mtp_head, &m.scratch.xn, &mut s.rot);
            b.matvecs(xn, [(&head.head, &mut m.logits)]);
            b.argmax(&m.logits, head.head.rows, i);
            b.end()?;
        }
        b.read_slots(drafts)
    }

    /// The MTP head on the batch's `tokens` (see `Mtp::run` in `qwen35`).
    #[expect(clippy::too_many_arguments, reason = "the head and its buffers")]
    fn run<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        head: &Mtp,
        m: &mut MtpState<B>,
        attn: &mut qwen35::AttentionBuffers<B>,
        ffn_buffers: &mut super::layers::FfnBuffers<B>,
        rot: &mut B::Vector,
        tokens: Embed<'_>,
        input: HeadInput,
    ) {
        let eps = c.rms_eps;
        let MtpState {
            cache,
            scratch,
            embd,
            h,
            pair,
            last_h,
            draft_h,
            ..
        } = m;
        let h = match input {
            HeadInput::Batch => &*h,
            HeadInput::Last => &*last_h,
            HeadInput::Draft => &*draft_h,
        };
        match tokens {
            Embed::Tokens(t) => b.embed(&head.embed, t, embd),
            Embed::Slot(slot) => b.embed_slot(&head.embed, slot, embd),
        }
        if let Some(r) = self.mtp_embd {
            b.unrotate(embd, r);
        }
        b.norm_pair(embd, head.enorm, h, head.hnorm, eps, pair);
        b.matvecs(pair, [(&head.eh_proj, &mut scratch.x)]);
        b.rmsnorm(&scratch.x, head.attn_norm, eps, &mut scratch.xn);
        let next = |norm: Param| Next { norm, eps };
        let r = &self.mtp;
        attention(
            b,
            c,
            &head.attn,
            r,
            cache,
            attn,
            scratch,
            rot,
            next(head.post_attention_norm),
        );
        ffn(
            b,
            &head.ffn,
            r,
            ffn_buffers,
            scratch,
            rot,
            next(head.head_norm),
        );
    }

    /// The MTP head over the tokens the trunk just evaluated (see
    /// `Mtp::catch_up` in `qwen35`).
    fn catch_up<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        head: &Mtp,
        s: &mut State<B>,
        tokens: &[u32],
        pos: usize,
    ) -> Result<()> {
        let st = &mut s.base;
        let Some(m) = &mut st.mtp else {
            return Ok(());
        };
        let (n, e) = (tokens.len(), c.n_embd);
        let trunk = &st.scratch.xn;
        b.kv_truncate(&mut m.cache, pos)?;
        b.begin(pos, n)?;
        b.copy_rows(&m.last_h, 0, &mut m.h, 0, 1, e);
        b.copy_rows(trunk, 0, &mut m.h, 1, n - 1, e);
        b.copy_rows(trunk, n - 1, &mut m.last_h, 0, 1, e);
        self.run(
            b,
            c,
            head,
            m,
            &mut st.attn,
            &mut st.ffn,
            &mut s.rot,
            Embed::Tokens(tokens),
            HeadInput::Batch,
        );
        m.batch = pos..pos + n;
        b.end()
    }
}

/// Gated attention over `s.xn` (see `GatedAttention::forward` in `qwen35`).
#[expect(clippy::too_many_arguments, reason = "the block and its buffers")]
fn attention<B: Backend>(
    b: &mut B,
    c: &ModelConfig,
    at: &GatedAttention,
    r: &Rotations,
    cache: &mut B::KvCache,
    a: &mut qwen35::AttentionBuffers<B>,
    s: &mut Scratch<B>,
    rot: &mut B::Vector,
    next: Next,
) {
    let x = rotated(b, r.mixer_in, &s.xn, rot);
    b.matvecs(
        x,
        [(&at.wq, &mut a.qg), (&at.wk, &mut a.k), (&at.wv, &mut a.v)],
    );
    let qk_norm = QkNorm {
        weights: Some((at.q_norm, at.k_norm)),
        eps: c.rms_eps,
        gated: true,
        multi_rope: true,
    };
    b.attention(&mut a.qg, &mut a.k, &a.v, qk_norm, cache, &mut a.att);
    add(b, s, rot, &at.wo, r.mixer_out, &a.att, next);
}

/// The dense SwiGLU feed-forward of `s.xn` (the registry refuses folded
/// mixture-of-experts files).
fn ffn<B: Backend>(
    b: &mut B,
    ffn: &Ffn,
    r: &Rotations,
    f: &mut super::layers::FfnBuffers<B>,
    s: &mut Scratch<B>,
    rot: &mut B::Vector,
    next: Next,
) {
    let Ffn::Dense(d) = ffn else {
        unreachable!("a folded mixture of experts was loaded");
    };
    let x = rotated(b, r.ffn_in, &s.xn, rot);
    b.swiglu(&d.gate, &d.up, x, &mut f.hidden, &mut f.up);
    add(b, s, rot, &d.down, r.ffn_down, &f.hidden, next);
}
