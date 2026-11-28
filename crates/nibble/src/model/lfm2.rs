//! LFM2 (Liquid's hybrid shortconv / attention decoders, `lfm2` and
//! `lfm2moe`).
//!
//! Follows llama.cpp's `src/models/lfm2.cpp`, `src/models/lfm2moe.cpp` and
//! `src/models/lfm-base.cpp`, and the CPU op
//! `ggml_compute_forward_ssm_conv`. Only the causal text models are
//! covered; the non-causal decision (D1) and bidirectional heads, which
//! llama.cpp runs through no-cache graphs, are rejected at load.
//!
//! `attention.head_count_kv` holds one count per block: 0 marks a
//! shortconv block, whose recurrent state is its convolution history (the
//! `lfm2.shortconv.l_cache - 1` values of `b * x` before the next
//! token). Sessions rewind through checkpoints, as for Qwen3.5.
//!
//! ```text
//! x = token_embd[token]
//! for each layer:
//!     x += mixer(rmsnorm(x) * attn_norm)          shortconv or attention
//!     x += ffn(rmsnorm(x) * ffn_norm)             SwiGLU or experts
//! logits = output * (rmsnorm(x) * token_embd_norm)
//!
//! shortconv, per token:
//!     [b | c | x] = in_proj * h                   3 rows of n_embd
//!     y = c * conv(b * x)                         causal, l_cache taps,
//!                                                oldest first
//!     out = out_proj * y
//! ```

use nibble_gguf::Gguf;

use super::bind::{experts, matrix, tensor_f32, vector};
use super::config::Meta;
use super::layers::{Attention, AttentionBuffers, FfnBuffers, Head, Next, Scratch, SwiGlu};
use super::{Architecture, Logits, ModelConfig};
use crate::backend::{Backend, Gating, Param, Params, ShortConvDims, ShortConvState};
use crate::cpu::{Experts, Weight};
use crate::error::{Error, Result};

pub(crate) struct Lfm2 {
    head: Head,
    blocks: Vec<Block>,
    /// The sizes of the shortconv layers.
    dims: ShortConvDims,
    /// Whether any block is a shortconv one (the state then cannot rewind).
    has_conv: bool,
    /// The sizes of the mixture-of-experts feed-forwards (`lfm2moe`).
    moe: Option<LfmMoeDims>,
}

/// A mixture-of-experts feed-forward of `lfm2moe`: sigmoid gating with a
/// selection bias, no shared expert (llama.cpp's `build_lfm_moe_ffn`).
#[derive(Clone, Copy, Debug)]
struct LfmMoeDims {
    n_expert: usize,
    /// Experts per token.
    k: usize,
    /// Hidden size of each expert.
    n_ff: usize,
    /// Blocks before this one run dense feed-forwards.
    dense_lead: usize,
}

struct Block {
    /// The operator norm (`blk.{i}.attn_norm`), which the mixer reads.
    attn_norm: Param,
    mixer: Mixer,
    ffn_norm: Param,
    ffn: LfmFfn,
}

enum Mixer {
    Conv(ShortConv),
    Attention(Attention),
}

enum LfmFfn {
    Dense(SwiGlu),
    Moe(Box<LfmMoe>),
}

/// LFM2's shortconv block: a gated causal depthwise convolution.
struct ShortConv {
    /// `[n_embd, 3 * n_embd]`: the gates `b`, then `c`, then the
    /// input `x`, each one row.
    in_proj: Weight,
    /// The convolution taps by time, oldest first (`shortconv.conv`
    /// stores each channel's taps together, transposed here as
    /// Qwen3.5's are).
    taps: Param,
    /// `[n_embd, n_embd]`.
    out_proj: Weight,
}

/// One mixture-of-experts feed-forward layer.
struct LfmMoe {
    router: Weight,
    gate: Experts,
    up: Experts,
    down: Experts,
    /// `ffn_exp_probs_b`: the selection bias, one value per expert.
    bias: Param,
    /// Experts per token.
    k: usize,
}

impl Lfm2 {
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, p: &mut Params) -> Result<Self> {
        let m = Meta::new(g, c.architecture);
        // The non-causal decision models (D1) run through a no-cache graph
        // llama.cpp serves elsewhere.
        if m.usize_opt("decision.block_count")?.is_some_and(|n| n > 0) {
            return Err(Error::UnsupportedArchitecture(
                "lfm2 decision models (non-causal heads)".into(),
            ));
        }
        // A sliding window would need masking in every backend's attention;
        // no LFM2 text file carries one, so refuse it outright.
        if m.usize_opt("attention.sliding_window")?
            .is_some_and(|n| n > 0)
        {
            return Err(m.invalid(
                "attention.sliding_window",
                "sliding-window LFM2 models are not supported",
            ));
        }
        if g.get(&m.key("attention.causal"))
            .is_some_and(|v| v.as_bool() == Some(false))
        {
            return Err(m.invalid(
                "attention.causal",
                "bidirectional LFM2 models are not supported",
            ));
        }
        let dims = ShortConvDims {
            taps: m.usize("shortconv.l_cache")?,
            n_embd: c.n_embd,
        };
        if dims.taps < 2 {
            return Err(m.invalid("shortconv.l_cache", "must be at least 2"));
        }
        let moe = match c.architecture {
            Architecture::Lfm2Moe => {
                // llama.cpp aborts on any other gating than the file's, and
                // no lfm2moe file carries another one.
                let gating = g.get(&m.key("expert_gating_func")).and_then(|v| v.as_str());
                if gating != Some("sigmoid") {
                    return Err(m.invalid("expert_gating_func", "only sigmoid gating is supported"));
                }
                let d = LfmMoeDims {
                    n_expert: m.usize("expert_count")?,
                    k: m.usize("expert_used_count")?,
                    n_ff: m.usize("expert_feed_forward_length")?,
                    dense_lead: m.usize_opt("leading_dense_block_count")?.unwrap_or(0),
                };
                if d.k == 0 || d.k > d.n_expert || d.n_ff == 0 || d.dense_lead > c.n_layer {
                    return Err(m.invalid(
                        "expert_used_count",
                        "expert counts and sizes must be non-zero, with at most expert_count used and at most block_count dense-led",
                    ));
                }
                Some(d)
            }
            Architecture::Lfm2 => None,
            _ => unreachable!("another architecture's module"),
        };
        let blocks = (0..c.n_layer)
            .map(|i| {
                let name = |n: &str| format!("blk.{i}.{n}.weight");
                let mixer = if is_conv_layer(g, &m, i)? {
                    Mixer::Conv(ShortConv::load(g, c, p, dims, i)?)
                } else {
                    Mixer::Attention(Attention::load(g, c, p, i, true)?)
                };
                let ffn = match moe {
                    Some(d) if i >= d.dense_lead => {
                        LfmFfn::Moe(Box::new(LfmMoe::load(g, c, p, d, i)?))
                    }
                    _ => LfmFfn::Dense(SwiGlu::load(g, c, i)?),
                };
                Ok(Block {
                    attn_norm: vector(g, p, &name("attn_norm"), c.n_embd)?,
                    mixer,
                    ffn_norm: vector(g, p, &name("ffn_norm"), c.n_embd)?,
                    ffn,
                })
            })
            .collect::<Result<Vec<Block>>>()?;
        let has_conv = blocks.iter().any(|b| matches!(b.mixer, Mixer::Conv(_)));
        Ok(Self {
            head: Head::load_with_norm(g, c, p, "token_embd_norm.weight")?,
            blocks,
            dims,
            has_conv,
            moe,
        })
    }

    /// Whether any block is a shortconv one.
    pub(crate) const fn has_conv(&self) -> bool {
        self.has_conv
    }

    /// The sizes of the shortconv layers.
    pub(crate) const fn dims(&self) -> ShortConvDims {
        self.dims
    }

    pub(crate) fn new_state<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        n_ctx: usize,
        n_batch: usize,
    ) -> Result<State<B>> {
        if self.moe.is_some() {
            b.check_moe()?;
        }
        let dims = self.dims;
        let layers = self
            .blocks
            .iter()
            .map(|block| {
                Ok(match block.mixer {
                    Mixer::Conv(_) => LayerState::Conv(ShortConvState {
                        conv: b.vector(dims.conv_len())?,
                        oldest: 0,
                    }),
                    Mixer::Attention(_) => LayerState::Attention(b.kv_cache(n_ctx)?),
                })
            })
            .collect::<Result<_>>()?;
        Ok(State {
            layers,
            scratch: Scratch::new(b, c, n_batch)?,
            ffn: FfnBuffers::new(b, c, None, n_batch)?,
            conv: ConvBuffers {
                bcx: b.vector(n_batch * 3 * c.n_embd)?,
                y: b.vector(n_batch * c.n_embd)?,
            },
            attn: AttentionBuffers::new(b, c, n_batch)?,
            moe: self
                .moe
                .map(|d| LfmMoeBuffers::new(b, c, d, n_batch))
                .transpose()?,
        })
    }

    /// Runs `tokens` at positions from `pos` (the number of tokens seen
    /// by the state) on.
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        s: &mut State<B>,
        tokens: &[u32],
        pos: usize,
        logits: Option<Logits<'_>>,
    ) -> Result<()> {
        let x = &mut s.scratch;
        b.begin(pos, tokens.len())?;
        self.head.embed(b, tokens, &mut x.x);
        let norm = |p: Param| Next {
            norm: p,
            eps: c.rms_eps,
        };
        let first = self
            .blocks
            .first()
            .map_or(self.head.output_norm, |b| b.attn_norm);
        b.rmsnorm(&x.x, first, c.rms_eps, &mut x.xn);
        for (i, (block, layer)) in self.blocks.iter().zip(&mut s.layers).enumerate() {
            let after_mixer = norm(block.ffn_norm);
            match (&block.mixer, layer) {
                (Mixer::Conv(sc), LayerState::Conv(state)) => {
                    b.matvecs(&x.xn, [(&sc.in_proj, &mut s.conv.bcx)]);
                    b.short_conv(self.dims, sc.taps, state, &s.conv.bcx, &mut s.conv.y);
                    x.add(b, &sc.out_proj, &s.conv.y, after_mixer);
                }
                (Mixer::Attention(attn), LayerState::Attention(cache)) => {
                    debug_assert_eq!(b.kv_len(cache), pos);
                    attn.forward(b, c, cache, &mut s.attn, x, after_mixer);
                }
                _ => unreachable!("layer state does not match its mixer"),
            }
            let next = self
                .blocks
                .get(i + 1)
                .map_or(self.head.output_norm, |b| b.attn_norm);
            match &block.ffn {
                LfmFfn::Dense(ffn) => ffn.forward(b, &mut s.ffn, x, norm(next)),
                LfmFfn::Moe(m) => {
                    let Some(f) = &mut s.moe else {
                        panic!("a mixture of experts without its buffers");
                    };
                    b.matvecs(&x.xn, [(&m.router, &mut f.logits)]);
                    b.moe_route(
                        &f.logits,
                        Gating::Sigmoid { bias: m.bias },
                        m.router.rows,
                        m.k,
                        &mut f.route,
                    );
                    b.moe_swiglu(
                        &m.gate,
                        &m.up,
                        &x.xn,
                        &f.route,
                        m.k,
                        &mut f.hidden,
                        &mut f.up,
                    );
                    b.moe_down(
                        &m.down,
                        &f.hidden,
                        &f.route,
                        m.k,
                        &mut f.products,
                        &mut f.out,
                    );
                    b.add_norm(&f.out, &mut x.x, next, c.rms_eps, &mut x.xn);
                }
            }
        }
        self.head.finish(b, &x.xn, logits)
    }
}

/// Whether block `i` is a shortconv one: `attention.head_count_kv` marks
/// it with 0 (a scalar names one count for every block).
fn is_conv_layer(g: &Gguf<'_>, m: &Meta<'_, '_>, i: usize) -> Result<bool> {
    Ok(g.get(&m.key("attention.head_count_kv"))
        .and_then(|v| v.as_array())
        .and_then(|a| a.get(i))
        .is_some_and(|e| e.as_u64() == Some(0)))
}

impl ShortConv {
    fn load(
        g: &Gguf<'_>,
        c: &ModelConfig,
        p: &mut Params,
        d: ShortConvDims,
        layer: usize,
    ) -> Result<Self> {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        let (k, channels) = (d.taps, d.n_embd);
        let by_channel = tensor_f32(g, &name("shortconv.conv"), &[k, channels])?;
        let mut conv = vec![0.0; k * channels];
        for (ch, taps) in by_channel.chunks_exact(k).enumerate() {
            for (t, &tap) in taps.iter().enumerate() {
                conv[t * channels + ch] = tap;
            }
        }
        Ok(Self {
            in_proj: matrix(g, &name("shortconv.in_proj"), c.n_embd, 3 * c.n_embd)?,
            taps: p.push(&conv),
            out_proj: matrix(g, &name("shortconv.out_proj"), c.n_embd, c.n_embd)?,
        })
    }
}

impl LfmMoe {
    fn load(
        g: &Gguf<'_>,
        c: &ModelConfig,
        p: &mut Params,
        d: LfmMoeDims,
        layer: usize,
    ) -> Result<Self> {
        let name = |n: &str| format!("blk.{layer}.{n}.weight");
        Ok(Self {
            router: matrix(g, &name("ffn_gate_inp"), c.n_embd, d.n_expert)?,
            gate: experts(g, &name("ffn_gate_exps"), c.n_embd, d.n_ff, d.n_expert)?,
            up: experts(g, &name("ffn_up_exps"), c.n_embd, d.n_ff, d.n_expert)?,
            down: experts(g, &name("ffn_down_exps"), d.n_ff, c.n_embd, d.n_expert)?,
            bias: vector(
                g,
                p,
                &format!("blk.{layer}.ffn_exp_probs_b.bias"),
                d.n_expert,
            )?,
            k: d.k,
        })
    }
}

/// Per-session state: a shortconv history or a KV cache per layer, plus
/// scratch.
pub(crate) struct State<B: Backend> {
    layers: Vec<LayerState<B>>,
    scratch: Scratch<B>,
    ffn: FfnBuffers<B>,
    conv: ConvBuffers<B>,
    attn: AttentionBuffers<B>,
    moe: Option<LfmMoeBuffers<B>>,
}

enum LayerState<B: Backend> {
    Conv(ShortConvState<B::Vector>),
    Attention(B::KvCache),
}

/// Buffers for the shortconv blocks, one row per token.
struct ConvBuffers<B: Backend> {
    /// `in_proj`'s output: the gates `b`, then `c`, then the input
    /// `x`.
    bcx: B::Vector,
    /// `c` gating the convolution of `b * x`.
    y: B::Vector,
}

/// Buffers for a mixture-of-experts feed-forward, one row per token
/// (or `k` rows per token, for the experts' hidden states).
struct LfmMoeBuffers<B: Backend> {
    logits: B::Vector,
    route: B::Vector,
    hidden: B::Vector,
    up: B::Vector,
    products: B::Vector,
    out: B::Vector,
}

impl<B: Backend> LfmMoeBuffers<B> {
    fn new(b: &mut B, c: &ModelConfig, d: LfmMoeDims, n_batch: usize) -> Result<Self> {
        let rows = |n: usize| n_batch * n;
        Ok(Self {
            logits: b.vector(rows(d.n_expert))?,
            route: b.vector(rows(2 * d.k))?,
            hidden: b.vector(rows(d.k * d.n_ff))?,
            up: b.vector(rows(d.k * d.n_ff))?,
            products: b.vector(rows(d.k * c.n_embd))?,
            out: b.vector(rows(c.n_embd))?,
        })
    }
}

/// The recurrent parts of a [`State`] at some position: every shortconv
/// layer's convolution window, oldest first, in layer order. The KV caches
/// need no copy: they rewind by truncation.
#[derive(Default)]
pub(crate) struct Snapshot {
    conv: Vec<f32>,
    /// Scratch for one layer's history.
    ring: Vec<f32>,
}

impl<B: Backend> State<B> {
    fn conv_layers(&self) -> impl Iterator<Item = &ShortConvState<B::Vector>> {
        self.layers.iter().filter_map(|l| match l {
            LayerState::Conv(s) => Some(s),
            LayerState::Attention(_) => None,
        })
    }

    /// Takes back the last `from - to` tokens. A state with shortconv
    /// layers cannot (their histories would need the dropped tokens'
    /// values): the caller replays from a checkpoint. Returns false,
    /// changing nothing, then.
    pub(crate) fn rewind(
        &mut self,
        b: &mut B,
        has_conv: bool,
        _from: usize,
        to: usize,
    ) -> Result<bool> {
        if has_conv {
            return Ok(false);
        }
        for layer in &mut self.layers {
            if let LayerState::Attention(cache) = layer {
                b.kv_truncate(cache, to)?;
            }
        }
        Ok(true)
    }

    /// Copies the recurrent state into `snapshot`, reusing its buffers.
    pub(crate) fn save(
        &mut self,
        b: &mut B,
        d: ShortConvDims,
        snapshot: &mut Snapshot,
    ) -> Result<()> {
        let (e, rows) = (d.n_embd, d.conv_rows());
        let n_layers = self.conv_layers().count();
        snapshot.conv.resize(n_layers * rows * e, 0.0);
        snapshot.ring.resize(d.conv_len(), 0.0);
        for (i, s) in self.conv_layers().enumerate() {
            b.read(&s.conv, &mut snapshot.ring)?;
            let out = &mut snapshot.conv[i * rows * e..(i + 1) * rows * e];
            for (r, out) in out.chunks_exact_mut(e).enumerate() {
                let row = (s.oldest + r) % rows;
                out.copy_from_slice(&snapshot.ring[row * e..(row + 1) * e]);
            }
        }
        Ok(())
    }

    /// Returns to a [`State::save`] taken `n_pos` tokens in: the saved
    /// histories go back to their rings' starts and the caches are
    /// truncated.
    pub(crate) fn restore(
        &mut self,
        b: &mut B,
        d: ShortConvDims,
        snapshot: &mut Snapshot,
        n_pos: usize,
    ) -> Result<()> {
        let (e, rows) = (d.n_embd, d.conv_rows());
        let mut i = 0;
        for layer in &mut self.layers {
            match layer {
                LayerState::Conv(s) => {
                    b.write(
                        &mut s.conv,
                        &snapshot.conv[i * rows * e..(i + 1) * rows * e],
                    )?;
                    s.oldest = 0;
                    i += 1;
                }
                LayerState::Attention(cache) => b.kv_truncate(cache, n_pos)?,
            }
        }
        Ok(())
    }
}
