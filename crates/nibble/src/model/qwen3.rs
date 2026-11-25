//! Qwen3 dense decoder (`qwen3`): pre-norm blocks of grouped-query attention
//! (with per-head Q/K RMSNorm and NeoX RoPE) and a SwiGLU feed-forward.
//!
//! ```text
//! x = token_embd[token]
//! for each layer:
//!     x += attention(rmsnorm(x) * attn_norm)
//!     x += swiglu(rmsnorm(x) * ffn_norm)
//! logits = output · (rmsnorm(x) * output_norm)        (output = token_embd when tied)
//! ```

use nibble_gguf::Gguf;

use super::ModelConfig;
use super::bind::vector;
use super::layers::{Attention, AttentionBuffers, FfnBuffers, Head, Next, Scratch, SwiGlu};
use crate::backend::{Backend, Param, Params};
use crate::error::Result;

pub(crate) struct Qwen3 {
    head: Head,
    blocks: Vec<Block>,
}

struct Block {
    attn_norm: Param,
    attn: Attention,
    ffn_norm: Param,
    ffn: SwiGlu,
}

/// Per-session state: one KV cache per layer plus scratch buffers.
pub(crate) struct State<B: Backend> {
    caches: Vec<B::KvCache>,
    attn: AttentionBuffers<B>,
    ffn: FfnBuffers<B>,
    scratch: Scratch<B>,
}

impl Qwen3 {
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig, p: &mut Params) -> Result<Self> {
        let blocks = (0..c.n_layer)
            .map(|i| {
                Ok(Block {
                    attn_norm: vector(g, p, &format!("blk.{i}.attn_norm.weight"), c.n_embd)?,
                    attn: Attention::load(g, c, p, i, true)?,
                    ffn_norm: vector(g, p, &format!("blk.{i}.ffn_norm.weight"), c.n_embd)?,
                    ffn: SwiGlu::load(g, c, i)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            head: Head::load(g, c, p)?,
            blocks,
        })
    }

    pub(crate) fn new_state<B: Backend>(
        b: &mut B,
        c: &ModelConfig,
        n_ctx: usize,
    ) -> Result<State<B>> {
        Ok(State {
            caches: (0..c.n_layer)
                .map(|_| b.kv_cache(n_ctx))
                .collect::<Result<_>>()?,
            attn: AttentionBuffers::new(b, c)?,
            ffn: FfnBuffers::new(b, c)?,
            scratch: Scratch::new(b, c)?,
        })
    }

    /// Runs `token` at position `pos` (the number of cached positions).
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        c: &ModelConfig,
        s: &mut State<B>,
        token: u32,
        pos: usize,
        logits: Option<&mut [f32]>,
    ) -> Result<()> {
        let x = &mut s.scratch;
        b.begin(pos)?;
        self.head.embed(b, token, &mut x.x);
        let norm = |p: Param| Next {
            norm: p,
            eps: c.rms_eps,
        };
        let first = self
            .blocks
            .first()
            .map_or(self.head.output_norm, |b| b.attn_norm);
        b.rmsnorm(&x.x, first, c.rms_eps, &mut x.xn);
        for (i, (block, cache)) in self.blocks.iter().zip(&mut s.caches).enumerate() {
            debug_assert_eq!(b.kv_len(cache), pos);
            let next = self
                .blocks
                .get(i + 1)
                .map_or(self.head.output_norm, |b| b.attn_norm);
            block
                .attn
                .forward(b, c, cache, &mut s.attn, x, norm(block.ffn_norm));
            block.ffn.forward(b, &mut s.ffn, x, norm(next));
        }
        self.head.finish(b, &x.xn, logits)
    }
}

impl<B: Backend> State<B> {
    /// Keeps the first `n_pos` positions.
    pub(crate) fn truncate(&mut self, b: &mut B, n_pos: usize) -> Result<()> {
        for cache in &mut self.caches {
            b.kv_truncate(cache, n_pos)?;
        }
        Ok(())
    }
}
