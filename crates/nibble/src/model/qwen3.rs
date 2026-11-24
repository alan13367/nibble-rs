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

use super::bind::vector;
use super::layers::{Attention, AttentionBuffers, FfnBuffers, Head, KvCache, Scratch, SwiGlu};
use super::{Ctx, ModelConfig};
use crate::cpu;
use crate::error::Result;

pub(crate) struct Qwen3 {
    head: Head,
    blocks: Vec<Block>,
}

struct Block {
    attn_norm: Vec<f32>,
    attn: Attention,
    ffn_norm: Vec<f32>,
    ffn: SwiGlu,
}

/// Per-session state: one KV cache per layer plus scratch buffers.
pub(crate) struct State {
    caches: Vec<KvCache>,
    x: Vec<f32>,
    attn: AttentionBuffers,
    ffn: FfnBuffers,
    scratch: Scratch,
}

impl Qwen3 {
    pub(crate) fn load(g: &Gguf<'_>, c: &ModelConfig) -> Result<Self> {
        let blocks = (0..c.n_layer)
            .map(|i| {
                Ok(Block {
                    attn_norm: vector(g, &format!("blk.{i}.attn_norm.weight"), c.n_embd)?,
                    attn: Attention::load(g, c, i, true)?,
                    ffn_norm: vector(g, &format!("blk.{i}.ffn_norm.weight"), c.n_embd)?,
                    ffn: SwiGlu::load(g, c, i)?,
                })
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            head: Head::load(g, c)?,
            blocks,
        })
    }

    pub(crate) fn new_state(c: &ModelConfig, n_ctx: usize) -> State {
        State {
            caches: (0..c.n_layer).map(|_| KvCache::new(c, n_ctx)).collect(),
            x: vec![0.0; c.n_embd],
            attn: AttentionBuffers::new(c, n_ctx),
            ffn: FfnBuffers::new(c),
            scratch: Scratch::new(c),
        }
    }

    /// Runs `token` at position `pos` (the number of cached positions).
    pub(crate) fn forward(
        &self,
        ctx: &Ctx<'_>,
        s: &mut State,
        token: u32,
        pos: usize,
        logits: Option<&mut [f32]>,
    ) {
        let c = ctx.config;
        self.head.embed(ctx, token, &mut s.x);
        cpu::rope_table(&mut s.scratch.rope, pos, c.rope_dims, c.rope_base);
        for (block, cache) in self.blocks.iter().zip(&mut s.caches) {
            debug_assert_eq!(cache.len(), pos);
            cpu::rmsnorm(&mut s.scratch.xn, &s.x, &block.attn_norm, c.rms_eps);
            block.attn.forward(ctx, cache, &mut s.attn, &mut s.scratch);
            cpu::add_in_place(&mut s.x, &s.scratch.out);

            cpu::rmsnorm(&mut s.scratch.xn, &s.x, &block.ffn_norm, c.rms_eps);
            block.ffn.forward(ctx, &mut s.ffn, &mut s.scratch);
            cpu::add_in_place(&mut s.x, &s.scratch.out);
        }
        if let Some(logits) = logits {
            self.head.logits(ctx, &s.x, &mut s.scratch, logits);
        }
    }
}

impl State {
    /// Keeps the first `n_pos` positions.
    pub(crate) fn truncate(&mut self, n_pos: usize) {
        for cache in &mut self.caches {
            cache.truncate(n_pos);
        }
    }
}
