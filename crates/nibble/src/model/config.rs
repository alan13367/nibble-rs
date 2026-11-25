//! Hyperparameters shared by decoder architectures, read from `{arch}.*` metadata.
//!
//! Architecture-specific hyperparameters (for example Qwen3.5's DeltaNet sizes)
//! are read by the architecture's own module with [`Meta`].

use nibble_gguf::{GgmlType, Gguf};

use super::Architecture;
use crate::error::{Error, Result};

/// Hyperparameters common to the supported decoder-only transformers.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ModelConfig {
    /// The architecture.
    pub architecture: Architecture,
    /// `general.name`, if present.
    pub name: Option<String>,
    /// Vocabulary size (rows of the embedding matrix).
    pub n_vocab: usize,
    /// Hidden size.
    pub n_embd: usize,
    /// Number of blocks run for each token. Multi-token-prediction blocks
    /// (`nextn_predict_layers`) stored after them are not counted.
    pub n_layer: usize,
    /// Query heads of the attention layers.
    pub n_head: usize,
    /// Key/value heads (grouped-query attention when fewer than `n_head`).
    pub n_head_kv: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Feed-forward hidden size.
    pub n_ff: usize,
    /// Context length the model was trained for.
    pub n_ctx_train: usize,
    /// RoPE frequency base.
    pub rope_base: f32,
    /// Dimensions of each head that RoPE rotates (may be fewer than `head_dim`).
    pub rope_dims: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Whether the output projection reuses the token embeddings.
    pub tied_embeddings: bool,
    /// Type of the main weight matrices (layer 0's largest projection).
    pub weight_type: GgmlType,
}

/// Typed access to `{arch}.*` metadata keys.
pub(crate) struct Meta<'g, 'a> {
    g: &'g Gguf<'a>,
    arch: &'static str,
}

impl<'g, 'a> Meta<'g, 'a> {
    pub(crate) fn new(g: &'g Gguf<'a>, architecture: Architecture) -> Self {
        Self {
            g,
            arch: architecture.name(),
        }
    }

    /// The full key for `suffix`, e.g. `qwen3.block_count`.
    pub(crate) fn key(&self, suffix: &str) -> String {
        format!("{}.{suffix}", self.arch)
    }

    pub(crate) fn invalid(&self, suffix: &str, reason: impl Into<String>) -> Error {
        Error::InvalidMetadata {
            key: self.key(suffix),
            reason: reason.into(),
        }
    }

    pub(crate) fn usize_opt(&self, suffix: &str) -> Result<Option<usize>> {
        match self.g.get(&self.key(suffix)) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .map(Some)
                .ok_or_else(|| {
                    self.invalid(suffix, format!("expected an unsigned integer, found {v:?}"))
                }),
        }
    }

    pub(crate) fn usize(&self, suffix: &str) -> Result<usize> {
        self.usize_opt(suffix)?
            .ok_or_else(|| Error::MissingMetadata(self.key(suffix)))
    }

    pub(crate) fn f32_or(&self, suffix: &str, default: f32) -> Result<f32> {
        match self.g.get(&self.key(suffix)) {
            None => Ok(default),
            Some(v) => v
                .as_f32()
                .ok_or_else(|| self.invalid(suffix, format!("expected a float, found {v:?}"))),
        }
    }
}

/// Reads the common hyperparameters, with llama.cpp's defaults where a key is optional.
pub(crate) fn read(g: &Gguf<'_>, architecture: Architecture) -> Result<ModelConfig> {
    let m = Meta::new(g, architecture);

    // Zero sizes would make every per-row and per-head loop degenerate.
    let non_zero = |key: &str, value: usize| {
        if value == 0 {
            Err(m.invalid(key, "must be non-zero"))
        } else {
            Ok(value)
        }
    };
    let n_embd = non_zero("embedding_length", m.usize("embedding_length")?)?;
    // Mixture-of-experts models may have no dense feed-forward size; their
    // buffers are then sized by the experts'.
    let n_ff = match m.usize_opt("feed_forward_length")? {
        Some(n) => n,
        None => m
            .usize_opt("expert_feed_forward_length")?
            .ok_or_else(|| Error::MissingMetadata(m.key("feed_forward_length")))?,
    };
    let n_ff = non_zero("feed_forward_length", n_ff)?;
    let n_head = m.usize("attention.head_count")?;
    let n_head_kv = m.usize_opt("attention.head_count_kv")?.unwrap_or(n_head);
    if n_head == 0 || n_head_kv == 0 || n_head % n_head_kv != 0 {
        return Err(m.invalid(
            "attention.head_count_kv",
            "must be non-zero and divide attention.head_count",
        ));
    }
    let head_dim = non_zero(
        "attention.key_length",
        m.usize_opt("attention.key_length")?
            .unwrap_or(n_embd / n_head),
    )?;
    if m.usize_opt("attention.value_length")?
        .is_some_and(|v| v != head_dim)
    {
        return Err(m.invalid("attention.value_length", "must equal attention.key_length"));
    }
    let rope_dims = m.usize_opt("rope.dimension_count")?.unwrap_or(head_dim);
    if rope_dims > head_dim || rope_dims % 2 != 0 {
        return Err(m.invalid(
            "rope.dimension_count",
            "must be even and at most the head size",
        ));
    }
    let block_count = m.usize("block_count")?;
    let nextn = m.usize_opt("nextn_predict_layers")?.unwrap_or(0);
    let n_layer = block_count
        .checked_sub(nextn)
        .ok_or_else(|| m.invalid("nextn_predict_layers", "exceeds block_count"))?;

    let token_embd = g
        .tensor("token_embd.weight")
        .ok_or_else(|| Error::MissingTensor("token_embd.weight".into()))?;
    let n_vocab =
        usize::try_from(token_embd.shape().ne(1)).map_err(|_| Error::InvalidMetadata {
            key: "token_embd.weight".into(),
            reason: "too many rows".into(),
        })?;
    if n_vocab == 0 {
        return Err(Error::InvalidMetadata {
            key: "token_embd.weight".into(),
            reason: "has no rows".into(),
        });
    }
    let weight_type = g
        .tensors()
        .iter()
        .filter(|t| t.name().starts_with("blk.0."))
        .max_by_key(|t| t.shape().n_elements())
        .map_or(token_embd.ggml_type(), |t| t.ggml_type());

    Ok(ModelConfig {
        architecture,
        name: g
            .get("general.name")
            .and_then(|v| v.as_str())
            .map(str::to_owned),
        n_vocab,
        n_embd,
        n_layer,
        n_head,
        n_head_kv,
        head_dim,
        n_ff,
        n_ctx_train: m.usize_opt("context_length")?.unwrap_or(4096),
        rope_base: m.f32_or("rope.freq_base", 10_000.0)?,
        rope_dims,
        rms_eps: m.f32_or("attention.layer_norm_rms_epsilon", 1e-6)?,
        tied_embeddings: g.tensor("output.weight").is_none(),
        weight_type,
    })
}
