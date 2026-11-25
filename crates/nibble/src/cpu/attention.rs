//! Attention on the CPU: the f16 KV cache and the softmax attention over it.

use super::{Cpu, Kernels};
use crate::model::ModelConfig;

/// The attention row is padded to a multiple of this many positions, with
/// masked (zero-weight) entries, as llama.cpp pads it to 256. 32 is the
/// largest step of the kernels involved (ggml's softmax and f16 dot), so the
/// real positions are computed in the same groups as in llama.cpp, and the
/// further padding there adds only exact zeros.
const KV_BLOCK: usize = 32;

/// Keys and values of one attention layer for up to `capacity` positions, in
/// f16 (llama.cpp's default cache type), laid out as llama.cpp lays them out
/// without flash attention: keys by position, values transposed. Both
/// attention products are then contiguous dot products of f16 cache entries
/// with f32 vectors.
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

    /// The keys of KV head `g`: one row of `head_dim` values per position,
    /// rows `stride` bytes apart, as `(bytes from the first row, stride)`.
    fn keys(&self, g: usize) -> (&[u8], usize) {
        (&self.k[2 * g * self.head_dim..], 2 * self.kv_dim)
    }

    /// The value rows from element `i` on: row `i + r` holds that element of
    /// every position, rows `stride` bytes apart, as `(bytes, stride)`.
    fn value_rows(&self, i: usize) -> (&[u8], usize) {
        (&self.vt[2 * i * self.stride..], 2 * self.stride)
    }
}

/// Buffers for [`attend`].
pub(crate) struct AttentionScratch {
    /// Per head: the attention scores, then the probabilities, of every
    /// position, padded to a whole block.
    heads: Vec<Vec<f32>>,
}

impl AttentionScratch {
    pub(crate) fn new(c: &ModelConfig, n_ctx: usize) -> Self {
        let padded = n_ctx.next_multiple_of(KV_BLOCK);
        Self {
            heads: vec![vec![0.0; padded]; c.n_head],
        }
    }
}

/// Causal softmax attention of the query heads `q` over every cached position,
/// with grouped-query sharing (query head `h` reads KV head `h / (n_head / n_head_kv)`).
/// Writes `n_head * head_dim` values to `out`.
///
/// As llama.cpp without flash attention (b12359 and later): the f32 queries
/// meet the f16 keys, the softmax runs over the padded row with the padding
/// masked to `-inf`, and each output value is the dot product of the f32
/// probabilities with a row of transposed f16 values. Heads run in parallel.
pub(crate) fn attend(
    cpu: &Cpu,
    c: &ModelConfig,
    q: &[f32],
    cache: &KvCache,
    s: &mut AttentionScratch,
    out: &mut [f32],
) {
    let hd = c.head_dim;
    let group = c.n_head / c.n_head_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let kernels = cpu.kernels();
    let n_pos = cache.len();
    let padded = n_pos.next_multiple_of(KV_BLOCK);
    cpu.pool()
        .for_each_chunk_pair(out, hd, &mut s.heads, 1, |h, out, head| {
            let kv_head = h / group;
            let q = &q[h * hd..(h + 1) * hd];
            let probs = &mut head[0][..padded];
            let (scores, masked) = probs.split_at_mut(n_pos);
            let (keys, key_stride) = cache.keys(kv_head);
            kernels.dot_f16_f32_rows(keys, key_stride, q, scores);
            for score in scores.iter_mut() {
                *score *= scale;
            }
            masked.fill(f32::NEG_INFINITY);
            kernels.softmax(probs);
            let (values, value_stride) = cache.value_rows(kv_head * hd);
            kernels.dot_f16_f32_rows(values, value_stride, probs, out);
        });
}
