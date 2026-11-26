//! Attention on the CPU: the KV cache in any of llama.cpp's cache types and
//! the softmax attention over it.

use super::{Cpu, Kernels};
use crate::backend::{KvCacheType, KvTypes};
use crate::model::ModelConfig;
use crate::quant::{self, DotType, blocks32};

/// The attention row is padded to a multiple of this many positions, with
/// masked (zero-weight) entries, as llama.cpp pads it to 256. 32 is the
/// largest step of the kernels involved (ggml's softmax and f16 dot), so the
/// real positions are computed in the same groups as in llama.cpp, and the
/// further padding there adds only exact zeros.
const KV_BLOCK: usize = 32;

/// Writes `x` to `out` as `ty`, as ggml's CPU backend stores it in a cache
/// of that type (the type's `from_float`: the NEON `quantize_row_q8_0` and
/// f16/bf16 conversions, the reference quantizers of the other types).
fn encode(kernels: Kernels, ty: KvCacheType, x: &[f32], out: &mut [u8]) {
    match ty {
        KvCacheType::F32 => {
            for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(x) {
                *o = v.to_le_bytes();
            }
        }
        KvCacheType::F16 => kernels.to_f16(x, out),
        KvCacheType::Bf16 => kernels.encode(DotType::Bf16, x, out),
        KvCacheType::Q8_0 => kernels.encode(DotType::Q8_0, x, out),
        other => blocks32::quantize_row_ref(other.ggml_type(), x, out),
    }
}

/// `a · b` for two bf16 vectors, as `ggml_vec_dot_bf16` computes it on
/// AArch64, where it has no SIMD path: each product in f32, summed in f64
/// in order. (nibble's bf16 weight kernels are faster and round
/// differently; the cache follows llama.cpp exactly.)
fn dot_bf16(a: &[u8], b: &[u8]) -> f32 {
    let value = |p: &[u8; 2]| quant::bf16_to_f32(u16::from_le_bytes(*p));
    let sum = a
        .as_chunks::<2>()
        .0
        .iter()
        .zip(b.as_chunks::<2>().0)
        .fold(0.0f64, |s, (x, y)| s + f64::from(value(x) * value(y)));
    sum as f32
}

/// `out[r] = row r · xq` for rows of a cache type, `xq` encoded in its dot
/// type: the weight kernels, except bf16 rows (see [`dot_bf16`]).
fn dot_rows(
    kernels: Kernels,
    ty: KvCacheType,
    rows: &[u8],
    row_bytes: usize,
    xq: &[u8],
    out: &mut [f32],
) {
    if ty == KvCacheType::Bf16 {
        for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
            *o = dot_bf16(row, xq);
        }
    } else {
        kernels.dot_rows(ty.ggml_type(), rows, row_bytes, xq, out);
    }
}

/// Keys and values of one attention layer for up to `capacity` positions,
/// in the cache types asked for, laid out for llama.cpp's two attention
/// paths:
///
/// - Values of a float type (f32, f16, bf16), for its path without flash
///   attention: keys by position, values transposed (row `i` holds element
///   `i` of every position's value), so both attention products are dot
///   products of cache rows with f32 (or bf16) vectors.
/// - Quantized values, for its flash-attention path: keys and values by
///   position, one row of a head's values per position.
///
/// Rows by position are stored head by head (each KV head's rows are
/// contiguous), which changes no arithmetic.
///
/// The buffers are allocated for the whole context up front, as llama.cpp
/// does, so the forward pass never reallocates them. The OS backs their
/// zeroed pages with memory only when they are written, though Windows
/// counts the whole allocation against its commit limit. Writing the first
/// position touches one page per value row of a transposed cache.
pub(crate) struct KvCache {
    types: KvTypes,
    /// For each KV head, `capacity` rows of `k_row` bytes.
    k: Vec<u8>,
    /// Transposed: `kv_dim` rows of `stride` values of `v_elem` bytes, whose
    /// entries past `len` are zero. By position: as `k`, rows of `v_row`.
    v: Vec<u8>,
    /// A transposed value being written, before it is scattered into `v`.
    staging: Vec<u8>,
    head_dim: usize,
    capacity: usize,
    /// `capacity` rounded up to a whole block.
    stride: usize,
    k_row: usize,
    v_row: usize,
    /// Bytes per value of a transposed cache.
    v_elem: usize,
    len: usize,
}

impl KvCache {
    pub(crate) fn new(c: &ModelConfig, capacity: usize, types: KvTypes) -> Self {
        let kv_dim = c.n_head_kv * c.head_dim;
        let stride = capacity.next_multiple_of(KV_BLOCK);
        let k_row = types.stored_k().bytes(c.head_dim);
        let v_row = types.v.bytes(c.head_dim);
        let v_elem = types.v.bytes(1);
        let v = if types.flash() {
            vec![0; c.n_head_kv * capacity * v_row]
        } else {
            vec![0; kv_dim * stride * v_elem]
        };
        Self {
            types,
            k: vec![0; c.n_head_kv * capacity * k_row],
            v,
            staging: vec![0; if types.flash() { 0 } else { kv_dim * v_elem }],
            head_dim: c.head_dim,
            capacity,
            stride,
            k_row,
            v_row,
            v_elem,
            len: 0,
        }
    }

    /// Bytes of the keys and values.
    pub(crate) fn bytes(&self) -> usize {
        self.k.len() + self.v.len()
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
        let hd = self.head_dim;
        let k_type = self.types.stored_k();
        for (g, head) in k.chunks_exact(hd).enumerate() {
            let at = (g * self.capacity + t) * self.k_row;
            encode(kernels, k_type, head, &mut self.k[at..at + self.k_row]);
        }
        if self.types.flash() {
            for (g, head) in v.chunks_exact(hd).enumerate() {
                let at = (g * self.capacity + t) * self.v_row;
                encode(
                    kernels,
                    self.types.v,
                    head,
                    &mut self.v[at..at + self.v_row],
                );
            }
        } else {
            let es = self.v_elem;
            encode(kernels, self.types.v, v, &mut self.staging);
            for (i, value) in self.staging.chunks_exact(es).enumerate() {
                let at = es * (i * self.stride + t);
                self.v[at..at + es].copy_from_slice(value);
            }
        }
        self.len += 1;
    }

    /// Keeps the first `n_pos` positions.
    pub(crate) fn truncate(&mut self, n_pos: usize) {
        if n_pos >= self.len {
            return;
        }
        if !self.types.flash() {
            // The value product reads the dropped entries of the last block,
            // weighted by zero, so they must not hold an infinity.
            let es = self.v_elem;
            for row in self.v.chunks_exact_mut(es * self.stride) {
                row[es * n_pos..es * self.len].fill(0);
            }
        }
        self.len = n_pos;
    }

    /// The first `n_pos` cached keys of KV head `g`, rows of `k_row` bytes.
    fn keys_up_to(&self, g: usize, n_pos: usize) -> &[u8] {
        let at = g * self.capacity * self.k_row;
        &self.k[at..at + n_pos * self.k_row]
    }

    /// Row `i` of the transposed values: element `i` of every position.
    fn value_row(&self, i: usize) -> &[u8] {
        let at = self.v_elem * i * self.stride;
        &self.v[at..at + self.v_elem * self.stride]
    }

    /// The value of KV head `g` at position `p`, stored by position.
    fn value_at(&self, g: usize, p: usize) -> &[u8] {
        let at = (g * self.capacity + p) * self.v_row;
        &self.v[at..at + self.v_row]
    }
}

/// One head's buffers for [`attend`].
struct HeadScratch {
    /// The attention scores, then the probabilities, of every position,
    /// padded to a whole block.
    probs: Vec<f32>,
    /// The query, or the probabilities, encoded for a dot type.
    encoded: Vec<u8>,
    /// Flash attention's output accumulator and dequantized value.
    acc: Vec<f32>,
    value: Vec<f32>,
}

/// Buffers for [`attend`], one set per query head.
pub(crate) struct AttentionScratch {
    heads: Vec<HeadScratch>,
}

impl AttentionScratch {
    pub(crate) fn new(c: &ModelConfig, n_ctx: usize) -> Self {
        let padded = n_ctx.next_multiple_of(KV_BLOCK);
        let heads = (0..c.n_head)
            .map(|_| HeadScratch {
                probs: vec![0.0; padded],
                encoded: vec![0; 4 * padded.max(c.head_dim)],
                acc: vec![0.0; c.head_dim],
                value: vec![0.0; c.head_dim],
            })
            .collect();
        Self { heads }
    }
}

/// Causal softmax attention for one query head over cached positions `0..n_pos`.
///
/// Each query head meets its keys as ggml's CPU `mul_mat` does: f16 and f32
/// keys meet the f32 query, the others the query encoded in their dot type
/// (`Q8_0`, `Q8_1`, `BF16`). Then, as llama.cpp (b12359 and later):
///
/// - Without flash attention (values of a float type): the softmax runs
///   over the padded row with the padding masked to `-inf`, and each output
///   value is the dot product of the probabilities (in the values' dot
///   type) with a row of transposed values.
/// - With flash attention (quantized values): ggml's
///   `ggml_compute_forward_flash_attn_ext_f16_one_chunk`, which every row
///   takes for these types: an online softmax over the positions in order,
///   each value dequantized and added with a fused multiply-add.
#[expect(clippy::too_many_arguments, reason = "one head attention step")]
fn attend_head(
    kernels: Kernels,
    cache: &KvCache,
    kv_head: usize,
    hd: usize,
    scale: f32,
    n_pos: usize,
    q: &[f32],
    head: &mut HeadScratch,
    out: &mut [f32],
) {
    let padded = n_pos.next_multiple_of(KV_BLOCK);
    let types = cache.types;
    let k_type = types.stored_k();
    let keys = cache.keys_up_to(kv_head, n_pos);
    let scores = &mut head.probs[..n_pos];
    if k_type == KvCacheType::F16 {
        kernels.dot_f16_f32_rows(keys, cache.k_row, q, scores);
    } else {
        let ty = k_type.ggml_type();
        let dt = DotType::of(ty);
        let xq = &mut head.encoded[..dt.encoded_len(hd)];
        kernels.encode(dt, q, xq);
        dot_rows(kernels, k_type, keys, cache.k_row, xq, scores);
    }
    if types.flash() {
        return flash(cache, kv_head, n_pos, scale, head, out);
    }
    let probs = &mut head.probs[..padded];
    let (scores, masked) = probs.split_at_mut(n_pos);
    for score in scores.iter_mut() {
        *score *= scale;
    }
    masked.fill(f32::NEG_INFINITY);
    kernels.softmax(probs);
    let es = cache.v_elem;
    if types.v == KvCacheType::F16 {
        let values = &cache.v[es * kv_head * hd * cache.stride..];
        kernels.dot_f16_f32_rows(values, es * cache.stride, probs, out);
        return;
    }
    let ty = types.v.ggml_type();
    let dt = DotType::of(ty);
    let xq = &mut head.encoded[..dt.encoded_len(padded)];
    kernels.encode(dt, probs, xq);
    for (i, o) in out.iter_mut().enumerate() {
        let row = &cache.value_row(kv_head * hd + i)[..es * padded];
        dot_rows(
            kernels,
            types.v,
            row,
            es * padded,
            xq,
            std::slice::from_mut(o),
        );
    }
}

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
    cpu.pool()
        .for_each_chunk_pair(out, hd, &mut s.heads, 1, |h, out, head| {
            attend_head(
                kernels,
                cache,
                h / group,
                hd,
                scale,
                n_pos,
                &q[h * hd..(h + 1) * hd],
                &mut head[0],
                out,
            );
        });
}

/// Batched causal softmax attention for `n` tokens whose keys and values have
/// already been appended at `start_pos..start_pos + n` in `cache`: one parallel
/// pass over query heads (instead of `n` thread-pool barriers), also applying
/// `rot_v` inverse Hadamard and `gated` sigmoid in the same pass.
#[expect(clippy::too_many_arguments, reason = "one batched attention pass")]
pub(crate) fn attend_batch(
    cpu: &Cpu,
    c: &ModelConfig,
    q: &[f32],
    q_stride: usize,
    gated: bool,
    start_pos: usize,
    n: usize,
    cache: &KvCache,
    rot_v: Option<usize>,
    s: &mut AttentionScratch,
    out: &mut [f32],
) {
    let hd = c.head_dim;
    let group = c.n_head / c.n_head_kv;
    let scale = 1.0 / (hd as f32).sqrt();
    let kernels = cpu.kernels();
    let q_row = c.n_head * q_stride;
    let out_split = super::Split {
        segments: n,
        chunk_len: hd,
    };
    let head_split = super::Split {
        segments: 1,
        chunk_len: 1,
    };
    cpu.pool().for_each_segment_chunk_pair(
        out,
        out_split,
        &mut s.heads,
        head_split,
        |h, out_h, head| {
            let head = &mut head.segment(0)[0];
            let kv_head = h / group;
            for t in 0..n {
                let q_at = t * q_row + h * q_stride;
                let o = out_h.segment(t);
                attend_head(
                    kernels,
                    cache,
                    kv_head,
                    hd,
                    scale,
                    start_pos + t + 1,
                    &q[q_at..q_at + hd],
                    head,
                    o,
                );
                if let Some(rot) = rot_v {
                    super::hadamard(o, rot);
                }
                if gated {
                    for (a, &g) in o.iter_mut().zip(&q[q_at + hd..q_at + 2 * hd]) {
                        *a *= super::sigmoid(g);
                    }
                }
            }
        },
    );
}

/// The flash-attention loop of one query head over the scores in
/// `head.probs` (ggml's online softmax, `ggml_vec_scale_f32` and
/// `ggml_vec_mad_f32`, the latter fused per value).
fn flash(
    cache: &KvCache,
    kv_head: usize,
    n_pos: usize,
    scale: f32,
    head: &mut HeadScratch,
    out: &mut [f32],
) {
    let ty = cache.types.v.ggml_type();
    let acc = &mut head.acc[..];
    let value = &mut head.value[..];
    acc.fill(0.0);
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    for (p, &score) in head.probs[..n_pos].iter().enumerate() {
        // The mask of a visible position adds zero.
        let s = score * scale;
        let (ms, vs) = if s > max {
            let ms = (max - s).exp();
            max = s;
            for a in acc.iter_mut() {
                *a *= ms;
            }
            (ms, 1.0)
        } else {
            (1.0, (s - max).exp())
        };
        quant::dequantize_row(ty, cache.value_at(kv_head, p), value);
        for (a, &v) in acc.iter_mut().zip(value.iter()) {
            *a = v.mul_add(vs, *a);
        }
        sum = sum.mul_add(ms, vs);
    }
    let inv = if sum == 0.0 { 0.0 } else { 1.0 / sum };
    for (o, &a) in out.iter_mut().zip(acc.iter()) {
        *o = a * inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The inputs of the C program the checksums below come from: 64
    /// blocks of 32 at magnitudes from 1e-3 to 1e3, with all-zero and
    /// constant blocks.
    fn inputs() -> Vec<f32> {
        let mut s = 12345u32;
        (0..32 * 64)
            .map(|i| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let b = i / 32;
                let mut f = (s >> 8) as f32 / 16_777_216.0 - 0.5;
                if b % 11 == 5 {
                    f = 0.0;
                }
                if b % 13 == 7 {
                    f = 0.25;
                }
                f * 10f32.powf((b % 7) as f32 - 3.0)
            })
            .collect()
    }

    fn fnv(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Every cache type is stored as ggml's CPU backend stores it: the FNV-1a
    /// hashes of `ggml_get_type_traits_cpu(type)->from_float` on the same
    /// inputs, from a program linked against llama.cpp b12722's
    /// `libggml-cpu` on an M4 Max.
    #[test]
    fn cache_encoders_match_ggml() {
        let x = inputs();
        let expected = [
            (KvCacheType::Q4_0, 0x09c8_4296_f77e_dd36),
            (KvCacheType::Q4_1, 0x3b3c_1b1e_c125_9a86),
            (KvCacheType::Q5_0, 0x0cae_cf80_f4ce_70a5),
            (KvCacheType::Q5_1, 0x657b_7586_7b65_fc49),
            (KvCacheType::Iq4Nl, 0xa003_2fba_53fa_b7eb),
            (KvCacheType::Q8_0, 0xf0b0_4d98_c1e9_385e),
            (KvCacheType::Bf16, 0xa4ae_624b_385e_cc7c),
            (KvCacheType::F16, 0x5de1_e993_7746_94a5),
        ];
        for kernels in [Kernels::Scalar, Kernels::detect()] {
            for (ty, hash) in expected {
                let mut out = vec![0; ty.bytes(x.len())];
                encode(kernels, ty, &x, &mut out);
                assert_eq!(fnv(&out), hash, "{ty} with {} kernels", kernels.name());
            }
        }
    }

    /// Each encoder decodes back to its input within the type's precision.
    #[test]
    fn cache_types_round_trip() {
        let x = inputs();
        for ty in KvCacheType::ALL {
            let mut bytes = vec![0; ty.bytes(x.len())];
            encode(Kernels::detect(), ty, &x, &mut bytes);
            let mut back = vec![0.0; x.len()];
            quant::dequantize_row(ty.ggml_type(), &bytes, &mut back);
            for (block, (xb, yb)) in x.chunks(32).zip(back.chunks(32)).enumerate() {
                let amax = xb.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let tol = match ty {
                    KvCacheType::F32 => 0.0,
                    KvCacheType::F16 => 1e-3,
                    KvCacheType::Bf16 => 8e-3,
                    KvCacheType::Q8_0 => 1e-2,
                    KvCacheType::Q5_0 | KvCacheType::Q5_1 => 7e-2,
                    _ => 0.15,
                } * amax
                    // A block scale below f16's normal range keeps fewer bits.
                    + if ty.is_quantized() { 128.0 * 6e-8 } else { 0.0 };
                for (a, b) in xb.iter().zip(yb) {
                    assert!((a - b).abs() <= tol, "{ty} block {block}: {a} -> {b}");
                }
            }
        }
    }
}
