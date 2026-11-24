//! CPU reference operations.
//!
//! Plain Rust, written for correctness. Matrix-vector products are split
//! across scoped threads by rows, so results do not depend on the thread count.
//! The SIMD kernels and persistent thread pool of Phase 4 replace this.

use std::thread;

use nibble_gguf::GgmlType;

use crate::quant;

/// A weight matrix inside the model bytes: `rows` rows of `cols` elements each
/// (ggml shape `[cols, rows]`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Weight {
    pub(crate) ty: GgmlType,
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    /// Absolute byte offset of row 0 in the model storage.
    pub(crate) offset: usize,
    pub(crate) row_bytes: usize,
}

impl Weight {
    /// The encoded bytes of row `r`.
    pub(crate) fn row<'a>(&self, bytes: &'a [u8], r: usize) -> &'a [u8] {
        let start = self.offset + r * self.row_bytes;
        &bytes[start..start + self.row_bytes]
    }

    /// Decodes row `r` into `out` (`cols` elements), e.g. an embedding lookup.
    pub(crate) fn dequantize_row(&self, bytes: &[u8], r: usize, out: &mut [f32]) {
        quant::dequantize_row(self.ty, self.row(bytes, r), out);
    }
}

/// Smallest matrix (in multiply-adds) worth splitting across threads.
const MIN_WORK_PER_THREAD: usize = 1 << 16;

/// `out[r] = dot(row r of w, x)` for every row of `w`.
///
/// `x` is first encoded in the weight type's dot type (see [`quant`]) into
/// the scratch buffer `xq`.
pub(crate) fn matvec(
    bytes: &[u8],
    w: &Weight,
    x: &[f32],
    out: &mut [f32],
    xq: &mut Vec<u8>,
    threads: usize,
) {
    debug_assert_eq!(x.len(), w.cols);
    debug_assert_eq!(out.len(), w.rows);
    quant::quantize_activation(w.ty, x, xq);
    let x = &xq[..];
    let threads = threads.clamp(1, (w.rows * w.cols / MIN_WORK_PER_THREAD).max(1));
    if threads == 1 {
        matvec_rows(bytes, w, x, out, 0);
        return;
    }
    let chunk = w.rows.div_ceil(threads);
    thread::scope(|s| {
        let mut chunks = out.chunks_mut(chunk).enumerate();
        let first = chunks.next();
        for (i, out) in chunks {
            s.spawn(move || matvec_rows(bytes, w, x, out, i * chunk));
        }
        // The calling thread takes the first chunk instead of idling.
        if let Some((_, out)) = first {
            matvec_rows(bytes, w, x, out, 0);
        }
    });
}

fn matvec_rows(bytes: &[u8], w: &Weight, x: &[u8], out: &mut [f32], first_row: usize) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = quant::dot_row(w.ty, w.row(bytes, first_row + i), x);
    }
}

/// RMSNorm: `out = x / sqrt(mean(x²) + eps) * weight`, summed in f64 as ggml does.
pub(crate) fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let scale = rms_scale(x, eps);
    for ((o, &v), &w) in out.iter_mut().zip(x).zip(weight) {
        *o = v * scale * w;
    }
}

/// RMSNorm in place, applied to each `weight.len()`-sized chunk of `x` (per-head norms).
pub(crate) fn rmsnorm_chunks(x: &mut [f32], weight: &[f32], eps: f32) {
    for chunk in x.chunks_exact_mut(weight.len()) {
        let scale = rms_scale(chunk, eps);
        for (v, &w) in chunk.iter_mut().zip(weight) {
            *v = *v * scale * w;
        }
    }
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let sum: f64 = x.iter().map(|&v| f64::from(v * v)).sum();
    let mean = (sum / x.len() as f64) as f32;
    1.0 / (mean + eps).sqrt()
}

/// Fills `table` with `(cos θ_i, sin θ_i)` for `θ_i = pos · base^(-2i/n_rot)`,
/// computed iteratively in f32 exactly as ggml's rope cache does.
pub(crate) fn rope_table(table: &mut Vec<(f32, f32)>, pos: usize, n_rot: usize, base: f32) {
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let mut theta = pos as f32;
    table.clear();
    for _ in 0..n_rot / 2 {
        table.push((theta.cos(), theta.sin()));
        theta *= theta_scale;
    }
}

/// NeoX-style RoPE: rotates pairs `(i, i + n_rot/2)` of every head in `x`.
pub(crate) fn rope_neox(x: &mut [f32], head_dim: usize, table: &[(f32, f32)]) {
    let half = table.len();
    for head in x.chunks_exact_mut(head_dim) {
        let (lo, hi) = head[..2 * half].split_at_mut(half);
        for ((a, b), &(cos, sin)) in lo.iter_mut().zip(hi.iter_mut()).zip(table) {
            let (x0, x1) = (*a, *b);
            *a = x0 * cos - x1 * sin;
            *b = x0 * sin + x1 * cos;
        }
    }
}

/// Numerically stable softmax in place.
pub(crate) fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f64;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += f64::from(*v);
    }
    let inv = (1.0 / sum) as f32;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// `a · b` for f16 vectors, accumulated in f32 with eight accumulators.
pub(crate) fn dot_f16(a: &[u16], b: &[u16]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        acc[i % 8] += quant::f16_to_f32(x) * quant::f16_to_f32(y);
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// `out += alpha * x` for an f16 `x`.
pub(crate) fn axpy_f16(out: &mut [f32], alpha: f32, x: &[u16]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += alpha * quant::f16_to_f32(v);
    }
}

/// `out += x`.
pub(crate) fn add_in_place(out: &mut [f32], x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += v;
    }
}

/// SiLU (swish): `x · sigmoid(x)`.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Logistic sigmoid.
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, linear above 20 as in ggml.
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// `a · b`, with eight accumulators.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        acc[i % 8] += x * y;
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// `out += alpha * x`.
pub(crate) fn axpy(out: &mut [f32], alpha: f32, x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += alpha * v;
    }
}

/// L2-normalizes each `n`-element chunk of `x`: `x / sqrt(Σx² + eps)`.
///
/// Computed as llama.cpp's DeltaNet does it: an RMSNorm with `eps / n`, then a
/// scale by `1 / sqrt(n)`.
pub(crate) fn l2_norm_chunks(x: &mut [f32], n: usize, eps: f32) {
    let inv_sqrt_n = 1.0 / (n as f32).sqrt();
    for chunk in x.chunks_exact_mut(n) {
        let scale = rms_scale(chunk, eps / n as f32);
        for v in chunk.iter_mut() {
            *v = *v * scale * inv_sqrt_n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_normalizes_to_unit_rms() {
        let x = [3.0f32, -4.0, 0.0, 5.0];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &[1.0; 4], 0.0);
        let rms = (out.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-6);
        assert!((out[1] / out[0] + 4.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn rope_is_identity_at_position_zero_and_preserves_norms() {
        let mut table = Vec::new();
        let head: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();

        rope_table(&mut table, 0, 8, 10_000.0);
        let mut x = head.clone();
        rope_neox(&mut x, 8, &table);
        assert_eq!(x, head);

        rope_table(&mut table, 17, 8, 10_000.0);
        let mut x = head.clone();
        rope_neox(&mut x, 8, &table);
        for i in 0..4 {
            let before = head[i].hypot(head[i + 4]);
            let after = x[i].hypot(x[i + 4]);
            assert!((before - after).abs() < 1e-5);
        }
        // Pair 0 rotates by exactly `pos` radians.
        let (sin, cos) = 17.0f32.sin_cos();
        assert!((x[0] - (head[0] * cos - head[4] * sin)).abs() < 1e-5);
        assert!((x[4] - (head[0] * sin + head[4] * cos)).abs() < 1e-5);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 1000.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > 0.999);
    }

    #[test]
    fn threaded_matvec_matches_single_threaded() {
        let (rows, cols) = (1000, 512);
        let data: Vec<u8> = (0..rows * cols)
            .flat_map(|i| (((i * 7919) % 97) as f32 / 97.0 - 0.5).to_le_bytes())
            .collect();
        let w = Weight {
            ty: GgmlType::F32,
            cols,
            rows,
            offset: 0,
            row_bytes: cols * 4,
        };
        let x: Vec<f32> = (0..cols).map(|i| (i as f32).sin()).collect();
        let mut one = vec![0.0; rows];
        let mut many = vec![0.0; rows];
        let mut xq = Vec::new();
        matvec(&data, &w, &x, &mut one, &mut xq, 1);
        matvec(&data, &w, &x, &mut many, &mut xq, 7);
        assert_eq!(one, many);
    }
}
