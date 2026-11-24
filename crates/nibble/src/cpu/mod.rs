//! CPU execution: the thread pool, the vector kernels, and the ops the
//! models are written in.
//!
//! - [`Cpu`] is what a session computes with: its [`ThreadPool`] and the
//!   [`Kernels`] chosen for this CPU. It provides the parallel matrix-vector
//!   products: single, grouped (several matrices that share an input, in one
//!   parallel pass), and the fused SwiGLU feed-forward.
//! - [`Kernels`] dispatches each vector kernel to its SIMD implementation
//!   (`avx2`) or to the scalar reference ([`scalar`], and [`crate::quant`]
//!   for the weight formats), which is also the correctness oracle.
//! - Cheap element-wise ops (norms, RoPE, residual adds) are plain functions
//!   here.
//!
//! Results do not depend on the thread count: work is split by output rows
//! or heads, and each output value is computed by one kernel call, whichever
//! thread makes it.

#[cfg(target_arch = "x86_64")]
mod avx2;
mod kernels;
mod pool;
pub(crate) mod scalar;
mod topology;

use std::ops::Range;

use nibble_gguf::GgmlType;

pub(crate) use kernels::Kernels;
pub(crate) use pool::ThreadPool;
pub(crate) use scalar::DeltaStep;
pub(crate) use topology::default_threads;

use crate::quant::{self, DotType};

/// Products smaller than this (in multiply-adds) run on the calling thread:
/// handing them to the pool costs more than it saves.
const PARALLEL_MIN_MACS: usize = 1 << 15;

/// Tasks per thread when a product is split, so that a thread that is
/// descheduled or slower (an SMT sibling) delays the product by a fraction
/// of its share.
const TASKS_PER_THREAD: usize = 8;

/// Output rows per task are a multiple of this: 16 f32 values are a 64-byte
/// cache line, so two tasks never write the same line.
const ROW_ALIGN: usize = 16;

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
    /// The encoded bytes of the rows in `rows`.
    pub(crate) fn rows_bytes<'a>(&self, bytes: &'a [u8], rows: Range<usize>) -> &'a [u8] {
        &bytes[self.offset + rows.start * self.row_bytes..self.offset + rows.end * self.row_bytes]
    }

    /// Decodes row `r` into `out` (`cols` elements), e.g. an embedding lookup.
    pub(crate) fn dequantize_row(&self, bytes: &[u8], r: usize, out: &mut [f32]) {
        quant::dequantize_row(self.ty, self.rows_bytes(bytes, r..r + 1), out);
    }
}

/// The input of a matrix product, encoded in the dot type of each weight
/// type it is multiplied with (see [`quant`]). Reused across products, so
/// encoding allocates only while the buffers grow to the largest input.
#[derive(Debug, Default)]
pub(crate) struct EncodedInput {
    by_dot_type: [Vec<u8>; DotType::COUNT],
}

impl EncodedInput {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Encodes `x` in the dot type of each weight type in `types`, once per
    /// dot type.
    fn encode(&mut self, kernels: Kernels, x: &[f32], types: impl IntoIterator<Item = GgmlType>) {
        let mut done = [false; DotType::COUNT];
        for dt in types.into_iter().map(DotType::of) {
            if std::mem::replace(&mut done[dt.index()], true) {
                continue;
            }
            let buf = &mut self.by_dot_type[dt.index()];
            buf.resize(dt.encoded_len(x.len()), 0);
            kernels.encode(dt, x, buf);
        }
    }

    /// The input as encoded for weights of type `ty`.
    fn get(&self, ty: GgmlType) -> &[u8] {
        &self.by_dot_type[DotType::of(ty).index()]
    }
}

/// The threads and kernels a session computes with.
#[derive(Debug)]
pub(crate) struct Cpu {
    pool: ThreadPool,
    kernels: Kernels,
}

impl Cpu {
    /// `threads` threads (the calling thread included) and the fastest
    /// kernels this CPU supports.
    pub(crate) fn new(threads: usize) -> Self {
        Self::with_kernels(threads, Kernels::detect())
    }

    /// `threads` threads and the given kernels.
    pub(crate) fn with_kernels(threads: usize, kernels: Kernels) -> Self {
        Self {
            pool: ThreadPool::new(threads),
            kernels,
        }
    }

    /// Threads that run ops, the calling thread included. Fewer than asked
    /// for if the system could not start them all.
    pub(crate) fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub(crate) const fn kernels(&self) -> Kernels {
        self.kernels
    }

    pub(crate) const fn pool(&self) -> &ThreadPool {
        &self.pool
    }

    /// `out = w · x`.
    pub(crate) fn matvec(
        &self,
        bytes: &[u8],
        w: &Weight,
        x: &[f32],
        out: &mut [f32],
        input: &mut EncodedInput,
    ) {
        self.matvecs(bytes, x, input, [(w, out)]);
    }

    /// `out_j = w_j · x` for every `(w_j, out_j)` in `products`, in one
    /// parallel pass: `x` is encoded once per dot type, and the threads meet
    /// once instead of once per matrix.
    pub(crate) fn matvecs<const N: usize>(
        &self,
        bytes: &[u8],
        x: &[f32],
        input: &mut EncodedInput,
        products: [(&Weight, &mut [f32]); N],
    ) {
        let weights = products.each_ref().map(|(w, _)| *w);
        let outs = products.map(|(_, out)| out);
        for (w, out) in weights.iter().zip(&outs) {
            debug_assert_eq!((w.cols, w.rows), (x.len(), out.len()));
        }
        input.encode(self.kernels, x, weights.iter().map(|w| w.ty));
        let input = &*input;
        let kernels = self.kernels;
        let product = |j: usize, first_row: usize, out: &mut [f32]| {
            let w = weights[j];
            let rows = w.rows_bytes(bytes, first_row..first_row + out.len());
            kernels.dot_rows(w.ty, rows, w.row_bytes, input.get(w.ty), out);
        };
        let rows: usize = weights.iter().map(|w| w.rows).sum();
        if !self.worth_splitting(rows, x.len()) {
            for (j, out) in outs.into_iter().enumerate() {
                product(j, 0, out);
            }
            return;
        }
        let chunk = self.rows_per_task(rows);
        self.pool
            .for_each_chunk(outs, chunk, |j, c, out| product(j, c * chunk, out));
    }

    /// `out = silu(gate · x) * (up · x)`: the two products of a SwiGLU
    /// feed-forward and ggml's `swiglu` of them, in one parallel pass. Each
    /// task computes the same rows of both products, then their SwiGLU;
    /// `up_out` receives `up · x`.
    #[expect(
        clippy::too_many_arguments,
        reason = "two weights and their two outputs"
    )]
    pub(crate) fn swiglu(
        &self,
        bytes: &[u8],
        gate: &Weight,
        up: &Weight,
        x: &[f32],
        input: &mut EncodedInput,
        out: &mut [f32],
        up_out: &mut [f32],
    ) {
        debug_assert_eq!((gate.cols, gate.rows), (x.len(), out.len()));
        debug_assert_eq!((up.cols, up.rows), (x.len(), up_out.len()));
        input.encode(self.kernels, x, [gate.ty, up.ty]);
        let input = &*input;
        let kernels = self.kernels;
        let task = |first_row: usize, g: &mut [f32], u: &mut [f32]| {
            let rows = first_row..first_row + g.len();
            let gate_rows = gate.rows_bytes(bytes, rows.clone());
            kernels.dot_rows(gate.ty, gate_rows, gate.row_bytes, input.get(gate.ty), g);
            let up_rows = up.rows_bytes(bytes, rows);
            kernels.dot_rows(up.ty, up_rows, up.row_bytes, input.get(up.ty), u);
            // Chunks are whole groups of 8 (ROW_ALIGN), so the SIMD kernels
            // take the same elements as ggml's one pass over the whole vector.
            kernels.swiglu(g, u);
        };
        if !self.worth_splitting(2 * gate.rows, x.len()) {
            task(0, out, up_out);
            return;
        }
        let chunk = self.rows_per_task(2 * gate.rows);
        self.pool
            .for_each_chunk_pair(out, chunk, up_out, chunk, |i, g, u| task(i * chunk, g, u));
    }

    fn worth_splitting(&self, rows: usize, cols: usize) -> bool {
        self.threads() > 1 && rows.saturating_mul(cols) >= PARALLEL_MIN_MACS
    }

    /// Rows per task for a product with `rows` output rows in all.
    fn rows_per_task(&self, rows: usize) -> usize {
        rows.div_ceil(TASKS_PER_THREAD * self.threads())
            .next_multiple_of(ROW_ALIGN)
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

/// `out += x`.
pub(crate) fn add_in_place(out: &mut [f32], x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += v;
    }
}

/// Logistic sigmoid (scalar in ggml too).
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `ln(1 + e^x)`, linear above 20 as in ggml.
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
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

    /// Weight matrices laid out back to back, as in a model file.
    struct Matrices {
        bytes: Vec<u8>,
        weights: Vec<Weight>,
    }

    fn matrices(shapes: &[(GgmlType, usize, usize)]) -> Matrices {
        let mut bytes = Vec::new();
        let mut weights = Vec::new();
        for (i, &(ty, cols, rows)) in shapes.iter().enumerate() {
            let offset = bytes.len();
            for r in 0..rows {
                let row: Vec<f32> = (0..cols)
                    .map(|c| ((r * 7919 + c * 104_729 + i * 31) % 97) as f32 / 97.0 - 0.5)
                    .collect();
                bytes.extend(quant::encoded(DotType::of(ty), &row));
            }
            weights.push(Weight {
                ty,
                cols,
                rows,
                offset,
                row_bytes: (bytes.len() - offset) / rows,
            });
        }
        Matrices { bytes, weights }
    }

    fn input(cols: usize) -> Vec<f32> {
        (0..cols).map(|i| (i as f32 * 0.37).sin()).collect()
    }

    /// Every output row computed one at a time with the scalar reference.
    fn reference(m: &Matrices, w: &Weight, x: &[f32]) -> Vec<f32> {
        let xq = quant::encoded(DotType::of(w.ty), x);
        (0..w.rows)
            .map(|r| quant::dot_row(w.ty, w.rows_bytes(&m.bytes, r..r + 1), &xq))
            .collect()
    }

    #[test]
    fn products_do_not_depend_on_the_thread_count() {
        let m = matrices(&[(GgmlType::Q8_0, 512, 1000)]);
        let w = &m.weights[0];
        let x = input(512);
        let mut results = Vec::new();
        for threads in [1, 2, 7] {
            let cpu = Cpu::new(threads);
            let mut out = vec![0.0; w.rows];
            cpu.matvec(&m.bytes, w, &x, &mut out, &mut EncodedInput::new());
            results.push(out);
        }
        assert_eq!(results[0], results[1]);
        assert_eq!(results[0], results[2]);
    }

    /// A group with mixed weight types needs one encoding per dot type, and
    /// each output must match the scalar reference row by row.
    #[test]
    fn grouped_products_of_mixed_types_match_the_reference() {
        let cols = 256;
        let m = matrices(&[
            (GgmlType::Q8_0, cols, 300),
            (GgmlType::F16, cols, 70),
            (GgmlType::Q8_0, cols, 16),
            (GgmlType::Bf16, cols, 129),
        ]);
        let x = input(cols);
        for kernels in [Kernels::Scalar, Kernels::detect()] {
            let cpu = Cpu::with_kernels(4, kernels);
            let mut outs: Vec<Vec<f32>> = m.weights.iter().map(|w| vec![0.0; w.rows]).collect();
            let [a, b, c, d] = &mut outs[..] else {
                unreachable!("four matrices")
            };
            let w = &m.weights;
            cpu.matvecs(
                &m.bytes,
                &x,
                &mut EncodedInput::new(),
                [
                    (&w[0], &mut a[..]),
                    (&w[1], &mut b[..]),
                    (&w[2], &mut c[..]),
                    (&w[3], &mut d[..]),
                ],
            );
            for (w, out) in m.weights.iter().zip(&outs) {
                let want = reference(&m, w, &x);
                for (r, (&got, &want)) in out.iter().zip(&want).enumerate() {
                    assert!(
                        (got - want).abs() <= 1e-5 * want.abs().max(1.0),
                        "{} {} row {r}: {got} vs {want}",
                        kernels.name(),
                        w.ty
                    );
                }
            }
        }
    }

    #[test]
    fn fused_swiglu_matches_separate_products() {
        let (cols, rows) = (128, 200);
        let m = matrices(&[(GgmlType::Q8_0, cols, rows), (GgmlType::F16, cols, rows)]);
        let (gate, up) = (&m.weights[0], &m.weights[1]);
        let x = input(cols);
        for threads in [1, 3] {
            let cpu = Cpu::new(threads);
            let (mut out, mut up_out) = (vec![0.0; rows], vec![0.0; rows]);
            let mut input = EncodedInput::new();
            cpu.swiglu(&m.bytes, gate, up, &x, &mut input, &mut out, &mut up_out);

            let (mut g, mut u) = (vec![0.0; rows], vec![0.0; rows]);
            cpu.matvec(&m.bytes, gate, &x, &mut g, &mut input);
            cpu.matvec(&m.bytes, up, &x, &mut u, &mut input);
            assert_eq!(up_out, u);
            cpu.kernels().swiglu(&mut g, &u);
            assert_eq!(out, g, "{threads} threads");
        }
    }
}
