//! CPU execution: the thread pool, the vector kernels, and the ops the
//! models are written in.
//!
//! - [`Cpu`] is what a session computes with: its [`ThreadPool`] and the
//!   [`Kernels`] chosen for this CPU. It provides the parallel matrix-vector
//!   products: single, grouped (several matrices that share an input, in one
//!   parallel pass), and the fused SwiGLU feed-forward.
//! - [`Kernels`] dispatches each vector kernel to its SIMD implementation
//!   (`avx2` or `neon`) or to the scalar reference ([`scalar`], and
//!   [`crate::quant`] for the weight formats), which is also the correctness oracle.
//! - Cheap element-wise ops (norms, RoPE, residual adds) are plain functions
//!   here.
//!
//! Results do not depend on the thread count: work is split by output rows
//! or heads, and each output value is computed by one kernel call, whichever
//! thread makes it. Nor do they depend on the batch: a product over several
//! tokens computes each token's output with the same kernel calls as a
//! product over that token alone.

mod attention;
#[cfg(target_arch = "x86_64")]
mod avx2;
mod backend;
mod kernels;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
mod neon;
mod pool;
pub(crate) mod scalar;
mod topology;

use std::ops::Range;

use nibble_gguf::GgmlType;

pub(crate) use backend::CpuBackend;
pub(crate) use kernels::Kernels;
pub(crate) use pool::{SegmentChunks, Split, ThreadPool};
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

/// A product over a batch multiplies this many bytes of weight rows with
/// every token's input before it moves on, so that they are read from
/// memory once per batch and stay in the core's L1 cache meanwhile.
const BATCH_BLOCK_BYTES: usize = 32 * 1024;

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

/// A stack of `count` expert matrices of one shape (a 3-D tensor): expert
/// `e`'s rows follow expert `e - 1`'s.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Experts {
    /// Expert 0.
    pub(crate) first: Weight,
    pub(crate) count: usize,
}

impl Experts {
    /// Expert `e`'s matrix.
    ///
    /// # Panics
    ///
    /// If `e` is not below `count` (an internal invariant: experts are
    /// chosen by the router).
    pub(crate) fn expert(&self, e: usize) -> Weight {
        assert!(e < self.count, "expert {e} of {}", self.count);
        Weight {
            offset: self.first.offset + e * self.first.rows * self.first.row_bytes,
            ..self.first
        }
    }

    /// Every expert as one matrix of `count * rows` rows.
    #[cfg(target_os = "macos")]
    pub(crate) fn all(&self) -> Weight {
        Weight {
            rows: self.count * self.first.rows,
            ..self.first
        }
    }
}

/// The input of a matrix product, encoded in the dot type of each weight
/// type it is multiplied with (see [`quant`]): one encoded vector per token
/// of the batch, one after another. Reused across products, so encoding
/// allocates only while the buffers grow to the largest batch.
#[derive(Debug, Default)]
pub(crate) struct EncodedInput {
    by_dot_type: [Vec<u8>; DotType::COUNT],
}

impl EncodedInput {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Encodes each `cols`-element vector of `x` in the dot type of each
    /// weight type in `types`, once per dot type, using `pool` for batches.
    fn encode(
        &mut self,
        pool: &ThreadPool,
        kernels: Kernels,
        x: &[f32],
        cols: usize,
        types: impl IntoIterator<Item = GgmlType>,
    ) {
        let n = x.len() / cols;
        let mut done = [false; DotType::COUNT];
        for dt in types.into_iter().map(DotType::of) {
            if std::mem::replace(&mut done[dt.index()], true) {
                continue;
            }
            let len = dt.encoded_len(cols);
            let buf = &mut self.by_dot_type[dt.index()];
            buf.resize(len * n, 0);
            if n >= 4 && pool.threads() > 1 && n * cols >= PARALLEL_MIN_MACS {
                let per_task = n.div_ceil(pool.threads() * 2).max(1);
                let split = Split {
                    segments: 1,
                    chunk_len: per_task * len,
                };
                pool.for_each_segment_chunk([buf.as_mut_slice()], split, |_, _, chunk| {
                    let t0 = chunk.start() / len;
                    let out = chunk.segment(0);
                    for (i, dst) in out.chunks_exact_mut(len).enumerate() {
                        let t = t0 + i;
                        kernels.encode(dt, &x[t * cols..(t + 1) * cols], dst);
                    }
                });
            } else {
                for (x, out) in x.chunks_exact(cols).zip(buf.chunks_exact_mut(len)) {
                    kernels.encode(dt, x, out);
                }
            }
        }
    }

    /// The input as encoded for weights of type `ty`: one vector of
    /// `encoded_len` bytes per token.
    fn get(&self, ty: GgmlType) -> &[u8] {
        &self.by_dot_type[DotType::of(ty).index()]
    }
}

/// The threads and kernels a session computes with.
#[derive(Debug)]
pub(crate) struct Cpu {
    pool: ThreadPool,
    /// A pool without workers, which runs its tasks on the calling thread:
    /// for products too small to share.
    serial: ThreadPool,
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
            serial: ThreadPool::new(1),
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

    /// `out = w · x` for every `w.cols`-element vector of `x` (one per token
    /// of a batch), the outputs one after another in `out`.
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

    /// `out_j = w_j · x` for every `(w_j, out_j)` in `products` and every
    /// token's vector in `x`, in one parallel pass: `x` is encoded once per
    /// dot type, and the threads meet once instead of once per matrix.
    /// Each task takes a range of output rows of one product for every token.
    ///
    /// # Panics
    ///
    /// If `N` is zero (an internal invariant).
    pub(crate) fn matvecs<const N: usize>(
        &self,
        bytes: &[u8],
        x: &[f32],
        input: &mut EncodedInput,
        products: [(&Weight, &mut [f32]); N],
    ) {
        let weights = products.each_ref().map(|(w, _)| *w);
        let outs = products.map(|(_, out)| out);
        let cols = weights[0].cols;
        let n = x.len() / cols;
        debug_assert_eq!(x.len(), n * cols);
        for (w, out) in weights.iter().zip(&outs) {
            debug_assert_eq!((w.cols, w.rows * n), (cols, out.len()));
        }
        if n == 0 {
            return;
        }
        input.encode(
            &self.pool,
            self.kernels,
            x,
            cols,
            weights.iter().map(|w| w.ty),
        );
        let input = &*input;
        let kernels = self.kernels;
        let rows: usize = weights.iter().map(|w| w.rows).sum();
        let (pool, chunk_len) = self.split_rows(rows, n * cols);
        let split = Split {
            segments: n,
            chunk_len: chunk_len.unwrap_or(rows).max(1),
        };
        pool.for_each_segment_chunk(outs, split, |j, _, out| {
            dot_batch(kernels, bytes, weights[j], input.get(weights[j].ty), out);
        });
    }

    /// `out = silu(gate · x) * (up · x)` for every token's vector in `x`:
    /// the two products of a SwiGLU feed-forward and ggml's `swiglu` of
    /// them, in one parallel pass. Each task computes the same rows of both
    /// products, then their SwiGLU; `up_out` receives `up · x`.
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
        let n = x.len() / gate.cols;
        debug_assert_eq!((gate.cols * n, gate.rows * n), (x.len(), out.len()));
        debug_assert_eq!((up.cols * n, up.rows * n), (x.len(), up_out.len()));
        if n == 0 {
            return;
        }
        input.encode(&self.pool, self.kernels, x, gate.cols, [gate.ty, up.ty]);
        let input = &*input;
        let kernels = self.kernels;
        let (pool, chunk_len) = self.split_rows(2 * gate.rows, x.len());
        let split = Split {
            segments: n,
            chunk_len: chunk_len.unwrap_or(gate.rows).max(1),
        };
        pool.for_each_segment_chunk_pair(out, split, up_out, split, |_, g, u| {
            dot_batch(kernels, bytes, gate, input.get(gate.ty), g);
            dot_batch(kernels, bytes, up, input.get(up.ty), u);
            // Chunks are whole groups of 8 (ROW_ALIGN), so the SIMD kernels
            // take the same elements as ggml's one pass over the whole vector.
            for t in 0..n {
                kernels.swiglu(g.segment(t), u.segment(t));
            }
        });
    }

    /// `out = silu(gate_e · x_t) * (up_e · x_t)` for every slot `s` of
    /// `ids`, where `e = ids[s]` and `x_t` is the vector of token
    /// `t = s / k` in `x` (`k` slots per token): segment `s` of `out`
    /// (`up_out` receives `up_e · x_t`). Each output row is one
    /// [`Kernels::dot_rows`], as in [`Cpu::swiglu`].
    #[expect(
        clippy::too_many_arguments,
        reason = "two weights and their two outputs"
    )]
    pub(crate) fn moe_swiglu(
        &self,
        bytes: &[u8],
        gate: &Experts,
        up: &Experts,
        ids: &[usize],
        x: &[f32],
        input: &mut EncodedInput,
        out: &mut [f32],
        up_out: &mut [f32],
    ) {
        let (g0, u0) = (gate.first, up.first);
        let n = x.len() / g0.cols;
        if n == 0 || ids.is_empty() {
            return;
        }
        let k = ids.len() / n;
        input.encode(&self.pool, self.kernels, x, g0.cols, [g0.ty, u0.ty]);
        let (xg, xu) = (input.get(g0.ty), input.get(u0.ty));
        let (lg, lu) = (xg.len() / n, xu.len() / n);
        let kernels = self.kernels;
        let (pool, chunk_len) = self.split_rows(g0.rows, 2 * ids.len() * g0.cols);
        let split = Split {
            segments: ids.len(),
            chunk_len: chunk_len.unwrap_or(g0.rows).max(1),
        };
        pool.for_each_segment_chunk_pair(out, split, up_out, split, |_, g, u| {
            let rows = g.start()..g.start() + g.len();
            for (s, &e) in ids.iter().enumerate() {
                let t = s / k;
                let (we, wu) = (gate.expert(e), up.expert(e));
                let (gs, us) = (g.segment(s), u.segment(s));
                let xg = &xg[t * lg..(t + 1) * lg];
                kernels.dot_rows(
                    we.ty,
                    we.rows_bytes(bytes, rows.clone()),
                    we.row_bytes,
                    xg,
                    gs,
                );
                let xu = &xu[t * lu..(t + 1) * lu];
                kernels.dot_rows(
                    wu.ty,
                    wu.rows_bytes(bytes, rows.clone()),
                    wu.row_bytes,
                    xu,
                    us,
                );
                kernels.swiglu(gs, us);
            }
        });
    }

    /// `out_s = down_e · h_s` for every slot `s` of `ids`, `e = ids[s]` and
    /// `h_s` the `s`-th vector of `hidden`: segment `s` of `out`.
    pub(crate) fn moe_products(
        &self,
        bytes: &[u8],
        down: &Experts,
        ids: &[usize],
        hidden: &[f32],
        input: &mut EncodedInput,
        out: &mut [f32],
    ) {
        let w0 = down.first;
        if ids.is_empty() {
            return;
        }
        input.encode(&self.pool, self.kernels, hidden, w0.cols, [w0.ty]);
        let xq = input.get(w0.ty);
        let len = xq.len() / ids.len();
        let kernels = self.kernels;
        let (pool, chunk_len) = self.split_rows(w0.rows, ids.len() * w0.cols);
        let split = Split {
            segments: ids.len(),
            chunk_len: chunk_len.unwrap_or(w0.rows).max(1),
        };
        pool.for_each_segment_chunk([out], split, |_, _, o| {
            let rows = o.start()..o.start() + o.len();
            for (s, &e) in ids.iter().enumerate() {
                let w = down.expert(e);
                let x = &xq[s * len..(s + 1) * len];
                kernels.dot_rows(
                    w.ty,
                    w.rows_bytes(bytes, rows.clone()),
                    w.row_bytes,
                    x,
                    o.segment(s),
                );
            }
        });
    }

    /// [`hadamard`] of every block of `n` values of `x`, blocks spread over
    /// the threads when there are enough of them (ggml splits by block too).
    pub(crate) fn hadamard(&self, x: &mut [f32], n: usize) {
        let blocks = x.len() / n;
        let work = x.len().saturating_mul(n.trailing_zeros() as usize);
        if self.threads() == 1 || blocks < 2 || work < PARALLEL_MIN_MACS {
            return hadamard(x, n);
        }
        let per_task = blocks.div_ceil(TASKS_PER_THREAD * self.threads()).max(1);
        let split = Split {
            segments: 1,
            chunk_len: per_task * n,
        };
        self.pool.for_each_segment_chunk([x], split, |_, _, chunk| {
            hadamard(chunk.segment(0), n);
        });
    }

    /// The pool for a product of `rows` output rows per token over `cols`
    /// input values in all, and the rows per task if it is worth splitting
    /// (`None`: one task per product, on the calling thread).
    fn split_rows(&self, rows: usize, cols: usize) -> (&ThreadPool, Option<usize>) {
        if self.threads() > 1 && rows.saturating_mul(cols) >= PARALLEL_MIN_MACS {
            let chunk = rows
                .div_ceil(TASKS_PER_THREAD * self.threads())
                .next_multiple_of(ROW_ALIGN);
            (&self.pool, Some(chunk))
        } else {
            (&self.serial, None)
        }
    }
}

/// The rows of `w` that `out` covers times every token's input in `xq` (one
/// encoded vector per segment of `out`). For a batch, each block of
/// [`BATCH_BLOCK_BYTES`] of weight rows meets every token before the next
/// block is read; each output is still one row's [`Kernels::dot_rows`], so
/// the result does not depend on the batch.
fn dot_batch(
    kernels: Kernels,
    bytes: &[u8],
    w: &Weight,
    xq: &[u8],
    out: &mut SegmentChunks<'_, f32>,
) {
    let first = out.start();
    let n = out.segments();
    if n == 1 {
        let rows = w.rows_bytes(bytes, first..first + out.len());
        return kernels.dot_rows(w.ty, rows, w.row_bytes, xq, out.segment(0));
    }
    let xq_len = xq.len() / n;
    let x = |t: usize| &xq[t * xq_len..(t + 1) * xq_len];
    let prompt = n > crate::backend::MAX_LOGIT_ROWS;
    // Keep weight blocks a multiple of 4 rows so 4-row GEMM tiles stay aligned.
    let block = (BATCH_BLOCK_BYTES / w.row_bytes.max(1))
        .max(4)
        .next_multiple_of(4);
    let mut b0 = 0;
    while b0 < out.len() {
        let b1 = (b0 + block).min(out.len());
        let rows = w.rows_bytes(bytes, first + b0..first + b1);
        let mut t = 0;
        if prompt {
            while t + 8 <= n {
                let o = out.segments_mut::<8>(t).map(|o| &mut o[b0..b1]);
                let xs = std::array::from_fn(|i| x(t + i));
                if !kernels.dot_rows_prompt(w.ty, rows, w.row_bytes, xs, o) {
                    break;
                }
                t += 8;
            }
            while t + 4 <= n {
                let o = out.segments_mut::<4>(t).map(|o| &mut o[b0..b1]);
                let xs = std::array::from_fn(|i| x(t + i));
                if !kernels.dot_rows_prompt(w.ty, rows, w.row_bytes, xs, o) {
                    break;
                }
                t += 4;
            }
        }
        // Groups of 4 inputs, then one of 3 or of 2, where the type has
        // such kernels (bit-identical to single-token decoding).
        while t + 4 <= n {
            let o = out.segments_mut::<4>(t).map(|o| &mut o[b0..b1]);
            if !kernels.dot_rows_n(
                w.ty,
                rows,
                w.row_bytes,
                [x(t), x(t + 1), x(t + 2), x(t + 3)],
                o,
            ) {
                break;
            }
            t += 4;
        }
        if t + 3 == n {
            let o = out.segments_mut::<3>(t).map(|o| &mut o[b0..b1]);
            if kernels.dot_rows_n(w.ty, rows, w.row_bytes, [x(t), x(t + 1), x(t + 2)], o) {
                t += 3;
            }
        }
        while t + 2 <= n {
            let o = out.segments_mut::<2>(t).map(|o| &mut o[b0..b1]);
            if !kernels.dot_rows_n(w.ty, rows, w.row_bytes, [x(t), x(t + 1)], o) {
                break;
            }
            t += 2;
        }
        for t in t..n {
            kernels.dot_rows(w.ty, rows, w.row_bytes, x(t), &mut out.segment(t)[b0..b1]);
        }
        b0 = b1;
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
/// one pair per rotated pair (`n_rot = 2 · table.len()`), computed
/// iteratively in f32 exactly as ggml's rope cache does.
pub(crate) fn rope_table(table: &mut [(f32, f32)], pos: usize, base: f32) {
    let theta_scale = base.powf(-2.0 / (2 * table.len()) as f32);
    let mut theta = pos as f32;
    for pair in table {
        *pair = (theta.cos(), theta.sin());
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
            // ggml's `x0*cos - x1*sin` and `x0*sin + x1*cos`, with the first
            // product fused into each, as clang compiles them.
            *a = x0.mul_add(cos, -(x1 * sin));
            *b = x0.mul_add(sin, x1 * cos);
        }
    }
}

/// `out += x`.
pub(crate) fn add_in_place(out: &mut [f32], x: &[f32]) {
    for (o, &v) in out.iter_mut().zip(x) {
        *o += v;
    }
}

/// The orthonormal Walsh-Hadamard transform of each block of `n` values
/// (a power of two), as ggml's CPU backend computes llama.cpp's Hadamard
/// rotations (`ggml_compute_forward_fwht`, for a quantized KV cache and for
/// folded weights): each value scaled by `1 / sqrt(n)`, then the butterflies
/// from pairs of neighbours up. ggml's SIMD passes compute `u - v` as
/// `fma(v, -1, u)`, which is exact, so the results are the same. The
/// transform is its own inverse.
pub(crate) fn hadamard(x: &mut [f32], n: usize) {
    let scale = 1.0 / (n as f32).sqrt();
    for row in x.chunks_exact_mut(n) {
        for v in row.iter_mut() {
            *v *= scale;
        }
        let mut len = 1;
        while len < n {
            // Contiguous halves, so the compiler vectorizes the wider passes.
            for pair in row.chunks_exact_mut(2 * len) {
                let (lo, hi) = pair.split_at_mut(len);
                for (a, b) in lo.iter_mut().zip(hi) {
                    let (u, v) = (*a, *b);
                    *a = u + v;
                    *b = u - v;
                }
            }
            len <<= 1;
        }
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
    use crate::quant::test_data::{f32_bytes, fnv1a, values};

    /// ggml's CPU `ggml_compute_forward_fwht` (a `mul_mat` with
    /// `GGML_HINT_SRC0_IS_HADAMARD`) in llama.cpp b12722's macOS arm64 and
    /// b12359's x64 release, which agree: the FNV hash of the output bits of
    /// `values(n * rows, seed)`, and its first, `n - 1`-th and last values.
    #[test]
    fn hadamard_matches_ggml_bit_for_bit() {
        #[rustfmt::skip]
        let cases: [(usize, usize, u32, u64, [u32; 4]); 3] = [
            (64, 3, 5, 0x3582_09D5_E67A_1444, [0x3EC4_A651, 0xBF0D_DEA5, 0xBFCC_C035, 0xBD8E_13B8]),
            (128, 2, 6, 0x7E37_B49A_8313_B4EB, [0x3FA5_B0D0, 0xBE43_E0B8, 0x3F06_9D2A, 0xBE47_6400]),
            (1024, 5, 7, 0xB0F6_A026_A4E3_E72A, [0xBF1E_2A98, 0xBE53_5334, 0xBE76_2E1D, 0x3F33_3015]),
        ];
        for threads in [1, 4] {
            let cpu = Cpu::new(threads);
            for (n, rows, seed, hash, at) in cases {
                let mut x = values(n * rows, seed);
                cpu.hadamard(&mut x, n);
                assert_eq!(fnv1a(&f32_bytes(&x)), hash, "n = {n}, {threads} threads");
                let got = [x[0], x[1], x[n - 1], x[n * rows - 1]].map(f32::to_bits);
                assert_eq!(got, at, "n = {n}");
            }
        }
    }

    /// `ggml_rope_ext` (NeoX, 128 rotated dims, base 1e6) in llama.cpp's
    /// b12359 macOS arm64 release, on two heads of `values(256, 1)`: an FNV
    /// hash of the output bits, and elements 0..4 and 64..68.
    #[test]
    fn rope_matches_ggml_bit_for_bit() {
        #[rustfmt::skip]
        let cases: [(usize, u64, [u32; 8]); 4] = [
            (0, 0x3271_9F08_4C63_9EE9, [0xBEBB_4AE4, 0x3F00_58CA, 0x3F3B_B5DA, 0x3DF6_1680, 0x3EF4_EA8C, 0x3F03_20BE, 0x3D2F_EEA0, 0xBF58_4E7A]),
            (1, 0xF6F4_D5CE_3B01_AB00, [0xBF19_A469, 0xBCB6_F03C, 0x3F0E_DAB5, 0x3F06_BDD5, 0xBD4A_2DFC, 0x3F37_65D6, 0x3EF4_87AB, 0xBF2B_FCA3]),
            (7, 0xA4AF_E07E_0A6E_B439, [0xBF17_0DB0, 0x3F35_51FF, 0xBDA2_73F8, 0xBF06_6E98, 0x3DF6_6070, 0x3DE0_E8FC, 0xBF3A_EEB2, 0x3F2C_3A99]),
            (300, 0xE9FD_71C1_E4E5_0857, [0x3EF8_FEB4, 0xBF12_7FB5, 0x3F3B_3295, 0x3D33_7723, 0x3EB5_D5C2, 0xBEDC_F50B, 0x3D8D_913A, 0xBF5A_31DB]),
        ];
        let mut table = vec![(0.0, 0.0); 64];
        for (pos, hash, bits) in cases {
            let mut x = values(256, 1);
            rope_table(&mut table, pos, 1e6);
            rope_neox(&mut x, 128, &table);
            let got: Vec<u32> = [0, 1, 2, 3, 64, 65, 66, 67]
                .iter()
                .map(|&i| x[i].to_bits())
                .collect();
            assert_eq!(got, bits, "pos {pos}");
            assert_eq!(fnv1a(&f32_bytes(&x)), hash, "pos {pos}");
        }
    }

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
        let mut table = vec![(0.0, 0.0); 4];
        let head: Vec<f32> = (0..8).map(|i| i as f32 - 3.5).collect();

        rope_table(&mut table, 0, 10_000.0);
        let mut x = head.clone();
        rope_neox(&mut x, 8, &table);
        assert_eq!(x, head);

        rope_table(&mut table, 17, 10_000.0);
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
                push_weight(&mut bytes, ty, &row);
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

    /// K-quant rows are not the activation dot type, so a deterministic block
    /// with a finite f16 scale stands in for a quantized row.
    fn push_weight(bytes: &mut Vec<u8>, ty: GgmlType, row: &[f32]) {
        if matches!(
            ty,
            GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q6K | GgmlType::Iq4Xs
        ) {
            let n = ty.layout().unwrap().type_size as usize;
            for b in 0..row.len() / 256 {
                let mut block = vec![0u8; n];
                for (i, slot) in block.iter_mut().enumerate() {
                    let v = row[(b * 17 + i) % row.len()];
                    *slot = (v.to_bits().wrapping_mul(i as u32 + 1) >> (i % 16)) as u8;
                }
                let scale = 0x3C00u16.to_le_bytes();
                match ty {
                    GgmlType::Q4K | GgmlType::Q5K => {
                        block[..2].copy_from_slice(&scale);
                        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                    }
                    GgmlType::Q6K => {
                        block[208..210].copy_from_slice(&scale);
                        for s in &mut block[192..208] {
                            *s %= 9;
                        }
                    }
                    GgmlType::Q3K => block[108..110].copy_from_slice(&scale),
                    GgmlType::Iq4Xs => block[..2].copy_from_slice(&scale),
                    _ => {}
                }
                bytes.extend(block);
            }
        } else if ty == GgmlType::F16 {
            // F16 weights are f16; their dot type is f32.
            bytes.extend(row.iter().flat_map(|&v| quant::f32_to_f16(v).to_le_bytes()));
        } else {
            bytes.extend(quant::encoded(DotType::of(ty), row));
        }
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
        for ty in [
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q3K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Iq4Xs,
        ] {
            let cols = if ty == GgmlType::Q8_0 { 512 } else { 256 };
            let m = matrices(&[(ty, cols, 1000)]);
            let w = &m.weights[0];
            let x = input(cols);
            let mut results = Vec::new();
            for threads in [1, 2, 7] {
                let cpu = Cpu::new(threads);
                let mut out = vec![0.0; w.rows];
                cpu.matvec(&m.bytes, w, &x, &mut out, &mut EncodedInput::new());
                results.push(out);
            }
            assert_eq!(results[0], results[1], "{ty}");
            assert_eq!(results[0], results[2], "{ty}");
        }
    }

    /// A group with mixed weight types needs one encoding per dot type, and
    /// each output must match the scalar reference row by row.
    #[test]
    fn grouped_products_of_mixed_types_match_the_reference() {
        let cols = 256;
        let m = matrices(&[
            (GgmlType::Q8_0, cols, 300),
            (GgmlType::F16, cols, 70),
            (GgmlType::Q4K, cols, 80),
            (GgmlType::Bf16, cols, 129),
            (GgmlType::Q6K, cols, 64),
            (GgmlType::Q3K, cols, 48),
            (GgmlType::Q5K, cols, 32),
        ]);
        let x = input(cols);
        for kernels in [Kernels::Scalar, Kernels::detect()] {
            let cpu = Cpu::with_kernels(4, kernels);
            let mut outs: Vec<Vec<f32>> = m.weights.iter().map(|w| vec![0.0; w.rows]).collect();
            let w = &m.weights;
            let mut slots = outs.iter_mut();
            let (s0, s1, s2, s3, s4, s5, s6) = (
                slots.next().unwrap(),
                slots.next().unwrap(),
                slots.next().unwrap(),
                slots.next().unwrap(),
                slots.next().unwrap(),
                slots.next().unwrap(),
                slots.next().unwrap(),
            );
            cpu.matvecs(
                &m.bytes,
                &x,
                &mut EncodedInput::new(),
                [
                    (&w[0], s0.as_mut_slice()),
                    (&w[1], s1.as_mut_slice()),
                    (&w[2], s2.as_mut_slice()),
                    (&w[3], s3.as_mut_slice()),
                    (&w[4], s4.as_mut_slice()),
                    (&w[5], s5.as_mut_slice()),
                    (&w[6], s6.as_mut_slice()),
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

    /// `n` inputs of `cols` values, different per token.
    fn batch_input(cols: usize, n: usize) -> Vec<f32> {
        (0..n * cols)
            .map(|i| ((i % cols) as f32 * 0.37 + (i / cols) as f32 * 1.3).sin())
            .collect()
    }

    /// Each token's output of a product over a batch, bit for bit, is that
    /// of the product over the token alone: on one thread (one task covers
    /// the matrix, so it walks several blocks of rows) and on four.
    #[test]
    fn batched_products_equal_per_token_products() {
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Bf16,
            GgmlType::Q8_0,
            GgmlType::Q4K,
            GgmlType::Q3K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Iq4Xs,
        ];
        let cols = 256;
        for ty in types {
            let m = matrices(&[(ty, cols, 1003)]);
            let w = &m.weights[0];
            for kernels in [Kernels::Scalar, Kernels::detect()] {
                for threads in [1, 4] {
                    let cpu = Cpu::with_kernels(threads, kernels);
                    for n in [2, 3, 5, 7] {
                        let x = batch_input(cols, n);
                        let mut input = EncodedInput::new();
                        let mut batch = vec![0.0; n * w.rows];
                        cpu.matvec(&m.bytes, w, &x, &mut batch, &mut input);
                        for (t, (x, got)) in x.chunks(cols).zip(batch.chunks(w.rows)).enumerate() {
                            let mut want = vec![0.0; w.rows];
                            cpu.matvec(&m.bytes, w, x, &mut want, &mut input);
                            let same = got
                                .iter()
                                .zip(&want)
                                .all(|(a, b)| a.to_bits() == b.to_bits());
                            assert!(
                                same,
                                "{ty} {} {threads} threads, token {t} of {n}",
                                kernels.name()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batched_swiglu_equals_per_token_swiglu() {
        for (gate_ty, up_ty) in [
            (GgmlType::Q8_0, GgmlType::F16),
            (GgmlType::Q4K, GgmlType::Q6K),
        ] {
            let (cols, rows, n) = (256, 200, 3);
            let m = matrices(&[(gate_ty, cols, rows), (up_ty, cols, rows)]);
            let (gate, up) = (&m.weights[0], &m.weights[1]);
            let x = batch_input(cols, n);
            for threads in [1, 3] {
                let cpu = Cpu::new(threads);
                let mut input = EncodedInput::new();
                let (mut out, mut up_out) = (vec![0.0; n * rows], vec![0.0; n * rows]);
                cpu.swiglu(&m.bytes, gate, up, &x, &mut input, &mut out, &mut up_out);
                for (t, x) in x.chunks(cols).enumerate() {
                    let (mut o, mut u) = (vec![0.0; rows], vec![0.0; rows]);
                    cpu.swiglu(&m.bytes, gate, up, x, &mut input, &mut o, &mut u);
                    assert_eq!(
                        out[t * rows..(t + 1) * rows],
                        o,
                        "{gate_ty} {up_ty} token {t}"
                    );
                    assert_eq!(
                        up_out[t * rows..(t + 1) * rows],
                        u,
                        "{gate_ty} {up_ty} token {t}"
                    );
                }
            }
        }
    }

    #[test]
    fn fused_swiglu_matches_separate_products() {
        for (gate_ty, up_ty, cols) in [
            (GgmlType::Q8_0, GgmlType::F16, 128),
            (GgmlType::Q4K, GgmlType::Q6K, 256),
            (GgmlType::Q3K, GgmlType::Q5K, 256),
        ] {
            let rows = 200;
            let m = matrices(&[(gate_ty, cols, rows), (up_ty, cols, rows)]);
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
                assert_eq!(up_out, u, "{gate_ty} {up_ty}");
                cpu.kernels().swiglu(&mut g, &u);
                assert_eq!(out, g, "{gate_ty} {up_ty}, {threads} threads");
            }
        }
    }
}
