//! Scalar reference implementations of the vector kernels.
//!
//! Each SIMD kernel in this directory is tested against the function of the
//! same name here, within the tolerance its test documents. These are also
//! the kernels used on CPUs without a SIMD implementation, so they are
//! written to vectorize where the summation order allows it.
//!
//! f16 vectors are little-endian byte buffers, the layout of GGUF weights.

use crate::quant::{f16_to_f32, f32_to_f16};

/// `a · b`, with eight accumulators.
pub(crate) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        acc[i % 8] += x * y;
    }
    sum8(acc)
}

/// `y += x * v`.
pub(crate) fn mad_f32(y: &mut [f32], x: &[f32], v: f32) {
    for (y, &x) in y.iter_mut().zip(x) {
        *y += x * v;
    }
}

/// `a · b` for f16 vectors, accumulated in f32 with eight accumulators.
pub(crate) fn dot_f16(a: &[u8], b: &[u8]) -> f32 {
    let mut acc = [0.0f32; 8];
    for (i, (x, y)) in a
        .as_chunks::<2>()
        .0
        .iter()
        .zip(b.as_chunks::<2>().0)
        .enumerate()
    {
        acc[i % 8] += f16_to_f32(u16::from_le_bytes(*x)) * f16_to_f32(u16::from_le_bytes(*y));
    }
    sum8(acc)
}

/// Writes `x` as f16 (round to nearest even) to `out`, two bytes per value.
pub(crate) fn to_f16(x: &[f32], out: &mut [u8]) {
    for (o, &v) in out.as_chunks_mut::<2>().0.iter_mut().zip(x) {
        *o = f32_to_f16(v).to_le_bytes();
    }
}

/// SiLU (swish): `x · sigmoid(x)`, as ggml's `ggml_silu_f32`.
pub(crate) fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// [`silu`] of every element.
pub(crate) fn silu_in_place(x: &mut [f32]) {
    for v in x {
        *v = silu(*v);
    }
}

/// `g = silu(g) * u`: ggml's `swiglu` with the gate in `g`.
pub(crate) fn swiglu(g: &mut [f32], u: &[f32]) {
    for (g, &u) in g.iter_mut().zip(u) {
        *g = silu(*g) * u;
    }
}

/// Softmax in place, as ggml's `soft_max`: subtract the maximum,
/// exponentiate, sum in f64 and scale by the reciprocal of the sum.
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

/// One head's inputs to a step of the gated delta rule.
pub(crate) struct DeltaStep<'a> {
    pub(crate) q: &'a [f32],
    pub(crate) k: &'a [f32],
    pub(crate) v: &'a [f32],
    /// `sigmoid(b)`, the write strength.
    pub(crate) beta: f32,
    /// `exp(g)`, the state's decay for this token.
    pub(crate) decay: f32,
    /// The output scale, `1 / sqrt(d_v)`.
    pub(crate) scale: f32,
}

/// One token of the gated delta rule for one head, as ggml's
/// `gated_delta_net` op. `state` holds `d_v` rows of `d_k`: row `j` is
/// column `j` of the head's state matrix `S` (ggml's transposed layout).
///
/// ```text
/// S *= decay;  delta = (v - Sᵀk) * beta;  S += k ⊗ delta;  out = Sᵀq * scale
/// ```
///
/// ggml makes four passes over the state; every step only reads and writes
/// row `j` for `delta[j]` and `out[j]`, so one pass per row computes the same.
pub(crate) fn delta_net_head(state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
    let d_k = step.k.len();
    for ((row, &v), o) in state.chunks_exact_mut(d_k).zip(step.v).zip(out) {
        for s in row.iter_mut() {
            *s *= step.decay;
        }
        let delta = (v - dot_f32(row, step.k)) * step.beta;
        mad_f32(row, step.k, delta);
        *o = dot_f32(row, step.q) * step.scale;
    }
}

/// Sum of eight accumulators, in a fixed pairwise order.
fn sum8(acc: [f32; 8]) -> f32 {
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0f32, 2.0, 3.0, 1000.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > 0.999);
    }

    #[test]
    fn delta_net_head_matches_the_matrix_form() {
        // d_k = d_v = 2, with a non-trivial state, decay and beta.
        let mut state = [0.5f32, -1.0, 2.0, 0.25]; // rows are columns of S
        let (q, k, v) = ([1.0f32, 2.0], [0.6f32, 0.8], [1.0f32, -1.0]);
        let step = DeltaStep {
            q: &q,
            k: &k,
            v: &v,
            beta: 0.5,
            decay: 0.9,
            scale: 0.5,
        };
        let mut out = [0.0; 2];
        delta_net_head(&mut state, &step, &mut out);

        // S[i][j] = state[j][i], updated in f64 as written in the paper.
        let mut s = [[0.0f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                s[i][j] = f64::from([0.5f32, -1.0, 2.0, 0.25][j * 2 + i]) * 0.9;
            }
        }
        for j in 0..2 {
            let kv: f64 = (0..2).map(|i| s[i][j] * f64::from(k[i])).sum();
            let delta = (f64::from(v[j]) - kv) * 0.5;
            for i in 0..2 {
                s[i][j] += f64::from(k[i]) * delta;
            }
        }
        for j in 0..2 {
            let o: f64 = (0..2).map(|i| s[i][j] * f64::from(q[i])).sum::<f64>() * 0.5;
            assert!((f64::from(out[j]) - o).abs() < 1e-6, "{out:?}");
            for i in 0..2 {
                assert!((f64::from(state[j * 2 + i]) - s[i][j]).abs() < 1e-6);
            }
        }
    }
}
