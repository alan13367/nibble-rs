//! AVX2 kernels (x86-64-v3: AVX2, FMA and F16C).
//!
//! Each kernel transcribes the ggml function named in its documentation,
//! keeping ggml's accumulator layout and reduction order, so results match
//! llama.cpp's CPU backend on the same machine (llama.cpp's x86 builds use
//! these code paths on any AVX2 CPU without AVX-512). The scalar code in
//! [`super::scalar`] and [`crate::quant`] is the reference each kernel is
//! tested against.
//!
//! Every function is compiled with `#[target_feature(enable = "avx2,fma,f16c")]`
//! so that the intrinsics inline. They are reachable only through [`Avx2`],
//! a token that exists only on CPUs with those features.

use std::arch::x86_64::*;

use nibble_gguf::GgmlType;

use super::scalar::{self, DeltaStep};
use crate::quant::{self, DotType, Q8_0_BLOCK_BYTES, QK8_0};

/// Proof that the CPU supports AVX2, FMA and F16C: only [`Avx2::detect`]
/// creates one, so its methods can run the kernels.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Avx2(());

impl Avx2 {
    /// The token, if this CPU has AVX2, FMA and F16C.
    #[must_use]
    pub(crate) fn detect() -> Option<Self> {
        let supported = is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c");
        supported.then_some(Self(()))
    }

    /// See [`super::Kernels::encode`].
    pub(crate) fn encode(self, dt: DotType, x: &[f32], out: &mut [u8]) {
        // SAFETY: `self` exists only if the CPU supports AVX2, FMA and F16C
        // (see `detect`), the features every kernel here is compiled with.
        // The same holds for every method below.
        unsafe { encode(dt, x, out) }
    }

    /// See [`super::Kernels::dot_rows`].
    pub(crate) fn dot_rows(
        self,
        ty: GgmlType,
        rows: &[u8],
        row_bytes: usize,
        xq: &[u8],
        out: &mut [f32],
    ) {
        // SAFETY: see `encode`.
        unsafe { dot_rows(ty, rows, row_bytes, xq, out) }
    }

    /// See [`super::Kernels::dot_f16_f32_rows`].
    pub(crate) fn dot_f16_f32_rows(self, a: &[u8], stride: usize, b: &[f32], out: &mut [f32]) {
        for (r, o) in out.iter_mut().enumerate() {
            let row = &a[r * stride..r * stride + 2 * b.len()];
            // SAFETY: see `encode`.
            *o = unsafe { dot_f16_f32(row, b) };
        }
    }

    /// See [`super::Kernels::to_f16`].
    pub(crate) fn to_f16(self, x: &[f32], out: &mut [u8]) {
        // SAFETY: see `encode`.
        unsafe { to_f16(x, out) }
    }

    /// See [`super::Kernels::silu`].
    pub(crate) fn silu(self, x: &mut [f32]) {
        // SAFETY: see `encode`.
        unsafe { silu(x) }
    }

    /// See [`super::Kernels::swiglu`].
    pub(crate) fn swiglu(self, g: &mut [f32], u: &[f32]) {
        // SAFETY: see `encode`.
        unsafe { swiglu(g, u) }
    }

    /// See [`super::Kernels::softmax`].
    pub(crate) fn softmax(self, x: &mut [f32]) {
        // SAFETY: see `encode`.
        unsafe { softmax(x) }
    }

    /// See [`super::Kernels::delta_net_head`].
    pub(crate) fn delta_net_head(self, state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
        // SAFETY: see `encode`.
        unsafe { delta_net_head(state, step, out) }
    }
}

// --- Loads, stores and reductions ---------------------------------------

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn load_f32(a: &[f32; 8]) -> __m256 {
    // SAFETY: `a` is 32 readable bytes; the load has no alignment requirement.
    unsafe { _mm256_loadu_ps(a.as_ptr()) }
}

/// Eight little-endian f32 values.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn load_f32_le(a: &[u8; 32]) -> __m256 {
    // SAFETY: as `load_f32`; x86 is little-endian.
    unsafe { _mm256_loadu_ps(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn store_f32(a: &mut [f32; 8], v: __m256) {
    // SAFETY: `a` is 32 writable bytes; the store has no alignment requirement.
    unsafe { _mm256_storeu_ps(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn load_u128(a: &[u8; 16]) -> __m128i {
    // SAFETY: `a` is 16 readable bytes; the load has no alignment requirement.
    unsafe { _mm_loadu_si128(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn store_u128(a: &mut [u8; 16], v: __m128i) {
    // SAFETY: `a` is 16 writable bytes; the store has no alignment requirement.
    unsafe { _mm_storeu_si128(a.as_mut_ptr().cast(), v) }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn load_u256(a: &[u8; 32]) -> __m256i {
    // SAFETY: `a` is 32 readable bytes; the load has no alignment requirement.
    unsafe { _mm256_loadu_si256(a.as_ptr().cast()) }
}

#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn store_u256(a: &mut [u8; 32], v: __m256i) {
    // SAFETY: `a` is 32 writable bytes; the store has no alignment requirement.
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast(), v) }
}

/// Eight f16 values (`GGML_F32Cx8_LOAD`).
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn load_f16(a: &[u8; 16]) -> __m256 {
    _mm256_cvtph_ps(load_u128(a))
}

/// Eight bf16 values: widened to 32 bits and shifted into the high half.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn load_bf16(a: &[u8; 16]) -> __m256 {
    _mm256_castsi256_ps(_mm256_slli_epi32::<16>(_mm256_cvtepu16_epi32(load_u128(a))))
}

/// One f16 value, converted exactly.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn f16_to_f32(h: [u8; 2]) -> f32 {
    _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(i32::from(
        u16::from_le_bytes(h),
    ))))
}

/// The four `M`-element parts of a step of `4 * M` elements (one vector each).
#[inline]
fn split4<T, const M: usize>(step: &[T]) -> [&[T; M]; 4] {
    let parts = step.as_chunks::<M>().0;
    [&parts[0], &parts[1], &parts[2], &parts[3]]
}

/// A vector of eight copies of the f32 with these bits.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn splat(bits: u32) -> __m256 {
    _mm256_set1_ps(f32::from_bits(bits))
}

/// ggml's `hsum_float_8`: the sum of eight lanes, as `(l0 + h0) + (l2 + h2)`
/// plus `(l1 + h1) + (l3 + h3)`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn hsum(x: __m256) -> f32 {
    let mut r = _mm_add_ps(_mm256_extractf128_ps::<1>(x), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    _mm_cvtss_f32(r)
}

/// `GGML_F32x8_REDUCE` over ggml's four accumulators: pairwise sums, then
/// the two halves, then two horizontal adds.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn reduce4(s: [__m256; 4]) -> f32 {
    let x = _mm256_add_ps(_mm256_add_ps(s[0], s[2]), _mm256_add_ps(s[1], s[3]));
    let t0 = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps::<1>(x));
    let t1 = _mm_hadd_ps(t0, t0);
    _mm_cvtss_f32(_mm_hadd_ps(t1, t1))
}

// --- Activation encoding -------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
fn encode(dt: DotType, x: &[f32], out: &mut [u8]) {
    match dt {
        // A copy; nothing to vectorize beyond what the compiler does.
        DotType::F32 => quant::encode_activation(dt, x, out),
        DotType::Bf16 => to_bf16(x, out),
        DotType::Q8_0 => quantize_q8_0(x, out),
        // The x86 AVX2 `Q8_K` kernel is not here yet; it has to be
        // transcribed and checked on an x86 machine. The scalar encoder
        // produces the bytes every architecture agrees on.
        DotType::Q8K | DotType::Q8_1 => quant::encode_activation(dt, x, out),
    }
}

/// f32 to f16 with round to nearest even, as `ggml_cpu_fp32_to_fp16`.
#[target_feature(enable = "avx2,fma,f16c")]
fn to_f16(x: &[f32], out: &mut [u8]) {
    let (xs, x_tail) = x.as_chunks::<8>();
    let (os, o_tail) = out.as_chunks_mut::<16>();
    for (o, x) in os.iter_mut().zip(xs) {
        store_u128(o, _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(load_f32(x)));
    }
    scalar::to_f16(x_tail, o_tail);
}

/// f32 to bf16 with round to nearest even and quiet NaNs, as
/// `GGML_FP32_TO_BF16` (ggml has no AVX2 version; this is exact).
#[target_feature(enable = "avx2,fma,f16c")]
fn to_bf16(x: &[f32], out: &mut [u8]) {
    let (xs, x_tail) = x.as_chunks::<8>();
    let (os, o_tail) = out.as_chunks_mut::<16>();
    let abs_mask = _mm256_set1_epi32(0x7FFF_FFFF);
    let inf = _mm256_set1_epi32(0x7F80_0000);
    for (o, x) in os.iter_mut().zip(xs) {
        let bits = _mm256_castps_si256(load_f32(x));
        let nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, abs_mask), inf);
        let odd = _mm256_and_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(1));
        let bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), odd);
        let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(bits, bias));
        let quiet = _mm256_or_si256(_mm256_srli_epi32::<16>(bits), _mm256_set1_epi32(0x40));
        let halves = _mm256_blendv_epi8(rounded, quiet, nan);
        // Pack the eight 16-bit results (each lane's values fit, so the
        // saturation never applies) and gather both 128-bit lanes.
        let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(halves, halves));
        store_u128(o, _mm256_castsi256_si128(packed));
    }
    for (o, &v) in o_tail.as_chunks_mut::<2>().0.iter_mut().zip(x_tail) {
        *o = quant::f32_to_bf16(v).to_le_bytes();
    }
}

/// `quantize_row_q8_0`, AVX2 path: per block, `d = max|x| / 127` and
/// `q = round(x * 127 / max|x|)`. Identical to the scalar encoder.
#[target_feature(enable = "avx2,fma,f16c")]
fn quantize_q8_0(x: &[f32], out: &mut [u8]) {
    let blocks = out.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter_mut().zip(x.as_chunks::<QK8_0>().0) {
        let [x0, x1, x2, x3] = split4::<_, 8>(x);
        let (v0, v1, v2, v3) = (load_f32(x0), load_f32(x1), load_f32(x2), load_f32(x3));
        let sign = _mm256_set1_ps(-0.0);
        let mut max = _mm256_andnot_ps(sign, v0);
        max = _mm256_max_ps(max, _mm256_andnot_ps(sign, v1));
        max = _mm256_max_ps(max, _mm256_andnot_ps(sign, v2));
        max = _mm256_max_ps(max, _mm256_andnot_ps(sign, v3));
        let mut max4 = _mm_max_ps(_mm256_extractf128_ps::<1>(max), _mm256_castps256_ps128(max));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        let amax = _mm_cvtss_f32(max4);

        let d = amax / 127.0;
        let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
        let mul = _mm256_set1_ps(id);
        let i0 = round_scaled(v0, mul);
        let i1 = round_scaled(v1, mul);
        let i2 = round_scaled(v2, mul);
        let i3 = round_scaled(v3, mul);
        // i32 -> i16 -> i8 packs work per 128-bit lane; the permutation
        // restores element order.
        let packed = _mm256_packs_epi16(_mm256_packs_epi32(i0, i1), _mm256_packs_epi32(i2, i3));
        let q = _mm256_permutevar8x32_epi32(packed, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

        let (scale, qs) = block.split_first_chunk_mut::<2>().unwrap();
        *scale = quant::f32_to_f16(d).to_le_bytes();
        store_u256(qs.try_into().unwrap(), q);
    }
}

/// `round(v * mul)` as integers, rounding half to even.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn round_scaled(v: __m256, mul: __m256) -> __m256i {
    _mm256_cvtps_epi32(_mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(_mm256_mul_ps(
        v, mul,
    )))
}

// --- Dot products ----------------------------------------------------------

#[target_feature(enable = "avx2,fma,f16c")]
fn dot_rows(ty: GgmlType, rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
    let rows = rows.chunks_exact(row_bytes);
    match ty {
        GgmlType::Q8_0 => {
            let x = xq.as_chunks::<Q8_0_BLOCK_BYTES>().0;
            for (o, row) in out.iter_mut().zip(rows) {
                *o = dot_q8_0(row.as_chunks().0, x);
            }
        }
        GgmlType::F16 => {
            for (o, row) in out.iter_mut().zip(rows) {
                *o = dot_f16_f32_le(row, xq);
            }
        }
        GgmlType::Bf16 => {
            for (o, row) in out.iter_mut().zip(rows) {
                *o = dot_bf16(row, xq);
            }
        }
        GgmlType::F32 => {
            for (o, row) in out.iter_mut().zip(rows) {
                *o = dot_f32_le(row, xq);
            }
        }
        // AVX2 kernels for the other types will be added and tested on an
        // x86 machine. Until then the scalar reference keeps them correct
        // on x86 too, instead of refusing them; its sums follow the NEON
        // kernels, so x86 results can differ from llama.cpp's in the last bits.
        _ => {
            for (o, row) in out.iter_mut().zip(rows) {
                *o = quant::dot_row(ty, row, xq);
            }
        }
    }
}

/// `ggml_vec_dot_q8_0_q8_0`, AVX2 path: per block, the exact integer dot of
/// the 32 pairs scaled by both block scales, accumulated with one FMA chain.
#[target_feature(enable = "avx2,fma,f16c")]
fn dot_q8_0(w: &[[u8; Q8_0_BLOCK_BYTES]], x: &[[u8; Q8_0_BLOCK_BYTES]]) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for (wb, xb) in w.iter().zip(x) {
        let (dw, qw) = wb.split_first_chunk::<2>().unwrap();
        let (dx, qx) = xb.split_first_chunk::<2>().unwrap();
        let d = _mm256_set1_ps(f16_to_f32(*dw) * f16_to_f32(*dx));
        let (qw, qx) = (
            load_u256(qw.try_into().unwrap()),
            load_u256(qx.try_into().unwrap()),
        );
        // `mul_sum_i8_pairs_float`: make the weights unsigned and move their
        // signs to the activations so `maddubs` can multiply the pairs.
        let products = _mm256_maddubs_epi16(_mm256_sign_epi8(qw, qw), _mm256_sign_epi8(qx, qw));
        let q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), products));
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    hsum(acc)
}

/// `ggml_vec_dot_f32`: four 8-lane FMA accumulators over steps of 32, the
/// `GGML_F32x8_REDUCE` reduction, then the leftovers in f32 (`sumf +=
/// x[i]*y[i]`, which the compiler fuses on an FMA target).
#[target_feature(enable = "avx2,fma,f16c")]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a_steps, a_tail) = a[..n].as_chunks::<32>();
    let (b_steps, b_tail) = b[..n].as_chunks::<32>();
    let mut sum = [_mm256_setzero_ps(); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        for ((s, a), b) in sum.iter_mut().zip(split4::<_, 8>(a)).zip(split4::<_, 8>(b)) {
            *s = _mm256_fmadd_ps(load_f32(a), load_f32(b), *s);
        }
    }
    let mut sumf = reduce4(sum);
    for (&a, &b) in a_tail.iter().zip(b_tail) {
        sumf = a.mul_add(b, sumf);
    }
    sumf
}

/// [`dot_f32`] over little-endian f32 bytes (F32 weight rows).
#[target_feature(enable = "avx2,fma,f16c")]
fn dot_f32_le(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len()) / 4 * 4;
    let (a_steps, a_tail) = a[..n].as_chunks::<128>();
    let (b_steps, b_tail) = b[..n].as_chunks::<128>();
    let mut sum = [_mm256_setzero_ps(); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        for ((s, a), b) in sum
            .iter_mut()
            .zip(split4::<_, 32>(a))
            .zip(split4::<_, 32>(b))
        {
            *s = _mm256_fmadd_ps(load_f32_le(a), load_f32_le(b), *s);
        }
    }
    let mut sumf = reduce4(sum);
    for (a, b) in a_tail
        .as_chunks::<4>()
        .0
        .iter()
        .zip(b_tail.as_chunks::<4>().0)
    {
        sumf = f32::from_le_bytes(*a).mul_add(f32::from_le_bytes(*b), sumf);
    }
    sumf
}

/// `ggml_vec_dot_f16_f32`, AVX + F16C path: `a` is f16, `b` f32. Four
/// 8-lane FMA accumulators over steps of 32, then steps of 8 into the first,
/// summed `(a0 + a1) + (a2 + a3)` and reduced as `hsum_float_8` into f64,
/// then the leftovers (each product rounded to f32) in f64.
#[target_feature(enable = "avx2,fma,f16c")]
fn dot_f16_f32(a: &[u8], b: &[f32]) -> f32 {
    let n = (a.len() / 2).min(b.len());
    let (a_steps, a_rest) = a[..2 * n].as_chunks::<64>();
    let (b_steps, b_rest) = b[..n].as_chunks::<32>();
    let mut acc = [_mm256_setzero_ps(); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        for ((s, a), b) in acc
            .iter_mut()
            .zip(split4::<_, 16>(a))
            .zip(split4::<_, 8>(b))
        {
            *s = _mm256_fmadd_ps(load_f16(a), load_f32(b), *s);
        }
    }
    let (a8, a_tail) = a_rest.as_chunks::<16>();
    let (b8, b_tail) = b_rest.as_chunks::<8>();
    for (a, b) in a8.iter().zip(b8) {
        acc[0] = _mm256_fmadd_ps(load_f16(a), load_f32(b), acc[0]);
    }
    let total = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut sumf = f64::from(hsum(total));
    for (a, &b) in a_tail.as_chunks::<2>().0.iter().zip(b_tail) {
        sumf += f64::from(f16_to_f32(*a) * b);
    }
    sumf as f32
}

/// [`dot_f16_f32`] for an F16 weight row and an activation of
/// little-endian f32 bytes.
#[target_feature(enable = "avx2,fma,f16c")]
fn dot_f16_f32_le(a: &[u8], b: &[u8]) -> f32 {
    let n = (a.len() / 2).min(b.len() / 4);
    let (a_steps, a_rest) = a[..2 * n].as_chunks::<64>();
    let (b_steps, b_rest) = b[..4 * n].as_chunks::<128>();
    let mut acc = [_mm256_setzero_ps(); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        for ((s, a), b) in acc
            .iter_mut()
            .zip(split4::<_, 16>(a))
            .zip(split4::<_, 32>(b))
        {
            *s = _mm256_fmadd_ps(load_f16(a), load_f32_le(b), *s);
        }
    }
    let (a8, a_tail) = a_rest.as_chunks::<16>();
    let (b8, b_tail) = b_rest.as_chunks::<32>();
    for (a, b) in a8.iter().zip(b8) {
        acc[0] = _mm256_fmadd_ps(load_f16(a), load_f32_le(b), acc[0]);
    }
    let total = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
    let mut sumf = f64::from(hsum(total));
    for (a, b) in a_tail
        .as_chunks::<2>()
        .0
        .iter()
        .zip(b_tail.as_chunks::<4>().0)
    {
        sumf += f64::from(f16_to_f32(*a) * f32::from_le_bytes(*b));
    }
    sumf as f32
}

/// `ggml_vec_dot_bf16`, AVX2 path: four 8-lane accumulators of separate
/// multiplies and adds over steps of 32, summed `(c0 + c2) + (c1 + c3)`,
/// reduced with `hsum_float_8` into f64, then the leftovers in f64.
#[target_feature(enable = "avx2,fma,f16c")]
fn dot_bf16(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len()) / 2 * 2;
    let (a_steps, a_tail) = a[..n].as_chunks::<64>();
    let (b_steps, b_tail) = b[..n].as_chunks::<64>();
    let mut c = [_mm256_setzero_ps(); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        for ((c, a), b) in c.iter_mut().zip(split4::<_, 16>(a)).zip(split4::<_, 16>(b)) {
            *c = _mm256_add_ps(_mm256_mul_ps(load_bf16(a), load_bf16(b)), *c);
        }
    }
    let total = _mm256_add_ps(_mm256_add_ps(c[0], c[2]), _mm256_add_ps(c[1], c[3]));
    let mut sumf = f64::from(hsum(total));
    for (a, b) in a_tail
        .as_chunks::<2>()
        .0
        .iter()
        .zip(b_tail.as_chunks::<2>().0)
    {
        let (a, b) = (u16::from_le_bytes(*a), u16::from_le_bytes(*b));
        sumf += f64::from(quant::bf16_to_f32(a) * quant::bf16_to_f32(b));
    }
    sumf as f32
}

// --- Element-wise functions ------------------------------------------------

/// `ggml_v_expf`: e^x with at most 1.45 ulp + 0.5 ulp of error (ggml's
/// figure); above 88.38 it overflows to infinity, below -103.97 it flushes
/// to zero. The hex-float constants of the C original are given as bits.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn exp(x: __m256) -> __m256 {
    let r = splat(0x4B40_0000); // 0x1.8p23f
    let z = _mm256_fmadd_ps(x, splat(0x3FB8_AA3B), r); // 0x1.715476p+0f
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        splat(0x35BF_BE8E),                         // 0x1.7f7d1cp-20f
        _mm256_fnmadd_ps(n, splat(0x3F31_7200), x), // 0x1.62e4p-1f
    );
    let e = _mm256_slli_epi32::<23>(_mm256_castps_si256(z));
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0)),
    ));
    let abs_n = _mm256_andnot_ps(_mm256_set1_ps(-0.0), n);
    let big = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(126.0));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(splat(0x3C07_2010), b, splat(0x3D2B_9F17)), // 0x1.0e4020p-7f, 0x1.573e2ep-5f
            u,
            _mm256_fmadd_ps(splat(0x3E2A_AF33), b, splat(0x3EFF_FEDB)), // 0x1.555e66p-3f, 0x1.fffdb6p-2f
        ),
        u,
        _mm256_mul_ps(splat(0x3F7F_FFF6), b), // 0x1.ffffecp-1f
    );
    if _mm256_movemask_ps(big) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }
    // Some lane is far from 1: scale in two steps to reach the extremes.
    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps::<_CMP_LE_OQ>(n, _mm256_setzero_ps())),
        _mm256_set1_epi32(0x8200_0000u32 as i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7F00_0000)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let huge = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_n, _mm256_set1_ps(192.0));
    _mm256_or_ps(
        _mm256_and_ps(huge, _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            huge,
            _mm256_or_ps(
                _mm256_and_ps(big, _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                _mm256_andnot_ps(big, _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

/// `ggml_v_silu`: `x / (1 + e^-x)` with [`exp`].
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn silu8(x: __m256) -> __m256 {
    let neg = _mm256_sub_ps(_mm256_setzero_ps(), x);
    _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0), exp(neg)))
}

/// `ggml_vec_silu_f32`: vectorized in groups of eight, the rest scalar.
#[target_feature(enable = "avx2,fma,f16c")]
fn silu(x: &mut [f32]) {
    let (groups, tail) = x.as_chunks_mut::<8>();
    for g in groups {
        store_f32(g, silu8(load_f32(g)));
    }
    scalar::silu_in_place(tail);
}

/// `ggml_vec_swiglu_f32`: `g = silu(g) * u` in groups of eight, the rest scalar.
#[target_feature(enable = "avx2,fma,f16c")]
fn swiglu(g: &mut [f32], u: &[f32]) {
    let n = g.len().min(u.len());
    let (gs, g_tail) = g[..n].as_chunks_mut::<8>();
    let (us, u_tail) = u[..n].as_chunks::<8>();
    for (g, u) in gs.iter_mut().zip(us) {
        store_f32(g, _mm256_mul_ps(silu8(load_f32(g)), load_f32(u)));
    }
    scalar::swiglu(g_tail, u_tail);
}

/// `ggml_compute_forward_soft_max` of one row: `ggml_vec_soft_max_f32`
/// exponentiates groups of eight with [`exp`] and adds each group's
/// `hsum_float_8` to an f64 sum, then the row is scaled by `1 / sum`.
///
/// llama.cpp pads the row to a multiple of 256 with masked (-inf) entries,
/// so its rows never have a scalar remainder. The last partial group here is
/// padded the same way; e^-inf is 0, so the padding changes nothing else.
#[target_feature(enable = "avx2,fma,f16c")]
fn softmax(x: &mut [f32]) {
    let max = _mm256_set1_ps(x.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let mut sum = 0.0f64;
    let (groups, tail) = x.as_chunks_mut::<8>();
    for g in groups {
        let e = exp(_mm256_sub_ps(load_f32(g), max));
        store_f32(g, e);
        sum += f64::from(hsum(e));
    }
    if !tail.is_empty() {
        let mut padded = [f32::NEG_INFINITY; 8];
        padded[..tail.len()].copy_from_slice(tail);
        let e = exp(_mm256_sub_ps(load_f32(&padded), max));
        store_f32(&mut padded, e);
        sum += f64::from(hsum(e));
        tail.copy_from_slice(&padded[..tail.len()]);
    }
    let inv = (1.0 / sum) as f32;
    for v in x {
        *v *= inv;
    }
}

// --- Gated delta rule --------------------------------------------------------

/// One head of `ggml_compute_forward_gated_delta_net` (see
/// [`scalar::delta_net_head`]), with ggml's vector kernels: the decay is
/// `ggml_vec_scale_f32`, the update `ggml_vec_mad_f32` (an FMA), and both
/// products [`dot_f32`].
#[target_feature(enable = "avx2,fma,f16c")]
fn delta_net_head(state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
    let d_k = step.k.len();
    for ((row, &v), o) in state.chunks_exact_mut(d_k).zip(step.v).zip(out) {
        scale(row, step.decay);
        let delta = (v - dot_f32(row, step.k)) * step.beta;
        mad(row, step.k, delta);
        *o = dot_f32(row, step.q) * step.scale;
    }
}

/// `ggml_vec_scale_f32`: `y *= v`.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn scale(y: &mut [f32], v: f32) {
    let (ys, tail) = y.as_chunks_mut::<8>();
    let vv = _mm256_set1_ps(v);
    for y in ys {
        store_f32(y, _mm256_mul_ps(load_f32(y), vv));
    }
    for y in tail {
        *y *= v;
    }
}

/// `ggml_vec_mad_f32`: `y = fma(x, v, y)` over steps of 32, the leftovers
/// as a separate multiply and add.
#[inline]
#[target_feature(enable = "avx2,fma,f16c")]
fn mad(y: &mut [f32], x: &[f32], v: f32) {
    let n = y.len().min(x.len());
    let (ys, y_tail) = y[..n].as_chunks_mut::<32>();
    let (xs, x_tail) = x[..n].as_chunks::<32>();
    let vv = _mm256_set1_ps(v);
    for (y, x) in ys.iter_mut().zip(xs) {
        let (ys, xs) = (y.as_chunks_mut::<8>().0, x.as_chunks::<8>().0);
        for (y, x) in ys.iter_mut().zip(xs) {
            store_f32(y, _mm256_fmadd_ps(load_f32(x), vv, load_f32(y)));
        }
    }
    scalar::mad_f32(y_tail, x_tail, v);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in [-1, 1).
    fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).max(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    /// `values` scaled to `[-scale, scale)`.
    fn scaled(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        values(n, seed).into_iter().map(|v| v * scale).collect()
    }

    fn f16(x: &[f32]) -> Vec<u8> {
        let mut out = vec![0; 2 * x.len()];
        scalar::to_f16(x, &mut out);
        out
    }

    /// `|got - want| <= rel * scale`, with a message.
    fn assert_near(got: f32, want: f32, rel: f64, scale: f64, what: &str) {
        let err = (f64::from(got) - f64::from(want)).abs();
        assert!(
            err <= rel * scale,
            "{what}: {got} vs {want} (error {err:e}, allowed {:e})",
            rel * scale
        );
    }

    fn avx2() -> Option<Avx2> {
        let k = Avx2::detect();
        if k.is_none() {
            eprintln!("skipped: this CPU has no AVX2/FMA/F16C");
        }
        k
    }

    /// Lengths that cover whole steps of 32, partial groups of 8 and tails.
    const LENGTHS: [usize; 8] = [1, 7, 8, 31, 32, 100, 256, 1024];

    #[test]
    fn hex_float_constants_have_the_right_values() {
        let cases = [
            (0x4B40_0000, 12_582_912.0),
            (0x3FB8_AA3B, std::f64::consts::LOG2_E),
            (0x3F31_7200, 0.693_145_751_953_125),
            (0x35BF_BE8E, 1.428_606_765_330_187e-6),
            (0x3F7F_FFF6, 0.999_999_403_953_552_2),
        ];
        for (bits, value) in cases {
            assert!(
                (f64::from(f32::from_bits(bits)) - value).abs() <= value * 1e-7,
                "{bits:#x}"
            );
        }
    }

    #[test]
    fn encoders_match_the_scalar_reference_exactly() {
        let Some(k) = avx2() else { return };
        for dt in [DotType::F32, DotType::Bf16, DotType::Q8_0] {
            for n in [32, 64, 96, 1024] {
                for seed in 1..4 {
                    // Mixed magnitudes, including an all-zero Q8_0 block and
                    // values that round to f16 subnormals.
                    let mut x = scaled(n, seed, 4.0);
                    x[..8].copy_from_slice(&[1e-6, -3e-5, 6.5e-5, 0.0, -0.0, 65504.0, 1e9, -1e-30]);
                    if n > 32 {
                        x[32..64].fill(0.0);
                    }
                    let want = quant::encoded(dt, &x);
                    let mut got = vec![0xAA; want.len()];
                    k.encode(dt, &x, &mut got);
                    assert_eq!(got, want, "{dt:?}, n = {n}, seed {seed}");
                }
            }
        }
        // The tail of the bf16 encoder.
        let x = values(13, 9);
        let want = quant::encoded(DotType::Bf16, &x);
        let mut got = vec![0; want.len()];
        k.encode(DotType::Bf16, &x, &mut got);
        assert_eq!(got, want, "bf16 tail");
        // NaN handling of the bf16 encoder.
        let x = [
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::from_bits(0x7F80_0001),
            1.0,
            -2.0,
            3.5,
            0.1,
        ];
        let mut got = vec![0; 16];
        k.encode(DotType::Bf16, &x, &mut got);
        assert_eq!(got, quant::encoded(DotType::Bf16, &x));
    }

    /// The tolerance is relative to `Σ|w||x|`: only the f32 summation order
    /// differs (and, for Q8_0, nothing but the order of the block sums).
    #[test]
    fn dot_rows_match_the_scalar_reference() {
        let Some(k) = avx2() else { return };
        for ty in [GgmlType::F32, GgmlType::F16, GgmlType::Bf16, GgmlType::Q8_0] {
            let dt = DotType::of(ty);
            let lengths: &[usize] = if ty == GgmlType::Q8_0 {
                &[32, 96, 1024]
            } else {
                &[8, 100, 1024]
            };
            for &n in lengths {
                let rows = 5;
                let w: Vec<f32> = (0..rows).flat_map(|r| values(n, 10 + r as u32)).collect();
                let x = values(n, 3);
                // F16 rows are f16; every other type's rows use its dot
                // type's encoding.
                let rows_q: Vec<u8> = w
                    .chunks(n)
                    .flat_map(|r| {
                        if ty == GgmlType::F16 {
                            f16(r)
                        } else {
                            quant::encoded(dt, r)
                        }
                    })
                    .collect();
                let xq = quant::encoded(dt, &x);
                let row_bytes = rows_q.len() / rows;
                let mut got = vec![0.0; rows];
                k.dot_rows(ty, &rows_q, row_bytes, &xq, &mut got);
                for (r, &g) in got.iter().enumerate() {
                    let row = &rows_q[r * row_bytes..(r + 1) * row_bytes];
                    let want = quant::dot_row(ty, row, &xq);
                    let scale: f64 = w[r * n..(r + 1) * n]
                        .iter()
                        .zip(&x)
                        .map(|(a, b)| f64::from(a.abs() * b.abs()))
                        .sum();
                    assert_near(g, want, 1e-6, scale, &format!("{ty} n = {n} row {r}"));
                }
            }
        }
    }

    /// Q8_0's per-block integer dot is exact, so one block agrees to the
    /// last bit, including the -128 weights that a converter may emit.
    #[test]
    fn q8_0_single_block_is_exact() {
        let Some(k) = avx2() else { return };
        let mut row = quant::encoded(DotType::Q8_0, &values(QK8_0, 7));
        row[2] = (-128i8) as u8;
        row[9] = (-128i8) as u8;
        let xq = quant::encoded(DotType::Q8_0, &values(QK8_0, 8));
        let mut got = [0.0];
        k.dot_rows(GgmlType::Q8_0, &row, row.len(), &xq, &mut got);
        assert_eq!(got[0], quant::dot_row(GgmlType::Q8_0, &row, &xq));
    }

    #[test]
    fn vector_dots_match_the_scalar_reference() {
        let Some(k) = avx2() else { return };
        for n in LENGTHS {
            let (a, b) = (values(n, 1), values(n, 2));
            let scale: f64 = a
                .iter()
                .zip(&b)
                .map(|(x, y)| f64::from(x.abs() * y.abs()))
                .sum();
            // SAFETY: `k` proves the features are available.
            let got = unsafe { dot_f32(&a, &b) };
            assert_near(
                got,
                scalar::dot_f32(&a, &b),
                1e-6,
                scale,
                &format!("dot_f32 n = {n}"),
            );
            let a16 = f16(&a);
            let mut got = [0.0];
            k.dot_f16_f32_rows(&a16, a16.len(), &b, &mut got);
            let got = got[0];
            assert_near(
                got,
                scalar::dot_f16_f32(&a16, &b),
                1e-6,
                scale,
                &format!("dot_f16_f32 n = {n}"),
            );
        }
    }

    #[test]
    fn to_f16_matches_the_scalar_reference_exactly() {
        let Some(k) = avx2() else { return };
        for n in LENGTHS {
            let x = scaled(n, 4, 70_000.0);
            let mut got = vec![0; 2 * n];
            k.to_f16(&x, &mut got);
            assert_eq!(got, f16(&x), "n = {n}");
        }
    }

    #[test]
    fn exp_is_accurate_and_handles_the_extremes() {
        let Some(_) = avx2() else { return };
        let mut xs: Vec<f32> = (-1040..=880).map(|i| i as f32 / 10.0 + 0.037).collect();
        xs.extend([
            0.0,
            -0.0,
            88.3,
            -87.3,
            -103.0,
            -104.0,
            89.0,
            f32::NEG_INFINITY,
            f32::INFINITY,
        ]);
        xs.resize(xs.len().next_multiple_of(8), 0.0);
        for group in xs.as_chunks::<8>().0 {
            let mut out = [0.0; 8];
            // SAFETY: `avx2()` returned the token, so the features are available.
            unsafe { store_f32(&mut out, exp(load_f32(group))) };
            for (&x, &got) in group.iter().zip(&out) {
                let want = f64::from(x).exp();
                if want > f64::from(f32::MAX) {
                    assert_eq!(got, f32::INFINITY, "e^{x}");
                } else if want < 1e-38 {
                    // Subnormal results flush to zero below -103.97.
                    assert!(f64::from(got) <= 2e-38, "e^{x} = {got}");
                } else {
                    let rel = (f64::from(got) - want).abs() / want;
                    assert!(
                        rel < 3e-7,
                        "e^{x}: {got} vs {want} (relative error {rel:e})"
                    );
                }
            }
        }
    }

    #[test]
    fn activations_match_the_scalar_reference() {
        let Some(k) = avx2() else { return };
        for n in LENGTHS {
            let g = scaled(n, 5, 20.0);
            let u = values(n, 6);
            let mut got = g.clone();
            k.silu(&mut got);
            let mut want = g.clone();
            scalar::silu_in_place(&mut want);
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert_near(
                    a,
                    b,
                    1e-6,
                    f64::from(b.abs()).max(1e-30),
                    &format!("silu[{i}] of {}", g[i]),
                );
            }
            let mut got = g.clone();
            k.swiglu(&mut got, &u);
            let mut want = g.clone();
            scalar::swiglu(&mut want, &u);
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert_near(
                    a,
                    b,
                    1e-6,
                    f64::from(b.abs()).max(1e-30),
                    &format!("swiglu[{i}]"),
                );
            }
        }
    }

    #[test]
    fn softmax_matches_the_scalar_reference() {
        let Some(k) = avx2() else { return };
        for n in LENGTHS {
            let x = scaled(n, 7, 30.0);
            let mut got = x.clone();
            k.softmax(&mut got);
            let mut want = x.clone();
            scalar::softmax(&mut want);
            for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert_near(
                    a,
                    b,
                    2e-6,
                    f64::from(b).max(1e-30),
                    &format!("softmax n = {n} [{i}]"),
                );
            }
            let total: f64 = got.iter().map(|&v| f64::from(v)).sum();
            assert!((total - 1.0).abs() < 1e-5, "n = {n}: sums to {total}");
        }
    }

    #[test]
    fn delta_net_head_matches_the_scalar_reference() {
        let Some(k) = avx2() else { return };
        // Unit-norm q and k, as after the model's l2 norm; with larger keys
        // the recurrence is unstable and any rounding difference explodes.
        let unit = |v: Vec<f32>| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect::<Vec<_>>()
        };
        for d in [8, 32, 128] {
            let (q, kk, v) = (unit(values(d, 1)), unit(values(d, 2)), values(d, 3));
            let step = DeltaStep {
                q: &q,
                k: &kk,
                v: &v,
                beta: 0.7,
                decay: 0.95,
                scale: 1.0 / (d as f32).sqrt(),
            };
            let state = values(d * d, 4);
            let (mut s_got, mut s_want) = (state.clone(), state);
            let (mut o_got, mut o_want) = (vec![0.0; d], vec![0.0; d]);
            // Several tokens, so the state differences would compound.
            for _ in 0..4 {
                k.delta_net_head(&mut s_got, &step, &mut o_got);
                scalar::delta_net_head(&mut s_want, &step, &mut o_want);
            }
            for (i, (&a, &b)) in o_got.iter().zip(&o_want).enumerate() {
                assert_near(a, b, 1e-5, 1.0, &format!("d = {d} out[{i}]"));
            }
            for (i, (&a, &b)) in s_got.iter().zip(&s_want).enumerate() {
                assert_near(a, b, 1e-5, 1.0, &format!("d = {d} state[{i}]"));
            }
        }
    }
}
