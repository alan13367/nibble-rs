//! AArch64 NEON kernels.
//!
//! Each kernel transcribes the ggml function named in its documentation:
//! the NEON `vec.cpp` / `arch/arm/quants.c` accumulator layout and reduction
//! order, and the fused multiply-adds clang makes of `a * b + c` there. The
//! reference is llama.cpp's macOS arm64 release build, which enables
//! `dotprod` but neither fp16 arithmetic nor i8mm, with weight repacking off
//! (`--no-repack`: repacked weights use other kernels). The tests compare
//! the kernels with that build's results bit for bit. [`dot_bf16`] is the
//! one deliberate exception.
//!
//! Integer dots use `vdotq_s32` when the CPU has `dotprod` and ggml's `vmull`
//! fallback otherwise; f16 conversions use `vcvt` when the CPU has `fp16`.
//! Both features are optional on baseline AArch64, so a machine without
//! them never executes those intrinsics.
//!
//! The scalar code in [`super::scalar`] and [`crate::quant`] is the reference
//! each kernel is tested against.

use std::arch::aarch64::*;

use nibble_gguf::GgmlType;

use super::scalar::{self, DeltaStep};
use crate::quant::{
    self, DotType, IQ4_XS_BLOCK_BYTES, Q3_K_BLOCK_BYTES, Q4_K_BLOCK_BYTES, Q5_K_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, Q8_0_BLOCK_BYTES, Q8_K_BLOCK_BYTES, QK_K, QK8_0,
};

/// Calls a NEON kernel. The comment sits on the `unsafe` block so every
/// expansion carries it.
macro_rules! neon_call {
    ($expr:expr) => {{
        // SAFETY: the caller holds a `Neon` token, so NEON is available, and
        // it only invokes a kernel whose extra features (`dotprod`, `fp16`)
        // it has already checked.
        unsafe { $expr }
    }};
}

/// Proof that this CPU has NEON. Only [`Neon::detect`] creates one.
/// `dotprod` and `fp16` are recorded separately: they are not part of
/// baseline AArch64, and the corresponding intrinsics are not executed
/// unless the flag is set.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Neon {
    dotprod: bool,
    fp16: bool,
    i8mm: bool,
}

impl Neon {
    /// The token, if this CPU has NEON.
    #[must_use]
    pub(crate) fn detect() -> Option<Self> {
        if !std::arch::is_aarch64_feature_detected!("neon") {
            return None;
        }
        Some(Self {
            dotprod: std::arch::is_aarch64_feature_detected!("dotprod"),
            fp16: std::arch::is_aarch64_feature_detected!("fp16"),
            i8mm: std::arch::is_aarch64_feature_detected!("i8mm"),
        })
    }

    /// See [`super::Kernels::encode`].
    pub(crate) fn encode(self, dt: DotType, x: &[f32], out: &mut [u8]) {
        match dt {
            DotType::F32 | DotType::Bf16 | DotType::Q8_1 => quant::encode_activation(dt, x, out),
            DotType::Q8_0 => neon_call!(quantize_q8_0(x, out)),
            DotType::Q8K => neon_call!(quantize_q8_k(x, out)),
        }
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
        let scalar = |out: &mut [f32]| {
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                *o = quant::dot_row(ty, row, xq);
            }
        };
        match ty {
            GgmlType::F32 => neon_call!(dot_f32_rows(rows, row_bytes, xq, out)),
            GgmlType::F16 => {
                if self.fp16 {
                    neon_call!(dot_f16_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_f16_rows_soft(rows, row_bytes, xq, out))
                }
            }
            GgmlType::Bf16 => neon_call!(dot_bf16_rows(rows, row_bytes, xq, out)),
            GgmlType::Q8_0 => {
                if self.dotprod && self.fp16 {
                    neon_call!(dot_q8_0_rows_fp16(rows, row_bytes, xq, out))
                } else if self.dotprod {
                    neon_call!(dot_q8_0_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_q8_0_rows_mul(rows, row_bytes, xq, out))
                }
            }
            GgmlType::Q4K => {
                if self.dotprod {
                    neon_call!(dot_q4_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_q4_rows_mul(rows, row_bytes, xq, out))
                }
            }
            GgmlType::Q5K => {
                if self.dotprod {
                    neon_call!(dot_q5_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_q5_rows_mul(rows, row_bytes, xq, out))
                }
            }
            GgmlType::Q6K => {
                if self.dotprod {
                    neon_call!(dot_q6_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_q6_rows_mul(rows, row_bytes, xq, out))
                }
            }
            GgmlType::Q3K => {
                if self.dotprod {
                    neon_call!(dot_q3_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_q3_rows_mul(rows, row_bytes, xq, out))
                }
            }
            GgmlType::Iq4Xs => {
                if self.dotprod {
                    neon_call!(dot_iq4_xs_rows(rows, row_bytes, xq, out))
                } else {
                    neon_call!(dot_iq4_xs_rows_mul(rows, row_bytes, xq, out))
                }
            }
            ty if quant::is_block32(ty) && self.dotprod => {
                neon_call!(dot_block32_rows(ty, rows, row_bytes, xq, out))
            }
            GgmlType::Q2_0 if self.dotprod && self.fp16 => {
                neon_call!(dot_q2_0_rows_fp16(rows, row_bytes, xq, out))
            }
            GgmlType::Q2_0 if self.dotprod => neon_call!(dot_q2_0_rows(rows, row_bytes, xq, out)),
            _ => scalar(out),
        }
    }

    /// See [`super::Kernels::dot_rows_n`].
    pub(crate) fn dot_rows_n<const N: usize>(
        self,
        ty: GgmlType,
        rows: &[u8],
        row_bytes: usize,
        xq: [&[u8]; N],
        out: [&mut [f32]; N],
    ) -> bool {
        match ty {
            GgmlType::Q8_0 if self.dotprod && self.fp16 => {
                neon_call!(dot_q8_0_rows_n(rows, row_bytes, xq, out));
            }
            GgmlType::Q4K if self.i8mm && N == 4 => {
                neon_call!(dot_q4_rows_i8mm::<N, 2>(rows, row_bytes, xq, out));
            }
            GgmlType::Q4K if self.dotprod => neon_call!(dot_q4_rows_n(rows, row_bytes, xq, out)),
            GgmlType::Q5K if self.i8mm && N == 4 => {
                neon_call!(dot_q5_rows_i8mm::<N, 2>(rows, row_bytes, xq, out));
            }
            GgmlType::Q5K if self.dotprod => neon_call!(dot_q5_rows_n(rows, row_bytes, xq, out)),
            GgmlType::Q6K if self.i8mm && N == 4 => {
                neon_call!(dot_q6_rows_i8mm::<N, 2>(rows, row_bytes, xq, out));
            }
            GgmlType::Q6K if self.dotprod => neon_call!(dot_q6_rows_n(rows, row_bytes, xq, out)),
            GgmlType::Bf16 => neon_call!(dot_bf16_rows_n(rows, row_bytes, xq, out)),
            GgmlType::F16 if self.fp16 => neon_call!(dot_f16_rows_n(rows, row_bytes, xq, out)),
            _ => return false,
        }
        true
    }

    /// See [`super::Kernels::dot_rows_prompt`].
    pub(crate) fn dot_rows_prompt<const N: usize>(
        self,
        ty: GgmlType,
        rows: &[u8],
        row_bytes: usize,
        xq: [&[u8]; N],
        out: [&mut [f32]; N],
    ) -> bool {
        if self.i8mm {
            match (ty, N) {
                (GgmlType::Q8_0, 8) if self.fp16 => {
                    neon_call!(dot_q8_0_rows_i8mm::<N, 4>(rows, row_bytes, xq, out));
                    return true;
                }
                (GgmlType::Q8_0, 4) if self.fp16 => {
                    neon_call!(dot_q8_0_rows_i8mm::<N, 2>(rows, row_bytes, xq, out));
                    return true;
                }
                (GgmlType::Q4K, 8) => {
                    neon_call!(dot_q4_rows_i8mm::<N, 4>(rows, row_bytes, xq, out));
                    return true;
                }
                (GgmlType::Q5K, 8) => {
                    neon_call!(dot_q5_rows_i8mm::<N, 4>(rows, row_bytes, xq, out));
                    return true;
                }
                (GgmlType::Q6K, 8) => {
                    neon_call!(dot_q6_rows_i8mm::<N, 4>(rows, row_bytes, xq, out));
                    return true;
                }
                _ => {}
            }
        }
        if N <= 4 {
            self.dot_rows_n(ty, rows, row_bytes, xq, out)
        } else {
            false
        }
    }

    /// See [`super::Kernels::dot_f16_f32_rows`].
    pub(crate) fn dot_f16_f32_rows(self, a: &[u8], stride: usize, b: &[f32], out: &mut [f32]) {
        if self.fp16 {
            neon_call!(dot_f16_f32_rows(a, stride, b, out));
        } else {
            for (r, o) in out.iter_mut().enumerate() {
                *o = neon_call!(dot_f16_f32_soft(
                    &a[r * stride..r * stride + 2 * b.len()],
                    b
                ));
            }
        }
    }

    /// See [`super::Kernels::to_f16`].
    pub(crate) fn to_f16(self, x: &[f32], out: &mut [u8]) {
        if self.fp16 {
            neon_call!(to_f16(x, out))
        } else {
            scalar::to_f16(x, out);
        }
    }

    /// See [`super::Kernels::silu`].
    pub(crate) fn silu(self, x: &mut [f32]) {
        neon_call!(silu(x))
    }

    /// See [`super::Kernels::swiglu`].
    pub(crate) fn swiglu(self, g: &mut [f32], u: &[f32]) {
        neon_call!(swiglu(g, u))
    }

    /// See [`super::Kernels::softmax`].
    pub(crate) fn softmax(self, x: &mut [f32]) {
        neon_call!(softmax(x))
    }

    /// See [`super::Kernels::delta_net_head`].
    pub(crate) fn delta_net_head(self, state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
        neon_call!(delta_net_head(state, step, out))
    }
}

// --- Loads and stores -------------------------------------------------------

#[inline]
#[target_feature(enable = "neon")]
fn load_f32(a: &[f32; 4]) -> float32x4_t {
    // SAFETY: `a` is 16 readable bytes; the load has no alignment requirement.
    unsafe { vld1q_f32(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_f32(a: &mut [f32; 4], v: float32x4_t) {
    // SAFETY: `a` is 16 writable bytes; the store has no alignment requirement.
    unsafe { vst1q_f32(a.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_u8(a: &[u8; 16]) -> uint8x16_t {
    // SAFETY: `a` is 16 readable bytes; the load has no alignment requirement.
    unsafe { vld1q_u8(a.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i8(a: &[u8; 16]) -> int8x16_t {
    // SAFETY: `a` is 16 readable bytes; the load has no alignment requirement.
    unsafe { vld1q_s8(a.as_ptr().cast()) }
}

/// Sixteen bytes at `at` inside `bytes`.
#[inline]
fn at16(bytes: &[u8], at: usize) -> &[u8; 16] {
    bytes[at..at + 16].try_into().unwrap()
}

/// Four f16 values, widened exactly.
#[inline]
#[target_feature(enable = "neon,fp16")]
fn load_f16_4(a: &[u8; 8]) -> float32x4_t {
    // SAFETY: `a` is 8 readable bytes; the load has no alignment requirement.
    let bits = unsafe { vld1_u16(a.as_ptr().cast()) };
    vcvt_f32_f16(vreinterpret_f16_u16(bits))
}

/// Four f32 values narrowed to f16, round to nearest even.
#[inline]
#[target_feature(enable = "neon,fp16")]
fn store_f16_4(a: &mut [u8; 8], v: float32x4_t) {
    let bits = vreinterpret_u16_f16(vcvt_f16_f32(v));
    // SAFETY: `a` is 8 writable bytes; the store has no alignment requirement.
    unsafe { vst1_u16(a.as_mut_ptr().cast(), bits) }
}

/// Eight bf16 values widened by shifting into the high half of an f32.
#[inline]
#[target_feature(enable = "neon")]
fn load_bf16_8(a: &[u8; 16]) -> (float32x4_t, float32x4_t) {
    // SAFETY: `a` is 16 readable bytes; the load has no alignment requirement.
    let u = unsafe { vld1q_u16(a.as_ptr().cast()) };
    let lo = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_low_u16(u))));
    let hi = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(vget_high_u16(u))));
    (lo, hi)
}

/// `dotprod` dot of 16 `i8` pairs, added into `acc`'s four lanes (four
/// products each), as `vdotq_s32`.
#[inline]
#[target_feature(enable = "neon,dotprod")]
fn i8_dot16(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    vdotq_s32(acc, a, b)
}

/// The same 16 products without `dotprod`: `vmull` plus a pairwise sum, as
/// ggml's `ggml_vdotq_s32` fallback. `vaddvq_s32` of the result is the exact
/// integer dot, but the lanes hold different groups of products than
/// `vdotq_s32` puts in them, which matters where lanes are scaled (Q8_0).
#[inline]
#[target_feature(enable = "neon")]
fn i8_dot16_mul(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let hi = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    vaddq_s32(acc, vaddq_s32(vpaddlq_s16(lo), vpaddlq_s16(hi)))
}

/// ggml's NEON `GGML_F32x4_REDUCE` over four accumulators: pairwise adds,
/// then `vaddvq`.
#[inline]
#[target_feature(enable = "neon")]
fn reduce4(mut s: [float32x4_t; 4]) -> f32 {
    s[0] = vaddq_f32(s[0], s[2]);
    s[1] = vaddq_f32(s[1], s[3]);
    s[0] = vaddq_f32(s[0], s[1]);
    vaddvq_f32(s[0])
}

// --- exp, SiLU, softmax -----------------------------------------------------

/// `ggml_v_expf` for NEON (`vec.h`). At most about 1.45 ulp of error; above
/// 88.38 the result overflows to infinity, below -103.97 it flushes to zero.
#[inline]
#[target_feature(enable = "neon")]
fn exp4(x: float32x4_t) -> float32x4_t {
    let r = vdupq_n_f32(f32::from_bits(0x4B40_0000));
    let z = vfmaq_f32(r, x, vdupq_n_f32(f32::from_bits(0x3FB8_AA3B)));
    let n = vsubq_f32(z, r);
    let b = vfmsq_f32(
        vfmsq_f32(x, n, vdupq_n_f32(f32::from_bits(0x3F31_7200))),
        n,
        vdupq_n_f32(f32::from_bits(0x35BF_BE8E)),
    );
    let e = vshlq_n_u32::<23>(vreinterpretq_u32_f32(z));
    let k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1.0))));
    let c = vcagtq_f32(n, vdupq_n_f32(126.0));
    let u = vmulq_f32(b, b);
    let j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(f32::from_bits(0x3F7F_FFF6)), b),
        vfmaq_f32(
            vfmaq_f32(
                vdupq_n_f32(f32::from_bits(0x3EFF_FEDB)),
                vdupq_n_f32(f32::from_bits(0x3E2A_AF33)),
                b,
            ),
            vfmaq_f32(
                vdupq_n_f32(f32::from_bits(0x3D2B_9F17)),
                vdupq_n_f32(f32::from_bits(0x3C07_2010)),
                b,
            ),
            u,
        ),
        u,
    );
    // Every lane is near 1: the two-step scaling below would be an identity.
    if vaddvq_u32(c) == 0 {
        return vfmaq_f32(k, j, k);
    }
    let d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x8200_0000));
    let s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7F00_0000)));
    let s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    vbslq_f32(
        vcagtq_f32(n, vdupq_n_f32(192.0)),
        vmulq_f32(s1, s1),
        vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, s2, j), s1), vfmaq_f32(k, k, j)),
    )
}

/// `ggml_v_silu`: `x / (1 + e^-x)`.
#[inline]
#[target_feature(enable = "neon")]
fn silu4(x: float32x4_t) -> float32x4_t {
    let neg = vsubq_f32(vdupq_n_f32(0.0), x);
    vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0), exp4(neg)))
}

/// `ggml_vec_silu_f32`, groups of four.
#[target_feature(enable = "neon")]
fn silu(x: &mut [f32]) {
    let (groups, tail) = x.as_chunks_mut::<4>();
    for g in groups {
        store_f32(g, silu4(load_f32(g)));
    }
    scalar::silu_in_place(tail);
}

/// `ggml_vec_swiglu_f32`: `g = silu(g) * u`.
#[target_feature(enable = "neon")]
fn swiglu(g: &mut [f32], u: &[f32]) {
    let n = g.len().min(u.len());
    let (gs, g_tail) = g[..n].as_chunks_mut::<4>();
    let (us, u_tail) = u[..n].as_chunks::<4>();
    for (g, u) in gs.iter_mut().zip(us) {
        store_f32(g, vmulq_f32(silu4(load_f32(g)), load_f32(u)));
    }
    scalar::swiglu(g_tail, u_tail);
}

/// `ggml_vec_soft_max_f32` for NEON: groups of four with [`exp4`], each
/// group's `vaddvq` added to an f64 sum. The tail uses the scalar `exp`,
/// as the leftover loop in `vec.cpp` does.
#[target_feature(enable = "neon")]
fn softmax(x: &mut [f32]) {
    let max = vdupq_n_f32(x.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let mut sum = 0.0f64;
    let (groups, tail) = x.as_chunks_mut::<4>();
    for g in groups {
        let e = exp4(vsubq_f32(load_f32(g), max));
        store_f32(g, e);
        sum += f64::from(vaddvq_f32(e));
    }
    let max_s = vgetq_lane_f32::<0>(max);
    for v in tail {
        *v = (*v - max_s).exp();
        sum += f64::from(*v);
    }
    let inv = (1.0 / sum) as f32;
    for v in x {
        *v *= inv;
    }
}

// --- Floating dots and stores ------------------------------------------------

/// `ggml_vec_dot_f32` for NEON: four `float32x4` accumulators, step 16,
/// [`reduce4`], then the tail in f32 with fused `sumf += x[i]*y[i]`.
#[target_feature(enable = "neon")]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (a_steps, a_tail) = a[..n].as_chunks::<16>();
    let (b_steps, b_tail) = b[..n].as_chunks::<16>();
    let mut sum = [vdupq_n_f32(0.0); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        let (a, b) = (a.as_chunks::<4>().0, b.as_chunks::<4>().0);
        for i in 0..4 {
            sum[i] = vfmaq_f32(sum[i], load_f32(&a[i]), load_f32(&b[i]));
        }
    }
    let mut sumf = reduce4(sum);
    for (&a, &b) in a_tail.iter().zip(b_tail) {
        sumf = a.mul_add(b, sumf);
    }
    sumf
}

/// Byte-wise f32 dot: little-endian loads, same reduction as [`dot_f32`].
#[target_feature(enable = "neon")]
fn dot_f32_bytes(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len()) / 4 * 4;
    let (a_steps, a_tail) = a[..n].as_chunks::<64>();
    let (b_steps, b_tail) = b[..n].as_chunks::<64>();
    let mut sum = [vdupq_n_f32(0.0); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        for i in 0..4 {
            // SAFETY: each slice is 16 readable bytes carved from a 64-byte chunk.
            let (av, bv) = unsafe {
                (
                    vld1q_f32(a[i * 16..].as_ptr().cast()),
                    vld1q_f32(b[i * 16..].as_ptr().cast()),
                )
            };
            sum[i] = vfmaq_f32(sum[i], av, bv);
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

#[target_feature(enable = "neon")]
fn dot_f32_rows(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
        *o = dot_f32_bytes(row, xq);
    }
}

/// Four f16 values widened with the scalar bit conversion, for CPUs
/// without `fp16`; the values are the same as [`load_f16_4`]'s.
#[inline]
#[target_feature(enable = "neon")]
fn load_f16_4_soft(a: &[u8; 8]) -> float32x4_t {
    let mut v = [0.0f32; 4];
    for (v, h) in v.iter_mut().zip(a.as_chunks::<2>().0) {
        *v = quant::f16_to_f32(u16::from_le_bytes(*h));
    }
    load_f32(&v)
}

/// Four little-endian f32 values.
#[inline]
#[target_feature(enable = "neon")]
fn load_f32_le(a: &[u8; 16]) -> float32x4_t {
    // SAFETY: `a` is 16 readable bytes; the load has no alignment
    // requirement, and the module only exists on little-endian targets.
    unsafe { vld1q_f32(a.as_ptr().cast()) }
}

/// Values per chunk of ggml's ARM `ggml_vec_dot_f16_f32`.
const F16_CHUNK: usize = 64;

// `ggml_vec_dot_f16_f32` on ARM (`GGML_SIMD` without AVX) widens the f16
// vector 64 values at a time and runs `ggml_vec_dot_f32` ([`dot_f32`]) on
// each chunk, adding the chunk sums in f64. Widening is exact, so the chunk
// dot is computed straight from the f16 values here, without the copy.
// Without `fp16` the widening uses the scalar conversion, which gives the
// same values.
macro_rules! define_f16_f32 {
    ($name:ident, $rows:ident, $feat:literal, $load:ident) => {
        /// `a · b` for f16 bytes `a` and f32 values `b`.
        #[target_feature(enable = $feat)]
        fn $name(a: &[u8], b: &[f32]) -> f32 {
            let n = (a.len() / 2).min(b.len());
            let mut sumf = 0.0f64;
            for (a, b) in a[..2 * n]
                .chunks(2 * F16_CHUNK)
                .zip(b[..n].chunks(F16_CHUNK))
            {
                // `ggml_vec_dot_f32` of the chunk: four accumulators, steps
                // of 16, [`reduce4`], then the tail with fused multiply-adds.
                let (a_steps, a_tail) = a.as_chunks::<32>();
                let (b_steps, b_tail) = b.as_chunks::<16>();
                let mut acc = [vdupq_n_f32(0.0); 4];
                for (a, b) in a_steps.iter().zip(b_steps) {
                    let (a, b) = (a.as_chunks::<8>().0, b.as_chunks::<4>().0);
                    for i in 0..4 {
                        acc[i] = vfmaq_f32(acc[i], $load(&a[i]), load_f32(&b[i]));
                    }
                }
                let mut chunk = reduce4(acc);
                for (a, &b) in a_tail.as_chunks::<2>().0.iter().zip(b_tail) {
                    chunk = quant::f16_to_f32(u16::from_le_bytes(*a)).mul_add(b, chunk);
                }
                sumf += f64::from(chunk);
            }
            sumf as f32
        }

        /// F16 weight rows times an activation of little-endian f32 bytes,
        /// in the same chunks.
        #[target_feature(enable = $feat)]
        fn $rows(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let n = (row.len() / 2).min(xq.len() / 4);
                let mut sumf = 0.0f64;
                for (a, b) in row[..2 * n]
                    .chunks(2 * F16_CHUNK)
                    .zip(xq[..4 * n].chunks(4 * F16_CHUNK))
                {
                    let (a_steps, a_tail) = a.as_chunks::<32>();
                    let (b_steps, b_tail) = b.as_chunks::<64>();
                    let mut acc = [vdupq_n_f32(0.0); 4];
                    for (a, b) in a_steps.iter().zip(b_steps) {
                        let (a, b) = (a.as_chunks::<8>().0, b.as_chunks::<16>().0);
                        for i in 0..4 {
                            acc[i] = vfmaq_f32(acc[i], $load(&a[i]), load_f32_le(&b[i]));
                        }
                    }
                    let mut chunk = reduce4(acc);
                    for (a, b) in a_tail
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .zip(b_tail.as_chunks::<4>().0)
                    {
                        let a = quant::f16_to_f32(u16::from_le_bytes(*a));
                        chunk = a.mul_add(f32::from_le_bytes(*b), chunk);
                    }
                    sumf += f64::from(chunk);
                }
                *o = sumf as f32;
            }
        }
    };
}

define_f16_f32!(dot_f16_f32, dot_f16_rows, "neon,fp16", load_f16_4);
define_f16_f32!(dot_f16_f32_soft, dot_f16_rows_soft, "neon", load_f16_4_soft);

/// [`dot_f16_f32`] of the f16 rows at `a[r * stride..]` with `b`, into
/// `out[r]`. Attention calls it with many short rows, so four rows go
/// through each chunk together and share the loads of `b`; every row keeps
/// its own accumulators, so each result is exactly the single-row one.
#[target_feature(enable = "neon,fp16")]
fn dot_f16_f32_rows(a: &[u8], stride: usize, b: &[f32], out: &mut [f32]) {
    let n = b.len();
    let row = |r: usize| &a[r * stride..r * stride + 2 * n];
    let (groups, rest) = out.as_chunks_mut::<4>();
    let done = 4 * groups.len();
    for (g, o) in groups.iter_mut().enumerate() {
        let rows = [row(4 * g), row(4 * g + 1), row(4 * g + 2), row(4 * g + 3)];
        let mut sumf = [0.0f64; 4];
        for (c, b) in b.chunks(F16_CHUNK).enumerate() {
            let at = 2 * F16_CHUNK * c;
            let chunks = rows.map(|r| &r[at..at + 2 * b.len()]);
            let steps = chunks.map(|c| c.as_chunks::<32>().0);
            let (b_steps, b_tail) = b.as_chunks::<16>();
            let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
            for (s, b) in b_steps.iter().enumerate() {
                let b = b.as_chunks::<4>().0;
                let bv = [
                    load_f32(&b[0]),
                    load_f32(&b[1]),
                    load_f32(&b[2]),
                    load_f32(&b[3]),
                ];
                for (acc, steps) in acc.iter_mut().zip(&steps) {
                    let a = steps[s].as_chunks::<8>().0;
                    for i in 0..4 {
                        acc[i] = vfmaq_f32(acc[i], load_f16_4(&a[i]), bv[i]);
                    }
                }
            }
            let tail = 2 * 16 * b_steps.len();
            for ((sumf, acc), chunk) in sumf.iter_mut().zip(acc).zip(&chunks) {
                let mut total = reduce4(acc);
                for (a, &b) in chunk[tail..].as_chunks::<2>().0.iter().zip(b_tail) {
                    total = quant::f16_to_f32(u16::from_le_bytes(*a)).mul_add(b, total);
                }
                *sumf += f64::from(total);
            }
        }
        for (o, s) in o.iter_mut().zip(sumf) {
            *o = s as f32;
        }
    }
    for (r, o) in rest.iter_mut().enumerate() {
        *o = dot_f16_f32(row(done + r), b);
    }
}

/// bf16 dot: widen by a 16-bit shift, four f32 accumulators, step 16, the
/// tail in f32.
///
/// This is the one deliberate departure from llama.cpp in this module. ggml
/// has no NEON `ggml_vec_dot_bf16`: on ARM it multiplies one element at a
/// time and sums in f64, and llama.cpp decodes the BF16 Qwen3-0.6B at about
/// 29 tok/s on an M4 Max where nibble reaches about 170. The sums differ in
/// the last bits, and since the next product rounds its input to bf16, that
/// can move an activation by a bf16 step: greedy BF16 output can differ from
/// llama.cpp on ARM at a near-tie (one of the 30 comparison runs does). On
/// x86 nibble follows ggml's AVX2 kernel exactly.
#[target_feature(enable = "neon")]
fn dot_bf16(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len()) / 2 * 2;
    let (a_steps, a_tail) = a[..n].as_chunks::<32>();
    let (b_steps, b_tail) = b[..n].as_chunks::<32>();
    let mut sum = [vdupq_n_f32(0.0); 4];
    for (a, b) in a_steps.iter().zip(b_steps) {
        let (a0, a1) = load_bf16_8((&a[..16]).try_into().unwrap());
        let (a2, a3) = load_bf16_8((&a[16..]).try_into().unwrap());
        let (b0, b1) = load_bf16_8((&b[..16]).try_into().unwrap());
        let (b2, b3) = load_bf16_8((&b[16..]).try_into().unwrap());
        sum[0] = vfmaq_f32(sum[0], a0, b0);
        sum[1] = vfmaq_f32(sum[1], a1, b1);
        sum[2] = vfmaq_f32(sum[2], a2, b2);
        sum[3] = vfmaq_f32(sum[3], a3, b3);
    }
    let mut sumf = reduce4(sum);
    for (a, b) in a_tail
        .as_chunks::<2>()
        .0
        .iter()
        .zip(b_tail.as_chunks::<2>().0)
    {
        sumf +=
            quant::bf16_to_f32(u16::from_le_bytes(*a)) * quant::bf16_to_f32(u16::from_le_bytes(*b));
    }
    sumf
}

#[target_feature(enable = "neon")]
fn dot_bf16_rows(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
    for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
        *o = dot_bf16(row, xq);
    }
}

/// f32 to f16, round to nearest even, groups of four.
#[target_feature(enable = "neon,fp16")]
fn to_f16(x: &[f32], out: &mut [u8]) {
    let (xs, x_tail) = x.as_chunks::<4>();
    let (os, o_tail) = out.as_chunks_mut::<8>();
    for (o, x) in os.iter_mut().zip(xs) {
        store_f16_4(o, load_f32(x));
    }
    scalar::to_f16(x_tail, o_tail);
}

// --- Activation quantizers ---------------------------------------------------

/// `quantize_row_q8_0`, NEON path. The horizontal max matches for finite
/// values; rounding is `vcvtn` (nearest, ties to even) and the narrow
/// saturates, which is what `as i8` does to the rounded f32.
#[target_feature(enable = "neon")]
fn quantize_q8_0(x: &[f32], out: &mut [u8]) {
    let blocks = out.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter_mut().zip(x.as_chunks::<QK8_0>().0) {
        let parts = x.as_chunks::<4>().0;
        let mut src = [vdupq_n_f32(0.0); 8];
        let mut abs = [vdupq_n_f32(0.0); 8];
        for j in 0..8 {
            src[j] = load_f32(&parts[j]);
            abs[j] = vabsq_f32(src[j]);
        }
        for j in 0..4 {
            abs[j] = vmaxq_f32(abs[j], abs[j + 4]);
        }
        for j in 0..2 {
            abs[j] = vmaxq_f32(abs[j], abs[j + 2]);
        }
        abs[0] = vmaxq_f32(abs[0], abs[1]);
        let amax = vmaxvq_f32(abs[0]);
        let d = amax / 127.0;
        // `id = d ? 1/d : 0`, not the `127/amax` of the x86 kernel: see
        // [`quant::Q8_0Inverse`].
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let (scale, qs) = block.split_at_mut(2);
        scale.copy_from_slice(&quant::f32_to_f16(d).to_le_bytes());
        let idv = vdupq_n_f32(id);
        for j in 0..8 {
            let vi = vcvtnq_s32_f32(vmulq_f32(src[j], idv));
            let narrow = vqmovn_s16(vcombine_s16(vqmovn_s32(vi), vdup_n_s16(0)));
            qs[4 * j] = vget_lane_s8::<0>(narrow) as u8;
            qs[4 * j + 1] = vget_lane_s8::<1>(narrow) as u8;
            qs[4 * j + 2] = vget_lane_s8::<2>(narrow) as u8;
            qs[4 * j + 3] = vget_lane_s8::<3>(narrow) as u8;
        }
    }
}

/// `quantize_row_q8_K_ref`. The signed extreme is found in scalar order so
/// the first tied maximum wins; the rounding is the same 2^23 bias as
/// `nearest_int`, one lane at a time, so the bytes match the reference.
#[target_feature(enable = "neon")]
fn quantize_q8_k(x: &[f32], out: &mut [u8]) {
    let blocks = out.as_chunks_mut::<Q8_K_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter_mut().zip(x.as_chunks::<QK_K>().0) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in x {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
                max = v;
            }
        }
        if amax == 0.0 {
            block.fill(0);
            continue;
        }
        let iscale = -127.0 / max;
        let d = 1.0 / iscale;
        block[..4].copy_from_slice(&d.to_le_bytes());
        let (qs, bsums) = block[4..].split_at_mut(QK_K);
        let bias = vdupq_n_f32(12_582_912.0);
        let scale = vdupq_n_f32(iscale);
        let parts = x.as_chunks::<4>().0;
        let qparts = qs.as_chunks_mut::<4>().0;
        for (q, src) in qparts.iter_mut().zip(parts) {
            let scaled = vmulq_f32(load_f32(src), scale);
            let bits = vreinterpretq_s32_f32(vaddq_f32(scaled, bias));
            let masked = vandq_u32(vreinterpretq_u32_s32(bits), vdupq_n_u32(0x007f_ffff));
            let n = vminq_s32(
                vsubq_s32(vreinterpretq_s32_u32(masked), vdupq_n_s32(0x0040_0000)),
                vdupq_n_s32(127),
            );
            q[0] = vgetq_lane_s32::<0>(n) as i8 as u8;
            q[1] = vgetq_lane_s32::<1>(n) as i8 as u8;
            q[2] = vgetq_lane_s32::<2>(n) as i8 as u8;
            q[3] = vgetq_lane_s32::<3>(n) as i8 as u8;
        }
        for (j, slot) in bsums.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let sum: i16 = qs[j * 16..(j + 1) * 16]
                .iter()
                .map(|&b| i16::from(b as i8))
                .sum();
            *slot = sum.to_le_bytes();
        }
    }
}

fn k4_table(table: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..8 {
        let (scale, min) = quant::scale_min_k4(table, j);
        scales[j] = scale;
        mins[j] = min;
    }
    (scales, mins)
}

fn q8k_bsums(block: &[u8; Q8_K_BLOCK_BYTES]) -> [i16; 16] {
    let mut bsums = [0i16; 16];
    for (slot, bytes) in bsums.iter_mut().zip(block[260..].as_chunks::<2>().0) {
        *slot = i16::from_le_bytes(*bytes);
    }
    bsums
}

// `dotprod` and `vmull` are separate functions so a CPU without `dotprod`
// never contains a call to `vdotq_s32`. The K-quant integer sums are reduced
// with `vaddvq` before scaling, so both selections match the scalar reference.

macro_rules! q8_partial {
    ($dot:ident, $w:expr, $x:expr) => {{
        let zero = vdupq_n_s32(0);
        vaddq_s32(
            $dot(zero, load_i8(at16(&$w[2..], 0)), load_i8(at16(&$x[2..], 0))),
            $dot(
                zero,
                load_i8(at16(&$w[2..], 16)),
                load_i8(at16(&$x[2..], 16)),
            ),
        )
    }};
}

macro_rules! define_q8_rows {
    ($name:ident, $feat:literal, $dot:ident, $scale:ident) => {
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            let x = xq.as_chunks::<Q8_0_BLOCK_BYTES>().0;
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let w = row.as_chunks::<Q8_0_BLOCK_BYTES>().0;
                let n = w.len().min(x.len());
                // Two running vectors, one per block of a pair: ggml's NEON
                // `ggml_vec_dot_q8_0_q8_0`. Each block's 32 products stay in
                // four lanes and are scaled there, then `vaddvq` sums the
                // lanes. clang compiles ggml's `vmlaq_n_f32` to a fused
                // `fmla`, hence `vfmaq_n_f32`.
                let mut sum0 = vdupq_n_f32(0.0);
                let mut sum1 = vdupq_n_f32(0.0);
                let mut i = 0;
                while i + 1 < n {
                    let d0 = $scale(&w[i]) * $scale(&x[i]);
                    let d1 = $scale(&w[i + 1]) * $scale(&x[i + 1]);
                    sum0 = vfmaq_n_f32(sum0, vcvtq_f32_s32(q8_partial!($dot, w[i], x[i])), d0);
                    sum1 = vfmaq_n_f32(
                        sum1,
                        vcvtq_f32_s32(q8_partial!($dot, w[i + 1], x[i + 1])),
                        d1,
                    );
                    i += 2;
                }
                let mut sum = vaddvq_f32(sum0) + vaddvq_f32(sum1);
                // The block left over from an odd count is ggml's scalar tail
                // loop: `sumf += sumi*(dx*dy)`, after the lanes are summed.
                if i < n {
                    let sumi = vaddvq_s32(q8_partial!($dot, w[i], x[i]));
                    sum = (sumi as f32).mul_add($scale(&w[i]) * $scale(&x[i]), sum);
                }
                *o = sum;
            }
        }
    };
}

// Kernels for several inputs at once: each row's weights are loaded and
// decoded once for all of them, and each input's arithmetic is that of the
// one-input kernel named, so each output is the same. In the K-quants the
// products are exact integer sums, which any grouping gives; the float
// update per super-block is the one-input kernel's.

/// [`dot_q4_rows`] for `N` inputs.
#[target_feature(enable = "neon,dotprod")]
fn dot_q4_rows_n<const N: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    mut out: [&mut [f32]; N],
) {
    let xs = xq.map(|x| x.as_chunks::<Q8_K_BLOCK_BYTES>().0);
    let m4b = vdupq_n_u8(0x0f);
    let zero = vdupq_n_s32(0);
    for (r, row) in rows.chunks_exact(row_bytes).enumerate().take(out[0].len()) {
        let mut sum = [0.0f32; N];
        for (b, w) in row.as_chunks::<Q4_K_BLOCK_BYTES>().0.iter().enumerate() {
            if xs.iter().any(|x| b >= x.len()) {
                break;
            }
            let d = quant::f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
            let dmin = quant::f16_to_f32(u16::from_le_bytes([w[2], w[3]]));
            let (scales, mins) = k4_table(&w[4..16]);
            let mut sumi = [0i32; N];
            let mut sum_mins = [0i32; N];
            for c in 0..N {
                let bsums = q8k_bsums(&xs[c][b]);
                for g in 0..8 {
                    sum_mins[c] += i32::from(bsums[2 * g]) * i32::from(mins[g]);
                    sum_mins[c] += i32::from(bsums[2 * g + 1]) * i32::from(mins[g]);
                }
            }
            let qs = &w[16..];
            for j in 0..4 {
                let q0 = load_u8(at16(qs, j * 32));
                let q1 = load_u8(at16(qs, j * 32 + 16));
                let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, m4b));
                let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, m4b));
                let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
                for c in 0..N {
                    let xqs = &xs[c][b][4..260];
                    let p1 = vdotq_s32(
                        vdotq_s32(zero, lo0, load_i8(at16(xqs, j * 64))),
                        lo1,
                        load_i8(at16(xqs, j * 64 + 16)),
                    );
                    let p2 = vdotq_s32(
                        vdotq_s32(zero, hi0, load_i8(at16(xqs, j * 64 + 32))),
                        hi1,
                        load_i8(at16(xqs, j * 64 + 48)),
                    );
                    sumi[c] += vaddvq_s32(p1) * i32::from(scales[2 * j])
                        + vaddvq_s32(p2) * i32::from(scales[2 * j + 1]);
                }
            }
            for c in 0..N {
                let x = &xs[c][b];
                let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                sum[c] = quant::q4_k_update(sum[c], xd * d, xd * dmin, sumi[c], sum_mins[c]);
            }
        }
        for (o, s) in out.iter_mut().zip(sum) {
            o[r] = s;
        }
    }
}

/// [`dot_q5_rows`] for `N` inputs.
#[target_feature(enable = "neon,dotprod")]
fn dot_q5_rows_n<const N: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    mut out: [&mut [f32]; N],
) {
    let xs = xq.map(|x| x.as_chunks::<Q8_K_BLOCK_BYTES>().0);
    let m4b = vdupq_n_u8(0x0f);
    let one = vdupq_n_u8(1);
    let two = vdupq_n_u8(2);
    let zero = vdupq_n_s32(0);
    for (r, row) in rows.chunks_exact(row_bytes).enumerate().take(out[0].len()) {
        let mut sum = [0.0f32; N];
        for (b, w) in row.as_chunks::<Q5_K_BLOCK_BYTES>().0.iter().enumerate() {
            if xs.iter().any(|x| b >= x.len()) {
                break;
            }
            let d = quant::f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
            let dmin = quant::f16_to_f32(u16::from_le_bytes([w[2], w[3]]));
            let (scales, mins) = k4_table(&w[4..16]);
            let mut isum = [0i32; N];
            let mut sum_mins = [0i32; N];
            for c in 0..N {
                let bsums = q8k_bsums(&xs[c][b]);
                for g in 0..8 {
                    sum_mins[c] += i32::from(bsums[2 * g]) * i32::from(mins[g]);
                    sum_mins[c] += i32::from(bsums[2 * g + 1]) * i32::from(mins[g]);
                }
            }
            let mut qh0 = load_u8(at16(&w[16..], 0));
            let mut qh1 = load_u8(at16(&w[16..], 16));
            let ql = &w[48..];
            for j in 0..4 {
                let q0 = load_u8(at16(ql, j * 32));
                let q1 = load_u8(at16(ql, j * 32 + 16));
                let h0 = vshlq_n_u8::<4>(vandq_u8(one, qh0));
                let h1 = vshlq_n_u8::<4>(vandq_u8(one, qh1));
                let h2 = vshlq_n_u8::<3>(vandq_u8(two, qh0));
                let h3 = vshlq_n_u8::<3>(vandq_u8(two, qh1));
                qh0 = vshrq_n_u8::<2>(qh0);
                qh1 = vshrq_n_u8::<2>(qh1);
                let b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4b), h0));
                let b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4b), h1));
                let b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q0), h2));
                let b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q1), h3));
                for c in 0..N {
                    let xqs = &xs[c][b][4..260];
                    let p0 = vdotq_s32(
                        vdotq_s32(zero, b0, load_i8(at16(xqs, j * 64))),
                        b1,
                        load_i8(at16(xqs, j * 64 + 16)),
                    );
                    let p1 = vdotq_s32(
                        vdotq_s32(zero, b2, load_i8(at16(xqs, j * 64 + 32))),
                        b3,
                        load_i8(at16(xqs, j * 64 + 48)),
                    );
                    isum[c] += vaddvq_s32(p0) * i32::from(scales[2 * j]);
                    isum[c] += vaddvq_s32(p1) * i32::from(scales[2 * j + 1]);
                }
            }
            for c in 0..N {
                let x = &xs[c][b];
                let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                sum[c] = quant::q5_k_update(sum[c], xd * d, xd * dmin, isum[c], sum_mins[c]);
            }
        }
        for (o, s) in out.iter_mut().zip(sum) {
            o[r] = s;
        }
    }
}

/// [`dot_q6_rows`] for `N` inputs.
#[target_feature(enable = "neon,dotprod")]
fn dot_q6_rows_n<const N: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    mut out: [&mut [f32]; N],
) {
    let xs = xq.map(|x| x.as_chunks::<Q8_K_BLOCK_BYTES>().0);
    let m4b = vdupq_n_u8(0x0f);
    let three = vdupq_n_u8(3);
    let zero = vdupq_n_s32(0);
    for (r, row) in rows.chunks_exact(row_bytes).enumerate().take(out[0].len()) {
        let mut sum = [0.0f32; N];
        for (b, w) in row.as_chunks::<Q6_K_BLOCK_BYTES>().0.iter().enumerate() {
            if xs.iter().any(|x| b >= x.len()) {
                break;
            }
            let d = quant::f16_to_f32(u16::from_le_bytes([w[208], w[209]]));
            let mut scales = [0i8; 16];
            for (slot, &v) in scales.iter_mut().zip(&w[192..208]) {
                *slot = v as i8;
            }
            let mut bias = [0i32; N];
            for c in 0..N {
                let bsums = q8k_bsums(&xs[c][b]);
                for j in 0..16 {
                    bias[c] += i32::from(bsums[j]) * i32::from(scales[j]);
                }
            }
            let mut isum = [0i32; N];
            let (mut ql, mut qh, mut xoff, mut sc) = (0, 128, 4, 0);
            for _ in 0..2 {
                let qh0 = load_u8(at16(w, qh));
                let qh1 = load_u8(at16(w, qh + 16));
                let q0 = load_u8(at16(w, ql));
                let q1 = load_u8(at16(w, ql + 16));
                let q2 = load_u8(at16(w, ql + 32));
                let q3 = load_u8(at16(w, ql + 48));
                let lo = [
                    vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(q0, m4b),
                        vshlq_n_u8::<4>(vandq_u8(three, qh0)),
                    )),
                    vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(q1, m4b),
                        vshlq_n_u8::<4>(vandq_u8(three, qh1)),
                    )),
                    vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(q2, m4b),
                        vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<2>(qh0))),
                    )),
                    vreinterpretq_s8_u8(vorrq_u8(
                        vandq_u8(q3, m4b),
                        vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<2>(qh1))),
                    )),
                ];
                let hi = [
                    vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8::<4>(q0),
                        vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<4>(qh0))),
                    )),
                    vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8::<4>(q1),
                        vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<4>(qh1))),
                    )),
                    vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8::<4>(q2),
                        vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<6>(qh0))),
                    )),
                    vreinterpretq_s8_u8(vorrq_u8(
                        vshrq_n_u8::<4>(q3),
                        vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<6>(qh1))),
                    )),
                ];
                for c in 0..N {
                    let x = &xs[c][b];
                    for k in 0..4 {
                        isum[c] +=
                            vaddvq_s32(vdotq_s32(zero, lo[k], load_i8(at16(x, xoff + 16 * k))))
                                * i32::from(scales[sc + k]);
                    }
                    for k in 0..4 {
                        isum[c] += vaddvq_s32(vdotq_s32(
                            zero,
                            hi[k],
                            load_i8(at16(x, xoff + 64 + 16 * k)),
                        )) * i32::from(scales[sc + 4 + k]);
                    }
                }
                ql += 64;
                qh += 32;
                xoff += 128;
                sc += 8;
            }
            for c in 0..N {
                let x = &xs[c][b];
                let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                sum[c] = (xd * d).mul_add((isum[c] - 32 * bias[c]) as f32, sum[c]);
            }
        }
        for (o, s) in out.iter_mut().zip(sum) {
            o[r] = s;
        }
    }
}

/// [`dot_bf16`] of each row with `N` inputs.
#[target_feature(enable = "neon")]
fn dot_bf16_rows_n<const N: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    out: [&mut [f32]; N],
) {
    for (r, a) in rows.chunks_exact(row_bytes).enumerate().take(out[0].len()) {
        let n = xq.iter().fold(a.len(), |m, x| m.min(x.len())) / 2 * 2;
        let (a_steps, a_tail) = a[..n].as_chunks::<32>();
        let mut sum = [[vdupq_n_f32(0.0); 4]; N];
        for (s, a) in a_steps.iter().enumerate() {
            let (a0, a1) = load_bf16_8((&a[..16]).try_into().unwrap());
            let (a2, a3) = load_bf16_8((&a[16..]).try_into().unwrap());
            for c in 0..N {
                let b = &xq[c][32 * s..32 * s + 32];
                let (b0, b1) = load_bf16_8((&b[..16]).try_into().unwrap());
                let (b2, b3) = load_bf16_8((&b[16..]).try_into().unwrap());
                sum[c][0] = vfmaq_f32(sum[c][0], a0, b0);
                sum[c][1] = vfmaq_f32(sum[c][1], a1, b1);
                sum[c][2] = vfmaq_f32(sum[c][2], a2, b2);
                sum[c][3] = vfmaq_f32(sum[c][3], a3, b3);
            }
        }
        for c in 0..N {
            let mut sumf = reduce4(sum[c]);
            let b_tail = &xq[c][32 * a_steps.len()..n];
            for (a, b) in a_tail
                .as_chunks::<2>()
                .0
                .iter()
                .zip(b_tail.as_chunks::<2>().0)
            {
                sumf += quant::bf16_to_f32(u16::from_le_bytes(*a))
                    * quant::bf16_to_f32(u16::from_le_bytes(*b));
            }
            out[c][r] = sumf;
        }
    }
}

/// [`dot_f16_rows`] (the `fp16` widening) of each row with `N` inputs of
/// little-endian f32 bytes, in its chunks.
#[target_feature(enable = "neon,fp16")]
fn dot_f16_rows_n<const N: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    out: [&mut [f32]; N],
) {
    for (r, row) in rows.chunks_exact(row_bytes).enumerate().take(out[0].len()) {
        let n = xq.iter().fold(row.len() / 2, |m, x| m.min(x.len() / 4));
        let mut sumf = [0.0f64; N];
        for (k, a) in row[..2 * n].chunks(2 * F16_CHUNK).enumerate() {
            let (a_steps, a_tail) = a.as_chunks::<32>();
            let mut acc = [[vdupq_n_f32(0.0); 4]; N];
            for (s, a) in a_steps.iter().enumerate() {
                let a = a.as_chunks::<8>().0;
                let av = [
                    load_f16_4(&a[0]),
                    load_f16_4(&a[1]),
                    load_f16_4(&a[2]),
                    load_f16_4(&a[3]),
                ];
                for c in 0..N {
                    let at = 4 * (k * F16_CHUNK + 16 * s);
                    let b = xq[c][at..at + 64].as_chunks::<16>().0;
                    for i in 0..4 {
                        acc[c][i] = vfmaq_f32(acc[c][i], av[i], load_f32_le(&b[i]));
                    }
                }
            }
            for c in 0..N {
                let mut chunk = reduce4(acc[c]);
                let at = 4 * (k * F16_CHUNK + 16 * a_steps.len());
                let b_tail = &xq[c][at..at + 2 * a_tail.len()];
                for (a, b) in a_tail
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .zip(b_tail.as_chunks::<4>().0)
                {
                    let a = quant::f16_to_f32(u16::from_le_bytes(*a));
                    chunk = a.mul_add(f32::from_le_bytes(*b), chunk);
                }
                sumf[c] += f64::from(chunk);
            }
        }
        for c in 0..N {
            out[c][r] = sumf[c] as f32;
        }
    }
}

/// [`dot_q8_0_rows`] for `N` inputs at once: each block of weights is
/// loaded once for two rows and all the inputs. Every (row, input) pair
/// keeps the one-input kernel's lanes, its two running vectors, its scale
/// products and its odd-block tail, so each output is the same; the two
/// 16-byte dots of a block are chained into one accumulator, which gives the
/// same exact integers as adding them.
#[target_feature(enable = "neon,dotprod,fp16")]
fn dot_q8_0_rows_n<const N: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    mut out: [&mut [f32]; N],
) {
    let x = xq.map(|x| x.as_chunks::<Q8_0_BLOCK_BYTES>().0);
    let blocks = |r: usize| {
        rows[r * row_bytes..(r + 1) * row_bytes]
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
    };
    let n_rows = out[0].len();
    let mut r = 0;
    while r + 2 <= n_rows {
        let sums = q8_0_tile::<2, N>([blocks(r), blocks(r + 1)], x);
        for (c, o) in out.iter_mut().enumerate() {
            o[r] = sums[0][c];
            o[r + 1] = sums[1][c];
        }
        r += 2;
    }
    if r < n_rows {
        let sums = q8_0_tile::<1, N>([blocks(r)], x);
        for (c, o) in out.iter_mut().enumerate() {
            o[r] = sums[0][c];
        }
    }
}

/// The dots of `R` rows of Q8_0 blocks with `N` encoded inputs, as
/// [`dot_q8_0_rows`] computes each.
#[inline]
#[target_feature(enable = "neon,dotprod,fp16")]
fn q8_0_tile<const R: usize, const N: usize>(
    w: [&[[u8; Q8_0_BLOCK_BYTES]]; R],
    x: [&[[u8; Q8_0_BLOCK_BYTES]]; N],
) -> [[f32; N]; R] {
    let n = w
        .iter()
        .map(|w| w.len())
        .chain(x.iter().map(|x| x.len()))
        .min()
        .unwrap_or(0);
    let zero = vdupq_n_s32(0);
    let mut sum0 = [[vdupq_n_f32(0.0); N]; R];
    let mut sum1 = [[vdupq_n_f32(0.0); N]; R];
    // One block's products for every pair, into `sums`.
    let block = |i: usize, sums: &mut [[float32x4_t; N]; R]| {
        let wl: [[int8x16_t; 2]; R] = std::array::from_fn(|r| {
            [
                load_i8(at16(&w[r][i][2..], 0)),
                load_i8(at16(&w[r][i][2..], 16)),
            ]
        });
        let ws: [f32; R] = std::array::from_fn(|r| q8_scale_fp16(&w[r][i]));
        if N == 4 {
            // The four inputs' scale products with a row in one multiply
            // (each lane the same `ws * xs` as below).
            let xs4: [f32; 4] = std::array::from_fn(|c| q8_scale_fp16(&x[c.min(N - 1)][i]));
            let xs4 = load_f32(&xs4);
            let d: [float32x4_t; R] = std::array::from_fn(|r| vmulq_n_f32(xs4, ws[r]));
            macro_rules! col {
                ($c:literal) => {{
                    let xl = [
                        load_i8(at16(&x[$c][i][2..], 0)),
                        load_i8(at16(&x[$c][i][2..], 16)),
                    ];
                    for r in 0..R {
                        let p = vdotq_s32(vdotq_s32(zero, wl[r][0], xl[0]), wl[r][1], xl[1]);
                        sums[r][$c] = vfmaq_laneq_f32::<$c>(sums[r][$c], vcvtq_f32_s32(p), d[r]);
                    }
                }};
            }
            col!(0);
            col!(1);
            col!(2);
            col!(3);
            return;
        }
        for c in 0..N {
            let xl = [
                load_i8(at16(&x[c][i][2..], 0)),
                load_i8(at16(&x[c][i][2..], 16)),
            ];
            let xs = q8_scale_fp16(&x[c][i]);
            for r in 0..R {
                let p = vdotq_s32(vdotq_s32(zero, wl[r][0], xl[0]), wl[r][1], xl[1]);
                sums[r][c] = vfmaq_n_f32(sums[r][c], vcvtq_f32_s32(p), ws[r] * xs);
            }
        }
    };
    let mut i = 0;
    while i + 1 < n {
        block(i, &mut sum0);
        block(i + 1, &mut sum1);
        i += 2;
    }
    let mut out = [[0.0; N]; R];
    for r in 0..R {
        for c in 0..N {
            let mut sum = vaddvq_f32(sum0[r][c]) + vaddvq_f32(sum1[r][c]);
            if i < n {
                let p = vdotq_s32(
                    vdotq_s32(
                        zero,
                        load_i8(at16(&w[r][i][2..], 0)),
                        load_i8(at16(&x[c][i][2..], 0)),
                    ),
                    load_i8(at16(&w[r][i][2..], 16)),
                    load_i8(at16(&x[c][i][2..], 16)),
                );
                let sumi = vaddvq_s32(p);
                sum = (sumi as f32).mul_add(q8_scale_fp16(&w[r][i]) * q8_scale_fp16(&x[c][i]), sum);
            }
            out[r][c] = sum;
        }
    }
    out
}

/// Signed `2×8 × 8×2 -> 2×2` int8 matrix multiply-accumulate (`SMMLA`).
/// `std::arch::aarch64::vmmlaq_s32` is still unstable on Rust 1.100, while
/// `target_feature(enable = "i8mm")` and `is_aarch64_feature_detected!("i8mm")`
/// are stable; inline assembly emits the exact instruction behind that token.
#[inline]
#[target_feature(enable = "neon,i8mm")]
fn smmla_s32(mut acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    // SAFETY: caller proved `i8mm` via `Neon::detect()`; `smmla` touches only
    // its three vector registers.
    unsafe {
        std::arch::asm!(
            "smmla {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    acc
}

/// Packs two 16-byte vectors `(r0, r1)` into two `2×8` matrices for `smmla_s32`:
/// first `[r0[0..8], r1[0..8]]`, second `[r0[8..16], r1[8..16]]`.
#[inline]
#[target_feature(enable = "neon")]
fn zip_2x8(r0: int8x16_t, r1: int8x16_t) -> (int8x16_t, int8x16_t) {
    let a = vreinterpretq_s64_s8(r0);
    let b = vreinterpretq_s64_s8(r1);
    (
        vreinterpretq_s8_s64(vzip1q_s64(a, b)),
        vreinterpretq_s8_s64(vzip2q_s64(a, b)),
    )
}

/// Loads 32 bytes from two rows `(r0, r1)` and packs them into four `2×8`
/// matrices for `vmmlaq_s32`.
#[inline]
#[target_feature(enable = "neon")]
fn load_2x32_i8mm(r0: &[u8], r1: &[u8]) -> [int8x16_t; 4] {
    let (a0, a1) = zip_2x8(load_i8(at16(r0, 0)), load_i8(at16(r1, 0)));
    let (a2, a3) = zip_2x8(load_i8(at16(r0, 16)), load_i8(at16(r1, 16)));
    [a0, a1, a2, a3]
}

/// Four `f16` block scales widened in a single `vcvt_f32_f16`.
#[inline]
#[target_feature(enable = "neon,fp16")]
fn q8_scales4_fp16(
    b0: &[u8; Q8_0_BLOCK_BYTES],
    b1: &[u8; Q8_0_BLOCK_BYTES],
    b2: &[u8; Q8_0_BLOCK_BYTES],
    b3: &[u8; Q8_0_BLOCK_BYTES],
) -> float32x4_t {
    let raw = u64::from(u16::from_le_bytes([b0[0], b0[1]]))
        | (u64::from(u16::from_le_bytes([b1[0], b1[1]])) << 16)
        | (u64::from(u16::from_le_bytes([b2[0], b2[1]])) << 32)
        | (u64::from(u16::from_le_bytes([b3[0], b3[1]])) << 48);
    vcvt_f32_f16(vreinterpret_f16_u64(vcreate_u64(raw)))
}

/// `[a, a, b, b]` in `i32` in pure registers (no stack round-trip).
#[inline]
#[target_feature(enable = "neon")]
fn dup2_s32(a: i32, b: i32) -> int32x4_t {
    vcombine_s32(vdup_n_s32(a), vdup_n_s32(b))
}

/// `Q8_0` prompt GEMM (`4 × N` tiles with `i8mm` `smmla`, `N = 2 * NC_PAIRS`).
#[target_feature(enable = "neon,dotprod,fp16,i8mm")]
fn dot_q8_0_rows_i8mm<const N: usize, const NC_PAIRS: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    out: [&mut [f32]; N],
) {
    debug_assert_eq!(N, 2 * NC_PAIRS);
    let x = xq.map(|x| x.as_chunks::<Q8_0_BLOCK_BYTES>().0);
    let blocks = |r: usize| {
        rows[r * row_bytes..(r + 1) * row_bytes]
            .as_chunks::<Q8_0_BLOCK_BYTES>()
            .0
    };
    let n_rows = out[0].len();
    let zero = vdupq_n_s32(0);
    let mut r = 0;
    while r + 4 <= n_rows {
        let w = [blocks(r), blocks(r + 1), blocks(r + 2), blocks(r + 3)];
        let nb = w[0].len().min(x[0].len());
        let mut sum0 = [[vdupq_n_f32(0.0); NC_PAIRS]; 2];
        let mut sum1 = [[vdupq_n_f32(0.0); NC_PAIRS]; 2];
        let step = |i: usize, sum: &mut [[float32x4_t; NC_PAIRS]; 2]| {
            let ws = q8_scales4_fp16(&w[0][i], &w[1][i], &w[2][i], &w[3][i]);
            let ws01 = vzip1q_f32(ws, ws);
            let ws23 = vzip2q_f32(ws, ws);
            let a0 = load_2x32_i8mm(&w[0][i][2..], &w[1][i][2..]);
            let a1 = load_2x32_i8mm(&w[2][i][2..], &w[3][i][2..]);
            for cp in 0..NC_PAIRS {
                let c0 = 2 * cp;
                let c1 = c0 + 1;
                let b = load_2x32_i8mm(&x[c0][i][2..], &x[c1][i][2..]);
                let p0 = smmla_s32(
                    smmla_s32(
                        smmla_s32(smmla_s32(zero, a0[0], b[0]), a0[1], b[1]),
                        a0[2],
                        b[2],
                    ),
                    a0[3],
                    b[3],
                );
                let p1 = smmla_s32(
                    smmla_s32(
                        smmla_s32(smmla_s32(zero, a1[0], b[0]), a1[1], b[1]),
                        a1[2],
                        b[2],
                    ),
                    a1[3],
                    b[3],
                );
                let xraw = u64::from(u16::from_le_bytes([x[c0][i][0], x[c0][i][1]]))
                    | (u64::from(u16::from_le_bytes([x[c1][i][0], x[c1][i][1]])) << 16);
                let xs_lo = vget_low_f32(vcvt_f32_f16(vreinterpret_f16_u64(vcreate_u64(xraw))));
                let xs = vcombine_f32(xs_lo, xs_lo);
                sum[0][cp] = vfmaq_f32(sum[0][cp], vcvtq_f32_s32(p0), vmulq_f32(ws01, xs));
                sum[1][cp] = vfmaq_f32(sum[1][cp], vcvtq_f32_s32(p1), vmulq_f32(ws23, xs));
            }
        };
        let mut i = 0;
        while i + 1 < nb {
            step(i, &mut sum0);
            step(i + 1, &mut sum1);
            i += 2;
        }
        if i < nb {
            step(i, &mut sum0);
        }
        for cp in 0..NC_PAIRS {
            let mut s0 = [0.0f32; 4];
            let mut s1 = [0.0f32; 4];
            store_f32(&mut s0, vaddq_f32(sum0[0][cp], sum1[0][cp]));
            store_f32(&mut s1, vaddq_f32(sum0[1][cp], sum1[1][cp]));
            let c0 = 2 * cp;
            let c1 = c0 + 1;
            out[c0][r] = s0[0];
            out[c1][r] = s0[1];
            out[c0][r + 1] = s0[2];
            out[c1][r + 1] = s0[3];
            out[c0][r + 2] = s1[0];
            out[c1][r + 2] = s1[1];
            out[c0][r + 3] = s1[2];
            out[c1][r + 3] = s1[3];
        }
        r += 4;
    }
    if r < n_rows {
        let rem_rows = &rows[r * row_bytes..];
        let rem_out = out.map(|o| &mut o[r..]);
        dot_q8_0_rows_n(rem_rows, row_bytes, xq, rem_out);
    }
}

/// `Q4_K` `4 × N` GEMM with `i8mm` `vmmlaq_s32`: computes the exact same
/// integer block dots and `f32` updates as [`dot_q4_rows`] (bit-identical).
#[target_feature(enable = "neon,dotprod,i8mm")]
fn dot_q4_rows_i8mm<const N: usize, const NC_PAIRS: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    out: [&mut [f32]; N],
) {
    debug_assert_eq!(N, 2 * NC_PAIRS);
    let xs = xq.map(|x| x.as_chunks::<Q8_K_BLOCK_BYTES>().0);
    let blocks = |r: usize| {
        rows[r * row_bytes..(r + 1) * row_bytes]
            .as_chunks::<Q4_K_BLOCK_BYTES>()
            .0
    };
    let m4b = vdupq_n_u8(0x0f);
    let zero = vdupq_n_s32(0);
    let n_rows = out[0].len();
    let mut r = 0;
    while r + 4 <= n_rows {
        let w = [blocks(r), blocks(r + 1), blocks(r + 2), blocks(r + 3)];
        let nb = w[0].len().min(xs[0].len());
        let mut sum = [[0.0f32; N]; 4];
        for b in 0..nb {
            let d = [0, 1, 2, 3]
                .map(|k| quant::f16_to_f32(u16::from_le_bytes([w[k][b][0], w[k][b][1]])));
            let dmin = [0, 1, 2, 3]
                .map(|k| quant::f16_to_f32(u16::from_le_bytes([w[k][b][2], w[k][b][3]])));
            let tables = [0, 1, 2, 3].map(|k| k4_table(&w[k][b][4..16]));
            let mut sum_mins = [[0i32; N]; 4];
            for c in 0..N {
                let bsums = q8k_bsums(&xs[c][b]);
                let pair_sums: [i32; 8] =
                    std::array::from_fn(|g| i32::from(bsums[2 * g]) + i32::from(bsums[2 * g + 1]));
                for k in 0..4 {
                    let mins = &tables[k].1;
                    let mut m = 0i32;
                    for g in 0..8 {
                        m += pair_sums[g] * i32::from(mins[g]);
                    }
                    sum_mins[k][c] = m;
                }
            }
            let mut sumi = [[zero; NC_PAIRS]; 2];
            for j in 0..4 {
                let unpack = |k: usize| {
                    let qs = &w[k][b][16..];
                    let q0 = load_u8(at16(qs, j * 32));
                    let q1 = load_u8(at16(qs, j * 32 + 16));
                    (
                        vreinterpretq_s8_u8(vandq_u8(q0, m4b)),
                        vreinterpretq_s8_u8(vandq_u8(q1, m4b)),
                        vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0)),
                        vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1)),
                    )
                };
                let (l0a, l0b, h0a, h0b) = unpack(0);
                let (l1a, l1b, h1a, h1b) = unpack(1);
                let (l2a, l2b, h2a, h2b) = unpack(2);
                let (l3a, l3b, h3a, h3b) = unpack(3);
                let (al00, al01) = zip_2x8(l0a, l1a);
                let (al02, al03) = zip_2x8(l0b, l1b);
                let (al10, al11) = zip_2x8(l2a, l3a);
                let (al12, al13) = zip_2x8(l2b, l3b);
                let sc01_lo =
                    dup2_s32(i32::from(tables[0].0[2 * j]), i32::from(tables[1].0[2 * j]));
                let sc23_lo =
                    dup2_s32(i32::from(tables[2].0[2 * j]), i32::from(tables[3].0[2 * j]));
                for cp in 0..NC_PAIRS {
                    let x0 = &xs[2 * cp][b][4 + j * 64..];
                    let x1 = &xs[2 * cp + 1][b][4 + j * 64..];
                    let bl = load_2x32_i8mm(x0, x1);
                    let p0 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, al00, bl[0]), al01, bl[1]),
                            al02,
                            bl[2],
                        ),
                        al03,
                        bl[3],
                    );
                    let p1 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, al10, bl[0]), al11, bl[1]),
                            al12,
                            bl[2],
                        ),
                        al13,
                        bl[3],
                    );
                    sumi[0][cp] = vmlaq_s32(sumi[0][cp], p0, sc01_lo);
                    sumi[1][cp] = vmlaq_s32(sumi[1][cp], p1, sc23_lo);
                }
                let (ah00, ah01) = zip_2x8(h0a, h1a);
                let (ah02, ah03) = zip_2x8(h0b, h1b);
                let (ah10, ah11) = zip_2x8(h2a, h3a);
                let (ah12, ah13) = zip_2x8(h2b, h3b);
                let sc01_hi = dup2_s32(
                    i32::from(tables[0].0[2 * j + 1]),
                    i32::from(tables[1].0[2 * j + 1]),
                );
                let sc23_hi = dup2_s32(
                    i32::from(tables[2].0[2 * j + 1]),
                    i32::from(tables[3].0[2 * j + 1]),
                );
                for cp in 0..NC_PAIRS {
                    let x0 = &xs[2 * cp][b][4 + j * 64 + 32..];
                    let x1 = &xs[2 * cp + 1][b][4 + j * 64 + 32..];
                    let bh = load_2x32_i8mm(x0, x1);
                    let p0 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, ah00, bh[0]), ah01, bh[1]),
                            ah02,
                            bh[2],
                        ),
                        ah03,
                        bh[3],
                    );
                    let p1 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, ah10, bh[0]), ah11, bh[1]),
                            ah12,
                            bh[2],
                        ),
                        ah13,
                        bh[3],
                    );
                    sumi[0][cp] = vmlaq_s32(sumi[0][cp], p0, sc01_hi);
                    sumi[1][cp] = vmlaq_s32(sumi[1][cp], p1, sc23_hi);
                }
            }
            for cp in 0..NC_PAIRS {
                let mut i0 = [0i32; 4];
                let mut i1 = [0i32; 4];
                // SAFETY: 16 writable bytes.
                unsafe {
                    vst1q_s32(i0.as_mut_ptr(), sumi[0][cp]);
                    vst1q_s32(i1.as_mut_ptr(), sumi[1][cp]);
                }
                let c0 = 2 * cp;
                let c1 = c0 + 1;
                let xd0 = f32::from_le_bytes(xs[c0][b][0..4].try_into().unwrap());
                let xd1 = f32::from_le_bytes(xs[c1][b][0..4].try_into().unwrap());
                sum[0][c0] = quant::q4_k_update(
                    sum[0][c0],
                    xd0 * d[0],
                    xd0 * dmin[0],
                    i0[0],
                    sum_mins[0][c0],
                );
                sum[0][c1] = quant::q4_k_update(
                    sum[0][c1],
                    xd1 * d[0],
                    xd1 * dmin[0],
                    i0[1],
                    sum_mins[0][c1],
                );
                sum[1][c0] = quant::q4_k_update(
                    sum[1][c0],
                    xd0 * d[1],
                    xd0 * dmin[1],
                    i0[2],
                    sum_mins[1][c0],
                );
                sum[1][c1] = quant::q4_k_update(
                    sum[1][c1],
                    xd1 * d[1],
                    xd1 * dmin[1],
                    i0[3],
                    sum_mins[1][c1],
                );
                sum[2][c0] = quant::q4_k_update(
                    sum[2][c0],
                    xd0 * d[2],
                    xd0 * dmin[2],
                    i1[0],
                    sum_mins[2][c0],
                );
                sum[2][c1] = quant::q4_k_update(
                    sum[2][c1],
                    xd1 * d[2],
                    xd1 * dmin[2],
                    i1[1],
                    sum_mins[2][c1],
                );
                sum[3][c0] = quant::q4_k_update(
                    sum[3][c0],
                    xd0 * d[3],
                    xd0 * dmin[3],
                    i1[2],
                    sum_mins[3][c0],
                );
                sum[3][c1] = quant::q4_k_update(
                    sum[3][c1],
                    xd1 * d[3],
                    xd1 * dmin[3],
                    i1[3],
                    sum_mins[3][c1],
                );
            }
        }
        for c in 0..N {
            out[c][r] = sum[0][c];
            out[c][r + 1] = sum[1][c];
            out[c][r + 2] = sum[2][c];
            out[c][r + 3] = sum[3][c];
        }
        r += 4;
    }
    if r < n_rows {
        let rem_rows = &rows[r * row_bytes..];
        let rem_out = out.map(|o| &mut o[r..]);
        dot_q4_rows_n(rem_rows, row_bytes, xq, rem_out);
    }
}

/// `Q5_K` `4 × N` GEMM with `i8mm` `vmmlaq_s32`: bit-identical to [`dot_q5_rows`].
#[target_feature(enable = "neon,dotprod,i8mm")]
fn dot_q5_rows_i8mm<const N: usize, const NC_PAIRS: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    out: [&mut [f32]; N],
) {
    debug_assert_eq!(N, 2 * NC_PAIRS);
    let xs = xq.map(|x| x.as_chunks::<Q8_K_BLOCK_BYTES>().0);
    let blocks = |r: usize| {
        rows[r * row_bytes..(r + 1) * row_bytes]
            .as_chunks::<Q5_K_BLOCK_BYTES>()
            .0
    };
    let m4b = vdupq_n_u8(0x0f);
    let one = vdupq_n_u8(1);
    let two = vdupq_n_u8(2);
    let zero = vdupq_n_s32(0);
    let n_rows = out[0].len();
    let mut r = 0;
    while r + 4 <= n_rows {
        let w = [blocks(r), blocks(r + 1), blocks(r + 2), blocks(r + 3)];
        let nb = w[0].len().min(xs[0].len());
        let mut sum = [[0.0f32; N]; 4];
        for b in 0..nb {
            let d = [0, 1, 2, 3]
                .map(|k| quant::f16_to_f32(u16::from_le_bytes([w[k][b][0], w[k][b][1]])));
            let dmin = [0, 1, 2, 3]
                .map(|k| quant::f16_to_f32(u16::from_le_bytes([w[k][b][2], w[k][b][3]])));
            let tables = [0, 1, 2, 3].map(|k| k4_table(&w[k][b][4..16]));
            let mut sum_mins = [[0i32; N]; 4];
            for c in 0..N {
                let bsums = q8k_bsums(&xs[c][b]);
                let pair_sums: [i32; 8] =
                    std::array::from_fn(|g| i32::from(bsums[2 * g]) + i32::from(bsums[2 * g + 1]));
                for k in 0..4 {
                    let mins = &tables[k].1;
                    let mut m = 0i32;
                    for g in 0..8 {
                        m += pair_sums[g] * i32::from(mins[g]);
                    }
                    sum_mins[k][c] = m;
                }
            }
            let mut qh0 = [0, 1, 2, 3].map(|k| load_u8(at16(&w[k][b][16..], 0)));
            let mut qh1 = [0, 1, 2, 3].map(|k| load_u8(at16(&w[k][b][16..], 16)));
            let mut isum = [[zero; NC_PAIRS]; 2];
            for j in 0..4 {
                let mut unpack = |k: usize| {
                    let ql = &w[k][b][48..];
                    let q0 = load_u8(at16(ql, j * 32));
                    let q1 = load_u8(at16(ql, j * 32 + 16));
                    let h0 = vshlq_n_u8::<4>(vandq_u8(one, qh0[k]));
                    let h1 = vshlq_n_u8::<4>(vandq_u8(one, qh1[k]));
                    let h2 = vshlq_n_u8::<3>(vandq_u8(two, qh0[k]));
                    let h3 = vshlq_n_u8::<3>(vandq_u8(two, qh1[k]));
                    qh0[k] = vshrq_n_u8::<2>(qh0[k]);
                    qh1[k] = vshrq_n_u8::<2>(qh1[k]);
                    (
                        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4b), h0)),
                        vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4b), h1)),
                        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q0), h2)),
                        vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q1), h3)),
                    )
                };
                let (l0a, l0b, h0a, h0b) = unpack(0);
                let (l1a, l1b, h1a, h1b) = unpack(1);
                let (l2a, l2b, h2a, h2b) = unpack(2);
                let (l3a, l3b, h3a, h3b) = unpack(3);
                let (al00, al01) = zip_2x8(l0a, l1a);
                let (al02, al03) = zip_2x8(l0b, l1b);
                let (al10, al11) = zip_2x8(l2a, l3a);
                let (al12, al13) = zip_2x8(l2b, l3b);
                let sc01_lo =
                    dup2_s32(i32::from(tables[0].0[2 * j]), i32::from(tables[1].0[2 * j]));
                let sc23_lo =
                    dup2_s32(i32::from(tables[2].0[2 * j]), i32::from(tables[3].0[2 * j]));
                for cp in 0..NC_PAIRS {
                    let x0 = &xs[2 * cp][b][4 + j * 64..];
                    let x1 = &xs[2 * cp + 1][b][4 + j * 64..];
                    let bl = load_2x32_i8mm(x0, x1);
                    let p0 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, al00, bl[0]), al01, bl[1]),
                            al02,
                            bl[2],
                        ),
                        al03,
                        bl[3],
                    );
                    let p1 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, al10, bl[0]), al11, bl[1]),
                            al12,
                            bl[2],
                        ),
                        al13,
                        bl[3],
                    );
                    isum[0][cp] = vmlaq_s32(isum[0][cp], p0, sc01_lo);
                    isum[1][cp] = vmlaq_s32(isum[1][cp], p1, sc23_lo);
                }
                let (ah00, ah01) = zip_2x8(h0a, h1a);
                let (ah02, ah03) = zip_2x8(h0b, h1b);
                let (ah10, ah11) = zip_2x8(h2a, h3a);
                let (ah12, ah13) = zip_2x8(h2b, h3b);
                let sc01_hi = dup2_s32(
                    i32::from(tables[0].0[2 * j + 1]),
                    i32::from(tables[1].0[2 * j + 1]),
                );
                let sc23_hi = dup2_s32(
                    i32::from(tables[2].0[2 * j + 1]),
                    i32::from(tables[3].0[2 * j + 1]),
                );
                for cp in 0..NC_PAIRS {
                    let x0 = &xs[2 * cp][b][4 + j * 64 + 32..];
                    let x1 = &xs[2 * cp + 1][b][4 + j * 64 + 32..];
                    let bh = load_2x32_i8mm(x0, x1);
                    let p0 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, ah00, bh[0]), ah01, bh[1]),
                            ah02,
                            bh[2],
                        ),
                        ah03,
                        bh[3],
                    );
                    let p1 = smmla_s32(
                        smmla_s32(
                            smmla_s32(smmla_s32(zero, ah10, bh[0]), ah11, bh[1]),
                            ah12,
                            bh[2],
                        ),
                        ah13,
                        bh[3],
                    );
                    isum[0][cp] = vmlaq_s32(isum[0][cp], p0, sc01_hi);
                    isum[1][cp] = vmlaq_s32(isum[1][cp], p1, sc23_hi);
                }
            }
            for cp in 0..NC_PAIRS {
                let mut i0 = [0i32; 4];
                let mut i1 = [0i32; 4];
                // SAFETY: 16 writable bytes.
                unsafe {
                    vst1q_s32(i0.as_mut_ptr(), isum[0][cp]);
                    vst1q_s32(i1.as_mut_ptr(), isum[1][cp]);
                }
                let c0 = 2 * cp;
                let c1 = c0 + 1;
                let xd0 = f32::from_le_bytes(xs[c0][b][0..4].try_into().unwrap());
                let xd1 = f32::from_le_bytes(xs[c1][b][0..4].try_into().unwrap());
                sum[0][c0] = quant::q5_k_update(
                    sum[0][c0],
                    xd0 * d[0],
                    xd0 * dmin[0],
                    i0[0],
                    sum_mins[0][c0],
                );
                sum[0][c1] = quant::q5_k_update(
                    sum[0][c1],
                    xd1 * d[0],
                    xd1 * dmin[0],
                    i0[1],
                    sum_mins[0][c1],
                );
                sum[1][c0] = quant::q5_k_update(
                    sum[1][c0],
                    xd0 * d[1],
                    xd0 * dmin[1],
                    i0[2],
                    sum_mins[1][c0],
                );
                sum[1][c1] = quant::q5_k_update(
                    sum[1][c1],
                    xd1 * d[1],
                    xd1 * dmin[1],
                    i0[3],
                    sum_mins[1][c1],
                );
                sum[2][c0] = quant::q5_k_update(
                    sum[2][c0],
                    xd0 * d[2],
                    xd0 * dmin[2],
                    i1[0],
                    sum_mins[2][c0],
                );
                sum[2][c1] = quant::q5_k_update(
                    sum[2][c1],
                    xd1 * d[2],
                    xd1 * dmin[2],
                    i1[1],
                    sum_mins[2][c1],
                );
                sum[3][c0] = quant::q5_k_update(
                    sum[3][c0],
                    xd0 * d[3],
                    xd0 * dmin[3],
                    i1[2],
                    sum_mins[3][c0],
                );
                sum[3][c1] = quant::q5_k_update(
                    sum[3][c1],
                    xd1 * d[3],
                    xd1 * dmin[3],
                    i1[3],
                    sum_mins[3][c1],
                );
            }
        }
        for c in 0..N {
            out[c][r] = sum[0][c];
            out[c][r + 1] = sum[1][c];
            out[c][r + 2] = sum[2][c];
            out[c][r + 3] = sum[3][c];
        }
        r += 4;
    }
    if r < n_rows {
        let rem_rows = &rows[r * row_bytes..];
        let rem_out = out.map(|o| &mut o[r..]);
        dot_q5_rows_n(rem_rows, row_bytes, xq, rem_out);
    }
}

/// `Q6_K` `4 × N` GEMM with `i8mm` `vmmlaq_s32`: bit-identical to [`dot_q6_rows`].
#[target_feature(enable = "neon,dotprod,i8mm")]
fn dot_q6_rows_i8mm<const N: usize, const NC_PAIRS: usize>(
    rows: &[u8],
    row_bytes: usize,
    xq: [&[u8]; N],
    out: [&mut [f32]; N],
) {
    debug_assert_eq!(N, 2 * NC_PAIRS);
    let xs = xq.map(|x| x.as_chunks::<Q8_K_BLOCK_BYTES>().0);
    let blocks = |r: usize| {
        rows[r * row_bytes..(r + 1) * row_bytes]
            .as_chunks::<Q6_K_BLOCK_BYTES>()
            .0
    };
    let m4b = vdupq_n_u8(0x0f);
    let three = vdupq_n_u8(3);
    let zero = vdupq_n_s32(0);
    let n_rows = out[0].len();
    let mut r = 0;
    while r + 4 <= n_rows {
        let w = [blocks(r), blocks(r + 1), blocks(r + 2), blocks(r + 3)];
        let nb = w[0].len().min(xs[0].len());
        let mut sum = [[0.0f32; N]; 4];
        for b in 0..nb {
            let d = [0, 1, 2, 3]
                .map(|k| quant::f16_to_f32(u16::from_le_bytes([w[k][b][208], w[k][b][209]])));
            let scales: [[i8; 16]; 4] =
                std::array::from_fn(|k| std::array::from_fn(|j| w[k][b][192 + j] as i8));
            let mut bias = [[0i32; N]; 4];
            for c in 0..N {
                let bsums = q8k_bsums(&xs[c][b]);
                for k in 0..4 {
                    let mut bi = 0i32;
                    for j in 0..16 {
                        bi += i32::from(bsums[j]) * i32::from(scales[k][j]);
                    }
                    bias[k][c] = bi;
                }
            }
            let mut isum = [[zero; NC_PAIRS]; 2];
            for half in 0..2 {
                let ql = half * 64;
                let qh = 128 + half * 32;
                let xoff = 4 + half * 128;
                let sc = half * 8;
                let unpack = |k: usize| -> [int8x16_t; 8] {
                    let wb = &w[k][b];
                    let qh0 = load_u8(at16(wb, qh));
                    let qh1 = load_u8(at16(wb, qh + 16));
                    let q0 = load_u8(at16(wb, ql));
                    let q1 = load_u8(at16(wb, ql + 16));
                    let q2 = load_u8(at16(wb, ql + 32));
                    let q3 = load_u8(at16(wb, ql + 48));
                    [
                        vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(q0, m4b),
                            vshlq_n_u8::<4>(vandq_u8(three, qh0)),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(q1, m4b),
                            vshlq_n_u8::<4>(vandq_u8(three, qh1)),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(q2, m4b),
                            vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<2>(qh0))),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vandq_u8(q3, m4b),
                            vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<2>(qh1))),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8::<4>(q0),
                            vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<4>(qh0))),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8::<4>(q1),
                            vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<4>(qh1))),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8::<4>(q2),
                            vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<6>(qh0))),
                        )),
                        vreinterpretq_s8_u8(vorrq_u8(
                            vshrq_n_u8::<4>(q3),
                            vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<6>(qh1))),
                        )),
                    ]
                };
                let u = [unpack(0), unpack(1), unpack(2), unpack(3)];
                for g in 0..8 {
                    let (a00, a01) = zip_2x8(u[0][g], u[1][g]);
                    let (a10, a11) = zip_2x8(u[2][g], u[3][g]);
                    let sc01 = dup2_s32(i32::from(scales[0][sc + g]), i32::from(scales[1][sc + g]));
                    let sc23 = dup2_s32(i32::from(scales[2][sc + g]), i32::from(scales[3][sc + g]));
                    for cp in 0..NC_PAIRS {
                        let (b0, b1) = zip_2x8(
                            load_i8(at16(&xs[2 * cp][b], xoff + 16 * g)),
                            load_i8(at16(&xs[2 * cp + 1][b], xoff + 16 * g)),
                        );
                        let p0 = smmla_s32(smmla_s32(zero, a00, b0), a01, b1);
                        let p1 = smmla_s32(smmla_s32(zero, a10, b0), a11, b1);
                        isum[0][cp] = vmlaq_s32(isum[0][cp], p0, sc01);
                        isum[1][cp] = vmlaq_s32(isum[1][cp], p1, sc23);
                    }
                }
            }
            for cp in 0..NC_PAIRS {
                let mut i0 = [0i32; 4];
                let mut i1 = [0i32; 4];
                // SAFETY: 16 writable bytes.
                unsafe {
                    vst1q_s32(i0.as_mut_ptr(), isum[0][cp]);
                    vst1q_s32(i1.as_mut_ptr(), isum[1][cp]);
                }
                let c0 = 2 * cp;
                let c1 = c0 + 1;
                let xd0 = f32::from_le_bytes(xs[c0][b][0..4].try_into().unwrap());
                let xd1 = f32::from_le_bytes(xs[c1][b][0..4].try_into().unwrap());
                sum[0][c0] = (xd0 * d[0]).mul_add((i0[0] - 32 * bias[0][c0]) as f32, sum[0][c0]);
                sum[0][c1] = (xd1 * d[0]).mul_add((i0[1] - 32 * bias[0][c1]) as f32, sum[0][c1]);
                sum[1][c0] = (xd0 * d[1]).mul_add((i0[2] - 32 * bias[1][c0]) as f32, sum[1][c0]);
                sum[1][c1] = (xd1 * d[1]).mul_add((i0[3] - 32 * bias[1][c1]) as f32, sum[1][c1]);
                sum[2][c0] = (xd0 * d[2]).mul_add((i1[0] - 32 * bias[2][c0]) as f32, sum[2][c0]);
                sum[2][c1] = (xd1 * d[2]).mul_add((i1[1] - 32 * bias[2][c1]) as f32, sum[2][c1]);
                sum[3][c0] = (xd0 * d[3]).mul_add((i1[2] - 32 * bias[3][c0]) as f32, sum[3][c0]);
                sum[3][c1] = (xd1 * d[3]).mul_add((i1[3] - 32 * bias[3][c1]) as f32, sum[3][c1]);
            }
        }
        for c in 0..N {
            out[c][r] = sum[0][c];
            out[c][r + 1] = sum[1][c];
            out[c][r + 2] = sum[2][c];
            out[c][r + 3] = sum[3][c];
        }
        r += 4;
    }
    if r < n_rows {
        let rem_rows = &rows[r * row_bytes..];
        let rem_out = out.map(|o| &mut o[r..]);
        dot_q6_rows_n(rem_rows, row_bytes, xq, rem_out);
    }
}

/// [`q8_scale`] with the hardware conversion: the same exact value in one
/// instruction instead of a software decode, which costs as much as a
/// block's products.
#[inline]
#[target_feature(enable = "neon,fp16")]
fn q8_scale_fp16(block: &[u8; Q8_0_BLOCK_BYTES]) -> f32 {
    let bits = vdup_n_u16(u16::from_le_bytes([block[0], block[1]]));
    vgetq_lane_f32::<0>(vcvt_f32_f16(vreinterpret_f16_u16(bits)))
}

fn q8_scale(block: &[u8; Q8_0_BLOCK_BYTES]) -> f32 {
    quant::f16_to_f32(u16::from_le_bytes([block[0], block[1]]))
}

define_q8_rows!(dot_q8_0_rows, "neon,dotprod", i8_dot16, q8_scale);
define_q8_rows!(
    dot_q8_0_rows_fp16,
    "neon,dotprod,fp16",
    i8_dot16,
    q8_scale_fp16
);
define_q8_rows!(dot_q8_0_rows_mul, "neon", i8_dot16_mul, q8_scale);

// --- Blocks of 32: Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, MXFP4 ---------------------

/// One block's 32 weights as the NEON kernels widen them (weights `0..16`
/// and `16..32`), its scale and its minimum (`Q4_1`/`Q5_1`).
#[inline]
#[target_feature(enable = "neon")]
fn block32_weights(ty: GgmlType, b: &[u8], values: int8x16_t) -> (int8x16_t, int8x16_t, f32, f32) {
    let f16 = |at: usize| quant::f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]));
    let m4b = vdupq_n_u8(0x0f);
    let nibbles = |at: usize| {
        let q = load_u8(at16(b, at));
        (vandq_u8(q, m4b), vshrq_n_u8::<4>(q))
    };
    // The fifth bits of weights `0..16` (`2..4` for `16..32`) as 0x10 bytes:
    // `table_b2b_0`'s bytes, here by testing each byte's bit.
    let fifth = |at: usize| {
        let qh = u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]);
        let bits = |word: u16| {
            let v = vcombine_u8(vdup_n_u8(word as u8), vdup_n_u8((word >> 8) as u8));
            let mask = vreinterpretq_u8_u64(vdupq_n_u64(0x8040_2010_0804_0201));
            vandq_u8(vtstq_u8(v, mask), vdupq_n_u8(0x10))
        };
        (bits(qh as u16), bits((qh >> 16) as u16))
    };
    let s = |v: uint8x16_t| vreinterpretq_s8_u8(v);
    match ty {
        GgmlType::Q4_0 => {
            let (lo, hi) = nibbles(2);
            let eight = vdupq_n_s8(8);
            (vsubq_s8(s(lo), eight), vsubq_s8(s(hi), eight), f16(0), 0.0)
        }
        GgmlType::Q4_1 => {
            let (lo, hi) = nibbles(4);
            (s(lo), s(hi), f16(0), f16(2))
        }
        GgmlType::Q5_0 => {
            let (lo, hi) = nibbles(6);
            let (hl, hh) = fifth(2);
            let sixteen = vdupq_n_s8(16);
            (
                vsubq_s8(s(vorrq_u8(lo, hl)), sixteen),
                vsubq_s8(s(vorrq_u8(hi, hh)), sixteen),
                f16(0),
                0.0,
            )
        }
        GgmlType::Q5_1 => {
            let (lo, hi) = nibbles(8);
            let (hl, hh) = fifth(4);
            (s(vorrq_u8(lo, hl)), s(vorrq_u8(hi, hh)), f16(0), f16(2))
        }
        GgmlType::Iq4Nl => {
            let (lo, hi) = nibbles(2);
            (vqtbl1q_s8(values, lo), vqtbl1q_s8(values, hi), f16(0), 0.0)
        }
        GgmlType::Mxfp4 => {
            let (lo, hi) = nibbles(1);
            let d = quant::blocks32::e8m0_half(b[0]);
            (vqtbl1q_s8(values, lo), vqtbl1q_s8(values, hi), d, 0.0)
        }
        other => unreachable!("{other} has no 32-weight blocks"),
    }
}

/// ggml's NEON `ggml_vec_dot_q4_0_q8_0` and its siblings for `Q4_1`, `Q5_0`,
/// `Q5_1` (with `Q8_1`), `IQ4_NL` and `MXFP4`: blocks in pairs, the even
/// block's `vdotq_s32` lanes into one float vector and the odd one's into
/// another (`vmlaq_n_f32`, fused), or for `IQ4_NL`/`MXFP4` the pair's two
/// sums added to the running sum; the scalar tail for an odd last block.
/// The float order is that of [`quant::blocks32::dot_row`], which is the
/// reference.
#[target_feature(enable = "neon,dotprod")]
fn dot_block32_rows(ty: GgmlType, rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
    let bb = quant::blocks32::block_bytes(ty);
    let with_min = matches!(ty, GgmlType::Q4_1 | GgmlType::Q5_1);
    let (xb, xo) = if with_min {
        (quant::Q8_1_BLOCK_BYTES, 4)
    } else {
        (Q8_0_BLOCK_BYTES, 2)
    };
    let table = if ty == GgmlType::Mxfp4 {
        quant::blocks32::KVALUES_MXFP4
    } else {
        quant::KVALUES_IQ4NL
    };
    let values = load_i8(&table.map(|v| v as u8));
    let zero = vdupq_n_s32(0);
    let nb = row_bytes / bb;
    let f16 = |b: &[u8], at: usize| quant::f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]));
    for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
        // Block `i`'s integer lanes, scale, min, and the activation's scale and sum.
        let block = |i: usize| {
            let w = &row[i * bb..(i + 1) * bb];
            let x = &xq[i * xb..(i + 1) * xb];
            let (lo, hi, d, m) = block32_weights(ty, w, values);
            let p = vdotq_s32(
                vdotq_s32(zero, lo, load_i8(at16(x, xo))),
                hi,
                load_i8(at16(x, xo + 16)),
            );
            let s = if with_min { f16(x, 2) } else { 0.0 };
            (p, d, m, f16(x, 0), s)
        };
        let mut sumf = 0.0f32;
        let mut ib = 0;
        if matches!(ty, GgmlType::Iq4Nl | GgmlType::Mxfp4) {
            while ib + 1 < nb {
                let (p0, d0, _, y0, _) = block(ib);
                let (p1, d1, _, y1, _) = block(ib + 1);
                let (i0, i1) = (vaddvq_s32(p0) as f32, vaddvq_s32(p1) as f32);
                sumf += (d0 * y0).mul_add(i0, (d1 * y1) * i1);
                ib += 2;
            }
            for ib in ib..nb {
                let (p, d, _, y, _) = block(ib);
                sumf = (y * d).mul_add(vaddvq_s32(p) as f32, sumf);
            }
        } else {
            let mut v0 = vdupq_n_f32(0.0);
            let mut v1 = vdupq_n_f32(0.0);
            let (mut summs, mut summs1) = (0.0f32, 0.0f32);
            while ib + 1 < nb {
                let (p0, d0, m0, y0, s0) = block(ib);
                let (p1, d1, m1, y1, s1) = block(ib + 1);
                v0 = vfmaq_n_f32(v0, vcvtq_f32_s32(p0), d0 * y0);
                v1 = vfmaq_n_f32(v1, vcvtq_f32_s32(p1), d1 * y1);
                match ty {
                    GgmlType::Q4_1 => summs += m0.mul_add(s0, m1 * s1),
                    GgmlType::Q5_1 => {
                        summs = m0.mul_add(s0, summs);
                        summs1 = m1.mul_add(s1, summs1);
                    }
                    _ => {}
                }
                ib += 2;
            }
            sumf = vaddvq_f32(v0) + vaddvq_f32(v1);
            if with_min {
                sumf = sumf + summs + summs1;
            }
            for ib in ib..nb {
                let (p, d, m, y, s) = block(ib);
                let sumi = vaddvq_s32(p) as f32;
                sumf = match ty {
                    GgmlType::Q4_0 => (sumi * d).mul_add(y, sumf),
                    GgmlType::Q4_1 | GgmlType::Q5_1 => sumf + (d * y).mul_add(sumi, m * s),
                    _ => (d * y).mul_add(sumi, sumf),
                };
            }
        }
        *o = sumf;
    }
}

// --- Q2_0 ---------------------------------------------------------------------

/// ggml's NEON `ggml_vec_dot_q2_0_q8_0`: each half of a block (32 weights,
/// 8 bytes of codes) has every byte repeated four times by a table lookup,
/// shifted right by 0, 2, 4 and 6 and masked to its code, less one; its two
/// `vdotq_s32` go into one float vector with the half's scale
/// (`vmlaq_n_f32`, fused), reduced once per row. The float order is that of
/// [`quant::q2_0::dot_row`], which is the reference. With `fp16` the scales
/// are converted by the hardware (exact either way).
macro_rules! define_q2_0_rows {
    ($name:ident, $feat:literal, $f16:ident) => {
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            use quant::q2_0::Q2_0_BLOCK_BYTES;
            const SHIFTS: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];
            let idx_lo = load_u8(&[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]);
            let idx_hi = load_u8(&[4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7]);
            let shifts = load_i8(&SHIFTS.map(|s| s as u8));
            let mask = vdupq_n_u8(3);
            let one = vdupq_n_s8(1);
            let zero = vdupq_n_s32(0);
            let xs = xq.as_chunks::<Q8_0_BLOCK_BYTES>().0.as_chunks::<2>().0;
            let codes = |raw16: uint8x16_t, idx: uint8x16_t| {
                let q = vandq_u8(vshlq_u8(vqtbl1q_u8(raw16, idx), shifts), mask);
                vsubq_s8(vreinterpretq_s8_u8(q), one)
            };
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let mut sumv = vdupq_n_f32(0.0);
                for (block, x) in row.as_chunks::<Q2_0_BLOCK_BYTES>().0.iter().zip(xs) {
                    let d0 = $f16([block[0], block[1]]);
                    for (half, y) in block[2..].as_chunks::<8>().0.iter().zip(x) {
                        let raw = vcreate_u8(u64::from_le_bytes(*half));
                        let raw16 = vcombine_u8(raw, raw);
                        let p = vdotq_s32(zero, codes(raw16, idx_lo), load_i8(at16(y, 2)));
                        let p = vdotq_s32(p, codes(raw16, idx_hi), load_i8(at16(y, 18)));
                        let d1 = $f16([y[0], y[1]]);
                        sumv = vfmaq_n_f32(sumv, vcvtq_f32_s32(p), d0 * d1);
                    }
                }
                *o = vaddvq_f32(sumv);
            }
        }
    };
}

/// An f16 scale converted in software.
fn f16_soft(b: [u8; 2]) -> f32 {
    quant::f16_to_f32(u16::from_le_bytes(b))
}

/// An f16 scale converted by the hardware.
#[inline]
#[target_feature(enable = "neon,fp16")]
fn f16_hard(b: [u8; 2]) -> f32 {
    let bits = vdup_n_u16(u16::from_le_bytes(b));
    vgetq_lane_f32::<0>(vcvt_f32_f16(vreinterpret_f16_u16(bits)))
}

define_q2_0_rows!(dot_q2_0_rows, "neon,dotprod", f16_soft);
define_q2_0_rows!(dot_q2_0_rows_fp16, "neon,dotprod,fp16", f16_hard);

macro_rules! define_q4_rows {
    ($name:ident, $feat:literal, $dot:ident) => {
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
            let m4b = vdupq_n_u8(0x0f);
            let zero = vdupq_n_s32(0);
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let mut sum = 0.0f32;
                for (w, x) in row.as_chunks::<Q4_K_BLOCK_BYTES>().0.iter().zip(xs) {
                    let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                    let d = quant::f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
                    let dmin = quant::f16_to_f32(u16::from_le_bytes([w[2], w[3]]));
                    let (scales, mins) = k4_table(&w[4..16]);
                    let bsums = q8k_bsums(x);
                    let mut sum_mins = 0i32;
                    for g in 0..8 {
                        sum_mins += i32::from(bsums[2 * g]) * i32::from(mins[g]);
                        sum_mins += i32::from(bsums[2 * g + 1]) * i32::from(mins[g]);
                    }
                    let mut sumi1 = 0i32;
                    let mut sumi2 = 0i32;
                    let qs = &w[16..];
                    let xqs = &x[4..260];
                    for j in 0..4 {
                        let q0 = load_u8(at16(qs, j * 32));
                        let q1 = load_u8(at16(qs, j * 32 + 16));
                        let lo0 = vreinterpretq_s8_u8(vandq_u8(q0, m4b));
                        let lo1 = vreinterpretq_s8_u8(vandq_u8(q1, m4b));
                        // Low nibbles are the first 32 weights of this group of 64.
                        let p1 = vaddq_s32(
                            $dot(zero, lo0, load_i8(at16(xqs, j * 64))),
                            $dot(zero, lo1, load_i8(at16(xqs, j * 64 + 16))),
                        );
                        sumi1 += vaddvq_s32(p1) * i32::from(scales[2 * j]);
                        let hi0 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q0));
                        let hi1 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(q1));
                        let p2 = vaddq_s32(
                            $dot(zero, hi0, load_i8(at16(xqs, j * 64 + 32))),
                            $dot(zero, hi1, load_i8(at16(xqs, j * 64 + 48))),
                        );
                        sumi2 += vaddvq_s32(p2) * i32::from(scales[2 * j + 1]);
                    }
                    sum = quant::q4_k_update(sum, xd * d, xd * dmin, sumi1 + sumi2, sum_mins);
                }
                *o = sum;
            }
        }
    };
}

define_q4_rows!(dot_q4_rows, "neon,dotprod", i8_dot16);
define_q4_rows!(dot_q4_rows_mul, "neon", i8_dot16_mul);

macro_rules! define_iq4_xs_rows {
    ($name:ident, $feat:literal, $dot:ident) => {
        /// ggml's NEON `ggml_vec_dot_iq4_xs_q8_K`: the nibbles through the
        /// value table (`vqtbl1q_s8`), each group's exact integer dot times
        /// its scale, and one fused `sumf += d * d8 * isum` per super-block.
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
            let values = load_i8(&quant::KVALUES_IQ4NL.map(|v| v as u8));
            let m4b = vdupq_n_u8(0x0f);
            let zero = vdupq_n_s32(0);
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let mut sum = 0.0f32;
                for (w, x) in row.as_chunks::<IQ4_XS_BLOCK_BYTES>().0.iter().zip(xs) {
                    let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                    let d = quant::f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
                    let mut h = u16::from_le_bytes([w[2], w[3]]);
                    let qs = &w[8..];
                    let xqs = &x[4..260];
                    let mut sumi1 = 0i32;
                    let mut sumi2 = 0i32;
                    for ib in 0..4 {
                        let q0 = load_u8(at16(qs, ib * 32));
                        let q1 = load_u8(at16(qs, ib * 32 + 16));
                        let b0 = vqtbl1q_s8(values, vandq_u8(q0, m4b));
                        let b1 = vqtbl1q_s8(values, vshrq_n_u8::<4>(q0));
                        let b2 = vqtbl1q_s8(values, vandq_u8(q1, m4b));
                        let b3 = vqtbl1q_s8(values, vshrq_n_u8::<4>(q1));
                        let p1 = $dot(
                            $dot(zero, b0, load_i8(at16(xqs, ib * 64))),
                            b1,
                            load_i8(at16(xqs, ib * 64 + 16)),
                        );
                        let p2 = $dot(
                            $dot(zero, b2, load_i8(at16(xqs, ib * 64 + 32))),
                            b3,
                            load_i8(at16(xqs, ib * 64 + 48)),
                        );
                        let ls1 = i32::from((w[4 + ib] & 0xf) | ((h << 4) & 0x30) as u8) - 32;
                        let ls2 = i32::from((w[4 + ib] >> 4) | ((h << 2) & 0x30) as u8) - 32;
                        h >>= 4;
                        sumi1 += vaddvq_s32(p1) * ls1;
                        sumi2 += vaddvq_s32(p2) * ls2;
                    }
                    sum = (d * xd).mul_add((sumi1 + sumi2) as f32, sum);
                }
                *o = sum;
            }
        }
    };
}

define_iq4_xs_rows!(dot_iq4_xs_rows, "neon,dotprod", i8_dot16);
define_iq4_xs_rows!(dot_iq4_xs_rows_mul, "neon", i8_dot16_mul);

macro_rules! define_q5_rows {
    ($name:ident, $feat:literal, $dot:ident) => {
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
            let m4b = vdupq_n_u8(0x0f);
            let one = vdupq_n_u8(1);
            let two = vdupq_n_u8(2);
            let zero = vdupq_n_s32(0);
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let mut sum = 0.0f32;
                for (w, x) in row.as_chunks::<Q5_K_BLOCK_BYTES>().0.iter().zip(xs) {
                    let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                    let d = quant::f16_to_f32(u16::from_le_bytes([w[0], w[1]]));
                    let dmin = quant::f16_to_f32(u16::from_le_bytes([w[2], w[3]]));
                    let (scales, mins) = k4_table(&w[4..16]);
                    let bsums = q8k_bsums(x);
                    let mut sum_mins = 0i32;
                    for g in 0..8 {
                        sum_mins += i32::from(bsums[2 * g]) * i32::from(mins[g]);
                        sum_mins += i32::from(bsums[2 * g + 1]) * i32::from(mins[g]);
                    }
                    // High bits walk up the same 32 mask bytes, two bits per
                    // group of 64 weights (`dequantize_row_q5_K`).
                    let mut qh0 = load_u8(at16(&w[16..], 0));
                    let mut qh1 = load_u8(at16(&w[16..], 16));
                    let ql = &w[48..];
                    let xqs = &x[4..260];
                    let mut isum = 0i32;
                    for j in 0..4 {
                        let q0 = load_u8(at16(ql, j * 32));
                        let q1 = load_u8(at16(ql, j * 32 + 16));
                        let h0 = vshlq_n_u8::<4>(vandq_u8(one, qh0));
                        let h1 = vshlq_n_u8::<4>(vandq_u8(one, qh1));
                        let h2 = vshlq_n_u8::<3>(vandq_u8(two, qh0));
                        let h3 = vshlq_n_u8::<3>(vandq_u8(two, qh1));
                        qh0 = vshrq_n_u8::<2>(qh0);
                        qh1 = vshrq_n_u8::<2>(qh1);
                        let b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4b), h0));
                        let b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4b), h1));
                        let b2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q0), h2));
                        let b3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q1), h3));
                        let p0 = vaddq_s32(
                            $dot(zero, b0, load_i8(at16(xqs, j * 64))),
                            $dot(zero, b1, load_i8(at16(xqs, j * 64 + 16))),
                        );
                        let p1 = vaddq_s32(
                            $dot(zero, b2, load_i8(at16(xqs, j * 64 + 32))),
                            $dot(zero, b3, load_i8(at16(xqs, j * 64 + 48))),
                        );
                        isum += vaddvq_s32(p0) * i32::from(scales[2 * j]);
                        isum += vaddvq_s32(p1) * i32::from(scales[2 * j + 1]);
                    }
                    sum = quant::q5_k_update(sum, xd * d, xd * dmin, isum, sum_mins);
                }
                *o = sum;
            }
        }
    };
}

define_q5_rows!(dot_q5_rows, "neon,dotprod", i8_dot16);
define_q5_rows!(dot_q5_rows_mul, "neon", i8_dot16_mul);

macro_rules! define_q6_rows {
    ($name:ident, $feat:literal, $dot:ident) => {
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
            let m4b = vdupq_n_u8(0x0f);
            let three = vdupq_n_u8(3);
            let zero = vdupq_n_s32(0);
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let mut sum = 0.0f32;
                for (w, x) in row.as_chunks::<Q6_K_BLOCK_BYTES>().0.iter().zip(xs) {
                    let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                    let d = quant::f16_to_f32(u16::from_le_bytes([w[208], w[209]]));
                    let mut scales = [0i8; 16];
                    for (slot, &b) in scales.iter_mut().zip(&w[192..208]) {
                        *slot = b as i8;
                    }
                    let bsums = q8k_bsums(x);
                    // `(q - 32) * scale * x = q * scale * x - 32 * scale * bsum`.
                    let mut bias = 0i32;
                    for j in 0..16 {
                        bias += i32::from(bsums[j]) * i32::from(scales[j]);
                    }
                    let mut isum = 0i32;
                    let mut ql = 0;
                    let mut qh = 128;
                    let mut xoff = 4;
                    let mut sc = 0;
                    for _ in 0..2 {
                        let qh0 = load_u8(at16(w, qh));
                        let qh1 = load_u8(at16(w, qh + 16));
                        let q0 = load_u8(at16(w, ql));
                        let q1 = load_u8(at16(w, ql + 16));
                        let q2 = load_u8(at16(w, ql + 32));
                        let q3 = load_u8(at16(w, ql + 48));
                        let h0 = vshlq_n_u8::<4>(vandq_u8(three, qh0));
                        let h1 = vshlq_n_u8::<4>(vandq_u8(three, qh1));
                        let h2 = vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<2>(qh0)));
                        let h3 = vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<2>(qh1)));
                        let b0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q0, m4b), h0));
                        let b1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q1, m4b), h1));
                        let b2 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q2, m4b), h2));
                        let b3 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(q3, m4b), h3));
                        isum += vaddvq_s32($dot(zero, b0, load_i8(at16(x, xoff))))
                            * i32::from(scales[sc]);
                        isum += vaddvq_s32($dot(zero, b1, load_i8(at16(x, xoff + 16))))
                            * i32::from(scales[sc + 1]);
                        isum += vaddvq_s32($dot(zero, b2, load_i8(at16(x, xoff + 32))))
                            * i32::from(scales[sc + 2]);
                        isum += vaddvq_s32($dot(zero, b3, load_i8(at16(x, xoff + 48))))
                            * i32::from(scales[sc + 3]);
                        let h4 = vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<4>(qh0)));
                        let h5 = vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<4>(qh1)));
                        let h6 = vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<6>(qh0)));
                        let h7 = vshlq_n_u8::<4>(vandq_u8(three, vshrq_n_u8::<6>(qh1)));
                        let c0 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q0), h4));
                        let c1 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q1), h5));
                        let c2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q2), h6));
                        let c3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(q3), h7));
                        isum += vaddvq_s32($dot(zero, c0, load_i8(at16(x, xoff + 64))))
                            * i32::from(scales[sc + 4]);
                        isum += vaddvq_s32($dot(zero, c1, load_i8(at16(x, xoff + 80))))
                            * i32::from(scales[sc + 5]);
                        isum += vaddvq_s32($dot(zero, c2, load_i8(at16(x, xoff + 96))))
                            * i32::from(scales[sc + 6]);
                        isum += vaddvq_s32($dot(zero, c3, load_i8(at16(x, xoff + 112))))
                            * i32::from(scales[sc + 7]);
                        ql += 64;
                        qh += 32;
                        xoff += 128;
                        sc += 8;
                    }
                    sum = (xd * d).mul_add((isum - 32 * bias) as f32, sum);
                }
                *o = sum;
            }
        }
    };
}

define_q6_rows!(dot_q6_rows, "neon,dotprod", i8_dot16);
define_q6_rows!(dot_q6_rows_mul, "neon", i8_dot16_mul);

macro_rules! define_q3_rows {
    ($name:ident, $feat:literal, $dot:ident) => {
        #[target_feature(enable = $feat)]
        fn $name(rows: &[u8], row_bytes: usize, xq: &[u8], out: &mut [f32]) {
            let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
            let m3b = vdupq_n_u8(0x3);
            let m0 = vdupq_n_u8(1);
            let m1 = vshlq_n_u8::<1>(m0);
            let m2 = vshlq_n_u8::<2>(m0);
            let m3 = vshlq_n_u8::<3>(m0);
            let zero = vdupq_n_s32(0);
            for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                let mut sum = 0.0f32;
                for (w, x) in row.as_chunks::<Q3_K_BLOCK_BYTES>().0.iter().zip(xs) {
                    let xd = f32::from_le_bytes([x[0], x[1], x[2], x[3]]);
                    let d = quant::f16_to_f32(u16::from_le_bytes([w[108], w[109]]));
                    let scales = quant::q3_scales(&w[96..108]);
                    let mut qh0 = load_u8(at16(w, 0));
                    let mut qh1 = load_u8(at16(w, 16));
                    let mut isum = 0i32;
                    let mut sc = 0;
                    for j in 0..2 {
                        let q0 = load_u8(at16(w, 32 + j * 32));
                        let q1 = load_u8(at16(w, 32 + j * 32 + 16));
                        let xbase = 4 + j * 128;
                        let h0 = vshlq_n_u8::<2>(vbicq_u8(m0, qh0));
                        let h1 = vshlq_n_u8::<2>(vbicq_u8(m0, qh1));
                        let h2 = vshlq_n_u8::<1>(vbicq_u8(m1, qh0));
                        let h3 = vshlq_n_u8::<1>(vbicq_u8(m1, qh1));
                        let b0 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(q0, m3b)),
                            vreinterpretq_s8_u8(h0),
                        );
                        let b1 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(q1, m3b)),
                            vreinterpretq_s8_u8(h1),
                        );
                        let b2 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(q0), m3b)),
                            vreinterpretq_s8_u8(h2),
                        );
                        let b3 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<2>(q1), m3b)),
                            vreinterpretq_s8_u8(h3),
                        );
                        isum += vaddvq_s32($dot(zero, b0, load_i8(at16(x, xbase))))
                            * (i32::from(scales[sc]) - 32);
                        isum += vaddvq_s32($dot(zero, b1, load_i8(at16(x, xbase + 16))))
                            * (i32::from(scales[sc + 1]) - 32);
                        isum += vaddvq_s32($dot(zero, b2, load_i8(at16(x, xbase + 32))))
                            * (i32::from(scales[sc + 2]) - 32);
                        isum += vaddvq_s32($dot(zero, b3, load_i8(at16(x, xbase + 48))))
                            * (i32::from(scales[sc + 3]) - 32);
                        let g0 = vbicq_u8(m2, qh0);
                        let g1 = vbicq_u8(m2, qh1);
                        let g2 = vshrq_n_u8::<1>(vbicq_u8(m3, qh0));
                        let g3 = vshrq_n_u8::<1>(vbicq_u8(m3, qh1));
                        let c0 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(q0), m3b)),
                            vreinterpretq_s8_u8(g0),
                        );
                        let c1 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<4>(q1), m3b)),
                            vreinterpretq_s8_u8(g1),
                        );
                        let c2 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<6>(q0), m3b)),
                            vreinterpretq_s8_u8(g2),
                        );
                        let c3 = vsubq_s8(
                            vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8::<6>(q1), m3b)),
                            vreinterpretq_s8_u8(g3),
                        );
                        isum += vaddvq_s32($dot(zero, c0, load_i8(at16(x, xbase + 64))))
                            * (i32::from(scales[sc + 4]) - 32);
                        isum += vaddvq_s32($dot(zero, c1, load_i8(at16(x, xbase + 80))))
                            * (i32::from(scales[sc + 5]) - 32);
                        isum += vaddvq_s32($dot(zero, c2, load_i8(at16(x, xbase + 96))))
                            * (i32::from(scales[sc + 6]) - 32);
                        isum += vaddvq_s32($dot(zero, c3, load_i8(at16(x, xbase + 112))))
                            * (i32::from(scales[sc + 7]) - 32);
                        if j == 0 {
                            qh0 = vshrq_n_u8::<4>(qh0);
                            qh1 = vshrq_n_u8::<4>(qh1);
                        }
                        sc += 8;
                    }
                    sum = (xd * d).mul_add(isum as f32, sum);
                }
                *o = sum;
            }
        }
    };
}

define_q3_rows!(dot_q3_rows, "neon,dotprod", i8_dot16);
define_q3_rows!(dot_q3_rows_mul, "neon", i8_dot16_mul);

/// One head of `ggml_compute_forward_gated_delta_net`, with NEON scale, FMA
/// and [`dot_f32`].
#[target_feature(enable = "neon")]
fn delta_net_head(state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
    let d_k = step.k.len();
    for ((row, &v), o) in state.chunks_exact_mut(d_k).zip(step.v).zip(out) {
        scale(row, step.decay);
        let delta = (v - dot_f32(row, step.k)) * step.beta;
        mad(row, step.k, delta);
        *o = dot_f32(row, step.q) * step.scale;
    }
}

/// `ggml_vec_scale_f32`.
#[target_feature(enable = "neon")]
fn scale(y: &mut [f32], v: f32) {
    let (ys, tail) = y.as_chunks_mut::<4>();
    let vv = vdupq_n_f32(v);
    for y in ys {
        store_f32(y, vmulq_f32(load_f32(y), vv));
    }
    for y in tail {
        *y *= v;
    }
}

/// `ggml_vec_mad_f32`: `y = fma(x, v, y)`.
#[target_feature(enable = "neon")]
fn mad(y: &mut [f32], x: &[f32], v: f32) {
    let n = y.len().min(x.len());
    let (ys, y_tail) = y[..n].as_chunks_mut::<4>();
    let (xs, x_tail) = x[..n].as_chunks::<4>();
    let vv = vdupq_n_f32(v);
    for (y, x) in ys.iter_mut().zip(xs) {
        store_f32(y, vfmaq_f32(load_f32(y), load_f32(x), vv));
    }
    scalar::mad_f32(y_tail, x_tail, v);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::test_data::{f32_bytes, fnv1a, k_row, scaled, values};

    fn f16_bytes(x: &[f32]) -> Vec<u8> {
        let mut out = vec![0; 2 * x.len()];
        scalar::to_f16(x, &mut out);
        out
    }

    fn bf16_bytes(x: &[f32]) -> Vec<u8> {
        quant::encoded(DotType::Bf16, x)
    }

    fn assert_near(got: f32, want: f32, rel: f64, scale: f64, what: &str) {
        let err = (f64::from(got) - f64::from(want)).abs();
        assert!(
            err <= rel * scale,
            "{what}: {got} vs {want} (error {err:e}, allowed {:e})",
            rel * scale
        );
    }

    /// The kernels to test: this CPU's, and the variants for CPUs without
    /// `dotprod` and `fp16`, which no Apple Silicon CPU would run otherwise.
    /// The latter use only baseline NEON, so they are sound wherever
    /// [`Neon::detect`] succeeds.
    fn variants() -> Vec<Neon> {
        let Some(k) = Neon::detect() else {
            eprintln!("skipped: this CPU has no NEON");
            return Vec::new();
        };
        let baseline = Neon {
            dotprod: false,
            fp16: false,
            i8mm: false,
        };
        if k.dotprod || k.fp16 {
            vec![k, baseline]
        } else {
            vec![k]
        }
    }

    const K_QUANTS: [GgmlType; 4] = [GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K];

    /// Lengths around the groups of 4 and the steps of 16.
    const LENGTHS: [usize; 10] = [1, 3, 4, 7, 15, 16, 17, 31, 100, 1024];

    #[test]
    fn encoders_match_the_scalar_reference_exactly() {
        for k in variants() {
            for dt in [DotType::F32, DotType::Bf16, DotType::Q8_0, DotType::Q8K] {
                let block = match dt {
                    DotType::Q8_0 => 32,
                    DotType::Q8K => 256,
                    _ => 1,
                };
                for n in [block, block * 2, block * 3, 1024 / block * block] {
                    if n == 0 {
                        continue;
                    }
                    for seed in 1..4 {
                        let mut x = scaled(n, seed, 4.0);
                        let head = [1e-6, -3e-5, 6.5e-5, 0.0, -0.0, 65504.0, 1e9, -1e-30];
                        x[..head.len().min(n)].copy_from_slice(&head[..head.len().min(n)]);
                        if dt == DotType::Q8_0 && n > 32 {
                            x[32..64.min(n)].fill(0.0);
                        }
                        if dt == DotType::Q8K && n > 256 {
                            x[256..512.min(n)].fill(0.0);
                        }
                        let want = quant::encoded(dt, &x);
                        let mut got = vec![0xAA; want.len()];
                        k.encode(dt, &x, &mut got);
                        assert_eq!(got, want, "{dt:?}, n = {n}, seed {seed}, {k:?}");
                    }
                }
            }
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
    }

    #[test]
    fn dot_rows_match_the_scalar_reference() {
        for k in variants() {
            for ty in [
                GgmlType::F32,
                GgmlType::F16,
                GgmlType::Bf16,
                GgmlType::Q8_0,
                GgmlType::Q3K,
                GgmlType::Q4K,
                GgmlType::Q5K,
                GgmlType::Q6K,
                GgmlType::Iq4Xs,
            ] {
                let quantized = K_QUANTS.contains(&ty) || ty == GgmlType::Iq4Xs;
                // Three super-blocks per K-quant row, so the block loop runs
                // over a zero and a negative scale; an odd number of Q8_0 blocks.
                let (n, rows) = match ty {
                    _ if quantized => (768, 3),
                    GgmlType::Q8_0 => (96, 5),
                    _ => (100, 5),
                };
                let dt = DotType::of(ty);
                let x = values(n, 3);
                let xq = quant::encoded(dt, &x);
                let rows_q: Vec<u8> = (0..rows)
                    .flat_map(|r| {
                        let w = values(n, 10 + r as u32);
                        match ty {
                            _ if quantized => k_row(ty, n / 256, 10 + r as u32),
                            GgmlType::F16 => f16_bytes(&w),
                            _ => quant::encoded(dt, &w),
                        }
                    })
                    .collect();
                let row_bytes = rows_q.len() / rows;
                let mut got = vec![0.0; rows];
                k.dot_rows(ty, &rows_q, row_bytes, &xq, &mut got);
                for (r, &g) in got.iter().enumerate() {
                    let row = &rows_q[r * row_bytes..(r + 1) * row_bytes];
                    let want = quant::dot_row(ty, row, &xq);
                    if quantized {
                        assert_eq!(
                            g.to_bits(),
                            want.to_bits(),
                            "{ty} row {r}: {g} vs {want}, {k:?}"
                        );
                    } else {
                        let scale = (want.abs() as f64).max(1.0);
                        assert_near(g, want, 1e-5, scale, &format!("{ty} row {r}, {k:?}"));
                    }
                }
            }
        }
    }

    /// The kernels for blocks of 32 give the scalar reference's results bit
    /// for bit (the reference matches ggml's, see `quant`), with an even and
    /// an odd number of blocks per row.
    #[test]
    fn block32_dots_equal_the_scalar_reference() {
        for k in variants() {
            for ty in [
                GgmlType::Q4_0,
                GgmlType::Q4_1,
                GgmlType::Q5_0,
                GgmlType::Q5_1,
                GgmlType::Iq4Nl,
                GgmlType::Mxfp4,
            ] {
                for n in [768, 800] {
                    let rows = 4;
                    let xq = quant::encoded(DotType::of(ty), &values(n, 3));
                    let w: Vec<u8> = (0..rows).flat_map(|r| k_row(ty, n / 32, 10 + r)).collect();
                    let row_bytes = w.len() / rows as usize;
                    let mut got = vec![0.0; rows as usize];
                    k.dot_rows(ty, &w, row_bytes, &xq, &mut got);
                    for (r, &g) in got.iter().enumerate() {
                        let want = quant::dot_row(ty, &w[r * row_bytes..(r + 1) * row_bytes], &xq);
                        assert_eq!(g.to_bits(), want.to_bits(), "{ty} n {n} row {r}, {k:?}");
                    }
                }
            }
        }
    }

    /// The `Q2_0` kernel gives the scalar reference's results bit for bit,
    /// which are ggml's NEON kernel's (llama.cpp b12722's `libggml-cpu` on
    /// rows `k_row(Q2_0, 25, 10 + r)` and the activation `values(1600, 3)`).
    #[test]
    fn q2_0_dots_equal_the_scalar_reference_and_ggml() {
        let ty = GgmlType::Q2_0;
        let ggml = [0xC03D_7C5Bu32, 0xC04F_E646, 0xBEC0_5C0E, 0xC04F_20DE];
        let mut kernels = variants();
        // Both scale conversions of the `dotprod` kernel.
        if let Some(&k) = kernels.first().filter(|k| k.dotprod && k.fp16) {
            kernels.push(Neon { fp16: false, ..k });
        }
        for k in kernels {
            for blocks in [1, 3, 25] {
                let n = 64 * blocks;
                let xq = quant::encoded(DotType::Q8_0, &values(n, 3));
                let w: Vec<u8> = (0..4).flat_map(|r| k_row(ty, blocks, 10 + r)).collect();
                let row_bytes = w.len() / 4;
                let mut got = vec![0.0; 4];
                k.dot_rows(ty, &w, row_bytes, &xq, &mut got);
                for (r, &g) in got.iter().enumerate() {
                    let want = quant::dot_row(ty, &w[r * row_bytes..(r + 1) * row_bytes], &xq);
                    assert_eq!(
                        g.to_bits(),
                        want.to_bits(),
                        "{blocks} blocks, row {r}, {k:?}"
                    );
                    if blocks == 25 {
                        assert_eq!(g.to_bits(), ggml[r], "row {r}: {g}, {k:?}");
                    }
                }
            }
        }
    }

    /// Bit patterns from ggml's own kernels in llama.cpp's b12359 macOS
    /// arm64 release (`ggml_get_type_traits_cpu(type)->vec_dot`, run through
    /// its `libggml-cpu`) on the same inputs. That build has `dotprod` but
    /// not fp16 arithmetic or i8mm.
    #[test]
    fn dots_match_ggml_bit_for_bit() {
        // Rows `values(n, 10 + r)` encoded as Q8_0, activation `values(n, 3)`:
        // 3 blocks (a pair and the scalar tail) and 33.
        #[rustfmt::skip]
        let q8_0: [(usize, [u32; 5]); 2] = [
            (96, [0x4005_9A7B, 0x4003_F608, 0x3FD7_37D7, 0x3E55_2C58, 0x401B_7D7F]),
            (1056, [0xC09D_2F37, 0x3FC7_F2F8, 0xC197_BA5E, 0xC099_FEAA, 0xBF72_EAD3]),
        ];
        // Rows `k_row(ty, 3, 10 + r)`, activation `values(768, 3)` as Q8_K.
        #[rustfmt::skip]
        let k_quants: [(GgmlType, [u32; 4]); 4] = [
            (GgmlType::Q3K, [0x41BA_C4D6, 0x415B_B436, 0x4117_A4D4, 0xC184_1DB7]),
            (GgmlType::Q4K, [0x4338_F907, 0xC34D_03FD, 0x42E3_1C27, 0x4208_D8D1]),
            (GgmlType::Q5K, [0x43AF_F341, 0xC396_F378, 0x438B_9C1A, 0xC2B1_C5B2]),
            (GgmlType::Q6K, [0x4542_9876, 0x4466_46AC, 0xC422_704C, 0x4463_55CC]),
        ];
        // `values(n, 1) · values(n, 2)` with the first vector in f16
        // (`ggml_vec_dot_f16_f32`), and both in f32.
        #[rustfmt::skip]
        let pairs: [(usize, u32, u32); 5] = [
            (100, 0x4075_10C2, 0x4075_1825),
            (128, 0x40BB_E4B7, 0x40BB_E7D6),
            (288, 0x3FA6_092D, 0x3FA6_25F4),
            (1024, 0x40DA_B188, 0x40DA_B60A),
            (1027, 0x40D3_A195, 0x40D3_A62D),
        ];
        for k in variants() {
            // Without `dotprod`, ggml's `ggml_vdotq_s32` fallback (the same
            // `vmull` sum as [`i8_dot16_mul`]) groups the products into
            // lanes differently, and Q8_0 scales the lanes before summing
            // them: its results are those of a ggml build without
            // `dotprod`, not of this one.
            for (n, want) in q8_0.into_iter().filter(|_| k.dotprod) {
                let xq = quant::encoded(DotType::Q8_0, &values(n, 3));
                let rows: Vec<u8> = (0..5)
                    .flat_map(|r| quant::encoded(DotType::Q8_0, &values(n, 10 + r)))
                    .collect();
                let mut got = [0.0f32; 5];
                k.dot_rows(GgmlType::Q8_0, &rows, rows.len() / 5, &xq, &mut got);
                assert_eq!(got.map(f32::to_bits), want, "Q8_0 n = {n}, {k:?}");
            }
            let xq = quant::encoded(DotType::Q8K, &values(768, 3));
            for (ty, want) in k_quants {
                let rows: Vec<u8> = (0..4).flat_map(|r| k_row(ty, 3, 10 + r)).collect();
                let mut got = [0.0f32; 4];
                k.dot_rows(ty, &rows, rows.len() / 4, &xq, &mut got);
                assert_eq!(got.map(f32::to_bits), want, "{ty}, {k:?}");
            }
            for (n, want_f16, want_f32) in pairs {
                let (a, b) = (values(n, 1), values(n, 2));
                let mut got = [0.0f32];
                k.dot_f16_f32_rows(&f16_bytes(&a), 2 * n, &b, &mut got);
                let got = got[0];
                assert_eq!(got.to_bits(), want_f16, "f16 n = {n}: {got}, {k:?}");
                let mut got = [0.0f32];
                k.dot_rows(
                    GgmlType::F16,
                    &f16_bytes(&a),
                    2 * n,
                    &f32_bytes(&b),
                    &mut got,
                );
                assert_eq!(got[0].to_bits(), want_f16, "f16 rows n = {n}: {}", got[0]);
                k.dot_rows(
                    GgmlType::F32,
                    &f32_bytes(&a),
                    4 * n,
                    &f32_bytes(&b),
                    &mut got,
                );
                assert_eq!(got[0].to_bits(), want_f32, "f32 rows n = {n}: {}", got[0]);
                // SAFETY: `variants` returned a token, so NEON is available.
                let got = unsafe { dot_f32(&a, &b) };
                assert_eq!(got.to_bits(), want_f32, "f32 n = {n}: {got}");
            }
        }
    }

    /// Rows computed four at a time give exactly the single-row results,
    /// for any row count and with rows further apart than their length.
    #[test]
    fn f16_f32_rows_match_single_rows_exactly() {
        for k in variants() {
            for n in [5, 64, 100, 288] {
                let stride = 2 * n + 6;
                for rows in [1, 3, 4, 5, 9] {
                    let mut a = vec![0u8; stride * rows];
                    for r in 0..rows {
                        let row = f16_bytes(&values(n, 40 + r as u32));
                        a[r * stride..r * stride + 2 * n].copy_from_slice(&row);
                    }
                    let b = values(n, 3);
                    let mut got = vec![0.0f32; rows];
                    k.dot_f16_f32_rows(&a, stride, &b, &mut got);
                    for (r, &g) in got.iter().enumerate() {
                        let mut want = [0.0f32];
                        k.dot_f16_f32_rows(&a[r * stride..], stride, &b, &mut want);
                        assert_eq!(
                            g.to_bits(),
                            want[0].to_bits(),
                            "n = {n}, row {r} of {rows}, {k:?}"
                        );
                    }
                }
            }
        }
    }

    /// The BF16 kernel departs from ggml's on purpose (see [`dot_bf16`]);
    /// this bounds the difference from ggml's f64 sums.
    #[test]
    fn bf16_dot_is_close_to_ggml() {
        let ggml = [
            (100, 3.824_654_3),
            (128, 5.869_657_5),
            (288, 1.290_095_3),
            (1024, 6.834_092_6),
            (1027, 6.613_547_3),
        ];
        for k in variants() {
            for (n, want) in ggml {
                let (a, b) = (values(n, 1), values(n, 2));
                let scale: f64 = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| f64::from(x.abs() * y.abs()))
                    .sum();
                let mut got = [0.0f32];
                k.dot_rows(
                    GgmlType::Bf16,
                    &bf16_bytes(&a),
                    2 * n,
                    &bf16_bytes(&b),
                    &mut got,
                );
                assert_near(got[0], want, 1e-6, scale, &format!("bf16 n = {n}"));
            }
        }
    }

    /// ggml's NEON `quantize_row_q8_0` on 4096 random blocks, and on the
    /// blocks where its `1 / d` inverse rounds differently from `127 / amax`.
    #[test]
    fn q8_0_encoder_matches_ggml() {
        for k in variants() {
            let x = scaled(4096 * QK8_0, 5, 3.0);
            let mut q = vec![0; 4096 * Q8_0_BLOCK_BYTES];
            k.encode(DotType::Q8_0, &x, &mut q);
            assert_eq!(fnv1a(&q), 0x71C3_A875_DC93_5F54);
            for (v, want) in [(0x3CCA_8FD5u32, 2i8), (0x3D28_CD32, 3)] {
                let mut x = [0.0f32; QK8_0];
                x[0] = f32::from_bits(0x4005_FC79);
                x[1] = f32::from_bits(v);
                x[2] = -f32::from_bits(v);
                let mut q = [0u8; Q8_0_BLOCK_BYTES];
                k.encode(DotType::Q8_0, &x, &mut q);
                assert_eq!([q[2] as i8, q[3] as i8, q[4] as i8], [127, want, -want]);
            }
        }
    }

    #[test]
    fn q8_0_minus_128_is_close_to_the_scalar_dot() {
        for k in variants() {
            let mut row = quant::encoded(DotType::Q8_0, &values(QK8_0, 7));
            row[2] = (-128i8) as u8;
            row[9] = (-128i8) as u8;
            let xq = quant::encoded(DotType::Q8_0, &values(QK8_0, 8));
            let mut got = [0.0];
            k.dot_rows(GgmlType::Q8_0, &row, row.len(), &xq, &mut got);
            let want = quant::dot_row(GgmlType::Q8_0, &row, &xq);
            assert_near(
                got[0],
                want,
                1e-6,
                f64::from(want.abs()).max(1e-6),
                "q8_0 -128",
            );
        }
    }

    #[test]
    fn k_quant_edges_match_the_scalar_reference() {
        for k in variants() {
            for ty in K_QUANTS {
                let n = ty.layout().unwrap().type_size as usize;
                // All-zero block, a subnormal f16 scale, and a -128 activation.
                let mut row = vec![0u8; n];
                let scale = 0x0001u16.to_le_bytes();
                match ty {
                    GgmlType::Q4K | GgmlType::Q5K => {
                        row[..2].copy_from_slice(&scale);
                        row[2..4].copy_from_slice(&0x8001u16.to_le_bytes());
                    }
                    GgmlType::Q6K => row[208..210].copy_from_slice(&scale),
                    GgmlType::Q3K => row[108..110].copy_from_slice(&scale),
                    _ => {}
                }
                let mut xq = quant::encoded(DotType::Q8K, &vec![0.25; 256]);
                xq[4] = (-128i8) as u8;
                let sum: i16 = xq[4..20].iter().map(|&q| i16::from(q as i8)).sum();
                xq[260..262].copy_from_slice(&sum.to_le_bytes());
                let mut got = [0.0];
                k.dot_rows(ty, &row, row.len(), &xq, &mut got);
                assert_eq!(
                    got[0].to_bits(),
                    quant::dot_row(ty, &row, &xq).to_bits(),
                    "{ty}, {k:?}"
                );
            }
        }
    }

    #[test]
    fn vector_dots_and_f16_store_match_the_scalar_reference() {
        for k in variants() {
            for n in LENGTHS {
                let (a, b) = (values(n, 1), values(n, 2));
                let scale: f64 = a
                    .iter()
                    .zip(&b)
                    .map(|(x, y)| f64::from(x.abs() * y.abs()))
                    .sum();
                // SAFETY: `variants` returned a token, so NEON is available.
                let got = unsafe { dot_f32(&a, &b) };
                assert_near(
                    got,
                    scalar::dot_f32(&a, &b),
                    1e-6,
                    scale.max(1e-12),
                    &format!("dot_f32 n = {n}"),
                );
                let a16 = f16_bytes(&a);
                let mut got = [0.0f32];
                k.dot_f16_f32_rows(&a16, a16.len(), &b, &mut got);
                assert_near(
                    got[0],
                    scalar::dot_f16_f32(&a16, &b),
                    1e-5,
                    scale.max(1e-12),
                    &format!("dot_f16_f32 n = {n}, {k:?}"),
                );
                let mut got = vec![0; 2 * n];
                k.to_f16(&scaled(n, 4, 70_000.0), &mut got);
                assert_eq!(
                    got,
                    f16_bytes(&scaled(n, 4, 70_000.0)),
                    "to_f16 n = {n}, {k:?}"
                );
            }
        }
    }

    #[test]
    fn exp_is_accurate_and_handles_the_extremes() {
        if variants().is_empty() {
            return;
        }
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
        xs.resize(xs.len().next_multiple_of(4), 0.0);
        for group in xs.as_chunks::<4>().0 {
            let mut out = [0.0; 4];
            // SAFETY: `variants` returned a token, so NEON is available.
            unsafe { store_f32(&mut out, exp4(load_f32(group))) };
            for (&x, &got) in group.iter().zip(&out) {
                let want = f64::from(x).exp();
                if want > f64::from(f32::MAX) {
                    assert_eq!(got, f32::INFINITY, "e^{x}");
                } else if want < 1e-38 {
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
    fn activations_and_softmax_match_the_scalar_reference() {
        let Some(&k) = variants().first() else { return };
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
                    &format!("silu[{i}]"),
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
        }
        // A NaN makes the normalizing sum NaN, so the whole row is NaN, as in
        // the scalar softmax.
        let mut x = [1.0f32, f32::NAN, 3.0, 0.5];
        let mut want = x;
        k.softmax(&mut x);
        scalar::softmax(&mut want);
        for (&a, &b) in x.iter().zip(&want) {
            assert!(a.is_nan() && b.is_nan());
        }
    }

    #[test]
    fn delta_net_head_matches_the_scalar_reference() {
        let Some(&k) = variants().first() else { return };
        let unit = |v: Vec<f32>| {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / norm).collect::<Vec<_>>()
        };
        for d in [4, 8, 32, 128] {
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

    #[test]
    fn dot_rows_prompt_matches_scalar_reference() {
        let Some(k) = Neon::detect() else { return };
        let cols = 512;
        let rows_count = 11;
        for ty in [GgmlType::Q8_0, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let mut rows = Vec::new();
            for r in 0..rows_count {
                rows.extend(quant::sample_row(ty, cols, 200 + r as u32));
            }
            let row_bytes = rows.len() / rows_count;
            let dt = DotType::of(ty);
            let xs: Vec<Vec<u8>> = (0..8)
                .map(|c| quant::encoded(dt, &values(cols, 50 + c as u32)))
                .collect();
            let want: Vec<Vec<f32>> = xs
                .iter()
                .map(|xq| {
                    rows.chunks_exact(row_bytes)
                        .map(|row| quant::dot_row(ty, row, xq))
                        .collect()
                })
                .collect();

            let mut out4 = std::array::from_fn::<_, 4, _>(|_| vec![0.0f32; rows_count]);
            let xq4: [&[u8]; 4] = std::array::from_fn(|c| xs[c].as_slice());
            assert!(k.dot_rows_prompt(
                ty,
                &rows,
                row_bytes,
                xq4,
                out4.each_mut().map(Vec::as_mut_slice),
            ));
            for c in 0..4 {
                for r in 0..rows_count {
                    assert_near(
                        out4[c][r],
                        want[c][r],
                        1e-5,
                        f64::from(want[c][r].abs().max(1.0)),
                        &format!("{ty} N=4 row {r} col {c}"),
                    );
                }
            }

            if k.i8mm {
                let mut out8 = std::array::from_fn::<_, 8, _>(|_| vec![0.0f32; rows_count]);
                let xq8: [&[u8]; 8] = std::array::from_fn(|c| xs[c].as_slice());
                assert!(k.dot_rows_prompt(
                    ty,
                    &rows,
                    row_bytes,
                    xq8,
                    out8.each_mut().map(Vec::as_mut_slice),
                ));
                for c in 0..8 {
                    for r in 0..rows_count {
                        assert_near(
                            out8[c][r],
                            want[c][r],
                            1e-5,
                            f64::from(want[c][r].abs().max(1.0)),
                            &format!("{ty} N=8 row {r} col {c}"),
                        );
                    }
                }
            }
        }
    }
}
