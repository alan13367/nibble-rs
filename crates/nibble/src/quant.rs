//! Scalar reference implementations of the weight formats.
//!
//! As in ggml's CPU backend, a dot product between a weight row and an
//! activation vector first converts the activation to the weight type's
//! "dot type" (`vec_dot_type`): BF16 weights use BF16 activations, Q8_0
//! weights use Q8_0 activations, and the K-quants (`Q3_K` .. `Q6_K`) share
//! one `Q8_K` activation. F32 and F16 weights use the f32 activation as it
//! is (ggml stopped rounding it to f16 for F16 weights in November 2026,
//! before llama.cpp b12359). The products themselves are the integer dots
//! from `ggml_vec_dot_q*_K_q8_K`. Matching this keeps greedy output in step
//! with llama.cpp, and it is the form the SIMD kernels compute in.
//!
//! These functions are the correctness oracle for every SIMD and GPU kernel,
//! so they favor clarity over speed.
//!
//! Where ggml's C code writes `a * b + c` in one expression, clang (which
//! builds llama.cpp's macOS releases, with the default `-ffp-contract=on`)
//! emits a fused multiply-add, so the code here uses [`f32::mul_add`] at
//! exactly those places. Rust never fuses on its own.

use nibble_gguf::GgmlType;

pub(crate) mod blocks32;
pub(crate) mod iq_grids;
pub(crate) mod iquants;

pub(crate) use blocks32::Q8_1_BLOCK_BYTES;

/// Elements per Q8_0 block.
pub(crate) const QK8_0: usize = 32;
/// Bytes per Q8_0 block: an f16 scale and 32 signed bytes.
pub(crate) const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

/// Elements per K-quant super-block.
pub(crate) const QK_K: usize = 256;
/// Bytes per `Q8_K` block: an f32 scale, 256 signed bytes and 16 i16 sums.
///
/// The sums are of each group of 16 quants. `Q4_K` and `Q5_K` use them to
/// subtract the block minimum without touching every element, and `Q6_K`
/// uses them to undo the +32 bias of its unsigned 6-bit quants.
pub(crate) const Q8_K_BLOCK_BYTES: usize = 4 + QK_K + QK_K / 8;
/// Bytes per `Q4_K` block: f16 scale, f16 min scale, 12 bytes of 6-bit
/// scales and mins, 128 bytes of nibbles.
pub(crate) const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 2;
/// Bytes per `Q5_K` block: the `Q4_K` layout plus 32 bytes of high bits,
/// between the scales and the low nibbles.
pub(crate) const Q5_K_BLOCK_BYTES: usize = 2 + 2 + 12 + QK_K / 8 + QK_K / 2;
/// Bytes per `Q6_K` block: low nibbles, high pairs, i8 scales, then the f16 scale.
pub(crate) const Q6_K_BLOCK_BYTES: usize = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
/// Bytes per `Q2_K` block: 4-bit scales and mins, 2-bit quants, f16 scale and min scale.
pub(crate) const Q2_K_BLOCK_BYTES: usize = QK_K / 16 + QK_K / 4 + 2 + 2;
/// Bytes per `Q3_K` block: high-bit mask, 2-bit quants, 6-bit scales, f16 scale.
pub(crate) const Q3_K_BLOCK_BYTES: usize = QK_K / 8 + QK_K / 4 + 12 + 2;
/// Bytes per `IQ4_XS` block: f16 scale, the high 2 bits of the eight 6-bit
/// group scales, their low 4 bits, then 128 bytes of nibbles that index
/// [`KVALUES_IQ4NL`].
pub(crate) const IQ4_XS_BLOCK_BYTES: usize = 2 + 2 + QK_K / 64 + QK_K / 2;

/// The 16 values an `IQ4_NL`/`IQ4_XS` nibble stands for (`kvalues_iq4nl`).
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Converts IEEE half-precision bits to `f32` (exact).
#[must_use]
pub(crate) fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);
    let magnitude = match exp {
        0 => {
            // Zero or subnormal: mantissa * 2^-24, exact in f32.
            let v = mantissa as f32 * (1.0 / 16_777_216.0);
            return if sign != 0 { -v } else { v };
        }
        0x1F => 0x7F80_0000 | (mantissa << 13), // inf / NaN
        _ => ((exp + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(sign | magnitude)
}

/// Converts `f32` to IEEE half-precision bits, rounding to nearest even
/// (the F16C `vcvtps2ph` result).
#[must_use]
pub(crate) fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exp == 0xFF {
        let nan = if mantissa != 0 {
            0x200 | (mantissa >> 13) as u16
        } else {
            0
        };
        return sign | 0x7C00 | nan;
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00; // overflow to infinity
    }
    let (value, shift) = if e > 0 {
        // Normal: keep 10 of 23 mantissa bits.
        (((e as u32) << 23) | mantissa, 13)
    } else {
        // Subnormal (or underflow): value / 2^-24, from the 24-bit significand.
        let shift = (14 - e) as u32;
        if shift > 24 {
            return sign;
        }
        (mantissa | 0x80_0000, shift)
    };
    let mut h = value >> shift;
    let rem = value & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rem > half || (rem == half && h & 1 == 1) {
        h += 1; // a carry into the exponent is still the right encoding
    }
    sign | h as u16
}

/// Converts bfloat16 bits to `f32` (exact).
#[must_use]
pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// Converts `f32` to bfloat16 bits, rounding to nearest even (ggml's `GGML_FP32_TO_BF16`).
#[must_use]
pub(crate) fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((bits >> 16) | 0x40) as u16; // quiet NaN
    }
    ((bits + (0x7FFF + ((bits >> 16) & 1))) >> 16) as u16
}

/// Weight types the CPU reference path can compute with.
pub(crate) const SUPPORTED: &[GgmlType] = &[
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Bf16,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::Q5_0,
    GgmlType::Q5_1,
    GgmlType::Iq4Nl,
    GgmlType::Mxfp4,
    GgmlType::Q2K,
    GgmlType::Q3K,
    GgmlType::Q4K,
    GgmlType::Q5K,
    GgmlType::Q6K,
    GgmlType::Iq4Xs,
    GgmlType::Iq2Xxs,
    GgmlType::Iq2Xs,
    GgmlType::Iq2S,
    GgmlType::Iq3Xxs,
    GgmlType::Iq3S,
    GgmlType::Iq1S,
    GgmlType::Iq1M,
];

/// The formats of 32-weight blocks in [`blocks32`].
pub(crate) const fn is_block32(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q5_0
            | GgmlType::Q5_1
            | GgmlType::Iq4Nl
            | GgmlType::Mxfp4
    )
}

/// The codebook formats in [`iquants`].
pub(crate) const fn is_codebook(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Iq2Xxs
            | GgmlType::Iq2Xs
            | GgmlType::Iq2S
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S
            | GgmlType::Iq1S
            | GgmlType::Iq1M
    )
}

/// Whether the CPU reference path can compute with this weight type.
#[must_use]
pub(crate) fn is_supported(ty: GgmlType) -> bool {
    SUPPORTED.contains(&ty)
}

/// Decodes one row of `out.len()` elements.
///
/// `row` must hold exactly one encoded row of a supported type.
pub(crate) fn dequantize_row(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, b) in out.iter_mut().zip(row.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*b);
            }
        }
        GgmlType::F16 => {
            for (o, b) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = f16_to_f32(u16::from_le_bytes(*b));
            }
        }
        GgmlType::Bf16 => {
            for (o, b) in out.iter_mut().zip(row.as_chunks::<2>().0) {
                *o = bf16_to_f32(u16::from_le_bytes(*b));
            }
        }
        GgmlType::Q8_0 => {
            let blocks = row.as_chunks::<Q8_0_BLOCK_BYTES>().0;
            for (o, block) in out.as_chunks_mut::<QK8_0>().0.iter_mut().zip(blocks) {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for (o, &q) in o.iter_mut().zip(&block[2..]) {
                    *o = d * f32::from(q as i8);
                }
            }
        }
        GgmlType::Q4K => dequantize_q4_k(row, out),
        GgmlType::Q5K => dequantize_q5_k(row, out),
        GgmlType::Q6K => dequantize_q6_k(row, out),
        GgmlType::Q3K => dequantize_q3_k(row, out),
        GgmlType::Iq4Xs => dequantize_iq4_xs(row, out),
        GgmlType::Q2K => dequantize_q2_k(row, out),
        ty if is_block32(ty) => blocks32::dequantize(ty, row, out),
        ty if is_codebook(ty) => iquants::dequantize(ty, row, out),
        other => unreachable!("dequantize_row called with unsupported type {other}"),
    }
}

/// The format an activation is encoded in before its dot products with the
/// rows of one weight type: ggml's `vec_dot_type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DotType {
    F32,
    Bf16,
    Q8_0,
    /// `Q8_K`, shared by the K-quants and the I-quants of 256-weight blocks.
    Q8K,
    /// `Q8_0` with each block's sum (`Q8_1`), for `Q4_1` and `Q5_1`.
    Q8_1,
}

impl DotType {
    /// Number of dot types, for tables indexed by [`DotType::index`].
    pub(crate) const COUNT: usize = 5;

    /// The dot type of weight type `ty`.
    ///
    /// # Panics
    ///
    /// If `ty` is not [supported](is_supported). Weights are checked when
    /// they are bound, so this is an internal invariant.
    #[must_use]
    pub(crate) fn of(ty: GgmlType) -> Self {
        match ty {
            GgmlType::F32 | GgmlType::F16 => Self::F32,
            GgmlType::Bf16 => Self::Bf16,
            GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q5_0
            | GgmlType::Iq4Nl
            | GgmlType::Mxfp4 => Self::Q8_0,
            GgmlType::Q4_1 | GgmlType::Q5_1 => Self::Q8_1,
            GgmlType::Q2K
            | GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Iq4Xs => Self::Q8K,
            ty if is_codebook(ty) => Self::Q8K,
            other => unreachable!("no dot type for unsupported weight type {other}"),
        }
    }

    /// A dense index in `0..COUNT`.
    #[must_use]
    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    /// Bytes that encode `n` values; `n` is a multiple of the block size.
    #[must_use]
    pub(crate) const fn encoded_len(self, n: usize) -> usize {
        match self {
            Self::F32 => 4 * n,
            Self::Bf16 => 2 * n,
            Self::Q8_0 => n / QK8_0 * Q8_0_BLOCK_BYTES,
            Self::Q8K => n / QK_K * Q8_K_BLOCK_BYTES,
            Self::Q8_1 => n / QK8_0 * Q8_1_BLOCK_BYTES,
        }
    }
}

/// Encodes the activation `x` as `dt` into `out`, which holds exactly
/// `dt.encoded_len(x.len())` bytes.
pub(crate) fn encode_activation(dt: DotType, x: &[f32], out: &mut [u8]) {
    debug_assert_eq!(out.len(), dt.encoded_len(x.len()));
    match dt {
        DotType::F32 => {
            for (o, &v) in out.as_chunks_mut::<4>().0.iter_mut().zip(x) {
                *o = v.to_le_bytes();
            }
        }
        DotType::Bf16 => {
            for (o, &v) in out.as_chunks_mut::<2>().0.iter_mut().zip(x) {
                *o = f32_to_bf16(v).to_le_bytes();
            }
        }
        DotType::Q8_0 => quantize_row_q8_0(x, out, Q8_0Inverse::NATIVE),
        DotType::Q8K => quantize_row_q8_k(x, out),
        DotType::Q8_1 => blocks32::quantize_row_q8_1(x, out, Q8_0Inverse::NATIVE),
    }
}

/// How `quantize_row_q8_0` computes the inverse scale `id` from the block's
/// largest magnitude `amax`. ggml's kernels disagree: for some `amax` the
/// two forms differ by an ulp, which moves a quant whose `x * id` is that
/// close to a rounding boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Q8_0Inverse {
    /// `127 / amax`, as the AVX2 kernel (`arch/x86/quants.c`).
    Direct,
    /// `1 / d` with `d = amax / 127`, as the NEON kernel (`arch/arm/quants.c`)
    /// and the reference `quantize_row_q8_0_ref`.
    OfScale,
}

impl Q8_0Inverse {
    /// The form of ggml's SIMD kernel for this target, so that the scalar
    /// encoder writes the same bytes as the SIMD one, and as llama.cpp.
    pub(crate) const NATIVE: Self = if cfg!(target_arch = "x86_64") {
        Self::Direct
    } else {
        Self::OfScale
    };
}

/// `quantize_row_q8_0`: per block of 32, `d = amax / 127` stored as f16, and
/// `q = round(x * id)` rounding ties to even, as both SIMD kernels do (the
/// scalar `_ref` rounds ties away from zero; no SIMD build of llama.cpp uses it).
pub(crate) fn quantize_row_q8_0(x: &[f32], out: &mut [u8], inverse: Q8_0Inverse) {
    let blocks = out.as_chunks_mut::<Q8_0_BLOCK_BYTES>().0;
    for (o, block) in blocks.iter_mut().zip(x.as_chunks::<QK8_0>().0) {
        let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = match inverse {
            Q8_0Inverse::Direct if amax != 0.0 => 127.0 / amax,
            Q8_0Inverse::OfScale if d != 0.0 => 1.0 / d,
            _ => 0.0,
        };
        let (scale, qs) = o.split_at_mut(2);
        scale.copy_from_slice(&f32_to_f16(d).to_le_bytes());
        for (q, &v) in qs.iter_mut().zip(block) {
            *q = (v * id).round_ties_even() as i8 as u8;
        }
    }
}

/// `x` encoded as `dt`, in a new buffer.
#[cfg(test)]
pub(crate) fn encoded(dt: DotType, x: &[f32]) -> Vec<u8> {
    let mut out = vec![0; dt.encoded_len(x.len())];
    encode_activation(dt, x, &mut out);
    out
}

/// Dot product of a weight row with an activation encoded by [`encode_activation`].
#[must_use]
pub(crate) fn dot_row(ty: GgmlType, row: &[u8], xq: &[u8]) -> f32 {
    match ty {
        GgmlType::F32 => dot_by(row.as_chunks::<4>().0, xq.as_chunks::<4>().0, |b| {
            f32::from_le_bytes(*b)
        }),
        GgmlType::F16 => {
            // The portable `ggml_vec_dot_f16_f32`: each product in f32, the
            // sum in f64.
            let mut sum = 0.0f64;
            for (w, x) in row.as_chunks::<2>().0.iter().zip(xq.as_chunks::<4>().0) {
                sum += f64::from(f16_to_f32(u16::from_le_bytes(*w)) * f32::from_le_bytes(*x));
            }
            sum as f32
        }
        GgmlType::Bf16 => dot_by(row.as_chunks::<2>().0, xq.as_chunks::<2>().0, |b| {
            bf16_to_f32(u16::from_le_bytes(*b))
        }),
        GgmlType::Q8_0 => {
            let w = row.as_chunks::<Q8_0_BLOCK_BYTES>().0;
            let x = xq.as_chunks::<Q8_0_BLOCK_BYTES>().0;
            let mut sum = 0.0f32;
            for (wb, xb) in w.iter().zip(x) {
                let sumi: i32 = wb[2..]
                    .iter()
                    .zip(&xb[2..])
                    .map(|(&a, &b)| i32::from(a as i8) * i32::from(b as i8))
                    .sum();
                let dw = f16_to_f32(u16::from_le_bytes([wb[0], wb[1]]));
                let dx = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
                // `sumf += sumi*(dx*dy)`, fused (`ggml_vec_dot_q8_0_q8_0_generic`).
                sum = (sumi as f32).mul_add(dw * dx, sum);
            }
            sum
        }
        GgmlType::Q4K => dot_q4_k(row, xq),
        GgmlType::Q5K => dot_q5_k(row, xq),
        GgmlType::Q6K => dot_q6_k(row, xq),
        GgmlType::Q3K => dot_q3_k(row, xq),
        GgmlType::Iq4Xs => dot_iq4_xs(row, xq),
        GgmlType::Q2K => dot_q2_k(row, xq),
        ty if is_block32(ty) => blocks32::dot_row(ty, row, xq),
        ty if is_codebook(ty) => iquants::dot_row(ty, row, xq),
        other => unreachable!("dot_row called with unsupported type {other}"),
    }
}

// --- IQ4_XS ----------------------------------------------------------------
//
// ggml's `block_iq4_xs`: 8 groups of 32 with 6-bit scales (bias 32), each
// nibble an index into a fixed non-linear table. Byte `j` of a group's 16
// holds value `j` in its low nibble and value `j + 16` in its high nibble.

/// The scale of group `g` (`0..8`), with its bias removed.
fn iq4_xs_scale(block: &[u8; IQ4_XS_BLOCK_BYTES], g: usize) -> i32 {
    let high = u16::from_le_bytes([block[2], block[3]]);
    let low = (block[4 + g / 2] >> (4 * (g % 2))) & 0xF;
    i32::from(low | (((high >> (2 * g)) & 3) as u8) << 4) - 32
}

/// `dequantize_row_iq4_xs`: `(d * scale) * value`.
fn dequantize_iq4_xs(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<IQ4_XS_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK_K>().0.iter_mut().zip(blocks) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (g, out) in out.as_chunks_mut::<32>().0.iter_mut().enumerate() {
            let dl = d * iq4_xs_scale(block, g) as f32;
            let qs = &block[8 + 16 * g..8 + 16 * g + 16];
            for (j, &q) in qs.iter().enumerate() {
                out[j] = dl * f32::from(KVALUES_IQ4NL[usize::from(q & 0xF)]);
                out[j + 16] = dl * f32::from(KVALUES_IQ4NL[usize::from(q >> 4)]);
            }
        }
    }
}

/// `ggml_vec_dot_iq4_xs_q8_K`, NEON path: each group's integer dot times its
/// scale, summed exactly over the super-block, then
/// `sumf += d * d8 * isum` (one fused operation).
fn dot_iq4_xs(row: &[u8], xq: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let blocks = row.as_chunks::<IQ4_XS_BLOCK_BYTES>().0;
    let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter().zip(xs) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let (xd, xqs, _) = q8k_block(x);
        let mut isum = 0i32;
        for g in 0..8 {
            let qs = &block[8 + 16 * g..8 + 16 * g + 16];
            let q8 = &xqs[32 * g..32 * g + 32];
            let mut dot = 0i32;
            for (j, &q) in qs.iter().enumerate() {
                dot += i32::from(KVALUES_IQ4NL[usize::from(q & 0xF)]) * i32::from(q8[j] as i8);
                dot += i32::from(KVALUES_IQ4NL[usize::from(q >> 4)]) * i32::from(q8[j + 16] as i8);
            }
            isum += dot * iq4_xs_scale(block, g);
        }
        sum = (d * xd).mul_add(isum as f32, sum);
    }
    sum
}

// --- K-quants --------------------------------------------------------------
//
// Block layouts are ggml's `block_q*_K` (`ggml-common.h`). Dequantization
// follows `dequantize_row_q*_K`. The dots follow the NEON
// `ggml_vec_dot_q*_K_q8_K` (`arch/arm/quants.c`): exact integer sums per
// super-block, minimums and the Q6 bias applied through `Q8_K.bsums`, then
// that kernel's f32 update of the running sum, fused multiply-adds included,
// so the NEON kernels can be compared with these bit for bit.

/// `nearest_int` from `ggml-quants.c`: round half to even via the 2^23 bias.
fn nearest_int(fval: f32) -> i32 {
    let bits = (fval + 12_582_912.0).to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

/// `quantize_row_q8_K_ref`. The scale is `-max/127` where `max` is the
/// value of largest magnitude (the first one, if several tie), so the
/// extreme quant is -127 and `d * q` restores its sign. An all-zero block
/// is stored as zeros, including the partial sums: the output buffer is
/// reused, and a stale sum would poison every `Q4_K`/`Q5_K`/`Q6_K` dot.
fn quantize_row_q8_k(x: &[f32], out: &mut [u8]) {
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
        for (q, &v) in qs.iter_mut().zip(x) {
            // `MIN(127, nearest_int(...))`, then the `int8_t` store, which wraps.
            *q = nearest_int(iscale * v).min(127) as i8 as u8;
        }
        for (j, slot) in bsums.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let sum: i16 = qs[j * 16..(j + 1) * 16]
                .iter()
                .map(|&q| i16::from(q as i8))
                .sum();
            *slot = sum.to_le_bytes();
        }
    }
}

/// Scale and the 256 signed quants of one `Q8_K` block, plus its 16 group sums.
fn q8k_block(block: &[u8; Q8_K_BLOCK_BYTES]) -> (f32, &[u8; QK_K], [i16; 16]) {
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    let qs: &[u8; QK_K] = block[4..4 + QK_K].try_into().unwrap();
    let mut bsums = [0i16; 16];
    for (slot, bytes) in bsums.iter_mut().zip(block[4 + QK_K..].as_chunks::<2>().0) {
        *slot = i16::from_le_bytes(*bytes);
    }
    (d, qs, bsums)
}

/// One 6-bit scale and its min from the 12-byte `Q4_K`/`Q5_K` table
/// (`get_scale_min_k4`). `j` is the group of 32 weights, `0..8`.
pub(crate) fn scale_min_k4(q: &[u8], j: usize) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let min = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (scale, min)
    }
}

/// The 16 signed `Q3_K` scales (`dequantize_row_q3_K`'s nibble shuffle).
pub(crate) fn q3_scales(scales: &[u8]) -> [i8; 16] {
    let mut aux = [
        u32::from_le_bytes(scales[0..4].try_into().unwrap()),
        u32::from_le_bytes(scales[4..8].try_into().unwrap()),
        u32::from_le_bytes(scales[8..12].try_into().unwrap()),
        0,
    ];
    let tmp = aux[2];
    const K1: u32 = 0x0303_0303;
    const K2: u32 = 0x0f0f_0f0f;
    aux[2] = ((aux[0] >> 4) & K2) | (((tmp >> 4) & K1) << 4);
    aux[3] = ((aux[1] >> 4) & K2) | (((tmp >> 6) & K1) << 4);
    aux[0] = (aux[0] & K2) | (((tmp) & K1) << 4);
    aux[1] = (aux[1] & K2) | (((tmp >> 2) & K1) << 4);
    let mut out = [0i8; 16];
    for (i, word) in aux.iter().enumerate() {
        let bytes = word.to_le_bytes();
        for k in 0..4 {
            out[i * 4 + k] = bytes[k] as i8;
        }
    }
    out
}

/// `y = d1 * q - m1` with `d1 = d * scale` and `m1 = dmin * min`, fused as
/// in `dequantize_row_q4_K` / `dequantize_row_q5_K`.
fn dequantize_k_mins(d: f32, dmin: f32, scale: u8, min: u8, q: u8) -> f32 {
    (d * f32::from(scale)).mul_add(f32::from(q), -(dmin * f32::from(min)))
}

fn dequantize_q4_k(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<Q4_K_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK_K>().0.iter_mut().zip(blocks) {
        let (d, dmin, scales, mins, q) = unpack_q4_k(block);
        for (i, o) in out.iter_mut().enumerate() {
            *o = dequantize_k_mins(d, dmin, scales[i / 32], mins[i / 32], q[i]);
        }
    }
}

fn dequantize_q5_k(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<Q5_K_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK_K>().0.iter_mut().zip(blocks) {
        let (d, dmin, scales, mins, q) = unpack_q5_k(block);
        for (i, o) in out.iter_mut().enumerate() {
            *o = dequantize_k_mins(d, dmin, scales[i / 32], mins[i / 32], q[i]);
        }
    }
}

fn dequantize_q6_k(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<Q6_K_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK_K>().0.iter_mut().zip(blocks) {
        let (d, scales, q) = unpack_q6_k(block);
        for (i, o) in out.iter_mut().enumerate() {
            *o = d * f32::from(scales[i / 16]) * f32::from(q[i]);
        }
    }
}

fn dequantize_q3_k(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<Q3_K_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK_K>().0.iter_mut().zip(blocks) {
        let (d, scales, q) = unpack_q3_k(block);
        for (i, o) in out.iter_mut().enumerate() {
            let scale = i32::from(scales[i / 16]) - 32;
            *o = d * scale as f32 * f32::from(q[i]);
        }
    }
}

/// `Q4_K`: 8 groups of 32. Quants are `0..15`.
fn unpack_q4_k(block: &[u8; Q4_K_BLOCK_BYTES]) -> (f32, f32, [u8; 8], [u8; 8], [u8; QK_K]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let table = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..8 {
        let (scale, min) = scale_min_k4(table, j);
        scales[j] = scale;
        mins[j] = min;
    }
    let mut q = [0u8; QK_K];
    let qs = &block[16..];
    for j in 0..4 {
        for l in 0..32 {
            q[j * 64 + l] = qs[j * 32 + l] & 0xF;
            q[j * 64 + 32 + l] = qs[j * 32 + l] >> 4;
        }
    }
    (d, dmin, scales, mins, q)
}

/// `Q5_K`: the high bit of each quant lives in `qh`, one pair of bits per
/// 32+32 group, walking up the same 32 mask bytes.
fn unpack_q5_k(block: &[u8; Q5_K_BLOCK_BYTES]) -> (f32, f32, [u8; 8], [u8; 8], [u8; QK_K]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let table = &block[4..16];
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];
    for j in 0..8 {
        let (scale, min) = scale_min_k4(table, j);
        scales[j] = scale;
        mins[j] = min;
    }
    let qh = &block[16..48];
    let ql = &block[48..];
    let mut q = [0u8; QK_K];
    // `u1`/`u2` shift by two after every 64 weights; wrapping matches the
    // `uint8_t` shift, which overflows on the unused iteration past the end.
    let mut bit_lo: u8 = 1;
    let mut bit_hi: u8 = 2;
    for j in 0..4 {
        for l in 0..32 {
            let lo = (ql[j * 32 + l] & 0xF) + u8::from(qh[l] & bit_lo != 0) * 16;
            let hi = (ql[j * 32 + l] >> 4) + u8::from(qh[l] & bit_hi != 0) * 16;
            q[j * 64 + l] = lo;
            q[j * 64 + 32 + l] = hi;
        }
        bit_lo = bit_lo.wrapping_shl(2);
        bit_hi = bit_hi.wrapping_shl(2);
    }
    (d, dmin, scales, mins, q)
}

/// `Q6_K`: quants are `nibble | (2 high bits) - 32`, in `-32..31`, with one
/// i8 scale per 16 weights. The four 32-wide slices of each 128 are
/// interleaved in `ql`/`qh` the way `dequantize_row_q6_K` reads them.
fn unpack_q6_k(block: &[u8; Q6_K_BLOCK_BYTES]) -> (f32, [i8; 16], [i8; QK_K]) {
    let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let mut scales = [0i8; 16];
    for (slot, &b) in scales.iter_mut().zip(&block[192..208]) {
        *slot = b as i8;
    }
    let ql = &block[..128];
    let qh = &block[128..192];
    let mut q = [0i8; QK_K];
    for n in 0..2 {
        let ql = &ql[n * 64..n * 64 + 64];
        let qh = &qh[n * 32..n * 32 + 32];
        let base = n * 128;
        for l in 0..32 {
            q[base + l] = q6_quant(ql[l] & 0xF, qh[l] & 3);
            q[base + 32 + l] = q6_quant(ql[l + 32] & 0xF, (qh[l] >> 2) & 3);
            q[base + 64 + l] = q6_quant(ql[l] >> 4, (qh[l] >> 4) & 3);
            q[base + 96 + l] = q6_quant(ql[l + 32] >> 4, (qh[l] >> 6) & 3);
        }
    }
    (d, scales, q)
}

/// A 6-bit field stored unsigned, returned as the signed quant (`- 32`).
fn q6_quant(low: u8, high: u8) -> i8 {
    let bits = i32::from((low & 0xF) | ((high & 3) << 4));
    (bits - 32) as i8
}

/// `Q3_K`: 2 bits in `qs` and a high bit in `hmask`. A clear high bit means
/// the quant is `bits - 4` (`-4..-1`); a set bit keeps `0..3`.
fn unpack_q3_k(block: &[u8; Q3_K_BLOCK_BYTES]) -> (f32, [i8; 16], [i8; QK_K]) {
    let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let scales = q3_scales(&block[96..108]);
    let hmask = &block[..32];
    let qs = &block[32..96];
    let mut q = [0i8; QK_K];
    let mut y = 0;
    let mut mask: u8 = 1;
    for half in 0..2 {
        let bytes = &qs[half * 32..half * 32 + 32];
        let mut shift = 0;
        for _ in 0..4 {
            for l in 0..16 {
                q[y] = q3_quant(bytes[l] >> shift, hmask[l] & mask != 0);
                y += 1;
            }
            for l in 0..16 {
                q[y] = q3_quant(bytes[l + 16] >> shift, hmask[l + 16] & mask != 0);
                y += 1;
            }
            shift += 2;
            mask = mask.wrapping_shl(1);
        }
    }
    (d, scales, q)
}

fn q3_quant(bits: u8, high: bool) -> i8 {
    let mag = (bits & 3) as i8;
    if high { mag } else { mag - 4 }
}

/// `Q4_K`/`Q5_K` dot: `d * Σ scale·q·x - dmin * Σ min·bsum`, with each min
/// covering two groups of 16 activations (one group of 32 weights).
fn dot_q4_k(row: &[u8], xq: &[u8]) -> f32 {
    dot_k_mins::<Q4_K_BLOCK_BYTES>(row, xq, unpack_q4_k, q4_k_update)
}

fn dot_q5_k(row: &[u8], xq: &[u8]) -> f32 {
    dot_k_mins::<Q5_K_BLOCK_BYTES>(row, xq, unpack_q5_k, q5_k_update)
}

/// The NEON `Q4_K` running sum after one super-block, with `d` and `dmin`
/// already multiplied by the activation scale: `sumf -= dmin * sum_mins`,
/// then `sumf += d * isum`, each statement one fused operation.
pub(crate) fn q4_k_update(sum: f32, d: f32, dmin: f32, isum: i32, sum_mins: i32) -> f32 {
    d.mul_add(isum as f32, dmin.mul_add(-(sum_mins as f32), sum))
}

/// The NEON `Q5_K` running sum after one super-block:
/// `sumf += d * isum - dmin * sum_mins`, where clang fuses the first product
/// into the difference and adds the block's term to the sum once.
pub(crate) fn q5_k_update(sum: f32, d: f32, dmin: f32, isum: i32, sum_mins: i32) -> f32 {
    sum + d.mul_add(isum as f32, -(dmin * sum_mins as f32))
}

fn dot_k_mins<const N: usize>(
    row: &[u8],
    xq: &[u8],
    unpack: impl Fn(&[u8; N]) -> (f32, f32, [u8; 8], [u8; 8], [u8; QK_K]),
    update: fn(f32, f32, f32, i32, i32) -> f32,
) -> f32 {
    let mut sum = 0.0f32;
    let blocks = row.as_chunks::<N>().0;
    let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter().zip(xs) {
        let (d, dmin, scales, mins, q) = unpack(block);
        let (xd, xqs, bsums) = q8k_block(x);
        let mut isum = 0i32;
        let mut sum_mins = 0i32;
        for g in 0..8 {
            let mut partial = 0i32;
            for i in 0..32 {
                partial += i32::from(q[g * 32 + i]) * i32::from(xqs[g * 32 + i] as i8);
            }
            isum += partial * i32::from(scales[g]);
            sum_mins += i32::from(bsums[2 * g]) * i32::from(mins[g]);
            sum_mins += i32::from(bsums[2 * g + 1]) * i32::from(mins[g]);
        }
        sum = update(sum, xd * d, xd * dmin, isum, sum_mins);
    }
    sum
}

fn dot_q6_k(row: &[u8], xq: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let blocks = row.as_chunks::<Q6_K_BLOCK_BYTES>().0;
    let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter().zip(xs) {
        let (d, scales, q) = unpack_q6_k(block);
        let (xd, xqs, _) = q8k_block(x);
        let mut isum = 0i32;
        for g in 0..16 {
            let mut partial = 0i32;
            for i in 0..16 {
                partial += i32::from(q[g * 16 + i]) * i32::from(xqs[g * 16 + i] as i8);
            }
            isum += partial * i32::from(scales[g]);
        }
        // `sum += d_all * y[i].d * isum`: the second product is fused.
        sum = (xd * d).mul_add(isum as f32, sum);
    }
    sum
}

/// `dequantize_row_q2_K`: group `g` of 16 has scale `d * (sc & 15)` and
/// min `dmin * (sc >> 4)`; weight `= dl * q - ml`, fused.
fn dequantize_q2_k(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<Q2_K_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK_K>().0.iter_mut().zip(blocks) {
        let (d, dmin, q) = unpack_q2_k(block);
        for (k, o) in out.iter_mut().enumerate() {
            let sc = block[k / 16];
            let (dl, ml) = (d * f32::from(sc & 0xF), dmin * f32::from(sc >> 4));
            *o = dl.mul_add(f32::from(q[k]), -ml);
        }
    }
}

/// The scale, min scale and 256 two-bit quants of a `Q2_K` block. Weight
/// `k` is bits `2s..2s + 2` of byte `32 * (k / 128) + k % 32` of the
/// quants, where `s = (k % 128) / 32`.
pub(crate) fn unpack_q2_k(block: &[u8; Q2_K_BLOCK_BYTES]) -> (f32, f32, [u8; QK_K]) {
    let d = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    let mut q = [0u8; QK_K];
    for (k, q) in q.iter_mut().enumerate() {
        let byte = block[16 + 32 * (k / 128) + k % 32];
        *q = (byte >> (2 * ((k % 128) / 32))) & 3;
    }
    (d, dmin, q)
}

/// `ggml_vec_dot_q2_K_q8_K`, NEON path: `sum += dmin * sum_mins` with
/// `dmin = -y.d * dmin`, then `sum += d * isum`, each fused.
fn dot_q2_k(row: &[u8], xq: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let blocks = row.as_chunks::<Q2_K_BLOCK_BYTES>().0;
    let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter().zip(xs) {
        let (d, dmin, q) = unpack_q2_k(block);
        let (xd, xqs, bsums) = q8k_block(x);
        let mut isum = 0i32;
        let mut sum_mins = 0i32;
        for g in 0..16 {
            let mut partial = 0i32;
            for i in 16 * g..16 * g + 16 {
                partial += i32::from(q[i]) * i32::from(xqs[i] as i8);
            }
            isum += partial * i32::from(block[g] & 0xF);
            sum_mins += i32::from(block[g] >> 4) * i32::from(bsums[g]);
        }
        sum = (-xd * dmin).mul_add(sum_mins as f32, sum);
        sum = (xd * d).mul_add(isum as f32, sum);
    }
    sum
}

fn dot_q3_k(row: &[u8], xq: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    let blocks = row.as_chunks::<Q3_K_BLOCK_BYTES>().0;
    let xs = xq.as_chunks::<Q8_K_BLOCK_BYTES>().0;
    for (block, x) in blocks.iter().zip(xs) {
        let (d, scales, q) = unpack_q3_k(block);
        let (xd, xqs, _) = q8k_block(x);
        let mut isum = 0i32;
        for g in 0..16 {
            let scale = i32::from(scales[g]) - 32;
            let mut partial = 0i32;
            for i in 0..16 {
                partial += i32::from(q[g * 16 + i]) * i32::from(xqs[g * 16 + i] as i8);
            }
            isum += partial * scale;
        }
        // `sum += d * isum`, fused.
        sum = (xd * d).mul_add(isum as f32, sum);
    }
    sum
}

/// Dot product of two encoded vectors of the same element format.
/// Eight accumulators fix the summation order and let the compiler vectorize.
#[inline]
fn dot_by<const N: usize>(a: &[[u8; N]], b: &[[u8; N]], decode: impl Fn(&[u8; N]) -> f32) -> f32 {
    let mut acc = [0.0f32; 8];
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        acc[i % 8] += decode(x) * decode(y);
    }
    ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]))
}

/// A row of `cols` weights of type `ty` (a supported type; `cols` a multiple
/// of its block size) from a fixed generator, with finite scales: inputs for
/// a backend to check its kernels with when it starts.
#[cfg(target_os = "macos")]
pub(crate) fn sample_row(ty: GgmlType, cols: usize, seed: u32) -> Vec<u8> {
    let mut s = seed.wrapping_mul(2_654_435_761).max(1);
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        s
    };
    let values: Vec<f32> = (0..cols)
        .map(|_| (next() >> 8) as f32 / 8_388_608.0 - 1.0)
        .collect();
    match ty {
        GgmlType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        GgmlType::F16 => values
            .iter()
            .flat_map(|&v| f32_to_f16(v).to_le_bytes())
            .collect(),
        GgmlType::Bf16 => values
            .iter()
            .flat_map(|&v| f32_to_bf16(v).to_le_bytes())
            .collect(),
        GgmlType::Q8_0 => {
            let mut out = vec![0; cols / QK8_0 * Q8_0_BLOCK_BYTES];
            quantize_row_q8_0(&values, &mut out, Q8_0Inverse::NATIVE);
            out
        }
        _ => {
            // Random bytes, with small finite scales.
            let (n, per) = ty
                .layout()
                .map_or((0, 1), |l| (l.type_size as usize, l.block_size as usize));
            let mut row: Vec<u8> = (0..cols / per * n).map(|_| (next() >> 11) as u8).collect();
            for block in row.chunks_exact_mut(n.max(1)) {
                set_block_scales(ty, block, 0x2E66, 0x2A3D);
            }
            row
        }
    }
}

/// Overwrites the scales of one block of type `ty` (any quantized type but
/// `Q8_0`) with the f16 `scale` and, where the type has one, the f16 min
/// scale `min`, so that a block of random bytes decodes to finite values.
/// `MXFP4`'s power-of-two scale is taken from the exponent of `scale`.
#[cfg(any(test, target_os = "macos"))]
pub(crate) fn set_block_scales(ty: GgmlType, block: &mut [u8], scale: u16, min: u16) {
    let (d, m) = (scale.to_le_bytes(), min.to_le_bytes());
    match ty {
        GgmlType::Q4K | GgmlType::Q5K | GgmlType::Q4_1 | GgmlType::Q5_1 => {
            block[..2].copy_from_slice(&d);
            block[2..4].copy_from_slice(&m);
        }
        GgmlType::Q2K => {
            block[80..82].copy_from_slice(&d);
            block[82..84].copy_from_slice(&m);
        }
        GgmlType::Q6K => block[208..210].copy_from_slice(&d),
        GgmlType::Q3K => block[108..110].copy_from_slice(&d),
        GgmlType::Mxfp4 => block[0] = 110 + ((scale >> 10) & 0x1F) as u8,
        GgmlType::Iq1M => {
            for k in 0..4 {
                let at = 48 + 2 * k;
                block[at + 1] = (block[at + 1] & 0x0F) | ((((scale >> (4 * k)) & 0xF) as u8) << 4);
            }
        }
        _ => block[..2].copy_from_slice(&d),
    }
}

/// Deterministic inputs for the kernel tests. The llama.cpp reference values
/// the tests compare with were computed from the same generators.
#[cfg(test)]
pub(crate) mod test_data {
    use nibble_gguf::GgmlType;

    fn seed_state(seed: u32) -> u32 {
        seed.wrapping_mul(2_654_435_761).max(1)
    }

    /// xorshift32 values in [-1, 1).
    pub(crate) fn values(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed_state(seed);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as f32 / 8_388_608.0 - 1.0
            })
            .collect()
    }

    /// [`values`] times `scale`.
    pub(crate) fn scaled(n: usize, seed: u32, scale: f32) -> Vec<f32> {
        values(n, seed).into_iter().map(|v| v * scale).collect()
    }

    /// A K-quant (or `IQ4_XS`) row of `blocks` super-blocks of random bytes,
    /// except that block `i` has the f16 scale `scales[i % 3]` and, for
    /// `Q4_K`/`Q5_K`, the min scale `dmin`.
    pub(crate) fn k_row_with(
        ty: GgmlType,
        blocks: usize,
        seed: u32,
        scales: [u16; 3],
        dmin: u16,
    ) -> Vec<u8> {
        let n = ty.layout().unwrap().type_size as usize;
        let mut row = vec![0u8; n * blocks];
        let mut s = seed_state(seed);
        for (i, b) in row.iter_mut().enumerate() {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *b = (s >> (i % 24)) as u8;
        }
        for (i, block) in row.chunks_exact_mut(n).enumerate() {
            super::set_block_scales(ty, block, scales[i % 3], dmin);
        }
        row
    }

    /// [`k_row_with`] with scales whose products round (about 0.0999, 0 and
    /// -0.0187, min scale 0.0444), so a changed rounding order shows.
    pub(crate) fn k_row(ty: GgmlType, blocks: usize, seed: u32) -> Vec<u8> {
        k_row_with(ty, blocks, seed, [0x2E66, 0x0000, 0xA4CD], 0x2A3D)
    }

    /// 64-bit FNV-1a, to compare long outputs with a recorded reference.
    pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// The little-endian bytes of `x`.
    pub(crate) fn f32_bytes(x: &[f32]) -> Vec<u8> {
        x.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_data::{f32_bytes, fnv1a, k_row, k_row_with, scaled};
    use super::*;

    #[test]
    fn f16_decoding_is_exact() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative());
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8); // smallest subnormal
        assert_eq!(f16_to_f32(0x03FF), 6.097_555e-5); // largest subnormal
        assert_eq!(f16_to_f32(0x0400), 6.103_515_6e-5); // smallest normal
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xFC00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn f16_encoding_round_trips_and_rounds_to_even() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if !f.is_nan() {
                assert_eq!(f32_to_f16(f), h, "{h:#06x}");
            }
        }
        let one = 1.0f32;
        let ulp = 2.0f32.powi(-10);
        assert_eq!(f32_to_f16(one + ulp / 2.0), 0x3C00); // tie: down to even
        assert_eq!(f32_to_f16(one + ulp * 1.5), 0x3C02); // tie: up to even
        assert_eq!(f32_to_f16(one + ulp * 0.75), 0x3C01);
        assert_eq!(f32_to_f16(70000.0), 0x7C00);
        assert_eq!(f32_to_f16(1e-9), 0x0000);
        assert_eq!(f32_to_f16(4.0e-8), 0x0001); // nearest subnormal
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_conversions() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC040), -3.0);
        assert_eq!(bf16_to_f32(0x7F80), f32::INFINITY);
        for h in (0..=u16::MAX).step_by(7) {
            let f = bf16_to_f32(h);
            if !f.is_nan() {
                assert_eq!(f32_to_bf16(f), h);
            }
        }
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80); // tie: down to even
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82); // tie: up to even
    }

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

    #[test]
    fn dot_row_matches_f64_reference_within_rounding() {
        let n = 128;
        let w = values(n, 1);
        let x = values(n, 2);
        // Worst-case relative rounding of each operand, per format.
        for (ty, eps) in [
            (GgmlType::F32, 1e-6),
            (GgmlType::F16, 1e-3),
            (GgmlType::Bf16, 8e-3),
            (GgmlType::Q8_0, 1e-2),
        ] {
            // F16 rows are f16; the other weight types are encoded as their
            // dot type is.
            let row = if ty == GgmlType::F16 {
                w.iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect()
            } else {
                encoded(DotType::of(ty), &w)
            };
            let xq = encoded(DotType::of(ty), &x);
            let got = f64::from(dot_row(ty, &row, &xq));
            let exact: f64 = w
                .iter()
                .zip(&x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let scale: f64 = w
                .iter()
                .zip(&x)
                .map(|(&a, &b)| f64::from(a.abs()) * f64::from(b.abs()))
                .sum();
            assert!(
                (got - exact).abs() <= 2.0 * eps * scale,
                "{ty}: {got} vs {exact}"
            );
        }
    }

    #[test]
    fn q8_0_round_trip() {
        let mut block = vec![0u8; Q8_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x3800u16.to_le_bytes()); // d = 0.5
        block[2] = 4;
        block[3] = (-6i8) as u8;
        block[33] = 127;
        let mut out = vec![0.0; QK8_0];
        dequantize_row(GgmlType::Q8_0, &block, &mut out);
        assert_eq!((out[0], out[1], out[2], out[31]), (2.0, -3.0, 0.0, 63.5));

        let x: Vec<f32> = (0..32).map(|i| i as f32 - 16.0).collect();
        let q = encoded(DotType::Q8_0, &x);
        assert_eq!(q[2] as i8, -127); // amax maps to ±127
        dequantize_row(GgmlType::Q8_0, &q, &mut out);
        for (a, b) in out.iter().zip(&x) {
            assert!((a - b).abs() <= 16.0 / 127.0, "{a} vs {b}");
        }
    }

    /// `nearest_int` is round-half-to-even, including the negative side.
    #[test]
    fn nearest_int_rounds_ties_to_even() {
        for (x, want) in [
            (0.0, 0),
            (0.5, 0),
            (1.5, 2),
            (-0.5, 0),
            (-1.5, -2),
            (63.5, 64),
            (-127.0, -127),
            (127.0, 127),
            (31.75, 32),
        ] {
            assert_eq!(nearest_int(x), want, "{x}");
        }
    }

    #[test]
    fn q8_k_scale_follows_the_signed_extreme_and_records_group_sums() {
        let mut x = vec![0.0f32; QK_K];
        x[0] = 1.0;
        x[1] = -2.0;
        x[2] = 0.5;
        let q = encoded(DotType::Q8K, &x);
        // The largest magnitude is -2, so iscale = -127/-2 and d = 2/127.
        let d = f32::from_le_bytes(q[..4].try_into().unwrap());
        assert!((d - 2.0 / 127.0).abs() < 1e-7, "{d}");
        assert_eq!(q[4] as i8, nearest_int(63.5) as i8); // 1 * 63.5 -> 64
        assert_eq!(q[5] as i8, -127);
        assert_eq!(q[6] as i8, nearest_int(31.75) as i8);
        let mut sum = 0i16;
        for &b in &q[4..20] {
            sum += i16::from(b as i8);
        }
        assert_eq!(i16::from_le_bytes(q[260..262].try_into().unwrap()), sum);

        let zeros = encoded(DotType::Q8K, &[0.0; QK_K]);
        assert!(zeros.iter().all(|&b| b == 0));
        let mut negzero = vec![0.0f32; QK_K];
        negzero[3] = -0.0;
        assert!(encoded(DotType::Q8K, &negzero).iter().all(|&b| b == 0));
    }

    /// Packs 8 scales and 8 mins the way `quantize_row_q4_K_ref` does, so the
    /// reader can be checked against a known table rather than against itself.
    fn pack_k4_scales(scales: [u8; 8], mins: [u8; 8]) -> [u8; 12] {
        let mut q = [0u8; 12];
        for j in 0..8 {
            let (ls, lm) = (scales[j] & 63, mins[j] & 63);
            if j < 4 {
                q[j] = ls;
                q[j + 4] = lm;
            } else {
                q[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                q[j - 4] |= (ls >> 4) << 6;
                q[j] |= (lm >> 4) << 6;
            }
        }
        q
    }

    #[test]
    fn k4_scales_round_trip_every_group() {
        let scales = [1u8, 63, 0, 17, 32, 48, 7, 21];
        let mins = [0u8, 5, 63, 8, 16, 3, 40, 9];
        let packed = pack_k4_scales(scales, mins);
        for j in 0..8 {
            assert_eq!(scale_min_k4(&packed, j), (scales[j], mins[j]), "group {j}");
        }
    }

    #[test]
    fn q4_k_dequant_matches_the_documented_formula() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1
        block[2..4].copy_from_slice(&0x3800u16.to_le_bytes()); // dmin = 0.5
        block[4..16].copy_from_slice(&pack_k4_scales(
            [1, 2, 3, 4, 0, 0, 0, 0],
            [0, 1, 0, 2, 0, 0, 0, 0],
        ));
        for b in &mut block[16..48] {
            *b = 0x21; // low nibble 1, high nibble 2, covering the first 64 weights twice
        }
        let mut out = vec![0.0; QK_K];
        dequantize_row(GgmlType::Q4K, &block, &mut out);
        // Group 0 (weights 0..32): scale 1, min 0, quant 1 -> 1*1*1 - 0.5*0 = 1
        assert_eq!(out[0], 1.0);
        assert_eq!(out[31], 1.0);
        // Group 1 (32..64): scale 2, min 1, quant 2 -> 2*2 - 0.5*1 = 3.5
        assert_eq!(out[32], 3.5);
        assert_eq!(out[63], 3.5);
        // Groups 4..7 have scale 0 and min 0.
        assert_eq!(out[128], 0.0);
        assert_eq!(out[255], 0.0);
    }

    #[test]
    fn q5_k_high_bit_adds_sixteen() {
        let mut block = vec![0u8; Q5_K_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        block[4..16].copy_from_slice(&pack_k4_scales([1; 8], [0; 8]));
        // qh is at [16..48], low nibbles at [48..]. Set quant 0's low nibble
        // to 3 and its high bit (qh[0] bit 0).
        block[48] = 0x03;
        block[16] = 0x01;
        let mut out = vec![0.0; QK_K];
        dequantize_row(GgmlType::Q5K, &block, &mut out);
        assert_eq!(out[0], 19.0); // 1 * (3 + 16) - 0
        assert_eq!(out[1], 0.0);
    }

    #[test]
    fn q6_k_quant_is_biased_by_32() {
        let mut block = vec![0u8; Q6_K_BLOCK_BYTES];
        block[208..210].copy_from_slice(&0x3C00u16.to_le_bytes()); // d = 1
        block[192] = 2; // scale of the first 16
        block[0] = 0x05; // low nibble of element 0, and (>> 4) of element 64
        block[128] = 0x01; // high bits of element 0 are 1 -> quant = (5 | 16) - 32 = -11
        let mut out = vec![0.0; QK_K];
        dequantize_row(GgmlType::Q6K, &block, &mut out);
        assert_eq!(out[0], 2.0 * -11.0);
        // A zero code is -32, times scale 0 for every later group.
        assert_eq!(out[16], 0.0);
    }

    #[test]
    fn q3_k_clear_high_bit_subtracts_four() {
        let mut block = vec![0u8; Q3_K_BLOCK_BYTES];
        block[108..110].copy_from_slice(&0x3C00u16.to_le_bytes());
        // Scale 0 is 33 (so `scale - 32` is 1): its low nibble is byte 0 of
        // the 12-byte table at offset 96 and its top two bits are bits 0..1
        // of byte 8. Only element 0 is checked.
        let mut raw = [0u8; 12];
        raw[0] = 33 & 0xF;
        raw[8] = (33 >> 4) & 3;
        block[96..108].copy_from_slice(&raw);
        // qs at offset 32. Element 0's low 2 bits are 1, high bit clear.
        block[32] = 0x01;
        let mut out = vec![0.0; QK_K];
        dequantize_row(GgmlType::Q3K, &block, &mut out);
        assert_eq!(q3_scales(&raw)[0], 33);
        assert_eq!(out[0], 1.0 * (1.0 - 4.0));
        // Set the high bit and the quant stays 1.
        block[0] = 0x01;
        dequantize_row(GgmlType::Q3K, &block, &mut out);
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn k_quant_dots_match_dequantized_f64() {
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for ty in [GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q5K, GgmlType::Q6K] {
            let n = ty.layout().unwrap().type_size as usize;
            for blocks in [1usize, 3] {
                let mut row = vec![0u8; n * blocks];
                for (i, b) in row.iter_mut().enumerate() {
                    *b = (next() >> (i % 24)) as u8;
                }
                // Finite, modest super-block scales, including a zero block
                // and a negative f16 scale. Quants stay whatever the bytes say,
                // which covers 0, the high codes and (for Q6) negative scales.
                for (i, block) in row.chunks_exact_mut(n).enumerate() {
                    let scale: u16 = match i % 3 {
                        0 => 0x3C00, // 1
                        1 => 0x0000, // 0
                        _ => 0xBC00, // -1
                    };
                    match ty {
                        GgmlType::Q4K | GgmlType::Q5K => {
                            block[..2].copy_from_slice(&scale.to_le_bytes());
                            block[2..4].copy_from_slice(&0x3800u16.to_le_bytes());
                        }
                        GgmlType::Q6K => block[208..210].copy_from_slice(&scale.to_le_bytes()),
                        GgmlType::Q3K => block[108..110].copy_from_slice(&scale.to_le_bytes()),
                        _ => unreachable!(),
                    }
                }
                let x = values(QK_K * blocks, 9 + n as u32);
                // A -128 activation quant, which the dot must not drop.
                let mut x_forced = x.clone();
                x_forced[0] = 1.0;
                x_forced[1] = -1.0;
                let mut xq = encoded(DotType::Q8K, &x_forced);
                // A converter may emit -128. The group sum has to follow it,
                // because Q4_K/Q5_K subtract mins through `bsums`.
                xq[4] = (-128i8) as u8;
                let sum: i16 = xq[4..20].iter().map(|&q| i16::from(q as i8)).sum();
                xq[260..262].copy_from_slice(&sum.to_le_bytes());

                let mut weight = vec![0.0; QK_K * blocks];
                dequantize_row(ty, &row, &mut weight);
                let mut act = vec![0.0; QK_K * blocks];
                for (block, out) in xq
                    .as_chunks::<Q8_K_BLOCK_BYTES>()
                    .0
                    .iter()
                    .zip(act.chunks_mut(QK_K))
                {
                    let d = f32::from_le_bytes(block[..4].try_into().unwrap());
                    for (o, &q) in out.iter_mut().zip(&block[4..4 + QK_K]) {
                        *o = d * f32::from(q as i8);
                    }
                }
                let got = f64::from(dot_row(ty, &row, &xq));
                let exact: f64 = weight
                    .iter()
                    .zip(&act)
                    .map(|(&w, &a)| f64::from(w) * f64::from(a))
                    .sum();
                let scale: f64 = weight
                    .iter()
                    .zip(&act)
                    .map(|(&w, &a)| f64::from(w.abs()) * f64::from(a.abs()))
                    .sum();
                assert!(
                    (got - exact).abs() <= 1e-5 * scale.max(1.0),
                    "{ty}: {got} vs {exact}"
                );
            }
        }
    }

    #[test]
    fn q8_k_dot_handles_subnormal_f16_scales_and_empty_rows() {
        let mut block = vec![0u8; Q4_K_BLOCK_BYTES];
        block[..2].copy_from_slice(&0x0001u16.to_le_bytes()); // smallest f16 subnormal
        block[2..4].copy_from_slice(&0x8001u16.to_le_bytes());
        let x = encoded(DotType::Q8K, &vec![0.25; QK_K]);
        let got = dot_row(GgmlType::Q4K, &block, &x);
        assert!(got.is_finite(), "{got}");
        assert_eq!(dot_row(GgmlType::Q6K, &[], &[]), 0.0);
    }

    // The tests below compare with values from llama.cpp: gguf-py 2.5.5
    // (`gguf.quants`), and ggml's C functions in the b12359 macOS arm64
    // release (`ggml_get_type_traits(type)->to_float` for dequantization,
    // `ggml_get_type_traits_cpu(type)->from_float` / `vec_dot` otherwise),
    // run on the inputs that `test_data` generates.

    /// With power-of-two scales every product is exact, so the fused and
    /// unfused forms agree and gguf-py (which does not fuse) checks the bit
    /// layout of each format; ggml's C functions give the same bytes.
    #[test]
    fn k_quant_dequantization_matches_gguf_py() {
        #[rustfmt::skip]
        let cases = [
            (GgmlType::Q3K, 0x0E2D_FE0D_C0B0_5916, [-0.0, 7.0, -0.0, -0.0, -0.0, -0.0, 1.3125, 1.75]),
            (GgmlType::Q4K, 0xEC61_A9BD_4F64_45D6, [6.90625, -0.15625, 39.03125, 38.4375, -1.625, -1.21875, -19.25, -35.90625]),
            (GgmlType::Q5K, 0xDB61_7551_1BAD_E514, [3.40625, -0.15625, 20.28125, 14.25, -1.9375, -1.8125, -74.28125, -61.5]),
            (GgmlType::Q6K, 0xDC67_99CE_91C9_2F31, [123.375, 25.1875, -38.0, 59.8125, 0.0, 0.0, 65.25, 36.0]),
        ];
        for (ty, hash, samples) in cases {
            // Scales 2^-4, 0 and -2^-4; min scale 2^-5.
            let row = k_row_with(ty, 3, 30, [0x2C00, 0x0000, 0xAC00], 0x2800);
            let mut out = vec![0.0; 3 * QK_K];
            dequantize_row(ty, &row, &mut out);
            for (&i, want) in [0, 37, 100, 255, 300, 511, 600, 767].iter().zip(samples) {
                assert_eq!(out[i].to_bits(), f32::to_bits(want), "{ty}[{i}]");
            }
            assert_eq!(fnv1a(&f32_bytes(&out)), hash, "{ty}");
        }
    }

    /// Scales whose products round: `Q4_K`/`Q5_K` dequantization is a fused
    /// multiply-add in ggml (`d1 * q - m1`).
    #[test]
    fn k_quant_dequantization_matches_ggml_rounding() {
        for (ty, hash) in [
            (GgmlType::Q3K, 0x703A_7961_0AD6_0122),
            (GgmlType::Q4K, 0x042C_0CB2_2253_75B9),
            (GgmlType::Q5K, 0x87CF_65B7_B43B_1BC8),
            (GgmlType::Q6K, 0x5557_5A22_77A3_5EC9),
        ] {
            let mut out = vec![0.0; 3 * QK_K];
            dequantize_row(ty, &k_row(ty, 3, 20), &mut out);
            assert_eq!(fnv1a(&f32_bytes(&out)), hash, "{ty}");
        }
        let mut q = vec![0; 3 * Q8_0_BLOCK_BYTES];
        quantize_row_q8_0(&test_data::values(96, 21), &mut q, Q8_0Inverse::OfScale);
        let mut out = vec![0.0; 96];
        dequantize_row(GgmlType::Q8_0, &q, &mut out);
        assert_eq!(fnv1a(&f32_bytes(&out)), 0x7385_D32D_EEE4_BDC1);
    }

    /// The scalar K-quant dots perform the float operations of ggml's NEON
    /// kernels, so they give its results exactly, on any CPU.
    #[test]
    fn k_quant_dots_match_ggml_neon() {
        let xq = encoded(DotType::Q8K, &test_data::values(3 * QK_K, 3));
        for (ty, want) in [
            (
                GgmlType::Q3K,
                [0x41BA_C4D6, 0x415B_B436, 0x4117_A4D4, 0xC184_1DB7],
            ),
            (
                GgmlType::Q4K,
                [0x4338_F907, 0xC34D_03FD, 0x42E3_1C27, 0x4208_D8D1],
            ),
            (
                GgmlType::Q5K,
                [0x43AF_F341, 0xC396_F378, 0x438B_9C1A, 0xC2B1_C5B2],
            ),
            (
                GgmlType::Q6K,
                [0x4542_9876, 0x4466_46AC, 0xC422_704C, 0x4463_55CC],
            ),
        ] {
            for (r, want) in want.into_iter().enumerate() {
                let got = dot_row(ty, &k_row(ty, 3, 10 + r as u32), &xq);
                assert_eq!(got.to_bits(), want, "{ty} row {r}: {got}");
            }
        }
    }

    /// `IQ4_XS` against ggml's `to_float` and NEON `vec_dot` (gguf-py has no
    /// `IQ4_XS` quantizer to compare with): the dequantized rows with exact
    /// and rounding scales, and four dots with a `Q8_K` activation.
    /// The formats added after the K-quants against llama.cpp b12722's
    /// `libggml-cpu` on this generator's rows (`k_row`, 25 blocks of 32 or 3
    /// of 256 weights): the hash of `to_float` of seed 20's row, the hash of
    /// the encoded activation `values(n, 3)`, and the NEON `vec_dot` of seeds
    /// 10..14 with it, bit for bit.
    #[test]
    fn other_formats_match_ggml() {
        #[rustfmt::skip]
        let cases: [(GgmlType, usize, u64, u64, [u32; 4]); 14] = [
            (GgmlType::Q4_0, 800, 0x1ee9d9faac8153da, 0x5e0b1604af85b5d4, [0x40e71b06, 0xc074d6fc, 0xc07dd5d5, 0x408233b7]),
            (GgmlType::Q4_1, 800, 0xeb924466f8bd6817, 0x3e3d3596683bccaf, [0xc090eb64, 0x4084df68, 0xc0da5d3f, 0xc1226862]),
            (GgmlType::Q5_0, 800, 0x032b143f6889f3fc, 0x5e0b1604af85b5d4, [0x403aa6ee, 0xbfbae96a, 0xbfefe64e, 0x410a971e]),
            (GgmlType::Q5_1, 800, 0x6bddfd4091f603de, 0x3e3d3596683bccaf, [0xbe396580, 0xc10008d8, 0xc187c78f, 0xc16c6560]),
            (GgmlType::Iq4Nl, 800, 0x997009f0dd224f23, 0x5e0b1604af85b5d4, [0x42dd09ae, 0xc23cb81b, 0xc27fe315, 0x4236f4e6]),
            (GgmlType::Mxfp4, 800, 0xb1261f274245ecd2, 0x5e0b1604af85b5d4, [0xbd696ac0, 0xbf00ee21, 0x3ed543e2, 0xbf5dd273]),
            (GgmlType::Q2K, 768, 0x9d9d7567cc11cf16, 0xc1a464cbf99c8d6a, [0x4139f5ff, 0xc1980745, 0x40ef88e8, 0x41476302]),
            (GgmlType::Iq2Xxs, 768, 0x9d3c5344dfb50178, 0xc1a464cbf99c8d6a, [0xc1a70cfd, 0x42a65147, 0x3f8c2871, 0xc28b6a4a]),
            (GgmlType::Iq2Xs, 768, 0x9e5966cc3775492b, 0xc1a464cbf99c8d6a, [0x41729145, 0xc28283f2, 0xc24f204b, 0x41a9e8e8]),
            (GgmlType::Iq2S, 768, 0x7a627ae0835da2f1, 0xc1a464cbf99c8d6a, [0x4286f4e5, 0x42244074, 0xc13d5391, 0xc24e8e19]),
            (GgmlType::Iq3Xxs, 768, 0x870b526689e7a566, 0xc1a464cbf99c8d6a, [0xc345d585, 0xc28dd079, 0xc30f134a, 0x430082fb]),
            (GgmlType::Iq3S, 768, 0x966c8c20d9dbae6b, 0xc1a464cbf99c8d6a, [0xc33c1d55, 0x40278b27, 0xc2b6606b, 0x431c334e]),
            (GgmlType::Iq1S, 768, 0xce25648b91356315, 0xc1a464cbf99c8d6a, [0x415876cf, 0xc06bc0eb, 0x40121ce0, 0x411472ab]),
            (GgmlType::Iq1M, 768, 0x36896040413b36af, 0xc1a464cbf99c8d6a, [0xc0c277ee, 0x403b7f66, 0x41040252, 0xc12b73b4]),
        ];
        for (ty, n, deq_hash, xq_hash, dots) in cases {
            let blocks = n / ty.layout().unwrap().block_size as usize;
            let mut out = vec![0.0; n];
            dequantize_row(ty, &k_row(ty, blocks, 20), &mut out);
            assert_eq!(fnv1a(&f32_bytes(&out)), deq_hash, "{ty} to_float");
            // The activation as ggml's NEON encoders write it, on any target.
            let x = test_data::values(n, 3);
            let dt = DotType::of(ty);
            let mut xq = vec![0; dt.encoded_len(n)];
            match dt {
                DotType::Q8_0 => quantize_row_q8_0(&x, &mut xq, Q8_0Inverse::OfScale),
                DotType::Q8_1 => blocks32::quantize_row_q8_1(&x, &mut xq, Q8_0Inverse::OfScale),
                _ => encode_activation(dt, &x, &mut xq),
            }
            assert_eq!(fnv1a(&xq), xq_hash, "{ty} activation");
            for (r, want) in dots.into_iter().enumerate() {
                let got = dot_row(ty, &k_row(ty, blocks, 10 + r as u32), &xq);
                assert_eq!(
                    got.to_bits(),
                    want,
                    "{ty} row {r}: {got} vs {}",
                    f32::from_bits(want)
                );
            }
        }
    }

    #[test]
    fn iq4_xs_matches_ggml() {
        let exact = k_row_with(GgmlType::Iq4Xs, 3, 30, [0x2C00, 0x0000, 0xAC00], 0x2800);
        let samples = [-42.75, -178.0, 28.25, 17.1875, -0.0, -0.0, 190.5, 111.125];
        for (row, hash) in [
            (exact, 0xF139_B264_6FF0_FBD6),
            (k_row(GgmlType::Iq4Xs, 3, 20), 0xD5F9_178C_AEEA_EFBD),
        ] {
            let mut out = vec![0.0; 3 * QK_K];
            dequantize_row(GgmlType::Iq4Xs, &row, &mut out);
            assert_eq!(fnv1a(&f32_bytes(&out)), hash);
            if hash == 0xF139_B264_6FF0_FBD6 {
                for (&i, want) in [0, 37, 100, 255, 300, 511, 600, 767].iter().zip(samples) {
                    assert_eq!(out[i].to_bits(), f32::to_bits(want), "[{i}]");
                }
            }
        }
        let xq = encoded(DotType::Q8K, &test_data::values(3 * QK_K, 3));
        for (r, want) in [0x44FE_12B0u32, 0xC49D_E191, 0xC49F_43E0, 0x44E6_DA04]
            .into_iter()
            .enumerate()
        {
            let got = dot_row(
                GgmlType::Iq4Xs,
                &k_row(GgmlType::Iq4Xs, 3, 10 + r as u32),
                &xq,
            );
            assert_eq!(got.to_bits(), want, "row {r}: {got}");
        }
    }

    #[test]
    fn q8_k_encoding_matches_ggml() {
        let q = encoded(DotType::Q8K, &scaled(64 * QK_K, 6, 5.0));
        assert_eq!(fnv1a(&q), 0xBE4C_C380_76BC_E7A5);
    }

    #[test]
    fn q8_0_encoding_matches_gguf_py_and_ggml() {
        // gguf-py rounds ties away from zero, ggml's SIMD kernels to even;
        // these inputs have no ties.
        let x = scaled(64 * QK8_0, 7, 2.0);
        let mut q = vec![0; 64 * Q8_0_BLOCK_BYTES];
        quantize_row_q8_0(&x, &mut q, Q8_0Inverse::OfScale);
        assert_eq!(fnv1a(&q), 0xCD94_EECB_25B8_A3D4);
        // ggml's NEON `quantize_row_q8_0`.
        let x = scaled(4096 * QK8_0, 5, 3.0);
        let mut q = vec![0; 4096 * Q8_0_BLOCK_BYTES];
        quantize_row_q8_0(&x, &mut q, Q8_0Inverse::OfScale);
        assert_eq!(fnv1a(&q), 0x71C3_A875_DC93_5F54);
    }

    /// Blocks where `1 / (amax / 127)` and `127 / amax` differ by an ulp and
    /// an element sits that close to a rounding boundary. ggml's NEON kernel
    /// writes 2 and 3; the `127 / amax` form (its AVX2 kernel) writes 1 and 2.
    #[test]
    fn q8_0_inverse_forms_round_boundary_values_as_ggml() {
        for (v, of_scale, direct) in [(0x3CCA_8FD5u32, 2i8, 1i8), (0x3D28_CD32, 3, 2)] {
            let mut x = [0.0f32; QK8_0];
            x[0] = f32::from_bits(0x4005_FC79);
            x[1] = f32::from_bits(v);
            x[2] = -f32::from_bits(v);
            for (inverse, want) in [
                (Q8_0Inverse::OfScale, of_scale),
                (Q8_0Inverse::Direct, direct),
            ] {
                let mut q = [0u8; Q8_0_BLOCK_BYTES];
                quantize_row_q8_0(&x, &mut q, inverse);
                assert_eq!(u16::from_le_bytes([q[0], q[1]]), 0x2438, "{inverse:?}");
                let got = [q[2] as i8, q[3] as i8, q[4] as i8];
                assert_eq!(got, [127, want, -want], "{inverse:?}, v = {v:#x}");
            }
        }
    }
}
