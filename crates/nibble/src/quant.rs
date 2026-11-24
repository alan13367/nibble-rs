//! Scalar reference implementations of the weight formats.
//!
//! As in ggml's CPU backend, a dot product between a weight row and an
//! activation vector first converts the activation to the weight type's
//! "dot type" (`vec_dot_type`): F16 weights use F16 activations, BF16 use BF16,
//! and Q8_0 weights use Q8_0-quantized activations with an integer dot product.
//! Matching this keeps greedy output in step with llama.cpp, and it is the
//! form the SIMD kernels compute in.
//!
//! These functions are the correctness oracle for every SIMD and GPU kernel,
//! so they favor clarity over speed.

use nibble_gguf::GgmlType;

/// Elements per Q8_0 block.
pub(crate) const QK8_0: usize = 32;
/// Bytes per Q8_0 block: an f16 scale and 32 signed bytes.
pub(crate) const Q8_0_BLOCK_BYTES: usize = 2 + QK8_0;

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
pub(crate) const SUPPORTED: &[GgmlType] =
    &[GgmlType::F32, GgmlType::F16, GgmlType::Bf16, GgmlType::Q8_0];

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
        other => unreachable!("dequantize_row called with unsupported type {other}"),
    }
}

/// Encodes the activation `x` in the dot type of weight type `ty`, into `out`.
pub(crate) fn quantize_activation(ty: GgmlType, x: &[f32], out: &mut Vec<u8>) {
    out.clear();
    match ty {
        GgmlType::F32 => out.extend(x.iter().flat_map(|v| v.to_le_bytes())),
        GgmlType::F16 => out.extend(x.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes())),
        GgmlType::Bf16 => out.extend(x.iter().flat_map(|&v| f32_to_bf16(v).to_le_bytes())),
        GgmlType::Q8_0 => {
            for block in x.as_chunks::<QK8_0>().0 {
                // As ggml's AVX2/NEON quantize_row_q8_0: d = amax/127, q = round_even(x * 127/amax).
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = amax / 127.0;
                let id = if amax != 0.0 { 127.0 / amax } else { 0.0 };
                out.extend(f32_to_f16(d).to_le_bytes());
                out.extend(
                    block
                        .iter()
                        .map(|&v| (v * id).round_ties_even() as i8 as u8),
                );
            }
        }
        other => unreachable!("quantize_activation called with unsupported type {other}"),
    }
}

/// Dot product of a weight row with an activation encoded by [`quantize_activation`].
#[must_use]
pub(crate) fn dot_row(ty: GgmlType, row: &[u8], xq: &[u8]) -> f32 {
    match ty {
        GgmlType::F32 => dot_by(row.as_chunks::<4>().0, xq.as_chunks::<4>().0, |b| {
            f32::from_le_bytes(*b)
        }),
        GgmlType::F16 => dot_by(row.as_chunks::<2>().0, xq.as_chunks::<2>().0, |b| {
            f16_to_f32(u16::from_le_bytes(*b))
        }),
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
                sum += sumi as f32 * (dw * dx);
            }
            sum
        }
        other => unreachable!("dot_row called with unsupported type {other}"),
    }
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

#[cfg(test)]
mod tests {
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
            // Weight rows use the same encoding as activations of their type.
            let mut row = Vec::new();
            quantize_activation(ty, &w, &mut row);
            let mut xq = Vec::new();
            quantize_activation(ty, &x, &mut xq);
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
        let mut q = Vec::new();
        quantize_activation(GgmlType::Q8_0, &x, &mut q);
        assert_eq!(q[2] as i8, -127); // amax maps to ±127
        dequantize_row(GgmlType::Q8_0, &q, &mut out);
        for (a, b) in out.iter().zip(&x) {
            assert!((a - b).abs() <= 16.0 / 127.0, "{a} vs {b}");
        }
    }
}
