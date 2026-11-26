//! The formats with blocks of 32 weights and one scale: `Q4_0`, `Q4_1`,
//! `Q5_0`, `Q5_1`, `IQ4_NL` and `MXFP4`, and the `Q8_1` activation that
//! `Q4_1` and `Q5_1` meet.
//!
//! Layouts are ggml's `block_*` (`ggml-common.h`): byte `j` of a block's 16
//! bytes of nibbles holds weight `j` in its low nibble and weight `j + 16`
//! in its high nibble; `Q5_*` keep the fifth bit of weight `j` in bit `j` of
//! a little-endian `u32`. Dequantization follows `dequantize_row_*`. The dots
//! follow the NEON `ggml_vec_dot_*` (`arch/arm/quants.c`), which take blocks
//! in pairs: the even block's four `vdotq_s32` lanes go to one float vector
//! and the odd block's to another (`sumv += lanes * d`, fused per lane), the
//! two are reduced pairwise and added, and an odd last block takes the
//! scalar tail. The integer lane sums are exact, so only that float order
//! matters, and [`dot_row`] repeats it.

use nibble_gguf::GgmlType;

use super::{KVALUES_IQ4NL, Q8_0_BLOCK_BYTES, Q8_0Inverse, QK8_0, f16_to_f32, f32_to_f16};

/// Bytes per `Q8_1` block: f16 scale `d`, f16 `s = d * sum(qs)`, 32 quants.
pub(crate) const Q8_1_BLOCK_BYTES: usize = 2 + 2 + QK8_0;

/// The 16 values an `MXFP4` nibble stands for: twice the E2M1 value
/// (`kvalues_mxfp4`), so the block scale is halved.
pub(crate) const KVALUES_MXFP4: [i8; 16] =
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Bytes per block of each format.
pub(crate) const fn block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Q4_0 | GgmlType::Iq4Nl => 2 + 16,
        GgmlType::Q4_1 => 2 + 2 + 16,
        GgmlType::Q5_0 => 2 + 4 + 16,
        GgmlType::Q5_1 => 2 + 2 + 4 + 16,
        GgmlType::Mxfp4 => 1 + 16,
        _ => 0,
    }
}

/// `GGML_E8M0_TO_FP32_HALF`: `2^(e - 128)`, the `MXFP4` block scale halved.
pub(crate) fn e8m0_half(e: u8) -> f32 {
    let bits = if e < 2 {
        0x0020_0000u32 << e
    } else {
        u32::from(e - 1) << 23
    };
    f32::from_bits(bits)
}

fn f16_at(b: &[u8], at: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[at], b[at + 1]]))
}

/// The fifth bits of a `Q5_*` block, bit `j` for weight `j`.
fn high_bits(block: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]])
}

/// The 32 weights of a block as the NEON kernel widens them to signed
/// bytes (before any scale), with its scale `d` and, for `Q4_1`/`Q5_1`, its
/// minimum `m`.
pub(crate) fn unpack(ty: GgmlType, block: &[u8]) -> ([i8; 32], f32, f32) {
    let mut q = [0i8; 32];
    let (nibbles, d, m, high) = match ty {
        GgmlType::Q4_0 | GgmlType::Iq4Nl => (&block[2..18], f16_at(block, 0), 0.0, None),
        GgmlType::Q4_1 => (&block[4..20], f16_at(block, 0), f16_at(block, 2), None),
        GgmlType::Q5_0 => (
            &block[6..22],
            f16_at(block, 0),
            0.0,
            Some(high_bits(block, 2)),
        ),
        GgmlType::Q5_1 => (
            &block[8..24],
            f16_at(block, 0),
            f16_at(block, 2),
            Some(high_bits(block, 4)),
        ),
        GgmlType::Mxfp4 => (&block[1..17], e8m0_half(block[0]), 0.0, None),
        other => unreachable!("{other} has no 32-weight blocks"),
    };
    for (j, &byte) in nibbles.iter().enumerate() {
        let (lo, hi) = (byte & 0xF, byte >> 4);
        let (lo, hi) = match ty {
            GgmlType::Q4_0 => (lo as i8 - 8, hi as i8 - 8),
            GgmlType::Q4_1 => (lo as i8, hi as i8),
            GgmlType::Iq4Nl => (
                KVALUES_IQ4NL[usize::from(lo)],
                KVALUES_IQ4NL[usize::from(hi)],
            ),
            GgmlType::Mxfp4 => (
                KVALUES_MXFP4[usize::from(lo)],
                KVALUES_MXFP4[usize::from(hi)],
            ),
            _ => {
                let h = high.unwrap_or(0);
                let lo = lo | (((h >> j) & 1) as u8) << 4;
                let hi = hi | (((h >> (j + 16)) & 1) as u8) << 4;
                if ty == GgmlType::Q5_0 {
                    (lo as i8 - 16, hi as i8 - 16)
                } else {
                    (lo as i8, hi as i8)
                }
            }
        };
        q[j] = lo;
        q[j + 16] = hi;
    }
    (q, d, m)
}

/// `dequantize_row_*`: `q * d`, or `q * d + m` (fused) with a minimum.
pub(crate) fn dequantize(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    let n = block_bytes(ty);
    for (out, block) in out
        .as_chunks_mut::<QK8_0>()
        .0
        .iter_mut()
        .zip(row.chunks_exact(n))
    {
        let (q, d, m) = unpack(ty, block);
        for (o, &q) in out.iter_mut().zip(&q) {
            *o = match ty {
                GgmlType::Q4_1 | GgmlType::Q5_1 => f32::from(q).mul_add(d, m),
                GgmlType::Iq4Nl => d * f32::from(q),
                _ => f32::from(q) * d,
            };
        }
    }
}

/// `quantize_row_q8_1`: [`super::quantize_row_q8_0`]'s quants and scale,
/// plus `s = d * sum(q)` stored as f16, from the f32 `d = amax / 127`.
pub(crate) fn quantize_row_q8_1(x: &[f32], out: &mut [u8], inverse: Q8_0Inverse) {
    let blocks = out.as_chunks_mut::<Q8_1_BLOCK_BYTES>().0;
    for (o, block) in blocks.iter_mut().zip(x.as_chunks::<QK8_0>().0) {
        let mut q8 = [0u8; Q8_0_BLOCK_BYTES];
        super::quantize_row_q8_0(block, &mut q8, inverse);
        let d = block.iter().fold(0.0f32, |m, v| m.max(v.abs())) / 127.0;
        let sum: i32 = q8[2..].iter().map(|&q| i32::from(q as i8)).sum();
        o[..2].copy_from_slice(&q8[..2]);
        o[2..4].copy_from_slice(&f32_to_f16(d * sum as f32).to_le_bytes());
        o[4..].copy_from_slice(&q8[2..]);
    }
}

/// The four `vdotq_s32` lanes of a block's dot: lane `l` sums weights
/// `4l..4l + 4` and `16 + 4l..16 + 4l + 4`.
pub(crate) fn lanes(q: &[i8; 32], y: &[u8]) -> [i32; 4] {
    let mut lanes = [0i32; 4];
    for (l, lane) in lanes.iter_mut().enumerate() {
        for k in 4 * l..4 * l + 4 {
            *lane += i32::from(q[k]) * i32::from(y[k] as i8);
            *lane += i32::from(q[k + 16]) * i32::from(y[k + 16] as i8);
        }
    }
    lanes
}

/// `vaddvq_f32`: `(v0 + v1) + (v2 + v3)`.
pub(crate) fn hsum(v: [f32; 4]) -> f32 {
    (v[0] + v[1]) + (v[2] + v[3])
}

/// Dot product of a row with an activation encoded as `Q8_0` (or `Q8_1`
/// for `Q4_1`/`Q5_1`).
pub(crate) fn dot_row(ty: GgmlType, row: &[u8], xq: &[u8]) -> f32 {
    let n = block_bytes(ty);
    let with_min = matches!(ty, GgmlType::Q4_1 | GgmlType::Q5_1);
    let xn = if with_min {
        Q8_1_BLOCK_BYTES
    } else {
        Q8_0_BLOCK_BYTES
    };
    let nb = row.len() / n;
    let w = |i: usize| &row[i * n..(i + 1) * n];
    let x = |i: usize| &xq[i * xn..(i + 1) * xn];
    // The activation's scale, `s` (with a minimum) and quants.
    let xs = |i: usize| -> (f32, f32, &[u8]) {
        let b = x(i);
        if with_min {
            (f16_at(b, 0), f16_at(b, 2), &b[4..])
        } else {
            (f16_at(b, 0), 0.0, &b[2..])
        }
    };
    let isum = |q: &[i8; 32], y: &[u8]| lanes(q, y).iter().sum::<i32>();

    let mut sumf = 0.0f32;
    let mut ib = 0;
    match ty {
        GgmlType::Iq4Nl | GgmlType::Mxfp4 => {
            // `sumf += d0 * y0 * isum0 + d1 * y1 * isum1`: the first product
            // is fused into the sum of the two.
            while ib + 1 < nb {
                let (q0, d0, _) = unpack(ty, w(ib));
                let (q1, d1, _) = unpack(ty, w(ib + 1));
                let (y0, _, yq0) = xs(ib);
                let (y1, _, yq1) = xs(ib + 1);
                let (s0, s1) = (d0 * y0, d1 * y1);
                sumf += s0.mul_add(isum(&q0, yq0) as f32, s1 * isum(&q1, yq1) as f32);
                ib += 2;
            }
            for ib in ib..nb {
                let (q, d, _) = unpack(ty, w(ib));
                let (y, _, yq) = xs(ib);
                sumf = (y * d).mul_add(isum(&q, yq) as f32, sumf);
            }
        }
        _ => {
            let mut v = [[0.0f32; 4]; 2];
            let (mut summs, mut summs1) = (0.0f32, 0.0f32);
            while ib + 1 < nb {
                for (k, v) in v.iter_mut().enumerate() {
                    let (q, d, _) = unpack(ty, w(ib + k));
                    let (y, _, yq) = xs(ib + k);
                    let scale = d * y;
                    for (v, l) in v.iter_mut().zip(lanes(&q, yq)) {
                        *v = (l as f32).mul_add(scale, *v);
                    }
                }
                let (m0, m1) = (unpack(ty, w(ib)).2, unpack(ty, w(ib + 1)).2);
                let (s0, s1) = (xs(ib).1, xs(ib + 1).1);
                match ty {
                    // `summs += m0 * s0 + m1 * s1`.
                    GgmlType::Q4_1 => summs += m0.mul_add(s0, m1 * s1),
                    // `summs0 += m0 * s0; summs1 += m1 * s1`.
                    GgmlType::Q5_1 => {
                        summs = m0.mul_add(s0, summs);
                        summs1 = m1.mul_add(s1, summs1);
                    }
                    _ => {}
                }
                ib += 2;
            }
            sumf = hsum(v[0]) + hsum(v[1]);
            if with_min {
                sumf = sumf + summs + summs1;
            }
            for ib in ib..nb {
                let (q, d, m) = unpack(ty, w(ib));
                let (y, s, yq) = xs(ib);
                let sumi = isum(&q, yq) as f32;
                sumf = match ty {
                    // `sumf += sumi * d * y`.
                    GgmlType::Q4_0 => (sumi * d).mul_add(y, sumf),
                    // `sumf += (d * y) * sumi + m * s`.
                    GgmlType::Q4_1 | GgmlType::Q5_1 => sumf + (d * y).mul_add(sumi, m * s),
                    // `sumf += (d * y) * sumi`.
                    _ => (d * y).mul_add(sumi, sumf),
                };
            }
        }
    }
    sumf
}

/// Encodes `x` (a multiple of 32 values) as `ty` into `out`, as ggml's
/// `quantize_row_*_ref` (`ggml-quants.c`) do, which is what its CPU
/// backend stores in a KV cache of that type (`from_float` of
/// `ggml_compute_forward_set_rows`). `IQ4_NL` takes the reference's single
/// guess of the scale (`ntry = -1`), not `quantize_iq4_nl`'s search.
///
/// # Panics
///
/// If `ty` is not `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1` or `IQ4_NL`, or `out` is
/// not exactly the bytes of `x` (internal invariants).
pub(crate) fn quantize_row_ref(ty: GgmlType, x: &[f32], out: &mut [u8]) {
    let bb = block_bytes(ty);
    assert!(
        bb > 0 && ty != GgmlType::Mxfp4,
        "no reference quantizer for {ty}"
    );
    assert_eq!(out.len(), x.len() / QK8_0 * bb, "{ty} row size");
    // `IQ4_NL`'s indices, which a block of zeros leaves as the block before
    // set them, as the reference does (its scale is then 0).
    let mut l = [0u8; QK8_0];
    for (xb, block) in x
        .as_chunks::<QK8_0>()
        .0
        .iter()
        .zip(out.chunks_exact_mut(bb))
    {
        match ty {
            GgmlType::Q4_0 | GgmlType::Q5_0 => quantize_block_sym(ty, xb, block),
            GgmlType::Q4_1 | GgmlType::Q5_1 => quantize_block_min(ty, xb, block),
            _ => quantize_block_iq4_nl(xb, block, &mut l),
        }
    }
}

/// The value of largest magnitude in `x` (the first of equal ones).
fn signed_max(x: &[f32; QK8_0]) -> f32 {
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in x {
        if amax < v.abs() {
            amax = v.abs();
            max = v;
        }
    }
    max
}

/// `quantize_row_q4_0_ref` / `quantize_row_q5_0_ref` for one block.
fn quantize_block_sym(ty: GgmlType, x: &[f32; QK8_0], block: &mut [u8]) {
    let five = ty == GgmlType::Q5_0;
    let (levels, bias, top) = if five {
        (-16.0f32, 16.5f32, 31u8)
    } else {
        (-8.0, 8.5, 15)
    };
    let d = signed_max(x) / levels;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    let qs_at = if five { 6 } else { 2 };
    let mut qh = 0u32;
    for j in 0..QK8_0 / 2 {
        let x0 = x[j] * id;
        let x1 = x[j + QK8_0 / 2] * id;
        // C's `(int8_t)` conversion truncates, as `as` does in range.
        let xi0 = ((x0 + bias) as i8 as u8).min(top);
        let xi1 = ((x1 + bias) as i8 as u8).min(top);
        block[qs_at + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= u32::from((xi0 & 0x10) >> 4) << j;
        qh |= u32::from((xi1 & 0x10) >> 4) << (j + QK8_0 / 2);
    }
    if five {
        block[2..6].copy_from_slice(&qh.to_le_bytes());
    }
}

/// `quantize_row_q4_1_ref` / `quantize_row_q5_1_ref` for one block.
fn quantize_block_min(ty: GgmlType, x: &[f32; QK8_0], block: &mut [u8]) {
    let five = ty == GgmlType::Q5_1;
    let (mut min, mut max) = (f32::MAX, -f32::MAX);
    for &v in x {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let d = (max - min) / if five { 31.0 } else { 15.0 };
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    block[2..4].copy_from_slice(&f32_to_f16(min).to_le_bytes());
    let qs_at = if five { 8 } else { 4 };
    let mut qh = 0u32;
    for j in 0..QK8_0 / 2 {
        let x0 = (x[j] - min) * id;
        let x1 = (x[j + QK8_0 / 2] - min) * id;
        let (xi0, xi1) = if five {
            ((x0 + 0.5) as u8, (x1 + 0.5) as u8)
        } else {
            (
                ((x0 + 0.5) as i8 as u8).min(15),
                ((x1 + 0.5) as i8 as u8).min(15),
            )
        };
        block[qs_at + j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
        qh |= u32::from((xi0 & 0x10) >> 4) << j;
        qh |= u32::from((xi1 & 0x10) >> 4) << (j + QK8_0 / 2);
    }
    if five {
        block[4..8].copy_from_slice(&qh.to_le_bytes());
    }
}

/// `quantize_row_iq4_nl_impl` for one block of 32 as
/// `quantize_row_iq4_nl_ref` calls it (no importance weights, `ntry = -1`):
/// the scale guessed from the largest magnitude, the nearest values, then
/// the least-squares scale for them.
fn quantize_block_iq4_nl(x: &[f32; QK8_0], block: &mut [u8], l: &mut [u8; QK8_0]) {
    let values = KVALUES_IQ4NL.map(f32::from);
    let mut mid = [0.0f32; 15];
    for k in 0..15 {
        mid[k] = 0.5 * (values[k] + values[k + 1]);
    }
    let best = |v: f32| -> usize { 15 - mid.iter().filter(|&&m| v < m).count() };
    let (mut amax, mut max) = (0.0f32, 0.0f32);
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    let mut d = 0.0f32;
    if amax >= 1e-15 {
        let id = 1.0 / (max / values[0]);
        let (mut sqx, mut sq2) = (0.0f32, 0.0f32);
        for (j, &v) in x.iter().enumerate() {
            let idx = best(id * v);
            l[j] = idx as u8;
            let w = v * v;
            let q = values[idx];
            // `sqx += w*q*xb[j]`, contracted to a fused multiply-add.
            sqx = (w * q).mul_add(v, sqx);
            sq2 = (w * q).mul_add(q, sq2);
        }
        d = if sq2 > 0.0 { sqx / sq2 } else { 0.0 };
    }
    block[..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    for j in 0..16 {
        block[2 + j] = l[j] | (l[j + 16] << 4);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e8m0_halves_powers_of_two() {
        assert_eq!(e8m0_half(127), 0.5);
        assert_eq!(e8m0_half(128), 1.0);
        assert_eq!(e8m0_half(130), 4.0);
        assert_eq!(e8m0_half(1), f32::from_bits(0x0040_0000)); // 2^-127
        assert_eq!(e8m0_half(0), f32::from_bits(0x0020_0000)); // 2^-128
    }

    #[test]
    fn fifth_bits_and_offsets_unpack_as_ggml_does() {
        let mut block = [0u8; 22];
        block[..2].copy_from_slice(&0x3C00u16.to_le_bytes());
        block[2..6].copy_from_slice(&0x0001_0001u32.to_le_bytes());
        block[6] = 0x3F; // weight 0: 0xF | 0x10, weight 16: 0x3 | 0x10
        block[7] = 0x21; // weight 1: 0x1, weight 17: 0x2
        let (q, d, _) = unpack(GgmlType::Q5_0, &block);
        assert_eq!(d, 1.0);
        assert_eq!((q[0], q[16], q[1], q[17]), (15, 3, -15, -14));
        let mut out = [0.0; 32];
        dequantize(GgmlType::Q5_0, &block, &mut out);
        assert_eq!((out[0], out[16], out[2]), (15.0, 3.0, -16.0));
    }

    #[test]
    fn q8_1_stores_the_scaled_sum() {
        let x: Vec<f32> = (0..32).map(|i| i as f32 - 8.0).collect();
        let mut out = [0u8; Q8_1_BLOCK_BYTES];
        quantize_row_q8_1(&x, &mut out, Q8_0Inverse::OfScale);
        let sum: i32 = out[4..].iter().map(|&q| i32::from(q as i8)).sum();
        let d = 23.0f32 / 127.0; // from the f32 `amax`, not the stored f16 scale
        assert_eq!(f16_at(&out, 2), f16_to_f32(f32_to_f16(d * sum as f32)));
        assert_eq!(out[4 + 31] as i8, 127);
    }
}
