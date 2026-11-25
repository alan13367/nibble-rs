//! The codebook I-quants: `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
//! `IQ1_S` and `IQ1_M`, all in super-blocks of 256 weights that meet a
//! `Q8_K` activation.
//!
//! Each weight's magnitude comes from a grid entry that packs 8 (`IQ2`,
//! `IQ1`) or 4 (`IQ3`) of them, and its sign from a bit (`IQ2`/`IQ3`) or the
//! grid itself (`IQ1`, whose values are -1, 0 and 1, shifted by a per-group
//! delta). Layouts are ggml's `block_iq*` (`ggml-common.h`); dequantization
//! follows `dequantize_row_iq*`. The dots follow the NEON
//! `ggml_vec_dot_iq*_q8_K` (`arch/arm/quants.c`): an exact integer dot per
//! group of 16 or 32 weights, combined with the group scales in that
//! kernel's order (integers where it stays in integers, its float updates,
//! fused, where it leaves them).

use super::iq_grids::{IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID};
use super::{Q8_K_BLOCK_BYTES, QK_K, f16_to_f32, q8k_block};
use nibble_gguf::GgmlType;

/// `IQ1S_DELTA` and `IQ1M_DELTA`: the shift of every `IQ1` grid value.
const IQ1_DELTA: f32 = 0.125;

/// `ksigns_iq2xs[i]`: the seven sign bits `i` and an eighth that makes the
/// number of set bits even.
pub(crate) const fn ksigns(i: usize) -> u8 {
    (i as u8 & 127) | (((i as u32 & 127).count_ones() as u8 & 1) << 7)
}

/// Bytes per super-block of each format.
pub(crate) const fn block_bytes(ty: GgmlType) -> usize {
    match ty {
        GgmlType::Iq2Xxs => 2 + QK_K / 4,
        GgmlType::Iq2Xs => 2 + QK_K / 4 + QK_K / 32,
        GgmlType::Iq2S => 2 + QK_K / 4 + QK_K / 16,
        GgmlType::Iq3Xxs => 2 + 3 * QK_K / 8,
        GgmlType::Iq3S => 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4,
        GgmlType::Iq1S => 2 + QK_K / 8 + QK_K / 16,
        GgmlType::Iq1M => QK_K / 8 + QK_K / 16 + QK_K / 32,
        _ => 0,
    }
}

fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Byte `j` of a grid entry.
fn byte(entry: u64, j: usize) -> u8 {
    (entry >> (8 * j)) as u8
}

/// The signed weights of a super-block (as the NEON kernels form them
/// before scaling; for `IQ1`, the grid value without its delta), in order.
/// `IQ2`/`IQ3` magnitudes are unsigned bytes up to 0x3E, so they and their
/// negations fit an `i8`.
pub(crate) fn weights(ty: GgmlType, block: &[u8]) -> [i8; QK_K] {
    let mut w = [0i8; QK_K];
    let signed = |mag: u8, signs: u8, j: usize| {
        if signs >> j & 1 != 0 {
            -(mag as i8)
        } else {
            mag as i8
        }
    };
    for g in 0..QK_K / 8 {
        let (ib32, l) = (g / 4, g % 4);
        let out = &mut w[8 * g..8 * g + 8];
        match ty {
            GgmlType::Iq2Xxs => {
                let aux1 = u32_at(block, 2 + 8 * ib32 + 4);
                let entry = IQ2XXS_GRID[usize::from(block[2 + 8 * ib32 + l])];
                let signs = ksigns(((aux1 >> (7 * l)) & 127) as usize);
                for (j, o) in out.iter_mut().enumerate() {
                    *o = signed(byte(entry, j), signs, j);
                }
            }
            GgmlType::Iq2Xs => {
                let q = u16_at(block, 2 + 2 * g);
                let entry = IQ2XS_GRID[usize::from(q & 511)];
                let signs = ksigns(usize::from(q >> 9));
                for (j, o) in out.iter_mut().enumerate() {
                    *o = signed(byte(entry, j), signs, j);
                }
            }
            GgmlType::Iq2S => {
                let qh = u16::from(block[2 + 64 + ib32]);
                let index = u16::from(block[2 + g]) | ((qh << (8 - 2 * l)) & 0x300);
                let entry = IQ2S_GRID[usize::from(index)];
                let signs = block[2 + 32 + g];
                for (j, o) in out.iter_mut().enumerate() {
                    *o = signed(byte(entry, j), signs, j);
                }
            }
            GgmlType::Iq3Xxs => {
                let aux = u32_at(block, 2 + 64 + 4 * ib32);
                let signs = ksigns(((aux >> (7 * l)) & 127) as usize);
                for half in 0..2 {
                    let entry =
                        u64::from(IQ3XXS_GRID[usize::from(block[2 + 8 * ib32 + 2 * l + half])]);
                    for j in 0..4 {
                        out[4 * half + j] = signed(byte(entry, j), signs, 4 * half + j);
                    }
                }
            }
            GgmlType::Iq3S => {
                let qh = u16::from(block[2 + 64 + ib32]);
                let signs = block[2 + 64 + 8 + g];
                for half in 0..2 {
                    let e = 2 * l + half;
                    let index = u16::from(block[2 + 8 * ib32 + e]) | ((qh << (8 - e)) & 256);
                    let entry = u64::from(IQ3S_GRID[usize::from(index)]);
                    for j in 0..4 {
                        out[4 * half + j] = signed(byte(entry, j), signs, 4 * half + j);
                    }
                }
            }
            GgmlType::Iq1S => {
                let qh = u16_at(block, 2 + 32 + 2 * ib32);
                let index = u16::from(block[2 + g]) | (((qh >> (3 * l)) & 7) << 8);
                let entry = IQ1S_GRID[usize::from(index)];
                for (j, o) in out.iter_mut().enumerate() {
                    *o = byte(entry, j) as i8;
                }
            }
            GgmlType::Iq1M => {
                let qh = u16::from(block[32 + g / 2]);
                let shift = if g % 2 == 0 { 8 } else { 4 };
                let index = u16::from(block[g]) | ((qh << shift) & 0x700);
                let entry = IQ1S_GRID[usize::from(index)];
                for (j, o) in out.iter_mut().enumerate() {
                    *o = byte(entry, j) as i8;
                }
            }
            other => unreachable!("{other} is not a codebook I-quant"),
        }
    }
    w
}

/// The f16 super-block scale of `IQ1_M`, spread over the top nibbles of its
/// four `u16` scale words.
fn iq1_m_scale(block: &[u8]) -> f32 {
    let sc = |k: usize| u16_at(block, 48 + 2 * k);
    f16_to_f32((sc(0) >> 12) | ((sc(1) >> 8) & 0x00F0) | ((sc(2) >> 4) & 0x0F00) | (sc(3) & 0xF000))
}

/// The `IQ1` delta of 8-weight group `g`: negative when its flag is set.
fn iq1_delta(ty: GgmlType, block: &[u8], g: usize) -> f32 {
    let negative = if ty == GgmlType::Iq1S {
        u16_at(block, 2 + 32 + 2 * (g / 4)) & 0x8000 != 0
    } else {
        block[32 + g / 2] & if g.is_multiple_of(2) { 0x08 } else { 0x80 } != 0
    };
    if negative { -IQ1_DELTA } else { IQ1_DELTA }
}

/// The integer scale of 16-weight group `t` of an `IQ2_XS`/`IQ2_S`
/// super-block (`2 * nibble + 1`), of 32-weight group `t` of `IQ3_S`
/// (`2 * nibble + 1`), and of 16-weight group `t` of `IQ1_M` (`2 * s + 1`
/// for a 3-bit `s`).
fn group_scale(ty: GgmlType, block: &[u8], t: usize) -> i32 {
    let nibble = |at: usize, k: usize| i32::from((block[at + k / 2] >> (4 * (k % 2))) & 0xF);
    match ty {
        GgmlType::Iq2Xs => 2 * nibble(2 + 64, t) + 1,
        GgmlType::Iq2S => 2 * nibble(2 + 64 + 8, t) + 1,
        GgmlType::Iq3S => {
            // Byte `t / 2`: low nibble for the even group of 32, high for the odd.
            2 * nibble(2 + 64 + 8 + 32, t) + 1
        }
        GgmlType::Iq1M => {
            let sc = u16_at(block, 48 + 2 * (t / 4));
            2 * i32::from((sc >> (3 * (t % 4))) & 7) + 1
        }
        other => unreachable!("{other} has no 4-bit group scales"),
    }
}

/// `dequantize_row_iq*`.
pub(crate) fn dequantize(ty: GgmlType, row: &[u8], out: &mut [f32]) {
    let n = block_bytes(ty);
    for (out, block) in out
        .as_chunks_mut::<QK_K>()
        .0
        .iter_mut()
        .zip(row.chunks_exact(n))
    {
        let w = weights(ty, block);
        let d = if ty == GgmlType::Iq1M {
            iq1_m_scale(block)
        } else {
            f16_to_f32(u16_at(block, 0))
        };
        for g in 0..QK_K / 8 {
            let ib32 = g / 4;
            // `db * grid * sign`, with the sign as a factor of ±1.
            let db = match ty {
                GgmlType::Iq2Xxs => {
                    let s = u32_at(block, 2 + 8 * ib32 + 4) >> 28;
                    d * (0.5 + s as f32) * 0.25
                }
                GgmlType::Iq2Xs | GgmlType::Iq2S => {
                    let at = if ty == GgmlType::Iq2Xs {
                        2 + 64
                    } else {
                        2 + 64 + 8
                    };
                    let s = (block[at + ib32] >> (4 * ((g % 4) / 2))) & 0xF;
                    d * (0.5 + f32::from(s)) * 0.25
                }
                GgmlType::Iq3Xxs => {
                    let s = u32_at(block, 2 + 64 + 4 * ib32) >> 28;
                    d * (0.5 + s as f32) * 0.5
                }
                GgmlType::Iq3S => d * group_scale(ty, block, ib32) as f32,
                GgmlType::Iq1S => {
                    let qh = u16_at(block, 2 + 32 + 2 * ib32);
                    d * (2 * ((qh >> 12) & 7) + 1) as f32
                }
                GgmlType::Iq1M => d * group_scale(ty, block, g / 2) as f32,
                _ => unreachable!(),
            };
            let out = &mut out[8 * g..8 * g + 8];
            let w = &w[8 * g..8 * g + 8];
            if matches!(ty, GgmlType::Iq1S | GgmlType::Iq1M) {
                let delta = iq1_delta(ty, block, g);
                for (o, &q) in out.iter_mut().zip(w) {
                    *o = db * (f32::from(q) + delta);
                }
            } else {
                for (o, &q) in out.iter_mut().zip(w) {
                    let sign = if q < 0 { -1.0 } else { 1.0 };
                    *o = db * f32::from(q.unsigned_abs()) * sign;
                }
            }
        }
    }
}

/// The exact integer dot of weights `w[a..b]` with the quants `q8[a..b]`.
fn idot(w: &[i8], q8: &[u8; QK_K], a: usize, b: usize) -> i32 {
    w[a..b]
        .iter()
        .zip(&q8[a..b])
        .map(|(&w, &q)| i32::from(w) * i32::from(q as i8))
        .sum()
}

/// Dot product of a row with a `Q8_K` activation.
pub(crate) fn dot_row(ty: GgmlType, row: &[u8], xq: &[u8]) -> f32 {
    let n = block_bytes(ty);
    let mut sumf = 0.0f32;
    for (block, x) in row
        .chunks_exact(n)
        .zip(xq.as_chunks::<Q8_K_BLOCK_BYTES>().0)
    {
        let (yd, q8, bsums) = q8k_block(x);
        let w = weights(ty, block);
        let xd = if ty == GgmlType::Iq1M {
            iq1_m_scale(block)
        } else {
            f16_to_f32(u16_at(block, 0))
        };
        match ty {
            GgmlType::Iq2Xxs | GgmlType::Iq3Xxs => {
                // `sumf1 += isum * (0.5 + s)` for even groups of 32, `sumf2`
                // for odd ones, then `sumf += d * (sumf1 + sumf2)`.
                let d = xd * yd;
                let mut sums = [0.0f32; 2];
                for ib32 in 0..8 {
                    let aux = if ty == GgmlType::Iq2Xxs {
                        u32_at(block, 2 + 8 * ib32 + 4)
                    } else {
                        u32_at(block, 2 + 64 + 4 * ib32)
                    };
                    let isum = idot(&w, q8, 32 * ib32, 32 * ib32 + 32) as f32;
                    sums[ib32 % 2] = isum.mul_add(0.5 + (aux >> 28) as f32, sums[ib32 % 2]);
                }
                sumf = d.mul_add(sums[0] + sums[1], sumf);
            }
            GgmlType::Iq2Xs | GgmlType::Iq2S => {
                let mut isum = 0i32;
                for t in 0..16 {
                    isum += idot(&w, q8, 16 * t, 16 * t + 16) * group_scale(ty, block, t);
                }
                sumf = (xd * yd).mul_add(isum as f32, sumf);
            }
            GgmlType::Iq3S => {
                let mut isum = 0i32;
                for ib32 in 0..8 {
                    isum += idot(&w, q8, 32 * ib32, 32 * ib32 + 32) * group_scale(ty, block, ib32);
                }
                sumf = (xd * yd).mul_add(isum as f32, sumf);
            }
            GgmlType::Iq1S => {
                // `sumf += y.d * d * (sumi1 + sumi2 + DELTA * sumi3)`, where
                // `sumi3` sums the activation (through `bsums`) with each
                // group's scale and delta sign.
                let (mut sumi, mut sumi3) = (0i32, 0i32);
                for ib in 0..8 {
                    let qh = u16_at(block, 2 + 32 + 2 * ib);
                    let ls = 2 * i32::from((qh >> 12) & 7) + 1;
                    let sign = if qh & 0x8000 != 0 { -1 } else { 1 };
                    sumi += idot(&w, q8, 32 * ib, 32 * ib + 32) * ls;
                    sumi3 += (i32::from(bsums[2 * ib]) + i32::from(bsums[2 * ib + 1])) * ls * sign;
                }
                let inner = IQ1_DELTA.mul_add(sumi3 as f32, sumi as f32);
                sumf = (yd * xd).mul_add(inner, sumf);
            }
            GgmlType::Iq1M => {
                let (mut sumi1, mut sumi2) = (0i32, 0i32);
                for t in 0..16 {
                    let scale = group_scale(ty, block, t);
                    sumi1 += scale * idot(&w, q8, 16 * t, 16 * t + 16);
                    let mut deltas = 0i32;
                    for g in 2 * t..2 * t + 2 {
                        let sum8: i32 = q8[8 * g..8 * g + 8]
                            .iter()
                            .map(|&q| i32::from(q as i8))
                            .sum();
                        deltas += if iq1_delta(ty, block, g) < 0.0 {
                            -sum8
                        } else {
                            sum8
                        };
                    }
                    sumi2 += scale * deltas;
                }
                let inner = IQ1_DELTA.mul_add(sumi2 as f32, sumi1 as f32);
                sumf = (yd * xd).mul_add(inner, sumf);
            }
            other => unreachable!("{other} is not a codebook I-quant"),
        }
    }
    match ty {
        GgmlType::Iq2Xxs => 0.25 * sumf,
        GgmlType::Iq2Xs | GgmlType::Iq2S => 0.125 * sumf,
        GgmlType::Iq3Xxs => 0.5 * sumf,
        _ => sumf,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_table_matches_ggml() {
        let head = [0u8, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139];
        for (i, &want) in head.iter().enumerate() {
            assert_eq!(ksigns(i), want);
        }
        assert_eq!(ksigns(127), 255);
        assert_eq!(ksigns(126), 126);
    }

    #[test]
    fn block_sizes_match_the_gguf_layouts() {
        for ty in [
            GgmlType::Iq2Xxs,
            GgmlType::Iq2Xs,
            GgmlType::Iq2S,
            GgmlType::Iq3Xxs,
            GgmlType::Iq3S,
            GgmlType::Iq1S,
            GgmlType::Iq1M,
        ] {
            assert_eq!(
                block_bytes(ty),
                ty.layout().unwrap().type_size as usize,
                "{ty}"
            );
        }
    }
}
