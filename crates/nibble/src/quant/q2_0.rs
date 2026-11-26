//! `Q2_0`: 64 weights per block, an f16 scale `d` and 2-bit codes (four per
//! byte, weight `j` in bits `2(j % 4)..` of byte `j / 4`) that stand for
//! `{-1, 0, 1, 2} · d` (`block_q2_0` and `dequantize_row_q2_0` in ggml).
//! Ternary models use the first three.
//!
//! A block meets two `Q8_0` activation blocks. The reference dot follows
//! ggml's NEON `ggml_vec_dot_q2_0_q8_0` (`arch/arm/quants.c`): each half of
//! a block is two `vdotq_s32` (weights `0..16` and `16..32`), whose four
//! integer lanes are added to one float vector with the half's scale
//! `d * d_y` (`vmlaq_n_f32`, fused), reduced by `vaddvq_f32` at the end.

use super::blocks32::{hsum, lanes};
use super::{Q8_0_BLOCK_BYTES, f16_to_f32};

/// Elements per `Q2_0` block.
pub(crate) const QK2_0: usize = 64;
/// Bytes per `Q2_0` block: the f16 scale and 16 bytes of codes.
pub(crate) const Q2_0_BLOCK_BYTES: usize = 2 + QK2_0 / 4;

/// The scale and the 64 weights of a block as signed integers (`code - 1`).
pub(crate) fn unpack(block: &[u8; Q2_0_BLOCK_BYTES]) -> (f32, [i8; QK2_0]) {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let mut q = [0i8; QK2_0];
    for (j, q) in q.iter_mut().enumerate() {
        *q = ((block[2 + j / 4] >> (2 * (j % 4))) & 3) as i8 - 1;
    }
    (d, q)
}

/// `dequantize_row_q2_0`: `(code - 1) * d`.
pub(crate) fn dequantize(row: &[u8], out: &mut [f32]) {
    let blocks = row.as_chunks::<Q2_0_BLOCK_BYTES>().0;
    for (out, block) in out.as_chunks_mut::<QK2_0>().0.iter_mut().zip(blocks) {
        let (d, q) = unpack(block);
        for (o, &q) in out.iter_mut().zip(&q) {
            *o = f32::from(q) * d;
        }
    }
}

/// Dot product of a row with an activation encoded as `Q8_0`.
pub(crate) fn dot_row(row: &[u8], xq: &[u8]) -> f32 {
    let blocks = row.as_chunks::<Q2_0_BLOCK_BYTES>().0;
    let xs = xq.as_chunks::<Q8_0_BLOCK_BYTES>().0;
    let mut v = [0.0f32; 4];
    for (block, x) in blocks.iter().zip(xs.as_chunks::<2>().0) {
        let (d, q) = unpack(block);
        for (q, y) in q.as_chunks::<32>().0.iter().zip(x) {
            let scale = d * f16_to_f32(u16::from_le_bytes([y[0], y[1]]));
            for (v, l) in v.iter_mut().zip(lanes(q, &y[2..])) {
                *v = (l as f32).mul_add(scale, *v);
            }
        }
    }
    hsum(v)
}
