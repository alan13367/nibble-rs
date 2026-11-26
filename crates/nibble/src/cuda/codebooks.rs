//! The I-quant tables uploaded once for the CUDA kernels.

use crate::quant::{iq_grids as grids, iquants};

/// Bytes of `Codebooks` in `kernels/common.cuh`: the grid entries in their
/// declared order, then the sign table. Every target is little-endian.
pub(crate) fn bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(25_728);
    for values in [
        &grids::IQ2XXS_GRID[..],
        &grids::IQ2XS_GRID[..],
        &grids::IQ2S_GRID[..],
    ] {
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    for values in [&grids::IQ3XXS_GRID[..], &grids::IQ3S_GRID[..]] {
        for &v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    // The CUDA dot products use ggml's packing: byte `j` holds grid values
    // `j` and `j + 4`, each plus one, in its low and high nibbles.
    for &entry in &grids::IQ1S_GRID {
        let value = |j: usize| ((entry >> (8 * j)) as u8 as i8 + 1) as u32;
        let packed = (0..4)
            .map(|j| (value(j) | (value(j + 4) << 4)) << (8 * j))
            .sum::<u32>();
        out.extend_from_slice(&packed.to_le_bytes());
    }
    for i in 0..128 {
        out.push(iquants::ksigns(i));
    }
    debug_assert_eq!(out.len(), 25_728);
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn layout_matches_the_cuda_struct() {
        let bytes = super::bytes();
        assert_eq!(bytes.len(), 25_728);
        assert_eq!(
            &bytes[..8],
            &crate::quant::iq_grids::IQ2XXS_GRID[0].to_le_bytes()
        );
        assert_eq!(bytes[25_600], crate::quant::iquants::ksigns(0));
        assert_eq!(bytes[25_727], crate::quant::iquants::ksigns(127));
    }
}
