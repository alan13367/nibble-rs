//! The model's bytes as GPU buffers.
//!
//! A Metal buffer holds at most `maxBufferLength` bytes (about 22 GB on a
//! 36 GB M4 Max, less on smaller Macs), so a larger model is split into
//! pieces that overlap by its largest tensor, as llama.cpp splits it: every
//! tensor then lies whole inside one piece. A memory-mapped file is wrapped
//! without a copy, so the overlap costs nothing; a model read into memory
//! is copied piece by piece.

use std::ops::Range;

use super::device::{Buffer, Device, Wrapped};
use crate::backend::ModelData;
use crate::cpu::Weight;

/// Environment variable that caps the size of each weight buffer, in
/// bytes, to test a split model on a Mac that would not need one.
pub(crate) const ENV_MAX_BUFFER: &str = "NIBBLE_METAL_MAX_BUFFER";

/// How `total` bytes are cut into pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Pieces {
    total: usize,
    /// Distance between the starts of consecutive pieces.
    step: usize,
    /// Length of every piece but the last.
    len: usize,
    count: usize,
}

impl Pieces {
    /// Pieces of at most `max_len` bytes that start on `page` boundaries and
    /// overlap by at least `overlap` bytes, or `None` if `overlap` leaves no
    /// room to advance.
    pub(crate) fn new(total: usize, max_len: usize, overlap: usize, page: usize) -> Option<Self> {
        let len = max_len / page * page;
        if total <= len {
            return Some(Self {
                total,
                step: total.max(1),
                len: total,
                count: 1,
            });
        }
        let step = len.checked_sub(overlap)? / page * page;
        if step == 0 {
            return None;
        }
        Some(Self {
            total,
            step,
            len,
            count: (total - len).div_ceil(step) + 1,
        })
    }

    pub(crate) fn count(&self) -> usize {
        self.count
    }

    /// The bytes of piece `k`.
    pub(crate) fn range(&self, k: usize) -> Range<usize> {
        let start = k * self.step;
        start..(start + self.len).min(self.total)
    }

    /// The piece holding bytes `offset..offset + len`, and the offset in it.
    pub(crate) fn locate(&self, offset: usize, len: usize) -> Option<(usize, usize)> {
        let end = offset.checked_add(len)?;
        let k = (offset / self.step).min(self.count - 1);
        let range = self.range(k);
        (end <= range.end).then(|| (k, offset - range.start))
    }
}

#[derive(Debug)]
enum Piece<'m> {
    /// Part of the memory map itself.
    Wrapped(Wrapped<'m>),
    /// A copy, for models read into memory.
    Copied(Buffer),
}

impl Piece<'_> {
    fn buffer(&self) -> &Buffer {
        match self {
            Self::Wrapped(w) => w.buffer(),
            Self::Copied(b) => b,
        }
    }
}

/// The model's bytes in one or more buffers.
#[derive(Debug)]
pub(crate) struct Weights<'m> {
    pieces: Pieces,
    buffers: Vec<Piece<'m>>,
}

impl<'m> Weights<'m> {
    /// Buffers of at most `max_len` bytes over `model.bytes`.
    pub(crate) fn new(
        device: &Device,
        model: ModelData<'m>,
        max_len: usize,
    ) -> Result<Self, String> {
        Self::build(
            device,
            model.bytes,
            model.mapped,
            model.largest_tensor,
            max_len,
        )
    }

    /// Buffers of at most `max_len` bytes over `bytes`, whose largest tensor
    /// is `largest` bytes, wrapped if `mapped` (as [`ModelData::mapped`]).
    pub(crate) fn build(
        device: &Device,
        bytes: &'m [u8],
        mapped: bool,
        largest: usize,
        max_len: usize,
    ) -> Result<Self, String> {
        let page = super::device::page_size();
        let pieces = Pieces::new(bytes.len(), max_len, largest, page).ok_or_else(|| {
            format!("a {largest}-byte tensor does not fit the device's {max_len}-byte buffers")
        })?;
        let wrapped: Option<Vec<Piece<'m>>> = mapped
            .then(|| {
                (0..pieces.count())
                    .map(|k| {
                        // SAFETY: `model.bytes` is a memory map of the whole
                        // file (`ModelData::mapped`): it starts on a page
                        // boundary and is readable to the end of its last
                        // page, and each piece starts on a page boundary
                        // within it. It lives for `'m`, as the buffers do,
                        // and the backend waits for the GPU before they go.
                        // The kernels only read weights.
                        unsafe { device.wrap(&bytes[pieces.range(k)]) }
                            .ok()
                            .map(Piece::Wrapped)
                    })
                    .collect()
            })
            .flatten();
        let buffers = match wrapped {
            Some(b) => b,
            None => (0..pieces.count())
                .map(|k| {
                    device
                        .buffer_with(&bytes[pieces.range(k)])
                        .map(Piece::Copied)
                })
                .collect::<Result<_, _>>()?,
        };
        Ok(Self { pieces, buffers })
    }

    /// Buffers it was split into.
    #[cfg(test)]
    pub(crate) fn pieces(&self) -> usize {
        self.buffers.len()
    }

    /// The buffer holding the matrix `w` and the offset of `w` in it.
    pub(crate) fn locate(&self, w: &Weight) -> Option<(&Buffer, usize)> {
        let len = w.rows.checked_mul(w.row_bytes)?;
        let (k, offset) = self.pieces.locate(w.offset, len)?;
        Some((self.buffers[k].buffer(), offset))
    }
}

#[cfg(test)]
mod tests {
    use super::Pieces;

    const PAGE: usize = 16384;

    #[test]
    fn small_models_are_one_piece() {
        let p = Pieces::new(1000, 4 * PAGE, 500, PAGE).unwrap();
        assert_eq!(p.count(), 1);
        assert_eq!(p.range(0), 0..1000);
        assert_eq!(p.locate(0, 1000), Some((0, 0)));
        assert_eq!(p.locate(10, 991), None);
    }

    /// Every range up to the overlap lies inside the piece `locate` names,
    /// pieces start on page boundaries and are never longer than allowed.
    #[test]
    fn every_tensor_lies_inside_one_piece() {
        for (total, max_len, overlap) in [
            (10 * PAGE + 123, 4 * PAGE, PAGE + 7),
            (100 * PAGE, 9 * PAGE + 100, 3 * PAGE),
            (33 * PAGE, 4 * PAGE, 3 * PAGE),
        ] {
            let p = Pieces::new(total, max_len, overlap, PAGE).unwrap();
            assert!(p.count() > 1);
            assert_eq!(p.range(p.count() - 1).end, total);
            for k in 0..p.count() {
                let r = p.range(k);
                assert_eq!(r.start % PAGE, 0);
                assert!(r.len() <= max_len);
            }
            for offset in (0..total).step_by(4093) {
                for len in [0, 1, overlap / 2, overlap] {
                    if offset + len > total {
                        continue;
                    }
                    let (k, inner) = p.locate(offset, len).unwrap();
                    let r = p.range(k);
                    assert_eq!(r.start + inner, offset);
                    assert!(offset + len <= r.end, "{offset}+{len} in piece {k} {r:?}");
                }
            }
            assert_eq!(p.locate(total - 1, 2), None);
        }
    }

    #[test]
    fn a_tensor_larger_than_a_buffer_cannot_be_placed() {
        assert_eq!(Pieces::new(100 * PAGE, 4 * PAGE, 4 * PAGE, PAGE), None);
    }
}
