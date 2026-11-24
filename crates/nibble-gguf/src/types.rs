use std::fmt;

/// Elements per super-block in the K-quant and most I-quant formats.
pub const QK_K: u64 = 256;

/// How a tensor type packs elements: `block_size` elements in `type_size` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockLayout {
    /// Elements stored in one block.
    pub block_size: u64,
    /// Bytes used by one block.
    pub type_size: u64,
}

impl BlockLayout {
    /// Bytes needed for a row of `n` elements, or `None` when `n` is not a
    /// multiple of the block size or the size overflows.
    #[must_use]
    pub const fn row_bytes(self, n: u64) -> Option<u64> {
        if !n.is_multiple_of(self.block_size) {
            return None;
        }
        (n / self.block_size).checked_mul(self.type_size)
    }
}

macro_rules! ggml_types {
    ($( $(#[$doc:meta])* $variant:ident = $id:literal, $name:literal, ($block:expr, $size:expr); )*) => {
        /// A tensor element type (`ggml_type`), including quantized block formats.
        ///
        /// The ids and block layouts match `gguf-py/gguf/constants.py` in llama.cpp.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[non_exhaustive]
        pub enum GgmlType {
            $( $(#[$doc])* $variant, )*
            /// A type id this crate does not know. Tensors of this type are listed
            /// but their data cannot be located. Create values with
            /// [`GgmlType::from_raw`] so known ids never land here.
            Unknown(u32),
        }

        impl GgmlType {
            /// Maps a raw `ggml_type` id to a type.
            #[must_use]
            pub const fn from_raw(id: u32) -> Self {
                match id {
                    $( $id => Self::$variant, )*
                    other => Self::Unknown(other),
                }
            }

            /// The raw `ggml_type` id.
            #[must_use]
            pub const fn to_raw(self) -> u32 {
                match self {
                    $( Self::$variant => $id, )*
                    Self::Unknown(id) => id,
                }
            }

            /// The name llama.cpp uses for this type, such as `Q4_K`.
            #[must_use]
            pub const fn name(self) -> &'static str {
                match self {
                    $( Self::$variant => $name, )*
                    Self::Unknown(_) => "UNKNOWN",
                }
            }

            /// The block layout, or `None` for [`GgmlType::Unknown`].
            #[must_use]
            pub const fn layout(self) -> Option<BlockLayout> {
                match self {
                    $( Self::$variant => Some(BlockLayout { block_size: $block, type_size: $size }), )*
                    Self::Unknown(_) => None,
                }
            }
        }
    };
}

ggml_types! {
    /// 32-bit IEEE float.
    F32 = 0, "F32", (1, 4);
    /// 16-bit IEEE float.
    F16 = 1, "F16", (1, 2);
    /// 4-bit, blocks of 32 with an f16 scale.
    Q4_0 = 2, "Q4_0", (32, 2 + 16);
    /// 4-bit, blocks of 32 with an f16 scale and minimum.
    Q4_1 = 3, "Q4_1", (32, 2 + 2 + 16);
    /// 5-bit, blocks of 32 with an f16 scale.
    Q5_0 = 6, "Q5_0", (32, 2 + 4 + 16);
    /// 5-bit, blocks of 32 with an f16 scale and minimum.
    Q5_1 = 7, "Q5_1", (32, 2 + 2 + 4 + 16);
    /// 8-bit, blocks of 32 with an f16 scale.
    Q8_0 = 8, "Q8_0", (32, 2 + 32);
    /// 8-bit, blocks of 32 with an f16 scale and sum (used for activations).
    Q8_1 = 9, "Q8_1", (32, 2 + 2 + 32);
    /// 2-bit K-quant.
    Q2K = 10, "Q2_K", (QK_K, 2 + 2 + QK_K / 16 + QK_K / 4);
    /// 3-bit K-quant.
    Q3K = 11, "Q3_K", (QK_K, 2 + QK_K / 4 + QK_K / 8 + 12);
    /// 4-bit K-quant.
    Q4K = 12, "Q4_K", (QK_K, 2 + 2 + QK_K / 2 + 12);
    /// 5-bit K-quant.
    Q5K = 13, "Q5_K", (QK_K, 2 + 2 + QK_K / 2 + QK_K / 8 + 12);
    /// 6-bit K-quant.
    Q6K = 14, "Q6_K", (QK_K, 2 + QK_K / 2 + QK_K / 4 + QK_K / 16);
    /// 8-bit K-quant (used for activations).
    Q8K = 15, "Q8_K", (QK_K, 4 + QK_K + QK_K / 8);
    /// 2.06 bits per weight I-quant.
    Iq2Xxs = 16, "IQ2_XXS", (QK_K, 2 + QK_K / 4);
    /// 2.31 bits per weight I-quant.
    Iq2Xs = 17, "IQ2_XS", (QK_K, 2 + QK_K / 4 + QK_K / 32);
    /// 3.06 bits per weight I-quant.
    Iq3Xxs = 18, "IQ3_XXS", (QK_K, 2 + QK_K / 4 + QK_K / 8);
    /// 1.56 bits per weight I-quant.
    Iq1S = 19, "IQ1_S", (QK_K, 2 + QK_K / 8 + QK_K / 16);
    /// 4-bit non-linear, blocks of 32.
    Iq4Nl = 20, "IQ4_NL", (32, 2 + 16);
    /// 3.44 bits per weight I-quant.
    Iq3S = 21, "IQ3_S", (QK_K, 2 + QK_K / 4 + QK_K / 8 + QK_K / 32 + 4);
    /// 2.5 bits per weight I-quant.
    Iq2S = 22, "IQ2_S", (QK_K, 2 + QK_K / 4 + QK_K / 16);
    /// 4.25 bits per weight non-linear I-quant.
    Iq4Xs = 23, "IQ4_XS", (QK_K, 2 + 2 + QK_K / 2 + QK_K / 64);
    /// 8-bit signed integer.
    I8 = 24, "I8", (1, 1);
    /// 16-bit signed integer.
    I16 = 25, "I16", (1, 2);
    /// 32-bit signed integer.
    I32 = 26, "I32", (1, 4);
    /// 64-bit signed integer.
    I64 = 27, "I64", (1, 8);
    /// 64-bit IEEE float.
    F64 = 28, "F64", (1, 8);
    /// 1.75 bits per weight I-quant.
    Iq1M = 29, "IQ1_M", (QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32);
    /// bfloat16.
    Bf16 = 30, "BF16", (1, 2);
    /// Ternary, 1.69 bits per weight.
    Tq1_0 = 34, "TQ1_0", (QK_K, 2 + 4 * 13);
    /// Ternary, 2.06 bits per weight.
    Tq2_0 = 35, "TQ2_0", (QK_K, 2 + 64);
    /// Microscaling FP4, blocks of 32 with an E8M0 scale.
    Mxfp4 = 39, "MXFP4", (32, 1 + 16);
    /// NVIDIA FP4, blocks of 64.
    Nvfp4 = 40, "NVFP4", (64, 4 + 32);
    /// 1-bit, blocks of 128.
    Q1_0 = 41, "Q1_0", (128, 2 + 16);
    /// 2-bit, blocks of 64.
    Q2_0 = 42, "Q2_0", (64, 2 + 16);
    /// 8-bit float, E4M3.
    F8E4M3 = 43, "F8_E4M3", (1, 1);
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(id) => write!(f, "type#{id}"),
            known => f.pad(known.name()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_ids_round_trip() {
        for id in 0..64 {
            assert_eq!(GgmlType::from_raw(id).to_raw(), id);
        }
    }

    #[test]
    fn layouts_match_llama_cpp() {
        let expect = [
            (GgmlType::Q4_0, 32, 18),
            (GgmlType::Q8_0, 32, 34),
            (GgmlType::Q2K, 256, 84),
            (GgmlType::Q3K, 256, 110),
            (GgmlType::Q4K, 256, 144),
            (GgmlType::Q5K, 256, 176),
            (GgmlType::Q6K, 256, 210),
            (GgmlType::Q8K, 256, 292),
            (GgmlType::Iq4Xs, 256, 136),
            (GgmlType::Iq1M, 256, 56),
            (GgmlType::Tq1_0, 256, 54),
        ];
        for (ty, block, size) in expect {
            let layout = ty.layout().unwrap();
            assert_eq!((layout.block_size, layout.type_size), (block, size), "{ty}");
        }
    }

    #[test]
    fn row_bytes_requires_whole_blocks() {
        let q8 = GgmlType::Q8_0.layout().unwrap();
        assert_eq!(q8.row_bytes(64), Some(68));
        assert_eq!(q8.row_bytes(33), None);
    }
}
