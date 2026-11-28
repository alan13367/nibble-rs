//! Embedded CUDA fatbin, kernel argument layouts and dispatch geometry.

use nibble_gguf::GgmlType;

use super::device::{Function, Module};

/// The offline-built kernels. Building nibble never invokes nvcc.
pub(crate) const FATBIN: &[u8] = include_bytes!("kernels/nibble.fatbin");

macro_rules! args {
    ($($name:ident { $($field:ident: $ty:ty),* $(,)? })*) => {
        $(
            #[derive(Clone, Copy, Debug)]
            #[repr(C)]
            pub(crate) struct $name {
                $(pub(crate) $field: $ty),*
            }
            const _: () = assert!(size_of::<$name>() == 4 * [$(stringify!($field)),*].len());
        )*
    };
}

args! {
    MatvecArgs { cols: u32, rows: u32, row_bytes: u32, ncols: u32, mode: u32 }
    GemmSplitK { blocks: u32, tiles_n: u32, tiles: u32, stages: u32 }
    MoeArgs { k: u32, per_slot: u32, expert_rows: u32, count: u32 }
    EmbedArgs { cols: u32, row: u32, row_bytes: u32, from_slot: u32 }
    CountArgs { n: u32 }
    RmsArgs { n: u32, eps: f32 }
    ShortConvArgs { n_embd: u32, taps: u32, rows: u32, oldest: u32, n: u32 }
    RouteArgs { n_expert: u32, k: u32, sigmoid: u32 }
    MoeSumArgs { n: u32, k: u32, eps: f32 }
    RotateArgs {
        len: u32, block: u32, signs: u32, head_dim: u32, n_k: u32, rep: u32, inverse: u32
    }
    AttentionArgs {
        n_pos: u32, n: u32, head_dim: u32, group: u32, kv_dim: u32, q_stride: u32,
        gated: u32, n_rot: u32, norm: u32, scale: f32, base: f32, eps: f32,
        k_type: u32, v_type: u32, k_row: u32, v_row: u32, rot_k: u32, rot_v: u32,
        n_head: u32, t0: u32, n_kv_max: u32, cap_pad: u32, multi_rope: u32,
        kq_mma: u32, n_cells: u32, v_trans: u32,
    }
    DeltaBatchArgs {
        n_k_heads: u32, d_k: u32, d_v: u32, key_dim: u32, conv_dim: u32, taps: u32,
        rows: u32, oldest: u32, replay: u32, n: u32, pending: u32, ba_read: u32,
        ba_write: u32, eps: f32, scale: f32, l2_eps: f32, l2_scale: f32,
        prepped: u32, tail: u32,
    }
}

pub(crate) const MATVEC_STORE: u32 = 0;
pub(crate) const MATVEC_ADD: u32 = 1;
pub(crate) const MATVEC_SWIGLU: u32 = 2;

/// A weight type's product and embedding entry points.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TypeKernels {
    pub(crate) matvec: Function,
    pub(crate) matvec_c4: Function,
    pub(crate) matvec_id: Function,
    pub(crate) embed: Function,
    /// The prompt-batch GEMMs (`kernels/gemm.cu`) for the tiles of
    /// `GEMM_TOKENS`, with their dynamic shared memory, for the types they
    /// cover.
    pub(crate) gemm: Option<[(Function, u32); 3]>,
    /// The fused gate/up GEMMs with SwiGLU (`gemm_glu_*`), as `gemm`.
    pub(crate) gemm_glu: Option<[(Function, u32); 3]>,
    /// `(values per block, warps, output rows per block)`.
    pub(crate) shape: (u32, u32, u32),
}

/// Every entry point the backend launches.
#[derive(Debug)]
pub(crate) struct Pipelines {
    types: Vec<(u32, TypeKernels)>,
    pub(crate) quantize_q8_1: Function,
    /// Activations for the GEMM (`quantize_mmq` in `kernels/gemm.cu`).
    pub(crate) quantize_mmq: Function,
    pub(crate) copy: Function,
    pub(crate) swiglu: Function,
    pub(crate) rmsnorm: Function,
    pub(crate) rotate: Function,
    pub(crate) argmax: Function,
    pub(crate) moe_route: Function,
    pub(crate) moe_combine: Function,
    pub(crate) moe_residual: Function,
    pub(crate) attention_prep: Function,
    pub(crate) attention: Function,
    pub(crate) attention_kq: Function,
    pub(crate) attention_kq_mma: Function,
    pub(crate) attention_softmax: Function,
    pub(crate) attention_kqv: Function,
    /// Prompt-batch flash attention for an F16 cache, by head size.
    pub(crate) fattn: [(usize, Function); 3],
    pub(crate) delta_net_batch: Function,
    pub(crate) delta_prep: Function,
    /// The sequential delta rule after `delta_prep`, by key head size.
    pub(crate) delta_seq: [(usize, Function); 3],
    pub(crate) delta_norm: Function,
    /// LFM2's shortconv convolution, and the ring store after it.
    pub(crate) lfm_short_conv: Function,
    pub(crate) lfm_conv_store: Function,
    /// `x += add`, then the next block's norm of `x`.
    pub(crate) add_norm: Function,
}

impl Pipelines {
    /// The kernels of `module` for a device of compute capability `major`.
    pub(crate) fn new(module: &Module, major: u32) -> Result<Self, String> {
        let get = |name: &str| module.function(name);
        let mut types = Vec::new();
        for (code, name, shape) in TYPE_SPECS {
            // The kernels' KQUANT_NWARPS: K-quants use 4 warps from Ampere on.
            let shape = &match *code {
                10..=14 if major >= 8 => (shape.0, 4, shape.2),
                _ => *shape,
            };
            types.push((
                *code,
                TypeKernels {
                    matvec: get(&format!("matvec_{name}"))?,
                    matvec_c4: get(&format!("matvec_{name}_c4"))?,
                    matvec_id: get(&format!("matvec_{name}_id"))?,
                    embed: get(&format!("embed_{name}"))?,
                    gemm: if GEMM_TYPES.contains(code) {
                        let tile = |t: usize| -> Result<(Function, u32), String> {
                            Ok((get(&format!("gemm_{name}_{t}"))?, gemm_shared(*code, t)))
                        };
                        Some([
                            tile(GEMM_TOKENS[0])?,
                            tile(GEMM_TOKENS[1])?,
                            tile(GEMM_TOKENS[2])?,
                        ])
                    } else {
                        None
                    },
                    gemm_glu: if GEMM_TYPES.contains(code) {
                        let tile = |t: usize| -> Result<(Function, u32), String> {
                            Ok((get(&format!("gemm_glu_{name}_{t}"))?, gemm_shared(*code, t)))
                        };
                        Some([
                            tile(GEMM_TOKENS[0])?,
                            tile(GEMM_TOKENS[1])?,
                            tile(GEMM_TOKENS[2])?,
                        ])
                    } else {
                        None
                    },
                    shape: *shape,
                },
            ));
        }
        Ok(Self {
            types,
            quantize_q8_1: get("quantize_q8_1")?,
            quantize_mmq: get("quantize_mmq")?,
            copy: get("copy_f32")?,
            swiglu: get("swiglu")?,
            rmsnorm: get("rmsnorm")?,
            rotate: get("rotate")?,
            argmax: get("argmax")?,
            moe_route: get("moe_route")?,
            moe_combine: get("moe_combine")?,
            moe_residual: get("moe_residual")?,
            attention_prep: get("attention_prep")?,
            attention: get("attention")?,
            attention_kq: get("attention_kq")?,
            attention_kq_mma: get("attention_kq_mma")?,
            attention_softmax: get("attention_softmax")?,
            attention_kqv: get("attention_kqv")?,
            fattn: [
                (64, get("fattn_f16_64")?),
                (128, get("fattn_f16_128")?),
                (256, get("fattn_f16_256")?),
            ],
            delta_net_batch: get("delta_net_batch")?,
            delta_prep: get("delta_prep")?,
            delta_seq: [
                (64, get("delta_seq_64")?),
                (128, get("delta_seq_128")?),
                (256, get("delta_seq_256")?),
            ],
            delta_norm: get("delta_norm")?,
            lfm_short_conv: get("lfm_short_conv")?,
            lfm_conv_store: get("lfm_conv_store")?,
            add_norm: get("add_norm")?,
        })
    }

    /// Every GEMM with its dynamic shared memory.
    pub(crate) fn gemms(&self) -> impl Iterator<Item = &(Function, u32)> {
        self.types
            .iter()
            .flat_map(|(_, k)| k.gemm.iter().chain(&k.gemm_glu).flatten())
    }

    pub(crate) fn get(&self, ty: GgmlType) -> Option<&TypeKernels> {
        let code = type_code(ty)?;
        self.types.iter().find(|(c, _)| *c == code).map(|(_, k)| k)
    }
}

/// The numeric type codes shared with `kernels/quant.cuh`.
pub(crate) const fn type_code(ty: GgmlType) -> Option<u32> {
    Some(match ty {
        GgmlType::F32 => 0,
        GgmlType::F16 => 1,
        GgmlType::Bf16 => 2,
        GgmlType::Q8_0 => 3,
        GgmlType::Q4_0 => 4,
        GgmlType::Q4_1 => 5,
        GgmlType::Q5_0 => 6,
        GgmlType::Q5_1 => 7,
        GgmlType::Iq4Nl => 8,
        GgmlType::Mxfp4 => 9,
        GgmlType::Q2K => 10,
        GgmlType::Q3K => 11,
        GgmlType::Q4K => 12,
        GgmlType::Q5K => 13,
        GgmlType::Q6K => 14,
        GgmlType::Iq4Xs => 15,
        GgmlType::Iq2Xxs => 16,
        GgmlType::Iq2Xs => 17,
        GgmlType::Iq2S => 18,
        GgmlType::Iq3Xxs => 19,
        GgmlType::Iq3S => 20,
        GgmlType::Iq1S => 21,
        GgmlType::Iq1M => 22,
        GgmlType::Q2_0 => 23,
        _ => return None,
    })
}

pub(crate) const fn is_quantized(ty: GgmlType) -> bool {
    match type_code(ty) {
        Some(code) => code >= 3,
        None => false,
    }
}

/// Dynamic shared memory of `fattn_f16_{head_dim}` (`FattnShape` in
/// `kernels/fattn.cu`).
pub(crate) const fn fattn_shared(head_dim: usize) -> usize {
    let kb = if head_dim >= 256 { 32 } else { 64 };
    2 * (64 * (head_dim + 8) + kb * (head_dim + 8) + head_dim * (kb + 8))
}

/// The type codes `kernels/gemm.cu` exports a GEMM for.
const GEMM_TYPES: [u32; 10] = [3, 4, 5, 6, 7, 10, 11, 12, 13, 14];

/// Weight rows of one GEMM block (`GEMM_BM`), and the token tiles of the
/// exported GEMMs, in the order of `TypeKernels::gemm`.
pub(crate) const GEMM_ROWS: usize = 128;
pub(crate) const GEMM_TOKENS: [usize; 3] = [32, 64, 128];
/// Tokens of the GEMM activations are padded to this (`GEMM_TOK_PAD`), and
/// a token's activations take this many bytes per 128 values (`GEMM_YS`).
pub(crate) const GEMM_TOKEN_PAD: usize = 128;
pub(crate) const GEMM_CHUNK_BYTES: usize = 4 * 44;

/// Dynamic shared memory of the GEMM for type `code` with tiles of
/// `tokens` (`GemmShared` in `kernels/gemm.cu`): weight rows of 100 words
/// for types with a min term (Q4_1, Q5_1, Q2_K, Q4_K, Q5_K), else 84, and
/// two buffers of activations.
const fn gemm_shared(code: u32, tokens: usize) -> u32 {
    let with_min = matches!(code, 5 | 7 | 10 | 12 | 13);
    (4 * (GEMM_ROWS * if with_min { 100 } else { 84 } + 2 * tokens * 44)) as u32
}

#[rustfmt::skip]
const TYPE_SPECS: &[(u32, &str, (u32, u32, u32))] = &[
    (0, "f32", (1, 4, 1)),
    (1, "f16", (1, 4, 1)),
    (2, "bf16", (1, 4, 1)),
    (3, "q8_0", (32, 4, 1)),
    (4, "q4_0", (32, 4, 1)),
    (5, "q4_1", (32, 4, 1)),
    (6, "q5_0", (32, 4, 1)),
    (7, "q5_1", (32, 4, 1)),
    (8, "iq4_nl", (32, 4, 1)),
    (9, "mxfp4", (32, 4, 1)),
    // K-quants: 2 warps on Turing (see `Pipelines::new`).
    (10, "q2_K", (256, 2, 2)),
    (11, "q3_K", (256, 2, 2)),
    (12, "q4_K", (256, 2, 1)),
    (13, "q5_K", (256, 2, 1)),
    (14, "q6_K", (256, 2, 1)),
    (15, "iq4_xs", (256, 4, 1)),
    (16, "iq2_xxs", (256, 4, 1)),
    (17, "iq2_xs", (256, 4, 1)),
    (18, "iq2_s", (256, 4, 1)),
    (19, "iq3_xxs", (256, 4, 1)),
    (20, "iq3_s", (256, 4, 1)),
    (21, "iq1_s", (256, 4, 2)),
    (22, "iq1_m", (256, 4, 2)),
    (23, "q2_0", (64, 4, 1)),
];
