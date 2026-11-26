//! The MSL kernels: their source, argument structs, pipelines and dispatch
//! geometry.

use nibble_gguf::GgmlType;

use super::device::{Device, KernelArgs, Pipeline, Size};

/// The shader source, compiled once per backend: the files in order, with
/// the I-quant codebooks ([`codebooks`]) after the shared definitions.
pub(crate) fn source() -> String {
    [
        include_str!("shaders/common.metal"),
        &codebooks(),
        include_str!("shaders/matvec.metal"),
        include_str!("shaders/kquants.metal"),
        include_str!("shaders/iquants.metal"),
        include_str!("shaders/matmul.metal"),
        include_str!("shaders/kv.metal"),
        include_str!("shaders/ops.metal"),
        include_str!("shaders/moe.metal"),
    ]
    .concat()
}

/// The I-quant codebooks as MSL `constant` arrays, from the CPU's tables.
/// `iq1s_grid_gpu` is ggml-metal's packing of `iq1s_grid`: byte `j` holds
/// value `j` plus one in its low nibble and value `j + 4` plus one in its
/// high nibble.
fn codebooks() -> String {
    use crate::quant::{iq_grids as g, iquants::ksigns};
    use std::fmt::Write;
    fn table<T: std::fmt::LowerHex>(out: &mut String, ty: &str, name: &str, values: &[T]) {
        let _ = write!(out, "constant {ty} {name}[{}] = {{", values.len());
        for v in values {
            let _ = write!(out, "0x{v:x},");
        }
        out.push_str("};\n");
    }
    let mut out = String::new();
    table(&mut out, "uint64_t", "iq2xxs_grid", &g::IQ2XXS_GRID);
    table(&mut out, "uint64_t", "iq2xs_grid", &g::IQ2XS_GRID);
    table(&mut out, "uint64_t", "iq2s_grid", &g::IQ2S_GRID);
    table(&mut out, "uint32_t", "iq3xxs_grid", &g::IQ3XXS_GRID);
    table(&mut out, "uint32_t", "iq3s_grid", &g::IQ3S_GRID);
    let gpu: Vec<u32> = g::IQ1S_GRID
        .iter()
        .map(|&e| {
            let v = |j: usize| ((e >> (8 * j)) as u8 as i8 + 1) as u32;
            (0..4).map(|j| (v(j) | v(j + 4) << 4) << (8 * j)).sum()
        })
        .collect();
    table(&mut out, "uint32_t", "iq1s_grid_gpu", &gpu);
    let signs: Vec<u8> = (0..128).map(ksigns).collect();
    table(&mut out, "uint8_t", "ksigns_iq2xs", &signs);
    out
}

/// Declares `repr(C)` argument structs of 4-byte fields, which have no
/// padding, so they are [`KernelArgs`].
macro_rules! kernel_args {
    ($($(#[$doc:meta])* struct $name:ident { $($field:ident: $ty:ty),* $(,)? })*) => {
        $(
            $(#[$doc])*
            #[derive(Clone, Copy, Debug)]
            #[repr(C)]
            pub(crate) struct $name {
                $(pub(crate) $field: $ty),*
            }

            // SAFETY: `repr(C)` with only `u32`/`f32` fields: 4-byte aligned
            // and without padding; the fields match the MSL struct of the
            // same name in `shaders/`.
            unsafe impl KernelArgs for $name {}
            const _: () = assert!(size_of::<$name>() == 4 * [$(stringify!($field)),*].len());
        )*
    };
}

kernel_args! {
    /// `dst = W · x`, or as `mode` says.
    struct MatvecArgs { cols: u32, rows: u32, row_bytes: u32, nsg: u32, mode: u32, ncols: u32 }
    /// `dst = W · x` (or `+=`, as `mode` says) for a batch of `n` inputs.
    struct MatmulArgs { cols: u32, rows: u32, row_bytes: u32, n: u32, mode: u32 }
    /// The tiles of a prompt batch's expert slots (`moe_map`).
    struct MoeMapArgs { n: u32, k: u32, n_expert: u32 }
    /// The expert products of a prompt batch (`matmul_*_id`).
    struct MatmulIdArgs {
        cols: u32, rows: u32, row_bytes: u32, expert_bytes: u32, k: u32, per_slot: u32,
    }
    /// Decode row `row` of a matrix.
    struct EmbedArgs { cols: u32, row: u32, row_bytes: u32, from_slot: u32 }
    struct RmsArgs { n: u32, eps: f32 }
    /// The experts of a slot (`MoeArgs` in `shaders/common.metal`).
    struct MoeArgs { k: u32, per_slot: u32, expert_rows: u32, count: u32 }
    struct RouteArgs { n_expert: u32, k: u32 }
    struct MoeSumArgs { n: u32, k: u32 }
    struct CountArgs { n: u32 }
    struct AttentionArgs {
        n_pos: u32, head_dim: u32, group: u32, kv_dim: u32, q_stride: u32, gated: u32,
        n_rot: u32, norm: u32, chunk: u32, n_chunks: u32, scale: f32, base: f32, eps: f32,
        k_type: u32, v_type: u32, k_row: u32, v_row: u32, rot_k: u32, rot_v: u32,
    }
    /// Attention over a prompt batch (`attention_prompt_*`).
    struct FlashArgs {
        n: u32, n_past: u32, kv_dim: u32, q_stride: u32, gated: u32, n_head: u32, scale: f32,
        group: u32, heads: u32, tiles: u32, k_type: u32, v_type: u32, k_row: u32, v_row: u32,
    }
    struct DeltaArgs {
        n_k_heads: u32, d_k: u32, d_v: u32, key_dim: u32, conv_dim: u32, taps: u32, rows: u32,
        oldest: u32, store: u32, pending: u32, eps: f32, scale: f32,
    }
    struct DeltaBatchArgs {
        n_k_heads: u32, d_k: u32, d_v: u32, key_dim: u32, conv_dim: u32, taps: u32, rows: u32,
        oldest: u32, replay: u32, n: u32, pending: u32, ba_read: u32, ba_write: u32, eps: f32,
        scale: f32,
    }
}

/// `MatvecArgs::mode`: `dst = W · x`, `dst += W · x`, and
/// `dst = silu(W · x) * (W2 · x)` (the `MATVEC_*` macros of the shaders).
pub(crate) const MATVEC_STORE: u32 = 0;
pub(crate) const MATVEC_ADD: u32 = 1;
pub(crate) const MATVEC_SWIGLU: u32 = 2;

/// A matrix-vector kernel and how ggml-metal dispatches it.
#[derive(Debug)]
pub(crate) struct Matvec {
    pipeline: Pipeline,
    /// The mixture-of-experts form of `pipeline` (`{name}_id`).
    pub(crate) id: Pipeline,
    /// Kernels that take several inputs at once (`{name}_c{NC}`), by `NC`,
    /// with their rows (as `nr0`).
    wide: Vec<(usize, Pipeline, usize)>,
    /// Simdgroups per threadgroup (fixed for quantized types; float types
    /// use `min(4, ceil(cols / 128))`).
    nsg: Option<u32>,
    /// Output rows per threadgroup, per simdgroup when `per_simdgroup`.
    /// Always 2, which `MATVEC_SWIGLU` takes as the pair.
    nr0: usize,
    per_simdgroup: bool,
    /// Threadgroup memory in bytes.
    smem: usize,
}

impl Matvec {
    /// The kernel for a product of `ncols` inputs, its rows (as `nr0`) and
    /// the inputs each threadgroup takes: the one-input kernel for one (it
    /// also takes several, one after another), else the widest kernel of at
    /// most 2 inputs for 2 and of at most 4 for more, whose threadgroups
    /// take a group of inputs each (more than 4 are measured faster as
    /// groups of 4 than in fewer, wider threadgroups).
    pub(crate) fn pipeline(&self, ncols: usize) -> (&Pipeline, usize, usize) {
        let want = match ncols {
            0 | 1 => return (&self.pipeline, self.nr0, 1),
            2 => 2,
            _ => 4,
        };
        self.wide
            .iter()
            .rfind(|(nc, ..)| *nc <= want)
            .map_or((&self.pipeline, self.nr0, 1), |(nc, p, nr0)| (p, *nr0, *nc))
    }

    /// Inputs per threadgroup of the kernels for several inputs.
    pub(crate) fn widths(&self) -> Vec<usize> {
        self.wide.iter().map(|(nc, ..)| *nc).collect()
    }

    /// Stops using the kernel for `nc` inputs.
    pub(crate) fn drop_width(&mut self, nc: usize) {
        self.wide.retain(|(n, ..)| *n != nc);
    }

    /// Simdgroups for a product with `cols` columns.
    pub(crate) fn nsg(&self, cols: usize) -> u32 {
        self.nsg
            .unwrap_or_else(|| cols.div_ceil(128).clamp(1, 4) as u32)
    }

    /// Threadgroups and threads per threadgroup for `rows` rows, with
    /// `nr0` rows per threadgroup (per simdgroup for some types), for
    /// `col_groups` groups of inputs.
    pub(crate) fn grid(
        &self,
        rows: usize,
        nsg: u32,
        nr0: usize,
        col_groups: usize,
    ) -> (Size, Size) {
        let per_group = if self.per_simdgroup {
            nr0 * nsg as usize
        } else {
            nr0
        };
        (
            Size::new(rows.div_ceil(per_group), col_groups, 1),
            Size::new(32, nsg as usize, 1),
        )
    }

    /// Threadgroup memory in bytes for `nr0` rows (the reduction of
    /// threadgroups whose simdgroups share rows takes one row of floats
    /// per row).
    pub(crate) fn smem(&self, nr0: usize) -> usize {
        self.smem / self.nr0 * nr0
    }
}

/// `(inputs, rows)` of the matrix-vector kernels that take several inputs at
/// once (`{name}_c{inputs}` in the shaders, instantiated with as many rows).
const WIDE: &[(usize, usize)] = &[(2, 2), (4, 2)];

/// A row of [`MATVECS`].
type MatvecSpec = (GgmlType, &'static str, Option<u32>, usize, bool, usize);

/// `(type, kernel name, simdgroups, rows, rows per simdgroup, smem)`, as
/// `ggml_metal_library_get_pipeline_mul_mv` sets them for one activation
/// row, except for two rows per simdgroup for Q5_K (ggml takes one).
#[rustfmt::skip]
const MATVECS: &[MatvecSpec] = &[
    (GgmlType::F32,  "matvec_f32",  None,    2, false, 32 * 4 * 2),
    (GgmlType::F16,  "matvec_f16",  None,    2, false, 32 * 4 * 2),
    (GgmlType::Bf16, "matvec_bf16", None,    2, false, 32 * 4 * 2),
    (GgmlType::Q8_0, "matvec_q8_0", Some(4), 2, false, 32 * 4 * 2),
    (GgmlType::Q3K, "matvec_q3_K", Some(2), 2, true,  0),
    (GgmlType::Q4K, "matvec_q4_K", Some(2), 2, true,  0),
    (GgmlType::Q5K, "matvec_q5_K", Some(2), 2, true,  0),
    (GgmlType::Q6K, "matvec_q6_K", Some(2), 2, true,  0),
    (GgmlType::Iq4Xs, "matvec_iq4_xs", Some(2), 2, true, 32 * 4),
    (GgmlType::Q4_0, "matvec_q4_0", Some(2), 4, true, 0),
    (GgmlType::Q4_1, "matvec_q4_1", Some(2), 4, true, 0),
    (GgmlType::Q5_0, "matvec_q5_0", Some(2), 4, true, 0),
    (GgmlType::Q5_1, "matvec_q5_1", Some(2), 4, true, 0),
    (GgmlType::Iq4Nl, "matvec_iq4_nl", Some(2), 2, true, 32 * 4),
    (GgmlType::Mxfp4, "matvec_mxfp4", Some(2), 2, true, 32 * 4),
    (GgmlType::Q2K, "matvec_q2_K", Some(2), 4, true, 0),
    // The codebooks some kernels copy to threadgroup memory: 256 or 512
    // grid entries and the 128 sign bytes.
    (GgmlType::Iq2Xxs, "matvec_iq2_xxs", Some(2), 4, true, 256 * 8 + 128),
    (GgmlType::Iq2Xs, "matvec_iq2_xs", Some(2), 4, true, 512 * 8 + 128),
    (GgmlType::Iq2S, "matvec_iq2_s", Some(2), 4, true, 0),
    (GgmlType::Iq3Xxs, "matvec_iq3_xxs", Some(2), 4, true, 256 * 4 + 128),
    (GgmlType::Iq3S, "matvec_iq3_s", Some(2), 4, true, 512 * 4),
    (GgmlType::Iq1S, "matvec_iq1_s", Some(2), 4, true, 0),
    (GgmlType::Iq1M, "matvec_iq1_m", Some(2), 4, true, 0),
];

/// The matrix-matrix kernels (`kernel_mul_mm` in ggml-metal), by weight type.
#[rustfmt::skip]
const MATMULS: &[(GgmlType, &str)] = &[
    (GgmlType::F32, "matmul_f32"),
    (GgmlType::F16, "matmul_f16"),
    (GgmlType::Bf16, "matmul_bf16"),
    (GgmlType::Q4_0, "matmul_q4_0"),
    (GgmlType::Q4_1, "matmul_q4_1"),
    (GgmlType::Q5_0, "matmul_q5_0"),
    (GgmlType::Q5_1, "matmul_q5_1"),
    (GgmlType::Q8_0, "matmul_q8_0"),
    (GgmlType::Mxfp4, "matmul_mxfp4"),
    (GgmlType::Q2K, "matmul_q2_K"),
    (GgmlType::Q3K, "matmul_q3_K"),
    (GgmlType::Q4K, "matmul_q4_K"),
    (GgmlType::Q5K, "matmul_q5_K"),
    (GgmlType::Q6K, "matmul_q6_K"),
    (GgmlType::Iq2Xxs, "matmul_iq2_xxs"),
    (GgmlType::Iq2Xs, "matmul_iq2_xs"),
    (GgmlType::Iq2S, "matmul_iq2_s"),
    (GgmlType::Iq3Xxs, "matmul_iq3_xxs"),
    (GgmlType::Iq3S, "matmul_iq3_s"),
    (GgmlType::Iq1S, "matmul_iq1_s"),
    (GgmlType::Iq1M, "matmul_iq1_m"),
    (GgmlType::Iq4Nl, "matmul_iq4_nl"),
    (GgmlType::Iq4Xs, "matmul_iq4_xs"),
];

/// Weight rows and inputs of a `matmul_*` threadgroup, its threads
/// (4 simdgroups) and its threadgroup memory (`MATMUL_SMEM`).
pub(crate) const MATMUL_ROWS: usize = 64;
pub(crate) const MATMUL_INPUTS: usize = 32;
pub(crate) const MATMUL_THREADS: usize = 128;
pub(crate) const MATMUL_SMEM: usize = 8192;

/// `(type, embedding kernel, values each thread writes)`.
#[rustfmt::skip]
const EMBEDS: &[(GgmlType, &str, usize)] = &[
    (GgmlType::F32,  "embed_f32",  1),
    (GgmlType::F16,  "embed_f16",  1),
    (GgmlType::Bf16, "embed_bf16", 1),
    (GgmlType::Q8_0, "embed_q8_0", 1),
    (GgmlType::Q3K, "embed_q3_K", 16),
    (GgmlType::Q4K, "embed_q4_K", 16),
    (GgmlType::Q5K, "embed_q5_K", 16),
    (GgmlType::Q6K, "embed_q6_K", 16),
    (GgmlType::Iq4Xs, "embed_iq4_xs", 16),
    (GgmlType::Q4_0, "embed_q4_0", 32),
    (GgmlType::Q4_1, "embed_q4_1", 32),
    (GgmlType::Q5_0, "embed_q5_0", 32),
    (GgmlType::Q5_1, "embed_q5_1", 32),
    (GgmlType::Iq4Nl, "embed_iq4_nl", 32),
    (GgmlType::Mxfp4, "embed_mxfp4", 32),
    (GgmlType::Q2K, "embed_q2_K", 8),
    (GgmlType::Iq2Xxs, "embed_iq2_xxs", 8),
    (GgmlType::Iq2Xs, "embed_iq2_xs", 8),
    (GgmlType::Iq2S, "embed_iq2_s", 8),
    (GgmlType::Iq3Xxs, "embed_iq3_xxs", 8),
    (GgmlType::Iq3S, "embed_iq3_s", 8),
    (GgmlType::Iq1S, "embed_iq1_s", 8),
    (GgmlType::Iq1M, "embed_iq1_m", 8),
];

/// The `attention_g{G}_d{D}` kernels: up to `G` query heads per key head,
/// heads of up to `128 * D` values.
const ATTENTIONS: [(usize, usize); 11] = [
    (1, 1),
    (2, 1),
    (4, 1),
    (8, 1),
    (1, 2),
    (2, 2),
    (4, 2),
    (8, 2),
    (1, 4),
    (2, 4),
    (4, 4),
];

/// The head sizes of the `attention_prompt_d{HD}` kernels.
const ATTENTION_PROMPTS: [usize; 3] = [64, 128, 256];

/// Cached positions `attention_prompt_*` reads at a time (`FLASH_C`); a KV
/// cache holds a whole number of them.
pub(crate) const FLASH_C: usize = 32;

/// How `attention_prompt_*` splits `group` query heads per key head of
/// `hd` values: the heads of a threadgroup (a divisor of `group`, at most
/// 4), its tiles of 8 tokens (4 simdgroups in all where the heads allow),
/// its simdgroups, and its threadgroup memory in bytes. A staged kernel also holds a block of dequantized keys or
/// values; where that does not fit, a threadgroup takes fewer tiles, then
/// fewer heads.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PromptAttention {
    pub(crate) heads: usize,
    pub(crate) tiles: usize,
    pub(crate) nsg: usize,
    pub(crate) smem: usize,
}

impl PromptAttention {
    /// The split, if one fits in `max_smem` bytes.
    pub(crate) fn new(group: usize, hd: usize, staged: bool, max_smem: usize) -> Option<Self> {
        let smem = |nsg: usize| {
            nsg * (8 * hd * 2 + (8 * FLASH_C + 64) * 4) + if staged { FLASH_C * hd * 2 } else { 0 }
        };
        let divisors = (1..=group.min(4))
            .rev()
            .filter(|h| group.is_multiple_of(*h));
        for heads in divisors {
            for tiles in (1..=4 / heads).rev() {
                let nsg = heads * tiles;
                if smem(nsg) <= max_smem {
                    return Some(Self {
                        heads,
                        tiles,
                        nsg,
                        smem: smem(nsg),
                    });
                }
            }
        }
        None
    }
}

/// Every compute pipeline of the backend.
#[derive(Debug)]
pub(crate) struct Pipelines {
    matvecs: Vec<(GgmlType, Matvec)>,
    /// `(type, matmul_*, matmul_*_id)`.
    matmuls: Vec<(GgmlType, Pipeline, Pipeline)>,
    embeds: Vec<(GgmlType, Pipeline, usize)>,
    pub(crate) rmsnorm: Pipeline,
    pub(crate) swiglu: Pipeline,
    pub(crate) copy: Pipeline,
    pub(crate) argmax: Pipeline,
    pub(crate) moe_route: Pipeline,
    pub(crate) moe_map: Pipeline,
    pub(crate) moe_combine: Pipeline,
    pub(crate) moe_residual: Pipeline,
    /// `(G, D, pipeline)` from [`ATTENTIONS`].
    attentions: Vec<(usize, usize, Pipeline)>,
    pub(crate) attention_reduce: Pipeline,
    pub(crate) attention_prep: Pipeline,
    /// `(HD, pipeline, staged pipeline)` from [`ATTENTION_PROMPTS`].
    attention_prompts: Vec<(usize, Pipeline, Pipeline)>,
    pub(crate) attention_unrotate: Pipeline,
    pub(crate) delta_net: Pipeline,
    pub(crate) delta_net_batch: Pipeline,
    pub(crate) delta_prep: Pipeline,
    pub(crate) delta_chunk_prep: Pipeline,
    pub(crate) delta_chunk_scan: Pipeline,
    /// `delta_scan_r1_c{CH}`, by `CH`, the float4s of a key per lane.
    delta_scans: Vec<(usize, Pipeline)>,
    pub(crate) delta_norm: Pipeline,
}

impl Pipelines {
    /// Compiles the shaders and creates every pipeline.
    pub(crate) fn new(device: &Device) -> Result<Self, String> {
        let library = device.library(&source())?;
        let get = |name: &str| device.pipeline(&library, name);
        // A weight type whose kernels are missing is unsupported (the ops
        // report it) rather than stopping the backend.
        let matvecs = MATVECS
            .iter()
            .filter_map(|&(ty, name, nsg, nr0, per_simdgroup, smem)| {
                let pipeline = get(name).ok()?;
                let id = get(&format!("{name}_id")).ok()?;
                let wide = WIDE
                    .iter()
                    .filter_map(|&(nc, rows)| Some((nc, get(&format!("{name}_c{nc}")).ok()?, rows)))
                    .collect();
                Some((
                    ty,
                    Matvec {
                        pipeline,
                        id,
                        wide,
                        nsg,
                        nr0,
                        per_simdgroup,
                        smem,
                    },
                ))
            })
            .collect();
        let matmuls = MATMULS
            .iter()
            .filter_map(|&(ty, name)| Some((ty, get(name).ok()?, get(&format!("{name}_id")).ok()?)))
            .collect();
        let embeds = EMBEDS
            .iter()
            .filter_map(|&(ty, name, per_thread)| Some((ty, get(name).ok()?, per_thread)))
            .collect();
        Ok(Self {
            matvecs,
            matmuls,
            embeds,
            rmsnorm: get("rmsnorm")?,
            swiglu: get("swiglu")?,
            copy: get("copy_f32")?,
            argmax: get("argmax")?,
            moe_route: get("moe_route")?,
            moe_map: get("moe_map")?,
            moe_combine: get("moe_combine")?,
            moe_residual: get("moe_residual")?,
            attentions: ATTENTIONS
                .iter()
                .map(|&(g, d)| Ok((g, d, get(&format!("attention_g{g}_d{d}"))?)))
                .collect::<Result<_, String>>()?,
            attention_reduce: get("attention_reduce")?,
            attention_prep: get("attention_prep")?,
            attention_prompts: ATTENTION_PROMPTS
                .iter()
                .map(|&hd| {
                    Ok((
                        hd,
                        get(&format!("attention_prompt_d{hd}"))?,
                        get(&format!("attention_prompt_staged_d{hd}"))?,
                    ))
                })
                .collect::<Result<_, String>>()?,
            attention_unrotate: get("attention_unrotate")?,
            delta_net: get("delta_net")?,
            delta_net_batch: get("delta_net_batch")?,
            delta_prep: get("delta_prep")?,
            delta_chunk_prep: get("delta_chunk_prep")?,
            delta_chunk_scan: get("delta_chunk_scan")?,
            delta_scans: [1, 2, 4]
                .iter()
                .map(|&c| Ok((c, get(&format!("delta_scan_r1_c{c}"))?)))
                .collect::<Result<_, String>>()?,
            delta_norm: get("delta_norm")?,
        })
    }

    /// The matrix-vector kernels, by weight type.
    pub(crate) fn matvecs_mut(&mut self) -> impl Iterator<Item = (GgmlType, &mut Matvec)> {
        self.matvecs.iter_mut().map(|(t, m)| (*t, m))
    }

    /// The matrix-vector kernel for weights of type `ty`.
    pub(crate) fn matvec(&self, ty: GgmlType) -> Option<&Matvec> {
        self.matvecs.iter().find(|(t, _)| *t == ty).map(|(_, m)| m)
    }

    /// The matrix-matrix kernel for weights of type `ty`.
    pub(crate) fn matmul(&self, ty: GgmlType) -> Option<&Pipeline> {
        self.matmuls.iter().find(|m| m.0 == ty).map(|m| &m.1)
    }

    /// The expert matrix-matrix kernel for weights of type `ty`.
    pub(crate) fn matmul_id(&self, ty: GgmlType) -> Option<&Pipeline> {
        self.matmuls.iter().find(|m| m.0 == ty).map(|m| &m.2)
    }

    /// The attention kernel for `group` query heads per key head and heads
    /// of `head_dim` values, and the `G` it was built for.
    pub(crate) fn attention(&self, group: usize, head_dim: usize) -> Option<(&Pipeline, usize)> {
        let g = group.checked_next_power_of_two()?;
        let d = head_dim.div_ceil(128).checked_next_power_of_two()?;
        self.attentions
            .iter()
            .find(|a| (a.0, a.1) == (g, d))
            .map(|a| (&a.2, a.0))
    }

    /// The prompt attention kernel for heads of `head_dim` values.
    pub(crate) fn attention_prompt(&self, head_dim: usize) -> Option<&Pipeline> {
        self.attention_prompts
            .iter()
            .find(|a| a.0 == head_dim)
            .map(|a| &a.1)
    }

    /// The prompt attention kernel for caches of any type.
    pub(crate) fn attention_prompt_staged(&self, head_dim: usize) -> Option<&Pipeline> {
        self.attention_prompts
            .iter()
            .find(|a| a.0 == head_dim)
            .map(|a| &a.2)
    }

    /// The DeltaNet scan for keys of `d_k` values (at most 512).
    pub(crate) fn delta_scan(&self, d_k: usize) -> Option<&Pipeline> {
        let ch = d_k.div_ceil(128).next_power_of_two();
        self.delta_scans
            .iter()
            .find(|(c, _)| *c == ch)
            .map(|(_, p)| p)
    }

    /// The embedding kernel for type `ty` and the values each thread writes.
    pub(crate) fn embed(&self, ty: GgmlType) -> Option<(&Pipeline, usize)> {
        self.embeds
            .iter()
            .find(|(t, ..)| *t == ty)
            .map(|(_, p, n)| (p, *n))
    }
}
