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
        pad: u32,
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

/// Every compute pipeline of the backend.
#[derive(Debug)]
pub(crate) struct Pipelines {
    matvecs: Vec<(GgmlType, Matvec)>,
    embeds: Vec<(GgmlType, Pipeline, usize)>,
    pub(crate) rmsnorm: Pipeline,
    pub(crate) swiglu: Pipeline,
    pub(crate) copy: Pipeline,
    pub(crate) argmax: Pipeline,
    pub(crate) moe_route: Pipeline,
    pub(crate) moe_combine: Pipeline,
    pub(crate) moe_residual: Pipeline,
    /// `(G, D, pipeline)` from [`ATTENTIONS`].
    attentions: Vec<(usize, usize, Pipeline)>,
    pub(crate) attention_reduce: Pipeline,
    pub(crate) delta_net: Pipeline,
    pub(crate) delta_net_batch: Pipeline,
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
        let embeds = EMBEDS
            .iter()
            .filter_map(|&(ty, name, per_thread)| Some((ty, get(name).ok()?, per_thread)))
            .collect();
        Ok(Self {
            matvecs,
            embeds,
            rmsnorm: get("rmsnorm")?,
            swiglu: get("swiglu")?,
            copy: get("copy_f32")?,
            argmax: get("argmax")?,
            moe_route: get("moe_route")?,
            moe_combine: get("moe_combine")?,
            moe_residual: get("moe_residual")?,
            attentions: ATTENTIONS
                .iter()
                .map(|&(g, d)| Ok((g, d, get(&format!("attention_g{g}_d{d}"))?)))
                .collect::<Result<_, String>>()?,
            attention_reduce: get("attention_reduce")?,
            delta_net: get("delta_net")?,
            delta_net_batch: get("delta_net_batch")?,
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

    /// The embedding kernel for type `ty` and the values each thread writes.
    pub(crate) fn embed(&self, ty: GgmlType) -> Option<(&Pipeline, usize)> {
        self.embeds
            .iter()
            .find(|(t, ..)| *t == ty)
            .map(|(_, p, n)| (p, *n))
    }
}
