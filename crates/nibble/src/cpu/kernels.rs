//! Kernel selection: the scalar reference, or the SIMD kernels this CPU supports.
//!
//! Every CPU op that has a SIMD version goes through [`Kernels`], so this is
//! the one place that knows which implementations exist. A new instruction
//! set adds a variant here, a module beside `avx2`, and a scalar-comparison
//! test for each kernel.

use std::sync::OnceLock;

use nibble_gguf::GgmlType;

#[cfg(target_arch = "x86_64")]
use super::avx2::Avx2;
#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
use super::neon::Neon;
use super::scalar::{self, DeltaStep};
use crate::quant::{self, DotType};

/// Environment variable that forces the scalar kernels when set to
/// `scalar`, to tell a kernel bug from a model bug when comparing with llama.cpp.
const ENV_OVERRIDE: &str = "NIBBLE_KERNELS";

/// The implementation of the vector kernels.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Kernels {
    /// The portable reference in [`scalar`] and [`quant`].
    Scalar,
    /// AVX2 + FMA + F16C.
    #[cfg(target_arch = "x86_64")]
    Avx2(Avx2),
    /// AArch64 NEON. `dotprod` and `fp16` are used when the CPU has them.
    /// Little-endian only: GGUF data is little-endian and the NEON loads are
    /// native-endian, so big-endian AArch64 uses the scalar kernels.
    #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
    Neon(Neon),
}

impl Kernels {
    /// The fastest kernels this CPU supports (detected once per process),
    /// unless `NIBBLE_KERNELS=scalar` is set.
    #[must_use]
    pub(crate) fn detect() -> Self {
        static DETECTED: OnceLock<Kernels> = OnceLock::new();
        *DETECTED.get_or_init(|| {
            if std::env::var_os(ENV_OVERRIDE).is_some_and(|v| v == "scalar") {
                return Self::Scalar;
            }
            #[cfg(target_arch = "x86_64")]
            if let Some(k) = Avx2::detect() {
                return Self::Avx2(k);
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            if let Some(k) = Neon::detect() {
                return Self::Neon(k);
            }
            Self::Scalar
        })
    }

    /// Short name for logs: `scalar`, `avx2` or `neon`.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => "avx2",
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(_) => "neon",
        }
    }

    /// Encodes the activation `x` as `dt` into `out` (exactly
    /// `dt.encoded_len(x.len())` bytes). All implementations produce
    /// identical bytes.
    pub(crate) fn encode(self, dt: DotType, x: &[f32], out: &mut [u8]) {
        match self {
            Self::Scalar => quant::encode_activation(dt, x, out),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.encode(dt, x, out),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.encode(dt, x, out),
        }
    }

    /// `out[r] = row r · xq` for the `out.len()` rows of `row_bytes` bytes
    /// in `rows`, which have type `ty`; `xq` is the activation encoded in
    /// `ty`'s dot type.
    pub(crate) fn dot_rows(
        self,
        ty: GgmlType,
        rows: &[u8],
        row_bytes: usize,
        xq: &[u8],
        out: &mut [f32],
    ) {
        debug_assert_eq!(rows.len(), out.len() * row_bytes);
        match self {
            Self::Scalar => {
                for (o, row) in out.iter_mut().zip(rows.chunks_exact(row_bytes)) {
                    *o = quant::dot_row(ty, row, xq);
                }
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.dot_rows(ty, rows, row_bytes, xq, out),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.dot_rows(ty, rows, row_bytes, xq, out),
        }
    }

    /// [`Kernels::dot_rows`] for `N` encoded inputs at once, one output each,
    /// if this type has a kernel that shares the weights' loads between
    /// them (false otherwise, writing nothing). Each output is what
    /// [`Kernels::dot_rows`] gives for its input.
    pub(crate) fn dot_rows_n<const N: usize>(
        self,
        ty: GgmlType,
        rows: &[u8],
        row_bytes: usize,
        xq: [&[u8]; N],
        out: [&mut [f32]; N],
    ) -> bool {
        match self {
            // No shared-load kernels: the caller takes the inputs one by one.
            Self::Scalar => {
                let _ = (ty, rows, row_bytes, xq, out);
                false
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => {
                let _ = (ty, rows, row_bytes, xq, out);
                false
            }
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.dot_rows_n(ty, rows, row_bytes, xq, out),
        }
    }

    /// `out[r] = a_r · b`, where `a_r` is the f16 vector of `b.len()`
    /// little-endian values at byte `r * stride` of `a` (the cached keys of
    /// a head, or its rows of transposed values) and `b` is f32. Each result
    /// is that of one ggml `ggml_vec_dot_f16_f32` call.
    ///
    /// # Panics
    ///
    /// If a row extends past the end of `a` (an internal invariant).
    pub(crate) fn dot_f16_f32_rows(self, a: &[u8], stride: usize, b: &[f32], out: &mut [f32]) {
        match self {
            Self::Scalar => {
                for (r, o) in out.iter_mut().enumerate() {
                    *o = scalar::dot_f16_f32(&a[r * stride..r * stride + 2 * b.len()], b);
                }
            }
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.dot_f16_f32_rows(a, stride, b, out),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.dot_f16_f32_rows(a, stride, b, out),
        }
    }

    /// Writes `x` as f16 (round to nearest even) to `out`, two bytes per
    /// value. All implementations produce identical bytes.
    pub(crate) fn to_f16(self, x: &[f32], out: &mut [u8]) {
        debug_assert_eq!(out.len(), 2 * x.len());
        match self {
            Self::Scalar => scalar::to_f16(x, out),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.to_f16(x, out),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.to_f16(x, out),
        }
    }

    /// SiLU of every element.
    pub(crate) fn silu(self, x: &mut [f32]) {
        match self {
            Self::Scalar => scalar::silu_in_place(x),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.silu(x),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.silu(x),
        }
    }

    /// `g = silu(g) * u`.
    pub(crate) fn swiglu(self, g: &mut [f32], u: &[f32]) {
        match self {
            Self::Scalar => scalar::swiglu(g, u),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.swiglu(g, u),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.swiglu(g, u),
        }
    }

    /// Softmax in place.
    pub(crate) fn softmax(self, x: &mut [f32]) {
        match self {
            Self::Scalar => scalar::softmax(x),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.softmax(x),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.softmax(x),
        }
    }

    /// One token of the gated delta rule for one head (see
    /// [`scalar::delta_net_head`]).
    pub(crate) fn delta_net_head(self, state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
        match self {
            Self::Scalar => scalar::delta_net_head(state, step, out),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.delta_net_head(state, step, out),
            #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
            Self::Neon(k) => k.delta_net_head(state, step, out),
        }
    }
}
