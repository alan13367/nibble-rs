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
            Self::Scalar
        })
    }

    /// Short name for logs: `scalar` or `avx2`.
    #[must_use]
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(_) => "avx2",
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
        }
    }

    /// `a · b` for f16 vectors stored as little-endian bytes.
    #[must_use]
    pub(crate) fn dot_f16(self, a: &[u8], b: &[u8]) -> f32 {
        match self {
            Self::Scalar => scalar::dot_f16(a, b),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.dot_f16(a, b),
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
        }
    }

    /// SiLU of every element.
    pub(crate) fn silu(self, x: &mut [f32]) {
        match self {
            Self::Scalar => scalar::silu_in_place(x),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.silu(x),
        }
    }

    /// `g = silu(g) * u`.
    pub(crate) fn swiglu(self, g: &mut [f32], u: &[f32]) {
        match self {
            Self::Scalar => scalar::swiglu(g, u),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.swiglu(g, u),
        }
    }

    /// Softmax in place.
    pub(crate) fn softmax(self, x: &mut [f32]) {
        match self {
            Self::Scalar => scalar::softmax(x),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.softmax(x),
        }
    }

    /// One token of the gated delta rule for one head (see
    /// [`scalar::delta_net_head`]).
    pub(crate) fn delta_net_head(self, state: &mut [f32], step: &DeltaStep<'_>, out: &mut [f32]) {
        match self {
            Self::Scalar => scalar::delta_net_head(state, step, out),
            #[cfg(target_arch = "x86_64")]
            Self::Avx2(k) => k.delta_net_head(state, step, out),
        }
    }
}
