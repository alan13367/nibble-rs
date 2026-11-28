//! The CUDA backend for NVIDIA GPUs on Windows and Linux.
//!
//! The driver is opened at runtime, model bytes and small parameters are
//! uploaded once, and every activation stays in device memory.
//!
//! The kernels (an offline-built fatbin, see `kernels/kernels.cu`) round as
//! ggml-cuda does when llama.cpp evaluates one token at a time without flash
//! attention (`-ngl 99 -fa off -ub 1`), so the logits match llama.cpp's CUDA
//! build bit for bit: `mul_mat_vec_q` with Q8_1 activations for quantized
//! weights, `mul_mat_vec_f` for F32/F16/BF16 ones, ggml's norms, RoPE (fused
//! with the Q/K norm for NEOX RoPE, as ggml-cuda fuses it), non-flash
//! attention over llama.cpp's padded cache, and its DeltaNet ops. A batch
//! gives every token the arithmetic it gets alone. Quantized KV caches,
//! which llama.cpp only runs with flash attention, use an online softmax.

mod codebooks;
pub(crate) mod device;
mod driver;
mod kernels;
#[cfg(test)]
mod tests;

use std::ops::Range;

use device::{Buffer, Device, Module, Stream};
use driver::DevicePtr;
use kernels::{
    AttentionArgs, CountArgs, DeltaBatchArgs, EmbedArgs, MatvecArgs, MoeArgs, MoeSumArgs,
    Pipelines, RmsArgs, RotateArgs, RouteArgs,
};

use crate::backend::{
    Backend, DeltaNetDims, DeltaNetParams, DeltaNetState, KvCacheType, KvTypes, MAX_LOGIT_ROWS,
    MAX_PENDING, ModelData, Param, QkNorm, Rotation,
};
use crate::cpu::{Experts, Weight};
use crate::error::{Error, Result};

const THREADS: u32 = 256;
/// Batches of more tokens than this multiply quantized weights with the
/// GEMM (as llama.cpp switches from `mul_mat_vec_q` to `mul_mat_q`), whose
/// rounding differs from the one-token kernels; shorter ones, such as drafts,
/// keep every token's one-token result.
const GEMM_MIN_TOKENS: usize = 8;
const MAX_EXPERTS: usize = 4096;
const DELTA_WARPS: u32 = 4;
const DELTA_ROWS: u32 = 4;
const DELTA_GROUP_ROWS: u32 = DELTA_WARPS * DELTA_ROWS;
/// Most bytes of attention scores held at once; longer batches are split.
const SCORE_BYTES: usize = 64 << 20;

/// llama.cpp's n_kv for a token at `pos` in a cache of `cap_pad` (its size
/// rounded up to 256) cells: the used cells rounded up to 256, at least 256.
fn att_n_kv(pos: usize, cap_pad: usize) -> usize {
    (pos + 1).next_multiple_of(256).clamp(256, cap_pad)
}

/// The block size ggml-cuda's `rms_norm_f32` uses for rows of `ncols` floats
/// (`rms_shape` in `kernels/common.cuh`).
fn rms_block_size(ncols: usize) -> u32 {
    match ncols / 4 {
        _ if !ncols.is_multiple_of(4) || ncols == 0 => {
            if ncols < 1024 {
                256
            } else {
                1024
            }
        }
        0..=32 => 32,
        33..=128 => 128,
        129..=256 => 256,
        257..=512 => 512,
        _ => 1024,
    }
}

/// `mul_mat_f`'s warps per block for rows of `ncols` pairs of values.
fn mmf_warps(ncols: usize) -> u32 {
    let (mut best, mut niter_best) = (1, ncols.div_ceil(64));
    for nwarps in 2..=8 {
        let niter = ncols.div_ceil(nwarps * 64);
        if niter < niter_best {
            (best, niter_best) = (nwarps, niter);
        }
    }
    best as u32
}

/// `mul_mat_vec_f`'s block size for rows of `ncols` values.
fn mmvf_block_size(ncols: usize) -> u32 {
    let (mut best, mut niter_best) = (32, ncols.div_ceil(64));
    for bs in (64..=256).step_by(32) {
        let niter = ncols.div_ceil(2 * bs);
        if niter < niter_best {
            (best, niter_best) = (bs, niter);
        }
    }
    best as u32
}

/// Whether attention follows ggml-cuda's path without flash attention
/// (`attention_kq`, `attention_softmax`, `attention_kqv`): caches that
/// llama.cpp runs without flash attention.
fn ggml_attention(kv: KvTypes) -> bool {
    let plain = |t| matches!(t, KvCacheType::F16 | KvCacheType::Bf16 | KvCacheType::F32);
    plain(kv.k) && plain(kv.v)
}

#[derive(Debug)]
pub(crate) struct CudaVector {
    buffer: Buffer,
    len: usize,
}

#[derive(Debug)]
pub(crate) struct CudaKvCache {
    k: Buffer,
    v: Buffer,
    len: usize,
    capacity: usize,
}

#[derive(Clone, Copy, Debug)]
enum Output<'a> {
    Store,
    Add,
    SwiGlu(&'a Weight),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EncodedKey {
    source: u64,
    at: usize,
    cols: usize,
    rows: usize,
}

pub(crate) struct CudaBackend<'m> {
    model: ModelData<'m>,
    device: Device,
    stream: Stream,
    pipelines: Pipelines,
    /// Owns the code behind every `Function` in `pipelines`.
    _module: Module,
    weights: Buffer,
    params: Buffer,
    codebooks: Buffer,
    tokens: Buffer,
    slots: Buffer,
    logits: Buffer,
    q8: Option<(Buffer, usize)>,
    encoded: Option<EncodedKey>,
    /// The GEMM activations (`quantize_mmq`), their size in bytes, and what
    /// they encode.
    mmq: Option<(Buffer, usize)>,
    mmq_encoded: Option<EncodedKey>,
    /// The partial tiles of stream-K GEMMs, and their size in bytes.
    gemm_partial: Option<(Buffer, usize)>,
    /// Per tile, how many stream-K blocks have finished it (`GemmSplitK` in
    /// `kernels/gemm.cu`; zero between GEMMs), and how many tiles it covers.
    gemm_counts: Option<(Buffer, usize)>,
    /// Tokens at the end of a long batch that DeltaNet layers leave pending
    /// (`Backend::set_delta_net_tail`).
    delta_tail: usize,
    sm_count: usize,
    delta_work: Option<Buffer>,
    /// Every token's DeltaNet convolution outputs for a long batch
    /// (`delta_prep`), and its size in bytes.
    delta_prep: Option<(Buffer, usize)>,
    /// Attention scores of a chunk of tokens, and its size in bytes.
    scores: Option<(Buffer, usize)>,
    kv: KvTypes,
    /// The device's compute capability.
    cc: (u32, u32),
    kv_bytes: usize,
    n: usize,
    error: Option<String>,
}

impl std::fmt::Debug for CudaBackend<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBackend")
            .field("device", &self.device.name())
            .finish_non_exhaustive()
    }
}

fn backend_error(e: String) -> Error {
    Error::Backend(format!("CUDA: {e}"))
}

fn u32_of(n: usize, what: &str) -> std::result::Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{what} ({n}) is too large for the CUDA kernels"))
}

fn floats_as_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no padding and every bit pattern is valid to copy as bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), 4 * values.len()) }
}

fn f32_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: as `f32_bytes`; the returned bytes cover exactly the live floats.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), 4 * values.len()) }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    // SAFETY: `u32` has no padding and every bit pattern is valid to copy as bytes.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), 4 * values.len()) }
}

fn u32_bytes_mut(values: &mut [u32]) -> &mut [u8] {
    // SAFETY: as `u32_bytes`; the returned bytes cover exactly the live integers.
    unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<u8>(), 4 * values.len()) }
}

const fn kv_code(ty: KvCacheType) -> u32 {
    match ty {
        KvCacheType::F16 => 0,
        KvCacheType::F32 => 1,
        KvCacheType::Bf16 => 2,
        KvCacheType::Q8_0 => 3,
        KvCacheType::Q4_0 => 4,
        KvCacheType::Q4_1 => 5,
        KvCacheType::Iq4Nl => 6,
        KvCacheType::Q5_0 => 7,
        KvCacheType::Q5_1 => 8,
    }
}

/// Builds CUDA kernel parameters: each value is a local variable holding a
/// device pointer or a plain `repr(C)` scalar struct.
macro_rules! launch {
    ($stream:expr, $function:expr, $grid:expr, $block:expr, $shared:expr, $($arg:ident),* $(,)?) => {{
        let mut params = [$(std::ptr::from_mut(&mut $arg).cast::<std::ffi::c_void>()),*];
        // SAFETY: every argument is a live local of the exact kernel ABI type;
        // the backend owns all device allocations and its module, and
        // synchronizes the stream before dropping them.
        unsafe { $stream.launch($function, $grid, $block, $shared, &mut params) }
    }};
}

impl<'m> CudaBackend<'m> {
    pub(crate) fn new(model: ModelData<'m>, n_ctx: usize, kv: KvTypes) -> Result<Self> {
        let device = Device::new(0).map_err(backend_error)?;
        let (major, minor) = device.compute_capability();
        if (major, minor) < (7, 5) {
            return Err(backend_error(format!(
                "{} has compute capability {major}.{minor}; nibble's CUDA kernels need 7.5 or newer",
                device.name()
            )));
        }
        let sm_count = device.sm_count() as usize;
        let module = device.module(kernels::FATBIN).map_err(backend_error)?;
        let pipelines = Pipelines::new(&module, major).map_err(backend_error)?;
        if (major, minor) >= (8, 0) {
            for (head_dim, f) in &pipelines.fattn {
                module
                    .allow_shared(f, kernels::fattn_shared(*head_dim))
                    .map_err(backend_error)?;
            }
            for (f, bytes) in pipelines.gemms() {
                module
                    .allow_shared(f, *bytes as usize)
                    .map_err(backend_error)?;
            }
        }
        let stream = device.stream().map_err(backend_error)?;
        let weights = device.buffer_with(model.bytes).map_err(backend_error)?;
        let params = device
            .buffer_with(&floats_as_bytes(model.params))
            .map_err(backend_error)?;
        let codebooks = device
            .buffer_with(&codebooks::bytes())
            .map_err(backend_error)?;
        let tokens = device.buffer(4 * n_ctx.max(1)).map_err(backend_error)?;
        let slots = device.buffer(4 * MAX_LOGIT_ROWS).map_err(backend_error)?;
        let logits = device
            .buffer(4 * MAX_LOGIT_ROWS * model.config.n_vocab)
            .map_err(backend_error)?;
        let c = model.config;
        if c.head_dim == 0
            || c.head_dim > 512
            || c.rope_dims > c.head_dim
            || c.n_head_kv == 0
            || !c.n_head.is_multiple_of(c.n_head_kv)
        {
            return Err(backend_error(format!(
                "no attention kernel for {} query heads of {} values per key head",
                c.n_head / c.n_head_kv.max(1),
                c.head_dim
            )));
        }
        Ok(Self {
            model,
            device,
            stream,
            pipelines,
            _module: module,
            weights,
            params,
            codebooks,
            tokens,
            slots,
            logits,
            q8: None,
            encoded: None,
            mmq: None,
            mmq_encoded: None,
            gemm_partial: None,
            gemm_counts: None,
            delta_tail: 0,
            sm_count,
            delta_work: None,
            delta_prep: None,
            scores: None,
            kv,
            cc: (major, minor),
            kv_bytes: 0,
            n: 0,
            error: None,
        })
    }

    pub(crate) fn device_name(&self) -> &str {
        self.device.name()
    }

    fn sync(&mut self) -> Result<()> {
        self.stream.synchronize().map_err(backend_error)
    }

    fn fail(&mut self, message: String) {
        self.error.get_or_insert(message);
    }

    fn record(&mut self, result: std::result::Result<(), String>) {
        if let Err(e) = result {
            self.fail(e);
        }
    }

    fn result(&mut self) -> Result<()> {
        self.error.take().map_or(Ok(()), |e| Err(backend_error(e)))
    }

    fn weight_ptr(&self, w: &Weight) -> std::result::Result<u64, String> {
        let len = w
            .rows
            .checked_mul(w.row_bytes)
            .ok_or("a weight's byte length overflows")?;
        let end = w
            .offset
            .checked_add(len)
            .ok_or("a weight's byte offset overflows")?;
        if end > self.model.bytes.len() {
            return Err(format!(
                "a weight at byte {} is outside the model",
                w.offset
            ));
        }
        self.weights.offset(w.offset)
    }

    fn param_ptr(&self, p: Param) -> std::result::Result<u64, String> {
        self.params.offset(4 * p.range().start)
    }

    fn invalidate_encoded(&mut self, dst: u64) {
        if self.encoded.is_some_and(|e| e.source == dst) {
            self.encoded = None;
        }
        if self.mmq_encoded.is_some_and(|e| e.source == dst) {
            self.mmq_encoded = None;
        }
    }

    /// The activations `x` (`rows` tokens of `cols` values, from float `at`)
    /// as the GEMM reads them, and the padded token count of their layout.
    fn ensure_mmq(
        &mut self,
        x: &Buffer,
        at: usize,
        cols: usize,
        rows: usize,
    ) -> Option<(u64, u32)> {
        let ystride = rows.next_multiple_of(kernels::GEMM_TOKEN_PAD);
        let chunks = cols.div_ceil(256) * 2;
        let key = EncodedKey {
            source: x.ptr(),
            at,
            cols,
            rows,
        };
        let sizes = (
            chunks
                .checked_mul(ystride)
                .and_then(|n| n.checked_mul(kernels::GEMM_CHUNK_BYTES)),
            u32::try_from(ystride),
            u32::try_from(chunks),
            u32::try_from(cols),
            u32::try_from(rows),
        );
        let (Some(bytes), Ok(stride), Ok(nchunks), Ok(ncols), Ok(ntok)) = sizes else {
            self.fail("the GEMM activation size overflows".to_owned());
            return None;
        };
        if self.mmq_encoded == Some(key) {
            return self.mmq.as_ref().map(|(b, _)| (b.ptr(), stride));
        }
        if self.mmq.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
            if let Err(e) = self.sync() {
                self.fail(e.to_string());
                return None;
            }
            match self.device.buffer(bytes) {
                Ok(b) => self.mmq = Some((b, bytes)),
                Err(e) => {
                    self.fail(e);
                    return None;
                }
            }
        }
        let mut out = self.mmq.as_ref().map(|(b, _)| b.ptr())?;
        let mut x = match x.offset(at) {
            Ok(x) => x,
            Err(e) => {
                self.fail(e);
                return None;
            }
        };
        let (mut ncols, mut ntok, mut stride_arg) = (ncols, ntok, stride);
        let function = &self.pipelines.quantize_mmq;
        self.record(launch!(
            self.stream,
            function,
            [stride, nchunks, 1],
            [128, 1, 1],
            0,
            x,
            out,
            ncols,
            ntok,
            stride_arg
        ));
        self.mmq_encoded = Some(key);
        Some((out, stride))
    }

    fn ensure_q8(&mut self, x: &Buffer, at: usize, cols: usize, rows: usize) -> Option<u64> {
        let key = EncodedKey {
            source: x.ptr(),
            at,
            cols,
            rows,
        };
        if self.encoded == Some(key) {
            return self.q8.as_ref().map(|(b, _)| b.ptr());
        }
        let blocks = match cols.checked_mul(rows).and_then(|n| n.checked_div(32)) {
            Some(n) => n,
            None => {
                self.fail("the Q8_1 activation size overflows".to_owned());
                return None;
            }
        };
        let bytes = match blocks.checked_mul(36) {
            Some(n) => n,
            None => {
                self.fail("the Q8_1 activation size overflows".to_owned());
                return None;
            }
        };
        if self.q8.as_ref().is_none_or(|(_, cap)| *cap < bytes) {
            if let Err(e) = self.sync() {
                self.fail(e.to_string());
                return None;
            }
            match self.device.buffer(bytes) {
                Ok(b) => self.q8 = Some((b, bytes)),
                Err(e) => {
                    self.fail(e);
                    return None;
                }
            }
        }
        let out = self.q8.as_ref().map(|(b, _)| b.ptr())?;
        let n = match u32_of(cols * rows, "an activation") {
            Ok(n) => n,
            Err(e) => {
                self.fail(e);
                return None;
            }
        };
        let mut x = match x.offset(at) {
            Ok(x) => x,
            Err(e) => {
                self.fail(e);
                return None;
            }
        };
        let mut out = out;
        let mut n = n;
        let function = &self.pipelines.quantize_q8_1;
        self.record(launch!(
            self.stream,
            function,
            [n.div_ceil(128), 1, 1],
            [32, 4, 1],
            0,
            x,
            out,
            n
        ));
        self.encoded = Some(key);
        Some(out)
    }

    fn encode_matvec(
        &mut self,
        w: &Weight,
        x: &Buffer,
        x_at: usize,
        out: DevicePtr,
        output: Output<'_>,
        ncols: usize,
    ) {
        if ncols == 0 {
            return;
        }
        let Some(k) = self.pipelines.get(w.ty).copied() else {
            return self.fail(format!("no CUDA kernel for {} weights", w.ty));
        };
        let args = match (
            u32_of(w.cols, "a matrix width"),
            u32_of(w.rows, "a matrix height"),
            u32_of(w.row_bytes, "a matrix row"),
            u32_of(ncols, "a batch"),
        ) {
            (Ok(cols), Ok(rows), Ok(row_bytes), Ok(n)) => MatvecArgs {
                cols,
                rows,
                row_bytes,
                ncols: n,
                mode: match output {
                    Output::Store => kernels::MATVEC_STORE,
                    Output::Add => kernels::MATVEC_ADD,
                    Output::SwiGlu(_) => kernels::MATVEC_SWIGLU,
                },
            },
            (Err(e), _, _, _) | (_, Err(e), _, _) | (_, _, Err(e), _) | (_, _, _, Err(e)) => {
                return self.fail(e);
            }
        };
        let mut wptr = match self.weight_ptr(w) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        if let (Some(gemms), false) = (
            self.gemm_for(&k, ncols),
            matches!(output, Output::SwiGlu(_)),
        ) {
            let Some((mut xq, mut ystride)) = self.ensure_mmq(x, x_at, w.cols, ncols) else {
                return;
            };
            // The smallest tile that covers the batch, up to 128 tokens.
            let tile = kernels::GEMM_TOKENS
                .iter()
                .position(|&t| ncols <= t)
                .unwrap_or(kernels::GEMM_TOKENS.len() - 1);
            let tokens = kernels::GEMM_TOKENS[tile];
            let mut out_ptr = out;
            let mut args = args;
            let (tiles_n, tiles_m) = (ncols.div_ceil(tokens), w.rows.div_ceil(kernels::GEMM_ROWS));
            let stages = w.cols.div_ceil(256);
            let (mut sk, grid) = self.gemm_split(tiles_n, tiles_m, stages);
            let (mut partial, mut counts) = (0, 0);
            if sk.blocks > 0 {
                // Two partial tiles per block, and a count per tile.
                let bytes = 2 * (grid[0] * grid[1]) as usize * kernels::GEMM_ROWS * tokens * 4;
                let tiles = tiles_n * tiles_m;
                if self
                    .gemm_partial
                    .as_ref()
                    .is_none_or(|(_, cap)| *cap < bytes)
                    || self
                        .gemm_counts
                        .as_ref()
                        .is_none_or(|(_, cap)| *cap < tiles)
                {
                    if let Err(e) = self.sync() {
                        return self.fail(e.to_string());
                    }
                    // New counts start at zero; each GEMM leaves them so.
                    match (self.device.buffer(bytes), self.device.buffer(4 * tiles)) {
                        (Ok(p), Ok(c)) => {
                            self.gemm_partial = Some((p, bytes));
                            self.gemm_counts = Some((c, tiles));
                        }
                        (Err(e), _) | (_, Err(e)) => return self.fail(e),
                    }
                }
                partial = self.gemm_partial.as_ref().map_or(0, |(b, _)| b.ptr());
                counts = self.gemm_counts.as_ref().map_or(0, |(b, _)| b.ptr());
            }
            let (gemm, shared) = &gemms[tile];
            self.record(launch!(
                self.stream,
                gemm,
                grid,
                [256, 1, 1],
                *shared,
                wptr,
                xq,
                out_ptr,
                args,
                ystride,
                sk,
                partial,
                counts
            ));
            self.invalidate_encoded(out);
            return;
        }
        let up = match output {
            Output::SwiGlu(up) => up,
            _ => w,
        };
        if matches!(output, Output::SwiGlu(_))
            && (up.ty, up.cols, up.rows, up.row_bytes) != (w.ty, w.cols, w.rows, w.row_bytes)
        {
            return self.fail("the SwiGLU products differ in type or shape".to_owned());
        }
        let mut w2 = match self.weight_ptr(up) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out_ptr = out;
        let (_, nwarps, rpb) = k.shape;
        let grid = [
            w.rows.div_ceil(rpb as usize) as u32,
            ncols.div_ceil(if ncols > 1 { 4 } else { 1 }) as u32,
            1,
        ];
        let block = if kernels::is_quantized(w.ty) {
            [32, nwarps, 1]
        } else {
            [mmvf_block_size(w.cols), 1, 1]
        };
        if kernels::is_quantized(w.ty) {
            let Some(mut xq) = self.ensure_q8(x, x_at, w.cols, ncols) else {
                return;
            };
            let mut args = args;
            let mut cb = self.codebooks.ptr();
            let function = if ncols > 1 { &k.matvec_c4 } else { &k.matvec };
            self.record(launch!(
                self.stream,
                function,
                grid,
                block,
                0,
                wptr,
                w2,
                xq,
                out_ptr,
                args,
                cb
            ));
        } else {
            let mut xf = match x.offset(x_at) {
                Ok(p) => p,
                Err(e) => return self.fail(e),
            };
            let mut args = args;
            let function = if ncols > 1 { &k.matvec_c4 } else { &k.matvec };
            self.record(launch!(
                self.stream,
                function,
                grid,
                block,
                0,
                wptr,
                w2,
                xf,
                out_ptr,
                args
            ));
        }
        self.invalidate_encoded(out);
    }

    #[expect(clippy::too_many_arguments, reason = "one expert dispatch")]
    fn encode_matvec_id(
        &mut self,
        experts: &Experts,
        up: Option<&Experts>,
        x: &Buffer,
        x_rows: usize,
        out: &Buffer,
        route: &Buffer,
        k: usize,
        per_slot: bool,
        output: Output<'_>,
    ) {
        let f = experts.first;
        let Some(kernels) = self.pipelines.get(f.ty).copied() else {
            return self.fail(format!("no CUDA expert kernel for {} weights", f.ty));
        };
        let slots = self.n * k;
        let args = match (
            u32_of(f.cols, "a matrix width"),
            u32_of(f.rows, "a matrix height"),
            u32_of(f.row_bytes, "a matrix row"),
            u32_of(slots, "a batch"),
            u32_of(k, "an expert count"),
            u32_of(experts.count, "an expert count"),
        ) {
            (Ok(cols), Ok(rows), Ok(row_bytes), Ok(ncols), Ok(k), Ok(count)) => (
                MatvecArgs {
                    cols,
                    rows,
                    row_bytes,
                    ncols,
                    mode: match output {
                        Output::Store => kernels::MATVEC_STORE,
                        Output::Add => kernels::MATVEC_ADD,
                        Output::SwiGlu(_) => kernels::MATVEC_SWIGLU,
                    },
                },
                MoeArgs {
                    k,
                    per_slot: u32::from(per_slot),
                    expert_rows: rows,
                    count,
                },
            ),
            _ => return self.fail("expert dimensions exceed the CUDA kernels".to_owned()),
        };
        let mut w = match self.weight_ptr(&experts.all()) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut w2 = match self.weight_ptr(&up.unwrap_or(experts).all()) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out_ptr = out.ptr();
        let mut route_ptr = route.ptr();
        let (_, nwarps, rpb) = kernels.shape;
        let grid = [f.rows.div_ceil(rpb as usize) as u32, slots as u32, 1];
        let block = if kernels::is_quantized(f.ty) {
            [32, nwarps, 1]
        } else {
            [mmvf_block_size(f.cols), 1, 1]
        };
        if kernels::is_quantized(f.ty) {
            let Some(mut xq) = self.ensure_q8(x, 0, f.cols, x_rows) else {
                return;
            };
            let (mut args, mut moe) = args;
            let mut cb = self.codebooks.ptr();
            self.record(launch!(
                self.stream,
                &kernels.matvec_id,
                grid,
                block,
                0,
                w,
                w2,
                xq,
                out_ptr,
                route_ptr,
                args,
                moe,
                cb
            ));
        } else {
            let mut xf = x.ptr();
            let (mut args, mut moe) = args;
            self.record(launch!(
                self.stream,
                &kernels.matvec_id,
                grid,
                block,
                0,
                w,
                w2,
                xf,
                out_ptr,
                route_ptr,
                args,
                moe
            ));
        }
        self.invalidate_encoded(out.ptr());
    }

    fn elementwise(&mut self, function: &device::Function, n: usize, buffers: &[&Buffer]) {
        let mut n = match u32_of(n, "a vector length") {
            Ok(n) => n,
            Err(e) => return self.fail(e),
        };
        let grid = [(n as usize).div_ceil(THREADS as usize) as u32, 1, 1];
        let result = match buffers {
            [a, b] => {
                let mut a = a.ptr();
                let mut b = b.ptr();
                launch!(self.stream, function, grid, [THREADS, 1, 1], 0, a, b, n)
            }
            _ => return self.fail("an element-wise kernel has the wrong buffers".to_owned()),
        };
        self.record(result);
        for b in buffers {
            self.invalidate_encoded(b.ptr());
        }
    }

    /// `rows` rows of `out = rmsnorm(x) * w`; the caller invalidates `out`.
    fn encode_rms(&mut self, x: DevicePtr, w: Param, eps: f32, out: DevicePtr, rows: usize) {
        let mut args = RmsArgs {
            n: w.len() as u32,
            eps,
        };
        let mut x = x;
        let mut norm = match self.param_ptr(w) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out_ptr = out;
        self.record(launch!(
            self.stream,
            &self.pipelines.rmsnorm,
            [rows as u32, 1, 1],
            [rms_block_size(w.len()), 1, 1],
            0,
            x,
            norm,
            out_ptr,
            args
        ));
    }

    fn encode_rotate(&mut self, x: &Buffer, r: Rotation, inverse: bool, out: &Buffer) {
        let order = r.heads.unwrap_or(crate::backend::HeadOrder {
            head_dim: 0,
            n_k: 0,
            rep: 0,
        });
        let mut args = RotateArgs {
            len: r.len as u32,
            block: r.block as u32,
            signs: u32::from(r.signs.is_some()),
            head_dim: order.head_dim as u32,
            n_k: order.n_k as u32,
            rep: order.rep as u32,
            inverse: u32::from(inverse),
        };
        let mut xptr = x.ptr();
        let mut signs = match r.signs.map_or(Ok(0), |p| self.param_ptr(p)) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out_ptr = out.ptr();
        self.record(launch!(
            self.stream,
            &self.pipelines.rotate,
            [(r.len / r.block) as u32, self.n as u32, 1],
            [(r.block / 2).clamp(1, THREADS as usize) as u32, 1, 1],
            (4 * r.block) as u32,
            xptr,
            signs,
            out_ptr,
            args
        ));
        self.invalidate_encoded(out.ptr());
    }

    /// Whether ggml-cuda computes KQ with `mul_mat_f` (tensor cores) once
    /// n_kv exceeds 512 (`ggml_cuda_should_use_mmvf` turns it down for F16
    /// and BF16 rows of more than 512 with several KV heads, and
    /// `ggml_cuda_should_use_mmf` takes it): an F16 cache from Turing on, a
    /// BF16 one from Ampere on, heads of a multiple of 64 values. Other heads
    /// would go to cuBLAS, which nibble does not reproduce.
    fn kq_mma(&self, k: KvCacheType) -> bool {
        let c = self.model.config;
        let arch = match k {
            KvCacheType::F16 => (7, 5),
            KvCacheType::Bf16 => (8, 0),
            _ => return false,
        };
        self.cc >= arch && c.n_head_kv > 1 && c.head_dim.is_multiple_of(64)
    }

    /// `out = silu(gate · x) * (up · x)` for the batch with the fused
    /// GEMMs `glu`; `gate` and `up` have the same type and shape.
    fn encode_glu_gemm(
        &mut self,
        glu: &[(device::Function, u32); 3],
        gate: &Weight,
        up: &Weight,
        x: &CudaVector,
        out: &mut CudaVector,
    ) {
        let ncols = self.n;
        let mut args = match (
            u32_of(gate.cols, "a matrix width"),
            u32_of(gate.rows, "a matrix height"),
            u32_of(gate.row_bytes, "a matrix row"),
            u32_of(ncols, "a batch"),
        ) {
            (Ok(cols), Ok(rows), Ok(row_bytes), Ok(n)) => MatvecArgs {
                cols,
                rows,
                row_bytes,
                ncols: n,
                mode: kernels::MATVEC_STORE,
            },
            (Err(e), _, _, _) | (_, Err(e), _, _) | (_, _, Err(e), _) | (_, _, _, Err(e)) => {
                return self.fail(e);
            }
        };
        let (mut gptr, mut uptr) = match (self.weight_ptr(gate), self.weight_ptr(up)) {
            (Ok(g), Ok(u)) => (g, u),
            (Err(e), _) | (_, Err(e)) => return self.fail(e),
        };
        let Some((mut xq, mut ystride)) = self.ensure_mmq(&x.buffer, 0, gate.cols, ncols) else {
            return;
        };
        let tile = kernels::GEMM_TOKENS
            .iter()
            .position(|&t| ncols <= t)
            .unwrap_or(kernels::GEMM_TOKENS.len() - 1);
        let grid = [
            ncols.div_ceil(kernels::GEMM_TOKENS[tile]) as u32,
            gate.rows.div_ceil(kernels::GEMM_ROWS / 2) as u32,
            1,
        ];
        let mut out_ptr = out.buffer.ptr();
        let (f, shared) = &glu[tile];
        self.record(launch!(
            self.stream,
            f,
            grid,
            [256, 1, 1],
            *shared,
            gptr,
            uptr,
            xq,
            out_ptr,
            args,
            ystride
        ));
        self.invalidate_encoded(out_ptr);
    }

    /// How a GEMM of `tiles_n` by `tiles_m` tiles of `stages` stages runs:
    /// one block per tile when that is expected to be faster, else stream-K
    /// over the row tiles with one block per SM in all (each holds a whole
    /// SM's registers).
    fn gemm_split(
        &self,
        tiles_n: usize,
        tiles_m: usize,
        stages: usize,
    ) -> (kernels::GemmSplitK, [u32; 3]) {
        let tiles = tiles_n * tiles_m;
        let sms = self.sm_count.max(1);
        let waves = tiles.div_ceil(sms);
        let tiled = (
            kernels::GemmSplitK {
                blocks: 0,
                tiles_n: 0,
                tiles: 0,
                stages: 0,
            },
            [tiles_n as u32, tiles_m as u32, 1],
        );
        // In stages per SM: tiling runs whole waves of tiles; stream-K runs
        // an even share, plus about two stages for the partial tiles
        // (measured on Qwen3 0.6B and Qwen3.5 0.8B and 4B).
        let sk_cost = (tiles * stages).div_ceil(sms) + 2;
        if waves * stages <= sk_cost || tiles_n > sms {
            return tiled;
        }
        let blocks = (sms / tiles_n).min(tiles_m * stages);
        match (
            u32::try_from(blocks),
            u32::try_from(tiles_n),
            u32::try_from(tiles_m),
            u32::try_from(stages),
            u32::try_from(tiles_m as u64 * stages as u64),
        ) {
            (Ok(blocks), Ok(n), Ok(m), Ok(stages), Ok(_)) => (
                kernels::GemmSplitK {
                    blocks,
                    tiles_n: n,
                    tiles: m,
                    stages,
                },
                [n, blocks, 1],
            ),
            _ => tiled,
        }
    }

    /// The GEMM for a batch of `ncols` tokens of a weight with kernels `k`,
    /// if the batch takes the GEMM path (its integer MMAs need Ampere).
    fn gemm_for(
        &self,
        k: &kernels::TypeKernels,
        ncols: usize,
    ) -> Option<[(device::Function, u32); 3]> {
        (ncols > GEMM_MIN_TOKENS && self.cc >= (8, 0))
            .then_some(k.gemm)
            .flatten()
    }

    /// Whether long batches of these DeltaNet sizes run `delta_seq`, which
    /// can leave a tail pending.
    fn delta_seq_supported(&self, d: DeltaNetDims) -> bool {
        Self::delta_supported(d) && self.pipelines.delta_seq.iter().any(|(dk, _)| *dk == d.d_k)
    }

    fn delta_supported(d: DeltaNetDims) -> bool {
        d.d_k.is_multiple_of(8) && d.d_k <= 512 && d.conv_kernel > 0
    }
}

impl Drop for CudaBackend<'_> {
    fn drop(&mut self) {
        let _ = self.sync();
    }
}

impl Backend for CudaBackend<'_> {
    type Vector = CudaVector;
    type KvCache = CudaKvCache;

    fn name(&self) -> &'static str {
        "cuda"
    }

    fn vector(&mut self, len: usize) -> Result<CudaVector> {
        let bytes = len
            .checked_mul(4)
            .ok_or_else(|| backend_error("a vector's byte length overflows".to_owned()))?;
        Ok(CudaVector {
            buffer: self.device.buffer(bytes).map_err(backend_error)?,
            len,
        })
    }

    fn kv_cache(&mut self, n_ctx: usize) -> Result<CudaKvCache> {
        let c = self.model.config;
        let kv_dim = c.n_head_kv * c.head_dim;
        let k_bytes = self.kv.k.bytes(kv_dim).checked_mul(n_ctx);
        let v_bytes = self.kv.v.bytes(kv_dim).checked_mul(n_ctx);
        let (Some(k_bytes), Some(v_bytes)) = (k_bytes, v_bytes) else {
            return Err(backend_error("a KV cache's size overflows".to_owned()));
        };
        let k = self.device.buffer(k_bytes).map_err(backend_error)?;
        let v = self.device.buffer(v_bytes).map_err(backend_error)?;
        self.kv_bytes += k_bytes + v_bytes;
        Ok(CudaKvCache {
            k,
            v,
            len: 0,
            capacity: n_ctx,
        })
    }

    fn kv_bytes(&self) -> usize {
        self.kv_bytes
    }

    fn check_delta_net(&self, d: DeltaNetDims) -> Result<()> {
        if Self::delta_supported(d) {
            Ok(())
        } else {
            Err(backend_error(format!(
                "no DeltaNet kernel for heads of {} keys",
                d.d_k
            )))
        }
    }

    fn delta_net_tail(&self, d: DeltaNetDims) -> bool {
        self.delta_seq_supported(d)
    }

    fn set_delta_net_tail(&mut self, tail: usize) {
        self.delta_tail = tail;
    }

    fn check_moe(&self) -> Result<()> {
        Ok(())
    }

    fn check_rotation(&self, r: Rotation) -> Result<()> {
        if r.block.is_power_of_two() && r.block <= 8192 && u32::try_from(r.len).is_ok() {
            Ok(())
        } else {
            Err(backend_error(format!(
                "no kernel for a Hadamard rotation of {} values in blocks of {}",
                r.len, r.block
            )))
        }
    }

    fn kv_len(&self, cache: &CudaKvCache) -> usize {
        cache.len
    }

    fn kv_truncate(&mut self, cache: &mut CudaKvCache, n_pos: usize) -> Result<()> {
        cache.len = cache.len.min(n_pos);
        Ok(())
    }

    fn read(&mut self, v: &CudaVector, out: &mut [f32]) -> Result<()> {
        self.sync()?;
        v.buffer
            .download(0, f32_bytes_mut(&mut out[..v.len]))
            .map_err(backend_error)
    }

    fn write(&mut self, v: &mut CudaVector, data: &[f32]) -> Result<()> {
        self.sync()?;
        self.invalidate_encoded(v.buffer.ptr());
        v.buffer
            .upload(0, f32_bytes(&data[..v.len]))
            .map_err(backend_error)
    }

    fn begin(&mut self, _pos: usize, n: usize) -> Result<()> {
        self.n = n;
        self.error = None;
        self.encoded = None;
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        self.result()
    }

    fn logits(
        &mut self,
        w: &Weight,
        x: &CudaVector,
        rows: Range<usize>,
        out: &mut [f32],
    ) -> Result<()> {
        if w.rows > self.model.config.n_vocab || rows.len() > MAX_LOGIT_ROWS {
            self.fail(format!(
                "{} rows of {} logits do not fit the logits buffer",
                rows.len(),
                w.rows
            ));
        } else {
            self.encode_matvec(
                w,
                &x.buffer,
                4 * rows.start * w.cols,
                self.logits.ptr(),
                Output::Store,
                rows.len(),
            );
        }
        self.result()?;
        self.sync()?;
        self.logits
            .download(0, f32_bytes_mut(&mut out[..rows.len() * w.rows]))
            .map_err(backend_error)
    }

    fn embed(&mut self, w: &Weight, tokens: &[u32], x: &mut CudaVector) {
        let Some(k) = self.pipelines.get(w.ty).copied() else {
            return self.fail(format!("no embedding kernel for {} weights", w.ty));
        };
        // On the backend's stream, after earlier embeddings that read the
        // buffer and before the kernel below.
        // SAFETY: `tokens` and `self.tokens` outlive the copy: the stream is
        // synchronized before `tokens` is released.
        let copied = unsafe {
            self.stream
                .memcpy_htod_async(&self.tokens, 0, u32_bytes(tokens))
        };
        if let Err(e) = copied.and_then(|()| self.stream.synchronize()) {
            return self.fail(e);
        }
        let mut wptr = match self.weight_ptr(w) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out = x.buffer.ptr();
        let mut token_ptr = self.tokens.ptr();
        let mut args = EmbedArgs {
            cols: w.cols as u32,
            row: 0,
            row_bytes: w.row_bytes as u32,
            from_slot: 1,
        };
        let function = &k.embed;
        let grid = [(w.cols / k.shape.0 as usize) as u32, tokens.len() as u32, 1];
        if kernels::is_quantized(w.ty) {
            let mut cb = self.codebooks.ptr();
            self.record(launch!(
                self.stream,
                function,
                grid,
                [128, 1, 1],
                0,
                wptr,
                out,
                token_ptr,
                args,
                cb
            ));
        } else {
            self.record(launch!(
                self.stream,
                function,
                grid,
                [128, 1, 1],
                0,
                wptr,
                out,
                token_ptr,
                args
            ));
        }
        self.invalidate_encoded(x.buffer.ptr());
    }

    fn embed_slot(&mut self, w: &Weight, slot: usize, x: &mut CudaVector) {
        let Some(k) = self.pipelines.get(w.ty).copied() else {
            return self.fail(format!("no embedding kernel for {} weights", w.ty));
        };
        if slot >= MAX_LOGIT_ROWS {
            return;
        }
        let mut wptr = match self.weight_ptr(w) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out = x.buffer.ptr();
        let mut token_ptr = match self.slots.offset(4 * slot) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut args = EmbedArgs {
            cols: w.cols as u32,
            row: 0,
            row_bytes: w.row_bytes as u32,
            from_slot: 1,
        };
        let mut cb = self.codebooks.ptr();
        let function = &k.embed;
        if kernels::is_quantized(w.ty) {
            self.record(launch!(
                self.stream,
                function,
                [(w.cols / k.shape.0 as usize) as u32, 1, 1],
                [128, 1, 1],
                0,
                wptr,
                out,
                token_ptr,
                args,
                cb
            ));
        } else {
            self.record(launch!(
                self.stream,
                function,
                [(w.cols / k.shape.0 as usize) as u32, 1, 1],
                [128, 1, 1],
                0,
                wptr,
                out,
                token_ptr,
                args
            ));
        }
        self.invalidate_encoded(x.buffer.ptr());
    }

    fn argmax(&mut self, x: &CudaVector, len: usize, slot: usize) {
        if slot >= MAX_LOGIT_ROWS {
            return;
        }
        let mut xptr = x.buffer.ptr();
        let mut slots = match self.slots.offset(4 * slot) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut args = CountArgs { n: len as u32 };
        self.record(launch!(
            self.stream,
            &self.pipelines.argmax,
            [1, 1, 1],
            [1024, 1, 1],
            0,
            xptr,
            slots,
            args
        ));
    }

    fn read_slots(&mut self, out: &mut [u32]) -> Result<()> {
        self.sync()?;
        let n = out.len().min(MAX_LOGIT_ROWS);
        self.slots
            .download(0, u32_bytes_mut(&mut out[..n]))
            .map_err(backend_error)
    }

    fn rotate(&mut self, x: &CudaVector, r: Rotation, out: &mut CudaVector) {
        self.encode_rotate(&x.buffer, r, false, &out.buffer);
    }

    fn unrotate(&mut self, x: &mut CudaVector, r: Rotation) {
        self.encode_rotate(&x.buffer, r, true, &x.buffer);
    }

    fn rmsnorm(&mut self, x: &CudaVector, w: Param, eps: f32, out: &mut CudaVector) {
        if self.n > 0 {
            self.encode_rms(x.buffer.ptr(), w, eps, out.buffer.ptr(), self.n);
            self.invalidate_encoded(out.buffer.ptr());
        }
    }

    fn norm_pair(
        &mut self,
        a: &CudaVector,
        wa: Param,
        b: &CudaVector,
        wb: Param,
        eps: f32,
        out: &mut CudaVector,
    ) {
        let (da, db) = (wa.len(), wb.len());
        for t in 0..self.n {
            let at = |v: &CudaVector, i: usize| v.buffer.offset(4 * i);
            match (
                at(a, t * da),
                at(b, t * db),
                at(out, t * (da + db)),
                at(out, t * (da + db) + da),
            ) {
                (Ok(xa), Ok(xb), Ok(oa), Ok(ob)) => {
                    self.encode_rms(xa, wa, eps, oa, 1);
                    self.encode_rms(xb, wb, eps, ob, 1);
                }
                (Err(e), ..) | (_, Err(e), ..) | (.., Err(e), _) | (.., Err(e)) => {
                    return self.fail(e);
                }
            }
        }
        self.invalidate_encoded(out.buffer.ptr());
    }

    fn copy_rows(
        &mut self,
        src: &CudaVector,
        from: usize,
        dst: &mut CudaVector,
        to: usize,
        rows: usize,
        len: usize,
    ) {
        let n = rows * len;
        if n == 0 {
            return;
        }
        let mut args = CountArgs { n: n as u32 };
        let mut src_ptr = match src.buffer.offset(4 * from * len) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut dst_ptr = match dst.buffer.offset(4 * to * len) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        self.record(launch!(
            self.stream,
            &self.pipelines.copy,
            [n.div_ceil(THREADS as usize) as u32, 1, 1],
            [THREADS, 1, 1],
            0,
            src_ptr,
            dst_ptr,
            args
        ));
        self.invalidate_encoded(dst.buffer.ptr());
    }

    fn residual(
        &mut self,
        w: &Weight,
        a: &CudaVector,
        x: &mut CudaVector,
        norm: Param,
        eps: f32,
        out: &mut CudaVector,
    ) {
        self.encode_matvec(w, &a.buffer, 0, x.buffer.ptr(), Output::Add, self.n);
        self.rmsnorm(x, norm, eps, out);
    }

    fn matvecs<const N: usize>(
        &mut self,
        x: &CudaVector,
        products: [(&Weight, &mut CudaVector); N],
    ) {
        for (w, out) in products {
            self.encode_matvec(w, &x.buffer, 0, out.buffer.ptr(), Output::Store, self.n);
        }
    }

    fn swiglu(
        &mut self,
        gate: &Weight,
        up: &Weight,
        x: &CudaVector,
        out: &mut CudaVector,
        up_out: &mut CudaVector,
    ) {
        let gemm = self
            .pipelines
            .get(gate.ty)
            .copied()
            .and_then(|k| self.gemm_for(&k, self.n).and(k.gemm_glu));
        let same = (gate.ty, gate.cols, gate.rows, gate.row_bytes)
            == (up.ty, up.cols, up.rows, up.row_bytes);
        if let (Some(glu), true) = (gemm, same) {
            self.encode_glu_gemm(&glu, gate, up, x, out);
        } else if gemm.is_none() && same {
            self.encode_matvec(
                gate,
                &x.buffer,
                0,
                out.buffer.ptr(),
                Output::SwiGlu(up),
                self.n,
            );
        } else {
            self.encode_matvec(gate, &x.buffer, 0, out.buffer.ptr(), Output::Store, self.n);
            self.encode_matvec(up, &x.buffer, 0, up_out.buffer.ptr(), Output::Store, self.n);
            let function = self.pipelines.swiglu;
            self.elementwise(
                &function,
                self.n * gate.rows,
                &[&out.buffer, &up_out.buffer],
            );
        }
    }

    fn moe_route(
        &mut self,
        logits: &CudaVector,
        n_expert: usize,
        k: usize,
        route: &mut CudaVector,
    ) {
        if n_expert > MAX_EXPERTS || self.n == 0 {
            if n_expert > MAX_EXPERTS {
                self.fail(format!("{n_expert} experts is more than {MAX_EXPERTS}"));
            }
            return;
        }
        let mut logits = logits.buffer.ptr();
        let mut out = route.buffer.ptr();
        let mut args = RouteArgs {
            n_expert: n_expert as u32,
            k: k as u32,
        };
        self.record(launch!(
            self.stream,
            &self.pipelines.moe_route,
            [self.n as u32, 1, 1],
            [32, 1, 1],
            (4 * n_expert.next_multiple_of(4)) as u32,
            logits,
            out,
            args
        ));
    }

    fn moe_swiglu(
        &mut self,
        gate: &Experts,
        up: &Experts,
        x: &CudaVector,
        route: &CudaVector,
        k: usize,
        out: &mut CudaVector,
        up_out: &mut CudaVector,
    ) {
        if (
            gate.first.ty,
            gate.first.cols,
            gate.first.rows,
            gate.first.row_bytes,
        ) == (
            up.first.ty,
            up.first.cols,
            up.first.rows,
            up.first.row_bytes,
        ) {
            self.encode_matvec_id(
                gate,
                Some(up),
                &x.buffer,
                self.n,
                &out.buffer,
                &route.buffer,
                k,
                false,
                Output::SwiGlu(&up.first),
            );
        } else {
            self.encode_matvec_id(
                gate,
                None,
                &x.buffer,
                self.n,
                &out.buffer,
                &route.buffer,
                k,
                false,
                Output::Store,
            );
            self.encode_matvec_id(
                up,
                None,
                &x.buffer,
                self.n,
                &up_out.buffer,
                &route.buffer,
                k,
                false,
                Output::Store,
            );
            let function = self.pipelines.swiglu;
            self.elementwise(
                &function,
                self.n * k * gate.first.rows,
                &[&out.buffer, &up_out.buffer],
            );
        }
    }

    fn moe_down(
        &mut self,
        down: &Experts,
        hidden: &CudaVector,
        route: &CudaVector,
        k: usize,
        products: &mut CudaVector,
        out: &mut CudaVector,
    ) {
        self.encode_matvec_id(
            down,
            None,
            &hidden.buffer,
            self.n * k,
            &products.buffer,
            &route.buffer,
            k,
            true,
            Output::Store,
        );
        let e = down.first.rows;
        let mut products_ptr = products.buffer.ptr();
        let mut route_ptr = route.buffer.ptr();
        let mut out_ptr = out.buffer.ptr();
        let mut args = MoeSumArgs {
            n: e as u32,
            k: k as u32,
            eps: 0.0,
        };
        self.record(launch!(
            self.stream,
            &self.pipelines.moe_combine,
            [e.div_ceil(THREADS as usize) as u32, self.n as u32, 1],
            [THREADS, 1, 1],
            0,
            products_ptr,
            route_ptr,
            out_ptr,
            args
        ));
        self.invalidate_encoded(out.buffer.ptr());
    }

    fn moe_residual(
        &mut self,
        moe: &CudaVector,
        shared: &CudaVector,
        gate: &CudaVector,
        x: &mut CudaVector,
        norm: Param,
        eps: f32,
        out: &mut CudaVector,
    ) {
        let e = norm.len();
        let mut moe = moe.buffer.ptr();
        let mut shared = shared.buffer.ptr();
        let mut gate = gate.buffer.ptr();
        let mut xptr = x.buffer.ptr();
        let mut norm_ptr = match self.param_ptr(norm) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut out_ptr = out.buffer.ptr();
        let mut args = MoeSumArgs {
            n: e as u32,
            k: 0,
            eps,
        };
        self.record(launch!(
            self.stream,
            &self.pipelines.moe_residual,
            [self.n as u32, 1, 1],
            [rms_block_size(e).max(THREADS), 1, 1],
            0,
            moe,
            shared,
            gate,
            xptr,
            norm_ptr,
            out_ptr,
            args
        ));
        self.invalidate_encoded(out.buffer.ptr());
        self.invalidate_encoded(x.buffer.ptr());
    }

    fn attention(
        &mut self,
        q: &mut CudaVector,
        k: &mut CudaVector,
        v: &CudaVector,
        norm: QkNorm,
        cache: &mut CudaKvCache,
        out: &mut CudaVector,
    ) {
        let c = self.model.config;
        let hd = c.head_dim;
        let n = self.n;
        let q_stride = if norm.gated { 2 * hd } else { hd };
        if cache.len + n > cache.capacity {
            return self.fail(format!("KV cache of {} positions is full", cache.capacity));
        }
        let (mut q_norm, mut k_norm) = match norm.weights {
            Some((q, k)) => match (self.param_ptr(q), self.param_ptr(k)) {
                (Ok(q), Ok(k)) => (q, k),
                (Err(e), _) | (_, Err(e)) => return self.fail(e),
            },
            None => (self.params.ptr(), self.params.ptr()),
        };
        let kv = self.kv;
        let mut args = AttentionArgs {
            n_pos: cache.len as u32,
            n: n as u32,
            head_dim: hd as u32,
            group: (c.n_head / c.n_head_kv) as u32,
            kv_dim: (c.n_head_kv * hd) as u32,
            q_stride: q_stride as u32,
            gated: u32::from(norm.gated),
            n_rot: c.rope_dims as u32,
            norm: u32::from(norm.weights.is_some()),
            scale: 1.0 / (hd as f32).sqrt(),
            // ggml's theta_scale, computed on the host as llama.cpp does.
            base: c.rope_base.powf(-2.0 / c.rope_dims as f32),
            eps: norm.eps,
            k_type: kv_code(kv.k),
            v_type: kv_code(kv.v),
            k_row: kv.k.bytes(c.n_head_kv * hd) as u32,
            v_row: kv.v.bytes(c.n_head_kv * hd) as u32,
            rot_k: kv.rot_k(hd).unwrap_or(0) as u32,
            rot_v: kv.rot_v(hd).unwrap_or(0) as u32,
            n_head: c.n_head as u32,
            t0: 0,
            n_kv_max: 0,
            cap_pad: cache.capacity.next_multiple_of(256) as u32,
            multi_rope: u32::from(norm.multi_rope),
            kq_mma: u32::from(self.kq_mma(kv.k)),
            n_cells: cache.capacity as u32,
            v_trans: u32::from(ggml_attention(kv)),
        };
        let mut qptr = q.buffer.ptr();
        let mut kptr = k.buffer.ptr();
        let mut vptr = v.buffer.ptr();
        let mut k_cache = cache.k.ptr();
        let mut v_cache = cache.v.ptr();
        self.record(launch!(
            self.stream,
            &self.pipelines.attention_prep,
            [(c.n_head + c.n_head_kv) as u32, n as u32, 1],
            [THREADS, 1, 1],
            (4 * hd) as u32,
            qptr,
            kptr,
            vptr,
            q_norm,
            k_norm,
            k_cache,
            v_cache,
            args
        ));
        let mut out_ptr = out.buffer.ptr();
        let fattn = self
            .pipelines
            .fattn
            .iter()
            .find(|(d, _)| *d == hd)
            .map(|(_, f)| *f)
            .filter(|_| {
                // It copies rows of V^T 16 bytes at a time.
                n > GEMM_MIN_TOKENS
                    && self.cc >= (8, 0)
                    && cache.capacity.is_multiple_of(8)
                    && (kv.k, kv.v) == (KvCacheType::F16, KvCacheType::F16)
            });
        if let Some(f) = fattn {
            self.record(launch!(
                self.stream,
                &f,
                [c.n_head as u32, n.div_ceil(64) as u32, 1],
                [128, 1, 1],
                kernels::fattn_shared(hd) as u32,
                qptr,
                k_cache,
                v_cache,
                out_ptr,
                args
            ));
        } else if ggml_attention(kv) && n > 0 {
            let cap_pad = cache.capacity.next_multiple_of(256);
            let n_kv_max = att_n_kv(cache.len + n - 1, cap_pad);
            let row_bytes = 4 * c.n_head * n_kv_max;
            let chunk = (SCORE_BYTES / row_bytes).clamp(1, n);
            let bytes = chunk * row_bytes;
            if self.scores.as_ref().is_none_or(|(_, have)| *have < bytes) {
                if let Err(e) = self.sync() {
                    return self.fail(e.to_string());
                }
                match self.device.buffer(bytes) {
                    Ok(b) => self.scores = Some((b, bytes)),
                    Err(e) => return self.fail(e),
                }
            }
            let mut scores = self.scores.as_ref().map_or(0, |(b, _)| b.ptr());
            args.n_kv_max = n_kv_max as u32;
            let softmax_block = n_kv_max.next_power_of_two().clamp(32, 1024) as u32;
            for t0 in (0..n).step_by(chunk) {
                let m = chunk.min(n - t0) as u32;
                let mut a = args;
                a.t0 = t0 as u32;
                if a.kq_mma != 0 && n_kv_max > 512 {
                    let nwarps = mmf_warps(hd / 2);
                    let shared =
                        (nwarps as usize * 16 * 36).max(8 * (nwarps as usize * 32 + 4)) * 4;
                    self.record(launch!(
                        self.stream,
                        &self.pipelines.attention_kq_mma,
                        [(n_kv_max / 32) as u32, c.n_head as u32, m],
                        [32, nwarps, 1],
                        shared as u32,
                        qptr,
                        k_cache,
                        scores,
                        a
                    ));
                }
                self.record(launch!(
                    self.stream,
                    &self.pipelines.attention_kq,
                    [n_kv_max as u32, c.n_head as u32, m],
                    [mmvf_block_size(hd), 1, 1],
                    0,
                    qptr,
                    k_cache,
                    scores,
                    a
                ));
                self.record(launch!(
                    self.stream,
                    &self.pipelines.attention_softmax,
                    [c.n_head as u32, m, 1],
                    [softmax_block, 1, 1],
                    0,
                    scores,
                    a
                ));
                self.record(launch!(
                    self.stream,
                    &self.pipelines.attention_kqv,
                    [hd as u32, c.n_head as u32, m],
                    [256, 1, 1],
                    0,
                    scores,
                    v_cache,
                    qptr,
                    out_ptr,
                    a
                ));
            }
        } else {
            self.record(launch!(
                self.stream,
                &self.pipelines.attention,
                [c.n_head as u32, n as u32, 1],
                [THREADS, 1, 1],
                0,
                qptr,
                k_cache,
                v_cache,
                out_ptr,
                args
            ));
        }
        cache.len += n;
        self.invalidate_encoded(out.buffer.ptr());
        self.invalidate_encoded(q.buffer.ptr());
        self.invalidate_encoded(k.buffer.ptr());
    }

    fn delta_net(
        &mut self,
        d: DeltaNetDims,
        p: DeltaNetParams,
        eps: f32,
        state: &mut DeltaNetState<CudaVector>,
        qkv: &CudaVector,
        _conv: &mut CudaVector,
        z: &mut CudaVector,
        beta: &mut CudaVector,
        alpha: &mut CudaVector,
        out: &mut CudaVector,
    ) {
        let n = self.n;
        if !Self::delta_supported(d)
            || state.conv.len != d.conv_len()
            || state.pending_ba.len != d.pending_len()
            || qkv.len < n * d.conv_dim()
            || out.len < n * d.value_dim()
            || p.norm.len() != d.d_v
        {
            return self.fail(format!("unsupported DeltaNet head size {}", d.d_k));
        }
        let pending = (2..=MAX_PENDING).contains(&n);
        let tail = if pending { 0 } else { self.delta_tail };
        if tail > MAX_PENDING || (tail > 0 && n < tail + d.conv_kernel - 1) {
            return self.fail(format!("a pending tail of {tail} in a batch of {n}"));
        }
        if pending && self.delta_work.is_none() {
            match self.device.buffer(4 * d.state_len()) {
                Ok(work) => self.delta_work = Some(work),
                Err(e) => return self.fail(e),
            }
        }
        let r = state.keep;
        let ba_write = if pending || tail > 0 {
            (state.pending_row + MAX_PENDING) % (2 * MAX_PENDING)
        } else {
            0
        };
        let mut args = DeltaBatchArgs {
            n_k_heads: d.n_k_heads as u32,
            d_k: d.d_k as u32,
            d_v: d.d_v as u32,
            key_dim: d.key_dim() as u32,
            conv_dim: d.conv_dim() as u32,
            taps: d.conv_kernel as u32,
            rows: d.conv_rows() as u32,
            oldest: state.oldest as u32,
            replay: r as u32,
            n: n as u32,
            pending: u32::from(pending),
            ba_read: state.pending_row as u32,
            ba_write: ba_write as u32,
            eps,
            scale: 1.0 / (d.d_v as f32).sqrt(),
            // llama.cpp's L2 norm: rms_norm with eps / d_k, scaled by 1 / sqrt(d_k).
            l2_eps: eps / d.d_k as f32,
            l2_scale: 1.0 / (d.d_k as f32).sqrt(),
            prepped: 0,
            tail: tail as u32,
        };
        let mut xptr = qkv.buffer.ptr();
        let mut ring = state.conv.buffer.ptr();
        let mut taps = match self.param_ptr(p.conv) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut beta_ptr = beta.buffer.ptr();
        let mut alpha_ptr = alpha.buffer.ptr();
        let mut dt_bias = match self.param_ptr(p.dt_bias) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut a_log = match self.param_ptr(p.a) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut ssm = state.ssm.buffer.ptr();
        let mut out_ptr = out.buffer.ptr();
        let state_out = if pending {
            self.delta_work.as_ref().unwrap().ptr()
        } else {
            ssm
        };
        let mut state_out_ptr = state_out;
        let mut pending_ba = state.pending_ba.buffer.ptr();
        let mut commit = ssm;
        // A long batch computes every token's convolution and norms in
        // parallel first (same arithmetic); the sequential kernel then only
        // applies the delta rule.
        let mut prep = 0;
        let mut gates = 0;
        let mut seq = None;
        if !pending && n > GEMM_MIN_TOKENS {
            let steps = r + n;
            let bytes = 4 * steps * (d.conv_dim() + 2 * d.n_v_heads);
            if self
                .delta_prep
                .as_ref()
                .is_none_or(|(_, have)| *have < bytes)
            {
                if let Err(e) = self.sync() {
                    return self.fail(e.to_string());
                }
                match self.device.buffer(bytes) {
                    Ok(b) => self.delta_prep = Some((b, bytes)),
                    Err(e) => return self.fail(e),
                }
            }
            prep = self.delta_prep.as_ref().map_or(0, |(b, _)| b.ptr());
            gates = prep + (4 * steps * d.conv_dim()) as u64;
            args.prepped = 1;
            let value_blocks = d.value_dim().div_ceil(128);
            self.record(launch!(
                self.stream,
                &self.pipelines.delta_prep,
                [steps as u32, (d.n_k_heads + value_blocks + 1) as u32, 1],
                [128, 1, 1],
                (8 * d.d_k) as u32,
                xptr,
                ring,
                taps,
                beta_ptr,
                alpha_ptr,
                dt_bias,
                a_log,
                pending_ba,
                prep,
                gates,
                args
            ));
            seq = self
                .pipelines
                .delta_seq
                .iter()
                .find(|(dk, _)| *dk == d.d_k)
                .map(|(_, f)| *f);
        }
        if tail > 0 && seq.is_none() {
            return self.fail(format!("no pending tail for DeltaNet heads of {}", d.d_k));
        }
        if let Some(f) = seq {
            self.record(launch!(
                self.stream,
                &f,
                [
                    d.n_v_heads as u32,
                    d.d_v.div_ceil(DELTA_GROUP_ROWS as usize) as u32,
                    1
                ],
                [32 * DELTA_WARPS, 1, 1],
                0,
                prep,
                gates,
                ssm,
                out_ptr,
                args
            ));
        } else {
            self.record(launch!(
                self.stream,
                &self.pipelines.delta_net_batch,
                [
                    d.n_v_heads as u32,
                    d.d_v.div_ceil(DELTA_GROUP_ROWS as usize) as u32,
                    1
                ],
                [32 * DELTA_WARPS, 1, 1],
                (4 * (2 * d.d_k + DELTA_GROUP_ROWS as usize)) as u32,
                xptr,
                ring,
                taps,
                beta_ptr,
                alpha_ptr,
                dt_bias,
                a_log,
                ssm,
                out_ptr,
                state_out_ptr,
                pending_ba,
                commit,
                prep,
                args
            ));
        }
        let mut norm_ptr = match self.param_ptr(p.norm) {
            Ok(p) => p,
            Err(e) => return self.fail(e),
        };
        let mut zptr = z.buffer.ptr();
        self.record(launch!(
            self.stream,
            &self.pipelines.delta_norm,
            [d.n_v_heads as u32, n as u32, 1],
            [128, 1, 1],
            (4 * d.d_v) as u32,
            out_ptr,
            zptr,
            norm_ptr,
            args
        ));
        // The convolution keeps its window before the first pending token,
        // then the pending tokens' inputs.
        let first = if pending {
            0
        } else {
            n.saturating_sub(d.conv_kernel - 1 + tail)
        };
        let slot = (state.oldest + d.conv_kernel - 1 + r + first) % d.conv_rows();
        let count = n - first;
        let before_wrap = count.min(d.conv_rows() - slot);
        let cd = d.conv_dim();
        self.copy_rows(qkv, first, &mut state.conv, slot, before_wrap, cd);
        self.copy_rows(
            qkv,
            first + before_wrap,
            &mut state.conv,
            0,
            count - before_wrap,
            cd,
        );
        if pending {
            state.oldest = (state.oldest + r) % d.conv_rows();
            state.pending = n;
            state.keep = n;
            state.pending_row = ba_write;
        } else {
            state.oldest = (state.oldest + r + n - tail) % d.conv_rows();
            state.pending = tail;
            state.keep = tail;
            state.pending_row = if tail > 0 { ba_write } else { 0 };
        }
        self.invalidate_encoded(out.buffer.ptr());
    }
}
