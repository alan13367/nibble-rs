//! The Metal backend for Apple GPUs.
//!
//! - [`objc`] loads the Objective-C runtime and the Metal framework at run
//!   time; nothing is linked, so the binary still runs on the CPU where
//!   Metal is missing.
//! - [`device`] wraps the few Metal objects used: the device and queue,
//!   pipelines, shared buffers, command buffers.
//! - [`kernels`] holds the MSL source (compiled when a session starts), the
//!   kernels' argument structs and their dispatch geometry.
//!
//! The ops of [`Backend`] encode dispatches into a compute encoder,
//! separated by memory barriers where one reads what another wrote. A batch
//! is a few command buffers: the first is committed after a few dispatches,
//! so the GPU starts while the rest of the batch is encoded, and the last
//! when the batch ends. Batches whose logits are not needed are not waited
//! for, so a prompt streams to the GPU. For now an op over a batch encodes
//! the dispatch of each token in turn, at that token's rows of the vectors:
//! each token gets exactly its result in a batch of its own. The model file's memory map is
//! wrapped in a buffer without a copy; activations are f32, and the KV
//! cache is f16.
//!
//! Decoding one token is a chain of dependent dispatches, each waiting for
//! the last (4–6 µs each on an M4 Max, which bounds small models more than
//! bandwidth), so the ops fuse what they can: a residual add into the
//! product that makes it, the SwiGLU into the gate and up products, an
//! attention step into one dispatch (two past 64 positions), and a DeltaNet
//! step into two.

pub(crate) mod device;
mod kernels;
pub(crate) mod objc;
#[cfg(test)]
mod tests;
mod weights;

use std::collections::VecDeque;
use std::ops::Range;

use device::{Buffer, Commands, Device, Size};
use kernels::{
    AttentionArgs, CountArgs, DeltaArgs, DeltaBatchArgs, EmbedArgs, MatvecArgs, Pipelines, RmsArgs,
};

use crate::backend::{
    Backend, DeltaNetDims, DeltaNetParams, DeltaNetState, MAX_LOGIT_ROWS, MAX_PENDING, ModelData,
    Param, QkNorm,
};
use crate::cpu::Weight;
use crate::error::{Error, Result};
use weights::Weights;

/// Threads per threadgroup of the element-wise and norm kernels.
const THREADS: usize = 256;

/// Simdgroups per threadgroup of `delta_net`, and the state rows the
/// threadgroup updates (`DELTA_ROWS` in the shader per simdgroup).
const DELTA_SIMDGROUPS: usize = 4;
const DELTA_GROUP_ROWS: usize = DELTA_SIMDGROUPS * 4;

/// Simdgroups per threadgroup of `attention` (`ATT_SIMDGROUPS` in the shader).
const ATTENTION_SIMDGROUPS: usize = 8;
/// Threadgroups `attention` aims for, and the positions each takes: a power
/// of two between the bounds (capped further by threadgroup memory).
const ATTENTION_GROUPS: usize = 512;
const ATTENTION_MIN_CHUNK: usize = 32;
const ATTENTION_MAX_CHUNK: usize = 256;

/// How one attention layer is dispatched.
#[derive(Clone, Copy, Debug)]
struct AttentionPlan {
    /// The `G` of the kernel: query heads per key head, rounded up.
    g: usize,
    /// The fewest and most positions a chunk may take (the most limited by
    /// threadgroup memory).
    min_chunk: usize,
    max_chunk: usize,
}

impl AttentionPlan {
    /// Positions per chunk for `n_pos` positions and `n_kv` key heads.
    fn chunk(self, n_pos: usize, n_kv: usize) -> usize {
        (n_pos * n_kv)
            .div_ceil(ATTENTION_GROUPS)
            .next_power_of_two()
            .clamp(self.min_chunk, self.max_chunk)
    }

    /// Threadgroup memory, in floats, for chunks of `chunk` positions.
    fn floats(self, head_dim: usize, chunk: usize) -> usize {
        (self.g + 3 + ATTENTION_SIMDGROUPS) * head_dim + self.g * chunk
    }
}

/// Dispatches after which a token's first commands are committed, so the GPU
/// starts while the CPU encodes the rest, and after which later ones are.
const FIRST_COMMIT: usize = 8;
const DISPATCHES_PER_COMMIT: usize = 64;

/// Committed command buffers kept to check, at most: a queue holds 64
/// unfinished ones by default.
const MAX_IN_FLIGHT: usize = 64;

/// A vector of `f32` in a shared buffer.
#[derive(Debug)]
pub(crate) struct MetalVector {
    buffer: Buffer,
    len: usize,
}

/// One attention layer's f16 keys and values, both by position.
#[derive(Debug)]
pub(crate) struct MetalKvCache {
    k: Buffer,
    v: Buffer,
    len: usize,
    capacity: usize,
}

/// What a matrix-vector dispatch writes.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Output<'a> {
    /// `out = w · x`.
    Store,
    /// `out += w · x`.
    Add,
    /// `out = silu(w · x) * (up · x)`, for `up` of the same type and shape.
    SwiGlu(&'a Weight),
}

/// The buffers of one token of a DeltaNet layer, with their byte offsets.
struct DeltaStep<'a> {
    /// The token's input (the projection of q, k and v).
    x: (&'a Buffer, usize),
    beta: (&'a Buffer, usize),
    alpha: (&'a Buffer, usize),
    /// Where the state is read from and written to.
    ssm: (&'a Buffer, &'a Buffer),
    out: (&'a Buffer, usize),
    /// The output gate, for the gated norm.
    z: (&'a Buffer, usize),
}

/// The Metal backend of one session.
pub(crate) struct MetalBackend<'m> {
    model: ModelData<'m>,
    device: Device,
    pipelines: Pipelines,
    weights: Weights<'m>,
    /// The model's parameter arena.
    params: Buffer,
    /// The token slots of [`Backend::argmax`], one `u32` each.
    slots: Buffer,
    logits: Buffer,
    /// Each attention chunk's output, maximum score and sum of exponentials.
    partials: Buffer,
    /// One DeltaNet layer's state, which a batch that stays pending runs
    /// on; allocated by the first batch that needs it.
    delta_work: Option<Buffer>,
    attention: AttentionPlan,
    n_ctx: usize,
    /// Tokens in the batch being encoded.
    n: usize,
    /// The token being encoded.
    commands: Option<Commands>,
    /// Dispatches encoded into `commands`.
    dispatches: usize,
    /// Command buffers of this token committed so far.
    chunks: usize,
    /// Committed commands, oldest first, until they are known to have
    /// finished. Command buffers run in the order they are committed.
    in_flight: VecDeque<Commands>,
    /// The first failure of committed commands, reported by the next op
    /// that returns a result.
    gpu_error: Option<String>,
    /// The first op of this token that could not be encoded.
    error: Option<String>,
}

impl std::fmt::Debug for MetalBackend<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBackend")
            .field("device", &self.device.name())
            .finish_non_exhaustive()
    }
}

fn backend_error(e: String) -> Error {
    Error::Backend(format!("Metal: {e}"))
}

fn u32_of(n: usize, what: &str) -> std::result::Result<u32, String> {
    u32::try_from(n).map_err(|_| format!("{what} ({n}) is too large for the Metal kernels"))
}

/// Copies `data` into the start of `buffer`.
///
/// # Safety
///
/// No command buffer that uses `buffer` may be running.
unsafe fn store(buffer: &mut Buffer, data: &[f32]) {
    // SAFETY: guaranteed by the caller.
    let bytes = unsafe { buffer.bytes_mut() };
    for (out, v) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(data) {
        *out = v.to_ne_bytes();
    }
}

/// Copies the start of `buffer` into `out`.
///
/// # Safety
///
/// No command buffer that writes `buffer` may be running.
unsafe fn load(buffer: &Buffer, out: &mut [f32]) {
    // SAFETY: guaranteed by the caller.
    let bytes = unsafe { buffer.bytes() };
    for (o, v) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *o = f32::from_ne_bytes(*v);
    }
}

impl<'m> MetalBackend<'m> {
    /// A backend on the system's default GPU for sessions of up to `n_ctx`
    /// positions of `model`.
    pub(crate) fn new(model: ModelData<'m>, n_ctx: usize) -> Result<Self> {
        let device = Device::system_default().map_err(backend_error)?;
        let mut pipelines = Pipelines::new(&device).map_err(backend_error)?;
        check_wide_matvecs(&device, &mut pipelines).map_err(backend_error)?;
        let max_buffer = std::env::var(weights::ENV_MAX_BUFFER)
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(device.max_buffer_length(), |cap: usize| {
                cap.min(device.max_buffer_length())
            });
        let weights = Weights::new(&device, model, max_buffer).map_err(backend_error)?;
        let floats = |n: usize| device.buffer(4 * n).map_err(backend_error);
        let mut params = floats(model.params.len())?;
        // SAFETY: the buffer is new; no command uses it yet.
        unsafe { store(&mut params, model.params) };
        let c = model.config;
        let logits = floats(MAX_LOGIT_ROWS * c.n_vocab)?;
        let slots = floats(MAX_LOGIT_ROWS)?;
        // Checked here so that `Auto` falls back to the CPU for a model the
        // kernels do not cover.
        let attention = Self::attention_plan(&device, &pipelines, c).ok_or_else(|| {
            backend_error(format!(
                "no attention kernel for {} query heads of {} values per key head",
                c.n_head / c.n_head_kv.max(1),
                c.head_dim
            ))
        })?;
        let max_chunks = (1..=n_ctx)
            .map(|n| n.div_ceil(attention.chunk(n, c.n_head_kv)))
            .max()
            .unwrap_or(1);
        let partials = floats(c.n_head * max_chunks * (c.head_dim + 2))?;
        Ok(Self {
            model,
            pipelines,
            weights,
            params,
            logits,
            slots,
            partials,
            delta_work: None,
            attention,
            n_ctx,
            n: 0,
            commands: None,
            dispatches: 0,
            chunks: 0,
            in_flight: VecDeque::with_capacity(MAX_IN_FLIGHT),
            gpu_error: None,
            error: None,
            device,
        })
    }

    /// How attention layers of `c` are dispatched, if the kernels cover them.
    fn attention_plan(
        device: &Device,
        pipelines: &Pipelines,
        c: &crate::model::ModelConfig,
    ) -> Option<AttentionPlan> {
        if c.head_dim == 0
            || !c.head_dim.is_multiple_of(4)
            || c.rope_dims > c.head_dim
            || c.n_head_kv == 0
            || !c.n_head.is_multiple_of(c.n_head_kv)
        {
            return None;
        }
        let (pipeline, g) = pipelines.attention(c.n_head / c.n_head_kv, c.head_dim)?;
        if pipeline.max_threads() < 32 * ATTENTION_SIMDGROUPS {
            return None;
        }
        let mut plan = AttentionPlan {
            g,
            min_chunk: ATTENTION_MIN_CHUNK,
            max_chunk: ATTENTION_MAX_CHUNK,
        };
        let memory = device.max_threadgroup_memory() / 4;
        while plan.floats(c.head_dim, plan.max_chunk) > memory {
            plan.max_chunk /= 2;
            if plan.max_chunk < ATTENTION_MIN_CHUNK {
                return None;
            }
        }
        Some(plan)
    }

    /// Makes every attention chunk `chunk` positions long, if that is within
    /// the plan's bounds (the partials are sized for its shortest chunks,
    /// threadgroup memory for its longest).
    #[cfg(test)]
    fn force_attention_chunk(&mut self, chunk: usize) -> bool {
        let plan = &mut self.attention;
        let ok = chunk >= plan.min_chunk && chunk <= plan.max_chunk;
        if ok {
            plan.min_chunk = chunk;
            plan.max_chunk = chunk;
        }
        ok
    }

    /// Whether the DeltaNet kernels cover layers of these sizes: a lane
    /// holds at most 4 float4s of a key (`DELTA_MAX_CHUNKS`).
    fn delta_net_supported(d: DeltaNetDims) -> bool {
        d.d_k.is_multiple_of(4) && d.d_k <= 512 && d.conv_kernel > 0
    }

    /// The GPU's name, e.g. "Apple M4 Max".
    pub(crate) fn device_name(&self) -> &str {
        self.device.name()
    }

    /// Waits for every committed token.
    fn sync(&mut self) -> Result<()> {
        while let Some(commands) = self.in_flight.pop_front() {
            if let Err(e) = commands.wait() {
                self.gpu_error.get_or_insert(e);
            }
        }
        self.gpu_result()
    }

    /// The first failure of committed commands since the last report.
    fn gpu_result(&mut self) -> Result<()> {
        self.gpu_error
            .take()
            .map_or(Ok(()), |e| Err(backend_error(e)))
    }

    /// Keeps `commands`, just committed, to check later; first forgets the
    /// oldest ones that have finished (recording a failure), and waits for
    /// the oldest if too many are kept.
    fn track(&mut self, commands: Commands) {
        while let Some(oldest) = self.in_flight.front() {
            let finished = match oldest.status() {
                Ok(done) => done,
                Err(e) => {
                    self.gpu_error.get_or_insert(e);
                    true
                }
            };
            if !finished && self.in_flight.len() < MAX_IN_FLIGHT {
                break;
            }
            if !finished && let Some(oldest) = self.in_flight.pop_front() {
                if let Err(e) = oldest.wait() {
                    self.gpu_error.get_or_insert(e);
                }
                continue;
            }
            self.in_flight.pop_front();
        }
        self.in_flight.push_back(commands);
    }

    /// Records the first failure of this token.
    fn fail(&mut self, message: String) {
        self.error.get_or_insert(message);
    }

    /// Commits the token's last commands, which are then in flight.
    fn commit(&mut self) -> Result<()> {
        let commands = self.commands.take();
        if let Some(e) = self.error.take() {
            return Err(backend_error(e));
        }
        let mut commands =
            commands.ok_or_else(|| backend_error("a token ended without beginning".into()))?;
        commands.commit();
        self.track(commands);
        self.gpu_result()
    }

    /// Makes room for `n` more dispatches in `commands`, committing the ones
    /// before first if there are enough of them. False after a failure.
    fn reserve(&mut self, n: usize) -> bool {
        if self.error.is_some() || self.commands.is_none() {
            return false;
        }
        let limit = if self.chunks == 0 {
            FIRST_COMMIT
        } else {
            DISPATCHES_PER_COMMIT
        };
        if self.dispatches >= limit {
            self.chunks += 1;
            let next = match self.device.commands() {
                Ok(c) => c,
                Err(e) => {
                    self.fail(e);
                    return false;
                }
            };
            if let Some(mut done) = self.commands.replace(next) {
                done.commit();
                self.track(done);
            }
            self.dispatches = 0;
        }
        self.dispatches += n;
        true
    }

    fn param_offset(p: Param) -> usize {
        4 * p.range().start
    }

    /// Encodes one token of a DeltaNet layer and its gated norm (see
    /// `delta_net` in the shaders) into the state's ring `conv` and
    /// `pending_ba`. False after a failure.
    fn encode_delta_step(
        &mut self,
        d: DeltaNetDims,
        p: DeltaNetParams,
        args: &DeltaArgs,
        (conv, pending_ba): (&Buffer, &Buffer),
        step: &DeltaStep<'_>,
    ) -> bool {
        if !self.reserve(2) {
            return false;
        }
        let Some(commands) = self.commands.as_mut() else {
            return false;
        };
        let params = &self.params;
        commands.barrier();
        commands.set_pipeline(&self.pipelines.delta_net);
        commands.set_bytes(0, args);
        commands.set_buffer(1, step.x.0, step.x.1);
        commands.set_buffer(2, conv, 0);
        commands.set_buffer(3, params, Self::param_offset(p.conv));
        commands.set_buffer(4, step.beta.0, step.beta.1);
        commands.set_buffer(5, step.alpha.0, step.alpha.1);
        commands.set_buffer(6, params, Self::param_offset(p.dt_bias));
        commands.set_buffer(7, params, Self::param_offset(p.a));
        commands.set_buffer(8, step.ssm.0, 0);
        commands.set_buffer(9, step.out.0, step.out.1);
        commands.set_buffer(10, step.ssm.1, 0);
        commands.set_buffer(11, pending_ba, 0);
        commands.set_threadgroup_memory(0, 4 * (2 * d.d_k + DELTA_GROUP_ROWS));
        commands.dispatch(
            Size::new(d.n_v_heads, d.d_v.div_ceil(DELTA_GROUP_ROWS), 1),
            Size::new(32 * DELTA_SIMDGROUPS, 1, 1),
        );
        commands.barrier();
        commands.set_pipeline(&self.pipelines.delta_norm);
        commands.set_bytes(0, args);
        commands.set_buffer(1, step.out.0, step.out.1);
        commands.set_buffer(2, step.z.0, step.z.1);
        commands.set_buffer(3, params, Self::param_offset(p.norm));
        commands.dispatch(Size::new(d.n_v_heads, 1, 1), Size::new(128, 1, 1));
        true
    }

    /// Threadgroups of [`THREADS`] covering `n` threads.
    fn groups(n: usize) -> (Size, Size) {
        (
            Size::new(n.div_ceil(THREADS), 1, 1),
            Size::new(THREADS, 1, 1),
        )
    }

    /// Encodes an element-wise kernel over `n` values with `buffers` at 1...
    fn elementwise(
        &mut self,
        pick: fn(&Pipelines) -> &device::Pipeline,
        n: usize,
        buffers: &[&Buffer],
    ) {
        let n32 = match u32_of(n, "a vector length") {
            Ok(n) => n,
            Err(e) => return self.fail(e),
        };
        if !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        let pipeline = pick(&self.pipelines);
        commands.barrier();
        commands.set_pipeline(pipeline);
        commands.set_bytes(0, &CountArgs { n: n32 });
        for (i, b) in buffers.iter().enumerate() {
            commands.set_buffer(i + 1, b, 0);
        }
        let (groups, threads) = Self::groups(n);
        commands.dispatch(groups, threads);
    }

    /// Encodes the product of `w` and every token's row of `x` into its row
    /// of `out`, in one dispatch, after a barrier if `barrier`.
    fn encode_matvec(
        &mut self,
        w: &Weight,
        x: &Buffer,
        out: &Buffer,
        output: Output<'_>,
        barrier: bool,
    ) {
        if self.n == 0 || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        if barrier {
            commands.barrier();
        }
        let (pipelines, weights, n) = (&self.pipelines, &self.weights, self.n);
        let result = encode_matvec(commands, pipelines, weights, w, (x, 0), (out, 0), output, n);
        if let Err(e) = result {
            self.fail(e);
        }
    }
}

/// Encodes `out = w · x` (or as `output` says) into `commands`, for the
/// vectors at the byte offsets given with their buffers.
#[expect(clippy::too_many_arguments, reason = "one dispatch")]
fn encode_matvec(
    commands: &mut Commands,
    pipelines: &Pipelines,
    weights: &Weights<'_>,
    w: &Weight,
    (x, x_at): (&Buffer, usize),
    (out, out_at): (&Buffer, usize),
    output: Output<'_>,
    ncols: usize,
) -> std::result::Result<(), String> {
    let mv = pipelines
        .matvec(w.ty)
        .ok_or_else(|| format!("no kernel for {} weights", w.ty))?;
    if !w.cols.is_multiple_of(4) {
        return Err(format!("{} columns is not a multiple of 4", w.cols));
    }
    let (mode, w2, virtual_rows) = match output {
        Output::Store => (kernels::MATVEC_STORE, w, w.rows),
        Output::Add => (kernels::MATVEC_ADD, w, w.rows),
        Output::SwiGlu(up) => {
            if (up.ty, up.cols, up.rows, up.row_bytes) != (w.ty, w.cols, w.rows, w.row_bytes) {
                return Err("the SwiGLU products differ in type or shape".to_owned());
            }
            (kernels::MATVEC_SWIGLU, up, 2 * w.rows)
        }
    };
    let nsg = mv.nsg(w.cols);
    let (pipeline, nr0, nc) = mv.pipeline(ncols);
    // The one-input kernel goes through the inputs itself.
    let col_groups = if nc == 1 { 1 } else { ncols.div_ceil(nc) };
    let args = MatvecArgs {
        cols: u32_of(w.cols, "a matrix width")?,
        rows: u32_of(w.rows, "a matrix height")?,
        row_bytes: u32_of(w.row_bytes, "a matrix row")?,
        nsg,
        mode,
        ncols: u32_of(ncols, "a batch")?,
    };
    commands.set_pipeline(pipeline);
    commands.set_buffer(2, x, x_at);
    let locate = |w: &Weight| {
        weights
            .locate(w)
            .ok_or_else(|| format!("a weight at byte {} is outside the model", w.offset))
    };
    let (w_buffer, w_offset) = locate(w)?;
    let (w2_buffer, w2_offset) = locate(w2)?;
    commands.set_bytes(0, &args);
    commands.set_buffer(1, w_buffer, w_offset);
    commands.set_buffer(3, out, out_at);
    commands.set_buffer(4, w2_buffer, w2_offset);
    if mv.smem(nr0) > 0 {
        commands.set_threadgroup_memory(0, mv.smem(nr0));
    }
    let (groups, threads) = mv.grid(virtual_rows, nsg, nr0, col_groups);
    commands.dispatch(groups, threads);
    Ok(())
}

/// Stops using the kernels for several inputs that do not give, bit for bit,
/// the one-input kernel's output for each input on this device. They
/// transcribe its arithmetic, but with fast math the shader compiler may
/// group sums differently (it does for some widths, which are therefore not
/// built). Each is run once on sample rows of its type.
fn check_wide_matvecs(
    device: &Device,
    pipelines: &mut Pipelines,
) -> std::result::Result<(), String> {
    let (rows, cols) = (40, 512);
    let mut checks = Vec::new();
    for (ty, mv) in pipelines.matvecs_mut() {
        for nc in mv.widths() {
            checks.push((ty, nc));
        }
    }
    for (ty, nc) in checks {
        let bytes: Vec<u8> = (0..rows)
            .flat_map(|r| crate::quant::sample_row(ty, cols, 100 + r as u32))
            .collect();
        let w = Weight {
            ty,
            cols,
            rows,
            offset: 0,
            row_bytes: bytes.len() / rows,
        };
        let weights = Weights::build(device, &bytes, false, bytes.len(), usize::MAX)?;
        let x: Vec<f32> = (0..nc * cols)
            .map(|i| ((i * 7919 % 1013) as f32 / 1013.0 - 0.5) * 2.0)
            .collect();
        let run = |pipelines: &Pipelines,
                   x: &[f32],
                   ncols: usize|
         -> std::result::Result<Vec<f32>, String> {
            let mut xbuf = device.buffer(4 * x.len())?;
            let out = device.buffer(4 * ncols * rows)?;
            // SAFETY: the buffers are new; no command uses them yet.
            unsafe { store(&mut xbuf, x) };
            let mut commands = device.commands()?;
            encode_matvec(
                &mut commands,
                pipelines,
                &weights,
                &w,
                (&xbuf, 0),
                (&out, 0),
                Output::Store,
                ncols,
            )?;
            commands.commit();
            commands.wait()?;
            let mut got = vec![0.0; ncols * rows];
            // SAFETY: the commands that wrote `out` have completed.
            unsafe { load(&out, &mut got) };
            Ok(got)
        };
        // With `nc` inputs the product takes the `nc` kernel (the narrower
        // ones were checked first).
        let wide = run(pipelines, &x, nc)?;
        let mut same = true;
        for c in 0..nc {
            let alone = run(pipelines, &x[c * cols..(c + 1) * cols], 1)?;
            same &= wide[c * rows..(c + 1) * rows]
                .iter()
                .zip(&alone)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        }
        if !same && let Some((_, mv)) = pipelines.matvecs_mut().find(|(t, _)| *t == ty) {
            mv.drop_width(nc);
        }
    }
    Ok(())
}

impl Drop for MetalBackend<'_> {
    fn drop(&mut self) {
        // The GPU may still be reading buffers that are about to be freed.
        let _ = self.sync();
    }
}

impl Backend for MetalBackend<'_> {
    type Vector = MetalVector;
    type KvCache = MetalKvCache;

    fn name(&self) -> &'static str {
        "metal"
    }

    fn vector(&mut self, len: usize) -> Result<MetalVector> {
        Ok(MetalVector {
            buffer: self.device.buffer(4 * len).map_err(backend_error)?,
            len,
        })
    }

    fn kv_cache(&mut self, n_ctx: usize) -> Result<MetalKvCache> {
        let c = self.model.config;
        let bytes = 2 * c.n_head_kv * c.head_dim * n_ctx;
        Ok(MetalKvCache {
            k: self.device.buffer(bytes).map_err(backend_error)?,
            v: self.device.buffer(bytes).map_err(backend_error)?,
            len: 0,
            capacity: n_ctx,
        })
    }

    fn check_delta_net(&self, d: DeltaNetDims) -> Result<()> {
        if Self::delta_net_supported(d) {
            Ok(())
        } else {
            Err(backend_error(format!(
                "no DeltaNet kernel for heads of {} keys",
                d.d_k
            )))
        }
    }

    fn kv_len(&self, cache: &MetalKvCache) -> usize {
        cache.len
    }

    fn kv_truncate(&mut self, cache: &mut MetalKvCache, n_pos: usize) -> Result<()> {
        // Attention reads only the first `len` positions.
        cache.len = cache.len.min(n_pos);
        Ok(())
    }

    fn read(&mut self, v: &MetalVector, out: &mut [f32]) -> Result<()> {
        self.sync()?;
        debug_assert!(self.commands.is_none(), "read inside a token");
        // SAFETY: every committed command buffer has completed, and no
        // token is being encoded, so nothing writes the buffer.
        unsafe { load(&v.buffer, &mut out[..v.len]) };
        Ok(())
    }

    fn write(&mut self, v: &mut MetalVector, data: &[f32]) -> Result<()> {
        self.sync()?;
        debug_assert!(self.commands.is_none(), "write inside a token");
        // SAFETY: as in `read`.
        unsafe { store(&mut v.buffer, &data[..v.len]) };
        Ok(())
    }

    fn begin(&mut self, _pos: usize, n: usize) -> Result<()> {
        // Positions come from the caches and states the ops update.
        self.n = n;
        self.error = None;
        self.dispatches = 0;
        self.chunks = 0;
        self.commands = Some(self.device.commands().map_err(backend_error)?);
        Ok(())
    }

    fn end(&mut self) -> Result<()> {
        // Batches queue up in order; only the latest is kept, to wait for.
        self.commit()
    }

    fn logits(
        &mut self,
        w: &Weight,
        x: &MetalVector,
        rows: Range<usize>,
        out: &mut [f32],
    ) -> Result<()> {
        if w.rows > self.model.config.n_vocab || rows.len() > MAX_LOGIT_ROWS {
            self.fail(format!(
                "{} rows of {} logits do not fit the logits buffer",
                rows.len(),
                w.rows
            ));
        }
        // The logits are encoded into the last commands, which `commit`
        // takes: no `reserve`, which could commit them early.
        if let Some(commands) = self.commands.as_mut().filter(|_| self.error.is_none()) {
            commands.barrier();
            let result = encode_matvec(
                commands,
                &self.pipelines,
                &self.weights,
                w,
                (&x.buffer, 4 * rows.start * w.cols),
                (&self.logits, 0),
                Output::Store,
                rows.len(),
            );
            if let Err(e) = result {
                self.fail(e);
            }
        }
        self.commit()?;
        self.sync()?;
        // SAFETY: the commands that wrote the logits have completed.
        unsafe { load(&self.logits, &mut out[..rows.len() * w.rows]) };
        Ok(())
    }

    fn embed(&mut self, w: &Weight, tokens: &[u32], x: &mut MetalVector) {
        let Some((_, per_thread)) = self.pipelines.embed(w.ty) else {
            return self.fail(format!("no embedding kernel for {} weights", w.ty));
        };
        let sizes = (|| {
            Ok::<_, String>((
                u32_of(w.cols, "the embedding size")?,
                u32_of(w.row_bytes, "an embedding row")?,
            ))
        })();
        let (cols, row_bytes) = match sizes {
            Ok(s) => s,
            Err(e) => return self.fail(e),
        };
        if self.weights.locate(w).is_none() {
            return self.fail(format!(
                "a weight at byte {} is outside the model",
                w.offset
            ));
        }
        for (t, &token) in tokens.iter().enumerate() {
            if !self.reserve(1) {
                return;
            }
            let (Some(commands), Some((weights, offset)), Some((pipeline, _))) = (
                self.commands.as_mut(),
                self.weights.locate(w),
                self.pipelines.embed(w.ty),
            ) else {
                return;
            };
            // The rows are independent: one barrier before them all.
            if t == 0 {
                commands.barrier();
            }
            commands.set_pipeline(pipeline);
            commands.set_bytes(
                0,
                &EmbedArgs {
                    cols,
                    row: token,
                    row_bytes,
                    from_slot: 0,
                },
            );
            commands.set_buffer(1, weights, offset);
            commands.set_buffer(2, &x.buffer, 4 * t * w.cols);
            commands.set_buffer(3, &self.slots, 0);
            let (groups, threads) = Self::groups(w.cols.div_ceil(per_thread));
            commands.dispatch(groups, threads);
        }
    }

    fn embed_slot(&mut self, w: &Weight, slot: usize, x: &mut MetalVector) {
        let Some((_, per_thread)) = self.pipelines.embed(w.ty) else {
            return self.fail(format!("no embedding kernel for {} weights", w.ty));
        };
        let args = match (|| {
            Ok::<_, String>(EmbedArgs {
                cols: u32_of(w.cols, "the embedding size")?,
                row: 0,
                row_bytes: u32_of(w.row_bytes, "an embedding row")?,
                from_slot: 1,
            })
        })() {
            Ok(a) => a,
            Err(e) => return self.fail(e),
        };
        if slot >= MAX_LOGIT_ROWS || !self.reserve(1) {
            return;
        }
        let (Some(commands), Some((weights, offset)), Some((pipeline, _))) = (
            self.commands.as_mut(),
            self.weights.locate(w),
            self.pipelines.embed(w.ty),
        ) else {
            return self.fail(format!(
                "a weight at byte {} is outside the model",
                w.offset
            ));
        };
        commands.barrier();
        commands.set_pipeline(pipeline);
        commands.set_bytes(0, &args);
        commands.set_buffer(1, weights, offset);
        commands.set_buffer(2, &x.buffer, 0);
        commands.set_buffer(3, &self.slots, 4 * slot);
        let (groups, threads) = Self::groups(w.cols.div_ceil(per_thread));
        commands.dispatch(groups, threads);
    }

    fn argmax(&mut self, x: &MetalVector, len: usize, slot: usize) {
        let n = match u32_of(len, "a vector length") {
            Ok(n) => n,
            Err(e) => return self.fail(e),
        };
        if slot >= MAX_LOGIT_ROWS || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        commands.barrier();
        commands.set_pipeline(&self.pipelines.argmax);
        commands.set_bytes(0, &CountArgs { n });
        commands.set_buffer(1, &x.buffer, 0);
        commands.set_buffer(2, &self.slots, 4 * slot);
        commands.dispatch(Size::new(1, 1, 1), Size::new(1024, 1, 1));
    }

    fn read_slots(&mut self, out: &mut [u32]) -> Result<()> {
        self.sync()?;
        let mut v = [0.0f32; MAX_LOGIT_ROWS];
        let n = out.len().min(MAX_LOGIT_ROWS);
        // SAFETY: every committed command buffer has completed.
        unsafe { load(&self.slots, &mut v[..n]) };
        for (o, v) in out.iter_mut().zip(&v[..n]) {
            *o = v.to_bits();
        }
        Ok(())
    }

    fn rmsnorm(&mut self, x: &MetalVector, w: Param, eps: f32, out: &mut MetalVector) {
        let args = RmsArgs {
            n: w.len() as u32,
            eps,
        };
        if self.n == 0 || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        // One threadgroup per row.
        commands.barrier();
        commands.set_pipeline(&self.pipelines.rmsnorm);
        commands.set_bytes(0, &args);
        commands.set_buffer(1, &x.buffer, 0);
        commands.set_buffer(2, &self.params, Self::param_offset(w));
        commands.set_buffer(3, &out.buffer, 0);
        commands.dispatch(Size::new(self.n, 1, 1), Size::new(THREADS, 1, 1));
    }

    fn norm_pair(
        &mut self,
        a: &MetalVector,
        wa: Param,
        b: &MetalVector,
        wb: Param,
        eps: f32,
        out: &mut MetalVector,
    ) {
        let (da, db) = (wa.len(), wb.len());
        // Every norm reads and writes rows of its own: one barrier first.
        for t in 0..self.n {
            let halves = [
                (&a.buffer, wa, 4 * t * da, 0),
                (&b.buffer, wb, 4 * t * db, 4 * da),
            ];
            for (i, (x, w, x_at, out_at)) in halves.into_iter().enumerate() {
                if !self.reserve(1) {
                    return;
                }
                let Some(commands) = self.commands.as_mut() else {
                    return;
                };
                if t == 0 && i == 0 {
                    commands.barrier();
                }
                commands.set_pipeline(&self.pipelines.rmsnorm);
                commands.set_bytes(
                    0,
                    &RmsArgs {
                        n: w.len() as u32,
                        eps,
                    },
                );
                commands.set_buffer(1, x, x_at);
                commands.set_buffer(2, &self.params, Self::param_offset(w));
                commands.set_buffer(3, &out.buffer, 4 * t * (da + db) + out_at);
                commands.dispatch(Size::new(1, 1, 1), Size::new(THREADS, 1, 1));
            }
        }
    }

    fn copy_rows(
        &mut self,
        src: &MetalVector,
        from: usize,
        dst: &mut MetalVector,
        to: usize,
        rows: usize,
        len: usize,
    ) {
        let n = rows * len;
        if n == 0 || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        commands.barrier();
        commands.set_pipeline(&self.pipelines.copy);
        commands.set_bytes(0, &CountArgs { n: n as u32 });
        commands.set_buffer(1, &src.buffer, 4 * from * len);
        commands.set_buffer(2, &dst.buffer, 4 * to * len);
        let (groups, threads) = Self::groups(n);
        commands.dispatch(groups, threads);
    }

    fn residual(
        &mut self,
        w: &Weight,
        a: &MetalVector,
        x: &mut MetalVector,
        norm: Param,
        eps: f32,
        out: &mut MetalVector,
    ) {
        self.encode_matvec(w, &a.buffer, &x.buffer, Output::Add, true);
        self.rmsnorm(x, norm, eps, out);
    }

    fn matvecs<const N: usize>(
        &mut self,
        x: &MetalVector,
        products: [(&Weight, &mut MetalVector); N],
    ) {
        // The products read the same input and write different outputs, so
        // they may run together: one barrier before them all.
        for (j, (w, out)) in products.into_iter().enumerate() {
            self.encode_matvec(w, &x.buffer, &out.buffer, Output::Store, j == 0);
        }
    }

    fn swiglu(
        &mut self,
        gate: &Weight,
        up: &Weight,
        x: &MetalVector,
        out: &mut MetalVector,
        up_out: &mut MetalVector,
    ) {
        let x = &x.buffer;
        if (gate.ty, gate.cols, gate.rows, gate.row_bytes)
            == (up.ty, up.cols, up.rows, up.row_bytes)
        {
            return self.encode_matvec(gate, x, &out.buffer, Output::SwiGlu(up), true);
        }
        self.encode_matvec(gate, x, &out.buffer, Output::Store, true);
        self.encode_matvec(up, x, &up_out.buffer, Output::Store, false);
        let n = self.n * gate.rows;
        self.elementwise(|p| &p.swiglu, n, &[&out.buffer, &up_out.buffer]);
    }

    fn attention(
        &mut self,
        q: &mut MetalVector,
        k: &mut MetalVector,
        v: &MetalVector,
        norm: QkNorm,
        cache: &mut MetalKvCache,
        out: &mut MetalVector,
    ) {
        let c = self.model.config;
        let hd = c.head_dim;
        let kv_dim = c.n_head_kv * hd;
        let q_stride = if norm.gated { 2 * hd } else { hd };
        let plan = self.attention;
        let n = self.n;
        let shapes_ok = q.len >= n * c.n_head * q_stride
            && k.len >= n * kv_dim
            && v.len >= n * kv_dim
            && out.len >= n * c.n_head * hd
            && cache.capacity <= self.n_ctx;
        if !shapes_ok {
            return self.fail("unsupported attention shape".to_owned());
        }
        if cache.len + n > cache.capacity {
            return self.fail(format!("KV cache of {} positions is full", cache.capacity));
        }
        let group = c.n_head / c.n_head_kv;
        let (q_norm, k_norm) = norm.weights.map_or((0, 0), |(q, k)| {
            (Self::param_offset(q), Self::param_offset(k))
        });
        for t in 0..n {
            // Each token appends its key and value, then attends to every
            // position up to its own.
            cache.len += 1;
            let chunk = plan.chunk(cache.len, c.n_head_kv);
            let n_chunks = cache.len.div_ceil(chunk);
            let args = AttentionArgs {
                n_pos: cache.len as u32,
                head_dim: hd as u32,
                group: group as u32,
                kv_dim: kv_dim as u32,
                q_stride: q_stride as u32,
                gated: u32::from(norm.gated),
                n_rot: c.rope_dims as u32,
                norm: u32::from(norm.weights.is_some()),
                chunk: chunk as u32,
                n_chunks: n_chunks as u32,
                scale: 1.0 / (hd as f32).sqrt(),
                base: c.rope_base,
                eps: norm.eps,
                pad: 0,
            };
            if !self.reserve(2) {
                return;
            }
            let Some(commands) = self.commands.as_mut() else {
                return;
            };
            let p = &self.pipelines;
            let Some((pipeline, _)) = p.attention(group, hd) else {
                return;
            };
            let (q_at, kv_at, out_at) = (
                4 * t * c.n_head * q_stride,
                4 * t * kv_dim,
                4 * t * c.n_head * hd,
            );
            commands.barrier();
            commands.set_pipeline(pipeline);
            commands.set_bytes(0, &args);
            commands.set_buffer(1, &q.buffer, q_at);
            commands.set_buffer(2, &k.buffer, kv_at);
            commands.set_buffer(3, &v.buffer, kv_at);
            commands.set_buffer(4, &self.params, q_norm);
            commands.set_buffer(5, &self.params, k_norm);
            commands.set_buffer(6, &cache.k, 0);
            commands.set_buffer(7, &cache.v, 0);
            commands.set_buffer(8, &self.partials, 0);
            commands.set_buffer(9, &out.buffer, out_at);
            commands.set_threadgroup_memory(0, 4 * plan.floats(hd, chunk));
            commands.dispatch(
                Size::new(c.n_head_kv, n_chunks, 1),
                Size::new(32 * ATTENTION_SIMDGROUPS, 1, 1),
            );
            if n_chunks > 1 {
                commands.barrier();
                commands.set_pipeline(&p.attention_reduce);
                commands.set_bytes(0, &args);
                commands.set_buffer(1, &q.buffer, q_at);
                commands.set_buffer(2, &self.partials, 0);
                commands.set_buffer(3, &out.buffer, out_at);
                commands.dispatch(Size::new(c.n_head, 1, 1), Size::new(hd.min(256), 1, 1));
            }
        }
    }

    fn delta_net(
        &mut self,
        d: DeltaNetDims,
        p: DeltaNetParams,
        eps: f32,
        state: &mut DeltaNetState<MetalVector>,
        qkv: &MetalVector,
        _conv: &mut MetalVector,
        z: &mut MetalVector,
        beta: &mut MetalVector,
        alpha: &mut MetalVector,
        out: &mut MetalVector,
    ) {
        let n = self.n;
        let rows = d.conv_rows();
        if !Self::delta_net_supported(d)
            || state.conv.len != d.conv_len()
            || state.pending_ba.len != d.pending_len()
            || qkv.len < n * d.conv_dim()
            || out.len < n * d.value_dim()
            || p.norm.len() != d.d_v
        {
            return self.fail(format!("unsupported DeltaNet head size {}", d.d_k));
        }
        let pending = (2..=MAX_PENDING).contains(&n);
        if pending && self.delta_work.is_none() {
            match self.device.buffer(4 * d.state_len()) {
                Ok(work) => self.delta_work = Some(work),
                Err(e) => return self.fail(e),
            }
        }
        let args = DeltaArgs {
            n_k_heads: d.n_k_heads as u32,
            d_k: d.d_k as u32,
            d_v: d.d_v as u32,
            key_dim: d.key_dim() as u32,
            conv_dim: d.conv_dim() as u32,
            taps: d.conv_kernel as u32,
            rows: rows as u32,
            oldest: state.oldest as u32,
            store: 1,
            pending: 0,
            eps,
            scale: 1.0 / (d.d_v as f32).sqrt(),
        };
        if n == 1 && state.keep == 0 {
            // Decoding: one step, which stores its input in the ring itself.
            let ssm = &state.ssm.buffer;
            let step = DeltaStep {
                x: (&qkv.buffer, 0),
                beta: (&beta.buffer, 0),
                alpha: (&alpha.buffer, 0),
                ssm: (ssm, ssm),
                out: (&out.buffer, 0),
                z: (&z.buffer, 0),
            };
            let ring = (&state.conv.buffer, &state.pending_ba.buffer);
            if self.encode_delta_step(d, p, &args, ring, &step) {
                state.oldest = (state.oldest + 1) % rows;
            }
            return;
        }

        // The kept pending tokens, then the batch, in one dispatch.
        let r = state.keep;
        let ba_write = if pending {
            (state.pending_row + MAX_PENDING) % (2 * MAX_PENDING)
        } else {
            0
        };
        let batch = DeltaBatchArgs {
            n_k_heads: args.n_k_heads,
            d_k: args.d_k,
            d_v: args.d_v,
            key_dim: args.key_dim,
            conv_dim: args.conv_dim,
            taps: args.taps,
            rows: args.rows,
            oldest: args.oldest,
            replay: r as u32,
            n: n as u32,
            pending: u32::from(pending),
            ba_read: state.pending_row as u32,
            ba_write: ba_write as u32,
            eps,
            scale: args.scale,
        };
        if !self.reserve(2) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        let ssm = &state.ssm.buffer;
        let state_out = match &self.delta_work {
            Some(work) if pending => work,
            _ => ssm,
        };
        let params = &self.params;
        commands.barrier();
        commands.set_pipeline(&self.pipelines.delta_net_batch);
        commands.set_bytes(0, &batch);
        commands.set_buffer(1, &qkv.buffer, 0);
        commands.set_buffer(2, &state.conv.buffer, 0);
        commands.set_buffer(3, params, Self::param_offset(p.conv));
        commands.set_buffer(4, &beta.buffer, 0);
        commands.set_buffer(5, &alpha.buffer, 0);
        commands.set_buffer(6, params, Self::param_offset(p.dt_bias));
        commands.set_buffer(7, params, Self::param_offset(p.a));
        commands.set_buffer(8, ssm, 0);
        commands.set_buffer(9, &out.buffer, 0);
        commands.set_buffer(10, state_out, 0);
        commands.set_buffer(11, &state.pending_ba.buffer, 0);
        commands.set_buffer(12, ssm, 0);
        commands.set_threadgroup_memory(0, 4 * (2 * d.d_k + DELTA_GROUP_ROWS));
        commands.dispatch(
            Size::new(d.n_v_heads, d.d_v.div_ceil(DELTA_GROUP_ROWS), 1),
            Size::new(32 * DELTA_SIMDGROUPS, 1, 1),
        );
        commands.barrier();
        commands.set_pipeline(&self.pipelines.delta_norm);
        commands.set_bytes(0, &args);
        commands.set_buffer(1, &out.buffer, 0);
        commands.set_buffer(2, &z.buffer, 0);
        commands.set_buffer(3, params, Self::param_offset(p.norm));
        commands.dispatch(Size::new(d.n_v_heads, n, 1), Size::new(128, 1, 1));

        // The ring, read by the dispatch above, then takes the inputs that
        // stay in it: a pending batch's, else those of the new window.
        let first = if pending {
            0
        } else {
            n.saturating_sub(d.conv_kernel - 1)
        };
        let slot = (state.oldest + d.conv_kernel - 1 + r + first) % rows;
        let count = n - first;
        let before_wrap = count.min(rows - slot);
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
            state.oldest = (state.oldest + r) % rows;
            state.pending = n;
            state.keep = n;
            state.pending_row = ba_write;
        } else {
            state.oldest = (state.oldest + r + n) % rows;
            state.pending = 0;
            state.keep = 0;
            state.pending_row = 0;
        }
    }
}
