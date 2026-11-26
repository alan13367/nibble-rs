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
//! for, so a prompt streams to the GPU. A batch of up to
//! [`MATMUL_MIN_BATCH`] tokens (a draft being verified) gives each token
//! exactly its result in a batch of its own; a longer one (a prompt)
//! multiplies matrices with simdgroup-matrix tiles, attends with a flash
//! attention kernel and splits each DeltaNet layer into a parallel pass and
//! a scan, rounding like llama.cpp's batched path rather than like
//! decoding. The model file's memory map is wrapped in a buffer without a
//! copy; activations are f32, and the KV cache is f16.
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

use device::{Buffer, Commands, Device, Pipeline, Size};
use kernels::{
    AttentionArgs, CountArgs, DeltaArgs, DeltaBatchArgs, EmbedArgs, FlashArgs, MatmulArgs,
    MatmulIdArgs, MatvecArgs, MoeArgs, MoeMapArgs, MoeSumArgs, Pipelines, RmsArgs, RotateArgs,
    RouteArgs,
};

use crate::backend::{
    Backend, DeltaNetDims, DeltaNetParams, DeltaNetState, KvCacheType, KvTypes, MAX_LOGIT_ROWS,
    MAX_PENDING, ModelData, Param, QkNorm, Rotation,
};
use crate::cpu::{Experts, Weight};
use crate::error::{Error, Result};
use weights::Weights;

/// Threads per threadgroup of the element-wise and norm kernels.
const THREADS: usize = 256;

/// The most experts a router may choose from (the routing kernel keeps
/// their probabilities in threadgroup memory).
const MAX_EXPERTS: usize = 4096;

/// Simdgroups per threadgroup of `delta_net`, and the state rows the
/// threadgroup updates (`DELTA_ROWS` in the shader per simdgroup).
const DELTA_SIMDGROUPS: usize = 4;
const DELTA_GROUP_ROWS: usize = DELTA_SIMDGROUPS * 4;
/// Tokens per chunk of `delta_chunk_*` (`DCHUNK`), the head size they
/// take (`DHEAD`), the floats a chunk's prep leaves (`DCHUNK_PREP`), and
/// the threadgroup memory of the scan (the state rows, two chunks by 32
/// rows, four diagonals, three values per token), in bytes.
const DCHUNK: usize = 32;
const DHEAD: usize = 128;
const DCHUNK_PREP: usize = 2 * DCHUNK * DCHUNK + DCHUNK;
const DCHUNK_SCAN_SMEM: usize = 4 * (32 * DHEAD + 2 * DCHUNK * 32 + 4 * 64 + 3 * DCHUNK);

/// Simdgroups per threadgroup of `delta_scan`, one state row each.
const SCAN_SIMDGROUPS: usize = 8;

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

/// Batches of more than this many tokens multiply with the matrix-matrix
/// kernels (ggml-metal's `mul_mm`, which rounds inputs to f16), smaller ones
/// with the matrix-vector kernels, bit-identical to decoding each token.
/// llama.cpp takes `mul_mm` above 16 inputs too (its few-row kernels cover
/// 2 to 16), and a draft being verified is at most [`MAX_LOGIT_ROWS`] long.
const MATMUL_MIN_BATCH: usize = MAX_LOGIT_ROWS;

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
    /// The types of the attention caches it makes, and their bytes so far.
    kv: KvTypes,
    kv_bytes: usize,
    /// The chunked delta rule's per-chunk matrices (`delta_chunk_prep`),
    /// and the floats they have room for.
    delta_chunks: Option<(Buffer, usize)>,
    /// The expert tiles and slot lists of a prompt batch (`moe_map`), and
    /// the slots they have room for; allocated by the first batch that
    /// needs them.
    moe_map: Option<(Buffer, Buffer, usize)>,
    /// Buffers replaced while commands that use them may still run (command
    /// buffers do not retain them), freed at the next sync.
    retired: Vec<Buffer>,
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

/// The `KV_*` code of a cache type in the shaders (`shaders/kv.metal`).
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
    pub(crate) fn new(model: ModelData<'m>, n_ctx: usize, kv: KvTypes) -> Result<Self> {
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
            kv,
            kv_bytes: 0,
            delta_chunks: None,
            moe_map: None,
            retired: Vec::new(),
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
        if self.commands.is_none() {
            self.retired.clear();
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

    /// Encodes `rotate` (see [`Backend::rotate`]) of every row of `x` into
    /// `out`, which may be `x` for the inverse.
    fn encode_rotate(&mut self, x: &Buffer, r: Rotation, inverse: bool, out: &Buffer) {
        let order = r.heads.unwrap_or(crate::backend::HeadOrder {
            head_dim: 0,
            n_k: 0,
            rep: 0,
        });
        // `check_rotation` bounds every size.
        let args = RotateArgs {
            len: r.len as u32,
            block: r.block as u32,
            signs: u32::from(r.signs.is_some()),
            head_dim: order.head_dim as u32,
            n_k: order.n_k as u32,
            rep: order.rep as u32,
            inverse: u32::from(inverse),
        };
        if self.n == 0 || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        commands.barrier();
        commands.set_pipeline(&self.pipelines.rotate);
        commands.set_bytes(0, &args);
        commands.set_buffer(1, x, 0);
        let signs = r.signs.map_or(0, Self::param_offset);
        commands.set_buffer(2, &self.params, signs);
        commands.set_buffer(3, out, 0);
        commands.set_threadgroup_memory(0, 4 * r.block);
        commands.dispatch(
            Size::new(r.len / r.block, self.n, 1),
            Size::new((r.block / 2).clamp(1, THREADS), 1, 1),
        );
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
        let result = if n > MATMUL_MIN_BATCH && matmul_covers(pipelines, w) {
            let add = match output {
                Output::Store => false,
                Output::Add => true,
                Output::SwiGlu(_) => {
                    return self.fail("a SwiGLU product of a prompt batch".to_owned());
                }
            };
            encode_matmul(commands, pipelines, weights, w, x, out, add, n)
        } else {
            encode_matvec(commands, pipelines, weights, w, (x, 0), (out, 0), output, n)
        };
        if let Err(e) = result {
            self.fail(e);
        }
    }
}

impl MetalBackend<'_> {
    /// Whether the expert products of this batch multiply tiles of slots
    /// (`matmul_*_id`) rather than one slot at a time: prompt batches, as
    /// llama.cpp's `mul_mm_id`, for routers the map kernel covers (one
    /// thread per expert).
    fn moe_matmul(&self, w: &Experts) -> bool {
        self.n > MATMUL_MIN_BATCH
            && w.first.cols.is_multiple_of(32)
            && self.pipelines.matmul_id(w.first.ty).is_some()
            && w.count <= self.pipelines.moe_map.max_threads()
    }

    /// Encodes `moe_map` for the batch's `route` of `k` slots per token
    /// over `n_expert` experts. False after a failure.
    fn encode_moe_map(&mut self, route: &Buffer, k: usize, n_expert: usize) -> bool {
        let slots = self.n * k;
        if self.moe_map.as_ref().is_none_or(|m| m.2 < slots) {
            // The most tiles: the full ones, and a partial one per expert.
            let tiles = slots.div_ceil(32) + n_expert.min(slots);
            let made = self
                .device
                .buffer(4 * (1 + 3 * tiles))
                .and_then(|t| Ok((t, self.device.buffer(4 * (slots + n_expert))?)));
            match made {
                Ok((t, i)) => {
                    if let Some((old_t, old_i, _)) = self.moe_map.replace((t, i, slots)) {
                        self.retired.extend([old_t, old_i]);
                    }
                }
                Err(e) => {
                    self.fail(e);
                    return false;
                }
            }
        }
        let args = MoeMapArgs {
            n: self.n as u32,
            k: k as u32,
            n_expert: n_expert as u32,
        };
        if !self.reserve(1) {
            return false;
        }
        let (Some(commands), Some((tiles, ids, _))) =
            (self.commands.as_mut(), self.moe_map.as_ref())
        else {
            return false;
        };
        commands.barrier();
        commands.set_pipeline(&self.pipelines.moe_map);
        commands.set_bytes(0, &args);
        commands.set_buffer(1, route, 0);
        commands.set_buffer(2, tiles, 0);
        commands.set_buffer(3, ids, 0);
        commands.set_threadgroup_memory(
            0,
            (2 * 4 * n_expert + 2 * n_expert * k).next_multiple_of(16),
        );
        commands.dispatch(Size::new(1, 1, 1), Size::new(n_expert, 1, 1));
        true
    }

    /// Encodes the expert products of every slot with `matmul_*_id`, after
    /// [`Self::encode_moe_map`]: `out_s = w_e · x`, the input row `s` of `x`
    /// with `per_slot`, else row `s / k`.
    fn encode_matmul_id(
        &mut self,
        w: &Experts,
        x: &Buffer,
        out: &Buffer,
        k: usize,
        per_slot: bool,
    ) {
        if !self.reserve(1) {
            return;
        }
        let f = &w.first;
        let result = (|| {
            let pipeline = self
                .pipelines
                .matmul_id(f.ty)
                .ok_or_else(|| format!("no expert matrix kernel for {} weights", f.ty))?;
            let (tiles, ids, _) = self.moe_map.as_ref().ok_or("no expert map")?;
            let args = MatmulIdArgs {
                cols: u32_of(f.cols, "a matrix width")?,
                rows: u32_of(f.rows, "a matrix height")?,
                row_bytes: u32_of(f.row_bytes, "a matrix row")?,
                expert_bytes: u32_of(f.rows * f.row_bytes, "an expert")?,
                k: u32_of(k, "an expert count")?,
                per_slot: u32::from(per_slot),
            };
            let (w_buffer, w_offset) = self
                .weights
                .locate(&w.all())
                .ok_or_else(|| format!("experts at byte {} are outside the model", f.offset))?;
            let slots = self.n * k;
            let max_tiles = slots.div_ceil(32) + w.count.min(slots);
            let commands = self.commands.as_mut().ok_or("no commands")?;
            commands.barrier();
            commands.set_pipeline(pipeline);
            commands.set_bytes(0, &args);
            commands.set_buffer(1, w_buffer, w_offset);
            commands.set_buffer(2, x, 0);
            commands.set_buffer(3, tiles, 0);
            commands.set_buffer(4, ids, 0);
            commands.set_buffer(5, out, 0);
            commands.set_threadgroup_memory(0, kernels::MATMUL_SMEM);
            commands.dispatch(
                Size::new(max_tiles, f.rows.div_ceil(kernels::MATMUL_ROWS), 1),
                Size::new(kernels::MATMUL_THREADS, 1, 1),
            );
            Ok::<_, String>(())
        })();
        if let Err(e) = result {
            self.fail(e);
        }
    }
}

impl MetalBackend<'_> {
    /// Encodes the mixture-of-experts products: for every slot `s` of the
    /// batch's `n * k`, `out_s = w_e · x` (or as `output` says, with `up`'s
    /// expert `e` as the second matrix), `e` the slot's expert in `route`;
    /// the input is row `s` of `x` with `per_slot`, else row `s / k`.
    #[expect(clippy::too_many_arguments, reason = "one dispatch")]
    fn encode_matvec_id(
        &mut self,
        w: &Experts,
        up: Option<&Experts>,
        x: &Buffer,
        out: &Buffer,
        route: &Buffer,
        (k, per_slot): (usize, bool),
        output: Output<'_>,
    ) {
        if self.n == 0 || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        commands.barrier();
        let result = (|| {
            let f = &w.first;
            let mv = self
                .pipelines
                .matvec(f.ty)
                .ok_or_else(|| format!("no kernel for {} weights", f.ty))?;
            if !f.cols.is_multiple_of(4) {
                return Err(format!("{} columns is not a multiple of 4", f.cols));
            }
            let (mode, virtual_rows) = match output {
                Output::SwiGlu(_) => (kernels::MATVEC_SWIGLU, 2 * f.rows),
                _ => (kernels::MATVEC_STORE, f.rows),
            };
            let nsg = mv.nsg(f.cols);
            let (_, nr0, _) = mv.pipeline(1);
            let args = MatvecArgs {
                cols: u32_of(f.cols, "a matrix width")?,
                rows: u32_of(f.rows, "a matrix height")?,
                row_bytes: u32_of(f.row_bytes, "a matrix row")?,
                nsg,
                mode,
                ncols: 1,
            };
            let moe = MoeArgs {
                k: u32_of(k, "an expert count")?,
                per_slot: u32::from(per_slot),
                expert_rows: u32_of(f.rows, "a matrix height")?,
                count: u32_of(w.count, "an expert count")?,
            };
            let locate = |e: &Experts| {
                self.weights.locate(&e.all()).ok_or_else(|| {
                    format!("experts at byte {} are outside the model", e.first.offset)
                })
            };
            let (w_buffer, w_offset) = locate(w)?;
            let (w2_buffer, w2_offset) = locate(up.unwrap_or(w))?;
            commands.set_pipeline(&mv.id);
            commands.set_bytes(0, &args);
            commands.set_buffer(1, w_buffer, w_offset);
            commands.set_buffer(2, x, 0);
            commands.set_buffer(3, out, 0);
            commands.set_buffer(4, w2_buffer, w2_offset);
            commands.set_buffer(5, route, 0);
            commands.set_bytes(6, &moe);
            if mv.smem(nr0) > 0 {
                commands.set_threadgroup_memory(0, mv.smem(nr0));
            }
            let (groups, threads) = mv.grid(virtual_rows, nsg, nr0, self.n * k);
            commands.dispatch(groups, threads);
            Ok(())
        })();
        if let Err(e) = result {
            self.fail(e);
        }
    }
}

/// Whether the matrix-matrix kernels cover products with `w`.
fn matmul_covers(pipelines: &Pipelines, w: &Weight) -> bool {
    w.cols.is_multiple_of(32) && pipelines.matmul(w.ty).is_some()
}

/// Encodes `out = w · x` (`out += w · x` with `add`) for the `n` rows of
/// `x` into `commands`, with the matrix-matrix kernel of `w`'s type.
#[expect(clippy::too_many_arguments, reason = "one dispatch")]
fn encode_matmul(
    commands: &mut Commands,
    pipelines: &Pipelines,
    weights: &Weights<'_>,
    w: &Weight,
    x: &Buffer,
    out: &Buffer,
    add: bool,
    n: usize,
) -> std::result::Result<(), String> {
    let pipeline = pipelines
        .matmul(w.ty)
        .ok_or_else(|| format!("no matrix kernel for {} weights", w.ty))?;
    if !w.cols.is_multiple_of(32) {
        return Err(format!("{} columns is not a multiple of 32", w.cols));
    }
    let args = MatmulArgs {
        cols: u32_of(w.cols, "a matrix width")?,
        rows: u32_of(w.rows, "a matrix height")?,
        row_bytes: u32_of(w.row_bytes, "a matrix row")?,
        n: u32_of(n, "a batch")?,
        mode: if add {
            kernels::MATVEC_ADD
        } else {
            kernels::MATVEC_STORE
        },
    };
    let (w_buffer, w_offset) = weights
        .locate(w)
        .ok_or_else(|| format!("a weight at byte {} is outside the model", w.offset))?;
    commands.set_pipeline(pipeline);
    commands.set_bytes(0, &args);
    commands.set_buffer(1, w_buffer, w_offset);
    commands.set_buffer(2, x, 0);
    commands.set_buffer(3, out, 0);
    commands.set_threadgroup_memory(0, kernels::MATMUL_SMEM);
    commands.dispatch(
        Size::new(
            n.div_ceil(kernels::MATMUL_INPUTS),
            w.rows.div_ceil(kernels::MATMUL_ROWS),
            1,
        ),
        Size::new(kernels::MATMUL_THREADS, 1, 1),
    );
    Ok(())
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
        // Prompt attention reads whole blocks of positions (the ones past
        // the end are masked).
        let positions = n_ctx.next_multiple_of(kernels::FLASH_C);
        let kv_dim = c.n_head_kv * c.head_dim;
        let buffer = |ty: KvCacheType| {
            self.device
                .buffer(ty.bytes(kv_dim) * positions)
                .map_err(backend_error)
        };
        let (k, v) = (buffer(self.kv.k)?, buffer(self.kv.v)?);
        self.kv_bytes += (self.kv.k.bytes(kv_dim) + self.kv.v.bytes(kv_dim)) * positions;
        Ok(MetalKvCache {
            k,
            v,
            len: 0,
            capacity: n_ctx,
        })
    }

    fn moe_route(
        &mut self,
        logits: &MetalVector,
        n_expert: usize,
        k: usize,
        route: &mut MetalVector,
    ) {
        let args = match (
            u32_of(n_expert, "an expert count"),
            u32_of(k, "an expert count"),
        ) {
            (Ok(n_expert), Ok(k)) => RouteArgs { n_expert, k },
            (Err(e), _) | (_, Err(e)) => return self.fail(e),
        };
        if n_expert > MAX_EXPERTS {
            return self.fail(format!("{n_expert} experts is more than {MAX_EXPERTS}"));
        }
        if self.n == 0 || !self.reserve(1) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        commands.barrier();
        commands.set_pipeline(&self.pipelines.moe_route);
        commands.set_bytes(0, &args);
        commands.set_buffer(1, &logits.buffer, 0);
        commands.set_buffer(2, &route.buffer, 0);
        commands.set_threadgroup_memory(0, 4 * n_expert.next_multiple_of(4));
        commands.dispatch(Size::new(self.n, 1, 1), Size::new(32, 1, 1));
    }

    fn moe_swiglu(
        &mut self,
        gate: &Experts,
        up: &Experts,
        x: &MetalVector,
        route: &MetalVector,
        k: usize,
        out: &mut MetalVector,
        up_out: &mut MetalVector,
    ) {
        let (g, u) = (&gate.first, &up.first);
        let route = &route.buffer;
        if self.moe_matmul(gate) && self.moe_matmul(up) {
            if self.encode_moe_map(route, k, gate.count) {
                self.encode_matmul_id(gate, &x.buffer, &out.buffer, k, false);
                self.encode_matmul_id(up, &x.buffer, &up_out.buffer, k, false);
                let n = self.n * k * g.rows;
                self.elementwise(|p| &p.swiglu, n, &[&out.buffer, &up_out.buffer]);
            }
            return;
        }
        if (g.ty, g.cols, g.rows, g.row_bytes) == (u.ty, u.cols, u.rows, u.row_bytes) {
            let output = Output::SwiGlu(&up.first);
            return self.encode_matvec_id(
                gate,
                Some(up),
                &x.buffer,
                &out.buffer,
                route,
                (k, false),
                output,
            );
        }
        self.encode_matvec_id(
            gate,
            None,
            &x.buffer,
            &out.buffer,
            route,
            (k, false),
            Output::Store,
        );
        self.encode_matvec_id(
            up,
            None,
            &x.buffer,
            &up_out.buffer,
            route,
            (k, false),
            Output::Store,
        );
        let n = self.n * k * g.rows;
        self.elementwise(|p| &p.swiglu, n, &[&out.buffer, &up_out.buffer]);
    }

    fn moe_down(
        &mut self,
        down: &Experts,
        hidden: &MetalVector,
        route: &MetalVector,
        k: usize,
        products: &mut MetalVector,
        out: &mut MetalVector,
    ) {
        let e = down.first.rows;
        let (route, products) = (&route.buffer, &products.buffer);
        if self.moe_matmul(down) {
            if self.encode_moe_map(route, k, down.count) {
                self.encode_matmul_id(down, &hidden.buffer, products, k, true);
            }
        } else {
            self.encode_matvec_id(
                down,
                None,
                &hidden.buffer,
                products,
                route,
                (k, true),
                Output::Store,
            );
        }
        let args = MoeSumArgs {
            n: e as u32,
            k: k as u32,
        };
        if self.n > 0
            && self.reserve(1)
            && let Some(commands) = self.commands.as_mut()
        {
            commands.barrier();
            commands.set_pipeline(&self.pipelines.moe_combine);
            commands.set_bytes(0, &args);
            commands.set_buffer(1, products, 0);
            commands.set_buffer(2, route, 0);
            commands.set_buffer(3, &out.buffer, 0);
            commands.dispatch(
                Size::new(e.div_ceil(THREADS), self.n, 1),
                Size::new(THREADS, 1, 1),
            );
        }
    }

    fn moe_residual(
        &mut self,
        moe: &MetalVector,
        shared: &MetalVector,
        gate: &MetalVector,
        x: &mut MetalVector,
        norm: Param,
        eps: f32,
        out: &mut MetalVector,
    ) {
        let e = norm.len();
        if self.n > 0
            && self.reserve(1)
            && let Some(commands) = self.commands.as_mut()
        {
            commands.barrier();
            commands.set_pipeline(&self.pipelines.moe_residual);
            commands.set_bytes(0, &MoeSumArgs { n: e as u32, k: 0 });
            commands.set_buffer(1, &moe.buffer, 0);
            commands.set_buffer(2, &shared.buffer, 0);
            commands.set_buffer(3, &gate.buffer, 0);
            commands.set_buffer(4, &x.buffer, 0);
            commands.dispatch(
                Size::new(e.div_ceil(THREADS), self.n, 1),
                Size::new(THREADS, 1, 1),
            );
        }
        self.rmsnorm(x, norm, eps, out);
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

    fn kv_bytes(&self) -> usize {
        self.kv_bytes
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

    fn check_rotation(&self, r: Rotation) -> Result<()> {
        let fits = r.block.is_power_of_two()
            && r.block <= 8192
            && 4 * r.block <= self.device.max_threadgroup_memory()
            && u32::try_from(r.len).is_ok();
        if fits {
            Ok(())
        } else {
            Err(backend_error(format!(
                "no kernel for a Hadamard rotation of {} values in blocks of {}",
                r.len, r.block
            )))
        }
    }

    fn rotate(&mut self, x: &MetalVector, r: Rotation, out: &mut MetalVector) {
        self.encode_rotate(&x.buffer, r, false, &out.buffer);
    }

    fn unrotate(&mut self, x: &mut MetalVector, r: Rotation) {
        self.encode_rotate(&x.buffer, r, true, &x.buffer);
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
        // A prompt batch multiplies both matrices, then applies the SwiGLU,
        // as llama.cpp does.
        let matmul = self.n > MATMUL_MIN_BATCH && matmul_covers(&self.pipelines, gate);
        if !matmul
            && (gate.ty, gate.cols, gate.rows, gate.row_bytes)
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
        let kv = self.kv;
        let rot_v = kv.rot_v(hd);
        // With a rotated value cache the output is rotated back, and gated,
        // by `attention_unrotate`.
        let gated = norm.gated && rot_v.is_none();
        let args = |n_pos: usize, chunk: usize, n_chunks: usize| AttentionArgs {
            n_pos: n_pos as u32,
            head_dim: hd as u32,
            group: group as u32,
            kv_dim: kv_dim as u32,
            q_stride: q_stride as u32,
            gated: u32::from(gated),
            n_rot: c.rope_dims as u32,
            norm: u32::from(norm.weights.is_some()),
            chunk: chunk as u32,
            n_chunks: n_chunks as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            base: c.rope_base,
            eps: norm.eps,
            k_type: kv_code(kv.k),
            v_type: kv_code(kv.v),
            k_row: kv.k.bytes(kv_dim) as u32,
            v_row: kv.v.bytes(kv_dim) as u32,
            rot_k: kv.rot_k(hd).unwrap_or(0) as u32,
            rot_v: rot_v.unwrap_or(0) as u32,
        };
        let first = cache.len;
        // f16 caches are multiplied from device memory, others from blocks
        // dequantized into threadgroup memory.
        let staged = kv != KvTypes::default();
        let shape =
            kernels::PromptAttention::new(group, hd, staged, self.device.max_threadgroup_memory());
        fn flash(p: &Pipelines, staged: bool, hd: usize) -> Option<&Pipeline> {
            if staged {
                p.attention_prompt_staged(hd)
            } else {
                p.attention_prompt(hd)
            }
        }
        let prompt = n > MATMUL_MIN_BATCH
            && shape.is_some_and(|s| {
                flash(&self.pipelines, staged, hd).is_some_and(|p| p.max_threads() >= 32 * s.nsg)
            });
        if let (true, Some(shape)) = (prompt, shape) {
            let prep = args(cache.len, 0, 0);
            let flash_args = FlashArgs {
                n: n as u32,
                n_past: cache.len as u32,
                kv_dim: kv_dim as u32,
                q_stride: q_stride as u32,
                gated: u32::from(gated),
                n_head: c.n_head as u32,
                scale: prep.scale,
                group: group as u32,
                heads: shape.heads as u32,
                tiles: shape.tiles as u32,
                k_type: prep.k_type,
                v_type: prep.v_type,
                k_row: prep.k_row,
                v_row: prep.v_row,
            };
            cache.len += n;
            if !self.reserve(2) {
                return;
            }
            let Some(commands) = self.commands.as_mut() else {
                return;
            };
            let p = &self.pipelines;
            let Some(flash) = flash(p, staged, hd) else {
                return;
            };
            commands.barrier();
            commands.set_pipeline(&p.attention_prep);
            commands.set_bytes(0, &prep);
            commands.set_buffer(1, &q.buffer, 0);
            commands.set_buffer(2, &k.buffer, 0);
            commands.set_buffer(3, &v.buffer, 0);
            commands.set_buffer(4, &self.params, q_norm);
            commands.set_buffer(5, &self.params, k_norm);
            commands.set_buffer(6, &cache.k, 0);
            commands.set_buffer(7, &cache.v, 0);
            commands.set_threadgroup_memory(0, 4 * 2 * hd);
            commands.dispatch(
                Size::new(c.n_head + c.n_head_kv, n, 1),
                Size::new(32 * ATTENTION_SIMDGROUPS, 1, 1),
            );
            commands.barrier();
            commands.set_pipeline(flash);
            commands.set_bytes(0, &flash_args);
            commands.set_buffer(1, &q.buffer, 0);
            commands.set_buffer(2, &cache.k, 0);
            commands.set_buffer(3, &cache.v, 0);
            commands.set_buffer(4, &out.buffer, 0);
            commands.set_threadgroup_memory(0, shape.smem);
            commands.dispatch(
                Size::new(
                    n.div_ceil(8 * shape.tiles),
                    c.n_head_kv * (group / shape.heads),
                    1,
                ),
                Size::new(32 * shape.nsg, 1, 1),
            );
        } else {
            for t in 0..n {
                // Each token appends its key and value, then attends to
                // every position up to its own.
                cache.len += 1;
                let chunk = plan.chunk(cache.len, c.n_head_kv);
                let n_chunks = cache.len.div_ceil(chunk);
                let args = args(cache.len, chunk, n_chunks);
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
        if rot_v.is_some() {
            let args = AttentionArgs {
                gated: u32::from(norm.gated),
                ..args(first, 0, 0)
            };
            if !self.reserve(1) {
                return;
            }
            let Some(commands) = self.commands.as_mut() else {
                return;
            };
            commands.barrier();
            commands.set_pipeline(&self.pipelines.attention_unrotate);
            commands.set_bytes(0, &args);
            commands.set_buffer(1, &q.buffer, 0);
            commands.set_buffer(2, &out.buffer, 0);
            commands.set_threadgroup_memory(0, 4 * hd);
            commands.dispatch(
                Size::new(c.n_head, n, 1),
                Size::new(32 * ATTENTION_SIMDGROUPS, 1, 1),
            );
        }
    }

    fn delta_net(
        &mut self,
        d: DeltaNetDims,
        p: DeltaNetParams,
        eps: f32,
        state: &mut DeltaNetState<MetalVector>,
        qkv: &MetalVector,
        conv: &mut MetalVector,
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
        // A prompt batch: the replayed tokens alone, then every token's
        // inputs in parallel, then the scan (see `delta_prep`), which stages
        // up to 16 tokens' q, k, v, beta and decay at a time.
        let scan_row = 2 * d.d_k + SCAN_SIMDGROUPS + 2;
        // Each thread loads `SCAN_LOADS` (4) float4s of q and k and one other
        // value of a block.
        let threads = 32 * SCAN_SIMDGROUPS;
        let stage = (self.device.max_threadgroup_memory() / (4 * scan_row))
            .min(4 * threads / (d.d_k / 2))
            .min(threads / (SCAN_SIMDGROUPS + 2))
            .min(16);
        let prompt = n > MATMUL_MIN_BATCH
            && conv.len >= n * d.conv_dim()
            && stage > 0
            && self.pipelines.delta_scan(d.d_k).is_some();
        // The chunked form reads whole chunks of the batch buffers.
        let n_chunks = n.div_ceil(DCHUNK);
        let chunked = prompt
            && d.d_k == DHEAD
            && d.d_v == DHEAD
            && conv.len >= n_chunks * DCHUNK * d.conv_dim()
            && beta.len >= n * d.n_v_heads
            && DCHUNK_SCAN_SMEM <= self.device.max_threadgroup_memory();
        let chunk_floats = d.n_v_heads * n_chunks * DCHUNK_PREP;
        if chunked
            && self
                .delta_chunks
                .as_ref()
                .is_none_or(|c| c.1 < chunk_floats)
        {
            match self.device.buffer(4 * chunk_floats) {
                Ok(b) => {
                    if let Some((old, _)) = self.delta_chunks.replace((b, chunk_floats)) {
                        self.retired.push(old);
                    }
                }
                Err(e) => return self.fail(e),
            }
        }
        if !self.reserve(if prompt { 5 } else { 2 }) {
            return;
        }
        let Some(commands) = self.commands.as_mut() else {
            return;
        };
        let ssm = &state.ssm.buffer;
        let params = &self.params;
        let groups = Size::new(d.n_v_heads, d.d_v.div_ceil(DELTA_GROUP_ROWS), 1);
        let threads = Size::new(32 * DELTA_SIMDGROUPS, 1, 1);
        if prompt {
            if r > 0 {
                let replay = DeltaBatchArgs { n: 0, ..batch };
                commands.barrier();
                commands.set_pipeline(&self.pipelines.delta_net_batch);
                commands.set_bytes(0, &replay);
                commands.set_buffer(1, &qkv.buffer, 0);
                commands.set_buffer(2, &state.conv.buffer, 0);
                commands.set_buffer(3, params, Self::param_offset(p.conv));
                commands.set_buffer(4, &beta.buffer, 0);
                commands.set_buffer(5, &alpha.buffer, 0);
                commands.set_buffer(6, params, Self::param_offset(p.dt_bias));
                commands.set_buffer(7, params, Self::param_offset(p.a));
                commands.set_buffer(8, ssm, 0);
                commands.set_buffer(9, &out.buffer, 0);
                commands.set_buffer(10, ssm, 0);
                commands.set_buffer(11, &state.pending_ba.buffer, 0);
                commands.set_buffer(12, ssm, 0);
                commands.set_threadgroup_memory(0, 4 * (2 * d.d_k + DELTA_GROUP_ROWS));
                commands.dispatch(groups, threads);
            }
            // Concurrent with the replay, which reads none of its outputs.
            if r == 0 {
                commands.barrier();
            }
            commands.set_pipeline(&self.pipelines.delta_prep);
            commands.set_bytes(0, &batch);
            commands.set_buffer(1, &qkv.buffer, 0);
            commands.set_buffer(2, &state.conv.buffer, 0);
            commands.set_buffer(3, params, Self::param_offset(p.conv));
            commands.set_buffer(4, &beta.buffer, 0);
            commands.set_buffer(5, &alpha.buffer, 0);
            commands.set_buffer(6, params, Self::param_offset(p.dt_bias));
            commands.set_buffer(7, params, Self::param_offset(p.a));
            commands.set_buffer(8, &conv.buffer, 0);
            commands.dispatch(Size::new(d.n_k_heads, n, 1), threads);
            commands.barrier();
            if let (true, Some((chunks, _))) = (chunked, self.delta_chunks.as_ref()) {
                commands.set_pipeline(&self.pipelines.delta_chunk_prep);
                commands.set_bytes(0, &batch);
                commands.set_buffer(1, &conv.buffer, 0);
                commands.set_buffer(2, &beta.buffer, 0);
                commands.set_buffer(3, &alpha.buffer, 0);
                commands.set_buffer(4, chunks, 0);
                commands.set_threadgroup_memory(0, 4 * (2 * DCHUNK * DCHUNK + 2 * DCHUNK));
                commands.dispatch(Size::new(n_chunks, d.n_v_heads, 1), Size::new(128, 1, 1));
                commands.barrier();
                commands.set_pipeline(&self.pipelines.delta_chunk_scan);
                commands.set_bytes(0, &batch);
                commands.set_buffer(1, &conv.buffer, 0);
                commands.set_buffer(2, &beta.buffer, 0);
                commands.set_buffer(3, chunks, 0);
                commands.set_buffer(4, ssm, 0);
                commands.set_buffer(5, &out.buffer, 0);
                commands.set_threadgroup_memory(0, DCHUNK_SCAN_SMEM);
                commands.dispatch(Size::new(d.n_v_heads, d.d_v / 32, 1), Size::new(128, 1, 1));
            }
            if !chunked {
                let Some(scan) = self.pipelines.delta_scan(d.d_k) else {
                    return;
                };
                commands.set_pipeline(scan);
                commands.set_bytes(6, &CountArgs { n: stage as u32 });
                commands.set_threadgroup_memory(0, 4 * stage * scan_row);
                commands.set_bytes(0, &batch);
                commands.set_buffer(1, &conv.buffer, 0);
                commands.set_buffer(2, &beta.buffer, 0);
                commands.set_buffer(3, &alpha.buffer, 0);
                commands.set_buffer(4, ssm, 0);
                commands.set_buffer(5, &out.buffer, 0);
                commands.dispatch(
                    Size::new(d.n_v_heads, d.d_v.div_ceil(SCAN_SIMDGROUPS), 1),
                    Size::new(32 * SCAN_SIMDGROUPS, 1, 1),
                );
            }
        }
        let state_out = match &self.delta_work {
            Some(work) if pending => work,
            _ => ssm,
        };
        if !prompt {
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
            commands.dispatch(groups, threads);
        }
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
