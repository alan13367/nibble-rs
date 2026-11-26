//! Inference sessions: one sequence's tokens, cache and logits.

use crate::backend::{Backend, BackendKind, KvCacheType, KvTypes, MAX_LOGIT_ROWS};
use crate::cpu::{self, CpuBackend};
use crate::error::{Error, Result};
#[cfg(target_os = "macos")]
use crate::metal::MetalBackend;
use crate::model::{Logits, Model, Snapshot, State};
use crate::sampler::Sampler;

/// Options for [`Model::session`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionOptions {
    /// Maximum tokens the session can hold. `0` uses the model's training context.
    /// Larger values are capped at the training context.
    pub n_ctx: usize,
    /// Threads that run the forward pass, the calling thread included. `0`
    /// uses the performance cores on Apple Silicon and one per physical core
    /// elsewhere (one per logical CPU where the OS does not report cores): a
    /// second thread per core does not make decoding faster.
    /// The threads live as long as the session.
    pub threads: usize,
    /// The hardware that runs the forward pass (default
    /// [`BackendKind::Auto`]: Metal where available, else the CPU).
    pub backend: BackendKind,
    /// Most tokens evaluated in one forward pass, which reads each weight
    /// once for all of them. `0` uses [`DEFAULT_BATCH`]. Each token of a
    /// batch needs its own activation buffers (about 140 KiB for
    /// Qwen3.5-0.8B), allocated when the session starts; results do not
    /// depend on the batch size.
    pub n_batch: usize,
    /// Most tokens [`Session::generate`] drafts with the model's
    /// multi-token-prediction head after each token, checking them all in
    /// one forward pass (speculative decoding). The tokens generated are
    /// the same either way; drafting is faster when the drafts are often
    /// right. `0`, the default, does not draft. Needs a model with a head
    /// ([`Model::has_mtp`]); at most [`MAX_EVAL_ALL`]` - 1` are drafted.
    pub draft_max: usize,
    /// The type of the cached attention keys (default
    /// [`KvCacheType::F16`]), as llama.cpp's `--cache-type-k`.
    pub cache_type_k: KvCacheType,
    /// The type of the cached attention values (default
    /// [`KvCacheType::F16`]), as llama.cpp's `--cache-type-v`. A quantized
    /// type makes attention follow llama.cpp's flash-attention path.
    pub cache_type_v: KvCacheType,
}

/// The default [`SessionOptions::n_batch`], llama.cpp's default batch.
pub const DEFAULT_BATCH: usize = 512;

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            threads: 0,
            backend: BackendKind::default(),
            n_batch: 0,
            draft_max: 0,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
        }
    }
}

/// How [`Session::generate`]'s drafts fared since the session started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct DraftStats {
    /// Tokens drafted.
    pub drafted: usize,
    /// Drafted tokens the model agreed with.
    pub accepted: usize,
    /// Forward passes that checked drafts.
    pub verifications: usize,
}

/// Why [`Session::generate`] stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The model produced an end-of-generation token (not passed to the callback).
    EndOfGeneration(u32),
    /// `max_tokens` tokens were generated.
    MaxTokens,
    /// The callback asked to stop.
    Stopped,
    /// The context is full.
    ContextFull,
}

/// Most tokens [`Session::eval_all`] evaluates at once.
pub const MAX_EVAL_ALL: usize = MAX_LOGIT_ROWS;

/// Checkpoints of recurrent state a session keeps; each costs the size of
/// the model's recurrent state (about 19 MiB for Qwen3.5-0.8B).
pub const MAX_CHECKPOINTS: usize = 4;

/// How many tokens before the end of an evaluation [`Session::truncate`]'s
/// checkpoint is taken: llama.cpp's choice. Chat prompts end with the reply
/// header (and possibly an empty think block), which the next turn's
/// prompt does not repeat.
pub const CHECKPOINT_TAIL: usize = 4;

/// What a recurrent state was after the first `n_pos` tokens.
struct Checkpoint {
    n_pos: usize,
    snapshot: Snapshot,
}

/// A backend and the model state it holds.
struct Run<B: Backend> {
    backend: B,
    state: State<B>,
}

/// The session's backend. Each [`Session`] method matches on it once and
/// runs generic code on the backend inside.
#[expect(
    clippy::large_enum_variant,
    reason = "one per session, matched on every call: boxing buys nothing"
)]
enum Engine<'m> {
    Cpu(Run<CpuBackend<'m>>),
    #[cfg(target_os = "macos")]
    Metal(Run<MetalBackend<'m>>),
}

/// Runs `$body` with `$b` and `$s` bound to the backend and state of `$engine`.
macro_rules! on_engine {
    ($engine:expr, |$b:ident, $s:ident| $body:expr) => {
        match $engine {
            Engine::Cpu(Run {
                backend: $b,
                state: $s,
            }) => $body,
            #[cfg(target_os = "macos")]
            Engine::Metal(Run {
                backend: $b,
                state: $s,
            }) => $body,
        }
    };
}

impl<'m> Engine<'m> {
    /// The engine, and why `Auto` did not pick the GPU if it did not.
    fn new(
        model: &'m Model,
        options: SessionOptions,
        n_ctx: usize,
        rows: usize,
    ) -> Result<(Self, Option<String>)> {
        let mtp = options.draft_max > 0;
        let kv = KvTypes {
            k: options.cache_type_k,
            v: options.cache_type_v,
        };
        let head_dim = model.config().head_dim;
        for (which, ty) in [("K", kv.k), ("V", kv.v)] {
            if !head_dim.is_multiple_of(ty.block_size()) {
                return Err(Error::UnsupportedCacheType {
                    which,
                    ty,
                    head_dim,
                });
            }
        }
        let cpu = |threads: usize| -> Result<Self> {
            let threads = match threads {
                0 => cpu::default_threads(),
                n => n,
            };
            let mut backend = CpuBackend::new(model.data(), n_ctx, threads, kv);
            let state = model.new_state(&mut backend, n_ctx, rows, mtp)?;
            Ok(Self::Cpu(Run { backend, state }))
        };
        match options.backend {
            BackendKind::Cpu => Ok((cpu(options.threads)?, None)),
            BackendKind::Metal => Ok((Self::metal(model, n_ctx, rows, mtp, kv)?, None)),
            // Only Metal exists to try, and only on macOS.
            BackendKind::Auto if !cfg!(target_os = "macos") => Ok((cpu(options.threads)?, None)),
            BackendKind::Auto => match Self::metal(model, n_ctx, rows, mtp, kv) {
                Ok(engine) => Ok((engine, None)),
                Err(e) => Ok((cpu(options.threads)?, Some(e.to_string()))),
            },
        }
    }

    #[cfg(target_os = "macos")]
    fn metal(model: &'m Model, n_ctx: usize, rows: usize, mtp: bool, kv: KvTypes) -> Result<Self> {
        let mut backend = MetalBackend::new(model.data(), n_ctx, kv)?;
        let state = model.new_state(&mut backend, n_ctx, rows, mtp)?;
        Ok(Self::Metal(Run { backend, state }))
    }

    #[cfg(not(target_os = "macos"))]
    fn metal(
        _model: &'m Model,
        _n_ctx: usize,
        _rows: usize,
        _mtp: bool,
        _kv: KvTypes,
    ) -> Result<Self> {
        Err(Error::Backend(
            "Metal is only available on macOS".to_owned(),
        ))
    }
}

/// A sequence being evaluated on a [`Model`]: tokens so far, cache, logits.
pub struct Session<'m> {
    model: &'m Model,
    n_ctx: usize,
    n_batch: usize,
    /// Rows of the forward pass's buffers: a batch of [`Session::eval`] or
    /// the most tokens of [`Session::eval_all`].
    rows: usize,
    engine: Engine<'m>,
    /// Why [`BackendKind::Auto`] fell back to the CPU.
    fallback: Option<String>,
    kv: KvTypes,
    tokens: Vec<u32>,
    /// The logits of the last evaluation: one row, or one per token after
    /// [`Session::eval_all`] (grown then).
    logits: Vec<f32>,
    /// The row of `logits` that predicts the next token.
    logits_row: usize,
    logits_valid: bool,
    /// Oldest first; positions increase.
    checkpoints: Vec<Checkpoint>,
    /// Most tokens drafted after each token (0: no drafting).
    draft_max: usize,
    /// The drafts of the current step, and the head's logits.
    drafts: Vec<u32>,
    draft_logits: Vec<f32>,
    stats: DraftStats,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("n_ctx", &self.n_ctx)
            .field("backend", &self.kernel())
            .field("tokens", &self.tokens.len())
            .finish_non_exhaustive()
    }
}

impl<'m> Session<'m> {
    /// A new, empty session.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if the backend asked for cannot start or cannot
    /// allocate the session's buffers, [`Error::NoDraftHead`] for drafting
    /// with a model that has no MTP head.
    pub fn new(model: &'m Model, options: SessionOptions) -> Result<Self> {
        if options.draft_max > 0 && !model.has_mtp() {
            return Err(Error::NoDraftHead);
        }
        let c = model.config();
        let n_ctx = match options.n_ctx {
            0 => c.n_ctx_train,
            n => n.min(c.n_ctx_train),
        };
        // A batch never needs more rows than the context holds.
        let n_batch = match options.n_batch {
            0 => DEFAULT_BATCH,
            n => n,
        }
        .min(n_ctx)
        .max(1);
        let rows = n_batch.max(MAX_EVAL_ALL).min(n_ctx);
        let (engine, fallback) = Engine::new(model, options, n_ctx, rows)?;
        Ok(Self {
            model,
            n_ctx,
            n_batch,
            rows,
            engine,
            fallback,
            kv: KvTypes {
                k: options.cache_type_k,
                v: options.cache_type_v,
            },
            tokens: Vec::new(),
            logits: vec![0.0; c.n_vocab],
            logits_row: 0,
            logits_valid: false,
            checkpoints: Vec::new(),
            draft_max: options.draft_max.min(MAX_EVAL_ALL - 1),
            drafts: Vec::new(),
            draft_logits: Vec::new(),
            stats: DraftStats::default(),
        })
    }

    /// The model this session runs on.
    #[must_use]
    pub const fn model(&self) -> &'m Model {
        self.model
    }

    /// Maximum number of tokens.
    #[must_use]
    pub const fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Most tokens evaluated in one forward pass.
    #[must_use]
    pub const fn n_batch(&self) -> usize {
        self.n_batch
    }

    /// Most tokens [`Session::generate`] drafts after each token (0: it
    /// does not draft).
    #[must_use]
    pub const fn draft_max(&self) -> usize {
        self.draft_max
    }

    /// How the drafts fared so far.
    #[must_use]
    pub const fn draft_stats(&self) -> DraftStats {
        self.stats
    }

    /// The kernels this session runs: `scalar`, `avx2` or `neon` on the
    /// CPU, `metal` on an Apple GPU.
    #[must_use]
    pub fn kernel(&self) -> &'static str {
        on_engine!(&self.engine, |b, _s| b.name())
    }

    /// The backend this session runs on (never [`BackendKind::Auto`]).
    #[must_use]
    pub fn backend(&self) -> BackendKind {
        match self.engine {
            Engine::Cpu(_) => BackendKind::Cpu,
            #[cfg(target_os = "macos")]
            Engine::Metal(_) => BackendKind::Metal,
        }
    }

    /// Why the session runs on the CPU although [`BackendKind::Auto`] was
    /// asked for: the GPU backend's error (on macOS, for instance, a device
    /// or model shape the Metal kernels do not cover). `None` otherwise.
    #[must_use]
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    /// Bytes of the attention layers' KV caches, allocated for the whole
    /// context when the session started.
    #[must_use]
    pub fn kv_cache_bytes(&self) -> usize {
        on_engine!(&self.engine, |b, _s| b.kv_bytes())
    }

    /// The types of the cached keys and values.
    #[must_use]
    pub fn cache_types(&self) -> (KvCacheType, KvCacheType) {
        (self.kv.k, self.kv.v)
    }

    /// What runs the session, for logs: `10 threads, neon kernels` on the
    /// CPU, `Metal on Apple M4 Max` on a GPU.
    #[must_use]
    pub fn compute(&self) -> String {
        match &self.engine {
            Engine::Cpu(run) => format!(
                "{} threads, {} kernels",
                run.backend.threads(),
                run.backend.name()
            ),
            #[cfg(target_os = "macos")]
            Engine::Metal(run) => format!("Metal on {}", run.backend.device_name()),
        }
    }

    /// CPU threads that run the forward pass, the calling thread included:
    /// fewer than requested if the system could not start them all, and 1
    /// on a GPU backend.
    #[must_use]
    pub fn threads(&self) -> usize {
        match &self.engine {
            Engine::Cpu(run) => run.backend.threads(),
            #[cfg(target_os = "macos")]
            Engine::Metal(_) => 1,
        }
    }

    /// Tokens evaluated so far.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Logits for the next token, if the last call evaluated something.
    #[must_use]
    pub fn logits(&self) -> Option<&[f32]> {
        self.logits_valid.then(|| self.next_logits())
    }

    /// The row of the logits that predicts the next token.
    fn next_logits(&self) -> &[f32] {
        let n_vocab = self.model.config().n_vocab;
        &self.logits[self.logits_row * n_vocab..(self.logits_row + 1) * n_vocab]
    }

    /// Checks that `tokens` can be appended.
    fn check(&self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Err(Error::EmptyInput);
        }
        if self.tokens.len() + tokens.len() > self.n_ctx {
            return Err(Error::ContextFull { n_ctx: self.n_ctx });
        }
        let n_vocab = self.model.config().n_vocab;
        if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= n_vocab) {
            return Err(Error::InvalidToken(bad));
        }
        Ok(())
    }

    /// Appends `tokens` and returns the logits predicting the token after them.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyInput`] for no tokens, [`Error::InvalidToken`] for ids
    /// outside the vocabulary, [`Error::ContextFull`] if they do not fit,
    /// [`Error::Backend`] if the backend fails.
    pub fn eval(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.check(tokens)?;
        self.logits_valid = false;
        self.logits_row = 0;
        let n_vocab = self.model.config().n_vocab;
        let model = self.model;
        // The checkpoint is taken before token `cut`, so a batch ends there.
        let cut = tokens.len().checked_sub(CHECKPOINT_TAIL);
        on_engine!(&mut self.engine, |b, state| {
            let mut start = 0;
            while start < tokens.len() {
                let mut end = (start + self.n_batch).min(tokens.len());
                if let Some(cut) = cut.filter(|&cut| start < cut && cut < end) {
                    end = cut;
                }
                if Some(start) == cut {
                    checkpoint(model, b, state, self.tokens.len(), &mut self.checkpoints)?;
                }
                // Only the last token's logits are needed.
                let logits = (end == tokens.len()).then(|| Logits {
                    rows: end - start - 1..end - start,
                    out: &mut self.logits[..n_vocab],
                });
                let batch = &tokens[start..end];
                model.forward(b, state, batch, self.tokens.len(), logits)?;
                self.tokens.extend_from_slice(batch);
                start = end;
            }
        });
        self.logits_valid = true;
        Ok(self.next_logits())
    }

    /// Appends `tokens` (1 to [`MAX_EVAL_ALL`]) in one forward pass and
    /// returns the logits after each of them: one row of `n_vocab` values
    /// per token, the last predicting the next token (and returned by
    /// [`Session::logits`]). This is how a draft is verified.
    ///
    /// Unlike [`Session::eval`], it takes no checkpoint. Until the next
    /// evaluation, [`Session::truncate`] can take back any of 2 or more such
    /// tokens without evaluating anything again, recurrent models included.
    ///
    /// # Errors
    ///
    /// As [`Session::eval`], and [`Error::TooManyTokens`] for more than
    /// [`MAX_EVAL_ALL`] tokens.
    pub fn eval_all(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        self.check(tokens)?;
        if tokens.len() > MAX_EVAL_ALL {
            return Err(Error::TooManyTokens {
                n: tokens.len(),
                max: MAX_EVAL_ALL,
            });
        }
        self.logits_valid = false;
        let n_vocab = self.model.config().n_vocab;
        let n = tokens.len();
        // Grows once, to the most rows asked for.
        if self.logits.len() < n * n_vocab {
            self.logits.resize(n * n_vocab, 0.0);
        }
        let model = self.model;
        let pos = self.tokens.len();
        on_engine!(&mut self.engine, |b, state| {
            let logits = Logits {
                rows: 0..n,
                out: &mut self.logits[..n * n_vocab],
            };
            model.forward(b, state, tokens, pos, Some(logits))?;
        });
        self.tokens.extend_from_slice(tokens);
        self.logits_row = n - 1;
        self.logits_valid = true;
        Ok(&self.logits[..n * n_vocab])
    }

    /// Keeps the first `len` tokens and drops the rest. Returns how many of
    /// the kept tokens had to be evaluated again.
    ///
    /// That is none for most models. Models with recurrent layers cannot
    /// rewind their state, except into the last evaluation when it had 2 to
    /// 16 tokens (a draft being verified), so otherwise the session goes
    /// back to the latest checkpoint at or before `len` and evaluates the
    /// tokens after it, then keeps a checkpoint at `len`. A checkpoint is
    /// also taken [`CHECKPOINT_TAIL`] tokens before the end of every
    /// evaluation of at least that many tokens: in a chat, the next turn's
    /// prompt usually diverges from the last one just there.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if the backend fails. The session is then empty.
    pub fn truncate(&mut self, len: usize) -> Result<usize> {
        if len >= self.tokens.len() {
            return Ok(0);
        }
        self.logits_valid = false;
        let result = self.rewind(len);
        if result.is_err() {
            self.tokens.clear();
            self.checkpoints.clear();
        }
        result
    }

    fn rewind(&mut self, len: usize) -> Result<usize> {
        let end = self.tokens.len();
        self.tokens.truncate(len);
        self.checkpoints.retain(|c| c.n_pos <= len);
        let model = self.model;
        let n_ctx = self.n_ctx;
        on_engine!(&mut self.engine, |b, state| {
            if model.truncate_state(b, state, end, len)? {
                return Ok(0);
            }
            let from = match self.checkpoints.last_mut() {
                Some(c) => {
                    model.restore_state(b, state, &mut c.snapshot, c.n_pos)?;
                    c.n_pos
                }
                None => {
                    *state = model.new_state(b, n_ctx, self.rows, self.draft_max > 0)?;
                    0
                }
            };
            let mut pos = from;
            for batch in self.tokens[from..].chunks(self.n_batch) {
                model.forward(b, state, batch, pos, None)?;
                pos += batch.len();
            }
            // Where the sequence was cut is where the next one is likely to
            // diverge too (a client resending a shared prefix, such as a
            // harness's system prompt): keep the state there.
            if len > from {
                checkpoint(model, b, state, len, &mut self.checkpoints)?;
            }
            Ok(len - from)
        })
    }

    /// Drops all tokens.
    ///
    /// # Errors
    ///
    /// [`Error::Backend`] if the backend cannot allocate a new state.
    pub fn clear(&mut self) -> Result<()> {
        self.tokens.clear();
        self.logits_valid = false;
        self.checkpoints.clear();
        let model = self.model;
        let n_ctx = self.n_ctx;
        let (rows, mtp) = (self.rows, self.draft_max > 0);
        on_engine!(&mut self.engine, |b, state| {
            *state = model.new_state(b, n_ctx, rows, mtp)?;
        });
        Ok(())
    }

    /// Evaluates `prompt`, then samples up to `max_tokens` tokens, passing each
    /// to `on_token` together with the logits it was sampled from (for
    /// log-probabilities). Return `false` from `on_token` to stop.
    ///
    /// Each token is evaluated after `on_token` accepts it, so the session ends
    /// up holding the prompt and every accepted token. An empty prompt
    /// continues from the current logits.
    ///
    /// # Errors
    ///
    /// As [`Session::eval`]; also [`Error::EmptyInput`] if the prompt is empty
    /// and nothing has been evaluated yet.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        sampler: &mut Sampler,
        max_tokens: usize,
        mut on_token: impl FnMut(u32, &[f32]) -> bool,
    ) -> Result<StopReason> {
        if !prompt.is_empty() {
            self.eval(prompt)?;
        } else if !self.logits_valid {
            return Err(Error::EmptyInput);
        }
        if self.draft_max > 0 {
            return self.generate_drafted(sampler, max_tokens, &mut on_token);
        }
        for _ in 0..max_tokens {
            let token = sampler.sample(self.next_logits(), &self.tokens);
            if self.model.tokenizer().is_end_of_generation(token) {
                return Ok(StopReason::EndOfGeneration(token));
            }
            if !on_token(token, self.next_logits()) {
                return Ok(StopReason::Stopped);
            }
            if self.tokens.len() >= self.n_ctx {
                return Ok(StopReason::ContextFull);
            }
            self.eval(&[token])?;
        }
        Ok(StopReason::MaxTokens)
    }

    /// [`Session::generate`] with drafts: each accepted token is evaluated
    /// together with the tokens the MTP head drafts after it, and every
    /// draft the sampler then picks itself (from the logits after the
    /// tokens before it, as without drafts) is a token gained. The first
    /// draft it does not pick is dropped, with the rest; its own pick is
    /// the next token. The sampler sees the same logits in the same order
    /// as without drafts, so the tokens are the same, and the session ends
    /// up holding the same tokens.
    fn generate_drafted(
        &mut self,
        sampler: &mut Sampler,
        max_tokens: usize,
        on_token: &mut impl FnMut(u32, &[f32]) -> bool,
    ) -> Result<StopReason> {
        let tokenizer = self.model.tokenizer();
        let mut generated = 0;
        // A token sampled from the current logits, not yet passed on.
        let mut sampled = None;
        loop {
            if generated == max_tokens {
                return Ok(StopReason::MaxTokens);
            }
            let token = match sampled.take() {
                Some(t) => t,
                None => sampler.sample(self.next_logits(), &self.tokens),
            };
            if tokenizer.is_end_of_generation(token) {
                return Ok(StopReason::EndOfGeneration(token));
            }
            if !on_token(token, self.next_logits()) {
                return Ok(StopReason::Stopped);
            }
            generated += 1;
            if self.tokens.len() >= self.n_ctx {
                return Ok(StopReason::ContextFull);
            }
            // Drafts beyond the context or the tokens still wanted are of no use.
            let room = (self.n_ctx - self.tokens.len() - 1)
                .min(max_tokens - generated)
                .min(self.draft_max);
            self.draft(token, room)?;
            let mut batch = Vec::with_capacity(1 + self.drafts.len());
            batch.push(token);
            batch.extend_from_slice(&self.drafts);
            self.eval_all(&batch)?;
            let base = self.tokens.len() - batch.len();
            if self.drafts.is_empty() {
                continue;
            }
            self.stats.drafted += self.drafts.len();
            self.stats.verifications += 1;
            let n_vocab = self.model.config().n_vocab;
            for (j, &draft) in batch[1..].iter().enumerate() {
                // Row `j` follows `batch[j]`, at `base + j`.
                let row = &self.logits[j * n_vocab..(j + 1) * n_vocab];
                let pick = sampler.sample(row, &self.tokens[..base + j + 1]);
                if pick != draft {
                    self.keep_drafts(base, j)?;
                    sampled = Some(pick);
                    break;
                }
                self.stats.accepted += 1;
                if tokenizer.is_end_of_generation(pick) {
                    self.keep_drafts(base, j)?;
                    return Ok(StopReason::EndOfGeneration(pick));
                }
                if !on_token(pick, row) {
                    self.keep_drafts(base, j)?;
                    return Ok(StopReason::Stopped);
                }
                generated += 1;
                if generated == max_tokens {
                    self.keep_drafts(base, j + 1)?;
                    return Ok(StopReason::MaxTokens);
                }
            }
        }
    }

    /// Keeps the first `kept` drafts of the batch evaluated at `base` (the
    /// token before them included) and drops the rest; the logits after
    /// the last kept token are then the current ones.
    fn keep_drafts(&mut self, base: usize, kept: usize) -> Result<()> {
        self.truncate(base + 1 + kept)?;
        self.logits_row = kept;
        self.logits_valid = true;
        Ok(())
    }

    /// The MTP head's logits for the token after `tokens`, which would
    /// follow the session's tokens: the first paired with the model's
    /// hidden state at the session's last token, each later one with the
    /// head's own output for the token before (as when drafting). The
    /// session's tokens and logits stay as they are.
    ///
    /// # Errors
    ///
    /// [`Error::NoDraftHead`] if the session does not draft
    /// ([`SessionOptions::draft_max`] is 0), [`Error::EmptyInput`] if
    /// nothing has been evaluated or `tokens` is empty, otherwise as
    /// [`Session::eval`].
    pub fn draft_logits(&mut self, tokens: &[u32]) -> Result<&[f32]> {
        if self.draft_max == 0 {
            return Err(Error::NoDraftHead);
        }
        if self.tokens.is_empty() {
            return Err(Error::EmptyInput);
        }
        self.check(tokens)?;
        let model = self.model;
        self.draft_logits.resize(model.config().n_vocab, 0.0);
        let pos = self.tokens.len();
        on_engine!(&mut self.engine, |b, state| {
            for (i, &t) in tokens.iter().enumerate() {
                model.draft(b, state, t, pos + i, i == 0, &mut self.draft_logits)?;
            }
        });
        Ok(&self.draft_logits)
    }

    /// Drafts up to `max` tokens after `token`, which is to be evaluated at
    /// the next position, into `self.drafts`: each the head's most likely
    /// token (chosen on the device, the steps not waited for one by one),
    /// cut after an end-of-generation token.
    fn draft(&mut self, token: u32, max: usize) -> Result<()> {
        self.drafts.clear();
        if max == 0 {
            return Ok(());
        }
        let model = self.model;
        let pos = self.tokens.len();
        self.drafts.resize(max, 0);
        on_engine!(&mut self.engine, |b, state| {
            model.draft_chain(b, state, token, pos, &mut self.drafts)?;
        });
        let tokenizer = model.tokenizer();
        if let Some(end) = self
            .drafts
            .iter()
            .position(|&t| tokenizer.is_end_of_generation(t))
        {
            self.drafts.truncate(end + 1);
        }
        Ok(())
    }
}

/// Saves what `state` cannot rewind at position `n_pos`, evicting the
/// oldest checkpoint (and reusing its buffers) when there are
/// [`MAX_CHECKPOINTS`]. Does nothing for models that rewind without them.
fn checkpoint<B: Backend>(
    model: &Model,
    b: &mut B,
    state: &State<B>,
    n_pos: usize,
    checkpoints: &mut Vec<Checkpoint>,
) -> Result<()> {
    if n_pos == 0 {
        return Ok(()); // a new state is as good
    }
    let mut snapshot =
        (checkpoints.len() == MAX_CHECKPOINTS).then(|| checkpoints.remove(0).snapshot);
    if model.save_state(b, state, &mut snapshot)?
        && let Some(snapshot) = snapshot
    {
        checkpoints.push(Checkpoint { n_pos, snapshot });
    }
    Ok(())
}
