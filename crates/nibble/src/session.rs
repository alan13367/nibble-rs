//! Inference sessions: one sequence's tokens, cache and logits.

use crate::backend::{Backend, BackendKind};
use crate::cpu::{self, CpuBackend};
use crate::error::{Error, Result};
#[cfg(target_os = "macos")]
use crate::metal::MetalBackend;
use crate::model::{Model, Snapshot, State};
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
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            threads: 0,
            backend: BackendKind::default(),
        }
    }
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
    ) -> Result<(Self, Option<String>)> {
        let cpu = |threads: usize| -> Result<Self> {
            let threads = match threads {
                0 => cpu::default_threads(),
                n => n,
            };
            let mut backend = CpuBackend::new(model.data(), n_ctx, threads);
            let state = model.new_state(&mut backend, n_ctx)?;
            Ok(Self::Cpu(Run { backend, state }))
        };
        match options.backend {
            BackendKind::Cpu => Ok((cpu(options.threads)?, None)),
            BackendKind::Metal => Ok((Self::metal(model, n_ctx)?, None)),
            // Only Metal exists to try, and only on macOS.
            BackendKind::Auto if !cfg!(target_os = "macos") => Ok((cpu(options.threads)?, None)),
            BackendKind::Auto => match Self::metal(model, n_ctx) {
                Ok(engine) => Ok((engine, None)),
                Err(e) => Ok((cpu(options.threads)?, Some(e.to_string()))),
            },
        }
    }

    #[cfg(target_os = "macos")]
    fn metal(model: &'m Model, n_ctx: usize) -> Result<Self> {
        let mut backend = MetalBackend::new(model.data(), n_ctx)?;
        let state = model.new_state(&mut backend, n_ctx)?;
        Ok(Self::Metal(Run { backend, state }))
    }

    #[cfg(not(target_os = "macos"))]
    fn metal(_model: &'m Model, _n_ctx: usize) -> Result<Self> {
        Err(Error::Backend(
            "Metal is only available on macOS".to_owned(),
        ))
    }
}

/// A sequence being evaluated on a [`Model`]: tokens so far, cache, logits.
pub struct Session<'m> {
    model: &'m Model,
    n_ctx: usize,
    engine: Engine<'m>,
    /// Why [`BackendKind::Auto`] fell back to the CPU.
    fallback: Option<String>,
    tokens: Vec<u32>,
    logits: Vec<f32>,
    logits_valid: bool,
    /// Oldest first; positions increase.
    checkpoints: Vec<Checkpoint>,
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
    /// allocate the session's buffers.
    pub fn new(model: &'m Model, options: SessionOptions) -> Result<Self> {
        let c = model.config();
        let n_ctx = match options.n_ctx {
            0 => c.n_ctx_train,
            n => n.min(c.n_ctx_train),
        };
        let (engine, fallback) = Engine::new(model, options, n_ctx)?;
        Ok(Self {
            model,
            n_ctx,
            engine,
            fallback,
            tokens: Vec::new(),
            logits: vec![0.0; c.n_vocab],
            logits_valid: false,
            checkpoints: Vec::new(),
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
        self.logits_valid.then_some(&self.logits[..])
    }

    /// Appends `tokens` and returns the logits predicting the token after them.
    ///
    /// # Errors
    ///
    /// [`Error::EmptyInput`] for no tokens, [`Error::InvalidToken`] for ids
    /// outside the vocabulary, [`Error::ContextFull`] if they do not fit,
    /// [`Error::Backend`] if the backend fails.
    pub fn eval(&mut self, tokens: &[u32]) -> Result<&[f32]> {
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
        self.logits_valid = false;
        let model = self.model;
        on_engine!(&mut self.engine, |b, state| {
            for (i, &token) in tokens.iter().enumerate() {
                if i + CHECKPOINT_TAIL == tokens.len() {
                    checkpoint(model, b, state, self.tokens.len(), &mut self.checkpoints)?;
                }
                // Only the last token's logits are needed.
                let logits = (i + 1 == tokens.len()).then_some(&mut self.logits[..]);
                model.forward(b, state, token, self.tokens.len(), logits)?;
                self.tokens.push(token);
            }
        });
        self.logits_valid = true;
        Ok(&self.logits)
    }

    /// Keeps the first `len` tokens and drops the rest. Returns how many of
    /// the kept tokens had to be evaluated again.
    ///
    /// That is none for most models. Models with recurrent layers cannot
    /// rewind their state, so the session goes back to the latest checkpoint
    /// at or before `len` and evaluates the tokens after it. A checkpoint is
    /// taken [`CHECKPOINT_TAIL`] tokens before the end of every evaluation of
    /// at least that many tokens: in a chat, the next turn's prompt usually
    /// diverges from the last one just there.
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
        self.tokens.truncate(len);
        let model = self.model;
        let n_ctx = self.n_ctx;
        on_engine!(&mut self.engine, |b, state| {
            if model.truncate_state(b, state, len)? {
                return Ok(0);
            }
            self.checkpoints.retain(|c| c.n_pos <= len);
            let from = match self.checkpoints.last() {
                Some(c) => {
                    model.restore_state(b, state, &c.snapshot, c.n_pos)?;
                    c.n_pos
                }
                None => {
                    *state = model.new_state(b, n_ctx)?;
                    0
                }
            };
            for (pos, &token) in self.tokens.iter().enumerate().skip(from) {
                model.forward(b, state, token, pos, None)?;
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
        on_engine!(&mut self.engine, |b, state| {
            *state = model.new_state(b, n_ctx)?;
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
        for _ in 0..max_tokens {
            let token = sampler.sample(&self.logits, &self.tokens);
            if self.model.tokenizer().is_end_of_generation(token) {
                return Ok(StopReason::EndOfGeneration(token));
            }
            if !on_token(token, &self.logits) {
                return Ok(StopReason::Stopped);
            }
            if self.tokens.len() >= self.n_ctx {
                return Ok(StopReason::ContextFull);
            }
            self.eval(&[token])?;
        }
        Ok(StopReason::MaxTokens)
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
