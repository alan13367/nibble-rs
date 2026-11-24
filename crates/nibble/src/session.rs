//! Inference sessions: one sequence's tokens, cache and logits.

use crate::cpu::{self, Cpu};
use crate::error::{Error, Result};
use crate::model::{Model, Snapshot, State};
use crate::sampler::Sampler;

/// Options for [`Model::session`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionOptions {
    /// Maximum tokens the session can hold. `0` uses the model's training context.
    /// Larger values are capped at the training context.
    pub n_ctx: usize,
    /// Threads that run the forward pass, the calling thread included. `0`
    /// uses one per physical core (where the OS reports cores, else one per
    /// logical CPU): a second thread per core does not make decoding faster.
    /// The threads live as long as the session.
    pub threads: usize,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            threads: 0,
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

/// A sequence being evaluated on a [`Model`]: tokens so far, cache, logits.
pub struct Session<'m> {
    model: &'m Model,
    n_ctx: usize,
    cpu: Cpu,
    tokens: Vec<u32>,
    logits: Vec<f32>,
    logits_valid: bool,
    state: State,
    /// Oldest first; positions increase.
    checkpoints: Vec<Checkpoint>,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("n_ctx", &self.n_ctx)
            .field("threads", &self.cpu.threads())
            .field("kernels", &self.cpu.kernels().name())
            .field("tokens", &self.tokens.len())
            .finish_non_exhaustive()
    }
}

impl<'m> Session<'m> {
    /// A new, empty session.
    #[must_use]
    pub fn new(model: &'m Model, options: SessionOptions) -> Self {
        let c = model.config();
        let n_ctx = match options.n_ctx {
            0 => c.n_ctx_train,
            n => n.min(c.n_ctx_train),
        };
        let threads = match options.threads {
            0 => cpu::default_threads(),
            n => n,
        };
        Self {
            model,
            n_ctx,
            cpu: Cpu::new(threads),
            tokens: Vec::new(),
            logits: vec![0.0; c.n_vocab],
            logits_valid: false,
            state: model.new_state(n_ctx),
            checkpoints: Vec::new(),
        }
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

    /// Threads that run the forward pass, the calling thread included.
    /// Fewer than requested if the system could not start them all.
    #[must_use]
    pub fn threads(&self) -> usize {
        self.cpu.threads()
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
    /// outside the vocabulary, [`Error::ContextFull`] if they do not fit.
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
        for (i, &token) in tokens.iter().enumerate() {
            if i + CHECKPOINT_TAIL == tokens.len() {
                self.checkpoint();
            }
            // Only the last token's logits are needed.
            let logits = (i + 1 == tokens.len()).then_some(&mut self.logits[..]);
            self.model
                .forward(&self.cpu, &mut self.state, token, self.tokens.len(), logits);
            self.tokens.push(token);
        }
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
    pub fn truncate(&mut self, len: usize) -> usize {
        if len >= self.tokens.len() {
            return 0;
        }
        self.tokens.truncate(len);
        self.logits_valid = false;
        if self.model.truncate_state(&mut self.state, len) {
            return 0;
        }
        self.checkpoints.retain(|c| c.n_pos <= len);
        let from = match self.checkpoints.last() {
            Some(c) => {
                self.model
                    .restore_state(&mut self.state, &c.snapshot, c.n_pos);
                c.n_pos
            }
            None => {
                self.state = self.model.new_state(self.n_ctx);
                0
            }
        };
        for pos in from..len {
            self.model
                .forward(&self.cpu, &mut self.state, self.tokens[pos], pos, None);
        }
        len - from
    }

    /// Drops all tokens.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.logits_valid = false;
        self.checkpoints.clear();
        self.state = self.model.new_state(self.n_ctx);
    }

    /// Saves what the state cannot rewind at the current position, evicting
    /// the oldest checkpoint (and reusing its buffers) when there are
    /// [`MAX_CHECKPOINTS`]. Does nothing for models that rewind without them.
    fn checkpoint(&mut self) {
        let n_pos = self.tokens.len();
        if n_pos == 0 {
            return; // a new state is as good
        }
        let mut snapshot = (self.checkpoints.len() == MAX_CHECKPOINTS)
            .then(|| self.checkpoints.remove(0).snapshot);
        if self.model.save_state(&self.state, &mut snapshot)
            && let Some(snapshot) = snapshot
        {
            self.checkpoints.push(Checkpoint { n_pos, snapshot });
        }
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
