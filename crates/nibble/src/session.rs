//! Inference sessions: one sequence's tokens, cache and logits.

use crate::error::{Error, Result};
use crate::model::{Model, State};
use crate::sampler::Sampler;

/// Options for [`Model::session`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SessionOptions {
    /// Maximum tokens the session can hold. `0` uses the model's training context.
    /// Larger values are capped at the training context.
    pub n_ctx: usize,
    /// Worker threads for matrix products. `0` uses all available cores.
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

/// A sequence being evaluated on a [`Model`]: tokens so far, cache, logits.
pub struct Session<'m> {
    model: &'m Model,
    n_ctx: usize,
    threads: usize,
    tokens: Vec<u32>,
    logits: Vec<f32>,
    logits_valid: bool,
    state: State,
}

impl std::fmt::Debug for Session<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Session")
            .field("n_ctx", &self.n_ctx)
            .field("threads", &self.threads)
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
            0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
            n => n,
        };
        Self {
            model,
            n_ctx,
            threads,
            tokens: Vec::new(),
            logits: vec![0.0; c.n_vocab],
            logits_valid: false,
            state: model.new_state(n_ctx),
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

    /// Worker threads used for matrix products.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
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
            // Only the last token's logits are needed.
            let logits = (i + 1 == tokens.len()).then_some(&mut self.logits[..]);
            self.model.forward(
                &mut self.state,
                token,
                self.tokens.len(),
                logits,
                self.threads,
            );
            self.tokens.push(token);
        }
        self.logits_valid = true;
        Ok(&self.logits)
    }

    /// Keeps the first `len` tokens and drops the rest.
    ///
    /// For models with recurrent layers, whose state cannot be rewound, this
    /// re-evaluates the kept tokens, which costs as much as evaluating them
    /// the first time.
    pub fn truncate(&mut self, len: usize) {
        if len >= self.tokens.len() {
            return;
        }
        self.tokens.truncate(len);
        self.logits_valid = false;
        if !self.model.truncate_state(&mut self.state, len) {
            self.state = self.model.new_state(self.n_ctx);
            for (pos, &token) in self.tokens.iter().enumerate() {
                self.model
                    .forward(&mut self.state, token, pos, None, self.threads);
            }
        }
    }

    /// Drops all tokens.
    pub fn clear(&mut self) {
        self.tokens.clear();
        self.logits_valid = false;
        self.state = self.model.new_state(self.n_ctx);
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
