//! Models: loading a GGUF file and dispatching to its architecture.
//!
//! This module is the only place that knows the list of architectures. To add
//! one:
//!
//! 1. add a variant to [`Architecture`] (and to `Architecture::ALL`);
//! 2. write a module with a model type (`load`, `new_state`, `forward`) and a
//!    state type, built from the blocks in `layers` and generic over the
//!    [`Backend`] it runs on;
//! 3. add arms to `Arch` and `State` and the dispatch functions below.
//!
//! Sessions, the CLI and the server never match on architectures.

mod bind;
mod config;
mod layers;
mod qwen3;
mod qwen35;

use std::ops::Range;
use std::path::Path;

use nibble_gguf::{Gguf, Storage};

pub use config::ModelConfig;

use crate::backend::{Backend, MAX_LOGIT_ROWS, ModelData, Params};
use crate::chat::ChatTemplate;
use crate::error::{Error, Result};
use crate::session::{Session, SessionOptions};
use crate::tokenizer::Tokenizer;

/// A supported model architecture (`general.architecture`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Architecture {
    /// Qwen3 dense models (`qwen3`).
    Qwen3,
    /// Qwen3.5 hybrid DeltaNet / attention models (`qwen35`).
    Qwen35,
    /// Qwen3.5 hybrid models with mixture-of-experts feed-forwards (`qwen35moe`).
    Qwen35Moe,
}

impl Architecture {
    /// Every supported architecture.
    pub const ALL: &'static [Self] = &[Self::Qwen3, Self::Qwen35, Self::Qwen35Moe];

    /// The GGUF architecture name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen35",
            Self::Qwen35Moe => "qwen35moe",
        }
    }

    /// The architecture with GGUF name `name`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|a| a.name() == name)
    }
}

impl std::fmt::Display for Architecture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The architecture-specific weights and forward pass.
enum Arch {
    Qwen3(qwen3::Qwen3),
    Qwen35(qwen35::Qwen35),
}

/// Per-session state of an architecture (KV caches, recurrent state, scratch)
/// on backend `B`.
pub(crate) enum State<B: Backend> {
    Qwen3(qwen3::State<B>),
    Qwen35(qwen35::State<B>),
}

/// A loaded model: weights (memory-mapped or in memory), tokenizer and config.
///
/// A `Model` is read-only and can serve many [`Session`]s.
pub struct Model {
    storage: Storage,
    config: ModelConfig,
    /// The small `f32` parameters (norm weights and the like) of every layer.
    params: Vec<f32>,
    /// Bytes of the largest tensor.
    largest_tensor: usize,
    arch: Arch,
    tokenizer: Tokenizer,
    chat_template: Option<ChatTemplate>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("config", &self.config)
            .field("storage", &self.storage)
            .finish_non_exhaustive()
    }
}

impl Model {
    /// Loads a GGUF model by reading the whole file into memory.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read, is not valid GGUF, or describes an
    /// unsupported model.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::from_storage(Storage::read(path)?)
    }

    /// Loads a GGUF model by memory-mapping it: fast, and the weights are
    /// shared with the OS page cache instead of copied.
    ///
    /// # Safety
    ///
    /// The file must not be modified or truncated while the model is alive
    /// (see [`Storage::open`]).
    ///
    /// # Errors
    ///
    /// As [`Model::load`].
    pub unsafe fn load_mmap(path: impl AsRef<Path>) -> Result<Self> {
        // SAFETY: the caller upholds `Storage::open`'s contract.
        Self::from_storage(unsafe { Storage::open(path)? })
    }

    /// Builds a model from GGUF bytes.
    ///
    /// # Errors
    ///
    /// As [`Model::load`].
    pub fn from_storage(storage: Storage) -> Result<Self> {
        let (config, params, largest_tensor, arch, tokenizer, chat_template) = {
            let g = Gguf::parse(&storage)?;
            let largest_tensor = g
                .tensors()
                .iter()
                .filter_map(|t| t.data().map(<[u8]>::len))
                .max()
                .unwrap_or(0);
            let name = g.architecture().unwrap_or("(missing)");
            let architecture = Architecture::from_name(name)
                .ok_or_else(|| Error::UnsupportedArchitecture(name.to_owned()))?;
            let config = config::read(&g, architecture)?;
            let mut params = Params::default();
            let p = &mut params;
            let arch = match architecture {
                Architecture::Qwen3 => Arch::Qwen3(qwen3::Qwen3::load(&g, &config, p)?),
                Architecture::Qwen35 | Architecture::Qwen35Moe => {
                    Arch::Qwen35(qwen35::Qwen35::load(&g, &config, p)?)
                }
            };
            let tokenizer = Tokenizer::from_gguf(&g)?;
            if tokenizer.vocab_size() < config.n_vocab {
                return Err(Error::Tokenizer(format!(
                    "vocabulary has {} tokens but the embedding matrix has {} rows",
                    tokenizer.vocab_size(),
                    config.n_vocab
                )));
            }
            let chat_template = g
                .get("tokenizer.chat_template")
                .and_then(|v| v.as_str())
                .and_then(ChatTemplate::detect);
            (
                config,
                params.into_vec(),
                largest_tensor,
                arch,
                tokenizer,
                chat_template,
            )
        };
        Ok(Self {
            storage,
            config,
            params,
            largest_tensor,
            arch,
            tokenizer,
            chat_template,
        })
    }

    /// The model's hyperparameters.
    #[must_use]
    pub const fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The model's tokenizer.
    #[must_use]
    pub const fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The chat template family, if the model's template is recognized.
    #[must_use]
    pub const fn chat_template(&self) -> Option<ChatTemplate> {
        self.chat_template
    }

    /// Whether the file has a multi-token-prediction (MTP) head, with which
    /// a session can draft tokens (see [`SessionOptions::draft_max`]).
    #[must_use]
    pub const fn has_mtp(&self) -> bool {
        match &self.arch {
            Arch::Qwen3(_) => false,
            Arch::Qwen35(m) => m.has_mtp(),
        }
    }

    /// Whether the weights are memory-mapped.
    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        self.storage.is_mapped()
    }

    /// Starts an inference session (its own cache) on this model.
    ///
    /// # Errors
    ///
    /// Fails if the backend asked for cannot start (no Metal device, for
    /// instance) or cannot allocate the session's buffers.
    pub fn session(&self, options: SessionOptions) -> Result<Session<'_>> {
        Session::new(self, options)
    }

    /// What backends read from this model.
    pub(crate) fn data(&self) -> ModelData<'_> {
        ModelData {
            bytes: &self.storage,
            mapped: self.storage.is_mapped(),
            params: &self.params,
            config: &self.config,
            largest_tensor: self.largest_tensor,
        }
    }

    /// Fresh per-session state on `b` for up to `n_ctx` positions, evaluated
    /// in batches of up to `n_batch` tokens; with the MTP head's state if
    /// `mtp` (and the model has one), which every forward pass then
    /// updates.
    pub(crate) fn new_state<B: Backend>(
        &self,
        b: &mut B,
        n_ctx: usize,
        n_batch: usize,
        mtp: bool,
    ) -> Result<State<B>> {
        let c = &self.config;
        Ok(match &self.arch {
            Arch::Qwen3(_) => State::Qwen3(qwen3::Qwen3::new_state(b, c, n_ctx, n_batch)?),
            Arch::Qwen35(m) => State::Qwen35(m.new_state(b, c, n_ctx, n_batch, mtp)?),
        })
    }

    /// `drafts.len()` greedy draft steps of the MTP head (the state has its
    /// state) after `token` at `pos`, chosen on the device.
    pub(crate) fn draft_chain<B: Backend>(
        &self,
        b: &mut B,
        state: &mut State<B>,
        token: u32,
        pos: usize,
        drafts: &mut [u32],
    ) -> Result<()> {
        let c = &self.config;
        match (&self.arch, state) {
            (Arch::Qwen35(m), State::Qwen35(s)) => m.draft_chain(b, c, s, token, pos, drafts),
            _ => unreachable!("drafting on a model without an MTP head"),
        }
    }

    /// One draft step of the MTP head (the state has its state): `token`
    /// at `pos`, the position after the state's, paired with the trunk's
    /// last hidden state (`first`) or the head's output at the step before;
    /// writes the logits of the token after it to `logits`.
    pub(crate) fn draft<B: Backend>(
        &self,
        b: &mut B,
        state: &mut State<B>,
        token: u32,
        pos: usize,
        first: bool,
        logits: &mut [f32],
    ) -> Result<()> {
        let c = &self.config;
        match (&self.arch, state) {
            (Arch::Qwen35(m), State::Qwen35(s)) => m.draft(b, c, s, token, pos, first, logits),
            _ => unreachable!("drafting on a model without an MTP head"),
        }
    }

    /// Runs `tokens` (at most the state's `n_batch`) at positions from `pos`
    /// on, on `b`, writing the logits asked for.
    pub(crate) fn forward<B: Backend>(
        &self,
        b: &mut B,
        state: &mut State<B>,
        tokens: &[u32],
        pos: usize,
        logits: Option<Logits<'_>>,
    ) -> Result<()> {
        debug_assert!(logits.as_ref().is_none_or(|l| {
            l.rows.end <= tokens.len()
                && l.rows.len() <= MAX_LOGIT_ROWS
                && l.out.len() >= l.rows.len() * self.config.n_vocab
        }));
        let c = &self.config;
        match (&self.arch, state) {
            (Arch::Qwen3(m), State::Qwen3(s)) => m.forward(b, c, s, tokens, pos, logits),
            (Arch::Qwen35(m), State::Qwen35(s)) => m.forward(b, c, s, tokens, pos, logits),
            _ => unreachable!("session state belongs to a different architecture"),
        }
    }

    /// Rewinds `state` from `from` to its first `to` positions. Returns
    /// false, changing nothing, when the architecture cannot (recurrent
    /// state, except into the pending tokens of a short last batch), in
    /// which case the caller must rebuild the state from a [`Snapshot`] or
    /// by evaluating the kept tokens again.
    pub(crate) fn truncate_state<B: Backend>(
        &self,
        b: &mut B,
        state: &mut State<B>,
        from: usize,
        to: usize,
    ) -> Result<bool> {
        match state {
            State::Qwen3(s) => {
                s.truncate(b, to)?;
                Ok(true)
            }
            State::Qwen35(s) => s.rewind(b, from, to),
        }
    }

    /// Copies the parts of `state` that cannot be rewound into `snapshot`,
    /// reusing its buffers. Returns false, leaving `snapshot` as it is, for
    /// architectures whose state rewinds without one.
    pub(crate) fn save_state<B: Backend>(
        &self,
        b: &mut B,
        state: &State<B>,
        snapshot: &mut Option<Snapshot>,
    ) -> Result<bool> {
        match (&self.arch, state) {
            (Arch::Qwen35(m), State::Qwen35(s)) => {
                if let Some(Snapshot::Qwen35(target)) = snapshot {
                    s.save(b, m.dims(), target)?;
                } else {
                    let mut target = qwen35::Snapshot::default();
                    s.save(b, m.dims(), &mut target)?;
                    *snapshot = Some(Snapshot::Qwen35(target));
                }
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    /// Returns `state` to the moment `snapshot` was taken, `n_pos` tokens
    /// in: the saved parts are copied back and the caches truncated.
    pub(crate) fn restore_state<B: Backend>(
        &self,
        b: &mut B,
        state: &mut State<B>,
        snapshot: &mut Snapshot,
        n_pos: usize,
    ) -> Result<()> {
        match (&self.arch, state, snapshot) {
            (Arch::Qwen35(m), State::Qwen35(s), Snapshot::Qwen35(saved)) => {
                s.restore(b, m.dims(), saved, n_pos)
            }
            _ => unreachable!("snapshot belongs to a different architecture"),
        }
    }
}

/// The logits a forward pass computes: those of the batch rows `rows` (at
/// most [`crate::backend::MAX_LOGIT_ROWS`]), one after another in `out`.
pub(crate) struct Logits<'a> {
    pub(crate) rows: Range<usize>,
    pub(crate) out: &'a mut [f32],
}

/// The parts of a [`State`] that cannot be rewound (recurrent layers), as
/// they were at some position.
pub(crate) enum Snapshot {
    Qwen35(qwen35::Snapshot),
}
