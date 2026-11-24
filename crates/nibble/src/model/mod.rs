//! Models: loading a GGUF file and dispatching to its architecture.
//!
//! This module is the only place that knows the list of architectures. To add
//! one:
//!
//! 1. add a variant to [`Architecture`] (and to `Architecture::ALL`);
//! 2. write a module with a model type (`load`, `new_state`, `forward`) and a
//!    state type, built from the blocks in `layers`;
//! 3. add arms to `Arch` and `State` and the dispatch functions below.
//!
//! Sessions, the CLI and the server never match on architectures.

mod bind;
mod config;
mod layers;
mod qwen3;
mod qwen35;

use std::path::Path;

use nibble_gguf::{Gguf, Storage};

pub use config::ModelConfig;

use crate::chat::ChatTemplate;
use crate::cpu::Cpu;
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
}

impl Architecture {
    /// Every supported architecture.
    pub const ALL: &'static [Self] = &[Self::Qwen3, Self::Qwen35];

    /// The GGUF architecture name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Qwen3 => "qwen3",
            Self::Qwen35 => "qwen35",
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

/// What a forward pass needs besides weights and state.
#[derive(Clone, Copy)]
pub(crate) struct Ctx<'a> {
    /// The model bytes that weight offsets point into.
    pub(crate) bytes: &'a [u8],
    pub(crate) config: &'a ModelConfig,
    /// The session's threads and kernels.
    pub(crate) cpu: &'a Cpu,
}

/// The architecture-specific weights and forward pass.
enum Arch {
    Qwen3(qwen3::Qwen3),
    Qwen35(qwen35::Qwen35),
}

/// Per-session state of an architecture (KV caches, recurrent state, scratch).
pub(crate) enum State {
    Qwen3(qwen3::State),
    Qwen35(qwen35::State),
}

/// A loaded model: weights (memory-mapped or in memory), tokenizer and config.
///
/// A `Model` is read-only and can serve many [`Session`]s.
pub struct Model {
    storage: Storage,
    config: ModelConfig,
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
        let (config, arch, tokenizer, chat_template) = {
            let g = Gguf::parse(&storage)?;
            let name = g.architecture().unwrap_or("(missing)");
            let architecture = Architecture::from_name(name)
                .ok_or_else(|| Error::UnsupportedArchitecture(name.to_owned()))?;
            let config = config::read(&g, architecture)?;
            let arch = match architecture {
                Architecture::Qwen3 => Arch::Qwen3(qwen3::Qwen3::load(&g, &config)?),
                Architecture::Qwen35 => Arch::Qwen35(qwen35::Qwen35::load(&g, &config)?),
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
            (config, arch, tokenizer, chat_template)
        };
        Ok(Self {
            storage,
            config,
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

    /// Whether the weights are memory-mapped.
    #[must_use]
    pub const fn is_mapped(&self) -> bool {
        self.storage.is_mapped()
    }

    /// Starts an inference session (its own cache) on this model.
    #[must_use]
    pub fn session(&self, options: SessionOptions) -> Session<'_> {
        Session::new(self, options)
    }

    /// Fresh per-session state for up to `n_ctx` positions.
    pub(crate) fn new_state(&self, n_ctx: usize) -> State {
        match &self.arch {
            Arch::Qwen3(_) => State::Qwen3(qwen3::Qwen3::new_state(&self.config, n_ctx)),
            Arch::Qwen35(m) => State::Qwen35(m.new_state(&self.config, n_ctx)),
        }
    }

    /// Runs `token` at position `pos` on `cpu`, writing the next-token logits
    /// when asked.
    pub(crate) fn forward(
        &self,
        cpu: &Cpu,
        state: &mut State,
        token: u32,
        pos: usize,
        logits: Option<&mut [f32]>,
    ) {
        let ctx = Ctx {
            bytes: &self.storage,
            config: &self.config,
            cpu,
        };
        match (&self.arch, state) {
            (Arch::Qwen3(m), State::Qwen3(s)) => m.forward(&ctx, s, token, pos, logits),
            (Arch::Qwen35(m), State::Qwen35(s)) => m.forward(&ctx, s, token, pos, logits),
            _ => unreachable!("session state belongs to a different architecture"),
        }
    }

    /// Rewinds `state` to its first `n_pos` positions. Returns false when the
    /// architecture cannot rewind (recurrent state), in which case the caller
    /// must rebuild the state from a [`Snapshot`] or by evaluating the kept
    /// tokens again.
    pub(crate) fn truncate_state(&self, state: &mut State, n_pos: usize) -> bool {
        match state {
            State::Qwen3(s) => {
                s.truncate(n_pos);
                true
            }
            // DeltaNet's recurrent state cannot be rewound.
            State::Qwen35(_) => false,
        }
    }

    /// Copies the parts of `state` that cannot be rewound into `snapshot`,
    /// reusing its buffers. Returns false, leaving `snapshot` as it is, for
    /// architectures whose state rewinds without one.
    pub(crate) fn save_state(&self, state: &State, snapshot: &mut Option<Snapshot>) -> bool {
        match state {
            State::Qwen3(_) => false,
            State::Qwen35(s) => {
                if let Some(Snapshot::Qwen35(target)) = snapshot {
                    s.save(target);
                } else {
                    let mut target = qwen35::Snapshot::default();
                    s.save(&mut target);
                    *snapshot = Some(Snapshot::Qwen35(target));
                }
                true
            }
        }
    }

    /// Returns `state` to the moment `snapshot` was taken, `n_pos` tokens
    /// in: the saved parts are copied back and the caches truncated.
    pub(crate) fn restore_state(&self, state: &mut State, snapshot: &Snapshot, n_pos: usize) {
        match (state, snapshot) {
            (State::Qwen35(s), Snapshot::Qwen35(saved)) => s.restore(saved, n_pos),
            _ => unreachable!("snapshot belongs to a different architecture"),
        }
    }
}

/// The parts of a [`State`] that cannot be rewound (recurrent layers), as
/// they were at some position.
pub(crate) enum Snapshot {
    Qwen35(qwen35::Snapshot),
}
