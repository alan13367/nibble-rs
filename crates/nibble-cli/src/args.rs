//! Minimal argument parsing and the options shared by several commands.

use std::error::Error;
use std::ffi::OsString;
use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use nibble::sampler::SamplerConfig;
use nibble::{Model, SessionOptions};

pub(crate) type CliResult<T> = Result<T, Box<dyn Error>>;

/// One command-line item.
pub(crate) enum Arg {
    /// `-x`, `--name` (a `--name=value` value is kept for [`Args::value`]).
    Flag(String),
    /// Anything else.
    Positional(OsString),
}

pub(crate) struct Args {
    items: std::vec::IntoIter<OsString>,
    inline_value: Option<(String, OsString)>,
}

impl Args {
    pub(crate) fn new(items: impl IntoIterator<Item = OsString>) -> Self {
        Self {
            items: items.into_iter().collect::<Vec<_>>().into_iter(),
            inline_value: None,
        }
    }

    pub(crate) fn next(&mut self) -> CliResult<Option<Arg>> {
        if let Some((flag, _)) = self.inline_value.take() {
            return Err(format!("{flag} does not take a value").into());
        }
        let Some(item) = self.items.next() else {
            return Ok(None);
        };
        let Some(s) = item.to_str().filter(|s| s.starts_with('-') && s.len() > 1) else {
            return Ok(Some(Arg::Positional(item)));
        };
        if s.starts_with("--")
            && let Some((flag, value)) = s.split_once('=')
        {
            self.inline_value = Some((flag.to_owned(), value.into()));
            return Ok(Some(Arg::Flag(flag.to_owned())));
        }
        Ok(Some(Arg::Flag(s.to_owned())))
    }

    /// The value of `flag`: inline (`--flag=v`) or the next item.
    pub(crate) fn value(&mut self, flag: &str) -> CliResult<OsString> {
        if let Some((_, v)) = self.inline_value.take() {
            return Ok(v);
        }
        self.items
            .next()
            .ok_or_else(|| format!("{flag} needs a value").into())
    }

    pub(crate) fn string(&mut self, flag: &str) -> CliResult<String> {
        self.value(flag)?
            .into_string()
            .map_err(|_| format!("{flag}: value is not valid UTF-8").into())
    }

    pub(crate) fn parse<T: FromStr>(&mut self, flag: &str) -> CliResult<T>
    where
        T::Err: Display,
    {
        let v = self.string(flag)?;
        v.parse()
            .map_err(|e| format!("invalid value `{v}` for {flag}: {e}").into())
    }
}

/// `-m`, `-c`, `-t`, `--read`.
pub(crate) struct ModelOptions {
    pub(crate) path: Option<PathBuf>,
    pub(crate) n_ctx: usize,
    pub(crate) threads: usize,
    pub(crate) mmap: bool,
}

pub(crate) const MODEL_HELP: &str = "\
  -m, --model FILE         GGUF model file (required)
  -c, --ctx N              Context size in tokens (default 4096, 0 = model maximum)
  -t, --threads N          Threads (default: all cores)
      --read               Read the model into memory instead of memory-mapping it
";

impl ModelOptions {
    pub(crate) fn new() -> Self {
        Self {
            path: None,
            n_ctx: 4096,
            threads: 0,
            mmap: true,
        }
    }

    /// Handles `flag` if it is a model option.
    pub(crate) fn parse(&mut self, flag: &str, args: &mut Args) -> CliResult<bool> {
        match flag {
            "-m" | "--model" => self.path = Some(args.value(flag)?.into()),
            "-c" | "--ctx" => self.n_ctx = args.parse(flag)?,
            "-t" | "--threads" => self.threads = args.parse(flag)?,
            "--read" => self.mmap = false,
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(crate) fn load(&self) -> CliResult<Model> {
        let path = self
            .path
            .as_ref()
            .ok_or("missing model file: pass -m <model.gguf>")?;
        let start = Instant::now();
        let model = if self.mmap {
            // SAFETY: model files are read-only inputs. As in llama.cpp, we
            // assume nothing modifies or truncates the file while it is mapped.
            unsafe { Model::load_mmap(path) }
        } else {
            Model::load(path)
        }
        .map_err(|e| format!("{}: {e}", path.display()))?;
        let c = model.config();
        eprintln!(
            "nibble: loaded {} ({}, {}, {} layers) in {:.0} ms{}",
            c.name.as_deref().unwrap_or("model"),
            c.architecture.name(),
            c.weight_type,
            c.n_layer,
            start.elapsed().as_secs_f64() * 1e3,
            if model.is_mapped() {
                ", memory-mapped"
            } else {
                ""
            },
        );
        Ok(model)
    }

    pub(crate) fn session_options(&self) -> SessionOptions {
        SessionOptions {
            n_ctx: self.n_ctx,
            threads: self.threads,
        }
    }
}

/// Sampling flags.
pub(crate) struct SamplingOptions {
    config: SamplerConfig,
    seed: Option<u64>,
}

pub(crate) const SAMPLING_HELP: &str = "\
      --temp T             Temperature (default 0.8; 0 = greedy)
      --top-k K            Top-k (default 40; 0 = off)
      --top-p P            Top-p (default 0.95; 1 = off)
      --min-p P            Min-p (default 0.05; 0 = off)
      --repeat-penalty F   Repetition penalty (default 1 = off)
      --presence-penalty F Presence penalty (default 0)
      --frequency-penalty F Frequency penalty (default 0)
      --seed N             Random seed (default: random)
      --greedy             Same as --temp 0
";

impl SamplingOptions {
    pub(crate) fn new() -> Self {
        Self {
            config: SamplerConfig::default(),
            seed: None,
        }
    }

    /// Handles `flag` if it is a sampling option.
    pub(crate) fn parse(&mut self, flag: &str, args: &mut Args) -> CliResult<bool> {
        let c = &mut self.config;
        match flag {
            "--temp" => c.temperature = args.parse(flag)?,
            "--top-k" => c.top_k = args.parse(flag)?,
            "--top-p" => c.top_p = args.parse(flag)?,
            "--min-p" => c.min_p = args.parse(flag)?,
            "--repeat-penalty" => c.repeat_penalty = args.parse(flag)?,
            "--presence-penalty" => c.presence_penalty = args.parse(flag)?,
            "--frequency-penalty" => c.frequency_penalty = args.parse(flag)?,
            "--seed" => self.seed = Some(args.parse(flag)?),
            "--greedy" => c.temperature = 0.0,
            _ => return Ok(false),
        }
        Ok(true)
    }

    pub(crate) fn config(&self) -> SamplerConfig {
        let seed = self.seed.unwrap_or_else(|| {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64);
            nanos ^ u64::from(std::process::id()).rotate_left(32)
        });
        SamplerConfig {
            seed,
            ..self.config
        }
    }
}

/// Reads the prompt from `-p TEXT` or `-f FILE`.
pub(crate) fn prompt_option(
    flag: &str,
    args: &mut Args,
    prompt: &mut Option<String>,
) -> CliResult<bool> {
    match flag {
        "-p" | "--prompt" => *prompt = Some(args.string(flag)?),
        "-f" | "--file" => {
            let path = PathBuf::from(args.value(flag)?);
            *prompt = Some(
                std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?,
            );
        }
        _ => return Ok(false),
    }
    Ok(true)
}
