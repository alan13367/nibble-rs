//! Minimal argument parsing and the options shared by several commands.

use std::error::Error;
use std::ffi::OsString;
use std::fmt::Display;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nibble::sampler::SamplerConfig;
use nibble::{BackendKind, KvCacheType, Model, Session, SessionOptions};

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

/// `-m`, `-c`, `-t`, `--backend`, `--draft`, `-ctk`, `-ctv`, `--read`.
pub(crate) struct ModelOptions {
    pub(crate) path: Option<PathBuf>,
    pub(crate) n_ctx: usize,
    pub(crate) threads: usize,
    pub(crate) backend: BackendKind,
    pub(crate) draft: usize,
    pub(crate) cache_type_k: KvCacheType,
    pub(crate) cache_type_v: KvCacheType,
    pub(crate) mmap: bool,
}

pub(crate) const MODEL_HELP: &str = "  -m, --model FILE         GGUF model file (required)
  -c, --ctx N              Context size in tokens (default 4096, 0 = model maximum)
  -t, --threads N          Threads (default: performance cores on Apple Silicon,
                           else one per physical core)
      --backend B          auto, cpu, metal or cuda (default auto: Metal or CUDA
                           where available, else the CPU)
      --draft N            Draft up to N tokens after each one with the model's
                           MTP head and check them in one pass (speculative
                           decoding; same output). Default 0 = off
  -ctk, --cache-type-k T   Type of the cached attention keys: f32, f16, bf16,
                           q8_0, q4_0, q4_1, iq4_nl, q5_0 or q5_1 (default f16)
  -ctv, --cache-type-v T   Type of the cached attention values (same types,
                           default f16). A quantized type computes attention as
                           llama.cpp's flash attention does
      --read               Read the model into memory instead of memory-mapping it
";

/// Says on stderr why a session asked to use the GPU where available runs
/// on the CPU.
pub(crate) fn report_fallback(session: &Session<'_>) {
    if let Some(reason) = session.fallback_reason() {
        eprintln!("nibble: not using the GPU ({reason}); running on the CPU");
    }
}

impl ModelOptions {
    pub(crate) fn new() -> Self {
        Self {
            path: None,
            n_ctx: 4096,
            threads: 0,
            backend: BackendKind::Auto,
            draft: 0,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            mmap: true,
        }
    }

    /// Handles `flag` if it is a model option.
    pub(crate) fn parse(&mut self, flag: &str, args: &mut Args) -> CliResult<bool> {
        match flag {
            "-m" | "--model" => self.path = Some(args.value(flag)?.into()),
            "-c" | "--ctx" => self.n_ctx = args.parse(flag)?,
            "-t" | "--threads" => self.threads = args.parse(flag)?,
            "--backend" => {
                let name = args.string(flag)?;
                self.backend = BackendKind::from_name(&name).ok_or_else(|| {
                    format!("unknown backend `{name}` (expected auto, cpu, metal or cuda)")
                })?;
            }
            "--draft" => self.draft = args.parse(flag)?,
            "-ctk" | "--cache-type-k" => self.cache_type_k = cache_type(flag, args)?,
            "-ctv" | "--cache-type-v" => self.cache_type_v = cache_type(flag, args)?,
            "--read" => self.mmap = false,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Loads the model and reports it on stderr.
    pub(crate) fn load(&self) -> CliResult<Model> {
        let (model, elapsed) = self.load_quietly()?;
        let c = model.config();
        eprintln!(
            "nibble: loaded {} ({}, {}, {} layers) in {:.0} ms{}",
            c.name.as_deref().unwrap_or("model"),
            c.architecture.name(),
            c.weight_type,
            c.n_layer,
            elapsed.as_secs_f64() * 1e3,
            if model.is_mapped() {
                ", memory-mapped"
            } else {
                ""
            },
        );
        Ok(model)
    }

    /// Loads the model without printing, and says how long it took.
    pub(crate) fn load_quietly(&self) -> CliResult<(Model, Duration)> {
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
        Ok((model, start.elapsed()))
    }

    pub(crate) fn session_options(&self) -> SessionOptions {
        SessionOptions {
            n_ctx: self.n_ctx,
            threads: self.threads,
            backend: self.backend,
            n_batch: 0,
            draft_max: self.draft,
            cache_type_k: self.cache_type_k,
            cache_type_v: self.cache_type_v,
        }
    }
}

/// The cache type named by the value of `flag`.
fn cache_type(flag: &str, args: &mut Args) -> CliResult<KvCacheType> {
    let name = args.string(flag)?;
    KvCacheType::from_name(&name).ok_or_else(|| {
        let names: Vec<&str> = KvCacheType::ALL.iter().map(|t| t.name()).collect();
        format!(
            "unknown cache type `{name}` for {flag} (expected one of {})",
            names.join(", ")
        )
        .into()
    })
}

/// Sampling flags.
pub(crate) struct SamplingOptions {
    config: SamplerConfig,
    seed: Option<u64>,
}

pub(crate) const SAMPLING_HELP: &str =
    "      --temp T             Temperature (default 0.8; 0 = greedy)
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
