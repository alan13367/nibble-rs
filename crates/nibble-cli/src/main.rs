//! The `nibble` command-line tool.

mod args;
mod bench;
mod chat;
mod inspect;
mod run;
mod serve;
mod style;
mod tokenize;

use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

use args::{Arg, Args, CliResult};

const USAGE: &str = "\
nibble - tiny GGUF inference engine

Usage:
  nibble run -m <model.gguf> -p <prompt>   Generate text from a prompt
  nibble chat -m <model.gguf>              Chat in the terminal
  nibble serve -m <model.gguf>             Serve the OpenAI API over HTTP
  nibble tokenize -m <model.gguf> -p TEXT  Show how text tokenizes
  nibble inspect <model.gguf>              Show a GGUF file's metadata and tensors
  nibble bench -m <model.gguf>             Measure prompt and decode speed

Run `nibble <command> --help` for a command's options.

Global options:
  -h, --help      Show this help
  -V, --version   Show the version
";

const INSPECT_HELP: &str = "\
Usage: nibble inspect <model.gguf> [options]

  --all           Print arrays and long strings in full
  --no-metadata   Skip the metadata listing
  --no-tensors    Skip the tensor table
  --read          Read the file into memory instead of memory-mapping it
";

fn main() -> ExitCode {
    match run(std::env::args_os().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        // Output piped into `head` and similar: stop quietly.
        Err(e) if is_broken_pipe(e.as_ref()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<OsString>) -> CliResult<()> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        eprint!("{USAGE}");
        return Err("missing command".into());
    };
    let rest = Args::new(args);
    match command.to_str() {
        Some("run") => run::run(rest),
        Some("chat") => chat::run(rest),
        Some("bench") => bench::run(rest),
        Some("serve") => serve::run(rest),
        Some("tokenize") => tokenize::run(rest),
        Some("inspect") => parse_inspect(rest).and_then(|opts| match opts {
            Some(opts) => inspect::run(&opts),
            None => {
                print!("{INSPECT_HELP}");
                Ok(())
            }
        }),
        Some("-h" | "--help" | "help") => {
            print!("{USAGE}");
            Ok(())
        }
        Some("-V" | "--version") => {
            println!("nibble {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        _ => Err(format!(
            "unknown command `{}` (see `nibble --help`)",
            command.to_string_lossy()
        )
        .into()),
    }
}

/// Parses `inspect` options; `None` means help was requested.
fn parse_inspect(mut args: Args) -> CliResult<Option<inspect::Options>> {
    let mut path = None;
    let mut options = inspect::Options {
        path: PathBuf::new(),
        all: false,
        metadata: true,
        tensors: true,
        mmap: true,
    };
    while let Some(arg) = args.next()? {
        match arg {
            Arg::Flag(f) => match f.as_str() {
                "--all" => options.all = true,
                "--no-metadata" => options.metadata = false,
                "--no-tensors" => options.tensors = false,
                "--read" => options.mmap = false,
                "-h" | "--help" => return Ok(None),
                other => {
                    return Err(
                        format!("unknown option `{other}` (see `nibble inspect --help`)").into(),
                    );
                }
            },
            Arg::Positional(p) if path.is_none() => path = Some(PathBuf::from(p)),
            Arg::Positional(_) => return Err("inspect takes one file".into()),
        }
    }
    options.path = path.ok_or("inspect needs a file: nibble inspect <model.gguf>")?;
    Ok(Some(options))
}

fn is_broken_pipe(e: &(dyn Error + 'static)) -> bool {
    e.downcast_ref::<io::Error>()
        .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
}
