//! The `nibble` command-line tool.

mod inspect;

use std::error::Error;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
nibble - tiny GGUF inference engine

Usage:
  nibble inspect <model.gguf> [options]   Show a GGUF file's metadata and tensors

Options for inspect:
  --all           Print arrays and long strings in full
  --no-metadata   Skip the metadata listing
  --no-tensors    Skip the tensor table
  --read          Read the file into memory instead of memory-mapping it

Global options:
  -h, --help      Show this help
  -V, --version   Show the version
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

fn run(args: Vec<OsString>) -> Result<(), Box<dyn Error>> {
    let mut args = args.into_iter();
    let Some(command) = args.next() else {
        eprint!("{USAGE}");
        return Err("missing command".into());
    };
    match command.to_str() {
        Some("inspect") => inspect::run(&parse_inspect(args)?),
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

fn parse_inspect(args: impl Iterator<Item = OsString>) -> Result<inspect::Options, Box<dyn Error>> {
    let mut path = None;
    let mut options = inspect::Options {
        path: PathBuf::new(),
        all: false,
        metadata: true,
        tensors: true,
        mmap: true,
    };
    for arg in args {
        match arg.to_str() {
            Some("--all") => options.all = true,
            Some("--no-metadata") => options.metadata = false,
            Some("--no-tensors") => options.tensors = false,
            Some("--read") => options.mmap = false,
            Some(flag) if flag.starts_with('-') => {
                return Err(format!("unknown option `{flag}`").into());
            }
            _ if path.is_none() => path = Some(PathBuf::from(arg)),
            _ => return Err("inspect takes one file".into()),
        }
    }
    options.path = path.ok_or("inspect needs a file: nibble inspect <model.gguf>")?;
    Ok(options)
}

fn is_broken_pipe(e: &(dyn Error + 'static)) -> bool {
    e.downcast_ref::<io::Error>()
        .is_some_and(|e| e.kind() == io::ErrorKind::BrokenPipe)
}
