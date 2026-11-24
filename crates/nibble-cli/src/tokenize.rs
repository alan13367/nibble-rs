//! `nibble tokenize`: show how text tokenizes (for comparing with llama-tokenize).

use std::io::{self, BufWriter, Write};

use nibble::gguf::{Gguf, Storage};
use nibble::tokenizer::Tokenizer;

use crate::args::{Arg, Args, CliResult, ModelOptions, prompt_option};

pub(crate) const HELP: &str = "\
Usage: nibble tokenize -m <model.gguf> (-p TEXT | -f FILE) [options]

  -p, --prompt TEXT        Text to tokenize
  -f, --file FILE          Read the text from a file
      --ids                Print only the ids, as a list
      --no-parse-special   Treat control tokens like <|im_start|> as plain text
      --specials           List the special tokens instead
";

pub(crate) fn run(mut args: Args) -> CliResult<()> {
    let mut model = ModelOptions::new();
    let mut prompt = None;
    let (mut ids_only, mut parse_special, mut list_specials) = (false, true, false);
    while let Some(arg) = args.next()? {
        match arg {
            Arg::Flag(f)
                if model.parse(&f, &mut args)? || prompt_option(&f, &mut args, &mut prompt)? => {}
            Arg::Flag(f) => match f.as_str() {
                "--ids" => ids_only = true,
                "--no-parse-special" => parse_special = false,
                "--specials" => list_specials = true,
                "-h" | "--help" => {
                    print!("{HELP}");
                    return Ok(());
                }
                other => {
                    return Err(
                        format!("unknown option `{other}` (see `nibble tokenize --help`)").into(),
                    );
                }
            },
            Arg::Positional(p) => {
                return Err(format!("unexpected argument `{}`", p.to_string_lossy()).into());
            }
        }
    }

    // Only the tokenizer is needed, so this works for any architecture.
    let path = model
        .path
        .ok_or("missing model file: pass -m <model.gguf>")?;
    let storage = if model.mmap {
        // SAFETY: model files are read-only inputs; nothing modifies them while mapped.
        unsafe { Storage::open(&path) }
    } else {
        Storage::read(&path)
    }
    .map_err(|e| format!("{}: {e}", path.display()))?;
    let gguf = Gguf::parse(&storage).map_err(|e| format!("{}: {e}", path.display()))?;
    let tok = Tokenizer::from_gguf(&gguf).map_err(|e| format!("{}: {e}", path.display()))?;
    let tok = &tok;
    let mut out = BufWriter::new(io::stdout().lock());

    if list_specials {
        for &id in tok.special_tokens() {
            let kind = tok
                .token_kind(id)
                .map_or_else(String::new, |k| format!("{k:?}"));
            let eog = if tok.is_end_of_generation(id) {
                "  end-of-generation"
            } else {
                ""
            };
            writeln!(
                out,
                "{id:>8}  {kind:<12} {:?}{eog}",
                tok.token_text(id).unwrap_or("")
            )?;
        }
        out.flush()?;
        return Ok(());
    }

    let text = prompt.ok_or("nothing to tokenize: pass -p TEXT or -f FILE")?;
    let ids = tok.encode(&text, parse_special);
    if ids_only {
        writeln!(out, "{ids:?}")?;
    } else {
        for &id in &ids {
            let piece = String::from_utf8_lossy(tok.token_bytes(id, true));
            writeln!(out, "{id:>8}  {piece:?}")?;
        }
        writeln!(out, "{} tokens", ids.len())?;
    }
    out.flush()?;
    Ok(())
}
