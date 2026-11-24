//! `nibble chat`: an interactive conversation in the terminal.

use std::io::{self, BufRead, Write};
use std::time::Instant;

use nibble::chat::{ChatOptions, Message};
use nibble::sampler::Sampler;
use nibble::tokenizer::StreamDecoder;
use nibble::{Session, StopReason};

use crate::args::{Arg, Args, CliResult, MODEL_HELP, ModelOptions, SAMPLING_HELP, SamplingOptions};

pub(crate) fn help() -> String {
    format!(
        "Usage: nibble chat -m <model.gguf> [options]\n\n\
         Type a message and press Enter. Commands: /clear (new conversation), /exit.\n\n\
         Chat:\n      \
         --system TEXT        System message\n      \
         --no-think           Disable thinking (for models that support it)\n  \
         -n, --max-tokens N       Maximum tokens per reply (default 2048)\n\n\
         Model:\n{MODEL_HELP}\nSampling:\n{SAMPLING_HELP}"
    )
}

pub(crate) fn run(mut args: Args) -> CliResult<()> {
    let mut model_opts = ModelOptions::new();
    let mut sampling = SamplingOptions::new();
    let mut system = None;
    let (mut think, mut max_tokens) = (true, 2048usize);
    while let Some(arg) = args.next()? {
        match arg {
            Arg::Flag(f)
                if model_opts.parse(&f, &mut args)? || sampling.parse(&f, &mut args)? => {}
            Arg::Flag(f) => match f.as_str() {
                "--system" => system = Some(args.string(&f)?),
                "--no-think" => think = false,
                "-n" | "--max-tokens" => max_tokens = args.parse(&f)?,
                "-h" | "--help" => {
                    print!("{}", help());
                    return Ok(());
                }
                other => {
                    return Err(
                        format!("unknown option `{other}` (see `nibble chat --help`)").into(),
                    );
                }
            },
            Arg::Positional(p) => {
                return Err(format!("unexpected argument `{}`", p.to_string_lossy()).into());
            }
        }
    }

    let model = model_opts.load()?;
    let template = model
        .chat_template()
        .ok_or("this model's chat template is not supported yet")?;
    let tok = model.tokenizer();
    let mut session = model.session(model_opts.session_options());
    let mut sampler = Sampler::new(sampling.config());
    let options = ChatOptions {
        add_generation_prompt: true,
        enable_thinking: think,
        ..ChatOptions::default()
    };
    let start_messages: Vec<Message> = system.into_iter().map(Message::system).collect();
    let mut messages = start_messages.clone();
    eprintln!("nibble: chat ready. /clear starts over, /exit quits.");

    let stdin = io::stdin();
    let mut line = String::new();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        line.clear();
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            return Ok(());
        }
        let input = line.trim();
        match input {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/clear" => {
                messages = start_messages.clone();
                session.clear();
                eprintln!("nibble: conversation cleared");
                continue;
            }
            _ => {}
        }
        messages.push(Message::user(input));
        let prompt = tok.encode(&template.render(&messages, options)?, true);
        let new_tokens = reuse_prefix(&mut session, &prompt);

        let start = Instant::now();
        let mut reply = String::new();
        let mut decoder = StreamDecoder::new();
        let mut stdout = io::stdout().lock();
        let mut generated = 0usize;
        let stop = session.generate(new_tokens, &mut sampler, max_tokens, |token, _| {
            generated += 1;
            let text = decoder.push(tok, token, false);
            reply.push_str(&text);
            stdout
                .write_all(text.as_bytes())
                .and_then(|()| stdout.flush())
                .is_ok()
        });
        let tail = decoder.finish();
        reply.push_str(&tail);
        writeln!(stdout, "{tail}")?;
        drop(stdout);
        match stop {
            Ok(StopReason::ContextFull) => {
                eprintln!("nibble: context full; use /clear to start over")
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("nibble: {e}");
                messages.pop();
                session.clear();
                continue;
            }
        }
        let secs = start.elapsed().as_secs_f64();
        // The time includes evaluating this turn's prompt.
        eprintln!("[{generated} tokens in {secs:.1} s]");
        messages.push(Message::assistant(reply));
    }
}

/// Keeps the longest cached prefix of `prompt` and returns the tokens still to evaluate.
/// At least one token is always re-evaluated, to get fresh logits.
fn reuse_prefix<'p>(session: &mut Session<'_>, prompt: &'p [u32]) -> &'p [u32] {
    let common = session
        .tokens()
        .iter()
        .zip(prompt)
        .take_while(|(a, b)| a == b)
        .count();
    let keep = common.min(prompt.len().saturating_sub(1));
    session.truncate(keep);
    &prompt[keep..]
}
