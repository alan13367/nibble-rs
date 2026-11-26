//! `nibble run`: generate a completion for one prompt.

use std::io::{self, Write};
use std::time::Instant;

use nibble::chat::{ChatOptions, Message};
use nibble::sampler::Sampler;
use nibble::tokenizer::StreamDecoder;
use nibble::{Model, StopReason};

use crate::args::{
    Arg, Args, CliResult, MODEL_HELP, ModelOptions, SAMPLING_HELP, SamplingOptions, prompt_option,
};

pub(crate) fn help() -> String {
    format!(
        "Usage: nibble run -m <model.gguf> (-p TEXT | -f FILE) [options]\n\n\
         Prompt:\n  \
         -p, --prompt TEXT        Prompt text (special tokens like <|im_start|> are recognized)\n  \
         -f, --file FILE          Read the prompt from a file\n      \
         --chat               Wrap the prompt as a user message in the model's chat template\n      \
         --system TEXT        System message (with --chat)\n      \
         --no-think           Disable thinking (with --chat, for models that support it)\n  \
         -n, --max-tokens N       Tokens to generate (default 256)\n      \
         --special            Print control tokens too\n      \
         --probs N            Print the N most likely tokens at each step (to stderr)\n\n\
         Model:\n{MODEL_HELP}\nSampling:\n{SAMPLING_HELP}"
    )
}

pub(crate) fn run(mut args: Args) -> CliResult<()> {
    let mut model_opts = ModelOptions::new();
    let mut sampling = SamplingOptions::new();
    let mut prompt = None;
    let mut system = None;
    let (mut chat, mut think, mut special, mut max_tokens, mut probs) =
        (false, true, false, 256usize, 0usize);
    while let Some(arg) = args.next()? {
        match arg {
            Arg::Flag(f)
                if model_opts.parse(&f, &mut args)?
                    || sampling.parse(&f, &mut args)?
                    || prompt_option(&f, &mut args, &mut prompt)? => {}
            Arg::Flag(f) => match f.as_str() {
                "--chat" => chat = true,
                "--system" => system = Some(args.string(&f)?),
                "--no-think" => think = false,
                "--special" => special = true,
                "-n" | "--max-tokens" => max_tokens = args.parse(&f)?,
                "--probs" => probs = args.parse(&f)?,
                "-h" | "--help" => {
                    print!("{}", help());
                    return Ok(());
                }
                other => {
                    return Err(
                        format!("unknown option `{other}` (see `nibble run --help`)").into(),
                    );
                }
            },
            Arg::Positional(p) => {
                return Err(format!("unexpected argument `{}`", p.to_string_lossy()).into());
            }
        }
    }
    let prompt = prompt.ok_or("missing prompt: pass -p TEXT or -f FILE")?;

    let model = model_opts.load()?;
    let tokens = prompt_tokens(&model, &prompt, chat, system, think)?;
    let mut session = model.session(model_opts.session_options())?;
    crate::args::report_fallback(&session);
    let mut sampler = Sampler::new(sampling.config());
    let (k, v) = session.cache_types();
    eprintln!(
        "nibble: {} prompt tokens, context {} (KV cache K {k}, V {v}, {:.0} MiB), {}, seed {}",
        tokens.len(),
        session.n_ctx(),
        session.kv_cache_bytes() as f64 / (1024.0 * 1024.0),
        session.compute(),
        sampler.config().seed
    );

    let start = Instant::now();
    session.eval(&tokens)?;
    let prompt_secs = start.elapsed().as_secs_f64();

    let tok = model.tokenizer();
    let mut decoder = StreamDecoder::new();
    let mut stdout = io::stdout().lock();
    let mut generated = 0usize;
    let mut write_error = None;
    let start = Instant::now();
    let stop = session.generate(&[], &mut sampler, max_tokens, |token, logits| {
        if probs > 0 {
            eprintln!("{}", top_probs(tok, token, logits, probs));
        }
        generated += 1;
        let text = decoder.push(tok, token, special);
        if let Err(e) = stdout
            .write_all(text.as_bytes())
            .and_then(|()| stdout.flush())
        {
            write_error = Some(e);
            return false;
        }
        true
    })?;
    let gen_secs = start.elapsed().as_secs_f64();
    if let Some(e) = write_error {
        return Err(e.into());
    }
    stdout.write_all(decoder.finish().as_bytes())?;
    writeln!(stdout)?;
    drop(stdout);

    let stop = match stop {
        StopReason::EndOfGeneration(t) => {
            format!("end of generation ({})", tok.token_text(t).unwrap_or("?"))
        }
        StopReason::MaxTokens => "max tokens".to_owned(),
        StopReason::Stopped => "stopped".to_owned(),
        StopReason::ContextFull => "context full".to_owned(),
    };
    eprintln!(
        "nibble: prompt {} tokens in {:.2} s ({:.1} tok/s), generated {} tokens in {:.2} s ({:.1} tok/s), stop: {stop}",
        tokens.len(),
        prompt_secs,
        tokens.len() as f64 / prompt_secs,
        generated,
        gen_secs,
        generated as f64 / gen_secs.max(1e-9),
    );
    let d = session.draft_stats();
    if d.verifications > 0 {
        eprintln!(
            "nibble: drafts: {} of {} accepted ({:.0}%), {:.2} tokens per pass",
            d.accepted,
            d.drafted,
            100.0 * d.accepted as f64 / d.drafted.max(1) as f64,
            1.0 + d.accepted as f64 / d.verifications as f64,
        );
    }
    Ok(())
}

/// `chosen | candidate p (logit) ...` for the `n` most likely tokens.
fn top_probs(tok: &nibble::tokenizer::Tokenizer, chosen: u32, logits: &[f32], n: usize) -> String {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    let n = n.min(ids.len());
    ids.select_nth_unstable_by(n - 1, |&a, &b| {
        logits[b as usize].total_cmp(&logits[a as usize])
    });
    ids.truncate(n);
    ids.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    let piece = |id: u32| format!("{:?}", String::from_utf8_lossy(tok.token_bytes(id, true)));
    let mut line = format!("{:>12} |", piece(chosen));
    for id in ids {
        let p = f64::from(logits[id as usize] - max).exp() / sum;
        line.push_str(&format!(
            " {} {p:.4} ({:.3})",
            piece(id),
            logits[id as usize]
        ));
    }
    line
}

fn prompt_tokens(
    model: &Model,
    prompt: &str,
    chat: bool,
    system: Option<String>,
    think: bool,
) -> CliResult<Vec<u32>> {
    let tok = model.tokenizer();
    let text = if chat {
        let template = model
            .chat_template()
            .ok_or("this model's chat template is not supported yet")?;
        let mut messages = Vec::new();
        if let Some(system) = system {
            messages.push(Message::system(system));
        }
        messages.push(Message::user(prompt));
        template.render(
            &messages,
            ChatOptions {
                enable_thinking: think,
                ..ChatOptions::default()
            },
        )?
    } else {
        prompt.to_owned()
    };
    let mut tokens = Vec::new();
    if tok.add_bos()
        && let Some(bos) = tok.bos()
    {
        tokens.push(bos);
    }
    tokens.extend(tok.encode(&text, true));
    if tokens.is_empty() {
        return Err("the prompt is empty".into());
    }
    Ok(tokens)
}
