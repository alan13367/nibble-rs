//! `nibble bench`: prompt-processing and decode speed, llama-bench style.
//!
//! Follows `llama-bench`'s methodology so the numbers compare directly on
//! the same machine and thread count:
//!
//! - `pp N`: each repetition evaluates `N` random tokens into an empty
//!   session, timed as a whole. nibble has no batched matrix products yet
//!   (Phase 7), so this is token-by-token evaluation and does not compare
//!   with llama-bench's batched `pp` until then.
//! - `tg N`: each repetition decodes `N` random tokens one at a time into an
//!   empty session, computing the full logits each time (llama-bench's
//!   `test_gen`).
//! - Tokens come from a fixed-seed generator, as llama-bench's unseeded
//!   `rand()`, so nothing is sampled or printed while timing.
//! - One untimed warm-up precedes the repetitions, the session is cleared
//!   (untimed) before each one, and the table shows the mean tokens/s over
//!   the repetitions with its sample standard deviation.

use std::time::Instant;

use nibble::Session;

use crate::args::{Arg, Args, CliResult, MODEL_HELP, ModelOptions};

pub(crate) fn help() -> String {
    format!(
        "Usage: nibble bench -m <model.gguf> [options]\n\n\
         Tests (as llama-bench):\n  \
         -p, --prompt N           Prompt tokens per repetition, `ppN` (default 512, 0 = skip)\n  \
         -n, --gen N              Tokens to decode per repetition, `tgN` (default 128, 0 = skip)\n  \
         -r, --reps N             Repetitions per test (default 5)\n      \
         --seed N             Seed for the random tokens (default 1)\n\n\
         Model:\n{MODEL_HELP}"
    )
}

pub(crate) fn run(mut args: Args) -> CliResult<()> {
    let mut model_opts = ModelOptions::new();
    let (mut pp, mut tg, mut reps, mut seed) = (512usize, 128usize, 5usize, 1u64);
    while let Some(arg) = args.next()? {
        match arg {
            Arg::Flag(f) if model_opts.parse(&f, &mut args)? => {}
            Arg::Flag(f) => match f.as_str() {
                "-p" | "--prompt" => pp = args.parse(&f)?,
                "-n" | "--gen" => tg = args.parse(&f)?,
                "-r" | "--reps" => reps = args.parse(&f)?,
                "--seed" => seed = args.parse(&f)?,
                "-h" | "--help" => {
                    print!("{}", help());
                    return Ok(());
                }
                other => {
                    return Err(
                        format!("unknown option `{other}` (see `nibble bench --help`)").into(),
                    );
                }
            },
            Arg::Positional(p) => {
                return Err(format!("unexpected argument `{}`", p.to_string_lossy()).into());
            }
        }
    }
    if pp == 0 && tg == 0 {
        return Err("nothing to do: pass -p N and/or -n N".into());
    }
    if reps == 0 {
        return Err("--reps must be at least 1".into());
    }

    let model = model_opts.load()?;
    // Each repetition fills the session from empty: raise a smaller `-c` to
    // fit the longer test (0 already means the model's maximum).
    let needed = pp.max(tg);
    let mut options = model_opts.session_options();
    if options.n_ctx != 0 {
        options.n_ctx = options.n_ctx.max(needed);
    }
    let mut session = model.session(options);
    if session.n_ctx() < needed {
        return Err(format!(
            "a test of {needed} tokens does not fit the model's context of {}",
            session.n_ctx()
        )
        .into());
    }

    let c = model.config();
    eprintln!(
        "nibble bench: {} ({}, {}), {} threads, {}, {reps} repetitions, seed {seed}",
        c.name.as_deref().unwrap_or("model"),
        c.architecture,
        c.weight_type,
        session.threads(),
        session.kernel(),
    );
    if pp > 0 {
        eprintln!(
            "nibble: pp evaluates tokens one at a time until batched prompt processing exists"
        );
    }

    let mut tokens = RandomTokens::new(seed, c.n_vocab);
    println!("| test | t/s | stdev | ms/token |");
    println!("|------|----:|------:|---------:|");
    if pp > 0 {
        let prompt: Vec<u32> = (0..pp).map(|_| tokens.next()).collect();
        let samples = repeat(&mut session, reps, |s| {
            s.eval(&prompt)?;
            Ok(())
        })?;
        print_row(&format!("pp{pp}"), &samples, pp);
    }
    if tg > 0 {
        let samples = repeat(&mut session, reps, |s| {
            for _ in 0..tg {
                s.eval(&[tokens.next()])?;
            }
            Ok(())
        })?;
        print_row(&format!("tg{tg}"), &samples, tg);
    }
    Ok(())
}

/// Seconds taken by each of `reps` runs of `test` on a cleared session,
/// after one untimed warm-up run.
fn repeat(
    session: &mut Session<'_>,
    reps: usize,
    mut test: impl FnMut(&mut Session<'_>) -> CliResult<()>,
) -> CliResult<Vec<f64>> {
    session.clear();
    test(session)?;
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        session.clear();
        let start = Instant::now();
        test(session)?;
        samples.push(start.elapsed().as_secs_f64());
    }
    Ok(samples)
}

/// Token ids from a fixed-seed xorshift generator.
struct RandomTokens {
    state: u64,
    n_vocab: u64,
}

impl RandomTokens {
    fn new(seed: u64, n_vocab: usize) -> Self {
        Self {
            // xorshift has no non-zero state that maps to zero.
            state: seed.max(1),
            n_vocab: n_vocab as u64,
        }
    }

    fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x % self.n_vocab) as u32
    }
}

/// llama-bench's statistics: the mean of the per-repetition rates and their
/// sample standard deviation.
fn print_row(name: &str, samples: &[f64], tokens: usize) {
    let rates: Vec<f64> = samples.iter().map(|&s| tokens as f64 / s).collect();
    let n = rates.len() as f64;
    let mean = rates.iter().sum::<f64>() / n;
    let stdev = if rates.len() > 1 {
        (rates.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
    } else {
        0.0
    };
    let ms_per_token = 1e3 * samples.iter().sum::<f64>() / n / tokens as f64;
    println!("| {name} | {mean:.2} | {stdev:.2} | {ms_per_token:.2} |");
}
