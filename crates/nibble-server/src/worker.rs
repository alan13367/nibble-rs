//! The generation worker: one thread owns the session and serves jobs in
//! arrival order, reusing the cached prefix between requests.

use std::sync::mpsc::{Receiver, Sender};

use nibble::sampler::{self, Sampler, SamplerConfig};
use nibble::tokenizer::{StreamDecoder, Tokenizer};
use nibble::{Model, Session, SessionOptions, StopReason};

use crate::text::StopMatcher;

/// One generation request.
pub(crate) struct Job {
    pub(crate) prompt: Vec<u32>,
    pub(crate) sampler: SamplerConfig,
    pub(crate) max_tokens: usize,
    pub(crate) stop: Vec<String>,
    /// Report each token's log-probability, with this many alternatives.
    pub(crate) logprobs: Option<usize>,
    pub(crate) events: Sender<Event>,
}

/// Why a generation finished, in OpenAI's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Finish {
    /// End-of-generation token or a stop sequence.
    Stop,
    /// `max_tokens` or the context size was reached.
    Length,
    /// The reply ended with tool calls (decided by the chat endpoint).
    ToolCalls,
}

impl Finish {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ToolCalls => "tool_calls",
        }
    }
}

/// A generated token's log-probability and the most likely alternatives,
/// under the model's raw distribution.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TokenLogprob {
    /// The token's bytes (special tokens as their text).
    pub(crate) bytes: Vec<u8>,
    pub(crate) logprob: f32,
    pub(crate) top: Vec<(Vec<u8>, f32)>,
}

impl TokenLogprob {
    fn new(tokenizer: &Tokenizer, token: u32, logits: &[f32], n_top: usize) -> Self {
        let lp = sampler::logprobs(logits, token, n_top);
        let bytes = |id: u32| tokenizer.token_bytes(id, true).to_vec();
        Self {
            bytes: bytes(token),
            logprob: lp.token,
            top: lp.top.into_iter().map(|(id, l)| (bytes(id), l)).collect(),
        }
    }
}

/// What the worker reports for a job, in order: text pieces, then `Done` or `Error`.
#[derive(Debug, PartialEq)]
pub(crate) enum Event {
    /// Generated text, with the log-probabilities of the tokens it completes
    /// (when requested). Tokens of a stop sequence are not reported.
    Text {
        text: String,
        logprobs: Vec<TokenLogprob>,
    },
    Done {
        finish: Finish,
        prompt_tokens: usize,
        /// Prompt tokens reused from the last request instead of evaluated.
        cached_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

/// Serves jobs until every sender is dropped.
pub(crate) fn run(model: &Model, options: SessionOptions, log: bool, jobs: &Receiver<Job>) {
    match model.session(options) {
        Ok(mut session) => {
            for job in jobs {
                serve(&mut session, job, log);
            }
        }
        Err(e) => {
            let message = format!("cannot start a session: {e}");
            for job in jobs {
                let _ = job.events.send(Event::Error(message.clone()));
            }
        }
    }
}

fn serve(session: &mut Session<'_>, job: Job, log: bool) {
    let started = std::time::Instant::now();
    let drafts_before = session.draft_stats();
    let n_ctx = session.n_ctx();
    if job.prompt.is_empty() {
        let _ = job.events.send(Event::Error("the prompt is empty".into()));
        return;
    }
    if job.prompt.len() >= n_ctx {
        let _ = job.events.send(Event::Error(format!(
            "the prompt has {} tokens, but the context holds {n_ctx}",
            job.prompt.len()
        )));
        return;
    }

    // Keep the cached prefix this prompt shares with the last request; at
    // least one token is evaluated again so there are fresh logits.
    let common = session
        .tokens()
        .iter()
        .zip(&job.prompt)
        .take_while(|(a, b)| a == b)
        .count();
    let keep = common.min(job.prompt.len() - 1);
    if let Err(e) = session.truncate(keep) {
        let _ = job.events.send(Event::Error(e.to_string()));
        return;
    }

    let tokenizer = session.model().tokenizer();
    let mut sampler = Sampler::new(job.sampler);
    let mut first_token = None;
    let mut decoder = StreamDecoder::new();
    let mut stops = StopMatcher::new(job.stop);
    // Log-probabilities of tokens whose text is not released yet.
    let mut pending = Vec::new();
    let mut completion_tokens = 0;
    let mut hit_stop = false;
    let mut client_gone = false;
    let result = session.generate(
        &job.prompt[keep..],
        &mut sampler,
        job.max_tokens,
        |token, logits| {
            first_token.get_or_insert_with(std::time::Instant::now);
            completion_tokens += 1;
            if let Some(n_top) = job.logprobs {
                pending.push(TokenLogprob::new(tokenizer, token, logits, n_top));
            }
            let (text, stopped) = stops.push(&decoder.push(tokenizer, token, false));
            if !text.is_empty() {
                let logprobs = std::mem::take(&mut pending);
                if job.events.send(Event::Text { text, logprobs }).is_err() {
                    client_gone = true;
                    return false;
                }
            }
            hit_stop = stopped;
            !stopped
        },
    );
    if client_gone {
        return;
    }
    let finish = match result {
        Ok(StopReason::EndOfGeneration(_)) => Finish::Stop,
        Ok(StopReason::Stopped) if hit_stop => Finish::Stop,
        Ok(_) => Finish::Length,
        Err(e) => {
            let _ = job.events.send(Event::Error(e.to_string()));
            return;
        }
    };
    if !hit_stop {
        let (tail, _) = stops.push(&decoder.finish());
        let text = tail + &stops.finish();
        if !text.is_empty() || !pending.is_empty() {
            let _ = job.events.send(Event::Text {
                text,
                logprobs: pending,
            });
        }
    }
    let _ = job.events.send(Event::Done {
        finish,
        prompt_tokens: job.prompt.len(),
        cached_tokens: keep,
        completion_tokens,
    });
    if log {
        let prompt_end = first_token.unwrap_or_else(std::time::Instant::now);
        let prompt_secs = (prompt_end - started).as_secs_f64();
        let gen_secs = prompt_end.elapsed().as_secs_f64();
        let new = job.prompt.len() - keep;
        let d = session.draft_stats();
        let drafts = if d.drafted > drafts_before.drafted {
            let (drafted, accepted) = (
                d.drafted - drafts_before.drafted,
                d.accepted - drafts_before.accepted,
            );
            format!(", drafts {accepted}/{drafted} accepted")
        } else {
            String::new()
        };
        eprintln!(
            "nibble: prompt {} tokens ({keep} cached, {new} in {prompt_secs:.2} s, {:.0} tok/s), generated {completion_tokens} in {gen_secs:.2} s ({:.1} tok/s), {}{drafts}",
            job.prompt.len(),
            new as f64 / prompt_secs.max(1e-9),
            completion_tokens.saturating_sub(1) as f64 / gen_secs.max(1e-9),
            finish.as_str(),
        );
    }
}
