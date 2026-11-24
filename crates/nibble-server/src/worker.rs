//! The generation worker: one thread owns the session and serves jobs in
//! arrival order, reusing the cached prefix between requests.

use std::sync::mpsc::{Receiver, Sender};

use nibble::sampler::{Sampler, SamplerConfig};
use nibble::tokenizer::StreamDecoder;
use nibble::{Model, Session, SessionOptions, StopReason};

use crate::text::StopMatcher;

/// One generation request.
pub(crate) struct Job {
    pub(crate) prompt: Vec<u32>,
    pub(crate) sampler: SamplerConfig,
    pub(crate) max_tokens: usize,
    pub(crate) stop: Vec<String>,
    pub(crate) events: Sender<Event>,
}

/// Why a generation finished, in OpenAI's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Finish {
    /// End-of-generation token or a stop sequence.
    Stop,
    /// `max_tokens` or the context size was reached.
    Length,
}

impl Finish {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

/// What the worker reports for a job, in order: text pieces, then `Done` or `Error`.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Event {
    Text(String),
    Done {
        finish: Finish,
        prompt_tokens: usize,
        completion_tokens: usize,
    },
    Error(String),
}

/// Serves jobs until every sender is dropped.
pub(crate) fn run(model: &Model, options: SessionOptions, jobs: &Receiver<Job>) {
    let mut session = model.session(options);
    for job in jobs {
        serve(&mut session, job);
    }
}

fn serve(session: &mut Session<'_>, job: Job) {
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
    session.truncate(keep);

    let tokenizer = session.model().tokenizer();
    let mut sampler = Sampler::new(job.sampler);
    let mut decoder = StreamDecoder::new();
    let mut stops = StopMatcher::new(job.stop);
    let mut completion_tokens = 0;
    let mut hit_stop = false;
    let mut client_gone = false;
    let result = session.generate(
        &job.prompt[keep..],
        &mut sampler,
        job.max_tokens,
        |token, _| {
            completion_tokens += 1;
            let (text, stopped) = stops.push(&decoder.push(tokenizer, token, false));
            if !text.is_empty() && job.events.send(Event::Text(text)).is_err() {
                client_gone = true;
                return false;
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
        let tail = tail + &stops.finish();
        if !tail.is_empty() {
            let _ = job.events.send(Event::Text(tail));
        }
    }
    let _ = job.events.send(Event::Done {
        finish,
        prompt_tokens: job.prompt.len(),
        completion_tokens,
    });
}
