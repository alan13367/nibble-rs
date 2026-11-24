//! Token sampling: penalties, top-k, top-p, min-p, temperature.
//!
//! The order matches llama.cpp's default chain: penalties → top-k → top-p →
//! min-p → temperature → draw.

/// Sampling parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerConfig {
    /// Softmax temperature. `0` (or less) means greedy: always the most likely token.
    pub temperature: f32,
    /// Keep the `top_k` most likely tokens; `0` keeps all.
    pub top_k: usize,
    /// Keep the smallest set of tokens whose probabilities sum to `top_p`; `1` disables.
    pub top_p: f32,
    /// Drop tokens less likely than `min_p` times the most likely one; `0` disables.
    pub min_p: f32,
    /// Divide (or multiply, if negative) the logits of recently seen tokens; `1` disables.
    pub repeat_penalty: f32,
    /// Subtract from the logit of every token seen recently; `0` disables.
    pub presence_penalty: f32,
    /// Subtract this times the count of each recently seen token; `0` disables.
    pub frequency_penalty: f32,
    /// How many recent tokens the penalties look at.
    pub penalty_last_n: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for SamplerConfig {
    /// llama.cpp's defaults.
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            penalty_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    /// Greedy decoding with no penalties.
    #[must_use]
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            ..Self::default()
        }
    }
}

/// Picks the next token from logits.
#[derive(Clone, Debug)]
pub struct Sampler {
    config: SamplerConfig,
    rng: Pcg32,
    candidates: Vec<(u32, f32)>,
    adjusted: Vec<f32>,
}

impl Sampler {
    /// A sampler with the given configuration.
    #[must_use]
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            config,
            rng: Pcg32::new(config.seed),
            candidates: Vec::new(),
            adjusted: Vec::new(),
        }
    }

    /// The configuration.
    #[must_use]
    pub const fn config(&self) -> &SamplerConfig {
        &self.config
    }

    /// Samples a token from `logits`; `history` holds the tokens so far, for penalties.
    ///
    /// # Panics
    ///
    /// Panics if `logits` is empty.
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        assert!(!logits.is_empty(), "no logits to sample from");
        let c = self.config;
        let logits: &[f32] = if apply_penalties(&c, logits, history, &mut self.adjusted) {
            &self.adjusted
        } else {
            logits
        };

        if c.temperature <= 0.0 {
            return argmax(logits);
        }

        let cand = &mut self.candidates;
        cand.clear();
        cand.extend(logits.iter().enumerate().map(|(i, &l)| (i as u32, l)));
        let by_logit_desc =
            |a: &(u32, f32), b: &(u32, f32)| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0));

        if c.top_k > 0 && c.top_k < cand.len() {
            cand.select_nth_unstable_by(c.top_k - 1, by_logit_desc);
            cand.truncate(c.top_k);
        }
        cand.sort_unstable_by(by_logit_desc);

        // top-p and min-p work on the probabilities before temperature.
        if c.top_p < 1.0 || c.min_p > 0.0 {
            let probs = softmax_of(cand);
            let mut keep = cand.len();
            if c.top_p < 1.0 {
                let mut cumulative = 0.0;
                for (i, p) in probs.iter().enumerate() {
                    cumulative += p;
                    if cumulative >= c.top_p {
                        keep = keep.min(i + 1);
                        break;
                    }
                }
            }
            if c.min_p > 0.0 {
                let threshold = probs[0] * c.min_p;
                keep = keep.min(probs.iter().take_while(|&&p| p >= threshold).count().max(1));
            }
            cand.truncate(keep);
        }

        for (_, l) in cand.iter_mut() {
            *l /= c.temperature;
        }
        let probs = softmax_of(cand);
        let r = self.rng.next_f32();
        let mut cumulative = 0.0;
        for (&(id, _), p) in cand.iter().zip(&probs) {
            cumulative += p;
            if r < cumulative {
                return id;
            }
        }
        cand.last().map_or(0, |&(id, _)| id)
    }
}

/// Writes penalized logits into `adjusted` and returns true, or returns false
/// when no penalty applies (so the original logits can be used as they are).
fn apply_penalties(
    c: &SamplerConfig,
    logits: &[f32],
    history: &[u32],
    adjusted: &mut Vec<f32>,
) -> bool {
    let active = c.repeat_penalty != 1.0 || c.presence_penalty != 0.0 || c.frequency_penalty != 0.0;
    if !active || c.penalty_last_n == 0 || history.is_empty() {
        return false;
    }
    adjusted.clear();
    adjusted.extend_from_slice(logits);
    let recent = &history[history.len().saturating_sub(c.penalty_last_n)..];
    let mut counts: Vec<(u32, u32)> = Vec::new();
    for &t in recent {
        match counts.iter_mut().find(|(id, _)| *id == t) {
            Some((_, n)) => *n += 1,
            None => counts.push((t, 1)),
        }
    }
    for (id, n) in counts {
        let Some(l) = adjusted.get_mut(id as usize) else {
            continue;
        };
        if *l > 0.0 {
            *l /= c.repeat_penalty;
        } else {
            *l *= c.repeat_penalty;
        }
        *l -= n as f32 * c.frequency_penalty + c.presence_penalty;
    }
    true
}

/// Index of the largest logit (lowest index on ties).
#[must_use]
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

/// Log-probabilities of the model's distribution over the next token, as
/// [`logprobs`] computes them.
#[derive(Clone, Debug, PartialEq)]
pub struct Logprobs {
    /// Log-probability of the chosen token.
    pub token: f32,
    /// The most likely tokens and their log-probabilities, most likely
    /// first (lowest id first on ties).
    pub top: Vec<(u32, f32)>,
}

/// The log-probability of `token` and of the `n` most likely tokens under
/// the softmax of `logits` (the raw model distribution, before any sampling
/// setting, as llama.cpp's server reports it).
///
/// # Panics
///
/// If `token` is not an index of `logits`.
#[must_use]
pub fn logprobs(logits: &[f32], token: u32, n: usize) -> Logprobs {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f64 = logits.iter().map(|&l| f64::from(l - max).exp()).sum();
    let log_sum = f64::from(max) + sum.ln();
    let logprob = |l: f32| (f64::from(l) - log_sum) as f32;
    // One pass keeping the best `n` in order; `n` is small.
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(n + 1);
    if n > 0 {
        for (i, &l) in logits.iter().enumerate() {
            if top.len() == n && l <= top[n - 1].1 {
                continue;
            }
            let at = top.partition_point(|&(_, t)| t >= l);
            top.insert(at, (i as u32, l));
            top.truncate(n);
        }
    }
    Logprobs {
        token: logprob(logits[token as usize]),
        top: top.into_iter().map(|(i, l)| (i, logprob(l))).collect(),
    }
}

fn softmax_of(cand: &[(u32, f32)]) -> Vec<f32> {
    let max = cand.first().map_or(0.0, |c| c.1);
    let mut probs: Vec<f32> = cand.iter().map(|&(_, l)| (l - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }
    probs
}

/// PCG32 (XSH-RR), a small, good-quality generator.
#[derive(Clone, Debug)]
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (0xDA3E_39CB_94B9_5BDB << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        xorshifted.rotate_right((old >> 59) as u32)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprobs_are_the_log_softmax() {
        let logits = [1.0f32, 3.0, 2.0, 3.0, -1.0];
        let total: f64 = logits.iter().map(|&l| f64::from(l).exp()).sum();
        let want = |i: usize| (f64::from(logits[i]).exp() / total).ln() as f32;
        let lp = logprobs(&logits, 2, 3);
        assert!((lp.token - want(2)).abs() < 1e-6);
        // Ties keep the lower id first.
        let ids: Vec<u32> = lp.top.iter().map(|&(i, _)| i).collect();
        assert_eq!(ids, [1, 3, 2]);
        for &(i, l) in &lp.top {
            assert!((l - want(i as usize)).abs() < 1e-6);
        }
        assert!(logprobs(&logits, 0, 0).top.is_empty());
        assert_eq!(logprobs(&logits, 0, 10).top.len(), logits.len());
    }

    #[test]
    fn greedy_picks_the_maximum() {
        let mut s = Sampler::new(SamplerConfig::greedy());
        assert_eq!(s.sample(&[0.1, 3.0, -1.0, 2.9], &[]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let config = SamplerConfig {
            top_k: 1,
            ..SamplerConfig::default()
        };
        let mut s = Sampler::new(config);
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.5, 0.4, 2.0, 1.9], &[]), 2);
        }
    }

    #[test]
    fn seeded_sampling_is_deterministic_and_follows_probabilities() {
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let config = SamplerConfig {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            temperature: 1.0,
            seed: 42,
            ..SamplerConfig::default()
        };
        let draw = |n| {
            let mut s = Sampler::new(config);
            (0..n).map(|_| s.sample(&logits, &[])).collect::<Vec<_>>()
        };
        assert_eq!(draw(50), draw(50));
        let counts = draw(4000).iter().fold([0usize; 4], |mut c, &t| {
            c[t as usize] += 1;
            c
        });
        // Softmax of [0,1,2,3] is about [0.032, 0.087, 0.237, 0.644].
        assert!(counts[3] > 2400 && counts[3] < 2750, "{counts:?}");
        assert!(counts[0] > 60 && counts[0] < 200, "{counts:?}");
    }

    #[test]
    fn repeat_penalty_discourages_seen_tokens() {
        let config = SamplerConfig {
            repeat_penalty: 2.0,
            ..SamplerConfig::greedy()
        };
        let mut s = Sampler::new(config);
        assert_eq!(s.sample(&[1.0, 1.5], &[1]), 0);
    }

    #[test]
    fn min_p_keeps_at_least_one_token() {
        let config = SamplerConfig {
            min_p: 1.0,
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            ..SamplerConfig::default()
        };
        let mut s = Sampler::new(config);
        assert_eq!(s.sample(&[0.0, 5.0, 1.0], &[]), 1);
    }
}
