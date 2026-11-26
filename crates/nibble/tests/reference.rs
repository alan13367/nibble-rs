//! Checks against llama.cpp on real model files.
//!
//! Ignored by default. Put the models in `models/` at the repository root, then:
//!
//! ```sh
//! cargo test --release -p nibble --test reference -- --ignored
//! ```
//!
//! Expected values were produced by llama.cpp's b12359 release builds
//! (`tools/llama.cpp/b12359`): `llama-tokenize --ids`, and
//!
//! ```sh
//! llama-completion -m MODEL -f PROMPT -n 64 --temp 0 -no-cnv --no-display-prompt \
//!     -ngl 0 --device none -c 512 -fa off --no-repack -b 512 -ub 1
//! ```
//!
//! on an M4 Max (the macOS arm64 build). `--no-repack` keeps the kernels
//! nibble transcribes (repacked weights use others), and `-ub 1` evaluates
//! the prompt one token at a time, as nibble does. The macOS x64 build (AVX2)
//! gives the same text for the BF16 and Q8_0 cases. The Qwen3.5 BF16 case is
//! from b12675 on x86 and was not rerun here.
//!
//! Every generation case also agrees with llama.cpp with flash attention on
//! and with weight repacking on, so none is a near-tie.

use std::path::PathBuf;

use nibble::gguf::{Gguf, Storage};
use nibble::sampler::{Sampler, SamplerConfig};
use nibble::tokenizer::{StreamDecoder, Tokenizer};
use nibble::{Model, SessionOptions, StopReason};

fn model_path(file: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models")
        .join(file);
    assert!(
        path.exists(),
        "missing {}: download it into models/ to run this test",
        path.display()
    );
    path
}

fn load_model(file: &str) -> Model {
    // SAFETY: test model files are not modified while the tests run.
    unsafe { Model::load_mmap(model_path(file)) }.unwrap()
}

fn load_tokenizer(file: &str) -> Tokenizer {
    // SAFETY: as above.
    let storage = unsafe { Storage::open(model_path(file)) }.unwrap();
    Tokenizer::from_gguf(&Gguf::parse(&storage).unwrap()).unwrap()
}

/// Greedy completion of `prompt` (special tokens parsed), as text, on the CPU.
fn greedy(model: &Model, prompt: &str, max_tokens: usize) -> (String, StopReason) {
    let (text, stop, _) = greedy_on(model, prompt, max_tokens, nibble::BackendKind::Cpu, 0);
    (text, stop)
}

/// [`greedy`] on `backend`, drafting up to `draft_max` tokens with the MTP
/// head, and how the drafts fared.
fn greedy_on(
    model: &Model,
    prompt: &str,
    max_tokens: usize,
    backend: nibble::BackendKind,
    draft_max: usize,
) -> (String, StopReason, nibble::DraftStats) {
    greedy_with(
        model,
        prompt,
        max_tokens,
        SessionOptions {
            n_ctx: 512,
            threads: 0,
            backend,
            n_batch: 0,
            draft_max,
            ..Default::default()
        },
    )
}

/// [`greedy`] in a session with `options`.
fn greedy_with(
    model: &Model,
    prompt: &str,
    max_tokens: usize,
    options: SessionOptions,
) -> (String, StopReason, nibble::DraftStats) {
    let tok = model.tokenizer();
    let tokens = tok.encode(prompt, true);
    let mut session = model.session(options).unwrap();
    let mut sampler = Sampler::new(SamplerConfig::greedy());
    let mut decoder = StreamDecoder::new();
    let mut text = String::new();
    let stop = session
        .generate(&tokens, &mut sampler, max_tokens, |t, _| {
            text.push_str(&decoder.push(tok, t, false));
            true
        })
        .unwrap();
    text.push_str(&decoder.finish());
    (text, stop, session.draft_stats())
}

const TOKENIZER_CASES: [&str; 7] = [
    "Hello world! I'm testing   spaces\n\nand 12345 numbers.",
    "Ça va? Привет мир! 日本語のテキスト。 مرحبا 😀👍🏽 naïve café",
    "They're here, we'll see. I'VE got it, don't you'd? rock'n'roll",
    "  leading\ttabs\t\tand trailing   \n\n\n  \r\n end  ",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "नमस्ते दुनिया สวัสดีครับ 안녕하세요 e\u{301}e\u{301}",
    "👨‍👩‍👧‍👦 family   ",
];

#[test]
#[ignore = "needs models/Qwen3-0.6B-Q8_0.gguf"]
fn qwen3_tokenizer_matches_llama_cpp() {
    let tok = load_tokenizer("Qwen3-0.6B-Q8_0.gguf");
    let expected: [&[u32]; 7] = [
        &[
            9707, 1879, 0, 358, 2776, 7497, 256, 12621, 271, 437, 220, 16, 17, 18, 19, 20, 5109, 13,
        ],
        &[
            45913, 64, 11164, 30, 79484, 26991, 8178, 137144, 0, 75402, 21894, 102819, 15767,
            56833, 61803, 70534, 1773, 23364, 126860, 124671, 90316, 144349, 145375, 94880, 586,
            51950,
        ],
        &[
            6865, 2299, 1588, 11, 582, 3278, 1490, 13, 358, 6, 4491, 2684, 432, 11, 1513, 944, 498,
            4172, 30, 6946, 43786, 6, 1100,
        ],
        &[
            220, 6388, 3244, 3435, 197, 52477, 27748, 262, 1406, 10419, 835, 256,
        ],
        &[
            151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
        ],
        &[
            60096, 87244, 78368, 30484, 97, 34370, 14925, 99, 72653, 60096, 42311, 107, 23868,
            129328, 37213, 23271, 125136, 28319, 123886, 83546, 95170, 144370, 91145, 384, 53839,
            68, 53839,
        ],
        &[
            145367, 378, 235, 145233, 378, 235, 145665, 378, 235, 145988, 2997, 262,
        ],
    ];
    for (text, ids) in TOKENIZER_CASES.iter().zip(expected) {
        assert_eq!(tok.encode(text, true), ids, "{text:?}");
        assert_eq!(tok.decode(ids, true), *text);
    }
}

#[test]
#[ignore = "needs models/Qwen3.5-0.8B-Q8_0.gguf"]
fn qwen35_tokenizer_matches_llama_cpp() {
    let tok = load_tokenizer("Qwen3.5-0.8B-Q8_0.gguf");
    let expected: [&[u32]; 7] = [
        &[
            9419, 1814, 0, 353, 2688, 7262, 256, 12258, 271, 429, 220, 16, 17, 18, 19, 20, 4947, 13,
        ],
        &[
            44382, 64, 10840, 30, 239029, 160202, 0, 220, 247359, 15303, 210342, 1710, 164422,
            150027, 87209, 9008, 239, 235, 9008, 237, 121, 91603, 571, 50203,
        ],
        &[
            6651, 2224, 1532, 11, 567, 3172, 1436, 13, 353, 6, 4343, 2597, 424, 11, 1459, 914, 488,
            4035, 30, 6730, 42336, 6, 1065,
        ],
        &[
            220, 6187, 3139, 3325, 197, 50711, 26849, 262, 1358, 10109, 809, 256,
        ],
        &[
            248045, 846, 198, 12675, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        ],
        &[
            58069, 84237, 150104, 153348, 184642, 235886, 245990, 153295, 191972, 378, 52033, 68,
            52033,
        ],
        &[
            9008, 239, 101, 373, 235, 9008, 239, 102, 373, 235, 9008, 239, 100, 373, 235, 9008,
            239, 99, 2902, 262,
        ],
    ];
    for (text, ids) in TOKENIZER_CASES.iter().zip(expected) {
        assert_eq!(tok.encode(text, true), ids, "{text:?}");
        assert_eq!(tok.decode(ids, true), *text);
    }
}

#[test]
#[ignore = "needs models/Qwen3-0.6B-BF16.gguf"]
fn qwen3_bf16_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3-0.6B-BF16.gguf");

    let chat = "<|im_start|>user\nWrite a haiku about the Rust programming language.<|im_end|>\n\
                <|im_start|>assistant\n<think>\n\n</think>\n";
    let (text, stop) = greedy(&model, chat, 64);
    assert_eq!(text, "A rust program  \nA rust program  \nA rust program.");
    assert!(matches!(stop, StopReason::EndOfGeneration(_)));

    let (text, stop) = greedy(&model, "The capital of France is", 64);
    assert_eq!(
        text,
        " Paris. The capital of Italy is Rome. The capital of Spain is Madrid. The capital of China is Beijing. \
         The capital of Japan is Tokyo. The capital of India is New Delhi. The capital of Brazil is Brasilia. \
         The capital of Egypt is Cairo. The capital of South Africa is Cape Town. The capital"
    );
    assert_eq!(stop, StopReason::MaxTokens);
}

#[test]
#[ignore = "needs models/Qwen3-0.6B-Q8_0.gguf"]
fn qwen3_q8_0_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3-0.6B-Q8_0.gguf");

    let (text, _) = greedy(&model, "List of prime numbers: 2, 3, 5, 7,", 64);
    assert_eq!(
        text,
        " 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,"
    );
}

#[test]
#[ignore = "needs models/Qwen3.5-0.8B-BF16.gguf"]
fn qwen35_bf16_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.5-0.8B-BF16.gguf");
    let (text, stop) = greedy(&model, "The capital of France is", 64);
    assert_eq!(
        text,
        format!(
            " Paris.{}\nThe capital of France is",
            "\nThe capital of France is Paris.".repeat(7)
        )
    );
    assert_eq!(stop, StopReason::MaxTokens);
}

#[test]
#[ignore = "needs models/Qwen3.5-0.8B-Q8_0.gguf"]
fn qwen35_q8_0_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.5-0.8B-Q8_0.gguf");

    let (text, _) = greedy(&model, "List of prime numbers: 2, 3, 5, 7,", 64);
    assert_eq!(
        text,
        " 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,"
    );

    let chat = "<|im_start|>user\nWrite a haiku about the Rust programming language.<|im_end|>\n\
                <|im_start|>assistant\n<think>\n\n</think>\n";
    let (text, stop) = greedy(&model, chat, 64);
    assert_eq!(
        text,
        "Red code, green eyes,\nSwift code, blue eyes,\nRust code, green eyes."
    );
    assert!(matches!(stop, StopReason::EndOfGeneration(_)));

    let (text, stop) = greedy(&model, "The capital of France is", 64);
    assert_eq!(
        text,
        format!(
            " Paris.{}\nThe capital of France is",
            "\nThe capital of France is Paris.".repeat(7)
        )
    );
    assert_eq!(stop, StopReason::MaxTokens);
}

#[test]
#[ignore = "needs models/Qwen3.5-0.8B-Q4_K_M.gguf"]
fn qwen35_q4_k_m_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.5-0.8B-Q4_K_M.gguf");

    let (text, stop) = greedy(&model, "The capital of France is", 64);
    assert_eq!(
        text,
        " the capital of the country.\n\
         The capital of France is the capital of the country.\n\
         The capital of France is the capital of the country.\n\
         The capital of France is the capital of the country.\n\
         The capital of France is the capital of the country.\n\
         The capital of France is the capital of the"
    );
    assert_eq!(stop, StopReason::MaxTokens);

    let (text, _) = greedy(&model, "List of prime numbers: 2, 3, 5, 7,", 64);
    assert_eq!(
        text,
        " 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,"
    );
}

/// A larger `qwen35` (32 value heads sharing 16 key heads) with `IQ4_XS`
/// weights among its K-quants (Unsloth's "UD" mix).
#[test]
#[ignore = "needs models/Qwen3.5-9B-UD-Q4_K_XL.gguf"]
fn qwen35_9b_ud_q4_k_xl_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.5-9B-UD-Q4_K_XL.gguf");

    let (text, stop) = greedy(&model, "The capital of France is", 64);
    assert_eq!(text, " Paris.\nThe capital of France is".repeat(8));
    assert_eq!(stop, StopReason::MaxTokens);

    let prompt = "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n";
    let (text, _) = greedy(&model, prompt, 64);
    assert_eq!(
        text,
        "    if n == 0:\n        return 0\n    elif n == 1:\n        return 1\n    \
         else:\n        return fibonacci(n - 1) + fibonacci(n - 2)\n\n# Example usage:\n\
         print(fibonacci(10))  # Output: "
    );
}

#[test]
#[ignore = "needs models/Qwen3.5-0.8B-Q3_K_M.gguf"]
fn qwen35_q3_k_m_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.5-0.8B-Q3_K_M.gguf");

    // The other comparison prompts are near-ties on this file: llama.cpp
    // itself changes its answer with flash attention or weight repacking.
    let (text, _) = greedy(&model, "List of prime numbers: 2, 3, 5, 7,", 64);
    assert_eq!(
        text,
        " 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71,"
    );
}

/// The 4B file carries an MTP head (`blk.32`): drafting with it gives the
/// tokens of plain decoding, which match llama.cpp's on the CPU, and the
/// drafts are mostly accepted (llama.cpp b12722's own MTP drafting accepts
/// 81% on Metal and 88% on the CPU for the chat prompt, 3 drafts per step).
#[test]
#[ignore = "needs models/Qwen3.5-4B-Q8_0.gguf"]
fn qwen35_4b_drafting_gives_the_tokens_of_plain_decoding() {
    let model = load_model("Qwen3.5-4B-Q8_0.gguf");
    assert!(model.has_mtp());
    let chat = "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n\
                <|im_start|>assistant\n";
    let (text, stop) = greedy(&model, "1, 1, 2, 3, 5, 8, 13,", 64);
    assert_eq!(
        text,
        " 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765"
    );
    // Digits are single tokens: these fill the 64.
    assert_eq!(stop, StopReason::MaxTokens);
    let (text, _) = greedy(&model, chat, 64);
    assert_eq!(
        text,
        "<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Topic: Why the \
         sky is blue.\n    *   Constraint: Exactly two sentences.\n\n2.  **Identify Key \
         Concepts:**\n    *   Sunlight is white (contains all colors).\n    *  "
    );

    let mut backends = vec![nibble::BackendKind::Cpu];
    if cfg!(target_os = "macos") {
        backends.push(nibble::BackendKind::Metal);
    }
    for backend in backends {
        for prompt in ["1, 1, 2, 3, 5, 8, 13,", chat] {
            let plain = greedy_on(&model, prompt, 64, backend, 0);
            for draft_max in [1, 3, 6] {
                let (text, stop, stats) = greedy_on(&model, prompt, 64, backend, draft_max);
                assert_eq!(
                    (&text, stop),
                    (&plain.0, plain.1),
                    "{backend}, {draft_max} drafts"
                );
                // A head that computed the wrong thing would get almost
                // none right; longer drafts are right less often.
                assert!(
                    stats.accepted * 3 > stats.drafted,
                    "{backend}, {draft_max} drafts: {stats:?}"
                );
            }
        }
    }
}

/// The chat prompt of the Qwen3.5 cases.
const SKY: &str = "<|im_start|>user\nExplain why the sky is blue in two sentences.<|im_end|>\n\
                   <|im_start|>assistant\n";

/// A mixture-of-experts model (`qwen35moe`), with Q4_K, Q5_K, Q6_K, Q8_0,
/// F32 and BF16 tensors. Expected text from llama.cpp b12722 on the CPU
/// (`-c 2048 -b 2048 -ub 1`, otherwise as above), which its Metal path also
/// gives; trailing newlines are not compared. Drafting with the MTP head,
/// itself a mixture of experts, leaves the text as it is.
#[test]
#[ignore = "needs models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"]
fn qwen36_35b_a3b_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf");
    assert!(model.has_mtp());
    let sky = "<think>\n\n</think>\n\nSunlight reaches Earth's atmosphere and is scattered in all \
               directions by gases and particles, a phenomenon known as Rayleigh scattering. \
               Because blue light is scattered more than other colors due to its shorter \
               wavelength, the sky appears blue to our eyes.";
    let code = "    match n {\n        0 => 0,\n        1 => 1,\n        _ => fibonacci(n - 1) + \
                fibonacci(n - 2),\n    }\n}\n\nfn main() {\n    let n = 10;\n    println!(\"The \
                {}-th Fibonacci number is: {}\", n, fibonacci(n));\n}";
    for draft_max in [0, 3] {
        for (prompt, want) in [(SKY, sky), ("fn fibonacci(n: u64) -> u64 {\n", code)] {
            let (text, ..) = greedy_on(&model, prompt, 128, nibble::BackendKind::Cpu, draft_max);
            assert_eq!(text.trim_end(), want, "drafting {draft_max}");
        }
    }
}

/// A dense Qwen3.5 model with `IQ4_NL` tensors (and Q5_K, IQ4_XS, Q6_K, Q4_K,
/// Q8_0, Q3_K). Expected text as for the 35B above.
#[test]
#[ignore = "needs models/Qwen3.8-27B-UD-Q4_K_XL.gguf"]
fn qwen38_27b_ud_q4_k_xl_greedy_matches_llama_cpp() {
    let model = load_model("Qwen3.8-27B-UD-Q4_K_XL.gguf");
    let (text, _) = greedy(&model, SKY, 64);
    assert_eq!(
        text.trim_end(),
        "<think>\nThe user wants a two-sentence explanation of why the sky is blue. This is \
         about Rayleigh scattering. I need to be concise and accurate in exactly two \
         sentences.\n\nKey points: Sunlight contains all colors. When it hits molecules in the \
         atmosphere, shorter wavelengths (blue/violet) scatter more than"
    );
    let (text, _) = greedy(&model, "fn fibonacci(n: u64) -> u64 {\n", 64);
    assert_eq!(
        text.trim_end(),
        "    if n <= 1 {\n        return n;\n    }\n    fibonacci(n - 1) + fibonacci(n - 2)\n}\n\n\
         fn main() {\n    let n = 10;\n    println!(\"The {}th Fibonacci number is: {}\", n, \
         fibonacci(n));"
    );
}

/// Quantized and bf16 KV caches give llama.cpp b12722's tokens, on the CPU
/// (`llama-completion -ngl 0 --device none --no-repack -ub 1 -ctk K -ctv V`,
/// `-fa on` exactly when V is quantized) and on Metal (`-ngl 99 -ub 1`).
/// With 4-bit keys this model's answer falls apart, in llama.cpp too: a
/// sensitive check. llama.cpp runs attention over `iq4_nl` keys on its CPU
/// even with `-ngl 99` (ggml-metal has no kernel for them), and 4-bit
/// caches diverge between its CPU and Metal paths, so Metal is checked on
/// the others.
#[test]
#[ignore = "needs models/Qwen3-0.6B-Q8_0.gguf"]
fn qwen3_cache_types_match_llama_cpp() {
    use nibble::{BackendKind, KvCacheType as T};
    let model = load_model("Qwen3-0.6B-Q8_0.gguf");
    let prompt = "<|im_start|>user\nWhat does this function do?\n\n```rust\n\
                  fn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        \
                  let t = a % b;\n        a = b;\n        b = t;\n    }\n    a\n}\n\
                  ```<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe function";
    let gcd = " `gcd(mut a: u64, mut b: u64) -> u64` is a **Greatest Common Divisor (GCD)** \
               function in Rust.\n\n### What it";
    let broken = format!(
        " of how how way how how to? \n\n</think>\n\n</think>\n\n{}Okay, I",
        "Okay, I am the best way.\n\n".repeat(3)
    );
    let iq4 = " `gcd(mut a: u64, mut b: u66) -> u64` in Rust does **compute the greatest \
               common divisor (GCD)** of two integers `a`";
    let cases = [
        (BackendKind::Cpu, T::Q8_0, T::Q8_0, gcd),
        (BackendKind::Cpu, T::Q4_0, T::Q4_0, broken.as_str()),
        (BackendKind::Cpu, T::Bf16, T::F16, gcd),
        (BackendKind::Cpu, T::Iq4Nl, T::Q5_1, iq4),
        (BackendKind::Metal, T::Q8_0, T::Q8_0, gcd),
        (BackendKind::Metal, T::Bf16, T::F16, gcd),
    ];
    for (backend, k, v, expected) in cases {
        if backend == BackendKind::Metal && !cfg!(target_os = "macos") {
            continue;
        }
        let options = SessionOptions {
            n_ctx: 512,
            backend,
            cache_type_k: k,
            cache_type_v: v,
            ..Default::default()
        };
        let (text, _, _) = greedy_with(&model, prompt, 40, options);
        assert_eq!(text, expected, "{backend} K {k} V {v}");
    }
}
