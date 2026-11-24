//! Checks against llama.cpp on real model files.
//!
//! Ignored by default. Put the models in `models/` at the repository root, then:
//!
//! ```sh
//! cargo test --release -p nibble --test reference -- --ignored
//! ```
//!
//! Expected values were produced by llama.cpp b12675 (`llama-tokenize --ids`,
//! and `llama-completion --temp 0 -fa off`). Generation cases were chosen to
//! agree with llama.cpp with flash attention both on and off, so they are not
//! near-ties.

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

/// Greedy completion of `prompt` (special tokens parsed), as text.
fn greedy(model: &Model, prompt: &str, max_tokens: usize) -> (String, StopReason) {
    let tok = model.tokenizer();
    let tokens = tok.encode(prompt, true);
    let mut session = model.session(SessionOptions {
        n_ctx: 512,
        threads: 0,
    });
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
    (text, stop)
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

    let chat = "<|im_start|>user\nWrite a haiku about the Rust programming language.<|im_end|>\n\
                <|im_start|>assistant\n<think>\n\n</think>\n";
    let (text, stop) = greedy(&model, chat, 64);
    assert_eq!(text, "A rust program  \nA rust program  \nA rust program.");
    assert!(matches!(stop, StopReason::EndOfGeneration(_)));
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
}
