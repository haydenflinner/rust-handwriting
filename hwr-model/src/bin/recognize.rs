//! Smoke-test CLI: recognize every line of an armrest-format corpus file and
//! report per-line output plus mean character error rate (CER).
//!
//! Analogous to `armrest/src/bin/test-tflite.rs`, but against the pure-Rust
//! candle recognizer. With `Recognizer::random()` weights the CER will be
//! near 1.0 (garbage) — this exists to validate the pipeline shape, not
//! accuracy, until a trained checkpoint exists (Phase 3).

use std::path::PathBuf;

use hwr_model::Recognizer;

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../armrest/data/inks/jabberwocky.txt"));

    let checkpoint = std::env::args().nth(2);

    let recognizer = match checkpoint {
        Some(path) => Recognizer::load(&path).expect("failed to load checkpoint"),
        None => {
            eprintln!("No checkpoint given; using random (untrained) weights.");
            Recognizer::random().expect("failed to build recognizer")
        }
    };

    let pairs = hwr_model::corpus::load_pairs(&path)
        .unwrap_or_else(|e| panic!("failed to load corpus {}: {e}", path.display()));

    println!("Loaded {} lines from {}", pairs.len(), path.display());

    let mut total_cer = 0.0f64;
    for (expected, ink) in &pairs {
        let actual = recognizer
            .recognize_greedy(ink)
            .unwrap_or_else(|e| panic!("recognition failed: {e}"));

        let dist = strsim::levenshtein(expected, &actual) as f64;
        let cer = if expected.is_empty() {
            0.0
        } else {
            dist / expected.len() as f64
        };
        total_cer += cer;

        println!("[{cer:.3}] {expected:?} -> {actual:?}");
    }

    if !pairs.is_empty() {
        println!("Mean CER: {:.4}", total_cer / pairs.len() as f64);
    }
}
