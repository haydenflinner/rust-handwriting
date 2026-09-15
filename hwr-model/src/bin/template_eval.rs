//! Evaluate the $1/$P-style nearest-neighbor template matcher (see
//! `template_match.rs`) on a held-out split, using the exact same
//! train/val split methodology as the CTC model's training run (shuffle
//! seed 1234, 10% held out) for a fair, apples-to-apples comparison.
//!
//! Usage: template_eval [SOURCE...]  (default: armrest/data/inks)

use std::path::PathBuf;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use hwr_model::corpus::load_source;
use hwr_model::template_match::TemplateLibrary;

fn main() {
    let mut sources: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if sources.is_empty() {
        sources.push(PathBuf::from("armrest/data/inks"));
    }

    let mut pairs = Vec::new();
    for source in &sources {
        load_source(source, &mut pairs);
    }
    println!("Loaded {} pairs from {} source(s)", pairs.len(), sources.len());

    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    pairs.shuffle(&mut rng);

    let val_count = ((pairs.len() as f64) * 0.1).round() as usize;
    let (val_pairs, train_pairs) = pairs.split_at(val_count);
    println!("Split: {} train (=template library), {} validation", train_pairs.len(), val_pairs.len());

    let library = TemplateLibrary::new(train_pairs);

    let mut exact_matches = 0usize;
    let mut total_cer = 0.0f64;
    let mut evaluated = 0usize;
    for (expected, ink) in val_pairs {
        let Some((predicted, distance)) = library.recognize(ink) else {
            continue;
        };
        let dist = strsim::levenshtein(expected, predicted) as f64;
        let cer = if expected.is_empty() {
            0.0
        } else {
            dist / expected.chars().count() as f64
        };
        total_cer += cer;
        evaluated += 1;
        if predicted == expected {
            exact_matches += 1;
        }
        if evaluated <= 30 {
            println!("  [{cer:.2}, dist={distance:.3}] {expected:?} -> {predicted:?}");
        }
    }

    println!(
        "\nExact match: {exact_matches}/{evaluated} ({:.1}%)",
        100.0 * exact_matches as f64 / evaluated.max(1) as f64
    );
    println!("Mean CER: {:.4}", total_cer / evaluated.max(1) as f64);
}
