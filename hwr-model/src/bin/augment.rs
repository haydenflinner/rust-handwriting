//! Expand a corpus of real `(text, ink)` recordings into a larger corpus of
//! augmented variants (rotation/scale/shear/jitter/time-warp — see
//! `hwr_model::augment`), so the writer doesn't have to write everything
//! multiple times by hand to get variety into training.
//!
//! Usage:
//!   augment --out PATH --count N [SOURCE...]
//!
//!   --out PATH   where to write the augmented corpus (armrest `text\tink`
//!                format — directly usable as a `train` source)
//!   --count N    augmented variants to generate per input sample (default: 8)
//!   SOURCE...    corpus files/dirs (default: armrest/data/inks)
//!
//! The output does NOT include the original samples — pass both the
//! original source(s) and the augmented output to `train` together, so the
//! model still sees the real, unperturbed ink too:
//!   augment --out checkpoints/augmented.txt --count 8 \
//!       armrest/data/inks ~/Library/Application\ Support/hwr/calibration.txt
//!   train armrest/data/inks ~/Library/Application\ Support/hwr/calibration.txt \
//!       checkpoints/augmented.txt

use std::path::PathBuf;

use rand::SeedableRng;

use hwr_model::augment::{augment_n, AugmentConfig};
use hwr_model::corpus::{load_source, save_pairs};

fn main() {
    let mut out_path = PathBuf::from("augmented.txt");
    let mut count = 8usize;
    let mut sources: Vec<PathBuf> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => out_path = PathBuf::from(args.next().expect("--out needs a path")),
            "--count" => {
                count = args
                    .next()
                    .expect("--count needs a number")
                    .parse()
                    .expect("count must be a number")
            }
            other => sources.push(PathBuf::from(other)),
        }
    }
    if sources.is_empty() {
        sources.push(PathBuf::from("armrest/data/inks"));
    }

    let mut pairs = Vec::new();
    for source in &sources {
        load_source(source, &mut pairs);
    }
    println!(
        "Loaded {} (text, ink) pairs from {} source(s)",
        pairs.len(),
        sources.len()
    );

    let config = AugmentConfig::default();
    let mut rng = rand::rngs::StdRng::seed_from_u64(9001);

    let mut augmented = Vec::with_capacity(pairs.len() * count);
    for (text, ink) in &pairs {
        augmented.extend(augment_n(text, ink, count, &config, &mut rng));
    }

    println!(
        "Generated {} augmented samples ({count} per input), writing to {}",
        augmented.len(),
        out_path.display()
    );
    save_pairs(&out_path, &augmented).expect("failed to write augmented corpus");
}
