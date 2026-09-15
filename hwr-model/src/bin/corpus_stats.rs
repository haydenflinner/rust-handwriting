//! Report the distribution of post-simplification encoded sequence lengths
//! (`spline::encode_vec` step counts) across one or more corpus sources —
//! used to characterize why BiLSTM training under Burn's autodiff is so
//! much slower than expected (graph-node count scales with sequence
//! length, and that scaling is super-linear — see `burn-spike`).
//!
//! Usage: corpus_stats [--max-steps N] [SOURCE...]  (default: armrest/data/inks)
//!
//! `--max-steps N`, if given, applies `corpus::cap_long_samples` before
//! reporting — so you can preview its effect on the length distribution.

use std::path::PathBuf;

use hwr_model::corpus::load_source;
use hwr_model::spline;

fn main() {
    let mut max_steps: Option<usize> = None;
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--max-steps" => {
                max_steps = Some(
                    args.next()
                        .expect("--max-steps needs a number")
                        .parse()
                        .expect("max steps must be a number"),
                )
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
    if let Some(max_steps) = max_steps {
        pairs = hwr_model::corpus::cap_long_samples(pairs, max_steps);
    }

    let mut lens: Vec<usize> = pairs
        .iter()
        .map(|(_, ink)| spline::encode_vec(ink).len() / spline::WIDTH)
        .collect();
    lens.sort_unstable();

    if lens.is_empty() {
        println!("no samples");
        return;
    }

    let n = lens.len();
    let pct = |p: f64| lens[((n as f64 - 1.0) * p).round() as usize];
    let sum: usize = lens.iter().sum();

    println!("samples: {n}");
    println!("min:     {}", lens[0]);
    println!("p50:     {}", pct(0.50));
    println!("p90:     {}", pct(0.90));
    println!("p99:     {}", pct(0.99));
    println!("max:     {}", lens[n - 1]);
    println!("mean:    {:.1}", sum as f64 / n as f64);

    // Text-length correlation: longest 5 samples, with their text and length.
    let mut by_len: Vec<(usize, &str)> = pairs
        .iter()
        .map(|(text, ink)| (spline::encode_vec(ink).len() / spline::WIDTH, text.as_str()))
        .collect();
    by_len.sort_by_key(|&(len, _)| std::cmp::Reverse(len));
    println!("\nlongest samples:");
    for (len, text) in by_len.iter().take(5) {
        let preview: String = text.chars().take(60).collect();
        println!("  {len:>5} steps  {:?}", preview);
    }
}
