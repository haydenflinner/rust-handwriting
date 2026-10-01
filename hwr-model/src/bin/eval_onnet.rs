//! Score the imported ONNET model (greedy CTC) against a labeled ink corpus —
//! by default the app's own calibration file, so this measures real user
//! handwriting rather than IAM-OnDB.
//!
//!   cargo run -p hwr-model --bin eval_onnet [path/to/calibration.txt]

use hwr_model::corpus;
use hwr_model::onnet::Onnet;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        format!(
            "{}/Library/Application Support/hwr/calibration.txt",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let pairs = corpus::load_pairs(&path).expect("read calibration corpus");
    eprintln!("eval_onnet: {} samples from {path}", pairs.len());

    let rec = Onnet::new();
    // Bucket by label length: ONNET was trained on full English text lines,
    // so isolated glyphs are far out of distribution — report separately.
    let mut buckets = [[0usize; 4]; 3]; // per bucket: n, exact, edits, chars
    let bucket = |len: usize| match len {
        0 | 1 => 0,
        2 | 3 => 1,
        _ => 2,
    };
    for (text, ink) in &pairs {
        let pred = rec.recognize(ink);
        let d = strsim::levenshtein(text, &pred);
        let b = &mut buckets[bucket(text.chars().count())];
        b[0] += 1;
        b[1] += (d == 0) as usize;
        b[2] += d;
        b[3] += text.chars().count();
        if d != 0 {
            println!("  {text:?} -> {pred:?}  (edit distance {d})");
        }
    }
    let report = |name: &str, b: [usize; 4]| {
        if b[0] == 0 {
            return;
        }
        println!(
            "{name:>24}: {} samples, {} exact ({:.0}%), CER {:.1}%",
            b[0],
            b[1],
            100.0 * b[1] as f64 / b[0] as f64,
            100.0 * b[2] as f64 / b[3].max(1) as f64,
        );
    };
    report("len 1 (single glyphs)", buckets[0]);
    report("len 2-3", buckets[1]);
    report("len 4+", buckets[2]);
    let mut all = [0; 4];
    for b in &buckets {
        for i in 0..4 {
            all[i] += b[i];
        }
    }
    report("ALL", all);
}
