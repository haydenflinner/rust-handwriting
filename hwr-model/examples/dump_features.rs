//! Dump `extract_features` output for a fixture .ink file as csv rows.
//! Usage: cargo run -p hwr-model --example dump_features -- <name>

use hwr_ink::ink::Ink;
use hwr_model::onnet::{extract_features, FEATURE_DIM};

fn main() {
    let name = std::env::args().nth(1).expect("fixture name");
    let text = std::fs::read_to_string(format!("tests/fixtures/{name}.ink")).unwrap();
    let ink = Ink::from_string(text.trim());
    let feats = extract_features(&ink);
    for row in feats.chunks(FEATURE_DIM) {
        println!(
            "{}",
            row.iter()
                .map(|v| format!("{v:.9}"))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}
