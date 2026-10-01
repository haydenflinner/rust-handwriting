//! End-to-end check of the imported IAMhwr ONNET model against fixtures
//! dumped from the original Python pipeline (scripts/dump_fixture.py):
//!
//!   .ink       — the raw stroke data as an `Ink::from_string` literal
//!   .features  — SCHEME6 features produced by the repo's own datarep.py
//!
//! `m05-507z-06` greedy-decodes exactly under the reference implementation
//! (~7% CER overall on the bundled t2 set), so the Rust path must reproduce
//! it verbatim. `b04-334z-07` is a known miss ('Surrey' -> 'Suries'); it
//! still exercises feature parity.

use hwr_ink::ink::Ink;
use hwr_model::onnet::{self, Onnet, FEATURE_DIM};

fn load_ink(name: &str) -> Ink {
    let text = std::fs::read_to_string(format!("tests/fixtures/{name}.ink")).unwrap();
    Ink::from_string(text.trim())
}

fn load_rows(name: &str, ext: &str) -> Vec<f32> {
    std::fs::read_to_string(format!("tests/fixtures/{name}.{ext}"))
        .unwrap()
        .lines()
        .flat_map(|row| row.split(',').map(|v| v.parse::<f32>().unwrap()))
        .collect()
}

fn load_features(name: &str) -> Vec<f32> {
    load_rows(name, "features")
}

/// Rows rarely differ by ±1: `resample_distance` compares `l > d` at
/// floating-point boundaries, and upstream coords already differ ~1e-15 from
/// numpy's (np.polyfit is SVD-based). Allow a few insertion/deletion rows
/// while requiring everything else to match closely.
fn assert_features_match(name: &str) {
    let ink = load_ink(name);
    let got = onnet::extract_features(&ink);
    let want = load_features(name);
    let (got, want) = (
        got.chunks(FEATURE_DIM).collect::<Vec<_>>(),
        want.chunks(FEATURE_DIM).collect::<Vec<_>>(),
    );

    // Compare x, y, dx, dy positionally — the down/up flags at a stroke
    // boundary legitimately shift by a row when an endpoint is duplicated.
    let close = |a: &[f32], b: &[f32]| a[..4].iter().zip(&b[..4]).all(|(x, y)| (x - y).abs() < 1e-4);
    let (mut i, mut j, mut skips, mut worst) = (0, 0, 0, 0.0f32);
    while i < got.len() && j < want.len() {
        if close(got[i], want[j]) {
            worst = worst.max(
                got[i][..4]
                    .iter()
                    .zip(&want[j][..4])
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max),
            );
            i += 1;
            j += 1;
        } else if j + 1 < want.len() && close(got[i], want[j + 1]) {
            j += 1;
            skips += 1;
        } else if i + 1 < got.len() && close(got[i + 1], want[j]) {
            i += 1;
            skips += 1;
        } else {
            panic!("{name}: row {i}/{j} diverges: {:?} vs {:?}", got[i], want[j]);
        }
    }
    skips += (got.len() - i) + (want.len() - j);
    assert!(skips <= 2, "{name}: {skips} unaligned feature rows");

    // Same number of stroke ends.
    let ups = |rows: &[&[f32]]| rows.iter().filter(|r| r[5] > 0.5).count();
    assert_eq!(ups(&got), ups(&want), "{name}: stroke count differs");
    eprintln!("{name}: {} rows aligned, {skips} skipped, max diff {worst:e}", i);
}

#[test]
fn features_match_python_reference() {
    assert_features_match("m05-507z-06");
    assert_features_match("b04-334z-07");
}

/// Burn (wgpu/Metal) output vs the onnxruntime reference for the same
/// input — catches subtle import mismatches (gate order, BN stats) that a
/// passing decode could hide.
#[test]
fn logits_match_onnxruntime() {
    let recognizer = Onnet::new();
    // Feed the reference features straight in — preprocessing parity is
    // covered separately, and a BiLSTM's every output depends on the whole
    // input, so the ±1 boundary row would smear a positional comparison.
    let feats = load_features("m05-507z-06");
    let got = recognizer.forward(&feats);
    let want = load_rows("m05-507z-06", "logits");
    assert_eq!(got.len(), want.len(), "logit element count differs");
    let n = got.len();
    let worst = got[..n]
        .iter()
        .zip(&want[..n])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("logit diff: {worst:e} over {n} values ({} rows)", n / 83);
    assert!(worst < 0.05, "burn logits diverge from onnxruntime: {worst}");
}

#[test]
fn decodes_known_line() {
    let recognizer = Onnet::new();
    let ink = load_ink("m05-507z-06");
    assert_eq!(recognizer.recognize(&ink), "the other's viewpoint . ");
}

#[test]
fn empty_ink_decodes_to_empty() {
    let recognizer = Onnet::new();
    assert_eq!(recognizer.recognize(&Ink::new()), "");
}
