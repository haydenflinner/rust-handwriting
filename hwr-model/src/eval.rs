//! Mean character error rate (CER) of a `model::Recognizer` against a corpus
//! of `(text, ink)` pairs — used both after training and by the `recognize`
//! CLI.

use burn::tensor::backend::Backend;
use hwr_ink::ink::Ink;

use crate::{decode, model, spline};

/// Historical TCN autotune floor. HAT attention is fine at batch=1; leftover
/// eval chunks still pad up to this so launch shapes stay uniform.
pub const SAFE_BATCH: usize = 16;

/// Distinct samples packed into one eval launch. Images are 224², so this
/// is lower than the old LSTM-only 128 to keep Metal memory bounded.
pub const EVAL_BATCH: usize = 32;

struct Encoded {
    text: String,
    encoded: Vec<f32>,
    steps: usize,
}

pub fn mean_cer<B: Backend>(
    net: &model::Recognizer<B>,
    pairs: &[(String, Ink)],
    device: &B::Device,
) -> f64 {
    mean_cer_batched(net, pairs, device, EVAL_BATCH)
}

pub fn mean_cer_batched<B: Backend>(
    net: &model::Recognizer<B>,
    pairs: &[(String, Ink)],
    device: &B::Device,
    batch: usize,
) -> f64 {
    mean_cers_batched(net, pairs, device, batch).0
}

/// Greedy CER and CTC prefix-beam CER from the same forwards.
/// `beam` uses width 8; if every gold label is digits-only, the LM is the
/// digit alphabet (blocks `B`/`O`/…), otherwise LM odds are 1 (alignment
/// summing only).
pub fn mean_cers_batched<B: Backend>(
    net: &model::Recognizer<B>,
    pairs: &[(String, Ink)],
    device: &B::Device,
    batch: usize,
) -> (f64, f64) {
    let batch = batch.max(SAFE_BATCH);
    let mut prepared: Vec<Encoded> = pairs
        .iter()
        .filter_map(|(text, ink)| {
            let encoded = spline::encode_strokes(ink);
            let steps = encoded.len() / spline::STROKE_DIM;
            if steps == 0 {
                None
            } else {
                Some(Encoded {
                    text: text.clone(),
                    encoded,
                    steps,
                })
            }
        })
        .collect();
    prepared.sort_by_key(|s| s.steps);
    let digits_only = !prepared.is_empty()
        && prepared
            .iter()
            .all(|s| s.text.chars().all(|c| c.is_ascii_digit()));

    let mut greedy_total = 0.0f64;
    let mut beam_total = 0.0f64;
    let mut count = 0usize;
    for chunk in prepared.chunks(batch) {
        let (g, b, n) = score_chunk(net, chunk, batch, device, digits_only);
        greedy_total += g;
        beam_total += b;
        count += n;
    }
    if count > 0 {
        (greedy_total / count as f64, beam_total / count as f64)
    } else {
        (f64::NAN, f64::NAN)
    }
}

fn score_chunk<B: Backend>(
    net: &model::Recognizer<B>,
    chunk: &[Encoded],
    launch: usize,
    device: &B::Device,
    digits_only: bool,
) -> (f64, f64, usize) {
    let n_real = chunk.len();
    if n_real == 0 {
        return (0.0, 0.0, 0);
    }
    let max_steps = chunk.iter().map(|s| s.steps).max().unwrap_or(0);
    if max_steps == 0 {
        return (0.0, 0.0, 0);
    }
    let launch = launch.max(n_real).max(SAFE_BATCH);
    let rows = chunk.iter().map(|s| (s.encoded.as_slice(), s.steps));
    let (strokes, images, pad) = spline::pack_hat_batch(rows, n_real, launch, max_steps);
    let (strokes, images, pad_mask) =
        model::packed_inputs(strokes, images, pad, launch, max_steps, device);
    let probs = net.forward(strokes, images, Some(pad_mask));
    let Ok(flat) = probs.into_data().to_vec::<f32>() else {
        return (0.0, 0.0, 0);
    };
    let (g, b, n) = score_flat(&flat, launch, max_steps, chunk, digits_only);
    (g, b, n)
}

/// Decode each *real* row using only its unpadded timesteps. The padded
/// tail is zeros, whose argmax is class 0 (space) and would inflate CER
/// if we ran greedy over the whole `max_steps`.
fn score_flat(
    flat: &[f32],
    launch: usize,
    max_steps: usize,
    chunk: &[Encoded],
    digits_only: bool,
) -> (f64, f64, usize) {
    let classes = decode::classes();
    let per_row = max_steps * classes;
    if per_row == 0 || flat.len() < launch * per_row {
        return (0.0, 0.0, 0);
    }
    let alphabet: Vec<char> = decode::CHARS.chars().collect();
    let digits: Vec<char> = ('0'..='9').collect();
    let mut greedy_total = 0.0f64;
    let mut beam_total = 0.0f64;
    let n_real = chunk.len();
    for (bi, sample) in chunk.iter().enumerate() {
        if bi >= launch {
            break;
        }
        let row = &flat[bi * per_row..(bi + 1) * per_row];
        let usable = sample.steps.min(max_steps) * classes;
        let slice = &row[..usable];
        let greedy = decode::greedy_decode(slice);
        let beam = beam_best(slice, &alphabet, digits_only.then_some(digits.as_slice()));
        greedy_total += sample_cer(&sample.text, &greedy);
        beam_total += sample_cer(&sample.text, &beam);
    }
    (greedy_total, beam_total, n_real)
}

fn beam_best(buffer: &[f32], alphabet: &[char], digit_lm: Option<&[char]>) -> String {
    const BEAM: usize = 8;
    let ranked = match digit_lm {
        Some(digits) => decode::beam_decode(buffer, BEAM, alphabet, &digits),
        None => decode::beam_decode(buffer, BEAM, alphabet, &true),
    };
    ranked
        .first()
        .map(|(s, _)| s.clone())
        .unwrap_or_default()
}

fn sample_cer(gold: &str, pred: &str) -> f64 {
    if gold.is_empty() {
        0.0
    } else {
        strsim::levenshtein(gold, pred) as f64 / gold.chars().count() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_hot_row(classes: usize, steps: usize, picks: &[usize]) -> Vec<f32> {
        let mut buf = vec![0.0f32; steps * classes];
        for (t, &c) in picks.iter().enumerate() {
            buf[t * classes + c] = 1.0;
        }
        buf
    }

    #[test]
    fn greedy_cer_from_one_hot() {
        let classes = decode::classes();
        let a = decode::CHARS.find('a').unwrap();
        let chunk = [Encoded {
            text: "a".into(),
            encoded: vec![0.0; spline::STROKE_DIM],
            steps: 1,
        }];
        let mut picks = vec![decode::classes() - 1]; // blank
        picks[0] = a;
        let flat = one_hot_row(classes, 1, &picks);
        let (sum, _, n) = score_flat(&flat, 1, 1, &chunk, false);
        assert_eq!(n, 1);
        assert!(sum < 1e-6, "sum={sum}");
    }
}
