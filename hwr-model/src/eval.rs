//! Mean character error rate (CER) of a `model::Recognizer` against a corpus
//! of `(text, ink)` pairs — used both after training and by the `recognize`
//! CLI.

use burn::tensor::{Tensor, TensorData};
use hwr_ink::ink::Ink;

use crate::fused_lstm::FusedLstm;
use crate::{decode, model, spline};

/// CubeCL's GPU autotuner for some ops (historically the TCN `Conv1d`)
/// crashes hard at `batch_size=1`. Training's own batch=16 is already
/// validated to work, so eval and probes pad out to this shape and discard
/// the extra rows rather than risk an untested size.
pub const SAFE_BATCH: usize = 16;

pub fn mean_cer<B: FusedLstm>(
    net: &model::Recognizer<B>,
    pairs: &[(String, Ink)],
    device: &B::Device,
) -> f64 {
    let mut total = 0.0f64;
    let mut count = 0usize;

    for (text, ink) in pairs {
        let encoded = spline::encode_vec(ink);
        let steps = encoded.len() / spline::WIDTH;
        if steps == 0 {
            continue;
        }
        // Replicate to SAFE_BATCH — see that constant's docs for why.
        let mut batched = Vec::with_capacity(encoded.len() * SAFE_BATCH);
        for _ in 0..SAFE_BATCH {
            batched.extend_from_slice(&encoded);
        }
        let input = Tensor::<B, 3>::from_data(
            TensorData::new(batched, [SAFE_BATCH, steps, spline::WIDTH]),
            device,
        );
        let probs = net.forward(input);
        let Ok(flat) = probs.into_data().to_vec::<f32>() else {
            continue;
        };
        let per_sample = flat.len() / SAFE_BATCH;
        let actual = decode::greedy_decode(&flat[..per_sample]);

        let dist = strsim::levenshtein(text, &actual) as f64;
        let cer = if text.is_empty() {
            0.0
        } else {
            dist / text.chars().count() as f64
        };
        total += cer;
        count += 1;
    }

    if count > 0 {
        total / count as f64
    } else {
        f64::NAN
    }
}
