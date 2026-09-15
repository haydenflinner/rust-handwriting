//! Ink -> model-input encoding.
//!
//! Ported from armrest's `ml.rs` `ModelInput<Spline> for Ink`. Each input step
//! is `(dx, dy, dt, pen_up)`, deltas relative to the previous point, after
//! normalizing to unit height and simplifying the stroke.

use hwr_ink::ink::Ink;
use hwr_ink::math;

pub const WIDTH: usize = 4;

/// Clamp bound for `dx`/`dy` deltas. `Ink::normalize` scales BOTH axes by a
/// factor derived from Y-height alone (preserving aspect ratio, deliberate —
/// see `template_match.rs`'s docs for why squashing to a fixed box would
/// hurt shape discrimination) — but for a wide/short line, that means X
/// deltas can end up far larger in magnitude than Y's, since nothing bounds
/// them independently. Measured on 425 real samples: `dy` std=0.183,
/// max≈1.0, matching the unit-height target — `dx` std=0.315 but max=22.9,
/// a 20x+ outlier. `BatchNorm` (see `model.rs`) computes shared statistics
/// jointly over `[batch, steps]` per channel, so one such outlier sample
/// anywhere in a batch skews normalization for every other sample at every
/// timestep in that batch — a real, previously-unexamined contributor to
/// why training was struggling, found by an audit of the whole pipeline
/// after hours of LR/architecture tuning alone didn't unstick it. Clamping
/// here is the surgical fix: bounds the outliers without changing the
/// aspect-ratio-preserving normalization itself.
const MAX_XY_DELTA: f32 = 3.0;

/// Separate, asymmetric bound for `dt`: time should never run backwards, so
/// unlike `dx`/`dy` this clamps to `[0, MAX_DT_DELTA]`, not
/// `[-MAX_DT_DELTA, MAX_DT_DELTA]`. The same audit found at least one
/// sample with a negative `dt` (min=-1.6s) in the real corpus — a genuine
/// data-quality issue (out-of-order timestamps within a stroke) that this
/// clamp papers over rather than fixes, but bounding it here is still
/// better than feeding a nonsensical negative time delta into training.
const MAX_DT_DELTA: f32 = 3.0;

/// Normalize + simplify an ink the way the model expects: unit height,
/// smoothed, minimum-distance filtered, then Douglas-Peucker simplified.
fn prepare(ink: &Ink) -> Ink {
    let mut normal = ink.clone();
    normal.normalize(1.0);
    normal.smooth(0.01);
    normal = math::min_distance(&normal, 0.05);
    math::douglas_peucker(&normal, 0.01)
}

/// Encode `ink` as a flat `[steps * WIDTH]` vec of `(dx, dy, dt, pen_up)` steps.
pub fn encode_vec(ink: &Ink) -> Vec<f32> {
    if ink.is_empty() {
        return vec![];
    }

    let normal = prepare(ink);
    let points = normal.points();
    let mut buffer = Vec::with_capacity(points.len() * WIDTH);
    let mut last_point = points[0];
    for (i, point) in points.iter().enumerate() {
        buffer.push((point.x - last_point.x).clamp(-MAX_XY_DELTA, MAX_XY_DELTA));
        buffer.push((point.y - last_point.y).clamp(-MAX_XY_DELTA, MAX_XY_DELTA));
        buffer.push((point.z - last_point.z).clamp(0.0, MAX_DT_DELTA));
        buffer.push(if normal.is_pen_up(i) { 1.0 } else { 0.0 });
        last_point = *point;
    }
    buffer
}

/// Encode `ink` into `buffer` (a flat `[steps * WIDTH]` array of `f32`s),
/// e.g. for building a fixed-width, zero-padded training batch.
/// Returns the number of steps written (`<= buffer.len() / WIDTH`).
pub fn encode(ink: &Ink, buffer: &mut [f32]) -> usize {
    if ink.is_empty() {
        return 0;
    }

    let normal = prepare(ink);
    let points = normal.points();
    let mut last_point = points[0];
    for (i, (slice, point)) in buffer
        .chunks_exact_mut(WIDTH)
        .zip(points.iter())
        .enumerate()
    {
        slice[0] = (point.x - last_point.x).clamp(-MAX_XY_DELTA, MAX_XY_DELTA);
        slice[1] = (point.y - last_point.y).clamp(-MAX_XY_DELTA, MAX_XY_DELTA);
        slice[2] = (point.z - last_point.z).clamp(0.0, MAX_DT_DELTA);
        slice[3] = if normal.is_pen_up(i) { 1.0 } else { 0.0 };

        last_point = *point;
    }

    normal.len().min(buffer.len() / WIDTH)
}

/// Convenience wrapper: allocate a `max_steps`-long buffer, encode into it,
/// and return `(buffer, steps_written)`.
pub fn encode_padded(ink: &Ink, max_steps: usize) -> (Vec<f32>, usize) {
    let mut buffer = vec![0f32; max_steps * WIDTH];
    let steps = encode(ink, &mut buffer);
    (buffer, steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_empty() {
        let ink = Ink::new();
        let (buffer, steps) = encode_padded(&ink, 16);
        assert_eq!(steps, 0);
        assert!(buffer.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_encode_line() {
        let mut ink = Ink::new();
        ink.push(0.0, 0.0, 0.0);
        ink.push(1.0, 1.0, 0.1);
        ink.push(2.0, 0.0, 0.2);
        ink.pen_up();
        let (_buffer, steps) = encode_padded(&ink, 16);
        assert!(steps > 0);
    }
}
