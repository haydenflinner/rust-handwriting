//! Ink -> model-input encoding.
//!
//! Ported from armrest's `ml.rs` `ModelInput<Spline> for Ink`. Each input step
//! is `(dx, dy, dt, pen_up)`, deltas relative to the previous point, after
//! normalizing to unit height and simplifying the stroke.

use hwr_ink::ink::Ink;
use hwr_ink::math;

pub const WIDTH: usize = 4;

/// HAT stroke token: `(x, y, pen_up)` after the same normalize/simplify as
/// [`encode_vec`]. Paper (Lodh et al. 2025, transformer-joint) §3 uses
/// absolute coordinates plus a discrete pen-state, not raw deltas.
pub const STROKE_DIM: usize = 3;

/// Offline crop size for the HAT image branch (`I` resized to 224×224).
pub const IMAGE_SIZE: usize = 224;
pub const IMAGE_CHANNELS: usize = 3;
pub const IMAGE_FLOATS: usize = IMAGE_CHANNELS * IMAGE_SIZE * IMAGE_SIZE;

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

/// Encode `ink` as a flat `[steps * STROKE_DIM]` vec of `(x, y, pen_up)`.
pub fn encode_strokes(ink: &Ink) -> Vec<f32> {
    if ink.is_empty() {
        return vec![];
    }
    let normal = prepare(ink);
    let mut buffer = Vec::with_capacity(normal.len() * STROKE_DIM);
    for (i, point) in normal.points().iter().enumerate() {
        buffer.push(point.x);
        buffer.push(point.y);
        buffer.push(if normal.is_pen_up(i) { 1.0 } else { 0.0 });
    }
    buffer
}

/// Rasterize prepared `(x, y, pen_up)` strokes onto a 224×224 grayscale
/// crop, replicated to 3 channels (CHW, values in `[0, 1]`). Matches the
/// paper's offline crop: bounding box of the glyph, aspect preserved,
/// letterboxed into 224².
pub fn rasterize_strokes(strokes: &[f32], steps: usize) -> Vec<f32> {
    let mut img = vec![0f32; IMAGE_SIZE * IMAGE_SIZE];
    if steps == 0 {
        return replicate_channels(&img);
    }
    let n = steps.min(strokes.len() / STROKE_DIM);
    if n == 0 {
        return replicate_channels(&img);
    }

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for i in 0..n {
        let x = strokes[i * STROKE_DIM];
        let y = strokes[i * STROKE_DIM + 1];
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    let dx = (max_x - min_x).max(1e-3);
    let dy = (max_y - min_y).max(1e-3);
    let margin = 16.0;
    let usable = IMAGE_SIZE as f32 - 2.0 * margin;
    let scale = usable / dx.max(dy);

    let map = |x: f32, y: f32| -> (i32, i32) {
        let px = margin + (x - min_x) * scale + 0.5 * (usable - dx * scale);
        let py = margin + (y - min_y) * scale + 0.5 * (usable - dy * scale);
        (px.round() as i32, py.round() as i32)
    };

    for i in 0..n.saturating_sub(1) {
        let off = i * STROKE_DIM;
        if strokes[off + 2] >= 0.5 {
            // Pen-up: this point ends a stroke; don't connect to the next.
            continue;
        }
        let (x0, y0) = map(strokes[off], strokes[off + 1]);
        let (x1, y1) = map(strokes[off + 3], strokes[off + 4]);
        draw_line(&mut img, x0, y0, x1, y1);
    }
    if n == 1 {
        let (x, y) = map(strokes[0], strokes[1]);
        stamp(&mut img, x, y);
    }

    replicate_channels(&img)
}

/// Normalize + simplify `ink`, then rasterize. Same points as [`encode_strokes`].
pub fn rasterize_ink(ink: &Ink) -> Vec<f32> {
    let strokes = encode_strokes(ink);
    let steps = strokes.len() / STROKE_DIM;
    rasterize_strokes(&strokes, steps)
}

fn replicate_channels(gray: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(IMAGE_FLOATS);
    for _ in 0..IMAGE_CHANNELS {
        out.extend_from_slice(gray);
    }
    out
}

fn stamp(img: &mut [f32], x: i32, y: i32) {
    let w = IMAGE_SIZE as i32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            let xx = x + dx;
            let yy = y + dy;
            if xx >= 0 && yy >= 0 && xx < w && yy < w {
                img[(yy as usize) * IMAGE_SIZE + xx as usize] = 1.0;
            }
        }
    }
}

fn draw_line(img: &mut [f32], mut x0: i32, mut y0: i32, x1: i32, y1: i32) {
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    loop {
        stamp(img, x0, y0);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Pack `launch` rows of HAT inputs (repeating the last real row to fill
/// the launch size). Returns `(strokes, images, pad_mask)` where
/// `strokes` is `[launch * max_steps * STROKE_DIM]`, `images` is
/// `[launch * IMAGE_FLOATS]` CHW, and `pad_mask` is `[launch * max_steps]`
/// with `1.0` = padded timestep (Burn MHA `mask_pad` convention).
pub fn pack_hat_batch<'a, I>(rows: I, n_real: usize, launch: usize, max_steps: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>)
where
    I: IntoIterator<Item = (&'a [f32], usize)>,
{
    let rows: Vec<(&'a [f32], usize)> = rows.into_iter().collect();
    let n_real = n_real.min(rows.len()).max(1);
    let mut strokes = vec![0f32; launch * max_steps * STROKE_DIM];
    let mut images = vec![0f32; launch * IMAGE_FLOATS];
    let mut pad_mask = vec![1f32; launch * max_steps];
    for bi in 0..launch {
        let (src, steps) = rows[bi.min(n_real - 1)];
        let steps = steps.min(max_steps).min(src.len() / STROKE_DIM);
        let dst = bi * max_steps * STROKE_DIM;
        let len = steps * STROKE_DIM;
        strokes[dst..dst + len].copy_from_slice(&src[..len]);
        let img = rasterize_strokes(src, steps);
        let idst = bi * IMAGE_FLOATS;
        images[idst..idst + IMAGE_FLOATS].copy_from_slice(&img);
        let mdst = bi * max_steps;
        for t in 0..steps {
            pad_mask[mdst + t] = 0.0;
        }
    }
    (strokes, images, pad_mask)
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

    #[test]
    fn test_strokes_and_raster_shapes() {
        let mut ink = Ink::new();
        ink.push(0.0, 0.0, 0.0);
        ink.push(1.0, 1.0, 0.1);
        ink.push(2.0, 0.0, 0.2);
        ink.pen_up();
        let strokes = encode_strokes(&ink);
        assert_eq!(strokes.len() % STROKE_DIM, 0);
        assert!(!strokes.is_empty());
        let img = rasterize_strokes(&strokes, strokes.len() / STROKE_DIM);
        assert_eq!(img.len(), IMAGE_FLOATS);
        assert!(img.iter().any(|&v| v > 0.0));
        assert!(img.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }
}
