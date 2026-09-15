//! Data augmentation: perturb a real stroke recording (rotation, scale,
//! shear, smooth spatial jitter, time-warp) to synthesize variants of ink
//! the writer only had to produce once. This is deliberately different from
//! the Hershey-font synthetic corpus explored earlier: it starts from real
//! pen dynamics (velocity/timing captured by an actual hand) rather than a
//! traced vector font, so the augmented samples should carry more realistic
//! signal — at the cost of only varying the *shape* of a word already
//! written, not producing new vocabulary.

use hwr_ink::ink::Ink;
use rand::{Rng, RngExt};

pub struct AugmentConfig {
    /// Max absolute rotation, radians.
    pub max_rotation: f32,
    /// Per-axis scale factor range, applied independently to x and y.
    pub scale_range: (f32, f32),
    /// Max absolute horizontal shear (slant) factor.
    pub max_shear: f32,
    /// Smooth jitter amplitude, as a fraction of the ink's height.
    pub jitter_amplitude: f32,
    /// Roughly how many jitter oscillations span the ink's height.
    pub jitter_frequency: f32,
    /// Max fractional speed-up/slow-down applied to the whole sample's
    /// timing (e.g. 0.15 means each variant is timed 0.85x-1.15x as fast).
    pub time_warp: f32,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        AugmentConfig {
            max_rotation: 0.12,
            scale_range: (0.85, 1.15),
            max_shear: 0.2,
            jitter_amplitude: 0.025,
            jitter_frequency: 1.2,
            time_warp: 0.15,
        }
    }
}

/// A smooth, low-frequency noise field built from a handful of random
/// sinusoids, evaluated as a function of a single scalar (here, horizontal
/// position along the whole ink) rather than per-point or per-stroke
/// independently. This is the same trick used for the Hershey-font
/// synthetic corpus, ported over after a bug there where per-stroke
/// independent noise tore apart multi-stroke glyphs (e.g. `{`, `}`, `~`,
/// the two strokes of `=`, the dot of `i`): a shared field keyed on
/// position keeps every stroke of one glyph moving together.
struct NoiseField {
    terms: Vec<(f32, f32, f32)>, // (angular freq, phase, amplitude)
}

impl NoiseField {
    fn random(rng: &mut impl Rng, n: usize, base_freq: f32, amplitude: f32) -> Self {
        let terms = (0..n)
            .map(|i| {
                let freq = base_freq * (0.5 + i as f32);
                let phase = rng.random_range(0.0..std::f32::consts::TAU);
                let amp = amplitude * rng.random_range(0.3..1.0) / (i as f32 + 1.0);
                (freq, phase, amp)
            })
            .collect();
        NoiseField { terms }
    }

    fn eval(&self, t: f32) -> f32 {
        self.terms
            .iter()
            .map(|&(freq, phase, amp)| amp * (freq * t + phase).sin())
            .sum()
    }
}

/// Produce one randomly-augmented variant of `ink`. The text label is
/// unchanged by any of these transforms (they only perturb geometry/timing),
/// so callers just keep pairing the result with the original text.
pub fn augment(ink: &Ink, config: &AugmentConfig, rng: &mut impl Rng) -> Ink {
    if ink.is_empty() {
        return ink.clone();
    }

    let height = (ink.y_range.max - ink.y_range.min).max(1e-3);
    let cx = (ink.x_range.min + ink.x_range.max) * 0.5;
    let cy = (ink.y_range.min + ink.y_range.max) * 0.5;

    let angle = rng.random_range(-config.max_rotation..=config.max_rotation);
    let (sin_a, cos_a) = angle.sin_cos();
    let sx = rng.random_range(config.scale_range.0..=config.scale_range.1);
    let sy = rng.random_range(config.scale_range.0..=config.scale_range.1);
    let shear = rng.random_range(-config.max_shear..=config.max_shear);
    let time_scale = 1.0 + rng.random_range(-config.time_warp..=config.time_warp);

    // Base frequency in "cycles per unit x", scaled so `jitter_frequency`
    // oscillations span the ink's height (our rough unit of glyph size).
    let base_freq = std::f32::consts::TAU * config.jitter_frequency / height;
    let x_noise = NoiseField::random(rng, 3, base_freq, config.jitter_amplitude * height);
    let y_noise = NoiseField::random(rng, 3, base_freq, config.jitter_amplitude * height);

    let t0 = ink.t_range.min;
    let mut out = Ink::new();
    for stroke in ink.strokes() {
        for p in stroke {
            // Field position keyed on the *original* x, so it's identical
            // across every stroke of a multi-stroke glyph regardless of the
            // transform applied afterward.
            let field_pos = p.x - ink.x_range.min;

            let mut x = p.x - cx;
            let mut y = p.y - cy;

            x += shear * y; // slant
            x *= sx;
            y *= sy;

            let (rx, ry) = (x * cos_a - y * sin_a, x * sin_a + y * cos_a);
            x = rx + cx + x_noise.eval(field_pos);
            y = ry + cy + y_noise.eval(field_pos);

            let t = t0 + (p.z - t0) * time_scale;

            out.push(x, y, t);
        }
        out.pen_up();
    }
    out
}

/// Generate `count` augmented variants of `(text, ink)`.
pub fn augment_n(
    text: &str,
    ink: &Ink,
    count: usize,
    config: &AugmentConfig,
    rng: &mut impl Rng,
) -> Vec<(String, Ink)> {
    (0..count)
        .map(|_| (text.to_string(), augment(ink, config, rng)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn augment_preserves_stroke_and_point_counts() {
        let mut ink = Ink::new();
        ink.push(0.0, 0.0, 0.0);
        ink.push(1.0, 1.0, 0.1);
        ink.push(2.0, 0.5, 0.2);
        ink.pen_up();
        ink.push(1.0, 2.0, 0.3);
        ink.push(1.0, 3.0, 0.4);
        ink.pen_up();

        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let out = augment(&ink, &AugmentConfig::default(), &mut rng);

        assert_eq!(out.len(), ink.len());
        assert_eq!(out.strokes().count(), ink.strokes().count());
    }

    #[test]
    fn augment_of_empty_ink_is_empty() {
        let ink = Ink::new();
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let out = augment(&ink, &AugmentConfig::default(), &mut rng);
        assert!(out.is_empty());
    }
}
