//! A $1/$P-style nearest-neighbor template matcher: no gradient-based
//! training, no alignment learning, no loss landscape to get stuck in — just
//! geometric normalization (resample, center, scale) plus a distance metric
//! against a library of previously-recorded `(text, ink)` examples. Meant as
//! a fast-to-build fallback for isolated single-token input (one word/symbol
//! per box, which is what the calibration UI already records), running
//! alongside — not instead of — the CTC model, which is still what you'd
//! want for free-form multi-word handwriting this approach can't segment.
//!
//! Deliberately NOT rotation-invariant like classic $1: text characters
//! aren't drawn at arbitrary angles the way freeform gestures are, and
//! rotation-normalizing would actively hurt by confusing rotationally
//! related glyphs (6/9, n/u, p/d). Also preserves aspect ratio when scaling
//! (uniform scale factor, not squash-to-square) for the same reason — a
//! tall 'l' and a wide '-' shouldn't become geometrically similar.

use hwr_ink::ink::Ink;

/// Points per template after resampling — enough to capture shape, cheap to
/// compare. $1 itself uses 64.
const RESAMPLE_POINTS: usize = 64;

/// A single normalized template: `RESAMPLE_POINTS` points, centered at the
/// origin, uniformly scaled so the larger bounding-box dimension is 1.0.
#[derive(Clone)]
pub struct Template {
    pub text: String,
    points: Vec<(f32, f32)>,
}

fn normalize(ink: &Ink) -> Vec<(f32, f32)> {
    if ink.is_empty() {
        return vec![(0.0, 0.0); RESAMPLE_POINTS];
    }

    // Reuse `Ink::resample`'s already-tested arc-length walk (per stroke,
    // pen-up transitions preserved) rather than reimplementing it — pick a
    // distance step from the target point count, then pad/truncate to
    // exactly `RESAMPLE_POINTS` (per-stroke rounding means the exact count
    // varies slightly for multi-stroke ink).
    let total_len = ink.ink_len().max(1e-6);
    let step = total_len / (RESAMPLE_POINTS.saturating_sub(1)).max(1) as f32;
    let resampled_ink = ink.resample(step.max(1e-6));
    let mut resampled: Vec<(f32, f32)> =
        resampled_ink.points().iter().map(|p| (p.x, p.y)).collect();

    if resampled.is_empty() {
        resampled.push((0.0, 0.0));
    }
    while resampled.len() < RESAMPLE_POINTS {
        resampled.push(*resampled.last().unwrap());
    }
    resampled.truncate(RESAMPLE_POINTS);

    // Center at centroid.
    let (sx, sy) = resampled
        .iter()
        .fold((0.0, 0.0), |(ax, ay), (x, y)| (ax + x, ay + y));
    let n = resampled.len() as f32;
    let (cx, cy) = (sx / n, sy / n);
    for p in &mut resampled {
        p.0 -= cx;
        p.1 -= cy;
    }

    // Uniform scale so the larger bounding-box dimension is 1.0 (aspect
    // ratio preserved — see module docs for why this differs from $1).
    let (mut max_x, mut max_y) = (0.0f32, 0.0f32);
    for &(x, y) in &resampled {
        max_x = max_x.max(x.abs());
        max_y = max_y.max(y.abs());
    }
    let scale = max_x.max(max_y).max(1e-6);
    for p in &mut resampled {
        p.0 /= scale;
        p.1 /= scale;
    }

    resampled
}

impl Template {
    pub fn from_ink(text: &str, ink: &Ink) -> Self {
        Template {
            text: text.to_string(),
            points: normalize(ink),
        }
    }
}

/// Average per-point Euclidean distance between two same-length normalized
/// point sequences — $1's core "path distance" metric. Rigid 1:1 index
/// correspondence: point `i` of `a` only ever compares against point `i` of
/// `b`, so two recordings of the same character written at different local
/// speeds (e.g. a slower curve on one stroke, faster on another) can end up
/// comparing geometrically mismatched points even though the overall shapes
/// agree — see [`dtw_distance`], which is what `TemplateLibrary` actually
/// uses, for the fix.
#[allow(dead_code)]
fn path_distance(a: &[(f32, f32)], b: &[(f32, f32)]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let sum: f32 = a
        .iter()
        .zip(b)
        .map(|(&(ax, ay), &(bx, by))| ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt())
        .sum();
    sum / a.len() as f32
}

/// Dynamic time warping distance: like [`path_distance`], but instead of a
/// rigid index-`i`-to-index-`i` correspondence, finds the lowest-cost
/// monotonic alignment path between the two point sequences (each point may
/// match against one or more consecutive points of the other sequence).
/// This is the standard fix for exactly the failure mode `path_distance`
/// has — two recordings of the same shape written at different local
/// speeds — and is cheap here (a 64x64 point DP table, ~4k cells).
/// Returned distance is normalized by the warping path's length, so it's
/// comparable in magnitude to `path_distance`'s average-per-point value
/// regardless of how much warping the optimal path used.
fn dtw_distance(a: &[(f32, f32)], b: &[(f32, f32)]) -> f32 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return f32::INFINITY;
    }

    let dist = |p: (f32, f32), q: (f32, f32)| ((p.0 - q.0).powi(2) + (p.1 - q.1).powi(2)).sqrt();

    // dp[i][j] = (cheapest cumulative cost to align a[..i] with b[..j],
    // number of cells on that path) — track path length alongside cost so
    // the final distance can be normalized, keeping it on the same scale as
    // `path_distance`'s per-point average regardless of sequence length or
    // how much warping occurred.
    let mut dp = vec![vec![(f32::INFINITY, 0u32); m + 1]; n + 1];
    dp[0][0] = (0.0, 0);
    for i in 1..=n {
        for j in 1..=m {
            let cost = dist(a[i - 1], b[j - 1]);
            let (best_prev, best_len) = [dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]]
                .into_iter()
                .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .unwrap();
            dp[i][j] = (best_prev + cost, best_len + 1);
        }
    }

    let (total_cost, path_len) = dp[n][m];
    total_cost / path_len.max(1) as f32
}

pub struct TemplateLibrary {
    templates: Vec<Template>,
}

impl TemplateLibrary {
    pub fn new(pairs: &[(String, Ink)]) -> Self {
        TemplateLibrary {
            templates: pairs
                .iter()
                .map(|(text, ink)| Template::from_ink(text, ink))
                .collect(),
        }
    }

    /// Nearest-neighbor match: the template with the smallest path distance
    /// to `ink`, plus that distance (useful as a confidence/rejection
    /// signal — large distances mean "doesn't look like anything we've
    /// seen").
    pub fn recognize(&self, ink: &Ink) -> Option<(&str, f32)> {
        let query = normalize(ink);
        self.templates
            .iter()
            .map(|t| (t.text.as_str(), dtw_distance(&query, &t.points)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
}
