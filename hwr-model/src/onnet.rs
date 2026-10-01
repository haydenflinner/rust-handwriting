//! ONNET (chunkyjasper/IAMhwr `pretrained-lstm`) — a TDNN + BiLSTM + CTC
//! recognizer imported from ONNX via `burn-onnx` (see ../build.rs).
//!
//! Base weights come from the repo's git history: the shipped
//! `pretrained-deep-lstm` checkpoint decodes to garbage under the current
//! preprocessing, but `pretrained-lstm` (deleted in the "swap pretrained
//! models" commit) verifies at ~7% CER greedy on the repo's bundled
//! IAM-OnDB t2 samples. scripts/onnet_torch.py rebuilds it in PyTorch,
//! scripts/export_onnx.py exports ONNX, and scripts/finetune_onnet.py
//! fine-tunes on the app's calibration corpus + synthesized stroke data
//! (Hershey fonts, glyph composition) with a widened 98-class head — that
//! fine-tuned model is onnet_lstm_finetuned.onnx next to this file.
//!
//! Input features are per-segment `[x, y, dx, dy, down, up]` after the
//! repo's SCHEME6 preprocessing, ported below from hwr/data/datarep.py.

use burn::backend::wgpu::WgpuDevice;
use burn::tensor::{Bytes, Tensor, TensorData};
use hwr_ink::ink::Ink;

use crate::Backend;

pub mod model {
    include!(concat!(env!("OUT_DIR"), "/onnet/onnet_lstm_finetuned.rs"));
}

/// Weights exported by burn-onnx's ModelGen at build time; embedding them
/// keeps the app free of runtime file dependencies (same trick as
/// `Recognizer::from_bytes`).
const WEIGHTS: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/onnet/onnet_lstm_finetuned.bpk"));

/// Output alphabet: IAM-OnDB MLF codes mapped to ASCII, extended by
/// scripts/finetune_onnet.py with 15 symbols the original vocabulary could
/// not spell (indices 82..96). CTC blank = 97. `''` entries are annotation
/// artifacts that decode to nothing.
pub const CHARS: &[&str] = &[
    "!", "\"", "", "", "&", "'", "/", "(", ")", "[", "]", "*", ",", "-", "+", ".", " ", ":", ";",
    "?", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "a",
    "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z",
    "=", "<", ">", "{", "}", "@", "#", "$", "%", "^", "_", "`", "|", "~", "\\",
];

pub const NUM_CLASSES: usize = CHARS.len() + 1;
pub const BLANK: usize = CHARS.len();
pub const FEATURE_DIM: usize = 6;

/// CTC best-path decode over per-step logits/probs `[steps][NUM_CLASSES]`.
pub fn greedy_decode(frame: &[f32]) -> String {
    let mut out = String::new();
    let mut last = usize::MAX;
    for step in frame.chunks(NUM_CLASSES) {
        let argmax = step
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(BLANK);
        if argmax != last && argmax != BLANK && argmax < CHARS.len() {
            out.push_str(CHARS[argmax]);
        }
        last = argmax;
    }
    out
}

// ---------------------------------------------------------------------------
// SCHEME6 preprocessing (hwr/data/datarep.py): slope correction, mean/sd_y
// normalization, cosine downsample, distance resample, short-stroke upsample.

const DOWN_COS: f64 = 0.975;
const RESAMPLE_D: f64 = 0.37;
const UP_SAMPLE: usize = 9;
const ADD_PAD: usize = 10;

#[derive(Clone, Copy, Debug)]
struct Pt {
    x: f64,
    y: f64,
}

type Stroke = Vec<Pt>;

fn len(a: Pt, b: Pt) -> f64 {
    ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt()
}

fn lerp(a: Pt, b: Pt, t: f64) -> Pt {
    Pt {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
    }
}

fn strokes_of(ink: &Ink) -> Vec<Stroke> {
    // strokes_with_open: include a trailing in-progress stroke — submitted
    // ink may not be pen_up'd yet.
    ink.strokes_with_open()
        .map(|s| {
            s.iter()
                .map(|p| Pt {
                    x: p.x as f64,
                    y: p.y as f64,
                })
                .collect()
        })
        .collect()
}

fn all_points(strokes: &[Stroke]) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for s in strokes {
        for p in s {
            x.push(p.x);
            y.push(p.y);
        }
    }
    (x, y)
}

/// Least-squares line fit (deg 1), then rotate all points by atan(slope)
/// about the origin — matches the current datarep.py (the pivot translation
/// is commented out there; normalization absorbs the offset anyway).
fn slope_correction(strokes: &mut [Stroke]) {
    let (x, y) = all_points(strokes);
    if x.len() < 2 {
        return;
    }
    let n = x.len() as f64;
    let (sx, sy) = (x.iter().sum::<f64>(), y.iter().sum::<f64>());
    let sxx = x.iter().map(|v| v * v).sum::<f64>();
    let sxy = x.iter().zip(&y).map(|(a, b)| a * b).sum::<f64>();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-9 {
        return;
    }
    let slope = (n * sxy - sx * sy) / denom;
    let rad = slope.atan();
    let (c, s) = (rad.cos(), rad.sin());
    // datarep.py applies the rotation as a row-vector matmul (co @ R), i.e.
    // x' = x*cos + y*sin, y' = -x*sin + y*cos — rotation by -rad.
    for stroke in strokes.iter_mut() {
        for p in stroke.iter_mut() {
            let (px, py) = (p.x, p.y);
            p.x = px * c + py * s;
            p.y = -px * s + py * c;
        }
    }
}

/// Length-weighted mean over segment midpoints; divide x and y by sd_y.
fn normalize(strokes: &mut [Stroke]) {
    let mut sum_px = 0.0;
    let mut sum_py = 0.0;
    let mut sum_l = 0.0;
    for s in strokes.iter() {
        for w in s.windows(2) {
            let l = len(w[0], w[1]);
            sum_l += l;
            sum_px += l * (w[0].x + w[1].x) / 2.0;
            sum_py += l * (w[0].y + w[1].y) / 2.0;
        }
    }
    if sum_l <= 0.0 {
        return;
    }
    let (mx, my) = (sum_px / sum_l, sum_py / sum_l);
    let mut var = 0.0;
    for s in strokes.iter() {
        for w in s.windows(2) {
            let l = len(w[0], w[1]);
            var += l / 3.0
                * ((w[1].y - my).powi(2) + (w[0].y - my).powi(2) + (w[0].y - my) * (w[1].y - my));
        }
    }
    let sd_y = (var / sum_l).sqrt();
    if sd_y <= 0.0 {
        return;
    }
    for s in strokes.iter_mut() {
        for p in s.iter_mut() {
            p.x = (p.x - mx) / sd_y;
            p.y = (p.y - my) / sd_y;
        }
    }
}

/// Drop points whose direction barely changes (cos between the
/// last-kept->current and current->next segments >= threshold).
fn down_sample_angle(strokes: &mut Vec<Stroke>, cos_th: f64) {
    for s in strokes.iter_mut() {
        let mut ret: Stroke = Vec::with_capacity(s.len());
        let mut removed = 0usize;
        for i in 0..s.len() {
            if i == 0 || i == s.len() - 1 {
                removed = 0;
                ret.push(s[i]);
            } else {
                let a = (s[i].x - s[i - 1 - removed].x, s[i].y - s[i - 1 - removed].y);
                let b = (s[i + 1].x - s[i].x, s[i + 1].y - s[i].y);
                let la = (a.0 * a.0 + a.1 * a.1).sqrt();
                let lb = (b.0 * b.0 + b.1 * b.1).sqrt();
                let cs = if la * lb < 1e-5 {
                    f64::INFINITY
                } else {
                    (a.0 * b.0 + a.1 * b.1) / (la * lb)
                };
                if cs < cos_th {
                    removed = 0;
                    ret.push(s[i]);
                } else {
                    removed += 1;
                }
            }
        }
        *s = ret;
    }
}

/// Resample each stroke so consecutive points are ~`d` apart (the original
/// walks the polyline and emits points at multiples of `d`, then the final
/// point — possibly duplicating it).
fn resample_distance(strokes: &mut Vec<Stroke>, d: f64) {
    for s in strokes.iter_mut() {
        let mut ret: Stroke = Vec::new();
        if !s.is_empty() {
            ret.push(s[0]);
        }
        for i in 1..s.len() {
            let last = *ret.last().unwrap();
            let l = len(last, s[i]);
            if l > d {
                let f = d / l;
                for j in 1..=(l / d) as usize {
                    ret.push(lerp(last, s[i], f * j as f64));
                }
            } else if l == d {
                ret.push(s[i]);
            }
        }
        if let Some(&last) = s.last() {
            ret.push(last);
        }
        *s = ret;
    }
}

/// Strokes shorter than `n` points get resampled up to `n` (single-point
/// strokes repeat the point).
fn up_sample_short_stroke(strokes: &mut Vec<Stroke>, n: usize) {
    for s in strokes.iter_mut() {
        if s.len() < n {
            if s.len() == 1 {
                *s = vec![s[0]; n];
            } else if !s.is_empty() {
                let total: f64 = s.windows(2).map(|w| len(w[0], w[1])).sum();
                let mut tmp = vec![s.clone()];
                resample_distance(&mut tmp, total / n as f64);
                *s = tmp.pop().unwrap();
            }
        }
    }
}

/// Ink -> `[steps][FEATURE_DIM]` features (plus ADD_PAD zero rows), the exact
/// input layout the ONNX model was trained on.
pub fn extract_features(ink: &Ink) -> Vec<f32> {
    let mut strokes = strokes_of(ink);
    slope_correction(&mut strokes);
    normalize(&mut strokes);
    down_sample_angle(&mut strokes, DOWN_COS);
    resample_distance(&mut strokes, RESAMPLE_D);
    up_sample_short_stroke(&mut strokes, UP_SAMPLE);

    let mut rows = 0usize;
    for s in &strokes {
        rows += s.len().saturating_sub(1);
    }
    let mut out = vec![0.0f32; (rows + ADD_PAD) * FEATURE_DIM];
    let mut i = 0usize;
    for s in &strokes {
        for j in 0..s.len().saturating_sub(1) {
            let last = j == s.len() - 2;
            out[i * FEATURE_DIM] = s[j].x as f32;
            out[i * FEATURE_DIM + 1] = s[j].y as f32;
            out[i * FEATURE_DIM + 2] = (s[j + 1].x - s[j].x) as f32;
            out[i * FEATURE_DIM + 3] = (s[j + 1].y - s[j].y) as f32;
            out[i * FEATURE_DIM + 4] = if last { 0.0 } else { 1.0 };
            out[i * FEATURE_DIM + 5] = if last { 1.0 } else { 0.0 };
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Ink -> features -> imported ONNX model -> greedy CTC decode.

/// The pretrained IAMhwr ONNET line recognizer (TDNN + 2×BiLSTM + CTC).
pub struct Onnet {
    model: model::Model<Backend>,
    device: WgpuDevice,
}

impl Onnet {
    pub fn new() -> Self {
        let device = WgpuDevice::default();
        let model = model::Model::from_bytes(Bytes::from_bytes_vec(WEIGHTS.to_vec()), &device);
        Onnet { model, device }
    }

    /// Raw per-step logits `[steps][NUM_CLASSES]` (~4x downsampled from the
    /// input feature rows by the two AvgPool layers).
    pub fn logits(&self, ink: &Ink) -> Vec<f32> {
        self.forward(&extract_features(ink))
    }

    /// Run the model on a pre-extracted feature matrix (`steps × 6`,
    /// row-major) — split out so tests can feed reference features
    /// directly, isolating model parity from preprocessing parity.
    pub fn forward(&self, feats: &[f32]) -> Vec<f32> {
        let steps = feats.len() / FEATURE_DIM;
        if steps == 0 {
            return Vec::new();
        }
        let input = Tensor::<Backend, 3>::from_data(
            TensorData::new(feats.to_vec(), [1, steps, FEATURE_DIM]),
            &self.device,
        );
        self.model
            .forward(input)
            .into_data()
            .to_vec()
            .expect("f32 tensor data should convert to Vec<f32>")
    }

    pub fn recognize(&self, ink: &Ink) -> String {
        greedy_decode(&self.logits(ink))
    }
}

impl Default for Onnet {
    fn default() -> Self {
        Self::new()
    }
}


