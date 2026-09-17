//! Training-time diagnostics: a cheap "what is this model actually doing?"
//! snapshot, taken once per epoch on a small probe batch.
//!
//! Scalar loss and CER are blunt. CTC in particular has a well-documented
//! "predict blank everywhere" basin where loss can keep dropping (the blank
//! path is a valid alignment) while greedy decode stays empty (`CER = 1`)
//! or, worse, starts emitting long garbage (`CER > 1`). Those two failure
//! modes look the same in a CER chart and opposite in a loss chart. The
//! numbers here split them:
//!
//! - `blank_mass` / `argmax_blank_frac`: how much of the softmax sits on
//!   the CTC blank, and how often blank wins the per-timestep argmax.
//! - `max_nonblank`: the strongest non-blank class probability anywhere in
//!   a timestep — can rise while blank still wins argmax, which is the
//!   "it *is* learning character identity, blank just still outscores it"
//!   signal that CER will never show.
//! - Per-layer activation RMS through the HAT stack (stroke transformer, fusion).
//! - Per-group weight / gradient RMS: are the weights actually moving, and
//!   is gradient reaching the early layers or dying at `classifier`?

use std::collections::BTreeMap;
use std::marker::PhantomData;

use burn::module::{Module, ModuleVisitor, Param};
use burn::optim::GradientsParams;
use burn::tensor::activation::softmax;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{ElementConversion, Tensor};
use hwr_ink::ink::Ink;

use crate::{decode, eval, model, spline};

/// One named activation tensor from [`model::Recognizer::forward_traced`].
#[derive(Clone, Debug)]
pub struct LayerAct {
    pub name: String,
    pub mean: f32,
    pub std: f32,
    pub rms: f32,
}

/// Output-side CTC / decode snapshot over the probe batch.
#[derive(Clone, Debug)]
pub struct OutputStats {
    /// Mean softmax mass on the CTC blank class, over real (unpadded) steps.
    pub blank_mass: f32,
    /// Mean categorical entropy of the per-step softmax (nats).
    pub entropy: f32,
    /// Fraction of real timesteps whose argmax is blank.
    pub argmax_blank_frac: f32,
    /// Mean over real timesteps of `max_{c ≠ blank} p[t, c]`.
    pub max_nonblank: f32,
    pub mean_decoded_len: f32,
    pub mean_target_len: f32,
    /// A few `(gold, greedy-decode)` pairs, for the log.
    pub samples: Vec<(String, String)>,
    /// Mean softmax mass per class, sorted descending — first entries are
    /// what the model is actually putting probability on.
    pub class_mass: Vec<(usize, f32)>,
}

#[derive(Clone, Debug)]
pub struct ProbeReport {
    pub layers: Vec<LayerAct>,
    pub weights: Vec<(String, f32)>,
    pub output: OutputStats,
}

impl ProbeReport {
    /// Parseable, human-readable log lines (leading two spaces so they
    /// don't match the `epoch N:` regex the dashboard already uses).
    pub fn log_lines(&self) -> Vec<String> {
        let o = &self.output;
        let mut lines = vec![format!(
            "  probe: blank={:.4} entropy={:.3} argmax_blank={:.3} max_nb={:.4} dec_len={:.2} tgt_len={:.2}",
            o.blank_mass, o.entropy, o.argmax_blank_frac, o.max_nonblank, o.mean_decoded_len, o.mean_target_len
        )];
        if !self.layers.is_empty() {
            lines.push(format!(
                "  layers: {}",
                join_pairs(self.layers.iter().map(|l| (l.name.as_str(), l.rms)))
            ));
        }
        if !self.weights.is_empty() {
            lines.push(format!(
                "  weights: {}",
                join_pairs(self.weights.iter().map(|(n, v)| (n.as_str(), *v)))
            ));
        }
        if !o.class_mass.is_empty() {
            let top = o.class_mass.iter().take(6);
            lines.push(format!(
                "  classes: {}",
                join_pairs(top.map(|(i, v)| (class_label(*i), *v)))
            ));
        }
        if !o.samples.is_empty() {
            let bits: Vec<String> = o
                .samples
                .iter()
                .map(|(g, p)| format!("{} -> {}", quote(g), quote(p)))
                .collect();
            lines.push(format!("  samples: {}", bits.join(" | ")));
        }
        lines
    }
}

/// Run a probe forward on up to [`eval::SAFE_BATCH`] samples and collect
/// layer / output / weight diagnostics. Pads the batch to `SAFE_BATCH`
/// (repeating the last row) for the same CubeCL autotune reason as
/// [`eval::mean_cer`].
pub fn probe<B: Backend>(
    net: &model::Recognizer<B>,
    pairs: &[(String, Ink)],
    device: &B::Device,
) -> Option<ProbeReport> {
    let prepared = prepare_probe_samples(pairs);
    if prepared.is_empty() {
        return None;
    }
    let n_real = prepared.len();
    let max_steps = prepared.iter().map(|s| s.steps).max().unwrap_or(0);
    if max_steps == 0 {
        return None;
    }

    let batch = eval::SAFE_BATCH;
    let rows = prepared.iter().map(|s| (s.encoded.as_slice(), s.steps));
    let (strokes, images, pad) = spline::pack_hat_batch(rows, n_real, batch, max_steps);
    let (strokes, images, pad_mask) =
        model::packed_inputs(strokes, images, pad, batch, max_steps, device);
    let (logits, traces) = net.forward_traced(strokes, images, Some(pad_mask));
    let layers: Vec<LayerAct> = traces
        .into_iter()
        .map(|(name, t)| {
            let (mean, std, rms) = mean_std_rms(t);
            LayerAct {
                name,
                mean,
                std,
                rms,
            }
        })
        .collect();

    let probs = softmax(logits, 2);
    let Ok(flat) = probs.into_data().to_vec::<f32>() else {
        return None;
    };
    let classes = decode::classes();
    let per_row = max_steps * classes;
    let mut real_flat = Vec::with_capacity(n_real * per_row);
    let mut texts = Vec::with_capacity(n_real);
    let mut real_steps = Vec::with_capacity(n_real);
    for (i, sample) in prepared.iter().enumerate() {
        let src = i * per_row;
        real_flat.extend_from_slice(&flat[src..src + per_row]);
        texts.push(sample.text.clone());
        real_steps.push(sample.steps);
    }
    let output = summarize_probs(&real_flat, classes, max_steps, &real_steps, &texts);
    Some(ProbeReport {
        layers,
        weights: grouped_weight_rms(net),
        output,
    })
}

/// Per-parameter-group RMS of `grads`, grouped the same way as
/// [`grouped_weight_rms`] (`lstm0`..`lstmN`, `dense`, optional `tcn*`).
/// Intended for the last successful batch of an epoch — pulling every
/// gradient tensor back to the CPU every step would dominate epoch time.
///
/// `GradientsParams::from_grads` registers each tensor as `B::InnerBackend`
/// (the Wgpu tensor, not the Autodiff wrapper). Looking them up as `B`
/// finds the id but fails Burn's `TensorContainer` downcast — which is a
/// panic, not `None` — and aborts the epoch before `on_epoch` can log.
pub fn grouped_grad_rms<B: AutodiffBackend, M: Module<B>>(
    module: &M,
    grads: &GradientsParams,
) -> Vec<(String, f32)> {
    let mut visitor = GradRms::<B> {
        path: Vec::new(),
        groups: BTreeMap::new(),
        grads,
        _b: PhantomData,
    };
    module.visit(&mut visitor);
    finish_groups(visitor.groups)
}

pub fn grouped_weight_rms<B: Backend, M: Module<B>>(module: &M) -> Vec<(String, f32)> {
    let mut visitor = GroupRms {
        path: Vec::new(),
        groups: BTreeMap::new(),
        _b: PhantomData,
    };
    module.visit(&mut visitor);
    finish_groups(visitor.groups)
}

struct ProbeSample {
    text: String,
    encoded: Vec<f32>,
    steps: usize,
}

fn prepare_probe_samples(pairs: &[(String, Ink)]) -> Vec<ProbeSample> {
    let mut out = Vec::new();
    for (text, ink) in pairs {
        if out.len() >= eval::SAFE_BATCH {
            break;
        }
        let encoded = spline::encode_strokes(ink);
        let steps = encoded.len() / spline::STROKE_DIM;
        if steps == 0 {
            continue;
        }
        out.push(ProbeSample {
            text: text.clone(),
            encoded,
            steps,
        });
    }
    out
}

fn mean_std_rms<B: Backend, const D: usize>(t: Tensor<B, D>) -> (f32, f32, f32) {
    let mean: f32 = t.clone().mean().into_scalar().elem();
    let mean_sq: f32 = t.powf_scalar(2.0).mean().into_scalar().elem();
    let var = (mean_sq - mean * mean).max(0.0);
    (mean, var.sqrt(), mean_sq.max(0.0).sqrt())
}

/// CPU-side summary of a `[n, max_steps, classes]` softmax buffer. Only
/// the first `real_steps[i]` timesteps of row `i` count (padding is
/// ignored), which is what makes `blank_mass` comparable across epochs
/// even if the probe batch's max length moves around.
pub(crate) fn summarize_probs(
    flat: &[f32],
    classes: usize,
    max_steps: usize,
    real_steps: &[usize],
    texts: &[String],
) -> OutputStats {
    let n = real_steps.len();
    let blank = classes.saturating_sub(1);
    let mut blank_mass = 0.0f64;
    let mut entropy = 0.0f64;
    let mut argmax_blank = 0.0f64;
    let mut max_nonblank = 0.0f64;
    let mut n_steps = 0.0f64;
    let mut class_mass = vec![0.0f64; classes];
    let mut decoded_len = 0.0f64;
    let mut target_len = 0.0f64;
    let mut samples = Vec::new();

    for (i, &steps) in real_steps.iter().enumerate() {
        let row = i * max_steps * classes;
        let usable = steps.min(max_steps);
        // Decode only real timesteps: the padded tail is zeros, whose
        // argmax is class 0 (space), which would inflate decoded length.
        let decoded = decode::greedy_decode(&flat[row..row + usable * classes]);
        decoded_len += decoded.chars().count() as f64;
        let text = texts.get(i).map(String::as_str).unwrap_or("");
        target_len += text.chars().count() as f64;
        if samples.len() < 6 {
            samples.push((text.to_string(), decoded));
        }
        for t in 0..usable {
            let off = row + t * classes;
            let step = &flat[off..off + classes];
            let mut best_i = 0usize;
            let mut best_p = f32::NEG_INFINITY;
            let mut best_nb = 0.0f32;
            let mut h = 0.0f64;
            for (c, &p) in step.iter().enumerate() {
                class_mass[c] += p as f64;
                if p > best_p {
                    best_p = p;
                    best_i = c;
                }
                if c != blank && p > best_nb {
                    best_nb = p;
                }
                if p > 0.0 {
                    h -= (p as f64) * (p as f64).ln();
                }
            }
            blank_mass += step[blank] as f64;
            entropy += h;
            max_nonblank += best_nb as f64;
            if best_i == blank {
                argmax_blank += 1.0;
            }
            n_steps += 1.0;
        }
    }

    let inv_steps = if n_steps > 0.0 { 1.0 / n_steps } else { 0.0 };
    let inv_n = if n > 0 { 1.0 / n as f64 } else { 0.0 };
    let mut class_mass: Vec<(usize, f32)> = class_mass
        .into_iter()
        .enumerate()
        .map(|(i, m)| (i, (m * inv_steps) as f32))
        .collect();
    class_mass.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    OutputStats {
        blank_mass: (blank_mass * inv_steps) as f32,
        entropy: (entropy * inv_steps) as f32,
        argmax_blank_frac: (argmax_blank * inv_steps) as f32,
        max_nonblank: (max_nonblank * inv_steps) as f32,
        mean_decoded_len: (decoded_len * inv_n) as f32,
        mean_target_len: (target_len * inv_n) as f32,
        samples,
        class_mass,
    }
}

struct GroupRms<B: Backend> {
    path: Vec<String>,
    groups: BTreeMap<String, (f64, usize)>,
    _b: PhantomData<B>,
}

impl<B: Backend> ModuleVisitor<B> for GroupRms<B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(group) = group_name(&self.path) else {
            return;
        };
        accumulate_rms(&mut self.groups, group, param.val());
    }
}

struct GradRms<'a, B: AutodiffBackend> {
    path: Vec<String>,
    groups: BTreeMap<String, (f64, usize)>,
    grads: &'a GradientsParams,
    _b: PhantomData<B>,
}

impl<B: AutodiffBackend> ModuleVisitor<B> for GradRms<'_, B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(group) = group_name(&self.path) else {
            return;
        };
        let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) else {
            return;
        };
        accumulate_rms(&mut self.groups, group, g);
    }
}

fn accumulate_rms<B: Backend, const D: usize>(
    groups: &mut BTreeMap<String, (f64, usize)>,
    group: String,
    t: Tensor<B, D>,
) {
    let n = t.shape().num_elements();
    let ss: f32 = t.powf_scalar(2.0).sum().into_scalar().elem();
    let entry = groups.entry(group).or_insert((0.0, 0));
    entry.0 += ss as f64;
    entry.1 += n;
}

fn finish_groups(groups: BTreeMap<String, (f64, usize)>) -> Vec<(String, f32)> {
    let mut out: Vec<(String, f32)> = groups
        .into_iter()
        .filter(|(_, (_, n))| *n > 0)
        .map(|(name, (ss, n))| (name, (ss / n as f64).sqrt() as f32))
        .collect();
    out.sort_by_key(|(name, _)| group_sort_key(name));
    out
}

fn group_name(path: &[String]) -> Option<String> {
    match path.first().map(String::as_str) {
        Some("img_convs") => Some(format!(
            "img{}",
            path.get(1).map(|s| s.as_str()).unwrap_or("0")
        )),
        Some("stroke_encoder") => Some("stroke_tf".to_string()),
        Some("pen_embed") | Some("stroke_proj") | Some("stroke_bn") => Some("stroke_in".to_string()),
        Some("cross_mha") | Some("cross_tf") => Some("fusion".to_string()),
        Some("latent_mha") | Some("latent_tf") | Some("latents") => Some("latents".to_string()),
        Some("classifier") => Some("classifier".to_string()),
        _ => None,
    }
}

fn group_sort_key(name: &str) -> (u8, u8) {
    match name {
        n if n.starts_with("img") => (0, n.trim_start_matches("img").parse().unwrap_or(0)),
        "stroke_in" => (1, 0),
        "stroke_tf" => (1, 1),
        "latents" => (2, 0),
        "fusion" => (3, 0),
        "classifier" => (4, 0),
        _ => (5, 0),
    }
}

fn join_pairs<S: AsRef<str>>(pairs: impl Iterator<Item = (S, f32)>) -> String {
    pairs
        .map(|(k, v)| format!("{}={:.4}", k.as_ref(), v))
        .collect::<Vec<_>>()
        .join(" ")
}

fn class_label(i: usize) -> String {
    if i >= decode::CHARS.chars().count() {
        return "blank".to_string();
    }
    match decode::CHARS.chars().nth(i) {
        Some(' ') => "space".to_string(),
        Some(c) => c.to_string(),
        None => format!("c{i}"),
    }
}

fn quote(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
    )
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
    fn all_blank_is_empty_decode_and_unit_blank_mass() {
        let classes = decode::classes();
        let blank = classes - 1;
        let picks = vec![blank; 4];
        let flat = one_hot_row(classes, 4, &picks);
        let stats = summarize_probs(&flat, classes, 4, &[4], &["5".to_string()]);
        assert!((stats.blank_mass - 1.0).abs() < 1e-5);
        assert!((stats.argmax_blank_frac - 1.0).abs() < 1e-5);
        assert_eq!(stats.samples[0].1, "");
        assert_eq!(stats.mean_decoded_len, 0.0);
        assert_eq!(stats.mean_target_len, 1.0);
    }

    #[test]
    fn peaked_digit_decodes_and_blank_mass_drops() {
        let classes = decode::classes();
        let five = decode::CHARS.chars().position(|c| c == '5').unwrap();
        let blank = classes - 1;
        // CTC collapse-break: blank, '5', blank, blank -> greedy "5".
        let picks = [blank, five, blank, blank];
        let flat = one_hot_row(classes, 4, &picks);
        let stats = summarize_probs(&flat, classes, 4, &[4], &["5".to_string()]);
        assert_eq!(stats.samples[0].1, "5");
        assert!((stats.blank_mass - 0.75).abs() < 1e-5);
        assert!((stats.argmax_blank_frac - 0.75).abs() < 1e-5);
        assert!(stats.class_mass[0].0 == blank || stats.class_mass[1].0 == five);
    }

    #[test]
    fn padding_timesteps_are_ignored() {
        let classes = decode::classes();
        let blank = classes - 1;
        let five = decode::CHARS.chars().position(|c| c == '5').unwrap();
        // max_steps=4 but only 2 real steps, both blank; padded tail is '5'
        // (would pollute blank_mass if we forgot to mask).
        let mut flat = one_hot_row(classes, 4, &[blank, blank, five, five]);
        // make the padded tail look like a confident non-blank so a bug
        // that counts it would move blank_mass well below 1.0.
        for t in 2..4 {
            let off = t * classes;
            for c in 0..classes {
                flat[off + c] = 0.0;
            }
            flat[off + five] = 1.0;
        }
        let stats = summarize_probs(&flat, classes, 4, &[2], &["x".to_string()]);
        assert!((stats.blank_mass - 1.0).abs() < 1e-5);
        assert_eq!(stats.samples[0].1, "");
    }
}
