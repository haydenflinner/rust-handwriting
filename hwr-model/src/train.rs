//! Training loop, built on Burn's native `CTCLoss` (see `crate` docs for why
//! this replaced the candle port's hand-rolled CTC forward-backward — in
//! short: Burn's CTC dispatches to a per-backend kernel, verified against
//! PyTorch's own output, so it runs GPU-resident with no CPU round-trip).
//!
//! Batched: samples are bucketed by encoded length (minimizes padding
//! waste), then chunked into fixed-size batches; batch *order* is
//! reshuffled each epoch, batch *membership* stays fixed. Padding caveat
//! carried over from the candle version: the bidirectional LSTM's backward
//! direction technically starts its reversed traversal at the padded tail,
//! so its hidden state at real positions has passed through a few padding
//! steps first — length-bucketing keeps this small. Unlike the candle
//! version, though, the *loss* itself is fully padding-aware: Burn's
//! `CTCLoss` takes explicit `input_lengths`/`target_lengths` tensors and
//! only ever computes over each sample's real (unpadded) extent.

use std::path::{Path, PathBuf};

use burn::backend::wgpu::WgpuDevice;
use burn::module::{AutodiffModule, Module};
use burn::nn::loss::{CTCLossConfig, Reduction};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::momentum::MomentumConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer, SgdConfig};
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::activation::log_softmax;
use burn::tensor::{Int, Tensor, TensorData};
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hwr_ink::ink::Ink;

use crate::{decode, model, spline, Backend, TrainBackend};

pub struct TrainConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    /// Global gradient-norm clip (per-parameter-tensor — Burn's own
    /// `GradientClippingConfig::Norm`, not the hand-rolled *combined*-norm
    /// clip the candle version used). armrest's own training pipeline
    /// (`script/training.py`) used 9.0 for the same architecture.
    pub max_grad_norm: f32,
    /// Samples per GPU launch. 16 was the long-standing CubeCL-safe size
    /// (TCN `Conv1d` used to crash at batch=1). No-TCN fused LSTM was
    /// smoke-tested through 256; 128 raises cube count without turning a
    /// short PBT stretch into a handful of giant steps.
    pub batch_size: usize,
    /// AdamW (the default) adapts each parameter's effective step size from
    /// its own gradient history, which converges faster in the common case
    /// but can behave very differently than plain SGD around a sharp local
    /// optimum — CTC's "predict blank everywhere" collapse is exactly that
    /// kind of optimum, and reference recipes (e.g.
    /// `lstm-ctc-ocr/3_phonernn.lua`) that successfully train CTC models
    /// use plain SGD with momentum, not an adaptive optimizer. Worth trying
    /// as a genuinely different optimization dynamic, not just a different
    /// learning rate, when AdamW is stuck.
    pub use_sgd: bool,
    /// One-cycle LR schedule (Smith 2017-style — a well-established,
    /// standard technique, not another ad hoc guess): ramp *up* from a
    /// small safe start to `peak_lr` over `warmup_steps`, then ramp back
    /// *down* to `learning_rate` (the stable long-run value) over
    /// `decay_steps`, then hold flat. The up-ramp specifically exists so a
    /// genuinely aggressive peak — deliberately higher than anything we've
    /// run stably for a long stretch — doesn't get hit at full strength on
    /// step one, which risked instant NaN before the schedule had a chance
    /// to do anything (see `train_batch_step`'s non-finite-loss skip, which
    /// backstops any batch that still goes bad regardless). `None` keeps
    /// the old flat-`learning_rate`-for-the-whole-run behavior.
    pub lr_schedule: Option<LrSchedule>,
    /// Unused by HAT (no TCN). Kept so existing `--no-tcn` CLI / PBT
    /// flags still parse; the field is ignored at model-build time.
    pub tcn_channels_override: Option<Option<usize>>,
}

#[derive(Clone, Copy, Debug)]
pub struct LrSchedule {
    pub warmup_steps: usize,
    pub peak_lr: f64,
    pub decay_steps: usize,
}

/// The LR to use at a given (cumulative, across all epochs) optimizer step.
/// `base_lr` is the schedule's post-decay, long-run-stable value —
/// `TrainConfig::learning_rate`.
fn lr_at_step(step: usize, base_lr: f64, schedule: &Option<LrSchedule>) -> f64 {
    const WARMUP_START_LR: f64 = 1e-3;
    let Some(s) = schedule else {
        return base_lr;
    };
    if step < s.warmup_steps {
        let t = step as f64 / s.warmup_steps.max(1) as f64;
        WARMUP_START_LR + t * (s.peak_lr - WARMUP_START_LR)
    } else if step < s.warmup_steps + s.decay_steps {
        let t = (step - s.warmup_steps) as f64 / s.decay_steps.max(1) as f64;
        s.peak_lr + t * (base_lr - s.peak_lr)
    } else {
        base_lr
    }
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            epochs: 10,
            learning_rate: 1e-3,
            max_grad_norm: 9.0,
            batch_size: 128,
            use_sgd: false,
            lr_schedule: None,
            tcn_channels_override: None,
        }
    }
}

pub struct EpochStats {
    pub epoch: usize,
    pub mean_loss: f64,
    pub samples: usize,
    pub skipped: usize,
    /// The LR in effect at the end of this epoch — mainly so a schedule's
    /// progress is visible in logs/dashboards, not just inferred from loss.
    pub lr: f64,
    /// Per-group gradient RMS from the last successful batch of this epoch
    /// (`lstm0`..`lstmN`, `dense`, optional `tcn*`). Empty if every batch
    /// was skipped. See `crate::probe` for the grouping.
    pub grad_rms: Vec<(String, f32)>,
}

/// A pre-encoded, pre-validated training example: spline-encoded ink plus
/// CTC-feasible labels (`steps >= 2*labels.len()+1`). Encoding ink is not
/// free (smoothing, simplification), so we do it once up front rather than
/// every epoch.
struct Sample {
    labels: Vec<usize>,
    encoded: Vec<f32>, // flat [steps * spline::STROKE_DIM]
    steps: usize,
}

fn prepare_samples(pairs: &[(String, Ink)]) -> (Vec<Sample>, usize) {
    let mut samples = Vec::with_capacity(pairs.len());
    let mut skipped = 0usize;
    for (text, ink) in pairs {
        let Some(labels) = decode::encode_labels(text) else {
            skipped += 1;
            continue;
        };
        if labels.is_empty() {
            // Burn's CTCLoss requires target_length >= 1.
            skipped += 1;
            continue;
        }
        let encoded = spline::encode_strokes(ink);
        let steps = encoded.len() / spline::STROKE_DIM;
        let ext_len = 2 * labels.len() + 1;
        if steps == 0 || steps < ext_len {
            skipped += 1;
            continue;
        }
        samples.push(Sample {
            labels,
            encoded,
            steps,
        });
    }
    (samples, skipped)
}

/// Sidecar path for optimizer moments next to a model checkpoint
/// (`foo.mpk` → `foo.optim.mpk`). AdamW first/second moments and SGD
/// momentum live here so a PBT stretch / `train_loop` restart does not
/// throw them away.
pub fn optimizer_sidecar(model_ckpt: impl AsRef<Path>) -> PathBuf {
    model_ckpt.as_ref().with_extension("optim.mpk")
}

fn optimizer_kind_path(optim_path: &Path) -> PathBuf {
    optim_path.with_extension("kind")
}

/// Copy `{from}.optim.mpk` (+ kind) onto `{to}`'s sidecar, or delete the
/// destination sidecar if the source has none.
pub fn copy_optimizer_sidecar(from_model: &Path, to_model: &Path) -> Result<(), String> {
    let src = optimizer_sidecar(from_model);
    let dst = optimizer_sidecar(to_model);
    if !src.exists() {
        clear_optimizer_sidecar(to_model);
        return Ok(());
    }
    std::fs::copy(&src, &dst).map_err(|e| e.to_string())?;
    let ksrc = optimizer_kind_path(&src);
    let kdst = optimizer_kind_path(&dst);
    if ksrc.exists() {
        std::fs::copy(&ksrc, &kdst).map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn clear_optimizer_sidecar(model_ckpt: &Path) {
    let p = optimizer_sidecar(model_ckpt);
    let _ = std::fs::remove_file(&p);
    let _ = std::fs::remove_file(optimizer_kind_path(&p));
}

fn save_optimizer<O>(optimizer: &O, path: &Path, use_sgd: bool) -> Result<(), String>
where
    O: Optimizer<model::Recognizer<TrainBackend>, TrainBackend>,
{
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    recorder
        .record(optimizer.to_record(), path.to_path_buf())
        .map_err(|e| e.to_string())?;
    let kind = if use_sgd { "sgd" } else { "adamw" };
    std::fs::write(optimizer_kind_path(path), kind).map_err(|e| e.to_string())
}

fn load_optimizer<O>(optimizer: O, path: &Path, use_sgd: bool, device: &WgpuDevice) -> O
where
    O: Optimizer<model::Recognizer<TrainBackend>, TrainBackend>,
{
    if !path.exists() {
        return optimizer;
    }
    let want = if use_sgd { "sgd" } else { "adamw" };
    let got = std::fs::read_to_string(optimizer_kind_path(path)).unwrap_or_default();
    if got.trim() != want {
        eprintln!(
            "warning: skipping {} (kind '{}' vs {want})",
            path.display(),
            got.trim()
        );
        return optimizer;
    }
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    match recorder.load(path.to_path_buf(), device) {
        Ok(record) => optimizer.load_record(record),
        Err(e) => {
            eprintln!("warning: failed to load optimizer {}: {e}", path.display());
            optimizer
        }
    }
}

/// Train a model on `pairs`, starting from `init_checkpoint`'s weights if
/// given (fine-tuning) or from a fresh random init otherwise. Calls
/// `on_epoch` after each epoch (e.g. for logging/checkpointing) with an
/// inference-ready (dropout-disabled) copy of the model and a `save_optim`
/// callback so the caller can persist AdamW/SGD moments next to the
/// weights they keep. `init_optim` reloads those moments when present and
/// the kind (adamw vs sgd) matches this run. Returns the final trained
/// model, also inference-ready.
pub fn train(
    pairs: &[(String, Ink)],
    config: &TrainConfig,
    init_checkpoint: Option<&Path>,
    init_optim: Option<&Path>,
    mut on_epoch: impl FnMut(
        &EpochStats,
        &model::Recognizer<Backend>,
        &mut dyn FnMut(&Path) -> Result<(), String>,
    ),
) -> model::Recognizer<Backend> {
    let device = Default::default();
    let model_config = model::Config::default();
    let mut model: model::Recognizer<TrainBackend> = model::Recognizer::new(model_config, &device);

    if let Some(path) = init_checkpoint {
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model = model.load_file(path.to_path_buf(), &recorder, &device).unwrap_or_else(|e| {
            panic!(
                "failed to load init checkpoint {}: {e} — HAT weights are a different layout than the old BiLSTM/TCN checkpoints",
                path.display()
            );
        });
    }

    let blank = decode::classes() - 1;
    // `zero_infinity` (PyTorch's own CTCLoss has the identical option, off
    // by default there too) masks only the *individual samples* in a batch
    // whose CTC loss came out infinite — a well-documented, expected
    // occurrence when the output distribution gets peaky/overconfident
    // (exactly what happens as a model slides toward blank-collapse: a
    // required non-blank alignment path can get assigned ~0 probability,
    // whose log is -inf) — rather than letting one such sample poison the
    // batch's *mean* loss into non-finite and discarding all 16 samples'
    // gradient signal, per the outer skip in `train_batch_step`. This is
    // strictly better than that all-or-nothing skip: the other, healthy
    // samples in the same batch still contribute a real gradient update
    // instead of being thrown away along with the poisoned one.
    let ctc = CTCLossConfig::new()
        .with_blank(blank)
        .with_zero_infinity(true)
        .init();

    let (mut samples, _statically_skipped) = prepare_samples(pairs);
    // Bucket by length: minimizes padding within a batch.
    samples.sort_by_key(|s| s.steps);

    let batch_size = config.batch_size.max(1);
    // Drop a final undersized batch rather than keep it: `BatchNorm`
    // computes (and folds into its running mean/var via an EMA) statistics
    // over [batch, steps] jointly, so a small trailing batch feeds noisy
    // statistics into that running average every single epoch — the same
    // batch, since only batch *order* is reshuffled below, not membership.
    // For a small dataset (few batches/epoch) that noise dominates and
    // shows up as epoch-to-epoch CER oscillation instead of steady
    // improvement; standard practice elsewhere (e.g. PyTorch's
    // `drop_last=True`) for exactly this reason.
    let mut batches: Vec<Vec<usize>> = (0..samples.len())
        .collect::<Vec<_>>()
        .chunks(batch_size)
        .filter(|c| c.len() == batch_size)
        .map(|c| c.to_vec())
        .collect();
    // Degenerate case (dataset smaller than one batch): dropping is worse
    // than an occasionally-noisy batch, so fall back to keeping it whole.
    if batches.is_empty() && !samples.is_empty() {
        batches.push((0..samples.len()).collect());
    }

    if config.use_sgd {
        let mut optimizer = SgdConfig::new()
            .with_momentum(Some(MomentumConfig::new().with_momentum(0.9)))
            .with_gradient_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)))
            .init();
        if let Some(path) = init_optim {
            optimizer = load_optimizer(optimizer, path, true, &device);
        }
        model = run_epochs(
            model,
            &mut optimizer,
            &ctc,
            &samples,
            &batches,
            config,
            pairs,
            &device,
            &mut on_epoch,
        );
    } else {
        let mut optimizer = AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)))
            .init();
        if let Some(path) = init_optim {
            optimizer = load_optimizer(optimizer, path, false, &device);
        }
        model = run_epochs(
            model,
            &mut optimizer,
            &ctc,
            &samples,
            &batches,
            config,
            pairs,
            &device,
            &mut on_epoch,
        );
    }

    model.valid()
}

/// The epoch loop itself, generic over the optimizer (`O`) so [`train`] can
/// pick AdamW or SGD+momentum without duplicating this — see
/// `TrainConfig::use_sgd`'s docs for why both exist.
#[allow(clippy::too_many_arguments)]
fn run_epochs<O>(
    mut model: model::Recognizer<TrainBackend>,
    optimizer: &mut O,
    ctc: &burn::nn::loss::CTCLoss,
    samples: &[Sample],
    batches: &[Vec<usize>],
    config: &TrainConfig,
    pairs: &[(String, Ink)],
    device: &WgpuDevice,
    on_epoch: &mut impl FnMut(
        &EpochStats,
        &model::Recognizer<Backend>,
        &mut dyn FnMut(&Path) -> Result<(), String>,
    ),
) -> model::Recognizer<TrainBackend>
where
    O: Optimizer<model::Recognizer<TrainBackend>, TrainBackend>,
{
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    // Counts every batch across the WHOLE run (not reset per epoch) — the
    // schedule is one continuous curve spanning all epochs, not a per-epoch
    // ramp.
    let mut global_step = 0usize;
    let mut current_lr = config.learning_rate;

    for epoch in 0..config.epochs {
        let mut total_loss = 0.0f64;
        let mut trained = 0usize;
        let mut last_grad_rms: Vec<(String, f32)> = Vec::new();

        let mut batch_order: Vec<usize> = (0..batches.len()).collect();
        batch_order.shuffle(&mut rng);

        for (_step_i, &bi) in batch_order.iter().enumerate() {
            current_lr = lr_at_step(global_step, config.learning_rate, &config.lr_schedule);
            let batch: Vec<&Sample> = batches[bi].iter().map(|&i| &samples[i]).collect();
            // Gradient RMS requires a GPU->CPU read of every parameter
            // tensor; do it only on the last batch of the epoch so the
            // dashboard can plot layer-wise gradient flow without paying
            // that cost on every step.
            // Left off: that sync was aborting/slowing every epoch (Burn
            // stores grads as inner-backend tensors; looking them up as
            // Autodiff panicked, and even the successful path is a full
            // param round-trip). Re-enable with
            // `capture_grads = step_i + 1 == batch_order.len()` if the
            // dashboard grads chart is needed again.
            let capture_grads = false;
            let (updated_model, loss_val, valid, grad_rms) = train_batch_step(
                model,
                optimizer,
                current_lr,
                ctc,
                &batch,
                device,
                capture_grads,
            );
            model = updated_model;
            total_loss += loss_val as f64;
            trained += valid;
            if !grad_rms.is_empty() {
                last_grad_rms = grad_rms;
            }
            global_step += 1;
        }

        let stats = EpochStats {
            epoch,
            mean_loss: if trained > 0 {
                total_loss / trained as f64
            } else {
                f64::NAN
            },
            samples: trained,
            skipped: pairs.len() - trained,
            lr: current_lr,
            grad_rms: last_grad_rms,
        };
        let inference_model = model.valid();
        let use_sgd = config.use_sgd;
        let mut save_optim = |path: &Path| save_optimizer(optimizer, path, use_sgd);
        on_epoch(&stats, &inference_model, &mut save_optim);
    }

    model
}

/// Run one training step on a batch of samples (already bucketed by the
/// caller). Returns `(updated model, summed loss, batch size, grad_rms)`.
/// `grad_rms` is only populated when `capture_grads` is set — see the
/// call in `run_epochs`.
fn train_batch_step<O>(
    model: model::Recognizer<TrainBackend>,
    optimizer: &mut O,
    lr: f64,
    ctc: &burn::nn::loss::CTCLoss,
    batch: &[&Sample],
    device: &WgpuDevice,
    capture_grads: bool,
) -> (
    model::Recognizer<TrainBackend>,
    f32,
    usize,
    Vec<(String, f32)>,
)
where
    O: Optimizer<model::Recognizer<TrainBackend>, TrainBackend>,
{
    let b = batch.len();
    let max_steps = batch.iter().map(|s| s.steps).max().unwrap_or(0);
    let max_target_len = batch.iter().map(|s| s.labels.len()).max().unwrap_or(0);
    if max_steps == 0 || max_target_len == 0 {
        return (model, 0.0, 0, Vec::new());
    }

    let mut target_buf = vec![0i32; b * max_target_len];
    let mut input_lengths = vec![0i32; b];
    let mut target_lengths = vec![0i32; b];

    for (bi, sample) in batch.iter().enumerate() {
        input_lengths[bi] = sample.steps as i32;
        let tdst = bi * max_target_len;
        for (j, &label) in sample.labels.iter().enumerate() {
            target_buf[tdst + j] = label as i32;
        }
        target_lengths[bi] = sample.labels.len() as i32;
    }

    let rows = batch.iter().map(|s| (s.encoded.as_slice(), s.steps));
    let (strokes, images, pad) = spline::pack_hat_batch(rows, b, b, max_steps);
    let (strokes, images, pad_mask) =
        model::packed_inputs(strokes, images, pad, b, max_steps, device);
    let logits = model.forward_logits(strokes, images, Some(pad_mask)); // [B, max_steps, classes]

    // Burn's CTCLoss wants log-probabilities shaped [time, batch, classes].
    let log_probs = log_softmax(logits, 2).swap_dims(0, 1);

    let targets = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(target_buf, [b, max_target_len]),
        device,
    );
    let input_lengths =
        Tensor::<TrainBackend, 1, Int>::from_data(TensorData::new(input_lengths, [b]), device);
    let target_lengths =
        Tensor::<TrainBackend, 1, Int>::from_data(TensorData::new(target_lengths, [b]), device);

    let loss = ctc.forward_with_reduction(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        Reduction::Mean,
    );
    let loss_val: f32 = loss
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("scalar loss should convert to Vec<f32>")[0];

    // A non-finite loss means this batch's gradient would be garbage (or
    // worse, would poison the optimizer's own running moment estimates,
    // e.g. AdamW's, in a way that can propagate indefinitely). We hit this
    // for real: AdamW at lr=0.1 ran clean for ~47 epochs (~7,000 steps),
    // then silently diverged to NaN and stayed there for 120+ more epochs
    // before anyone noticed — gradient-norm clipping doesn't catch this
    // since it bounds the gradient, not the already-corrupted state. Skip
    // the update entirely and leave the model (and optimizer state)
    // untouched, so one bad batch can't derail or waste the rest of a long
    // unattended run.
    if !loss_val.is_finite() {
        eprintln!("warning: skipping batch with non-finite loss ({loss_val})");
        return (model, 0.0, 0, Vec::new());
    }

    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let grad_rms = if capture_grads {
        crate::probe::grouped_grad_rms(&model, &grads)
    } else {
        Vec::new()
    };
    let model = optimizer.step(lr, model, grads);

    (model, loss_val, b, grad_rms)
}

/// Save a checkpoint in Burn's Named-MessagePack format.
pub fn save(model: &model::Recognizer<Backend>, path: impl AsRef<Path>) -> Result<(), String> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(path.as_ref().to_path_buf(), &recorder)
        .map_err(|e| e.to_string())
}
