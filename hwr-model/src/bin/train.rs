//! Train (or fine-tune) the recognizer on one or more armrest-format
//! corpus sources (a `.txt` file, or a directory of them).
//!
//! Usage:
//!   train [--out PATH] [--init PATH] [--epochs N] [--lr F] [--batch-size N] [--max-steps N] [SOURCE...]
//!
//!   --out PATH        where to save the trained checkpoint (Burn
//!                      Named-MessagePack format; default: checkpoint.mpk)
//!   --init PATH       start from this checkpoint's weights instead of
//!                      random init — i.e. fine-tune rather than train from
//!                      scratch
//!   --epochs N        number of training epochs (default: 15)
//!   --lr F            learning rate (default: 1e-3)
//!   --batch-size N    samples per batch (default: 16) — see `train::train`
//!                      module docs for why/how batching works here
//!   --max-steps N     samples with more encoded steps than this (default:
//!                      150) get split into per-word samples via
//!                      `corpus::cap_long_samples` — see its docs for why:
//!                      a few full-sentence/poem-line outliers in armrest's
//!                      corpus made Burn's autodiff graph bookkeeping (which
//!                      scales super-linearly with sequence length) the
//!                      dominant cost of an epoch
//!   SOURCE...         corpus files/dirs (default: armrest/data/inks)
//!
//! Runs on `hwr_model::Backend` (Wgpu, targeting Metal on macOS) — unlike
//! the candle version, there's no `--device` flag: Burn's backend is a
//! compile-time type parameter, not a runtime switch, and Wgpu is the only
//! backend this crate builds against.
//!
//! Example — fine-tune the shipped checkpoint on calibration data, on top
//! of the original corpus so it doesn't forget everything else:
//!   train --out checkpoints/model-finetuned.mpk \
//!         --init checkpoints/model.mpk --epochs 5 \
//!         armrest/data/inks ~/Library/Application\ Support/hwr/calibration.txt

use std::path::PathBuf;

use burn::module::Module;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use hwr_ink::ink::Ink;
use hwr_model::corpus::load_source;
use hwr_model::train::{train, LrSchedule, TrainConfig};

fn main() {
    let mut checkpoint_out = PathBuf::from("checkpoint.mpk");
    let mut init_checkpoint: Option<PathBuf> = None;
    let mut epochs = 15usize;
    let mut learning_rate = 1e-3f64;
    let mut batch_size = TrainConfig::default().batch_size;
    let mut max_steps = 150usize;
    let mut max_grad_norm = TrainConfig::default().max_grad_norm;
    let mut use_sgd = TrainConfig::default().use_sgd;
    let mut peak_lr: Option<f64> = None;
    let mut warmup_steps = 100usize;
    let mut decay_steps = 2000usize;
    // TCN off unless `--tcn` is added later. Same as `--no-tcn`.
    let mut tcn_channels_override: Option<Option<usize>> = Some(None);
    let mut sources: Vec<PathBuf> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out" => checkpoint_out = PathBuf::from(args.next().expect("--out needs a path")),
            "--init" => {
                init_checkpoint = Some(PathBuf::from(args.next().expect("--init needs a path")))
            }
            "--epochs" => {
                epochs = args
                    .next()
                    .expect("--epochs needs a number")
                    .parse()
                    .expect("epochs must be a number")
            }
            "--lr" => {
                learning_rate = args
                    .next()
                    .expect("--lr needs a number")
                    .parse()
                    .expect("lr must be a number")
            }
            "--batch-size" => {
                batch_size = args
                    .next()
                    .expect("--batch-size needs a number")
                    .parse()
                    .expect("batch size must be a number")
            }
            "--max-steps" => {
                max_steps = args
                    .next()
                    .expect("--max-steps needs a number")
                    .parse()
                    .expect("max steps must be a number")
            }
            "--max-grad-norm" => {
                max_grad_norm = args
                    .next()
                    .expect("--max-grad-norm needs a number")
                    .parse()
                    .expect("max grad norm must be a number")
            }
            "--sgd" => use_sgd = true,
            "--no-tcn" => tcn_channels_override = Some(None),
            "--peak-lr" => {
                peak_lr = Some(
                    args.next()
                        .expect("--peak-lr needs a number")
                        .parse()
                        .expect("peak lr must be a number"),
                )
            }
            "--warmup-steps" => {
                warmup_steps = args
                    .next()
                    .expect("--warmup-steps needs a number")
                    .parse()
                    .expect("warmup steps must be a number")
            }
            "--decay-steps" => {
                decay_steps = args
                    .next()
                    .expect("--decay-steps needs a number")
                    .parse()
                    .expect("decay steps must be a number")
            }
            other => sources.push(PathBuf::from(other)),
        }
    }
    if sources.is_empty() {
        sources.push(PathBuf::from("armrest/data/inks"));
    }

    let mut pairs = Vec::new();
    for source in &sources {
        load_source(source, &mut pairs);
    }
    println!(
        "Total: {} (text, ink) pairs from {} source(s)",
        pairs.len(),
        sources.len()
    );
    pairs = hwr_model::corpus::cap_long_samples(pairs, max_steps);
    if let Some(init) = &init_checkpoint {
        println!("Fine-tuning from checkpoint: {}", init.display());
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    pairs.shuffle(&mut rng);

    let val_fraction = 0.1;
    let val_count = ((pairs.len() as f64) * val_fraction).round() as usize;
    let (val_pairs, train_pairs) = pairs.split_at(val_count);
    println!(
        "Split: {} train, {} validation",
        train_pairs.len(),
        val_pairs.len()
    );

    let lr_schedule = peak_lr.map(|peak_lr| LrSchedule {
        warmup_steps,
        peak_lr,
        decay_steps,
    });
    let config = TrainConfig {
        epochs,
        learning_rate,
        batch_size,
        max_grad_norm,
        use_sgd,
        lr_schedule,
        tcn_channels_override,
    };
    println!(
        "Batch size: {batch_size}, max_grad_norm: {max_grad_norm}, optimizer: {}, architecture: HAT",
        if use_sgd { "SGD+momentum" } else { "AdamW" },
    );
    if let Some(s) = &lr_schedule {
        println!(
            "LR schedule: 1e-3 -> {} over {} steps, then {} -> {} over {} steps, then flat",
            s.peak_lr, s.warmup_steps, s.peak_lr, learning_rate, s.decay_steps
        );
    } else {
        println!("LR: flat {learning_rate}");
    }

    let device = Default::default();

    // A fixed subsample of the training set, so we can cheaply track
    // train-set CER (overfitting signal) alongside held-out val CER.
    let train_sample: Vec<(String, Ink)> = train_pairs.iter().take(200).cloned().collect();

    // Best-checkpoint selection: only overwrite `checkpoint_out` when val CER
    // improves, so a later epoch that's started overfitting (which dropout
    // reduces but doesn't eliminate) can never clobber a better earlier one.
    //
    // When fine-tuning (`--init`), seed this from the *loaded* checkpoint's
    // own val CER rather than starting at infinity: this process is one link
    // in a chain of restarts (see the training-loop wrapper script — each
    // process only runs a few epochs before a fresh one picks up from the
    // last checkpoint, to bound a per-process performance regression we hit
    // in Burn's fusion/dispatch queue), and without this, a fresh process's
    // epoch 0 would always look like "best so far" against infinity and
    // could silently overwrite a strictly-better checkpoint with a slightly
    // worse one just because it happened to run first in a new process.
    let mut best_val_cer = match &init_checkpoint {
        Some(path) => {
            let recorder =
                burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new();
            let baseline_model_config = hwr_model::model::Config::default();
            let init_model = hwr_model::model::Recognizer::<hwr_model::Backend>::new(
                baseline_model_config,
                &device,
            )
            .load_file(path.to_path_buf(), &recorder, &device)
            .expect("failed to load init checkpoint for baseline val CER — LSTM/TCN checkpoints cannot load into HAT");
            let baseline = hwr_model::eval::mean_cer(&init_model, val_pairs, &device);
            println!("Baseline val_cer from init checkpoint: {baseline:.4}");
            baseline
        }
        None => f64::INFINITY,
    };
    let mut best_epoch = 0usize;

    let start = std::time::Instant::now();
    let _final_model = train(
        train_pairs,
        &config,
        init_checkpoint.as_deref(),
        None,
        |stats, net, save_optim| {
            let val_cer = hwr_model::eval::mean_cer(net, val_pairs, &device);
            let train_cer = hwr_model::eval::mean_cer(net, &train_sample, &device);
            let improved = val_cer < best_val_cer;
            println!(
                "epoch {:>3}: mean_loss={:.4} samples={} skipped={} train_cer={:.4} val_cer={:.4} elapsed={:.1}s lr={:.5}{}",
                stats.epoch,
                stats.mean_loss,
                stats.samples,
                stats.skipped,
                train_cer,
                val_cer,
                start.elapsed().as_secs_f64(),
                stats.lr,
                if improved { "  <- best so far, saving" } else { "" },
            );
            // Layer / CTC-output snapshot on a fixed 16-sample slice of the
            // train subsample. Cheap next to mean_cer, and the numbers the
            // dashboard needs to tell blank-collapse apart from "emitting
            // garbage" apart from "actually learning characters".
            // Not on the training path: the extra forward + last-batch
            // grad RMS were stalling epochs. `hwr_model::probe` is still
            // there for a one-off look; loss/val_cer are enough while we
            // just need more training.
            if improved {
                best_val_cer = val_cer;
                best_epoch = stats.epoch;
                if let Err(e) = hwr_model::train::save(net, &checkpoint_out) {
                    eprintln!("failed to save checkpoint: {e}");
                } else {
                    let sidecar = hwr_model::train::optimizer_sidecar(&checkpoint_out);
                    if let Err(e) = save_optim(&sidecar) {
                        eprintln!("failed to save optimizer sidecar: {e}");
                    }
                }
            }
        },
    );

    println!(
        "Best validation CER: {best_val_cer:.4} (epoch {best_epoch}), saved to {}",
        checkpoint_out.display()
    );
}
