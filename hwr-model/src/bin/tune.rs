//! Hyperparameter search via a real sampler (TPE, from the `rustuna` crate
//! — a from-scratch Rust port of Optuna, vendored under `third_party/`)
//! instead of hand-launched "variant A/B/C" parallel guesses. Each trial
//! fine-tunes `--init`'s checkpoint for a short, fixed epoch budget under a
//! sampled `(learning_rate, use_sgd)` combination — flat LR, no hand-built
//! one-cycle schedule (see `run_one_trial`'s comment for why) — and
//! reports the best validation CER seen during that budget; TPE uses the
//! growing history of (params -> result) pairs to bias later samples
//! toward promising regions, the way a person doing this by hand would
//! after the first few trials — just faster and without the bias of only
//! trying what we already guessed might work.
//!
//! `--workers N` runs N trials concurrently as OS threads *within this one
//! process*, all pulling from one shared, in-memory `rustuna_core::Study`
//! (`Arc<Study>` — its `storage`/`sampler`/`queue` fields are themselves
//! `Arc<RwLock<...>>`/`Arc<dyn ...>`, so cloning the `Arc<Study>` handle
//! into each worker thread is enough; no IPC or on-disk storage needed).
//! That's a real shared trial queue, not the uncoordinated "launch 3
//! separate `tune` processes with different seeds" approach tried earlier
//! tonight — every worker's completed trial feeds the *same* TPE history,
//! so later trials (on any thread) get to use what every other thread has
//! learned so far, not just their own. Justified by measurement, not
//! guesswork: with 4 concurrent training processes already running
//! tonight, per-process CPU usage never exceeded ~40% of one core (this is
//! an M3 Max, 14 cores) and fans never spun up — this workload is
//! dispatch-latency-bound, not compute-bound, so concurrent workers mostly
//! fill each other's idle gaps rather than genuinely contend.
//!
//! This does NOT replace the long-running `train` processes — it's a
//! cheap(er), short-budget search to find a better *starting* LR schedule,
//! whose winner can then be handed to `train`/`train_once.sh` for a full
//! run. A short per-trial budget is a real tradeoff: a combination that
//! only pays off after 60+ epochs will look no better than one that
//! plateaus immediately. Treat the winner as a promising lead, not a proof.
//! For *continuous* adaptation while training (copy weights from better
//! replicas, mutate their HPs, keep going), see `bin/pbt.rs`.
//!
//! `--init` is optional: without it, every trial trains from a fresh
//! random init instead of fine-tuning a shared checkpoint — necessary when
//! no checkpoint exists yet for the architecture/corpus being searched
//! (e.g. right after an architecture change, or a corpus never trained
//! before). `--max-samples N` caps how many corpus samples each trial
//! trains on, trading trial fidelity for trial throughput — the same idea
//! as a short `--budget-epochs`, just applied to corpus size instead of
//! epoch count, so a large corpus doesn't make every trial expensive.
//!
//! Usage:
//!   tune [--init checkpoints/model.mpk] [--trials N] [--budget-epochs N] [--max-samples N] [--workers N] [--seed N] [SOURCE...]

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use rand::seq::SliceRandom;
use rand::SeedableRng;

use hwr_ink::ink::Ink;
use hwr_model::corpus::load_source;
use hwr_model::train::{train, TrainConfig};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study, Direction, Study};
use rustuna_sampler::tpe::TpeSampler;

struct TrialResult {
    number: u32,
    best_val_cer: f64,
    learning_rate: f64,
    use_sgd: bool,
}

/// Runs one trial to completion: ask the shared study for params, fine-tune
/// `init_checkpoint` for `budget_epochs` under them, tell the study the
/// result. Returns `Ok(false)` when the study has no more trials to give
/// out (`remaining` hit zero) — the signal for a worker thread to stop.
fn run_one_trial(
    study: &Study,
    remaining: &AtomicUsize,
    init_checkpoint: Option<&std::path::Path>,
    train_pairs: &[(String, Ink)],
    val_pairs: &[(String, Ink)],
    budget_epochs: usize,
    worker_id: usize,
) -> bool {
    // Claim a trial slot before asking the study for one, so concurrent
    // workers can't collectively run more than `--trials` total (the last
    // few threads to check in just see 0 remaining and stop).
    loop {
        let cur = remaining.load(Ordering::SeqCst);
        if cur == 0 {
            return false;
        }
        if remaining
            .compare_exchange(cur, cur - 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            break;
        }
    }

    // Flat learning rate, no hand-designed one-cycle warmup/peak/decay
    // shape: that schedule was a manually-picked heuristic from before we
    // had a real search tool, and layering it on top of TPE just adds 3
    // more correlated dimensions (peak_lr, warmup_steps, decay_steps) for
    // the sampler to untangle instead of directly searching the one knob
    // that actually matters. TPE is allowed to try the same aggressive
    // values the old schedule's peak reached (up to 0.5) directly, flat
    // from step 0 — the non-finite-loss skip in train_batch_step already
    // backstops any individual step that goes bad, and a genuinely-too-hot
    // flat LR trial just reports a bad val_cer and gets naturally avoided
    // by later trials, same as any other bad hyperparameter region.
    let mut trial = study.ask().expect("study.ask failed");
    let learning_rate = trial
        .suggest(
            "learning_rate",
            &Distribution::new_float(0.001, 0.5, None, true),
        )
        .expect("suggest learning_rate");
    let use_sgd = *trial
        .suggest_categorical("use_sgd", &[false, true])
        .expect("suggest use_sgd");
    let trial_number = trial.number;

    println!("\n=== [w{worker_id}] trial {trial_number}: lr={learning_rate:.5} sgd={use_sgd} ===");

    let config = TrainConfig {
        epochs: budget_epochs,
        learning_rate,
        max_grad_norm: TrainConfig::default().max_grad_norm,
        batch_size: TrainConfig::default().batch_size,
        use_sgd,
        lr_schedule: None,
        // Not searched here: every trial shares one `--init` checkpoint,
        // whose architecture (TCN on/off) is fixed at whatever it was
        // trained with — sampling a different tcn_channels per trial would
        // crash loading that checkpoint's weights into a mismatched model
        // shape. The TCN-vs-no-TCN question is a separate, from-scratch
        // ablation (see train.rs's --no-tcn flag), not part of this
        // fine-tune-from-a-checkpoint LR/optimizer search.
        // Explicit no-TCN: Config default is already None, but pass
        // `Some(None)` so a future default-on cannot silently re-enable it.
        tcn_channels_override: Some(None),
    };

    let device = Default::default();
    let mut best_val_cer = f64::INFINITY;
    let _ = train(train_pairs, &config, init_checkpoint, None, |stats, net, _save_optim| {
        let val_cer = hwr_model::eval::mean_cer(net, val_pairs, &device);
        if val_cer < best_val_cer {
            best_val_cer = val_cer;
        }
        println!(
            "  [w{worker_id}] trial {trial_number} epoch {:>3}: mean_loss={:.4} val_cer={:.4} (best this trial: {:.4})",
            stats.epoch, stats.mean_loss, val_cer, best_val_cer
        );
    });

    study
        .tell(
            trial_number,
            rustuna_core::trial::TrialStateValues::Complete(vec![best_val_cer]),
        )
        .expect("study.tell failed");

    // Side-channel for the final ranked-results printout — `tell` only
    // hands the score back into rustuna's own storage, not to us.
    RESULTS.get().unwrap().lock().unwrap().push(TrialResult {
        number: trial_number,
        best_val_cer,
        learning_rate,
        use_sgd,
    });

    true
}

static RESULTS: std::sync::OnceLock<Mutex<Vec<TrialResult>>> = std::sync::OnceLock::new();

fn main() {
    let mut init_checkpoint: Option<PathBuf> = None;
    let mut n_trials = 20usize;
    let mut budget_epochs = 15usize;
    let mut max_steps = 150usize;
    let mut workers = 1usize;
    // Seeds the TPE sampler only — the corpus shuffle below stays fixed at
    // 1234 always, so every run's val_cer is comparable across seeds/runs.
    let mut seed = 1234u64;
    // Caps how many (already-shuffled) samples each trial trains on, so a
    // large corpus doesn't make every trial expensive — trading trial
    // fidelity for trial *throughput*, which is the right trade for a
    // cheap short-budget search: more (params -> result) data points per
    // wall-clock hour lets TPE narrow in faster, same idea as
    // `--budget-epochs` already trading full-training fidelity for speed.
    // `None` (default) uses the whole corpus.
    let mut max_samples: Option<usize> = None;
    let mut sources: Vec<PathBuf> = Vec::new();

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--init" => {
                init_checkpoint = Some(PathBuf::from(args.next().expect("--init needs a path")))
            }
            "--trials" => {
                n_trials = args
                    .next()
                    .expect("--trials needs a number")
                    .parse()
                    .expect("trials must be a number")
            }
            "--budget-epochs" => {
                budget_epochs = args
                    .next()
                    .expect("--budget-epochs needs a number")
                    .parse()
                    .expect("budget-epochs must be a number")
            }
            "--max-steps" => {
                max_steps = args
                    .next()
                    .expect("--max-steps needs a number")
                    .parse()
                    .expect("max steps must be a number")
            }
            "--workers" => {
                workers = args
                    .next()
                    .expect("--workers needs a number")
                    .parse()
                    .expect("workers must be a number")
            }
            "--seed" => {
                seed = args
                    .next()
                    .expect("--seed needs a number")
                    .parse()
                    .expect("seed must be a number")
            }
            "--max-samples" => {
                max_samples = Some(
                    args.next()
                        .expect("--max-samples needs a number")
                        .parse()
                        .expect("max-samples must be a number"),
                )
            }
            other => sources.push(PathBuf::from(other)),
        }
    }
    if sources.is_empty() {
        sources.push(PathBuf::from("armrest/data/inks"));
    }
    // `--init` is now optional: with none, each trial trains from a fresh
    // random init instead of fine-tuning a shared checkpoint. Needed for
    // searching hyperparameters for an architecture (or corpus) that has
    // no checkpoint yet at all — e.g. right after dropping the TCN
    // front-end, or a corpus (like a from-scratch digits-only run) that's
    // never been trained before. Combine with `--max-samples`/a small
    // `--budget-epochs` to keep from-scratch trials cheap.
    RESULTS.set(Mutex::new(Vec::new())).ok();

    let mut pairs = Vec::new();
    for source in &sources {
        load_source(source, &mut pairs);
    }
    pairs = hwr_model::corpus::cap_long_samples(pairs, max_steps);

    // Same seed/split methodology as bin/train.rs, so a trial's val_cer is
    // directly comparable to the long-running processes' logged numbers.
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    pairs.shuffle(&mut rng);
    if let Some(n) = max_samples {
        pairs.truncate(n);
    }
    let val_fraction = 0.1;
    let val_count = ((pairs.len() as f64) * val_fraction).round() as usize;
    println!(
        "Split: {} train, {} validation. {n_trials} trials across {workers} worker thread(s), {budget_epochs} epochs/trial, {}",
        pairs.len() - val_count,
        val_count,
        match &init_checkpoint {
            Some(p) => format!("starting from {}", p.display()),
            None => "from scratch (no --init)".to_string(),
        }
    );

    let pairs = Arc::new(pairs);
    let storage = InMemoryStorage::new();
    let sampler = TpeSampler::seed_from_u64(seed);
    let study = Arc::new(
        create_study(
            "hwr-lr-schedule-search",
            storage,
            sampler,
            vec![Direction::Minimize],
        )
        .expect("failed to create study"),
    );
    let remaining = Arc::new(AtomicUsize::new(n_trials));

    let handles: Vec<_> = (0..workers.max(1))
        .map(|worker_id| {
            let study = Arc::clone(&study);
            let remaining = Arc::clone(&remaining);
            let pairs = Arc::clone(&pairs);
            let init_checkpoint = init_checkpoint.clone();
            std::thread::spawn(move || {
                let (val_pairs, train_pairs) = pairs.split_at(val_count);
                while run_one_trial(
                    &study,
                    &remaining,
                    init_checkpoint.as_deref(),
                    train_pairs,
                    val_pairs,
                    budget_epochs,
                    worker_id,
                ) {}
            })
        })
        .collect();
    for h in handles {
        let _ = h.join();
    }

    let mut results = RESULTS.get().unwrap().lock().unwrap();
    results.sort_by(|a, b| a.best_val_cer.partial_cmp(&b.best_val_cer).unwrap());
    println!("\n=== ranked results ({} trials) ===", results.len());
    for r in results.iter() {
        println!(
            "trial {:>3}: best_val_cer={:.4}  lr={:.5} sgd={}",
            r.number, r.best_val_cer, r.learning_rate, r.use_sgd
        );
    }
    if let Some(best) = results.first() {
        println!(
            "\nBest: trial {} val_cer={:.4} -- reuse with train's --lr {:.5}{}",
            best.number,
            best.best_val_cer,
            best.learning_rate,
            if best.use_sgd { " --sgd" } else { "" }
        );
    }
}
