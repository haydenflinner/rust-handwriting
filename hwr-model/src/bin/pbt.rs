//! Population-Based Training: a population of replicas trains in parallel,
//! and every `--ready-epochs` the worse ones copy weights from the better
//! ones and mutate hyperparameters (Jaderberg et al. 2017; Ray Tune
//! truncation). This is the continuous-adaptation counterpart of
//! `bin/tune.rs` — rustuna TPE is a discrete search that then hands a
//! winner to a separate long `train` run; PBT *is* the long run, with
//! hyperparameters still moving.
//!
//! Each ready interval calls `train::train` for `--ready-epochs` starting
//! from that member's checkpoint (or `--init` / random on generation 0).
//! AdamW moments and SGD momentum are persisted in `*.optim.mpk` next to
//! the weight checkpoint and reloaded on the next stretch (and copied on
//! exploit when the optimizer family matches).
//!
//! `--workers N` runs up to N member stretches as **child processes**, not
//! OS threads in this process. `bin/tune.rs` can keep several `train()`
//! calls alive on threads in one process because TPE trials start and
//! finish on a staggered loop; PBT's generation barrier starts every
//! replica's first backward together, and Burn 0.21's global autodiff
//! `TensorContainer` then panics (`downcast` of a `None` at
//! `container.rs:50`) on every worker. Isolated processes each get their
//! own Wgpu/Autodiff state. The parent only copies checkpoints and mutates
//! hyperparameters between generations.
//!
//! Memory: each worker is a full Wgpu/Autodiff process on unified
//! RAM+VRAM. Five of those *also* used to re-parse the raw corpora
//! (glyphs+augmented+MNIST ≈ 900MB of text) at every generation start —
//! that combination swapped the machine to death. The parent now writes a
//! capped `workdir/split.txt` once; workers load only that file. Default
//! `--workers` is 2 (not 5); HAT overnight digits uses 8 workers = 8
//! members so a generation is one wave, not a queue. 10-wide was too
//! close to 36GB unified RAM (~2.8GB/worker). Prefer `--batch-size 32`
//! on word corpora.
//!
//! Usage:
//!   pbt [--out PATH] [--workdir DIR] [--init PATH]
//!       [--population N] [--workers N] [--ready-epochs N] [--generations N]
//!       [--quantile F] [--max-samples N] [--per-source-cap N] [--max-steps N]
//!       [--batch-size N] [--no-tcn] [--seed N] [--lr-min F] [--lr-max F]
//!       [SOURCE...]
//!
//! Example — cheaper search-style PBT on a subsample, then inspect
//! `--out` and continue with a larger `--generations` / no `--max-samples`:
//!   pbt --out checkpoints/pbt-best.mpk --workdir checkpoints/pbt \
//!       --population 4 --workers 2 --ready-epochs 4 --generations 25 \
//!       --max-samples 8000 --batch-size 32 --no-tcn synthetic/mnist_train.txt

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

use burn::module::Module;
use rand::rngs::StdRng;
use rand::SeedableRng;

use hwr_ink::ink::Ink;
use hwr_model::corpus::{load_pairs, load_sources_capped, save_pairs};
use hwr_model::pbt::{
    apply_exploits, cer_sort_key, sample_hparams, truncation_exploit, Hyperparams, MemberState,
    PbtConfig,
};
use hwr_model::train::{
    clear_optimizer_sidecar, copy_optimizer_sidecar, optimizer_sidecar, train, TrainConfig,
};

struct Member {
    id: usize,
    hparams: Hyperparams,
    ckpt: PathBuf,
    val_cer: f64,
}

fn member_train_config(
    hp: Hyperparams,
    ready_epochs: usize,
    tcn_channels_override: Option<Option<usize>>,
    batch_size: usize,
) -> TrainConfig {
    TrainConfig {
        epochs: ready_epochs,
        learning_rate: hp.learning_rate,
        max_grad_norm: TrainConfig::default().max_grad_norm,
        batch_size: batch_size.max(1),
        use_sgd: hp.use_sgd,
        // Flat LR: PBT's explore step *is* the schedule. A one-cycle
        // warmup/peak/decay on every stretch would fight the mutated LR
        // the population just settled on.
        lr_schedule: None,
        tcn_channels_override,
    }
}

fn init_for_member(ckpt: &Path, shared_init: Option<&Path>) -> Option<PathBuf> {
    if ckpt.exists() {
        Some(ckpt.to_path_buf())
    } else {
        shared_init.map(|p| p.to_path_buf())
    }
}

#[derive(Clone)]
struct StretchOutcome {
    id: usize,
    val_cer: f64,
}

fn run_stretch(
    gen: usize,
    ready_epochs: usize,
    member_id: usize,
    hp: Hyperparams,
    ckpt: &Path,
    shared_init: Option<&Path>,
    train_pairs: &[(String, Ink)],
    val_pairs: &[(String, Ink)],
    tcn_channels_override: Option<Option<usize>>,
    batch_size: usize,
) -> StretchOutcome {
    let device = Default::default();
    let config = member_train_config(hp, ready_epochs, tcn_channels_override, batch_size);
    let init = init_for_member(ckpt, shared_init);
    let init_optim = {
        let p = optimizer_sidecar(ckpt);
        if p.exists() {
            Some(p)
        } else {
            None
        }
    };
    let last_val = Arc::new(Mutex::new(f64::NAN));
    let last_val_cb = Arc::clone(&last_val);
    let best_val = Arc::new(Mutex::new(f64::INFINITY));
    let best_val_cb = Arc::clone(&best_val);
    let ckpt_cb = ckpt.to_path_buf();
    let trained = train(
        train_pairs,
        &config,
        init.as_deref(),
        init_optim.as_deref(),
        |stats, net, save_optim| {
            let (greedy_cer, beam_cer) =
                hwr_model::eval::mean_cers_batched(net, val_pairs, &device, batch_size);
            // Rank / checkpoint on beam: greedy is stuck at CER=1 whenever
            // blank wins argmax, which is the usual early-CTC basin and
            // makes every member look tied. Beam sums CTC alignments (and
            // a digit allow-list on MNIST).
            *last_val_cb.lock().unwrap() = beam_cer;
            // High-LR stretches swing a lot across the ready window (we saw
            // 0.13 then 0.53 in one gen). Keep the best snapshot on disk so
            // PBT ranking / --out are not stuck with the last, worse epoch.
            {
                let mut best = best_val_cb.lock().unwrap();
                if save_member_if_best(net, &ckpt_cb, beam_cer, &mut best) {
                    let optim_path = optimizer_sidecar(&ckpt_cb);
                    if let Err(e) = save_optim(&optim_path) {
                        eprintln!(
                            "warning: failed to save optimizer {}: {e}",
                            optim_path.display()
                        );
                    }
                }
            }
            let total_epoch = gen * ready_epochs + stats.epoch;
            println!(
                "  [m{member_id}] gen {gen} epoch {:>3}: mean_loss={:.4} val_cer={:.4} greedy_cer={:.4} beam_cer={:.4} lr={:.5} (total_epoch={total_epoch})",
                stats.epoch, stats.mean_loss, beam_cer, greedy_cer, beam_cer, stats.lr
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        },
    );
    let last = *last_val.lock().unwrap();
    let mut best = *best_val.lock().unwrap();
    save_member_if_best(&trained, ckpt, last, &mut best);
    let val_cer = if best.is_finite() { best } else { last };
    if let Err(e) = write_cer_sidecar(ckpt, val_cer) {
        eprintln!("warning: failed to write CER sidecar for member {member_id}: {e}");
    }
    println!(
        "  [m{member_id}] stretch done val_cer={:.4} lr={:.5} sgd={}",
        val_cer, hp.learning_rate, hp.use_sgd
    );
    let _ = std::io::Write::flush(&mut std::io::stdout());
    StretchOutcome {
        id: member_id,
        val_cer,
    }
}

/// Same seed/split methodology as `bin/train.rs` / `bin/tune.rs`, so a
/// member's val_cer is comparable to those processes' logged numbers.
fn load_split(
    sources: &[PathBuf],
    max_steps: usize,
    max_samples: Option<usize>,
    per_source_cap: Option<usize>,
) -> (Vec<(String, Ink)>, usize) {
    let mut pairs = load_sources_capped(sources, max_steps, per_source_cap);
    if let Some(n) = max_samples {
        pairs.truncate(n);
    }
    let val_fraction = 0.1;
    let val_count = ((pairs.len() as f64) * val_fraction).round() as usize;
    (pairs, val_count)
}

fn load_cached_split(path: &Path, val_count: usize) -> (Vec<(String, Ink)>, usize) {
    let pairs = load_pairs(path).unwrap_or_else(|e| {
        panic!("failed to load cached PBT split {}: {e}", path.display())
    });
    println!(
        "Loaded cached split {} ({} samples, val={val_count})",
        path.display(),
        pairs.len()
    );
    let val_count = val_count.min(pairs.len());
    (pairs, val_count)
}

fn cer_sidecar(ckpt: &Path) -> PathBuf {
    ckpt.with_extension("cer")
}

fn write_cer_sidecar(ckpt: &Path, val_cer: f64) -> std::io::Result<()> {
    std::fs::write(cer_sidecar(ckpt), format!("{val_cer}"))
}

fn read_cer_sidecar(ckpt: &Path) -> Option<f64> {
    std::fs::read_to_string(cer_sidecar(ckpt))
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

/// Seed `best_val` from an existing `--out` so a restarted process cannot
/// clobber a better checkpoint just because this process's `best_val`
/// started at infinity (same idea as `bin/train.rs`'s baseline val CER).
fn seed_best_val(
    checkpoint_out: &Path,
    pairs: &[(String, Ink)],
    val_count: usize,
    tcn_channels_override: Option<Option<usize>>,
) -> f64 {
    if !checkpoint_out.exists() {
        return f64::INFINITY;
    }
    if let Some(v) = read_cer_sidecar(checkpoint_out) {
        if v.is_finite() {
            println!(
                "Existing {} val_cer={:.4} (sidecar); --out is only overwritten if a member beats this",
                checkpoint_out.display(),
                v
            );
            return v;
        }
    }
    let device = Default::default();
    let (val_pairs, _) = pairs.split_at(val_count);
    match eval_checkpoint(checkpoint_out, val_pairs, &device, tcn_channels_override) {
        Ok(v) => {
            println!(
                "Existing {} val_cer={:.4} (evaluated); --out is only overwritten if a member beats this",
                checkpoint_out.display(),
                v
            );
            if let Err(e) = write_cer_sidecar(checkpoint_out, v) {
                eprintln!("warning: failed to write CER sidecar for {}: {e}", checkpoint_out.display());
            }
            v
        }
        Err(e) => {
            eprintln!(
                "warning: could not score existing {}: {e}; refusing to overwrite it",
                checkpoint_out.display()
            );
            f64::NEG_INFINITY
        }
    }
}

fn eval_checkpoint(
    path: &Path,
    val_pairs: &[(String, Ink)],
    device: &burn::backend::wgpu::WgpuDevice,
    tcn_channels_override: Option<Option<usize>>,
) -> Result<f64, String> {
    let _ = tcn_channels_override;
    let recorder =
        burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new();
    let cfg = hwr_model::model::Config::default();
    let model = hwr_model::model::Recognizer::<hwr_model::Backend>::new(cfg, device)
        .load_file(path.to_path_buf(), &recorder, device)
        .map_err(|e| e.to_string())?;
    Ok(hwr_model::eval::mean_cer(&model, val_pairs, device))
}

fn save_member_if_best(
    net: &hwr_model::model::Recognizer<hwr_model::Backend>,
    ckpt: &Path,
    val_cer: f64,
    best_val: &mut f64,
) -> bool {
    if !val_cer.is_finite() || val_cer >= *best_val {
        return false;
    }
    match hwr_model::train::save(net, ckpt) {
        Ok(()) => {
            *best_val = val_cer;
            true
        }
        Err(e) => {
            eprintln!(
                "warning: failed to save member checkpoint {}: {e}",
                ckpt.display()
            );
            false
        }
    }
}

fn save_if_best(
    member_ckpt: &Path,
    checkpoint_out: &Path,
    val_cer: f64,
    member_id: usize,
    best_val: &mut f64,
    best_member: &mut usize,
) {
    if !val_cer.is_finite() || val_cer >= *best_val {
        return;
    }
    if let Err(e) = std::fs::copy(member_ckpt, checkpoint_out) {
        eprintln!(
            "warning: failed to copy best checkpoint to {}: {e}",
            checkpoint_out.display()
        );
        return;
    }
    if let Err(e) = write_cer_sidecar(checkpoint_out, val_cer) {
        eprintln!(
            "warning: failed to write CER sidecar for {}: {e}",
            checkpoint_out.display()
        );
    }
    *best_val = val_cer;
    *best_member = member_id;
    println!(
        "  saved best so far to {} (m{member_id}, val_cer={val_cer:.4})",
        checkpoint_out.display()
    );
}

/// Flags the parent passes through so a `--stretch` child reloads the same
/// cached split the orchestrator wrote (not the raw corpora).
struct StretchSpawn {
    gen: usize,
    ready_epochs: usize,
    shared_init: Option<PathBuf>,
    split: PathBuf,
    val_count: usize,
    batch_size: usize,
}

fn spawn_stretch(
    exe: &Path,
    member_id: usize,
    hp: Hyperparams,
    ckpt: &Path,
    spec: &StretchSpawn,
) -> std::io::Result<std::process::Child> {
    let mut cmd = Command::new(exe);
    cmd.arg("--stretch")
        .arg("--member-id")
        .arg(member_id.to_string())
        .arg("--gen")
        .arg(spec.gen.to_string())
        .arg("--lr")
        .arg(hp.learning_rate.to_string())
        .arg("--ckpt")
        .arg(ckpt)
        .arg("--ready-epochs")
        .arg(spec.ready_epochs.to_string())
        .arg("--split")
        .arg(&spec.split)
        .arg("--val-count")
        .arg(spec.val_count.to_string())
        .arg("--batch-size")
        .arg(spec.batch_size.to_string())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    if hp.use_sgd {
        cmd.arg("--sgd");
    }
    cmd.arg("--no-tcn");
    if let Some(init) = &spec.shared_init {
        cmd.arg("--init").arg(init);
    }
    cmd.spawn()
}

fn stretch_outcome(id: usize, ckpt: &Path, ok: bool) -> StretchOutcome {
    let val_cer = if ok {
        read_cer_sidecar(ckpt).unwrap_or(f64::INFINITY)
    } else {
        eprintln!("warning: stretch m{id} exited unsuccessfully");
        f64::INFINITY
    };
    StretchOutcome { id, val_cer }
}

/// Reap whichever in-flight worker has already exited. FIFO-wait on the
/// oldest spawn left a finished member occupying a slot while a slower
/// earlier one was still training — with pop > workers that is idle time
/// we already paid RAM for.
fn wait_one_finished(
    inflight: &mut Vec<(usize, PathBuf, std::process::Child)>,
) -> StretchOutcome {
    loop {
        for i in 0..inflight.len() {
            match inflight[i].2.try_wait() {
                Ok(Some(status)) => {
                    let (id, ckpt, _child) = inflight.remove(i);
                    return stretch_outcome(id, &ckpt, status.success());
                }
                Ok(None) => {}
                Err(e) => {
                    let (id, ckpt, _child) = inflight.remove(i);
                    eprintln!("warning: failed to wait for stretch m{id}: {e}");
                    return stretch_outcome(id, &ckpt, false);
                }
            }
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

fn run_generation_parallel(
    members: &[Member],
    workers: usize,
    spec: &StretchSpawn,
) -> Vec<StretchOutcome> {
    let exe = std::env::current_exe().expect("current_exe for pbt stretch workers");
    let n_workers = workers.max(1).min(members.len().max(1));
    let mut outcomes = Vec::with_capacity(members.len());
    let mut inflight: Vec<(usize, PathBuf, std::process::Child)> = Vec::new();
    for m in members {
        while inflight.len() >= n_workers {
            outcomes.push(wait_one_finished(&mut inflight));
        }
        match spawn_stretch(&exe, m.id, m.hparams, &m.ckpt, spec) {
            Ok(child) => inflight.push((m.id, m.ckpt.clone(), child)),
            Err(e) => {
                eprintln!("warning: failed to spawn stretch m{}: {e}", m.id);
                outcomes.push(StretchOutcome {
                    id: m.id,
                    val_cer: f64::INFINITY,
                });
            }
        }
    }
    while !inflight.is_empty() {
        outcomes.push(wait_one_finished(&mut inflight));
    }
    outcomes
}

fn main() {
    let mut checkpoint_out = PathBuf::from("checkpoints/pbt-best.mpk");
    let mut workdir = PathBuf::from("checkpoints/pbt");
    let mut init_checkpoint: Option<PathBuf> = None;
    let mut population = 4usize;
    let mut workers = 2usize;
    let mut ready_epochs = 4usize;
    let mut generations = 25usize;
    let mut max_steps = 150usize;
    let mut seed = 1234u64;
    let mut max_samples: Option<usize> = None;
    let mut per_source_cap: Option<usize> = None;
    // 32, not TrainConfig's 128: PBT runs several Wgpu processes at once
    // on unified memory, and full-vocab sequences are ~150 steps. 128 was
    // fine for a single train process on MNIST digits.
    let mut batch_size = 32usize;
    let mut split_path: Option<PathBuf> = None;
    let mut val_count_arg: Option<usize> = None;
    // TCN stays off unless someone later adds a `--tcn` flag. Defaulting
    // to `Some(None)` (same as `--no-tcn`) so a future Config default-on
    // cannot silently put the front-end back.
    let mut tcn_channels_override: Option<Option<usize>> = Some(None);
    let mut pbt_cfg = PbtConfig::default();
    let mut sources: Vec<PathBuf> = Vec::new();
    let mut stretch = false;
    let mut stretch_member_id = 0usize;
    let mut stretch_gen = 0usize;
    let mut stretch_lr = None;
    let mut stretch_ckpt: Option<PathBuf> = None;
    let mut use_sgd = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--stretch" => stretch = true,
            "--member-id" => {
                stretch_member_id = args
                    .next()
                    .expect("--member-id needs a number")
                    .parse()
                    .expect("member-id must be a number")
            }
            "--gen" => {
                stretch_gen = args
                    .next()
                    .expect("--gen needs a number")
                    .parse()
                    .expect("gen must be a number")
            }
            "--lr" => {
                stretch_lr = Some(
                    args.next()
                        .expect("--lr needs a number")
                        .parse()
                        .expect("lr must be a number"),
                )
            }
            "--ckpt" => stretch_ckpt = Some(PathBuf::from(args.next().expect("--ckpt needs a path"))),
            "--sgd" => use_sgd = true,
            "--out" => checkpoint_out = PathBuf::from(args.next().expect("--out needs a path")),
            "--workdir" => workdir = PathBuf::from(args.next().expect("--workdir needs a path")),
            "--init" => {
                init_checkpoint = Some(PathBuf::from(args.next().expect("--init needs a path")))
            }
            "--population" => {
                population = args
                    .next()
                    .expect("--population needs a number")
                    .parse()
                    .expect("population must be a number")
            }
            "--workers" => {
                workers = args
                    .next()
                    .expect("--workers needs a number")
                    .parse()
                    .expect("workers must be a number")
            }
            "--ready-epochs" => {
                ready_epochs = args
                    .next()
                    .expect("--ready-epochs needs a number")
                    .parse()
                    .expect("ready-epochs must be a number")
            }
            "--generations" => {
                generations = args
                    .next()
                    .expect("--generations needs a number")
                    .parse()
                    .expect("generations must be a number")
            }
            "--quantile" => {
                pbt_cfg.quantile = args
                    .next()
                    .expect("--quantile needs a number")
                    .parse()
                    .expect("quantile must be a number")
            }
            "--max-steps" => {
                max_steps = args
                    .next()
                    .expect("--max-steps needs a number")
                    .parse()
                    .expect("max-steps must be a number")
            }
            "--max-samples" => {
                max_samples = Some(
                    args.next()
                        .expect("--max-samples needs a number")
                        .parse()
                        .expect("max-samples must be a number"),
                )
            }
            "--per-source-cap" => {
                per_source_cap = Some(
                    args.next()
                        .expect("--per-source-cap needs a number")
                        .parse()
                        .expect("per-source-cap must be a number"),
                )
            }
            "--batch-size" => {
                batch_size = args
                    .next()
                    .expect("--batch-size needs a number")
                    .parse()
                    .expect("batch-size must be a number")
            }
            "--split" => split_path = Some(PathBuf::from(args.next().expect("--split needs a path"))),
            "--val-count" => {
                val_count_arg = Some(
                    args.next()
                        .expect("--val-count needs a number")
                        .parse()
                        .expect("val-count must be a number"),
                )
            }
            "--seed" => {
                seed = args
                    .next()
                    .expect("--seed needs a number")
                    .parse()
                    .expect("seed must be a number")
            }
            "--lr-min" => {
                pbt_cfg.lr_min = args
                    .next()
                    .expect("--lr-min needs a number")
                    .parse()
                    .expect("lr-min must be a number")
            }
            "--lr-max" => {
                pbt_cfg.lr_max = args
                    .next()
                    .expect("--lr-max needs a number")
                    .parse()
                    .expect("lr-max must be a number")
            }
            "--no-tcn" => tcn_channels_override = Some(None),
            other => sources.push(PathBuf::from(other)),
        }
    }
    if sources.is_empty() && split_path.is_none() {
        sources.push(PathBuf::from("armrest/data/inks"));
    }
    ready_epochs = ready_epochs.max(1);
    batch_size = batch_size.max(1);

    if stretch {
        let ckpt = stretch_ckpt.expect("--stretch requires --ckpt");
        let hp = Hyperparams {
            learning_rate: stretch_lr.expect("--stretch requires --lr"),
            use_sgd,
        };
        let (pairs, val_count) = if let Some(path) = &split_path {
            load_cached_split(path, val_count_arg.unwrap_or(0))
        } else {
            load_split(&sources, max_steps, max_samples, per_source_cap)
        };
        let (val_pairs, train_pairs) = pairs.split_at(val_count);
        run_stretch(
            stretch_gen,
            ready_epochs,
            stretch_member_id,
            hp,
            &ckpt,
            init_checkpoint.as_deref(),
            train_pairs,
            val_pairs,
            tcn_channels_override,
            batch_size,
        );
        return;
    }

    population = population.max(2);
    generations = generations.max(1);
    if pbt_cfg.lr_min > pbt_cfg.lr_max {
        std::mem::swap(&mut pbt_cfg.lr_min, &mut pbt_cfg.lr_max);
    }

    std::fs::create_dir_all(&workdir).expect("create pbt workdir");

    let (pairs, val_count) = load_split(&sources, max_steps, max_samples, per_source_cap);
    let train_n = pairs.len() - val_count;
    let cached_split = workdir.join("split.txt");
    save_pairs(&cached_split, &pairs).unwrap_or_else(|e| {
        panic!("failed to write cached PBT split {}: {e}", cached_split.display())
    });
    println!(
        "Wrote {} ({} samples) so workers reload this file, not the raw corpora",
        cached_split.display(),
        pairs.len()
    );

    let mut hp_rng = StdRng::seed_from_u64(seed);
    let mut members: Vec<Member> = (0..population)
        .map(|id| Member {
            id,
            hparams: sample_hparams(&mut hp_rng, &pbt_cfg),
            ckpt: workdir.join(format!("member_{id}.mpk")),
            val_cer: f64::INFINITY,
        })
        .collect();

    println!(
        "=== pbt run started === population={population} workers={} ready_epochs={ready_epochs} generations={generations} quantile={:.2} lr=[{:.4},{:.4}] batch={batch_size} train={train_n} val={val_count} {}",
        workers.max(1).min(population),
        pbt_cfg.quantile,
        pbt_cfg.lr_min,
        pbt_cfg.lr_max,
        match &init_checkpoint {
            Some(p) => format!("init={}", p.display()),
            None => "from scratch (no --init)".to_string(),
        }
    );
    println!(
        "Split: {train_n} train, {val_count} validation. architecture: HAT (Lodh et al. 2025)"
    );
    for m in &members {
        println!(
            "  member {}: lr={:.5} sgd={}",
            m.id, m.hparams.learning_rate, m.hparams.use_sgd
        );
    }

    let mut best_val = seed_best_val(
        &checkpoint_out,
        &pairs,
        val_count,
        tcn_channels_override,
    );
    drop(pairs);
    let mut best_member = 0usize;

    for gen in 0..generations {
        println!("=== gen {gen} start ===");
        let spec = StretchSpawn {
            gen,
            ready_epochs,
            shared_init: init_checkpoint.clone(),
            split: cached_split.clone(),
            val_count,
            batch_size,
        };
        let outcomes = run_generation_parallel(&members, workers, &spec);
        for out in outcomes {
            members[out.id].val_cer = out.val_cer;
        }

        let mut ranked: Vec<&Member> = members.iter().collect();
        ranked.sort_by(|a, b| {
            cer_sort_key(a.val_cer)
                .partial_cmp(&cer_sort_key(b.val_cer))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let gen_best = ranked[0];
        println!(
            "=== gen {gen} ready: best=m{} val_cer={:.4} ===",
            gen_best.id, gen_best.val_cer
        );
        save_if_best(
            &gen_best.ckpt,
            &checkpoint_out,
            gen_best.val_cer,
            gen_best.id,
            &mut best_val,
            &mut best_member,
        );

        if gen + 1 == generations {
            break;
        }
        let states: Vec<MemberState> = members
            .iter()
            .map(|m| MemberState {
                id: m.id,
                hparams: m.hparams,
                val_cer: m.val_cer,
            })
            .collect();
        let exploits = truncation_exploit(&states, &pbt_cfg, &mut hp_rng);
        if exploits.is_empty() {
            println!("=== gen {gen} exploit: none (quantile band empty) ===");
        }
        for e in &exploits {
            println!("{}", e.log_line(gen));
            let parent = &members[e.parent].ckpt;
            let victim = &members[e.victim].ckpt;
            if parent.exists() {
                if let Err(err) = std::fs::copy(parent, victim) {
                    eprintln!(
                        "warning: failed to copy m{} weights onto m{}: {err}",
                        e.parent, e.victim
                    );
                }
                if let Err(err) = copy_optimizer_sidecar(parent, victim) {
                    eprintln!(
                        "warning: failed to copy m{} optimizer onto m{}: {err}",
                        e.parent, e.victim
                    );
                }
            } else {
                eprintln!(
                    "warning: parent m{} has no checkpoint yet; skip weight copy",
                    e.parent
                );
                clear_optimizer_sidecar(victim);
            }
        }
        let mut states = states;
        apply_exploits(&mut states, &exploits);
        for s in states {
            members[s.id].hparams = s.hparams;
        }
    }

    println!(
        "\n=== pbt finished === best val_cer={:.4} (member {best_member}), saved to {}",
        best_val,
        checkpoint_out.display()
    );
    for m in &members {
        println!(
            "  member {}: val_cer={:.4} lr={:.5} sgd={}",
            m.id, m.val_cer, m.hparams.learning_rate, m.hparams.use_sgd
        );
    }
}
