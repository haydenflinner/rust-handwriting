//! Population-Based Training (Jaderberg et al. 2017; Ray Tune's truncation
//! mode is the widely-used recipe).
//!
//! rustuna / Optuna samplers cannot do this. They propose hyperparameters
//! for *independent* trials that start from the same init and die after a
//! budget — that's `bin/tune.rs`. PBT keeps a population of replicas
//! training in parallel and, at a ready interval, copies weights from
//! better members into worse ones and mutates the copied hyperparameters
//! so the next stretch of training continues from a stronger point, not
//! from scratch. There isn't an established Rust crate for that (Ray Tune
//! is the reference implementation, in Python), so the exploit/explore
//! step lives here, and `bin/pbt.rs` is the training loop around it.
//!
//! Truncation selection (Ray's default): rank by score, bottom `quantile`
//! of the population is replaced by a random member of the top `quantile`,
//! then hyperparameters are perturbed (`×1.2` / `×0.8`) or occasionally
//! resampled. Categorical knobs (`use_sgd`) can't be multiplied; they are
//! either kept from the parent or resampled.
//!
//! Optimizer state (AdamW moments / SGD momentum) is persisted next to
//! each member checkpoint (`member_N.optim.mpk`) and copied on exploit
//! when the parent and victim share the same optimizer family. A
//! SGD↔AdamW switch drops the sidecar so we don't load the wrong record.

use rand::{Rng, RngExt};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hyperparams {
    pub learning_rate: f64,
    pub use_sgd: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct PbtConfig {
    /// Fraction of the population treated as elite / as underperformers.
    /// Clamped so the two bands never overlap (`band <= n/2`).
    pub quantile: f64,
    /// Multiplicative perturbation: with equal probability LR is multiplied
    /// by this or by `1/this`… except the paper/Ray use 1.2 and 0.8, which
    /// are not reciprocals. We follow that pair, not a strict inverse.
    pub perturb_up: f64,
    pub perturb_down: f64,
    /// Chance, per hyperparameter, to resample from the original prior
    /// instead of perturbing the inherited value.
    pub resample_probability: f64,
    pub lr_min: f64,
    pub lr_max: f64,
}

impl Default for PbtConfig {
    fn default() -> Self {
        PbtConfig {
            quantile: 0.25,
            perturb_up: 1.2,
            perturb_down: 0.8,
            resample_probability: 0.25,
            lr_min: 1e-3,
            lr_max: 0.5,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemberState {
    pub id: usize,
    pub hparams: Hyperparams,
    pub val_cer: f64,
}

/// One truncation replacement: `victim` will load `parent`'s weights and
/// train next under `to` (a perturbation of `parent`'s hyperparameters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exploit {
    pub victim: usize,
    pub parent: usize,
    pub from: Hyperparams,
    pub to: Hyperparams,
}

impl Exploit {
    pub fn log_line(&self, gen: usize) -> String {
        format!(
            "=== gen {gen} exploit: m{} <- m{} lr {:.5} -> {:.5} sgd {} -> {} ===",
            self.victim,
            self.parent,
            self.from.learning_rate,
            self.to.learning_rate,
            self.from.use_sgd,
            self.to.use_sgd
        )
    }
}

pub fn sample_log_uniform<R: Rng>(rng: &mut R, low: f64, high: f64) -> f64 {
    let log_low = low.ln();
    let log_high = high.ln();
    (log_low + rng.random::<f64>() * (log_high - log_low)).exp()
}

pub fn sample_hparams<R: Rng>(rng: &mut R, cfg: &PbtConfig) -> Hyperparams {
    Hyperparams {
        learning_rate: sample_log_uniform(rng, cfg.lr_min, cfg.lr_max),
        use_sgd: rng.random::<bool>(),
    }
}

/// Perturb inherited hyperparameters (the "explore" half of PBT).
pub fn explore<R: Rng>(parent: Hyperparams, cfg: &PbtConfig, rng: &mut R) -> Hyperparams {
    let learning_rate = if rng.random::<f64>() < cfg.resample_probability {
        sample_log_uniform(rng, cfg.lr_min, cfg.lr_max)
    } else if rng.random::<bool>() {
        (parent.learning_rate * cfg.perturb_up).clamp(cfg.lr_min, cfg.lr_max)
    } else {
        (parent.learning_rate * cfg.perturb_down).clamp(cfg.lr_min, cfg.lr_max)
    };
    let use_sgd = if rng.random::<f64>() < cfg.resample_probability {
        rng.random::<bool>()
    } else {
        parent.use_sgd
    };
    Hyperparams {
        learning_rate,
        use_sgd,
    }
}

pub fn cer_sort_key(c: f64) -> f64 {
    if c.is_finite() {
        c
    } else {
        f64::INFINITY
    }
}

/// Rank the population and return the replacements to apply. Does not
/// mutate `members` — the caller copies checkpoints and then applies
/// [`apply_exploits`].
pub fn truncation_exploit<R: Rng>(
    members: &[MemberState],
    cfg: &PbtConfig,
    rng: &mut R,
) -> Vec<Exploit> {
    let n = members.len();
    if n < 2 {
        return Vec::new();
    }
    let mut band = ((n as f64) * cfg.quantile.clamp(0.0, 0.5)).floor() as usize;
    band = band.min(n / 2);
    if band == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        cer_sort_key(members[a].val_cer)
            .partial_cmp(&cer_sort_key(members[b].val_cer))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let elite = &order[..band];
    let bottom = &order[n - band..];

    let mut exploits = Vec::with_capacity(band);
    for &victim in bottom {
        let parent = elite[rng.random_range(0..band)];
        let to = explore(members[parent].hparams, cfg, rng);
        exploits.push(Exploit {
            victim,
            parent,
            from: members[victim].hparams,
            to,
        });
    }
    exploits
}

pub fn apply_exploits(members: &mut [MemberState], exploits: &[Exploit]) {
    for e in exploits {
        members[e.victim].hparams = e.to;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn cfg() -> PbtConfig {
        PbtConfig::default()
    }

    fn member(id: usize, cer: f64, lr: f64) -> MemberState {
        MemberState {
            id,
            hparams: Hyperparams {
                learning_rate: lr,
                use_sgd: false,
            },
            val_cer: cer,
        }
    }

    #[test]
    fn truncation_replaces_worst_from_best() {
        let members = [
            member(0, 1.00, 0.1),
            member(1, 0.20, 0.2), // best
            member(2, 0.90, 0.3),
            member(3, 0.30, 0.4), // second-best
            member(4, 1.50, 0.5), // worst
            member(5, 0.80, 0.6),
            member(6, 1.20, 0.7), // second-worst
            member(7, 0.50, 0.8),
        ];
        // n=8, quantile=0.25 -> band=2. elite={1,3}, bottom={6,4}.
        let mut rng = StdRng::seed_from_u64(1);
        let exploits = truncation_exploit(&members, &cfg(), &mut rng);
        assert_eq!(exploits.len(), 2);
        let mut victims: Vec<_> = exploits.iter().map(|e| e.victim).collect();
        victims.sort();
        assert_eq!(victims, vec![4, 6]);
        for e in &exploits {
            assert!(e.parent == 1 || e.parent == 3);
            assert!(e.to.learning_rate >= cfg().lr_min);
            assert!(e.to.learning_rate <= cfg().lr_max);
        }
    }

    #[test]
    fn small_population_with_tiny_quantile_does_nothing() {
        let members = [member(0, 1.0, 0.1), member(1, 0.5, 0.2)];
        let mut cfg = cfg();
        cfg.quantile = 0.1; // floor(2*0.1)=0
        let mut rng = StdRng::seed_from_u64(0);
        assert!(truncation_exploit(&members, &cfg, &mut rng).is_empty());
    }

    #[test]
    fn perturb_clamps_to_lr_range() {
        let mut rng = StdRng::seed_from_u64(0);
        let parent = Hyperparams {
            learning_rate: 0.5,
            use_sgd: true,
        };
        let mut cfg = cfg();
        cfg.resample_probability = 0.0; // always perturb
        for _ in 0..40 {
            let child = explore(parent, &cfg, &mut rng);
            assert!(child.learning_rate >= cfg.lr_min);
            assert!(child.learning_rate <= cfg.lr_max);
            assert!(child.use_sgd); // resample_probability=0 keeps parent's sgd
        }
    }

    #[test]
    fn nan_cer_ranks_as_worst() {
        let members = [
            member(0, 0.5, 0.1),
            member(1, f64::NAN, 0.2),
            member(2, 0.4, 0.3),
            member(3, 0.6, 0.4),
        ];
        let mut rng = StdRng::seed_from_u64(2);
        let exploits = truncation_exploit(&members, &cfg(), &mut rng);
        assert_eq!(exploits.len(), 1);
        assert_eq!(exploits[0].victim, 1);
        assert!(exploits[0].parent == 2 || exploits[0].parent == 0);
    }

    #[test]
    fn apply_exploits_updates_only_victims() {
        let mut members = vec![member(0, 0.2, 0.1), member(1, 1.0, 0.9)];
        let to = Hyperparams {
            learning_rate: 0.12,
            use_sgd: true,
        };
        let from = members[1].hparams;
        apply_exploits(
            &mut members,
            &[Exploit {
                victim: 1,
                parent: 0,
                from,
                to,
            }],
        );
        assert_eq!(members[0].hparams.learning_rate, 0.1);
        assert_eq!(members[1].hparams, to);
    }

    #[test]
    fn exploit_log_line_is_dashboard_parseable() {
        let e = Exploit {
            victim: 3,
            parent: 0,
            from: Hyperparams {
                learning_rate: 0.05,
                use_sgd: false,
            },
            to: Hyperparams {
                learning_rate: 0.06,
                use_sgd: false,
            },
        };
        assert_eq!(
            e.log_line(4),
            "=== gen 4 exploit: m3 <- m0 lr 0.05000 -> 0.06000 sgd false -> false ==="
        );
    }
}
