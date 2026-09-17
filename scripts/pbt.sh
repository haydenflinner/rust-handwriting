#!/bin/sh
# Population-Based Training launcher (bin/pbt.rs). Unlike `tune` (rustuna
# TPE: independent short trials, then a separate long `train` run), this
# *is* the long run: a population of replicas trains in parallel, and every
# READY_EPOCHS the worse ones copy weights from the better ones and mutate
# LR / SGD.
#
# Usage:
#   scripts/pbt.sh digits-smoke|digits|alnum|fullvocab
#
# Memory: never run 5 Wgpu workers against the raw corpora. Each worker
# is a full GPU process; they used to re-parse glyphs+augmented+MNIST
# (~900MB of text each) at every generation and swapped the machine to
# death. `bin/pbt` now caches a capped split.txt and defaults to 2
# workers / batch 32. HAT digits overnight used 8 workers = 8 members
# (~2.8GB each on a 36GB M3 Max). HAT fullvocab is heavier (word
# sequences + 224² image branch): ~4.5GB phys/worker, and 8-wide filled
# 16GB of swap then got SIGTERM. Stay at 1 worker / batch 16 on that
# corpus until the footprint drops.
#
# Curriculum: digits → alnum (numbers+letters) → fullvocab (words too).
# Each stage `--init`s from the previous `--out`. `--out` is only
# overwritten when a generation's best val CER beats both this run *and*
# any previously saved --out (sidecar, or an eval of the file on restart).
#
# Safe to stop (Ctrl+C) at any time: the current generation's member
# checkpoints live under checkpoints/pbt_<name>/.
cd "$(dirname "$0")/.."

# Point hwr-app at the current curriculum checkpoint. The app loads
# checkpoints/active.mpk from disk at startup (HWR_CHECKPOINT overrides).
link_active() {
  ckpt="$1"
  if [ -f "$ckpt" ]; then
    ln -sfn "$(basename "$ckpt")" checkpoints/active.mpk
  fi
}

MODE="${1:-fullvocab}"
case "$MODE" in
  digits-smoke)
    # Tiny HAT digits run so the dashboard moves in seconds, not an hour.
    # 192 traces (~10% val), 1-epoch stretches, 2 workers. Same architecture
    # and LR band as `digits`; not a substitute for the 32k run.
    OUT=checkpoints/pbt_digits_hat_smoke.mpk
    WORK=checkpoints/pbt_digits_hat_smoke
    LOG=checkpoints/pbt_digits_hat_smoke.log
    mkdir -p "$WORK"
    echo "=== pbt run started ($(date)) === digits HAT smoke (192 samples) ===" >> "$LOG"
    rm -f "$WORK"/member_*.mpk "$WORK"/member_*.cer "$WORK"/member_*.optim.mpk
    ./target/release/pbt --out "$OUT" --workdir "$WORK" \
      --population 4 --workers 2 --ready-epochs 1 --generations 40 \
      --max-samples 192 --batch-size 16 --no-tcn \
      --seed 2026 --lr-min 0.00003 --lr-max 0.0005 \
      synthetic/mnist_train.txt >> "$LOG" 2>&1
    ;;
  digits)
    # HAT (Lodh et al. 2025) is a different Module layout than the BiLSTM
    # `pbt_digits.mpk` (~7% CER). That file stays put; this run writes a
    # new --out so seed_best_val cannot lock us behind an unscorable LSTM
    # sidecar. Prefer an existing 32k HAT --out (resume); else the 192-sample
    # smoke HAT checkpoint (val_cer 0.2105).
    #
    # 8 workers = 8 members so a generation is one wave (the smoke's 2
    # workers queued half the population; 4-wide was still headroom).
    # 8-epoch stretches so spawn / GPU-init / val is a small fraction of
    # wall time. ~2.8GB/worker on a 36GB M3 Max; 10-wide (~28GB) plus
    # OS/compressor is swap territory.
    OUT=checkpoints/pbt_digits_hat.mpk
    WORK=checkpoints/pbt_digits_hat
    LOG=checkpoints/pbt_digits_hat.log
    if [ -f "$OUT" ]; then
      INIT="$OUT"
    else
      INIT=checkpoints/pbt_digits_hat_smoke.mpk
    fi
    mkdir -p "$WORK"
    # Leave checkpoints/active.mpk on the LSTM digits model so the app
    # still recognizes until this run produces a HAT --out.
    echo "=== pbt run started ($(date)) === digits HAT (32k, 8 workers, 8-epoch stretches, init=$INIT) ===" >> "$LOG"
    rm -f "$WORK"/member_*.mpk "$WORK"/member_*.cer "$WORK"/member_*.optim.mpk "$WORK"/member_*.optim.kind
    # Transformer-scale LR (paper AdamW 1e-4), not the LSTM band 0.008–0.04.
    ./target/release/pbt --out "$OUT" --workdir "$WORK" \
      --init "$INIT" \
      --population 8 --workers 8 --ready-epochs 8 --generations 40 \
      --max-samples 32000 --batch-size 32 --no-tcn \
      --seed 2026 --lr-min 0.00003 --lr-max 0.0005 \
      synthetic/mnist_train.txt >> "$LOG" 2>&1
    ;;
  alnum)
    # HAT letters+digits. LSTM `pbt_alnum.mpk` / `pbt_digits.mpk` will not
    # load into this Module — new --out, init from HAT digits (or this
    # run's own --out on restart). Same 8-wide / 8-epoch / transformer LR
    # band as digits.
    OUT=checkpoints/pbt_alnum_hat.mpk
    WORK=checkpoints/pbt_alnum_hat
    LOG=checkpoints/pbt_alnum_hat.log
    if [ -f "$OUT" ]; then
      INIT="$OUT"
    elif [ -f checkpoints/pbt_digits_hat.mpk ]; then
      INIT=checkpoints/pbt_digits_hat.mpk
    else
      INIT=checkpoints/pbt_digits_hat_smoke.mpk
    fi
    mkdir -p "$WORK"
    # Keep the app on LSTM digits; a mid-adaptation HAT alnum snapshot is
    # worse for live digit testing than the 7% LSTM digits checkpoint.
    echo "=== pbt run started ($(date)) === alnum HAT (numbers+letters, 8 workers, init=$INIT) ===" >> "$LOG"
    rm -f "$WORK"/member_*.mpk "$WORK"/member_*.cer "$WORK"/member_*.optim.mpk "$WORK"/member_*.optim.kind
    # Numbers + letters, init from HAT digits. --per-source-cap keeps MNIST
    # from drowning glyphs.txt (8k letter-words vs 60k digits). Calibration
    # is tiny (355) so it is included in full; it is the user's own ink.
    # eights_real/aug overweight the user's handwritten 8 (the live digits
    # model misses it; calibration only has 4 real 8s).
    ./target/release/pbt --out "$OUT" --workdir "$WORK" \
      --init "$INIT" \
      --population 8 --workers 8 --ready-epochs 8 --generations 30 \
      --per-source-cap 8000 --batch-size 32 --no-tcn \
      --seed 2027 --lr-min 0.00003 --lr-max 0.0005 \
      synthetic/glyphs.txt synthetic/mnist_train.txt \
      "/Users/wow/Library/Application Support/hwr/calibration.txt" \
      synthetic/eights_real.txt synthetic/eights_aug.txt \
      >> "$LOG" 2>&1
    ;;
  fullvocab)
    # HAT words+alnum. LSTM `pbt_fullvocab.mpk` stays put (incompatible).
    OUT=checkpoints/pbt_fullvocab_hat.mpk
    WORK=checkpoints/pbt_fullvocab_hat
    LOG=checkpoints/pbt_fullvocab_hat.log
    if [ -f "$OUT" ]; then
      INIT="$OUT"
    elif [ -f checkpoints/pbt_alnum_hat.mpk ]; then
      INIT=checkpoints/pbt_alnum_hat.mpk
    else
      INIT=checkpoints/pbt_digits_hat.mpk
    fi
    mkdir -p "$WORK"
    # Keep the app on LSTM digits until a HAT --out is worth pointing at.
    echo "=== pbt run started ($(date)) === fullvocab HAT (1 worker, batch 16, init=$INIT) ===" >> "$LOG"
    rm -f "$WORK"/member_*.mpk "$WORK"/member_*.cer "$WORK"/member_*.optim.mpk "$WORK"/member_*.optim.kind
    # eights_*: the live digits model misses the user's handwritten 8
    # (only 4 real samples in calibration). Those four plus 64× geometry
    # scrambles are their own source so MNIST 8s cannot drown them.
    # 1 worker / batch 16: measured 4.5GB phys, zero new swapouts. 8-wide
    # at batch 32 swapped ~16GB and was SIGTERM'd.
    ./target/release/pbt --out "$OUT" --workdir "$WORK" \
      --init "$INIT" \
      --population 2 --workers 1 --ready-epochs 8 --generations 20 \
      --per-source-cap 6000 --batch-size 16 --no-tcn \
      --seed 2028 --lr-min 0.00003 --lr-max 0.0005 \
      armrest/data/inks \
      "/Users/wow/Library/Application Support/hwr/calibration.txt" \
      synthetic/augmented.txt synthetic/glyphs.txt synthetic/mnist_train.txt \
      synthetic/eights_real.txt synthetic/eights_aug.txt \
      >> "$LOG" 2>&1
    ;;
  *)
    echo "usage: $0 digits-smoke|digits|alnum|fullvocab" >&2
    exit 1
    ;;
esac
echo "=== pbt process exited (code $?) at $(date) ===" >> "$LOG"
