#!/bin/sh
# Restart `train` every 50 epochs, each time resuming from the checkpoint
# the previous run just saved.
#
# Originally this restarted every 8 epochs to work around a per-process
# performance regression in Burn's fusion/dispatch operation queue (epoch
# time was growing ~140s/epoch within one long-running process, tied to
# `FusedBiLstmLayer`'s internals still issuing one GPU dispatch per op per
# timestep). That internal implementation was replaced with a genuine
# on-GPU fused CubeCL kernel (see `fused_lstm_kernel.rs`) that runs the
# whole per-timestep recurrence in a single kernel launch — confirmed via a
# 6-epoch real-corpus smoke test to hold a flat ~20s/epoch with zero growth
# (vs. the old path's ~150s-750s+ and climbing), so the tight restart
# cadence is no longer needed for that reason. Restarting every 50 epochs
# now is just a residual safety net (an untested-at-scale process is
# cheaper to recover from than to debug at 4am) while giving Adam's
# momentum (not persisted across restarts — only model weights are
# checkpointed) far more room to build than the original 8-epoch chunks did
# — relevant since LR=1e-3 measurably failed to break CTC's "predict blank
# everywhere" local optimum even after ~745 steps spread across 1-epoch
# restarts, while LR=0.1 broke symmetry within ~150 steps uninterrupted.
#
# Safe to stop (Ctrl+C / kill) at any time: `train` only overwrites
# checkpoints/model.mpk when validation CER actually improves (and seeds
# that comparison from the *loaded* checkpoint's real val CER, not infinity
# — so a restart can never silently save a worse model over a better one).
set -e
cd "$(dirname "$0")/.."

CKPT=checkpoints/model.mpk
LOG=checkpoints/train_loop.log

i=0
while true; do
  i=$((i + 1))
  echo "=== restart $i ($(date)) ===" >> "$LOG"
  if [ -f "$CKPT" ]; then
    ./target/release/train --out "$CKPT" --init "$CKPT" --epochs 50 --lr 0.1 \
      --batch-size 16 --max-steps 150 \
      armrest/data/inks "/Users/wow/Library/Application Support/hwr/calibration.txt" \
      >> "$LOG" 2>&1
  else
    ./target/release/train --out "$CKPT" --epochs 50 --lr 0.1 \
      --batch-size 16 --max-steps 150 \
      armrest/data/inks "/Users/wow/Library/Application Support/hwr/calibration.txt" \
      >> "$LOG" 2>&1
  fi
done
