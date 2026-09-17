#!/bin/sh
# Run `train` as ONE continuous process, no scripted restarts.
#
# The earlier restart-loop (train_loop.sh) existed to work around a
# per-process performance regression that made epoch time grow ~140s/epoch
# within one long-running process. That was tied to the old `FusedBiLstmLayer`
# internals issuing one GPU dispatch per op per timestep; it's since been
# replaced with a genuine on-GPU fused CubeCL kernel (fused_lstm_kernel.rs)
# that runs the whole per-timestep recurrence in a single kernel launch,
# confirmed via a 6-epoch real-corpus smoke test to hold a flat ~20s/epoch
# with zero growth. With that root cause gone, periodic restarts only cost
# us something (Adam's momentum resets every restart — optimizer state
# isn't checkpointed, only model weights) for no benefit, so this lets Adam
# run uninterrupted instead.
#
# LR dropped 0.1 -> 0.03 as the long-run stable target: the 0.1 run silently
# diverged to NaN loss after ~47 clean epochs (~7,000 steps) and sat there
# for 120+ more epochs before anyone noticed — gradient-norm clipping
# doesn't catch this since it bounds the gradient, not an already-corrupted
# optimizer/weight state. 0.03 never diverged in any of tonight's
# many-thousand-step runs. `train.rs` also now skips (rather than applies)
# any batch whose loss comes back non-finite, so a future recurrence can't
# silently waste hours again.
#
# --peak-lr/--warmup-steps/--decay-steps: a one-cycle LR schedule (standard,
# well-established technique — not another ad hoc guess) layered on top of
# that stable 0.03 target: ramp up to a genuinely aggressive peak (0.3, 3x
# anything run stably tonight) over the first 100 steps, then back down to
# 0.03 over the next 2000, then flat. The aggressive peak is a deliberate,
# bounded push to try to break CTC's "predict blank everywhere" local
# optimum, which many thousands of steps at fixed 0.01-0.1 LRs never did;
# the short warmup keeps it from hitting full strength on step one (which
# risked instant NaN before the schedule had a chance to do anything), and
# the non-finite-loss skip above backstops any individual step that still
# goes bad regardless.
#
# No automatic restart-on-crash: if this process dies or hangs, that's a
# real signal something needs attention, not something to paper over by
# silently restarting. On exit (crash or otherwise) this appends a clear
# "=== process exited ===" marker to the log so it shows up for whoever's
# watching, then stops — it does not loop.
#
# Safe to stop (Ctrl+C / kill) at any time: `train` only overwrites
# checkpoints/model.mpk when validation CER actually improves (and seeds
# that comparison from the *loaded* checkpoint's real val CER, not infinity
# — so a restart can never silently save a worse model over a better one).
cd "$(dirname "$0")/.."

CKPT=checkpoints/model.mpk
LOG=checkpoints/train_loop.log

echo "=== single continuous run started ($(date)) ===" >> "$LOG"
if [ -f "$CKPT" ]; then
  ./target/release/train --out "$CKPT" --init "$CKPT" --epochs 100000 --lr 0.03 \
    --peak-lr 0.3 --warmup-steps 100 --decay-steps 2000 \
    --batch-size 16 --max-steps 150 --no-tcn \
    armrest/data/inks "/Users/wow/Library/Application Support/hwr/calibration.txt" \
    >> "$LOG" 2>&1
else
  ./target/release/train --out "$CKPT" --epochs 100000 --lr 0.03 \
    --peak-lr 0.3 --warmup-steps 100 --decay-steps 2000 \
    --batch-size 16 --max-steps 150 --no-tcn \
    armrest/data/inks "/Users/wow/Library/Application Support/hwr/calibration.txt" \
    >> "$LOG" 2>&1
fi
echo "=== process exited (code $?) at $(date) ===" >> "$LOG"
