#!/bin/sh
# Overnight HAT curriculum: digits until val CER plateaus, then alnum
# (letters+digits), then fullvocab if time remains. Stops at 10:00 AM
# America/New_York on 2026-09-17. Safe to SIGTERM: member checkpoints
# live under checkpoints/pbt_*_hat/.
#
# Detach this (start_new_session) so an aborted Cursor terminal does not
# take training with it — that is why the 4-wide 32k run vanished from
# Activity Monitor.
cd "$(dirname "$0")/.."

LOG=checkpoints/night_train.log
DEADLINE=$(date -j -f "%Y-%m-%d %H:%M:%S" "2026-09-17 10:00:00" "+%s")

still_time() {
  [ "$(date +%s)" -lt "$DEADLINE" ]
}

kill_pbt() {
  # Parent + stretch children. pbt.sh does not trap SIGTERM itself.
  pkill -TERM -f '/target/release/pbt ' 2>/dev/null || true
  pkill -TERM -f 'scripts/pbt.sh (digits|alnum|fullvocab)' 2>/dev/null || true
  sleep 2
  pkill -KILL -f '/target/release/pbt ' 2>/dev/null || true
}

plateaued() {
  # 0 = plateau (stall), 1 = still learning / not enough gens.
  python3 - "$1" <<'PY'
import re, sys
from pathlib import Path
path = Path(sys.argv[1])
min_gens = 6
stall_gens = 4
min_delta = 0.005
if not path.exists():
    raise SystemExit(1)
text = path.read_text(errors="replace")
chunk = text.split("=== pbt run started")[-1]
ready = []
for line in chunk.splitlines():
    m = re.match(r"^=== gen (\d+) ready: best=m\d+ val_cer=(\S+)", line)
    if m:
        try:
            ready.append((int(m.group(1)), float(m.group(3))))
        except ValueError:
            pass
if len(ready) < min_gens:
    raise SystemExit(1)
bests = []
running = float("inf")
for _, cer in ready:
    running = min(running, cer)
    bests.append(running)
# Still blank-dominated: keep digits, don't jump to letters.
if bests[-1] >= 0.999:
    raise SystemExit(1)
# No improvement of min_delta over the last stall_gens completed gens.
if bests[-stall_gens - 1] - bests[-1] < min_delta:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

heartbeat() {
  rss=$(ps -axo rss,command | awk '/target\/release\/pbt/ {s+=$1; n++} END {printf "pbt_n=%d rss_gb=%.2f", n, s/1024/1024}')
  echo "=== night heartbeat $(date) stage=$STAGE $rss ===" >> "$LOG"
}

run_stage() {
  STAGE="$1"
  stage_log="$2"
  echo "=== night starting stage=$STAGE at $(date) ===" >> "$LOG"
  ./scripts/pbt.sh "$STAGE" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if ! still_time; then
      echo "=== night deadline 10:00 AM; stopping $STAGE ===" >> "$LOG"
      kill_pbt
      wait "$pid" 2>/dev/null || true
      return 2
    fi
    if plateaued "$stage_log"; then
      echo "=== night plateau on $STAGE; advancing curriculum ===" >> "$LOG"
      kill_pbt
      wait "$pid" 2>/dev/null || true
      return 0
    fi
    heartbeat
    sleep 30
  done
  wait "$pid"
  rc=$?
  echo "=== night pbt.sh $STAGE exited rc=$rc at $(date) ===" >> "$LOG"
  # Finished its generation budget. Treat a stall as plateau; otherwise
  # the caller may re-enter the same stage (pbt.sh --init from --out).
  if plateaued "$stage_log"; then
    return 0
  fi
  return 1
}

echo "=== night_train started $(date); deadline=$(date -r "$DEADLINE") ===" >> "$LOG"

# digits until plateau (re-enter while still improving and time remains)
while still_time; do
  run_stage digits checkpoints/pbt_digits_hat.log
  rc=$?
  [ "$rc" -eq 2 ] && break
  [ "$rc" -eq 0 ] && break
done

while still_time; do
  run_stage alnum checkpoints/pbt_alnum_hat.log
  rc=$?
  [ "$rc" -eq 2 ] && break
  [ "$rc" -eq 0 ] && break
done

while still_time; do
  run_stage fullvocab checkpoints/pbt_fullvocab_hat.log
  rc=$?
  [ "$rc" -eq 2 ] && break
  [ "$rc" -eq 0 ] && break
done

kill_pbt
echo "=== night_train finished $(date) ===" >> "$LOG"
