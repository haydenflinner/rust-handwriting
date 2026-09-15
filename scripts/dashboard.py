#!/usr/bin/env python3
"""Regenerate a local HTML dashboard from one or more train_loop*.log files —
pure stdlib, no dependencies, no server: the HTML file just re-reads itself
every 5s via <meta refresh>, so leaving it open in a browser tab gives a
live view of training without touching any running `train` process at all.

Supports multiple concurrent runs (see RUNS below) rendered as side-by-side
sections, for comparing parallel experiments (different LR schedules, etc.)
against each other.

For each log, only the CURRENT run within it is plotted (from the last
"=== single continuous run started" marker onward) — earlier runs in the
same file used different architectures/LRs and mixing them into one chart
would be misleading, not informative.
"""
import re
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "checkpoints" / "dashboard.html"

# (label, log file path) — add/remove entries here for whichever parallel
# runs are currently active. A missing file is just skipped, not an error.
RUNS = [
    ("G: digits-only (MNIST, 60k), no-TCN, from scratch", REPO / "checkpoints" / "train_loop_g.log"),
    ("H: full vocab +augmented +glyphs, no-TCN, from scratch", REPO / "checkpoints" / "train_loop_h.log"),
]
# Retired (TCN-era, killed after deciding to drop the TCN front-end):
# A/B/C/D (train_loop.log, train_loop_a/b/d.log) — logs left on disk, just
# no longer polled here.

# TPE hyperparameter search (bin/tune.rs, rustuna) — worker threads inside
# ONE process now share a single in-memory Study (a real shared trial
# queue), so there's one log again, not one per uncoordinated process (the
# old tune2.log/tune3.log from the earlier three-separate-processes
# approach are stale/frozen, left alone rather than deleted). Shown as a
# trial table, not a line chart: the log holds many short trials
# interleaved from multiple worker threads (each line tagged "[wN]"), not
# one continuous run.
TUNE_RUNS = [
    ("tune-digits: from scratch, MNIST (8k subsample)", REPO / "checkpoints" / "tune_digits.log"),
    ("tune-fullvocab: from scratch, expanded corpus (6k subsample)", REPO / "checkpoints" / "tune_fullvocab.log"),
]
# Retired: tune.log (fine-tune-from-checkpoint search, paused — no
# no-TCN checkpoint exists yet for it to fine-tune from).

TRIAL_START_RE = re.compile(r"^=== \[w(\d+)\] trial (\d+): lr=(\S+) sgd=(\S+) ===")
TUNE_RUN_START_RE = re.compile(r"^=== .*tune.*started", re.IGNORECASE)
TUNE_EPOCH_RE = re.compile(
    r"^\s*\[w(\d+)\] trial (\d+) epoch\s+(\d+):\s+mean_loss=(\S+)\s+val_cer=(\S+)\s+\(best this trial:\s+(\S+)\)"
)

EPOCH_RE = re.compile(
    r"^epoch\s+(\d+):\s+mean_loss=(\S+)\s+samples=(\d+)\s+skipped=(\d+)\s+"
    r"train_cer=(\S+)\s+val_cer=(\S+)\s+elapsed=([\d.]+)s(?:\s+lr=(\S+))?"
)
RUN_START_RE = re.compile(r"^=== single continuous run started")


def parse(log_path: pathlib.Path):
    if not log_path.exists():
        return []
    lines = log_path.read_text(errors="replace").splitlines()
    # Only the current run: everything after the LAST run-start marker.
    last_start = 0
    for i, line in enumerate(lines):
        if RUN_START_RE.match(line):
            last_start = i
    rows = []
    for line in lines[last_start:]:
        m = EPOCH_RE.match(line)
        if not m:
            continue
        epoch, loss, samples, skipped, train_cer, val_cer, elapsed, lr = m.groups()
        try:
            rows.append(
                dict(
                    epoch=int(epoch),
                    loss=float(loss) if loss != "NaN" else None,
                    train_cer=float(train_cer),
                    val_cer=float(val_cer),
                    elapsed=float(elapsed),
                    lr=float(lr) if lr is not None else None,
                )
            )
        except ValueError:
            continue
    return rows


def svg_line(rows, key, width=420, height=170, pad=34, color="#4f8fef", clamp_max=None):
    vals = [r[key] for r in rows if r[key] is not None]
    if not vals:
        return f'<svg width="{width}" height="{height}"></svg>'
    lo, hi = min(vals), max(vals)
    if clamp_max is not None:
        hi = min(hi, clamp_max) if hi > clamp_max else hi
    if hi - lo < 1e-9:
        hi = lo + 1.0
    n = len(rows)

    def x(i):
        return pad + (i / max(n - 1, 1)) * (width - 2 * pad)

    def y(v):
        v = min(v, hi) if clamp_max is not None else v
        return height - pad - ((v - lo) / (hi - lo)) * (height - 2 * pad)

    pts = []
    for i, r in enumerate(rows):
        if r[key] is None:
            continue
        pts.append(f"{x(i):.1f},{y(r[key]):.1f}")
    path = " ".join(pts)

    ticks = 3
    grid = []
    for t in range(ticks + 1):
        frac = t / ticks
        val = lo + frac * (hi - lo)
        yy = height - pad - frac * (height - 2 * pad)
        grid.append(
            f'<line x1="{pad}" y1="{yy:.1f}" x2="{width-pad}" y2="{yy:.1f}" '
            f'stroke="#e5e5e5" stroke-width="1"/>'
            f'<text x="2" y="{yy+4:.1f}" font-size="10" fill="#888">{val:.3f}</text>'
        )

    return (
        f'<svg width="{width}" height="{height}" style="background:#fff">'
        + "".join(grid)
        + f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
        + f'<text x="{pad}" y="14" font-size="11" fill="#555">{key}</text>'
        + "</svg>"
    )


def parse_tune(log_path: pathlib.Path):
    """Returns (trials, n_finished, total) — trials is a dict of trial_number
    -> trial dict (worker, params, best_val_cer, n_epochs, done). Multiple
    trials run concurrently (one per worker thread), so lines from
    different trials interleave — every line carries its own trial number,
    used as the dict key rather than assuming "the last trial that
    started" like a single-trial-at-a-time log would allow.

    Like the main RUNS parser, only the CURRENT run within the log is
    considered (everything from the last "=== ...tune...started" marker
    onward) — tune.log has been appended to across several separate `tune`
    process launches tonight, and each one's trial numbers restart at 0
    (a fresh rustuna Study each time), so without this boundary, trial 0
    from an old finished run and trial 0 from the current in-progress run
    collide in the same dict entry and corrupt each other's fields (this
    was live-diagnosed from a dashboard that showed "0 trials complete"
    and "?" lrs even though trials were actually completing)."""
    if not log_path.exists():
        return {}, 0, 0
    full_text = log_path.read_text(errors="replace")
    lines = full_text.splitlines()
    last_start = 0
    for i, line in enumerate(lines):
        if TUNE_RUN_START_RE.match(line):
            last_start = i
    text = "\n".join(lines[last_start:])
    trials = {}
    for line in text.splitlines():
        m = TRIAL_START_RE.match(line)
        if m:
            w, n, lr, sgd = m.groups()
            n = int(n)
            trials[n] = dict(
                number=n,
                worker=w,
                lr=lr,
                sgd=sgd,
                best_val_cer=float("inf"),
                epochs=0,
                done=False,
            )
            continue
        m = TUNE_EPOCH_RE.match(line)
        if m:
            w, n, _, _, _, best = m.groups()
            n = int(n)
            t = trials.setdefault(
                n,
                dict(
                    number=n, worker=w, lr="?",
                    sgd="?", best_val_cer=float("inf"), epochs=0, done=False,
                ),
            )
            t["best_val_cer"] = float(best)
            t["epochs"] += 1
    total = 0
    for line in text.splitlines():
        m = re.match(r"^Split:.*?(\d+) trials across (\d+) worker", line)
        if m:
            total = int(m.group(1))
        m2 = re.match(r"^trial\s+(\d+): best_val_cer=(\S+)", line)
        if m2 and int(m2.group(1)) in trials:
            trials[int(m2.group(1))]["done"] = True
            trials[int(m2.group(1))]["best_val_cer"] = float(m2.group(2))
    return trials, sum(1 for t in trials.values() if t["done"]), total or len(trials)


def render_tune_run(label: str, log_path: pathlib.Path) -> str:
    trials_dict, n_finished, total = parse_tune(log_path)
    trials = list(trials_dict.values())
    if not trials:
        return f"<section><h2>{label}</h2><p><i>no data yet ({log_path.name})</i></p></section>"
    finished = [t for t in trials if t["best_val_cer"] != float("inf")]
    best = min(finished, key=lambda t: t["best_val_cer"], default=None)
    rows = []
    for t in sorted(trials, key=lambda t: t["best_val_cer"]):
        tag = "" if t["done"] else f' &middot; <i>running (w{t["worker"]})</i>'
        bv = f'{t["best_val_cer"]:.4f}' if t["best_val_cer"] != float("inf") else "-"
        rows.append(
            f'<tr><td>{t["number"]}</td><td>{bv}</td><td>{t["lr"]}</td>'
            f'<td>{t["sgd"]}</td><td>{t["epochs"]}{tag}</td></tr>'
        )
    summary = (
        f'<p><b>{n_finished}/{total} trials complete</b>'
        + (f' &middot; best val_cer={best["best_val_cer"]:.4f} (trial {best["number"]})' if best else "")
        + "</p>"
    )
    table = (
        '<table style="border-collapse:collapse;font-size:12px" cellpadding="4">'
        "<tr style=\"text-align:left;border-bottom:1px solid #ccc\">"
        "<th>trial</th><th>best val_cer</th><th>lr</th><th>sgd</th><th>epochs</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    return f"<section><h2>{label}</h2>{summary}{table}</section>"


def render_run(label: str, log_path: pathlib.Path) -> str:
    rows = parse(log_path)
    if not rows:
        return f"<section><h2>{label}</h2><p><i>no data yet ({log_path.name})</i></p></section>"

    last = rows[-1]
    best_val = min((r["val_cer"] for r in rows), default=float("nan"))
    nan_count = sum(1 for r in rows if r["loss"] is None)
    lr_str = f'{last["lr"]:.5f}' if last.get("lr") is not None else "n/a"
    summary = (
        f'<p><b>Epoch {last["epoch"]}</b> &middot; '
        f'loss={last["loss"] if last["loss"] is not None else "NaN"} &middot; '
        f'train_cer={last["train_cer"]:.4f} &middot; '
        f'val_cer={last["val_cer"]:.4f} &middot; '
        f'best={best_val:.4f} &middot; '
        f'lr={lr_str} &middot; '
        f'{last["elapsed"]:.0f}s'
        + (f' &middot; <span style="color:#c33">{nan_count} NaN</span>' if nan_count else "")
        + "</p>"
    )
    has_lr = any(r.get("lr") is not None for r in rows)
    charts = (
        '<div style="display:flex;gap:10px;flex-wrap:wrap">'
        + svg_line(rows, "loss", color="#4f8fef")
        + svg_line(rows, "val_cer", color="#e0653a", clamp_max=2.0)
        + svg_line(rows, "train_cer", color="#8a5fd6", clamp_max=2.0)
        + (svg_line(rows, "lr", color="#2fa84f") if has_lr else "")
        + "</div>"
    )
    return f"<section><h2>{label}</h2>{summary}{charts}</section>"


def render():
    sections = [render_run(label, path) for label, path in RUNS]
    tune_sections = [render_tune_run(label, path) for label, path in TUNE_RUNS]
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="5">
<title>Training Dashboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 24px; color: #222; }}
h2 {{ font-size: 15px; margin: 0 0 4px 0; }}
section {{ margin-bottom: 28px; padding-bottom: 18px; border-bottom: 1px solid #eee; }}
table {{ background: #fff; }}
td, th {{ border-bottom: 1px solid #f0f0f0; }}
</style>
</head>
<body>
<h1 style="font-size:18px">hwr-model training — live view (auto-refreshes every 5s)</h1>
{''.join(sections)}
<h1 style="font-size:18px">TPE hyperparameter search (rustuna)</h1>
{''.join(tune_sections)}
</body></html>
"""
    OUT.write_text(html)


if __name__ == "__main__":
    render()
    print(f"wrote {OUT}")
