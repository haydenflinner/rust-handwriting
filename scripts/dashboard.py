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

When the log contains per-epoch `probe:` / `layers:` / `weights:` / `grads:`
lines (emitted by `train` from `hwr_model::probe`), extra charts show CTC
blank-mass, layer activation RMS, and gradient flow — the lens that tells
"loss is dropping because it's collapsing to blank" apart from "it's
actually learning character identity".

`bin/pbt.rs` logs are plotted in a separate Population-based training
section (per-member val_cer over time, current HPs, recent exploits).
"""
import re
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
OUT = REPO / "checkpoints" / "dashboard.html"

# Live view is PBT (curriculum fine-tunes). The from-scratch `train` loops
# and rustuna TPE searches are retired — logs stay on disk, just not
# plotted at the top of the dashboard.
RUNS = []
# Retired from-scratch train loops (left on disk):
#   G: train_loop_g.log  digits-only MNIST, no-TCN, from scratch
#   H: train_loop_h.log  full vocab +augmented +glyphs, no-TCN, from scratch
# TCN-era A/B/C/D (train_loop.log, train_loop_a/b/d.log) already retired.

TUNE_RUNS = []
# Retired: tune_digits.log / tune_fullvocab.log (from-scratch TPE),
# tune.log (fine-tune-from-checkpoint search, paused).

# Population-Based Training (bin/pbt.rs) — parallel replicas that copy
# weights from better members and mutate HPs every ready interval. Live
# run is HAT digits 32k (init from smoke). LSTM checkpoints will not load.
# A missing file is skipped.
PBT_RUNS = [
    ("pbt-digits HAT (32k, overnight)", REPO / "checkpoints" / "pbt_digits_hat.log"),
    ("pbt-alnum HAT (letters+digits)", REPO / "checkpoints" / "pbt_alnum_hat.log"),
    ("pbt-fullvocab HAT", REPO / "checkpoints" / "pbt_fullvocab_hat.log"),
    ("pbt-digits HAT smoke (192, done)", REPO / "checkpoints" / "pbt_digits_hat_smoke.log"),
    ("pbt-fullvocab (LSTM, paused)", REPO / "checkpoints" / "pbt_fullvocab.log"),
    ("pbt-alnum (LSTM, paused)", REPO / "checkpoints" / "pbt_alnum.log"),
    ("pbt-digits (LSTM)", REPO / "checkpoints" / "pbt_digits.log"),
]

PBT_START_RE = re.compile(r"^=== pbt run started")
PBT_EPOCH_RE = re.compile(
    r"^\s*\[m(\d+)\] gen (\d+) epoch\s+(\d+):\s+mean_loss=(\S+)\s+val_cer=(\S+)"
    r"(?:\s+greedy_cer=\S+\s+beam_cer=\S+)?"
    r"\s+lr=(\S+)"
    r"(?:\s+\(total_epoch=(\d+)\))?"
)
PBT_READY_RE = re.compile(r"^=== gen (\d+) ready: best=m(\d+) val_cer=(\S+)")
PBT_EXPLOIT_RE = re.compile(
    r"^=== gen (\d+) exploit: m(\d+) <- m(\d+) lr (\S+) -> (\S+) sgd (\S+) -> (\S+)"
)
PBT_STRETCH_RE = re.compile(
    r"^\s*\[m(\d+)\] stretch done val_cer=(\S+) lr=(\S+) sgd=(\S+)"
)

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
KV_RE = re.compile(r"(\S+)=(\S+)")
SAMPLE_RE = re.compile(r'"((?:\\.|[^"\\])*)"\s*->\s*"((?:\\.|[^"\\])*)"')
LAYER_COLORS = ["#4f8fef", "#e0653a", "#8a5fd6", "#2fa84f", "#c9a227", "#5aa", "#888", "#c45"]


def parse_pairs(rest: str) -> dict:
    out = {}
    for m in KV_RE.finditer(rest):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return out


def unescape(s: str) -> str:
    return s.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")


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
        if m:
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
            continue
        if not rows:
            continue
        # Indented probe lines belonging to the epoch just parsed.
        if line.startswith("  probe:"):
            rows[-1]["probe"] = parse_pairs(line[len("  probe:") :])
        elif line.startswith("  layers:"):
            rows[-1]["layers"] = parse_pairs(line[len("  layers:") :])
        elif line.startswith("  weights:"):
            rows[-1]["weights"] = parse_pairs(line[len("  weights:") :])
        elif line.startswith("  grads:"):
            rows[-1]["grads"] = parse_pairs(line[len("  grads:") :])
        elif line.startswith("  classes:"):
            rows[-1]["classes"] = parse_pairs(line[len("  classes:") :])
        elif line.startswith("  samples:"):
            rows[-1]["samples"] = [
                (unescape(a), unescape(b)) for a, b in SAMPLE_RE.findall(line)
            ]
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


def _series_vals(rows, series):
    """series is either a row key ('loss') or (dict_key, inner_key) for nested probe maps."""
    vals = []
    for r in rows:
        if isinstance(series, tuple):
            bag, inner = series
            v = (r.get(bag) or {}).get(inner)
        else:
            v = r.get(series)
        vals.append(v)
    return vals


def svg_multi(rows, series, title, width=640, height=190, pad=34, clamp_max=None):
    """Multi-line chart. `series` is a list of (label, key) where key is a
    row field or (dict_field, inner_key)."""
    if not rows:
        return ""
    all_vals = []
    extracted = []
    for label, key in series:
        vals = _series_vals(rows, key)
        extracted.append((label, vals))
        all_vals.extend(v for v in vals if v is not None)
    if not all_vals:
        return ""
    lo, hi = min(all_vals), max(all_vals)
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

    polylines = []
    legend = []
    lx = pad
    for i, (label, vals) in enumerate(extracted):
        color = LAYER_COLORS[i % len(LAYER_COLORS)]
        pts = []
        for j, v in enumerate(vals):
            if v is None:
                continue
            pts.append(f"{x(j):.1f},{y(v):.1f}")
        if pts:
            polylines.append(
                f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
        legend.append(
            f'<rect x="{lx}" y="{height - 16}" width="8" height="8" fill="{color}"/>'
            f'<text x="{lx+11}" y="{height - 9}" font-size="10" fill="#555">{label}</text>'
        )
        lx += 8 + 7 * len(label) + 18

    return (
        f'<svg width="{width}" height="{height}" style="background:#fff" '
        f'role="img" aria-label="{title}">'
        + "".join(grid)
        + "".join(polylines)
        + f'<text x="{pad}" y="14" font-size="11" fill="#555">{title}</text>'
        + "".join(legend)
        + "</svg>"
    )


def collect_keys(rows, bag_key):
    names = []
    seen = set()
    for r in rows:
        for k in r.get(bag_key) or {}:
            if k not in seen:
                seen.add(k)
                names.append(k)
    return names


def html_heatmap(rows, bag_key, title):
    """Epoch × name grid, color intensity = value / max. Omits itself if no data."""
    names = collect_keys(rows, bag_key)
    if not names:
        return ""
    vals = []
    for r in rows:
        bag = r.get(bag_key) or {}
        vals.extend(bag.get(n) for n in names if bag.get(n) is not None)
    if not vals:
        return ""
    hi = max(vals) or 1.0
    # Downsample columns if the run is long so the table stays readable.
    stride = max(1, (len(rows) + 39) // 40)
    shown = list(enumerate(rows))[::stride]
    if shown[-1][0] != len(rows) - 1:
        shown.append((len(rows) - 1, rows[-1]))
    head = "".join(
        f'<th style="font-weight:400;color:#888;font-size:10px">{r["epoch"]}</th>'
        for _, r in shown
    )
    body = []
    for name in names:
        cells = []
        for _, r in shown:
            v = (r.get(bag_key) or {}).get(name)
            if v is None:
                cells.append('<td></td>')
                continue
            t = min(1.0, v / hi)
            # Blue intensity on white; text stays readable at both ends.
            bg = f"rgba(79,143,239,{0.08 + 0.72 * t:.2f})"
            cells.append(
                f'<td title="{name} epoch {r["epoch"]}: {v:.4f}" '
                f'style="background:{bg};font-size:10px;text-align:center;'
                f'padding:3px 4px">{v:.2f}</td>'
            )
        body.append(
            f'<tr><th style="text-align:left;padding-right:8px;font-weight:500;'
            f'font-size:11px">{name}</th>{"".join(cells)}</tr>'
        )
    return (
        f'<div style="margin-top:8px"><div style="font-size:11px;color:#555;'
        f'margin-bottom:4px">{title} (max {hi:.3f}'
        + (f", every {stride} epochs" if stride > 1 else "")
        + ")</div>"
        f'<table style="border-collapse:collapse;font-size:11px">'
        f'<tr><th></th>{head}</tr>{"".join(body)}</table></div>'
    )


def html_samples(row):
    samples = row.get("samples") or []
    if not samples:
        return ""
    rows_html = []
    for gold, pred in samples:
        gold_e = gold.replace("&", "&amp;").replace("<", "&lt;")
        pred_e = pred.replace("&", "&amp;").replace("<", "&lt;")
        empty = ' style="color:#c33"' if pred == "" else ""
        rows_html.append(
            f'<tr><td style="font-family:ui-monospace,monospace">{gold_e}</td>'
            f'<td{empty} style="font-family:ui-monospace,monospace">{pred_e or "(empty)"}</td></tr>'
        )
    return (
        '<div style="margin-top:8px"><div style="font-size:11px;color:#555;'
        'margin-bottom:4px">greedy decode on probe batch (gold → pred)</div>'
        '<table style="border-collapse:collapse;font-size:12px" cellpadding="4">'
        '<tr style="text-align:left;border-bottom:1px solid #ccc">'
        "<th>gold</th><th>pred</th></tr>"
        + "".join(rows_html)
        + "</table></div>"
    )


def html_classes(row):
    classes = row.get("classes") or {}
    if not classes:
        return ""
    parts = []
    for name, v in list(classes.items())[:8]:
        parts.append(f"{name}={v:.3f}")
    return (
        f'<p style="font-size:12px;color:#555;margin:6px 0 0 0">'
        f'mean softmax mass: {" · ".join(parts)}</p>'
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


def _parse_pbt_slice(lines):
    """Parse one PBT run (from a start marker through the following lines)."""
    by_t = {}
    ready = []
    exploits = []
    latest = {}
    for line in lines:
        m = PBT_EPOCH_RE.match(line)
        if m:
            mid, gen, epoch, loss, val_cer, lr, total = m.groups()
            mid = int(mid)
            try:
                val = float(val_cer)
                lr_v = float(lr)
            except ValueError:
                continue
            if total is not None:
                t = int(total)
            else:
                t = int(gen) * 1000 + int(epoch)
            by_t.setdefault(t, {})[f"m{mid}"] = val
            by_t[t][f"m{mid}_lr"] = lr_v
            continue
        m = PBT_READY_RE.match(line)
        if m:
            gen, mid, val = m.groups()
            try:
                ready.append(dict(gen=int(gen), member=int(mid), val_cer=float(val)))
            except ValueError:
                pass
            continue
        m = PBT_EXPLOIT_RE.match(line)
        if m:
            gen, victim, parent, lr_from, lr_to, sgd_from, sgd_to = m.groups()
            exploits.append(
                dict(
                    gen=int(gen),
                    victim=int(victim),
                    parent=int(parent),
                    lr_from=lr_from,
                    lr_to=lr_to,
                    sgd_from=sgd_from,
                    sgd_to=sgd_to,
                )
            )
            continue
        m = PBT_STRETCH_RE.match(line)
        if m:
            mid, val, lr, sgd = m.groups()
            try:
                latest[int(mid)] = dict(val_cer=float(val), lr=lr, sgd=sgd)
            except ValueError:
                pass
    if not by_t and not ready:
        return None
    times = sorted(by_t)
    members = []
    seen = set()
    for t in times:
        for k in by_t[t]:
            if k.startswith("m") and not k.endswith("_lr") and k not in seen:
                seen.add(k)
                members.append(k)
    rows = []
    for t in times:
        row = dict(total_epoch=t)
        row.update(by_t[t])
        rows.append(row)
    return dict(rows=rows, members=members, ready=ready, exploits=exploits, latest=latest)


def parse_pbt(log_path: pathlib.Path):
    """Per-member val_cer over total_epoch, plus generation-best and exploit events.

    Prefer the latest run. If that run has not logged an epoch yet (GPU
    warmup after a resume), fall back to the previous run that did so the
    dashboard is not blank for an hour.
    """
    if not log_path.exists():
        return None
    lines = log_path.read_text(errors="replace").splitlines()
    starts = [i for i, line in enumerate(lines) if PBT_START_RE.match(line)] or [0]
    for start in reversed(starts):
        data = _parse_pbt_slice(lines[start:])
        if data:
            return data
    return None


def render_pbt_run(label: str, log_path: pathlib.Path) -> str:
    data = parse_pbt(log_path)
    if not data:
        return f"<section><h2>{label}</h2><p><i>no data yet ({log_path.name})</i></p></section>"
    rows = data["rows"]
    ready = data["ready"]
    best = min((r["val_cer"] for r in ready), default=float("nan"))
    last_ready = ready[-1] if ready else None
    summary = "<p>"
    if last_ready:
        summary += (
            f'<b>gen {last_ready["gen"]}</b> &middot; '
            f'best this gen=m{last_ready["member"]} val_cer={last_ready["val_cer"]:.4f}'
        )
        if best == best:
            summary += f" &middot; best ever={best:.4f}"
        summary += f' &middot; {len(data["exploits"])} exploits'
    summary += "</p>"
    charts = ""
    if rows and data["members"]:
        charts = (
            '<div style="display:flex;gap:10px;flex-wrap:wrap">'
            + svg_multi(
                rows,
                [(k, k) for k in data["members"]],
                "member val_cer over total epoch",
                width=640,
                clamp_max=2.0,
            )
            + "</div>"
        )
    pop_rows = []
    for mid in sorted(data["latest"]):
        s = data["latest"][mid]
        pop_rows.append(
            f'<tr><td>m{mid}</td><td>{s["val_cer"]:.4f}</td>'
            f'<td>{s["lr"]}</td><td>{s["sgd"]}</td></tr>'
        )
    table = ""
    if pop_rows:
        table = (
            '<table style="border-collapse:collapse;font-size:12px;margin-top:8px" cellpadding="4">'
            '<tr style="text-align:left;border-bottom:1px solid #ccc">'
            "<th>member</th><th>last val_cer</th><th>lr</th><th>sgd</th></tr>"
            + "".join(pop_rows)
            + "</table>"
        )
    exploit_bits = []
    for e in data["exploits"][-8:]:
        exploit_bits.append(
            f'g{e["gen"]}: m{e["victim"]}&lt;-m{e["parent"]} lr {e["lr_from"]}&rarr;{e["lr_to"]}'
        )
    exploit_html = ""
    if exploit_bits:
        exploit_html = (
            f'<p style="font-size:12px;color:#555;margin:6px 0 0 0">'
            f'recent exploits: {" · ".join(exploit_bits)}</p>'
        )
    return f"<section><h2>{label}</h2>{summary}{charts}{table}{exploit_html}</section>"


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
        + (f' &middot; blank={last["probe"]["blank"]:.3f}' if last.get("probe") and "blank" in last["probe"] else "")
        + (f' &middot; <span style="color:#c33">{nan_count} NaN</span>' if nan_count else "")
        + "</p>"
    )
    has_lr = any(r.get("lr") is not None for r in rows)
    has_probe = any(r.get("probe") for r in rows)
    charts = (
        '<div style="display:flex;gap:10px;flex-wrap:wrap">'
        + svg_line(rows, "loss", color="#4f8fef")
        + svg_line(rows, "val_cer", color="#e0653a", clamp_max=2.0)
        + svg_line(rows, "train_cer", color="#8a5fd6", clamp_max=2.0)
        + (svg_line(rows, "lr", color="#2fa84f") if has_lr else "")
        + "</div>"
    )
    extra = ""
    if has_probe:
        extra += (
            '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px">'
            + svg_multi(
                rows,
                [
                    ("blank mass", ("probe", "blank")),
                    ("argmax blank", ("probe", "argmax_blank")),
                    ("max non-blank", ("probe", "max_nb")),
                ],
                "CTC output: blank vs non-blank (collapse lens)",
            )
            + svg_multi(
                rows,
                [
                    ("decoded len", ("probe", "dec_len")),
                    ("target len", ("probe", "tgt_len")),
                    ("entropy", ("probe", "entropy")),
                ],
                "decoded vs target length, softmax entropy (nats)",
            )
            + "</div>"
        )
        layer_keys = collect_keys(rows, "layers")
        if layer_keys:
            extra += (
                '<div style="margin-top:8px">'
                + svg_multi(
                    rows,
                    [(k, ("layers", k)) for k in layer_keys],
                    "layer activation RMS",
                    width=860,
                )
                + html_heatmap(rows, "layers", "layer activation RMS over epochs")
                + "</div>"
            )
        if any(r.get("weights") for r in rows):
            wkeys = collect_keys(rows, "weights")
            extra += (
                '<div style="margin-top:8px">'
                + svg_multi(
                    rows,
                    [(k, ("weights", k)) for k in wkeys],
                    "weight RMS by parameter group",
                    width=860,
                )
                + html_heatmap(rows, "weights", "weight RMS over epochs")
                + "</div>"
            )
        if any(r.get("grads") for r in rows):
            gkeys = collect_keys(rows, "grads")
            extra += (
                '<div style="margin-top:8px">'
                + svg_multi(
                    rows,
                    [(k, ("grads", k)) for k in gkeys],
                    "gradient RMS by parameter group (last batch of epoch)",
                    width=860,
                )
                + html_heatmap(rows, "grads", "gradient RMS over epochs")
                + "</div>"
            )
        extra += html_classes(last)
        sample_row = next((r for r in reversed(rows) if r.get("samples")), last)
        extra += html_samples(sample_row)
    return f"<section><h2>{label}</h2>{summary}{charts}{extra}</section>"


def render():
    pbt_sections = [render_pbt_run(label, path) for label, path in PBT_RUNS]
    sections = [render_run(label, path) for label, path in RUNS]
    tune_sections = [render_tune_run(label, path) for label, path in TUNE_RUNS]
    extra = ""
    if sections:
        extra += (
            '<h1 style="font-size:18px">Retired train loops</h1>'
            + "".join(sections)
        )
    if tune_sections:
        extra += (
            '<h1 style="font-size:18px">Retired TPE search</h1>'
            + "".join(tune_sections)
        )
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
<h1 style="font-size:18px">Population-based training — live view (auto-refreshes every 5s)</h1>
{''.join(pbt_sections)}
{extra}
</body></html>
"""
    OUT.write_text(html)


if __name__ == "__main__":
    render()
    print(f"wrote {OUT}")
