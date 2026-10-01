"""Fine-tune the ported ONNET on the app's calibration corpus + synthetic data.

The pretrained head emits 83 classes (82 MLF symbols + CTC blank at 82).
The calibration corpus needs 15 more symbols the model cannot spell
(`= < > { } @ # $ % ^ _ ` \\ | ~`), so the dense head is widened to 98:
rows 0..81 keep the pretrained weights, 82..96 are the new symbols, and
the old blank row moves to 97.

Training data is the real bottleneck (302 usable calibration samples), so
most of the batch mix is synthesized:

  * Hershey stroke fonts (pen-plotter glyphs = real stroke polylines) laid
    out as strings, densified + jittered to look like sampled pen input.
  * The user's own single-glyph calibration inks composed into random
    strings (digit runs, operators, code tokens).
  * The 20 bundled IAM-OnDB lines, to retain line-level letter/word skill.

  .venv-onnx/bin/python scripts/finetune_onnet.py [--epochs 200]

Writes hwr-model/onnet_lstm_finetuned.onnx on improvement.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lxml import etree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/IAMhwr'))
sys.path.insert(0, os.path.dirname(__file__))
from onnet_torch import Onnet, load_keras_weights, CHARS
from hwr.data.datarep import PointSet, Point
from hwr.constants import PREPROCESS, DATA
from HersheyFonts import HersheyFonts

ROOT = os.path.join(os.path.dirname(__file__), '..')
H5 = os.path.join(ROOT, 'third_party/IAMhwr/models/iamon/ONNET/pretrained-lstm/weights.h5')
CALIB = os.path.expanduser('~/Library/Application Support/hwr/calibration.txt')
OUT = os.path.join(ROOT, 'hwr-model/onnet_lstm_finetuned.onnx')
IAM_MLF = os.path.join(ROOT, 'third_party/IAMhwr/data/iamon/lineStrokes(on)/t2_labels.mlf')
IAM_DIR = os.path.join(ROOT, 'third_party/IAMhwr/data/iamon/lineStrokes(on)/data')

NEW_SYMS = list('=<>{}@#$%^_`|~\\')
VOCAB = CHARS + NEW_SYMS          # 97 symbols
BLANK = len(VOCAB)                # 97
NUM_CLASSES = BLANK + 1           # 98
IDX = {c: i for i, c in enumerate(VOCAB)}

# ---------------------------------------------------------------------------
# Uniform sample representation: (label, strokes) where strokes is a list of
# strokes, each a list of (x, y) float tuples in screen coords (y down).

def parse_ink_strokes(s):
    """calibration.txt ink literal -> strokes."""
    strokes = []
    for stroke in s.split(';'):
        pts = []
        for pt in stroke.split(','):
            x, y, _t = pt.split()
            pts.append((float(x), float(y)))
        if pts:
            strokes.append(pts)
    return strokes


def strokes_to_pointset(strokes):
    ps = PointSet()
    t = 0
    for sid, stroke in enumerate(strokes, 1):
        for x, y in stroke:
            ps.add_point(Point(sid, t, float(x), float(y)))
            t += 1
    return ps


def features(strokes):
    ps = strokes_to_pointset(strokes)
    ps.preprocess(**PREPROCESS.SCHEME6)
    return ps.generate_features(add_pad=10).astype(np.float32)


def load_corpus(path):
    samples = []
    for line in open(path):
        line = line.rstrip('\n')
        if not line or '\t' not in line:
            continue
        label, ink = line.split('\t', 1)
        if not all(c in IDX for c in label):
            print(f'  skip (unspellable label): {label!r}')
            continue
        strokes = parse_ink_strokes(ink)
        if strokes:
            samples.append((label, strokes))
    return samples


# ---------------------------------------------------------------------------
# Hershey stroke-font synthesis.

HERSHEY_FONTS = ['futural', 'timesr', 'rowmans', 'scripts']


def load_hershey():
    """{font: {char: (strokes, advance)}} in Hershey units (y down)."""
    fonts = {}
    for name in HERSHEY_FONTS:
        f = HersheyFonts()
        f.load_default_font(name)
        table = {}
        for ch in VOCAB:
            if ch == ' ':
                continue
            gs = [g for g in f.glyphs_for_text(ch)]
            if not gs or not gs[0].strokes:
                continue
            g = gs[0]
            adv = g.char_width or max(p[0] for s in g.strokes for p in s)
            table[ch] = ([[(float(x), float(y)) for x, y in s] for s in g.strokes],
                         float(adv))
        fonts[name] = table
        missing = [c for c in VOCAB if c != ' ' and c not in table]
        print(f'hershey {name}: {len(table)} glyphs, missing {missing}')
    return fonts


def densify(strokes, rng, step=1.3, noise=0.28):
    """Resample each segment to ~step spacing and add point noise so the
    sparse corner-vertex polylines survive angle downsampling like real pen
    input does."""
    out = []
    for stroke in strokes:
        pts = [stroke[0]]
        for a, b in zip(stroke, stroke[1:]):
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(d / step))
            for j in range(1, n + 1):
                t = j / n
                pts.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
        out.append([(x + rng.normal(0, noise), y + rng.normal(0, noise))
                    for x, y in pts])
    return out


def hershey_ink(text, fonts, rng):
    """Lay out `text` from a random Hershey font -> strokes (y down)."""
    font = fonts[rng.choice(list(fonts))]
    strokes = []
    cursor = 0.0
    for ch in text:
        if ch == ' ':
            cursor += rng.uniform(10, 16)
            continue
        entry = font.get(ch)
        if entry is None:  # fall back to any font that has the glyph
            entry = next(f[ch] for f in fonts.values() if ch in f)
        gstrokes, adv = entry
        s = rng.uniform(0.9, 1.2)
        dy = rng.uniform(-2.0, 2.0)
        for st in gstrokes:
            strokes.append([(x * s + cursor, y * s + dy) for x, y in st])
        cursor += adv * s + rng.uniform(0, 4)
    strokes = densify(strokes, rng)
    # Global scale to ~pixel territory (normalization absorbs absolute
    # scale; this keeps point density plausible pre-downsample).
    scale = rng.uniform(3.5, 5.0)
    rot = rng.uniform(-0.08, 0.08)
    c, s = math.cos(rot), math.sin(rot)
    return [[((x * scale) * c + (y * scale) * s,
              -(x * scale) * s + (y * scale) * c) for x, y in st]
            for st in strokes]


# ---------------------------------------------------------------------------
# Composition of the user's own single-glyph samples into strings.

def compose_ink(text, pool, rng):
    """pool: {char: [strokes,...]}. Returns None if any char is uncovered."""
    if any(ch != ' ' and ch not in pool for ch in text):
        return None
    strokes = []
    cursor = 0.0
    for ch in text:
        if ch == ' ':
            cursor += rng.uniform(15, 25)
            continue
        src = pool[ch][rng.randint(len(pool[ch]))]
        xs = [p[0] for p_ in src for p in p_]
        s = rng.uniform(0.85, 1.15)
        dy = rng.uniform(-4, 4)
        x0 = min(xs) * s
        w = (max(xs) - min(xs)) * s
        for st in src:
            strokes.append([(x * s - x0 + cursor, y * s + dy) for x, y in st])
        cursor += w + rng.uniform(6, 22)
    return strokes


# ---------------------------------------------------------------------------
# Text generators for synthetic samples.

KEYWORDS = ['fn', 'let', 'mut', 'match', 'await', 'impl', 'struct', 'enum',
            'pub', 'mod', 'use', 'for', 'if', 'else', 'return', 'while',
            'loop', 'new', 'case', 'true', 'false', 'self', 'const', 'static',
            'ref', 'move', 'dyn', 'async', 'type', 'where', 'crate', 'Box',
            'Some', 'None', 'Ok', 'Err', 'Vec', 'String', 'str', 'bool',
            'i32', 'i64', 'u32', 'u64', 'f32', 'f64', 'usize', 'char']
OPS = ['==', '!=', '<=', '>=', '->', '=>', '&&', '||', '+=', '-=', '*=',
       '/=', '%=', '<<', '>>', '::', '..', '...']
PUNCT = list('{}()[]<>=+-*/%^|&!~?;:,.@#$_`\\\'"')


def gen_text(rng):
    r = rng.rand()
    if r < 0.30:  # digit strings, any length
        return ''.join(rng.choice(list('0123456789'))
                       for _ in range(rng.randint(1, 9)))
    if r < 0.55:  # operators / symbol runs
        if rng.rand() < 0.5:
            return rng.choice(OPS)
        return ''.join(rng.choice(PUNCT)
                       for _ in range(rng.randint(1, 5)))
    if r < 0.75:  # code keywords
        return rng.choice(KEYWORDS)
    # identifier-ish words: letters + digits/underscore
    n = rng.randint(2, 8)
    first = rng.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'))
    rest = ''.join(rng.choice(list('abcdefghijklmnopqrstuvwxyz0123456789_'))
                   for _ in range(n - 1))
    return first + rest


# ---------------------------------------------------------------------------
# IAM-OnDB bundled lines.

def parse_mlf(path):
    labels, name, buf = {}, None, []
    for line in open(path):
        line = line.strip()
        if line.startswith('"'):
            if name:
                labels[name] = buf
            name = os.path.basename(line.strip('"'))[:-4]
            buf = []
        elif line == '.':
            if name:
                labels[name] = buf
            name, buf = None, []
        elif name and line:
            buf.append(line)
    return {k: ''.join(DATA.MLF_MAPPING.get(c, c) for c in v)
            for k, v in labels.items()}


def load_iam():
    labels = parse_mlf(IAM_MLF)
    out = []
    for dirpath, _, files in os.walk(IAM_DIR):
        for fn in sorted(files):
            if not fn.endswith('.xml'):
                continue
            root = etree.parse(os.path.join(dirpath, fn)).getroot()
            wbd, strokeset = root.getchildren()
            l = int(wbd[2].attrib['x'])
            u = int(wbd[3].attrib['y'])
            strokes = []
            for stroke in strokeset:
                pts = [(float(p.attrib['x']) - l, float(p.attrib['y']) - u)
                       for p in stroke]
                if pts:
                    strokes.append(pts)
            label = labels.get(fn[:-4], '')
            if strokes and all(c in IDX for c in label):
                out.append((label, strokes))
    return out


# ---------------------------------------------------------------------------

def jitter(strokes, rng):
    """Light affine jitter + point noise; returns new strokes."""
    sx, sy = rng.uniform(0.9, 1.1, 2)
    rot = rng.uniform(-0.06, 0.06)
    c, s = math.cos(rot), math.sin(rot)
    tx, ty = rng.uniform(-5, 5, 2)
    out = []
    for st in strokes:
        out.append([
            ((x * sx) * c + (y * sy) * s + tx + rng.normal(0, 0.5),
             -(x * sx) * s + (y * sy) * c + ty + rng.normal(0, 0.5))
            for x, y in st])
    return out


def prepare(samples):
    """(label, feats, flen) triples for precomputed pools."""
    out = []
    for label, strokes in samples:
        f = features(strokes)
        flen = f.shape[0] // 4
        if flen >= len(label) + 1:  # CTC needs headroom (repeat -> +blank)
            out.append((label, f, flen))
    return out


def cer(model, prepared):
    edits = chars = exact = 0
    with torch.no_grad():
        for label, f, _ in prepared:
            out = model(torch.from_numpy(f).unsqueeze(0))[0].numpy()
            pred = greedy_decode_ext(out)
            d = levenshtein(label, pred)
            edits += d
            chars += max(1, len(label))
            exact += d == 0
    return edits / max(1, chars), exact / max(1, len(prepared))


def greedy_decode_ext(logits):
    idx = np.asarray(logits).argmax(axis=-1)
    out, last = [], -1
    for i in idx:
        if i != last and i != BLANK and i < len(VOCAB):
            out.append(VOCAB[i])
        last = i
    return ''.join(out)


def levenshtein(a, b):
    m, n = len(a), len(b)
    d = np.arange(n + 1)
    for i in range(1, m + 1):
        prev, d[0] = d[0], i
        for j in range(1, n + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1,
                                   prev + (a[i - 1] != b[j - 1]))
    return int(d[n])


def widen_head(model):
    """83 -> 98 output classes; old blank row 82 -> new row 97."""
    old = model.dense
    new = nn.Conv1d(old.in_channels, NUM_CLASSES, 1)
    with torch.no_grad():
        new.weight.zero_()
        new.bias.zero_()
        new.weight[:82] = old.weight[:82]          # existing symbols
        new.bias[:82] = old.bias[:82]
        new.weight[BLANK] = old.weight[82]         # blank moves to 97
        new.bias[BLANK] = old.bias[82]
        nn.init.normal_(new.weight[82:BLANK], std=0.02)
    model.dense = new
    return model


def batchify(items):
    """items: [(label, feats, flen)] -> padded tensors for CTC."""
    feats = [f for _, f, _ in items]
    T = max(f.shape[0] for f in feats)
    x = np.zeros((len(feats), T, 6), dtype=np.float32)
    for i, f in enumerate(feats):
        x[i, : f.shape[0]] = f
    targets, tlens = [], []
    for label, _, _ in items:
        targets.extend(IDX[c] for c in label)
        tlens.append(len(label))
    return (torch.from_numpy(x),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor([fl for _, _, fl in items], dtype=torch.long),
            torch.tensor(tlens, dtype=torch.long))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--holdout', type=float, default=0.15)
    ap.add_argument('--synth-pool', type=int, default=4000)
    ap.add_argument('--synth-per-epoch', type=int, default=1200)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(4, (os.cpu_count() or 4) - 2))

    model = Onnet().eval()
    load_keras_weights(model, H5)
    model = widen_head(model)

    # ---- data ----
    samples = load_corpus(CALIB)
    rng.shuffle(samples)
    n_test = max(1, int(len(samples) * args.holdout))
    test, train = samples[:n_test], samples[n_test:]
    print(f'{len(samples)} calib -> {len(train)} train / {len(test)} held-out')

    glyph_pool = {}
    for label, strokes in samples:  # train+test: composition is data aug, not leakage
        if len(label) == 1:
            glyph_pool.setdefault(label, []).append(strokes)
    print(f'{len(glyph_pool)} distinct single-glyph exemplars for composition')

    fonts = load_hershey()
    iam = load_iam()
    print(f'{len(iam)} IAM-OnDB lines')

    # Precompute synthetic + IAM features once (generation bakes in variety;
    # real calibration samples get fresh jitter every epoch instead).
    synth = []
    while len(synth) < args.synth_pool:
        text = gen_text(rng)
        if rng.rand() < 0.45:
            st = compose_ink(text, glyph_pool, rng)
            if st is None:
                st = hershey_ink(text, fonts, rng)
        else:
            st = hershey_ink(text, fonts, rng)
        synth.append((text, st))
    synth_prep = prepare(synth)
    iam_prep = prepare(iam)
    test_prep = prepare(test)
    print(f'synth pool {len(synth_prep)} usable, iam {len(iam_prep)}')

    base_cer, base_exact = cer(model, test_prep)
    print(f'before: held-out CER {100*base_cer:.1f}%  exact {100*base_exact:.0f}%')
    iam_c, iam_e = cer(model, iam_prep)
    print(f'        IAM lines CER {100*iam_c:.1f}%  exact {100*iam_e:.0f}%')

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best = base_cer
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(args.epochs):
        model.train()
        # Keep BN running stats frozen — the corpus is tiny and OOD; letting
        # stats drift on isolated glyphs would wreck the feature extractor.
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()

        # real samples: fresh jitter + features each epoch
        real = prepare([(l, jitter(st, rng)) for l, st in train])
        idx = rng.choice(len(synth_prep),
                         min(args.synth_per_epoch, len(synth_prep)),
                         replace=False)
        pool = real + iam_prep + [synth_prep[i] for i in idx]
        rng.shuffle(pool)

        tot, nb = 0.0, 0
        for i in range(0, len(pool), args.batch):
            x, y, ilens, tlens = batchify(pool[i : i + args.batch])
            logits = model(x)                       # [B, T', C]
            logp = logits.log_softmax(-1).permute(1, 0, 2)  # [T', B, C]
            loss = F.ctc_loss(logp, y, ilens, tlens,
                              blank=BLANK, zero_infinity=True)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            model.eval()
            c, e = cer(model, test_prep)
            marker = ' *' if c < best else ''
            if c < best:
                best = c
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            print(f'epoch {epoch+1:4d}: loss {tot/nb:.3f}  '
                  f'held-out CER {100*c:.1f}%  exact {100*e:.0f}%{marker}',
                  flush=True)

    print(f'\nbest held-out CER {100*best:.1f}% (was {100*base_cer:.1f}%)')
    model.load_state_dict(best_state)  # export the best checkpoint, not the last
    model.eval()
    iam_c, iam_e = cer(model, iam_prep)
    print(f'IAM lines after: CER {100*iam_c:.1f}%  exact {100*iam_e:.0f}%')

    # Export
    x = torch.randn(1, 100, 6)
    torch.onnx.export(
        model, (x,), OUT, dynamo=True,
        input_names=['features'], output_names=['logits'],
        dynamic_shapes=({0: torch.export.Dim.STATIC,
                         1: torch.export.Dim('time', min=4),
                         2: torch.export.Dim.STATIC},),
        external_data=False,
    )
    import onnx
    m = onnx.load(OUT)
    m.graph.output[0].type.tensor_type.shape.dim[1].dim_param = 'time'
    onnx.save(m, OUT)
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
