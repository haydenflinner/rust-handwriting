"""Dump an IAM-OnDB sample as Rust test fixtures.

Writes:
  hwr-model/tests/fixtures/<name>.ink      — Ink::from_string literal
  hwr-model/tests/fixtures/<name>.features — SCHEME6 feature rows (csv)
  hwr-model/tests/fixtures/<name>.logits   — onnxruntime output (csv)

Usage: .venv-onnx/bin/python scripts/dump_fixture.py [name ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/IAMhwr'))
sys.path.insert(0, os.path.dirname(__file__))
from verify_onnet import load_pointset, parse_mlf, MLF, DATADIR
from hwr.constants import PREPROCESS

ONNX = os.path.join(os.path.dirname(__file__), '../hwr-model/onnet_lstm_finetuned.onnx')

OUT = os.path.join(os.path.dirname(__file__), '../hwr-model/tests/fixtures')


def find_xml(name):
    for dirpath, _, files in os.walk(DATADIR):
        if name + '.xml' in files:
            return os.path.join(dirpath, name + '.xml')
    raise FileNotFoundError(name)


def ink_literal(ps):
    """x y z per point, ',' between points, ';' between strokes."""
    strokes = {}
    for p in ps.points:
        strokes.setdefault(p.stroke, []).append(p)
    return ';'.join(
        ','.join(f'{p.x} {p.y} {p.time}' for p in pts)
        for _, pts in sorted(strokes.items())
    )


def main(names):
    import onnxruntime as ort
    sess = ort.InferenceSession(ONNX)
    os.makedirs(OUT, exist_ok=True)
    labels = parse_mlf(MLF)
    for name in names:
        ps = load_pointset(find_xml(name))
        with open(os.path.join(OUT, name + '.ink'), 'w') as f:
            f.write(ink_literal(ps))
        ps.preprocess(**PREPROCESS.SCHEME6)
        feats = ps.generate_features(add_pad=10)
        np.savetxt(os.path.join(OUT, name + '.features'), feats, fmt='%.9g', delimiter=',')
        logits = sess.run(None, {'features': feats.astype(np.float32)[None]})[0][0]
        np.savetxt(os.path.join(OUT, name + '.logits'), logits, fmt='%.9g', delimiter=',')
        print(f'{name}: {feats.shape} gt={labels.get(name)!r}')


if __name__ == '__main__':
    main(sys.argv[1:] or ['m05-507z-06'])
