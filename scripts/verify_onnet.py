"""Verify the PyTorch port of ONNET against real IAM-OnDB samples.

Uses the repo's own datarep.py for preprocessing (SCHEME6) and the shipped
t2_labels.mlf for ground truth, then greedy-CTC-decodes the torch output.
"""
import os
import sys
from decimal import Decimal

import numpy as np
import torch
from lxml import etree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/IAMhwr'))
from hwr.constants import PREPROCESS, DATA
from hwr.data.datarep import PointSet, Point
from onnet_torch import Onnet, load_keras_weights, greedy_decode, NUM_CLASSES

REPO = os.path.join(os.path.dirname(__file__), '../third_party/IAMhwr')
MLF = os.path.join(REPO, 'data/iamon/lineStrokes(on)/t2_labels.mlf')
DATADIR = os.path.join(REPO, 'data/iamon/lineStrokes(on)/data')


def parse_mlf(path):
    """Return {sample_name: text}."""
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
    return {k: ''.join(DATA.MLF_MAPPING.get(c, c) for c in v) for k, v in labels.items()}


def load_pointset(xml_path):
    root = etree.parse(xml_path).getroot()
    wbd, strokeset = root.getchildren()
    l = int(wbd[2].attrib['x'])  # vertically-opposite x = left edge
    u = int(wbd[3].attrib['y'])  # horizontally-opposite y = upper edge
    points, sid = [], 1
    first = strokeset[0][0]
    t0 = Decimal(first.attrib['time'])
    for stroke in strokeset:
        for p in stroke:
            t = (Decimal(p.attrib['time']) - t0) * 1000
            points.append(Point(sid, int(t), int(p.attrib['x']) - l, int(p.attrib['y']) - u))
        sid += 1
    return PointSet(points=points)


def main():
    torch.manual_seed(0)
    model = Onnet().eval()
    load_keras_weights(model, os.path.join(REPO, 'models/iamon/ONNET/pretrained-lstm/weights.h5'))

    labels = parse_mlf(MLF)
    correct, total = 0, 0
    for dirpath, _, files in os.walk(DATADIR):
        for fn in sorted(files):
            if not fn.endswith('.xml'):
                continue
            name = fn[:-4]
            ps = load_pointset(os.path.join(dirpath, fn))
            ps.preprocess(**PREPROCESS.SCHEME6)
            feats = ps.generate_features(add_pad=10)
            x = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits = model(x)[0].numpy()
            pred = greedy_decode(logits)
            gt = labels.get(name, '?')
            match = 'OK ' if pred == gt else '   '
            if pred == gt:
                correct += 1
            total += 1
            print(f'{match}{name}: pred={pred!r} gt={gt!r}')
    print(f'\nexact match: {correct}/{total}')


if __name__ == '__main__':
    main()
