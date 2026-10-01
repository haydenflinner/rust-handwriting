"""Export the ported ONNET (pretrained-lstm) to ONNX for burn-onnx.

Output: hwr-model/onnet_lstm.onnx (+ .onnx.data external weights), with a
dynamic time axis. Then sanity-checks the export against torch + onnxruntime.

  .venv-onnx/bin/python scripts/export_onnx.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from onnet_torch import Onnet, load_keras_weights, greedy_decode

ROOT = os.path.join(os.path.dirname(__file__), '..')
H5 = os.path.join(ROOT, 'third_party/IAMhwr/models/iamon/ONNET/pretrained-lstm/weights.h5')
OUT = os.path.join(ROOT, 'hwr-model/onnet_lstm.onnx')


def main():
    torch.manual_seed(0)
    model = Onnet().eval()
    load_keras_weights(model, H5)

    x = torch.randn(1, 100, 6)
    torch.onnx.export(
        model,
        (x,),
        OUT,
        dynamo=True,
        input_names=['features'],
        output_names=['logits'],
        dynamic_shapes=({0: torch.export.Dim.STATIC, 1: torch.export.Dim('time', min=4), 2: torch.export.Dim.STATIC},),
        external_data=True,
    )

    # The exporter leaves the output's time dim static — patch the metadata
    # to match the dynamic input.
    import onnx
    m = onnx.load(OUT)
    m.graph.output[0].type.tensor_type.shape.dim[1].dim_param = 'time'
    onnx.save(m, OUT, save_as_external_data=True, all_tensors_to_one_file=True,
              location='onnet_lstm.onnx.data')

    import onnxruntime as ort
    sess = ort.InferenceSession(OUT)
    for t in (40, 100, 353):
        xin = np.random.RandomState(0).randn(1, t, 6).astype(np.float32)
        got = sess.run(None, {'features': xin})[0]
        want = model(torch.from_numpy(xin)).detach().numpy()
        assert got.shape == want.shape, (got.shape, want.shape)
        print(f'T={t}: out {got.shape} maxdiff {np.abs(got - want).max():.2e}')

    # decode sanity on the fixture features
    feats = np.loadtxt(os.path.join(ROOT, 'hwr-model/tests/fixtures/m05-507z-06.features'), delimiter=',')
    xin = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        print('torch:', repr(greedy_decode(model(xin)[0].numpy())))
        print('onnx :', repr(greedy_decode(sess.run(None, {'features': xin.numpy()})[0][0])))


if __name__ == '__main__':
    main()
