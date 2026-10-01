"""PyTorch reimplementation of IAMhwr's ONNET `pretrained-lstm` checkpoint.

The checkpoint (recovered from git history — the shipped `pretrained-deep-lstm`
CuDNN checkpoint decodes to garbage; this one gives ~6.8% CER greedy on the
repo's bundled IAM-OnDB t2 samples) pairs with the model config from the
init-era hwr/models/ONNET.py:

  input [B, T, 6]
  Conv1D(60,k=7,same)+BN+ReLU; Conv1D(80,k=5)+BN+ReLU; Conv1D(100,k=5)+BN+ReLU
  AvgPool(2)
  Conv1D(100,k=3)+BN+ReLU; Conv1D(130,k=3)+BN+ReLU; Conv1D(160,k=3)+BN+ReLU
  AvgPool(2)
  2 x BiLSTM(80)  (concat fwd/bwd -> 160)
  BatchNorm, Dense(83), (softmax in Keras; we export logits)

Keras LSTM layout: kernel (in, 4h), recurrent_kernel (h, 4h), bias (4h,)
single — maps to torch b_ih=bias, b_hh=0. Gate order i,f,c,o == torch i,f,g,o.
bi_rnn creates the forward LSTM first: lstm_{2i} fwd, lstm_{2i+1} bwd.
"""

import h5py
import numpy as np
import torch
import torch.nn as nn

CHARS_MLF = ['ex', 'qu', 'ga', 'do', 'am', 'ti', 'sl', 'lb', 'rb', 'ls', 'rs', 'sr', 'cm', 'mi',
             'pl', 'pt', 'sp', 'cl', 'sc', 'qm', 'n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7',
             'n8', 'n9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
             'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
             'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
             'y', 'z']
MLF_MAPPING = {"ex": "!", "qu": '"', "ga": "", "do": "", "am": "&", "ti": "'", "sl": "/",
               "lb": "(", "rb": ")", "ls": "[", "rs": "]", "sr": "*", "cm": ",", "mi": "-",
               "pl": "+", "pt": ".", "sp": " ", "cl": ":", "sc": ";", "qm": "?",
               "n0": "0", "n1": "1", "n2": "2", "n3": "3", "n4": "4", "n5": "5",
               "n6": "6", "n7": "7", "n8": "8", "n9": "9"}
CHARS = [MLF_MAPPING.get(c, c) for c in CHARS_MLF]  # 82 entries; index 82 is CTC blank
NUM_CLASSES = len(CHARS_MLF) + 1  # 83
BLANK = len(CHARS_MLF)            # 82


class Onnet(nn.Module):
    def __init__(self):
        super().__init__()
        eps = 1e-3  # Keras BatchNormalization default

        def cbr(cin, cout, k):
            return nn.Sequential(
                nn.Conv1d(cin, cout, k, padding=k // 2),
                nn.BatchNorm1d(cout, eps=eps),
                nn.ReLU(),
            )

        self.tdnn = nn.Sequential(
            cbr(6, 60, 7), cbr(60, 80, 5), cbr(80, 100, 5),
            nn.AvgPool1d(2),
            cbr(100, 100, 3), cbr(100, 130, 3), cbr(130, 160, 3),
            nn.AvgPool1d(2),
        )
        self.lstms = nn.ModuleList([
            nn.LSTM(160, 80, batch_first=True, bidirectional=True),
            nn.LSTM(160, 80, batch_first=True, bidirectional=True),
        ])
        self.bn = nn.BatchNorm1d(160, eps=eps)
        # Pointwise conv instead of Linear so the ONNX graph needs no
        # reshape (the dynamo exporter bakes T into Linear's gemm reshape).
        self.dense = nn.Conv1d(160, NUM_CLASSES, 1)

    def forward(self, x):  # x: [B, T, 6] -> logits [B, T/4, 83]
        h = self.tdnn(x.transpose(1, 2)).transpose(1, 2)
        for lstm in self.lstms:
            h, _ = lstm(h)
        h = self.bn(h.transpose(1, 2))
        return self.dense(h).transpose(1, 2)


def _lname(base, idx):
    return base if idx == 0 else f'{base}_{idx}'


def load_keras_weights(model, h5_path):
    f = h5py.File(h5_path, 'r')

    def arr(layer, name):
        return f[f'{layer}/{layer}/{name}:0'][()]

    convs = [m[0] for m in model.tdnn if isinstance(m, nn.Sequential)]
    bns = [m[1] for m in model.tdnn if isinstance(m, nn.Sequential)]
    for i, conv in enumerate(convs):
        lname = _lname('conv1d', i)
        conv.weight.data = torch.from_numpy(arr(lname, 'kernel').transpose(2, 1, 0).copy())
        conv.bias.data = torch.from_numpy(arr(lname, 'bias'))
    for i, bn in enumerate(bns):
        lname = _lname('batch_normalization', i)
        bn.weight.data = torch.from_numpy(arr(lname, 'gamma'))
        bn.bias.data = torch.from_numpy(arr(lname, 'beta'))
        bn.running_mean.data = torch.from_numpy(arr(lname, 'moving_mean'))
        bn.running_var.data = torch.from_numpy(arr(lname, 'moving_variance'))
    bn = model.bn
    lname = 'batch_normalization_6'
    bn.weight.data = torch.from_numpy(arr(lname, 'gamma'))
    bn.bias.data = torch.from_numpy(arr(lname, 'beta'))
    bn.running_mean.data = torch.from_numpy(arr(lname, 'moving_mean'))
    bn.running_var.data = torch.from_numpy(arr(lname, 'moving_variance'))

    for i, lstm in enumerate(model.lstms):
        for direction, suffix in ((0, ''), (1, '_reverse')):
            lname = _lname('lstm', i * 2 + direction)
            k = arr(lname, 'kernel')           # (in, 4h)
            rk = arr(lname, 'recurrent_kernel')  # (h, 4h)
            b = arr(lname, 'bias')             # (4h,) single bias
            setattr(lstm, f'weight_ih_l0{suffix}',
                    nn.Parameter(torch.from_numpy(k.T.copy())))
            setattr(lstm, f'weight_hh_l0{suffix}',
                    nn.Parameter(torch.from_numpy(rk.T.copy())))
            setattr(lstm, f'bias_ih_l0{suffix}',
                    nn.Parameter(torch.from_numpy(b.copy())))
            setattr(lstm, f'bias_hh_l0{suffix}',
                    nn.Parameter(torch.zeros(4 * lstm.hidden_size)))
        lstm.flatten_parameters()

    model.dense.weight.data = torch.from_numpy(arr('dense', 'kernel').T.copy()).unsqueeze(-1)
    model.dense.bias.data = torch.from_numpy(arr('dense', 'bias'))
    return model


def greedy_decode(logits):
    """logits/probs: [T, C] -> str (CTC best-path)."""
    idx = np.asarray(logits).argmax(axis=-1)
    out, last = [], -1
    for i in idx:
        if i != last and i != BLANK:
            out.append(CHARS[i])
        last = i
    return ''.join(out)
