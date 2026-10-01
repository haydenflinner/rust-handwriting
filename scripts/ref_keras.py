"""Reference ONNET forward in TF2/Keras, loading the TF1-era CuDNN weights.h5.

Prints per-stage stats and final logits for comparison with the torch port.
"""
import os
import sys

import h5py
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras import layers, Model

H5 = os.path.join(os.path.dirname(__file__),
                  '../third_party/IAMhwr/models/iamon/ONNET/pretrained-deep-lstm/weights.h5')
NUM_CLASSES = 83


def tdnn_bn_relu(x, filters, k, conv_name, bn_name):
    x = layers.Conv1D(filters, k, padding='same', name=conv_name)(x)
    x = layers.BatchNormalization(name=bn_name)(x)
    return layers.Activation('relu')(x)


def build_model():
    inputs = layers.Input(shape=(None, 6), name='xs')
    x = tdnn_bn_relu(inputs, 60, 7, 'conv1d', 'batch_normalization')
    x = tdnn_bn_relu(x, 90, 5, 'conv1d_1', 'batch_normalization_1')
    x = tdnn_bn_relu(x, 120, 5, 'conv1d_2', 'batch_normalization_2')
    x = layers.AveragePooling1D(2)(x)
    x = tdnn_bn_relu(x, 120, 3, 'conv1d_3', 'batch_normalization_3')
    x = tdnn_bn_relu(x, 160, 3, 'conv1d_4', 'batch_normalization_4')
    x = tdnn_bn_relu(x, 200, 3, 'conv1d_5', 'batch_normalization_5')
    x = layers.AveragePooling1D(2)(x)
    for i in range(4):
        n1, n2 = f'cu_dnnlstm_{2*i}', f'cu_dnnlstm_{2*i+1}'
        if 2 * i == 0:
            n1 = 'cu_dnnlstm'
        a = layers.LSTM(60, return_sequences=True, name=n1)(x)
        b = layers.LSTM(60, return_sequences=True, go_backwards=True, name=n2)(x)
        x = layers.Concatenate()([a, b])
    x = layers.BatchNormalization(name='batch_normalization_6')(x)
    x = layers.Dense(NUM_CLASSES, name='dense')(x)
    out = layers.Activation('softmax', name='softmax')(x)
    return Model(inputs, out)


def load_weights(model):
    f = h5py.File(H5, 'r')

    def arr(layer, name):
        return f[f'{layer}/{layer}/{name}:0'][()]

    for layer in model.layers:
        n = layer.name
        if n.startswith('conv1d'):
            layer.set_weights([arr(n, 'kernel'), arr(n, 'bias')])
        elif n.startswith('batch_normalization'):
            layer.set_weights([arr(n, 'gamma'), arr(n, 'beta'),
                               arr(n, 'moving_mean'), arr(n, 'moving_variance')])
        elif n.startswith('cu_dnnlstm'):
            k = arr(n, 'kernel')          # (in, 4h), gates i,f,c,o
            rk = arr(n, 'recurrent_kernel')
            b = arr(n, 'bias')            # (8h) = [b_ih | b_hh]
            h = rk.shape[0]
            layer.set_weights([k, rk, b[:4 * h] + b[4 * h:]])
        elif n == 'dense':
            layer.set_weights([arr(n, 'kernel'), arr(n, 'bias')])
    return model


def features_for(xml_name):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../third_party/IAMhwr'))
    sys.path.insert(0, os.path.dirname(__file__))
    from verify_onnet import load_pointset, REPO
    from hwr.constants import PREPROCESS
    ps = load_pointset(os.path.join(
        REPO, 'data/iamon/lineStrokes(on)/data', xml_name))
    ps.preprocess(**PREPROCESS.SCHEME6)
    return ps.generate_features(add_pad=10).astype(np.float32)


if __name__ == '__main__':
    model = load_weights(build_model())
    feats = features_for('b04/b04-334/b04-334z-07.xml')
    probs = model.predict(feats[None], verbose=0)[0]
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print('probs shape', probs.shape)
    am = probs.argmax(-1)
    print('argmax', am)
    # also dump stage outputs for comparison
    stage_names = ['conv1d_5', 'cu_dnnlstm_6', 'cu_dnnlstm_7', 'batch_normalization_6', 'dense']
    for sn in stage_names:
        inter = Model(model.input, model.get_layer(sn).output)
        out = inter.predict(feats[None], verbose=0)
        np.save(f'/tmp/ref_{sn}.npy', out)
        print(sn, out.shape, 'mean/std', out.mean(), out.std())
    np.save('/tmp/ref_probs.npy', probs)
    np.save('/tmp/ref_feats.npy', feats)
