"""Diff the C++ scalar backend against PyTorch on the same records.

This runs once, to prove include/model.hpp computes the same forward pass
the model was trained with. Once it passes, the scalar backend becomes the
oracle and later backends (int8 NEON, CUDA) are diffed against it instead.

    cd python
    python diff_scalar.py           # defaults to 1000 records

Expect max |diff| around 1e-6 to 1e-5 -- float32 summation order differs
between the two implementations and that is fine. Anything at 1e-2 or
larger is structural: normalization applied on one side only, a transposed
weight matrix, or a missing bias.
"""

import sys

import numpy as np
import torch

from dataset import load
from train import MLP

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

FEATURES = '../data/aapl.bin'
WEIGHTS = 'weights.npz'
CPP_LOGITS = '../data/logits.bin'


def build_model(w):
    """Rebuild the trained model from the exported arrays.

    Loading from weights.npz rather than a checkpoint means this diffs
    against exactly what export.py wrote -- if the export dropped or
    transposed something, that shows up here rather than staying hidden.
    """
    model = MLP(n_in=w['w0'].shape[1],
                n_hidden=w['w0'].shape[0],
                n_out=w['w4'].shape[0])
    for src, idx in (('0', 0), ('2', 2), ('4', 4)):
        model.net[idx].weight.data = torch.from_numpy(w[f'w{src}'].copy())
        model.net[idx].bias.data = torch.from_numpy(w[f'b{src}'].copy())
    model.eval()
    return model


def main():
    d = load(FEATURES)[:N]
    X_raw = d['f'].astype(np.float32)

    w = np.load(WEIGHTS)
    model = build_model(w)

    # The C++ backend normalizes inside forward(), so Python must apply the
    # identical transform or the two are not computing the same function.
    X = (X_raw - w['norm_mean']) / w['norm_std']
    X = X.astype(np.float32)

    with torch.no_grad():
        py = model(torch.from_numpy(X)).numpy()

    cpp = np.fromfile(CPP_LOGITS, dtype='<f4')
    if cpp.size % 3 != 0:
        raise ValueError(f'{CPP_LOGITS} holds {cpp.size} floats, not a multiple of 3')
    cpp = cpp.reshape(-1, 3)

    n = min(len(py), len(cpp))
    if n == 0:
        raise ValueError('nothing to compare -- did infer_dump run?')
    if len(py) != len(cpp):
        print(f'note: pytorch has {len(py)} rows, C++ has {len(cpp)}; '
              f'comparing the first {n}')
    py, cpp = py[:n], cpp[:n]

    diff = np.abs(py - cpp)
    max_diff = diff.max()
    mean_diff = diff.mean()

    py_arg = py.argmax(axis=1)
    cpp_arg = cpp.argmax(axis=1)
    disagree = int((py_arg != cpp_arg).sum())

    print(f'compared {n} records\n')
    print(f'  max  |diff| : {max_diff:.3e}')
    print(f'  mean |diff| : {mean_diff:.3e}')
    print(f'  argmax disagreements: {disagree} / {n}')

    print('\nfirst row:')
    print(f'  pytorch: {py[0][0]:12.6f} {py[0][1]:12.6f} {py[0][2]:12.6f}')
    print(f'  c++    : {cpp[0][0]:12.6f} {cpp[0][1]:12.6f} {cpp[0][2]:12.6f}')

    if max_diff > 1e-3:
        worst = np.unravel_index(diff.argmax(), diff.shape)
        print(f'\nworst row {worst[0]}, logit {worst[1]}:')
        print(f'  pytorch: {py[worst[0]]}')
        print(f'  c++    : {cpp[worst[0]]}')
        print('\nFAIL -- this is too large for float32 rounding. Check, in '
              'order: normalization applied on both sides with the same '
              'stats; weight matrices not transposed; every bias added.')
        sys.exit(1)

    if disagree:
        print(f'\n{disagree} rows pick a different class. With max diff '
              f'{max_diff:.1e} these are near-ties, which is expected in '
              f'small numbers; if the count is large, the logits are not '
              f'actually close.')

    print('\nPASS -- the scalar backend matches PyTorch within float32 '
          'rounding. It can now serve as the oracle for other backends.')


if __name__ == '__main__':
    main()