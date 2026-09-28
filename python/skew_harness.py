"""Train/serve skew harness.

The int8 inference scheme is defined twice: once in python/calibrate.py
(which computes the scales and writes the weight file) and once in
include/model_int8.hpp (which reads it and runs the arithmetic). Nothing
enforces that those two agree -- not the compiler, not the file format,
only the version number.

This reimplements the C++ forward pass in numpy, from the same binary
weight file, and asserts the two produce the same logits. If a scale moves,
a layout changes, or a rounding rule diverges, this fails and names the
stage.

Run after regenerating weights or touching either side:

    cd python
    ./../build/infer_int8 ../data/weights_int8.bin ../data/aapl.bin \\
        ../data/logits_int8.bin 1000
    python skew_harness.py
"""

import struct
import sys

import numpy as np

from dataset import load, RECORD_DTYPE

INT8_MAGIC = 0x53544D51
INT8_VERSION = 1

MARKET_OPEN = int(9.5 * 3600 * 1e9)
MARKET_CLOSE = int(16.0 * 3600 * 1e9)

WEIGHTS = '../data/weights_int8.bin'
FEATURES = '../data/aapl.bin'
CPP_LOGITS = '../data/logits_int8.bin'
N = 1000


def load_int8_weights(path):
    """Read the same binary include/model_int8.hpp reads.

    Deliberately parsed independently rather than reusing weights.npz --
    the point is to validate the file that C++ actually consumes, so a bad
    export shows up here rather than staying hidden behind a shared source.
    """
    with open(path, 'rb') as f:
        magic, version, n_in, n_hidden, n_out = struct.unpack('<5I', f.read(20))
        if magic != INT8_MAGIC:
            raise ValueError(f'bad magic {magic:#x}, want {INT8_MAGIC:#x}')
        if version != INT8_VERSION:
            raise ValueError(f'version {version}, this script expects {INT8_VERSION}')

        def rf(n):
            return np.frombuffer(f.read(4 * n), dtype='<f4')

        def rq(n):
            return np.frombuffer(f.read(n), dtype=np.int8)

        w = {
            'n_in': n_in, 'n_hidden': n_hidden, 'n_out': n_out,
            'norm_mean': rf(n_in),
            'norm_std': rf(n_in),
        }
        w['x_scale'], w['h1_scale'], w['h2_scale'] = rf(3)
        w['w0_scale'], w['w2_scale'], w['w4_scale'] = rf(3)

        w['w0'] = rq(n_hidden * n_in).reshape(n_hidden, n_in)
        w['b0'] = rf(n_hidden)
        w['w2'] = rq(n_hidden * n_hidden).reshape(n_hidden, n_hidden)
        w['b2'] = rf(n_hidden)
        w['w4'] = rq(n_out * n_hidden).reshape(n_out, n_hidden)
        w['b4'] = rf(n_out)

        if f.read(1):
            raise ValueError('trailing bytes -- layout mismatch')
    return w


def quantize(v, scale):
    """Match include/model_int8.hpp's quantize_inv.

    np.round and lrintf both round ties to even, so the codes should agree
    exactly. Clamping happens before the int8 cast on both sides -- casting
    first would wrap 300 to 44 rather than saturating to 127.
    """
    q = np.round(v / scale)
    return np.clip(q, -127, 127).astype(np.int8)


def forward_int8(w, x_raw):
    """Reimplementation of model::Int8Backend::forward, vectorized over rows.

    Every step mirrors the C++: normalize in float, quantize, integer
    matmul into int32, one float multiply per neuron to leave integer
    space, bias, relu, requantize at the NEXT layer's scale.
    """
    s0 = w['x_scale'] * w['w0_scale']
    s2 = w['h1_scale'] * w['w2_scale']
    s4 = w['h2_scale'] * w['w4_scale']

    x = (x_raw - w['norm_mean']) / w['norm_std']
    x_q = quantize(x, w['x_scale'])

    # int32 accumulation. astype before the matmul or numpy would overflow
    # int8 on the intermediate products.
    acc = x_q.astype(np.int32) @ w['w0'].astype(np.int32).T
    h1 = acc.astype(np.float32) * s0 + w['b0']
    h1 = np.maximum(h1, 0.0)
    h1_q = quantize(h1, w['h1_scale'])

    acc = h1_q.astype(np.int32) @ w['w2'].astype(np.int32).T
    h2 = acc.astype(np.float32) * s2 + w['b2']
    h2 = np.maximum(h2, 0.0)
    h2_q = quantize(h2, w['h2_scale'])

    acc = h2_q.astype(np.int32) @ w['w4'].astype(np.int32).T
    return acc.astype(np.float32) * s4 + w['b4']


def main():
    w = load_int8_weights(WEIGHTS)
    print(f'weights: n_in={w["n_in"]} n_hidden={w["n_hidden"]} '
          f'n_out={w["n_out"]}')
    print(f'  activation scales: x={w["x_scale"]:.6f} '
          f'h1={w["h1_scale"]:.6f} h2={w["h2_scale"]:.6f}')
    print(f'  weight scales:    w0={w["w0_scale"]:.6f} '
          f'w2={w["w2_scale"]:.6f} w4={w["w4_scale"]:.6f}')

    # Same record selection the C++ infer programs use: market hours only,
    # first N kept. A mismatch here would silently compare different rows.
    d = load(FEATURES)
    keep = (d['ts'] >= MARKET_OPEN) & (d['ts'] < MARKET_CLOSE)
    d = d[keep][:N]
    X = d['f'].astype(np.float32)
    print(f'\ncomparing {len(X)} market-hours records')

    py = forward_int8(w, X)

    cpp = np.fromfile(CPP_LOGITS, dtype='<f4')
    if cpp.size % w['n_out'] != 0:
        raise ValueError(f'{CPP_LOGITS}: {cpp.size} floats, not a multiple '
                         f'of {w["n_out"]}')
    cpp = cpp.reshape(-1, w['n_out'])

    n = min(len(py), len(cpp))
    if len(py) != len(cpp):
        print(f'note: python {len(py)} rows, C++ {len(cpp)}; comparing {n}')
    py, cpp = py[:n], cpp[:n]

    diff = np.abs(py - cpp)
    disagree = int((py.argmax(1) != cpp.argmax(1)).sum())

    print(f'\n  max  |diff| : {diff.max():.3e}')
    print(f'  mean |diff| : {diff.mean():.3e}')
    print(f'  argmax disagreements: {disagree} / {n}')
    print(f'\nfirst row:')
    print(f'  python: {py[0][0]:12.6f} {py[0][1]:12.6f} {py[0][2]:12.6f}')
    print(f'  c++   : {cpp[0][0]:12.6f} {cpp[0][1]:12.6f} {cpp[0][2]:12.6f}')

    # The integer arithmetic is identical on both sides, so the only
    # residual is float32 rounding in the per-neuron scale/bias step, plus
    # the reciprocal the C++ uses where this divides. 1e-3 on logits of
    # magnitude ~2 is generous; a structural bug would be far larger.
    if diff.max() > 1e-3 or disagree > 0:
        worst = int(diff.max(axis=1).argmax())
        print(f'\nworst row {worst}:')
        print(f'  python: {py[worst]}')
        print(f'  c++   : {cpp[worst]}')
        print('\nFAIL -- the offline and online int8 paths disagree. Check, '
              'in order: the scales printed above against calibrate.py\'s '
              'output; the weight file layout; the rounding rule in '
              'quantize.')
        sys.exit(1)

    print('\nPASS -- the numpy reimplementation and the C++ int8 backend '
          'agree. The scales, file layout, and rounding rules match.')


if __name__ == '__main__':
    main()