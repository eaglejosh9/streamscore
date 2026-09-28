"""Find activation ranges, then export everything the int8 kernel needs.

Weights are fixed, so their range is known offline. Activations depend on
the input, so their range has to be observed by pushing real data through
the float model.

Too small a range clips large activations; too large wastes the int8 range
on values that never occur. Clipping is where most of the accuracy loss in
a real quantization scheme comes from -- which is why the weight-only
measurement in quantize.py showed near-zero loss and this step needs its
own number.

    cd python
    python calibrate.py
"""

import struct

import numpy as np
import torch

from dataset import load, filter_rows, make_labels, features, normalize, split
from train import MLP
from quantize import quantize_tensor

MAGIC = 0x53544D51   # "STMQ"
VERSION = 1

CALIB_N = 100_000
PERCENTILE = 99.99


def build_float_model(w):
    model = MLP(n_in=w['w0'].shape[1],
                n_hidden=w['w0'].shape[0],
                n_out=w['w4'].shape[0])
    for src, idx in (('0', 0), ('2', 2), ('4', 4)):
        model.net[idx].weight.data = torch.from_numpy(w[f'w{src}'].copy())
        model.net[idx].bias.data = torch.from_numpy(w[f'b{src}'].copy())
    model.eval()
    return model


def activations(model, X):
    """Run the layers by hand so the intermediates are visible.

    model(X) would compute h1 and h2 and discard them -- they only exist
    inside the forward pass. Calling the Linear layers individually keeps
    them.
    """
    with torch.no_grad():
        x = torch.from_numpy(X)
        h1 = torch.relu(model.net[0](x))
        h2 = torch.relu(model.net[2](h1))
    return x.numpy(), h1.numpy(), h2.numpy()


def collect_ranges(model, X, percentile=PERCENTILE):
    """Max abs value at each quantization point: input, h1, h2.

    Logits are not measured -- nothing downstream consumes them as int8,
    so they stay float.
    """
    x, h1, h2 = activations(model, X)
    return tuple(float(np.percentile(np.abs(a), percentile))
                 for a in (x, h1, h2))


def clip_report(model, X, percentile=PERCENTILE):
    """How much the percentile cutoff actually costs.

    Prints, per quantization point, the true max, the percentile cutoff,
    and the fraction of values that would clip. A large gap between the two
    maxima means outliers exist; a tiny clip fraction means cutting them is
    cheap.
    """
    x, h1, h2 = activations(model, X)
    print(f'\nactivation ranges (calibrated on {len(X):,} training rows, '
          f'p{percentile}):')
    print(f'{"":>8} {"true max":>12} {"p-cutoff":>12} {"scale":>12} '
          f'{"clipped":>10}')

    ranges = {}
    for name, a in (('x', x), ('h1', h1), ('h2', h2)):
        mag = np.abs(a)
        true_max = float(mag.max())
        cutoff = float(np.percentile(mag, percentile))
        clipped = float((mag > cutoff).mean())
        scale = cutoff / 127.0
        ranges[name] = cutoff
        print(f'{name:>8} {true_max:12.4f} {cutoff:12.4f} {scale:12.6f} '
              f'{clipped:9.4%}')

    print('\nA large gap between true max and cutoff means outliers exist. '
          'Clipping them is usually worth it: one rare sample degrades so '
          'that every common value keeps its precision.')
    return ranges


def main():
    d = filter_rows(load('../data/aapl.bin'))
    y, n = make_labels(d)
    X = features(d)[:n]
    (Xtr, ytr), _, _ = split(X, y)
    Xtr, stats = normalize(Xtr)

    w = np.load('weights.npz')
    model = build_float_model(w)

    # Calibrate on TRAINING data only. Using test data here would leak the
    # test distribution into the deployed model, same as computing
    # normalization stats over everything.
    sample = Xtr[:CALIB_N]
    ranges = clip_report(model, sample)

    x_scale = np.float32(ranges['x'] / 127.0)
    h1_scale = np.float32(ranges['h1'] / 127.0)
    h2_scale = np.float32(ranges['h2'] / 127.0)

    # Per-tensor weight scales -- quantize.py showed per-row halves the
    # reconstruction error but makes no measurable accuracy difference at
    # this size, so take the simpler kernel.
    q0, s0 = quantize_tensor(w['w0'], per_row=False)
    q2, s2 = quantize_tensor(w['w2'], per_row=False)
    q4, s4 = quantize_tensor(w['w4'], per_row=False)

    n_in = w['w0'].shape[1]
    n_hidden = w['w0'].shape[0]
    n_out = w['w4'].shape[0]

    print(f'\nweight scales: w0={float(s0):.6f} w2={float(s2):.6f} '
          f'w4={float(s4):.6f}')

    with open('../data/weights_int8.bin', 'wb') as f:
        f.write(struct.pack('<5I', MAGIC, VERSION, n_in, n_hidden, n_out))

        # Normalization stats -- the kernel applies these before quantizing
        # the input, exactly as ScalarBackend does.
        w['norm_mean'].astype('<f4').tofile(f)
        w['norm_std'].astype('<f4').tofile(f)

        # Activation scales, then weight scales. Order is the contract with
        # include/model_int8.hpp.
        np.array([x_scale, h1_scale, h2_scale], dtype='<f4').tofile(f)
        np.array([float(s0), float(s2), float(s4)], dtype='<f4').tofile(f)

        # int8 weights, float32 biases. Biases are 67 values total --
        # quantizing them would add error for no saving worth having.
        q0.astype(np.int8).tofile(f)
        w['b0'].astype('<f4').tofile(f)
        q2.astype(np.int8).tofile(f)
        w['b2'].astype('<f4').tofile(f)
        q4.astype(np.int8).tofile(f)
        w['b4'].astype('<f4').tofile(f)

    expected = (20
                + 4 * (2 * n_in)
                + 4 * 6
                + n_hidden * n_in + 4 * n_hidden
                + n_hidden * n_hidden + 4 * n_hidden
                + n_out * n_hidden + 4 * n_out)

    import os
    actual = os.path.getsize('../data/weights_int8.bin')
    print(f'\nwrote ../data/weights_int8.bin')
    print(f'  size: {actual} bytes (expected {expected})')
    assert actual == expected, 'size mismatch -- layout is wrong'

    float_bytes = os.path.getsize('../data/weights.bin')
    print(f'  float32 file was {float_bytes} bytes '
          f'({float_bytes / actual:.2f}x larger)')


if __name__ == '__main__':
    main()