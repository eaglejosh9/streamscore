# python/export.py
"""Convert weights.npz into a flat binary the C++ backends can fread.

Layout is fixed and versioned; include/model.hpp must agree exactly.
"""

import struct
import numpy as np

MAGIC = 0x53544D57   # "STMW"
VERSION = 1


def main():

    """
    Byte budget
    header	5 x 4	20	magic, version, n_in, n_hidden, n_out
    norm_mean	16 x 4	64	z-score mean per feature
    norm_std	16 x 4	64	z-score std per feature
    w0	512 x 4	2048	(32, 16)
    b0	32 x 4	128	(32,)
    w2	1024 x 4	4096	(32, 32)
    b2	32 x 4	128	(32,)
    w4	96 x 4	384	(3, 32)
    b4	3 x 4	12	(3,)
    total	6944	
    """
    w = np.load('weights.npz')

    n_in = w['w0'].shape[1]
    n_hidden = w['w0'].shape[0]
    n_out = w['w4'].shape[0]

    # Shapes must be exactly what the C++ side expects. Assert rather than
    # trust -- a silent shape mismatch would read garbage over there.
    assert w['w0'].shape == (n_hidden, n_in)
    assert w['b0'].shape == (n_hidden,)
    assert w['w2'].shape == (n_hidden, n_hidden)
    assert w['b2'].shape == (n_hidden,)
    assert w['w4'].shape == (n_out, n_hidden)
    assert w['b4'].shape == (n_out,)
    assert w['norm_mean'].shape == (n_in,)
    assert w['norm_std'].shape == (n_in,)

    with open('../data/weights.bin', 'wb') as f:
        # Header: magic, version, then the three dimensions.
        f.write(struct.pack('<5I', MAGIC, VERSION, n_in, n_hidden, n_out))

        # tofile writes raw bytes with no numpy header. '<f4' forces
        # little-endian float32. Row-major, so each neuron's weights are
        # contiguous -- which is the order the kernel loops in.
        for key in ['norm_mean', 'norm_std',
                    'w0', 'b0', 'w2', 'b2', 'w4', 'b4']:
            w[key].astype('<f4').tofile(f)

    expected = 20 + 4 * (2 * n_in
                         + n_hidden * n_in + n_hidden
                         + n_hidden * n_hidden + n_hidden
                         + n_out * n_hidden + n_out)
    import os
    actual = os.path.getsize('../data/weights.bin')
    print(f'wrote ../data/weights.bin')
    print(f'  dims: n_in={n_in} n_hidden={n_hidden} n_out={n_out}')
    print(f'  size: {actual} bytes (expected {expected})')
    assert actual == expected, 'size mismatch -- layout is wrong'


if __name__ == '__main__':
    main()