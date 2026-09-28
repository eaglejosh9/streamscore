import numpy as np
a = np.fromfile('../data/logits_int8.bin', '<f4').reshape(-1, 3)
b = np.fromfile('../data/logits_neon.bin', '<f4').reshape(-1, 3)
assert a.shape == b.shape, f'{a.shape} vs {b.shape}'
exact = np.array_equal(a, b)
print('rows compared:', len(a))
print('bitwise identical:', exact)
if not exact:
    d = np.abs(a - b)
    print('max |diff|:', d.max())
    print('rows differing:', (d.max(axis=1) > 0).sum())
    i = np.unravel_index(d.argmax(), d.shape)
    print(f'worst row {i[0]}: int8 {a[i[0]]}  neon {b[i[0]]}')
