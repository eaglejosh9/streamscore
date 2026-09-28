import numpy as np
a = np.fromfile('../data/logits.bin', '<f4').reshape(-1, 3)
b = np.fromfile('../data/logits_int8.bin', '<f4').reshape(-1, 3)
print('max |diff|', np.abs(a-b).max())
print('relative', np.abs(a-b).max() / np.abs(a).max())
print('argmax disagreements', (a.argmax(1) != b.argmax(1)).sum(), '/', len(a))
