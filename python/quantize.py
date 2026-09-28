"""Quantize the trained weights to int8 and measure the accuracy cost.

Run before writing any SIMD: if int8 costs several points of accuracy, the
kernel is not worth writing as designed and the fix is a different scheme
(per-row scales, or int8 weights with float activations).
"""

import numpy as np
import torch

from dataset import load, filter_rows, make_labels, features, normalize, split
from train import MLP, evaluate, majority_baseline


def quantize_tensor(t, per_row=False):
    if per_row:
        scale = np.abs(t).max(axis=1, keepdims=True) / 127.0
    else:
        scale = np.abs(t).max() / 127.0

    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.round(t / scale), -127, 127).astype(np.int8)
    return q, np.asarray(scale, dtype=np.float32)


def dequantize(q, scale):
    """q * scale, back to float32."""
    return (q.astype(np.float32) * scale).astype(np.float32)


def quantization_error(t, per_row=False):
    """Report how much a tensor changed. Useful for spotting which layer
    is worst before looking at end-to-end accuracy."""
    q, scale = quantize_tensor(t, per_row)
    deq = dequantize(q, scale)
    max_abs = float(np.abs(t - deq).max())
    rel = float(np.linalg.norm(t - deq) / np.linalg.norm(t))
    return max_abs, rel

def eval_quantized(w, Xte, yte, per_row):
    """Build a model whose weights have been round-tripped through int8."""
    model = MLP(n_in=16)
    for src, idx in (('0', 0), ('2', 2), ('4', 4)):
        q, scale = quantize_tensor(w[f'w{src}'], per_row)
        deq = dequantize(q, scale)
        model.net[idx].weight.data = torch.from_numpy(deq)
        # Biases stay float32 -- 67 values total, quantizing them adds
        # error for no memory saving worth having.
        model.net[idx].bias.data = torch.from_numpy(w[f'b{src}'].copy())
    model.eval()
    acc, _, _, _ = evaluate(model, Xte, yte, 'cpu')
    return acc


def main():
    d = filter_rows(load('../data/aapl.bin'))
    y, n = make_labels(d)
    X = features(d)[:n]

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    Xtr, stats = normalize(Xtr)
    Xte, _ = normalize(Xte, stats)

    w = np.load('weights.npz')

    # Baseline: the float model, so the comparison is apples to apples.
    model = MLP(n_in=16)
    for src, idx in (('0', 0), ('2', 2), ('4', 4)):
        model.net[idx].weight.data = torch.from_numpy(w[f'w{src}'].copy())
        model.net[idx].bias.data = torch.from_numpy(w[f'b{src}'].copy())
    model.eval()
    float_acc, _, _, _ = evaluate(model, Xte, yte, 'cpu')
    print(f'float32 test accuracy: {float_acc:.4f}')

    # TODO 1. Quantize w0, w2, w4. Print the per-tensor error for each so
    #         you can see which layer suffers most.
    #         Leave the biases in float32 -- they are 67 values total and
    #         quantizing them buys nothing while adding error.
    print('\nquantization error by layer:')
    for name in ['w0', 'w2', 'w4']:
        pt_max, pt_rel = quantization_error(w[name], per_row=False)
        pr_max, pr_rel = quantization_error(w[name], per_row=True)
        print(f'  {name}: per-tensor {pt_rel:.3%}  per-row {pr_rel:.3%}')

    # TODO 2. Build a model with the DEQUANTIZED weights (round-tripped
    #         through int8) and evaluate it. This measures the accuracy
    #         cost of the precision loss alone, separate from any kernel
    #         bug -- it is the number that decides whether to proceed.

    # TODO 3. Repeat with per_row=True and compare. Report both.
    pt_acc = eval_quantized(w, Xte, yte, per_row=False)
    pr_acc = eval_quantized(w, Xte, yte, per_row=True)

    print(f'\nfloat32:          {float_acc:.4f}')
    print(f'int8 per-tensor:  {pt_acc:.4f}  ({pt_acc - float_acc:+.4f})')
    print(f'int8 per-row:     {pr_acc:.4f}  ({pr_acc - float_acc:+.4f})')

    # TODO 4. Write the int8 weights and scales to ../data/weights_int8.bin
    #         for the NEON kernel. Same discipline as export.py: magic,
    #         version, dims, then the arrays. Pick a new magic.
    #         Decide: do activations get quantized too, or do you keep them
    #         float and only quantize weights? The latter is simpler and
    #         still gets you the memory win; the former is what makes SDOT
    #         usable. Note which you chose and why.


if __name__ == '__main__':
    main()