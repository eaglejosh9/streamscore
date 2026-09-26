"""Train a small MLP on the replayed feature dump, then ablate features.

The model is deliberately tiny: phase 5 reimplements this forward pass by
hand in C++ with int8 weights, and every extra layer is work there.

Run from the python/ directory:
    python train.py ../data/aapl.bin
    python train.py ../data/aapl.bin --ablate
"""

import sys

import numpy as np
import torch
import torch.nn as nn

from dataset import (
    load, filter_rows, make_labels, features, normalize, split,
)

torch.manual_seed(0)
np.random.seed(0)

CLASS_NAMES = ['down', 'flat', 'up']

FEATURE_NAMES = [
    'imb1', 'imb3', 'imb5', 'rel_spread', 'microdev', 'mom_fast', 'mom_slow',
    'signed_vol', 'trade_rate', 'log_bid_depth', 'log_ask_depth',
    'bid_slope', 'ask_slope', 'bid_n', 'ask_n', 'realized_vol',
]

# Validation accuracy peaks around epoch 5 and declines after -- the signal
# does not support more fitting than this.
EPOCHS = 8
BATCH = 4096


class MLP(nn.Module):
    def __init__(self, n_in=16, n_hidden=32, n_out=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def majority_baseline(y_train, y_eval):
    majority = np.bincount(y_train, minlength=3).argmax()
    return (y_eval == majority).mean(), majority


def evaluate(model, X, y, device):
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        pred = logits.argmax(dim=1).cpu().numpy()

    overall = (pred == y).mean()

    per_class = {}
    for c in range(3):
        mask = y == c
        per_class[CLASS_NAMES[c]] = (
            (pred[mask] == c).mean() if mask.sum() else float('nan')
        )

    confusion = np.zeros((3, 3), dtype=np.int64)
    for true_c in range(3):
        for pred_c in range(3):
            confusion[true_c, pred_c] = ((y == true_c) & (pred == pred_c)).sum()

    return overall, per_class, confusion, pred


def train_model(Xtr, ytr, Xva, yva, device='cpu', epochs=EPOCHS, verbose=True,
                seed=0):
    """Train one MLP and return the weights from the best validation epoch.

    Shared by main() and ablate() so the two paths cannot drift apart --
    an ablation trained differently from the baseline would not be a
    comparison of features, it would be a comparison of training runs.
    """
    torch.manual_seed(seed)

    model = MLP(n_in=Xtr.shape[1]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)

    best_val = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        total_loss = 0.0

        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)

        val_acc, _, _, _ = evaluate(model, Xva, yva, device)

        if verbose:
            train_acc, _, _, _ = evaluate(model, Xtr, ytr, device)
            print(f'epoch {epoch + 1:2d}  loss {total_loss / len(Xtr):.4f}  '
                  f'train {train_acc:.4f}  val {val_acc:.4f}')

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, best_val


def ablate(Xtr, ytr, Xva, yva, Xte, yte, base_acc, device='cpu'):
    """Zero one feature at a time, retrain, and report the accuracy delta.

    Zeroing rather than dropping the column keeps the input width at 16, so
    every run trains an identically-shaped model. A negative delta means the
    feature was carrying signal; a delta near zero means it was not.
    """
    print(f'\n{"feature":>16} {"test acc":>10} {"delta":>10}')
    print('-' * 38)

    results = []
    for i, name in enumerate(FEATURE_NAMES):
        Xtr_a, Xva_a, Xte_a = Xtr.copy(), Xva.copy(), Xte.copy()
        Xtr_a[:, i] = 0.0
        Xva_a[:, i] = 0.0
        Xte_a[:, i] = 0.0

        model, _ = train_model(Xtr_a, ytr, Xva_a, yva, device, verbose=False)
        acc, _, _, _ = evaluate(model, Xte_a, yte, device)
        delta = acc - base_acc
        results.append((name, acc, delta))
        print(f'{name:>16} {acc:10.4f} {delta:+10.4f}')

    print('\nmost important (largest drop when removed):')
    for name, acc, delta in sorted(results, key=lambda r: r[2])[:5]:
        print(f'  {name:>16} {delta:+.4f}')


def print_confusion(cm):
    print('        pred:  down     flat       up')
    for i, name in enumerate(CLASS_NAMES):
        row = '  '.join(f'{v:8d}' for v in cm[i])
        print(f'  true {name:>4}:  {row}')


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    path = args[0] if args else '../data/aapl.bin'
    do_ablate = '--ablate' in sys.argv
    device = 'cpu'

    d = load(path)
    print(f'loaded {len(d):,} records')

    d = filter_rows(d)
    print(f'after market-hours + spread filter: {len(d):,}')

    y, n = make_labels(d)
    X = features(d)[:n]

    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(X, y)
    Xtr, stats = normalize(Xtr)
    Xva, _ = normalize(Xva, stats)
    Xte, _ = normalize(Xte, stats)

    dist = np.bincount(ytr, minlength=3) / len(ytr)
    print(f'train split: {len(Xtr):,}   val: {len(Xva):,}   test: {len(Xte):,}')
    print(f'train label distribution: down={dist[0]:.3f} '
          f'flat={dist[1]:.3f} up={dist[2]:.3f}')

    base_acc, majority = majority_baseline(ytr, yte)
    print(f'majority-class baseline on test '
          f'(always predict "{CLASS_NAMES[majority]}"): {base_acc:.4f}')
    print()

    model, best_val = train_model(Xtr, ytr, Xva, yva, device)
    print(f'\nrestored best val epoch (val {best_val:.4f})')

    test_acc, per_class, cm, _ = evaluate(model, Xte, yte, device)
    print()
    print(f'test accuracy:   {test_acc:.4f}')
    print(f'baseline:        {base_acc:.4f}')
    print(f'lift:            {test_acc - base_acc:+.4f}')
    print()
    print('per-class recall:')
    for name, acc in per_class.items():
        print(f'  {name:>4}: {acc:.4f}')
    print()
    print_confusion(cm)

    weights = {}
    for i, layer in enumerate(model.net):
        if isinstance(layer, nn.Linear):
            weights[f'w{i}'] = layer.weight.detach().cpu().numpy()
            weights[f'b{i}'] = layer.bias.detach().cpu().numpy()
    weights['norm_mean'] = stats[0]
    weights['norm_std'] = stats[1]
    np.savez('weights.npz', **weights)
    print('\nwrote weights.npz')

    if do_ablate:
        ablate(Xtr, ytr, Xva, yva, Xte, yte, test_acc, device)


if __name__ == '__main__':
    main()