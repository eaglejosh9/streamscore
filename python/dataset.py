"""Load the replay dump, filter it, label it, and split it chronologically.

The dump format is defined by feat::Record in include/features.hpp. If that
struct changes, DUMP_VERSION must be bumped and RECORD_DTYPE updated here.
"""

import numpy as np

N_FEATURES = 16

RECORD_DTYPE = np.dtype([
    ('ts',     '<u8'),
    ('mid',    '<i4'),
    ('spread', '<i4'),
    ('f',      '<f4', (N_FEATURES,)),
])

DUMP_MAGIC = 0x53544D32
DUMP_VERSION = 2

# Regular market hours, ns since midnight ET.
MARKET_OPEN = int(9.5 * 3600 * 1e9)
MARKET_CLOSE = int(16.0 * 3600 * 1e9)

# Rows wider than this are thin-book noise. 22 cents is roughly the p99 of
# the full dump; inside market hours almost nothing exceeds it.
MAX_SPREAD = 2200


def load(path):
    with open(path, 'rb') as f:
        magic, version = np.fromfile(f, dtype='<u4', count=2)
        if magic != DUMP_MAGIC:
            raise ValueError(f'bad magic {magic:#x}')
        if version != DUMP_VERSION:
            raise ValueError(f'unsupported dump version {version}')
        return np.fromfile(f, dtype=RECORD_DTYPE)


def filter_rows(d):
    """Keep regular market hours and drop implausibly wide books.

    Pre-market and after-hours books are thin enough that the mid is not a
    meaningful price -- the widest row in the dump has an $18 spread at 8pm.
    Training on those teaches the model about a regime we do not care about.
    """
    keep = (
        (d['ts'] >= MARKET_OPEN)
        & (d['ts'] < MARKET_CLOSE)
        & (d['spread'] > 0)
        & (d['spread'] <= MAX_SPREAD)
    )
    return d[keep]


def make_labels(d, horizon=100, threshold=50):
    """Label each row by the mid price `horizon` rows later.

    0 = down, 1 = flat, 2 = up. `threshold` is in price units (4 implied
    decimals), so 50 is half a cent. Without a threshold most windows have
    zero net change and the task becomes classifying rounding noise.

    Returns (labels, n) where n is the number of rows that have a future to
    compare against. The caller must truncate features to the same n.
    """
    n = len(d) - horizon
    if n <= 0:
        raise ValueError('not enough rows for this horizon')

    current = d['mid'][:n].astype(np.int64)
    future = d['mid'][horizon:horizon + n].astype(np.int64)
    delta = future - current

    labels = np.full(n, 1, dtype=np.int64)
    labels[delta < -threshold] = 0
    labels[delta > threshold] = 2
    return labels, n


def features(d):
    """The 16 floats are already contiguous and in order in the dump."""
    return d['f'].astype(np.float32)


def normalize(X, stats=None):
    """Z-score each column.

    stats must be computed on the training split only and reused for val and
    test. Computing it over the whole array leaks the test distribution into
    training.
    """
    if stats is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        stats = (mean, std)
    mean, std = stats
    return ((X - mean) / std).astype(np.float32), stats


def split(X, y, train_frac=0.7, val_frac=0.15):
    """Chronological split -- deliberately not random.

    Adjacent rows are milliseconds apart and nearly identical. A random split
    puts near-duplicates in both train and test, which inflates test accuracy
    into fiction.
    """
    n = len(X)
    i = int(n * train_frac)
    j = int(n * (train_frac + val_frac))
    return (X[:i], y[:i]), (X[i:j], y[i:j]), (X[j:], y[j:])
