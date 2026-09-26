import numpy as np
from dataset import load, RECORD_DTYPE, N_FEATURES

d = load('../data/aapl.bin')
f = d['f']
print(f'{len(d):,} records, {f.shape[1]} features')

# Nothing may be NaN or inf — a single one wrecks training.
bad = ~np.isfinite(f)
print('non-finite values:', bad.sum())
if bad.sum():
    print('  columns affected:', np.unique(np.where(bad)[1]))

names = ['imb1','imb3','imb5','rel_spread','microdev','mom_fast','mom_slow',
         'signed_vol','trade_rate','log_bid_depth','log_ask_depth',
         'bid_slope','ask_slope','bid_n','ask_n','realized_vol']
print(f'\n{"":>16} {"min":>12} {"p1":>12} {"p50":>12} {"p99":>12} {"max":>12}')
for i, nm in enumerate(names):
    c = f[:, i]
    print(f'{nm:>16} {c.min():12.4f} {np.percentile(c,1):12.4f} '
          f'{np.median(c):12.4f} {np.percentile(c,99):12.4f} {c.max():12.4f}')

print('\nfirst row momentum (should be exactly 0.0):', f[0, 5], f[0, 6])
print('imbalances in [-1,1]:', np.abs(f[:, 0:3]).max() <= 1.0)
print('level counts in [0.2,1]:', f[:, 13:15].min() >= 0.2, f[:, 13:15].max() <= 1.0)
