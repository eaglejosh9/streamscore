import numpy as np
d = np.fromfile('data/aapl.bin', dtype=np.dtype([
    ('ts','<u8'),('mid','<i4'),('spread','<i4'),
    ('imbalance','<f4'),('ema_diff','<f4'),('signed_vol','<f4')]), offset=8)
print(d[:5])
print('mid range', d['mid'].min(), d['mid'].max())
print('spread p50', np.median(d['spread']))
print('imbalance range', d['imbalance'].min(), d['imbalance'].max())
print('nans', np.isnan(d['imbalance']).sum(), np.isnan(d['ema_diff']).sum())
i = d['mid'].argmax()
print('widest mid row:', d[i])
print('spread max', d['spread'].max(), 'p99', np.percentile(d['spread'], 99))
