"""Collision-locked packet rate per transmitter (decoded CSI frames / s) and RSSI of other devices."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
from check_rssi_events import test

FS = 5.0  # 0.2 s bins
for name in ['m01', 'x02']:
    idx = sd.index_pcap(name)
    rel = idx['ts'] - idx['ts'][0]
    for mac in [sd.ROUTER, sd.LAPTOP, 'de:62:98:3b:a3:9c']:
        s = idx['macs'] == mac
        if s.sum() < 1000:
            continue
        b = (rel[s] * FS).astype(int)
        n = np.bincount(b, minlength=int(rel[-1] * FS) + 1).astype(float) * FS
        tt = np.arange(len(n)) / FS
        k = 3
        rate = np.convolve(n, np.ones(k) / k, 'same')
        feats = {'rate': rate}
        if mac.startswith('de'):
            r = np.bincount(b, idx['rssi'][s].astype(float), minlength=len(n)) / np.maximum(np.bincount(b, minlength=len(n)), 1)
            ok = np.bincount(b, minlength=len(n)) > 0
            feats['rssi'] = np.interp(tt, tt[ok], r[ok])
        for key, x in feats.items():
            for win, base in [((0, 1), (-2.5, -0.5)), ((-0.5, 1.5), (-4.5, -2.5))]:
                n_, up, me, mc, sc, p = test(name, tt, x, win, base)
                print(f'{name} {mac[:8]} {key:5s} win{win}: up {up}/{n_}  ev {me:+8.2f}  ctrl {mc:+8.2f}±{sc:7.2f}  two-sided p {p:.3f}')
