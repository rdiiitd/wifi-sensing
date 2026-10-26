import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
for name, t0, t1, lab in [('m_empty', 100, 200, 'static'), ('m01', 50, 150, 'pre-walk'), ('m01', 360, 460, 'walking/collisions'), ('x02', 150, 250, 'walking')]:
    for mac in [sd.ROUTER, sd.LAPTOP]:
        t, c, r = sd.load_csi(name, mac, t0, t1)
        lab_m, _ = sd.track_modes(t, c)
        m = lab_m == 0
        p = 10 * np.log10(np.mean(np.abs(c[m]) ** 2, 1))
        rr = r[m].astype(float)
        # 0.5 s bins
        b = ((t[m] - t0) // 0.5).astype(int)
        n = np.bincount(b)
        ok = n > 20
        pb = (np.bincount(b, p) / np.maximum(n, 1))[ok]
        rb = (np.bincount(b, rr) / np.maximum(n, 1))[ok]
        cc = np.corrcoef(pb, rb)[0, 1]
        slope = np.polyfit(rb, pb, 1)[0]
        print(f'{name:7s} {lab:20s} {mac[:2]}  RSSI sd {rb.std():4.2f} dB  CSI-power sd {pb.std():4.2f} dB  corr {cc:+.2f}  slope dCSI/dRSSI {slope:+.2f}'
              f'   per-pkt: CSI-pwr sd {p.std():.2f} dB, RSSI sd {rr.std():.2f}')
