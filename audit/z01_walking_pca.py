"""z01: CornerSense's 'remove your own motion first' step (two-stage PCA) with our one-antenna CSI.

CornerSense (He et al., IMWUT 2025) splits the CSI change over a 2.5 s window into patterns across subcarriers
(PCA). The strongest pattern is usually the receiver's own motion; they drop it and look for the person in
the next ones. We have SHARP-cleaned complex CSI (already referenced to one path, the role of their antenna
division), so per link and per 2.5 s window (0.25 s step):
  - remove the mean, PCA across the 53 subcarriers
  - share of the change in components 2-3 (what is left after the strongest pattern)
  - Doppler of the component-2 time series: share of power faster than 0.7 m/s
Scored on the walking parts with the same tests as z01_walking.py. Uses the segment cache of
z01_walking_resolution.py (run that first).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
import make_figures as mf
import z01_imu as zi
import z01_walking as zw
import z01_walking_resolution as zr
from z01_check import NAME

WIN_S = 2.5


def pca_features(hu, valid, t0, hop=0.25):
    n = int(WIN_S * sd.FS)
    step = int(hop * sd.FS)
    vel = (np.arange(256) - 128) * sd.FS / 256 * sd.LAMBDA
    out = []
    for s in range(0, len(hu) - n, step):
        if valid[s:s + n].mean() < 0.8:
            out.append((t0 + (s + n / 2) / sd.FS, np.nan, np.nan, np.nan))
            continue
        X = hu[s:s + n] - hu[s:s + n].mean(0)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        e = S ** 2 / np.sum(S ** 2)
        pc2 = U[:, 1] * S[1]
        F = np.abs(np.fft.fftshift(np.fft.fft(pc2 * np.hanning(n), 256))) ** 2
        out.append((t0 + (s + n / 2) / sd.FS, e[0], e[1] + e[2], F[np.abs(vel) > 0.7].sum() / F.sum()))
    return np.array(out)


if __name__ == '__main__':
    tf0, _, _ = zw.link_profiles()
    imu0 = zw.imu_frames(tf0)
    tc = np.arange(imu0['ti0'] + 1, imu0['ti1'] - 1, zi.STEP)
    td, dist = zi.load_distance(zw.to_pcap)
    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])
    ends = np.array([t for t, e in ev if e == 'collision_end'])
    E = zw.Evaluator(tc, zi.at(tc, td, dist), starts, ends, td, dist)

    rows = []
    for lab in zr.SEGMENTS:
        per_link = []
        for mac, mode, _ in mf.LINKS:
            t0, hu, valid = zr.cleaned(lab, mac, mode)
            per_link.append(pca_features(hu, valid, t0))
        n = min(len(p) for p in per_link)
        rows.append(np.c_[per_link[0][:n, :1], np.nanmean(np.stack([p[:n, 1:] for p in per_link]), 0)])
    R = np.concatenate(rows)
    print('== share of CSI change in the strongest pattern (own motion?) and in the next two ==')
    for lab, (v0, v1) in zr.SEGMENTS.items():
        a, b = zw.to_pcap([v0, v1])
        m = (R[:, 0] >= a) & (R[:, 0] < b)
        print(f'  {lab:17s} strongest {np.nanmean(R[m, 1]):.2f}   next two {np.nanmean(R[m, 2]):.2f}   '
              f'pattern-2 fast Doppler {np.nanmean(R[m, 3]):.3f}')
    for j, name in [(1, 'strongest pattern share'), (2, 'next two patterns share'), (3, 'pattern 2: fast Doppler share')]:
        x = np.interp(tc, R[:, 0], R[:, j], left=np.nan, right=np.nan)
        x[np.isnan(np.interp(tc, R[:, 0], np.isnan(R[:, j]).astype(float)))] = np.nan
        gap = np.min(np.abs(tc[:, None] - R[None, :, 0]), 1) > 0.2
        x[gap] = np.nan
        print(zw.fmt(name, E.run(x)))
