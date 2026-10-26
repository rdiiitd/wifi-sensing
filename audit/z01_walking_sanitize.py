"""z01: is the walking Doppler real motion, or the phase cleaning breaking while the wearer walks?

Same walking parts and tests as z01_walking.py, with the Doppler computed from three versions of the CSI:
  sharp  : SHARP path-referenced phase (our pipeline; cached full-capture spectrograms)
  linear : per-packet linear phase removed (simple, no path model)
  power  : amplitude only (|h|^2, no phase at all; immune to phase errors, cannot tell towards from away)
If walking's fast power is a phase-cleaning artefact, it should shrink a lot in 'power'.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
import make_figures as mf
import z01_imu as zi
import z01_walking as zw
from z01_check import NAME

SEGMENTS = {'still alone': (45, 85), 'walking alone': (205, 265), 'walking approach': (445, 505)}


def segment_profiles(method):
    """4-link mean sum-normalised profiles over the SEGMENTS (20 ms hop), concatenated in time."""
    ts, Ps = [], []
    for lab, (v0, v1) in SEGMENTS.items():
        t0, t1 = zw.to_pcap([v0, v1])
        acc, cnt, tref = None, None, None
        for mac, mode, _ in mf.LINKS:
            t, vel, D, valid = sd.link_spectrogram(NAME, mac, mode, t0, t1, method=method, hop=mf.HOP,
                                                   noise_db=mf.RAW_DB)
            P = D / D.sum(1, keepdims=True)
            if tref is None:
                tref, acc, cnt = t, np.zeros_like(P), np.zeros(len(t))
            n = min(len(t), len(tref))
            acc[:n] += np.where(valid[:n, None], P[:n], 0)
            cnt[:n] += valid[:n]
        ok = cnt > 0
        ts.append(tref[ok])
        Ps.append(acc[ok] / cnt[ok, None])
    return np.concatenate(ts), vel, np.concatenate(Ps)


if __name__ == '__main__':
    tf_all, vel, LP = zw.link_profiles()
    imu_all = zw.imu_frames(tf_all)
    tc = np.arange(imu_all['ti0'] + 1, imu_all['ti1'] - 1, zi.STEP)
    td, dist = zi.load_distance(zw.to_pcap)
    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])
    ends = np.array([t for t, e in ev if e == 'collision_end'])
    E = zw.Evaluator(tc, zi.at(tc, td, dist), starts, ends, td, dist)

    for method in ['sharp', 'linear', 'power']:
        if method == 'sharp':
            tf, P = tf_all, np.nanmean(np.stack(LP), 0)
        else:
            tf, vel, P = segment_profiles(method)
        av = np.abs(vel)
        imu = zw.imu_frames(tf)
        gate = np.maximum(0.7, zw.ss.SELF_FACTOR * imu['va'] + zw.ss.MARGIN)
        print(f'== {method} ==')
        for lab, (v0, v1) in SEGMENTS.items():
            t0, t1 = zw.to_pcap([v0, v1])
            m = (tf >= t0) & (tf < t1)
            print(f'  {lab:17s} power 0.7-2.5 m/s {P[m][:, (av > 0.7) & (av <= 2.5)].sum(1).mean():.4f}   '
                  f'> 4 m/s {P[m][:, av > 4].sum(1).mean():.4f}   '
                  f'edge (98 %) {np.mean(zw.ss.edge_speed(vel, P[m])):.2f} m/s')
        raw = zw.windows(tc, tf, P[:, av > 0.7].sum(1))
        gated = zw.windows(tc, tf, np.sum(P * (av[None] > gate[:, None]), 1))
        print(zw.fmt('motion score', E.run(raw)))
        print(zw.fmt('IMU speed gate', E.run(gated)))
