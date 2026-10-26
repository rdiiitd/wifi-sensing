"""z01: does sharper speed resolution separate the approacher from the wearer's own walking?

The 180 ms Doppler window smears every speed by +-0.64 m/s (Hann main lobe), about the whole gap between
the wearer's antenna speed (~0.3-0.6 m/s while walking) and the approacher's reflection (up to ~2x their
0.3-0.65 m/s walking speed). Longer windows give finer speed: 360 ms -> +-0.32 m/s, 500 ms -> +-0.23 m/s,
at the cost of blurring changes in time.

Segments (video s): still alone 45-85, walking alone 205-265, head turns approach 345-445, walking approach 445-505.
SHARP-cleaned CSI per link is cached per segment (cache/seg_*.npz), then spectra with several windows.
Gate: share of power faster than max(win-lobe, 2 x IMU antenna speed (same window) + win-lobe).
Same tests as z01_walking.py (the head-turn approach part is only used for the T2/T3 check).
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

SEGMENTS = {'still alone': (45, 85), 'walking alone': (205, 265), 'turns approach': (345, 445),
            'walking approach': (445, 505)}
WINDOWS = [(36, 100), (72, 200), (100, 200)]   # (samples at 200 Hz, FFT size)


def cleaned(lab, mac, mode):
    """SHARP-cleaned, 200 Hz CSI of one link over one segment (cached)."""
    v0, v1 = SEGMENTS[lab]
    t0, t1 = zw.to_pcap([v0, v1])
    p = os.path.join(sd.CACHE_DIR, f'seg_{NAME}_{v0}_{v1}_{mac[:2]}_{mode}.npz')
    if os.path.exists(p):
        z = np.load(p)
        return t0, z['h'], z['valid']
    t, csi, _ = sd.load_csi(NAME, mac, t0 - 1, t1 + 1)
    lab_m, _ = sd.track_modes(t, csi)
    t, csi = t[lab_m == mode], csi[lab_m == mode]
    h = sd.sanitize(csi, 'sharp', t)
    _, hu, valid = sd.resample_uniform(t, h, t0, t1)
    np.savez_compressed(p, h=hu.astype(np.complex64), valid=valid)
    return t0, hu, valid


def profiles(n_win, n_fft):
    """4-link mean sum-normalised Doppler profiles, 20 ms hop, all segments concatenated."""
    ts, Ps = [], []
    for lab in SEGMENTS:
        acc, cnt, tref = None, None, None
        for mac, mode, _ in mf.LINKS:
            t0, hu, valid = cleaned(lab, mac, mode)
            tc, vel, D = sd.doppler_spectrogram(hu, n_win=n_win, n_fft=n_fft, hop=mf.HOP, noise_db=mf.RAW_DB)
            ok = valid[np.round(tc * sd.FS).astype(int).clip(0, len(valid) - 1)]
            P = D / D.sum(1, keepdims=True)
            if tref is None:
                tref, acc, cnt = t0 + tc, np.zeros_like(P), np.zeros(len(tc))
            n = min(len(tc), len(tref))
            acc[:n] += np.where(ok[:n, None], P[:n], 0)
            cnt[:n] += ok[:n]
        good = cnt > 0
        ts.append(tref[good])
        Ps.append(acc[good] / cnt[good, None])
    return np.concatenate(ts), vel, np.concatenate(Ps)


if __name__ == '__main__':
    tf0, _, _ = zw.link_profiles()
    imu0 = zw.imu_frames(tf0)
    tc = np.arange(imu0['ti0'] + 1, imu0['ti1'] - 1, zi.STEP)
    td, dist = zi.load_distance(zw.to_pcap)
    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])
    ends = np.array([t for t, e in ev if e == 'collision_end'])
    E = zw.Evaluator(tc, zi.at(tc, td, dist), starts, ends, td, dist)
    Z = None

    for n_win, n_fft in WINDOWS:
        lobe = 2 * sd.FS / n_win * sd.LAMBDA
        tf, vel, P = profiles(n_win, n_fft)
        av = np.abs(vel)
        ti, acc, gyr = zw.ss.load_imu_raw()
        va_s, _, _, _, _ = zw.ss.antenna_speed(ti, acc, gyr, zw.to_pcap)
        # antenna speed averaged over this window length (antenna_speed uses 180 ms)
        va_s = np.convolve(va_s, np.ones(int(n_win / sd.FS * 800)) / int(n_win / sd.FS * 800), mode='same')
        va = np.interp(tf, ti, va_s)
        gate = np.maximum(lobe, zw.ss.SELF_FACTOR * va + lobe)
        print(f'== window {n_win / sd.FS * 1000:.0f} ms: speed smear +-{lobe:.2f} m/s ==')
        for lab, (v0, v1) in SEGMENTS.items():
            t0, t1 = zw.to_pcap([v0, v1])
            m = (tf >= t0) & (tf < t1)
            print(f'  {lab:17s} antenna {va[m].mean():.2f} m/s   edge (98 %) {np.mean(zw.ss.edge_speed(vel, P[m])):.2f} m/s'
                  f'   gated {np.sum(P[m] * (av[None] > gate[m][:, None]), 1).mean():.4f}')
        raw = zw.windows(tc, tf, P[:, av > max(0.7, lobe)].sum(1))
        gated = zw.windows(tc, tf, np.sum(P * (av[None] > gate[:, None]), 1))
        if Z is None:
            W = lambda x: zw.windows(tc, tf0, x)
            Z = np.c_[W(imu0['turn']), W(imu0['shake']), W(imu0['va'])]
            Z = (np.log1p(Z) - np.nanmean(np.log1p(Z), 0)) / np.nanstd(np.log1p(Z), 0)
        print(zw.fmt('motion score', E.run(raw)))
        print(zw.fmt('IMU speed gate', E.run(gated)))
        print(zw.fmt('IMU speed gate, IMU-matched', E.run(zw.imu_matched(tc, gated, Z, E.sets['walking']['alone']))))
        print(zw.fmt('IMU speed gate, 1 s', E.run(zw.smooth(gated, 4))))
        r = E.run(gated, 'head turns', ci=False)
        print(f'     head-turn approaches (check): T2 {r["T2"][0]:.2f}  T3 {r["T3"][0]}/{r["T3"][1]}')
