"""z01 with the raw headset IMU: can the wearer's own motion be taken out of the Doppler?

Inputs: data/z01/imu/{imu.csv, clock.csv} (Quest 3S, 800 Hz), our cached 4-link spectrograms
(audit/z01_check.py), the collector's auto distance (YOLO, video time) and events.

1. IMU clock -> Pi clock -> PCAP time (clock.csv line fit, then meta 'imu_offset_s').
   Check: the lag between head turn rate and our motion score should be ~0.
2. Per 0.5 s window (0.25 s step): Doppler profile averaged over the 4 links, and the head's
   motion from the IMU (turn rate, shaking = accel minus gravity, both mean and max).
3. IMU baseline: for each window, the 15 wearer-alone windows with the closest head motion
   (none within 3 s of it). Their mean profile = what the wearer's own motion alone looks like.
   - residual score = motion score - the neighbours' motion score
   - gated score    = share of power faster than the neighbours' 95 % speed (at least 0.7 m/s)
4. Tests in the approach part, per kind of wearer motion:
   AUC near (< 0.9 m) vs far (> 1.2 m); contact vs the farthest moment between two contacts;
   did the wearer move differently alone and in the approach part; touch reactions.
Saves 10_z01_imu.png.
"""
import csv
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
import make_figures as mf
from z01_check import NAME, KIND, video_to_pcap

IMU_DIR = os.path.join(sd.DATA_DIR, 'z01', 'imu')
DIST_CSV = os.path.join(sd.DATA_DIR, 'z01', 'z_folder', 'z01_test_auto_distance.csv')
WIN, STEP, K, EXCL = 0.5, 0.25, 15, 3.0
NEAR, FAR = 0.9, 1.2


def boxcar(x, n):
    return np.convolve(x, np.ones(n) / n, mode='same')


def load_imu():
    """Returns PCAP time, turn rate [rad/s], shaking [m/s^2] (accel minus a 0.5 s running mean)."""
    meta = json.load(open(os.path.join(sd.DATA_DIR, f'{NAME}_meta.json')))
    x = np.loadtxt(os.path.join(IMU_DIR, 'imu.csv'), delimiter=',', skiprows=1, usecols=(0, 4, 5, 6, 7, 8, 9))
    clk = np.loadtxt(os.path.join(IMU_DIR, 'clock.csv'), delimiter=',', skiprows=1)
    slope, icpt = np.polyfit(clk[:, 1] * 1e-6, clk[:, 0], 1)
    fit_rms = np.std(clk[:, 0] - (slope * clk[:, 1] * 1e-6 + icpt))
    t = slope * x[:, 0] * 1e-6 + icpt + meta['imu_offset_s'] - sd.index_pcap(NAME)['ts'][0]
    fs = 1 / np.median(np.diff(t))
    acc = x[:, 1:4]
    lin = acc - np.stack([boxcar(acc[:, i], int(0.5 * fs)) for i in range(3)], 1)
    turn = np.linalg.norm(x[:, 4:7], axis=1)
    shake = np.linalg.norm(lin, axis=1)
    print(f'IMU: {len(t)} samples at {fs:.0f} Hz, PCAP {t[0]:.1f}-{t[-1]:.1f} s, clock fit rms {fit_rms * 1e3:.1f} ms, '
          f'drift {(slope - 1) * 1e6:+.0f} ppm')
    return t, turn, shake


def frame_profiles():
    """Per 20 ms Doppler vector: mean (over valid links) of the sum-normalised profile."""
    acc, cnt, t_ref = None, None, None
    for mac, mode, _ in mf.LINKS:
        t, vel, D, valid = mf.load_full(NAME, mac, mode)
        P = D / D.sum(1, keepdims=True)
        if t_ref is None:
            t_ref, acc, cnt = t, np.zeros_like(P), np.zeros(len(t))
        n = min(len(t_ref), len(t))
        acc[:n] += np.where(valid[:n, None], P[:n], 0)
        cnt[:n] += valid[:n]
    ok = cnt > 0
    return t_ref[ok], vel, acc[ok] / cnt[ok, None]


def check_lag(ti, turn, tf, fd, a, b):
    """Lag [s] that best lines up turn rate with our motion score between a and b (positive: CSI later)."""
    m = (tf >= a) & (tf < b)
    g = np.interp(tf[m], ti, boxcar(turn, 144))
    f = fd[m]
    g, f = (g - g.mean()) / g.std(), (f - f.mean()) / f.std()
    lags = np.arange(-50, 51)
    r = [np.mean(g[max(0, -l):len(g) - max(0, l)] * f[max(0, l):len(f) - max(0, -l)]) for l in lags]
    i = int(np.argmax(r))
    return lags[i] * mf.HOP / sd.FS, r[i]


def windows(tf, P, vel, ti, turn, shake):
    starts = np.arange(max(tf[0], ti[0]), min(tf[-1], ti[-1]) - WIN, STEP)
    jf = np.searchsorted(tf, np.r_[starts, starts + WIN])
    ji = np.searchsorted(ti, np.r_[starts, starts + WIN])
    n = len(starts)
    prof, imu, keep = [], [], []
    for k in range(n):
        f0, f1, i0, i1 = jf[k], jf[n + k], ji[k], ji[n + k]
        keep.append(f1 - f0 >= 15)
        prof.append(P[f0:f1].mean(0) if f1 > f0 else np.full(len(vel), np.nan))
        imu.append([turn[i0:i1].mean(), turn[i0:i1].max(), shake[i0:i1].mean(), shake[i0:i1].max()])
    keep = np.array(keep)
    return starts[keep] + WIN / 2, np.array(prof)[keep], np.array(imu)[keep]


def speed95(vel, prof):
    """Speed below which 95 % of the power lies (|v|, both directions folded)."""
    order = np.argsort(np.abs(vel))
    c = np.cumsum(prof[:, order], 1)
    return np.abs(vel)[order][np.argmax(c >= 0.95 * c[:, -1:], 1)]


def imu_baseline(tc, prof, imu, vel, alone):
    """For every window: the K wearer-alone windows with the closest head motion, none within EXCL s."""
    z = np.log1p(imu)
    mu, sdv = z[alone].mean(0), z[alone].std(0)
    z = (z - mu) / sdv
    ia = np.flatnonzero(alone)
    fast = np.abs(vel) > 0.7
    score = prof[:, fast].sum(1)
    v95 = speed95(vel, prof)
    pred, gate = np.empty(len(tc)), np.empty(len(tc))
    for k in range(len(tc)):
        d = np.sum((z[ia] - z[k]) ** 2, 1)
        d[np.abs(tc[ia] - tc[k]) < EXCL] = np.inf
        nn = ia[np.argsort(d)[:K]]
        pred[k] = score[nn].mean()
        gate[k] = max(0.7, np.median(v95[nn]))
    gated = np.array([prof[k, np.abs(vel) > gate[k]].sum() for k in range(len(tc))])
    return score, score - pred, gated, gate


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (np.sum(pos[:, None] > neg[None]) + 0.5 * np.sum(pos[:, None] == neg[None])) / (len(pos) * len(neg))


def sign_p(k, n):
    """One-sided sign test, P(at least k of n by chance)."""
    from math import comb
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def load_distance(to_pcap):
    r = list(csv.DictReader(open(DIST_CSV)))
    tv = np.array([float(x['time_sec']) for x in r])
    d = np.array([float(x['distance_m']) if x['distance_m'] else np.nan for x in r])
    return to_pcap(tv), d


def at(tq, t, x, half=WIN / 2):
    """Median of x within +-half of each query time (nan if none)."""
    out = np.full(len(tq), np.nan)
    for k, q in enumerate(tq):
        m = (t >= q - half) & (t < q + half) & ~np.isnan(x)
        if m.any():
            out[k] = np.median(x[m])
    return out


if __name__ == '__main__':
    to_pcap = video_to_pcap(NAME)
    ti, turn, shake = load_imu()
    tf, vel, P = frame_profiles()
    fd = P[:, np.abs(vel) > 0.7].sum(1)

    A0, A1 = to_pcap([45, 265])
    print('== sync check: lag of our motion score behind head turn rate (after the collector offset) ==')
    for lab, (a, b) in {'alone, head turns': to_pcap([85, 205]), 'alone, all': (A0, A1)}.items():
        lag, r = check_lag(ti, turn, tf, fd, a, b)
        print(f'  {lab:18s} lag {lag:+.2f} s  r {r:.2f}')

    tc, prof, imu = windows(tf, P, vel, ti, turn, shake)
    alone = (tc >= A0) & (tc < A1)
    score, resid, gated, gate = imu_baseline(tc, prof, imu, vel, alone)
    td, dist = load_distance(to_pcap)
    dw = at(tc, td, dist)

    print('== how much of the wearer-alone motion score the IMU explains (neighbours from >3 s away) ==')
    r2 = 1 - np.var(resid[alone]) / np.var(score[alone])
    print(f'  R^2 {r2:.2f}')

    print('== head motion per part: turn rate mean [rad/s], shaking mean [m/s^2]; alone vs approach ==')
    parts = {}
    for kind, (a0, a1, p0, p1) in KIND.items():
        ma = (tc >= to_pcap(a0)) & (tc < to_pcap(a1))
        mp = (tc >= to_pcap(p0)) & (tc < to_pcap(p1))
        parts[kind] = (ma, mp)
        print(f'  {kind:10s} turn {imu[ma, 0].mean():.2f} vs {imu[mp, 0].mean():.2f}   '
              f'shake {imu[ma, 2].mean():.2f} vs {imu[mp, 2].mean():.2f}   '
              f'gate speed {gate[ma].mean():.2f} vs {gate[mp].mean():.2f} m/s')

    print('== scores per part: alone vs approach (mean) ==')
    for kind, (ma, mp) in parts.items():
        print(f'  {kind:10s} raw {score[ma].mean():.3f} vs {score[mp].mean():.3f}   '
              f'residual {resid[ma].mean():+.3f} vs {resid[mp].mean():+.3f}   '
              f'gated {gated[ma].mean():.3f} vs {gated[mp].mean():.3f}')

    print(f'== approach part, windows near (< {NEAR} m) vs far (> {FAR} m): AUC (0.5 = chance) ==')
    for kind, (ma, mp) in parts.items():
        near, far = mp & (dw < NEAR), mp & (dw > FAR)
        line = f'  {kind:10s} near {near.sum():3d} far {far.sum():3d}  '
        for lab, s in [('raw', score), ('residual', resid), ('gated', gated), ('turn rate', imu[:, 0]),
                       ('shaking', imu[:, 2])]:
            line += f'{lab} {auc(s[near], s[far]):.2f}  '
        print(line)

    # contact vs the farthest moment between two contacts (paired)
    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])
    ends = np.array([t for t, e in ev if e == 'collision_end'])
    for c0, c1, what in [(0, 1, '1 s from collision_start'),
                         (-1, 0, '1 s before collision_start, before any touch reaction')]:
        print(f'== contact ({what}) vs farthest moment between contacts (1 s), paired ==')
        for kind, (a0, a1, p0, p1) in KIND.items():
            P0, P1 = to_pcap([p0, p1])
            pairs = []
            for o in starts[(starts >= P0) & (starts < P1)]:
                prev = ends[ends < o]
                if not len(prev) or prev[-1] < P0:
                    continue
                m = (td > prev[-1]) & (td < o) & ~np.isnan(dist)
                if not m.any():
                    continue
                far_t = td[m][np.argmax(dist[m])]
                wc = (tc >= o + c0) & (tc < o + c1)
                wf = (tc >= far_t - 0.5) & (tc < far_t + 0.5)
                if wc.any() and wf.any():
                    pairs.append((dist[m].max(), [s[wc].mean() for s in (score, resid, gated)],
                                  [s[wf].mean() for s in (score, resid, gated)]))
            if not pairs:
                continue
            c, f = np.array([p[1] for p in pairs]), np.array([p[2] for p in pairs])
            n = len(pairs)
            wins = (c > f).sum(0)
            txt = '  '.join(f'{lab} {wins[j]}/{n} (p {sign_p(int(wins[j]), n):.3f})'
                            for j, lab in enumerate(['raw', 'residual', 'gated']))
            print(f'  {kind:10s} farthest {np.median([p[0] for p in pairs]):.2f} m   {txt}')

    print('== approach part, near vs far AUC, split by whether the head is quiet (turn < 0.15 rad/s, shaking < 0.2) ==')
    quiet = (imu[:, 0] < 0.15) & (imu[:, 2] < 0.2)
    mp = (tc >= to_pcap(265)) & (tc < to_pcap(517))
    for lab, m in [('quiet head', mp & quiet), ('moving head', mp & ~quiet)]:
        near, far = m & (dw < NEAR), m & (dw > FAR)
        print(f'  {lab:11s} near {near.sum():3d} far {far.sum():3d}  raw {auc(score[near], score[far]):.2f}  '
              f'residual {auc(resid[near], resid[far]):.2f}  turn rate {auc(imu[near, 0], imu[far, 0]):.2f}')
    print(f'  quiet head, mean raw score: alone {score[alone & quiet].mean():.3f}  '
          f'approach far {score[mp & quiet & (dw > FAR)].mean():.3f}  near {score[mp & quiet & (dw < NEAR)].mean():.3f}')

    print('== touch reactions: head turn rate 1 s before vs 1 s after collision_start ==')
    for kind, (a0, a1, p0, p1) in KIND.items():
        P0, P1 = to_pcap([p0, p1])
        ons = starts[(starts >= P0) & (starts < P1)]
        b = np.array([turn[(ti >= o - 1) & (ti < o)].mean() for o in ons])
        a = np.array([turn[(ti >= o) & (ti < o + 1)].mean() for o in ons])
        k = int((a > b).sum())
        print(f'  {kind:10s} before {b.mean():.2f}  after {a.mean():.2f}  rad/s   ({k}/{len(ons)} higher after, p {sign_p(k, len(ons)):.3f})')

    np.savez(os.path.join(sd.CACHE_DIR, 'z01_imu_windows.npz'), tc=tc, score=score, resid=resid, gated=gated,
             gate=gate, imu=imu, dist=dw)

    # figure: head motion, raw score with the IMU baseline, residual, distance
    fig, axs = plt.subplots(4, 1, figsize=(18, 11), sharex=True)
    axs[0].plot(tc, imu[:, 0], color='tab:orange', lw=0.7, label='head turn rate [rad/s]')
    axs[0].plot(tc, imu[:, 2], color='tab:purple', lw=0.7, label='head shaking [m/s²]')
    axs[0].set_ylabel('headset IMU')
    axs[0].legend(loc='upper left', fontsize=8)
    axs[1].plot(tc, score, color='k', lw=0.7, label='motion score (measured)')
    axs[1].plot(tc, score - resid, color='tab:blue', lw=0.7, label='expected from head motion alone')
    axs[1].set_ylabel('motion score')
    axs[1].legend(loc='upper left', fontsize=8)
    axs[2].plot(tc, boxcar(resid, 3), color='tab:red', lw=0.7)
    axs[2].axhline(0, color='grey', lw=0.5)
    axs[2].set_ylabel('measured - expected')
    axs[3].plot(td, dist, '.', ms=1.5, color='tab:green')
    axs[3].axhline(NEAR, color='r', lw=0.5)
    axs[3].set_ylabel('distance (video) [m]')
    axs[3].set_ylim(0, 2.5)
    axs[3].set_xlabel('time in the capture [s]')
    colours = {'still': 'tab:green', 'head turns': 'tab:orange', 'walking': 'tab:purple'}
    for kind, (a0, a1, p0, p1) in KIND.items():
        for x0, x1, lab in [(a0, a1, f'{kind}, alone'), (p0, p1, f'{kind}, approaches')]:
            X0, X1 = to_pcap([x0, x1])
            for ax in axs:
                ax.axvspan(X0, X1, color=colours[kind], alpha=0.06 if 'alone' in lab else 0.15, lw=0)
            axs[0].text((X0 + X1) / 2, axs[0].get_ylim()[1] * 0.95, lab, ha='center', va='top', fontsize=8)
    for s in starts:
        for ax in axs[1:]:
            ax.axvline(s, color='r', lw=0.4, alpha=0.5)
    axs[0].set_title('z01: head motion from the headset IMU, and what is left of the motion score after taking it out '
                     '(red lines: contact)')
    mf.save(fig, '10_z01_imu.png')
