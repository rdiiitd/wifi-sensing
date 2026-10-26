"""First look at z01_test, the first worn session with a headset IMU.

z01 was recorded with the Pi on the headset and the Quest IMU running. Its video parts are
(meta 'parts_video_s'): wearer alone 45-265 s (still 45-85, head turns 85-205, walking
205-265), then approaches 265-517 s (still ~265-345, head turns ~345-445, walking ~445-505).
Labels: the collector's auto labels (YOLO distance < 0.9 m), converted to data/z01_test_events.csv.

1. Our 4-link Doppler spectrograms of the whole capture (cached like the other captures).
2. Can the router's two antenna modes still be told apart while the Pi is worn?
   Per 0.5 s chunk: distance between the two mode templates / spread inside a mode.
3. Motion score per part, and at each approach versus the wearer-alone stretch with the
   same kind of wearer motion (the question: does an approach add anything on top of
   the wearer's own motion?).
Saves 09_z01_session.png.
"""
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

NAME = 'z01_test'
KIND = {'still': (45, 85, 265, 345), 'head turns': (85, 205, 345, 445), 'walking': (205, 265, 445, 505)}


def compute_full(name):
    idx = sd.index_pcap(name)
    dur = idx['ts'][-1] - idx['ts'][0]
    for mac, mode, _ in mf.LINKS:
        p = mf.full_path(name, mac, mode)
        if os.path.exists(p):
            continue
        t, vel, D, valid = sd.link_spectrogram(name, mac, mode, 0.0, dur, pad=0.0, hop=mf.HOP, noise_db=mf.RAW_DB)
        np.savez_compressed(p, t=t, vel=vel, D=D.astype(np.float16), valid=valid)
        print('saved', p, D.shape, flush=True)


def video_to_pcap(name):
    meta = json.load(open(os.path.join(sd.DATA_DIR, f'{name}_meta.json')))
    t0 = sd.index_pcap(name)['ts'][0]
    return lambda tv: meta['epoch_first_frame'] + np.asarray(tv, float) - t0


def mode_separation(name, n_chunks=40, chunk_s=0.5, seed=0):
    """Median over random 0.5 s chunks of (1 - cos between the 2 templates) / mean (1 - cos to own template)."""
    idx = sd.index_pcap(name)
    dur = idx['ts'][-1] - idx['ts'][0]
    rng = np.random.default_rng(seed)
    out = []
    for s in rng.uniform(5, dur - 5, n_chunks):
        t, csi, _ = sd.load_csi(name, sd.ROUTER, s, s + chunk_s)
        if len(t) < 50:
            continue
        a = np.abs(csi).astype(np.float64)
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        cen = sd._kmeans2(a)
        lab = np.argmax(a @ cen.T, 1)
        within = np.mean(1 - np.sum(a * cen[lab], 1))
        out.append((1 - cen[0] @ cen[1]) / max(within, 1e-9))
    return np.median(out), np.percentile(out, [25, 75])


def window_mean(tf, fd, a, b):
    m = (tf >= a) & (tf < b)
    return fd[m].mean() if m.any() else np.nan


if __name__ == '__main__':
    compute_full(NAME)
    to_pcap = video_to_pcap(NAME)

    print('== antenna-mode separation (router), median [IQR] over 40 random 0.5 s chunks ==')
    for name in ['m_empty', 'm01', 'x02', NAME]:
        med, iqr = mode_separation(name)
        print(f'  {name:9s} {med:7.1f}  [{iqr[0]:.1f}, {iqr[1]:.1f}]')

    tf, fd = mf.fused_dynamic(NAME, smooth_s=0.2)
    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])

    print('== motion score (share of Doppler power faster than 0.7 m/s), mean per part ==')
    for kind, (a0, a1, p0, p1) in KIND.items():
        print(f'  {kind:10s} wearer alone {window_mean(tf, fd, *to_pcap([a0, a1])):.3f}'
              f'   with approaches {window_mean(tf, fd, *to_pcap([p0, p1])):.3f}')

    print('== at each approach: 1 s after collision_start vs 1 s windows of the wearer alone, same motion ==')
    rng = np.random.default_rng(1)
    rows = []
    for kind, (a0, a1, p0, p1) in KIND.items():
        A0, A1, P0, P1 = to_pcap([a0, a1, p0, p1])
        ons = starts[(starts >= P0) & (starts < P1)]
        ev_v = np.array([window_mean(tf, fd, o, o + 1) for o in ons])
        alone = np.array([window_mean(tf, fd, s, s + 1) for s in np.arange(A0, A1 - 1, 0.5)])
        pct = np.array([np.mean(alone < v) for v in ev_v])
        null = np.array([rng.choice(alone, len(ev_v)).mean() for _ in range(5000)])
        p = (np.sum(null >= ev_v.mean()) + 1) / 5001
        rows.append((kind, ons, ev_v, alone))
        print(f'  {kind:10s} approaches {len(ons):2d}  mean {ev_v.mean():.3f} vs alone {alone.mean():.3f}'
              f'  ({int((pct > 0.5).sum())}/{len(ons)} above the alone median, p {p:.3f})')

    # figure: router mode A spectrogram with parts + approaches, motion score below
    t, vel, D, valid = mf.load_full(NAME, sd.ROUTER, 0)
    fig, axs = plt.subplots(2, 1, figsize=(18, 8), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    step = 5
    im = mf.show(axs[0], t[::step], vel, D[::step], 'z01: router, antenna mode A. Shaded: what the wearer is doing; '
                 'red lines: an approach reaches arm\'s length (< 0.9 m)')
    colours = {'still': 'tab:green', 'head turns': 'tab:orange', 'walking': 'tab:purple'}
    for kind, (a0, a1, p0, p1) in KIND.items():
        for x0, x1, lab in [(a0, a1, f'{kind}, alone'), (p0, p1, f'{kind}, approaches')]:
            X0, X1 = to_pcap([x0, x1])
            axs[1].axvspan(X0, X1, color=colours[kind], alpha=0.12 if 'alone' in lab else 0.25, lw=0)
            axs[1].text((X0 + X1) / 2, 0.95, lab, ha='center', va='top', fontsize=8)
    for s in starts:
        axs[0].axvline(s, color='r', lw=0.7)
        axs[1].axvline(s, color='r', lw=0.5, alpha=0.6)
    axs[1].plot(tf, fd, color='k', lw=0.6)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel('motion score')
    axs[1].set_xlabel('time in the capture [s]')
    fig.colorbar(im, ax=axs, label=mf.POWER, shrink=0.6, anchor=(0, 1))
    mf.save(fig, '09_z01_session.png')
