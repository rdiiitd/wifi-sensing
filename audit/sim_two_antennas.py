"""Does CornerSense's two-antenna processing keep a nearby person that SHARP's one-antenna reference step loses?

Same scene as sim_approach.py (static Rx, Tx 5 m away, 3 wall paths, person B walking at the Rx from 4 m
to 0.3 m at 1 m/s), but the Rx now has two antennas 2.5 cm apart (CornerSense's Intel 5300 spacing).
Per-packet clock errors (random phase and detection delay) are common to both antennas, as on real hardware.

Compared, all at 20 MHz with the same Doppler step:
  sharp_1ant : our SHARP pipeline on antenna 0 (Lasso + reference to the strongest copy)
  power_1ant : signal power |H|^2 on antenna 0 after per-packet gain normalisation (what AGC leaves)
  conj_2ant  : CornerSense / IndoTrack style H0 * conj(H1); the common clock error cancels, nothing is subtracted
Output: how strong B appears at B's true speed, relative to the strongest part of the spectrum, vs distance.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd

TX, RX = np.array([0.0, 0.0]), np.array([5.0, 0.0])
ANT = [np.array([0.0, -0.0125]), np.array([0.0, 0.0125])]   # 2.5 cm apart, across the Tx-Rx line
WALLS = [(0.5, 25e-9, 0.4, 35.0), (0.35, 48e-9, 2.1, -60.0), (0.25, 90e-9, 1.0, 120.0)]  # amp, excess delay, phase, angle


def synth(angle_deg=0.0, v=1.0, d0=4.0, d1=0.3, pps=500, sigma=1.0, seed=0, room=None):
    """room: seed for random wall phases (None keeps WALLS as listed)."""
    rng = np.random.default_rng(seed)
    walls = WALLS if room is None else [(a, dt, ph, ang) for (a, dt, _, ang), ph in
                                        zip(WALLS, np.random.default_rng(100 + room).uniform(0, 2 * np.pi, len(WALLS)))]
    dur = (d0 - d1) / v
    t = np.sort(rng.uniform(0, dur, int(dur * pps)))
    u = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    pos = RX[None] + u[None] * (d0 - v * t)[:, None]
    f = sd.K20 * sd.DELTA_F
    d_los = np.linalg.norm(RX - TX)
    cfo = np.exp(1j * rng.uniform(0, 2 * np.pi, len(t)))[:, None]
    pdd = np.exp(-2j * np.pi * f[None] * rng.uniform(-100e-9, 400e-9, len(t))[:, None])
    H = []
    for off in ANT:
        rx = RX + off
        L = np.linalg.norm(rx - TX)
        h = np.exp(-2j * np.pi * (sd.FC + f) * L / sd.C)[None].repeat(len(t), 0)
        for a, dt, ph, ang in walls:
            w = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
            Lw = d_los + dt * sd.C + w @ off            # plane wave from the wall's direction
            h = h + a * np.exp(1j * ph) * np.exp(-2j * np.pi * (sd.FC + f) * Lw / sd.C)[None]
        da = np.linalg.norm(pos - TX, axis=1)
        db = np.linalg.norm(pos - rx, axis=1)
        amp = np.sqrt(sigma / (4 * np.pi)) * d_los / (da * db)
        h = h + amp[:, None] * np.exp(-2j * np.pi * (sd.FC + f[None]) * ((da + db) / sd.C)[:, None])
        h = h * cfo * pdd
        h = h + 0.01 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
        H.append((1000 * h).astype(np.complex64))
    dist = np.linalg.norm(pos - RX, axis=1)
    da = np.linalg.norm(pos - TX, axis=1)
    v_true = -np.gradient(da + dist, t)
    truth_db = 20 * np.log10(np.sqrt(sigma / (4 * np.pi)) * d_los / (da * dist))
    return t, H, dist, v_true, truth_db


def spectrogram(t, x):
    tu, xu, _ = sd.resample_uniform(t, x, t[0] + 0.1, t[-1] - 0.1)
    xu = xu - xu.mean(0, keepdims=True) * 0  # keep the static line, as in our figures
    ts, vel, D = sd.doppler_spectrogram(xu, noise_db=-60)
    return ts + tu[0], vel, D


def level_at_truth(ts, vel, D, t, dist, v_true):
    d = np.interp(ts, t, dist)
    vt = np.interp(ts, t, v_true)
    lvl = np.array([10 * np.log10(max(D[i, np.argmin(np.abs(vel - vt[i]))],
                                      D[i, np.argmin(np.abs(vel + vt[i]))]) + 1e-12) for i in range(len(ts))])
    return d, lvl


def run(angle_deg, room=None):
    t, H, dist, v_true, truth_db = synth(angle_deg, room=room)
    out = {}
    h_sharp, _ = sd.sanitize_sharp(H[0], t)
    out['sharp_1ant'] = h_sharp
    a = np.abs(H[0]) / np.mean(np.abs(H[0]), axis=1, keepdims=True)      # per-packet gain normalisation (AGC)
    out['power_1ant'] = (a ** 2).astype(np.complex64)
    g = np.mean(np.abs(H[0]) * np.abs(H[1]), axis=1, keepdims=True)
    out['conj_2ant'] = (H[0] * np.conj(H[1]) / g).astype(np.complex64)
    res = {}
    for k, x in out.items():
        ts, vel, D = spectrogram(t, x)
        res[k] = (ts, vel, D) + level_at_truth(ts, vel, D, t, dist, v_true)
    return res, t, dist, truth_db


if __name__ == '__main__':
    names = {'sharp_1ant': 'SHARP, 1 antenna (ours now)', 'power_1ant': 'signal power, 1 antenna',
             'conj_2ant': 'CornerSense-style, 2 antennas'}
    fig, axs = plt.subplots(2, 4, figsize=(22, 8), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    for row, ang in enumerate([0.0, 90.0]):
        res, t, dist, truth_db = run(ang)
        for j, k in enumerate(names):
            ts, vel, D, d, lvl = res[k]
            ax = axs[row, j]
            keep = np.abs(vel) <= 3
            ax.pcolormesh(d, vel[keep], 10 * np.log10(np.maximum(D[:, keep], 1e-3)).T, vmin=-30, vmax=0, shading='auto')
            ax.set_xlim(d.max(), d.min())
            ax.set_title(f'{names[k]}\nB from {"behind" if ang == 0 else "the side"}', fontsize=10)
            ax.set_xlabel('distance of B from the receiver [m]')
            if j == 0:
                ax.set_ylabel('speed towards / away [m/s]')
        ax = axs[row, 3]
        ax.plot(dist, truth_db, 'k--', label='how strong B really is')
        for k in names:
            ts, vel, D, d, lvl = res[k]
            ax.plot(d, lvl, label=names[k])
        ax.set_xlim(dist.max(), dist.min()); ax.set_ylim(-45, 0)
        ax.set_xlabel('distance of B from the receiver [m]'); ax.set_ylabel('how strong B looks [dB]')
        ax.legend(fontsize=8)
        for dd in [3.0, 2.0, 1.0, 0.5]:
            row_txt = [f'{ang:3.0f} deg d={dd:.1f} m truth {np.interp(-dd, -dist, truth_db):6.1f}']
            for k in names:
                ts, vel, D, d, lvl = res[k]
                i = np.argmin(np.abs(d - dd))
                row_txt.append(f'{k} {lvl[i]:6.1f}')
            print('  '.join(row_txt))
    # robustness: 10 random rooms (wall phases), both approach directions
    print('median [min, max] over 10 rooms x 2 directions, B level minus truth [dB]:')
    loss = {k: {dd: [] for dd in (3.0, 2.0, 1.0, 0.5)} for k in names}
    for room in range(10):
        for ang in (0.0, 90.0):
            res, t, dist, truth_db = run(ang, room)
            for k in names:
                ts, vel, D, d, lvl = res[k]
                for dd in loss[k]:
                    i = np.argmin(np.abs(d - dd))
                    loss[k][dd].append(lvl[i] - np.interp(-dd, -dist, truth_db))
    for k in names:
        print(f'  {k:11s}', '  '.join(f'{dd:.1f} m: {np.median(v):5.1f} [{min(v):5.1f}, {max(v):5.1f}]'
                                       for dd, v in loss[k].items()))
    fig.tight_layout()
    os.makedirs(sd.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(sd.FIG_DIR, '08_one_vs_two_antennas.png'), dpi=100, bbox_inches='tight')
