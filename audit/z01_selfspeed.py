"""z01: speed gate from the headset IMU (option 1).

Physics: if only the wearer moves, every still reflection reaches the moving antenna with a Doppler of
v_antenna . (direction of that path). The cleaned CSI is referenced to one path (SHARP; CornerSense does the
same), and that path moves with the antenna too, so what we see is v (cos a_path - cos a_reference):
|Doppler speed| <= 2 |v_antenna|. Power faster than that must come from something else that moves.

1. Gravity direction from the IMU (gyro + accel complementary filter), vertical and horizontal accel.
2. Antenna speed = walking speed (steps: Weinberg step length K * (a_max - a_min)^(1/4) / step time)
                   + head rotation (|turn rate| * R_ANT, R_ANT = antenna distance from the neck axis, assumed).
   K is fitted to the video wearer speed (YOLO positions, 1 s smoothing) in the walking-approach part.
3. Checks in the wearer-alone parts: does the Doppler edge follow the IMU speed, and how much power is
   left above the gate when nobody else is there?
4. Gated score = share of power faster than max(0.7, SELF_FACTOR * v_antenna + MARGIN); same tests as z01_imu.py.
Saves 12_z01_speed_gate.png.
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
import make_figures as mf
import z01_imu as zi
from z01_check import KIND, NAME, video_to_pcap

R_ANT = 0.18   # m: the Pi sits just above the headset front box (collector); ~15-20 cm in front of the turning axis, estimated
SELF_FACTOR = 2.0  # reference-path phase: own motion appears at up to 2x the antenna speed (see top)
MARGIN = 2 * sd.FS / sd.N_WIN * sd.LAMBDA  # 0.64 m/s: half-width of the Hann main lobe, a still path smears this far
G = 9.81


def load_imu_raw():
    meta = json.load(open(os.path.join(sd.DATA_DIR, f'{NAME}_meta.json')))
    x = np.loadtxt(os.path.join(zi.IMU_DIR, 'imu.csv'), delimiter=',', skiprows=1, usecols=(0, 4, 5, 6, 7, 8, 9))
    clk = np.loadtxt(os.path.join(zi.IMU_DIR, 'clock.csv'), delimiter=',', skiprows=1)
    slope, icpt = np.polyfit(clk[:, 1] * 1e-6, clk[:, 0], 1)
    t = slope * x[:, 0] * 1e-6 + icpt + meta['imu_offset_s'] - sd.index_pcap(NAME)['ts'][0]
    return t, x[:, 1:4], x[:, 4:7]


def gravity(t, acc, gyr, tau=1.0):
    """Unit 'up' vector in the sensor frame: rotate with the gyro, pull slowly towards the accel."""
    dt = np.r_[np.median(np.diff(t)), np.diff(t)]
    up = np.empty_like(acc)
    g = acc[0] / np.linalg.norm(acc[0])
    for i in range(len(t)):
        g = g - np.cross(gyr[i], g) * dt[i]
        a = acc[i] / np.linalg.norm(acc[i])
        g = g + (a - g) * (dt[i] / tau)
        g /= np.linalg.norm(g)
        up[i] = g
    return up


def walking_speed(t, a_vert, k):
    """Per-sample walking speed from steps; 0 when there is no step within 1.2 s."""
    fs = 1 / np.median(np.diff(t))
    # band-pass around the step rate; a fixed height missed most of the slow, soft steps in z01
    av = sosfiltfilt(butter(2, [0.7, 3.5], btype='band', fs=fs, output='sos'), a_vert)
    pk, _ = find_peaks(av, prominence=0.5, distance=int(0.3 * fs))
    v = np.zeros(len(t))
    for i0, i1 in zip(pk[:-1], pk[1:]):
        dur = t[i1] - t[i0]
        if dur > 1.2:
            continue
        seg = av[i0:i1 + 1]
        v[i0:i1] = k * (seg.max() - seg.min()) ** 0.25 / dur
    return v, pk


def video_speed(to_pcap, smooth_s=1.0):
    r = list(csv.DictReader(open(zi.DIST_CSV)))
    tv = np.array([float(x['time_sec']) for x in r])
    f = lambda key: np.array([float(x[key]) if x[key] else np.nan for x in r])
    wx, wz = f('wearer_x'), f('wearer_z')
    ok = ~np.isnan(wx)
    grid = np.arange(tv[0], tv[-1], 0.1)
    n = int(smooth_s / 0.1)
    px = np.convolve(np.interp(grid, tv[ok], wx[ok]), np.ones(n) / n, mode='same')
    pz = np.convolve(np.interp(grid, tv[ok], wz[ok]), np.ones(n) / n, mode='same')
    return to_pcap(grid[1:]), np.hypot(np.diff(px), np.diff(pz)) / 0.1


def edge_speed(vel, P, keep=0.98):
    """Speed below which `keep` of the power lies (both directions folded)."""
    order = np.argsort(np.abs(vel))
    c = np.cumsum(P[:, order], 1)
    return np.abs(vel)[order][np.argmax(c >= keep * c[:, -1:], 1)]


def window_means(tc, tf, *xs):
    out = []
    for x in xs:
        out.append(np.array([x[(tf >= c - zi.WIN / 2) & (tf < c + zi.WIN / 2)].mean() for c in tc]))
    return out


def antenna_speed(ti, acc, gyr, to_pcap):
    """Antenna speed per IMU sample: walking (K fitted to the video speed in the walking-approach part)
    + head rotation * R_ANT, averaged over the 180 ms Doppler window."""
    up = gravity(ti, acc, gyr)
    a_vert = np.sum(acc * up, 1) - G
    v1, pk = walking_speed(ti, a_vert, 1.0)
    tvid, svid = video_speed(to_pcap)
    W0, W1 = to_pcap([445, 505])
    m = (tvid >= W0) & (tvid < W1)
    x = np.interp(tvid[m], ti, np.convolve(v1, np.ones(800) / 800, mode='same'))  # 1 s smoothing like the video
    K = float(np.sum(x * svid[m]) / np.sum(x * x))
    r_vid = np.corrcoef(K * x, svid[m])[0, 1]
    n = int(0.18 * 800)
    v_ant = np.convolve(K * v1 + R_ANT * np.linalg.norm(gyr, axis=1), np.ones(n) / n, mode='same')
    return v_ant, pk, K, r_vid, (np.mean(K * x), np.mean(svid[m]))


if __name__ == '__main__':
    to_pcap = video_to_pcap(NAME)
    ti, acc, gyr = load_imu_raw()
    v_ant, pk, K, r_vid, means = antenna_speed(ti, acc, gyr, to_pcap)
    turn = np.linalg.norm(gyr, axis=1)
    print(f'steps found {len(pk)}; K {K:.2f} (fitted to the video, walking-approach part); '
          f'IMU vs video speed r {r_vid:.2f}, mean {means[0]:.2f} vs {means[1]:.2f} m/s')

    tf, vel, P = zi.frame_profiles()
    va = np.interp(tf, ti, v_ant, left=np.nan, right=np.nan)
    gate = np.maximum(0.7, SELF_FACTOR * va + MARGIN)
    fast = np.abs(vel)[None, :] > gate[:, None]
    gated = np.sum(P * fast, 1)
    raw = P[:, np.abs(vel) > 0.7].sum(1)
    edge = edge_speed(vel, P)

    print('== wearer alone: IMU antenna speed, Doppler edge (98 % power), share of power above the gate ==')
    for kind, (a0, a1, p0, p1) in KIND.items():
        for part, (x0, x1) in [('alone', (a0, a1)), ('approach', (p0, p1))]:
            X0, X1 = to_pcap([x0, x1])
            mm = (tf >= X0) & (tf < X1) & ~np.isnan(va)
            print(f'  {kind:10s} {part:8s} antenna {va[mm].mean():.2f} m/s  edge {edge[mm].mean():.2f} m/s  '
                  f'raw {raw[mm].mean():.4f}  gated {gated[mm].mean():.4f}')
    A0, A1 = to_pcap([45, 265])
    ma = (tf >= A0) & (tf < A1) & ~np.isnan(va)
    print(f'  wearer alone, all: corr(antenna speed, Doppler edge) r {np.corrcoef(va[ma], edge[ma])[0, 1]:.2f};  '
          f'edge above the gate in {np.mean(edge[ma] > gate[ma]) * 100:.1f} % of frames')

    # window-level tests, same windows and labels as z01_imu.py
    tc = np.arange(max(tf[0], ti[0]) + zi.WIN / 2, min(tf[-1], ti[-1]) - zi.WIN / 2, zi.STEP)
    gw, rw, vw, tw = window_means(tc, tf, gated, raw, va, np.interp(tf, ti, turn))
    td, dist = zi.load_distance(to_pcap)
    dw = zi.at(tc, td, dist)

    print(f'== approach part, near (< {zi.NEAR} m) vs far (> {zi.FAR} m) windows: AUC ==')
    P0, P1 = to_pcap([265, 517])
    mp = (tc >= P0) & (tc < P1)
    quiet = tw < 0.15
    groups = {kind: (tc >= to_pcap(p0)) & (tc < to_pcap(p1)) for kind, (a0, a1, p0, p1) in KIND.items()}
    groups.update({'all approaches': mp, 'head quiet': mp & quiet, 'head moving': mp & ~quiet})
    for lab, g in groups.items():
        near, far = g & (dw < zi.NEAR), g & (dw > zi.FAR)
        print(f'  {lab:14s} near {near.sum():3d} far {far.sum():3d}   raw {zi.auc(rw[near], rw[far]):.2f}   '
              f'gated {zi.auc(gw[near], gw[far]):.2f}')

    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])
    ends = np.array([t for t, e in ev if e == 'collision_end'])
    print('== 1 s before contact vs 1 s at the farthest point between contacts, paired ==')
    for kind, (a0, a1, p0, p1) in KIND.items():
        Q0, Q1 = to_pcap([p0, p1])
        c, f = [], []
        for o in starts[(starts >= Q0) & (starts < Q1)]:
            prev = ends[ends < o]
            if not len(prev) or prev[-1] < Q0:
                continue
            mm = (td > prev[-1]) & (td < o) & ~np.isnan(dist)
            if not mm.any():
                continue
            ft = td[mm][np.argmax(dist[mm])]
            wc, wf = (tf >= o - 1) & (tf < o), (tf >= ft - 0.5) & (tf < ft + 0.5)
            c.append([raw[wc].mean(), gated[wc].mean()])
            f.append([raw[wf].mean(), gated[wf].mean()])
        c, f = np.array(c), np.array(f)
        wins = (c > f).sum(0)
        print(f'  {kind:10s} raw {wins[0]}/{len(c)} (p {zi.sign_p(int(wins[0]), len(c)):.3f})   '
              f'gated {wins[1]}/{len(c)} (p {zi.sign_p(int(wins[1]), len(c)):.3f})')

    # figure: spectrogram (router A) with the gate drawn on it, then raw vs gated score
    t, v, D, _ = mf.load_full(NAME, sd.ROUTER, 0)
    fig, axs = plt.subplots(3, 1, figsize=(18, 10), sharex=True, gridspec_kw=dict(height_ratios=[2, 1, 1]))
    step = 5
    im = mf.show(axs[0], t[::step], v, D[::step], 'z01, router antenna mode A. White: the speed gate '
                 f'(2 x wearer antenna speed from the IMU + {MARGIN:.2f} m/s); power outside it cannot come from the wearer alone')
    g = np.interp(t, tf, gate)
    axs[0].plot(t, g, color='w', lw=0.6)
    axs[0].plot(t, -g, color='w', lw=0.6)
    axs[1].plot(tf, np.interp(tf, tf, va), color='tab:purple', lw=0.5, label='antenna speed from IMU [m/s]')
    axs[1].plot(tf, edge, color='k', lw=0.4, alpha=0.6, label='Doppler edge (98 % of power) [m/s]')
    axs[1].set_ylim(0, 3)
    axs[1].legend(loc='upper left', fontsize=8)
    k = 10
    axs[2].plot(tf, np.convolve(raw, np.ones(k) / k, mode='same'), color='grey', lw=0.6, label='motion score (> 0.7 m/s)')
    axs[2].plot(tf, np.convolve(gated, np.ones(k) / k, mode='same'), color='tab:red', lw=0.7,
                label='gated score (faster than the wearer)')
    axs[2].set_ylim(0, 0.25)
    axs[2].legend(loc='upper left', fontsize=8)
    axs[2].set_xlabel('time in the capture [s]')
    colours = {'still': 'tab:green', 'head turns': 'tab:orange', 'walking': 'tab:purple'}
    for kind, (a0, a1, p0, p1) in KIND.items():
        for x0, x1, alpha in [(a0, a1, 0.06), (p0, p1, 0.15)]:
            X0, X1 = to_pcap([x0, x1])
            for ax in axs[1:]:
                ax.axvspan(X0, X1, color=colours[kind], alpha=alpha, lw=0)
    for s in starts:
        for ax in axs[1:]:
            ax.axvline(s, color='r', lw=0.4, alpha=0.5)
    fig.colorbar(im, ax=axs, label=mf.POWER, shrink=0.5, anchor=(0, 1))
    mf.save(fig, '12_z01_speed_gate.png')
