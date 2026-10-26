"""How much does antenna-mode sorting go wrong while the Pi is worn (z01)?

1. Packet header: does the Nexmon core/stream field tell the two router antennas apart? (no: always 0)
2. Does the router switch antennas in a pattern? Run lengths of the sorted modes in the static m_empty
   (coin flip per packet -> runs halve each step, mean 2) and their link to the sequence number.
3. Sorting statistics per z01 part (only rules out a sorter stuck on one mode: random labels also give 50/50).
4. The real test: wrong sorts smear power flat over all speeds. Power above 4 m/s (no head moves that fast)
   per part, against the same still stretch with the two modes mixed on purpose.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
import z01_imu as zi
from z01_check import KIND, NAME, video_to_pcap


def header(name, t0, t1):
    idx = sd.index_pcap(name)
    rel = idx['ts'] - idx['ts'][0]
    sel = np.flatnonzero((idx['macs'] == sd.ROUTER) & ((idx['lens'] == 1084) | (idx['lens'] == 1088))
                         & (rel >= t0) & (rel < t1))
    mm = np.memmap(os.path.join(sd.DATA_DIR, f'{name}.pcap'), dtype=np.uint8, mode='r')
    h = mm[(idx['offs'][sel] + 42)[:, None] + np.arange(18)].astype(int)
    return h[:, 10] | (h[:, 11] << 8), h[:, 12] | (h[:, 13] << 8)


def run_stats(lab):
    lab = lab.astype(float)
    ch = np.flatnonzero(np.diff(lab) != 0)
    runs = np.diff(np.r_[0, ch + 1, len(lab)])
    return lab.mean(), runs, np.mean(np.diff(lab) != 0)


if __name__ == '__main__':
    to_pcap = video_to_pcap(NAME)

    print('== 1. Nexmon core/stream field (router packets, 100-110 s) ==')
    for name in ['m_empty', NAME]:
        _, core_ss = header(name, 100, 110)
        print(f'  {name:9s} values {sorted(set(core_ss.tolist()))}')

    print('== 2. pattern of antenna switching, m_empty 100-160 s ==')
    seq, _ = header('m_empty', 100, 160)
    t, csi, _ = sd.load_csi('m_empty', sd.ROUTER, 100, 160)
    lab, _ = sd.track_modes(t, csi)
    a, runs, sw = run_stats(lab)
    counts = np.bincount(np.minimum(runs, 8))[1:]
    print(f'  run lengths 1..8+: {counts.tolist()}  (coin flip: each about half the one before)')
    print(f'  mode A share by sequence number mod 2: {[round(lab[(seq >> 4) % 2 == r].mean(), 2) for r in (0, 1)]}')

    print('== 3. sorting statistics per z01 part (correct sorting: A share 0.50, mean run 2.0) ==')
    for kind, (a0, a1, p0, p1) in KIND.items():
        for part, (x0, x1) in [('alone', (a0, a1)), ('approach', (p0, p1))]:
            t, csi, _ = sd.load_csi(NAME, sd.ROUTER, *to_pcap([x0, x1]))
            a, runs, sw = run_stats(sd.track_modes(t, csi)[0])
            print(f'  {kind:10s} {part:8s} A share {a:.2f}  mean run {runs.mean():.2f}  switch rate {sw:.2f}')

    print('== 4. share of Doppler power at 0.7-2.5 m/s (real motion) | above 4 m/s (impossible for a head) ==')
    tf, vel, P = zi.frame_profiles()
    mid, hi = (np.abs(vel) > 0.7) & (np.abs(vel) <= 2.5), np.abs(vel) > 4
    for kind, (a0, a1, p0, p1) in KIND.items():
        for part, (x0, x1) in [('alone', (a0, a1)), ('approach', (p0, p1))]:
            X0, X1 = to_pcap([x0, x1])
            m = (tf >= X0) & (tf < X1)
            print(f'  {kind:10s} {part:8s} {P[m][:, mid].sum(1).mean():.4f} | {P[m][:, hi].sum(1).mean():.4f}')
    X0, X1 = to_pcap([50, 80])
    for mode, lab in [(0, 'sorted (mode A)'), (None, 'modes mixed on purpose')]:
        t, v, D, _ = sd.link_spectrogram(NAME, sd.ROUTER, mode, X0, X1, hop=4, noise_db=-60)
        Q = D / D.sum(1, keepdims=True)
        print(f'  still alone, router {lab:22s} {Q[:, (np.abs(v) > 0.7) & (np.abs(v) <= 2.5)].sum(1).mean():.4f} | '
              f'{Q[:, np.abs(v) > 4].sum(1).mean():.4f}')
