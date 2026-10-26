"""Can SHARP-style processing at 20 MHz see a person walking up to the receiver?

Static Rx (best case for SHARP), Tx 5 m away, 3 static wall paths. Person B walks
straight at the Rx from 4 m to 0.3 m at 1 m/s, coming from the side opposite the Tx
(largest excess delay = most favourable). The CSI is built path by path (bistatic radar
amplitude, sigma = 1 m^2), with random per-packet CFO and detection delay as in the real
captures, and passed through the same pipeline as the measurements.
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
WALLS = [(0.5, 25e-9, 0.4), (0.35, 48e-9, 2.1), (0.25, 90e-9, 1.0)]  # (amp rel. LOS, excess delay, phase)


def synth_approach(v=1.0, d0=4.0, d1=0.3, angle_deg=0.0, pps=500, sigma=1.0, seed=0):
    """angle 0: B comes from behind the Rx (away from Tx); 90: from the side."""
    rng = np.random.default_rng(seed)
    dur = (d0 - d1) / v
    t = np.sort(rng.uniform(0, dur, int(dur * pps)))
    u = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    pos = RX[None] + u[None] * (d0 - v * t)[:, None]
    f = sd.K20 * sd.DELTA_F
    d_los = np.linalg.norm(RX - TX)
    tau_los = d_los / sd.C
    h = np.exp(-2j * np.pi * f * tau_los)[None].repeat(len(t), 0)
    for a, dt, ph in WALLS:
        h += a * np.exp(1j * ph) * np.exp(-2j * np.pi * f * (tau_los + dt))[None]
    da, db = np.linalg.norm(pos - TX, axis=1), np.linalg.norm(pos - RX, axis=1)
    amp = np.sqrt(sigma / (4 * np.pi)) * d_los / (da * db)
    tau_b = (da + db) / sd.C
    h += amp[:, None] * np.exp(-2j * np.pi * (sd.FC + f[None]) * tau_b[:, None])
    h *= np.exp(1j * rng.uniform(0, 2 * np.pi, len(t)))[:, None] * np.exp(-2j * np.pi * f[None] * rng.uniform(-100e-9, 400e-9, len(t))[:, None])
    h += 0.01 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    excess = (da + db - d_los) / sd.C
    v_true = -np.gradient(da + db, t)          # SHARP sign: approaching path -> positive? report |v|
    return t, (1000 * h).astype(np.complex64), db, excess, amp, v_true


def run(angle_deg):
    t, csi, db, excess, amp, v_true = synth_approach(angle_deg=angle_deg)
    H, _ = sd.sanitize_sharp(csi, t)
    tu, hu, _ = sd.resample_uniform(t, H, t[0] + 0.1, t[-1] - 0.1)
    ts, vel, D = sd.doppler_spectrogram(hu, noise_db=-60)
    ts = ts + tu[0]
    dist = np.interp(ts, t, db)
    vt = np.interp(ts, t, v_true)
    # level at the true path-length rate (either sign) relative to the zero-velocity peak
    lvl = np.array([10 * np.log10(max(D[i, np.argmin(np.abs(vel - vt[i]))], D[i, np.argmin(np.abs(vel + vt[i]))]) + 1e-12) for i in range(len(ts))])
    rel_amp = 20 * np.log10(np.interp(ts, t, amp))
    return ts, vel, D, dist, vt, lvl, rel_amp, np.interp(ts, t, excess)


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    for j, ang in enumerate([0.0, 90.0]):
        ts, vel, D, dist, vt, lvl, rel_amp, exc = run(ang)
        ax = axs[0, j]
        im = ax.pcolormesh(dist, vel, 10 * np.log10(D.T + 1e-12), vmin=-30, vmax=0, shading='auto')
        ax.plot(dist, vt, 'w--', lw=0.8, label='where B should appear')
        ax.plot(dist, -vt, 'w:', lw=0.8)
        ax.set_xlim(dist.max(), dist.min()); ax.set_ylim(-3, 3)
        ax.set_xlabel('distance of person B from the receiver [m]'); ax.set_ylabel('speed towards / away [m/s]')
        ax.set_title(f'Simulated: B walks up to a still receiver from {"behind" if ang == 0 else "the side"}'); ax.legend(fontsize=8)
        ax = axs[1, j]
        ax.plot(dist, lvl, label='how strong B looks after SHARP [dB]')
        ax.plot(dist, rel_amp, label='how strong B really is [dB]')
        ax2 = ax.twinx(); ax2.plot(dist, exc * 1e9, 'k:', label='delay behind the direct copy [ns]'); ax2.set_ylabel('B delay behind direct copy [ns] (dotted)')
        ax.axhline(-12, color='r', lw=0.6, ls='--')
        ax.set_xlim(dist.max(), dist.min()); ax.set_xlabel('distance of person B from the receiver [m]'); ax.legend(fontsize=8, loc='lower left')
        for d in [3.0, 2.0, 1.0, 0.5]:
            i = np.argmin(np.abs(dist - d))
            print(f'angle {ang:3.0f}  d={d:.1f} m  excess {exc[i]*1e9:5.1f} ns  truth {rel_amp[i]:6.1f} dB  SHARP at true v {lvl[i]:6.1f} dB  |v| {abs(vt[i]):.2f}')
    fig.colorbar(im, ax=axs[0], label='strength vs. strongest [dB]')
    os.makedirs(sd.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(sd.FIG_DIR, '07_why_close_person_is_hidden.png'), dpi=100, bbox_inches='tight')
