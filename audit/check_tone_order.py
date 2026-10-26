"""Ascending vs mirrored tone order: real multipath arrives *after* the first path, so the
power-delay profile (aligned on its leading tap) must have its tail at positive delay.
Mirrored order flips the delay axis."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd

NFFT = 1024
dt_ns = 1e9 / (NFFT * sd.DELTA_F)
for name, t0, t1 in [('m_empty', 70, 150), ('m01', 50, 150), ('x02', 10, 40)]:
    for mac in [sd.ROUTER, sd.LAPTOP]:
        t, c, _ = sd.load_csi(name, mac, t0, t1)
        lab, _ = sd.track_modes(t, c)
        for m in (0, 1):
            x = c[lab == m][::5]
            spec = np.zeros((len(x), NFFT), complex)
            spec[:, sd.K20 % NFFT] = x / np.abs(x).mean(1, keepdims=True)
            p = np.abs(np.fft.ifft(spec, axis=1)) ** 2
            # align each packet on its strongest tap, average
            sh = np.array([np.roll(r, NFFT // 2 - np.argmax(r)) for r in p]).mean(0)
            tau = (np.arange(NFFT) - NFFT // 2) * dt_ns
            pk = sh.max()
            prof = 10 * np.log10(sh / pk)
            sec = np.argmax(np.where(np.abs(tau) > 120, sh, 0))
            pos = sh[(tau > 60) & (tau < 160)].sum(); neg = sh[(tau < -60) & (tau > -160)].sum()
            print(f'{name:7s} {mac[:2]} mode{m}: 2nd tap {tau[sec]:+6.0f} ns ({prof[sec]:5.1f} dB)   '
                  f'energy 60-160 ns after/before main tap {10*np.log10(pos/neg):+5.1f} dB')
