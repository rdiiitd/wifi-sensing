import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd


STATIC_A = [(1.0, 30e-9, 0.3), (0.5, 90e-9, 1.1), (0.3, 160e-9, 2.0)]
STATIC_B = [(1.0, 20e-9, 2.5), (0.7, 70e-9, 0.2), (0.4, 210e-9, 1.4)]


def synth(v_dyn=1.2, a_dyn=0.4, dur=3.0, pps=500, seed=0, static=STATIC_A, dyn_delay=250e-9):
    """20 MHz channel: 3 static paths + 1 moving path, random per-packet CFO phase and detection delay."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, dur, int(dur * pps)))
    f = sd.K20 * sd.DELTA_F
    h = np.zeros((len(t), len(f)), complex)
    for a, tau, ph in static:
        h += a * np.exp(1j * ph) * np.exp(-2j * np.pi * f * tau)
    path_rate = v_dyn  # d(path length)/dt
    tau_d = dyn_delay + path_rate * t / sd.C
    h += a_dyn * np.exp(-2j * np.pi * (sd.FC + f[None, :]) * tau_d[:, None])
    cfo = np.exp(1j * rng.uniform(0, 2 * np.pi, len(t)))[:, None]
    pdd = rng.uniform(-100e-9, 400e-9, len(t))[:, None]
    h = h * cfo * np.exp(-2j * np.pi * f[None, :] * pdd)
    h += 0.02 * (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape))
    return t, (1000 * h).astype(np.complex64)


def profile_db(t, csi, method):
    h = sd.sanitize(csi, method)
    _, hu, _ = sd.resample_uniform(t, h, 0.2, 2.8)
    _, vel, D = sd.doppler_spectrogram(hu, noise_db=-40)
    return vel, 10 * np.log10(D.mean(0))


def level_at(vel, p, v):
    return p[np.argmin(np.abs(vel - v))]


def test_sharp_recovers_resolvable_moving_path():
    # moving path 220 ns from the strongest static path: resolvable at 20 MHz
    t, csi = synth(v_dyn=1.2, dyn_delay=250e-9)
    vel, p = profile_db(t, csi, 'sharp')
    p_moving = p.copy()
    p_moving[np.abs(vel) < 0.7] = -99
    assert abs(abs(vel[np.argmax(p_moving)]) - 1.2) < 0.15
    assert level_at(vel, p, -1.16) > -11.0          # true level ~ -9.2 dB
    assert level_at(vel, p, +1.16) < level_at(vel, p, -1.16) - 10   # mirror image suppressed


def test_unresolvable_moving_path_is_attenuated_and_mirrored():
    # documents the 20 MHz limitation: path 30 ns from the reference leaks into it
    t, csi = synth(v_dyn=1.2, dyn_delay=60e-9)
    vel, p = profile_db(t, csi, 'sharp')
    assert level_at(vel, p, -1.16) < -12.0
    assert level_at(vel, p, +1.16) > level_at(vel, p, -1.16) - 5


def test_static_channel_has_no_power_outside_zero_lobe():
    t, csi = synth(a_dyn=0.0)
    vel, p = profile_db(t, csi, 'sharp')
    assert p[np.abs(vel) > 0.7].max() < -20


def test_raw_phase_is_broadband():
    t, csi = synth()
    vel, p = profile_db(t, csi, 'raw')
    assert np.median(p) > -6  # random per-packet CFO spreads power over all bins


def test_mode_tracker_separates_two_interleaved_links():
    rng = np.random.default_rng(1)
    t, a = synth(a_dyn=0.0, seed=2)
    _, b = synth(a_dyn=0.0, seed=2, static=STATIC_B)
    truth = rng.integers(0, 2, len(t))
    mixed = np.where(truth[:, None] == 1, b, a)
    lab, _ = sd.track_modes(t, mixed)
    acc = max(np.mean(lab == truth), np.mean(lab != truth))
    assert acc > 0.99, acc


def test_consistent_reference_handles_cyclic_shift_diversity():
    # two equal-power Tx chains 205 ns apart (CSD), as seen on the router
    csd_static = [(1.0, 30e-9, 0.3), (0.95, 30e-9 - 205e-9, 1.7), (0.3, 120e-9, 2.0)]
    t, csi = synth(a_dyn=0.0, static=csd_static)

    def coh(H):
        return float(np.mean(np.abs(H.mean(0)) ** 2 / np.mean(np.abs(H) ** 2, 0)))

    strongest = coh(sd.sanitize_sharp(csi, ref='strongest', ref_halfwidth_ns=0.0)[0])
    consistent = coh(sd.sanitize_sharp(csi)[0])
    assert consistent > 0.95
    assert consistent > strongest + 0.1
