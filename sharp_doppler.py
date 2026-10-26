"""SHARP-style Doppler spectrograms for Nexmon (bcm43455c0) PCAP captures.

Adapted to this dataset:
  * 80 MHz capture, but every frame is 20 MHz on primary ch36 -> 52 usable subcarriers.
  * Each transmitter alternates between two Tx antenna configurations ("modes")
    per packet; each (tx, mode) pair is processed as an independent link.
  * Packets are irregular, so sanitized CSI is resampled to a uniform grid.
"""
import csv
import json
import os

import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sharp_output')
FIG_DIR = os.path.join(OUT_DIR, 'figures')
CACHE_DIR = os.path.join(OUT_DIR, 'cache')

ROUTER = '8c:86:dd:b7:46:73'
LAPTOP = '50:bb:b5:f1:91:24'

# 80 MHz (fftshifted) index of 20 MHz subcarrier k is k - 96 + 128
K20 = np.r_[np.arange(-26, 0), np.arange(1, 27)]
ACT = K20 - 96 + 128
K20_FULL = np.arange(-26, 27)  # reconstruction includes DC, like SHARP's 245 = 242 + 3
# SHARP as published reads the 242 data/pilot tones of an 80 MHz frame; here only 52 of them carry signal
K80 = np.r_[np.arange(-122, -1), np.arange(2, 123)]
ACT80 = K80 + 128
K80_FULL = np.arange(-122, 123)
DELTA_F = 312.5e3
FC = 5.18e9
C = 3e8
LAMBDA = C / FC

FS = 200.0           # uniform resampling rate [Hz]
N_WIN = 36           # 36 * 5 ms = 180 ms, same duration as SHARP's 31 * 5.8 ms
N_FFT = 100
NOISE_DB = -12.0


# ---------------------------------------------------------------- PCAP I/O
def index_pcap(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f'index_{name}.npz')
    if os.path.exists(cache):
        return dict(np.load(cache))
    mm = np.memmap(os.path.join(DATA_DIR, f'{name}.pcap'), dtype=np.uint8, mode='r')
    size = mm.shape[0]
    ts, lens, offs = [], [], []
    ptr = 24
    while ptr + 16 <= size:
        hdr = mm[ptr:ptr + 16].view(np.uint32)
        incl = int(hdr[2])
        if ptr + 16 + incl > size:
            break
        ts.append(hdr[0] + hdr[1] * 1e-6)
        lens.append(incl)
        offs.append(ptr + 16)
        ptr += 16 + incl
    offs = np.array(offs, dtype=np.int64)
    p = offs + 42
    mac = mm[p[:, None] + np.arange(4, 10)]
    macs = np.array([':'.join(f'{b:02x}' for b in r) for r in mac])
    idx = dict(ts=np.array(ts), lens=np.array(lens), offs=offs, macs=macs,
               rssi=mm[p + 2].view(np.int8).copy())
    np.savez(cache, **idx)
    return idx


def load_csi(name, mac, t_start=None, t_end=None, tones=ACT):
    """Returns (t_rel, csi[n, len(tones)] complex64, rssi). t_rel is relative to the first packet of the capture."""
    idx = index_pcap(name)
    rel = idx['ts'] - idx['ts'][0]
    sel = (idx['macs'] == mac) & ((idx['lens'] == 1084) | (idx['lens'] == 1088))
    if t_start is not None:
        sel &= rel >= t_start
    if t_end is not None:
        sel &= rel < t_end
    mm = np.memmap(os.path.join(DATA_DIR, f'{name}.pcap'), dtype=np.uint8, mode='r')
    offs = idx['offs'][sel] + 42 + 18
    out = np.empty((len(offs), 256), np.complex64)
    for i in range(0, len(offs), 50000):
        o = offs[i:i + 50000]
        raw = mm[o[:, None] + np.arange(1024)].view(np.int16).reshape(-1, 512).astype(np.float32)
        out[i:i + 50000] = np.fft.fftshift(raw[:, ::2] + 1j * raw[:, 1::2], axes=1)
    return rel[sel], out[:, tones], idx['rssi'][sel]


def event_times(name):
    """Collision events mapped to PCAP-relative time (needs <name>_meta.json with video sync)."""
    meta_p = os.path.join(DATA_DIR, f'{name}_meta.json')
    ev_p = os.path.join(DATA_DIR, f'{name}_events.csv')
    if not (os.path.exists(meta_p) and os.path.exists(ev_p)):
        return None
    meta = json.load(open(meta_p))
    if meta.get('epoch_first_frame') is None:
        return None
    t0 = index_pcap(name)['ts'][0]
    a = meta.get('video_clock_scale', 1.0)
    out = []
    for row in csv.DictReader(open(ev_p)):
        out.append((meta['epoch_first_frame'] + a * float(row['time_sec']) - t0, row['event_type']))
    return out


# ---------------------------------------------------------------- Tx-mode separation
def _kmeans2(a):
    cen = np.vstack([a[0], a[np.argmax(1 - a @ a[0])]])
    for _ in range(15):
        lab = np.argmax(a @ cen.T, 1)
        cen = np.array([a[lab == j].mean(0) if np.any(lab == j) else cen[j] for j in range(2)])
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    return cen


def track_modes(t, csi, alpha=0.5, reinit_gap=0.05):
    """Assigns each packet to one of the transmitter's two antenna modes by tracking
    per-mode amplitude templates (same-mode packets are ~2 ms apart, so templates stay valid)."""
    a = np.abs(csi).astype(np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    lab = np.zeros(len(a), np.int8)
    margin = np.zeros(len(a))
    tmpl, last_t = None, -1e9
    for i in range(len(a)):
        if tmpl is None or t[i] - last_t > reinit_gap:
            j = max(np.searchsorted(t, t[i] + 0.03), i + 8)
            new = _kmeans2(a[i:j])
            if tmpl is not None and new[0] @ tmpl[0] + new[1] @ tmpl[1] < new[0] @ tmpl[1] + new[1] @ tmpl[0]:
                new = new[::-1]
            tmpl = new.copy()
        s = tmpl @ a[i]
        k = int(s[1] > s[0])
        lab[i], margin[i] = k, abs(s[1] - s[0])
        tmpl[k] = alpha * tmpl[k] + (1 - alpha) * a[i]
        tmpl[k] /= np.linalg.norm(tmpl[k])
        last_t = t[i]
    return lab, margin


# ---------------------------------------------------------------- phase sanitization
def _steering(k, tau):
    return np.exp(-2j * np.pi * DELTA_F * np.outer(k, tau)).astype(np.complex64)


def _lasso_paths(h, T, lam, n_iter):
    """Batched FISTA for min ||h - T r||^2 + lam (|Re r|_1 + |Im r|_1), i.e. SHARP's real-extended Lasso (Eq. 8-11)."""
    Th = T.conj().T
    G = (Th @ T).T
    L = 2 * np.linalg.norm(T, 2) ** 2
    thr = lam / L
    Thh = h @ Th.T
    r = np.zeros((len(h), T.shape[1]), np.complex64)
    y, tk = r.copy(), 1.0
    for _ in range(n_iter):
        g = y + (2.0 / L) * (Thh - y @ G)
        r_new = (np.sign(g.real) * np.maximum(np.abs(g.real) - thr, 0)
                 + 1j * np.sign(g.imag) * np.maximum(np.abs(g.imag) - thr, 0)).astype(np.complex64)
        tk_new = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        y = r_new + ((tk - 1) / tk_new) * (r_new - r)
        r, tk = r_new, tk_new
    return r


def _window_argmax(absr, centre, lo, hi):
    """Per-row argmax of absr restricted to columns centre+lo .. centre+hi."""
    cols = np.arange(absr.shape[1])[None, :]
    m = (cols >= (centre + lo)[:, None]) & (cols <= (centre + hi)[:, None])
    masked = np.where(m, absr, -1.0)
    return np.argmax(masked, 1), masked.max(1)


def sanitize_sharp(csi, t=None, lam=0.1, n_iter=60, step_ns=5.0, half_win_ns=500.0, csd_ns=205.0,
                   ref='tracked', ref_halfwidth_ns=20.0, reseed_gap=0.2, chunk=20000, k=K20, k_full=K20_FULL):
    """SHARP Sec. 3.1: Lasso path decomposition h = T r, then reference all paths to one path (Eq. 14-15).

    ref='strongest'  : SHARP's rule, the single strongest atom.
    ref='consistent' : these transmitters use 2-chain cyclic-shift diversity, so the channel has two
                       near-equal taps ~csd_ns apart and 'strongest' flips between them packet to packet.
                       Here the reference is always the same tap of the CSD pair (the one that is stronger
                       on average over the call), decided per packet from where the partner tap sits.
    ref='tracked'    : like 'consistent', but the tap is chosen by continuity with the previous packet's
                       referenced CFR, so multipath changes cannot flip the reference (re-seeded after
                       gaps > reseed_gap seconds when t is given).
    ref_halfwidth_ns : >0 uses the sum of atoms within +-halfwidth of the reference as the reference
                       (stabilises the phase when the Lasso splits a path over neighbouring atoms).
    Returns sanitized CFR on 53 subcarriers (k = -26..26, DC reconstructed) and the reference delay."""
    h_all = (csi / np.mean(np.abs(csi), axis=1, keepdims=True)).astype(np.complex64)
    coarse = np.arange(-1600, 1600, 10.0) * 1e-9
    S_coarse = _steering(k, coarse).conj()
    tau = np.arange(-half_win_ns, half_win_ns + step_ns / 2, step_ns) * 1e-9
    T = _steering(k, tau)
    Tfull = _steering(k_full, tau)
    csd = int(round(csd_ns / step_ns))
    tol = int(round(30 / step_ns))
    hw = int(round(ref_halfwidth_ns / step_ns))
    rs, picks = [], []
    for i in range(0, len(csi), chunk):
        h = h_all[i:i + chunk]
        peak = coarse[np.argmax(np.abs(h @ S_coarse), 1)]
        h = h * np.exp(2j * np.pi * DELTA_F * np.outer(peak, k)).astype(np.complex64)
        r = _lasso_paths(h, T, lam, n_iter)
        absr = np.abs(r)
        a = np.argmax(absr, 1)
        if ref == 'strongest':
            picks.append((a, a, np.ones(len(a), bool)))
        else:
            b_minus, e_minus = _window_argmax(absr, a, -csd - tol, -csd + tol)
            b_plus, e_plus = _window_argmax(absr, a, csd - tol, csd + tol)
            a_is_late = e_minus > e_plus
            no_pair = np.maximum(e_minus, e_plus) < 0.3 * absr[np.arange(len(a)), a]
            late = np.where(a_is_late | no_pair, a, b_plus)
            early = np.where(~a_is_late | no_pair, a, b_minus)
            picks.append((late, early, a_is_late))
        rs.append(r)
    if ref == 'strongest':
        choose_late = True
    else:
        amp_late = np.concatenate([np.abs(r[np.arange(len(r)), p[0]]) for r, p in zip(rs, picks)])
        amp_early = np.concatenate([np.abs(r[np.arange(len(r)), p[1]]) for r, p in zip(rs, picks)])
        choose_late = np.median(amp_late) >= np.median(amp_early)

    def referenced(r, p):
        if hw > 0:
            cols = np.arange(r.shape[1])[None, :]
            ref_k = (np.where(np.abs(cols - p[:, None]) <= hw, r, 0) @ Tfull.T).conj()
        else:
            ref_k = (Tfull[:, p].T * r[np.arange(len(r)), p][:, None]).conj()
        hh = (r @ Tfull.T) * ref_k
        return hh / np.mean(np.abs(hh), axis=1, keepdims=True)

    out = np.empty((len(csi), len(k_full)), np.complex64)
    ref_tau = np.empty(len(csi))
    pos, prev, last_t = 0, None, -np.inf
    for r, (late, early, _) in zip(rs, picks):
        first, second = (late, early) if choose_late else (early, late)
        n = len(r)
        if ref != 'tracked':
            out[pos:pos + n] = referenced(r, first)
            ref_tau[pos:pos + n] = tau[first]
        else:
            # pick, packet by packet, the CSD tap whose referenced CFR continues the previous packet
            c1, c2 = referenced(r, first), referenced(r, second)
            for j in range(n):
                tj = t[pos + j] if t is not None else 0.0
                if prev is None or tj - last_t > reseed_gap:
                    use_second = False
                else:
                    use_second = np.real(np.vdot(prev, c2[j])) > np.real(np.vdot(prev, c1[j]))
                prev = c2[j] if use_second else c1[j]
                out[pos + j] = prev
                ref_tau[pos + j] = tau[second[j] if use_second else first[j]]
                last_t = tj
        pos += n
    return out, ref_tau


def sanitize_linear(csi):
    """Baseline: remove per-packet linear phase (offset + slope) across subcarriers."""
    h = csi / np.mean(np.abs(csi), axis=1, keepdims=True)
    ph = np.unwrap(np.angle(h), axis=1)
    A = np.vstack([np.ones_like(K20), K20]).T.astype(float)
    coef = np.linalg.lstsq(A, ph.T, rcond=None)[0]
    return np.abs(h) * np.exp(1j * (ph - (A @ coef).T))


def sanitize(csi, method, t=None):
    if method == 'sharp':
        return sanitize_sharp(csi, t)[0]
    if method == 'sharp_consistent':
        return sanitize_sharp(csi, t, ref='consistent')[0]
    if method == 'sharp_strongest':
        return sanitize_sharp(csi, t, ref='strongest', ref_halfwidth_ns=0.0)[0]
    if method == 'sharp_published':  # expects the 242 tones of load_csi(..., tones=ACT80)
        return sanitize_sharp(csi, t, ref='strongest', ref_halfwidth_ns=0.0, k=K80, k_full=K80_FULL)[0]
    if method == 'linear':
        return sanitize_linear(csi)
    if method == 'raw':
        return csi / np.mean(np.abs(csi), axis=1, keepdims=True)
    if method == 'power':
        h = np.abs(csi) / np.mean(np.abs(csi), axis=1, keepdims=True)
        return h ** 2 - np.mean(h ** 2, axis=0, keepdims=True)
    raise ValueError(method)


# ---------------------------------------------------------------- Doppler
def resample_uniform(t, x, t0, t1, fs=FS, max_gap=0.05):
    """Bin-average onto a uniform grid; empty bins are linearly interpolated, and bins
    farther than max_gap from any packet are flagged invalid."""
    grid = np.arange(t0, t1, 1 / fs)
    b = np.floor((t - t0) * fs).astype(int)
    ok = (b >= 0) & (b < len(grid))
    b, xx = b[ok], x[ok]
    cnt = np.bincount(b, minlength=len(grid))
    y = np.zeros((len(grid), x.shape[1]), np.complex64)
    np.add.at(y, b, xx)
    have = cnt > 0
    y[have] /= cnt[have, None]
    if have.sum() >= 2:
        hi = np.flatnonzero(have)
        for j in range(x.shape[1]):
            y[~have, j] = (np.interp(grid[~have], grid[hi], y[hi, j].real)
                           + 1j * np.interp(grid[~have], grid[hi], y[hi, j].imag))
    tp = t[ok]
    j = np.clip(np.searchsorted(tp, grid), 1, len(tp) - 1)
    near = np.minimum(np.abs(grid - tp[j - 1]), np.abs(grid - tp[j])) <= max_gap
    return grid, y, near


def doppler_spectrogram(h, n_win=N_WIN, n_fft=N_FFT, hop=1, noise_db=NOISE_DB, fs=FS):
    """SHARP Eq. (20): d(u) = sum_k |FFT_n{H(k,n) w(n)}|^2, Hann window, zero-padded, max-normalised, floored."""
    w = np.hanning(n_win).astype(np.float32)[:, None]
    starts = np.arange(0, len(h) - n_win + 1, hop)
    D = np.empty((len(starts), n_fft), np.float32)
    for i in range(0, len(starts), 2000):
        s = starts[i:i + 2000]
        seg = h[s[:, None] + np.arange(n_win)] * w
        F = np.fft.fftshift(np.fft.fft(seg, n=n_fft, axis=1), axes=1)
        D[i:i + 2000] = np.sum(np.abs(F) ** 2, axis=2)
    D /= np.max(D, axis=1, keepdims=True)
    D = np.maximum(D, 10 ** (noise_db / 10))
    t_centre = (starts + n_win / 2) / fs
    vel = (np.arange(n_fft) - n_fft // 2) * fs / n_fft * LAMBDA
    return t_centre, vel, D


def link_spectrogram(name, mac, mode, t0, t1, method='sharp', pad=1.0, hop=1, noise_db=NOISE_DB, tones=ACT):
    """Spectrogram of one link (tx, antenna mode) between t0 and t1 [PCAP-relative seconds].
    mode=None mixes both antenna modes (to show why separation matters).
    Returns (t, velocity, D[time, velocity], valid[time])."""
    t, csi, _ = load_csi(name, mac, t0 - pad, t1 + pad, tones)
    if mode is not None:
        lab, _ = track_modes(t, csi)
        t, csi = t[lab == mode], csi[lab == mode]
    h = sanitize(csi, method, t)
    _, hu, valid = resample_uniform(t, h, t0, t1)
    tc, vel, D = doppler_spectrogram(hu, hop=hop, noise_db=noise_db)
    centre = np.round(tc * FS).astype(int).clip(0, len(valid) - 1)
    return t0 + tc, vel, D, valid[centre]


def dynamic_fraction(vel, D, v_min=0.7):
    """Share of Doppler power outside the zero-velocity lobe (|v| > v_min), per Doppler vector."""
    return D[:, np.abs(vel) > v_min].sum(1) / D.sum(1)
