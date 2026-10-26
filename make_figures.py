"""Figures for the SHARP-style Doppler analysis.

python3 make_figures.py full   # cache full-capture spectrograms (all links, all captures)
python3 make_figures.py figs   # render figures into sharp_output/figures/

Figures (simple names, see sharp_output/README.md):
  01_sharp_before_after      SHARP as published vs our version, step by step, same 6 s
  02_what_motion_looks_like  empty room / standing / walking / collision, our version
  03_one_collision_all_links one collision seen by all four links
  04_all_collisions_<cap>    every collision of a capture, 3 s before to 3 s after
  05_collision_test_<cap>    motion around collisions vs random moments of walking
  06_full_sessions           whole captures with the collisions marked
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import sharp_doppler as sd

LINKS = [(sd.ROUTER, 0, 'router, antenna mode A'), (sd.ROUTER, 1, 'router, antenna mode B'),
         (sd.LAPTOP, 0, 'laptop, antenna mode A'), (sd.LAPTOP, 1, 'laptop, antenna mode B')]
CAPTURES = ['m_empty', 'm01', 'x02']
HOP = 4  # 20 ms between Doppler vectors in the cached full-capture spectrograms
RAW_DB = -60.0      # spectra are computed unfloored; floors are applied for display only
DISPLAY_DB = -30.0  # extended view: motion here sits 15-25 dB below the reference path
WALK_START = {'m01': 190.0, 'x02': 90.0}  # PCAP time from which control onsets are drawn
SPEED = 'speed towards / away [m/s]'
POWER = 'strength vs. strongest [dB]'


def full_path(name, mac, mode):
    return os.path.join(sd.CACHE_DIR, f'full_{name}_{mac[:2]}_{mode}.npz')


def save(fig, fname, dpi=110):
    os.makedirs(sd.FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(sd.FIG_DIR, fname), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname, flush=True)


def compute_full():
    os.makedirs(sd.CACHE_DIR, exist_ok=True)
    for name in CAPTURES:
        idx = sd.index_pcap(name)
        dur = idx['ts'][-1] - idx['ts'][0]
        for mac, mode, label in LINKS:
            p = full_path(name, mac, mode)
            if os.path.exists(p):
                continue
            t, vel, D, valid = sd.link_spectrogram(name, mac, mode, 0.0, dur, pad=0.0, hop=HOP, noise_db=RAW_DB)
            np.savez_compressed(p, t=t, vel=vel, D=D.astype(np.float16), valid=valid)
            print('saved', p, D.shape, flush=True)


def load_full(name, mac, mode):
    z = np.load(full_path(name, mac, mode))
    return z['t'], z['vel'], z['D'].astype(np.float32), z['valid']


def show(ax, t, vel, D, title, vmax_v=4.0, events=None, floor_db=DISPLAY_DB):
    keep = np.abs(vel) <= vmax_v
    Dd = np.maximum(D[:, keep], 10 ** (floor_db / 10))
    im = ax.pcolormesh(t, vel[keep], 10 * np.log10(Dd.T), shading='auto', cmap='viridis', vmin=floor_db, vmax=0)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(SPEED, fontsize=8)
    if events:
        for te, e in events:
            if t[0] <= te <= t[-1] and e.startswith('collision'):
                ax.axvline(te, color='r' if e.endswith('start') else 'w', lw=1.0, ls='--')
    return im


def collisions(name):
    ev = sd.event_times(name) or []
    starts = [t for t, e in ev if e == 'collision_start']
    ends = [t for t, e in ev if e == 'collision_end']
    return ev, list(zip(starts, ends))


def fig_before_after():
    """SHARP as published, then each change we made, on the same 6 s around m01 collision 1."""
    ev, col = collisions('m01')
    c0 = col[0][0]
    t0, t1 = c0 - 3.0, c0 + 3.0
    rows = [  # (method, antenna mode, tones, display floor, title)
        ('sharp_published', None, sd.ACT80, sd.NOISE_DB,
         'a) SHARP exactly as published: all 242 values, both antenna modes mixed, '
         'strongest-copy reference, hide below -12 dB'),
        ('sharp_published', None, sd.ACT80, DISPLAY_DB, 'b) same, but showing down to -30 dB'),
        ('sharp_strongest', None, sd.ACT, DISPLAY_DB, 'c) + keep only the 52 values that carry signal'),
        ('sharp_strongest', 0, sd.ACT, DISPLAY_DB, 'd) + split the two antenna modes (one mode shown)'),
        ('sharp', 0, sd.ACT, DISPLAY_DB, 'e) + steady reference copy = OUR PIPELINE'),
    ]
    fig, axs = plt.subplots(len(rows), 1, figsize=(12, 14), sharex=True)
    for ax, (method, mode, tones, floor, title) in zip(axs, rows):
        t, vel, D, _ = sd.link_spectrogram('m01', sd.ROUTER, mode, t0, t1, method=method, noise_db=RAW_DB, tones=tones)
        im = show(ax, t - c0, vel, D, title, events=[(te - c0, e) for te, e in ev], floor_db=floor)
        fig.colorbar(im, ax=ax, label=POWER, pad=0.01)
    axs[-1].set_xlabel('time from the start of collision 1 in m01 [s]   (red line = collision starts, white = ends)')
    save(fig, '01_sharp_before_after.png')


def fig_what_motion_looks_like():
    """SHARP Fig. 3 analogue, our pipeline, router link mode A, 3 s each."""
    ev, col = collisions('m01')
    c0 = col[0][0]
    cases = [('m_empty', 150.0, 'empty room'),
             ('m01', 100.0, 'm01, before walking'),
             ('m01', 300.0, 'm01, walking (calm moment)'),
             ('m01', 190.0, 'm01, walking (busy moment)'),
             ('m01', c0 - 1.5, 'm01, collision 1'),
             ('x02', 200.0, 'x02, walking')]
    fig, axs = plt.subplots(1, len(cases), figsize=(22, 3.8), sharey=True)
    for ax, (name, t0, title) in zip(axs, cases):
        t, vel, D, _ = sd.link_spectrogram(name, sd.ROUTER, 0, t0, t0 + 3.0, noise_db=RAW_DB)
        evs = [(te - t0, e) for te, e in ev] if name == 'm01' else None
        im = show(ax, t - t0, vel, D, title, events=evs)
        ax.set_xlabel('time [s]')
    for ax in axs[1:]:
        ax.set_ylabel('')
    fig.colorbar(im, ax=axs, label=POWER, pad=0.01)
    save(fig, '02_what_motion_looks_like.png')


def fig_one_collision_all_links():
    ev, col = collisions('m01')
    c0 = col[0][0]
    fig, axs = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    for ax, (mac, mode, label) in zip(axs, LINKS):
        t, vel, D, valid = sd.link_spectrogram('m01', mac, mode, c0 - 3, c0 + 3, noise_db=RAW_DB)
        D = D.copy()
        D[~valid] = 10 ** (RAW_DB / 10)  # dark = too few packets to trust
        im = show(ax, t - c0, vel, D, label, events=[(te - c0, e) for te, e in ev])
    axs[-1].set_xlabel('time from the start of collision 1 in m01 [s]   (dark gaps = not enough packets)')
    fig.colorbar(im, ax=axs, label=POWER)
    save(fig, '03_one_collision_all_links.png')


def fused_dynamic(name, smooth_s=0.5):
    """Mean dynamic fraction over the 4 links (only valid samples), smoothed."""
    acc, cnt, t_ref = None, None, None
    for mac, mode, _ in LINKS:
        t, vel, D, valid = load_full(name, mac, mode)
        f = sd.dynamic_fraction(vel, D)
        if t_ref is None:
            t_ref, acc, cnt = t, np.zeros_like(t), np.zeros_like(t)
        n = min(len(t_ref), len(t))
        acc[:n] += np.where(valid[:n], f[:n], 0)
        cnt[:n] += valid[:n]
    fused = acc / np.maximum(cnt, 1)
    k = max(1, int(smooth_s / (HOP / sd.FS)))
    return t_ref, np.convolve(fused, np.ones(k) / k, mode='same')


def fig_full_sessions():
    fig, axs = plt.subplots(len(CAPTURES), 1, figsize=(18, 12))
    titles = {'m_empty': 'm_empty: empty room', 'm01': 'm01: two people walking, collisions marked',
              'x02': 'x02: two people walking, collisions marked'}
    for ax, name in zip(axs, CAPTURES):
        t, vel, D, valid = load_full(name, sd.ROUTER, 0)
        step = 5
        im = show(ax, t[::step], vel, D[::step], titles[name], events=sd.event_times(name))
        ax2 = ax.twinx()
        tf, fd = fused_dynamic(name)
        ax2.plot(tf, fd, color='w', lw=0.8)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('motion score (white line)')
    axs[-1].set_xlabel('time in the capture [s]')
    fig.colorbar(im, ax=axs, label=POWER)
    save(fig, '06_full_sessions.png', dpi=90)


def collisions_in_capture(name, margin=6.0):
    ev, col = collisions(name)
    t_end = sd.index_pcap(name)['ts'][-1] - sd.index_pcap(name)['ts'][0]
    return ev, [(s, e) for s, e in col if s - margin >= 0 and e + margin <= t_end], t_end


def event_locked(name='m01', pre=6.0, post=6.0, n_ctrl=400, seed=0):
    """Collision-locked dynamic fraction, against random control onsets drawn from the same
    walking period but >= 3 s away from any collision."""
    ev, col, t_end = collisions_in_capture(name, max(pre, post))
    tf, fd = fused_dynamic(name, smooth_s=0.2)
    dt = tf[1] - tf[0]
    lag = np.arange(-pre, post, dt)
    starts = np.array([s for s, _ in col])
    walk0 = WALK_START[name]
    walk1 = min([t for t, e in ev if e == 'ignore_start' and t > starts[-1]] + [t_end])

    def locked(onsets):
        return np.array([np.interp(o + lag, tf, fd) for o in onsets])

    rng = np.random.default_rng(seed)
    cand = rng.uniform(walk0 + pre, walk1 - post, n_ctrl * 5)
    cand = cand[np.min(np.abs(cand[:, None] - starts[None, :]), 1) > 3.0][:n_ctrl]
    return lag, locked(starts), locked(cand)


def fig_collision_test(name='m01', win_s=(0.0, 1.0), base_s=(-2.5, -0.5)):
    lag, ev_tr, ctrl = event_locked(name)
    win = (lag >= win_s[0]) & (lag <= win_s[1])
    base = (lag >= base_s[0]) & (lag <= base_s[1])
    d_ev = ev_tr[:, win].mean(1) - ev_tr[:, base].mean(1)
    d_ct = ctrl[:, win].mean(1) - ctrl[:, base].mean(1)
    rng = np.random.default_rng(1)
    null = np.array([rng.choice(d_ct, len(d_ev), replace=False).mean() for _ in range(5000)])
    p_mean = (np.sum(null >= d_ev.mean()) + 1) / (len(null) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    ax = axs[0]
    for tr in ev_tr:
        ax.plot(lag, tr, color='tab:red', alpha=0.2, lw=0.7)
    ax.plot(lag, ev_tr.mean(0), color='tab:red', lw=2, label=f'average over {len(ev_tr)} collisions')
    lo, hi = np.percentile(ctrl, [2.5, 97.5], axis=0)
    ax.fill_between(lag, lo, hi, color='gray', alpha=0.3, label='range at random walking moments')
    ax.plot(lag, ctrl.mean(0), color='k', lw=1.5, label='average at random walking moments')
    ax.axvline(0, color='k', ls='--', lw=0.8)
    ax.set_xlabel('time from collision start [s]'); ax.set_ylabel('motion score')
    ax.legend(fontsize=8); ax.set_title(f'{name}: motion score around collisions')

    ax = axs[1]
    ax.hist(d_ct, bins=40, color='gray', alpha=0.7, density=True, label='random walking moments')
    for x in d_ev:
        ax.axvline(x, color='tab:red', alpha=0.6, lw=1)
    ax.axvline(d_ev.mean(), color='tab:red', lw=2.5, label=f'collisions, average (p = {p_mean:.2f})')
    ax.set_xlabel('rise in motion score: first 1 s of collision minus 2 s before'); ax.legend(fontsize=8)
    ax.set_title(f'{int(np.sum(d_ev > 0))} of {len(d_ev)} collisions rise (p < 0.05 would be a real effect)')
    save(fig, f'05_collision_test_{name}.png')
    return dict(n=len(d_ev), n_up=int(np.sum(d_ev > 0)), d_ev_mean=d_ev.mean(), d_ct_mean=d_ct.mean(), p_mean=p_mean)


def fig_all_collisions(name='m01'):
    ev, col, _ = collisions_in_capture(name, 3.0)
    n = len(col)
    cols = 4
    fig, axs = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(18, 3 * np.ceil(n / cols)), sharey=True)
    t, vel, D, valid = load_full(name, sd.ROUTER, 0)
    for i, (ax, (s, e)) in enumerate(zip(axs.ravel(), col)):
        m = (t >= s - 3) & (t <= s + 3)
        show(ax, t[m] - s, vel, D[m], f'{name} collision {i + 1} (lasts {e - s:.1f} s)',
             events=[(0, 'collision_start'), (e - s, 'collision_end')])
    for ax in axs.ravel()[n:]:
        ax.axis('off')
    fig.suptitle(f'{name}: every collision, router link, 3 s before to 3 s after (red = start, white = end)', y=1.0)
    fig.tight_layout()
    save(fig, f'04_all_collisions_{name}.png', dpi=90)


if __name__ == '__main__':
    if sys.argv[1] == 'full':
        compute_full()
    else:
        fig_before_after()
        fig_what_motion_looks_like()
        fig_one_collision_all_links()
        for name in ('m01', 'x02'):
            fig_all_collisions(name)
            print(name, fig_collision_test(name))
        fig_full_sessions()
