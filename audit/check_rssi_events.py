import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd, make_figures as mf
FS = 20.0  # 50 ms bins

def series(name, mac):
    idx = sd.index_pcap(name)
    rel = idx['ts'] - idx['ts'][0]
    s = idx['macs'] == mac
    t, r = rel[s], idx['rssi'][s].astype(float)
    b = (t * FS).astype(int)
    n = np.bincount(b)
    mean = np.bincount(b, r) / np.maximum(n, 1)
    sq = np.bincount(b, r * r) / np.maximum(n, 1)
    tt = np.arange(len(n)) / FS
    ok = n >= 5
    mean = np.interp(tt, tt[ok], mean[ok]); sd_ = np.sqrt(np.maximum(np.interp(tt, tt[ok], sq[ok]) - mean ** 2, 0))
    k = int(0.5 * FS)
    # fast RSSI fluctuation: |x - 0.5 s moving average|
    ma = np.convolve(mean, np.ones(k) / k, 'same')
    return tt, {'rssi': mean, 'rssi_fast': np.abs(mean - ma), 'rssi_std': sd_}

def test(name, sig_t, sig, win=(0, 1.0), base=(-2.5, -0.5), n_ctrl=400):
    ev, col, t_end = mf.collisions_in_capture(name, 6.0)
    starts = np.array([s for s, _ in col])
    walk1 = min([t for t, e in ev if e == 'ignore_start' and t > starts[-1]] + [t_end])
    rng = np.random.default_rng(0)
    cand = rng.uniform(mf.WALK_START[name] + 6, walk1 - 6, n_ctrl * 5)
    cand = cand[np.min(np.abs(cand[:, None] - starts[None]), 1) > 3][:n_ctrl]
    def d(o):
        w = np.interp(o + np.linspace(*win, 20), sig_t, sig).mean()
        b = np.interp(o + np.linspace(*base, 40), sig_t, sig).mean()
        return w - b
    de = np.array([d(o) for o in starts]); dc = np.array([d(o) for o in cand])
    null = np.array([rng.choice(dc, len(de), replace=False).mean() for _ in range(4000)])
    p2 = (np.sum(np.abs(null - dc.mean()) >= abs(de.mean() - dc.mean())) + 1) / (len(null) + 1)
    return len(de), np.sum(de > 0), de.mean(), dc.mean(), dc.std(), p2

if __name__ == "__main__":
    for name in ["m01", "x02"]:
      for mac in [sd.ROUTER, sd.LAPTOP]:
          tt, S = series(name, mac)
          for key, x in S.items():
              for win, base in [((0, 1), (-2.5, -0.5)), ((-0.5, 1.5), (-4.5, -2.5))]:
                  n, up, me, mc, sc, p = test(name, tt, x, win, base)
                  print(f'{name} {mac[:2]} {key:10s} win{win}: up {up}/{n}  ev {me:+.3f}  ctrl {mc:+.3f}±{sc:.3f}  two-sided p {p:.3f}')
