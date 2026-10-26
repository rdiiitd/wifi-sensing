"""z01: everything we can try on the walking problem with one session.

The question: while the wearer walks, can we tell that another person is moving close by?
Data: wearer walking alone (video 205-265 s) and walking with approaches (445-505 s, 9 paired approaches).

Common evaluation (half-second windows, 0.25 s step):
  T1     walking with approaches vs walking alone (AUC; is another person around at all?)
  T1pre  the second before a contact vs walking alone (AUC; the moment that matters)
  T2     near (< 0.9 m) vs far (> 1.2 m) inside the walking-approach part (AUC)
  T3     second before contact vs the farthest point between contacts, paired (k of 9)
  AUC CIs: 2.5-97.5 % of a bootstrap over 5 s blocks (windows overlap, so single windows are not independent).
Also run on the head-turn part as a check that a method does not only fit walking.

Ideas:
  0  baselines: motion score, speed bands, direction asymmetry, per-link scores, IMU speed gate,
     the collector's CSI features and IMU-regressed leftovers
  1  quiet moments: only windows or frames where the head moves least
  2  gait template: what the wearer's own Doppler looks like at each point of a step (learned while walking
     alone, cross-fitted), subtracted
  3  self-reference: change against the same wearer's own recent walking (last L s)
  4  small learned model on the band features + IMU (logistic regression, cross-validated by approach / by block)
"""
import os
import sys

import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
import make_figures as mf
import z01_imu as zi
import z01_selfspeed as ss
from z01_check import NAME, video_to_pcap

to_pcap = video_to_pcap(NAME)
PARTS = {'walking': ((205, 265), (445, 505)), 'head turns': ((85, 205), (345, 445))}
BANDS = [(0.3, 0.7), (0.7, 1.2), (1.2, 2.0), (2.0, 3.0), (3.0, 5.0)]
BLOCK = 5.0
rng = np.random.default_rng(0)


# ---------- data ----------

def link_profiles():
    """Per link: sum-normalised Doppler profiles on the common 20 ms grid (nan where the link has no data)."""
    out, t_ref = [], None
    for mac, mode, _ in mf.LINKS:
        t, vel, D, valid = mf.load_full(NAME, mac, mode)
        P = (D / D.sum(1, keepdims=True)).astype(np.float32)
        if t_ref is None:
            t_ref = t
        n = min(len(t_ref), len(t))
        Q = np.full((len(t_ref), D.shape[1]), np.nan, np.float32)
        Q[:n] = np.where(valid[:n, None], P[:n], np.nan)
        out.append(Q)
    return t_ref, vel, out


def imu_frames(tf):
    ti, acc, gyr = ss.load_imu_raw()
    va, pk, _, _, _ = ss.antenna_speed(ti, acc, gyr, to_pcap)
    turn = np.linalg.norm(gyr, axis=1)
    shake = np.linalg.norm(acc - np.stack([zi.boxcar(acc[:, i], 400) for i in range(3)], 1), axis=1)
    f = lambda x: np.interp(tf, ti, x, left=np.nan, right=np.nan)
    return dict(va=f(va), turn=f(zi.boxcar(turn, 144)), shake=f(zi.boxcar(shake, 144)),
                steps=ti[pk], ti0=ti[0], ti1=ti[-1])


def gait_phase(tf, steps, max_gap=1.2):
    """0..1 between consecutive step peaks, nan when no step within max_gap."""
    ph = np.full(len(tf), np.nan)
    j = np.searchsorted(steps, tf) - 1
    ok = (j >= 0) & (j < len(steps) - 1)
    a, b = steps[np.clip(j, 0, len(steps) - 2)], steps[np.clip(j + 1, 1, len(steps) - 1)]
    ok &= (b - a) <= max_gap
    ph[ok] = ((tf - a) / (b - a))[ok]
    return ph


# ---------- evaluation ----------

def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return np.nan
    r = rankdata(np.r_[pos, neg])
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(x, pos, neg, tc, n_boot=300):
    """AUC and a 5 s block bootstrap interval."""
    a = auc(x[pos], x[neg])
    bp, bn = np.floor(tc[pos] / BLOCK), np.floor(tc[neg] / BLOCK)
    up, un = np.unique(bp), np.unique(bn)
    xp, xn = x[pos], x[neg]
    gp = [xp[bp == u] for u in up]
    gn = [xn[bn == u] for u in un]
    bs = []
    for _ in range(n_boot):
        p = np.concatenate([gp[i] for i in rng.integers(0, len(gp), len(gp))])
        q = np.concatenate([gn[i] for i in rng.integers(0, len(gn), len(gn))])
        bs.append(auc(p, q))
    return a, np.nanpercentile(bs, [2.5, 97.5])


class Evaluator:
    def __init__(self, tc, dw, starts, ends, td, dist):
        self.tc, self.dw = tc, dw
        self.sets = {}
        for kind, ((a0, a1), (p0, p1)) in PARTS.items():
            A0, A1, P0, P1 = to_pcap([a0, a1, p0, p1])
            alone = (tc >= A0 + 1) & (tc < A1 - 1)
            appr = (tc >= P0) & (tc < P1)
            pre = np.zeros(len(tc), bool)
            pairs = []
            for o in starts[(starts >= P0) & (starts < P1)]:
                pre |= (tc >= o - 1) & (tc < o)
                prev = ends[ends < o]
                if not len(prev) or prev[-1] < P0:
                    continue
                m = (td > prev[-1]) & (td < o) & ~np.isnan(dist)
                if m.any():
                    ft = td[m][np.argmax(dist[m])]
                    pairs.append(((tc >= o - 1) & (tc < o), (tc >= ft - 0.5) & (tc < ft + 0.5)))
            self.sets[kind] = dict(alone=alone, appr=appr, pre=pre & appr, near=appr & (dw < zi.NEAR),
                                   far=appr & (dw > zi.FAR), pairs=pairs)

    def run(self, x, kind='walking', ci=True):
        s = self.sets[kind]
        r = {}
        for key, (p, n) in {'T1': ('appr', 'alone'), 'T1pre': ('pre', 'alone'), 'T2': ('near', 'far')}.items():
            r[key] = auc_ci(x, s[p], s[n], self.tc) if ci else (auc(x[s[p]], x[s[n]]), None)
        c = np.array([np.nanmean(x[a]) for a, b in s['pairs']])
        f = np.array([np.nanmean(x[b]) for a, b in s['pairs']])
        k = int(np.sum(c > f))
        r['T3'] = (k, len(c), zi.sign_p(k, len(c)))
        return r


def fmt(name, r):
    def a(v):
        return f'{v[0]:.2f} [{v[1][0]:.2f}-{v[1][1]:.2f}]' if v[1] is not None else f'{v[0]:.2f}'
    k, n, p = r['T3']
    return f'  {name:38s} T1 {a(r["T1"]):17s} T1pre {a(r["T1pre"]):17s} T2 {a(r["T2"]):17s} T3 {k}/{n} p {p:.3f}'


def windows(tc, tf, x, sel=None):
    """Mean of a per-frame series over each window; sel (bool per frame) restricts which frames count."""
    j0 = np.searchsorted(tf, tc - zi.WIN / 2)
    j1 = np.searchsorted(tf, tc + zi.WIN / 2)
    out = np.full(len(tc), np.nan)
    for k in range(len(tc)):
        v = x[j0[k]:j1[k]]
        if sel is not None:
            v = v[sel[j0[k]:j1[k]]]
        v = v[~np.isnan(v)]
        if len(v) >= 3:
            out[k] = v.mean()
    return out


def self_reference(tc, x, L, gap=0.5, mode='diff'):
    """x minus (or over) the median of the same series over [t - gap - L, t - gap]."""
    out = np.full(len(tc), np.nan)
    for k in range(len(tc)):
        m = (tc >= tc[k] - gap - L) & (tc < tc[k] - gap)
        v = x[m]
        v = v[~np.isnan(v)]
        if len(v) >= max(3, int(0.5 * L / zi.STEP)):
            ref = np.median(v)
            out[k] = x[k] - ref if mode == 'diff' else x[k] / max(ref, 1e-6)
    return out


def imu_matched(tc, x, Z, ref, k=15, excl=3.0):
    """x minus the mean of x over the k reference windows with the most similar head motion (none within excl s)."""
    out = np.full(len(tc), np.nan)
    ir = np.flatnonzero(ref & ~np.isnan(x) & ~np.isnan(Z).any(1))
    for j in np.flatnonzero(~np.isnan(x) & ~np.isnan(Z).any(1)):
        d = np.sum((Z[ir] - Z[j]) ** 2, 1)
        d[np.abs(tc[ir] - tc[j]) < excl] = np.inf
        nn = ir[np.argsort(d)[:k]]
        out[j] = x[j] - x[nn].mean()
    return out


def smooth(x, n):
    """Moving mean over n consecutive windows (nan-aware): a longer decision window."""
    v = np.where(np.isnan(x), 0, x)
    c = np.convolve(~np.isnan(x), np.ones(n), mode='same')
    out = np.convolve(v, np.ones(n), mode='same') / np.maximum(c, 1)
    out[c < n / 2] = np.nan
    return out


def logreg(X, y, lam=1.0, iters=300):
    """L2 logistic regression by Newton steps; X already standardised; returns weights incl. bias."""
    Xb = np.c_[np.ones(len(X)), X]
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) + lam * np.r_[0, w[1:]]
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + lam * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-6:
            break
    return w


if __name__ == '__main__':
    tf, vel, LP = link_profiles()
    P = np.nanmean(np.stack(LP), 0)
    ok = ~np.isnan(P).any(1)
    imu = imu_frames(tf)
    av = np.abs(vel)

    tc = np.arange(imu['ti0'] + 1, imu['ti1'] - 1, zi.STEP)
    td, dist = zi.load_distance(to_pcap)
    dw = zi.at(tc, td, dist)
    ev = sd.event_times(NAME)
    starts = np.array([t for t, e in ev if e == 'collision_start'])
    ends = np.array([t for t, e in ev if e == 'collision_end'])
    E = Evaluator(tc, dw, starts, ends, td, dist)
    W = lambda x, sel=None: windows(tc, tf, x, sel)

    A0, A1 = to_pcap(PARTS['walking'][0])
    walk_alone_f = (tf >= A0) & (tf < A1) & ok
    results = {}

    def report(name, x, kinds=('walking',)):
        for kind in kinds:
            r = E.run(x, kind)
            results[(name, kind)] = r
            print(fmt(name if kind == 'walking' else f'   (head-turn part) {name}', r))

    # ---------- 0. baselines ----------
    print('== 0. baselines ==')
    raw_f = P[:, av > 0.7].sum(1)
    raw = W(raw_f)
    report('motion score (> 0.7 m/s)', raw, ('walking', 'head turns'))
    for lo, hi in BANDS:
        report(f'band {lo}-{hi} m/s', W(P[:, (av > lo) & (av <= hi)].sum(1)))
    pos, neg = P[:, vel > 0.7].sum(1), P[:, vel < -0.7].sum(1)
    report('direction: towards minus away', W(pos - neg))
    report('direction: |towards - away| / sum', W(np.abs(pos - neg) / (pos + neg)))
    for (mac, mode, lab), Q in zip(mf.LINKS, LP):
        report(f'link {lab}', W(Q[:, av > 0.7].sum(1)))
    gate = np.maximum(0.7, ss.SELF_FACTOR * imu['va'] + ss.MARGIN)
    gated_f = np.sum(P * (av[None] > gate[:, None]), 1)
    gated = W(gated_f)
    report('IMU speed gate (option 1)', gated, ('walking', 'head turns'))
    report('IMU only: head turn rate', W(imu['turn']))
    report('IMU only: head shaking', W(imu['shake']))

    tab = np.load(os.path.join(sd.DATA_DIR, 'z01', 'z_folder', 'z01_test_table.npz'), allow_pickle=True)
    t_tab = tab['t_pi'] - sd.index_pcap(NAME)['ts'][0]
    for grp in ('csi', 'left'):
        for j, nm in enumerate(tab['csi_names']):
            x = np.interp(tc, t_tab, tab[grp][:, j], left=np.nan, right=np.nan)
            report(f'collector {grp}: {nm}', x)

    # ---------- 1. quiet moments ----------
    print('== 1. quiet moments ==')
    act = W(imu['turn'] / np.nanstd(imu['turn']) + imu['shake'] / np.nanstd(imu['shake']))
    for kind in ['walking']:
        s = E.sets[kind]
        for q in (0.5, 0.33, 0.2):
            thr = np.nanquantile(act[s['alone'] | s['appr']], q)
            quiet = act <= thr
            x = np.where(quiet, raw, np.nan)
            xg = np.where(quiet, gated, np.nan)
            print(f'  quietest {int(q * 100)} % of walking windows: alone {np.sum(quiet & s["alone"])}, '
                  f'approach {np.sum(quiet & s["appr"])}, pre-contact {np.sum(quiet & s["pre"])}')
            report(f'  quiet {int(q * 100)} %: motion score', x)
            report(f'  quiet {int(q * 100)} %: gated', xg)
    for frac in (0.5, 0.3):
        # inside each window, average only the frames with the lowest antenna speed
        thr = np.nanquantile(imu['va'][walk_alone_f], frac)
        sel = imu['va'] <= thr
        report(f'slowest {int(frac * 100)} % of frames (va <= {thr:.2f}): score', W(raw_f, sel))

    # ---------- 2. gait template ----------
    print('== 2. gait template ==')
    ph = gait_phase(tf, imu['steps'])
    print(f'  frames inside a step: walking alone {np.mean(~np.isnan(ph[walk_alone_f])) * 100:.0f} %, '
          f'walking approach {np.mean(~np.isnan(ph[(tf >= to_pcap(445)) & (tf < to_pcap(505))])) * 100:.0f} %')
    NB = 12
    pb = np.where(np.isnan(ph), -1, np.minimum((ph * NB).astype(int), NB - 1))
    # is there a step-locked part at all? share of the variance of the score explained by the phase bin,
    # against phase series shifted in time (keeps their structure, breaks the link to the Doppler)
    m = walk_alone_f & (pb >= 0)
    for lab, x in [('motion score', raw_f), ('band 1.2-2', P[:, (av > 1.2) & (av <= 2)].sum(1)),
                   ('band 0.7-1.2', P[:, (av > 0.7) & (av <= 1.2)].sum(1))]:
        def eta2(b):
            mu = np.array([x[m][b == i].mean() if np.any(b == i) else 0 for i in range(NB)])
            return np.var(mu[b]) / np.var(x[m])
        e = eta2(pb[m])
        null = [eta2(np.roll(pb[m], int(s))) for s in rng.integers(200, m.sum() - 200, 200)]
        print(f'  step-locked share of variance, walking alone, {lab:12s} {e * 100:.1f} %  '
              f'(shifted phases 95 %: {np.percentile(null, 95) * 100:.1f} %)')

    def template(train):
        T = np.zeros((NB, len(vel)))
        for i in range(NB):
            T[i] = P[train & (pb == i)].mean(0)
        return T

    A_mid = (A0 + A1) / 2
    folds = [(walk_alone_f & (tf < A_mid)), (walk_alone_f & (tf >= A_mid))]
    resid_f = np.full(len(tf), np.nan)
    resid_pos_f = np.full(len(tf), np.nan)
    for k in range(2):
        T = template(folds[1 - k] & (pb >= 0))
        tgt = folds[k] & (pb >= 0)
        R = P[tgt] - T[pb[tgt]]
        resid_f[tgt] = R[:, av > 0.7].sum(1)
        resid_pos_f[tgt] = np.clip(R[:, av > 0.7], 0, None).sum(1)
    T = template(walk_alone_f & (pb >= 0))
    rest = ~walk_alone_f & (pb >= 0) & ok
    R = P[rest] - T[pb[rest]]
    resid_f[rest] = R[:, av > 0.7].sum(1)
    resid_pos_f[rest] = np.clip(R[:, av > 0.7], 0, None).sum(1)
    report('gait template: score minus template', W(resid_f))
    report('gait template: positive excess only', W(resid_pos_f))
    # quietest phase of the step (lowest template score), only those frames
    q_bins = np.argsort(T[:, av > 0.7].sum(1))[:NB // 3]
    print(f'  quietest step phases: {sorted((q_bins / NB).round(2).tolist())}')
    report('quietest third of each step: score', W(raw_f, np.isin(pb, q_bins)))

    # ---------- 3. self-reference ----------
    print('== 3. self-reference: change against the last L s of the same wearer ==')
    for base_name, base in [('motion score', raw), ('gated', gated),
                            ('band 1.2-2', W(P[:, (av > 1.2) & (av <= 2)].sum(1)))]:
        for L in (2, 4, 8):
            for mode in ('diff', 'ratio'):
                report(f'{base_name}, L {L} s, {mode}', self_reference(tc, base, L, mode=mode))
    # the best of these on the head-turn part too
    report('motion score, L 4 s, ratio', self_reference(tc, raw, 4, mode='ratio'), ('head turns',))

    # ---------- 5. round 2: fair comparison, longer decisions, gate variants, combinations ----------
    print('== 5a. IMU-matched: each window minus walking-alone windows with the most similar head motion ==')
    Z = np.c_[W(imu['turn']), W(imu['shake']), W(imu['va'])]
    Z = (np.log1p(Z) - np.nanmean(np.log1p(Z), 0)) / np.nanstd(np.log1p(Z), 0)
    ref = E.sets['walking']['alone']
    print(f'  check: head motion before contact vs walking alone (AUC, 0.5 = same): '
          f'turn {auc(Z[E.sets["walking"]["pre"], 0], Z[ref, 0]):.2f} shake {auc(Z[E.sets["walking"]["pre"], 1], Z[ref, 1]):.2f} '
          f'speed {auc(Z[E.sets["walking"]["pre"], 2], Z[ref, 2]):.2f}')
    matched = {}
    for lab, x in [('motion score', raw), ('gated', gated), ('band 1.2-2', W(P[:, (av > 1.2) & (av <= 2)].sum(1))),
                   ('band 2-3', W(P[:, (av > 2) & (av <= 3)].sum(1)))]:
        matched[lab] = imu_matched(tc, x, Z, ref)
        report(f'matched: {lab}', matched[lab])

    print('== 5b. longer decision windows (moving mean over 1 s / 2 s) ==')
    for lab, x in [('gated', gated), ('matched gated', matched['gated']), ('motion score', raw)]:
        for n, sec_ in [(4, '1 s'), (8, '2 s')]:
            report(f'{lab}, {sec_}', smooth(x, n))

    print('== 5c. gate variants ==')
    for f_m in (0.5, 1.5, 2.0):
        g = np.maximum(0.7, ss.SELF_FACTOR * imu['va'] + f_m * ss.MARGIN)
        report(f'gate margin x{f_m} ({f_m * ss.MARGIN:.2f} m/s)', W(np.sum(P * (av[None] > g[:, None]), 1)))
    g = np.maximum(0.7, ss.SELF_FACTOR * imu['va'] + ss.MARGIN)
    report('gate, only power above 1.2 m/s too', W(np.sum(P * ((av[None] > g[:, None]) & (av[None] > 1.2)), 1)))
    for (mac, mode, lab), Q in zip(mf.LINKS, LP):
        report(f'gate on link {lab}', W(np.nansum(Q * (av[None] > g[:, None]), 1) + np.where(np.isnan(Q[:, 0]), np.nan, 0)))
    lap = np.nanmean(np.stack(LP[2:]), 0)
    rou = np.nanmean(np.stack(LP[:2]), 0)
    report('gate, laptop links only', W(np.sum(lap * (av[None] > g[:, None]), 1)))
    report('gate, router links only', W(np.sum(rou * (av[None] > g[:, None]), 1)))

    print('== 5d. combinations ==')
    s_w = E.sets['walking']
    for q in (0.5, 0.33):
        thr = np.nanquantile(act[s_w['alone'] | s_w['appr']], q)
        quiet = act <= thr
        report(f'quiet {int(q * 100)} % + matched gated', np.where(quiet, matched['gated'], np.nan))
    sel = imu['va'] <= np.nanquantile(imu['va'][walk_alone_f], 0.5)
    slow_g = W(gated_f, sel)
    report('slowest 50 % frames, gated', slow_g)
    report('slowest 50 % frames, gated, 2 s', smooth(slow_g, 8))
    report('slowest 50 % frames, gated, matched', imu_matched(tc, slow_g, Z, ref))
    report('matched gated, 2 s', smooth(matched['gated'], 8), ('walking',))
    # does the best combination keep working where it is easy (head turns)?
    Zt = Z
    report('check on head turns: gated, 2 s', smooth(gated, 8), ('head turns',))

    # ---------- 4. small learned model ----------
    print('== 4. logistic regression on Doppler bands + IMU (cross-validated) ==')
    feats = [W(P[:, (av > lo) & (av <= hi)].sum(1)) for lo, hi in BANDS]
    feats += [W(pos - neg), W(imu['turn']), W(imu['shake']), W(imu['va'])]
    X = np.log(np.abs(np.c_[tuple(feats)]) + 1e-4) * np.sign(np.c_[tuple(feats)])
    s = E.sets['walking']
    # (a) near vs far inside walking approaches, leave one approach cycle out
    cyc = np.searchsorted(starts, tc)
    use = (s['near'] | s['far']) & ~np.isnan(X).any(1)
    y = s['near'].astype(float)
    out = np.full(len(tc), np.nan)
    for c in np.unique(cyc[use]):
        tr, te = use & (cyc != c), use & (cyc == c)
        mu, sdv = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logreg((X[tr] - mu) / sdv, y[tr])
        out[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sdv] @ w
    print(f'  near vs far, leave-one-approach-out: AUC {auc(out[s["near"]], out[s["far"]]):.2f}')
    # (b) walking with approaches vs alone, 10 s blocks left out (time trends can leak in: read with care)
    use = (s['appr'] | s['alone']) & ~np.isnan(X).any(1)
    y = s['appr'].astype(float)
    blk = np.floor(tc / 10)
    out = np.full(len(tc), np.nan)
    for b in np.unique(blk[use]):
        tr, te = use & (blk != b), use & (blk == b)
        mu, sdv = X[tr].mean(0), X[tr].std(0) + 1e-9
        w = logreg((X[tr] - mu) / sdv, y[tr])
        out[te] = np.c_[np.ones(te.sum()), (X[te] - mu) / sdv] @ w
    print(f'  with approaches vs alone, 10 s blocks out: AUC {auc(out[s["appr"]], out[s["alone"]]):.2f}, '
          f'pre-contact vs alone {auc(out[s["pre"]], out[s["alone"]]):.2f}')

    np.savez(os.path.join(sd.CACHE_DIR, 'z01_walking_results.npz'),
             names=np.array([f'{k[0]} | {k[1]}' for k in results]),
             T1=np.array([r['T1'][0] for r in results.values()]),
             T1pre=np.array([r['T1pre'][0] for r in results.values()]),
             T2=np.array([r['T2'][0] for r in results.values()]),
             T3=np.array([r['T3'][0] / max(r['T3'][1], 1) for r in results.values()]))
