import os, sys, numpy as np, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
for name in ['m_empty', 'm01', 'x02']:
    idx = sd.index_pcap(name)
    mm = np.memmap(f'data/{name}.pcap', dtype=np.uint8, mode='r')
    p = idx['offs'] + 42
    dur = idx['ts'][-1] - idx['ts'][0]
    print(f'==== {name}: {len(p)} pkts, {dur:.1f} s, lens {collections.Counter(idx["lens"]).most_common(4)}')
    magic = mm[p[:, None] + np.arange(2)].view(np.uint16).ravel()
    fctl = mm[p + 3]
    seq = mm[p[:, None] + np.arange(10, 12)].view(np.uint16).ravel()
    core = mm[p[:, None] + np.arange(12, 14)].view(np.uint16).ravel()
    chs = mm[p[:, None] + np.arange(14, 16)].view(np.uint16).ravel()
    chip = mm[p[:, None] + np.arange(16, 18)].view(np.uint16).ravel()
    print(' magic', {hex(k): v for k, v in collections.Counter(magic).items()}, 'core/stream', dict(collections.Counter(core)),
          'chanspec', {hex(k): v for k, v in collections.Counter(chs).items()}, 'chip', {hex(k): v for k, v in collections.Counter(chip).items()})
    for mac, n in collections.Counter(idx['macs']).most_common(6):
        s = idx['macs'] == mac
        t = idx['ts'][s]
        dt = np.diff(t)
        fc = collections.Counter(fctl[s]).most_common(3)
        sq = (seq[s] >> 4).astype(int)
        dsq = np.diff(sq) % 4096
        r = idx['rssi'][s]
        print(f' {mac}: {n} ({n/dur:.0f}/s) fctl {[(hex(a), b) for a, b in fc]} rssi med {np.median(r):.0f} '
              f'[{np.percentile(r,5):.0f},{np.percentile(r,95):.0f}]')
        if n > 1000:
            print(f'    dt ms p10/50/90/99: {np.round(np.percentile(dt, [10,50,90,99])*1e3,2)}  gaps>50ms: {np.sum(dt>0.05)} ({dt[dt>0.05].sum()/dur*100:.1f}% of time)'
                  f'  seq step==1: {np.mean(dsq==1):.3f} step==0(retry/dup): {np.mean(dsq==0):.3f} est. lost frac {1 - n/(np.sum(np.where(dsq<200,dsq,1))+1):.3f}')
