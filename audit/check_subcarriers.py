import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sharp_doppler as sd
name = 'm01'
idx = sd.index_pcap(name)
mm = np.memmap(f'data/{name}.pcap', dtype=np.uint8, mode='r')
for mac in [sd.ROUTER, sd.LAPTOP]:
    sel = np.where((idx['macs'] == mac) & np.isin(idx['lens'], [1084, 1088]))[0][::50][:4000]
    o = idx['offs'][sel] + 60
    raw = mm[o[:, None] + np.arange(1024)].view(np.int16).reshape(-1, 512).astype(np.float32)
    c = np.fft.fftshift(raw[:, ::2] + 1j * raw[:, 1::2], axes=1)
    pw = 10 * np.log10(np.mean(np.abs(c) ** 2, 0) + 1e-9)
    ref = np.median(pw[sd.ACT])
    k20 = np.arange(256) - 128 + 96   # 20 MHz subcarrier index around ch36 centre
    print(mac, 'median |H|^2 at k20=±1..26 (dB rel):  0')
    for k in [-29, -28, -27, -26, 0, 26, 27, 28, 29]:
        i = k - 96 + 128
        print(f'   k20={k:+d}: {pw[i]-ref:6.1f} dB')
    for b, lo in enumerate([0, 64, 128, 192]):
        print(f'   80MHz sub-band {b} (20 MHz chunk): {np.median(pw[lo+6:lo+58])-ref:6.1f} dB')
    # per-frame: is the 2nd/3rd/4th sub-band ever active?
    sb = np.array([np.mean(np.abs(c[:, lo+6:lo+58])**2, 1) for lo in [0, 64, 128, 192]])
    act = sb / sb[0]  # sub-band 0 (lowest, fftshifted) is primary ch36
    print('   frames with any other sub-band within 10 dB of primary:', np.mean((act[[1,2,3]] > 0.1).any(0)).round(4))
