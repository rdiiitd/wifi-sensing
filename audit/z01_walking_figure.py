"""Summary figure of the walking experiments (13_z01_walking_summary.png).
Values are copied from sharp_output/cache/z01_walking.log and z01_walking_resolution.log (AUC and 5 s block
bootstrap 95 % interval); rerun those scripts and update the table if anything changes."""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import make_figures as mf

BLUE, GREY, INK, MUTED = '#2a78d6', '#8a8984', '#0b0b0b', '#52514e'

# name, T1 (approaches vs alone) AUC, lo, hi, T2 (near vs far) AUC, lo, hi
ROWS = [
    ('Motion score (current pipeline)', 0.52, 0.42, 0.60, 0.64, 0.50, 0.75),
    ('IMU speed gate (2x antenna speed)', 0.63, 0.54, 0.71, 0.63, 0.48, 0.75),
    ('IMU speed gate, fair comparison*', 0.58, 0.48, 0.67, 0.65, 0.51, 0.77),
    ('Gate, 500 ms window (sharper speed)', 0.64, 0.52, 0.72, 0.64, 0.48, 0.77),
    ('Gate, 500 ms window, 1 s decision', 0.65, 0.53, 0.75, 0.68, 0.50, 0.86),
    ('Quiet third of moments + gate', 0.65, 0.45, 0.82, 0.70, 0.47, 0.85),
    ('Quiet third + gate, fair comparison*', 0.63, 0.45, 0.79, 0.72, 0.48, 0.88),
    ('Step-pattern subtraction', 0.53, 0.45, 0.62, 0.68, 0.52, 0.82),
    ('Change vs own last 4 s', 0.48, 0.40, 0.56, 0.66, 0.47, 0.80),
    ("Collector's leftover, 30-80 Hz", 0.63, 0.53, 0.72, 0.41, 0.26, 0.56),
]
REF = ('Same motion score, head-turn part', 0.92, 0.89, 0.95, 0.65, 0.56, 0.72)

fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
rows = ROWS + [REF]
y = list(range(len(rows)))[::-1]
for ax, (k, title) in zip(axs, [(1, 'T1: is someone moving around me while I walk?\n(walking with approaches vs walking alone)'),
                                 (4, 'T2: is that person near (< 0.9 m) or far (> 1.2 m)?\n(inside the walking approaches)')]):
    ax.axvline(0.5, color=GREY, lw=1, ls=(0, (3, 3)))
    ax.text(0.505, -0.75, 'chance', color=MUTED, fontsize=8, ha='left')
    for yi, r in zip(y, rows):
        c = GREY if r is REF else BLUE
        ax.plot([r[k + 1], r[k + 2]], [yi, yi], color=c, lw=2, solid_capstyle='round')
        ax.plot(r[k], yi, 'o', ms=8, color=c, markeredgecolor='white', markeredgewidth=2)
        ax.text(r[k + 2] + 0.012, yi, f'{r[k]:.2f}', va='center', fontsize=8, color=INK)
    ax.axhline(0.5, color=GREY, lw=0.6)
    ax.set_xlim(0.2, 1.02)
    ax.set_title(title, fontsize=10, loc='left', color=INK)
    ax.set_xlabel('AUC (dot) with 95 % interval (line); 1.0 = perfect', fontsize=9, color=MUTED)
    ax.grid(axis='x', color='#e6e5e1', lw=0.6)
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
axs[0].set_yticks(y)
axs[0].set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
fig.suptitle('z01, wearer walking: every method tried (9 approaches, one session)', x=0.01, ha='left',
             fontsize=12, color=INK)
fig.text(0.01, 0.01, '* fair comparison: each moment is compared with walking-alone moments that had similar head '
         'motion, because the wearer walks differently just before a contact. Grey: the head-turn part, for scale.',
         fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
mf.save(fig, '13_z01_walking_summary.png')
