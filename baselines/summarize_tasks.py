"""Combine eval_setup.py results across tasks into one table and chart.

Reads baselines/results/setup/<group>/summary.csv (long format) for each group and reports, per task, each method's
step accuracy on different-creator pairs (the realistic case: a user's setup never matches the instructor's) and the
paired differences between methods, with 95% intervals.

Usage:
    python baselines/summarize_tasks.py --groups ReplaceSIMCard-1 MakeBurger-1 BoilNoodles-1 ...
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_lib import ROOT

LABELS = {'clipL': 'CLIP ViT-L/14', 'clipB': 'CLIP ViT-B/32', 'hand': 'Hand-crafted',
          'linear': 'Linear stretch (no features)'}
COLORS = {'clipB': '#2a78d6', 'hand': '#eb6834', 'clipL': '#1baf7a', 'linear': '#8a8984'}
ORDER = ['clipL', 'clipB', 'hand', 'linear']
SETUP = 'Different creators'

parser = argparse.ArgumentParser()
parser.add_argument('--groups', nargs='+', required=True)
parser.add_argument('--results', default=os.path.join(ROOT, 'baselines', 'results', 'setup'))
args = parser.parse_args()

rows = {}
for g in args.groups:
    path = os.path.join(args.results, g, 'summary.csv')
    if not os.path.exists(path):
        print(f'missing {path}, skipping')
        continue
    for r in csv.DictReader(open(path)):
        rows[(g, r['metric'], r['group'], r['method'])] = r
groups = [g for g in args.groups if any(k[0] == g and k[2] == SETUP for k in rows)]
methods = [m for m in ORDER if any(k[3] == m for k in rows)]
diffs = sorted({k[3] for k in rows if ' - ' in k[3]}, key=lambda d: [ORDER.index(x) for x in d.split(' - ')])
f = lambda r, k: float(r[k])

with open(os.path.join(args.results, 'all_tasks.csv'), 'w', newline='') as out:
    writer = csv.writer(out)
    writer.writerow(['group', 'metric', 'setup', 'pairs', 'method', 'mean', 'ci_low', 'ci_high', 'better_share', 'chance'])
    for (g, metric, setup, m), r in rows.items():
        if g in groups:
            writer.writerow([g, metric, setup, r['pairs'], m, r['mean'], r['ci_low'], r['ci_high'], r['better_share'], r['chance']])

for metric, title in (('step_acc', 'Step accuracy'), ('steps_matched', 'Steps matched')):
    print(f'\n{title}, different-creator pairs')
    print(f'{"task":20} {"pairs":>6} ' + ' '.join(f'{m:>13}' for m in methods) + f' {"random":>7}')
    for g in groups:
        r0 = rows[(g, metric, SETUP, methods[0])]
        print(f'{g:20} {int(r0["pairs"]):6} ' + ' '.join(f'{f(rows[(g, metric, SETUP, m)], "mean"):13.0%}' for m in methods) +
              f' {f(r0, "chance"):7.0%}')
    means = {m: np.mean([f(rows[(g, metric, SETUP, m)], 'mean') for g in groups]) for m in methods}
    print(f'{"mean over tasks":20} {"":6} ' + ' '.join(f'{means[m]:13.0%}' for m in methods))
    print('  paired differences (95% CI); tasks where the interval is above zero:')
    for d in diffs:
        cells = []
        for g in groups:
            r = rows.get((g, metric, SETUP, d))
            if r:
                cells.append((g, f(r, 'mean'), f(r, 'ci_low'), f(r, 'ci_high')))
        above = sum(lo > 0 for _, _, lo, _ in cells)
        below = sum(hi < 0 for _, _, _, hi in cells)
        print(f'    {d:26} ' + ', '.join(f'{g.rsplit("-", 1)[0]} {m:+.0%} ({lo:+.0%} to {hi:+.0%})' for g, m, lo, hi in cells) +
              f'  -> clearly better in {above}/{len(cells)}, clearly worse in {below}/{len(cells)}')

# ------------------------------------------------------------ figure
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
k = len(methods)
height = 1.6 + len(groups) * (0.25 * k + 0.35)
fig, axes = plt.subplots(1, 2, figsize=(11, height), sharey=True)
for ax, (metric, title) in zip(axes, (('step_acc', 'Step accuracy (frames in the right step)'),
                                      ('steps_matched', 'Steps matched (majority of frames)'))):
    for gi, g in enumerate(groups):
        for si, m in enumerate(methods):
            r = rows[(g, metric, SETUP, m)]
            y = gi + (si - (k - 1) / 2) * min(0.8 / k, 0.2)
            ax.plot([f(r, 'ci_low'), f(r, 'ci_high')], [y, y], color=COLORS[m], lw=2, solid_capstyle='round')
            ax.plot(f(r, 'mean'), y, 'o' if m != 'linear' else 'D', ms=6, color=COLORS[m], mec=SURFACE, mew=1.2,
                    label=LABELS[m] if gi == 0 else None)
            if m != 'linear':
                ax.annotate(f'{f(r, "mean"):.0%}', (f(r, 'ci_high'), y), xytext=(4, 0), textcoords='offset points',
                            va='center', fontsize=7, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    ax.grid(axis='x', color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.tick_params(left=False)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
labels = [f'{g.rsplit("-", 1)[0]}\n{int(rows[(g, "step_acc", SETUP, methods[0])]["pairs"]):,} pairs' for g in groups]
axes[0].set_yticks(range(len(groups)), labels)
axes[0].set_ylim(len(groups) - 0.5, -0.5)
axes[0].legend(loc='upper left', bbox_to_anchor=(0, -0.35 / height * 3), ncol=3, frameon=False, fontsize=8.5)
fig.suptitle('Aligning videos from different creators, per task', x=0.01, ha='left', fontsize=11, color=INK)
fig.text(0.01, 1 - 0.5 / height, 'Global DTW over all ordered pairs of videos from different YouTube channels. '
         'Dots = mean over pairs, bars = 95% interval (bootstrap over videos).', fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 1 - 0.7 / height))
fig.savefig(os.path.join(args.results, 'all_tasks.png'), dpi=150, bbox_inches='tight')
print(f'\nsaved all_tasks.csv, all_tasks.png to {os.path.relpath(args.results, ROOT)}')
