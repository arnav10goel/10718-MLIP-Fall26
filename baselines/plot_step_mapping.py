"""Step-level view of an alignment: which annotated segments of the reference map to which segments of the query.

For one pair, each panel draws the reference timeline (top) and the query timeline (bottom), split into their
annotated steps, and joins them with ribbons that follow the alignment route. A ribbon takes the step's color
when both ends are the same step (correct), light gray when both ends are unannotated, and is hatched when the
ends disagree. Panels: the annotation itself (what a perfect alignment would join), CLIP and hand-crafted DTW
(global). Also prints, for each query step, where its frames landed.

Usage (guideme env; features come from the cache in data/COIN/features):
    python baselines/plot_step_mapping.py --ref jwFQu_beN_U --query CG8Xh4bRw5Q
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Polygon, Rectangle

from align_lib import HANDCRAFTED, ROOT, align, distance, features, labels, load_db, prepare, score, task_steps

parser = argparse.ArgumentParser()
parser.add_argument('--ref', default='jwFQu_beN_U')
parser.add_argument('--query', default='CG8Xh4bRw5Q')
parser.add_argument('--fps', type=float, default=5.0)
parser.add_argument('--note', default='', help='describes the pair, shown in the title')
parser.add_argument('--out', default=os.path.join(ROOT, 'baselines', 'results', 'step_mapping'))
args = parser.parse_args()

STEP_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BACKGROUND, SURFACE, INK, INK2, MUTED = '#dcdbd6', '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984'
fps = args.fps

db = load_db()
task = db[args.ref]['class']
index, names = task_steps(db, task)
fq, fr = (features(task, yid, fps, HANDCRAFTED + ['clip']) for yid in (args.query, args.ref))
lq = labels(db[args.query]['annotation'], len(fq['clip']), fps, index)
lr = labels(db[args.ref]['annotation'], len(fr['clip']), fps, index)
ks = sorted(set(lq[lq > 0]) | set(lr[lr > 0]))


def run(use):
    maps, g = align(distance(prepare(fq, use, True, 1, fps), prepare(fr, use, True, 1, fps), use))
    return maps, g


clip_maps, clip_g = run(['clip'])
hand_maps, hand_g = run(HANDCRAFTED)
acc = lambda m: score(m, lq, lr)['step_acc']
panels = [
    ('Annotation: what a perfect alignment joins', None, None),
    (f'CLIP + DTW (global): {acc(clip_maps["dtw-global"]):.0%} of step frames correct',
     list(zip(clip_g.index1, clip_g.index2)), clip_maps['dtw-global']),
    (f'Hand-crafted + DTW (global): {acc(hand_maps["dtw-global"]):.0%} of step frames correct',
     list(zip(hand_g.index1, hand_g.index2)), hand_maps['dtw-global']),
]


def step_color(k):
    return BACKGROUND if k == 0 else STEP_COLORS[(k - 1) % len(STEP_COLORS)]


def runs(lab):
    edges = np.flatnonzero(np.diff(lab)) + 1
    return [(s, e, lab[s]) for s, e in zip(np.r_[0, edges], np.r_[edges, len(lab)])]


def ribbons(pairs):
    """Group consecutive route pairs into ribbons with constant (query label, reference label) and no jumps."""
    out = []
    for i, j in pairs:
        r = out[-1] if out else None
        if r and (lq[i], lr[j]) == (r['kq'], r['kr']) and i - r['i1'] <= 1 and abs(j - r['jl']) <= fps:
            r.update(i1=i, j0=min(r['j0'], j), j1=max(r['j1'], j), jl=j)
        else:
            out.append(dict(i0=i, i1=i, j0=j, j1=j, jl=j, kq=lq[i], kr=lr[j]))
    return out


def band(ax, q0, q1, r0, r1, **style):
    t = np.linspace(0, 1, 40)
    s = t * t * (3 - 2 * t)  # smoothstep, so ribbons leave and enter the bars vertically
    left = np.c_[q0 + (r0 - q0) * s, t]
    right = np.c_[q1 + (r1 - q1) * s, t][::-1]
    ax.add_patch(Polygon(np.r_[left, right], closed=True, lw=0, **style))


def bar(ax, lab, y, h):
    for s, e, k in runs(lab):
        ax.add_patch(Rectangle((s / fps, y), (e - s) / fps, h, color=step_color(k), lw=0))
        if k and (e - s) / fps >= 3:
            ax.text((s + e) / 2 / fps, y + h / 2, str(k), ha='center', va='center', fontsize=8, color=SURFACE, weight='bold')


plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                     'savefig.facecolor': SURFACE, 'hatch.color': MUTED, 'hatch.linewidth': 0.8})
T = max(len(lq), len(lr)) / fps
fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.35 * len(panels)), sharex=True)
for ax, (title, pairs, m) in zip(axes, panels):
    H = 0.16
    bar(ax, lr, 1, H)
    bar(ax, lq, -H, H)
    if pairs is None:
        for k in ks:
            qi, ri = np.flatnonzero(lq == k), np.flatnonzero(lr == k)
            if len(qi) and len(ri):
                band(ax, qi[0] / fps, (qi[-1] + 1) / fps, ri[0] / fps, (ri[-1] + 1) / fps, color=step_color(k), alpha=0.55)
    else:
        for r in ribbons(pairs):
            q0, q1, r0, r1 = r['i0'] / fps, (r['i1'] + 1) / fps, r['j0'] / fps, (r['j1'] + 1) / fps
            if r['kq'] == r['kr'] == 0:
                band(ax, q0, q1, r0, r1, color=BACKGROUND, alpha=0.45)
            elif r['kq'] == r['kr']:
                band(ax, q0, q1, r0, r1, color=step_color(r['kq']), alpha=0.6)
            else:
                band(ax, q0, q1, r0, r1, facecolor='none', hatch='////', edgecolor=MUTED, alpha=0.9)
        for k in ks:  # per-step accuracy under each query step
            qi = np.flatnonzero(lq == k)
            if len(qi):
                ax.text((qi[0] + qi[-1] + 1) / 2 / fps, -H - 0.05, f'{np.mean(lr[m[qi]] == k):.0%}', ha='center', va='top',
                        fontsize=7.5, color=INK2)
    ax.text(-0.01, 1 + H / 2, f'Reference\n{args.ref}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8, color=INK2)
    ax.text(-0.01, -H / 2, f'Query\n{args.query}', transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8, color=INK2)
    ax.set_ylim(-H - 0.3, 1 + H + 0.05)
    ax.set_xlim(0, T)
    ax.set_yticks([])
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, loc='left', fontsize=10, color=INK)
axes[-1].set_xlabel('Time in each video (s)')
legend = [Patch(color=step_color(k), label=f'{k}  {names[k]}') for k in ks] + [
    Patch(color=BACKGROUND, label='no step annotated'),
    Patch(facecolor='none', edgecolor=MUTED, hatch='////', label='ribbon joins different steps (wrong)')]
fig.legend(handles=legend, loc='lower left', bbox_to_anchor=(0.08, -0.06), ncol=3, frameon=False, fontsize=8)
fig.suptitle(f'{task}: ' + (args.note or 'which part of the reference each part of the query was aligned to'), x=0.08, ha='left', fontsize=12, color=INK, y=0.995)
fig.text(0.08, 0.957, 'Colored ribbon = both ends in the same step (correct). % under each query step = its frames '
         'mapped to the same step.', fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0.06, 0, 1, 0.935))
os.makedirs(args.out, exist_ok=True)
path = os.path.join(args.out, f'{args.query}_vs_{args.ref}.png')
fig.savefig(path, dpi=150, bbox_inches='tight')

print(f'{task}: query {args.query} ({db[args.query]["subset"]}), reference {args.ref} ({db[args.ref]["subset"]})')
print('where each query step\'s frames landed in the reference (share of that step\'s frames):')
for title, pairs, m in panels[1:]:
    print('  ' + title.split(':')[0])
    for k in ks:
        qi = np.flatnonzero(lq == k)
        if len(qi):
            dest = {j: np.mean(lr[m[qi]] == j) for j in [0] + ks}
            print(f'    query step {k} -> ' + ', '.join(f'{"no step" if j == 0 else "step " + str(j)} {v:.0%}' for j, v in dest.items() if v > 0))
print(f'saved {os.path.relpath(path, ROOT)}')
