"""Score alignment methods on every ordered pair of videos in one step group, split by same vs different creator.

Aligns every ordered pair of videos in one step group (default ReplaceSIMCard-1, videos with each step annotated
once) with global DTW, once per method, all on the same pairs. Pairs are split by whether the two videos come from
the same YouTube channel (same creator, usually the same room, camera and phone) or from different channels.

Methods (--methods; linear stretch is always included):
  hand          hand-crafted color + HOG + optical flow, centered per video, 1 s window
  clipB, clipL  frozen CLIP ViT-B/32 or ViT-L/14, centered per video, 1 s window
All settings were fixed before evaluation. Metrics (align_lib.score): step accuracy (frames) and steps matched.
95% intervals come from a bootstrap that resamples videos, since each video is in many pairs.

Outputs in baselines/results/setup/<group>/: pairs.csv, summary.csv (long format: one row per metric, creator group
and method or method difference), setup.png, example_pairs.txt.

Usage (guideme env; features come from the cache in data/COIN/features):
    python baselines/eval_setup.py --group ReplaceSIMCard-1 --methods hand clipB clipL
"""
import argparse
import csv
import json
import os
import time

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')  # one thread per worker; set before numpy loads
import urllib.request
from multiprocessing import Pool

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from align_lib import COIN, HANDCRAFTED, ROOT, align, chance, distance, features, labels, load_db, prepare, score, task_steps

LABELS = {'clipL': 'CLIP ViT-L/14', 'clipB': 'CLIP ViT-B/32', 'hand': 'Hand-crafted',
          'linear': 'Linear stretch (no features)'}
COLORS = {'clipB': '#2a78d6', 'hand': '#eb6834', 'clipL': '#1baf7a', 'linear': '#8a8984'}
ORDER = ['clipL', 'clipB', 'hand', 'linear']
COMPARISONS = [('hand', 'linear'), ('clipB', 'hand'), ('clipL', 'clipB')]
METRICS = {'step_acc': 'Step accuracy (frames in the right step)', 'steps_matched': 'Steps matched (majority of frames)'}

parser = argparse.ArgumentParser()
parser.add_argument('--group', default='ReplaceSIMCard-1')
parser.add_argument('--methods', nargs='+', default=['hand', 'clipB', 'clipL'], choices=[m for m in ORDER if m != 'linear'])
parser.add_argument('--fps', type=float, default=5.0)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--boot', type=int, default=2000)
parser.add_argument('--out', default=os.path.join(ROOT, 'baselines', 'results', 'setup'))
args = parser.parse_args()
methods = [m for m in ORDER if m in args.methods] + ['linear']

db = load_db()
group = [r for r in csv.DictReader(open(os.path.join(COIN, 'step_groups.csv'))) if r['group'] == args.group]
task, n_steps = group[0]['task'], len(group[0]['step_ids'].split('>'))
# each step annotated once; skip videos whose length differs from COIN.json (their timestamps may be shifted)
videos = sorted(r['youtube_id'] for r in group if len(db[r['youtube_id']]['annotation']) == n_steps and not r['note'])
index, names = task_steps(db, task)


def channels(yids):
    """YouTube channel of each video, cached in data/COIN/youtube_channels.json (fetched via oEmbed if missing)."""
    path = os.path.join(COIN, 'youtube_channels.json')
    cache = json.load(open(path)) if os.path.exists(path) else {}
    for y in yids:
        if y not in cache:
            url = f'https://www.youtube.com/oembed?format=json&url=https://www.youtube.com/watch?v={y}'
            try:
                cache[y] = json.load(urllib.request.urlopen(url, timeout=10))['author_name']
            except Exception:
                cache[y] = f'(unknown {y})'  # unique, so it never counts as a same-channel pair
            json.dump(cache, open(path, 'w'), indent=1)
    return {y: cache[y] for y in yids}


chan = channels(videos)
L = {}
P = {}  # method -> video -> prepared features
USE = {'hand': HANDCRAFTED, 'clipB': ['clip'], 'clipL': ['clip']}
extract_workers = max(1, args.workers // 4)
threads = max(1, args.workers // extract_workers)


def extract(yid):
    import cv2
    cv2.setNumThreads(threads)
    try:
        features(task, yid, args.fps, HANDCRAFTED + ['clip'], threads=threads)
        return yid, ''
    except Exception as e:  # one unreadable video shouldn't sink the whole run
        return yid, f'{type(e).__name__}: {e}'


def load_method(m, y):
    """Raw per-frame features of video y for method m ({name: frames x dims})."""
    if m == 'hand':
        return features(task, y, args.fps, HANDCRAFTED)
    return features(task, y, args.fps, ['clip'], clip_model={'clipB': 'ViT-B-32', 'clipL': 'ViT-L-14'}[m])


def evaluate(pair):
    q, r = pair
    row = {'query': q, 'ref': r, 'same_channel': chan[q] == chan[r], 'chance': chance(L[q], L[r])}
    for m in methods:
        if m == 'linear':
            continue
        maps, _ = align(distance(P[m][q], P[m][r], USE[m]))
        for k, v in score(maps['dtw-global'], L[q], L[r]).items():
            row[f'{m}_{k}'] = v
    for k, v in score(maps['linear'], L[q], L[r]).items():  # feature-free baseline
        row[f'linear_{k}'] = v
    return row


def boot(stats, rng):
    """Bootstrap over videos. stats: list of (A, M) with A a videos x videos matrix of per-pair values and M the
    mask of pairs in the group; returns resampled weighted means, one column per statistic."""
    n, out = len(videos), []
    for _ in range(args.boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        W = np.outer(w, w)
        out.append([(W * M * np.nan_to_num(A)).sum() / (W * M).sum() if (W * M).sum() > 0 else np.nan for A, M in stats])
    return np.array(out)


if __name__ == '__main__':
    out = os.path.join(args.out, args.group)
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    with Pool(extract_workers) as pool:
        failed = {y: e for y, e in pool.imap_unordered(extract, videos) if e}
    for y, e in failed.items():
        print(f'  skipping {y}: {e}', flush=True)
    videos = [y for y in videos if y not in failed]
    print(f'features ready in {time.time() - t0:.0f}s', flush=True)
    raw = {m: {y: load_method(m, y) for y in videos} for m in methods if m != 'linear'}

    def length(feats):
        return len(next(iter(feats.values())))

    for y in videos:  # decoders can disagree by a frame; use the common length
        n = min(length(raw[m][y]) for m in raw)
        L[y] = labels(db[y]['annotation'], n, args.fps, index)
        for m in raw:
            raw[m][y] = {k: a[:n] for k, a in raw[m][y].items()}
    for m in raw:
        P[m] = {y: prepare(raw[m][y], USE[m], True, 1, args.fps) for y in videos}
    pairs = [(q, r) for q in videos for r in videos if q != r]
    n_ch = len(set(chan.values()))
    print(f'{task} / {args.group}: {len(videos)} videos from {n_ch} YouTube channels, {len(pairs)} ordered pairs '
          f'({sum(chan[q] == chan[r] for q, r in pairs)} same channel); methods {methods}', flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        rows = pool.map(evaluate, pairs, chunksize=16)
    print(f'aligned in {time.time() - t0:.0f}s', flush=True)
    with open(os.path.join(out, 'pairs.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------ summary
    vi = {y: i for i, y in enumerate(videos)}

    def matrix(key):
        A = np.full((len(videos), len(videos)), np.nan)
        for row in rows:
            A[vi[row['query']], vi[row['ref']]] = row[key]
        return A

    valid = ~np.eye(len(videos), dtype=bool)
    same = np.array([[chan[a] == chan[b] for b in videos] for a in videos]) & valid
    groups = {g: M for g, M in {'Same creator': same, 'Different creators': valid & ~same}.items() if M.any()}
    comparisons = [(a, b) for a, b in COMPARISONS if a in methods and b in methods]
    rng = np.random.default_rng(0)
    summary = []
    for metric in METRICS:
        print(f'\n{METRICS[metric]}')
        A = {m: matrix(f'{m}_{metric}') for m in methods}
        for g, M in groups.items():
            stats = [(A[m], M) for m in methods] + [(A[a] - A[b], M) for a, b in comparisons]
            ci = np.nanpercentile(boot(stats, rng), [2.5, 97.5], axis=0)
            names_ = methods + [f'{a} - {b}' for a, b in comparisons]
            print(f'  {g} ({int(M.sum())} pairs; random mapping {np.nanmean(matrix("chance")[M]):.0%})')
            for i, (nm, (S, _)) in enumerate(zip(names_, stats)):
                mean = np.nanmean(S[M])
                row = {'metric': metric, 'group': g, 'pairs': int(M.sum()), 'method': nm, 'mean': mean,
                       'ci_low': ci[0][i], 'ci_high': ci[1][i], 'better_share': '', 'chance': np.nanmean(matrix('chance')[M])}
                if ' - ' in nm:
                    row['better_share'] = float(np.mean(S[M] > 0))
                summary.append(row)
                extra = f'   first better in {row["better_share"]:.0%} of pairs' if ' - ' in nm else ''
                print(f'    {nm:26} {mean:+7.1%} ({ci[0][i]:+6.1%} to {ci[1][i]:+6.1%}){extra}' if ' - ' in nm else
                      f'    {LABELS[nm]:26} {mean:7.1%} ({ci[0][i]:6.1%} to {ci[1][i]:6.1%})')
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # ------------------------------------------------------------ example pairs for step-mapping diagrams
    if 'clipB' in methods and 'hand' in methods and 'Different creators' in groups:
        D = matrix('clipB_step_acc') - matrix('hand_step_acc')
        picks = []
        for g, M in groups.items():
            idx, d = np.argwhere(M), D[M]
            picks.append((f'{g}: typical (median CLIP - hand difference)', idx[np.argmin(np.abs(d - np.median(d)))]))
        idx, d = np.argwhere(groups['Different creators']), D[groups['Different creators']]
        picks += [('Different creators: biggest CLIP win', idx[np.argmax(d)]),
                  ('Different creators: biggest CLIP loss', idx[np.argmin(d)])]
        with open(os.path.join(out, 'example_pairs.txt'), 'w') as f:
            for why, (i, j) in picks:
                q, r = videos[i], videos[j]
                f.write(f'{q} {r}  # {why}; query {chan[q]}, reference {chan[r]}; CLIP {matrix("clipB_step_acc")[i, j]:.0%}, '
                        f'hand-crafted {matrix("hand_step_acc")[i, j]:.0%}\n')

    # ------------------------------------------------------------ figure
    SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE})
    k = len(methods)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 1.2 + 0.32 * k * len(groups) + 0.6), sharey=True)
    for ax, metric in zip(axes, METRICS):
        for gi, g in enumerate(groups):
            for si, m in enumerate(methods):
                s = next(x for x in summary if x['metric'] == metric and x['group'] == g and x['method'] == m)
                y = gi + (si - (k - 1) / 2) * min(0.8 / k, 0.2)
                ax.plot([s['ci_low'], s['ci_high']], [y, y], color=COLORS[m], lw=2, solid_capstyle='round')
                ax.plot(s['mean'], y, 'o' if m != 'linear' else 'D', ms=6.5, color=COLORS[m], mec=SURFACE, mew=1.2,
                        label=LABELS[m] if gi == 0 else None)
                if m != 'linear':
                    ax.annotate(f'{s["mean"]:.0%}', (s['ci_high'], y), xytext=(4, 0), textcoords='offset points',
                                va='center', fontsize=7.5, color=INK)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.grid(axis='x', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
        ax.tick_params(left=False)
        ax.set_title(METRICS[metric], loc='left', fontsize=10, color=INK)
    counts = {g: int(M.sum()) for g, M in groups.items()}
    axes[0].set_yticks(range(len(groups)), [f'{g}\n({counts[g]:,} pairs)' for g in groups])
    axes[0].set_ylim(len(groups) - 0.5, -0.5)
    axes[0].legend(loc='upper left', bbox_to_anchor=(0, -0.12), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f'{task}: alignment quality by method, same creator vs. different creators', x=0.01, ha='left',
                 fontsize=11, color=INK)
    fig.text(0.01, 0.93, f'Global DTW over all ordered pairs of {len(videos)} videos ({n_ch} YouTube channels). '
             'Dots = mean over pairs, bars = 95% interval (bootstrap over videos).', fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(out, 'setup.png'), dpi=150, bbox_inches='tight')
    print(f'\nsaved pairs.csv, summary.csv, setup.png to {os.path.relpath(out, ROOT)}')
