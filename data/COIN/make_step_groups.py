"""Group the downloaded COIN videos of each task by their ordered step sequence -> step_groups.csv.

Two videos are in the same group when they go through the same steps in the same order (a step annotated as several
back-to-back segments counts once). Groups with at least two videos are kept, numbered per task by size (largest
first). Videos whose downloaded length differs from COIN.json (> max(2 s, 2%)) are flagged in the 'note' column,
since their step timestamps may be shifted; the evaluation scripts skip them.

Usage (after download_from_hf.py):
    python data/COIN/make_step_groups.py
"""
import collections
import csv
import itertools
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor

import imageio_ffmpeg

TASKS = ['ShaveBeard', 'UseJack', 'ChangeBikeTires', 'ParkParallel', 'ReplaceSIMCard', 'CleanToilet', 'MakeCandle',
         'MakeHomemadeIceCream', 'MakeBurger', 'BoilNoodles', 'WashDish']
here = os.path.dirname(os.path.abspath(__file__))
db = json.load(open(os.path.join(here, 'COIN.json')))['database']
have = {r['youtube_id']: r['path'] for r in csv.DictReader(open(os.path.join(here, 'manifest.csv')))
        if r['status'] in ('ok', 'exists')}
ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()


def length_mismatch(yid):
    info = subprocess.run([ffmpeg, '-hide_banner', '-i', os.path.join(here, have[yid])], capture_output=True, text=True).stderr
    m = re.search(r'Duration: (\d+):(\d+):([\d.]+)', info)
    expected = db[yid]['duration']
    return m is None or abs(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) - expected) > max(2.0, 0.02 * expected)


def sequence(v):
    segs = sorted(v['annotation'], key=lambda a: a['segment'][0])
    return [i for i, _ in itertools.groupby(a['id'] for a in segs)], segs


with ThreadPoolExecutor(8) as pool:
    mismatch = {y for y, bad in zip(have, pool.map(length_mismatch, have)) if bad}

rows = []
for t in TASKS:
    vs = {k: v for k, v in db.items() if v['class'] == t and k in have}
    labels = {a['id']: a['label'] for v in vs.values() for a in v['annotation']}
    groups = collections.defaultdict(list)
    for k, v in vs.items():
        groups[tuple(sequence(v)[0])].append(k)
    ranked = sorted((g for g in groups.items() if len(g[1]) >= 2), key=lambda g: (-len(g[1]), -len(g[0])))
    for gi, (s, ks) in enumerate(ranked, 1):
        for k in sorted(ks, key=lambda k: (db[k]['subset'] != 'training', k)):
            _, segs = sequence(db[k])
            rows.append({'task': t, 'group': f'{t}-{gi}', 'n_videos': len(ks), 'step_ids': '>'.join(s),
                         'steps': ' > '.join(labels[i] for i in s), 'youtube_id': k, 'subset': db[k]['subset'],
                         'duration': round(db[k]['duration'], 2), 'span_start': segs[0]['segment'][0],
                         'span_end': segs[-1]['segment'][1], 'path': f'videos/{t}/{k}.mp4',
                         'note': 'length differs from COIN.json' if k in mismatch else ''})

with open(os.path.join(here, 'step_groups.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f'{len(rows)} rows, {len({r["group"] for r in rows})} groups; {len(mismatch)} videos with mismatched length: '
      f'{sorted(mismatch)}')
