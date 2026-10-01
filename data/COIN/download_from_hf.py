"""Fetch COIN videos for the GuideMe proposal tasks from community re-uploads on the Hugging Face Hub.

YouTube blocks bulk downloads from cluster IPs and many COIN links are dead, so this pulls the same
full-length videos from public HF dataset repos instead: ttyue/COIN_Dataset first, then the pengxiang
repos for anything it lacks. These are unofficial re-uploads with no stated license.

Fetches the official COIN annotations (COIN.json) if missing, then writes videos/<Task>/<youtube_id>.mp4 and
manifest.csv listing every task video, where it came from, and whether we have it.

Usage:
    python download_from_hf.py
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('HF_XET_CHUNK_CACHE_SIZE_BYTES', '0')  # don't duplicate ~24 GB into ~/.cache
from huggingface_hub import HfApi, hf_hub_download

TASKS = [
    'ShaveBeard', 'ChangeBikeTires', 'ParkParallel', 'UseJack', 'ReplaceSIMCard', 'CleanToilet',
    'MakeCandle', 'MakeHomemadeIceCream', 'MakeBurger', 'BoilNoodles', 'WashDish',
]
# (repo, prefix) in priority order; each repo stores <prefix><recipe_type>/<youtube_id>.<ext>
SOURCES = [('ttyue/COIN_Dataset', 'videos/'), ('pengxiang/coins_new', ''), ('pengxiang/COINs', '')]

here = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('--json', default=os.path.join(here, 'COIN.json'))
parser.add_argument('--out', default=os.path.join(here, 'videos'))
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--tasks', nargs='+', default=TASKS)
args = parser.parse_args()

if not os.path.exists(args.json):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/coin-dataset/annotations/master/COIN.json', args.json)
data = json.load(open(args.json, 'r'))['database']
wanted = {yid: info for yid, info in data.items() if info['class'] in args.tasks}
staging = os.path.join(here, '.hf_staging')
try:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    ffmpeg = 'ffmpeg'

# Pick one source file per wanted video, preferring earlier repos.
api = HfApi()
source = {}
for repo, prefix in SOURCES:
    for rt in sorted({info['recipe_type'] for info in wanted.values()}):
        for f in api.list_repo_tree(repo, path_in_repo=f'{prefix}{rt}', repo_type='dataset'):
            name = f.path.rsplit('/', 1)[-1]
            yid = name[:11]  # YouTube ids are 11 chars; files are <id>.mp4, <id>.mp4.mkv, ...
            if yid in wanted and yid not in source and not name.endswith(('.part', '.ytdl')):
                source[yid] = (repo, f.path, f.size)

lock = threading.Lock()
done = [0]


def fetch(yid, info):
    dst = os.path.join(args.out, info['class'], yid + '.mp4')
    repo, remote, size = source.get(yid, ('', '', 0))
    row = {'youtube_id': yid, 'class': info['class'], 'recipe_type': info['recipe_type'],
           'subset': info['subset'], 'duration': info['duration'], 'path': os.path.relpath(dst, here),
           'status': 'ok', 'source': repo, 'error': ''}
    if os.path.exists(dst):
        row['status'] = 'exists'
    elif not repo:
        row['status'] = 'missing'
        row['path'] = ''
    else:
        try:
            local = hf_hub_download(repo, remote, repo_type='dataset', local_dir=os.path.join(staging, repo))
            if os.path.getsize(local) != size:
                raise IOError(f'size mismatch: got {os.path.getsize(local)}, expected {size}')
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if remote.endswith('.mp4'):
                os.replace(local, dst)
            else:  # .mkv / .webm: remux to mp4 without re-encoding
                subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', local, '-c', 'copy', dst + '.tmp.mp4'], check=True)
                os.replace(dst + '.tmp.mp4', dst)
                os.remove(local)
        except Exception as e:
            row['status'] = 'failed'
            row['error'] = str(e).splitlines()[0][:300]
    with lock:
        done[0] += 1
        print(f"[{done[0]}/{len(wanted)}] {row['status']:7} {info['class']:22} {yid} {row['source']} {row['error'][:80]}", flush=True)
    return row


with ThreadPoolExecutor(max_workers=args.workers) as pool:
    rows = list(pool.map(lambda item: fetch(*item), wanted.items()))
shutil.rmtree(staging, ignore_errors=True)

with open(os.path.join(here, 'manifest.csv'), 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print('\nclass                  have  missing  failed')
for task in args.tasks:
    rs = [r for r in rows if r['class'] == task]
    have = sum(r['status'] in ('ok', 'exists') for r in rs)
    print(f"{task:22} {have:4}/{len(rs):<3} {sum(r['status'] == 'missing' for r in rs):6} {sum(r['status'] == 'failed' for r in rs):7}")
print(f"TOTAL {sum(r['status'] in ('ok', 'exists') for r in rows)}/{len(rows)}")
