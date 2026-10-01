"""Extract and cache CLIP features at 5 fps for every downloaded COIN video, on a GPU if available.

Each video is decoded once and encoded with every requested CLIP model; files that already exist are skipped, so a
preempted job resumes. Several worker processes share the GPU so that video decoding (CPU) keeps it busy.
Features are cached in data/COIN/features/fps5/.

Usage (see run_extract_features.sbatch):
    python baselines/extract_features.py --models ViT-B-32:openai ViT-L-14:openai --workers 6
"""
import argparse
import csv
import os
import time
from multiprocessing import get_context

from align_lib import COIN, features, load_db

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', default=['ViT-B-32:openai', 'ViT-L-14:openai'], help='model:pretrained')
parser.add_argument('--fps', type=float, default=5.0)
parser.add_argument('--workers', type=int, default=6)
args = parser.parse_args()
models = [tuple(m.split(':', 1)) for m in args.models]


def work(job):
    import cv2
    cv2.setNumThreads(2)
    task, yid = job
    t = time.time()
    try:
        features(task, yid, args.fps, [], threads=2, also_clip=models)
        return yid, time.time() - t, ''
    except Exception as e:  # one unreadable video shouldn't sink the whole run
        return yid, time.time() - t, f'{type(e).__name__}: {e}'


if __name__ == '__main__':
    db = load_db()
    todo = [(db[r['youtube_id']]['class'], r['youtube_id']) for r in csv.DictReader(open(os.path.join(COIN, 'manifest.csv')))
            if r['status'] in ('ok', 'exists')]
    print(f'{len(todo)} COIN videos, models {models}, {args.workers} workers', flush=True)
    t0, failed = time.time(), []
    with get_context('spawn').Pool(args.workers) as pool:  # spawn: each worker gets its own CUDA context
        for i, (yid, dt, err) in enumerate(pool.imap_unordered(work, todo), 1):
            if err:
                failed.append((yid, err))
            if err or i % 25 == 0 or i == len(todo):
                print(f'  [{i}/{len(todo)}] {yid} {dt:.0f}s {err} (elapsed {time.time() - t0:.0f}s)', flush=True)
    print(f'done in {time.time() - t0:.0f}s; {len(failed)} failed: {failed[:10]}', flush=True)
