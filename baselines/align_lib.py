"""Shared pieces of the COIN DTW alignment baselines: frame sampling, features, DTW, and scoring against steps.

Feature types (one vector per frame, sampled at 5 fps):
  color : HSV color histogram (hand-crafted appearance)
  hog   : histogram of oriented gradients (hand-crafted shape / layout)
  flow  : Farneback optical-flow orientation histogram on a 2x2 grid (hand-crafted motion)
  clip  : CLIP image embedding (pretrained encoder; the proposal's simple ML baseline)
Features are cached per video (data/COIN/features/ by default) so every pair can reuse them.

Mapping each query frame to a reference frame:
  dtw-global : global DTW over both whole videos (ends pinned)
  linear     : linear stretch of the query's duration onto the reference's (no pixels used)
"""
import json
import os

import cv2
import numpy as np
from dtw import dtw, symmetric2
from scipy.ndimage import uniform_filter1d

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COIN = os.path.join(ROOT, 'data', 'COIN')
CACHE = os.path.join(COIN, 'features')
SIZE = (160, 120)
HANDCRAFTED = ['color', 'hog', 'flow']


def load_db():
    return json.load(open(os.path.join(COIN, 'COIN.json')))['database']


def task_steps(db, task):
    """Map COIN step ids to 1..K in step-id order (0 = no step), and 1..K to step labels."""
    labels = {a['id']: a['label'] for v in db.values() if v['class'] == task for a in v['annotation']}
    index = {sid: k + 1 for k, sid in enumerate(sorted(labels, key=int))}
    return index, {index[sid]: label for sid, label in labels.items()}


def video_path(task, yid):
    return os.path.join(COIN, 'videos', task, yid + '.mp4')


def read_frames(path, fps, clip=False):
    """Sample frames at fps: small BGR frames for hand-crafted features, 224px-short-side RGB for CLIP.

    Falls back to the standalone ffmpeg binary when OpenCV's build can't decode the video (e.g. AV1)."""
    cap = cv2.VideoCapture(path)
    vfps = cap.get(cv2.CAP_PROP_FPS)
    small, big, idx, next_t = [], [], 0, 0.0
    while cap.grab():
        if idx / vfps >= next_t - 1e-6:
            ok, frame = cap.retrieve()
            if ok:
                small.append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
                if clip:
                    s = 224 / min(frame.shape[:2])
                    big.append(cv2.cvtColor(cv2.resize(frame, (round(frame.shape[1] * s), round(frame.shape[0] * s)),
                                                       interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            next_t += 1 / fps
        idx += 1
    cap.release()
    if len(small) < 2:
        return _read_frames_ffmpeg(path, fps, clip)
    return small, big


def _read_frames_ffmpeg(path, fps, clip):
    import re
    import subprocess
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    info = subprocess.run([ff, '-hide_banner', '-i', path], capture_output=True, text=True).stderr
    w, h = map(int, re.search(r'Video: .*?(\d{2,5})x(\d{2,5})', info).groups())
    s = 224 / min(w, h)
    ow, oh = round(w * s / 2) * 2, round(h * s / 2) * 2
    proc = subprocess.Popen([ff, '-v', 'error', '-i', path, '-vf', f'fps={fps},scale={ow}:{oh}', '-f', 'rawvideo',
                             '-pix_fmt', 'bgr24', '-'], stdout=subprocess.PIPE)
    small, big, size = [], [], ow * oh * 3
    while len(buf := proc.stdout.read(size)) == size:
        frame = np.frombuffer(buf, np.uint8).reshape(oh, ow, 3)
        small.append(cv2.resize(frame, SIZE, interpolation=cv2.INTER_AREA))
        if clip:
            big.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    proc.wait()
    return small, big


def handcrafted(frames):
    from skimage.feature import hog
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    color = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        h = np.concatenate([cv2.calcHist([hsv], [0, 1], None, [16, 4], [0, 180, 0, 256]).ravel(),
                            cv2.calcHist([hsv], [2], None, [8], [0, 256]).ravel()])
        color.append(np.sqrt(h / h.sum()))  # Hellinger
    shape = [hog(g, orientations=9, pixels_per_cell=(20, 20), cells_per_block=(2, 2)) for g in gray]
    flow = []
    rows, cols = np.array_split(np.arange(SIZE[1]), 2), np.array_split(np.arange(SIZE[0]), 2)
    for a, b in zip(gray[:-1], gray[1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(fl[..., 0], fl[..., 1])
        bins = (ang / (2 * np.pi) * 8).astype(int) % 8
        flow.append(np.concatenate([np.bincount(bins[np.ix_(r, c)].ravel(), mag[np.ix_(r, c)].ravel(), minlength=8)
                                    for r in rows for c in cols]) / (len(rows[0]) * len(cols[0])))
    flow.insert(0, flow[0])
    return {'color': np.array(color), 'hog': np.array(shape), 'flow': np.array(flow)}


_encoders = {}


def clip_encoder(model='ViT-B-32', pretrained='openai', threads=8):
    """Return a function mapping RGB frames to L2-normalized CLIP embeddings (model loaded once per process)."""
    if (model, pretrained) not in _encoders:
        import open_clip
        import torch
        from PIL import Image
        torch.set_num_threads(threads)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        net, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        net.eval()
        bs = 256 if device == 'cuda' else 64

        def encode(frames):
            out = []
            with torch.no_grad(), torch.autocast(device, enabled=device == 'cuda'):
                for i in range(0, len(frames), bs):
                    batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i:i + bs]]).to(device)
                    out.append(net.encode_image(batch).float().cpu().numpy())
            x = np.concatenate(out)
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        _encoders[(model, pretrained)] = encode
    return _encoders[(model, pretrained)]


def _save(path, write):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        write(f)
    os.replace(tmp, path)  # atomic, so a preempted job never leaves a half-written cache file


def clip_file(d, yid, model, pretrained):
    return os.path.join(d, f'{yid}.clip-{model}-{pretrained}.npy')


def features(task, yid, fps, kinds, clip_model='ViT-B-32', clip_pretrained='openai', threads=8, also_clip=()):
    """Load a video's features from the cache, computing and caching whichever kinds are missing.

    also_clip lists extra (model, pretrained) CLIP encoders to cache from the same decoded frames; only clip_model is
    returned as 'clip'."""
    d = os.path.join(CACHE, f'fps{fps:g}')
    os.makedirs(d, exist_ok=True)
    hand_file = os.path.join(d, f'{yid}.handcrafted.npz')
    want_hand = bool(set(kinds) & set(HANDCRAFTED))
    clips = ([(clip_model, clip_pretrained)] if 'clip' in kinds else []) + list(also_clip)
    need_hand = want_hand and not os.path.exists(hand_file)
    need_clip = [(m, p) for m, p in clips if not os.path.exists(clip_file(d, yid, m, p))]
    if need_hand or need_clip:
        small, big = read_frames(video_path(task, yid), fps, clip=bool(need_clip))
        if need_hand:
            _save(hand_file, lambda f: np.savez(f, **handcrafted(small)))
        for m, p in need_clip:  # stored as float16 to save space
            _save(clip_file(d, yid, m, p), lambda f: np.save(f, clip_encoder(m, p, threads)(big).astype(np.float16)))
    feats = {}
    if want_hand:
        with np.load(hand_file) as z:
            feats.update({k: z[k] for k in HANDCRAFTED})
    if 'clip' in kinds:
        feats['clip'] = np.load(clip_file(d, yid, clip_model, clip_pretrained)).astype(np.float32)
    return feats


def prepare(feats, use, center, window, fps):
    """Average each feature over a sliding window (seconds; 0 = single frame), center per video, L2-normalize."""
    out = {}
    for n in use:
        x = uniform_filter1d(feats[n].astype(np.float64), size=max(1, round(window * fps)), axis=0, mode='nearest')
        if center:
            x = x - x.mean(0)
        out[n] = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    return out


def distance(q, r, use):
    return np.mean([1 - q[n] @ r[n].T for n in use], axis=0)  # mean cosine distance over feature types


def align(C):
    """Map every query frame (rows of the distance matrix C) to a reference frame (columns)."""
    N, M = C.shape
    g = dtw(C, step_pattern=symmetric2)
    # median reference frame matched to each query frame (the path is monotonic, so each query frame's matches are
    # a contiguous, sorted run of the path)
    s, e = np.searchsorted(g.index1, np.arange(N), 'left'), np.searchsorted(g.index1, np.arange(N), 'right')
    maps = {'dtw-global': (g.index2[(s + e - 1) // 2] + g.index2[(s + e) // 2]) // 2,
            'linear': np.round(np.arange(N) * (M - 1) / (N - 1)).astype(int)}
    return maps, g


def labels(annotation, n, fps, index):
    lab, t = np.zeros(n, int), np.arange(n) / fps
    for a in annotation:
        lab[(t >= a['segment'][0]) & (t < a['segment'][1])] = index[a['id']]
    return lab


def score(m, lq, lr):
    """m maps query frames to reference frames; lq, lr are per-frame step labels (0 = no step).
    step_acc      : share of the query's annotated frames mapped to a reference frame of the same step
    steps_matched : share of the query's steps whose frames mostly land in the same reference step"""
    on = lq > 0
    steps = [k for k in np.unique(lq) if k > 0]
    return {'step_acc': float(np.mean(lr[m][on] == lq[on])),
            'steps_matched': float(np.mean([np.bincount(lr[m][lq == k]).argmax() == k for k in steps]))}


def chance(lq, lr):
    """Expected step accuracy if every annotated query frame were mapped to a random reference frame."""
    on = lq > 0
    return float(sum(np.mean(lq[on] == k) * np.mean(lr == k) for k in set(lq[on])))
