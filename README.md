# 10718-MLIP-Fall26
Repository for 10718 Course Project (GuideMe).

## DTW + CLIP alignment baselines on COIN

The proposal's baselines for tracking which step of an instructional video a user is on: align the user's video to a
reference video of the same task with DTW, using hand-crafted frame features (non-ML baseline) or a pretrained CLIP
encoder (simple ML baseline). Step annotations are used only to score an alignment, never to compute it.

**Method** (`baselines/align_lib.py`)
- Frames sampled at 5 fps. Features: hand-crafted (HSV color histogram + HOG + optical-flow histogram) or CLIP
  ViT-B/32 / ViT-L/14 image embeddings. Each feature is averaged over a 1 s window, centered per video and
  L2-normalized; frames are compared by cosine distance.
- Global DTW (`dtw-python`, symmetric2) maps every query frame to a reference frame. Linear stretch (map by relative
  time, no pixels) is the feature-free floor.
- **Step accuracy**: share of the query's frames inside annotated steps that are mapped to a reference frame of the
  same step. **Steps matched**: share of the query's steps whose frames mostly land in the same reference step.
  95% intervals: bootstrap over videos (each video appears in many pairs).
- **Pairs**: every ordered pair of videos in a step group (same steps in the same order, each step annotated once),
  split by same vs. different YouTube channel.

**Results** (step accuracy, pairs of videos from different YouTube channels)

| Task (step group) | Pairs | CLIP ViT-L/14 + DTW | CLIP ViT-B/32 + DTW | Hand-crafted + DTW | Linear stretch |
|---|---|---|---|---|---|
| ReplaceSIMCard-1 | 2,206 | 51% | 41% | 22% | 21% |
| BoilNoodles-1 | 990 | 24% | 21% | 11% | 13% |
| MakeBurger-1 | 370 | 46% | 48% | 28% | 24% |
| ChangeBikeTires-1 | 124 | 58% | 49% | 52% | 62% |
| ParkParallel-1 | 90 | 40% | 45% | 44% | 50% |
| WashDish-1 | 72 | 23% | 20% | 13% | 7% |
| **Mean** | | **40%** | **37%** | **28%** | **30%** |

## Reproducing

Set up the environment (Python 3.11), then run from the repo root with it activated. Slurm jobs use the activated
environment's `python` and write logs to `logs/`.
```
conda create -n guideme python=3.11 && conda activate guideme && pip install -r requirements.txt
```

1. **Data** (~23 GB in `data/COIN/videos/`). Downloads the official annotations (`COIN.json`) and the videos of the
   11 proposal tasks from Hugging Face re-uploads (YouTube blocks bulk downloads from the cluster), then groups the
   videos by step sequence (`step_groups.csv`).
   ```
   python data/COIN/download_from_hf.py
   python data/COIN/make_step_groups.py
   ```
2. **CLIP features** (GPU; cached in `data/COIN/features/`, reruns skip finished videos):
   ```
   sbatch baselines/run_extract_features.sbatch
   ```
3. **Evaluate** (CPU; hand-crafted features are computed and cached on first use; YouTube channel names are fetched
   once into `data/COIN/youtube_channels.json`). Writes `baselines/results/setup/<group>/` (per-pair and summary CSVs,
   chart) and `baselines/results/setup/all_tasks.csv` / `.png`:
   ```
   sbatch baselines/run_eval_setup.sbatch ReplaceSIMCard-1 BoilNoodles-1 MakeBurger-1 ChangeBikeTires-1 ParkParallel-1 WashDish-1
   ```
4. **Step-level alignment diagram** for one pair (which reference step each query step was aligned to):
   ```
   python baselines/plot_step_mapping.py --query DRKF_C-dA5o --ref aOAnYbHdqsI
   ```
