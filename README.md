# Human-Safety-Aware Trajectory Planning for UAVs

Master thesis project on **predicting human trajectories** for **UAV safety**, with reliable ground-truth acquisition via **Visual-Inertial Odometry (VIO)**.

> Built on top of [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) (Salzmann et al., ECCV 2020). The original upstream README is preserved at [`TRAJECTRON_UPSTREAM_README.md`](./TRAJECTRON_UPSTREAM_README.md).

---

## Goal

A UAV that flies near humans needs to **predict where people will be in 1–3 seconds** so it can plan a safe path. This project builds an end-to-end pipeline:

1. **Perception** — detect humans in the UAV camera feed
2. **Ground truth** — convert pixel detections to metric world coordinates (VIO)
3. **Forecasting** — predict future trajectories (Trajectron++ fine-tuned on collected data)
4. **Hand-gesture intent** (separate module) — interpret pedestrian intent from hand motion
5. **Trajectory ranking** — pick the safest / most likely prediction for the UAV's planner

---

## Current architecture

```
┌────────────────────┐
│   UAV camera       │
│   (or iPhone for   │
│    data collection)│
└─────────┬──────────┘
          │ video stream
          ▼
┌────────────────────┐     ┌──────────────────┐
│   YOLOv8           │     │   ARKit (VIO)    │
│   person detection │     │   camera 6-DoF   │
│   → foot pixels    │     │   pose per frame │
└─────────┬──────────┘     └─────────┬────────┘
          │                          │
          └─────────┬────────────────┘
                    ▼
          ┌──────────────────┐
          │  Ray-casting     │
          │  pixel + pose →  │
          │  world (x,y,z)   │
          └─────────┬────────┘
                    │ metric trajectory
                    ▼
          ┌──────────────────┐
          │  Trajectron++    │
          │  (fine-tuned)    │
          └─────────┬────────┘
                    │ GMM + 20 samples
                    ▼
          ┌──────────────────┐
          │  Ranking module  │
          │  (likelihood,    │
          │   risk, …)       │
          └──────────────────┘
```

A separate **hand-gesture module** (MediaPipe Hands + classifier) runs in parallel and can supply intent priors to the trajectory predictor.

---

## Status

| Component | Status | Notes |
|---|---|---|
| YOLOv8 person detection | done | `experiments/pedestrians/` and `~/Desktop/uav_perception/` |
| Custom CSV → Trajectron++ `.pkl` converter | done | `experiments/pedestrians/csv_to_trajectron_data.py` |
| ETH base pre-training (100 epochs) | done | `models_08_Mar_2026_23_53_32_eth_100ep` |
| UAV fine-tuning (multiple variants v1–v8) | done | latest: `models_10_Mar_2026_00_42_50_uav_ft_v8_persp` |
| Trajectory ranking (likelihood, KDE, mean-prox, endpoint, history-consistency, risk) | done | `prediction_ranking*.ipynb` |
| Homography-based ground-truth calibration | done as **proof-of-concept** | `calibration_demo.ipynb` (lives in `uav_perception/`) |
| **VIO-based ground-truth (ARKit / CamTrackAR)** | in progress | see [`docs/VIO_RESEARCH.md`](docs/VIO_RESEARCH.md) |
| Hand-gesture module | planned | separate module, MediaPipe → classifier |
| UAV deployment | future | not in scope of this thesis |

For a detailed breakdown of what was done when and why, see [`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## Key documents

- [`docs/NOTEBOOKS_GUIDE.md`](docs/NOTEBOOKS_GUIDE.md) — **Start here.** Short narrative for each experiment notebook: question → method → result → why we moved on.
- [`docs/VIO_RESEARCH.md`](docs/VIO_RESEARCH.md) — Why VIO, how ARKit works, validation plan.
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — Module-by-module architecture and data flow.
- [`docs/PROGRESS.md`](docs/PROGRESS.md) — Timeline of decisions and experiments.

---

## Notebooks (read in this order)

The story told by the notebooks goes: *can we predict trajectories?* → *can we get metric ground truth?* → *can we get reliable ground truth, even for a moving camera?* Full narrative in [`docs/NOTEBOOKS_GUIDE.md`](docs/NOTEBOOKS_GUIDE.md).

| # | Notebook | What it shows |
|---|---|---|
| 1 | `notebooks/01_pipeline_results.ipynb` | End-to-end pipeline result with fine-tuned Trajectron++ and six trajectory-ranking methods (predictions in pixel space). |
| 2 | `notebooks/02_calibration_demo.ipynb` | First attempt at metric ground truth via homography. Three videos, ETH base + v1–v8 fine-tunes compared. Shows what works and what *doesn't* — motivation for VIO. |
| 3 | `notebooks/03_vio_pilot.ipynb` *(in progress)* | Current attempt: ARKit / VIO based ground truth. Not yet conclusive — first field test pending. |

Auxiliary notebooks (analysis, ablations):

- `experiments/pedestrians/prediction_analysis.ipynb` — v6 model evaluation.
- `experiments/pedestrians/prediction_analysis_v8.ipynb` — v6 vs v8 (perspective-corrected) comparison.
- `prediction_ranking.ipynb`, `prediction_ranking_uav.ipynb` — ranking-method experiments.

---

## Repository layout

```
.
├── trajectron/                     # core Trajectron++ model code
├── experiments/
│   └── pedestrians/                # ETH/UCY + custom UAV data experiments
│       ├── models/                 # trained checkpoints (large; many gitignored)
│       ├── *.ipynb                 # analysis notebooks
│       ├── csv_to_trajectron_data.py
│       └── prepare_finetune_data.py
├── notebooks/                      # experiment notebooks for advisor / collaborators
│   ├── 01_pipeline_results.ipynb   # fine-tuned Trajectron++ + ranking (pixel space)
│   ├── 02_calibration_demo.ipynb   # homography ground-truth attempt (3 videos)
│   └── 03_vio_pilot.ipynb          # VIO ground-truth attempt (in progress)
├── docs/
│   ├── NOTEBOOKS_GUIDE.md          # short narrative for each notebook (read first)
│   ├── VIO_RESEARCH.md             # why VIO, theory, validation plan
│   ├── PIPELINE.md                 # module architecture
│   └── PROGRESS.md                 # timeline of decisions
├── prediction_ranking*.ipynb       # ranking experiments (root-level)
├── trajectory_*.ipynb              # visualization experiments
├── TRAJECTRON_UPSTREAM_README.md   # original Trajectron++ README
└── README.md                       # this file
```

> **Note:** the perception / calibration / VIO scripts currently live in a separate local folder (`~/Desktop/uav_perception/`). They will be merged into this repo under `perception/` before the next milestone — see [`docs/PROGRESS.md`](docs/PROGRESS.md).

---

## Quick start

```bash
# 1. environment
conda create -n trajectron python=3.6 -y
conda activate trajectron
pip install -r requirements.txt

# 2. run an existing notebook
jupyter notebook experiments/pedestrians/prediction_analysis_v8.ipynb
```

For training / evaluation commands see [`TRAJECTRON_UPSTREAM_README.md`](./TRAJECTRON_UPSTREAM_README.md).

---

## Citation

If you use this work, please also cite the original Trajectron++ paper:

```bibtex
@inproceedings{Salzmann2020,
  title={Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data},
  author={Salzmann, Tim and Ivanovic, Boris and Chakravarty, Punarjay and Pavone, Marco},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

---

## Author

Master thesis, 2026. Contact: see institute page.
