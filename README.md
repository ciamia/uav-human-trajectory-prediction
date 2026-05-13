# Human-Safety-Aware Trajectory Planning for UAVs

Semester thesis project on predicting human trajectories for UAV safety.

> Built on top of [Trajectron++](https://github.com/StanfordASL/Trajectron-plus-plus) (Salzmann et al., ECCV 2020). The original upstream README is preserved at [`TRAJECTRON_UPSTREAM_README.md`](./TRAJECTRON_UPSTREAM_README.md).

---

## Goal

A UAV that flies near humans needs to predict where people will be, so it can plan a safe path. This project builds an end-to-end pipeline:

1. **Perception** — detect humans in the UAV camera feed
2. **Ground truth** — convert pixel detections to metric world coordinates
3. **Forecasting** — predict future trajectories (Trajectron++ fine-tuned on collected data)
4. **Hand-gesture intent** (separate module) — interpret pedestrian intent from hand motion
5. **Trajectory Planning** 

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
│   → foot pixels    │     │   pose per frame (on - trial for gt) │
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

A separate **hand-gesture module** (MediaPipe Hands + classifier) runs in parallel.

---



---

## Key documents

- [`docs/NOTEBOOKS_GUIDE.md`](docs/NOTEBOOKS_GUIDE.md) — **Start here.** Short narrative for each experiment notebook: question → method → result → why we moved on.
- [`docs/VIO_RESEARCH.md`](docs/VIO_RESEARCH.md) — Why VIO, how ARKit works, validation plan.
- [`docs/GESTURE_MODULE.md`](docs/GESTURE_MODULE.md) — Plan for the separate hand-gesture / intent module (MediaPipe Hands + classifier).
- [`docs/PIPELINE.md`](docs/PIPELINE.md) — Module-by-module architecture and data flow.
- [`docs/PROGRESS.md`](docs/PROGRESS.md) — Timeline of decisions and experiments.

---

## Notebooks (read in this order)

The story told by the notebooks goes: *can we predict trajectories?* → *can we get metric ground truth?* → *can we get reliable ground truth, even for a moving camera?* 

| # | Notebook | What it shows |
|---|---|---|
| 1 | `notebooks/01_pipeline_results.ipynb` | End-to-end pipeline result with fine-tuned Trajectron++ and six trajectory-ranking methods (predictions in pixel space). |
| 2 | `notebooks/02_calibration_demo.ipynb` | First attempt at metric ground truth via homography. Three videos, ETH base + v1–v8 fine-tunes compared. Shows what works and what *doesn't* — motivation for VIO. |
| 3 | `notebooks/03_vio_pilot.ipynb` *(in progress)* | Current attempt: ARKit / VIO based ground truth. |


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
│   ├── GESTURE_MODULE.md           # plan for hand-gesture / intent module
│   ├── PIPELINE.md                 # module architecture
│   └── PROGRESS.md                 # timeline of decisions
├── prediction_ranking*.ipynb       # ranking experiments (root-level)
├── trajectory_*.ipynb              # visualization experiments
├── TRAJECTRON_UPSTREAM_README.md   # original Trajectron++ README
└── README.md                       # this file
```



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


