# Pipeline Architecture

End-to-end data flow for the thesis, broken into independent modules. Following advisor feedback (2026-05), **hand tracking** and **trajectory forecasting** are two separate modules that share only a common world frame.

---

## High-level overview

```
                                  ┌──────────────────────────┐
                                  │   World frame (metres)   │
                                  │   provided by VIO        │
                                  └────────────┬─────────────┘
                                               │
                  ┌────────────────────────────┼────────────────────────────┐
                  │                            │                            │
                  ▼                            ▼                            ▼
        ┌──────────────────┐         ┌──────────────────┐         ┌──────────────────┐
        │  Module A        │         │  Module B        │         │  Module C        │
        │  Perception      │         │  Forecasting     │         │  Hand-gesture    │
        │  (per-person     │         │  (Trajectron++,  │         │  (MediaPipe +    │
        │   trajectories)  │         │   GMM over 20    │         │   classifier)    │
        │                  │         │   samples)       │         │                  │
        └────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
                 │                            │                            │
                 │ metric trajectory          │ predicted trajectory       │ intent label
                 │                            │                            │
                 └────────────┬───────────────┴───────────┬────────────────┘
                              │                           │
                              ▼                           ▼
                       ┌──────────────────┐      ┌──────────────────┐
                       │  Ranking module  │      │  UAV planner     │
                       │  (likelihood,    │ ───▶ │  (out of scope)  │
                       │   risk, …)       │      │                  │
                       └──────────────────┘      └──────────────────┘
```

The **dotted line below** is the data-collection path that produced today's results; the **solid path** is the eventual UAV deployment.

---

## Module A — Perception

**Input:** RGB video stream (UAV camera, or iPhone for data collection).
**Output:** for each frame `t`, a list of detections `[(track_id, foot_pixel_u, foot_pixel_v), …]`.

### Steps

1. **YOLOv8s** (`yolov8s.pt`, COCO classes) on every frame; keep only `class == person`.
2. **Foot point** = bottom-centre of the YOLO bounding box, **not** the centre — gives the contact point with the ground.
3. **Tracking** by simple IoU + Hungarian assignment (this can be upgraded to a learned tracker later, e.g. BoT-SORT).
4. **Outlier filter:** drop tracks shorter than 8 frames (less than one observation window for Trajectron++).

### Key files

- `~/Desktop/uav_perception/yolo_simple_overlay.py` — runs YOLO and writes overlay video + CSV
- `~/Desktop/uav_perception/process_calibration_video*.py` — variants per dataset
- `experiments/pedestrians/csv_to_trajectron_data.py` — converts the CSV into the `.pkl` Trajectron++ expects

### Known issues

- Long-range detections drop out when the person is < ~30 px tall.
- A second person briefly entering the frame creates a phantom track in 2 – 3 of the 26 collected videos. Documented in `docs/PROGRESS.md`.

---

## Module B — Ground truth (VIO)

**Input:** RGB video + IMU stream (or the equivalent CamTrackAR export).
**Output:** for every frame `t`, the camera pose `(C_t ∈ ℝ³, R_t ∈ SO(3))` in the world frame, plus a known ground plane `Y = 0`.

### Steps (current setup, iPhone + CamTrackAR)

1. Record video with CamTrackAR — ARKit produces `CameraKeyframes.CSV`.
2. Optionally place anchors at known physical reference points (court corners) for cross-video alignment.
3. Save the export folder under `data/vio/<session_id>/`.

### Steps (future, on-board UAV)

The same role can be filled by:

- An on-board IMU + downward-facing camera running VINS-Mono or ORB-SLAM3.
- DJI / PX4 VIO output, if the UAV exposes pose.

The interface to Module C stays the same — `(C_t, R_t)` per frame.

### Ray-cast to ground

Given the foot pixel from Module A and the pose from Module B:

```python
def pixel_to_world_floor(uv, pose, fx, fy, cx, cy):
    u, v = uv
    cam_pos, R = pose                          # R is 3x3 rotation matrix
    ray_cam = np.array([(u - cx) / fx,
                        (v - cy) / fy,
                        1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    ray_world = R @ ray_cam
    if abs(ray_world[1]) < 1e-6:               # ray parallel to floor
        return None
    t = -cam_pos[1] / ray_world[1]
    return cam_pos + t * ray_world             # (x, 0, z)
```

This gives the **metric `(x, z)`** position of the foot at every frame. Stacked over time, this is the **trajectory ground truth** that feeds Module C.

---

## Module C — Forecasting (Trajectron++)

**Input:** for each person, an 8-step history of `(x, y)` positions in metres (`Δt = 0.4 s` by default, i.e. 2.5 Hz observations covering ~3.2 s of history).
**Output:** a GMM over 12 future steps (~4.8 s horizon), plus 20 sampled trajectories.

### Current models

| Tag | Notes |
|---|---|
| `models_08_Mar_2026_23_53_32_eth_100ep` | ETH base, 100 epochs |
| `models_08_Mar_2026_23_36_01_uav_finetune` | first naive fine-tune |
| `models_09_Mar_2026_*_uav_ft_v2…v5` | fine-tune hyperparameter sweep |
| `models_09_Mar_2026_16_10_23_uav_ft_v6_aug` | + data augmentation (flips, 90 / 270 rotations, gaussian noise) — current "good" model |
| `models_10_Mar_2026_00_42_50_uav_ft_v8_persp` | + perspective-corrected ground truth (homography) |

`v6` is the production model for evaluation. `v8` is the current "do better ground truth helps?" answer (yes, ADE / FDE improve modestly).

### Training command

See `TRAJECTRON_UPSTREAM_README.md` for the exact `train.py` invocation. Custom args added in this fork: see `trajectron/argument_parser.py` (the `M` in `git status`).

---

## Module D — Hand-gesture intent (planned, separate)

**Input:** RGB video (same camera).
**Output:** for each frame and each detected person, a discrete gesture / intent label.

### Why separate

Advisor feedback (2026-05): keeping hand tracking and trajectory forecasting in **separate modules** that share only the world frame:

- lets either module fail or be replaced without breaking the other,
- matches how production robotic systems are structured,
- removes the need to backpropagate through a large multi-modal joint model.

### Sketch of pipeline

1. **MediaPipe Hands** — detect 21 3-D keypoints per hand, every frame.
2. Buffer the last `N` frames (~1 s window).
3. **Classifier** (small Transformer or ST-GCN) → gesture label `{stop, point-left, point-right, wave, none}`.
4. Publish the label to the planner alongside the trajectory predictions.

This module is *planned*, not yet implemented. First step: choose a public dataset (e.g. **SHREC**, **DHG-14/28**) for the classifier.

---

## Module E — Ranking (already implemented)

Given the 20 sampled trajectories from Module C, the ranking module selects the "best" / "safest" one. Six methods are implemented and compared in `prediction_ranking*.ipynb`:

| Method | Idea | Best for |
|---|---|---|
| Log-likelihood | per-sample log-prob under the GMM | mode-finding |
| Mean proximity | distance to the per-step mean | low-spread sets |
| KDE consensus | density of nearby samples | multi-modal sets |
| Endpoint centrality | how central the final point is | endpoint-critical tasks |
| History consistency | similarity to the observed history | inertial paths |
| **Risk** | weighted distance to UAV + collision time | UAV-specific |

The UAV-specific risk ranker is the one that ultimately matters for planning.

---

## Data flow at runtime

For a single 30 s clip processed offline today:

```
clip.mp4 ─┬──> YOLOv8 ─────────> foot_pixels.csv
          │
          └──> ARKit (CamTrackAR) ──> CameraKeyframes.CSV
                                           │
foot_pixels.csv + CameraKeyframes.CSV ─────┴──> ray-cast ─> world_traj.csv
                                                                  │
                                                                  └──> Trajectron++ ─> predictions.npz
                                                                                              │
                                                                                              └──> ranking ─> chosen_path.csv
```

Wall-clock cost on an M1 laptop, for a 30 s @ 30 fps clip:

- YOLO: ~12 s (GPU not available; CPU fallback)
- VIO: real-time on the phone, free during recording
- Ray-cast: < 1 s for 900 frames
- Trajectron++ inference: ~3 s for ~30 trajectory windows
- Ranking: < 1 s

Total: ~20 s per 30 s clip. Practical for iterating during the thesis.
