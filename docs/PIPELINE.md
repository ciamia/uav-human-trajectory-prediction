# Pipeline Architecture

End-to-end data flow, broken into independent modules. Hand tracking (gesture classifier) and trajectory forecasting are two separate modules.


```
                                  
        ┌──────────  ──────          ┌──────────────────┐         ┌──────────────────┐
        │  Module A        │         │  Module B        │         │  Module C        │
        │  Perception      │         │  Forecasting     │         │  Hand-gesture    │
        │  (per-person     │         │  (Trajectron++,  │         │  (MediaPipe +    │
        │   trajectories)  │         │   GMM over 20    │         │   classifier)    │
        │                  │         │   samples)       │         │                  │
        └────────┬─────────┘         └────────┬─────────┘         └────────┬─────────┘
                 │                            │                            │
                 │ metric trajectory          │ predicted trajectory                   │ intent label
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



---

## Module A — Perception

**Input:** RGB video stream (UAV camera, or iPhone for data collection).
**Output:** for each frame `t`, a list of detections `[(track_id, foot_pixel_u, foot_pixel_v), …]`.

### Steps

1. **YOLOv8s** (`yolov8s.pt`, COCO classes) on every frame; keep only `class == person`.
2. **Foot point** = bottom-centre of the YOLO bounding box, **not** the centre — gives the contact point with the ground.
3. **Tracking** by simple IoU + Hungarian assignment.
4. Outlier filter.

### Key files

- `~/Desktop/uav_perception/yolo_simple_overlay.py` — runs YOLO and writes overlay video + CSV
- `~/Desktop/uav_perception/process_calibration_video*.py` — variants per dataset
- `experiments/pedestrians/csv_to_trajectron_data.py` — converts the CSV into the `.pkl` Trajectron++ expects

### Known issues

- Long-range detections drop out.
- A second person briefly entering the frame creates a phantom track in 2 – 3 of the 26 collected videos. Documented in `docs/PROGRESS.md`.

---

## Module B — Ground truth (VIO) Trial Right Now

**Input:** RGB video + IMU stream (or the equivalent CamTrackAR export).
**Output:** for every frame `t`, the camera pose `(C_t ∈ ℝ³, R_t ∈ SO(3))` in the world frame, plus a known ground plane `Y = 0`.

### Steps (current setup, iPhone + CamTrackAR)

Record video with CamTrackAR — ARKit produces `CameraKeyframes.CSV`.



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
**Output:** a GMM over future steps, plus 20 sampled trajectories.

### Current models

| Tag | Notes |
|---|---|
| `models_08_Mar_2026_23_53_32_eth_100ep` | ETH base, 100 epochs |
| `models_08_Mar_2026_23_36_01_uav_finetune` | first naive fine-tune |
| `models_09_Mar_2026_*_uav_ft_v2…v5` | fine-tune hyperparameter sweep |
| `models_09_Mar_2026_16_10_23_uav_ft_v6_aug` | + data augmentation (flips, 90 / 270 rotations, gaussian noise) — current "good" model |
| `models_10_Mar_2026_00_42_50_uav_ft_v8_persp` | + perspective-corrected ground truth (homography) |

`v6` is the production model for evaluation. `v8` is the current "do better ground truth helps?" answer (yes, ADE / FDE improve modestly). With the better achieved ground truth the model can be and should be trained with the collected data.

### Training command

See `TRAJECTRON_UPSTREAM_README.md` for the exact `train.py` invocation. Custom args added in this fork: see `trajectron/argument_parser.py` (the `M` in `git status`).

---

## Module D — Hand-gesture intent (planned, separate)

**Input:** RGB video (same camera) + the YOLO boxes from Module A.
**Output:** for each frame and each detected person, a discrete gesture / intent label with confidence.

### Why separate

keeping hand tracking and trajectory forecasting in **separate modules**.

- lets either module fail or be replaced without breaking the other,
- matches how production robotic systems are structured,
- removes the need to backpropagate through a large multi-modal joint model.
-for now simpler.

### Sketch of pipeline

 **MediaPipe Hands** — detect 21 3-D keypoints per hand, every frame.
 Buffer the last `N ≈ 30` frames (~1 s window) of keypoints.
 **Classifier** → gesture label `{stop, follow_me, go_away, point_left, point_right, wave, none}`.
 Publish `(person_id, intent_label, confidence)` to the planner alongside the trajectory predictions.



This module is *planned*, not yet implemented. Working on the MediaPipe phase yet.

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

