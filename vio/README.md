# `vio/` — VIO ground-truth utilities

Driver scripts that turn CamTrackAR exports + YOLO detections into a metric
world trajectory, and (optionally) feed that trajectory into Trajectron++.
See `docs/VIO_RESEARCH.md` for the theory and `notebooks/03_vio_pilot.ipynb`
for the end-to-end walkthrough.

## Install

```bash
pip install numpy pandas matplotlib opencv-python ultralytics torch dill
```

`numpy / pandas / matplotlib` cover validation + analysis. `opencv-python` and
`ultralytics` are only needed for `run_yolo_foot.py`. `torch / dill` are only
needed for `predict_with_trajectron.py` and require the Trajectron++ repo on
your `PYTHONPATH` (this script appends it automatically).

## What's here

| File | Purpose |
|---|---|
| `validate_camera_walk.py` | Camera displacement summary from a `CameraKeyframes.CSV` — used for the tape-measure tests where no `Anchors.CSV` was recorded. |
| `validate_anchors.py` | Same idea but using two anchors from an `Anchors.CSV`. |
| `run_yolo_foot.py` | YOLOv8 person detection on a video → `foot_pixels.csv` (with optional overlay video). |
| `pixel_to_world.py` | Foot pixels + camera pose → `world_trajectory.csv` (metric world XZ). |
| `analyze_field.py` | Summary stats + top-down plot of a `world_trajectory.csv`, with optional rolling-median smoothing. |
| `predict_with_trajectron.py` | Resample to `dt = 0.4 s`, run Trajectron++ on a sliding window, save samples to `.npz`. |
| `plot_predictions.py` | Per-window panels (history, samples, mean, GT) from a predictions `.npz`. |

---

## Tape-measure validation

Two ways to do this:

**With anchors.** Place two physical anchors at a known distance, drop ARKit anchors at each, export from CamTrackAR, then:

```bash
python -m vio.validate_anchors \
    --anchors-csv data/vio/validation/test_1m/Anchors.CSV \
    --expected 1.00
```

**Without anchors (just walk).** Hold the phone steady, walk a measured straight line, stop. The phone's own VIO pose between the first and last keyframe is the measurement:

```bash
python -m vio.validate_camera_walk \
    --camera-csv ~/Desktop/uav_perception/vio_raw_video/5meters_2026_05_18_1452443210/CameraKeyframes.CSV \
    --expected 5.0
```

Example output (the May 2026 batch):

```
=== 5meters_..._CameraKeyframes.CSV ===
  keyframes:        947
  duration:         15.77 s
  start:            (-0.023, +1.204, -0.042) m
  end:              (-1.116, +1.185, -4.883) m
  3D displacement:  4.963 m
  XZ displacement:  4.963 m
  3D path length:   5.143 m
  expected:         5.000 m
  error (XZ disp):  -0.037 m  (-0.75 %)
```

Repeat for 1 m, 5 m, 10 m → three-row accuracy table for the meeting (see `notebooks/03_vio_pilot.ipynb` § 1).

---

## End-to-end pipeline on one clip

```bash
# 1. YOLOv8 person detection → foot pixels
python -m vio.run_yolo_foot \
    --video       ~/Desktop/uav_perception/vio_raw_video/<clip>/<clip>.mp4 \
    --output      data/vio/<clip>/foot_pixels.csv \
    --overlay-video data/vio/<clip>/overlay.mp4 \
    --model       yolov8s.pt

# 2. Ray-cast foot pixels onto the ARKit floor plane
python -m vio.pixel_to_world \
    --camera-csv  data/vio/<clip>/CameraKeyframes.CSV \
    --pixels-csv  data/vio/<clip>/foot_pixels.csv \
    --video-width  1920 --video-height 1440 \
    --output      data/vio/<clip>/world_trajectory.csv

# 3. Summary + top-down plot (also computes a 0.5 s smoothed version)
python -m vio.analyze_field \
    --trajectory data/vio/<clip>/world_trajectory.csv \
    --output     data/vio/<clip>/trajectory_plot.png

# 4. Trajectron++ forecasts on a sliding window
python -m vio.predict_with_trajectron \
    --trajectory data/vio/<clip>/world_trajectory.csv \
    --model-dir  ~/Trajectron-plus-plus/experiments/pedestrians/models/models_10_Mar_2026_00_42_50_uav_ft_v8_persp \
    --checkpoint 45 \
    --output     data/vio/<clip>/predictions_v8.npz \
    --num-samples 20 --stride 2

# 5. Plot the predictions (6 windows + summary)
python -m vio.plot_predictions \
    --predictions data/vio/<clip>/predictions_v8.npz \
    --output      data/vio/<clip>/predictions_plot.png
```

`world_trajectory.csv` has one row per frame per track:

```
frame, time_seconds, track_id, u, v, world_x, world_y, world_z,
cam_pos_x, cam_pos_y, cam_pos_z
```

`world_y` is always ~0 (we project onto the floor). The trajectory is `(world_x, world_z)` in metres.

---

## Notes / caveats

* **FOV.** CamTrackAR exports a per-frame FOV (horizontal, in degrees).
  We use that to build the intrinsics. If you have a more accurate
  intrinsic calibration (e.g. from a checkerboard), pass `--fov` to
  override.
* **Floor plane.** `--floor-y 0` works after you press *Set Floor* in
  CamTrackAR. If you forgot to, set `--floor-y` to whatever Y value
  corresponds to the ground (e.g. `--floor-y -1.5` for a phone held at
  1.5 m above the ground).
* **No rolling-shutter correction.** Fine at walking speeds; would matter
  for fast motion or a drone in a sharp turn.
* **One person per frame.** `run_yolo_foot.py` keeps only the largest YOLO box per frame and assigns it `track_id = 0`. For multi-person scenes you'd want a proper tracker (BoT-SORT / ByteTrack) — see `docs/PIPELINE.md`.
* **Long-range ray-cast amplifies pixel noise.** A YOLO foot detection that wiggles by 1 px at 30 m produces ~30 cm of world noise (ratio ≈ depth / camera_height). For Field 2 we mitigate this with a 2 s rolling-median filter before forecasting; see the notebook for details.
