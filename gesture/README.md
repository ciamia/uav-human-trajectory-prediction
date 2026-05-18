# `gesture/` — Hand-gesture / intent module (scaffolding)

This folder will hold the gesture-recognition pipeline described in
[`docs/GESTURE_MODULE.md`](../docs/GESTURE_MODULE.md). For now, just the
Phase-1 sanity check is implemented.

## Install

```bash
pip install mediapipe opencv-python numpy pandas
```

(MediaPipe currently supports Python 3.8 – 3.11. A separate conda env
from the Trajectron++ env is the cleanest option.)

```bash
conda create -n gesture python=3.10 -y
conda activate gesture
pip install mediapipe opencv-python numpy pandas matplotlib jupyter
```

## What's here

| File | Purpose |
|---|---|
| `mediapipe_overlay.py` | Phase 1: run MediaPipe Hands on a video → overlay video + keypoint CSV. |

Later phases (classifier training, integration with VIO output) will add
more files; see the implementation plan in `docs/GESTURE_MODULE.md`.

---

## Phase 1: distance-stratified sanity (1 / 2 / 3 / 4 m)

Four controlled clips were captured under `~/Desktop/uav_perception/media_hands/`
(`1meters.MOV`, `2meters.MOV`, `3meters.MOV`, `4meters.MOV`), all with the same
phone in the same room and a single person making hand gestures at each
distance.

Run MediaPipe Hands on each clip:

```bash
for d in 1 2 3 4; do
  python -m gesture.mediapipe_overlay \
    --video  ~/Desktop/uav_perception/media_hands/${d}meters.MOV \
    --output-prefix data/gesture/${d}m \
    --max-hands 2 --detection-confidence 0.5 --tracking-confidence 0.5
done
```

This produces, for each distance `d`:

* `data/gesture/${d}m_overlay.mp4` — input video with hand skeletons drawn.
* `data/gesture/${d}m_keypoints.csv` — one row per detected hand per frame,
  with 21 × 3 landmark coordinates (normalised image coordinates ∈ [0, 1]
  for x and y, scale-free for z).

CSV columns:

```
frame, time_seconds, hand_index, handedness, score,
kp0_x, kp0_y, kp0_z, kp1_x, kp1_y, kp1_z, ..., kp20_x, kp20_y, kp20_z
```

`hand_index` is 0 for the first hand detected in the frame, 1 for the
second. `handedness` is `"Left"` or `"Right"` (in camera frame, not mirrored).

### Measured behaviour (`notebooks/04_mediapipe_sanity.ipynb`)

| distance | detection rate | mean score | median hand span (% of frame) |
|---:|---:|---:|---:|
| 1 m | 76.3 % | 0.962 | 34.5 % |
| 2 m | 79.0 % | 0.959 | 13.2 % |
| 3 m | 56.2 % | 0.931 |  8.4 % |
| 4 m | 31.0 % | 0.912 |  5.4 % |

**Hand span tracks the expected `~ 1/d` law** (constant hand size, constant
camera) and **detection rate halves between 2 m and 4 m**. The classifier
working range is therefore ≤ 2–3 m at this lens / resolution; beyond that
the input keypoints become spatially aliased and finger geometry is no
longer recoverable.

These numbers determine the Phase-2 setup:

* Train at 1–2 m where keypoints are reliable.
* Add a crop-then-classify head so the downstream model always sees a
  resolution-normalised hand patch, independent of subject distance.
* Use the 3 m clips as a held-out stress test; deprioritise 4 m for now.
