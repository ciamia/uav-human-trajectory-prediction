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

## Phase 1: sanity check on existing thesis footage

```bash
python -m gesture.mediapipe_overlay \
    --video        /Users/simayyalcin/Desktop/uav_perception/raw_videos/01.mp4 \
    --output-prefix data/gesture/01
```

This produces:

* `data/gesture/01_overlay.mp4` — your input video with hand landmarks
  drawn on every frame, ready to share.
* `data/gesture/01_keypoints.csv` — one row per detected hand per frame,
  with 21 × 3 landmark coordinates (normalised image coordinates ∈ [0, 1]
  for x and y, scale-free for z).

CSV columns:

```
frame, time_seconds, hand_index, handedness, score,
kp0_x, kp0_y, kp0_z,
kp1_x, kp1_y, kp1_z,
...
kp20_x, kp20_y, kp20_z
```

`hand_index` is 0 for the first hand detected in the frame, 1 for the
second. `handedness` is `"Left"` or `"Right"` (in camera frame, not
mirrored).

See `notebooks/04_mediapipe_sanity.ipynb` for plots and summary.

---

## What this tells us

* **Detection rate** — fraction of frames with at least one hand detected.
  For a UAV-style clip (subject 2 – 5 m away, walking), expect 40 – 80 %.
* **Confidence distribution** — if mean score is below ~0.5, the framing
  is too far / too small for MediaPipe.
* **Failure modes** — the overlay video makes failure cases visually
  obvious (subject too far, hand occluded, blurred during motion).

These three numbers decide whether Phase 2 needs custom data collection
(close-up) or whether the existing 26 thesis videos are usable for
fine-tuning.
