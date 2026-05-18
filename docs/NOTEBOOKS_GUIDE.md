# Notebooks guide


The notebooks should be read in this order, because each one motivated the next:

1. [Pipeline results](#1-pipeline-results) — does our fine-tuned model produce reasonable predictions?
2. [Calibration demo (homography)](#2-calibration-demo-homography) — first attempt at metric ground truth.
3. [VIO pilot](#3-vio-pilot-in-progress) — current attempt at ground truth, motivated by the limitations of #2.
4. [MediaPipe sanity](#4-mediapipe-sanity-phase-1-of-the-gesture-module) — Phase 1 of the separate gesture module.

---

## 1. Pipeline results

**File:** `notebooks/01_pipeline_results.ipynb`

**Question.** Does Trajectron++, fine-tuned on our own UAV-style footage, produce reasonable trajectory predictions?

**What's inside.**
- 26 self-collected single-person walking videos (iPhone, volleyball court and around campus).
- Pre-training: ETH base model, 100 epochs.
- Fine-tuning: 8 variants (v1 – v8) sweeping over KL weight, data augmentation, and perspective correction.
- Six ground-truth-free ranking methods over the 20 sample trajectories: log-likelihood, mean-proximity, KDE consensus, endpoint centrality, history consistency, UAV-specific risk.
- Visualisations: observed history → predicted samples → ranked best path.

**Main results.**
- Predictions are *shape-correct* (the model learns the general direction and curvature of human walks).
- v6 (`+aug`) outperformed v1 – v5; v8 (`+perspective`) gave a further small improvement.


**Why this notebook motivated the next.**

The ADE / FDE numbers in this notebook are reported in pixels, not metres, because we didn't yet have a reliable pixel→metre mapping. That's exactly the gap the calibration notebook tries to fill.

---

## 2. Calibration demo (homography)

**File:** `notebooks/02_calibration_demo.ipynb`

**Question.** If we hand-calibrate the camera with a 4-point homography, can we recover metric ground truth from pixel detections?

**Why we tried this.** After the pipeline above, evaluating Trajectron++ in pixels is meaningless. ADE / FDE only have a clear interpretation when expressed in metres. A homography between four pixel corners and four tape-measured world corners is the simplest possible way to get there for a static camera looking at a planar region (a court).

**What's inside.** Three videos, processed end-to-end:

| Video | Scene | World size | Camera position |
|---|---|---|---|
| `IMG_5925.MOV` | Sand volleyball court | 20 m × 10.3 m | partial-court view, near corners cropped |
| `IMG_5948.MOV` | Same court, full-court take | 20 m × 10.3 m | all four corners visible |
| `IMG_5946.MOV` | Smaller paved area | 10.3 m × 7.5 m | close, low angle |

For each video:
1. YOLOv8 → pedestrian detection → **foot pixel** (bottom-centre of the box).
2. Hand-annotated court corners → `cv2.findHomography` → 3 × 3 matrix.
3. Pixel trajectory → metric trajectory.
4. Trajectory played back through Trajectron++ (ETH base + v1 – v8 fine-tunes) → ADE / FDE in metres.

**Main results.**
- "In-court" rate after the foot-point fix: > 90 % (was ~30 % when using the box centre).
- Best metric error: **Video 3**, the closest camera angle. Geometry dominates: the further the subject from the camera, the more a 1-px error becomes several centimetres in metres.


**Why this isn't enough.** Homography:

1. Only works for a static camera. The angle should never change and is not reliable.
2. Needs manual annotation. Hard to gather data.
3. Pixel-error sensitivity is bad at the far end of the court..
4. Requires a known planar region. 

The advisor's verdict (early May 2026): not reliable enough as thesis ground truth. Hence the move to VIO.

---

## 3. VIO pilot

**File:** `notebooks/03_vio_pilot.ipynb`

**Question.** Can we replace homography with Visual-Inertial Odometry (VIO) and recover metric ground truth automatically, including for moving cameras?

**Why.** Section 2 (homography) had four structural problems: static camera only, manual annotation, far-field pixel-error blow-up, and requires a known planar region. VIO removes all four.

**What's inside.**

1. **Validation: tape-measure walks (1 m / 5 m / 10 m).** Held the phone in hand, walked a measured straight line in CamTrackAR; the VIO-reported camera displacement is compared against the tape measure. Result: sub-percent error at 5 m and 10 m (the 1 m result is dominated by start-of-trace body sway).
2. **Field clip 1.** Static phone, single person walking close (3–13 m) for 44 s. End-to-end pipeline: YOLOv8 → `pixel_to_world.py` (ray-cast to ARKit floor) → `predict_with_trajectron.py` (sliding-window CVAE on dt=0.4 s). Clean trajectory; Trajectron++ v8 gives ADE 1.16 m / FDE 2.05 m / best-of-20 ADE 0.78 m on a 4.8 s horizon.
3. **Field clip 2.** Same setup but the person walked further away (22–60 m). Per-frame ray-cast becomes noisy at that range (1 px → tens of cm world error); a 2 s rolling-median filter is required before forecasting. Predictions are correspondingly worse (ADE 3.78 m / FDE 5.79 m), which is the right qualitative answer for this scenario.

The driver scripts all live in [`vio/`](../vio/README.md): `validate_camera_walk.py`, `run_yolo_foot.py`, `pixel_to_world.py`, `analyze_field.py`, `predict_with_trajectron.py`, `plot_predictions.py`.

**Main result.** VIO is the right ground-truth source for this thesis. The remaining work is engineering: keep subjects within a sensible depth range (or move to multi-view ground truth), add a proper multi-person tracker, and re-evaluate.

---

## 4. MediaPipe sanity (Phase 1 of the gesture module)

**File:** `notebooks/04_mediapipe_sanity.ipynb`

**Question.** Does MediaPipe Hands reliably detect hand landmarks on our existing thesis footage (people walking, distance 2 – 5 m, outdoor lighting)?

**Why now.** Phase 1 of the gesture module (see [`GESTURE_MODULE.md`](GESTURE_MODULE.md)) is just a feasibility check. Before training any classifier, we need to know if the keypoint extractor itself works on our framing. This is one day of work and unblocks the rest of the module.

**What's inside.** Picks one existing video, runs `gesture/mediapipe_overlay.py`, writes an overlay video + a keypoint CSV, then reports detection rate, mean confidence, and a sample frame.

**Output we'll show in the meeting.** Three numbers (detection rate, mean confidence, observed failure modes) and one frame from the overlay video.

---

## Quick reference — what each notebook answers

| Notebook | Question | Status |
|---|---|---|
| `01_pipeline_results.ipynb` | Does fine-tuned Trajectron++ work on our data? | Done — yes (pixel-space). |
| `02_calibration_demo.ipynb` | Can homography give metric ground truth? | Done — partially, but limitations are structural. |
| `03_vio_pilot.ipynb` | Can VIO give metric ground truth, including for moving cameras? | Done — yes; validated + two field clips end-to-end. |
| `04_mediapipe_sanity.ipynb` | Does MediaPipe Hands detect hands on our footage? | Scaffold ready; first run pending. |

