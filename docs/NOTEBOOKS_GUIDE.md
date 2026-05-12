# Notebooks guide

Short reading guide for the experiment notebooks in this repository, written for collaborators (advisor + co-supervised PhD student) who want to quickly see *what we tried, what worked, and where we are now* — without reading every cell.

The notebooks should be read in this order, because each one motivated the next:

1. [Pipeline results](#1-pipeline-results) — does our fine-tuned model produce reasonable predictions?
2. [Calibration demo (homography)](#2-calibration-demo-homography) — first attempt at metric ground truth.
3. [VIO pilot](#3-vio-pilot-in-progress) — current attempt, motivated by the limitations of #2.

---

## 1. Pipeline results

**File:** `notebooks/01_pipeline_results.ipynb`
*(source also kept at `experiments/pedestrians/presentation_pipeline_results.ipynb`)*

**Question.** Does Trajectron++, fine-tuned on our own UAV-style footage, produce reasonable trajectory predictions?

**What's inside.**
- 26 self-collected single-person walking videos (iPhone, volleyball court and around campus).
- Pre-training: ETH base model, 100 epochs.
- Fine-tuning: 8 variants (v1 – v8) sweeping over KL weight, data augmentation, and perspective correction.
- Six **ground-truth-free ranking methods** over the 20 sample trajectories: log-likelihood, mean-proximity, KDE consensus, endpoint centrality, history consistency, UAV-specific risk.
- Visualisations: observed history → predicted samples → ranked best path.

**Main results.**
- Predictions are *shape-correct* (the model learns the general direction and curvature of human walks).
- v6 (`+aug`) outperformed v1 – v5; v8 (`+perspective`) gave a further small improvement.
- The biggest single quality win was a code fix, not a hyperparameter — switching from bounding-box centre to **foot point** moved the projection from the torso to the actual ground contact.

**Why this notebook motivated the next.**

The ADE / FDE numbers in this notebook are reported in **pixels**, not metres, because we didn't yet have a reliable pixel→metre mapping. That's exactly the gap the calibration notebook tries to fill.

---

## 2. Calibration demo (homography)

**File:** `notebooks/02_calibration_demo.ipynb`

**Question.** If we hand-calibrate the camera with a 4-point homography, can we recover metric ground truth from pixel detections?

**Why we tried this.** After the pipeline above, the advisor pointed out (correctly) that *evaluating Trajectron++ in pixels is meaningless*. ADE / FDE only have a clear interpretation when expressed in metres. A homography between four pixel corners and four tape-measured world corners is the simplest possible way to get there for a static camera looking at a planar region (a court).

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
- Cross-video ADE / FDE table sits in the last markdown cell of the notebook.

**Why this isn't enough.** This is the conversation we are having with the advisor right now. Homography:

1. **Only works for a static camera.** A UAV is never static. The homography would have to be re-estimated every frame — i.e. it's the wrong tool.
2. **Needs manual annotation.** Four corners per video, by hand. Doesn't scale to a thesis-grade dataset.
3. **Pixel-error sensitivity is bad at the far end of the court.** A 2 px corner error becomes 20 – 30 cm in metres.
4. **Requires a known planar region.** A volleyball court works; a random outdoor scene does not.

The advisor's verdict (early May 2026): *not reliable enough as thesis ground truth.* Hence the move to VIO.

---

## 3. VIO pilot (in progress)

**File:** `notebooks/03_vio_pilot.ipynb` — *not yet committed, results currently being collected.*

**Question.** Can we replace homography with Visual-Inertial Odometry (VIO) and recover metric ground truth automatically, even when the camera moves?

**Why we're trying this.** A VIO system fuses the smartphone camera with the built-in IMU. The IMU provides metric scale (from gravity), the camera provides drift-corrected motion, and the combined output is a **6-DoF camera pose every frame** — in the right units, with no manual calibration, and crucially still working when the camera moves.

That removes every objection from Section 2.

**Status (today).**
- Selected **ARKit on iPhone 11** via the free **CamTrackAR** app as the entry point — same algorithm class as VINS-Mono, deployable today with no extra hardware.
- In-room sanity check: walked a tape-measured 1.72 m, ARKit reported 1.73 m → **0.6 % error**.
- Designed the pipeline: keep YOLO and Trajectron++ as-is, replace the homography step with a per-frame ray-cast `pixel → world floor` using the ARKit camera pose. Full theory + validation plan in [`VIO_RESEARCH.md`](VIO_RESEARCH.md).

**What this notebook *will* contain when it lands.**

1. Load one CamTrackAR clip (`.mp4` + `CameraKeyframes.CSV` + `Anchors.CSV`).
2. Run YOLOv8 on the video → foot pixels.
3. For each frame, look up the camera pose from `CameraKeyframes.CSV`, build the camera-frame ray from the foot pixel, rotate to world frame, intersect with `Y = 0`.
4. Resulting metric trajectory, plot side-by-side with the homography output on the *same* clip.
5. Quantitative comparison: ADE / FDE of Trajectron++ predictions against the two ground truths.

**Important caveat for readers.** This is an experiment, **not a confirmed result**. We do not yet claim that VIO is the final answer — we claim it is the next thing to try, because it removes the structural problems of homography. The notebook will be updated with real numbers once the first field recording is processed (planned this week).

---

## Quick reference — what each notebook answers

| Notebook | Question | Status |
|---|---|---|
| `01_pipeline_results.ipynb` | Does fine-tuned Trajectron++ work on our data? | Done — yes (pixel-space). |
| `02_calibration_demo.ipynb` | Can homography give metric ground truth? | Done — partially, but limitations are structural. |
| `03_vio_pilot.ipynb` | Can VIO give metric ground truth, including for moving cameras? | In progress — first field test pending. |

For the longer story (timeline, why each step was taken, what's planned next), see [`PROGRESS.md`](PROGRESS.md).
