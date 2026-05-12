# Hand-gesture / intent module (planned)

This module is **separate** from the trajectory forecasting module. The advisor's guidance (May 2026) was that *hand tracking* and *trajectory prediction* should be two independent components that share only a common world frame — they should not be combined in a single end-to-end model.

This document describes the planned design, datasets, and implementation phases. The actual code lives under `gesture/` (to be added).

---

## 1. Why a separate module?

Three reasons:

1. **Robustness.** If one module fails (e.g. the gesture classifier outputs garbage because the person is far away), the trajectory module still works. End-to-end joint models tend to fail together.
2. **Reusability.** The gesture module can be swapped, retrained, or replaced with off-the-shelf models without re-training Trajectron++.
3. **Practical engineering.** This matches how production robot stacks are built (ROS-style: each capability is a node that publishes to a shared world model).

The two modules communicate via a shared world frame (provided by the VIO module, see [`VIO_RESEARCH.md`](VIO_RESEARCH.md)). The trajectory module publishes `(person_id, future_trajectory)`; the gesture module publishes `(person_id, intent_label, confidence)`. A downstream planner combines both.

---

## 2. What "intent" means for a UAV

For UAV safety, the useful intents are not chess gestures — they are *coarse signals* a person nearby could use, intentionally or not, to indicate where they are about to move or what they want the UAV to do:

| Intent | Hand / body cue | UAV reaction |
|---|---|---|
| `stop` | open palm raised toward the camera | freeze in place, wait |
| `follow_me` | beckoning hand (closing fingers toward palm) | follow the person at a safe distance |
| `go_away` | shooing motion (open palm pushing forward) | retreat, increase clearance |
| `point_left` / `point_right` | extended finger pointing in a direction | bias the planner away from that direction |
| `wave` | repeated lateral hand motion | attention signal, hold pattern |
| `none` | no salient hand motion | use only trajectory prediction |

The exact label set will be finalised after a small pilot study (see Phase 1 below). The label set deliberately stays small so a lightweight model is enough.

---

## 3. Building blocks

### 3.1 Keypoint extraction — MediaPipe Hands

[MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) is a free, real-time hand tracker from Google. For each detected hand, it returns:

- 21 3-D landmark coordinates per hand (wrist + 4 joints × 5 fingers).
- Confidence score, handedness (left / right).
- Runs at ~30 fps on the iPhone CPU; on a laptop GPU even faster.

This is the standard choice for hand-gesture pipelines in 2025 – 2026 and avoids re-inventing a keypoint detector.

Alternative we might also want: **MediaPipe Pose** for the full body — useful because shoulder / elbow orientation often disambiguates "pointing" gestures.

### 3.2 Temporal window

Most useful gestures are not static poses; they are short motions over ~0.5 – 1.5 s. So the gesture module buffers the last `N ≈ 30` frames (~1 s at 30 fps) of keypoints before classifying.

### 3.3 Classifier

Several architectures fit, ranked by complexity:

| Model | Pros | Cons |
|---|---|---|
| **Rule-based** (joint angles + thresholds) | trivial to debug, deterministic | brittle, doesn't generalise |
| **MLP on flattened keypoints** | tiny, fast | ignores temporal structure |
| **1D CNN on keypoint sequence** | cheap, decent baseline | limited long-range context |
| **LSTM / GRU** | natural for sequences | slow, hard to train on small data |
| **Small Transformer (encoder-only)** | strong, parallelisable | needs more data |
| **ST-GCN** ([Yan et al., AAAI 2018](https://arxiv.org/abs/1801.07455)) | designed for skeleton sequences, SOTA on hand gestures | more complex to implement |

**Plan:** start with a 1D CNN baseline, upgrade to ST-GCN if accuracy is insufficient.

### 3.4 Output schema

Per frame `t`, per person `id`, the module emits:

```json
{
  "person_id": 3,
  "world_position": [x_m, z_m],
  "intent": "point_left",
  "confidence": 0.87,
  "raw_logits": {"stop": 0.02, "follow_me": 0.04, "go_away": 0.03,
                 "point_left": 0.87, "point_right": 0.01, "wave": 0.01,
                 "none": 0.02}
}
```

This is the **only** thing the trajectory / planner module needs to consume. Implementation details (which model, which dataset) stay internal to the gesture module.

---

## 4. Datasets we can use

The training data must be **video clips of people doing the target gestures**. Two strategies:

### 4.1 Public datasets (start here)

| Dataset | Size | Gestures | License | Suitability |
|---|---|---|---|---|
| **20BN-Jester** | ~148 k clips, 27 classes | swipe, push, drop, thumbs-up, … | research-only | very large, off-the-shelf good baseline |
| **SHREC-2017** | ~2.8 k sequences, 14 / 28 classes | grab, pinch, swipe, tap, … | research-only | designed for skeleton-based methods |
| **DHG-14/28** | 2.8 k sequences | dynamic hand gestures | research-only | clean depth + skeleton |
| **IPN Hand** | ~4 k clips, 13 classes | "human–computer interaction" gestures | research-only | naturalistic |
| **ChaLearn IsoGD** | ~47 k RGB-D videos | 249 gestures | research-only | huge, but classes are micro-gestures |

Strategy: pre-train a 27-class classifier on **20BN-Jester**, fine-tune on a small custom dataset with the 6 – 7 UAV-relevant labels above.

### 4.2 Custom small dataset (only as needed)

If the public-pre-trained model doesn't generalise to our scenario (different distance, lighting, framing), record a **small** custom set:

- 3 – 5 subjects (us + lab members)
- ~20 repetitions per gesture
- Two camera angles (eye-level + slight high angle, simulating a low UAV altitude)
- Same recording setup as the trajectory data so the modules are inter-operable

That's ~500 – 1000 clips, recordable in a single afternoon.

---

## 5. Integration with the trajectory module

```
        ┌──────────────────────────┐
        │ Shared world frame       │
        │ (VIO module — Y = 0      │
        │  is the ground plane)    │
        └────────────┬─────────────┘
                     │
       ┌─────────────┴─────────────┐
       │                           │
       ▼                           ▼
┌──────────────┐            ┌──────────────┐
│ Trajectory   │            │ Gesture      │
│ module       │            │ module       │
│              │            │              │
│ YOLO → ray-  │            │ YOLO →       │
│ cast → past  │            │ MediaPipe →  │
│ positions →  │            │ keypoint     │
│ Trajectron++ │            │ window →     │
│              │            │ classifier   │
└──────┬───────┘            └──────┬───────┘
       │                           │
       │ (person_id,               │ (person_id,
       │  future_traj)             │  intent_label)
       │                           │
       └─────────────┬─────────────┘
                     ▼
            ┌─────────────────┐
            │ Planner / safety│
            │ layer           │
            │ (out of scope   │
            │  for thesis)    │
            └─────────────────┘
```

Key design choice: **the gesture module reuses the YOLO bounding boxes from the trajectory module**, so there is no extra per-frame detection cost. Each YOLO box is cropped, MediaPipe Hands is run on the crop, and the resulting keypoints are matched to the person via their bounding-box ID.

This keeps the modules independent in terms of code, but cheap in terms of compute (one YOLO pass per frame).

---

## 6. Implementation phases

A realistic order of milestones. Each phase produces a runnable artefact, even if the next phase isn't done yet.

### Phase 1 — Sanity check (1 day)

- Install `mediapipe`, run it on one of the existing 26 thesis videos.
- Confirm keypoints are detected on the walking person.
- Plot the keypoints overlaid on the video frame.
- **Output:** `notebooks/04_mediapipe_sanity.ipynb`.

### Phase 2 — Public-dataset baseline (3 – 5 days)

- Download 20BN-Jester subset (the full set is ~22 GB; a 5 – 10 class subset is enough).
- Train a 1D CNN on the keypoint sequences (MediaPipe Hands on each frame → 21 × 3 × T tensor).
- Report top-1 / top-3 accuracy.
- **Output:** `notebooks/05_gesture_baseline.ipynb`, `gesture/train_baseline.py`, a checkpoint.

### Phase 3 — UAV-relevant label set (3 – 5 days)

- Define the final UAV gesture vocabulary (the 6 – 7 labels in Section 2).
- Either (a) re-map Jester labels (e.g. "Thumb Up" → `none`, "Pushing Hand Away" → `go_away`) or (b) collect a small custom dataset as in Section 4.2.
- Fine-tune the Phase-2 model on the new labels.
- **Output:** confusion matrix, per-class precision / recall.

### Phase 4 — Integration with VIO + Trajectron++ (2 – 3 days)

- Wire the gesture module into the same processing loop as the trajectory module.
- Both modules read the same YOLO output and the same VIO pose.
- Both modules write to a single JSON-lines log file per video.
- **Output:** `notebooks/06_integrated_pipeline.ipynb` — one notebook running everything end-to-end on a single field recording.

### Phase 5 — Evaluation (1 – 2 days)

- Quantitative: confusion matrix on a held-out test split.
- Qualitative: side-by-side video — left half shows trajectory predictions, right half shows detected gestures as text overlays.
- **Output:** plots + a short evaluation video for the next advisor meeting.

Total estimated effort: **~2 – 3 weeks**, parallelisable with the VIO data collection. The first phase is one day of work and is the right thing to do *next week*, after the VIO pilot.

---

## 7. Open design choices

The following decisions are **deliberately deferred** until we have first results from Phase 1:

- Whether to keep **MediaPipe Hands** (21 keypoints per hand) or move to **MediaPipe Holistic** (face + pose + hands). Holistic is heavier but might be needed if "pointing" gestures only become reliable with elbow orientation.
- Whether the classifier outputs a single label per frame or a label per *gesture instance* (segmentation in time). The latter is harder but more useful for the planner.
- Whether to also include **facial cues** (e.g. eye gaze direction). Doable with MediaPipe Face Mesh; useful for "is the person looking at the UAV?" but adds complexity. Probably out of scope for the thesis.

These will be revisited at the end of Phase 2.

---

## 8. What this gives the thesis

Concretely, by the time of submission the thesis chapter on this module will contain:

1. The architecture diagram in Section 5.
2. A quantitative table: gesture classifier accuracy on a held-out set.
3. An evaluation video showing the integrated pipeline (trajectory + gesture + VIO) running on a real field recording.
4. A short comparison: planner decisions with vs. without the gesture module as auxiliary input. Even if a "decision" is a heuristic for now, showing that the gesture intent could plausibly change the chosen plan demonstrates the value of the module.

This is *enough* for the thesis claim that the system can take human intent into account, without requiring a fully-deployed UAV.
