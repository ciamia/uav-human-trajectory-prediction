# Hand-gesture / intent module (planned)

This module is **separate** from the trajectory forecasting module. It was decided that *hand tracking* and *trajectory prediction* should be two independent components.

This document describes the planned design, datasets, and implementation phases. The actual code lives under `gesture/` (to be added).

---

## 1. Why a separate module?

Three reasons:

1. **Robustness.** If one module fails (e.g. the gesture classifier outputs garbage because the person is far away), the trajectory module still works. End-to-end joint models tend to fail together.
2. **Reusability.** The gesture module can be swapped, retrained, or replaced with off-the-shelf models without re-training Trajectron++.
3. **Practical engineering.** This matches how production robot stacks are built (ROS-style: each capability is a node that publishes to a shared world model).



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


----------

Dataset Samples Classes Subjects Scenes Resolution Annotations Annotation Method
LaRED, 2014 [13] 243,000 81 10 10 640 × 480 masks automatically
OUHANDS, 2016 [22] 3,000 10 23 various 640 × 480 masks, boxes automatically
HANDS, 2021 [25] 12,000 29 5 5 960 × 540 boxes –
SHAPE, 2022 [2] 33,471 32 20 various 4128 × 3096 masks, boxes manually
HaGRID, 2023 554,800 18 + 1 37,583 ⩾ 37,583 1920 × 1080 boxes manually




---

## 5. Integration with the trajectory module

```
        
       ┌─────────────--────────────┐
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
                 │
            └─────────────────┘
```





---

