# Visual-Inertial Odometry (VIO) for Ground Truth

This document summarises the research and design decisions for transitioning from **manual homography calibration** to **VIO-based ground truth** for the human trajectory data.

---

## 1. Why this change?

### 1.1 The original problem

Trajectron++ predicts pedestrian positions in **metric world coordinates**. To train and evaluate it on our own UAV footage, we need each detected person's `(x, y)` in metres — i.e. a *metric ground truth*.

### 1.2 First attempt — homography

For static iPhone footage of a sand volleyball court, a 4-point homography was computed:

- 4 pixel corners of the court (annotated by hand)
- 4 corresponding world-frame metres (`10.3 m × 20 m`, measured with a tape measure)
- `cv2.findHomography` → 3 × 3 matrix

This worked as a proof-of-concept (see `calibration_demo.ipynb`) and produced metric trajectories that gave **plausible** ADE / FDE numbers — especially after switching the projection point from the bounding-box centre to the **foot point** (bottom-centre).

### 1.3 Why homography is not enough

Advisor feedback after the demo (2026-05): the homography approach is **not reliable as a thesis-grade ground truth** because:

| Issue | Reason |
|---|---|
| **Static camera only** | A homography is a single 3 × 3 matrix — valid only while the camera does not move. The moment the UAV moves, the mapping is invalid. |
| **Manual annotation** | Four corners must be clicked by hand on every video. Small pixel errors propagate to large metric errors (perspective). |
| **Requires a known planar region** | A volleyball court works; an arbitrary outdoor scene does not. |
| **Distance / scale ambiguity** | A single camera cannot recover scale on its own. Homography "imports" the scale from the measured world corners — so its accuracy is bounded by the tape measure. |
| **Not scalable** | We cannot collect dozens of videos in different locations if each one needs manual calibration. |

For a UAV that **moves continuously** this is a non-starter.

---

## 2. What VIO solves

### 2.1 The core idea

A **Visual-Inertial Odometry** system fuses:

- **Visual** information — feature points tracked between consecutive frames from a normal RGB camera
- **Inertial** information — accelerometer + gyroscope readings from an IMU (every smartphone has one)

The output is a **6-DoF pose** for the camera at every frame, in a metric world frame:

```
pose_t = (x_t, y_t, z_t, q_x, q_y, q_z, q_w)
         └─ position ─┘   └─ orientation (quaternion) ─┘
```

### 2.2 Why metric scale works

Monocular vision alone has a **scale ambiguity** (a small object close looks the same as a big object far away). The IMU provides the missing scale:

- The accelerometer measures gravity (≈ 9.81 m/s²) — a known physical constant.
- Integrating the accelerometer over short windows gives metric displacement.
- The visual feature tracks constrain that displacement.
- An Extended Kalman Filter (or similar) fuses the two streams.

The result is a **metric, drift-corrected** camera trajectory — without GPS, without a checkerboard, without manual calibration.

### 2.3 Open analogues

Apple's ARKit is closed-source, but the algorithm class is well-studied in the literature:

- **VINS-Mono** — Qin et al., IEEE T-RO 2018. Open-source visual-inertial system from HKUST. Very similar in spirit to ARKit.
- **OKVIS** — Leutenegger et al., IJRR 2015. Earlier keyframe-based VI SLAM.
- **ORB-SLAM3** — Campos et al., IEEE T-RO 2021. State-of-the-art multi-map SLAM with visual-inertial mode.

Reference papers:
- T. Qin, P. Li, S. Shen. *VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator.* IEEE Transactions on Robotics, 2018.
- S. Leutenegger et al. *Keyframe-based visual-inertial odometry using nonlinear optimization.* IJRR, 2015.
- C. Campos et al. *ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM.* IEEE T-RO, 2021.

---

## 3. Practical implementation: ARKit via CamTrackAR

For this thesis, the easiest entry point is **Apple ARKit** on an iPhone, accessed through the free **CamTrackAR** app.

### 3.1 Why CamTrackAR

- Free, runs on iPhone 11 and newer.
- Exports per-frame **camera pose** as CSV, plus the raw `.mp4`.
- Lets the user place **anchors** (named world-frame points) and **set the floor** (defines `Y = 0`).
- No coding required for data collection — researcher can record many videos quickly.

### 3.2 What it gives you

After a recording session, the export folder contains:

```
2026_05_10_192138/
├── 2026_05_10_192138.mp4         # raw video, 1920 × 1440 @ 60 fps
├── CameraKeyframes.CSV           # per-frame pose
└── Anchors.CSV                   # 3-D positions of user-placed anchors
```

`CameraKeyframes.CSV` columns:

| Column | Meaning |
|---|---|
| `TimeSeconds` | seconds since session start |
| `PosX, PosY, PosZ` | camera position in metres, world frame |
| `QuatX, QuatY, QuatZ, QuatW` | camera orientation as a unit quaternion |
| `FOV` | horizontal field of view in degrees |
| `LensZoom` | lens zoom factor (usually 1.0) |

ARKit's world frame:

- `+Y` = up (gravity-aligned)
- `+X` = right (in the camera's initial pose)
- `+Z` = toward the user (camera looks down `-Z`)
- Origin: device pose at the moment ARKit initialised; after **Set Floor**, `Y = 0` is moved to the floor plane.

### 3.3 What it does NOT give you

- It does **not** track people in the scene. CamTrackAR only tracks the **camera (phone)** itself.
- To recover the **person's** world coordinates we still need YOLO + ray-casting (see below).

---

## 4. Pipeline: pixel → world

Given a YOLO foot-pixel `(u, v)` at frame `t`, the corresponding camera pose `(C_t, R_t)` from `CameraKeyframes.CSV`, and the camera intrinsics derived from `FOV` and the video resolution, the person's world coordinate is:

```text
1.  Build pixel ray in camera frame
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
        ray_cam = normalize( [x_n, y_n, 1] )       # OpenCV-style convention

2.  Rotate ray into the world frame
        ray_world = R_t @ ray_cam

3.  Intersect ray with the floor plane (Y = 0)
        t* = -C_t.y / ray_world.y           # scalar where ray meets floor
        P_world = C_t + t* * ray_world      # (x, 0, z)
```

`P_world.x` and `P_world.z` are the person's metric position. Drop the `y` component — they are on the ground.

This calculation is per-frame, so it works equally well for static or moving cameras (UAV), as long as `(C_t, R_t)` is available — which VIO provides.

---

## 5. Validation plan

The thesis claim is: *VIO gives reliable metric ground truth for human trajectories.* To support this with evidence, the following experiments will be reported.

### 5.1 In-room sanity checks (done)

| Test | Procedure | Expected | Measured (2026-05) |
|---|---|---|---|
| Tape-measure walk | mark 1.72 m on the floor, set anchor at A, walk to B, set anchor | `‖B − A‖ ≈ 1.72 m` | `dz = 1.73 m` (0.6 % error) |
| Lateral sway | observe `X` while walking straight | `< 20 cm` | ~60 cm (handheld; will drop with tripod) |
| Vertical drift | `Y` while walking | `< 5 cm` | sub-cm |

**Result:** ARKit gave 1.73 m for a 1.72 m walk — well inside published ARKit accuracy (~1 – 3 %).

### 5.2 Field validation (planned, before next milestone)

| Test | Procedure | Target |
|---|---|---|
| Static camera, known walk | Tripod-mount phone on sand court; subject walks marked 10 m line; compute trajectory length from VIO | error `< 5 %` of true distance |
| Cross-video consistency | Same court, same setup, two separate recordings; align via two corner anchors; compare reported lengths | scale match within 2 % |
| Comparison with homography | Same clip processed with (a) homography from `calibration_demo.ipynb` and (b) VIO pipeline | report side-by-side ADE / FDE on the same Trajectron++ model |

### 5.3 What we are *not* claiming

- We are not claiming sub-centimetre accuracy.
- We are not claiming this generalises to large outdoor scenes without anchors.
- We are not claiming the UAV deployment is solved — the thesis scope is the **data-collection and forecasting** side.

---

## 6. Open questions / decisions still to make

1. **Origin convention across recordings.** Each ARKit session has its own `(0, 0, 0)` at the device's initial pose. For multi-video analysis we will either (a) place a fixed-location physical marker and an anchor on it at every recording, or (b) post-process: align trajectories to court corners detected automatically in the first frame.
2. **Tripod vs handheld.** Tripod is cleaner for static-camera UAV-simulation data, but the eventual UAV case is moving. A small set of handheld "moving camera" recordings should be added to demonstrate that the pipeline still works.
3. **Drift over long recordings.** ARKit drifts noticeably after > 1 – 2 minutes of continuous tracking. For now, all recordings should stay under 60 s. If longer sequences are needed, consider **loop-closure anchors** every 30 s.
4. **Alternative to CamTrackAR.** If app limitations become an issue (e.g. it disappears from the App Store, or stops exporting raw CSV), the fallback is to write a small SwiftUI iOS app using the public ARKit API directly. Estimated effort: 2 – 3 days.

---

## 7. Next concrete steps

| # | Step | Output |
|---|---|---|
| 1 | Record a single 30 – 60 s static-camera clip of one person walking on the volleyball court | `.mp4`, `CameraKeyframes.CSV`, `Anchors.CSV` |
| 2 | Run YOLOv8 on the `.mp4` → foot-pixel CSV | `foot_pixels.csv` |
| 3 | Implement `pixel_to_world_floor(...)` and project frame-by-frame | `world_trajectory.csv` |
| 4 | Plot the resulting trajectory; sanity-check length against in-court tape distance | matplotlib figure + length report |
| 5 | Feed `world_trajectory.csv` into the v8 Trajectron++ model; compute ADE / FDE against the VIO ground truth | numbers + plots |
| 6 | Add a section to `calibration_demo.ipynb` that puts homography and VIO side-by-side on the same clip | comparison cell |

Once step 6 is finished, the result is a single notebook the advisor can run end-to-end. That is the artefact for the next meeting.
