# Progress log

Running log of what was done, when, and why. Most recent at the top.

---

## 2026-05 — Transition to VIO

**Trigger.** Advisor meeting (early May): homography-based ground truth is not reliable enough for a thesis claim. Suggested direction: Visual-Inertial Odometry.

**Done so far:**

- Read into VIO theory (VINS-Mono, OKVIS, ORB-SLAM3). Summary in [`VIO_RESEARCH.md`](VIO_RESEARCH.md).
- Decided to use **ARKit on iPhone 11** via the free **CamTrackAR** app for the proof-of-concept. Reasons: zero hardware cost, IMU + camera already present, exports per-frame pose CSV.
- Installed CamTrackAR, recorded a small in-room validation clip.
- Validation result: walked a tape-measured **1.72 m** → ARKit reported **1.73 m** along the Z-axis → ~0.6 % error. Lateral (X) deviation ~60 cm explained by handheld body sway; vertical drift sub-cm.
- Designed the new pipeline: keep YOLO and Trajectron++ as-is, replace homography with a `pixel_to_world_floor()` ray-cast that uses the per-frame ARKit pose.
- Decided that **hand tracking and trajectory forecasting** will live in two **independent modules** that share only the world frame (advisor recommendation). See [`PIPELINE.md`](PIPELINE.md).

**Planned before next milestone:**

1. Record a 30 – 60 s static-camera clip on the volleyball court.
2. Implement `pixel_to_world_floor()` and run it on that clip.
3. Compare resulting world trajectories with the previous homography output, side-by-side.
4. Add a "VIO" section to `calibration_demo.ipynb`.
5. Start the gesture module (Module D) — design document already written, see [`GESTURE_MODULE.md`](GESTURE_MODULE.md). First concrete step: MediaPipe Hands sanity check on the existing 26 videos.

---

## 2026-03 — Ranking module + perspective correction (v8)

- Implemented six GT-free ranking methods (log-lik, mean prox, KDE, endpoint centrality, history consistency, risk).
- Realised that the bounding-box centre was being used as the person's position; switched to **foot point** (bottom-centre of the YOLO box). Court "in-bounds" rate jumped from ~30 % to > 90 %.
- Fine-tuned a perspective-aware model **v8** (`models_10_Mar_2026_00_42_50_uav_ft_v8_persp`) on perspective-corrected metric trajectories from the homography pipeline.
- Built `prediction_analysis_v8.ipynb` + `presentation_pipeline_results.ipynb` for the advisor demo.
- Built `calibration_demo.ipynb` which walks through three calibrated videos end-to-end.

**Outcome.** v8 modestly outperforms v6 on the held-out clip; the bigger win was the foot-point fix, not the perspective correction itself.

---

## 2026-03 — Data collection + first fine-tune

- Recorded **26 iPhone clips** of single-person walking (volleyball court and around the campus, 10 s – 80 s each).
- Ran YOLOv8s on all 26, manually inspected the overlay videos:
    - 8 of 26 have detection drop-out when the subject is far away. Acceptable.
    - 3 of 26 have a phantom track from a passer-by. Filtered out in post-processing.
- Split into train / val / test, converted to Trajectron++ `.pkl`:
    - The 26 raw videos become **42 scenes** after gap-splitting (a clip with a 5+-frame detection gap is cut at the gap) and data augmentation (horizontal flip + 90°/270° rotation).
- Pre-trained an ETH base model for 100 epochs (`models_08_Mar_2026_23_53_32_eth_100ep`).
- Fine-tuned the base on the 42 scenes (v1 – v6, sweep of KL weight and augmentation).
- v6 (`+aug`) became the production model.

**Lesson.** A single-environment fine-tune is fragile without augmentation; flipping and 90° rotation roughly doubled the effective dataset and was the biggest single quality win.

---

## 2026-02 — First end-to-end attempt

- Got Trajectron++ running on the official ETH pedestrian dataset.
- Wrote `experiments/pedestrians/csv_to_trajectron_data.py` to import arbitrary CSV detections into the model's `.pkl` format.
- First fine-tune on a tiny self-collected clip → predictions were correct in shape but unrealistically small. Diagnosed as a scaling problem (pixel-space training, metre-space inference).

**Lesson.** Always train and infer in the same units — this motivated the entire calibration line of work.

---

## 2025-12 — Project setup

- Forked `StanfordASL/Trajectron-plus-plus`.
- Installed the upstream conda environment (`trajectron++`, Python 3.6).
- Reproduced the ETH results.
- Read the original paper and supplementary.

---

## Known issues / debt

1. **Code lives in two places.** `~/Desktop/uav_perception/` is not yet under version control with this repo. Plan: move it under `perception/` in a follow-up commit, with a thin compatibility shim.
2. **Some notebooks have stale outputs.** They'll re-execute cleanly with the current `conda` env, but I haven't re-run all of them since the foot-point fix.
3. **Model checkpoints are large** (~6 – 16 MB each). Listed in `.gitignore`; the few important ones can be uploaded to a separate releases page later.
4. **Hand-gesture module not started.** First milestone after VIO.
5. **No proper test set on real UAV footage.** All videos are iPhone-mounted simulations. A real (small) UAV recording session is planned for the next field session.

---

## Future work

- Replace CamTrackAR with a custom iOS app (more control over export format, possibility of streaming).
- Migrate to a more modern forecasting model (e.g. Adaptive Prediction, the official Trajectron++ successor referenced in the upstream README).
- Add the hand-gesture module and evaluate gesture-conditioned forecasting.
- Quantitative comparison: homography GT vs VIO GT on the *same* clip, *same* Trajectron++ model.
