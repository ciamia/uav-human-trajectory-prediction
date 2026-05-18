# Progress log

Running log of what was done, when, and why. Most recent at the top.

---

## VIO pilot — three validations + two field clips (May 2026, just before the meeting)

Recorded with the iPhone + CamTrackAR. All artefacts under `data/vio/`, all driver scripts under `vio/`, full notebook walkthrough in [`notebooks/03_vio_pilot.ipynb`](../notebooks/03_vio_pilot.ipynb).

**Validation (tape-measure walks, camera in hand).**

| target | XZ displacement reported by VIO | error |
| --- | --- | --- |
| 1 m  | 1.097 m | +9.7 % |
| 5 m  | 4.963 m | −0.75 % |
| 10 m | 9.861 m | −1.39 % |

Sub-percent error from 5 m onward → ARKit metric scale is good enough for our walking-distance ground truth. The 1 m result is biased by start-of-trace body sway (the walked distance and the sway have similar magnitudes).

**Field clip 1 (`data/vio/field_001/`)** — phone on a table, single person walks for 44 s within 3–13 m. YOLOv8s detected the person in 83.6 % of frames; ray-cast to the ARKit floor plane gave a clean trajectory. Trajectron++ v8 fine-tune (ckpt 45) on the resampled track:

- ADE (mean of samples) 1.16 m, FDE 2.05 m, best-of-20 ADE 0.78 m on a 4.8 s horizon.

**Field clip 2 (`data/vio/field_002/`)** — same setup, person walked further (22–60 m). Per-frame ray-cast became noisy at that range (1 px YOLO error → tens of cm world error because the rays are nearly parallel to the floor). After a 2 s rolling-median filter the trajectory looks physical (≈ 2.45 m/s, brisk walk). Trajectron++ predictions are correspondingly worse:

- ADE 3.78 m, FDE 5.79 m, best-of-20 ADE 2.99 m.

**Conclusions for the meeting.**

1. VIO replaces homography as the metric ground-truth source; no per-scene calibration required.
2. The model works end-to-end on VIO outputs without retraining.
3. The next data-collection pass needs the person to stay closer to the camera (or move to multi-view ground truth) to avoid the long-range ray-cast amplification.

---

## VIO research + module split (planning, mid-May 2026)

- Read into VIO theory (VINS-Mono, OKVIS, ORB-SLAM3). Summary in [`VIO_RESEARCH.md`](VIO_RESEARCH.md).
- Decided to use **ARKit on iPhone 11** for now via the free **CamTrackAR** app. Reasons: zero hardware cost, IMU + camera already present, exports per-frame pose CSV.
- Installed CamTrackAR, recorded a small in-room validation clip.
- First-look validation: walked a tape-measured **1.72 m** → ARKit reported **1.73 m** along the Z-axis → ~0.6 % error. Lateral (X) deviation ~60 cm explained by handheld body sway; vertical drift sub-cm.
- Designed the new pipeline: keep YOLO and Trajectron++ as-is, replace homography with a `pixel_to_world_floor()` ray-cast that uses the per-frame ARKit pose until the next meeting.
- Decided that **hand tracking and trajectory forecasting** will live in two **independent modules**.



---

## Ranking module + perspective correction + calibration (homogropy trials)

- Implemented six GT-free ranking methods (log-lik, mean prox, KDE, endpoint centrality, history consistency, risk).
- Realised that the bounding-box centre was being used as the person's position; switched to **foot point** (bottom-centre of the YOLO box).
- Built `prediction_analysis_v8.ipynb` + `presentation_pipeline_results.ipynb` for the advisor demo.
- Built `calibration_demo.ipynb` which walks through three calibrated videos end-to-end.


---

## Data collection + first fine-tune

- Recorded **26 iPhone clips** of single-person walking (volleyball court and around the campus, 10 s – 80 s each).
- Ran YOLOv8s on all 26, manually inspected the overlay videos:
    - 8 of 26 have detection drop-out when the subject is far away. Acceptable.
    - 3 of 26 have a phantom track from a passer-by. Filtered out in post-processing.
- Split into train / val / test, converted to Trajectron++ `.pkl`:
    - The 26 raw videos become **42 scenes** after gap-splitting (a clip with a 5+-frame detection gap is cut at the gap) and data augmentation (horizontal flip + 90°/270° rotation).
- Pre-trained an ETH base model for 100 epochs (`models_08_Mar_2026_23_53_32_eth_100ep`).
- Fine-tuned the base on the 42 scenes (v1 – v6, sweep of KL weight and augmentation).
- v6 (`+aug`) became the production model.



---

## First end-to-end attempt

- Got Trajectron++ running on the official ETH pedestrian dataset.
- Wrote `experiments/pedestrians/csv_to_trajectron_data.py` to import arbitrary CSV detections into the model's `.pkl` format.
- First fine-tune on a tiny self-collected clip → predictions were correct in shape but unrealistically small. Diagnosed as a scaling problem (pixel-space training, metre-space inference).

**Lesson.** Always train and infer in the same units — this motivated the entire calibration line of work.

---

## Project setup

- Forked `StanfordASL/Trajectron-plus-plus`.
- Installed the upstream conda environment (`trajectron++`, Python 3.6).
- Reproduced the ETH results.
- Read the original paper and supplementary.

---

issues

1. **Code lives in two places for now to be fixed.** `~/Desktop/uav_perception/` is not yet under version control with this repo. Plan: move it under `perception/` in a follow-up commit, with a thin compatibility shim.
2. **Some notebooks have stale outputs.** They'll re-execute cleanly with the current `conda` env, but I haven't re-run all of them since the foot-point fix.
3. **Model checkpoints are large** (~6 – 16 MB each). Listed in `.gitignore`; the few important ones can be uploaded to a separate releases page later.
4. **Hand-gesture module just started.** First milestone after VIO.
5. **No proper test set on real UAV footage.** All videos are iPhone-mounted simulations. A real (small) UAV recording session is planned for the next field session.

---


