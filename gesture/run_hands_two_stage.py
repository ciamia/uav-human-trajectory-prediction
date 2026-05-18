"""Experiment 2 — Two-stage **YOLOv8-person + MediaPipe Hands** runner.

The key insight: MediaPipe Hands drops detection rate sharply past 2 m
because the hand is then only ~5 % of the frame. If we **first detect
the person with YOLO**, **crop tightly to that person**, and run MediaPipe
on the crop, then from MediaPipe's perspective the hand suddenly fills
20 - 40 % of *its* input again — i.e. we recover the near-range regime.

Pipeline per frame:
    1. YOLOv8 person detection (class 0 in COCO) -> bbox(es).
    2. Pick the largest person bbox (assumes a single subject of interest).
    3. Pad the bbox by 10 % and crop the frame -> `crop`.
    4. Run MediaPipe Hands on `crop` -> hand landmarks in crop coords.
    5. Re-project the landmarks back into the **full-frame** coordinates
       (so the CSV is comparable to the other runners).
    6. Draw both the YOLO bbox and the re-projected hand skeleton on the
       full-resolution overlay.

Outputs match the schema of `gesture/mediapipe_overlay.py` exactly:

    `<output_prefix>_overlay.mp4`
    `<output_prefix>_keypoints.csv`

Usage:
    python -m gesture.run_hands_two_stage \
        --video /path/to/clip.mov \
        --output-prefix data/gesture/twostage/1m \
        --yolo-weights ~/Desktop/uav_perception/yolov8s.pt
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

NUM_LANDMARKS = 21


def _keypoint_header() -> list[str]:
    header = ["frame", "time_seconds", "hand_index", "handedness", "score"]
    for i in range(NUM_LANDMARKS):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"]
    return header


def _expand_bbox(x1, y1, x2, y2, w, h, pad_ratio=0.10):
    bw = x2 - x1
    bh = y2 - y1
    x1 = max(0, int(x1 - pad_ratio * bw))
    y1 = max(0, int(y1 - pad_ratio * bh))
    x2 = min(w, int(x2 + pad_ratio * bw))
    y2 = min(h, int(y2 + pad_ratio * bh))
    return x1, y1, x2, y2


def _largest_person_box(yolo_result, frame_w, frame_h):
    if yolo_result is None or yolo_result.boxes is None:
        return None
    best = None
    best_area = 0
    for b in yolo_result.boxes:
        cls = int(b.cls.item())
        if cls != 0:  # COCO 0 = person
            continue
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2, float(b.conf.item()))
    return best


def run(
    video_path: Path,
    output_prefix: Path,
    yolo_weights: Path,
    max_hands: int = 2,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
    person_conf: float = 0.30,
    crop_pad: float = 0.10,
) -> tuple[Path, Path]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    overlay_path = output_prefix.with_name(output_prefix.name + "_overlay.mp4")
    keypoints_path = output_prefix.with_name(output_prefix.name + "_keypoints.csv")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {video_path}")
    print(f"  resolution = {width}x{height}, fps = {fps:.2f}, frames = {total_frames}")
    print(f"YOLO weights: {yolo_weights}")
    print(f"Overlay video -> {overlay_path}")
    print(f"Keypoint CSV  -> {keypoints_path}")

    yolo = YOLO(str(yolo_weights))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    mp_hands = mp.solutions.hands

    csv_file = open(keypoints_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(_keypoint_header())

    n_frames_with_person = 0
    n_frames_with_hand = 0
    n_hand_instances = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=max_hands,
        min_detection_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    ) as hands:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            time_seconds = frame_idx / fps if fps > 0 else 0.0

            # Stage 1: YOLO person detection (largest bbox only).
            yres = yolo.predict(frame_bgr, conf=person_conf, classes=[0], verbose=False)[0]
            person_box = _largest_person_box(yres, width, height)

            if person_box is not None:
                n_frames_with_person += 1
                x1, y1, x2, y2, p_conf = person_box
                cx1, cy1, cx2, cy2 = _expand_bbox(x1, y1, x2, y2, width, height, crop_pad)
                crop = frame_bgr[cy1:cy2, cx1:cx2]

                # Stage 2: MediaPipe Hands on the crop.
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_rgb.flags.writeable = False
                results = hands.process(crop_rgb)
                crop_rgb.flags.writeable = True

                if results.multi_hand_landmarks:
                    n_frames_with_hand += 1
                    for hand_idx, hl in enumerate(results.multi_hand_landmarks):
                        handedness_label = ""
                        handedness_score = 0.0
                        if results.multi_handedness and hand_idx < len(results.multi_handedness):
                            cls = results.multi_handedness[hand_idx].classification[0]
                            handedness_label = cls.label
                            handedness_score = float(cls.score)

                        # Re-project crop-relative normalised coords into full-frame normalised coords.
                        crop_w = cx2 - cx1
                        crop_h = cy2 - cy1
                        kp_xy_world = []
                        for lm in hl.landmark:
                            x_full = (cx1 + lm.x * crop_w) / width
                            y_full = (cy1 + lm.y * crop_h) / height
                            kp_xy_world.append((x_full, y_full, lm.z))

                        row = [frame_idx, time_seconds, hand_idx, handedness_label, handedness_score]
                        for x, y, z in kp_xy_world:
                            row += [x, y, z]
                        csv_writer.writerow(row)
                        n_hand_instances += 1

                        # Draw on full-resolution overlay.
                        pts_px = [(int(x * width), int(y * height)) for x, y, _ in kp_xy_world]
                        for a, b in [
                            (0, 1), (1, 2), (2, 3), (3, 4),
                            (0, 5), (5, 6), (6, 7), (7, 8),
                            (5, 9), (9, 10), (10, 11), (11, 12),
                            (9, 13), (13, 14), (14, 15), (15, 16),
                            (13, 17), (17, 18), (18, 19), (19, 20),
                            (0, 17),
                        ]:
                            cv2.line(frame_bgr, pts_px[a], pts_px[b], (0, 200, 60), 2, cv2.LINE_AA)
                        for x, y in pts_px:
                            cv2.circle(frame_bgr, (x, y), 4, (40, 40, 220), -1, cv2.LINE_AA)

                # Draw the person bbox + crop region on the overlay.
                cv2.rectangle(frame_bgr, (cx1, cy1), (cx2, cy2), (180, 180, 0), 2)
                cv2.putText(frame_bgr, f"person {p_conf:.2f}", (cx1, max(15, cy1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 0), 2, cv2.LINE_AA)

            writer.write(frame_bgr)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  ... processed {frame_idx} / {total_frames} frames")

    cap.release()
    writer.release()
    csv_file.close()

    p_rate = 100.0 * n_frames_with_person / max(1, frame_idx)
    h_rate = 100.0 * n_frames_with_hand / max(1, frame_idx)
    print()
    print(f"Done. Frames processed: {frame_idx}")
    print(f"  Frames with a person:           {n_frames_with_person} ({p_rate:.1f} %)")
    print(f"  Frames with >= 1 hand detected: {n_frames_with_hand} ({h_rate:.1f} %)")
    print(f"  Total hand instances written:   {n_hand_instances}")
    return overlay_path, keypoints_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output-prefix", required=True, type=Path)
    p.add_argument("--yolo-weights", required=True, type=Path,
                   help="Path to a YOLOv8 weights file (e.g. yolov8s.pt)")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--detection-confidence", type=float, default=0.5)
    p.add_argument("--tracking-confidence", type=float, default=0.5)
    p.add_argument("--person-conf", type=float, default=0.30)
    p.add_argument("--crop-pad", type=float, default=0.10,
                   help="Padding ratio when expanding the person bbox before crop.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    run(
        video_path=a.video,
        output_prefix=a.output_prefix,
        yolo_weights=a.yolo_weights,
        max_hands=a.max_hands,
        detection_confidence=a.detection_confidence,
        tracking_confidence=a.tracking_confidence,
        person_conf=a.person_conf,
        crop_pad=a.crop_pad,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
