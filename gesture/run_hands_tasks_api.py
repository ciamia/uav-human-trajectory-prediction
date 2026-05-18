"""Experiment 1 — MediaPipe **Tasks API** HandLandmarker runner.

Drop-in replacement for `gesture/mediapipe_overlay.py` but uses the newer
`mediapipe.tasks.vision.HandLandmarker` (TFLite-based, GPU delegate
support, more configurable than the legacy `mp.solutions.hands`).

Outputs use the same schema as the legacy runner so downstream notebooks
can load both side-by-side:

    `<output_prefix>_overlay.mp4`
    `<output_prefix>_keypoints.csv`   # same columns: frame, time_seconds,
                                       # hand_index, handedness, score,
                                       # kp0_x, kp0_y, kp0_z, ..., kp20_z

Usage:
    python -m gesture.run_hands_tasks_api \
        --video /path/to/clip.mov \
        --output-prefix data/gesture/tasks/1m \
        --model models/hand_landmarker.task
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

NUM_LANDMARKS = 21

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def _keypoint_header() -> list[str]:
    header = ["frame", "time_seconds", "hand_index", "handedness", "score"]
    for i in range(NUM_LANDMARKS):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"]
    return header


def _draw_landmarks(frame_bgr, landmarks_norm, w, h):
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_norm]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, pts[a], pts[b], (0, 200, 60), 2, cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(frame_bgr, (x, y), 3, (40, 40, 220), -1, cv2.LINE_AA)


def run(
    video_path: Path,
    output_prefix: Path,
    model_path: Path,
    max_hands: int = 2,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
) -> tuple[Path, Path]:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

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
    print(f"Overlay video -> {overlay_path}")
    print(f"Keypoint CSV  -> {keypoints_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=max_hands,
        min_hand_detection_confidence=detection_confidence,
        min_hand_presence_confidence=detection_confidence,
        min_tracking_confidence=tracking_confidence,
    )

    csv_file = open(keypoints_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(_keypoint_header())

    n_detected_frames = 0
    n_hand_instances = 0

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            time_seconds = frame_idx / fps if fps > 0 else 0.0
            timestamp_ms = int(time_seconds * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                n_detected_frames += 1
                for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):
                    handedness_label = ""
                    handedness_score = 0.0
                    if result.handedness and hand_idx < len(result.handedness):
                        cls = result.handedness[hand_idx][0]
                        handedness_label = cls.category_name
                        handedness_score = float(cls.score)

                    row = [frame_idx, time_seconds, hand_idx, handedness_label, handedness_score]
                    for lm in hand_landmarks:
                        row += [lm.x, lm.y, lm.z]
                    csv_writer.writerow(row)
                    n_hand_instances += 1

                    _draw_landmarks(frame_bgr, hand_landmarks, width, height)

            writer.write(frame_bgr)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  ... processed {frame_idx} / {total_frames} frames")

    cap.release()
    writer.release()
    csv_file.close()

    rate = 100.0 * n_detected_frames / max(1, frame_idx)
    print()
    print(f"Done. Frames processed: {frame_idx}")
    print(f"  Frames with >= 1 hand detected: {n_detected_frames} ({rate:.1f} %)")
    print(f"  Total hand instances written:   {n_hand_instances}")
    return overlay_path, keypoints_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output-prefix", required=True, type=Path)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--detection-confidence", type=float, default=0.5)
    p.add_argument("--tracking-confidence", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = _parse_args(argv)
    run(
        video_path=a.video,
        output_prefix=a.output_prefix,
        model_path=a.model,
        max_hands=a.max_hands,
        detection_confidence=a.detection_confidence,
        tracking_confidence=a.tracking_confidence,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
