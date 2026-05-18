"""Run MediaPipe Hands on a video and produce an overlay video + keypoint CSV.

This is the "Phase 1 sanity check" for the gesture / intent module
(see `docs/GESTURE_MODULE.md`). It just confirms that hand detection works
on our footage; classification comes in a later phase.

Outputs:
    * `<output>_overlay.mp4` — original video with detected hand landmarks
      and connections drawn on every frame.
    * `<output>_keypoints.csv` — one row per detected hand per frame:
        frame, time_seconds, hand_index, handedness, label, score, kp0_x, kp0_y, kp0_z, ..., kp20_x, kp20_y, kp20_z

Usage:
    python -m gesture.mediapipe_overlay \
        --video /path/to/clip.mp4 \
        --output-prefix /path/to/out/clip

Requires `mediapipe` and `opencv-python`:
    pip install mediapipe opencv-python
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

try:
    import cv2
except ImportError as e:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python", file=sys.stderr)
    raise

try:
    import mediapipe as mp
except ImportError as e:
    print("ERROR: mediapipe is required. Install with: pip install mediapipe", file=sys.stderr)
    raise


NUM_LANDMARKS = 21  # MediaPipe Hands returns 21 3-D landmarks per hand.


def _keypoint_header() -> list[str]:
    header = ["frame", "time_seconds", "hand_index", "handedness", "score"]
    for i in range(NUM_LANDMARKS):
        header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"]
    return header


def _keypoint_row(
    frame_idx: int,
    time_seconds: float,
    hand_index: int,
    handedness: str,
    score: float,
    landmarks,
) -> list:
    row = [frame_idx, time_seconds, hand_index, handedness, score]
    for lm in landmarks.landmark:
        row += [lm.x, lm.y, lm.z]
    return row


def run(
    video_path: Path,
    output_prefix: Path,
    max_hands: int = 2,
    detection_confidence: float = 0.5,
    tracking_confidence: float = 0.5,
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
    print(f"Overlay video → {overlay_path}")
    print(f"Keypoint CSV → {keypoints_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    csv_file = open(keypoints_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(_keypoint_header())

    n_detected_frames = 0
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
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                n_detected_frames += 1
                for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness_label = ""
                    handedness_score = 0.0
                    if results.multi_handedness and hand_idx < len(results.multi_handedness):
                        classification = results.multi_handedness[hand_idx].classification[0]
                        handedness_label = classification.label
                        handedness_score = float(classification.score)

                    csv_writer.writerow(_keypoint_row(
                        frame_idx, time_seconds, hand_idx,
                        handedness_label, handedness_score, hand_landmarks,
                    ))
                    n_hand_instances += 1

                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style(),
                    )

            writer.write(frame_bgr)
            frame_idx += 1

            if frame_idx % 100 == 0:
                print(f"  ... processed {frame_idx} / {total_frames} frames")

    cap.release()
    writer.release()
    csv_file.close()

    detection_rate = 100.0 * n_detected_frames / max(1, frame_idx)
    print()
    print(f"Done. Frames processed: {frame_idx}")
    print(f"  Frames with >= 1 hand detected: {n_detected_frames} ({detection_rate:.1f} %)")
    print(f"  Total hand instances written:   {n_hand_instances}")

    return overlay_path, keypoints_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output-prefix", required=True, type=Path,
                   help="Without extension. The script appends _overlay.mp4 and _keypoints.csv")
    p.add_argument("--max-hands", type=int, default=2)
    p.add_argument("--detection-confidence", type=float, default=0.5)
    p.add_argument("--tracking-confidence", type=float, default=0.5)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        video_path=args.video,
        output_prefix=args.output_prefix,
        max_hands=args.max_hands,
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
