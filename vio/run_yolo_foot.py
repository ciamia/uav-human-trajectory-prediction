"""Run YOLOv8 person detection on a video and write a foot-pixel CSV.

The output CSV has exactly the columns expected by `vio/pixel_to_world.py`:

    frame, time_seconds, track_id, u, v

`(u, v)` is the foot pixel (bottom-centre of the bounding box).

For single-person videos (which is what we have), this script picks the
largest person detection in each frame and assigns it `track_id = 0`.
For multi-person scenes a proper tracker (BoT-SORT, ByteTrack) would be
required — out of scope for the thesis pilot.

Usage:
    python -m vio.run_yolo_foot \
        --video       /path/to/clip.mp4 \
        --output      /path/to/foot_pixels.csv \
        --overlay-video /path/to/overlay.mp4    # optional

Requires `ultralytics` (YOLOv8) and `opencv-python`:
    pip install ultralytics opencv-python
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics is required. Install with: pip install ultralytics", file=sys.stderr)
    raise


PERSON_CLASS = 0  # COCO class id for "person"


def run(
    video_path: Path,
    output_csv: Path,
    overlay_video: Path | None = None,
    model_name: str = "yolov8s.pt",
    conf: float = 0.35,
    imgsz: int = 640,
) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}\n  resolution = {w}x{h}, fps = {fps:.2f}, frames = {total}")

    writer = None
    if overlay_video is not None:
        overlay_video.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(overlay_video), fourcc, fps, (w, h))

    f = open(output_csv, "w", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "time_seconds", "track_id", "u", "v",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "score"])

    frame_idx = 0
    n_detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / fps if fps > 0 else 0.0
        results = model.predict(frame, imgsz=imgsz, conf=conf, classes=[PERSON_CLASS], verbose=False)

        best = None
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int(areas.argmax())
            best = (boxes[i], float(scores[i]))

        if best is not None:
            (x1, y1, x2, y2), score = best
            u = float((x1 + x2) / 2.0)
            v = float(y2)
            csv_writer.writerow([frame_idx, f"{t:.6f}", 0, f"{u:.2f}", f"{v:.2f}",
                                 f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}", f"{score:.3f}"])
            n_detected += 1

            if writer is not None:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.circle(frame, (int(u), int(v)), 6, (0, 0, 255), -1)
                cv2.putText(frame, f"conf {score:.2f}", (int(x1), max(20, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if writer is not None:
            writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  processed {frame_idx}/{total}")

    cap.release()
    f.close()
    if writer is not None:
        writer.release()

    rate = 100 * n_detected / max(1, frame_idx)
    print(f"Done. Detections: {n_detected}/{frame_idx} ({rate:.1f}%) -> {output_csv}")
    if overlay_video is not None:
        print(f"Overlay video: {overlay_video}")
    return output_csv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="Foot-pixel CSV path")
    p.add_argument("--overlay-video", type=Path, default=None, help="Optional overlay mp4")
    p.add_argument("--model", default="yolov8s.pt")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        video_path=args.video,
        output_csv=args.output,
        overlay_video=args.overlay_video,
        model_name=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
