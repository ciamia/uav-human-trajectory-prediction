"""Render YOLO overlay video from an existing detection CSV.

This avoids rerunning YOLO when we already have `foot_pixels.csv`.

Usage:
    python -m vio.render_overlay_from_csv \
        --video  /path/to/input.mp4 \
        --csv    data/vio/field_001/foot_pixels.csv \
        --output data/vio/field_001/overlay_from_csv.mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


def _load_detections(csv_path: Path) -> dict[int, dict]:
    det = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["frame"]))
            det[frame] = {
                "u": float(row["u"]),
                "v": float(row["v"]),
                "x1": float(row.get("bbox_x1", row["u"])),
                "y1": float(row.get("bbox_y1", row["v"])),
                "x2": float(row.get("bbox_x2", row["u"])),
                "y2": float(row.get("bbox_y2", row["v"])),
                "score": float(row.get("score", 0.0)),
                "track_id": int(float(row.get("track_id", 0))),
            }
    return det


def run(video: Path, csv_path: Path, output: Path) -> Path:
    det = _load_detections(csv_path)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        d = det.get(frame_idx)
        if d is not None:
            x1, y1, x2, y2 = int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])
            u, v = int(d["u"]), int(d["v"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 230, 0), 2)
            cv2.circle(frame, (u, v), 6, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"id {d['track_id']} conf {d['score']:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
        cv2.putText(
            frame,
            f"frame {frame_idx}/{max(1, n-1)}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Wrote overlay: {output}")
    return output


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.video, args.csv, args.output)
