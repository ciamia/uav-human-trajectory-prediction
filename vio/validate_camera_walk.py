"""Measure the distance the camera (phone) moved during a CamTrackAR recording.

Use this for the tape-measure validation tests where you held the phone
and walked a known distance — you didn't drop two anchors, you just walked
and recorded. We then compare the total walked distance (or, optionally,
the straight-line first-to-last distance) against the expected ground truth.

Usage:
    python -m vio.validate_camera_walk \
        --camera-csv /path/to/CameraKeyframes.CSV \
        --expected 1.00
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_keyframes(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=["PosX", "PosY", "PosZ"])
    return df.sort_values("TimeSeconds").reset_index(drop=True)


def summarise(camera_csv: Path, expected: float | None = None) -> dict:
    df = load_keyframes(camera_csv)
    if len(df) < 2:
        raise ValueError(f"{camera_csv}: not enough keyframes ({len(df)})")

    pos = df[["PosX", "PosY", "PosZ"]].to_numpy()
    first = pos[0]
    last = pos[-1]

    # Straight-line displacement first -> last (3D and floor projection).
    disp3 = float(np.linalg.norm(last - first))
    disp_xz = float(np.linalg.norm([last[0] - first[0], last[2] - first[2]]))

    # Cumulative path length (sum of per-frame deltas).
    deltas = np.diff(pos, axis=0)
    step_3d = np.linalg.norm(deltas, axis=1)
    step_xz = np.linalg.norm(deltas[:, [0, 2]], axis=1)
    path_3d = float(np.sum(step_3d))
    path_xz = float(np.sum(step_xz))

    duration = float(df["TimeSeconds"].iloc[-1] - df["TimeSeconds"].iloc[0])

    result = {
        "file": str(camera_csv),
        "n_frames": len(df),
        "duration_s": duration,
        "displacement_3d_m": disp3,
        "displacement_xz_m": disp_xz,
        "path_length_3d_m": path_3d,
        "path_length_xz_m": path_xz,
        "start_pos": first.tolist(),
        "end_pos": last.tolist(),
    }

    if expected is not None:
        result["expected_m"] = float(expected)
        # The user walked in a straight line, so straight-line displacement is
        # the right number to compare with the tape measure.
        result["error_xz_m"] = result["displacement_xz_m"] - expected
        result["error_xz_pct"] = 100.0 * result["error_xz_m"] / expected
        result["error_3d_m"] = result["displacement_3d_m"] - expected
        result["error_3d_pct"] = 100.0 * result["error_3d_m"] / expected

    return result


def pretty_print(result: dict) -> None:
    print(f"\n=== {result['file']} ===")
    print(f"  keyframes:        {result['n_frames']}")
    print(f"  duration:         {result['duration_s']:.2f} s")
    print(f"  start:            ({result['start_pos'][0]:+.3f}, {result['start_pos'][1]:+.3f}, {result['start_pos'][2]:+.3f}) m")
    print(f"  end:              ({result['end_pos'][0]:+.3f}, {result['end_pos'][1]:+.3f}, {result['end_pos'][2]:+.3f}) m")
    print(f"  3D displacement:  {result['displacement_3d_m']:.3f} m")
    print(f"  XZ displacement:  {result['displacement_xz_m']:.3f} m  (drops the vertical component)")
    print(f"  3D path length:   {result['path_length_3d_m']:.3f} m  (sum of per-frame steps)")
    print(f"  XZ path length:   {result['path_length_xz_m']:.3f} m")
    if "expected_m" in result:
        print(f"  expected:         {result['expected_m']:.3f} m")
        print(f"  error (XZ disp):  {result['error_xz_m']:+.3f} m  ({result['error_xz_pct']:+.2f} %)")
        print(f"  error (3D disp):  {result['error_3d_m']:+.3f} m  ({result['error_3d_pct']:+.2f} %)")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-csv", required=True, type=Path)
    p.add_argument("--expected", type=float, default=None,
                   help="Expected straight-line distance walked, in metres.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    res = summarise(args.camera_csv, expected=args.expected)
    pretty_print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
