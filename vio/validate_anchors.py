"""Compute distances between anchors in a CamTrackAR Anchors.CSV.

Use this to validate VIO accuracy: place two physical anchors at a known
tape-measured distance, then compare to what the app reports.

Usage:
    python -m vio.validate_anchors --anchors-csv /path/to/Anchors.CSV
    python -m vio.validate_anchors --anchors-csv /path/to/Anchors.CSV --expected 1.72

If `--expected` is given, the script reports the error in metres and %.
If there are more than two anchors, distances are reported pairwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def horizontal(a: np.ndarray, b: np.ndarray) -> float:
    """Distance ignoring Y (floor plane). Useful when subject stayed at one height."""
    a_h = np.array([a[0], 0.0, a[2]])
    b_h = np.array([b[0], 0.0, b[2]])
    return float(np.linalg.norm(b_h - a_h))


def run(anchors_csv: Path, expected: float | None = None) -> None:
    df = pd.read_csv(anchors_csv)
    df.columns = [c.strip() for c in df.columns]

    required = {"PosX", "PosY", "PosZ"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Anchors CSV is missing columns: {sorted(missing)}")

    if len(df) < 2:
        print(f"Need at least 2 anchors, got {len(df)}")
        return

    print(f"Loaded {len(df)} anchors from {anchors_csv}")
    print()
    print(f"{'Name':<20} {'PosX':>8} {'PosY':>8} {'PosZ':>8}")
    print("-" * 50)
    for _, row in df.iterrows():
        name = str(row.get("Name", "?"))
        print(f"{name:<20} {row.PosX:>8.3f} {row.PosY:>8.3f} {row.PosZ:>8.3f}")
    print()

    points = df[["PosX", "PosY", "PosZ"]].to_numpy()
    names = df.get("Name", pd.Series([str(i) for i in range(len(df))])).to_numpy()

    print(f"{'From':<20} {'To':<20} {'3D dist':>10} {'XZ dist':>10}")
    print("-" * 65)
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            d3 = euclidean(points[i], points[j])
            dxz = horizontal(points[i], points[j])
            print(f"{str(names[i]):<20} {str(names[j]):<20} {d3:>10.4f} {dxz:>10.4f}")
    print()

    if expected is not None and len(df) == 2:
        d3 = euclidean(points[0], points[1])
        dxz = horizontal(points[0], points[1])
        err3 = d3 - expected
        errxz = dxz - expected
        pct3 = 100 * err3 / expected
        pctxz = 100 * errxz / expected
        print(f"Expected distance: {expected:.4f} m")
        print(f"  3D measured:    {d3:.4f} m   error {err3:+.4f} m ({pct3:+.2f} %)")
        print(f"  XZ (floor) measured: {dxz:.4f} m   error {errxz:+.4f} m ({pctxz:+.2f} %)")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--anchors-csv", required=True, type=Path)
    p.add_argument("--expected", type=float, default=None,
                   help="Expected distance in metres (only used when exactly 2 anchors)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.anchors_csv, expected=args.expected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
