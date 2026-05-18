"""Convert pixel detections to metric world coordinates using ARKit / CamTrackAR camera pose.

Pipeline:
    1. Load per-frame camera pose from CamTrackAR's `CameraKeyframes.CSV`.
    2. Load per-frame foot pixels (from YOLO) for the tracked person.
    3. For each frame, build a ray from the camera through the foot pixel,
       transform it into the world frame, and intersect it with the ground
       plane Y = 0 (calibrated by CamTrackAR's "Set Floor" feature).
    4. Write a CSV with the resulting world coordinates (x, z) in metres.

ARKit conventions:
    World frame:   +X right, +Y up (gravity-aligned), +Z toward the user.
    Camera frame:  +X right, +Y up, camera looks down -Z (OpenGL convention).
    The pose given by ARKit is `(world position, world rotation)` of the camera.

Usage:
    python -m vio.pixel_to_world \
        --camera-csv /path/to/CameraKeyframes.CSV \
        --pixels-csv /path/to/foot_pixels.csv \
        --video-width 1920 --video-height 1440 \
        --output /path/to/world_trajectory.csv

`foot_pixels.csv` is expected to have columns:
    frame, time_seconds, track_id, u, v

where (u, v) is the foot pixel (bottom-centre of the YOLO bounding box).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  Math helpers
# --------------------------------------------------------------------------- #

def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Returns the rotation that maps a vector in the camera frame to the
    world frame.
    """
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if n < 1e-9:
        return np.eye(3)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz),     2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [    2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz),     2 * (qy * qz - qx * qw)],
        [    2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])


def intrinsics_from_fov(fov: float, w: int, h: int, fov_unit: str = "radians") -> tuple[float, float, float, float]:
    """Build pinhole intrinsics (fx, fy, cx, cy) from a horizontal FOV.

    `fov_unit` is either "radians" (CamTrackAR's CSV) or "degrees" (user override).
    """
    fov_rad = fov if fov_unit == "radians" else math.radians(fov)
    fx = (w / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


def intrinsics_from_lens_zoom(lens_zoom_px: float, w: int, h: int) -> tuple[float, float, float, float]:
    """CamTrackAR's `LensZoom` is the focal length in pixels for this video resolution.

    This is the most direct source of intrinsics — no FOV unit ambiguity.
    """
    return float(lens_zoom_px), float(lens_zoom_px), w / 2.0, h / 2.0


def pixel_to_world_floor(
    uv: tuple[float, float],
    cam_pos: np.ndarray,
    R_world_from_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    floor_y: float = 0.0,
) -> np.ndarray | None:
    """Cast a ray from the camera through pixel (u, v), intersect with Y = floor_y.

    Returns the 3D world coordinates of the intersection, or None if the ray
    is parallel to the floor or points away from it.
    """
    u, v = uv

    # Ray in camera frame.
    # ARKit / OpenGL convention: +X right, +Y up, camera looks down -Z.
    # Pixel v grows downward, so we flip its sign to align with camera +Y.
    x_cam = (u - cx) / fx
    y_cam = -(v - cy) / fy
    z_cam = -1.0

    ray_cam = np.array([x_cam, y_cam, z_cam], dtype=float)
    ray_cam /= np.linalg.norm(ray_cam)

    # Transform to world frame.
    ray_world = R_world_from_cam @ ray_cam

    # Intersect with the horizontal plane Y = floor_y.
    if abs(ray_world[1]) < 1e-6:
        return None
    t = (floor_y - cam_pos[1]) / ray_world[1]
    if t <= 0:
        # Ray points away from the floor (above or behind the camera).
        return None

    return cam_pos + t * ray_world


# --------------------------------------------------------------------------- #
#  Pose lookup
# --------------------------------------------------------------------------- #

def load_camera_keyframes(csv_path: Path) -> pd.DataFrame:
    """Load CamTrackAR's CameraKeyframes.CSV and normalise column names."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    required = {"TimeSeconds", "PosX", "PosY", "PosZ", "QuatX", "QuatY", "QuatZ", "QuatW"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Camera CSV is missing columns: {sorted(missing)}")
    return df.sort_values("TimeSeconds").reset_index(drop=True)


def interpolate_pose(
    time_seconds: float,
    cam_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    """Return (camera position, rotation matrix, FOV [rad], LensZoom [px]) at the given time.

    Position is linearly interpolated between the two surrounding keyframes;
    rotation is taken from the nearest keyframe (good enough at 60+ Hz).
    FOV and LensZoom are taken from the nearest keyframe.
    """
    t_arr = cam_df["TimeSeconds"].values
    idx = np.searchsorted(t_arr, time_seconds)
    idx = max(0, min(idx, len(cam_df) - 1))
    lower = max(0, idx - 1)
    upper = min(len(cam_df) - 1, idx)

    if lower == upper:
        row = cam_df.iloc[lower]
        cam_pos = np.array([row.PosX, row.PosY, row.PosZ])
    else:
        a = cam_df.iloc[lower]
        b = cam_df.iloc[upper]
        denom = b.TimeSeconds - a.TimeSeconds
        if denom <= 0:
            alpha = 0.0
        else:
            alpha = (time_seconds - a.TimeSeconds) / denom
            alpha = max(0.0, min(1.0, alpha))
        cam_pos = np.array([
            a.PosX + alpha * (b.PosX - a.PosX),
            a.PosY + alpha * (b.PosY - a.PosY),
            a.PosZ + alpha * (b.PosZ - a.PosZ),
        ])

    # Use the nearest keyframe for rotation and FOV.
    nearest = cam_df.iloc[upper]
    R = quat_to_rotmat(nearest.QuatX, nearest.QuatY, nearest.QuatZ, nearest.QuatW)
    fov_rad = float(nearest.FOV) if "FOV" in cam_df.columns else math.radians(60.0)
    lens_zoom = float(nearest.LensZoom) if "LensZoom" in cam_df.columns else None
    return cam_pos, R, fov_rad, lens_zoom


# --------------------------------------------------------------------------- #
#  Main pipeline
# --------------------------------------------------------------------------- #

def run(
    camera_csv: Path,
    pixels_csv: Path,
    video_width: int,
    video_height: int,
    output_csv: Path,
    fov_override: float | None = None,
    floor_y: float = 0.0,
) -> pd.DataFrame:
    """Project pixel detections to world coordinates and write a CSV."""
    cam_df = load_camera_keyframes(camera_csv)

    px_df = pd.read_csv(pixels_csv)
    px_df.columns = [c.strip() for c in px_df.columns]
    required = {"frame", "time_seconds", "u", "v"}
    missing = required - set(px_df.columns)
    if missing:
        raise ValueError(f"Pixel CSV is missing columns: {sorted(missing)}")

    if "track_id" not in px_df.columns:
        px_df["track_id"] = 0

    rows = []
    skipped = 0
    for _, row in px_df.iterrows():
        cam_pos, R, fov_rad, lens_zoom = interpolate_pose(float(row.time_seconds), cam_df)
        if fov_override is not None:
            # User passed an FOV in degrees on the CLI.
            fx, fy, cx, cy = intrinsics_from_fov(fov_override, video_width, video_height, fov_unit="degrees")
        elif lens_zoom is not None:
            # Prefer LensZoom (already in pixels) — no FOV unit ambiguity.
            fx, fy, cx, cy = intrinsics_from_lens_zoom(lens_zoom, video_width, video_height)
        else:
            # Fall back to FOV column (radians).
            fx, fy, cx, cy = intrinsics_from_fov(fov_rad, video_width, video_height, fov_unit="radians")

        p = pixel_to_world_floor(
            (float(row.u), float(row.v)),
            cam_pos, R,
            fx, fy, cx, cy,
            floor_y=floor_y,
        )
        if p is None:
            skipped += 1
            continue

        rows.append({
            "frame": int(row.frame),
            "time_seconds": float(row.time_seconds),
            "track_id": int(row.track_id),
            "u": float(row.u),
            "v": float(row.v),
            "world_x": float(p[0]),
            "world_y": float(p[1]),  # should be ~floor_y
            "world_z": float(p[2]),
            "cam_pos_x": float(cam_pos[0]),
            "cam_pos_y": float(cam_pos[1]),
            "cam_pos_z": float(cam_pos[2]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(output_csv, index=False)
    print(f"Wrote {len(out)} world-frame samples to {output_csv}")
    if skipped:
        print(f"Skipped {skipped} samples (ray did not intersect floor)")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera-csv", required=True, type=Path, help="CamTrackAR CameraKeyframes.CSV")
    p.add_argument("--pixels-csv", required=True, type=Path, help="YOLO foot-pixel CSV (frame, time_seconds, track_id, u, v)")
    p.add_argument("--video-width", required=True, type=int)
    p.add_argument("--video-height", required=True, type=int)
    p.add_argument("--output", required=True, type=Path, help="Output world-trajectory CSV")
    p.add_argument("--fov", type=float, default=None, help="Override FOV (degrees). Default: read from camera CSV")
    p.add_argument("--floor-y", type=float, default=0.0, help="Y value of the floor plane (default 0)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(
        camera_csv=args.camera_csv,
        pixels_csv=args.pixels_csv,
        video_width=args.video_width,
        video_height=args.video_height,
        output_csv=args.output,
        fov_override=args.fov,
        floor_y=args.floor_y,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
