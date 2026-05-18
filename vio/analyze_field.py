"""Quick analysis + plot of a VIO field clip.

Reads a world-trajectory CSV produced by `vio/pixel_to_world.py` and prints
summary stats + writes a top-down trajectory plot PNG.

Usage:
    python -m vio.analyze_field \
        --trajectory data/vio/field_001/world_trajectory.csv \
        --output     data/vio/field_001/trajectory_plot.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def smooth(df: pd.DataFrame, window: int = 31) -> pd.DataFrame:
    """Rolling-median smoothing of the world XZ track (robust to ray-cast outliers).
    `window` should be odd; ~0.5 s at 60 fps == 31."""
    out = df.copy()
    out["world_x"] = df["world_x"].rolling(window, center=True, min_periods=1).median()
    out["world_z"] = df["world_z"].rolling(window, center=True, min_periods=1).median()
    return out


def summarise(df: pd.DataFrame, smoothed_window: int = 31) -> dict:
    dx = df["world_x"].diff().fillna(0).to_numpy()
    dz = df["world_z"].diff().fillna(0).to_numpy()
    step = np.sqrt(dx ** 2 + dz ** 2)

    df_s = smooth(df, window=smoothed_window)
    dxs = df_s["world_x"].diff().fillna(0).to_numpy()
    dzs = df_s["world_z"].diff().fillna(0).to_numpy()
    step_s = np.sqrt(dxs ** 2 + dzs ** 2)

    duration = float(df["time_seconds"].max() - df["time_seconds"].min())
    return {
        "n_frames": int(len(df)),
        "duration_s": duration,
        "x_min": float(df["world_x"].min()),
        "x_max": float(df["world_x"].max()),
        "z_min": float(df["world_z"].min()),
        "z_max": float(df["world_z"].max()),
        "total_walked_m": float(step.sum()),
        "mean_speed_m_s": float(step.sum() / max(1e-3, duration)),
        "total_walked_smoothed_m": float(step_s.sum()),
        "mean_speed_smoothed_m_s": float(step_s.sum() / max(1e-3, duration)),
        "cam_pos_x_mean": float(df["cam_pos_x"].mean()),
        "cam_pos_y_mean": float(df["cam_pos_y"].mean()),
        "cam_pos_z_mean": float(df["cam_pos_z"].mean()),
        "cam_pos_x_std": float(df["cam_pos_x"].std()),
        "cam_pos_z_std": float(df["cam_pos_z"].std()),
    }


def _trim_for_display(df: pd.DataFrame, q_low: float = 0.01, q_high: float = 0.99) -> pd.DataFrame:
    """Trim extreme outliers for cleaner visualization only."""
    x0, x1 = df["world_x"].quantile(q_low), df["world_x"].quantile(q_high)
    z0, z1 = df["world_z"].quantile(q_low), df["world_z"].quantile(q_high)
    return df[(df["world_x"].between(x0, x1)) & (df["world_z"].between(z0, z1))].copy()


def plot(df: pd.DataFrame, summary: dict, output: Path, title: str,
         smoothed_window: int = 31, show_raw: bool = True) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    df_s = smooth(df, window=smoothed_window)
    df_raw_viz = _trim_for_display(df) if show_raw else df

    fig, ax = plt.subplots(figsize=(10, 8))

    if show_raw:
        ax.scatter(df_raw_viz["world_x"], df_raw_viz["world_z"], c="#b0b0b0", s=5, alpha=0.35,
                   label=f"raw detections ({len(df_raw_viz)}/{len(df)} shown)")

    sc = ax.scatter(df_s["world_x"], df_s["world_z"], c=df_s["time_seconds"],
                    s=12, cmap="viridis", zorder=3,
                    label=f"smoothed (median, {smoothed_window} fr)")
    ax.plot(df_s["world_x"], df_s["world_z"], "-", color="#3a3a3a", alpha=0.55, linewidth=1.4)

    ax.scatter([summary["cam_pos_x_mean"]], [summary["cam_pos_z_mean"]],
               marker="^", s=200, color="red", edgecolor="black", zorder=5,
               label=f"camera (Y={summary['cam_pos_y_mean']:.2f} m)")

    ax.scatter([df_s["world_x"].iloc[0]], [df_s["world_z"].iloc[0]],
               marker="o", s=120, color="green", edgecolor="black", zorder=5, label="start")
    ax.scatter([df_s["world_x"].iloc[-1]], [df_s["world_z"].iloc[-1]],
               marker="s", s=120, color="blue", edgecolor="black", zorder=5, label="end")

    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Z (m)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="best", fontsize=9)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("time (s)")

    plt.tight_layout()
    plt.savefig(output, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path, help="PNG output")
    p.add_argument("--title", default=None)
    p.add_argument("--smoothing-frames", type=int, default=31)
    p.add_argument("--hide-raw", action="store_true")
    args = p.parse_args(argv)

    df = pd.read_csv(args.trajectory)
    summary = summarise(df, smoothed_window=args.smoothing_frames)

    print(f"=== {args.trajectory} ===")
    print(f"  detections:       {summary['n_frames']}")
    print(f"  duration:         {summary['duration_s']:.2f} s")
    print(f"  X range:          [{summary['x_min']:+.2f}, {summary['x_max']:+.2f}] m  (width {summary['x_max']-summary['x_min']:.2f} m)")
    print(f"  Z range:          [{summary['z_min']:+.2f}, {summary['z_max']:+.2f}] m  (depth {summary['z_max']-summary['z_min']:.2f} m)")
    print(f"  total walked:     {summary['total_walked_m']:.2f} m (raw)  / "
          f"{summary['total_walked_smoothed_m']:.2f} m (smoothed)")
    print(f"  mean speed:       {summary['mean_speed_m_s']:.2f} m/s (raw) / "
          f"{summary['mean_speed_smoothed_m_s']:.2f} m/s (smoothed)")
    print(f"  camera position:  ({summary['cam_pos_x_mean']:+.3f}, {summary['cam_pos_y_mean']:+.3f}, {summary['cam_pos_z_mean']:+.3f}) m")
    print(f"  camera motion:    X std {summary['cam_pos_x_std']:.4f} m, Z std {summary['cam_pos_z_std']:.4f} m (low => static)")

    title = args.title or f"World-frame foot trajectory ({args.trajectory.parent.name})"
    plot(
        df,
        summary,
        args.output,
        title,
        smoothed_window=args.smoothing_frames,
        show_raw=not args.hide_raw,
    )
    print(f"  plot written to:  {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
