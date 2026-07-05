from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np


def _float_or_nan(value: str | None) -> float:
    if value in (None, ""):
        return float("nan")
    return float(value)


def load_training_log(path: str | Path) -> dict[str, np.ndarray]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in training log: {path}")
    return {
        "epoch": np.array([_float_or_nan(row.get("epoch")) for row in rows], dtype=np.float32),
        "step": np.array([_float_or_nan(row.get("step")) for row in rows], dtype=np.float32),
        "loss": np.array([_float_or_nan(row.get("loss")) for row in rows], dtype=np.float32),
        "collision_rate": np.array([_float_or_nan(row.get("collision_rate")) for row in rows], dtype=np.float32),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.isfinite(values)
    if window <= 1 or finite.sum() < window:
        return values
    cleaned = values.copy()
    cleaned[~finite] = np.interp(np.flatnonzero(~finite), np.flatnonzero(finite), cleaned[finite])
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(cleaned, kernel, mode="same")


def plot_training_curves(
    log_path: str | Path = "outputs/train_loss.csv",
    output: str | Path = "outputs/training_curves.png",
    smooth_window: int = 5,
) -> Path:
    data = load_training_log(log_path)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    x = data["step"]
    axes[0].plot(x, data["loss"], color="tab:blue", alpha=0.35, label="loss")
    axes[0].plot(x, moving_average(data["loss"], smooth_window), color="tab:blue", label="smoothed")
    axes[0].set_ylabel("CFM loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    collision = data["collision_rate"]
    if np.isfinite(collision).any():
        axes[1].plot(x, collision, "o-", color="tab:red", label="collision rate")
    axes[1].set_ylabel("collision rate")
    axes[1].set_xlabel("training step")
    axes[1].set_ylim(bottom=0.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"saved training curves to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="outputs/train_loss.csv")
    parser.add_argument("--output", default="outputs/training_curves.png")
    parser.add_argument("--smooth-window", type=int, default=5)
    args = parser.parse_args()
    plot_training_curves(args.log, args.output, smooth_window=args.smooth_window)


if __name__ == "__main__":
    main()
