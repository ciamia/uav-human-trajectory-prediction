"""Plot Trajectron++ predictions over the VIO world trajectory.

Produces a figure with N small panels — one per prediction window — and
one summary panel with all of them on a single map.

Usage:
    python -m vio.plot_predictions \
        --predictions data/vio/field_001/predictions_v8.npz \
        --output      data/vio/field_001/predictions_plot.png \
        --n-windows   6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_panels(
    data: dict,
    output: Path,
    n_windows: int = 6,
    title: str = "Trajectron++ predictions on VIO world trajectory",
) -> None:
    history = data["history"]      # [N, H, 2]
    future = data["future_gt"]     # [N, F, 2]
    samples = data["samples"]      # [N, S, F, 2]
    timesteps = data["timesteps"]  # [N]

    N = len(history)
    if N == 0:
        raise RuntimeError("No prediction windows.")

    # Pick evenly spaced windows.
    pick = np.linspace(0, N - 1, n_windows, dtype=int)

    cols = 3
    rows = int(np.ceil((len(pick) + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, i in enumerate(pick):
        ax = axes[ax_idx]
        hist = history[i]
        gt = future[i]
        samps = samples[i]  # [S, F, 2]

        for s in samps:
            ax.plot(s[:, 0], s[:, 1], "-", color="C0", alpha=0.18, linewidth=1)
        mean = samps.mean(axis=0)
        ax.plot(mean[:, 0], mean[:, 1], "-", color="C0", linewidth=2.5, label="mean sample")
        ax.plot(gt[:, 0], gt[:, 1], "-", color="C3", linewidth=2.5, label="ground truth")
        ax.plot(hist[:, 0], hist[:, 1], "-", color="black", linewidth=2.5, label="history")
        ax.scatter([hist[-1, 0]], [hist[-1, 1]], color="black", s=60, zorder=5)

        ade = float(np.linalg.norm(mean - gt, axis=-1).mean())
        ax.set_title(f"window {i} (ts={int(timesteps[i])})  ADE={ade:.2f} m")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("world X (m)")
        ax.set_ylabel("world Z (m)")
        if ax_idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Summary panel — all GT + all mean predictions on one map.
    sum_ax = axes[len(pick)]
    for i in range(N):
        sum_ax.plot(future[i, :, 0], future[i, :, 1], "-", color="C3", alpha=0.3, linewidth=1)
        m = samples[i].mean(axis=0)
        sum_ax.plot(m[:, 0], m[:, 1], "-", color="C0", alpha=0.3, linewidth=1)
    sum_ax.plot([], [], color="C3", label="GT futures (all windows)")
    sum_ax.plot([], [], color="C0", label="mean prediction (all windows)")
    sum_ax.set_title("All windows overlay")
    sum_ax.set_aspect("equal")
    sum_ax.grid(True, alpha=0.3)
    sum_ax.set_xlabel("world X (m)")
    sum_ax.set_ylabel("world Z (m)")
    sum_ax.legend(loc="best", fontsize=8)

    # Hide unused axes.
    for j in range(len(pick) + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=120)
    plt.close(fig)
    print(f"Wrote {output}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n-windows", type=int, default=6)
    p.add_argument("--title", default=None)
    args = p.parse_args(argv)

    npz = np.load(args.predictions)
    data = {k: npz[k] for k in npz.files}

    print(f"history {data['history'].shape}, future {data['future_gt'].shape}, "
          f"samples {data['samples'].shape}, dt={float(data['dt']):.3f} s")

    title = args.title or f"Trajectron++ predictions ({args.predictions.parent.name})"
    plot_panels(data, args.output, n_windows=args.n_windows, title=title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
