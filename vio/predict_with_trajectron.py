"""Run Trajectron++ inference on a VIO world trajectory.

Takes a world_trajectory.csv (produced by `vio.pixel_to_world`) and a
fine-tuned Trajectron++ checkpoint, downsamples to the model's dt, and
generates sample predictions on a sliding window.

The output is a `.npz` file with three keys:
    history    [N_windows, H, 2]    last H GT points used as input
    future_gt  [N_windows, F, 2]    next F GT points  (so you can plot residuals)
    samples    [N_windows, S, F, 2] S random samples from the model

Plus a CSV summarising the windows.

Usage:
    python -m vio.predict_with_trajectron \
        --trajectory data/vio/field_001/world_trajectory.csv \
        --model-dir  ~/Trajectron-plus-plus/experiments/pedestrians/models/models_10_Mar_2026_00_42_50_uav_ft_v8_persp \
        --checkpoint 45 \
        --output     data/vio/field_001/predictions.npz \
        --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT / "trajectron"))

from environment import Environment, Scene, Node  # noqa: E402
from environment import derivative_of  # noqa: E402
from model.model_registrar import ModelRegistrar  # noqa: E402
from model.trajectron import Trajectron  # noqa: E402
from utils import prediction_output_to_trajectories  # noqa: E402


DT_MODEL = 0.4
DATA_COLUMNS = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])
STANDARDIZATION = {
    "PEDESTRIAN": {
        "position":     {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
        "velocity":     {"x": {"mean": 0, "std": 2}, "y": {"mean": 0, "std": 2}},
        "acceleration": {"x": {"mean": 0, "std": 1}, "y": {"mean": 0, "std": 1}},
    }
}


def downsample(df: pd.DataFrame, dt_target: float) -> pd.DataFrame:
    """Resample to a fixed dt by linear interpolation in time."""
    t = df["time_seconds"].to_numpy()
    x = df["world_x"].to_numpy()
    z = df["world_z"].to_numpy()

    t0, t1 = float(t.min()), float(t.max())
    n = int(np.floor((t1 - t0) / dt_target)) + 1
    tt = t0 + dt_target * np.arange(n)

    xx = np.interp(tt, t, x)
    zz = np.interp(tt, t, z)
    return pd.DataFrame({"frame_id": np.arange(n), "time_seconds": tt, "pos_x": xx, "pos_y": zz})


def build_scene(df_resampled: pd.DataFrame) -> tuple[Environment, Scene]:
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=STANDARDIZATION)
    env.attention_radius = {(env.NodeType.PEDESTRIAN, env.NodeType.PEDESTRIAN): 3.0}

    scene = Scene(timesteps=int(df_resampled["frame_id"].max()) + 1, dt=DT_MODEL, name="vio_field")

    x = df_resampled["pos_x"].to_numpy()
    y = df_resampled["pos_y"].to_numpy()
    vx = derivative_of(x, scene.dt)
    vy = derivative_of(y, scene.dt)
    ax = derivative_of(vx, scene.dt)
    ay = derivative_of(vy, scene.dt)

    data = {
        ("position", "x"): x, ("position", "y"): y,
        ("velocity", "x"): vx, ("velocity", "y"): vy,
        ("acceleration", "x"): ax, ("acceleration", "y"): ay,
    }
    node_data = pd.DataFrame(data, columns=DATA_COLUMNS)
    node = Node(node_type=env.NodeType.PEDESTRIAN, node_id="0", data=node_data)
    node.first_timestep = 0
    scene.nodes.append(node)
    env.scenes = [scene]
    return env, scene


def load_trajectron(model_dir: Path, checkpoint: int, env: Environment, device: str = "cpu"):
    with open(model_dir / "config.json", "r") as f:
        hp = json.load(f)
    mr = ModelRegistrar(str(model_dir), device)
    traj = Trajectron(mr, hp, None, device)
    traj.set_environment(env)
    traj.set_annealing_params()

    cp = torch.load(model_dir / f"model_registrar-{checkpoint}.pt", map_location=device)
    mr.model_dict.load_state_dict(cp.state_dict())
    mr.eval()
    return traj, hp


def run(
    trajectory_csv: Path,
    model_dir: Path,
    checkpoint: int,
    output_npz: Path,
    num_samples: int = 20,
    stride: int = 1,
    device: str = "cpu",
) -> dict:
    raw = pd.read_csv(trajectory_csv)
    raw = raw.sort_values("time_seconds").drop_duplicates("time_seconds").reset_index(drop=True)
    print(f"Raw VIO points: {len(raw)} (duration {raw.time_seconds.max() - raw.time_seconds.min():.1f}s)")

    rs = downsample(raw, DT_MODEL)
    print(f"Resampled to dt={DT_MODEL}s -> {len(rs)} steps")

    env, scene = build_scene(rs)
    traj_model, hp = load_trajectron(model_dir, checkpoint, env, device=device)

    ph = int(hp.get("prediction_horizon", 12))
    max_hl = int(hp.get("maximum_history_length", 7))
    print(f"Model: ph={ph}, max_hl={max_hl}, num_samples={num_samples}")

    history_buf = []
    future_gt_buf = []
    samples_buf = []
    ts_buf = []

    valid_range = list(range(max_hl, scene.timesteps - ph, stride))
    print(f"Running prediction at {len(valid_range)} windows...")

    with torch.no_grad():
        for ts in valid_range:
            try:
                predictions = traj_model.predict(
                    scene, np.array([ts]), ph,
                    num_samples=num_samples,
                    min_future_timesteps=ph,
                    min_history_timesteps=1,
                    z_mode=False, gmm_mode=False, full_dist=False,
                )
            except Exception as exc:
                print(f"  ts={ts}: predict failed ({exc!s})")
                continue
            if not predictions:
                continue
            pred_dict, hist_dict, fut_dict = prediction_output_to_trajectories(
                predictions, scene.dt, max_hl, ph)
            if ts not in pred_dict:
                continue
            for node, samps in pred_dict[ts].items():
                samps = np.array(samps)
                if samps.ndim == 4:
                    samps = samps[0]  # drop the leading "scenes" axis
                gt = np.array(fut_dict[ts][node])
                hist = np.array(hist_dict[ts][node])
                if gt is None or len(gt) < ph:
                    continue
                history_buf.append(hist)
                future_gt_buf.append(gt[:ph])
                samples_buf.append(samps[:, :ph, :])
                ts_buf.append(ts)

    if not samples_buf:
        raise RuntimeError("No successful predictions.")

    history = np.stack(history_buf)
    future_gt = np.stack(future_gt_buf)
    samples = np.stack(samples_buf)
    ts_arr = np.array(ts_buf, dtype=int)
    print(f"history {history.shape}, future_gt {future_gt.shape}, samples {samples.shape}")

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz,
                        history=history, future_gt=future_gt, samples=samples,
                        timesteps=ts_arr, dt=DT_MODEL)

    # Quick ADE/FDE on the deterministic mean.
    mean = samples.mean(axis=1)
    ade = np.linalg.norm(mean - future_gt, axis=-1).mean()
    fde = np.linalg.norm(mean[:, -1] - future_gt[:, -1], axis=-1).mean()
    best = np.linalg.norm(samples - future_gt[:, None], axis=-1).mean(axis=-1).min(axis=-1).mean()
    print(f"\nADE (mean of samples):       {ade:.3f} m")
    print(f"FDE (mean of samples):       {fde:.3f} m")
    print(f"Best-of-{num_samples} ADE:   {best:.3f} m")
    print(f"\nWrote: {output_npz}")
    return {"history": history, "future_gt": future_gt, "samples": samples,
            "ade": ade, "fde": fde, "best_ade": best, "dt": DT_MODEL}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trajectory", required=True, type=Path)
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    run(args.trajectory, args.model_dir, args.checkpoint, args.output,
        num_samples=args.num_samples, stride=args.stride, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
