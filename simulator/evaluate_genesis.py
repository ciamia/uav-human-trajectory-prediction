from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import torch

from data.synthetic_dataset import SyntheticDroneTrajectoryDataset
from env.genesis_sim import GenesisSimBackend, GenesisSimConfig
from models.flow_transformer import FlowTransformer
from safety.collision_check import min_distance_to_obstacles
from scripts.sample_genesis import (
    build_model,
    load_checkpoint,
    load_config,
    plot_genesis_sample,
    project_and_refine,
    resolve_device,
    sample_candidate_trajectory,
)


def path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum())


def smoothness_cost(acceleration_refs: np.ndarray, time_refs: np.ndarray) -> float:
    if len(acceleration_refs) < 2:
        return 0.0
    jerk = np.gradient(acceleration_refs, time_refs, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=-1) ** 2))


def max_norm(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=-1).max())


def min_obstacle_distance(positions: np.ndarray, obstacles: np.ndarray) -> float:
    distances = min_distance_to_obstacles(torch.from_numpy(positions.astype(np.float32)), torch.from_numpy(obstacles.astype(np.float32)))
    return float(distances.min())


def build_eval_dataset(cfg: dict[str, Any], num_scenes: int) -> SyntheticDroneTrajectoryDataset:
    dataset_cfg = dict(cfg.get("dataset", {}))
    dataset_cfg["num_samples"] = int(num_scenes)
    dataset_cfg["seed"] = int(cfg.get("sampling", {}).get("seed", cfg["seed"] + 1000))
    return SyntheticDroneTrajectoryDataset.from_config(
        dataset_cfg,
        sim_config=cfg.get("sim"),
        planning_config=cfg.get("planning"),
    )


def make_headless_backend(cfg: dict[str, Any]) -> GenesisSimBackend:
    sim_cfg = cfg.get("sim", {})
    return GenesisSimBackend(
        GenesisSimConfig(
            dt=float(sim_cfg.get("dt", 0.02)),
            show_viewer=False,
            add_ground=True,
            drone_radius=float(sim_cfg.get("drone_radius", 0.15)),
            auto_build=False,
            launch_scene=False,
        )
    )


@torch.no_grad()
def evaluate_scene(
    idx: int,
    scene: dict[str, torch.Tensor],
    model: FlowTransformer,
    backend: GenesisSimBackend,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    planning_start = time.perf_counter()
    raw = sample_candidate_trajectory(model, scene, cfg, device=device)
    planning_time = time.perf_counter() - planning_start

    refinement_start = time.perf_counter()
    refined = project_and_refine(raw, scene["obstacles"], cfg)
    refinement_time = time.perf_counter() - refinement_start

    refined_state = np.concatenate([refined.position_refs, refined.velocity_refs], axis=-1).astype(np.float32)
    backend.reset(scene["start"].numpy(), scene["goal"].numpy(), scene["obstacles"].numpy())
    rollout = backend.rollout_trajectory(refined_state)

    positions = refined.position_refs
    collision = bool(rollout["collision"])
    min_distance = min_obstacle_distance(positions, scene["obstacles"].numpy())
    success = (not collision) and min_distance > 0.0
    metrics = {
        "scene": idx,
        "success": success,
        "collision": collision,
        "min_obstacle_distance": min_distance,
        "path_length": path_length(positions),
        "smoothness_cost": smoothness_cost(refined.acceleration_refs, refined.time),
        "max_velocity": max_norm(refined.velocity_refs),
        "max_acceleration": max_norm(refined.acceleration_refs),
        "planning_time": planning_time,
        "refinement_time": refinement_time,
    }
    return {
        "metrics": metrics,
        "raw": raw[:, :3].numpy(),
        "refined": positions,
        "start": scene["start"].numpy()[:3],
        "goal": scene["goal"].numpy()[:3],
        "obstacles": scene["obstacles"].numpy(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    numeric_keys = [
        "min_obstacle_distance",
        "path_length",
        "smoothness_cost",
        "max_velocity",
        "max_acceleration",
        "planning_time",
        "refinement_time",
    ]
    summary: dict[str, Any] = {
        "num_scenes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "collision_rate": float(np.mean([row["collision"] for row in rows])),
    }
    for key in numeric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
    return summary


def write_outputs(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
    plot_payloads: list[dict[str, Any]],
    world_size: list[float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_genesis.json"
    csv_path = output_dir / "eval_genesis.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "scenes": rows}, handle, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ranked = sorted(plot_payloads, key=lambda item: (item["metrics"]["success"], item["metrics"]["min_obstacle_distance"]))
    worst = ranked[0]
    best = ranked[-1]
    for name, payload in (("worst", worst), ("best", best)):
        plot_genesis_sample(
            payload["raw"],
            payload["refined"],
            payload["start"],
            payload["goal"],
            payload["obstacles"],
            output_dir / f"eval_genesis_{name}.png",
            world_size,
        )
    return json_path, csv_path


def evaluate(
    config_path: str | Path,
    checkpoint_path: str | Path,
    num_scenes: int,
    output_dir: str | Path = "outputs",
) -> dict[str, Any]:
    cfg = load_config(config_path)
    device = resolve_device(cfg["device"])
    checkpoint = load_checkpoint(checkpoint_path, device)
    ckpt_cfg = checkpoint.get("config")
    if ckpt_cfg is not None:
        cfg["model"] = ckpt_cfg.get("model", cfg["model"])

    horizon = int(cfg.get("sim", {}).get("horizon", cfg["dataset"].get("horizon", 64)))
    model = build_model(cfg, horizon=horizon).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise KeyError("checkpoint must contain 'model_state_dict'")
    model.load_state_dict(state_dict)
    model.eval()

    dataset = build_eval_dataset(cfg, num_scenes=num_scenes)
    backend = make_headless_backend(cfg)
    payloads = [evaluate_scene(idx, dataset[idx], model, backend, cfg, device) for idx in range(num_scenes)]
    backend.close()

    rows = [payload["metrics"] for payload in payloads]
    summary = summarize(rows)
    json_path, csv_path = write_outputs(
        rows,
        summary,
        Path(output_dir),
        payloads,
        world_size=cfg.get("sim", {}).get("world_size", [10.0, 10.0, 4.0]),
    )
    return {"summary": summary, "json": json_path, "csv": csv_path}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/genesis.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best.pt")
    parser.add_argument("--num_scenes", type=int, default=100)
    parser.add_argument("--output_dir", default="outputs")
    args = parser.parse_args()

    result = evaluate(args.config, args.ckpt, num_scenes=args.num_scenes, output_dir=args.output_dir)
    print(f"saved JSON to {result['json']}")
    print(f"saved CSV to {result['csv']}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
