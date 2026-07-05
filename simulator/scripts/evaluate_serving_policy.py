from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch

from data.expert_generator import ExpertGeneratorConfig, generate_expert_trajectory, trajectory_collision_mask
from flow.sampler import sample
from planning.refiner import RefinedTrajectory
from scripts.sample_serving_trajectory import (
    SampleServingConfig,
    anchor_trajectory,
    build_scene_condition,
    load_model,
    make_world,
    refine_candidate,
    tensors_from_condition,
)
from scripts.train_serving_fm import TrainServingConfig, resolve_device


@dataclass(frozen=True)
class EvalServingConfig:
    """Configuration for Milestone 9 serving policy evaluation."""

    checkpoint: str = "outputs/serving_fm/checkpoints/best.pt"
    output_dir: str = "outputs/serving_eval"
    num_scenes: int = 100
    device: str = "cpu"
    seed: int = 0
    horizon: int = 96
    sample_steps: int = 32
    sample_method: str = "heun"
    headless: bool = True
    safety_margin: float = 0.35
    max_velocity: float = 3.0
    max_acceleration: float = 5.0
    hover_tolerance: float = 0.5


def evaluate(config: EvalServingConfig) -> dict[str, Any]:
    """Evaluate the serving planner over random scenes and save metrics/plots."""
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    model, train_config, horizon = load_model(config.checkpoint, device, config.horizon)

    payloads: list[dict[str, Any]] = []
    for scene_idx in range(config.num_scenes):
        payloads.append(evaluate_scene(scene_idx, model, train_config, horizon, device, config))

    rows = [payload["metrics"] for payload in payloads]
    summary = summarize(rows)
    output_dir = Path(config.output_dir)
    json_path, csv_path = write_metrics(rows, summary, output_dir)
    plot_paths = write_best_worst_plots(payloads, output_dir)
    return {"summary": summary, "json": str(json_path), "csv": str(csv_path), "plots": [str(path) for path in plot_paths]}


@torch.no_grad()
def evaluate_scene(
    scene_idx: int,
    model: torch.nn.Module,
    train_config: TrainServingConfig,
    horizon: int,
    device: torch.device,
    config: EvalServingConfig,
) -> dict[str, Any]:
    """Evaluate one random serving scene."""
    scene_seed = config.seed + scene_idx
    torch.manual_seed(scene_seed)
    rng = np.random.default_rng(scene_seed)
    sample_config = SampleServingConfig(
        checkpoint=config.checkpoint,
        device=config.device,
        seed=scene_seed,
        horizon=horizon,
        sample_steps=config.sample_steps,
        sample_method=config.sample_method,
        headless=config.headless,
        safety_margin=config.safety_margin,
        max_velocity=config.max_velocity,
        max_acceleration=config.max_acceleration,
    )
    world = make_world(sample_config, train_config, rng)
    try:
        condition_np = build_scene_condition(world, horizon, config.safety_margin)
        condition = tensors_from_condition(condition_np, device)

        planning_start = time.perf_counter()
        noise = torch.randn(1, horizon, 6, device=device)
        sampled = sample(model, noise, condition, steps=config.sample_steps, method=config.sample_method)[0].cpu().numpy()
        planning_time = time.perf_counter() - planning_start
        sampled = anchor_trajectory(sampled, condition_np["drone_start"][0], condition_np["hover_goal"][0])

        refinement_start = time.perf_counter()
        refined = refine_candidate(sampled, horizon, sample_config)
        predicted_people = condition_np["future_people_predictions"][0]
        sampled_collision = bool(
            trajectory_collision_mask(
                refined.position_refs,
                world.buildings,
                predicted_people,
                ExpertGeneratorConfig(world_size=tuple(float(v) for v in world.world_size), safety_margin=config.safety_margin),
            ).any()
        )
        used_expert_fallback = False
        if sampled_collision or not np.isfinite(refined.position_refs).all():
            used_expert_fallback = True
            try:
                expert = generate_expert_trajectory(
                    start=condition_np["drone_start"][0],
                    hover_goal=condition_np["hover_goal"][0, :3],
                    buildings=world.buildings,
                    people=world.people,
                    config=ExpertGeneratorConfig(
                        world_size=tuple(float(v) for v in world.world_size),
                        horizon=horizon,
                        safety_margin=config.safety_margin,
                        max_velocity=config.max_velocity,
                        max_acceleration=config.max_acceleration,
                    ),
                )
                refined = refine_candidate(expert.as_state_trajectory(), horizon, sample_config)
            except RuntimeError:
                refined = straight_line_refinement(condition_np["drone_start"][0], condition_np["hover_goal"][0, :3], horizon, sample_config)
        refinement_time = time.perf_counter() - refinement_start

        distances = obstacle_distances(refined.position_refs, world.buildings, predicted_people, config.safety_margin)
        min_distance = float(distances.min()) if distances.size else float("inf")
        collision = min_distance <= 0.0
        hover_error = float(np.linalg.norm(refined.position_refs[-1] - condition_np["hover_goal"][0, :3]))
        success = (not collision) and hover_error <= config.hover_tolerance
        metrics = {
            "scene": scene_idx,
            "success": success,
            "collision": collision,
            "sampled_collision": sampled_collision,
            "used_expert_fallback": used_expert_fallback,
            "expert_fallback_failed": used_expert_fallback and sampled_collision and collision,
            "min_obstacle_distance": min_distance,
            "path_length": path_length(refined.position_refs),
            "smoothness": smoothness_cost(refined.acceleration_refs, refined.time),
            "inference_time": planning_time,
            "refinement_time": refinement_time,
            "final_hover_error": hover_error,
        }
        return {
            "metrics": metrics,
            "condition": condition_np,
            "sampled": sampled,
            "refined": refined,
            "predicted_people": predicted_people,
            "building_geometry": condition_np["building_geometry"][0],
            "world_size": world.world_size.copy(),
        }
    finally:
        world.close()


def obstacle_distances(
    positions: np.ndarray,
    building_geometry: Any,
    predicted_people: np.ndarray,
    safety_margin: float,
    person_radius: float = 0.3,
) -> np.ndarray:
    """Return signed distances to inflated buildings and predicted people."""
    buildings = _building_array(building_geometry)
    all_distances: list[np.ndarray] = []
    if buildings.size:
        centers = buildings[None, :, :3]
        half_sizes = buildings[None, :, 3:6] * 0.5
        delta = np.abs(positions[:, None, :] - centers) - half_sizes
        outside = np.maximum(delta, 0.0)
        outside_distance = np.linalg.norm(outside, axis=-1)
        inside_distance = np.minimum(np.max(delta, axis=-1), 0.0)
        all_distances.append(outside_distance + inside_distance - float(safety_margin))
    if predicted_people.size:
        time_ids = np.linspace(0, predicted_people.shape[1] - 1, len(positions)).round().astype(int)
        people_at_time = predicted_people[:, time_ids, :2].transpose(1, 0, 2)
        distances = np.linalg.norm(positions[:, None, :2] - people_at_time, axis=-1)
        all_distances.append(distances - (float(safety_margin) + float(person_radius)))
    if not all_distances:
        return np.array([np.inf], dtype=np.float32)
    return np.concatenate([values.reshape(len(positions), -1) for values in all_distances], axis=1)


def straight_line_refinement(
    start_state: np.ndarray,
    hover_goal: np.ndarray,
    horizon: int,
    sample_config: SampleServingConfig,
) -> RefinedTrajectory:
    """Last-resort trajectory so evaluation records a failed scene instead of crashing."""
    start = np.asarray(start_state, dtype=np.float32)
    goal_state = np.zeros(6, dtype=np.float32)
    goal_state[:3] = np.asarray(hover_goal, dtype=np.float32)[:3]
    waypoints = np.stack([start, goal_state]).astype(np.float32)
    return refine_candidate(waypoints, horizon, sample_config)


def path_length(positions: np.ndarray) -> float:
    """Return trajectory path length."""
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=-1).sum())


def smoothness_cost(accelerations: np.ndarray, times: np.ndarray) -> float:
    """Return mean squared jerk as a smoothness cost."""
    if len(accelerations) < 2:
        return 0.0
    jerk = np.gradient(accelerations, times, axis=0)
    return float(np.mean(np.linalg.norm(jerk, axis=-1) ** 2))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-scene serving evaluation rows."""
    numeric_keys = [
        "min_obstacle_distance",
        "path_length",
        "smoothness",
        "inference_time",
        "refinement_time",
        "final_hover_error",
    ]
    summary: dict[str, Any] = {
        "num_scenes": len(rows),
        "success_rate": float(np.mean([row["success"] for row in rows])) if rows else 0.0,
        "collision_rate": float(np.mean([row["collision"] for row in rows])) if rows else 0.0,
        "fallback_rate": float(np.mean([row["used_expert_fallback"] for row in rows])) if rows else 0.0,
        "expert_fallback_failure_rate": float(np.mean([row["expert_fallback_failed"] for row in rows])) if rows else 0.0,
    }
    for key in numeric_keys:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        summary[f"{key}_mean"] = float(values.mean()) if len(values) else 0.0
        summary[f"{key}_min"] = float(values.min()) if len(values) else 0.0
        summary[f"{key}_max"] = float(values.max()) if len(values) else 0.0
    return summary


def write_metrics(rows: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    """Write JSON and CSV evaluation metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_serving_policy.json"
    csv_path = output_dir / "eval_serving_policy.csv"
    json_path.write_text(json.dumps({"summary": summary, "scenes": rows}, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def write_best_worst_plots(payloads: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    """Save best and worst trajectory plots."""
    if not payloads:
        return []
    ranked = sorted(
        payloads,
        key=lambda item: (
            bool(item["metrics"]["success"]),
            float(item["metrics"]["min_obstacle_distance"]),
            -float(item["metrics"]["final_hover_error"]),
        ),
    )
    outputs = []
    for name, payload in (("worst", ranked[0]), ("best", ranked[-1])):
        output = output_dir / f"eval_serving_policy_{name}.png"
        plot_evaluation_scene(payload, output)
        outputs.append(output)
    return outputs


def plot_evaluation_scene(payload: dict[str, Any], output: str | Path) -> Path:
    """Plot one evaluated serving scene."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    condition = payload["condition"]
    refined: RefinedTrajectory = payload["refined"]
    sampled = payload["sampled"]
    predicted_people = payload["predicted_people"]
    buildings = payload["building_geometry"]
    world_size = np.asarray(payload.get("world_size", [30.0, 30.0, 20.0]), dtype=np.float32)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for building in buildings:
        _plot_box(ax, building[:3], building[3:6])
    for idx, prediction in enumerate(predicted_people):
        color = "tab:red" if idx == 2 else "tab:orange"
        ax.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color=color, alpha=0.35, linewidth=1.0)
    ax.plot(sampled[:, 0], sampled[:, 1], sampled[:, 2], "o--", color="tab:gray", alpha=0.5, label="sampled")
    ax.plot(refined.position_refs[:, 0], refined.position_refs[:, 1], refined.position_refs[:, 2], color="tab:blue", linewidth=2.3, label="refined")
    start = condition["drone_start"][0, :3]
    hover = condition["hover_goal"][0, :3]
    ax.scatter(start[0], start[1], start[2], color="tab:green", s=70, label="start")
    ax.scatter(hover[0], hover[1], hover[2], color="tab:purple", marker="*", s=130, label="hover")
    ax.set_xlim(-float(world_size[0]) / 2.0, float(world_size[0]) / 2.0)
    ax.set_ylim(-float(world_size[1]) / 2.0, float(world_size[1]) / 2.0)
    ax.set_zlim(0.0, float(world_size[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_box(ax: plt.Axes, center: np.ndarray, size: np.ndarray) -> None:
    half = size / 2.0
    x = [center[0] - half[0], center[0] + half[0]]
    y = [center[1] - half[1], center[1] + half[1]]
    z = [center[2] - half[2], center[2] + half[2]]
    corners = np.array(
        [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for start, end in edges:
        ax.plot(*zip(corners[start], corners[end]), color="0.35", linewidth=1.0)


def _building_array(building_geometry: Any) -> np.ndarray:
    if isinstance(building_geometry, np.ndarray):
        values = building_geometry.astype(np.float32)
        if values.ndim == 2 and values.shape[-1] == 6:
            return values
    return np.stack([np.concatenate([building.center, building.size]).astype(np.float32) for building in building_geometry])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/serving_fm/checkpoints/best.pt")
    parser.add_argument("--output-dir", default="outputs/serving_eval")
    parser.add_argument("--num-scenes", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--render", action="store_true", help="Launch Genesis viewer instead of headless evaluation.")
    args = parser.parse_args()

    result = evaluate(
        EvalServingConfig(
            checkpoint=args.ckpt,
            output_dir=args.output_dir,
            num_scenes=args.num_scenes,
            device=args.device,
            seed=args.seed,
            sample_steps=args.sample_steps,
            headless=not args.render,
        )
    )
    print(f"saved JSON to {result['json']}")
    print(f"saved CSV to {result['csv']}")
    for plot in result["plots"]:
        print(f"saved plot to {plot}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
