from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.synthetic_dataset import SyntheticDroneTrajectoryDataset
from flow.sampler import sample as sample_ode
from models.flow_transformer import FlowTransformer
from safety.projection import project_waypoints


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def with_sampling_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg.setdefault("sampling", {})
    cfg["sampling"].setdefault("num_samples", 8)
    cfg["sampling"].setdefault("steps", 50)
    cfg["sampling"].setdefault("method", "heun")
    cfg["sampling"].setdefault("projection_margin", cfg["dataset"].get("obstacle_clearance", 0.25))
    cfg["sampling"].setdefault("projection_iterations", 6)
    cfg["sampling"].setdefault("output_dir", str(Path(cfg["training"]["output_dir"]) / "samples"))
    cfg["sampling"].setdefault("seed", int(cfg["seed"]) + 1000)
    return cfg


def build_model(cfg: dict[str, Any], horizon: int) -> FlowTransformer:
    return FlowTransformer(
        state_dim=cfg["state_dim"],
        obstacle_dim=cfg["obstacle_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        max_horizon=max(horizon, cfg["max_horizon"]),
    )


def build_scene_loader(cfg: dict[str, Any]) -> DataLoader:
    dataset_cfg = dict(cfg["dataset"])
    dataset_cfg["num_samples"] = cfg["sampling"]["num_samples"]
    dataset_cfg["seed"] = cfg["sampling"]["seed"]
    dataset = SyntheticDroneTrajectoryDataset(**dataset_cfg)
    return DataLoader(dataset, batch_size=cfg["sampling"]["num_samples"], shuffle=False)


def enforce_endpoints(trajectories: torch.Tensor, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    trajectories = trajectories.clone()
    trajectories[:, 0] = start
    trajectories[:, -1] = goal
    return trajectories


def plot_scene(
    trajectory: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    output_path: Path,
    world_min: list[float],
    world_max: list[float],
) -> None:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    positions = trajectory[:, :3]
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:blue", linewidth=2.0)
    ax.scatter(start[0], start[1], start[2], color="tab:green", s=55, label="start")
    ax.scatter(goal[0], goal[1], goal[2], color="tab:red", s=55, label="goal")

    u, v = np.mgrid[0 : 2 * np.pi : 18j, 0 : np.pi : 10j]
    for obstacle in obstacles:
        center = obstacle[:3]
        radius = obstacle[3]
        x = center[0] + radius * np.cos(u) * np.sin(v)
        y = center[1] + radius * np.sin(u) * np.sin(v)
        z = center[2] + radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color="black", alpha=0.28, linewidth=0.5)

    ax.set_xlim(world_min[0], world_max[0])
    ax.set_ylim(world_min[1], world_max[1])
    ax.set_zlim(world_min[2], world_max[2])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def sample_from_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path = "configs/base.yaml",
    output_dir: str | Path | None = None,
) -> list[Path]:
    fallback_cfg = load_config(config_path)
    device = resolve_device(fallback_cfg["device"])
    checkpoint = load_checkpoint(checkpoint_path, device)
    cfg = with_sampling_defaults(checkpoint.get("config", fallback_cfg))
    device = resolve_device(cfg["device"])

    torch.manual_seed(int(cfg["sampling"]["seed"]))
    loader = build_scene_loader(cfg)
    batch = next(iter(loader))

    model = build_model(cfg["model"], horizon=cfg["dataset"]["horizon"]).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state_dict is None:
        raise KeyError("checkpoint must contain 'model_state_dict'")
    model.load_state_dict(state_dict)
    model.eval()

    start = batch["start"].to(device)
    goal = batch["goal"].to(device)
    obstacles = batch["obstacles"].to(device)
    condition = {"start": start, "goal": goal, "obstacles": obstacles}
    noise = torch.randn(
        start.shape[0],
        cfg["dataset"]["horizon"],
        cfg["model"]["state_dim"],
        device=device,
    )

    trajectories = sample_ode(
        model,
        noise,
        condition,
        steps=cfg["sampling"]["steps"],
        method=cfg["sampling"]["method"],
    )
    trajectories = enforce_endpoints(trajectories, start, goal)
    trajectories = project_waypoints(
        trajectories,
        obstacles,
        margin=cfg["sampling"]["projection_margin"],
        max_iterations=cfg["sampling"]["projection_iterations"],
    )
    trajectories = enforce_endpoints(trajectories, start, goal)

    plot_dir = Path(output_dir) if output_dir is not None else Path(cfg["sampling"]["output_dir"])
    saved_paths: list[Path] = []
    trajectories_np = trajectories.cpu().numpy()
    starts_np = start.cpu().numpy()
    goals_np = goal.cpu().numpy()
    obstacles_np = obstacles.cpu().numpy()
    for idx in range(trajectories_np.shape[0]):
        path = plot_dir / f"sample_{idx:03d}.png"
        plot_scene(
            trajectories_np[idx],
            starts_np[idx],
            goals_np[idx],
            obstacles_np[idx],
            path,
            world_min=cfg["dataset"]["world_min"],
            world_max=cfg["dataset"]["world_max"],
        )
        saved_paths.append(path)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/latest.pt")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    paths = sample_from_checkpoint(args.checkpoint, config_path=args.config, output_dir=args.output_dir)
    print(f"saved {len(paths)} plots to {paths[0].parent if paths else args.output_dir}")


if __name__ == "__main__":
    main()
