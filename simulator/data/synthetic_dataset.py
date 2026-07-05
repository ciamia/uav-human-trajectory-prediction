from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticDatasetConfig:
    num_samples: int = 2048
    horizon: int = 32
    num_obstacles: int = 6
    world_min: tuple[float, float, float] = (-5.0, -5.0, 0.0)
    world_max: tuple[float, float, float] = (5.0, 5.0, 5.0)
    obstacle_radius_range: tuple[float, float] = (0.35, 0.9)
    obstacle_clearance: float = 0.35
    min_start_goal_distance: float = 2.0
    max_speed: float = 1.5
    seed: int = 0


def _as_float_array(value: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _collides(point: np.ndarray, obstacles: np.ndarray, clearance: float = 0.0) -> bool:
    if obstacles.size == 0:
        return False
    distances = np.linalg.norm(obstacles[:, :3] - point[None, :], axis=-1)
    return bool(np.any(distances <= obstacles[:, 3] + clearance))


def _sample_free_position(
    rng: np.random.Generator,
    world_min: np.ndarray,
    world_max: np.ndarray,
    obstacles: np.ndarray,
    clearance: float,
    max_tries: int = 500,
) -> np.ndarray:
    for _ in range(max_tries):
        position = rng.uniform(world_min, world_max).astype(np.float32)
        if not _collides(position, obstacles, clearance):
            return position
    raise RuntimeError("Unable to sample a collision-free position.")


def _sample_world(
    rng: np.random.Generator,
    world_min: np.ndarray,
    world_max: np.ndarray,
    num_obstacles: int,
    radius_range: tuple[float, float],
) -> np.ndarray:
    obstacles: list[list[float]] = []
    for _ in range(num_obstacles):
        radius = float(rng.uniform(radius_range[0], radius_range[1]))
        low = world_min + radius
        high = world_max - radius
        center = rng.uniform(low, high).astype(np.float32)
        obstacles.append([float(center[0]), float(center[1]), float(center[2]), radius])
    return np.asarray(obstacles, dtype=np.float32)


def _sample_state(
    rng: np.random.Generator,
    position: np.ndarray,
    preferred_direction: np.ndarray,
    max_speed: float,
) -> np.ndarray:
    direction_norm = np.linalg.norm(preferred_direction)
    if direction_norm < 1e-6:
        direction = rng.normal(size=3).astype(np.float32)
        direction = direction / max(np.linalg.norm(direction), 1e-6)
    else:
        direction = preferred_direction / direction_norm
    velocity_noise = rng.normal(0.0, 0.35, size=3).astype(np.float32)
    speed = float(rng.uniform(0.15 * max_speed, max_speed))
    velocity = speed * direction + velocity_noise
    norm = np.linalg.norm(velocity)
    if norm > max_speed:
        velocity = velocity / norm * max_speed
    return np.concatenate([position.astype(np.float32), velocity.astype(np.float32)], axis=0)


def cubic_hermite_positions(start: np.ndarray, goal: np.ndarray, time: np.ndarray) -> np.ndarray:
    p0, v0 = start[:3], start[3:]
    p1, v1 = goal[:3], goal[3:]
    t = time[:, None]
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    return (h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1).astype(np.float32)


def apply_obstacle_avoidance_perturbations(
    positions: np.ndarray,
    obstacles: np.ndarray,
    world_min: np.ndarray,
    world_max: np.ndarray,
    clearance: float,
) -> np.ndarray:
    if obstacles.size == 0:
        return positions

    time = np.linspace(0.0, 1.0, len(positions), dtype=np.float32)
    perturbed = positions.astype(np.float32).copy()
    endpoint_taper = (np.sin(np.pi * time) ** 2)[:, None]

    for obstacle in obstacles:
        center = obstacle[:3]
        radius = float(obstacle[3])
        distances = np.linalg.norm(positions - center[None, :], axis=-1)
        closest_idx = int(np.argmin(distances))
        trigger_distance = radius + clearance + 0.65
        if distances[closest_idx] > trigger_distance:
            continue

        away = positions[closest_idx] - center
        away_norm = np.linalg.norm(away)
        if away_norm < 1e-6:
            away = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            away = away / away_norm

        width = 0.16
        bump = np.exp(-0.5 * ((time - time[closest_idx]) / width) ** 2)[:, None]
        strength = max(trigger_distance - float(distances[closest_idx]), 0.0)
        perturbed += endpoint_taper * bump * away[None, :] * strength

    perturbed[0] = positions[0]
    perturbed[-1] = positions[-1]
    return np.clip(perturbed, world_min, world_max).astype(np.float32)


def make_expert_trajectory(
    start: np.ndarray,
    goal: np.ndarray,
    obstacles: np.ndarray,
    horizon: int,
    world_min: np.ndarray,
    world_max: np.ndarray,
    clearance: float,
) -> np.ndarray:
    time = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    positions = cubic_hermite_positions(start, goal, time)
    positions = apply_obstacle_avoidance_perturbations(
        positions,
        obstacles=obstacles,
        world_min=world_min,
        world_max=world_max,
        clearance=clearance,
    )
    velocities = np.gradient(positions, time, axis=0).astype(np.float32)
    velocities[0] = start[3:]
    velocities[-1] = goal[3:]
    trajectory = np.concatenate([positions, velocities], axis=-1).astype(np.float32)
    trajectory[0] = start
    trajectory[-1] = goal
    return trajectory


class SyntheticDroneTrajectoryDataset(Dataset):
    """Random 3D obstacle worlds with smooth state-space expert trajectories."""

    def __init__(
        self,
        num_samples: int = 2048,
        horizon: int = 32,
        num_obstacles: int = 6,
        world_min: tuple[float, float, float] | list[float] | np.ndarray = (-5.0, -5.0, 0.0),
        world_max: tuple[float, float, float] | list[float] | np.ndarray = (5.0, 5.0, 5.0),
        obstacle_radius_range: tuple[float, float] = (0.35, 0.9),
        obstacle_clearance: float = 0.35,
        min_start_goal_distance: float = 2.0,
        max_speed: float = 1.5,
        seed: int = 0,
    ) -> None:
        self.config = SyntheticDatasetConfig(
            num_samples=num_samples,
            horizon=horizon,
            num_obstacles=num_obstacles,
            world_min=tuple(float(v) for v in world_min),
            world_max=tuple(float(v) for v in world_max),
            obstacle_radius_range=obstacle_radius_range,
            obstacle_clearance=obstacle_clearance,
            min_start_goal_distance=min_start_goal_distance,
            max_speed=max_speed,
            seed=seed,
        )
        self.world_min = _as_float_array(world_min)
        self.world_max = _as_float_array(world_max)
        self.rng = np.random.default_rng(seed)
        self.samples = [self._generate_sample() for _ in range(num_samples)]

    @classmethod
    def from_config(
        cls,
        dataset_config: dict,
        sim_config: dict | None = None,
        planning_config: dict | None = None,
    ) -> "SyntheticDroneTrajectoryDataset":
        """Build a dataset from either legacy dataset config or Genesis-style sim config."""
        dataset_config = dict(dataset_config)
        sim_config = dict(sim_config or {})
        planning_config = dict(planning_config or {})

        if sim_config:
            world_size = sim_config.get("world_size")
            if world_size is not None:
                world_size = _as_float_array(world_size)
                dataset_config["world_min"] = [-world_size[0] / 2.0, -world_size[1] / 2.0, 0.0]
                dataset_config["world_max"] = [world_size[0] / 2.0, world_size[1] / 2.0, world_size[2]]
            dataset_config["horizon"] = sim_config.get("horizon", dataset_config.get("horizon", 32))
            dataset_config["num_obstacles"] = sim_config.get(
                "obstacle_count",
                dataset_config.get("num_obstacles", 6),
            )
            dataset_config["obstacle_radius_range"] = sim_config.get(
                "obstacle_radius_range",
                dataset_config.get("obstacle_radius_range", (0.35, 0.9)),
            )
            if "safety_margin" in planning_config:
                dataset_config["obstacle_clearance"] = planning_config["safety_margin"]
            if "max_velocity" in planning_config:
                dataset_config["max_speed"] = planning_config["max_velocity"]

        return cls(
            num_samples=dataset_config.get("num_samples", 2048),
            horizon=dataset_config.get("horizon", 32),
            num_obstacles=dataset_config.get("num_obstacles", 6),
            world_min=dataset_config.get("world_min", (-5.0, -5.0, 0.0)),
            world_max=dataset_config.get("world_max", (5.0, 5.0, 5.0)),
            obstacle_radius_range=tuple(dataset_config.get("obstacle_radius_range", (0.35, 0.9))),
            obstacle_clearance=dataset_config.get("obstacle_clearance", 0.35),
            min_start_goal_distance=dataset_config.get("min_start_goal_distance", 2.0),
            max_speed=dataset_config.get("max_speed", 1.5),
            seed=dataset_config.get("seed", 0),
        )

    def _generate_sample(self) -> dict[str, torch.Tensor]:
        cfg = self.config
        obstacles = _sample_world(
            self.rng,
            self.world_min,
            self.world_max,
            cfg.num_obstacles,
            cfg.obstacle_radius_range,
        )

        for _ in range(500):
            start_pos = _sample_free_position(
                self.rng,
                self.world_min,
                self.world_max,
                obstacles,
                cfg.obstacle_clearance,
            )
            goal_pos = _sample_free_position(
                self.rng,
                self.world_min,
                self.world_max,
                obstacles,
                cfg.obstacle_clearance,
            )
            if np.linalg.norm(goal_pos - start_pos) >= cfg.min_start_goal_distance:
                break
        else:
            raise RuntimeError("Unable to sample a valid start/goal pair.")

        start = _sample_state(self.rng, start_pos, goal_pos - start_pos, cfg.max_speed)
        goal = _sample_state(self.rng, goal_pos, start_pos - goal_pos, cfg.max_speed)
        trajectory = make_expert_trajectory(
            start,
            goal,
            obstacles,
            horizon=cfg.horizon,
            world_min=self.world_min,
            world_max=self.world_max,
            clearance=cfg.obstacle_clearance,
        )

        return {
            "trajectory": torch.from_numpy(trajectory),
            "start": torch.from_numpy(start.astype(np.float32)),
            "goal": torch.from_numpy(goal.astype(np.float32)),
            "obstacles": torch.from_numpy(obstacles.astype(np.float32)),
        }

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]
