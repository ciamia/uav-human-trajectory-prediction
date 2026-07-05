from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.expert_generator import ExpertGeneratorConfig, generate_expert_trajectory, predict_people_positions
from env.buildings import BoxBuilding
from env.drone import Drone
from env.people import PersonAgent
from env.setup_genesis_world import setup_people_initial_position, \
                                    make_buildings, \
                                     WORLD_SIZE
from planning.hover_goal import compute_hover_goal, is_position_in_obstacles
from planning.target_selection import select_coffee_target


@dataclass(frozen=True)
class ServingDatasetConfig:
    """Configuration for Milestone 4 serving-trajectory dataset generation."""

    num_samples: int = 64
    horizon: int = 96
    person_count: int = 5
    world_size: tuple[float, float, float] = tuple(float(value) for value in WORLD_SIZE)
    seed: int = 0
    safety_margin: float = 0.35
    hover_distance: float = 1.0
    hover_height: float = 1.5
    max_velocity: float = 3.0
    max_acceleration: float = 5.0
    grid_resolution: float = 0.35
    max_retries: int = 100


class ServingTrajectoryDataset:
    """Lightweight loader for saved serving trajectory datasets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = dict(np.load(self.path, allow_pickle=False))
        self.metadata = json.loads(str(self.data["metadata"].item()))

    def __len__(self) -> int:
        return int(self.data["drone_start"].shape[0])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {
            key: value[index]
            for key, value in self.data.items()
            if key != "metadata" and value.shape[0] == len(self)
        }


def generate_serving_dataset(config: ServingDatasetConfig | None = None, progress: bool = False) -> dict[str, np.ndarray]:
    """Generate a fixed-shape training dataset for the serving task."""
    cfg = config or ServingDatasetConfig()
    rng = np.random.default_rng(cfg.seed)
    samples = []
    for sample_idx in range(cfg.num_samples):
        if progress:
            print(f"generating sample {sample_idx + 1}/{cfg.num_samples}", flush=True)
        samples.append(_generate_sample(cfg, rng, sample_idx))
    dataset = _stack_samples(samples)
    dataset["metadata"] = np.array(json.dumps(asdict(cfg)), dtype=np.str_)
    return dataset


def save_serving_dataset(dataset: dict[str, np.ndarray], path: str | Path) -> Path:
    """Save a generated serving dataset to a compressed NumPy archive."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **dataset)
    return output


def _generate_sample(cfg: ServingDatasetConfig, rng: np.random.Generator, sample_idx: int) -> dict[str, np.ndarray]:
    world_size = np.asarray(cfg.world_size, dtype=np.float32)
    buildings = _sample_buildings(rng, cfg.world_size)

    for _ in range(cfg.max_retries):
        target_index = int(rng.integers(0, cfg.person_count))
        drone = Drone(position=_sample_drone_position(rng, world_size, buildings, cfg.safety_margin))
        people = setup_people_initial_position(
            drone.position,
            world_size,
            cfg.person_count,
            buildings=buildings,
            margin=cfg.safety_margin,
            seed=cfg.seed + sample_idx,
        )
        if any(is_position_in_obstacles(person.position, buildings, margin=cfg.safety_margin) for person in people):
            continue
        people[target_index].command = "coffee"

        selected = select_coffee_target(people)
        try:
            hover_goal = compute_hover_goal(
                selected.person.position,
                approach_position=drone.position,
                obstacles=buildings,
                d_s=cfg.hover_distance,
                hover_height=cfg.hover_height,
                margin=cfg.safety_margin,
                world_size=world_size,
            )
        except ValueError:
            continue
        if is_position_in_obstacles(hover_goal.position, buildings, margin=cfg.safety_margin):
            continue

        expert_cfg = ExpertGeneratorConfig(
            world_size=tuple(float(v) for v in cfg.world_size),
            grid_resolution=cfg.grid_resolution,
            safety_margin=cfg.safety_margin,
            horizon=cfg.horizon,
            max_velocity=cfg.max_velocity,
            max_acceleration=cfg.max_acceleration,
        )
        try:
            expert = generate_expert_trajectory(
                start=_state_from_position(drone.position),
                hover_goal=hover_goal.position,
                buildings=buildings,
                people=people,
                config=expert_cfg,
            )
        except RuntimeError:
            continue

        people_states = _people_states(people)
        future_predictions = predict_people_positions(people, expert.time)
        return {
            "drone_start": _state_from_position(drone.position),
            "hover_goal": np.array([hover_goal.position[0], hover_goal.position[1], hover_goal.position[2], hover_goal.yaw], dtype=np.float32),
            "target_person_state": people_states[selected.index],
            "people_states": people_states,
            "future_people_predictions": future_predictions,
            "building_geometry": _building_geometry(buildings),
            "command_embedding": np.array([1.0, 0.0], dtype=np.float32),
            "target_index": np.array(selected.index, dtype=np.int64),
            "expert_trajectory": np.concatenate([expert.position, expert.velocity, expert.acceleration], axis=-1).astype(np.float32),
        }

    raise RuntimeError("Unable to generate a valid serving dataset sample.")


def _sample_buildings(rng: np.random.Generator, world_size: tuple[float, float, float]) -> list[BoxBuilding]:
    """Return setup-style buildings for serving dataset generation."""
    del rng, world_size
    return make_buildings()


def _sample_drone_position(
    rng: np.random.Generator,
    world_size: np.ndarray,
    buildings: list[BoxBuilding],
    margin: float,
) -> np.ndarray:
    half_x = float(world_size[0]) / 2.0
    half_y = float(world_size[1]) / 2.0
    for _ in range(200):
        position = np.array(
            [
                rng.uniform(-half_x + margin, half_x - margin),
                rng.uniform(-half_y + margin, half_y - margin),
                rng.uniform(0.8, max(0.9, float(world_size[2]) - margin)),
            ],
            dtype=np.float32,
        )
        if not is_position_in_obstacles(position, buildings, margin=margin):
            return position
    return np.array([-half_x + 1.0, -half_y + 1.0, 1.0], dtype=np.float32)


def _state_from_position(position: Any) -> np.ndarray:
    values = np.asarray(position, dtype=np.float32).reshape(-1)
    state = np.zeros(6, dtype=np.float32)
    state[:3] = values[:3]
    return state


def _people_states(people: list[PersonAgent]) -> np.ndarray:
    return np.stack([np.concatenate([person.position, person.velocity]).astype(np.float32) for person in people])


def _building_geometry(buildings: list[BoxBuilding]) -> np.ndarray:
    return np.stack([np.concatenate([building.center, building.size]).astype(np.float32) for building in buildings])


def _stack_samples(samples: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = samples[0].keys()
    return {key: np.stack([sample[key] for sample in samples]) for key in keys}
