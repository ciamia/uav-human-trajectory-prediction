from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from data.serving_dataset import ServingDatasetConfig, generate_serving_dataset, save_serving_dataset


def run_generation(
    output: str | Path = "outputs/serving_dataset.npz",
    config_path: str | Path | None = None,
    num_samples: int | None = None,
    seed: int | None = None,
) -> Path:
    """Generate and save a Milestone 4 serving dataset."""
    config = _load_config(config_path)
    if num_samples is not None:
        config = ServingDatasetConfig(**{**config.__dict__, "num_samples": int(num_samples)})
    if seed is not None:
        config = ServingDatasetConfig(**{**config.__dict__, "seed": int(seed)})

    dataset = generate_serving_dataset(config, progress=True)
    output_path = save_serving_dataset(dataset, output)
    print(f"saved dataset to {output_path}")
    print(f"samples: {dataset['drone_start'].shape[0]}")
    print(f"horizon: {dataset['expert_trajectory'].shape[1]}")
    print(f"people: {dataset['people_states'].shape[1]}")
    print(f"buildings: {dataset['building_geometry'].shape[1]}")
    return output_path


def _load_config(config_path: str | Path | None) -> ServingDatasetConfig:
    if config_path is None:
        return ServingDatasetConfig()
    with open(config_path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_mapping(raw)


def config_from_mapping(raw: dict[str, Any]) -> ServingDatasetConfig:
    """Build a serving dataset config from a project YAML mapping."""
    dataset = dict(raw.get("serving_dataset", raw.get("dataset", {})))
    sim = dict(raw.get("sim", {}))
    planning = dict(raw.get("planning", {}))

    values: dict[str, Any] = {}
    if "num_samples" in dataset:
        values["num_samples"] = int(dataset["num_samples"])
    if "horizon" in dataset or "horizon" in sim:
        values["horizon"] = int(dataset.get("horizon", sim.get("horizon")))
    if "seed" in dataset or "seed" in raw:
        values["seed"] = int(dataset.get("seed", raw.get("seed", 0)))
    if "person_count" in dataset:
        values["person_count"] = int(dataset["person_count"])
    if "world_size" in dataset or "world_size" in sim:
        values["world_size"] = tuple(float(v) for v in dataset.get("world_size", sim.get("world_size")))
    if "safety_margin" in dataset or "safety_margin" in planning:
        values["safety_margin"] = float(dataset.get("safety_margin", planning.get("safety_margin")))
    if "hover_distance" in dataset:
        values["hover_distance"] = float(dataset["hover_distance"])
    if "hover_height" in dataset:
        values["hover_height"] = float(dataset["hover_height"])
    if "max_velocity" in dataset or "max_velocity" in planning:
        values["max_velocity"] = float(dataset.get("max_velocity", planning.get("max_velocity")))
    if "max_acceleration" in dataset or "max_acceleration" in planning:
        values["max_acceleration"] = float(dataset.get("max_acceleration", planning.get("max_acceleration")))
    if "grid_resolution" in dataset:
        values["grid_resolution"] = float(dataset["grid_resolution"])

    return ServingDatasetConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Optional YAML config.")
    parser.add_argument("--output", default="outputs/serving_dataset.npz")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    run_generation(output=args.output, config_path=args.config, num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
