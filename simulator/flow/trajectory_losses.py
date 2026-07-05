from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TrajectoryLossWeights:
    """Weights for Milestone 6 soft trajectory objectives."""

    goal: float = 1.0
    safe_hover: float = 0.5
    smoothness: float = 0.1
    path_time: float = 0.05
    building_collision: float = 2.0
    people_collision: float = 2.0
    altitude: float = 1.0


def trajectory_objective_losses(
    trajectory: torch.Tensor,
    condition: Mapping[str, torch.Tensor],
    weights: TrajectoryLossWeights | None = None,
    safety_margin: float = 0.35,
    person_radius: float = 0.3,
    hover_distance: float = 1.0,
    min_altitude: float = 0.2,
    max_altitude: float = 4.0,
) -> dict[str, torch.Tensor]:
    """Compute weighted Milestone 6 soft trajectory objective losses."""
    w = weights or TrajectoryLossWeights()
    unweighted = {
        "goal": goal_loss(trajectory, condition["hover_goal"]),
        "safe_hover": safe_hover_distance_loss(
            trajectory,
            condition["target_person_state"],
            desired_distance=hover_distance,
        ),
        "smoothness": smoothness_loss(trajectory),
        "path_time": path_time_loss(trajectory),
        "building_collision": building_collision_loss(
            trajectory,
            condition.get("building_geometry"),
            margin=safety_margin,
        ),
        "people_collision": people_collision_loss(
            trajectory,
            condition.get("future_people_predictions"),
            margin=safety_margin,
            person_radius=person_radius,
        ),
        "altitude": altitude_loss(trajectory, min_altitude=min_altitude, max_altitude=max_altitude),
    }
    weighted = {
        "goal": unweighted["goal"] * w.goal,
        "safe_hover": unweighted["safe_hover"] * w.safe_hover,
        "smoothness": unweighted["smoothness"] * w.smoothness,
        "path_time": unweighted["path_time"] * w.path_time,
        "building_collision": unweighted["building_collision"] * w.building_collision,
        "people_collision": unweighted["people_collision"] * w.people_collision,
        "altitude": unweighted["altitude"] * w.altitude,
    }
    weighted["total"] = sum(weighted.values())
    return weighted


def goal_loss(trajectory: torch.Tensor, hover_goal: torch.Tensor) -> torch.Tensor:
    """Penalize final trajectory position error to hover goal position."""
    positions = _positions(trajectory)
    goal_position = _matrix(hover_goal, positions.shape[0], "hover_goal")[..., :3]
    return F.mse_loss(positions[:, -1], goal_position)


def safe_hover_distance_loss(
    trajectory: torch.Tensor,
    target_person_state: torch.Tensor,
    desired_distance: float = 1.0,
) -> torch.Tensor:
    """Penalize final horizontal distance from target person away from desired hover distance."""
    positions = _positions(trajectory)
    target = _matrix(target_person_state, positions.shape[0], "target_person_state")[..., :3]
    horizontal_distance = torch.linalg.norm(positions[:, -1, :2] - target[:, :2], dim=-1)
    return ((horizontal_distance - float(desired_distance)) ** 2).mean()


def smoothness_loss(trajectory: torch.Tensor) -> torch.Tensor:
    """Penalize acceleration-like and jerk-like finite differences."""
    positions = _positions(trajectory)
    velocities = _velocities(trajectory)
    if positions.shape[1] < 3:
        return positions.new_zeros(())
    second_difference = positions[:, 2:] - 2.0 * positions[:, 1:-1] + positions[:, :-2]
    velocity_difference = velocities[:, 1:] - velocities[:, :-1]
    return second_difference.square().mean() + velocity_difference.square().mean()


def path_time_loss(trajectory: torch.Tensor) -> torch.Tensor:
    """Softly prefer shorter paths and lower speeds for a fixed-horizon trajectory."""
    positions = _positions(trajectory)
    velocities = _velocities(trajectory)
    if positions.shape[1] < 2:
        return velocities.square().mean()
    segment_lengths = torch.linalg.norm(positions[:, 1:] - positions[:, :-1], dim=-1)
    return segment_lengths.mean() + 0.05 * velocities.square().mean()


def building_collision_loss(
    trajectory: torch.Tensor,
    building_geometry: torch.Tensor | None,
    margin: float = 0.35,
) -> torch.Tensor:
    """Soft box-collision penalty for building geometry [B, N, 6]."""
    positions = _positions(trajectory)
    if building_geometry is None:
        return positions.new_zeros(())
    buildings = _sequence(building_geometry, positions.shape[0], 6, "building_geometry").to(positions)
    if buildings.shape[1] == 0:
        return positions.new_zeros(())

    centers = buildings[:, None, :, :3]
    half_sizes = buildings[:, None, :, 3:6] * 0.5
    delta = torch.abs(positions[:, :, None, :] - centers) - half_sizes
    outside = torch.clamp(delta, min=0.0)
    outside_distance = torch.linalg.norm(outside, dim=-1)
    inside_distance = torch.minimum(delta.max(dim=-1).values, torch.zeros_like(outside_distance))
    signed_distance = outside_distance + inside_distance
    return F.relu(float(margin) - signed_distance).square().mean()


def people_collision_loss(
    trajectory: torch.Tensor,
    future_people_predictions: torch.Tensor | None,
    margin: float = 0.35,
    person_radius: float = 0.3,
) -> torch.Tensor:
    """Soft dynamic-people collision penalty using horizontal distance."""
    positions = _positions(trajectory)
    if future_people_predictions is None:
        return positions.new_zeros(())
    people = _sequence(future_people_predictions, positions.shape[0], 3, "future_people_predictions").to(positions)
    if people.shape[1] == 0:
        return positions.new_zeros(())

    people = _match_people_horizon(people, positions.shape[1])
    distances = torch.linalg.norm(positions[:, None, :, :2] - people[..., :2], dim=-1)
    clearance = float(margin) + float(person_radius)
    return F.relu(clearance - distances).square().mean()


def altitude_loss(
    trajectory: torch.Tensor,
    min_altitude: float = 0.2,
    max_altitude: float = 4.0,
) -> torch.Tensor:
    """Softly enforce altitude bounds."""
    z = _positions(trajectory)[..., 2]
    low = F.relu(float(min_altitude) - z).square()
    high = F.relu(z - float(max_altitude)).square()
    return (low + high).mean()


def _positions(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.ndim != 3 or trajectory.shape[-1] < 3:
        raise ValueError("trajectory must have shape [B, T, D] with D >= 3")
    return trajectory[..., :3]


def _velocities(trajectory: torch.Tensor) -> torch.Tensor:
    if trajectory.shape[-1] >= 6:
        return trajectory[..., 3:6]
    return torch.zeros_like(_positions(trajectory))


def _matrix(value: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] != batch or value.shape[-1] < 3:
        raise ValueError(f"{name} must have shape [B, D] with D >= 3")
    return value


def _sequence(value: torch.Tensor, batch: int, dim: int, name: str) -> torch.Tensor:
    if value.ndim < 3 or value.shape[0] != batch or value.shape[-1] != dim:
        raise ValueError(f"{name} must have shape [B, ..., {dim}]")
    return value


def _match_people_horizon(people: torch.Tensor, horizon: int) -> torch.Tensor:
    people = people.reshape(people.shape[0], people.shape[1], -1, people.shape[-1])
    if people.shape[2] == horizon:
        return people
    indices = torch.linspace(0, people.shape[2] - 1, horizon, device=people.device).round().long()
    return people.index_select(2, indices)
