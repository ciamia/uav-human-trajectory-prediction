from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class HoverGoal:
    """Hover pose near a target person."""

    position: np.ndarray
    yaw: float


def compute_hover_goal(
    target_position: Any,
    approach_position: Any | None = None,
    obstacles: Sequence[Any] | None = None,
    d_s: float = 1.0,
    hover_height: float = 1.5,
    margin: float = 0.25,
    world_size: Sequence[float] | None = None,
) -> HoverGoal:
    """Compute a collision-free hover position and yaw facing the target."""
    target = _vec3(target_position, "target_position")
    approach = target + np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if approach_position is not None:
        approach = _vec3(approach_position, "approach_position")

    base_direction = approach[:2] - target[:2]
    if np.linalg.norm(base_direction) < 1e-6:
        base_direction = np.array([1.0, 0.0], dtype=np.float32)
    base_angle = float(np.arctan2(base_direction[1], base_direction[0]))

    obstacle_list = list(obstacles or [])
    for radius_scale in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
        radius = float(d_s) * radius_scale
        for angle in _candidate_angles(base_angle):
            position = np.array(
                [
                    target[0] + radius * np.cos(angle),
                    target[1] + radius * np.sin(angle),
                    float(hover_height),
                ],
                dtype=np.float32,
            )
            position = _clamp_to_world(position, world_size, margin)
            if not is_position_in_obstacles(position, obstacle_list, margin):
                return HoverGoal(position=position, yaw=_yaw_facing(position, target))

    raise ValueError("Could not find a collision-free hover goal")


def is_position_in_obstacles(position: Any, obstacles: Sequence[Any], margin: float = 0.25) -> bool:
    """Return True when a 3D position lies inside any supported obstacle."""
    point = _vec3(position, "position")
    return any(_inside_obstacle(point, obstacle, margin) for obstacle in obstacles)


def _candidate_angles(base_angle: float) -> list[float]:
    offsets = [0.0]
    for step in range(1, 13):
        offset = step * np.pi / 12.0
        offsets.extend([offset, -offset])
    return [base_angle + offset for offset in offsets]


def _inside_obstacle(point: np.ndarray, obstacle: Any, margin: float) -> bool:
    if hasattr(obstacle, "center") and hasattr(obstacle, "size"):
        center = _vec3(obstacle.center, "obstacle.center")
        size = _vec3(obstacle.size, "obstacle.size")
        half_size = size / 2.0 + float(margin)
        return bool(np.all(np.abs(point - center) <= half_size))

    values = np.asarray(obstacle, dtype=np.float32).reshape(-1)
    if values.shape[0] == 4:
        center = values[:3]
        radius = float(values[3]) + float(margin)
        return bool(np.linalg.norm(point - center) <= radius)
    if values.shape[0] == 6:
        center = values[:3]
        size = values[3:6]
        half_size = size / 2.0 + float(margin)
        return bool(np.all(np.abs(point - center) <= half_size))
    raise ValueError("obstacles must be BoxBuilding-like objects, [x,y,z,r], or [x,y,z,sx,sy,sz]")


def _clamp_to_world(position: np.ndarray, world_size: Sequence[float] | None, margin: float) -> np.ndarray:
    if world_size is None:
        return position
    size = np.asarray(world_size, dtype=np.float32).reshape(-1)
    if size.shape[0] < 2:
        raise ValueError("world_size must contain at least x and y extents")
    clamped = position.copy()
    half_x = float(size[0]) / 2.0 - float(margin)
    half_y = float(size[1]) / 2.0 - float(margin)
    clamped[0] = np.clip(clamped[0], -half_x, half_x)
    clamped[1] = np.clip(clamped[1], -half_y, half_y)
    return clamped


def _yaw_facing(position: np.ndarray, target: np.ndarray) -> float:
    delta = target[:2] - position[:2]
    return float(np.arctan2(delta[1], delta[0]))


def _vec3(value: Any, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape[0] < 3:
        raise ValueError(f"{name} must contain at least 3 values")
    return vector[:3].copy()
