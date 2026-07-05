from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Iterable, Sequence

import numpy as np

from planning.refiner import RefinedTrajectory, refine_waypoints


@dataclass(frozen=True)
class ExpertGeneratorConfig:
    """Configuration for the Milestone 3 heuristic expert trajectory generator."""

    world_size: tuple[float, float, float] = (30.0, 30.0, 20.0)
    grid_resolution: float = 0.35
    safety_margin: float = 0.35
    person_radius: float = 0.3
    horizon: int = 96
    total_duration: float | None = None
    max_velocity: float = 3.0
    max_acceleration: float = 5.0
    altitude_bump: float = 0.35


@dataclass(frozen=True)
class ExpertTrajectory:
    """Generated expert references and the coarse waypoints used to create them."""

    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    waypoints: np.ndarray

    def as_state_trajectory(self) -> np.ndarray:
        """Return the trajectory as [x, y, z, vx, vy, vz]."""
        return np.concatenate([self.position, self.velocity], axis=-1).astype(np.float32)


@dataclass(frozen=True)
class BoxObstacle:
    """Axis-aligned box obstacle normalized from Genesis building objects."""

    center: np.ndarray
    size: np.ndarray


def generate_expert_trajectory(
    start: Any,
    hover_goal: Any,
    buildings: Sequence[Any] | None = None,
    people: Sequence[Any] | None = None,
    people_predictions: Any | None = None,
    config: ExpertGeneratorConfig | None = None,
) -> ExpertTrajectory:
    """Generate a smooth expert trajectory from drone start to hover goal.

    The planner is intentionally simple for Milestone 3: a conservative 2D A*
    route around building footprints and predicted people motion, followed by
    cubic/quintic smoothing into position, velocity, and acceleration references.
    """
    cfg = config or ExpertGeneratorConfig()
    start_position = _vec3(start, "start")
    goal_position = _vec3(hover_goal, "hover_goal")
    boxes = [_box_obstacle(item) for item in buildings or []]
    prediction_times = _planning_times(cfg)
    predicted_people = predict_people_positions(people or [], prediction_times, people_predictions)

    xy_path = _astar_xy_path(start_position, goal_position, boxes, predicted_people, cfg)
    waypoints = _lift_xy_path(xy_path, start_position, goal_position, cfg)
    duration = cfg.total_duration if cfg.total_duration is not None else _duration_from_path(waypoints, cfg.max_velocity)

    refined = refine_waypoints(
        waypoints,
        total_duration=duration,
        num_samples=cfg.horizon,
        method="quintic",
        max_velocity=cfg.max_velocity,
        max_acceleration=cfg.max_acceleration,
    )
    return ExpertTrajectory(
        time=refined.time,
        position=refined.position_refs,
        velocity=refined.velocity_refs,
        acceleration=refined.acceleration_refs,
        waypoints=waypoints.astype(np.float32),
    )


def predict_people_positions(
    people: Sequence[Any],
    times: np.ndarray,
    people_predictions: Any | None = None,
) -> np.ndarray:
    """Return predicted people positions with shape [P, K, 3]."""
    if people_predictions is not None:
        predictions = _to_numpy(people_predictions)
        if predictions.ndim == 2 and predictions.shape[1] >= 3:
            predictions = predictions[None, :, :3]
        if predictions.ndim != 3 or predictions.shape[-1] < 3:
            raise ValueError("people_predictions must have shape [K, 3] or [P, K, 3]")
        return predictions[..., :3].astype(np.float32)

    predicted: list[np.ndarray] = []
    for person in people:
        position = _vec3(getattr(person, "position", person), "person.position")
        velocity = _vec3(getattr(person, "velocity", np.zeros(3, dtype=np.float32)), "person.velocity")
        predicted.append(position[None, :] + times[:, None] * velocity[None, :])
    if not predicted:
        return np.zeros((0, len(times), 3), dtype=np.float32)
    return np.stack(predicted).astype(np.float32)


def building_collision_mask(positions: Any, buildings: Sequence[Any], margin: float = 0.35) -> np.ndarray:
    """Return True for positions inside any inflated box building."""
    points = _to_numpy(positions).reshape(-1, 3)
    mask = np.zeros(points.shape[0], dtype=bool)
    for building in buildings:
        box = _box_obstacle(building)
        half_size = box.size / 2.0 + float(margin)
        mask |= np.all(np.abs(points - box.center[None, :]) <= half_size[None, :], axis=-1)
    return mask


def people_collision_mask(
    positions: Any,
    predicted_people: Any,
    margin: float = 0.35,
    person_radius: float = 0.3,
) -> np.ndarray:
    """Return True for positions too close to predicted people in the horizontal plane."""
    points = _to_numpy(positions).reshape(-1, 3)
    predictions = _to_numpy(predicted_people)
    if predictions.size == 0:
        return np.zeros(points.shape[0], dtype=bool)
    if predictions.ndim == 2:
        predictions = predictions[None, :, :3]
    if predictions.ndim != 3:
        raise ValueError("predicted_people must have shape [P, K, 3]")

    time_ids = np.linspace(0, predictions.shape[1] - 1, points.shape[0]).round().astype(int)
    clearance = float(margin) + float(person_radius)
    mask = np.zeros(points.shape[0], dtype=bool)
    for idx, point in enumerate(points):
        people_xy = predictions[:, time_ids[idx], :2]
        distances = np.linalg.norm(people_xy - point[None, :2], axis=-1)
        mask[idx] = bool(np.any(distances <= clearance))
    return mask


def trajectory_collision_mask(
    positions: Any,
    buildings: Sequence[Any] | None = None,
    predicted_people: Any | None = None,
    config: ExpertGeneratorConfig | None = None,
) -> np.ndarray:
    """Return True where a trajectory violates building or predicted-person clearance."""
    cfg = config or ExpertGeneratorConfig()
    points = _to_numpy(positions).reshape(-1, 3)
    mask = building_collision_mask(points, buildings or [], margin=cfg.safety_margin)
    if predicted_people is not None:
        mask |= people_collision_mask(
            points,
            predicted_people,
            margin=cfg.safety_margin,
            person_radius=cfg.person_radius,
        )
    return mask


def _planning_times(cfg: ExpertGeneratorConfig) -> np.ndarray:
    duration = cfg.total_duration if cfg.total_duration is not None else 1.0
    return np.linspace(0.0, float(duration), cfg.horizon, dtype=np.float32)


def _astar_xy_path(
    start: np.ndarray,
    goal: np.ndarray,
    buildings: Sequence[BoxObstacle],
    predicted_people: np.ndarray,
    cfg: ExpertGeneratorConfig,
) -> np.ndarray:
    world = np.asarray(cfg.world_size, dtype=np.float32)
    half_x = float(world[0]) / 2.0
    half_y = float(world[1]) / 2.0
    resolution = float(cfg.grid_resolution)
    xs = np.arange(-half_x, half_x + resolution * 0.5, resolution, dtype=np.float32)
    ys = np.arange(-half_y, half_y + resolution * 0.5, resolution, dtype=np.float32)

    start_node = _point_to_node(start[:2], xs, ys)
    goal_node = _point_to_node(goal[:2], xs, ys)
    allowed = {start_node, goal_node}

    frontier: list[tuple[float, tuple[int, int]]] = []
    heappush(frontier, (0.0, start_node))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_node: None}
    cost_so_far: dict[tuple[int, int], float] = {start_node: 0.0}

    while frontier:
        _, current = heappop(frontier)
        if current == goal_node:
            break
        for neighbor, step_cost in _neighbors(current, len(xs), len(ys)):
            if neighbor not in allowed and _blocked_xy(np.array([xs[neighbor[0]], ys[neighbor[1]]], dtype=np.float32), buildings, predicted_people, cfg):
                continue
            new_cost = cost_so_far[current] + step_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + _node_distance(neighbor, goal_node)
                heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    if goal_node not in came_from:
        raise RuntimeError("Unable to generate an expert path through the current world.")

    nodes = _reconstruct_nodes(came_from, goal_node)
    points = np.array([[xs[node[0]], ys[node[1]]] for node in nodes], dtype=np.float32)
    if len(points) == 1:
        return np.stack([start[:2], goal[:2]]).astype(np.float32)
    points[0] = start[:2]
    points[-1] = goal[:2]
    return _simplify_xy(points)


def _blocked_xy(
    xy: np.ndarray,
    buildings: Sequence[BoxObstacle],
    predicted_people: np.ndarray,
    cfg: ExpertGeneratorConfig,
) -> bool:
    clearance = float(cfg.safety_margin) + float(cfg.grid_resolution)
    for building in buildings:
        half = building.size[:2] / 2.0 + clearance
        if bool(np.all(np.abs(xy - building.center[:2]) <= half)):
            return True
    if predicted_people.size:
        distances = np.linalg.norm(predicted_people[..., :2].reshape(-1, 2) - xy[None, :], axis=-1)
        if bool(np.any(distances <= float(cfg.person_radius) + clearance)):
            return True
    return False


def _lift_xy_path(
    xy_path: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    cfg: ExpertGeneratorConfig,
) -> np.ndarray:
    count = len(xy_path)
    alpha = np.linspace(0.0, 1.0, count, dtype=np.float32)
    z = (1.0 - alpha) * start[2] + alpha * goal[2]
    z += float(cfg.altitude_bump) * np.sin(np.pi * alpha)
    z = np.clip(z, 0.2, float(cfg.world_size[2]) - 0.2)
    waypoints = np.column_stack([xy_path, z]).astype(np.float32)
    waypoints[0] = start
    waypoints[-1] = goal
    return waypoints


def _simplify_xy(points: np.ndarray) -> np.ndarray:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    previous_direction = None
    for idx in range(1, len(points) - 1):
        direction = np.sign(points[idx + 1] - points[idx]).astype(np.int8)
        if previous_direction is None or not np.array_equal(direction, previous_direction):
            simplified.append(points[idx])
        previous_direction = direction
    simplified.append(points[-1])
    return np.asarray(simplified, dtype=np.float32)


def _duration_from_path(waypoints: np.ndarray, max_velocity: float) -> float:
    length = float(np.linalg.norm(np.diff(waypoints, axis=0), axis=-1).sum())
    return max(2.0, length / max(float(max_velocity) * 0.55, 1e-6))


def _neighbors(node: tuple[int, int], nx: int, ny: int) -> Iterable[tuple[tuple[int, int], float]]:
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            next_node = (node[0] + dx, node[1] + dy)
            if 0 <= next_node[0] < nx and 0 <= next_node[1] < ny:
                yield next_node, float(np.hypot(dx, dy))


def _point_to_node(point_xy: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> tuple[int, int]:
    x_idx = int(np.argmin(np.abs(xs - float(point_xy[0]))))
    y_idx = int(np.argmin(np.abs(ys - float(point_xy[1]))))
    return x_idx, y_idx


def _node_distance(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _reconstruct_nodes(
    came_from: dict[tuple[int, int], tuple[int, int] | None],
    goal_node: tuple[int, int],
) -> list[tuple[int, int]]:
    node: tuple[int, int] | None = goal_node
    path: list[tuple[int, int]] = []
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


def _box_obstacle(value: Any) -> BoxObstacle:
    if hasattr(value, "center") and hasattr(value, "size"):
        return BoxObstacle(center=_vec3(value.center, "building.center"), size=_vec3(value.size, "building.size"))
    array = _to_numpy(value).reshape(-1)
    if array.shape[0] != 6:
        raise ValueError("buildings must be BoxBuilding-like objects or [x, y, z, sx, sy, sz]")
    return BoxObstacle(center=array[:3].copy(), size=array[3:6].copy())


def _vec3(value: Any, name: str) -> np.ndarray:
    values = _to_numpy(value).reshape(-1)
    if values.shape[0] < 3:
        raise ValueError(f"{name} must contain at least x, y, z")
    return values[:3].astype(np.float32).copy()


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, RefinedTrajectory):
        return value.position_refs
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
