from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Any

from flow.sampler import sample
from planning.hover_goal import compute_hover_goal
from planning.refiner import RefinedTrajectory, refine_waypoints
from planning.target_selection import select_coffee_target


import torch
import timeit
import numpy as np
from pathlib import Path
from dataclasses import asdict, dataclass
import utils.trajectory_generator as traj_fun

if TYPE_CHECKING:
    from models.flow_transformer import FlowTransformer
    from scripts.train_serving_fm import TrainServingConfig

class SampleServingConfig:
    """Configuration for Milestone 8 serving trajectory sampling demo."""

    checkpoint: str = "outputs/serving_fm/checkpoints/best.pt"
    output: str = "outputs/sample_serving_trajectory.png"
    video: str = "outputs/sample_serving_trajectory.mp4"
    device: str = "auto"
    seed: int = 0
    horizon: int = 96
    sample_steps: int = 32
    sample_method: str = "heun"
    headless: bool = False
    safety_margin: float = 0.35
    max_velocity: float = 3.0
    max_acceleration: float = 5.0
    use_mpc: bool = True
    mpc_position_control: bool = False
    mpc_fullpose_control: bool = True
    mpc_horizon: int = 12
    genesis_config: str = "configs/task_config.yaml"
    keep_open: bool = False


@dataclass
class BoxESDF:
    """Signed-distance field backed by axis-aligned box obstacles."""

    centers: np.ndarray
    sizes: np.ndarray
    center: np.ndarray | None
    resolution: float
    local_radius: float

    def __post_init__(self) -> None:
        self.centers = np.asarray(self.centers, dtype=np.float32).reshape(-1, 3)
        self.sizes = np.asarray(self.sizes, dtype=np.float32).reshape(-1, 3)
        if self.center is not None:
            self.center = np.asarray(self.center, dtype=np.float32).reshape(3)
        if self.centers.shape != self.sizes.shape:
            raise ValueError("centers and sizes must both have shape [N, 3]")

    def distance(self, point: np.ndarray) -> float:
        """Return positive free-space distance and non-positive inside distance."""
        point = np.asarray(point, dtype=np.float32).reshape(3)
        if len(self.centers) == 0:
            return float("inf")
        half_sizes = 0.5 * self.sizes
        delta = np.abs(point[None, :] - self.centers) - half_sizes
        outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
        inside = np.minimum(np.max(delta, axis=1), 0.0)
        return float(np.min(outside + inside))

    def bounds(self, start: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return planning bounds that cover the query and current obstacle set."""
        start = np.asarray(start, dtype=np.float32).reshape(3)
        goal = np.asarray(goal, dtype=np.float32).reshape(3)
        points = [start, goal]
        if self.center is not None:
            points.append(self.center)
        if len(self.centers):
            half_sizes = 0.5 * self.sizes
            points.append(np.min(self.centers - half_sizes, axis=0))
            points.append(np.max(self.centers + half_sizes, axis=0))
        stacked = np.vstack(points)
        margin = max(float(self.local_radius), 4.0 * float(self.resolution))
        lower = np.min(stacked, axis=0) - margin
        upper = np.max(stacked, axis=0) + margin
        lower[2] = max(0.0, lower[2])
        return lower.astype(np.float32), upper.astype(np.float32)


@dataclass
class SplineTrajectory:
    """Trajectory wrapper exposing position, velocity, and acceleration queries."""

    refined: RefinedTrajectory

    def _interp(self, values: np.ndarray, t: float | np.ndarray) -> np.ndarray:
        query = np.asarray(t, dtype=np.float32)
        flat = query.reshape(-1)
        result = np.stack(
            [np.interp(flat, self.refined.time, values[:, axis]) for axis in range(3)],
            axis=-1,
        ).astype(np.float32)
        return result[0] if query.ndim == 0 else result.reshape(query.shape + (3,))

    def evaluate(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate position at t."""
        return self._interp(self.refined.position_refs, t)

    def velocity(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate velocity at t."""
        return self._interp(self.refined.velocity_refs, t)

    def acceleration(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate acceleration at t."""
        return self._interp(self.refined.acceleration_refs, t)


@dataclass
class MINCOTrajectory:
    """Trajectory wrapper exposing exact MINCO position, velocity, and acceleration queries."""

    coeffs: np.ndarray
    durations: np.ndarray
    refined: RefinedTrajectory

    def __post_init__(self) -> None:
        self.coeffs = np.asarray(self.coeffs, dtype=np.float32)
        self.durations = np.asarray(self.durations, dtype=np.float32).reshape(-1)
        self.time = np.concatenate([[0.0], np.cumsum(self.durations)]).astype(np.float32)

    def _evaluate(self, t: float | np.ndarray, derivative: int) -> np.ndarray:
        query = np.asarray(t, dtype=np.float32)
        flat = np.clip(query.reshape(-1), self.time[0], self.time[-1])
        indices = np.searchsorted(self.time[1:], flat, side="right")
        indices = np.minimum(indices, len(self.durations) - 1)
        local_t = flat - self.time[indices]
        result = np.zeros((flat.shape[0], 3), dtype=np.float32)

        for piece_index in np.unique(indices):
            mask = indices == piece_index
            coeffs = self.coeffs[piece_index]
            if derivative == 0:
                factors = np.ones(coeffs.shape[0], dtype=np.float32)
                active_coeffs = coeffs
            elif derivative == 1:
                factors = np.arange(1, coeffs.shape[0], dtype=np.float32)
                active_coeffs = coeffs[1:] * factors[:, None]
            elif derivative == 2:
                factors = np.arange(2, coeffs.shape[0], dtype=np.float32)
                active_coeffs = coeffs[2:] * (factors * (factors - 1.0))[:, None]
            else:
                raise ValueError("derivative must be 0, 1, or 2")
            powers = np.stack(
                [local_t[mask] ** power for power in range(active_coeffs.shape[0])],
                axis=1,
            )
            result[mask] = powers @ active_coeffs

        return result[0] if query.ndim == 0 else result.reshape(query.shape + (3,))

    def evaluate(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate position at t."""
        return self._evaluate(t, derivative=0)

    def velocity(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate velocity at t."""
        return self._evaluate(t, derivative=1)

    def acceleration(self, t: float | np.ndarray) -> np.ndarray:
        """Evaluate acceleration at t."""
        return self._evaluate(t, derivative=2)


def build_esdf_from_boxes(
    obstacles: dict,
    center: np.ndarray | None,
    resolution: float,
    local_radius: float,
    inflation_radius: float,
) -> BoxESDF:
    """Build a lightweight ESDF from world obstacle boxes."""
    centers = []
    sizes = []
    for group in obstacles.values():
        group_centers = np.asarray(group.get("positions", []), dtype=np.float32).reshape(-1, 3)
        group_sizes = np.asarray(group.get("sizes", []), dtype=np.float32).reshape(-1, 3)
        if len(group_centers) == 0:
            continue
        centers.append(group_centers)
        sizes.append(group_sizes + 2.0 * float(inflation_radius))
    if centers:
        all_centers = np.concatenate(centers, axis=0)
        all_sizes = np.concatenate(sizes, axis=0)
    else:
        all_centers = np.zeros((0, 3), dtype=np.float32)
        all_sizes = np.zeros((0, 3), dtype=np.float32)
    return BoxESDF(all_centers, all_sizes, center, float(resolution), float(local_radius))

def load_model(checkpoint: str | Path, device: torch.device, fallback_horizon: int) -> tuple[FlowTransformer, TrainServingConfig, int]:
    """Load a trained serving Flow Transformer checkpoint."""
    from scripts.train_serving_fm import TrainServingConfig, build_model

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=device)
    config_values = payload.get("config", {})
    train_config = TrainServingConfig(**{**asdict(TrainServingConfig()), **{k: v for k, v in config_values.items() if k in TrainServingConfig.__dataclass_fields__}})
    state_dict = payload.get("model_state_dict", payload.get("model"))
    if state_dict is None:
        raise ValueError("checkpoint must contain model_state_dict or model")
    horizon = int(state_dict.get("position_embedding", torch.empty(1, fallback_horizon)).shape[1])
    model = build_model(train_config, horizon).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, train_config, horizon

class TrajectorPlanner:
    """
    Real-time 3D trajectory planner for drone navigation.

    Main usage:
        planner.env_ESDF_map()
        waypoints = planner.waypoints_A_star(start, goal)
        traj = planner.B_spline(ts, waypoints)

    Recommended online pipeline:
        1. Build/update local ESDF map.
        2. Generate candidate waypoints.
        3. Simplify waypoints with shortcutting.
        4. Validate all waypoint segments.
        5. Generate smooth trajectory.
        6. Check trajectory collision and dynamic constraints.
        7. Return trajectory or fallback.
    """

    def __init__(self, world, config: SampleServingConfig):
        from scripts.train_serving_fm import resolve_device

        self.world = world
        self.buildings = world.buildings

        # Convert generator to list.
        # Original code used parentheses, which creates generators.
        self.building_position = [
            building.center() if callable(building.center) else building.center
            for building in self.buildings
        ]
        self.building_size = [
            building.size() if callable(building.size) else building.size
            for building in self.buildings
        ]

        # world.buildings
        # [BoxBuilding(center=array([  5.5, -10. ,   0. ], dtype=float32), size=array([ 8.,  4.,...e=float32), color=(0.35, 0.48, 0.78, 1.0)), 
        #  BoxBuilding(center=array([12., 10.,  0.], dtype=float32), size=array([6., 6., 5.], dt...e=float32), color=(0.72, 0.42, 0.28, 1.0))]
        # world.people
        # [PersonAgent(position=array([ 4.054066 , -6.8143134,  0.668    ], dtype=float32), velo...re='standing', routine_index=0, name_id=1), 
        #  PersonAgent(position=array([-13.587184, -14.310782,   0.7     ], dtype=float32), velo...osture='wide', routine_index=0, name_id=2), 
        #  PersonAgent(position=array([3.156419 , 6.7930984, 0.6888   ], dtype=float32), velocit...re='reaching', routine_index=0, name_id=3), 
        #  PersonAgent(position=array([ 1.2912998, 12.878144 ,  0.44384  ], dtype=float32), velo...ture='crouch', routine_index=0, name_id=4), 
        #  PersonAgent(position=array([  9.349265, -14.718941,   0.716   ], dtype=float32), velo...osture='lean', routine_index=0, name_id=5)]


        self.ESDF_map = None
        self.waypoints = None
        self.velocities = None
        self.ts = None

        # config values.
        self.radius_people = getattr(config, "radius_people", 0.2)
        self.radius_drone  = getattr(config, "radius_drone", 0.1)
        self.safety_radius = getattr(config, "safety_margin", 0.25)
        self.map_resolution = getattr(config, "map_resolution", 0.5)
        self.local_map_radius = getattr(config, "local_map_radius", 10.0)
        self.max_vel = getattr(config, "max_velocity", 3.0)
        self.max_acc = getattr(config, "max_acceleration", 3.0)

        device = resolve_device(config.device)
        model, train_config, horizon = load_model(
            config.checkpoint,
            device,
            config.horizon,
        )

        self.device = device
        self.flow_match_model = model
        self.train_config = train_config
        self.horizon = horizon

    # ------------------------------------------------------------------
    # ESDF / collision checking
    # ------------------------------------------------------------------

    def env_ESDF_map(self, ego_drone=None, target_person=None):
        """
        Build or update the local ESDF map.

        Object handling:
            ego drone      -> map center, NOT obstacle
            target person  -> goal, NOT obstacle
            other people   -> dynamic obstacles
            buildings      -> static box obstacles
            other drones   -> dynamic obstacles
        """

        # -----------------------------
        # People obstacles
        # -----------------------------
        position_people = []
        size_people = []
        for person in self.world.people:
            if target_person is not None and person is target_person:
                continue

            p = np.asarray(person.get_position(), dtype=np.float32)

            # People are approximated as vertical boxes/cylinders.
            s = np.asarray(
                [
                    2.0 * self.radius_people,
                    2.0 * self.radius_people,
                    2.0 * self.radius_people,
                ],
                dtype=np.float32,
            )

            position_people.append(p)
            size_people.append(s)

        # -----------------------------
        # Building obstacles
        # -----------------------------
        position_build = []
        size_build = []

        for building in self.world.buildings:
            # Your BoxBuilding already has center and size.
            if hasattr(building, "center"):
                center = building.center() if callable(building.center) else building.center
            else:
                center = building.position()

            if hasattr(building, "size"):
                size = building.size() if callable(building.size) else building.size
            else:
                size = building.get_size()

            position_build.append(np.asarray(center, dtype=np.float32))
            size_build.append(np.asarray(size, dtype=np.float32))

        # -----------------------------
        # Drone obstacles
        # -----------------------------
        position_drone = []
        size_drone = []

        # for drone in getattr(self.world, "drones", []):
        #     if ego_drone is not None and drone is ego_drone:
        #         continue

        #     p = np.asarray(drone.get_position(), dtype=np.float32)

        #     s = np.asarray(
        #         [
        #             2.0 * self.radius_drone,
        #             2.0 * self.radius_drone,
        #             self.radius_drone,
        #         ],
        #         dtype=np.float32,
        #     )

        #     position_drone.append(p)
        #     size_drone.append(s)

        obstacles = {
            "people": {
                "positions": np.asarray(position_people, dtype=np.float32),
                "sizes": np.asarray(size_people, dtype=np.float32),
                "type": "dynamic_box",
            },
            "buildings": {
                "positions": np.asarray(position_build, dtype=np.float32),
                "sizes": np.asarray(size_build, dtype=np.float32),
                "type": "static_box",
            },
            "drones": {
                "positions": np.asarray(position_drone, dtype=np.float32),
                "sizes": np.asarray(size_drone, dtype=np.float32),
                "type": "dynamic_box",
            },
        }

        # Ego drone position defines local map center.
        if ego_drone is not None:
            map_center = ego_drone
        else:
            map_center = None

        self.ESDF_map = build_esdf_from_boxes(
            obstacles=obstacles,
            center=map_center,
            resolution=self.map_resolution,
            local_radius=self.local_map_radius,
            inflation_radius=self.safety_radius,
        )

        return self.ESDF_map


    def query_esdf(self, point):
        """
        Return distance from point to nearest obstacle.

        """
        if self.ESDF_map is None:
            raise RuntimeError("ESDF map has not been built yet.")
        return self.ESDF_map.distance(point)

    def is_point_valid(self, point):
        """
        Check whether one 3D point is collision-free.
        """
        return self.query_esdf(point) > self.safety_radius

    def is_segment_valid(self, p0, p1, step=None):
        """
        Check whether the straight-line segment p0 -> p1 is collision-free.

        """
        p0 = np.asarray(p0, dtype=np.float32).reshape(3)
        p1 = np.asarray(p1, dtype=np.float32).reshape(3)
        if step is None:
            step = self.map_resolution
        distance = float(np.linalg.norm(p1 - p0))
        sample_count = max(int(np.ceil(distance / float(step))) + 1, 2)
        alphas = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)
        for alpha in alphas:
            point = (1.0 - alpha) * p0 + alpha * p1
            if not self.is_point_valid(point):
                return False
        return True

    def shortcut_waypoints(self, waypoints, max_dis=1) -> np.ndarray:
        """
        Simplify waypoint list using line-of-sight shortcutting.

        Example:
            [A, p1, p2, p3, B] -> [A, p3, B]
        if A -> p3 and p3 -> B are collision-free.
        """
        # TODO: please modify this function, ensure collosion free while the distance between two adjacent waypoints is less than max_dis, if larger, please interpolate 

        waypoints = self._as_waypoints(waypoints)
        if len(waypoints) <= 2:
            return waypoints

        simplified = [waypoints[0]]
        i = 0
        while i < len(waypoints) - 1:
            j = len(waypoints) - 1
            while j > i + 1 and not self.is_segment_valid(waypoints[i], waypoints[j]):
                j -= 1
            simplified.append(waypoints[j])
            i = j

        # if len(simplified) < 3:
        #     start = waypoints[0]
        #     goal = waypoints[-1]
        #     fallback = waypoints[len(waypoints) // 2]
        #     for candidate in waypoints[1:-1]:
        #         if self.is_segment_valid(start, candidate) and self.is_segment_valid(candidate, goal):
        #             fallback = candidate
        #             break
        #     simplified = [start, fallback, goal]

        return np.asarray(simplified, dtype=np.float32)

    # ------------------------------------------------------------------
    # Waypoint planners
    # ------------------------------------------------------------------

    def waypoints_A_star(self, start, goal):
        """
        Use 3D A* on voxel grid / ESDF to generate waypoints.
        """
        self._ensure_esdf()
        start = np.asarray(start, dtype=np.float32).reshape(3)
        goal = np.asarray(goal, dtype=np.float32).reshape(3)
        if not self.is_point_valid(start) or not self.is_point_valid(goal):
            raise ValueError("start and goal must be collision-free")
        if self.is_segment_valid(start, goal):
            self.waypoints = np.asarray([start, goal], dtype=np.float32)
            return self.waypoints

        lower, upper = self.ESDF_map.bounds(start, goal)
        resolution = float(self.map_resolution)
        max_index = np.floor((upper - lower) / resolution).astype(np.int32)

        def to_grid(point):
            return tuple(np.round((point - lower) / resolution).astype(np.int32))

        def to_world(index):
            return lower + resolution * np.asarray(index, dtype=np.float32)

        start_index = to_grid(start)
        goal_index = to_grid(goal)
        neighbors = [
            np.asarray([dx, dy, dz], dtype=np.int32)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if (dx, dy, dz) != (0, 0, 0)
        ]
        open_heap = [(0.0, start_index)]
        came_from = {}
        g_score = {start_index: 0.0}
        closed = set()

        for _ in range(60000):
            if not open_heap:
                break
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal_index:
                raw_path = self._reconstruct_grid_path(came_from, current, to_world, start, goal)
                self.waypoints = self.shortcut_waypoints(raw_path)
                return self.waypoints
            closed.add(current)
            current_array = np.asarray(current, dtype=np.int32)
            for offset in neighbors:
                next_array = current_array + offset
                if np.any(next_array < 0) or np.any(next_array > max_index):
                    continue
                neighbor = tuple(int(value) for value in next_array)
                if neighbor in closed:
                    continue
                point = to_world(neighbor)
                if not self.is_point_valid(point):
                    continue
                tentative = g_score[current] + float(np.linalg.norm(offset)) * resolution
                if tentative >= g_score.get(neighbor, float("inf")):
                    continue
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                priority = tentative + float(np.linalg.norm(point - goal))
                heapq.heappush(open_heap, (priority, neighbor))

        return self.waypoints_RRT_connect(start, goal)

    def waypoints_Kinodynamic_A_star(self, start_state, goal_state):
        """
        Use kinodynamic A* to generate dynamically feasible waypoints.

        start_state:
            position, velocity, optionally acceleration

        goal_state:
            target position, optional target velocity

        """
        start_state = np.asarray(start_state, dtype=np.float32).reshape(-1)
        goal_state = np.asarray(goal_state, dtype=np.float32).reshape(-1)
        if start_state.shape[0] < 3 or goal_state.shape[0] < 3:
            raise ValueError("start_state and goal_state must include at least xyz")
        path = self.waypoints_A_star(start_state[:3], goal_state[:3])
        distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
        ts = np.maximum(distances / max(float(self.max_vel), 1e-3), 0.1).astype(np.float32)
        velocities = np.zeros_like(path, dtype=np.float32)
        if start_state.shape[0] >= 6:
            velocities[0] = start_state[3:6]
        if goal_state.shape[0] >= 6:
            velocities[-1] = goal_state[3:6]
        for idx in range(1, len(path) - 1):
            velocities[idx] = (path[idx + 1] - path[idx - 1]) / max(float(ts[idx - 1] + ts[idx]), 1e-3)
        self.waypoints = path
        self.velocities = velocities
        self.ts = ts
        return self.waypoints

    def waypoints_RRT_connect(self, start, goal):
        """
        Use RRT-Connect to find a feasible 3D path.
        """
        self._ensure_esdf()
        start = np.asarray(start, dtype=np.float32).reshape(3)
        goal = np.asarray(goal, dtype=np.float32).reshape(3)
        if not self.is_point_valid(start) or not self.is_point_valid(goal):
            raise ValueError("start and goal must be collision-free")
        if self.is_segment_valid(start, goal):
            self.waypoints = np.asarray([start, goal], dtype=np.float32)
            return self.waypoints

        lower, upper = self.ESDF_map.bounds(start, goal)
        rng = np.random.default_rng(0)
        step = max(0.5, 5.0 * float(self.map_resolution))
        trees = [[start], [goal]]
        parents = [[-1], [-1]]

        for iteration in range(4000):
            active = iteration % 2
            other = 1 - active
            sample_point = goal if iteration % 8 == 0 else rng.uniform(lower, upper).astype(np.float32)
            if not self.is_point_valid(sample_point):
                continue
            new_index = self._rrt_extend(trees[active], parents[active], sample_point, step)
            if new_index is None:
                continue
            connect_index = self._nearest_index(trees[other], trees[active][new_index])
            if self.is_segment_valid(trees[active][new_index], trees[other][connect_index]):
                path_a = self._trace_tree(trees[active], parents[active], new_index)
                path_b = self._trace_tree(trees[other], parents[other], connect_index)
                raw_path = path_a + path_b[::-1] if active == 0 else path_b + path_a[::-1]
                self.waypoints = self.shortcut_waypoints(raw_path)
                return self.waypoints

        raise RuntimeError("RRT-Connect failed to find a valid path")

    def waypoints_flow_matching(self, start, goal, num_candidates=8):
        """
        Use flow matching model to propose candidate waypoint paths.

        Important:
        The flow model should NOT be trusted directly.
        It only proposes waypoint candidates.

        """
        start = np.asarray(start, dtype=np.float32).reshape(3)
        goal = np.asarray(goal, dtype=np.float32).reshape(3)
        sample_count = min(max(int(self.horizon), 2), 32)
        candidates = [np.linspace(start, goal, sample_count, dtype=np.float32)]

        valid_candidates = [path for path in candidates if self.validate_waypoint_path(path)]
        if not valid_candidates:
            try:
                return self.waypoints_A_star(start, goal)
            except Exception:
                return self.waypoints_RRT_connect(start, goal)

        best = min(valid_candidates, key=self.path_cost)
        self.waypoints = self.shortcut_waypoints(best)
        return self.waypoints

    def validate_waypoint_path(self, waypoints):
        """
        Check whether all waypoints and all connecting segments are valid.
        """
        waypoints = self._as_waypoints(waypoints)
        for point in waypoints:
            if not self.is_point_valid(point):
                return False
        for p0, p1 in zip(waypoints[:-1], waypoints[1:]):
            if not self.is_segment_valid(p0, p1):
                return False
        return True

    def path_cost(self, waypoints):
        """
        Score a path.

        Lower is better.

        Suggested cost:
            path length
            + smoothness penalty
            - clearance reward
        """
        waypoints = self._as_waypoints(waypoints)
        segment_vectors = np.diff(waypoints, axis=0)
        path_length = float(np.linalg.norm(segment_vectors, axis=1).sum())
        smoothness = 0.0
        if len(waypoints) > 2:
            smoothness = float(np.linalg.norm(waypoints[2:] - 2.0 * waypoints[1:-1] + waypoints[:-2], axis=1).sum())
        clearances = np.asarray([self.query_esdf(point) for point in waypoints], dtype=np.float32)
        clearance_penalty = float(np.maximum(float(self.safety_radius) - clearances, 0.0).sum())
        clearance_reward = float(np.minimum(clearances, float(self.local_map_radius)).mean())
        return path_length + 0.25 * smoothness + 10.0 * clearance_penalty - 0.05 * clearance_reward

    # ------------------------------------------------------------------
    # Trajectory generators
    # ------------------------------------------------------------------

    def B_spline(self, ts, waypoints, velocities=None):
        """
        Generate a constrained B-spline trajectory.
        """
        waypoints = self._as_waypoints(waypoints)
        ts = self._as_segment_times(ts, len(waypoints) - 1)
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=np.float32)
            if velocities.shape != waypoints.shape:
                raise ValueError("velocities must have shape [M, 3]")
            spline_points = np.concatenate([waypoints, velocities], axis=1)
        else:
            spline_points = waypoints

        timestamps = np.concatenate([[0.0], np.cumsum(ts)]).astype(np.float32)
        sample_dt = max(float(self.map_resolution), 1e-3)
        num_samples = max(2, int(np.ceil(float(timestamps[-1]) / sample_dt)) + 1)
        refined = refine_waypoints(
            spline_points,
            timestamps=timestamps,
            num_samples=num_samples,
            method="cubic",
            max_velocity=self.max_vel,
            max_acceleration=self.max_acc,
            collision_checker=lambda positions: np.asarray(
                [not self.is_point_valid(position) for position in positions],
                dtype=bool,
            ),
        )
        if refined.collision_info is not None and np.any(refined.collision_info):
            raise RuntimeError("generated B-spline trajectory is in collision")
        return SplineTrajectory(refined)

    def _ensure_esdf(self) -> None:
        """Raise if the local ESDF has not been built."""
        if self.ESDF_map is None:
            raise RuntimeError("ESDF map has not been built yet. Call env_ESDF_map() first.")

    def _as_waypoints(self, waypoints) -> np.ndarray:
        """Convert waypoint inputs to a finite [M, 3] float32 array."""
        values = np.asarray(waypoints, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("waypoints must have shape [M, 3]")
        if values.shape[0] < 2:
            raise ValueError("at least two waypoints are required")
        if not np.isfinite(values).all():
            raise ValueError("waypoints must be finite")
        return values

    def _as_segment_times(self, ts, segment_count: int) -> np.ndarray:
        """Convert segment durations to finite positive [M - 1] values."""
        values = np.asarray(ts, dtype=np.float32).reshape(-1)
        if values.shape[0] != segment_count:
            raise ValueError(f"ts must contain {segment_count} segment durations")
        if not np.isfinite(values).all() or np.any(values <= 0.0):
            raise ValueError("ts must contain finite positive durations")
        return values

    def _reconstruct_grid_path(self, came_from, current, to_world, start, goal) -> np.ndarray:
        """Reconstruct a world-coordinate A* path."""
        indices = [current]
        while current in came_from:
            current = came_from[current]
            indices.append(current)
        indices.reverse()
        path = np.asarray([to_world(index) for index in indices], dtype=np.float32)
        path[0] = start
        path[-1] = goal
        return path

    def _nearest_index(self, tree: list[np.ndarray], point: np.ndarray) -> int:
        """Return the nearest node index in an RRT tree."""
        distances = [float(np.linalg.norm(node - point)) for node in tree]
        return int(np.argmin(distances))

    def _rrt_extend(self, tree: list[np.ndarray], parents: list[int], target: np.ndarray, step: float) -> int | None:
        """Extend one RRT tree toward target and return the new node index."""
        nearest_index = self._nearest_index(tree, target)
        nearest = tree[nearest_index]
        delta = target - nearest
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return None
        new_point = target if distance <= step else nearest + (delta / distance) * step
        new_point = np.asarray(new_point, dtype=np.float32)
        if not self.is_point_valid(new_point) or not self.is_segment_valid(nearest, new_point):
            return None
        tree.append(new_point)
        parents.append(nearest_index)
        return len(tree) - 1

    def _trace_tree(self, tree: list[np.ndarray], parents: list[int], index: int) -> list[np.ndarray]:
        """Trace an RRT branch from root to index."""
        path = []
        while index >= 0:
            path.append(tree[index])
            index = parents[index]
        path.reverse()
        return path

    def MINCO_S3NU(self, ts, waypoints, velocities=None) -> MINCOTrajectory:
        """
        Generate MINCO_S3NU trajectory.

        Inputs:
            waypoints: [M, 3], including start and goal
            velocities: optional [M, 3]
            ts: [M - 1] segment durations

        MINCO convention:
            N = number of polynomial segments = len(waypoints) - 1

        headPVA:
            initial position, velocity, acceleration

        tailPVA:
            final position, velocity, acceleration

        pos:
            intermediate waypoint positions, excluding start and goal

        vel:
            intermediate waypoint velocities, only for MINCO_S3NU_PV

        t:
            segment durations
        """
        waypoints_np = self._as_waypoints(waypoints)
        segment_times = self._as_segment_times(ts, len(waypoints_np) - 1)
        waypoints_tensor = torch.as_tensor(waypoints_np, dtype=torch.float32)
        times_tensor = torch.as_tensor(segment_times[:, None], dtype=torch.float32)

        point_count = waypoints_tensor.shape[0]
        segment_count = point_count - 1
        zero_vec = torch.zeros(3, dtype=torch.float32)

        if velocities is not None:
            velocities_np = np.asarray(velocities, dtype=np.float32)
            if velocities_np.shape != waypoints_np.shape:
                raise ValueError("velocities must have the same shape as waypoints: [M, 3]")
            velocities_np = velocities_np.copy()
            velocities_np[0] = 0.0
            velocities_np[-1] = 0.0
            velocities_tensor = torch.as_tensor(velocities_np, dtype=torch.float32)
            head_vel = zero_vec
            tail_vel = zero_vec
            inner_vel = velocities_tensor[1:-1].unsqueeze(0)
            model = traj_fun.MINCO_S3NU_PV(segment_count)

        else:
            head_vel = zero_vec
            tail_vel = zero_vec
            inner_vel = None
            model = traj_fun.MINCO_S3NU(segment_count)

        headPVA = torch.stack(
            [
                waypoints_tensor[0],
                head_vel,
                zero_vec,
            ],
            dim=0,
        ).unsqueeze(0)

        tailPVA = torch.stack(
            [
                waypoints_tensor[-1],
                tail_vel,
                zero_vec,
            ],
            dim=0,
        ).unsqueeze(0)

        # Intermediate positions only.
        pos = waypoints_tensor[1:-1].unsqueeze(0)

        # Segment durations.
        t = times_tensor.unsqueeze(0)

        start = timeit.default_timer()
        with torch.no_grad():
            if velocities is not None:
                model(headPVA, tailPVA, pos, inner_vel, t)
            else:
                model(headPVA, tailPVA, pos, t)
        tim = timeit.default_timer() - start
        print(f"solver time is : {tim}")
        return model.get_trajectory()[0]
        # minco = model.get_trajectory()[0]
        # coeffs = minco.coeffs.detach().cpu().numpy().astype(np.float32)
        # sample_dt = 0.05
        # total_time = float(np.sum(segment_times))
        # sample_times = np.linspace(
        #     0.0,
        #     total_time,
        #     max(2, int(np.ceil(total_time / sample_dt)) + 1),
        #     dtype=np.float32,
        # )
        # trajectory = MINCOTrajectory(
        #     coeffs=coeffs,
        #     durations=segment_times,
        #     refined=RefinedTrajectory(
        #         time=sample_times,
        #         position_refs=np.zeros((sample_times.shape[0], 3), dtype=np.float32),
        #         velocity_refs=np.zeros((sample_times.shape[0], 3), dtype=np.float32),
        #         acceleration_refs=np.zeros((sample_times.shape[0], 3), dtype=np.float32),
        #     ),
        # )
        # position_refs = trajectory.evaluate(sample_times)
        # velocity_refs = trajectory.velocity(sample_times)
        # acceleration_refs = trajectory.acceleration(sample_times)
        # collision_info = None
        # if getattr(self, "ESDF_map", None) is not None:
        #     collision_info = np.asarray(
        #         [not self.is_point_valid(position) for position in position_refs],
        #         dtype=bool,
        #     )
        # trajectory.refined = RefinedTrajectory(
        #     time=sample_times,
        #     position_refs=position_refs,
        #     velocity_refs=velocity_refs,
        #     acceleration_refs=acceleration_refs,
        #     collision_info=collision_info,
        # )
        return trajectory
    
