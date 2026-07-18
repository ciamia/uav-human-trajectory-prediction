from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from env.drone import DroneSim
from env.sim_backend import SimBackend


class GenesisSimUnavailableError(RuntimeError):
    pass


def _import_genesis() -> Any:
    try:
        import genesis as gs
    except ImportError as exc:
        raise GenesisSimUnavailableError(
            "genesis-world is required for GenesisSimBackend. Install it with `pip install genesis-world`."
        ) from exc
    return gs


@dataclass(frozen=True)
class GenesisSimConfig:
    backend: str = "cpu"
    dt: float = 0.02
    show_viewer: bool = False
    add_ground: bool = True
    drone_radius: float = 0.12
    auto_build: bool = True
    launch_scene: bool = True


@dataclass(frozen=True)
class ParsedObstacle:
    kind: str
    center: np.ndarray
    size: np.ndarray


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _position_view(trajectory: Any) -> np.ndarray:
    values = _to_numpy(trajectory)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("trajectory must have shape [T, 3] or [T, D] with D >= 3")
    return values[:, :3]


def parse_obstacles(obstacles: Any) -> list[ParsedObstacle]:
    if obstacles is None:
        return []
    if isinstance(obstacles, dict):
        obstacles = [obstacles]

    parsed: list[ParsedObstacle] = []
    if isinstance(obstacles, (list, tuple)) and obstacles and isinstance(obstacles[0], dict):
        for obstacle in obstacles:
            kind = obstacle.get("type", obstacle.get("kind", "sphere"))
            center = _to_numpy(obstacle["center"]).reshape(3)
            if kind == "sphere":
                radius = float(obstacle.get("radius", obstacle.get("r")))
                parsed.append(ParsedObstacle("sphere", center, np.array([radius], dtype=np.float32)))
            elif kind == "box":
                size = _to_numpy(obstacle.get("size", obstacle.get("half_extents"))).reshape(3)
                parsed.append(ParsedObstacle("box", center, size.astype(np.float32)))
            else:
                raise ValueError(f"Unsupported obstacle type: {kind}")
        return parsed

    values = _to_numpy(obstacles)
    if values.size == 0:
        return []
    if values.ndim != 2 or values.shape[1] not in (4, 6):
        raise ValueError("array obstacles must have shape [N, 4] for spheres or [N, 6] for boxes")
    for row in values:
        center = row[:3].astype(np.float32)
        if values.shape[1] == 4:
            parsed.append(ParsedObstacle("sphere", center, np.array([float(row[3])], dtype=np.float32)))
        else:
            parsed.append(ParsedObstacle("box", center, row[3:6].astype(np.float32)))
    return parsed


def _sphere_collision(positions: np.ndarray, obstacle: ParsedObstacle, radius: float, margin: float) -> np.ndarray:
    distances = np.linalg.norm(positions - obstacle.center[None, :], axis=-1)
    return distances <= float(obstacle.size[0]) + radius + margin


def _box_collision(positions: np.ndarray, obstacle: ParsedObstacle, radius: float, margin: float) -> np.ndarray:
    half_size = obstacle.size * 0.5
    delta = np.abs(positions - obstacle.center[None, :]) - half_size[None, :]
    outside = np.maximum(delta, 0.0)
    outside_distance = np.linalg.norm(outside, axis=-1)
    inside_distance = np.minimum(np.maximum.reduce(delta, axis=-1), 0.0)
    signed_distance = outside_distance + inside_distance
    return signed_distance <= radius + margin


class GenesisSimBackend(SimBackend):
    """Genesis simulator backend with a simple Genesis drone proxy.

    This backend does not require ROS2 or Gazebo.
    """

    def __init__(self, config: GenesisSimConfig | None = None) -> None:
        self.config = config or GenesisSimConfig()
        self.gs: Any = None
        self.scene: Any = None
        self.ground_entity: Any = None
        self.drone_entity: Any = None
        self.drone_proxy: DroneSim | None = None
        self.obstacle_entities: list[Any] = []
        self.start: np.ndarray | None = None
        self.goal: np.ndarray | None = None
        self.obstacles: list[ParsedObstacle] = []
        self.last_trajectory: np.ndarray | None = None
        self.last_collision_mask: np.ndarray | None = None
        self.frames: list[Any] = []

    @property
    def available(self) -> bool:
        try:
            _import_genesis()
        except GenesisSimUnavailableError:
            return False
        return True

    def initialize(self) -> None:
        if not self.config.launch_scene:
            return
        self.gs = _import_genesis()
        backend = getattr(self.gs, self.config.backend, None)
        kwargs = {"backend": backend} if backend is not None else {}
        try:
            self.gs.init(**kwargs)
        except Exception:
            pass

    def _new_scene(self) -> Any:
        self.initialize()
        if self.gs is None:
            return None
        try:
            return self.gs.Scene(
                show_viewer=self.config.show_viewer,
                sim_options=self.gs.options.SimOptions(dt=self.config.dt),
            )
        except TypeError:
            return self.gs.Scene(show_viewer=self.config.show_viewer)

    def _add_ground(self) -> None:
        if self.scene is None or self.gs is None or not self.config.add_ground:
            return
        if hasattr(self.gs, "morphs") and hasattr(self.gs.morphs, "Plane"):
            self.ground_entity = self.scene.add_entity(self.gs.morphs.Plane())

    def _add_sphere(self, center: np.ndarray, radius: float, name: str | None = None) -> Any:
        if self.scene is None or self.gs is None or not hasattr(self.gs.morphs, "Sphere"):
            return None
        morph = self.gs.morphs.Sphere(pos=center.tolist(), radius=float(radius))
        try:
            return self.scene.add_entity(morph, name=name)
        except TypeError:
            return self.scene.add_entity(morph)

    def _add_box(self, center: np.ndarray, size: np.ndarray, name: str | None = None) -> Any:
        if self.scene is None or self.gs is None or not hasattr(self.gs.morphs, "Box"):
            return None
        morph = self.gs.morphs.Box(pos=center.tolist(), size=size.tolist())
        try:
            return self.scene.add_entity(morph, name=name)
        except TypeError:
            return self.scene.add_entity(morph)

    def _set_drone_position(self, position: np.ndarray) -> None:
        if self.drone_proxy is not None:
            self.drone_proxy.set_position(position)
            return
        if self.drone_entity is None:
            return
        for method_name in ("set_pos", "set_position"):
            if hasattr(self.drone_entity, method_name):
                getattr(self.drone_entity, method_name)(position.tolist())
                return
        if hasattr(self.drone_entity, "set_qpos"):
            self.drone_entity.set_qpos(position.tolist())

    def reset(self, start: Any, goal: Any, obstacles: Any) -> dict[str, Any]:
        self.start = _to_numpy(start).reshape(-1)
        self.goal = _to_numpy(goal).reshape(-1)
        self.obstacles = parse_obstacles(obstacles)
        self.last_trajectory = None
        self.last_collision_mask = None
        self.frames = []

        self.scene = self._new_scene()
        self.obstacle_entities = []
        self.ground_entity = None
        self.drone_entity = None
        self.drone_proxy = None
        if self.scene is not None:
            self._add_ground()
            for idx, obstacle in enumerate(self.obstacles):
                if obstacle.kind == "sphere":
                    entity = self._add_sphere(obstacle.center, float(obstacle.size[0]), name=f"obstacle_sphere_{idx}")
                else:
                    entity = self._add_box(obstacle.center, obstacle.size, name=f"obstacle_box_{idx}")
                self.obstacle_entities.append(entity)
            self.drone_proxy = DroneSim(position=self.start[:3], radius=self.config.drone_radius)
            self.drone_entity = self.drone_proxy.add_to_scene(self.gs, self.scene, name="drone_proxy")
            if self.config.auto_build and hasattr(self.scene, "build"):
                self.scene.build()
        return self.get_observation()

    def rollout_trajectory(self, trajectory: Any) -> dict[str, Any]:
        positions = _position_view(trajectory)
        self.last_trajectory = _to_numpy(trajectory)
        collisions = self.check_collisions(trajectory)
        self.last_collision_mask = collisions

        for position in positions:
            self._set_drone_position(position)
            if self.scene is not None and hasattr(self.scene, "step"):
                self.scene.step()
            frame = self.render()
            if frame is not None:
                self.frames.append(frame)
        return {
            "trajectory": self.last_trajectory,
            "collision_mask": collisions,
            "collision": bool(collisions.any()),
            "num_steps": int(len(positions)),
        }

    def check_collisions(self, trajectory: Any) -> np.ndarray:
        positions = _position_view(trajectory)
        mask = np.zeros(len(positions), dtype=bool)
        for obstacle in self.obstacles:
            if obstacle.kind == "sphere":
                mask |= _sphere_collision(positions, obstacle, self.config.drone_radius, margin=0.0)
            elif obstacle.kind == "box":
                mask |= _box_collision(positions, obstacle, self.config.drone_radius, margin=0.0)
            else:
                raise ValueError(f"Unsupported obstacle type: {obstacle.kind}")
        return mask

    def get_observation(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "goal": self.goal,
            "obstacles": self.obstacles,
            "last_trajectory": self.last_trajectory,
            "last_collision_mask": self.last_collision_mask,
        }

    def render(self) -> Any:
        if self.scene is None:
            return None
        if hasattr(self.scene, "render"):
            return self.scene.render()
        if hasattr(self.scene, "visualizer") and hasattr(self.scene.visualizer, "render"):
            return self.scene.visualizer.render()
        if hasattr(self.scene, "viewer") and hasattr(self.scene.viewer, "render"):
            return self.scene.viewer.render()
        return None

    def save_video(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            output.touch()
            return output
        try:
            import imageio.v2 as imageio

            imageio.mimsave(output, self.frames)
        except ImportError as exc:
            raise RuntimeError("Saving video requires imageio.") from exc
        return output

    def close(self) -> None:
        if self.scene is not None:
            if hasattr(self.scene, "destroy"):
                self.scene.destroy()
            elif hasattr(self.scene, "close"):
                self.scene.close()
        self.scene = None
        self.ground_entity = None
        self.drone_entity = None
        self.drone_proxy = None
        self.obstacle_entities = []
