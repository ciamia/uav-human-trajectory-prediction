from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import pyquaternion
import numpy as np
import torch
import contextlib
import logging
import math
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
from typing import Any, Iterator, Optional, Union

try:
    import genesis as gs
except ImportError:
    gs = None


def _mute_genesis_loggers(genesis_module: ModuleType) -> None:
    """Disable Genesis stream loggers so FPS messages do not spam the terminal."""
    for logger in _iter_genesis_python_loggers(genesis_module):
        logger.disabled = True
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.stream = sys.__stdout__

def _iter_genesis_python_loggers(genesis_module: ModuleType) -> Iterator[logging.Logger]:
    """Yield Python loggers wrapped by Genesis logger objects when present."""
    candidates = [
        getattr(genesis_module, "logger", None),
        getattr(getattr(genesis_module, "logging", None), "logger", None),
    ]
    for candidate in candidates:
        current = candidate
        for _ in range(3):
            if isinstance(current, logging.Logger):
                yield current
                break
            current = getattr(current, "_logger", None)
            if current is None:
                break

def _as_tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert numpy, list, or tensor values to a detached float tensor."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _require_genesis() -> Any:
    """Import Genesis only when a Genesis-backed world is being built."""
    global gs
    if gs is None:
        import genesis as genesis_module

        gs = genesis_module
    return gs


def _genesis_geom() -> Any:
    """Import Genesis geometry helpers lazily."""
    return _require_genesis().utils.geom

def quaternion_to_euler(q):
    q = pyquaternion.Quaternion(w=q[0], x=q[1], y=q[2], z=q[3])
    yaw, pitch, roll = q.yaw_pitch_roll
    return [roll, pitch, yaw]

def yaw_only_quaternion(quat_wxyz: Any) -> np.ndarray:
    """Return a [w, x, y, z] quaternion with current yaw and zero roll/pitch."""
    quat = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1, 4)[0]
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    w, x, y, z = (quat / norm).tolist()
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half_yaw = 0.5 * yaw
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float32)

def transform_by_quat(vector: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions using Genesis when available."""
    return _genesis_geom().transform_by_quat(vector, quat)

def inv_quat(quat: torch.Tensor) -> torch.Tensor:
    """Return inverse quaternions using Genesis when available."""
    return _genesis_geom().inv_quat(quat)

def xyz_to_quat(xyz: torch.Tensor, rpy: bool = False) -> torch.Tensor:
    """Convert xyz/rpy angles to quaternions using Genesis when available."""
    return _genesis_geom().xyz_to_quat(xyz, rpy=rpy)


def quat_omega_dot(quat: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Return quaternion derivative for body angular velocity."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = quat.unbind(dim=-1)
    ox, oy, oz = omega.unbind(dim=-1)
    return 0.5 * torch.stack(
        [
            -x * ox - y * oy - z * oz,
            w * ox + y * oz - z * oy,
            w * oy + z * ox - x * oz,
            w * oz + x * oy - y * ox,
        ],
        dim=-1,
    )


def transform_by_quat_diff(vector: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by quaternions for derivative quantities."""
    return transform_by_quat(vector, quat)

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)
    
def load_yaml_config(value: Any) -> dict[str, Any]:
    """Return a config mapping from either an inline dict or a YAML path."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return load_yaml(value) or {}


LOCAL_GENESIS_ASSETS_DIR = Path("/home/yuxia/lab_ws/Genesis/genesis/assets/urdf")
DRONE_MODEL_FILES = {
    "CF2X": "cf2x.urdf",
    "CF2P": "cf2p.urdf",
    "RACE": "racer.urdf",
}


class ControlMode(Enum):
    """Supported simplified drone control command interfaces."""

    ACC_YAWRATE = "acc_yawrate"
    THRUST_BODYRATE = "thrust_bodyrate"
    MOTOR_THRUSTS = "motor_thrusts"
    FORCE_TORQUE = "force_torque"


def _to_position(value: Any) -> np.ndarray:
    """Convert a 3D position-like value to float32."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    position = np.asarray(value, dtype=np.float32).reshape(-1)
    if position.shape[0] < 3:
        raise ValueError("position must contain at least x, y, z")
    return position[:3].copy()


def _to_vector3(value: Any, name: str) -> np.ndarray:
    """Convert a scalar or 3-vector to a float32 3-vector."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] == 1:
        return np.repeat(array[0], 3).astype(np.float32)
    if array.shape[0] != 3:
        raise ValueError(f"{name} must be a scalar or contain exactly 3 values")
    return array.copy()


def _wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _rotation_matrix_from_euler(attitude: np.ndarray) -> np.ndarray:
    """Return body-to-world rotation for roll, pitch, yaw Euler angles."""
    roll, pitch, yaw = [float(v) for v in attitude]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    rotation_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rotation_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (rotation_z @ rotation_y @ rotation_x).astype(np.float32)


def _quaternion_wxyz_from_euler(attitude: np.ndarray) -> np.ndarray:
    """Return a [w, x, y, z] quaternion from roll, pitch, yaw."""
    roll, pitch, yaw = [float(v) for v in attitude]
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _make_surface(gs: Any, color: Optional[Tuple[float, float, float, float]]) -> Any:
    """Create a simple Genesis visual surface when supported."""
    if gs is None or color is None or not hasattr(gs, "surfaces"):
        return None
    for surface_name in ("Default", "Rough", "Smooth"):
        surface_cls = getattr(gs.surfaces, surface_name, None)
        if surface_cls is None:
            continue
        for color_value in (color, color[:3]):
            try:
                return surface_cls(color=color_value)
            except TypeError:
                pass
    return None


def make_visual_morph(morph_cls: Any, **kwargs: Any) -> Any:
    """Create a Genesis morph, using fixed/collision options only if supported."""
    for extra_kwargs in ({"fixed": True, "collision": False}, {"fixed": True}, {"collision": False}, {}):
        merged = dict(kwargs)
        merged.update(extra_kwargs)
        try:
            return morph_cls(**merged)
        except TypeError:
            continue
        except Exception as exc:
            msg = str(exc)
            if "Unrecognized attribute" in msg or "unexpected keyword" in msg:
                continue
            raise
    return morph_cls(**kwargs)


def _try_morph(morph_cls: Any, kwargs_options: list[dict[str, Any]]) -> Any:
    """Instantiate a Genesis morph with the first compatible kwargs."""
    for kwargs in kwargs_options:
        try:
            return make_visual_morph(morph_cls, **kwargs)
        except Exception:
            continue
    return None


def _add_entity(
    scene: Any,
    morph: Any,
    name: Optional[str] = None,
    gs: Any = None,
    color: Optional[Tuple[float, float, float, float]] = None,
) -> Any:
    """Add a Genesis entity while tolerating visual API differences."""
    surface = _make_surface(gs, color)
    kwargs_options: list[dict[str, Any]] = []
    if name is not None and surface is not None:
        kwargs_options.append({"name": name, "surface": surface})
    if surface is not None:
        kwargs_options.append({"surface": surface})
    if name is not None:
        kwargs_options.append({"name": name})
    kwargs_options.append({})
    for kwargs in kwargs_options:
        try:
            return scene.add_entity(morph, **kwargs)
        except TypeError:
            continue
    return scene.add_entity(morph)


def set_entity_position(entity: Any, position: np.ndarray) -> None:
    """Move a simple Genesis proxy entity if the available API supports it."""
    if entity is None:
        return
    position_list = np.asarray(position, dtype=np.float32).reshape(3).tolist()
    for method_name in ("set_pos", "set_position"):
        if hasattr(entity, method_name):
            getattr(entity, method_name)(position_list)
            return
    if hasattr(entity, "set_qpos"):
        if hasattr(entity, "get_qpos"):
            try:
                qpos = np.asarray(entity.get_qpos(), dtype=np.float32).reshape(-1)
                if qpos.shape[0] >= 3:
                    qpos[:3] = np.asarray(position_list, dtype=np.float32)
                    entity.set_qpos(qpos.tolist())
                    return
            except Exception:
                pass
        for qpos in (position_list + [1.0, 0.0, 0.0, 0.0], position_list):
            try:
                entity.set_qpos(qpos)
                return
            except Exception:
                continue


def set_entity_pose(entity: Any, position: np.ndarray, attitude: np.ndarray) -> None:
    """Move and orient a Genesis entity when pose APIs are available."""
    if entity is None:
        return
    position_list = np.asarray(position, dtype=np.float32).reshape(3).tolist()
    quaternion = _quaternion_wxyz_from_euler(np.asarray(attitude, dtype=np.float32).reshape(3))
    quaternion_list = quaternion.tolist()

    moved = False
    if hasattr(entity, "set_pos"):
        entity.set_pos(position_list)
        moved = True
    elif hasattr(entity, "set_position"):
        entity.set_position(position_list)
        moved = True

    for method_name in ("set_quat", "set_orientation", "set_quaternion"):
        if hasattr(entity, method_name):
            try:
                getattr(entity, method_name)(quaternion_list)
                return
            except Exception:
                break

    if hasattr(entity, "set_qpos"):
        if hasattr(entity, "get_qpos"):
            try:
                qpos = np.asarray(entity.get_qpos(), dtype=np.float32).reshape(-1)
                if qpos.shape[0] >= 7:
                    qpos[:3] = np.asarray(position_list, dtype=np.float32)
                    qpos[3:7] = quaternion
                    entity.set_qpos(qpos.tolist())
                    return
            except Exception:
                pass
        try:
            entity.set_qpos(position_list + quaternion_list)
            return
        except Exception:
            pass
    if not moved:
        set_entity_position(entity, position)


class DroneGenesisConfig:
    def __init__(self, config_path) -> None:

        self.backend = "cpu"
        self.show_viewer = True
        self.dt = 0.01
        self.quiet: bool = True
        self.add_ground: bool = True
        self.mass: float = 1.0
        self.gravity: float = -9.81
        self.start_pos: tuple[float, float, float] = (0.0, 0.0, 1.0)
        self.prefer_genesis_drone: bool = True
        self.genesis_drone_model: str = "CF2X"
        self.model_scale: float = 3.0
        self.model_file: Optional[Union[str, Path]] = None
        self.camera_pos: tuple[float, float, float] = (2.5, -3.0, 2.0)

        load_config = load_yaml(config_path)

        self.num_drones   = int(load_config.get("num_drones", 1))
        self.is_swingload     = bool(load_config.get("is_swingload", False))
        self.which_controller = str(load_config.get("which_controller", "mpc"))
        self.which_task       = str(load_config.get("which_task", "go_to_point"))

        self._get_quad_cfg(load_config.get("drone"))
        self._get_task_cfg(load_config.get("task"))
        self._get_pid_cfg(load_config.get("pid"))
        self._get_mpc_cfg(load_config.get("mpc"))

    def _get_quad_cfg(self, config: Any) -> None:
        """Load drone parameters into an attribute-accessible config."""
        _yaml = load_yaml_config(config)
        _path = config
        quad_cfg = SimpleNamespace()

        quad_cfg.quad_path = _path
        quad_cfg.mass = float(_yaml.get("mass"))
        inertia = np.asarray(_yaml.get("inertia"), dtype=np.float32)
        quad_cfg.inertia = np.diag(inertia) if inertia.size == 3 else inertia.reshape(3, 3)
        quad_cfg.gravity = float(_yaml.get("gravity", 9.81))
        quad_cfg.thrust_min = float(_yaml.get("thrust_min", 0))
        quad_cfg.thrust_max = float(_yaml.get("thrust_max", 8.5))
        quad_cfg.thrust_max_scale = float(_yaml.get("thrust_max_scale", quad_cfg.thrust_max))
        quad_cfg.torque_limit = float(_yaml.get("torque_limit", 0.25))
        quad_cfg.tilt_max = float(_yaml.get("tilt_max", 0.45))
        quad_cfg.thrust_map = np.asarray(_yaml.get("thrust_map", [0.0, 0.0, 0.0]), dtype=np.float32)
        quad_cfg.motor_omega_max = float(_yaml.get("motor_omega_max", 0.0))
        quad_cfg.kappa = float(_yaml.get("kappa", 0.0))

        quad_cfg.rotor_cmd_min = float(_yaml.get("rotor_cmd_min", 0.0))
        quad_cfg.rotor_cmd_max = float(_yaml.get("rotor_cmd_max", 1.0))
        quad_cfg.body_z_acc_min = float(_yaml.get("body_z_acc_min", -5.0))
        quad_cfg.body_z_acc_max = float(_yaml.get("body_z_acc_max",  5.0))
        quad_cfg.ang_acc_min = float(_yaml.get("ang_acc_min", -5.0))
        quad_cfg.ang_acc_max = float(_yaml.get("ang_acc_max",  5.0))
        quad_cfg.velocity_abs_max = float(_yaml.get("velocity_abs_max", 5.0))
        quad_cfg.omega_abs_max = float(_yaml.get("omega_abs_max", 10))


        # load the parameters of the angribird
        # quad = load_quad("angrybird")

        self.quad_cfg = quad_cfg
        # self.quad = quad

    def _get_task_cfg(self, config: Any) -> None:
        """Load task definition parameters into an attribute-accessible config."""
        _yaml = load_yaml_config(config)
        _path = config
        task_cfg = SimpleNamespace()

        task_cfg.quad_path = _path
        task_cfg.task_id = int(_yaml.get("task_id"))
        task_cfg.target_position = np.asarray(_yaml.get("target_position"),  dtype=np.float32).reshape(3)
        task_cfg.target_euler = np.asarray(_yaml.get("target_euler"),  dtype=np.float32).reshape(3)
        task_cfg.position_tolerance = float(_yaml.get("position_tolerance", 0.05))
        task_cfg.velocity_tolerance = float(_yaml.get("velocity_tolerance", 0.05))
        task_cfg.yaw_tolerance = float(_yaml.get("yaw_tolerance", 0.05))
        task_cfg.speed_max     = float(_yaml.get("speed_max", 2.5))

        self.task_cfg = task_cfg

    def _get_mpc_cfg(self, config: Any) -> None:
        """Load MPC parameters from YAML into an attribute-accessible config."""
        _yaml = load_yaml_config(config)
        _path = config
        mpc_cfg = SimpleNamespace()

        mpc_cfg.quad_path = _path

        mpc_cfg.enabled = bool(_yaml.get("enabled", True))
        mpc_cfg.use_solver_controls = bool(_yaml.get("use_solver_controls", False))
        mpc_cfg.dt = float(_yaml.get("dt", 0.05))
        mpc_cfg.nodes = int(_yaml.get("nodes", 10))
        mpc_cfg.duration = float(_yaml.get("duration", mpc_cfg.nodes * mpc_cfg.dt))
        mpc_cfg.episode_steps = max(1, int(round(mpc_cfg.duration / mpc_cfg.dt)))
        mpc_cfg.horizon = int(_yaml.get("horizon", mpc_cfg.episode_steps))
        mpc_cfg.receding = bool(_yaml.get("receding", False))
        mpc_cfg.solver_max_iter = int(_yaml.get("solver_max_iter", 40))
        mpc_cfg.position_tolerance = float (_yaml.get("position_tolerance", 0.05))
        mpc_cfg.velocity_tolerance = float(_yaml.get("velocity_tolerance", 0.15))
        mpc_cfg.angle_tolerance = float(_yaml.get("angle_tolerance", 0.08))
        mpc_cfg.omega_tolerance = float(_yaml.get("omega_tolerance", 0.01))
        mpc_cfg.thrust_max_scale = float(_yaml.get("thrust_max_scale"))

        state_cost = _yaml.get("state_cost", {})
        mpc_cfg.state_cost = SimpleNamespace()
        mpc_cfg.state_cost.position_weight    = np.asarray(state_cost.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.state_cost.velocity_weight    = np.asarray(state_cost.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.state_cost.attitude_weight    = np.asarray(state_cost.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.state_cost.omega_weight         = np.asarray(state_cost.get("omega_gain",    [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.state_cost.control_weight       = float(state_cost.get("control_gain", 0.001))
        terminal_cost = _yaml.get("terminal_cost", {})
        mpc_cfg.terminal_cost = SimpleNamespace()
        mpc_cfg.terminal_cost.position_weight = np.asarray(terminal_cost.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.terminal_cost.velocity_weight = np.asarray(terminal_cost.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.terminal_cost.attitude_weight = np.asarray(terminal_cost.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.terminal_cost.omega_weight      = np.asarray(terminal_cost.get("omega_gain",    [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        mpc_cfg.terminal_cost.control_weight    = float(terminal_cost.get("control_gain", 0.001))
        self.mpc_cfg = mpc_cfg

    def _get_pid_cfg(self, config: Any) -> None:
        """Load PID/PD fallback parameters into an attribute-accessible config."""
        _yaml = load_yaml_config(config)
        _path = config
        pid_cfg = SimpleNamespace()

        pid_cfg.quad_path = _path

        pid_cfg.position_weight = np.asarray(_yaml.get("position_gain", [1.2, 1.2, 8.0]),  dtype=np.float32).reshape(3)
        pid_cfg.velocity_weight = np.asarray(_yaml.get("velocity_gain", [1.8, 1.8, 4.0]),dtype=np.float32).reshape(3)
        pid_cfg.attitude_weight = np.asarray(_yaml.get("attitude_gain", [0.12, 0.12, 0.12]),dtype=np.float32).reshape(3)
        pid_cfg.omega_weight    = np.asarray(_yaml.get("omega_gain",    [0.025, 0.025, 0.025]),dtype=np.float32).reshape(3)
        pid_cfg.control_weight  = float(_yaml.get("control_gain", 0.001))
        pid_cfg.thrust_max_scale = float(_yaml.get("thrust_max_scale", 8.5))
        
        self.pid_cfg = pid_cfg


@dataclass
class DroneSim:
    """Trajectory-following drone proxy with only configuration and helpers.

    The proxy follows waypoint states directly and can add a simple visual to a
    Genesis scene supplied by a world object. It does not initialize Genesis or
    build scenes.
    """

    position: np.ndarray
    velocity: Optional[np.ndarray] = None
    attitude: Optional[np.ndarray] = None
    body_rates: Optional[np.ndarray] = None
    radius: float = 0.22
    body_color: tuple[float, float, float, float] = (0.1, 0.35, 1.0, 1.0)
    prefer_genesis_model: bool = True
    model_file: Optional[Union[str, Path]] = None
    model_scale: float = 1.0
    genesis_drone_model: str = "CF2X"
    propeller_rpm: float = 6500.0
    mass: float = 1.0
    inertia: Union[float, np.ndarray] = 0.02
    arm_length: float = 0.2
    k_torque: float = 0.01
    gravity: float = 9.81
    entity: Any = field(default=None, init=False, repr=False)
    model_kind: str = field(default="none", init=False)

    def __post_init__(self) -> None:
        self.position = _to_position(self.position)
        self.velocity = np.zeros(3, dtype=np.float32) if self.velocity is None else _to_position(self.velocity)
        self.attitude = np.zeros(3, dtype=np.float32) if self.attitude is None else _to_position(self.attitude)
        self.body_rates = np.zeros(3, dtype=np.float32) if self.body_rates is None else _to_position(self.body_rates)
        self.inertia = _to_vector3(self.inertia, "inertia")
        if self.mass <= 0.0:
            raise ValueError("mass must be positive")
        if np.any(np.asarray(self.inertia, dtype=np.float32) <= 0.0):
            raise ValueError("inertia values must be positive")

    def add_to_scene(self, gs: Any, scene: Any, name: str = "serving_drone") -> Any:
        """Add the drone proxy to an existing Genesis scene."""
        if scene is None or gs is None or not hasattr(gs, "morphs"):
            return None
        morph = self._make_drone_morph(gs)
        if morph is None:
            return None
        self.entity = _add_entity(scene, morph, name=name, gs=gs, color=self.body_color)
        return self.entity

    def _make_drone_morph(self, gs: Any) -> Any:
        """Create a Genesis drone morph with a visible sphere fallback."""
        if self.prefer_genesis_model:
            morph = self._make_file_drone_morph(gs)
            if morph is not None:
                return morph
            morph = self._make_builtin_drone_morph(gs)
            if morph is not None:
                return morph
        if hasattr(gs.morphs, "Sphere"):
            self.model_kind = "sphere"
            return make_visual_morph(gs.morphs.Sphere, pos=self.position.tolist(), radius=float(self.radius))
        return None

    def _make_builtin_drone_morph(self, gs: Any) -> Any:
        """Try common non-file-backed Genesis drone morph names."""
        kwargs_options = [
            {"pos": self.position.tolist(), "scale": float(self.model_scale)},
            {"pos": self.position.tolist()},
        ]
        for morph_name in ("Quadrotor", "Quadcopter", "UAV"):
            morph_cls = getattr(gs.morphs, morph_name, None)
            if morph_cls is None:
                continue
            morph = _try_morph(morph_cls, kwargs_options)
            if morph is not None:
                self.model_kind = morph_name
                return morph
        return None

    def _make_file_drone_morph(self, gs: Any) -> Any:
        """Try loading an optional URDF/MJCF drone model through Genesis."""

        # self.drone = self.scene.add_entity(
        #     morph=genesis_module.morphs.Drone(
        #         file="urdf/drones/cf2x.urdf",
        #         pos=(0.0, 0, 0.5),  # Start a bit higher
        #         scale =1.0,
        #         propellers_link_name=("prop0_link", "prop2_link", "prop1_link", "prop3_link"),
        #         propellers_spin=(-1, -1, 1, 1),
        #     ),
        # )

        model_path = str(self._resolve_model_file(gs))
        if not model_path:
            return None
        for loader_name in ("Drone", "URDF", "MJCF"):
            loader_cls = getattr(gs.morphs, loader_name, None)
            if loader_cls is None:
                continue
            kwargs_options: list[dict[str, Any]] = []
            if loader_name == "Drone":
                kwargs_options.extend(
                    [
                        {
                            "file": model_path,
                            "pos": tuple(self.position.tolist()),
                            "scale": float(self.model_scale),
                            "model": self.genesis_drone_model,
                            "propellers_link_name": ("prop0_link", "prop1_link", "prop2_link", "prop3_link"),
                        },
                        {
                            "file": model_path,
                            "pos": tuple(self.position.tolist()),
                            "scale": float(self.model_scale),
                            "model": self.genesis_drone_model,
                        },
                        {"file": model_path, "pos": tuple(self.position.tolist()), "scale": float(self.model_scale)},
                        {"file": model_path, "pos": tuple(self.position.tolist())},
                    ]
                )
            else:
                for file_key in ("file", "filename", "path"):
                    kwargs_options.extend(
                        [
                            {file_key: model_path, "pos": tuple(self.position.tolist()), "scale": float(self.model_scale)},
                            {file_key: model_path, "pos": tuple(self.position.tolist())},
                        ]
                    )
            morph = _try_morph(loader_cls, kwargs_options)
            if morph is not None:
                self.model_kind = loader_name
                return morph
        return None

    def _resolve_model_file(self, gs: Any) -> Union[Path, str]:
        """Resolve the configured or bundled Genesis drone URDF path."""
        if self.model_file is not None:
            return self.model_file

        model_filename = DRONE_MODEL_FILES.get(self.genesis_drone_model, "cf2x.urdf")
        candidates = [
            LOCAL_GENESIS_ASSETS_DIR / "drones" / model_filename,
            self._installed_genesis_assets_dir(gs) / "drones" / model_filename,
        ]
        for candidate in candidates:
            if candidate is not None and Path(candidate).exists():
                return candidate
        return ""

    @staticmethod
    def _installed_genesis_assets_dir(gs: Any) -> Path:
        """Return the installed Genesis package asset directory when discoverable."""
        module_file = getattr(gs, "__file__", None)
        if module_file is None:
            return Path()
        return Path(module_file).resolve().parent / "assets" / "urdf"

    @property
    def state(self) -> np.ndarray:
        """Return the current trajectory state [x, y, z, vx, vy, vz]."""
        return np.concatenate([self.position, np.asarray(self.velocity, dtype=np.float32)]).astype(np.float32)

    def set_state(self, state: Any) -> None:
        """Set position and velocity state without simulating motor dynamics."""
        values = np.asarray(state.detach().cpu().numpy() if hasattr(state, "detach") else state, dtype=np.float32).reshape(-1)
        if values.shape[0] < 3:
            raise ValueError("state must contain at least x, y, z")
        self.position = values[:3].copy()
        if values.shape[0] >= 6:
            self.velocity = values[3:6].copy()
        set_entity_pose(self.entity, self.position, np.asarray(self.attitude, dtype=np.float32))
        self.animate_propellers()

    def step_dynamics(self, acceleration: Any, dt: float, max_velocity: Optional[float] = None) -> None:
        """Advance a simple double-integrator drone state by one control step."""
        accel = _to_position(acceleration)
        dt_value = float(dt)
        velocity = np.asarray(self.velocity, dtype=np.float32) + accel * dt_value
        if max_velocity is not None:
            speed = float(np.linalg.norm(velocity))
            if speed > float(max_velocity):
                velocity = velocity / max(speed, 1e-6) * float(max_velocity)
        self.position = self.position + velocity * dt_value
        self.velocity = velocity.astype(np.float32)
        set_entity_pose(self.entity, self.position, np.asarray(self.attitude, dtype=np.float32))
        self.animate_propellers()

    def apply_control(
        self,
        u: np.ndarray,
        mode: ControlMode,
        dt: float,
        max_velocity: Optional[float] = None,
    ) -> None:
        """Apply a simplified low-level control command to the drone proxy."""
        command = np.asarray(u, dtype=np.float32).reshape(-1)
        control_mode = ControlMode(mode)
        dt_value = float(dt)
        if dt_value <= 0.0:
            raise ValueError("dt must be positive")

        if control_mode == ControlMode.ACC_YAWRATE:
            if command.shape[0] != 4:
                raise ValueError("ACC_YAWRATE command must be [ax, ay, az, yaw_rate]")
            self.body_rates = np.array([0.0, 0.0, command[3]], dtype=np.float32)
            self.attitude[2] = _wrap_angle(float(self.attitude[2]) + float(command[3]) * dt_value)
            self._integrate_translation(command[:3], dt_value, max_velocity)
        elif control_mode == ControlMode.THRUST_BODYRATE:
            if command.shape[0] != 4:
                raise ValueError("THRUST_BODYRATE command must be [T, p, q, r]")
            self.body_rates = command[1:4].astype(np.float32)
            self._integrate_attitude(self.body_rates, dt_value)
            self._integrate_translation(self._world_acceleration_from_thrust(float(command[0])), dt_value, max_velocity)
        elif control_mode == ControlMode.MOTOR_THRUSTS:
            if command.shape[0] != 4:
                raise ValueError("MOTOR_THRUSTS command must be [f1, f2, f3, f4]")
            motor_thrusts = np.maximum(command[:4], 0.0).astype(np.float32)
            angular_acceleration = self._motor_torques(motor_thrusts) / np.asarray(self.inertia, dtype=np.float32)
            self.body_rates = np.asarray(self.body_rates, dtype=np.float32) + angular_acceleration * dt_value
            self._integrate_attitude(self.body_rates, dt_value)
            self._integrate_translation(self._world_acceleration_from_thrust(float(motor_thrusts.sum())), dt_value, max_velocity)
        elif control_mode == ControlMode.FORCE_TORQUE:
            if command.shape[0] != 4:
                raise ValueError("FORCE_TORQUE command must be [F, tau_x, tau_y, tau_z]")
            torque = command[1:4].astype(np.float32)
            angular_acceleration = torque / np.asarray(self.inertia, dtype=np.float32)
            self.body_rates = np.asarray(self.body_rates, dtype=np.float32) + angular_acceleration * dt_value
            self._integrate_attitude(self.body_rates, dt_value)
            self._integrate_translation(self._world_acceleration_from_thrust(max(float(command[0]), 0.0)), dt_value, max_velocity)
        else:
            raise ValueError(f"Unsupported control mode: {mode}")

        set_entity_pose(self.entity, self.position, np.asarray(self.attitude, dtype=np.float32))
        self.animate_propellers()

    def _integrate_translation(self, acceleration: np.ndarray, dt: float, max_velocity: Optional[float]) -> None:
        """Integrate linear velocity and position with optional velocity clipping."""
        velocity = np.asarray(self.velocity, dtype=np.float32) + np.asarray(acceleration, dtype=np.float32).reshape(3) * float(dt)
        if max_velocity is not None:
            speed = float(np.linalg.norm(velocity))
            if speed > float(max_velocity):
                velocity = velocity / max(speed, 1e-6) * float(max_velocity)
        self.position = self.position + velocity * float(dt)
        self.velocity = velocity.astype(np.float32)

    def _integrate_attitude(self, body_rates: np.ndarray, dt: float) -> None:
        """Integrate roll, pitch, yaw from body rates."""
        roll, pitch, yaw = [float(v) for v in np.asarray(self.attitude, dtype=np.float32)]
        p, q, r = [float(v) for v in np.asarray(body_rates, dtype=np.float32)]
        cos_pitch = float(np.cos(pitch))
        if abs(cos_pitch) < 1e-4:
            cos_pitch = 1e-4 if cos_pitch >= 0.0 else -1e-4
        tan_pitch = float(np.sin(pitch) / cos_pitch)

        roll_dot = p + np.sin(roll) * tan_pitch * q + np.cos(roll) * tan_pitch * r
        pitch_dot = np.cos(roll) * q - np.sin(roll) * r
        yaw_dot = (np.sin(roll) / cos_pitch) * q + (np.cos(roll) / cos_pitch) * r
        self.attitude = np.array(
            [
                _wrap_angle(roll + float(roll_dot) * dt),
                float(np.clip(pitch + float(pitch_dot) * dt, -1.45, 1.45)),
                _wrap_angle(yaw + float(yaw_dot) * dt),
            ],
            dtype=np.float32,
        )

    def _world_acceleration_from_thrust(self, collective_thrust: float) -> np.ndarray:
        """Map body-z collective thrust to world acceleration and add gravity."""
        body_acceleration = np.array([0.0, 0.0, float(collective_thrust) / float(self.mass)], dtype=np.float32)
        gravity = np.array([0.0, 0.0, -float(self.gravity)], dtype=np.float32)
        return _rotation_matrix_from_euler(np.asarray(self.attitude, dtype=np.float32)) @ body_acceleration + gravity

    def _motor_torques(self, motor_thrusts: np.ndarray) -> np.ndarray:
        """Compute simple X-quad body torques from four motor thrusts."""
        f1, f2, f3, f4 = [float(v) for v in motor_thrusts]
        arm = float(self.arm_length) / np.sqrt(2.0)
        return np.array(
            [
                arm * (f1 + f4 - f2 - f3),
                arm * (f1 + f2 - f3 - f4),
                float(self.k_torque) * (f1 - f2 + f3 - f4),
            ],
            dtype=np.float32,
        )

    def animate_propellers(self) -> None:
        """Request Genesis DroneEntity propeller animation when available."""
        if self.entity is None:
            return
        if hasattr(self.entity, "set_propellers_rpm"):
            propeller_count = int(getattr(self.entity, "n_propellers", 4))
            with contextlib.suppress(Exception):
                self.entity.set_propellers_rpm(np.full(propeller_count, float(self.propeller_rpm), dtype=np.float32))
        if hasattr(self.entity, "update_propeller_vgeoms"):
            with contextlib.suppress(Exception):
                self.entity.update_propeller_vgeoms()

    def reset(self, position: Any) -> None:
        """Reset the proxy position."""
        self.velocity = np.zeros(3, dtype=np.float32)
        self.body_rates = np.zeros(3, dtype=np.float32)
        self.set_position(position)

    def set_position(self, position: Any) -> None:
        """Move the proxy to a new position without simulating motor dynamics."""
        self.position = _to_position(position)
        set_entity_pose(self.entity, self.position, np.asarray(self.attitude, dtype=np.float32))

@dataclass
class Drone:
    """Genesis-backed drone world.

    State:  [pos(3), quat(4), vel(3), ang_vel(3)]
    Action: [body_z_acc(1), ang_acc_body(3)] 
    Command: [Fz(1), tau_x(1), tau_y(1), tau_z(1)]
    Goal:   [px, py, pz, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    """
    def __init__(self, config: str = None, scene_setup: Any = None, show_viewer: bool | None = None) -> None:
        self.device = torch.device("cpu")
        self.num_drones = 1
        self.linear_velocity_abs_max = 5.0
        self.angular_velocity_abs_max = 10.0
        self.drone_cfg: dict[str, Any] = {}
        self.task_cfg: dict[str, Any] = {"is_swingload": False}
        self.is_swingload = False
        self.num_obs = 0
        self.num_task_states = 12
        self.base_thrust = 14475.8  # Base RPM for constant hover
        self.gravity: float = -9.81
        self.ground_height: float = 0.0
        self.ground_contact_height: float = 0.15
        self.is_landed: bool = False
        self.max_acc = 2
        self.max_thrust = self.max_acc * self.base_thrust

        self.base_ctrlMode = "acc"
        self.controller = SimpleNamespace(
            cost_parameters=SimpleNamespace(state=SimpleNamespace(), terminal=SimpleNamespace())
        )
        self.action_limits = torch.tensor(
            [[-5.0, 5.0], [-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]],
            device=self.device,
            dtype=torch.float32,
        )
        self.rotor_cmd_limits = torch.tensor([[0.0, 1.0]] * 4, device=self.device, dtype=torch.float32)
        
        self.scene: Any = None
        self.ground: Any = None
        self.entity: Any = None
        self.frames: list[Any] = []
        self.gs   = gs
        self.time = 0.0

        if config is not None:
            if isinstance(config, str):
                self.config = DroneGenesisConfig(config)
            else:
                self.config = config
        else:
            self.config = None
        if self.config is not None and show_viewer is not None:
            self.config.show_viewer = bool(show_viewer)
        if self.config is not None:
            self.base_thrust = self._hover_rpm_from_config()
            self.max_thrust = self._max_propeller_rpm()
        

    def build_scene(self, scene_setup: Any = None) -> None:
        """Initialize Genesis, create the drone scene, and build it."""
        gs.init(backend=gs.cpu)
        self.scene = self.gs.Scene(
            sim_options=self.gs.options.SimOptions(
                dt=0.01,
                gravity=(0, 0, -9.81),
            ),
            viewer_options=self.gs.options.ViewerOptions(
                # camera_pos=(0.0, -2.0, 1.0),
                # camera_lookat=(0.0, 0.0, 0.3),
                # camera_fov=45,
                camera_pos=(3.0, -5.0, 3.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=45,
                max_FPS=60,
            ),
            vis_options=self.gs.options.VisOptions(
                show_world_frame=False,
            ),
            show_viewer=True if self.config is None else bool(self.config.show_viewer),
            show_FPS=False,
        )
        self.scene.add_entity(self.gs.morphs.Plane())
        ## for multiple drones
        # drones = []
        # for drone_id in range(self.num_drones):
        #     drones.append(self.scene.add_entity(
        #         morph=gs.morphs.Drone(
        #             file="urdf/drones/cf2x.urdf",
        #             pos=(0.0, 0, 0.5),  # Start a bit higher
        #             scale =1.0,
        #             propellers_link_name=("prop0_link", "prop2_link", "prop1_link", "prop3_link"),
        #             propellers_spin=(-1, -1, 1, 1), 
        #         ),
        #         name=f"drone_{drone_id}",
        #     ))
        self.entity = self.scene.add_entity(
            morph=self.gs.morphs.Drone(
                file="urdf/drones/cf2x.urdf",
                pos=(0.0, 0, 0.5),  # Start a bit higher
                scale =1.0,
                propellers_link_name=("prop0_link", "prop2_link", "prop1_link", "prop3_link"),
                propellers_spin=(-1, -1, 1, 1),
            ),
        )
        if scene_setup is not None:
            scene_setup(self.gs, self.scene)
        # self.scene.viewer.follow_entity(self.entity) # fix the viwer
        self.scene.build()

    def render(self) -> Any:
        """Render the Genesis scene when a render API is available."""
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
        """Save captured frames to a video file when imageio is available."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.frames:
            output.touch()
            return output
        try:
            import imageio.v2 as imageio

            imageio.mimsave(output, self.frames)
        except ImportError:
            output.touch()
        return output

    def close(self) -> None:
        """Release scene references when Genesis exposes a close API."""
        if self.scene is not None:
            if hasattr(self.scene, "destroy"):
                self.scene.destroy()
            elif hasattr(self.scene, "close"):
                self.scene.close()
        self.scene = None

    def thrusts_to_rpms(self, thrusts):
        """Convert normalized motor thrust commands to Genesis propeller RPMs."""
        normalized_thrusts = np.clip(np.asarray(thrusts, dtype=np.float32), 0.0, 1.0)
        thrusts_newton = normalized_thrusts * self._motor_thrust_max()
        rpms = np.sqrt(thrusts_newton / max(self._thrust_coefficient_rpm(), 1e-12))
        rpms =  np.clip(rpms, 0.0, self._max_propeller_rpm()).astype(np.float32)
        self.entity.set_propellers_rpm(rpms)
        return rpms

    def _motor_thrust_max(self) -> float:
        """Return the configured maximum thrust in Newtons for one motor."""
        return float(getattr(self.config.quad_cfg, "thrust_max", 0.148989))

    def _thrust_coefficient_rpm(self) -> float:
        """Return Genesis/URDF KF in N/RPM^2."""
        drone = getattr(self, "drone", None)
        if drone is not None and hasattr(drone, "KF"):
            with contextlib.suppress(Exception):
                return float(drone.KF)
        thrust_map = np.asarray(getattr(self.config.quad_cfg, "thrust_map", [0.0]), dtype=np.float32).reshape(-1)
        if thrust_map.size and float(thrust_map[0]) > 0.0:
            return float(thrust_map[0]) * (2.0 * math.pi / 60.0) ** 2
        return 3.16e-10

    def _max_propeller_rpm(self) -> float:
        """Return the configured maximum propeller speed in RPM."""
        omega_max = float(getattr(self.config.quad_cfg, "motor_omega_max", 0.0))
        if omega_max > 0.0:
            return omega_max * 60.0 / (2.0 * math.pi)
        return math.sqrt(self._motor_thrust_max() / max(self._thrust_coefficient_rpm(), 1e-12))

    def _hover_rpm_from_config(self) -> float:
        """Return the per-motor hover RPM implied by the configured drone parameters."""
        hover_thrust = float(self.config.quad_cfg.mass) * float(self.config.quad_cfg.gravity) / 4.0
        return math.sqrt(max(hover_thrust, 0.0) / max(self._thrust_coefficient_rpm(), 1e-12))
    
    def _hover_mass(self) -> float:
        """Return the built entity mass when available, otherwise the configured mass."""
        if self.entity is not None and hasattr(self.entity, "get_mass"):
            with contextlib.suppress(Exception):
                mass = np.asarray(self.entity.get_mass(), dtype=np.float32).reshape(-1)[0]
                if np.isfinite(mass) and float(mass) > 0.0:
                    return float(mass)
        return float(self.config.mass)
    
    def hover_command(self) -> np.ndarray:
        # """Return a level-hover [Fz, tau_x, tau_y, tau_z] command."""
        # return np.array([-self._hover_mass() * self.gravity, 0.0, 0.0, 0.0], dtype=np.float32)
        """Return a level-hover rmps command."""
        return np.array([self.base_thrust, self.base_thrust, self.base_thrust, self.base_thrust], dtype=np.float32)

    def set_hover_pose(self) -> None:
        """Keep current position/yaw, set roll and pitch to zero, and clear velocity."""
        position = self.entity.get_pos()
        quat = self.entity.get_quat()
        hover_quat = yaw_only_quaternion(quat)
        if hasattr(self.entity, "set_qpos"):
            qpos = np.concatenate([position, hover_quat]).astype(np.float32)
            with contextlib.suppress(TypeError):
                self.entity.set_qpos(qpos, zero_velocity=True)
                return
            self.entity.set_qpos(qpos)
            self.entity.set_dofs_velocity(None)
            return
        self.entity.set_dofs_velocity(None)

    def zero_velocity(self) -> None:
        """Clear drone linear and angular velocity when the API supports it."""
        if hasattr(self.entity, "zero_velocity"):
            with contextlib.suppress(Exception):
                self.entity.zero_velocity()
                return
        if hasattr(self.entity, "set_dofs_velocity"):
            with contextlib.suppress(Exception):
                self.entity.set_dofs_velocity(None)

    def settle_on_ground_if_landed(self, ground_height: float | None = None) -> bool:
        """Stop the drone and keep it settled after it reaches the ground."""
        ground_z = self.ground_height if ground_height is None else float(ground_height)
        position = self.get_position()
        if self.is_landed or float(position[2]) <= ground_z + self.ground_contact_height:
            self.is_landed = True
            self.zero_velocity()
            if hasattr(self.entity, "set_propellers_rpm"):
                with contextlib.suppress(Exception):
                    self.entity.set_propellers_rpm(np.zeros(4, dtype=np.float32))
            if float(position[2]) < ground_z and hasattr(self.entity, "set_pos"):
                position[2] = ground_z
                with contextlib.suppress(TypeError):
                    self.entity.set_pos(position, zero_velocity=True)
                    return True
                with contextlib.suppress(Exception):
                    self.entity.set_pos(position)
            return True
        return False

    def get_hover_pose(self) -> np.ndarray:
        """Get current position/yaw, set roll and pitch to zero, and clear velocity."""
        position = self.entity.get_pos().numpy()
        quat = self.entity.get_quat().numpy()
        hover_quat = yaw_only_quaternion(quat)
        return np.hstack([position, hover_quat, np.zeros(3), np.zeros(3)]) 
    
    def return_to_hover(
        self,
        reset_velocity: bool = True,
        reset_angular_velocity: bool = True,
        reset_pose: bool = True,
    ) -> np.ndarray:
        """Level the drone, clear residual motion, and return hover command."""
        if reset_pose:
            self.set_hover_pose()
        elif reset_velocity or reset_angular_velocity:
            self.zero_velocity()
        return self.hover_command()
    
    def get_state(self):
        pos = self.entity.get_pos()
        quat = self.entity.get_quat()
        vel = self.entity.get_vel()
        omega_world = self.entity.get_ang()
        omega_body = transform_by_quat(omega_world, inv_quat(quat))

        state_parts = np.hstack([pos, quat, vel, omega_body])
        return state_parts
        if self.is_swingload:
            theta = self.entity.get_dofs_position(dofs_idx_local=self._swingload_dofs_idx_local)
            theta_dot = self.entity.get_dofs_velocity(dofs_idx_local=self._swingload_dofs_idx_local)
            state_parts.extend([theta, theta_dot])

        return torch.cat(state_parts, dim=-1)
    
    def get_state_derivative(self):
        v_WB_W = self.entity.get_vel()
        q_WB = self.entity.get_quat()        
        q_BW = inv_quat(q_WB)
        omega_W = self.entity.get_ang()
        omega_B = transform_by_quat(omega_W, q_BW) 
        q_WB_dot = quat_omega_dot(q_WB, omega_B)

        a_WB_W_res = self.entity.get_links_acc(links_idx_local = self.entity.base_link.idx_local).squeeze(1)
        alpha_WB_W_res = self.entity.get_links_acc_ang(links_idx_local = self.entity.base_link.idx_local).squeeze(1)
        alpha_WB_B_res = transform_by_quat_diff(alpha_WB_W_res, q_BW)
        
        state_derivative_parts = [v_WB_W, q_WB_dot, a_WB_W_res, alpha_WB_B_res]
        if self.is_swingload:
            theta_dot = self.entity.get_dofs_velocity(dofs_idx_local=self._swingload_dofs_idx_local)
            theta_ddot = torch.zeros_like(theta_dot)
            state_derivative_parts.extend([theta_dot, theta_ddot])

        return torch.cat(state_derivative_parts, dim=-1)
    
    def get_position(self) -> np.ndarray:
        """Return the current drone position as a numpy vector."""
        if hasattr(self.entity, "get_pos"):
            return _as_tensor(self.entity.get_pos()).reshape(-1, 3)[0].cpu().numpy()
        return np.asarray(self.config.start_pos, dtype=np.float32)
    
    def get_quat(self) -> np.ndarray:
        """Return the current drone orientation as a [w, x, y, z] numpy quaternion."""
        if hasattr(self.entity, "get_quat"):
            return _as_tensor(self.entity.get_quat()).reshape(-1, 4)[0].cpu().numpy()
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    def get_euler(self) -> np.ndarray:
        """Return the current drone orientation as a [roll, pitch, yaw] numpy vector."""
        if hasattr(self.entity, "get_euler"):
            return _as_tensor(self.entity.get_euler()).reshape(-1, 3)[0].cpu().numpy()
        else:
            eulers = quaternion_to_euler(self.entity.get_quat())
            return _as_tensor(eulers).reshape(-1, 3)[0].cpu().numpy()
        return np.zeros(3, dtype=np.float32)

    def get_velocity(self) -> np.ndarray:
        """Return the current drone linear velocity as a numpy vector."""
        if hasattr(self.entity, "get_vel"):
            return _as_tensor(self.entity.get_vel()).reshape(-1, 3)[0].cpu().numpy()
        return np.zeros(3, dtype=np.float32)

    def get_angular_velocity(self) -> np.ndarray:
        """Return the current drone angular velocity as a numpy vector."""
        if hasattr(self.entity, "get_ang"):
            return _as_tensor(self.entity.get_ang()).reshape(-1, 3)[0].cpu().numpy()
        return np.zeros(3, dtype=np.float32)

    def set_state(self, x: torch.Tensor, env_ids):
        pos = x[:, self.state_idx.pos]
        quat = x[:, self.state_idx.att]
        quat = quat / torch.norm(quat, dim=1, keepdim=True).clamp(min=1e-8)
        vel = x[:, self.state_idx.vel]
        omega_body = x[:, self.state_idx.ang_vel]
        omega_world = transform_by_quat(omega_body, quat)

        self.entity.set_pos(pos, envs_idx=env_ids)
        self.entity.set_quat(quat, envs_idx=env_ids)

        if self._root_dof_idx_local is not None:
            root_vel = torch.cat([vel, omega_world], dim=-1)
            self.entity.set_dofs_velocity(root_vel, dofs_idx_local=self._root_dof_idx_local, envs_idx=env_ids)

        if self.is_swingload:
            self.entity.set_dofs_position(
                x[:, self.state_idx.theta],
                dofs_idx_local=self._swingload_dofs_idx_local,
                envs_idx=env_ids,
            )
            self.entity.set_dofs_velocity(
                x[:, self.state_idx.theta_dot],
                dofs_idx_local=self._swingload_dofs_idx_local,
                envs_idx=env_ids,
            )


if __name__ == "__main__":
    world = Drone()
    world.build_scene(scene_setup=None)
    while True:
        world.scene.step()
