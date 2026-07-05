from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Optional, Tuple, Union

import numpy as np

from env.drone import ControlMode

try:
    import casadi as ca
except ImportError:  # pragma: no cover - exercised only when CasADi is absent.
    ca = None


@dataclass(frozen=True)
class QuadrotorMPCConfig:
    """Configuration for a CasADi trajectory MPC over [p, q, v, w] states."""

    dt: float = 0.05
    horizon: int = 12
    receding: bool = True
    mass: float = 1.0
    inertia: Union[float, Tuple[float, float, float]] = 0.02
    arm_length: float = 0.2
    k_torque: float = 0.01
    gravity: float = 9.81
    position_weight: float = 20.0
    velocity_weight: float = 4.0
    attitude_weight: float = 3.0
    angular_velocity_weight: float = 0.5
    control_weight: float = 0.02
    terminal_weight: float = 8.0
    max_velocity: float = 5.0
    max_acceleration: float = 8.0
    yaw_rate_limit: float = 3.0
    thrust_min: float = 0.0
    thrust_max: float = 25.0
    body_rate_limit: float = 6.0
    torque_limit: float = 1.0
    motor_thrust_min: float = 0.0
    motor_thrust_max: float = 8.0
    solver_max_iter: int = 80
    verbose: bool = False


@dataclass(frozen=True)
class QuadrotorMPCResult:
    """State, control, reference, and tracking-error rollout from quadrotor MPC."""

    states: np.ndarray
    controls: np.ndarray
    references: np.ndarray
    tracking_error: np.ndarray


def _require_casadi() -> Any:
    """Return CasADi or raise a clear dependency error."""
    if ca is None:
        raise ImportError("QuadrotorMPCController requires CasADi. Install it with `pip install casadi`.")
    return ca


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or arrays to float32 numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _as_vec3(value: Union[float, Tuple[float, float, float]]) -> np.ndarray:
    """Convert a scalar or 3-tuple to a positive float32 3-vector."""
    values = np.asarray(value, dtype=np.float32).reshape(-1)
    if values.shape[0] == 1:
        values = np.repeat(values[0], 3)
    if values.shape[0] != 3:
        raise ValueError("inertia must be a scalar or 3-vector")
    if np.any(values <= 0.0):
        raise ValueError("inertia values must be positive")
    return values.astype(np.float32)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize a [w, x, y, z] quaternion."""
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _normalize_references(reference: Any) -> np.ndarray:
    """Validate and normalize [T, 13] references."""
    values = _to_numpy(reference)
    if values.ndim != 2 or values.shape[1] != 13:
        raise ValueError("reference must have shape [T, 13] for [p, q, v, w]")
    if values.shape[0] < 2:
        raise ValueError("reference must contain at least two states")
    values = values.copy()
    for idx in range(len(values)):
        values[idx, 3:7] = _normalize_quaternion(values[idx, 3:7])
    return values


def _clip_norm(value: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip vector norm without changing direction."""
    norm = float(np.linalg.norm(value))
    if norm <= float(max_norm):
        return value.astype(np.float32)
    return (value / max(norm, 1e-8) * float(max_norm)).astype(np.float32)


def _quat_multiply_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _rotation_matrix_np(q: np.ndarray) -> np.ndarray:
    """Return body-to-world rotation matrix from a [w, x, y, z] quaternion."""
    w, x, y, z = [float(v) for v in _normalize_quaternion(q)]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _quat_multiply_sym(q1: Any, q2: Any) -> Any:
    """CasADi quaternion product for [w, x, y, z]."""
    c = _require_casadi()
    return c.vertcat(
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0],
    )


def _normalize_quaternion_sym(q: Any) -> Any:
    """CasADi quaternion normalization."""
    c = _require_casadi()
    return q / c.sqrt(c.sumsqr(q) + 1e-9)


def _rotation_matrix_sym(q: Any) -> Any:
    """CasADi body-to-world rotation matrix from [w, x, y, z]."""
    c = _require_casadi()
    q = _normalize_quaternion_sym(q)
    w, x, y, z = q[0], q[1], q[2], q[3]
    return c.vertcat(
        c.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        c.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        c.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


class QuadrotorMPCController:
    """CasADi MPC for quadrotor references in [position, quaternion, velocity, angular velocity]."""

    def __init__(self, config: Optional[QuadrotorMPCConfig] = None) -> None:
        self.config = config or QuadrotorMPCConfig()
        if self.config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.config.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.config.mass <= 0.0:
            raise ValueError("mass must be positive")
        self.inertia = _as_vec3(self.config.inertia)

    def solve(
        self,
        reference: Any,
        initial_state: Optional[Any] = None,
        mode: ControlMode = ControlMode.ACC_YAWRATE,
    ) -> QuadrotorMPCResult:
        """Solve MPC against a [T, 13] reference and return state/control sequences."""
        _require_casadi()
        references = _normalize_references(reference)
        state = references[0].copy() if initial_state is None else _to_numpy(initial_state).reshape(-1)[:13].astype(np.float32)
        if state.shape[0] != 13:
            raise ValueError("initial_state must have shape [13]")
        state[3:7] = _normalize_quaternion(state[3:7])
        control_mode = ControlMode(mode)

        if self.config.receding and len(references) - 1 > self.config.horizon:
            states = np.zeros_like(references)
            controls = np.zeros((len(references) - 1, 4), dtype=np.float32)
            states[0] = state
            for idx in range(len(references) - 1):
                window = self._reference_window(references, idx)
                result = self._solve_window(states[idx], window, control_mode)
                controls[idx] = result.controls[0]
                states[idx + 1] = self._step_numpy(states[idx], controls[idx], control_mode)
            tracking_error = np.linalg.norm(states[:, :3] - references[:, :3], axis=-1).astype(np.float32)
            return QuadrotorMPCResult(states=states, controls=controls, references=references, tracking_error=tracking_error)

        window = references[: min(len(references), self.config.horizon + 1)]
        return self._solve_window(state, window, control_mode)

    def track(
        self,
        reference: Any,
        initial_state: Optional[Any] = None,
        mode: ControlMode = ControlMode.ACC_YAWRATE,
    ) -> QuadrotorMPCResult:
        """Alias for :meth:`solve` to mirror other project controllers."""
        return self.solve(reference, initial_state=initial_state, mode=mode)

    def _reference_window(self, references: np.ndarray, start_idx: int) -> np.ndarray:
        count = min(self.config.horizon + 1, len(references) - start_idx)
        return references[start_idx : start_idx + count]

    def _solve_window(self, initial_state: np.ndarray, references: np.ndarray, mode: ControlMode) -> QuadrotorMPCResult:
        """Build and solve one CasADi optimal-control window."""
        c = _require_casadi()
        steps = len(references) - 1
        opti = c.Opti()
        states = opti.variable(13, steps + 1)
        controls = opti.variable(4, steps)

        opti.subject_to(states[:, 0] == initial_state)
        objective = 0
        for idx in range(steps):
            next_state = self._dynamics_sym(states[:, idx], controls[:, idx], mode)
            opti.subject_to(states[:, idx + 1] == next_state)
            self._add_control_constraints(opti, controls[:, idx], mode)
            if self.config.max_velocity is not None:
                opti.subject_to(c.sumsqr(states[7:10, idx]) <= float(self.config.max_velocity) ** 2)
            objective += self._stage_cost(states[:, idx], controls[:, idx], references[idx], terminal=False, mode=mode)

        objective += self._stage_cost(states[:, steps], None, references[steps], terminal=True, mode=mode)
        opti.minimize(objective)
        self._set_initial_guess(opti, states, controls, references, mode)

        solver_options = {
            "print_time": bool(self.config.verbose),
            "ipopt": {
                "print_level": 5 if self.config.verbose else 0,
                "max_iter": int(self.config.solver_max_iter),
                "sb": "yes",
            },
        }
        opti.solver("ipopt", solver_options)
        solution = opti.solve()

        state_values = np.asarray(solution.value(states), dtype=np.float32).T
        control_values = np.asarray(solution.value(controls), dtype=np.float32).T
        for idx in range(len(state_values)):
            state_values[idx, 3:7] = _normalize_quaternion(state_values[idx, 3:7])
        tracking_error = np.linalg.norm(state_values[:, :3] - references[:, :3], axis=-1).astype(np.float32)
        return QuadrotorMPCResult(
            states=state_values,
            controls=control_values,
            references=references.astype(np.float32),
            tracking_error=tracking_error,
        )

    def _dynamics_sym(self, state: Any, control: Any, mode: ControlMode) -> Any:
        """CasADi discrete dynamics for one MPC step."""
        c = _require_casadi()
        dt = float(self.config.dt)
        position = state[0:3]
        quat = _normalize_quaternion_sym(state[3:7])
        velocity = state[7:10]
        omega = state[10:13]

        if mode == ControlMode.ACC_YAWRATE:
            acceleration = control[0:3]
            next_omega = c.vertcat(0.0, 0.0, control[3])
            quat_dot = 0.5 * _quat_multiply_sym(quat, c.vertcat(0.0, next_omega))
        elif mode == ControlMode.THRUST_BODYRATE:
            thrust = control[0]
            next_omega = control[1:4]
            acceleration = _rotation_matrix_sym(quat) @ c.vertcat(0.0, 0.0, thrust / float(self.config.mass))
            acceleration += c.vertcat(0.0, 0.0, -float(self.config.gravity))
            quat_dot = 0.5 * _quat_multiply_sym(quat, c.vertcat(0.0, next_omega))
        elif mode == ControlMode.MOTOR_THRUSTS:
            motor_thrusts = control[0:4]
            thrust = c.sum1(motor_thrusts)
            torque = self._motor_torques_sym(motor_thrusts)
            angular_acceleration = c.vertcat(
                torque[0] / float(self.inertia[0]),
                torque[1] / float(self.inertia[1]),
                torque[2] / float(self.inertia[2]),
            )
            next_omega = omega + angular_acceleration * dt
            acceleration = _rotation_matrix_sym(quat) @ c.vertcat(0.0, 0.0, thrust / float(self.config.mass))
            acceleration += c.vertcat(0.0, 0.0, -float(self.config.gravity))
            quat_dot = 0.5 * _quat_multiply_sym(quat, c.vertcat(0.0, next_omega))
        elif mode == ControlMode.FORCE_TORQUE:
            thrust = control[0]
            torque = control[1:4]
            angular_acceleration = c.vertcat(
                torque[0] / float(self.inertia[0]),
                torque[1] / float(self.inertia[1]),
                torque[2] / float(self.inertia[2]),
            )
            next_omega = omega + angular_acceleration * dt
            acceleration = _rotation_matrix_sym(quat) @ c.vertcat(0.0, 0.0, thrust / float(self.config.mass))
            acceleration += c.vertcat(0.0, 0.0, -float(self.config.gravity))
            quat_dot = 0.5 * _quat_multiply_sym(quat, c.vertcat(0.0, next_omega))
        else:
            raise ValueError(f"Unsupported control mode: {mode}")

        next_position = position + velocity * dt + 0.5 * acceleration * dt * dt
        next_velocity = velocity + acceleration * dt
        next_quat = _normalize_quaternion_sym(quat + quat_dot * dt)
        return c.vertcat(next_position, next_quat, next_velocity, next_omega)

    def _stage_cost(
        self,
        state: Any,
        control: Optional[Any],
        reference: np.ndarray,
        terminal: bool,
        mode: ControlMode,
    ) -> Any:
        """Quadratic tracking and control-effort objective."""
        c = _require_casadi()
        scale = float(self.config.terminal_weight) if terminal else 1.0
        ref_quat = _normalize_quaternion(np.asarray(reference[3:7], dtype=np.float32))
        ref = c.DM(reference)
        quat = _normalize_quaternion_sym(state[3:7])
        quat_alignment = c.dot(quat, c.DM(ref_quat))
        cost = 0
        cost += scale * float(self.config.position_weight) * c.sumsqr(state[0:3] - ref[0:3])
        cost += scale * float(self.config.velocity_weight) * c.sumsqr(state[7:10] - ref[7:10])
        cost += scale * float(self.config.attitude_weight) * (1.0 - quat_alignment * quat_alignment)
        cost += scale * float(self.config.angular_velocity_weight) * c.sumsqr(state[10:13] - ref[10:13])
        if control is not None:
            cost += float(self.config.control_weight) * c.sumsqr(control - self._hover_control_sym(mode))
        return cost

    def _hover_control_sym(self, mode: ControlMode) -> Any:
        """Return the neutral hover command for a control mode."""
        c = _require_casadi()
        hover_thrust = float(self.config.mass) * float(self.config.gravity)
        if mode == ControlMode.ACC_YAWRATE:
            return c.DM.zeros(4)
        if mode == ControlMode.THRUST_BODYRATE:
            return c.DM([hover_thrust, 0.0, 0.0, 0.0])
        if mode == ControlMode.MOTOR_THRUSTS:
            return c.DM([hover_thrust / 4.0] * 4)
        if mode == ControlMode.FORCE_TORQUE:
            return c.DM([hover_thrust, 0.0, 0.0, 0.0])
        raise ValueError(f"Unsupported control mode: {mode}")

    def _add_control_constraints(self, opti: Any, control: Any, mode: ControlMode) -> None:
        """Add mode-specific actuator constraints."""
        c = _require_casadi()
        if mode == ControlMode.ACC_YAWRATE:
            opti.subject_to(opti.bounded(-float(self.config.max_acceleration), control[0:3], float(self.config.max_acceleration)))
            opti.subject_to(opti.bounded(-float(self.config.yaw_rate_limit), control[3], float(self.config.yaw_rate_limit)))
        elif mode == ControlMode.THRUST_BODYRATE:
            opti.subject_to(opti.bounded(float(self.config.thrust_min), control[0], float(self.config.thrust_max)))
            opti.subject_to(opti.bounded(-float(self.config.body_rate_limit), control[1:4], float(self.config.body_rate_limit)))
        elif mode == ControlMode.MOTOR_THRUSTS:
            opti.subject_to(opti.bounded(float(self.config.motor_thrust_min), control, float(self.config.motor_thrust_max)))
        elif mode == ControlMode.FORCE_TORQUE:
            opti.subject_to(opti.bounded(float(self.config.thrust_min), control[0], float(self.config.thrust_max)))
            opti.subject_to(opti.bounded(-float(self.config.torque_limit), control[1:4], float(self.config.torque_limit)))
        else:
            raise ValueError(f"Unsupported control mode: {mode}")

    def _set_initial_guess(self, opti: Any, states: Any, controls: Any, references: np.ndarray, mode: ControlMode) -> None:
        """Seed the NLP with references and hover-like controls."""
        opti.set_initial(states, references.T)
        if controls.shape[1] == 0:
            return
        guess = np.zeros((4, controls.shape[1]), dtype=np.float32)
        if mode == ControlMode.THRUST_BODYRATE:
            guess[0, :] = float(self.config.mass) * float(self.config.gravity)
        elif mode == ControlMode.MOTOR_THRUSTS:
            guess[:, :] = float(self.config.mass) * float(self.config.gravity) / 4.0
        elif mode == ControlMode.FORCE_TORQUE:
            guess[0, :] = float(self.config.mass) * float(self.config.gravity)
        opti.set_initial(controls, guess)

    def _motor_torques_sym(self, motor_thrusts: Any) -> Any:
        """CasADi X-quad torque mixer."""
        c = _require_casadi()
        f1, f2, f3, f4 = motor_thrusts[0], motor_thrusts[1], motor_thrusts[2], motor_thrusts[3]
        arm = float(self.config.arm_length) / np.sqrt(2.0)
        return c.vertcat(
            arm * (f1 + f4 - f2 - f3),
            arm * (f1 + f2 - f3 - f4),
            float(self.config.k_torque) * (f1 - f2 + f3 - f4),
        )

    def _step_numpy(self, state: np.ndarray, control: np.ndarray, mode: ControlMode) -> np.ndarray:
        """Numpy dynamics used to roll out the first MPC command in receding mode."""
        dt = float(self.config.dt)
        next_state = np.asarray(state, dtype=np.float32).copy()
        quat = _normalize_quaternion(next_state[3:7])
        omega = next_state[10:13]

        if mode == ControlMode.ACC_YAWRATE:
            acceleration = control[:3]
            next_omega = np.array([0.0, 0.0, control[3]], dtype=np.float32)
        elif mode == ControlMode.THRUST_BODYRATE:
            next_omega = control[1:4].astype(np.float32)
            acceleration = _rotation_matrix_np(quat) @ np.array([0.0, 0.0, control[0] / self.config.mass], dtype=np.float32)
            acceleration += np.array([0.0, 0.0, -self.config.gravity], dtype=np.float32)
        elif mode == ControlMode.MOTOR_THRUSTS:
            motor_thrusts = np.maximum(control[:4], 0.0).astype(np.float32)
            torque = self._motor_torques_np(motor_thrusts)
            next_omega = omega + torque / self.inertia * dt
            acceleration = _rotation_matrix_np(quat) @ np.array([0.0, 0.0, motor_thrusts.sum() / self.config.mass], dtype=np.float32)
            acceleration += np.array([0.0, 0.0, -self.config.gravity], dtype=np.float32)
        elif mode == ControlMode.FORCE_TORQUE:
            thrust = max(float(control[0]), 0.0)
            torque = control[1:4].astype(np.float32)
            next_omega = omega + torque / self.inertia * dt
            acceleration = _rotation_matrix_np(quat) @ np.array([0.0, 0.0, thrust / self.config.mass], dtype=np.float32)
            acceleration += np.array([0.0, 0.0, -self.config.gravity], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported control mode: {mode}")

        next_state[:3] = next_state[:3] + next_state[7:10] * dt + 0.5 * acceleration * dt * dt
        next_state[7:10] = _clip_norm(next_state[7:10] + acceleration * dt, self.config.max_velocity)
        quat_dot = 0.5 * _quat_multiply_np(quat, np.array([0.0, *next_omega], dtype=np.float32))
        next_state[3:7] = _normalize_quaternion(quat + quat_dot * dt)
        next_state[10:13] = next_omega
        return next_state.astype(np.float32)

    def _motor_torques_np(self, motor_thrusts: np.ndarray) -> np.ndarray:
        """Numpy X-quad torque mixer."""
        f1, f2, f3, f4 = [float(v) for v in motor_thrusts]
        arm = float(self.config.arm_length) / np.sqrt(2.0)
        return np.array(
            [
                arm * (f1 + f4 - f2 - f3),
                arm * (f1 + f2 - f3 - f4),
                float(self.config.k_torque) * (f1 - f2 + f3 - f4),
            ],
            dtype=np.float32,
        )


def main() -> None:
    """Run a small ACC_YAWRATE MPC tracking demo and save result artifacts."""
    dt = 0.02
    duration = 1.0
    steps = int(round(duration / dt)) + 1
    horizon = min(10, steps - 1)
    start = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    goal = np.array([0.5, -0.25, 1.25], dtype=np.float32)
    times = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    blend = 3.0 * times**2 - 2.0 * times**3

    reference = np.zeros((steps, 13), dtype=np.float32)
    reference[:, :3] = start[None, :] + blend[:, None] * (goal - start)[None, :]
    reference[:, 3] = 1.0
    reference[:, 7:10] = np.gradient(reference[:, :3], dt, axis=0).astype(np.float32)
    reference[0, 7:10] = 0.0
    reference[-1, 7:10] = 0.0

    controller = QuadrotorMPCController(
        QuadrotorMPCConfig(
            dt=dt,
            horizon=horizon,
            receding=True,
            max_acceleration=10.0,
            solver_max_iter=80,
        )
    )
    result = controller.solve(reference, initial_state=reference[0], mode=ControlMode.ACC_YAWRATE)

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "quad_mpc_demo.npz"
    plot_path = output_dir / "quad_mpc_demo.png"
    np.savez_compressed(
        data_path,
        states=result.states,
        controls=result.controls,
        references=result.references,
        tracking_error=result.tracking_error,
    )

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    # matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    state_steps = np.arange(result.states.shape[0])
    control_steps = np.arange(result.controls.shape[0])
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)

    for idx, label in enumerate(("x", "y", "z")):
        axes[0].plot(state_steps, result.states[:, idx], label=f"{label} state")
        axes[0].plot(state_steps, result.references[:, idx], "--", label=f"{label} ref")
    axes[0].set_ylabel("position")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3, fontsize=8)

    for idx, label in enumerate(("vx", "vy", "vz"), start=7):
        axes[1].plot(state_steps, result.states[:, idx], label=f"{label} state")
        axes[1].plot(state_steps, result.references[:, idx], "--", label=f"{label} ref")
    axes[1].plot(state_steps, result.tracking_error, color="black", linewidth=1.6, label="position error")
    axes[1].set_ylabel("velocity / error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=4, fontsize=8)

    for idx, label in enumerate(("ax", "ay", "az", "yaw_rate")):
        axes[2].step(control_steps, result.controls[:, idx], where="post", label=label)
    axes[2].set_xlabel("MPC step")
    axes[2].set_ylabel("control")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=4, fontsize=8)

    fig.tight_layout()
    plt.show()
    # fig.savefig(plot_path, dpi=150)
    # plt.close(fig)
    print(f"Saved MPC demo data to {data_path}")
    print(f"Saved MPC tracking plot to {plot_path}")


if __name__ == "__main__":
    main()
