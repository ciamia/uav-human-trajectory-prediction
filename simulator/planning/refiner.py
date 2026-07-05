from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


CollisionChecker = Callable[[np.ndarray], Any]


@dataclass(frozen=True)
class RefinedTrajectory:
    time: np.ndarray
    position_refs: np.ndarray
    velocity_refs: np.ndarray
    acceleration_refs: np.ndarray
    collision_info: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "position_refs": self.position_refs,
            "velocity_refs": self.velocity_refs,
            "acceleration_refs": self.acceleration_refs,
            "collision_info": self.collision_info,
        }


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float32)


def _positions(waypoints: Any) -> tuple[np.ndarray, np.ndarray | None]:
    values = _to_numpy(waypoints)
    if values.ndim != 2 or values.shape[1] not in (3, 6):
        raise ValueError("waypoints must have shape [T, 3] or [T, 6]")
    if values.shape[0] < 2:
        raise ValueError("at least two waypoints are required")
    velocities = values[:, 3:6].copy() if values.shape[1] == 6 else None
    return values[:, :3].copy(), velocities


def _knot_times(num_waypoints: int, timestamps: Any | None, total_duration: float | None) -> np.ndarray:
    if timestamps is not None:
        times = _to_numpy(timestamps).reshape(-1)
        if times.shape[0] != num_waypoints:
            raise ValueError("timestamps must have one value per waypoint")
    else:
        duration = float(total_duration) if total_duration is not None else float(num_waypoints - 1)
        if duration <= 0.0:
            raise ValueError("total_duration must be positive")
        times = np.linspace(0.0, duration, num_waypoints, dtype=np.float32)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("timestamps must be strictly increasing")
    return times.astype(np.float32)


def _sample_times(knot_times: np.ndarray, num_samples: int | None, dt: float | None) -> np.ndarray:
    if num_samples is not None and dt is not None:
        raise ValueError("provide either num_samples or dt, not both")
    if dt is not None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        count = int(np.floor((knot_times[-1] - knot_times[0]) / dt)) + 1
        samples = knot_times[0] + np.arange(count, dtype=np.float32) * dt
        if samples[-1] < knot_times[-1]:
            samples = np.concatenate([samples, knot_times[-1:]])
        return samples.astype(np.float32)
    count = int(num_samples) if num_samples is not None else max(2, (len(knot_times) - 1) * 10 + 1)
    if count < 2:
        raise ValueError("num_samples must be at least 2")
    return np.linspace(knot_times[0], knot_times[-1], count, dtype=np.float32)


def _estimate_velocities(positions: np.ndarray, times: np.ndarray, provided: np.ndarray | None) -> np.ndarray:
    if provided is not None:
        return provided.astype(np.float32)
    velocities = np.zeros_like(positions)
    velocities[0] = (positions[1] - positions[0]) / (times[1] - times[0])
    velocities[-1] = (positions[-1] - positions[-2]) / (times[-1] - times[-2])
    for idx in range(1, len(positions) - 1):
        velocities[idx] = (positions[idx + 1] - positions[idx - 1]) / (times[idx + 1] - times[idx - 1])
    return velocities.astype(np.float32)


def _estimate_accelerations(velocities: np.ndarray, times: np.ndarray) -> np.ndarray:
    accelerations = np.zeros_like(velocities)
    accelerations[0] = (velocities[1] - velocities[0]) / (times[1] - times[0])
    accelerations[-1] = (velocities[-1] - velocities[-2]) / (times[-1] - times[-2])
    for idx in range(1, len(velocities) - 1):
        accelerations[idx] = (velocities[idx + 1] - velocities[idx - 1]) / (times[idx + 1] - times[idx - 1])
    return accelerations.astype(np.float32)


def _segment_indices(knot_times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(knot_times, query_times, side="right") - 1
    return np.clip(indices, 0, len(knot_times) - 2)


def _cubic_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    duration: float,
    s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2 = s * s
    s3 = s2 * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    dh00 = 6.0 * s2 - 6.0 * s
    dh10 = 3.0 * s2 - 4.0 * s + 1.0
    dh01 = -6.0 * s2 + 6.0 * s
    dh11 = 3.0 * s2 - 2.0 * s
    ddh00 = 12.0 * s - 6.0
    ddh10 = 6.0 * s - 4.0
    ddh01 = -12.0 * s + 6.0
    ddh11 = 6.0 * s - 2.0

    position = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1
    velocity = (dh00 * p0 + dh10 * duration * v0 + dh01 * p1 + dh11 * duration * v1) / duration
    acceleration = (ddh00 * p0 + ddh10 * duration * v0 + ddh01 * p1 + ddh11 * duration * v1) / (duration**2)
    return position, velocity, acceleration


def _quintic_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    duration: float,
    s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s2 = s * s
    s3 = s2 * s
    s4 = s3 * s
    s5 = s4 * s
    h00 = 1.0 - 10.0 * s3 + 15.0 * s4 - 6.0 * s5
    h10 = s - 6.0 * s3 + 8.0 * s4 - 3.0 * s5
    h20 = 0.5 * (s2 - 3.0 * s3 + 3.0 * s4 - s5)
    h01 = 10.0 * s3 - 15.0 * s4 + 6.0 * s5
    h11 = -4.0 * s3 + 7.0 * s4 - 3.0 * s5
    h21 = 0.5 * (s3 - 2.0 * s4 + s5)

    dh00 = -30.0 * s2 + 60.0 * s3 - 30.0 * s4
    dh10 = 1.0 - 18.0 * s2 + 32.0 * s3 - 15.0 * s4
    dh20 = 0.5 * (2.0 * s - 9.0 * s2 + 12.0 * s3 - 5.0 * s4)
    dh01 = 30.0 * s2 - 60.0 * s3 + 30.0 * s4
    dh11 = -12.0 * s2 + 28.0 * s3 - 15.0 * s4
    dh21 = 0.5 * (3.0 * s2 - 8.0 * s3 + 5.0 * s4)

    ddh00 = -60.0 * s + 180.0 * s2 - 120.0 * s3
    ddh10 = -36.0 * s + 96.0 * s2 - 60.0 * s3
    ddh20 = 0.5 * (2.0 - 18.0 * s + 36.0 * s2 - 20.0 * s3)
    ddh01 = 60.0 * s - 180.0 * s2 + 120.0 * s3
    ddh11 = -24.0 * s + 84.0 * s2 - 60.0 * s3
    ddh21 = 0.5 * (6.0 * s - 24.0 * s2 + 20.0 * s3)

    d = duration
    position = h00 * p0 + h10 * d * v0 + h20 * d**2 * a0 + h01 * p1 + h11 * d * v1 + h21 * d**2 * a1
    velocity = (dh00 * p0 + dh10 * d * v0 + dh20 * d**2 * a0 + dh01 * p1 + dh11 * d * v1 + dh21 * d**2 * a1) / d
    acceleration = (ddh00 * p0 + ddh10 * d * v0 + ddh20 * d**2 * a0 + ddh01 * p1 + ddh11 * d * v1 + ddh21 * d**2 * a1) / (d**2)
    return position, velocity, acceleration


def _interpolate(
    positions: np.ndarray,
    velocities: np.ndarray,
    accelerations: np.ndarray,
    knot_times: np.ndarray,
    query_times: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    refs = np.zeros((len(query_times), 3), dtype=np.float32)
    vel_refs = np.zeros_like(refs)
    acc_refs = np.zeros_like(refs)
    indices = _segment_indices(knot_times, query_times)
    for out_idx, seg_idx in enumerate(indices):
        duration = float(knot_times[seg_idx + 1] - knot_times[seg_idx])
        s = float((query_times[out_idx] - knot_times[seg_idx]) / duration)
        if method == "cubic":
            result = _cubic_segment(
                positions[seg_idx],
                positions[seg_idx + 1],
                velocities[seg_idx],
                velocities[seg_idx + 1],
                duration,
                s,
            )
        elif method == "quintic":
            result = _quintic_segment(
                positions[seg_idx],
                positions[seg_idx + 1],
                velocities[seg_idx],
                velocities[seg_idx + 1],
                accelerations[seg_idx],
                accelerations[seg_idx + 1],
                duration,
                s,
            )
        else:
            raise ValueError("method must be 'cubic' or 'quintic'")
        refs[out_idx], vel_refs[out_idx], acc_refs[out_idx] = result
    refs[0] = positions[0]
    refs[-1] = positions[-1]
    return refs, vel_refs, acc_refs


def _max_norm(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.linalg.norm(values, axis=-1).max())


def _jerk_refs(acceleration_refs: np.ndarray, time_refs: np.ndarray) -> np.ndarray:
    if len(acceleration_refs) < 2:
        return np.zeros_like(acceleration_refs)
    return np.gradient(acceleration_refs, time_refs, axis=0).astype(np.float32)


def _limit_time_scale(
    velocity_refs: np.ndarray,
    acceleration_refs: np.ndarray,
    time_refs: np.ndarray,
    max_velocity: float | None,
    max_acceleration: float | None,
    max_jerk: float | None,
) -> float:
    scale = 1.0
    if max_velocity is not None:
        scale = max(scale, _max_norm(velocity_refs) / float(max_velocity))
    if max_acceleration is not None:
        scale = max(scale, np.sqrt(_max_norm(acceleration_refs) / float(max_acceleration)))
    if max_jerk is not None:
        jerk = _jerk_refs(acceleration_refs, time_refs)
        scale = max(scale, np.cbrt(_max_norm(jerk) / float(max_jerk)))
    return float(max(scale, 1.0))


def refine_waypoints(
    waypoints: Any,
    timestamps: Any | None = None,
    total_duration: float | None = None,
    num_samples: int | None = None,
    dt: float | None = None,
    method: str = "cubic",
    max_velocity: float | None = None,
    max_acceleration: float | None = None,
    max_jerk: float | None = None,
    collision_checker: CollisionChecker | None = None,
) -> RefinedTrajectory:
    """Refine raw trajectory waypoints into smooth position, velocity, and acceleration references.

    Collision checking is exposed as a hook via collision_checker(position_refs).
    This is simulation-only and deliberately does not integrate any flight stack.
    """
    positions, provided_velocities = _positions(waypoints)
    knot_times = _knot_times(len(positions), timestamps, total_duration)
    query_times = _sample_times(knot_times, num_samples=num_samples, dt=dt)
    velocities = _estimate_velocities(positions, knot_times, provided_velocities)
    accelerations = _estimate_accelerations(velocities, knot_times)
    position_refs, velocity_refs, acceleration_refs = _interpolate(
        positions,
        velocities,
        accelerations,
        knot_times,
        query_times,
        method=method,
    )

    scale = _limit_time_scale(
        velocity_refs,
        acceleration_refs,
        query_times,
        max_velocity=max_velocity,
        max_acceleration=max_acceleration,
        max_jerk=max_jerk,
    )
    if scale > 1.0:
        scaled_knot_times = knot_times[0] + (knot_times - knot_times[0]) * scale
        scaled_query_times = _sample_times(scaled_knot_times, num_samples=num_samples, dt=dt)
        scaled_velocities = velocities / scale
        scaled_accelerations = accelerations / (scale**2)
        position_refs, velocity_refs, acceleration_refs = _interpolate(
            positions,
            scaled_velocities,
            scaled_accelerations,
            scaled_knot_times,
            scaled_query_times,
            method=method,
        )
        query_times = scaled_query_times

    collision_info = collision_checker(position_refs) if collision_checker is not None else None
    return RefinedTrajectory(
        time=query_times.astype(np.float32),
        position_refs=position_refs.astype(np.float32),
        velocity_refs=velocity_refs.astype(np.float32),
        acceleration_refs=acceleration_refs.astype(np.float32),
        collision_info=collision_info,
    )
