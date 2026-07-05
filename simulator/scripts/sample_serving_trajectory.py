from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import os
import select
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import torch
import timeit

from control.mpc_controller import DroneMPCConfig, DroneMPCController, DroneMPCResult
from data.expert_generator import (
    ExpertGeneratorConfig,
    generate_expert_trajectory,
    predict_people_positions,
    trajectory_collision_mask,
)
from env.drone import Drone
from env.setup_genesis_world import (
    DEFAULT_DRONE_POSITION,
    WORLD_SIZE,
    _add_scene_entities,
    assign_ids,
    make_buildings,
    setup_people_initial_position,
    SetupGenesisWorld
)
from flow.sampler import sample
from models.flow_transformer import FlowTransformer
from planning.hover_goal import compute_hover_goal
from planning.refiner import RefinedTrajectory, refine_waypoints
from planning.target_selection import select_coffee_target
from scripts.train_serving_fm import TrainServingConfig, build_model, resolve_device

import utils.trajectories as traj_utils
import utils.utils as utils

DEFAULT_COLOR = (1.0, 0.05, 0.05, 1.0)
@dataclass(frozen=True)
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


def run_sample(config: SampleServingConfig) -> dict[str, Any]:
    """Sample, refine, visualize, and roll out one serving trajectory."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    model, train_config, horizon = load_model(config.checkpoint, device, config.horizon)

    # world = make_world(config, train_config, rng)
    world = SetupGenesisWorld()
    should_close = True
    try:
        condition_np = build_scene_condition(world, horizon, config.safety_margin)
        condition = tensors_from_condition(condition_np, device)
        noise = torch.randn(1, horizon, 6, device=device)
        sampled = sample(model, noise, condition, steps=config.sample_steps, method=config.sample_method)[0].detach().cpu().numpy()
        sampled = anchor_trajectory(sampled, condition_np["drone_start"][0], condition_np["hover_goal"][0])

        refined = refine_candidate(sampled, horizon, config)
        predicted_people = condition_np["future_people_predictions"][0]
        sampled_collision_free = not bool(
            trajectory_collision_mask(
                refined.position_refs,
                world.buildings,
                predicted_people,
                ExpertGeneratorConfig(world_size=tuple(float(v) for v in world.world_size), safety_margin=config.safety_margin),
            ).any()
        )

        used_expert_fallback = False
        if not sampled_collision_free or not np.isfinite(refined.position_refs).all():
            used_expert_fallback = True
            expert = generate_expert_trajectory(
                start=condition_np["drone_start"][0],
                hover_goal=condition_np["hover_goal"][0, :3],
                buildings=world.buildings,
                people=world.people,
                config=ExpertGeneratorConfig(
                    world_size=tuple(float(v) for v in world.world_size),
                    horizon=horizon,
                    safety_margin=config.safety_margin,
                    max_velocity=config.max_velocity,
                    max_acceleration=config.max_acceleration,
                ),
            )
            refined = refine_candidate(expert.as_state_trajectory(), horizon, config)

        collision_free = not bool(
            trajectory_collision_mask(
                refined.position_refs,
                world.buildings,
                predicted_people,
                ExpertGeneratorConfig(world_size=tuple(float(v) for v in world.world_size), safety_margin=config.safety_margin),
            ).any()
        )
        tracking = rollout_flight(world, refined, config)
        plot_path = plot_sample(
            world,
            condition_np,
            sampled,
            refined,
            predicted_people,
            config.output,
            used_expert_fallback,
            tracked_positions=tracking.states[:, :3],
        )
        video_path = world.save_video(config.video)
        if config.keep_open and world.scene is not None:
            should_close = False
            hover_state = None if config.use_mpc and config.mpc_fullpose_control else final_hover_state(refined)
            if hover_state is not None:
                set_drone_position_state(world, hover_state)
            print("Genesis viewer kept open. Close the viewer or press Ctrl+C when finished.")
            hold_viewer_open(world, hover_state)
    finally:
        if should_close:
            world.close()

    print(f"sampled_collision_free: {sampled_collision_free}")
    print(f"used_expert_fallback: {used_expert_fallback}")
    print(f"collision_free: {collision_free}")
    print(f"hover_error: {np.linalg.norm(refined.position_refs[-1] - condition_np['hover_goal'][0, :3]):.3f}")
    print(f"mpc_max_tracking_error: {float(tracking.tracking_error.max()):.3f}")
    print(f"saved plot to {plot_path}")
    print(f"saved video to {video_path}")
    return {
        "plot": str(plot_path),
        "video": str(video_path),
        "collision_free": collision_free,
        "used_expert_fallback": used_expert_fallback,
        "sampled_collision_free": sampled_collision_free,
        "mpc_max_tracking_error": float(tracking.tracking_error.max()),
    }

def run_missions(config: SampleServingConfig) -> dict[str, Any]:
    """Run realtime FM replanning with background trajectory generation."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = resolve_device(config.device)
    model, train_config, horizon = load_model(config.checkpoint, device, config.horizon)

    # world = make_world(config, train_config, rng)
    world = SetupGenesisWorld()
    dt = 0.05    # Time step
    horizon = 1
    N = 20      # Horizontal length
    quad_name = "cf2x"
    MPCController = DroneMPCController(t_horizon = horizon, n_nodes = N, opt = dt, quad_name = quad_name)
    controller = MPCController.controller
    quad       = MPCController.quad
    com_dt     = MPCController.controller.dt
    com_N      = MPCController.controller.N

    mission_results: list[dict[str, Any]] = []

    print("Enter missions as '<mission> <person_id>' such as 'follow 2'. Type q to quit.")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    mission_mode = False
    request_mode = False
    tracking_active = False
    reference_traj = np.empty((0, 13), dtype=float)
    traj_X = None
    recover_color = True
    target = world.get_hover_pose()
    hover_goal = world.get_hover_pose()
    ref_idx = -1
    follow_replan_interval = 0.5
    follow_target_threshold = 0.35
    last_follow_plan_time = -float("inf")
    last_follow_target: np.ndarray | None = None
    target_changed = False
    rolling_segment_distance = 2.0
    p2p_speed = 2.0
    p2p_goal_tolerance = 0.25
    pending_p2p_plan: concurrent.futures.Future[dict[str, Any]] | None = None
    desired_p2p_target: np.ndarray | None = None
    last_submitted_p2p_target: np.ndarray | None = None
    p2p_request_id = 0
    active_p2p_request_id = 0
    last_p2p_submit = -float("inf")
    try:
        while True:
            request = poll_mission_request()
            target_changed = False

            if request in {"hover", "stop_mission"}:
                mission_mode = False
                request_mode = False
                desired_p2p_target = None
                last_submitted_p2p_target = None
                tracking_active = False
                reference_traj = np.empty((0, 13), dtype=float)
                ref_idx = -1
                if pending_p2p_plan is not None:
                    pending_p2p_plan.cancel()
                    pending_p2p_plan = None
                request = None
            if request in {"q", "quit", "exit"}:
                break
            if request:                                                                                                 
                try:
                    # assignment = parse_mission_request(request)
                    request_dic = parse_request(request)
                    if request_dic["type"] == "mission":
                        mission_mode = True
                        request_mode = False
                        tracking_active = False
                        ref_idx = -1
                        last_follow_plan_time = -float("inf")
                        last_follow_target = None
                        last_submitted_p2p_target = None
                        desired_p2p_target = None
                    else: 
                        traj_X, target = apply_request(world, request_dic) # will return the command for drone: traj_X = "hover"/"p2p"
                        mission_mode = False
                        request_mode = True
                        desired_p2p_target = np.asarray(target, dtype=float).copy() if traj_X == "p2p" else None
                        last_submitted_p2p_target = None
                        print(f"traj_x: {traj_X}, target: {target}")
                except ValueError as exc:
                    print(exc)
                    continue
            
            if mission_mode:
                people_by_id = {getattr(person, "id", f"person_{idx}"): person for idx, person in enumerate(world.people)}
                target_id = request_dic["id"]
                person=people_by_id[f"person_{target_id}"]
                
                mission_name = request_dic["name"]
                if mission_name == "circle":
                    traj_X = "circle"
                    desired_p2p_target = None
                    last_submitted_p2p_target = None
                    reference_traj, t_ref, reference_u = \
                    traj_utils.get_trajectory_X(quad, traj_X, com_dt)
                else:
                    if mission_name in {"hover", "follow"}: # , "circle"
                        target_ = person.get_position()
                    elif mission_name in {"land"}:
                        target_ = world.get_position()
                        target_[2] = 0.1  # land the drone at current position 
                    elif mission_name in {"back"}:
                        target_ = DEFAULT_DRONE_POSITION
                    else:
                        target_ = target
                        print(f"Please define missipn -- {mission_name} -- first!")
                    if last_follow_target is not None:
                        target_changed = np.linalg.norm(target_[:3] - last_follow_target) > follow_target_threshold
                        if target_changed:
                            target = np.hstack([target_[:3], 0.0])
                    else:
                        target = np.hstack([target_[:3], 0.0])
                        traj_X = None 
                    desired_p2p_target = np.asarray(target, dtype=float).copy()

                person.color = (1.0, 1.0, 1.0, 1.0)
                recover_color = False
            else:
                recover_color = True
            if recover_color:
                for person in world.people:
                    person.color = DEFAULT_COLOR

            if pending_p2p_plan is not None and pending_p2p_plan.done():
                try:
                    plan = pending_p2p_plan.result()
                except Exception as exc:
                    print(f"p2p planner failed: {exc}")
                else:
                    plan_target_is_current = (
                        desired_p2p_target is not None
                        and np.linalg.norm(plan["target"][:3] - desired_p2p_target[:3]) <= follow_target_threshold
                    )
                    if plan["request_id"] == active_p2p_request_id and plan_target_is_current:
                        reference_traj = plan["reference_traj"]
                        last_follow_target = plan["target"]
                        tracking_active = reference_traj.shape[0] > 0
                        ref_idx = 0 if tracking_active else -1
                        traj_X = None
                        print(
                            f"new rolling p2p segment: dist={plan['distance']:.2f}m "
                            f"segment_end={plan['segment_end'].round(3).tolist()} "
                            f"samples={reference_traj.shape[0]} plan_time={plan['planning_time']:.4f}s"
                        )
                    else:
                        print("ignored stale p2p segment")
                pending_p2p_plan = None

            if desired_p2p_target is not None:
                position_cur = _as_numpy_vector(world.drone.get_pos())
                distance_to_target = float(np.linalg.norm(desired_p2p_target[:3] - position_cur))
                if distance_to_target <= p2p_goal_tolerance and not tracking_active:
                    desired_p2p_target = None
                    last_submitted_p2p_target = None
                    hover_goal = np.hstack([position_cur, [1, 0, 0, 0], np.zeros(3), np.zeros(3)])
                else:
                    now = time.monotonic()
                    submitted_target_changed = (
                        last_submitted_p2p_target is None
                        or np.linalg.norm(desired_p2p_target[:3] - last_submitted_p2p_target[:3]) > follow_target_threshold
                    )
                    should_submit_p2p = (
                        pending_p2p_plan is None
                        and (
                            not tracking_active
                            or (
                                submitted_target_changed
                                and now - last_p2p_submit >= follow_replan_interval
                            )
                        )
                    )
                    if should_submit_p2p:
                        euler_cur = utils.quaternion_to_euler(world.drone.get_quat())
                        p2p_request_id += 1
                        active_p2p_request_id = p2p_request_id
                        last_submitted_p2p_target = desired_p2p_target.copy()
                        pending_p2p_plan = executor.submit(
                            build_rolling_p2p_reference,
                            request_id=active_p2p_request_id,
                            dt=com_dt,
                            p_start=position_cur.copy(),
                            p_end=desired_p2p_target[:3].copy(),
                            yaw_start_deg=float(np.rad2deg(euler_cur[2])),
                            yaw_end_deg=float(desired_p2p_target[3]),
                            speed=p2p_speed,
                            segment_distance=rolling_segment_distance,
                        )
                        last_p2p_submit = now

            if ref_idx >= reference_traj.shape[0]:
                tracking_active = False
                traj_X = None
                if reference_traj.shape[0] > 0:
                    hover_goal = np.hstack([reference_traj[-1, 0:3], [1,0,0,0], np.zeros(3), np.zeros(3)])
                ref_idx = -1
                print(f"reach the last reference, hover at: {hover_goal}")
            if tracking_active:
                goal = get_ref(reference_traj, ref_idx, com_N)
                ref_idx += 1
            else:
                # goal = np.tile(hover_goal, (com_N+1, 1))
                goal = np.vstack([hover_goal] * (com_N+1))
            sim_dt = float(getattr(world.config, "dt", com_dt))
            world_ratio = max(1, int(round(com_dt / max(sim_dt, 1e-6))))
            for t in range(world_ratio):
                current = world.get_state()
                thrusts = controller.run_optimization(initial_state=current, goal=goal, mode='traj')[:4]
                rpms = world.thrusts_to_rpms(thrusts)
                # print(f"current: {current}")
                # print(f"thrusts: {thrusts}")
                world.drone.set_propellers_rpm(rpms)

                next_position = None
                update_people_state(sim_dt, rng, target_id, next_position, world)
             # Step simulation
                # for person in world.people:
                #     person.update(sim_dt*5, rng, WORLD_SIZE)
                world.scene.step()
                world.settle_on_ground_if_landed()

            if "PYTEST_VERSION" in os.environ:
                break
            
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        world.close()

    return {
        "missions": mission_results,
        "count": len(mission_results),
        "collision_free": all(result["collision_free"] for result in mission_results),
    }


def _as_numpy_vector(value: Any) -> np.ndarray:
    """Convert tensor-like simulator values to a flat float numpy vector."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=float).reshape(-1)

def update_people_state(world, sim_dt, rng, id, position):
    people_by_id = {getattr(person, "id", f"person_{idx}"): person for idx, person in enumerate(world.people)}
    person=people_by_id[f"person_{id}"]

    for idx , person in enumerate(world.people):
        if idx == id -1 and position is not None:
            person.set_position(position)
        else:
            person.update(sim_dt*5, rng, WORLD_SIZE)


def build_rolling_p2p_reference(
    request_id: int,
    dt: float,
    p_start: np.ndarray,
    p_end: np.ndarray,
    yaw_start_deg: float,
    yaw_end_deg: float,
    speed: float,
    segment_distance: float,
) -> dict[str, Any]:
    """Build one short p2p reference segment toward the latest target."""
    start_time = timeit.default_timer()
    p_start = np.asarray(p_start, dtype=float).reshape(3)
    p_end = np.asarray(p_end, dtype=float).reshape(3)
    direction = p_end - p_start
    distance = float(np.linalg.norm(direction))

    if distance > segment_distance:
        segment_end = p_start + direction / distance * segment_distance
        yaw_end = yaw_start_deg + (yaw_end_deg - yaw_start_deg) * (segment_distance / distance)
    else:
        segment_end = p_end.copy()
        yaw_end = yaw_end_deg

    midpoint = 0.5 * (p_start + p_end)
    # waypoints = np.stack([p_start, midpoint, segment_end], axis=0)
    waypoints = np.stack([p_start, midpoint, p_end], axis=0)
    yaw_targets = np.deg2rad(np.array([yaw_start_deg, 0.5 * (yaw_start_deg + yaw_end), yaw_end], dtype=float))
    segment_length = max(float(np.linalg.norm(segment_end - p_start)), 1e-6)
    segment_dt = max(segment_length / max(speed, 1e-6) / 2.0, float(dt))
    poly = traj_utils.fit_multi_segment_polynomial_trajectory(waypoints.T, yaw_targets)
    traj, yaw_traj, _ = traj_utils.get_full_traj(poly, target_dt=segment_dt, int_dt=float(dt))

    sample_count = traj.shape[2]
    reference_traj = np.zeros((sample_count, 13), dtype=float)
    reference_traj[:, 0:3] = traj[0].T
    reference_traj[:, 7:10] = traj[1].T
    for idx, yaw in enumerate(yaw_traj[0]):
        reference_traj[idx, 3:7] = utils.euler_to_quaternion(0.0, 0.0, float(yaw))
    reference_traj[:, 10] = yaw_traj[1]

    return {
        "request_id": request_id,
        "reference_traj": reference_traj,
        "target": p_end,
        "segment_end": segment_end,
        "distance": distance,
        "planning_time": timeit.default_timer() - start_time,
    }


def poll_mission_request() -> str | None:
    """Poll for a terminal mission request without pausing the control loop."""
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        return sys.stdin.readline().strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def parse_mission_request(request: str) -> dict[str, Any]:
    """Parse a terminal request and build the mission assignment."""
    parts = request.replace(",", " ").split()
    if not parts:
        raise ValueError("Empty mission request.")
    mission = parts[0]
    person_id = parts[1] if len(parts) > 1 else None
    assignment = mission_assign(person_id, mission)
    if assignment["mission"] is None:
        raise ValueError("Mission request did not produce a drone mission.")
    return assignment
from typing import Any


def get_ref(ref_traj, i, N):
    pos_ref   = np.array(ref_traj[:,:3])
    quat_ref  = np.array(ref_traj[:,3:7])
    vel_ref   = np.array(ref_traj[:,7:10])
    omega_ref = np.array(ref_traj[:,10:13])

    n_steps = len(pos_ref)
    i_end = min(i + N + 1, n_steps)
    ref_pos = pos_ref[i:i_end, :]
    ref_vel = vel_ref[i:i_end, :]
    ref_quat = quat_ref[i:i_end, :]
    ref_omega = omega_ref[i:i_end, :]

    n_available = i_end - i
    # This creates consistent final condition: constant position, zero velocity
    if n_available < N + 1:
        pad_len = N + 1 - n_available
        ref_pos = np.vstack([ref_pos, np.tile(pos_ref[-1], (pad_len, 1))])
        ref_vel = np.vstack([ref_vel, np.zeros((pad_len, 3))])  # ZERO velocity for padding!
        ref_quat = np.vstack([ref_quat, np.tile([1, 0, 0, 0], (pad_len, 1))])
        ref_omega = np.vstack([ref_omega, np.zeros((pad_len, 3))])
        

    state_goal = np.hstack([ref_pos, ref_quat, ref_vel, ref_omega])
    return state_goal

def parse_request(request: str) -> dict[str, Any]:
    request = request.strip()
    parts = request.replace(",", " ").split()

    if not parts:
        raise ValueError("Empty request")

    mission_names = {"hover", "circle", "follow", "land", "back"}
    drone_requests = {"state", "pose", "twist"}
    drone_commands = {"hover", "goto"}
    people_requests = {"state", "pose", "twist"}
    people_commands = {"goto", "velocity"}

    category = parts[0].lower()

    # Mission
    if category == "mission":
        if len(parts) != 3:
            raise ValueError(
                "Mission format: mission <hover|circle|follow|land|back> <target_id>"
            )

        mission_name = parts[1].lower()
        target_id = int(parts[2])

        if mission_name not in mission_names:
            raise ValueError(f"Invalid mission name: {mission_name}")

        return {
            "which": "drone",
            "type": "mission",
            "name": mission_name,
            "id": target_id,
        }

    # Drone
    if category == "drone":
        if len(parts) < 2:
            raise ValueError("Drone request missing name")

        name = parts[1].lower()

        if name in drone_requests:
            return {
                "which": "drone",
                "type": "request",
                "name": name,
            }

        if name == "hover":
            return {
                "which": "drone",
                "type": "command",
                "name": "hover",
            }

        if name == "goto":
            values = [float(v) for v in parts[2:]]

            if len(values) == 3:
                cmd = np.hstack([values, 0])
            elif len(values) == 4:
                cmd = np.array(values)
            else:
                raise ValueError(
                    "Drone goto format: drone goto x y z [yaw]"
                )

            return {
                "which": "drone",
                "type": "command",
                "name": "goto",
                "command": cmd,
            }

    # People
    if category == "people":
        if len(parts) < 2:
            raise ValueError("People request missing name")
        target_id = int(parts[1])
        name = parts[2].lower()
        

        if name in people_requests:
            return {
                "which": "people",
                "type": "request",
                "name": name,
                "id": target_id,
            }

        if name == "goto":
            values = [float(v) for v in parts[3:]]

            if len(values) != 3:
                raise ValueError(
                    "People goto format: people goto x y z"
                )

            return {
                "which": "people",
                "type": "command",
                "name": "goto",
                "command": np.array(values),
                "id": target_id,
            }

        if name == "velocity":
            values = [float(v) for v in parts[3:]]

            if len(values) != 3:
                raise ValueError(
                    "People velocity format: people velocity vx vy vz"
                )

            return {
                "which": "people",
                "type": "command",
                "name": "velocity",
                "command": np.array(values),
                "id": target_id,
            }

    raise ValueError(f"Invalid request: {request}")

def test_example():
    # ==========================
    # Demo Tests
    # ==========================
    tests = [
        "mission hover 1",
        "mission follow 3",
        "drone state",
        "drone pose",
        "drone twist",
        "drone hover",
        "drone goto 1 2 3",
        "drone goto 1 2 3 90",
        "people 1 state",
        "people 1 pose", #  "velocity",
        "people 1 twist",
        "people 1 goto 4 5 6",
        "people 1 velocity 0.5 0.0 -0.2",
    ]

    for req in tests:
        print("=" * 60)
        print("Input :", req)
        print("Output:", parse_request(req))


    # ==========================
    # Interactive Demo
    # ==========================
    print("\nInteractive mode (type 'quit' to exit)\n")

    while True:
        req = input("> ")

        if req.lower() in {"quit", "exit"}:
            break

        try:
            result = parse_request(req)
            print(result)
        except Exception as e:
            print("ERROR:", e)
        

def mission_target(world: Drone, mission: dict[str, Any]) -> Any:
    """Return the live target person for a mission config."""
    target_id = str(mission["person_id"])
    people_by_id = {getattr(person, "id", f"person_{idx}"): person for idx, person in enumerate(world.people)}
    if target_id in people_by_id:
        return people_by_id[target_id]
    if target_id.startswith("person_") and target_id.split("_", 1)[1].isdigit():
        index = int(target_id.split("_", 1)[1])
        if 0 <= index < len(world.people):
            return world.people[index]
        if 1 <= index <= len(world.people):
            return world.people[index - 1]
    raise ValueError(f"Unknown target person id: {target_id}")


def realtime_idle_step(world: Drone, dt: float) -> None:
    """Advance the world while waiting for the first mission."""
    for person in world.people:
        person.update(dt, world.rng, world.world_size)
    if world.scene is not None and hasattr(world.scene, "step"):
        world.scene.step()
    frame = world.render()
    if frame is not None:
        world.frames.append(frame)
    world.time += dt


def build_realtime_condition(
    world: Drone,
    mission: dict[str, Any],
    horizon: int,
    planning_duration: float,
    safety_margin: float,
) -> dict[str, np.ndarray]:
    """Build Flow Matching condition arrays from live drone, people, and obstacle state."""
    target = mission_target(world, mission)
    mission["person_id"] = getattr(target, "id", mission["person_id"])
    current = _full_drone_state(world)
    drone_position = current[:3]
    drone_velocity = current[7:10]
    hover_goal = compute_hover_goal(
        target.position,
        approach_position=drone_position,
        obstacles=world.buildings,
        d_s=float(mission.get("safe_distance", 1.0)),
        hover_height=1.5,
        margin=safety_margin,
        world_size=world.world_size,
    )
    times = np.linspace(0.0, planning_duration, horizon, dtype=np.float32)
    people_predictions = predict_people_positions(world.people, times)
    people_states = np.stack([np.concatenate([person.position, person.velocity]).astype(np.float32) for person in world.people])
    buildings = np.stack([np.concatenate([building.center, building.size]).astype(np.float32) for building in world.buildings])
    drone_start = np.zeros((1, 6), dtype=np.float32)
    drone_start[0, :3] = drone_position
    drone_start[0, 3:6] = drone_velocity
    target_index = next((idx for idx, person in enumerate(world.people) if person is target), -1)
    if target_index < 0:
        raise ValueError(f"Target person is not in world.people: {getattr(target, 'id', target)}")
    hover = np.array([[hover_goal.position[0], hover_goal.position[1], hover_goal.position[2], hover_goal.yaw]], dtype=np.float32)
    return {
        "drone_start": drone_start,
        "hover_goal": hover,
        "target_person_state": people_states[target_index][None, :],
        "people_states": people_states[None, :, :],
        "future_people_predictions": people_predictions[None, :, :, :],
        "building_geometry": buildings[None, :, :],
        "command_embedding": np.array([[1.0, 0.0]], dtype=np.float32),
    }


def plan_realtime_candidate(
    world: Drone,
    config: SampleServingConfig,
    model: FlowTransformer,
    device: torch.device,
    condition_np: dict[str, np.ndarray],
    predicted_people: np.ndarray,
    previous_refined: RefinedTrajectory | None,
    horizon: int,
) -> tuple[RefinedTrajectory, np.ndarray, bool, RefinedTrajectory]:
    """Sample, refine, safety-check, and fallback one realtime trajectory candidate."""
    condition = tensors_from_condition(condition_np, device)
    noise = torch.randn(1, horizon, 6, device=device)
    sampled = sample(model, noise, condition, steps=config.sample_steps, method=config.sample_method)[0].detach().cpu().numpy()
    sampled = anchor_trajectory(sampled, condition_np["drone_start"][0], condition_np["hover_goal"][0])
    refined = refine_candidate(sampled, horizon, config)
    safe = np.isfinite(refined.position_refs).all() and not bool(
        trajectory_collision_mask(
            refined.position_refs,
            world.buildings,
            predicted_people,
            ExpertGeneratorConfig(world_size=tuple(float(v) for v in world.world_size), safety_margin=config.safety_margin),
        ).any()
    )
    if safe:
        return refined, sampled, False, refined

    with contextlib.suppress(Exception):
        expert = generate_expert_trajectory(
            start=condition_np["drone_start"][0],
            hover_goal=condition_np["hover_goal"][0, :3],
            buildings=world.buildings,
            people=world.people,
            config=ExpertGeneratorConfig(
                world_size=tuple(float(v) for v in world.world_size),
                horizon=horizon,
                total_duration=5.0,
                safety_margin=config.safety_margin,
                max_velocity=config.max_velocity,
                max_acceleration=config.max_acceleration,
            ),
        )
        refined = refine_candidate(expert.as_state_trajectory(), horizon, config)
        return refined, expert.as_state_trajectory(), True, refined

    if previous_refined is not None:
        return previous_refined, np.concatenate([previous_refined.position_refs, previous_refined.velocity_refs], axis=1), True, previous_refined

    hover = condition_np["drone_start"][0].copy()
    hover[3:6] = 0.0
    sampled = np.repeat(hover[None, :], horizon, axis=0).astype(np.float32)
    refined = refine_candidate(sampled, horizon, config)
    return refined, sampled, True, refined


def full_reference_from_refined(
    refined: RefinedTrajectory,
    controller: Any,
    quad: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert refined waypoints to a full-pose MPC reference on controller dt."""
    refined_time = np.asarray(refined.time, dtype=np.float32).reshape(-1)
    if refined_time.shape[0] < 2:
        raise ValueError("refined trajectory must contain at least two time samples")
    dt = float(getattr(controller, "dt", 0.05))
    t_end = float(refined_time[-1] - refined_time[0])
    t_ref = np.arange(int(np.ceil(t_end / dt)) + 1, dtype=np.float32) * dt
    source_time = refined_time - refined_time[0]
    position_refs = _interp_time_series(source_time, refined.position_refs, t_ref)
    velocity_refs = _interp_time_series(source_time, refined.velocity_refs, t_ref)
    acceleration_refs = _interp_time_series(source_time, refined.acceleration_refs, t_ref)
    jerk_refs = np.gradient(acceleration_refs, dt, axis=0).astype(np.float32)
    traj_derivatives = np.stack([position_refs.T, velocity_refs.T, acceleration_refs.T, jerk_refs.T], axis=0)
    yaw_derivatives = np.zeros((2, position_refs.shape[0]), dtype=np.float32)
    ref_traj, _, reference_u = traj_utils.minimum_snap_trajectory_generator(
        traj_derivatives,
        yaw_derivatives,
        t_ref,
        quad,
        None,
        False,
        adjust_pos=False,
    )
    return ref_traj.astype(np.float32), np.asarray(reference_u, dtype=np.float32)


def execute_realtime_segment(
    world: Drone,
    controller: Any,
    ref_traj: np.ndarray,
    replan_interval: float,
) -> dict[str, float]:
    """Execute the first short reference segment with full-pose MPC."""
    dt = float(getattr(controller, "dt", 0.05))
    steps = min(max(1, int(np.ceil(replan_interval / dt))), len(ref_traj))
    tracking_error = 0.0
    for idx in range(steps):
        state_goal = fullpose_goal_window(ref_traj, idx, int(controller.N))
        current = _full_drone_state(world)
        thrust = controller.run_optimization(initial_state=current, goal=state_goal, mode="traj")[:4]
        rpms = world.thrusts_to_rpms(thrust)
        sim_dt = float(getattr(world.config, "dt", dt))
        sim_steps = max(1, int(round(dt / max(sim_dt, 1e-6))))
        for _ in range(sim_steps):
            for person in world.people:
                person.update(sim_dt, world.rng, world.world_size)
            if world.scene is not None and hasattr(world.scene, "step"):
                if hasattr(world.drone, "set_propellers_rpm"):
                    world.drone.set_propellers_rpm(rpms)
                elif hasattr(world, "set_propellers_rpm"):
                    world.set_propellers_rpm(rpms)
                world.scene.step()
            world.time += sim_dt
        current = _full_drone_state(world)
        tracking_error = float(np.linalg.norm(current[:3] - ref_traj[idx, :3]))
        frame = world.render()
        if frame is not None:
            world.frames.append(frame)
    return {"tracking_error": tracking_error}


def fullpose_goal_window(ref_traj: np.ndarray, start: int, horizon: int) -> np.ndarray:
    """Return a padded [pos, quat, vel, omega] goal window for full-pose MPC."""
    end = min(start + horizon + 1, len(ref_traj))
    window = np.asarray(ref_traj[start:end], dtype=np.float32)
    if len(window) < horizon + 1:
        pad = np.repeat(window[-1][None, :], horizon + 1 - len(window), axis=0)
        pad[:, 7:13] = 0.0
        window = np.vstack([window, pad])
    return window[:, :13]


def plan_realtime_reference(
    request_id: int,
    mission: dict[str, Any],
    config: SampleServingConfig,
    model: FlowTransformer,
    device: torch.device,
    condition_np: dict[str, np.ndarray],
    predicted_people: np.ndarray,
    buildings: tuple[Any, ...],
    people_snapshot: list[dict[str, Any]],
    world_size: np.ndarray,
    previous_refined: RefinedTrajectory | None,
    horizon: int,
    controller: Any,
    quad: Any,
) -> dict[str, Any]:
    """Plan one realtime full-pose reference from immutable world snapshots."""
    start_time = time.monotonic()
    refined, sampled, used_fallback, latest_refined = plan_realtime_candidate_from_snapshot(
        config=config,
        model=model,
        device=device,
        condition_np=condition_np,
        predicted_people=predicted_people,
        buildings=buildings,
        people_snapshot=people_snapshot,
        world_size=world_size,
        previous_refined=previous_refined,
        horizon=horizon,
    )
    ref_traj, _ = full_reference_from_refined(refined, controller, quad)
    collision_free = not bool(
        trajectory_collision_mask(
            refined.position_refs,
            buildings,
            predicted_people,
            ExpertGeneratorConfig(world_size=tuple(float(v) for v in world_size), safety_margin=config.safety_margin),
        ).any()
    )
    return {
        "request_id": request_id,
        "mission": mission["name"],
        "target_person": mission["person_id"],
        "ref_traj": ref_traj,
        "refined": latest_refined,
        "sampled": sampled,
        "collision_free": collision_free,
        "used_fallback": used_fallback,
        "planning_time": time.monotonic() - start_time,
    }


def plan_realtime_candidate_from_snapshot(
    config: SampleServingConfig,
    model: FlowTransformer,
    device: torch.device,
    condition_np: dict[str, np.ndarray],
    predicted_people: np.ndarray,
    buildings: tuple[Any, ...],
    people_snapshot: list[dict[str, Any]],
    world_size: np.ndarray,
    previous_refined: RefinedTrajectory | None,
    horizon: int,
) -> tuple[RefinedTrajectory, np.ndarray, bool, RefinedTrajectory]:
    """Sample and refine a trajectory without reading live Genesis state."""
    condition = tensors_from_condition(condition_np, device)
    noise = torch.randn(1, horizon, 6, device=device)
    with torch.inference_mode():
        sampled = sample(model, noise, condition, steps=config.sample_steps, method=config.sample_method)[0].detach().cpu().numpy()
    sampled = anchor_trajectory(sampled, condition_np["drone_start"][0], condition_np["hover_goal"][0])
    refined = refine_candidate(sampled, horizon, config)
    safe = np.isfinite(refined.position_refs).all() and not bool(
        trajectory_collision_mask(
            refined.position_refs,
            buildings,
            predicted_people,
            ExpertGeneratorConfig(world_size=tuple(float(v) for v in world_size), safety_margin=config.safety_margin),
        ).any()
    )
    if safe:
        return refined, sampled, False, refined

    snapshot_people = [
        SimpleNamespace(
            position=np.asarray(person["position"], dtype=np.float32),
            velocity=np.asarray(person["velocity"], dtype=np.float32),
            radius=float(person.get("radius", 0.3)),
        )
        for person in people_snapshot
    ]
    with contextlib.suppress(Exception):
        expert = generate_expert_trajectory(
            start=condition_np["drone_start"][0],
            hover_goal=condition_np["hover_goal"][0, :3],
            buildings=buildings,
            people=snapshot_people,
            config=ExpertGeneratorConfig(
                world_size=tuple(float(v) for v in world_size),
                horizon=horizon,
                total_duration=5.0,
                safety_margin=config.safety_margin,
                max_velocity=config.max_velocity,
                max_acceleration=config.max_acceleration,
            ),
        )
        refined = refine_candidate(expert.as_state_trajectory(), horizon, config)
        return refined, expert.as_state_trajectory(), True, refined

    if previous_refined is not None:
        sampled_previous = np.concatenate([previous_refined.position_refs, previous_refined.velocity_refs], axis=1)
        return previous_refined, sampled_previous, True, previous_refined

    hover = condition_np["drone_start"][0].copy()
    hover[3:6] = 0.0
    sampled_hover = np.repeat(hover[None, :], horizon, axis=0).astype(np.float32)
    refined = refine_candidate(sampled_hover, horizon, config)
    return refined, sampled_hover, True, refined

def realtime_hover_step(world: Drone, controller: Any, dt: float) -> None:
    """Hold approximately in place while waiting for a planned reference."""
    del controller
    hover_rpm = getattr(world, "_hover_rpm_from_config", lambda: 0.0)()
    rpms = np.full(4, float(hover_rpm), dtype=np.float32)
    step_world_with_command(world, rpms, dt)


def step_world_with_command(world: Drone, rpms: np.ndarray, dt: float) -> None:
    """Advance people and Genesis for one controller step using a motor command."""
    sim_dt = float(getattr(world.config, "dt", dt))
    sim_steps = max(1, int(round(float(dt) / max(sim_dt, 1e-6))))
    for _ in range(sim_steps):
        for person in world.people:
            person.update(sim_dt, world.rng, world.world_size)
        if world.scene is not None and hasattr(world.scene, "step"):
            if hasattr(world.drone, "set_propellers_rpm"):
                world.drone.set_propellers_rpm(rpms)
            elif hasattr(world, "set_propellers_rpm"):
                world.set_propellers_rpm(rpms)
            world.scene.step()
        world.time += sim_dt
    frame = world.render()
    if frame is not None:
        world.frames.append(frame)


def apply_mission(world: Drone, request_dic: dict[str, Any]) -> None:
    """ if the request is a mission for a drone """
    people_by_id = {getattr(person, "id", f"person_{idx}"): person for idx, person in enumerate(world.people)}
    target_id = request_dic["id"]
    person=people_by_id[f"person_{target_id}"]
    
    mission_name = request_dic["name"]
    if mission_name in {"hover", "circle", "follow"}:
        target = person.get_position()
    elif mission_name in {"land"}:
        target = world.get_position
        target[2] = 0.1  # land the drone at current position 
    elif mission_name in {"back"}:
        target = DEFAULT_DRONE_POSITION
    else:
        print(f"Please define missipn -- {mission_name} -- first!")

    traj_X = mission_name
    target = target_id
    person.color = (1.0, 1.0, 1.0, 1.0)


def apply_request(world: Drone, request_dic: dict[str, Any]) -> None:
    """Apply the request for drone or people."""

    traj_X = None
    target = None
    if request_dic["which"] == "drone":
        if request_dic["type"] == "request":
            lowered = request_dic["name"].lower()
            if lowered in {"state"}:
                print(
                    f"position=        {world.get_position().round(3).tolist()} \n"
                    f"euler=           {world.get_euler().round(3).tolist()}    \n"
                    f"velocity=        {world.get_velocity().round(3).tolist()} \n"
                    f"angular_velocity={world.get_angular_velocity().round(3).tolist()} \n"
                    )
            elif lowered in {"pose"}:
                print(f"position={world.get_position().round(3).tolist()}\n"
                    f"euler=   {world.get_euler().round(3).tolist()}  \n"
                    )
            elif lowered in {"twist"}:
                print(
                    f"velocity=        {world.get_velocity().round(3).tolist()} \n"
                    f"angular_velocity={world.get_angular_velocity().round(3).tolist()}\n"
                    )
            else:
                print(f"Please define request -- {lowered} -- first!")

        elif request_dic["type"] == "command":
            lowered = request_dic["name"].lower()
            if lowered in {"hover", "level"}:
                world.return_to_hover(
                            reset_velocity=True,
                            reset_angular_velocity=True,
                            reset_pose=True,
                        )
                traj_X = "hover"
                target = world.get_hover_pose()
                print(f"Returning to hover at {target}")
            elif lowered in {"goto"}:
                # otherwise, the command received, the drone conduct the received mission
                target = request_dic["command"]
                traj_X = "p2p"
                print(f"Go to: position -- {target[:3]} and yaw -- {target[3]}")
            
            else:
                print(f"Please define command -- {lowered} -- first!")
        else:
            print(f"Please define the request type -- {lowered} -- first!")

    elif request_dic["which"] == "people":
        people_by_id = {getattr(person, "id", f"person_{idx}"): person for idx, person in enumerate(world.people)}
        if request_dic["type"] == "request":
            lowered = request_dic["name"].lower()
            if lowered in {"state", "position", "velocity"}:
                for person in world.people:
                    print(
                        f"person: {person.name_id}\n"
                        f"position=        {person.get_position().round(3).tolist()} \n"
                        f"velocity=        {person.get_velocity().round(3).tolist()} \n"
                        )
            else:
                print(f"Please define request {lowered} first!")
        elif request_dic["type"] == "command":
            lowered = request_dic["name"].lower()
            target = request_dic["command"]
            target_id = request_dic["id"]
            person=people_by_id[f"person_{target_id}"]
            if lowered in {"goto"}:
                person.set_position(target)
                print(f"Person {target_id}: go to: position -- {target[:3]}")
                
            elif lowered in {"velocity"}:
                person.set_velocity(target)
                print(f"Person {target_id}: set velocity -- {target[:3]}")
        else:
            print(f"Please define the command type -- {lowered} -- first!")

    return traj_X, target


def conduct_drone_mission(
    world: Drone,
    config: SampleServingConfig,
    model: FlowTransformer,
    device: torch.device,
    horizon: int,
    mission_config: dict[str, Any],
    mission_index: int,
) -> dict[str, Any]:
    """Plan and roll out one assigned drone mission."""
    condition_np = build_scene_condition(world, horizon, config.safety_margin)
    predicted_people = condition_np["future_people_predictions"][0]
    mission_name = str(mission_config["name"])
    used_expert_fallback = False

    if mission_name in {"hover", "follow"}:
        condition = tensors_from_condition(condition_np, device)
        noise = torch.randn(1, horizon, 6, device=device)
        sampled = sample(model, noise, condition, steps=config.sample_steps, method=config.sample_method)[0].detach().cpu().numpy()
        sampled = anchor_trajectory(sampled, condition_np["drone_start"][0], condition_np["hover_goal"][0])
        refined = refine_candidate(sampled, horizon, config)
        sampled_collision_free = not bool(
            trajectory_collision_mask(
                refined.position_refs,
                world.buildings,
                predicted_people,
                ExpertGeneratorConfig(world_size=tuple(float(v) for v in world.world_size), safety_margin=config.safety_margin),
            ).any()
        )
        if not sampled_collision_free or not np.isfinite(refined.position_refs).all():
            used_expert_fallback = True
            expert = generate_expert_trajectory(
                start=condition_np["drone_start"][0],
                hover_goal=condition_np["hover_goal"][0, :3],
                buildings=world.buildings,
                people=world.people,
                config=ExpertGeneratorConfig(
                    world_size=tuple(float(v) for v in world.world_size),
                    horizon=horizon,
                    safety_margin=config.safety_margin,
                    max_velocity=config.max_velocity,
                    max_acceleration=config.max_acceleration,
                ),
            )
            sampled = expert.as_state_trajectory()
            refined = refine_candidate(sampled, horizon, config)
    else:
        sampled = mission_waypoints(world, mission_config, condition_np["hover_goal"][0, :3], horizon)
        sampled_collision_free = True
        refined = refine_candidate(sampled, horizon, config)

    collision_free = not bool(
        trajectory_collision_mask(
            refined.position_refs,
            world.buildings,
            predicted_people,
            ExpertGeneratorConfig(world_size=tuple(float(v) for v in world.world_size), safety_margin=config.safety_margin),
        ).any()
    )
    tracking = rollout_flight(world, refined, config)
    plot_path = plot_sample(
        world,
        condition_np,
        sampled,
        refined,
        predicted_people,
        _mission_output_path(config.output, mission_index),
        used_expert_fallback,
        tracked_positions=tracking.states[:, :3],
        show=False,
    )
    video_path = world.save_video(_mission_output_path(config.video, mission_index))
    return {
        "mission": mission_name,
        "target_person": mission_config.get("person_id"),
        "plot": str(plot_path),
        "video": str(video_path),
        "collision_free": collision_free,
        "used_expert_fallback": used_expert_fallback,
        "sampled_collision_free": sampled_collision_free,
        "mpc_max_tracking_error": float(tracking.tracking_error.max()),
    }


def mission_waypoints(world: Drone, mission_config: dict[str, Any], hover_goal: np.ndarray, horizon: int) -> np.ndarray:
    """Build simple state waypoints for non-serving mission types."""
    start = position_state_from_world(world)
    start_position = start[:3]
    mission_name = str(mission_config["name"])
    if mission_name == "land":
        goal = start_position.copy()
        goal[2] = 0.15
        positions = np.linspace(start_position, goal, horizon, dtype=np.float32)
    elif mission_name == "back":
        goal = np.asarray(mission_config.get("target_position", DEFAULT_DRONE_POSITION), dtype=np.float32)
        positions = np.linspace(start_position, goal, horizon, dtype=np.float32)
    elif mission_name == "circle":
        center = hover_goal.copy()
        radius = float(mission_config.get("radius", 1.2))
        angles = np.linspace(0.0, 2.0 * np.pi, horizon, dtype=np.float32)
        positions = np.empty((horizon, 3), dtype=np.float32)
        positions[:, 0] = center[0] + radius * np.cos(angles)
        positions[:, 1] = center[1] + radius * np.sin(angles)
        positions[:, 2] = center[2]
        blend = np.linspace(0.0, 1.0, min(10, horizon), dtype=np.float32)[:, None]
        positions[: len(blend)] = (1.0 - blend) * start_position + blend * positions[: len(blend)]
    else:
        positions = np.linspace(start_position, hover_goal, horizon, dtype=np.float32)
    velocities = np.gradient(positions, axis=0).astype(np.float32)
    return np.concatenate([positions, velocities], axis=1).astype(np.float32)


def _mission_output_path(path: str | Path, mission_index: int) -> str:
    """Return an output path with a mission index suffix."""
    output = Path(path)
    return str(output.with_name(f"{output.stem}_mission_{mission_index:02d}{output.suffix}"))


def mission_assign(person_id: str | int | None, mission: str | None) -> dict[str, Any]:
    """Assign random people motion and an optional drone mission target."""
    allowed_missions = {"hover", "circle", "follow", "land", "back"}
    mission_name = None if mission is None else str(mission).lower()
    if mission_name is not None and mission_name not in allowed_missions:
        raise ValueError(f"mission must be one of {sorted(allowed_missions)} or None")

    rng = np.random.default_rng()
    buildings = make_buildings()
    world_xy = np.asarray(WORLD_SIZE[:2], dtype=np.float32)
    low = -world_xy / 2.0 + 0.5
    high = world_xy / 2.0 - 0.5

    def sample_point() -> np.ndarray:
        """Sample one person waypoint outside the simple building footprints."""
        for _ in range(64):
            point = np.array([rng.uniform(low[0], high[0]), rng.uniform(low[1], high[1]), 0.85], dtype=np.float32)
            if not trajectory_collision_mask(point[None, :], buildings=buildings).any():
                return point
        return np.array([rng.uniform(low[0], high[0]), rng.uniform(low[1], high[1]), 0.85], dtype=np.float32)

    people = [
        {
            "id": f"person_{idx}",
            "behavior": behavior,
            "velocity": np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), 0.0], dtype=np.float32),
            "routine_points": np.stack([sample_point() for _ in range(4)]).astype(np.float32) if behavior == "routine" else None,
            "avoid_buildings": True,
            "avoid_people": True,
        }
        for idx in range(5)
        for behavior in ("random_walk" if rng.random() < 0.5 else "routine",)
    ]

    assignment: dict[str, Any] = {"people": people, "mission": None}
    if mission_name is None:
        return assignment

    if person_id is None:
        raise ValueError("person id is required for drone missions")
    if isinstance(person_id, int) or str(person_id).isdigit():
        target_index = int(person_id)
        if target_index < 0 or target_index >= len(people):
            raise ValueError(f"person id must be between 1 and {len(people)}")
        target_id = f"person_{target_index}"
    else:
        target_id = str(person_id)
        if target_id not in {person["id"] for person in people}:
            raise ValueError(f"Unknown target person id: {target_id}")
    mission_config: dict[str, Any] = {"name": mission_name, "person_id": target_id}
    if mission_name == "hover":
        mission_config.update({"hold_current": True})
    elif mission_name == "circle":
        mission_config.update({"center_person_id": target_id, "radius": 1.2, "angular_speed": 0.5})
    elif mission_name == "follow":
        mission_config.update({"target_person_id": target_id, "safe_distance": 1.0, "height_mode": "person_height"})
    elif mission_name == "land":
        mission_config.update({"land_in_place": True})
    elif mission_name == "back":
        mission_config.update({"target_position": DEFAULT_DRONE_POSITION.copy()})
    assignment["mission"] = mission_config
    return assignment


def load_model(checkpoint: str | Path, device: torch.device, fallback_horizon: int) -> tuple[FlowTransformer, TrainServingConfig, int]:
    """Load a trained serving Flow Transformer checkpoint."""
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


def make_world(config: SampleServingConfig, train_config: TrainServingConfig, rng: np.random.Generator) -> Drone:
    """Create a Drone serving world with attached planning metadata."""
    del train_config
    world_size = np.asarray(WORLD_SIZE, dtype=np.float32)
    buildings = make_buildings()
    people = setup_people_initial_position(
        DEFAULT_DRONE_POSITION.copy(),
        world_size,
        count=5,
        buildings=buildings,
        seed=config.seed,
    )
    assign_ids(buildings, "building")
    assign_ids(people, "person")
    world = Drone(
        config=config.genesis_config,
        scene_setup=lambda gs, scene: _add_scene_entities(gs, scene, buildings, people),
        show_viewer=not config.headless,
    )
    world.world_size = world_size
    world.rng = rng
    world.buildings = buildings
    world.people = people
    world.frames = []
    world.time = 0.0
    target_index = min(2, len(world.people) - 1)
    world.people[target_index].command = "coffee"
    world.people[target_index].color = (1.0, 0.05, 0.05, 1.0)
    return world


def final_hover_state(refined: RefinedTrajectory) -> np.ndarray:
    """Return a zero-velocity hover state at the final refined waypoint."""
    hover_state = np.zeros(6, dtype=np.float32)
    hover_state[:3] = refined.position_refs[-1]
    return hover_state


def hold_viewer_open(world: Drone, hover_state: np.ndarray | None = None) -> None:
    """Keep the Genesis viewer alive and hold the drone at the final hover state."""
    try:
        while True:
            if hover_state is not None:
                set_drone_position_state(world, hover_state)
            if world.scene is not None and hasattr(world.scene, "step"):
                world.scene.step()
            if hover_state is not None:
                set_drone_position_state(world, hover_state)
            frame = world.render()
            if frame is not None:
                world.frames.append(frame)
            time.sleep(max(float(world.config.dt), 0.02))
    except KeyboardInterrupt:
        pass


def build_scene_condition(world: Drone, horizon: int, safety_margin: float) -> dict[str, np.ndarray]:
    """Build Milestone 5 condition arrays from the current serving world."""
    selected = select_coffee_target(world.people)
    if world.drone is None:
        raise RuntimeError("World reset did not create a drone.")
    drone_position = world.get_position()
    hover_goal = compute_hover_goal(
        selected.person.position,
        approach_position=drone_position,
        obstacles=world.buildings,
        d_s=1.0,
        hover_height=1.5,
        margin=safety_margin,
        world_size=world.world_size,
    )
    times = np.linspace(0.0, 1.0, horizon, dtype=np.float32)
    people_predictions = predict_people_positions(world.people, times)
    people_states = np.stack([np.concatenate([person.position, person.velocity]).astype(np.float32) for person in world.people])
    buildings = np.stack([np.concatenate([building.center, building.size]).astype(np.float32) for building in world.buildings])
    drone_start = np.zeros((1, 6), dtype=np.float32)
    drone_start[0, :3] = drone_position
    hover = np.array([[hover_goal.position[0], hover_goal.position[1], hover_goal.position[2], hover_goal.yaw]], dtype=np.float32)
    return {
        "drone_start": drone_start,
        "hover_goal": hover,
        "target_person_state": people_states[selected.index][None, :],
        "people_states": people_states[None, :, :],
        "future_people_predictions": people_predictions[None, :, :, :],
        "building_geometry": buildings[None, :, :],
        "command_embedding": np.array([[1.0, 0.0]], dtype=np.float32),
    }


def tensors_from_condition(condition: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Convert condition arrays to torch tensors."""
    keys = ["drone_start", "hover_goal", "target_person_state", "future_people_predictions", "building_geometry", "command_embedding"]
    return {key: torch.as_tensor(condition[key], dtype=torch.float32, device=device) for key in keys}


def anchor_trajectory(sampled: np.ndarray, start: np.ndarray, hover_goal: np.ndarray) -> np.ndarray:
    """Force sampled trajectory endpoints to the current task endpoints."""
    trajectory = np.asarray(sampled, dtype=np.float32).copy()
    trajectory[0] = start
    trajectory[-1, :3] = hover_goal[:3]
    trajectory[-1, 3:6] = 0.0
    return trajectory


def refine_candidate(sampled: np.ndarray, horizon: int, config: SampleServingConfig) -> RefinedTrajectory:
    """Smooth sampled waypoints into position, velocity, and acceleration references."""
    return refine_waypoints(
        sampled,
        total_duration=5.0,
        num_samples=horizon,
        method="quintic",
        max_velocity=config.max_velocity,
        max_acceleration=config.max_acceleration,
    )


def rollout_flight(world: Drone, refined: RefinedTrajectory, config: SampleServingConfig) -> DroneMPCResult:
    """Track refined Flow Matching references with MPC and capture Genesis frames."""
    if world.drone is None:
        raise RuntimeError("World reset did not create a drone.")
    if len(refined.time) > 1:
        dt = float(np.mean(np.diff(refined.time)))
    else:
        dt = float(world.config.dt)
    reference = np.concatenate([refined.position_refs, refined.velocity_refs], axis=1).astype(np.float32)
    if config.use_mpc:
        if config.mpc_position_control:
            controller = DroneMPCController(
                DroneMPCConfig(
                    dt=dt,
                    horizon=config.mpc_horizon,
                    max_velocity=config.max_velocity,
                    max_acceleration=config.max_acceleration,
                )
            )
            tracking = controller.track(reference, initial_state=position_state_from_world(world))
            states = tracking.states
        elif config.mpc_fullpose_control:
            controller = DroneMPCController(
                DroneMPCConfig(
                    dt=dt,
                    horizon=config.mpc_horizon,
                    max_velocity=config.max_velocity,
                    max_acceleration=config.max_acceleration,
                )
            )
            return drone_track(refined, controller.controller, controller.quad, world, world)
        else:
            raise RuntimeError("Define a control mode and controller.")
    else:
        tracking = DroneMPCResult(
            states=reference,
            controls=refined.acceleration_refs,
            references=reference,
            tracking_error=np.zeros(len(reference), dtype=np.float32),
        )
        states = reference
    # for state in states:
    #     set_drone_position_state(world, state)
    #     for person in world.people:
    #         person.update(dt, world.rng, world.world_size)
    #     if world.scene is not None and hasattr(world.scene, "step"):
    #         world.scene.step()
    #     frame = world.render()
    #     if frame is not None:
    #         world.frames.append(frame)
    #     world.time += dt
    return tracking

def drone_track(
    refined: RefinedTrajectory,
    controller: Any,
    quad: Any,
    drone: Drone,
    world: Drone,
) -> DroneMPCResult:
    """Track Flow Matching position/velocity references with the full-pose MPC."""
    refined_time = np.asarray(refined.time, dtype=np.float32).reshape(-1)
    if refined_time.shape[0] < 2:
        raise ValueError("refined trajectory must contain at least two time samples")
    if np.any(np.diff(refined_time) <= 0.0):
        raise ValueError("refined trajectory time samples must be strictly increasing")

    dt = float(getattr(controller, "dt", 0.05))
    t_end = float(refined_time[-1] - refined_time[0])
    sample_count = int(np.ceil(t_end / dt)) + 1
    t_ref = np.arange(sample_count, dtype=np.float32) * dt
    source_time = refined_time - refined_time[0]
    position_refs = _interp_time_series(source_time, refined.position_refs, t_ref)
    velocity_refs = _interp_time_series(source_time, refined.velocity_refs, t_ref)
    acceleration_refs = _interp_time_series(source_time, refined.acceleration_refs, t_ref)
    jerk_refs = np.gradient(acceleration_refs, dt, axis=0).astype(np.float32)
    traj_derivatives = np.stack([position_refs.T, velocity_refs.T, acceleration_refs.T, jerk_refs.T], axis=0)
    yaw_derivatives = np.zeros((2, position_refs.shape[0]), dtype=np.float32)
    ref_traj, t_ref, reference_u = traj_utils.minimum_snap_trajectory_generator(
        traj_derivatives,
        yaw_derivatives,
        t_ref,
        quad,
        None,
        False,
        adjust_pos=False,
    )
    pos_ref   = np.array(ref_traj[:, 0:3])
    quat_ref  = np.array(ref_traj[:, 3:7])
    vel_ref   = np.array(ref_traj[:, 7:10])
    omega_ref = np.array(ref_traj[:, 10:13])
    n_steps = len(pos_ref)
    N_drone = controller.N
    states = np.zeros((n_steps, 13), dtype=np.float32)
    controls = np.asarray(reference_u, dtype=np.float32)
    tracking_error = np.zeros(n_steps, dtype=np.float32)
    for i in range(n_steps):
        # Extract reference window  
        i_end = min(i + N_drone + 1, n_steps)
        ref_pos = pos_ref[i:i_end, :]
        ref_vel = vel_ref[i:i_end, :]
        ref_quat = quat_ref[i:i_end, :]
        ref_omega = omega_ref[i:i_end, :]

        n_available = i_end - i
        # This creates consistent final condition: constant position, zero velocity
        if n_available < N_drone + 1:
            pad_len = N_drone + 1 - n_available
            ref_pos = np.vstack([ref_pos, np.tile(pos_ref[-1], (pad_len, 1))])
            ref_vel = np.vstack([ref_vel, np.zeros((pad_len, 3))])  # ZERO velocity for padding!
            ref_quat = np.vstack([ref_quat, np.tile([1, 0, 0, 0], (pad_len, 1))])
            ref_omega = np.vstack([ref_omega, np.zeros((pad_len, 3))])
            

        state_goal = np.hstack([ref_pos, ref_quat, ref_vel, ref_omega])

        current = _full_drone_state(drone)
        thrust = controller.run_optimization(initial_state=current, goal=state_goal, mode='traj')[:4]
        rpms = drone.thrusts_to_rpms(thrust)
        
        sim_dt = float(getattr(world.config, "dt", dt))
        sim_steps = max(1, int(round(dt / max(sim_dt, 1e-6))))
        for _ in range(sim_steps):
            for person in world.people:
                person.update(sim_dt, world.rng, world.world_size)
            if world.scene is not None and hasattr(world.scene, "step"):
                drone.drone.set_propellers_rpm(rpms)
                world.scene.step()
            world.time += sim_dt

        states[i] = _full_drone_state(drone)
        tracking_error[i] = float(np.linalg.norm(states[i, :3] - pos_ref[i]))
        controls[i] = thrust
        frame = world.render()
        if frame is not None:
            world.frames.append(frame)
    return DroneMPCResult(states=states, controls=controls, references=ref_traj.astype(np.float32), tracking_error=tracking_error)

def _interp_time_series(source_time: np.ndarray, values: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """Interpolate vector references from refined timestamps to MPC timestamps."""
    values = np.asarray(values, dtype=np.float32)
    return np.stack(
        [np.interp(target_time, source_time, values[:, axis]) for axis in range(values.shape[1])],
        axis=1,
    ).astype(np.float32)


def _full_drone_state(drone: Drone) -> np.ndarray:
    """Return a full [pos, quat, vel, omega] state for full-pose MPC."""
    if hasattr(drone, "get_state"):
        try:
            return np.asarray(drone.get_state(), dtype=np.float32).reshape(-1)[:13]
        except AttributeError:
            pass
    if hasattr(drone, "get_full_state"):
        try:
            return np.asarray(drone.get_full_state(), dtype=np.float32).reshape(-1)[:13]
        except AttributeError:
            pass
    position = np.asarray(drone.position, dtype=np.float32)
    velocity = np.asarray(drone.velocity, dtype=np.float32)
    omega = np.asarray(getattr(drone, "body_rates", np.zeros(3)), dtype=np.float32)
    return np.concatenate([position, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), velocity, omega])


def position_state_from_world(world: Drone) -> np.ndarray:
    """Return the current Drone [x, y, z, vx, vy, vz] state."""
    return np.concatenate([world.get_position(), world.get_velocity()]).astype(np.float32)


def set_drone_position_state(world: Drone, state: np.ndarray) -> None:
    """Move the Genesis drone entity from a simplified [p, v] trajectory state."""
    values = np.asarray(state, dtype=np.float32).reshape(-1)
    if values.shape[0] < 3:
        raise ValueError("state must contain at least x, y, z")
    if hasattr(world.drone, "set_pos"):
        world.drone.set_pos(values[:3])
    if values.shape[0] >= 6 and hasattr(world.drone, "set_dofs_velocity"):
        velocity = np.zeros(6, dtype=np.float32)
        velocity[:3] = values[3:6]
        with contextlib.suppress(Exception):
            world.drone.set_dofs_velocity(velocity)


def plot_sample(
    world: Drone,
    condition: dict[str, np.ndarray],
    sampled: np.ndarray,
    refined: RefinedTrajectory,
    predicted_people: np.ndarray,
    output: str | Path,
    used_expert_fallback: bool,
    tracked_positions: np.ndarray | None = None,
    show=True,
) -> Path:
    """Save a 3D plot of the sampled and refined serving trajectory."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for building in world.buildings:
        _plot_box(ax, building.center, building.size)
    for idx, prediction in enumerate(predicted_people):
        color = "tab:red" if idx == min(2, len(predicted_people) - 1) else "tab:orange"
        ax.plot(prediction[:, 0], prediction[:, 1], prediction[:, 2], color=color, alpha=0.35, linewidth=1.0)
    ax.plot(sampled[:, 0], sampled[:, 1], sampled[:, 2], "o--", color="tab:gray", alpha=0.55, label="sampled FM")
    label = "refined fallback" if used_expert_fallback else "refined sampled"
    ax.plot(refined.position_refs[:, 0], refined.position_refs[:, 1], refined.position_refs[:, 2], color="tab:blue", linewidth=2.4, label=label)
    if tracked_positions is not None:
        ax.plot(
            tracked_positions[:, 0],
            tracked_positions[:, 1],
            tracked_positions[:, 2],
            color="tab:cyan",
            linewidth=2.0,
            linestyle=":",
            label="MPC tracked",
        )
    start = condition["drone_start"][0, :3]
    hover = condition["hover_goal"][0, :3]
    ax.scatter(start[0], start[1], start[2], color="tab:green", s=75, label="drone start")
    ax.scatter(hover[0], hover[1], hover[2], color="tab:purple", marker="*", s=140, label="hover goal")
    half_x = float(world.world_size[0]) / 2.0
    half_y = float(world.world_size[1]) / 2.0
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_zlim(0.0, float(world.world_size[2]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_box(ax: plt.Axes, center: np.ndarray, size: np.ndarray) -> None:
    """Draw a wireframe building box."""
    half = size / 2.0
    x = [center[0] - half[0], center[0] + half[0]]
    y = [center[1] - half[1], center[1] + half[1]]
    z = [center[2] - half[2], center[2] + half[2]]
    corners = np.array(
        [
            [x[0], y[0], z[0]],
            [x[1], y[0], z[0]],
            [x[1], y[1], z[0]],
            [x[0], y[1], z[0]],
            [x[0], y[0], z[1]],
            [x[1], y[0], z[1]],
            [x[1], y[1], z[1]],
            [x[0], y[1], z[1]],
        ],
        dtype=np.float32,
    )
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for start, end in edges:
        ax.plot(*zip(corners[start], corners[end]), color="0.35", linewidth=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="outputs/serving_fm/checkpoints/best.pt")
    parser.add_argument("--output", default="outputs/sample_serving_trajectory.png")
    parser.add_argument("--video", default="outputs/sample_serving_trajectory.mp4")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mpc", action="store_true")
    parser.add_argument("--mpc-horizon", type=int, default=12)
    parser.add_argument("--genesis-config", default="configs/task_config.yaml")
    parser.add_argument(
        "--no-keep-open",
        action="store_true",
        help="Close the Genesis viewer after rollout instead of holding hover until Ctrl+C.",
    )
    args = parser.parse_args()
    run_sample(
    # run_missions(
        SampleServingConfig(
            checkpoint=args.ckpt,
            output=args.output,
            video=args.video,
            headless=args.headless,
            sample_steps=args.sample_steps,
            seed=args.seed,
            use_mpc=not args.no_mpc,
            mpc_horizon=args.mpc_horizon,
            genesis_config=args.genesis_config,
            keep_open=not args.no_keep_open,
        )
    )


if __name__ == "__main__":
    main()
