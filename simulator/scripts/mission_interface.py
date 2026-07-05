from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import os

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
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import timeit

from control.mpc_controller import DroneMPCController
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
from planning.path_traj import TrajectorPlanner

import utils.parse_terminal_command as parse_command

DEFAULT_COLOR = (1.0, 0.05, 0.05, 1.0)
TARGET_COLOR  = (1.0, 0.05, 0.05, 1.0)

def update_people_position(people, next_position):
    # this is the funtion to update the target person next postion 
    # people: people = [world.people[person_idx]] -- person_idx starts from 0
    # next_state: the next position of the target person, need to be 3D nuppy array
    if isinstance(people, (list, tuple)):
        for person, next_step in zip(people, next_position):
            person.set_position(next_step)
    else:
        people.set_position(next_position)

def give_drone_cmd(drone, cmd):
    # this is the funtion to update the target person next postion 
    # drone: drone = [world.drone[person_idx]] -- person_idx starts from 0
    # cmd: the r, need to be 4D thrust command 
    if isinstance(drone, (list, tuple)):
        for _drone, command in zip(drone, cmd):
            rpms = _drone.thrusts_to_rpms(command)
            _drone.set_position(rpms)
    else:
       rpms = drone.thrusts_to_rpms(cmd)
       drone.set_propellers_rpm(rpms)



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
    safety_margin: float = 0.3
    map_resolution: float = 0.2
    local_map_radius: float = 15
    max_velocity: float = 3.0
    max_acceleration: float = 5.0
    radius_people: float = 0.2
    radius_drone: float = 0.1
    use_mpc: bool = True
    mpc_position_control: bool = False
    mpc_fullpose_control: bool = True
    mpc_horizon: int = 12
    genesis_config: str = "configs/task_config.yaml"
    keep_open: bool = False


def optimize_traj(get_waypoints, build_map, traj_solver,
                      max_velocity,
                      start_position, target_position, 
                      target_person=None):
    """Build a sampled realtime trajectory reference for MPC execution."""

    build_map(ego_drone=start_position, target_person=target_person) 
    waypoints = get_waypoints(start_position, target_position)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = np.maximum(segment_lengths / max(float(max_velocity), 1e-3), 0.1)
    trajectory = traj_solver(segment_times, waypoints)

    return trajectory, segment_times, waypoints

def _rotation_matrices_to_quaternions(rotations: np.ndarray) -> np.ndarray:
    """Convert body-to-world rotation matrices to normalized [w, x, y, z] quaternions."""
    quaternions = np.empty((rotations.shape[0], 4), dtype=np.float32)
    for idx, rot in enumerate(rotations):
        trace = float(np.trace(rot))
        if trace > 0.0:
            scale = np.sqrt(trace + 1.0) * 2.0
            quaternions[idx] = [
                0.25 * scale,
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
            ]
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quaternions[idx] = [
                (rot[2, 1] - rot[1, 2]) / scale,
                0.25 * scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
            ]
        elif rot[1, 1] > rot[2, 2]:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quaternions[idx] = [
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[0, 1] + rot[1, 0]) / scale,
                0.25 * scale,
                (rot[1, 2] + rot[2, 1]) / scale,
            ]
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quaternions[idx] = [
                (rot[1, 0] - rot[0, 1]) / scale,
                (rot[0, 2] + rot[2, 0]) / scale,
                (rot[1, 2] + rot[2, 1]) / scale,
                0.25 * scale,
            ]
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / np.maximum(norms, 1e-8)

def get_realtime_traj(config, get_waypoints, build_map, traj_solver,
                      start_t, duration, sample_t, max_velocity,
                      start_position, target_position, 
                      start_yaw, end_yaw, yaw_rate=None, target_person=None):
    """Build a sampled realtime trajectory reference for MPC execution."""

    build_map(ego_drone=start_position, target_person=target_person) 
    waypoints = get_waypoints(start_position, target_position)
    segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    segment_times = np.maximum(segment_lengths / max(float(max_velocity), 1e-3), 0.1)
    trajectory = traj_solver(segment_times, waypoints)

    if duration > segment_times[-1]:
        duration = segment_times[-1]
    
    sample_times = torch.linspace(start_t, duration, int((duration- start_t)/sample_t)+1)
    if hasattr(trajectory, "ts") and sample_times.numel() > 0:
        trajectory_end = float(np.asarray(trajectory.ts)[-1])
        if trajectory_end > float(start_t) and float(sample_times[-1]) >= trajectory_end:
            sample_times[-1] = trajectory_end - max(1e-6, abs(trajectory_end) * 1e-6)
    # start = timeit.default_timer()
    pos = torch.as_tensor(trajectory.pos(sample_times), dtype=torch.float32)
    vel = torch.as_tensor(trajectory.vel(sample_times), dtype=torch.float32)
    acc = torch.as_tensor(trajectory.acc(sample_times), dtype=torch.float32)
    jerk = torch.as_tensor(trajectory.jerk(sample_times), dtype=torch.float32)
    sample_count = min(sample_times.shape[0], pos.shape[0], vel.shape[0], acc.shape[0], jerk.shape[0])
    sample_times = sample_times[:sample_count]
    sample_times_np = sample_times.detach().cpu().numpy()
    pos = pos[:sample_count]
    vel = vel[:sample_count]
    acc = acc[:sample_count]
    jerk = jerk[:sample_count]
    snap = torch.zeros_like(acc)
    # tim = timeit.default_timer()-start

    # print(f"get traj time :{tim}")

    if yaw_rate is not None:
        yaw_rate_values = np.asarray(yaw_rate, dtype=np.float32)
        if yaw_rate_values.ndim == 0:
            yaw_rate_values = np.full(sample_count, float(yaw_rate_values), dtype=np.float32)
        else:
            yaw_rate_values = np.resize(yaw_rate_values.reshape(-1), sample_count).astype(np.float32)
        yaw = float(start_yaw) + np.cumsum(yaw_rate_values) * float(sample_t)
        yaw -= yaw_rate_values[0] * float(sample_t)
        yaw_acceleration = np.gradient(yaw_rate_values, sample_times_np, edge_order=1).astype(np.float32)
    else:
        if start_yaw == end_yaw:
            yaw              = start_yaw * np.ones(sample_count, dtype=np.float32)
            yaw_rate_values  = np.zeros(sample_count, dtype=np.float32)
            yaw_acceleration = np.zeros(sample_count, dtype=np.float32)
        else:
            yaw = np.linspace(float(start_yaw), float(end_yaw), sample_count, dtype=np.float32)
            elapsed = max(float(duration - start_t), 1e-6)
            yaw_rate_values = np.full(sample_count, (float(end_yaw) - float(start_yaw)) / elapsed, dtype=np.float32)
            yaw_acceleration = np.gradient(yaw_rate_values, sample_times_np, edge_order=1).astype(np.float32)
   

    y = torch.cat([pos, vel, acc, jerk, snap], dim=1)
    y_psi = torch.as_tensor(
        np.stack([yaw, yaw_rate_values, yaw_acceleration], axis=1),
        dtype=torch.float32,
    )
    quad_cfg = getattr(config, "quad_cfg", None)
    inertia = np.asarray(
        getattr(quad_cfg, "inertia", getattr(config, "inertia", np.eye(3))),
        dtype=np.float32,
    )
    if inertia.size == 3:
        inertia = np.diag(inertia.reshape(3))
    params = SimpleNamespace(
        m=float(getattr(quad_cfg, "mass", getattr(config, "mass", 1.0))),
        J=torch.as_tensor(inertia.reshape(3, 3), dtype=torch.float32),
        g_vec=torch.tensor(
            [0.0, 0.0, abs(float(getattr(quad_cfg, "gravity", getattr(config, "gravity", 9.81))))],
            dtype=torch.float32,
        ),
    )

    flat_traj = traj_utils.flatOutputToStateControl(params, y, y_psi).detach().cpu().numpy()
    body_axes = flat_traj[:, 6:15].reshape(sample_count, 3, 3).transpose(0, 2, 1)
    u, _, vh = np.linalg.svd(body_axes)
    body_axes = u @ vh
    negative_det = np.linalg.det(body_axes) < 0.0
    if np.any(negative_det):
        u[negative_det, :, -1] *= -1.0
        body_axes[negative_det] = u[negative_det] @ vh[negative_det]
    quaternions = _rotation_matrices_to_quaternions(body_axes)

    traj = np.zeros((sample_count, 13), dtype=np.float32)
    traj[:, 0:3] = flat_traj[:, 0:3]
    traj[:, 3:7] = quaternions
    traj[:, 7:10] = flat_traj[:, 3:6]
    traj[:, 10:13] = flat_traj[:, 15:18]
    return traj, waypoints 

def test_traj(config: SampleServingConfig) -> dict[str, Any]:
    """Run a small waypoint and trajectory planning demo for a typed target."""
    torch.manual_seed(config.seed)
    world = SetupGenesisWorld(config=config.genesis_config, seed=config.seed)
    planner = TrajectorPlanner(world, config)

    drone_idx = 0
    start_t = 0
    cmd_dt    = 0.05

    planner.env_ESDF_map

    methods = [ planner.waypoints_A_star,
                planner.waypoints_RRT_connect,
                planner.waypoints_flow_matching]

    traj_solver = planner.MINCO_S3NU
                # , planner.B_spline]
    build_map = planner.env_ESDF_map
    while True:

        request = input("start position x y z yaw: ").strip()
        parts = request.replace(",", " ").split()
        if len(parts) == 4:
            start_position = np.asarray([float(value) for value in parts[0:3]], dtype=np.float32)
            start_yaw = float(parts[3])
        else:
            continue
        request = input("target postion x y z yaw: ").strip()
        parts = request.replace(",", " ").split()
        if len(parts) == 4:
            target_position = np.asarray([float(value) for value in parts[0:3]], dtype=np.float32)
            end_yaw = float(parts[3])
        else:
            continue

        request = input("duration t: ").strip()
        parts = request.replace(",", " ").split()
        if len(parts) == 1:
            duration = float(parts[0])
        else:
            continue

        output_path = Path(config.output)
        pdf_path = output_path.with_name(f"{output_path.stem}_realtime_states.pdf")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with PdfPages(pdf_path) as pdf:
            for get_waypoints in methods:
                # for planner_idx in range(2):
                #     planner = planners[planner_idx]
                start = timeit.default_timer()
                traj, waypoints = get_realtime_traj(world.config, get_waypoints, build_map, traj_solver,
                        start_t, duration, cmd_dt, config.max_velocity,
                        start_position, target_position, 
                        start_yaw, end_yaw, yaw_rate=None)
                tim = timeit.default_timer()-start
                times = start_t + np.arange(traj.shape[0], dtype=np.float32) * cmd_dt
                waypoint_name = getattr(get_waypoints, "__name__", get_waypoints.__class__.__name__)
                traj_solver_name = getattr(traj_solver, "__name__", traj_solver.__class__.__name__)

                print(f"waypoint method is: {waypoint_name}, planner is: {traj_solver_name}, time is: {tim}")
                fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                fig.suptitle(f"{traj_solver_name} with {waypoint_name}")
                for axis, values, labels, ylabel in (
                    (axes[0], traj[:, 0:3], ("x", "y", "z"), "pos"),
                    (axes[1], traj[:, 7:10], ("vx", "vy", "vz"), "vel"),
                    (axes[2], traj[:, 3:7], ("qw", "qx", "qy", "qz"), "quat"),
                    (axes[3], traj[:, 10:13], ("wx", "wy", "wz"), "omega"),
                ):
                    for col_idx, label in enumerate(labels):
                        axis.plot(times, values[:, col_idx], label=label)
                    axis.set_ylabel(ylabel)
                    axis.grid(True, alpha=0.25)
                    axis.legend(loc="best")
                axes[-1].set_xlabel("time [s]")
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            print(f"{waypoint_name} get waypoints as: {waypoints}")
        print(f"Saved realtime trajectory state plots to {pdf_path}")


def run_missions(config: SampleServingConfig) -> dict[str, Any]:
    """Run realtime FM replanning with background trajectory generation."""

    ## setup the genesis world
    torch.manual_seed(config.seed)
    device = resolve_device(config.device)
    world = SetupGenesisWorld()
    rng = np.random.default_rng(config.seed)
    world_dt = world.dt

    ## configure the controller 
    dt = 0.05    # Time step
    horizon = 1
    N = 20      # Horizontal length
    quad_name = "cf2x"
    MPCController = DroneMPCController(t_horizon = horizon, n_nodes = N, opt = dt, quad_name = quad_name)
    controller = MPCController.controller
    quad       = MPCController.quad
    com_dt     = MPCController.controller.dt
    com_N      = MPCController.controller.N

    # the update ratio between controller and the simulation world
    update_world = int(com_dt/world_dt)

    ## setup the trajectory planner 
    planner = TrajectorPlanner(world, config)
    get_waypoints = [planner.waypoints_A_star,
                    planner.waypoints_RRT_connect,
                    planner.waypoints_flow_matching]
    traj_solvers = [planner.MINCO_S3NU,
                  planner.B_spline]
    build_map = planner.env_ESDF_map
    get_waypoint = planner.waypoints_flow_matching
    traj_solver  = planner.MINCO_S3NU
    p2p_speed = 5.0

    ## some parameters for the mission
    mission_results: list[dict[str, Any]] = []
    mission_mode = False
    request_mode = False
    command_mode = False
    tracking_active = False
    reference_traj = np.empty((0, 13), dtype=float)
    traj_X = None
    recover_color = True
    ref_idx = -1
    follow_replan_interval = 0.5
    follow_target_threshold = 0.2 + config.safety_margin
    target_changed = False
    rolling_segment_duration = 2.0
    p2p_goal_tolerance = 0.25
    pending_p2p_plan: concurrent.futures.Future[tuple[Any, np.ndarray, np.ndarray]] | None = None
    desired_p2p_target: np.ndarray | None = None
    last_submitted_p2p_target: np.ndarray | None = None
    pending_p2p_target: np.ndarray | None = None
    pending_yaw_start_deg = 0.0
    pending_yaw_end_deg = 0.0
    active_trajectory = None
    active_segment_times: np.ndarray | None = None
    active_p2p_target: np.ndarray | None = None
    active_yaw_start_deg = 0.0
    active_yaw_end_deg = 0.0
    cmd_t = 0.0
    p2p_request_id = 0
    active_p2p_request_id = 0
    last_p2p_submit = -float("inf")
    # configure a real time and run backward executor for the trajectory generation
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    # the controlled drone id
    drone_idx = 0
    target_id = None
    hover_min_z = 0.4
    hover_goal = world.drones[drone_idx].get_hover_pose()
    if hover_goal[2] < hover_min_z:
        hover_goal[2] = hover_min_z
    target     =  hover_goal # 

    drone = world.drones[drone_idx]
    try:
        while True:
            request = parse_command.poll_mission_request()
            target_changed = False

            if request in {"hover", "stop_mission"}:
                mission_mode = False
                request_mode = False
                command_mode = False
                traj_X       = "hover"
                if pending_p2p_plan is not None:
                    pending_p2p_plan.cancel()
                    pending_p2p_plan = None
                request = None
            if request in {"q", "quit", "exit"}:
                break
        
            if request:                                                                                                 
                try:
                    parts = request.strip().replace(",", " ").split()
                    if parts[0].lower() == "drone_id":
                        drone_idx = int(parts[1])
                        drone = world.drones[drone_idx]
                        print(f"set the controlled drone is as: {drone_idx}, Maximum is {world.config.num_drones - 1}")
                        continue
                    request_dic = parse_command.parse_request(request, drone_idx)
                    com_req = request_dic["type"]
                    # requae to print state
                    if com_req == "request":
                        _ = apply_request(world, request_dic, drone_idx)
                        mission_mode = False
                        request_mode = True 
                        command_mode = False
                    elif com_req == "command":
                        traj_X, target = apply_command(world, request_dic, drone_idx) # will return the command for drone: traj_X = "hover"/"p2p"
                        mission_mode = False
                        request_mode = False 
                        command_mode = False
                        if traj_X == "p2p":
                            command_mode = True
                    elif com_req == "mission":
                        mission_name = request_dic["name"]
                        target_id     = request_dic["target_id"]
                        target_person =world.people[target_id]

                        if mission_name in {"follow", "hover"}:
                            mission_mode = True 
                            request_mode = False 
                            command_mode = False
                            build_map(ego_drone=drone.get_position(), target_person=target_person)
                            traj_X, target, pre_position = \
                            conduct_mission(config, drone, target_person, planner, hover_min_z)

                        else:
                            mission_mode = False
                            request_mode = False 
                            command_mode = False
                            if mission_name in {"circle"}:
                                command_mode = True
                                traj_X = "circle"  # need to be extended
                                target = np.asarray(target_person.get_position(), dtype=float).copy()
                            elif mission_name in {"land"}:
                                command_mode = True
                                traj_X = "land"
                                target_ = drone.get_position()
                                target_[2] = 0.1  # land the drone at current position 
                            elif mission_name in {"back"}:
                                command_mode = True
                                traj_X = "back"
                                target_ = DEFAULT_DRONE_POSITION
                            else: 
                                print(f"Unkown requestion type: {com_req}, please enter: request/command/mission! \n")
                                continue
                    if mission_mode:
                        target_person.color = (1.0, 1.0, 1.0, 1.0)
                    else:
                        if command_mode:
                            p_cur = drone.get_position()
                            euler_cur = utils.quaternion_to_euler(drone.get_quat())
                            heading = [float(np.rad2deg(euler_cur[2])),target[3]]
                            reference_traj, t_ref, reference_u = \
                            traj_utils.get_trajectory_X(quad, traj_X, com_dt, 
                                                        speed=p2p_speed, 
                                                        p_start=p_cur, 
                                                        p_end=target[0:3], 
                                                        heading=heading)
                            ref_idx = 0
                        for person in world.people:
                            person.color =  DEFAULT_COLOR

                except ValueError as exc:
                    print(exc)
                    continue
            
            if mission_mode:
                traj_X, target, pre_position = \
                            conduct_mission(config, drone, target_person, planner, hover_min_z)

                if desired_p2p_target is not None:
                    position_cur = _as_numpy_vector(drone.get_position())
                    old_distance = np.linalg.norm(position_cur[:3] - desired_p2p_target[:3])
                    new_distance = np.linalg.norm(position_cur[:3] - target[:3])
                    target_changed = abs(old_distance - new_distance) > follow_target_threshold
                    if target_changed:
                        desired_p2p_target = np.asarray(target, dtype=float).copy()
                else:
                    desired_p2p_target = target    

                if pending_p2p_plan is not None and pending_p2p_plan.done():
                    try:
                        trajectory, segment_times, waypoints = pending_p2p_plan.result()
                    except Exception as exc:
                        print(f"p2p planner failed: {exc}")
                    else:
                        plan_target_is_current = (
                            desired_p2p_target is not None
                            and pending_p2p_target is not None
                            and np.linalg.norm(pending_p2p_target[:3] - desired_p2p_target[:3]) <= follow_target_threshold
                        )
                        if plan_target_is_current:
                            active_trajectory = trajectory
                            active_segment_times = np.asarray(segment_times, dtype=float).reshape(-1)
                            active_p2p_target = pending_p2p_target.copy()
                            active_yaw_start_deg = pending_yaw_start_deg
                            active_yaw_end_deg = pending_yaw_end_deg
                            cmd_t = 0.0
                            last_follow_target = active_p2p_target[:3].copy()
                            tracking_active = True
                            reference_traj = np.empty((0, 13), dtype=float)
                            ref_idx = -1
                            traj_X = None
                            print(
                                f"new rolling duration: dist={rolling_segment_duration}"
                                f"end position is: {active_p2p_target[:3].round(3).tolist()} "
                                f"waypoints={len(waypoints)}"
                            )
                        else:
                            print("ignored stale p2p segment")
                    pending_p2p_plan = None
                    pending_p2p_target = None

                # 
                if desired_p2p_target is not None:
                    position_cur = _as_numpy_vector(drone.get_position())
                    distance_to_target = float(np.linalg.norm(desired_p2p_target[:3] - position_cur))
                    if distance_to_target <= p2p_goal_tolerance and not tracking_active:
                        desired_p2p_target = None
                        last_submitted_p2p_target = None
                        hover_goal = np.hstack([position_cur, [1, 0, 0, 0], np.zeros(3), np.zeros(3)])
                    else:
                        now = time.monotonic()
                        if last_submitted_p2p_target is None:
                            submitted_target_changed = True
                        else:
                            submitted_old_distance = np.linalg.norm(position_cur[:3] - last_submitted_p2p_target[:3])
                            submitted_new_distance = np.linalg.norm(position_cur[:3] - desired_p2p_target[:3])
                            submitted_target_changed = abs(submitted_old_distance - submitted_new_distance) > follow_target_threshold
                        should_submit_p2p = (
                            pending_p2p_plan is None
                            and (
                                active_trajectory is None
                                or (
                                    submitted_target_changed
                                    and now - last_p2p_submit >= follow_replan_interval
                                )
                            )
                        )
                        if should_submit_p2p:
                            euler_cur = utils.quaternion_to_euler(drone.get_quat())
                            p2p_request_id += 1
                            active_p2p_request_id = p2p_request_id
                            last_submitted_p2p_target = desired_p2p_target.copy()
                            pending_p2p_target = desired_p2p_target.copy()
                            pending_yaw_start_deg = float(np.rad2deg(euler_cur[2]))
                            if target_person is not None:
                                person_position = np.asarray(target_person.get_position(), dtype=float)
                                yaw_delta = person_position[:2] - desired_p2p_target[:2]
                                pending_yaw_end_deg = float(np.rad2deg(np.arctan2(yaw_delta[1], yaw_delta[0])))
                            else:
                                pending_yaw_end_deg = float(desired_p2p_target[3])
                            pending_p2p_plan = executor.submit(
                                optimize_traj,
                                get_waypoints=get_waypoint, 
                                build_map=build_map, 
                                traj_solver=traj_solver,
                                max_velocity = p2p_speed,
                                start_position=position_cur.copy(),
                                target_position=desired_p2p_target[:3].copy(),
                                target_person=target_person,
                            )

                            last_p2p_submit = now

                if active_trajectory is not None and active_segment_times is not None and active_p2p_target is not None:
                    reference = build_reference(
                        config,
                        active_trajectory,
                        active_segment_times,
                        request_id=active_p2p_request_id,
                        dt=com_dt,
                        yaw_start_deg=active_yaw_start_deg,
                        yaw_end_deg=active_yaw_end_deg,
                        cmd_t=cmd_t,
                        horizon=com_N,
                        target_position=active_p2p_target[:3],
                    )
                    goal = reference["goal"]
                    hover_goal = reference["hover_goal"]
                    tracking_active = not reference["finished"]
                    cmd_t += com_dt

                    if reference["finished"]:
                        active_trajectory = None
                        active_segment_times = None
                        active_p2p_target = None
                        traj_X = None
                        if request_mode or command_mode:
                            desired_p2p_target = None
                            last_submitted_p2p_target = None
                        print(f"reach the last reference, hover at: {hover_goal}")
                else:
                    goal = np.vstack([hover_goal] * (com_N+1))

            if command_mode:
                if ref_idx >= reference_traj.shape[0]:
                    traj_X = None
                    command_mode = False
                    hover_goal = np.hstack([reference_traj[-1, 0:3], [1,0,0,0], np.zeros(3), np.zeros(3)])
                    ref_idx = -1
                    print(f"reach the last reference, hover at: {hover_goal}")
                else:
                    if reference_traj.shape[0] > 1:
                        goal = get_ref(reference_traj, ref_idx, com_N)
                        ref_idx += 1
                    else:
                        goal = np.vstack([hover_goal] * (com_N+1))
                        ref_idx = -1
            if traj_X is None or traj_X in {"hover"}:
                goal = np.vstack([hover_goal] * (com_N+1))

            for t in range(update_world):
                current = drone.get_state()
                thrusts = controller.run_optimization(initial_state=current, goal=goal, mode='traj')[:4]
                rpms = drone.thrusts_to_rpms(thrusts)
                # print(f"current: {current}")
                # print(f"thrusts: {thrusts}")
                world.scene.step()
                # drone.settle_on_ground_if_landed()
                # Step simulation
                
                for person in world.people:

                    # if target_person is not None and person is target_person:
                        # if you want to operate the target_person 
                    person.update(world_dt, rng, WORLD_SIZE)

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


def build_reference(config,
                    trajectory, 
                    segment_times,   
                    request_id: int,
                    dt: float,
                    yaw_start_deg: float,
                    yaw_end_deg: float,
                    cmd_t: float,
                    horizon: int,
                    target_position: np.ndarray,
                    yaw_rate=None):
    """Sample the active optimized trajectory into an MPC goal horizon."""
    del request_id
    segment_times = np.asarray(segment_times, dtype=float).reshape(-1)
    trajectory_end = float(np.sum(segment_times))
    if hasattr(trajectory, "ts"):
        trajectory_end = min(trajectory_end, float(np.asarray(trajectory.ts)[-1]))
    trajectory_end = max(trajectory_end, float(dt))

    sample_times_np = float(cmd_t) + np.arange(int(horizon) + 1, dtype=np.float32) * float(dt)
    finished = float(cmd_t) >= trajectory_end
    after_end = sample_times_np >= trajectory_end
    query_end = max(trajectory_end - max(1e-6, trajectory_end * 1e-6), 0.0)
    query_times_np = np.minimum(sample_times_np, query_end).astype(np.float32)
    sample_times = torch.as_tensor(query_times_np, dtype=torch.float32)

    pos = torch.as_tensor(trajectory.pos(sample_times), dtype=torch.float32)
    vel = torch.as_tensor(trajectory.vel(sample_times), dtype=torch.float32)
    acc = torch.as_tensor(trajectory.acc(sample_times), dtype=torch.float32)
    jerk = torch.as_tensor(trajectory.jerk(sample_times), dtype=torch.float32)
    sample_count = int(horizon) + 1

    target_position = np.asarray(target_position, dtype=np.float32).reshape(3)
    if np.any(after_end):
        after_end_tensor = torch.as_tensor(after_end, dtype=torch.bool)
        pos[after_end_tensor] = torch.as_tensor(target_position, dtype=torch.float32)
        vel[after_end_tensor] = 0.0
        acc[after_end_tensor] = 0.0
        jerk[after_end_tensor] = 0.0
    snap = torch.zeros_like(acc)

    if yaw_rate is not None:
        yaw_rate_values = np.asarray(yaw_rate, dtype=np.float32)
        if yaw_rate_values.ndim == 0:
            yaw_rate_values = np.full(sample_count, float(yaw_rate_values), dtype=np.float32)
        else:
            yaw_rate_values = np.resize(yaw_rate_values.reshape(-1), sample_count).astype(np.float32)
        yaw = np.deg2rad(float(yaw_start_deg)) + np.cumsum(yaw_rate_values) * float(dt)
        yaw -= yaw_rate_values[0] * float(dt)
        yaw_acceleration = np.gradient(yaw_rate_values, sample_times_np, edge_order=1).astype(np.float32)
    else:
        start_yaw = np.deg2rad(float(yaw_start_deg))
        end_yaw = np.deg2rad(float(yaw_end_deg))
        if yaw_start_deg == yaw_end_deg:
            yaw              = start_yaw * np.ones(sample_count, dtype=np.float32)
            yaw_rate_values  = np.zeros(sample_count, dtype=np.float32)
            yaw_acceleration = np.zeros(sample_count, dtype=np.float32)
        else:
            yaw_alpha = np.clip(sample_times_np / max(trajectory_end, 1e-6), 0.0, 1.0).astype(np.float32)
            yaw = start_yaw + yaw_alpha * (end_yaw - start_yaw)
            yaw_rate_values = np.full(
                sample_count,
                (end_yaw - start_yaw) / max(trajectory_end, 1e-6),
                dtype=np.float32,
            )
            yaw_rate_values[after_end] = 0.0
            yaw_acceleration = np.zeros(sample_count, dtype=np.float32)
   

    y = torch.cat([pos, vel, acc, jerk, snap], dim=1)
    y_psi = torch.as_tensor(
        np.stack([yaw, yaw_rate_values, yaw_acceleration], axis=1),
        dtype=torch.float32,
    )
    quad_cfg = getattr(config, "quad_cfg", None)
    inertia = np.asarray(
        getattr(quad_cfg, "inertia", getattr(config, "inertia", np.eye(3))),
        dtype=np.float32,
    )
    if inertia.size == 3:
        inertia = np.diag(inertia.reshape(3))
    params = SimpleNamespace(
        m=float(getattr(quad_cfg, "mass", getattr(config, "mass", 1.0))),
        J=torch.as_tensor(inertia.reshape(3, 3), dtype=torch.float32),
        g_vec=torch.tensor(
            [0.0, 0.0, abs(float(getattr(quad_cfg, "gravity", getattr(config, "gravity", 9.81))))],
            dtype=torch.float32,
        ),
    )

    flat_traj = traj_utils.flatOutputToStateControl(params, y, y_psi).detach().cpu().numpy()
    body_axes = flat_traj[:, 6:15].reshape(sample_count, 3, 3).transpose(0, 2, 1)
    u, _, vh = np.linalg.svd(body_axes)
    body_axes = u @ vh
    negative_det = np.linalg.det(body_axes) < 0.0
    if np.any(negative_det):
        u[negative_det, :, -1] *= -1.0
        body_axes[negative_det] = u[negative_det] @ vh[negative_det]
    quaternions = _rotation_matrices_to_quaternions(body_axes)

    reference_traj = np.zeros((sample_count, 13), dtype=np.float32)
    reference_traj[:, 0:3] = flat_traj[:, 0:3]
    reference_traj[:, 3:7] = quaternions
    reference_traj[:, 7:10] = flat_traj[:, 3:6]
    reference_traj[:, 10:13] = flat_traj[:, 15:18]

    if np.any(after_end):
        reference_traj[after_end, 0:3] = target_position
        reference_traj[after_end, 3:7] = utils.euler_to_quaternion(0.0, 0.0, np.deg2rad(float(yaw_end_deg)))
        reference_traj[after_end, 7:13] = 0.0

    hover_goal = np.hstack([
        target_position,
        utils.euler_to_quaternion(0.0, 0.0, np.deg2rad(float(yaw_end_deg))),
        np.zeros(3),
        np.zeros(3),
    ])
    return {
        "goal": reference_traj,
        "hover_goal": hover_goal,
        "target": target_position,
        "finished": finished,
    }

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

def apply_request(world, request_dic, drone_idx):
    """Apply the request for drone or people."""
    target_id = request_dic["target_id"]
    request   = request_dic["name"]
    object    = world.drones[drone_idx] if request_dic["which"] in {"drone"} else world.people[target_id]
    
    if request in {"state"}:
        print(
            f"position=        {object.get_position().round(3).tolist()} \n"
            f"velocity=        {object.get_velocity().round(3).tolist()} \n"
            )
    elif request in {"attitude"}:  # only apply for drone
        print(f"Please be aware attitude is only for drones \n")
        print(
            f"quaternion           = {object.get_quat().round(4).tolist()} \n"
            f"Euler angle          = {object.get_euler().round(3).tolist()} \n"
            f"get_angular_velocity = {object.get_angular_velocity().round(3).tolist()} \n"
            )
    elif request in {"pose"}:
        print(f"position={object.get_position().round(3).tolist()}\n"
            )
    elif request in {"velocity"}:
        print(
            f"velocity=        {object.get_velocity().round(3).tolist()} \n"
            )
    else:
        print(f"Please enter a valis request: state/attitude/pose/velocity") 

def apply_command(world, request_dic, drone_idx) -> None:
    """Apply the command for drone or people."""
    traj_X = None
    target = None
    name = request_dic["name"].lower()
    if request_dic["which"] == "drone":
        if name in {"hover", "level"}:
            traj_X = "hover"
            target = world.drones[drone_idx].get_hover_pose()
            print(f"Returning to hover at {target}")
        elif name in {"goto"}:
            # otherwise, the command received, the drone conduct the received mission
            target = request_dic["command"]
            traj_X = "p2p"
            print(f"Go to: position -- {target[:3]} and yaw -- {target[3]}")
        else:
            print(f"Please define command -- {name} -- first!")

    elif request_dic["which"] == "people":
        target = request_dic["command"]
        target_id = request_dic["target_id"]
        person=world.people[target_id]
        if name in {"goto"}:
            person.set_position(target)
            print(f"Person {target_id}: go to: position -- {target[:3]}")
            
        elif name in {"velocity"}:
            person.set_velocity(target)
            print(f"Person {target_id}: set velocity -- {target[:3]}")

    return traj_X, target

def conduct_mission(config, drone, person,  
                    planner, hover_min_z):

    # if mission_name in {"hover", "follow"}:
    person_position = np.asarray(person.get_position(), dtype=float).copy()
    drone_position = np.asarray(drone.get_position(), dtype=float).copy()

    safe_distance = max(
        float(config.safety_margin),
        float(getattr(config, "map_resolution", 0.2)),
        1e-3,
    )

    # Direction from person to drone in XY plane
    direction = drone_position[:2] - person_position[:2]
    distance = np.linalg.norm(direction)

    if distance > 1e-6:
        direction = direction / distance
    else:
        direction = np.array([1.0, 0.0])

    # Closest point on circle to the drone
    target_ = person_position.copy()
    target_[:2] = person_position[:2] + safe_distance * direction
    target_[2] = person_position[2]

    # If that point is invalid, sample the circle and choose the closest valid point
    if not planner.is_point_valid(target_[:3]):
        best_target = None
        best_distance = np.inf

        for angle in np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False):
            candidate = person_position.copy()
            candidate[0] = person_position[0] + safe_distance * np.cos(angle)
            candidate[1] = person_position[1] + safe_distance * np.sin(angle)
            candidate[2] = person_position[2]

            if planner.is_point_valid(candidate[:3]):
                d = np.linalg.norm(candidate[:3] - drone_position[:3])
                if d < best_distance:
                    best_distance = d
                    best_target = candidate.copy()

        if best_target is not None:
            target_ = best_target
        elif planner.is_point_valid(drone_position[:3]):
            target_ = drone_position.copy()
    
    if target_[2] < hover_min_z:
        target_[2] = hover_min_z
    target = np.hstack([target_[:3], 0.0])
    traj_X = "p2p" 

    return traj_X, target, person_position

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


def plot_planning_demo(
    world: Drone,
    waypoints: np.ndarray,
    trajectory_positions: np.ndarray,
    start_position: np.ndarray,
    target_position: np.ndarray,
    title: str,
    output: str | Path,
) -> Path:
    """Save one waypoint and trajectory planning demo plot."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for building in world.buildings:
        _plot_box(ax, building.center, building.size)
    for person in world.people:
        position = np.asarray(person.get_position(), dtype=np.float32)
        ax.scatter(position[0], position[1], position[2], color="tab:orange", s=28, alpha=0.75)
    ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2], "o--", color="tab:gray", label="waypoints")
    ax.plot(
        trajectory_positions[:, 0],
        trajectory_positions[:, 1],
        trajectory_positions[:, 2],
        color="tab:blue",
        linewidth=2.2,
        label="trajectory",
    )
    ax.scatter(start_position[0], start_position[1], start_position[2], color="tab:green", s=70, label="start")
    ax.scatter(target_position[0], target_position[1], target_position[2], color="tab:red", marker="*", s=130, label="target")
    half_x = float(WORLD_SIZE[0]) / 2.0
    half_y = float(WORLD_SIZE[1]) / 2.0
    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_zlim(0.0, float(WORLD_SIZE[2]))
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(loc="upper right")
    fig.tight_layout()
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
    # run_sample(
    run_missions(
    # test_traj(
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
