import os

import genesis as gs
import math
import select

import numpy as np
import torch

import utils.utils as utils 
import utils.trajectories as traj_utils

from control.mpc_controller import DroneMPCController

from utils.quad_param_loader import TaskConfig

from env.drone import Drone, Drone


def _parse_terminal_command(line: str) -> tuple[np.ndarray, float]:
    """Parse terminal input as `Fz tx ty tz t` or `Fz tx ty tz`."""
    values = line.replace(",", " ").replace("[", " ").replace("]", " ").split()
    if len(values) not in (3, 4, 5):
        raise ValueError("expected: Fz tx ty tz [duration_seconds]")
    command = np.zeros(4,)
    command[:3] = np.asarray(values[:3], dtype=float)
    command[3] = float(values[3]) if len(values) >=4 else 0
    duration = float(values[4]) if len(values) == 5 else math.inf
    return command, duration

def yaw_quaternion(yaw: float) -> np.ndarray:
    """Return a level [w, x, y, z] quaternion for yaw in radians."""
    half_yaw = 0.5 * float(yaw)
    return np.array([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float32)


def main():
    
    task_file = "configs/task_config.yaml"
    config = TaskConfig(task_file)
    world = Drone(task_file)
    # Drone(task_config=config, build_scene=True)

    # Initialize controller
    # controller = DroneController()
    dt = 0.1    # Time step
    horizon = 1
    N = 20      # Horizontal length
    quad_name = "cf2x"
    drone_control = DroneMPCController(t_horizon = horizon, n_nodes = N, opt = dt, quad_name = quad_name)
        
    # quad = Quadrotor3D()    # Quadrotor model
    # controller = Controller(quad, t_horizon=2*N*dt, n_nodes=N)  # Initialize MPC controller

    is_running = True

    # Run simulation
    try:
        target = None
        hover_goal = world.get_hover_pose()
        print(f"Current position -- {hover_goal[:3]}")
        while is_running:
            if world.settle_on_ground_if_landed():
                world.scene.step()
                if "PYTEST_VERSION" in os.environ:
                    break
                continue

            # Update and apply RPMs based on current direction
            # rpms = controller.update_rpms()
            readable, _, _ = select.select([0], [], [], 0.0)
            if readable:
                line = input().strip()
                if line:
                    lowered = line.lower()
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
                    elif lowered in {"hover", "level"}:
                        command = world.return_to_hover(
                            reset_velocity=True,
                            reset_angular_velocity=True,
                            reset_pose=True,
                        )
                        hover_goal = world.get_hover_pose()
                        target = None
                        command_until = math.inf
                        command_active = False
                        print(f"Returning to hover command={command.tolist()}")
                    else:
                        # otherwise, the command received, the drone conduct the received mission
                        target, duration = _parse_terminal_command(line) # taget here is [x,y,z,yaw]
                        position_cur = world.drone.get_pos()
                        euler_cur    = utils.quaternion_to_euler(world.drone.get_quat())
                        p_end  = target[:3]
                        heading = [target[3], euler_cur[2]]
                        traj_X = "p2p"

                        # reference_traj, t_ref, reference_u = traj_utils.get_trajectory_X(quad, traj_X, dt, 
                        #                                                                     p_start=position_cur, p_end=p_end, heading=heading)
                        # command_until = math.inf if math.isinf(t_ref[-1]) else time.monotonic() + max(t_ref[-1], 0.0)
                        command_active = True
                        # print(f"Leght of the trajectory is ={len(t_ref)} for t={t_ref[-1]}")
                        print(f"Go to: position -- {target[:3]} and yaw -- {target[3]}")
                        
            if target is not None:
                yaw_quat = yaw_quaternion(target[3])
                goal = np.hstack([target[:3], yaw_quat, np.zeros(3), np.zeros(3)])
            else:
                goal = hover_goal
            current = world.get_state()   
            # if np.linalg.norm(current[:3]-target[:3]) > 0.1:
            # current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
            thrusts = drone_control.controller.run_optimization(initial_state=current, goal=goal)[:4]
            rpms = world.thrusts_to_rpms(thrusts)
            print(f"current: {current}")
            print(f"thrusts: {thrusts}")
            world.drone.set_propellers_rpm(rpms)

            # Step simulation
            world.scene.step()
            world.settle_on_ground_if_landed()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


if __name__ == "__main__":
    main()
