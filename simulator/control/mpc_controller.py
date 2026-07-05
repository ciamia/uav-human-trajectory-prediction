from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from utils.quad_param_loader import TaskConfig
import timeit
import copy

def system_dic():

    state_load_dic = {"t":[],
                      "num": 1,
                      "pos": [],
                      "quat": [],
                      "vel": [],
                      "omega": [],
                      "acc":[],
                      "euler":[]
                    }
    state_cable_dic = {"t":[],
                       "num": 3,
                      "qi": [],
                      "wi": [],
                      "wi_d": [],
                      "wi_dd": [],
                      "Fi": [],
                      "fi":[],
                      "fi_d":[]
                    }
    state_drone_dic = {"t":[],
                       "num": 3,
                      "pos": [],
                      "quat": [],
                      "vel": [],
                      "omega": [],
                      "acc":[],
                      "euler":[],
                      "Fi":[]
                    }
    cmd_load_dic = {"t":[],
                    "num": 1,
                    "fi":[],
                    "wi_d":[],
                    "Fi":[],
                    "Fi_d":[],
                    "fi_d":[],
                    "fi_dd":[],
                    "wi_ddd":[]
                    }
    cmd_drone_dic = {"t":[],
                    "num": 3,
                    "Fi":[],
                    "f"    :[],
                    "Tau_ui":[],
                    "thrust":[]
                    }
    data_dic = {"state_load_dic":  state_load_dic,
                "state_load_est_dic": state_load_dic,
                "state_cable_dic": state_cable_dic,
                "state_drone_dic": state_drone_dic,
                "state_load_ref_dic":  state_load_dic,
                "state_cable_ref_dic": state_cable_dic,
                "state_drone_ref_dic": state_drone_dic,
                "cmd_load_dic":    cmd_load_dic,
                "cmd_drone_dic":   cmd_drone_dic,
                "time_record_load": [],
                "time_record_estimator": [],
                "time_record_drones": [],
                "obs_distance": []
    } 
    return data_dic, state_load_dic, state_cable_dic, state_drone_dic, cmd_load_dic, cmd_drone_dic


@dataclass(frozen=True)
class DroneMPCConfig:
    """Configuration for a lightweight double-integrator MPC tracker."""

    dt: float = 0.05
    horizon: int = 12
    position_weight: float = 12.0
    velocity_weight: float = 2.0
    acceleration_weight: float = 0.2
    terminal_position_weight: float = 30.0
    terminal_velocity_weight: float = 6.0
    max_velocity: float = 3.0
    max_acceleration: float = 5.0


@dataclass(frozen=True)
class DroneMPCResult:
    """Tracked state/control rollout returned by the MPC tracker."""

    states: np.ndarray
    controls: np.ndarray
    references: np.ndarray
    tracking_error: np.ndarray


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or arrays to float32 numpy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _as_state_reference(reference: Any, dt: float) -> np.ndarray:
    """Convert [T, 3] or [T, 6] references to [T, 6]."""
    values = _to_numpy(reference)
    if values.ndim != 2 or values.shape[1] not in (3, 6):
        raise ValueError("reference must have shape [T, 3] or [T, 6]")
    if values.shape[0] < 2:
        raise ValueError("reference must contain at least two waypoints")
    if values.shape[1] == 6:
        return values.copy()

    states = np.zeros((values.shape[0], 6), dtype=np.float32)
    states[:, :3] = values
    states[:, 3:6] = np.gradient(values, float(dt), axis=0).astype(np.float32)
    return states


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a vector to a maximum Euclidean norm."""
    norm = float(np.linalg.norm(vector))
    if norm <= float(max_norm):
        return vector.astype(np.float32)
    return (vector / max(norm, 1e-6) * float(max_norm)).astype(np.float32)


class DroneMPCController:
    """Receding-horizon MPC tracker for the simplified Genesis drone.

    The controller uses a double-integrator model with acceleration commands.
    It is intentionally simulation-only and does not model motor thrust,
    attitude or propellers.
    """

    def __init__(self, config_position: DroneMPCConfig | None = None, 
                 config_fullpose: TaskConfig | None = None,
                 t_horizon=1, n_nodes=20, opt=0.1, quad_name = "cf2x",
                 point_reference=False, models=None,
                 model_conf=None, rdrv=None) -> None:

        self.config = config_position or DroneMPCConfig()
        if self.config.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.config.horizon < 1:
            raise ValueError("horizon must be at least 1")
        
        # here add the acados controller for the full pose control, the control command is single thrusts
        config = "configs/task_config.yaml"
        self.config_all = TaskConfig(config)

        quad = custom_quad_param_loader("cf2x")

        # Initialize quad MPC
        if point_reference:
            acados_config = {
                "solver_type": "SQP",
                "terminal_cost": True
            }
        else:
            acados_config = {
                "solver_type": "SQP_RTI",
                "terminal_cost": False
            }

        q_diagonal = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        r_diagonal = np.array([1.0, 1.0, 1.0, 1.0])

        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

        quad_mpc = Controller(quad, t_horizon=t_horizon, n_nodes=n_nodes,
                             q_mask=q_mask, q_cost=q_diagonal, r_cost=r_diagonal, rdrv_d_mat=rdrv,
                            model_name=quad_name, solver_options=acados_config)

        self.quad_name = quad_name
        self.quad = quad
        self.controller = quad_mpc

        # Last optimization
        self.last_w = None

    def drone_track(self, ref_traj, filename=None):
        
        # self.drone  = Drone_3D()
        # self.controller_drone = Drone_Controller(self.drone , t_horizon=N*dt, n_nodes=N)  # Initialize MPC self.controller_drone
        pos_ref   = np.array(ref_traj["pos"])
        quat_ref  = np.array(ref_traj["quat"])
        vel_ref   = np.array(ref_traj["vel"])
        omega_ref = np.array(ref_traj["omega"])

        # xref, yref, zref = traj_util.get_drone_trajX(0, sim_time, dt)
        

        n_steps = len(pos_ref)
        print(f"Reference trajectory has {n_steps} steps")
        print(f"Reference final state: pos={pos_ref[-1]}, vel={vel_ref[-1]}\n")
        
        # Initialize to reference start
        self.drone .pos   = pos_ref[0, :]
        self.drone .quat  = quat_ref[0, :]
        self.drone .vel   = vel_ref[0, :]
        self.drone .omega = omega_ref[0, :]

        data_dic, _, _, state_drone_dic, _, cmd_drone_dic = system_dic()
        state_drone_ref_dic = copy.deepcopy(state_drone_dic)
        state_drone_dic["num"] = 1
        cmd_drone_dic["num"] = 1
        state_drone_ref_dic["num"] = 1

        path = []
        system_u = []
        time_record = []
        for i in range(n_steps):
            # Extract reference window  
            current_time = i * self.dt_drone
            i_end = min(i + self.N_drone + 1, n_steps)
            ref_pos = pos_ref[i:i_end, :]
            ref_vel = vel_ref[i:i_end, :]
            ref_quat = quat_ref[i:i_end, :]
            ref_omega = omega_ref[i:i_end, :]

            n_available = i_end - i
            # This creates consistent final condition: constant position, zero velocity
            if n_available < self.N_drone + 1:
                pad_len = self.N_drone + 1 - n_available
                ref_pos = np.vstack([ref_pos, np.tile(pos_ref[-1], (pad_len, 1))])
                ref_vel = np.vstack([ref_vel, np.zeros((pad_len, 3))])  # ZERO velocity for padding!
                ref_quat = np.vstack([ref_quat, np.tile([1, 0, 0, 0], (pad_len, 1))])
                ref_omega = np.vstack([ref_omega, np.zeros((pad_len, 3))])
                

            state_goal = np.hstack([ref_pos, ref_quat, ref_vel, ref_omega])

            current = np.concatenate([self.drone .pos, self.drone .quat, self.drone .vel, self.drone .omega])
            start = timeit.default_timer()
            thrust = self.controller.run_optimization(initial_state=current, goal=state_goal, mode='traj')[:4]
            time_record.append(timeit.default_timer() - start)
            self.drone .update(thrust, self.dt_drone)
            path.append(self.drone.pos)
            system_u.append(thrust)

            state_drone_dic["t"].append(current_time)
            state_drone_dic["pos"].append(self.drone.pos)
            state_drone_dic["quat"].append(self.drone.quat)
            state_drone_dic["vel"].append(self.drone.vel)
            state_drone_dic["omega"].append(self.drone.omega)

            cmd_drone_dic["t"].append(current_time)
            cmd_drone_dic["thrust"].append(thrust)
            # state_drone_dic["acc"].append(np.array(acc)) 

        state_drone_ref_dic["t"] = np.linspace(0, (n_steps-1) * self.dt_drone, n_steps)
        state_drone_ref_dic["pos"] = ref_traj["pos"]
        state_drone_ref_dic["quat"] = ref_traj["quat"]
        state_drone_ref_dic["vel"] = ref_traj["vel"]
        state_drone_ref_dic["omega"] = ref_traj["omega"]
        state_drone_ref_dic["acc"] = ref_traj["acc"]

        data_dic["state_drone_dic"] = state_drone_dic
        data_dic["state_drone_ref_dic"] = state_drone_ref_dic
        data_dic["cmd_drone_dic"] = cmd_drone_dic

        # CPU time
        print("average estimation time is {:.5f}".format(np.array(time_record).mean()))
        print("max estimation time is {:.5f}".format(np.array(time_record).max()))
        print("min estimation time is {:.5f}".format(np.array(time_record).min()))

        # Visualization
        path = np.array(path)
        # # print(path)
        # plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.plot(pos_ref[:, 0], pos_ref[:, 1], pos_ref[:, 2], c=[1,0,0], label='goal')
        # ax.plot(path[:,0], path[:,1], path[:,2], label='sim')
        # ax.axis('equal')
        # ax.set_xlabel('x [m]')
        # ax.set_ylabel('y [m]')
        # ax.set_zlabel('z [m]')
        # ax.legend()

        # plt.figure()
        # plt.plot(time_record)
        # plt.legend()
        # plt.ylabel('CPU Time [s]')
        # # plt.yscale("log")

        # plt.show()

        return path, system_u, data_dic

    def track(self, reference: Any, initial_state: Any | None = None) -> DroneMPCResult:
        """Track a Flow Matching/reference trajectory and return simulated states."""
        references = _as_state_reference(reference, self.config.dt)
        state = references[0].copy() if initial_state is None else _to_numpy(initial_state).reshape(-1)[:6].astype(np.float32)
        if state.shape[0] != 6:
            raise ValueError("initial_state must have shape [6]")

        states = np.zeros_like(references)
        controls = np.zeros((len(references), 3), dtype=np.float32)
        states[0] = state
        for idx in range(len(references) - 1):
            window = self._reference_window(references, idx)
            acceleration = self._first_control(state, window)
            acceleration = _clip_norm(acceleration, self.config.max_acceleration)
            state = self._step(state, acceleration)
            states[idx + 1] = state
            controls[idx] = acceleration

        tracking_error = np.linalg.norm(states[:, :3] - references[:, :3], axis=-1).astype(np.float32)
        return DroneMPCResult(states=states, controls=controls, references=references, tracking_error=tracking_error)

    def _reference_window(self, references: np.ndarray, start_idx: int) -> np.ndarray:
        """Return a horizon-sized reference window padded with the final state."""
        count = self.config.horizon + 1
        window = references[start_idx : start_idx + count]
        if len(window) == count:
            return window
        padding = np.repeat(references[-1][None, :], count - len(window), axis=0)
        return np.concatenate([window, padding], axis=0)

    def _first_control(self, state: np.ndarray, references: np.ndarray) -> np.ndarray:
        """Solve the finite-horizon LQR subproblem and return the first action."""
        dt = float(self.config.dt)
        identity = np.eye(3, dtype=np.float32)
        dynamics_a = np.block([[identity, dt * identity], [np.zeros((3, 3), dtype=np.float32), identity]]).astype(np.float32)
        dynamics_b = np.concatenate([0.5 * dt * dt * identity, dt * identity], axis=0).astype(np.float32)
        q = np.diag([self.config.position_weight] * 3 + [self.config.velocity_weight] * 3).astype(np.float32)
        q_terminal = np.diag([self.config.terminal_position_weight] * 3 + [self.config.terminal_velocity_weight] * 3).astype(np.float32)
        r = np.eye(3, dtype=np.float32) * float(self.config.acceleration_weight)

        p_matrix = q_terminal.copy()
        p_vector = -q_terminal @ references[-1]
        gains: list[tuple[np.ndarray, np.ndarray]] = []
        for ref in reversed(references[:-1]):
            control_hessian = r + dynamics_b.T @ p_matrix @ dynamics_b
            inv_control = np.linalg.pinv(control_hessian)
            gain = inv_control @ dynamics_b.T @ p_matrix @ dynamics_a
            feedforward = inv_control @ dynamics_b.T @ p_vector
            gains.append((gain.astype(np.float32), feedforward.astype(np.float32)))

            closed_term = dynamics_a.T @ p_matrix @ dynamics_b @ inv_control @ dynamics_b.T
            p_matrix = q + dynamics_a.T @ p_matrix @ dynamics_a - closed_term @ p_matrix @ dynamics_a
            p_vector = -q @ ref + dynamics_a.T @ p_vector - closed_term @ p_vector

        first_gain, first_feedforward = gains[-1]
        return (-first_gain @ state - first_feedforward).astype(np.float32)

    def _step(self, state: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
        """Advance the simplified drone state by one double-integrator step."""
        dt = float(self.config.dt)
        next_state = state.copy()
        next_state[:3] = state[:3] + state[3:6] * dt + 0.5 * acceleration * dt * dt
        next_state[3:6] = state[3:6] + acceleration * dt
        next_state[3:6] = _clip_norm(next_state[3:6], self.config.max_velocity)
        return next_state.astype(np.float32)


from multirotors.quadrotor.controller import Controller
from utils.quad_param_loader import custom_quad_param_loader
class QuadMPC:
    def __init__(self, t_horizon, n_mpc_nodes, opt_dt, quad_name, point_reference=False, models=None,
                 model_conf=None, rdrv=None):

        quad = custom_quad_param_loader(quad_name)

        # Initialize quad MPC
        if point_reference:
            acados_config = {
                "solver_type": "SQP",
                "terminal_cost": True
            }
        else:
            acados_config = {
                "solver_type": "SQP_RTI",
                "terminal_cost": False
            }

        q_diagonal = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        r_diagonal = np.array([1.0, 1.0, 1.0, 1.0])

        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

        quad_mpc = Controller(quad, t_horizon=t_horizon, n_nodes=n_mpc_nodes,
                             q_mask=q_mask, q_cost=q_diagonal, r_cost=r_diagonal, rdrv_d_mat=rdrv,
                            model_name=quad_name, solver_options=acados_config)

        self.quad_name = quad_name
        self.quad = quad
        self.controller = quad_mpc

        # Last optimization
        self.last_w = None
