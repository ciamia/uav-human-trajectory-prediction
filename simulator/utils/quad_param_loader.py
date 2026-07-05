
import numpy as np
import os
import yaml
import sys
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import sys
import yaml
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, Union

from multirotors.quadrotor.quadrotor import Quadrotor3D


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return yaml.safe_load(handle)

class TaskConfig:
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
        _yaml = load_yaml(config)
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
        quad_cfg.arm_length = float(_yaml.get("arm_length", 0.0397))
        quad_cfg.k_force = float(_yaml.get("k_force", 3.16e-10))
        quad_cfg.k_torque = float(_yaml.get("k_torque", 7.94e-12))


        # load the parameters of the angribird
        # quad = load_quad("angrybird")

        self.quad_cfg = quad_cfg
        # self.quad = quad

    def _get_task_cfg(self, config: Any) -> None:
        """Load task definition parameters into an attribute-accessible config."""
        _yaml = load_yaml(config)
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
        _yaml = load_yaml(config)
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
        _yaml = load_yaml(config)
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


def custom_quad_param_loader(quad_name):

    this_path = os.path.dirname(os.path.realpath(__file__))
    # params_file = os.path.join(this_path, 'configs', quad_name + '.yaml')
    params_file = os.path.join(this_path, '../configs', quad_name + '.yaml')
    # quad_name = "angrybird"  # or get this dynamically as needed
    # params_file = os.path.join(os.getcwd(), quad_name + '.yaml')


    # Get parameters for drone
    with open(params_file, "r") as stream:
        attrib = yaml.safe_load(stream)


    quad = Quadrotor3D(noisy=False, drag=False, payload=False, motor_noise=False)

    quad.mv = float(attrib['mass']) + (float(attrib['mass_rotor']) if 'mass_rotor' in attrib else 0) * 4
    quad.J = np.array(attrib['inertia'])
    quad.g = [0,0,9.81]

    quad.thrust_map = attrib["thrust_map"]

    if 'thrust_max' in attrib:
        quad.max_thrust = attrib['thrust_max']
    else:
        float(attrib["motor_omega_max"]) ** 2 * quad.thrust_map[0]
    quad.c = float(attrib['kappa'])

    if 'arm_length' in attrib and attrib['rotors_config'] == 'cross':
        quad.length = float(attrib['arm_length'])
        h = np.cos(np.pi / 4) * quad.length
        quad.x_f = np.array([h, -h, -h, h])
        quad.y_f = np.array([-h, h, -h, h])
    elif 'arm_length' in attrib and attrib['rotors_config'] == 'plus':
        quad.length = float(attrib['arm_length'])
        quad.x_f = np.array([0, 0, -quad.length, quad.length])
        quad.y_f = np.array([-quad.length, quad.length, 0, 0])
    else:
        tbm_fr = np.array(attrib['tbm_fr'])
        tbm_bl = np.array(attrib['tbm_bl'])
        tbm_br = np.array(attrib['tbm_br'])
        tbm_fl = np.array(attrib['tbm_fl'])
        quad.length = np.linalg.norm(tbm_fr)
        quad.x_f = np.array([tbm_fr[0], tbm_bl[0], tbm_br[0], tbm_fl[0]])
        quad.y_f = np.array([tbm_fr[1], tbm_bl[1], tbm_br[1], tbm_fl[1]])
    quad.z_l_tau = np.array([-quad.c, -quad.c, quad.c, quad.c])

    if 'motor_tau' in attrib:
        quad.motor_tau = attrib['motor_tau']

    if 'comm_delay' in attrib:
        quad.comm_delay = attrib['comm_delay']
     
    quad.thre = attrib['cable_tole']     # small threshold
    quad.eps = 1e-10      # Small epsilon to avoid division by zero 

    # quad.CUAV = attrib['cuav']
    # quad.ml = attrib['load_mass']
    # quad.cl = attrib['cable_length']
    # quad.OFFSET = attrib['offset']
    # quad.r = np.array(attrib['offset_vector'])  

    return quad
