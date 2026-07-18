from __future__ import annotations

import argparse
import contextlib
import time
from typing import Any

import numpy as np
import math
import select
import os

from env.buildings import BoxBuilding
from env.drone import Drone
from env.people import PersonAgent, \
                        _assign_ids, \
                        _add_person_id_label, \
                        _matchstick_ground_z
from planning.hover_goal import is_position_in_obstacles

from control.mpc_controller import DroneMPCController
from control.quad_genesis import _parse_terminal_command, yaw_quaternion
import utils.utils as utils
import utils.trajectories as traj_utils

from utils.quad_param_loader import TaskConfig
import genesis as gs

import utils.parse_terminal_command as parse_command

WORLD_SIZE = np.array([30.0, 30.0, 20.0], dtype=np.float32)
DEFAULT_DRONE_POSITION = np.array([0.0, 0.0, 0.5], dtype=np.float32)

class SetupGenesisWorld:
    """Build a Genesis scene around a Drone world."""

    def __init__(self, config: str = "configs/task_config.yaml", seed: int = 0) -> None:
        self.setup_genesis_world(config, seed)
        self.gs = gs

    def setup_genesis_world(self, config: str = "configs/task_config.yaml", seed: int = 0) -> Drone:
        """Create a quadrotor Genesis world with buildings and people attached."""
        
        task_config = TaskConfig(config)
        
        buildings = make_buildings()
        # people = make_people_around_position(
        #     np.array([0.0, 0.0, 0.5], dtype=np.float32),
        #     WORLD_SIZE=WORLD_SIZE)

        people = setup_people_initial_position(
        DEFAULT_DRONE_POSITION.copy(),
        WORLD_SIZE,
        count=5,
        buildings=buildings,
        seed=seed,
    )
        
        assign_ids(buildings, "building")
        assign_ids(people, "person")

        drone_num = task_config.num_drones
        drones = []
        for id in range(drone_num):
            drone = Drone(config=task_config)
            drones.append(drone)
        
        self.dt        = 0.01
        self.build_scene(scene_setup=lambda gs, 
                        scene: _add_scene_entities(gs, scene, buildings, people, drones))
        self.drones    = drones
        self.buildings = buildings
        self.people    = people
        self.config    = task_config
        

    def __getattr__(self, name: str) -> Any:
        """Forward unknown attributes to the underlying Drone world."""
        return getattr(self.world, name)
    
    def build_scene(self, scene_setup: Any = None, show_viewer=True) -> None:
        """Initialize Genesis, create the drone scene, and build it."""
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                gravity=(0, 0, -9.81),
            ),
            viewer_options=gs.options.ViewerOptions(
                # camera_pos=(0.0, -2.0, 1.0),
                # camera_lookat=(0.0, 0.0, 0.3),
                # camera_fov=45,
                camera_pos=(3.0, -5.0, 3.0),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=45,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                show_world_frame=False,
            ),
            show_viewer=show_viewer,
            show_FPS=False,
        )
        self.scene.add_entity(gs.morphs.Plane())
        if scene_setup is not None:
            scene_setup(gs, self.scene)
        # self.scene.viewer.follow_entity(self.drone) # fix the viwer
        self.scene.build()


def _add_scene_entities(gs: Any, scene: Any, 
                        buildings: list[BoxBuilding], 
                        people: list[PersonAgent], 
                        drones: list[Drone]) -> None:
    """Add simple visual entities for buildings and people when Genesis is active."""
    if scene is None or not hasattr(gs, "morphs"):
        return

    for building in buildings:
        if hasattr(gs.morphs, "Box"):
            with contextlib.suppress(Exception):
                surface = _make_surface(gs, building.color)
                building.entity = scene.add_entity(
                    morph=gs.morphs.Box(pos=building.center.tolist(), size=building.size.tolist(), fixed=True),
                    surface=surface,
                    name=building.id,
                )

    for person in people:
        _add_person_entity(gs, scene, person)
        add_person_id_label(scene, gs, person)

    for drone in drones:
        with contextlib.suppress(Exception):
            drone.entity = scene.add_entity(
                        morph=gs.morphs.Drone(
                            file="urdf/drones/cf2x.urdf",
                            pos=(0.0, 0, 0.5),  # Start a bit higher
                            scale =1.0,
                            propellers_link_name=("prop0_link", "prop2_link", "prop1_link", "prop3_link"),
                            propellers_spin=(-1, -1, 1, 1),
                        ),
                        )
            # drone.scene = scene
            # drone.gs = gs
        
def _add_person_entity(gs: Any, scene: Any, person: PersonAgent) -> None:
    """Add a matchstick person proxy to the scene."""
    parts = person._make_matchstick_parts(gs)
    if parts:
        person.entities = []
        person.entity_offsets = []
        for suffix, morph, offset, color in parts:
            with contextlib.suppress(Exception):
                entity = scene.add_entity(morph=morph, surface=_make_surface(gs, color), name=f"{person.id}_{suffix}")
                person.entities.append(entity)
                person.entity_offsets.append(offset)
                person.entity = entity
                person.entity_offset = offset
        return

    if hasattr(gs.morphs, "Sphere"):
        with contextlib.suppress(Exception):
            person.entity = scene.add_entity(
                morph=gs.morphs.Sphere(pos=person.position.tolist(), radius=float(person.radius)),
                surface=_make_surface(gs, person.color),
                name=person.id,
            )


def _make_surface(gs: Any, color: tuple[float, float, float, float]) -> Any:
    """Create a Genesis default surface with the requested color."""
    if hasattr(gs, "surfaces") and hasattr(gs.surfaces, "Default"):
        with contextlib.suppress(Exception):
            return gs.surfaces.Default(color=color)
    return None

def make_buildings() -> list[BoxBuilding]:
    """Return two simple box buildings for the Genesis scene."""
    return [
        BoxBuilding(
            center=np.array([5.5, -10.0, 0]),
            size=np.array([8., 4., 10.0]),
            color=(0.35, 0.48, 0.78, 1.0),
        ),
        BoxBuilding(
            center=np.array([12.0, 10.0, 0]),
            size=np.array([6, 6, 5]),
            color=(0.72, 0.42, 0.28, 1.0),
        ),
        # BoxBuilding(
        #     center=np.array([-8.0, 5.0, 0]),
        #     size=np.array([8, 6, 15]),
        #     color=(0.12, 0.42, 0.88, 1.0),
        # ),
    ]

def make_people_around_position(drone_position: np.ndarray, WORLD_SIZE: np.ndarray) -> list[PersonAgent]:
    """Place five people in a small ring around the given drone position."""
    angles = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    specs = [
        (0.20, 1.55, "standing", (0.95, 0.18, 0.20, 1.0), (1.0, 0.78, 0.55, 1.0)),
        (0.24, 1.75, "wide", (0.18, 0.55, 0.95, 1.0), (0.86, 0.62, 0.42, 1.0)),
        (0.18, 1.68, "reaching", (0.20, 0.72, 0.34, 1.0), (0.72, 0.48, 0.32, 1.0)),
        (0.26, 1.45, "crouch", (0.85, 0.66, 0.16, 1.0), (0.95, 0.70, 0.48, 1.0)),
        (0.22, 1.85, "lean", (0.58, 0.28, 0.78, 1.0), (0.68, 0.46, 0.30, 1.0)),
    ]
  
    people: list[PersonAgent] = []
    idx = 0
    for angle, (radius, height, posture, color, head_color) in zip(angles, specs):
        offset = np.array([0.8 * np.cos(angle), 1.5 * np.sin(angle), 0.35], dtype=np.float32)
        position = drone_position + offset
        position[0] = np.clip(position[0], -WORLD_SIZE[0] / 2.0 + 0.2, WORLD_SIZE[0] / 2.0 - 0.2)
        position[1] = np.clip(position[1], -WORLD_SIZE[1] / 2.0 + 0.2, WORLD_SIZE[1] / 2.0 - 0.2)
        position[2] = matchstick_ground_z(height, posture)
        people.append(
            PersonAgent(
                position=position,
                velocity=np.zeros(3, dtype=np.float32),
                radius=radius,
                height=height,
                color=color,
                head_color=head_color,
                posture=posture,
                name_id=idx+1,
                center_z=position[2],
            )
        )
        idx += 1
    return people


def setup_people_initial_position(
    drone_position: np.ndarray,
    world_size: np.ndarray | tuple[float, float, float],
    count: int = 5,
    buildings: list[BoxBuilding] | None = None,
    margin: float = 0.25,
    seed: int | None = None,
) -> list[PersonAgent]:
    """Place setup-style matchstick people at random positions in the world.

    Avoid positions inside any building boxes when obstacles are provided.
    """
    world = np.asarray(world_size, dtype=np.float32)
    specs = [
        (0.20, 1.55, "standing", (0.95, 0.18, 0.20, 1.0), (1.0, 0.78, 0.55, 1.0)),
        (0.24, 1.75, "wide", (0.18, 0.55, 0.95, 1.0), (0.86, 0.62, 0.42, 1.0)),
        (0.18, 1.68, "reaching", (0.20, 0.72, 0.34, 1.0), (0.72, 0.48, 0.32, 1.0)),
        (0.26, 1.45, "crouch", (0.85, 0.66, 0.16, 1.0), (0.95, 0.70, 0.48, 1.0)),
        (0.22, 1.85, "lean", (0.58, 0.28, 0.78, 1.0), (0.68, 0.46, 0.30, 1.0)),
    ]
    people: list[PersonAgent] = []
    rng = np.random.default_rng(seed)
    for idx in range(max(int(count), 0)):
        radius, height, posture, color, head_color = specs[idx % len(specs)]
        for attempt in range(200):
            position = np.array(
                [
                    rng.uniform(-world[0] / 2.0 + 0.2, world[0] / 2.0 - 0.2),
                    rng.uniform(-world[1] / 2.0 + 0.2, world[1] / 2.0 - 0.2),
                    0.0,
                ],
                dtype=np.float32,
            )
            position[2] = matchstick_ground_z(height, posture)
            if buildings is None or not is_position_in_obstacles(position, buildings, margin):
                break
        else:
            raise ValueError(
                "Could not sample a non-building person position after 200 attempts"
            )
        people.append(
            PersonAgent(
                position=position,
                velocity=np.zeros(3, dtype=np.float32),
                radius=radius,
                height=height,
                color=color,
                head_color=head_color,
                posture=posture,
                name_id=idx+1,
                center_z=position[2],
            )
        )

    # people = [
    #     PersonAgent(
    #         position=np.array([x, y, _matchstick_ground_z(height, posture)], dtype=np.float32),
    #         velocity=np.zeros(3, dtype=np.float32),
    #         radius=radius,
    #         height=height,
    #         color=color,
    #         head_color=head_color,
    #         posture=posture,
    #         center_z=position[2],
    #     )
    #     for x, y, radius, height, posture, color, head_color in [
    #         (-0.8, -0.5, 0.20, 1.55, "standing", (0.95, 0.18, 0.20, 1.0), (1.0, 0.78, 0.55, 1.0)),
    #         (0.0, -0.7, 0.24, 1.75, "wide", (0.18, 0.55, 0.95, 1.0), (0.86, 0.62, 0.42, 1.0)),
    #         (0.8, -0.5, 0.18, 1.68, "reaching", (0.20, 0.72, 0.34, 1.0), (0.72, 0.48, 0.32, 1.0)),
    #         (-0.4, 0.35, 0.26, 1.45, "crouch", (0.85, 0.66, 0.16, 1.0), (0.95, 0.70, 0.48, 1.0)),
    #         (0.45, 0.35, 0.22, 1.85, "lean", (0.58, 0.28, 0.78, 1.0), (0.68, 0.46, 0.30, 1.0)),
    #     ]
    # ]
        
    return people

def matchstick_ground_z(height: float, posture: str) -> float:
    return _matchstick_ground_z(height, posture)

def add_person_id_label(scene: Any, gs: Any, person: PersonAgent):
    return _add_person_id_label(scene, gs, person)

def assign_ids(items: list[Any], prefix: str) -> None:
    """Attach stable ids to world objects."""
    _assign_ids(items, prefix)

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




def main() -> None:
    """Launch the configured Genesis world and keep the viewer alive briefly."""
    parser = argparse.ArgumentParser(description="Create a Genesis quadrotor world with buildings and people.")
    parser.add_argument("--config", default="configs/task_config.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=1000000.0)
    args = parser.parse_args()

    world = SetupGenesisWorld(config=args.config, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    world_size = np.array([4.0, 4.0, 2.0], dtype=np.float32)
    world_dt = 0.02

    dt = 0.1    # Time step
    horizon = 1
    N = 20      # Horizontal length
    quad_name = "cf2x"
    MPCController = DroneMPCController(t_horizon = horizon, n_nodes = N, opt = dt, quad_name = quad_name)
    controller = MPCController.controller
    quad       = MPCController.quad
    com_dt     = MPCController.controller.dt
    com_N      = MPCController.controller.N

    update_world = int(com_dt/world_dt)

    ref_idx = -1
    drone_idx = 0
    reference_traj = np.array([]); t_ref = np.array([]); reference_u = np.array([])
    command_active = False
    # Run simulation
    try:
        target = None
        hover_goal = world.drones[drone_idx].get_hover_pose()
        print(f"Current position -- {hover_goal[:3]}")
        while True:
            if world.drones[drone_idx].settle_on_ground_if_landed():
                world.scene.step()
                if "PYTEST_VERSION" in os.environ:
                    break
                continue

            # Update and apply RPMs based on current direction
            # rpms = controller.update_rpms()
            request = parse_command.poll_mission_request()
            request_dic = parse_command.parse_request(request) if request else None
            if request_dic:                
                com_req = request_dic["type"]
                obj_idx = request_dic["id"]
                name = request_dic["name"]
                object = world.drones[obj_idx] if request_dic["which"] in {"drone"} else world.people[obj_idx]
                if com_req == "request":
                    parse_command.apply_request(object, name)
                elif com_req == "command" and request_dic["which"] == "people":
                    value = request_dic["command"]
                    if name in {"goto"}:
                        object.set_position(value)
                    if name in  "velocity":
                        object.set_velocity(value)
                else:
                    if request_dic["which"] == "drone":
                        # otherwise, the command received, the drone conduct the received mission
                        position_cur = world.drones[obj_idx].get_position()
                        euler_cur    = utils.quaternion_to_euler(world.drones[obj_idx].get_quat())
                        target = request_dic["command"] if request_dic["which"]=="goto" else position_cur
                        p_end  = target[:3]
                        heading = [target[3], euler_cur[2]]
                        traj_X = "p2p"
                        reference_traj, t_ref, reference_u = traj_utils.get_trajectory_X(
                                                    quad, traj_X, com_dt, 
                                                    p_start=position_cur, p_end=p_end, heading=heading)
                        
                        ref_idx = 0
                        # command_until = math.inf if math.isinf(t_ref[-1]) else time.monotonic() + max(t_ref[-1], 0.0)
                        command_active = True
                        # print(f"Leght of the trajectory is ={len(t_ref)} for t={t_ref[-1]}")
                        print(f"Go to: position -- {target[:3]} and yaw -- \
                              {target[3]}, shape of reference_traj -- \
                              {reference_traj.shape}, t -- {t_ref[-1]}")
                        
            current = world.drones[drone_idx].get_state()
            if ref_idx >= reference_traj.shape[drone_idx]:
                command_active = False
                hover_goal = np.hstack([reference_traj[-1, 0:3], [1,0,0,0], np.zeros(3), np.zeros(3)])
                ref_idx = -1
                print(f"reach the last reference, hover at: {hover_goal}")
            if command_active: #  and reference_traj.shape[drone_idx] > com_N:
                goal = get_ref(reference_traj, ref_idx, com_N)
                ref_idx += 1
            else:
                # goal = np.tile(hover_goal, (com_N+1, 1))
                goal = np.vstack([hover_goal] * (com_N+1))

            for up_t in range(update_world):
                thrusts = controller.run_optimization(initial_state=current, goal=goal, mode='traj')[:4]
                rpms = world.drones[drone_idx].thrusts_to_rpms(thrusts)
                # print(f"current: {current}")
                # print(f"thrusts: {thrusts}")
                
                # Step simulation
                for person in world.people:
                    person.update(world_dt, rng, WORLD_SIZE)
                world.scene.step()
                world.drones[drone_idx].settle_on_ground_if_landed()

            if "PYTEST_VERSION" in os.environ:
                break
    except KeyboardInterrupt:
        gs.logger.info("Simulation interrupted, exiting.")
    finally:
        gs.logger.info("Simulation finished.")


    if args.seconds > 0.0:
        end_time = time.time() + args.seconds
        while time.time() < end_time:
            with contextlib.suppress(Exception):
                for person in world.people:
                    person.update(world_dt, rng, world_size)
                world.scene.step()
                world.drones[drone_idx].settle_on_ground_if_landed()


if __name__ == "__main__":
    main()
