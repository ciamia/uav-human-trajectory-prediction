from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import contextlib
import time

import numpy as np

def _vec3(value: Any, name: str) -> np.ndarray:
    """Convert a value to a 3D float32 vector."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 3:
        raise ValueError(f"{name} must have exactly 3 values")
    return vector.copy()


def make_visual_morph(morph_cls: Any, **kwargs: Any) -> Any:
    """Create a state-settable fixed rigid Genesis morph when supported."""
    for extra_kwargs in ({"fixed": True, "collision": False}, {"fixed": True}, {"collision": False}, {}):
        merged = dict(kwargs)
        merged.update(extra_kwargs)
        try:
            return morph_cls(**merged)
        except TypeError:
            continue
    return morph_cls(**kwargs)


def _add_entity(scene: Any, morph: Any, name: str, gs: Any, color: tuple[float, float, float, float]) -> Any:
    """Add a named visual entity to a Genesis scene."""
    surface = None
    if hasattr(gs, "surfaces") and hasattr(gs.surfaces, "Default"):
        with contextlib.suppress(Exception):
            surface = gs.surfaces.Default(color=color)
    try:
        if surface is not None:
            return scene.add_entity(morph=morph, surface=surface, name=name)
        return scene.add_entity(morph=morph, name=name)
    except TypeError:
        if surface is not None:
            return scene.add_entity(morph, surface=surface)
        return scene.add_entity(morph)


def set_entity_position(entity: Any, position: np.ndarray) -> None:
    """Move a Genesis entity if it exposes a position setter."""
    if entity is None:
        return
    if hasattr(entity, "set_pos"):
        with contextlib.suppress(Exception):
            entity.set_pos(np.asarray(position, dtype=np.float32))


@dataclass
class PersonAgent:
    """Simple moving person represented as a matchstick-style figure."""

    position: np.ndarray
    velocity: np.ndarray
    radius: float = 0.25
    height: float = 1.7
    center_z: float = 1.5
    behavior: str = "random_walk"
    routine_points: list[np.ndarray] | None = None
    command: str | None = None
    color: tuple[float, float, float, float] = (1.0, 0.55, 0.05, 1.0)
    head_color: tuple[float, float, float, float] = (1.0, 0.78, 0.45, 1.0)
    posture: str = "standing"
    entity: Any = field(default=None, init=False, repr=False)
    entity_offset: np.ndarray = field(default_factory=list, init=False, repr=False)
    entities: list[Any] = field(default_factory=list, init=False, repr=False)
    entity_offsets: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    routine_index: int = field(default=0, init=False)
    name_id: int = 1

    def __post_init__(self) -> None:
        self.position = _vec3(self.position, "position")
        self.velocity = _vec3(self.velocity, "velocity")
        if self.routine_points is not None:
            self.routine_points = [_vec3(point, "routine point") for point in self.routine_points]

    def add_to_scene(self, gs: Any, scene: Any, name: str = "person") -> Any:
        """Add the person proxy to a Genesis scene."""
        if scene is None or gs is None or not hasattr(gs, "morphs"):
            return None
        parts = self._make_matchstick_parts(gs)
        if parts:
            self.entities = []
            self.entity_offsets = []
            for suffix, morph, offset, color in parts:
                entity = _add_entity(scene, morph, name=f"{name}_{suffix}", gs=gs, color=color)
                self.entities.append(entity)
                self.entity_offsets.append(offset)
            self.entity = self.entities[0] if self.entities else None
            return self.entities

        morph = self._make_person_fallback_morph(gs)
        if morph is None:
            return None
        self.entity = _add_entity(scene, morph, name=name, gs=gs, color=self.color)
        self.entities = [self.entity]
        self.entity_offsets = [np.zeros(3, dtype=np.float32)]
        return self.entity

    def _make_matchstick_parts(
        self, gs: Any
    ) -> list[tuple[str, Any, np.ndarray, tuple[float, float, float, float]]]:
        """Create head, torso, arms, and legs from simple Genesis primitives."""
        if not hasattr(gs.morphs, "Box"):
            return []

        parts: list[tuple[str, Any, np.ndarray, tuple[float, float, float, float]]] = []
        stick_width = max(0.035, float(self.radius) * 0.18)
        head_radius = max(0.09, float(self.radius) * 0.45)
        shoulder_width = max(0.45, float(self.radius) * 1.8)
        torso_height = float(self.height) * 0.38
        leg_height = float(self.height) * 0.32
        torso_offset = np.array([0.0, 0.0, 0.10], dtype=np.float32)
        arms_offset = np.array([0.0, 0.0, 0.22], dtype=np.float32)
        left_leg_offset = np.array([-stick_width * 1.6, 0.0, -0.42], dtype=np.float32)
        right_leg_offset = np.array([stick_width * 1.6, 0.0, -0.42], dtype=np.float32)
        arm_size = np.array([shoulder_width, stick_width, stick_width], dtype=np.float32)
        left_leg_size = np.array([stick_width, stick_width, leg_height], dtype=np.float32)
        right_leg_size = np.array([stick_width, stick_width, leg_height], dtype=np.float32)

        if self.posture == "wide":
            left_leg_offset[0] *= 2.2
            right_leg_offset[0] *= 2.2
            arm_size[0] *= 1.25
        elif self.posture == "reaching":
            arms_offset = np.array([0.16, 0.0, 0.38], dtype=np.float32)
            arm_size = np.array([stick_width, stick_width, shoulder_width * 0.85], dtype=np.float32)
        elif self.posture == "crouch":
            torso_offset[2] -= 0.14
            arms_offset[2] -= 0.08
            left_leg_offset[2] += 0.12
            right_leg_offset[2] += 0.12
            left_leg_size[2] *= 0.62
            right_leg_size[2] *= 0.62
        elif self.posture == "lean":
            torso_offset[0] += 0.08
            arms_offset[0] += 0.14
            left_leg_offset[0] -= 0.03
            right_leg_offset[0] += 0.10

        if hasattr(gs.morphs, "Sphere"):
            head_offset = np.array([0.0, 0.0, float(self.height) * 0.43], dtype=np.float32)
            if self.posture == "crouch":
                head_offset[2] -= 0.12
            elif self.posture == "lean":
                head_offset[0] += 0.12
            head = make_visual_morph(
                gs.morphs.Sphere,
                pos=(self.position + head_offset).tolist(),
                radius=head_radius,
            )
            parts.append(("head", head, head_offset, self.head_color))

        box_specs = [
            ("torso", torso_offset, np.array([stick_width * 1.4, stick_width, torso_height], dtype=np.float32)),
            ("arms", arms_offset, arm_size),
            ("left_leg", left_leg_offset, left_leg_size),
            ("right_leg", right_leg_offset, right_leg_size),
        ]
        for suffix, offset, size in box_specs:
            morph = make_visual_morph(
                gs.morphs.Box,
                pos=(self.position + offset).tolist(),
                size=size.tolist(),
            )
            parts.append((suffix, morph, offset, self.color))
        return parts

    def _make_person_fallback_morph(self, gs: Any) -> Any:
        """Create a capsule/cylinder proxy with sphere fallback."""
        center = self.position.tolist()
        for morph_name in ("Capsule", "Cylinder"):
            if hasattr(gs.morphs, morph_name):
                morph_cls = getattr(gs.morphs, morph_name)
                try:
                    return make_visual_morph(morph_cls, pos=center, radius=float(self.radius), height=float(self.height))
                except TypeError:
                    try:
                        return make_visual_morph(morph_cls, radius=float(self.radius), height=float(self.height), pos=center)
                    except TypeError:
                        pass
        if hasattr(gs.morphs, "Sphere"):
            return make_visual_morph(gs.morphs.Sphere, pos=center, radius=float(self.radius))
        return None

    def update(self, dt: float, rng: np.random.Generator, world_size: np.ndarray, behavior="random_walk") -> None:
        """Advance the simple person behavior by one timestep."""
        self.behavior = behavior
        if self.behavior == "random_walk":
            self._random_walk(dt, rng)
        elif self.behavior == "routine":
            self._routine_step(dt)
        else:
            raise ValueError(f"Unsupported person behavior: {self.behavior}")
        self._keep_inside_world(world_size)
        self._set_visual_position()

    def _set_visual_position(self) -> None:
        """Move all matchstick parts to the current agent position."""
        if self.entities and len(self.entities) == len(self.entity_offsets):
            for entity, offset in zip(self.entities, self.entity_offsets):
                set_entity_position(entity, self.position + offset)
            return
        set_entity_position(self.entity, self.position)

    def get_state(self) -> np.ndarray:
        """Return the grouped person state [x, y, z, vx, vy, vz]."""
        if self.entity is None or self._has_composite_visual():
            return np.concatenate([self.position, self.velocity]).astype(np.float32)
        if hasattr(self.entity, "get_pos"):
            self.position = self.entity.get_pos().detach().cpu().numpy()
        if hasattr(self.entity, "get_vel"):
            self.velocity = self.entity.get_vel().detach().cpu().numpy()
        return np.concatenate([self.position, self.velocity]).astype(np.float32)

    def get_position(self) -> np.ndarray:
        """Return the person position [x, y, z]."""
        if self.entity is None or self._has_composite_visual():
            return self.position
        if hasattr(self.entity, "get_pos"):
            self.position = self.entity.get_pos().detach().cpu().numpy()
        return self.position
    
    def get_velocity(self) -> np.ndarray:
        """Return the person velocity [vx, vy, vz]."""
        if self.entity is None or self._has_composite_visual():
            return self.velocity
        if hasattr(self.entity, "get_vel"):
            self.velocity = self.entity.get_vel().detach().cpu().numpy()
        return self.velocity

    def set_position(self, position: Any) -> None:
        """Set the person position [x, y, z] and move its visual parts."""
        self.position = _vec3(position , "position")
        self.position[2] = self.center_z
        if self.entity is None:
            return
        if self._has_composite_visual():
            self._set_visual_position()
            return
        postion = self.position.copy()
        if hasattr(self.entity, "set_pos"):
            self.entity.set_pos(np.asarray(postion, dtype=np.float32))

    def set_velocity(self, velocity: Any) -> None:
        """Set the person velocity [vx, vy, vz]."""
        self.velocity = _vec3(velocity, "velocity")
        self.velocity[2] = 0 #  = np.asarray([self.velocity[0], self.velocity[1], 0])
        if self.entity is None or self._has_composite_visual():
            return
        if hasattr(self.entity, "set_vel"):
            self.entity.set_vel(np.asarray(velocity, dtype=np.float32))

    def _has_composite_visual(self) -> bool:
        """Return True when the person is represented by multiple visual parts."""
        return bool(self.entities and len(self.entities) > 1)

    def get_position_velocity(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of the person position and velocity."""
        return self.get_position(), self.get_velocity()

    def set_state(self, state: Any) -> None:
        """Set the grouped matchstick rigid proxy state and move every part."""
        values = np.asarray(state.detach().cpu().numpy() if hasattr(state, "detach") else state, dtype=np.float32).reshape(-1)
        if values.shape[0] < 3:
            raise ValueError("person state must contain at least x, y, z")
        self.set_position(values[:3].copy())
        if values.shape[0] >= 6:
            self.set_velocity(values[3:6].copy())

    def _random_walk(self, dt: float, rng: np.random.Generator) -> None:
        noise = rng.normal(0.0, 0.4, size=3).astype(np.float32)
        noise[2] = 0.0
        self.velocity = 0.92 * self.velocity + 0.08 * noise
        speed = np.linalg.norm(self.velocity[:2])
        if speed > 1.0:
            self.velocity[:2] = self.velocity[:2] / speed
        self.position += self.velocity * float(dt)
        self.position[2] = self.center_z

    def _routine_step(self, dt: float) -> None:
        if not self.routine_points:
            self.position += self.velocity * float(dt)
            return
        target = self.routine_points[self.routine_index]
        delta = target - self.position
        delta[2] = 0.0
        distance = np.linalg.norm(delta)
        if distance < 0.15:
            self.routine_index = (self.routine_index + 1) % len(self.routine_points)
            return
        direction = delta / max(distance, 1e-6)
        self.velocity = direction.astype(np.float32)
        self.position += self.velocity * float(dt)
        self.position[2] = self.center_z

    def _keep_inside_world(self, world_size: np.ndarray) -> None:
        half_x = float(world_size[0]) / 2.0
        half_y = float(world_size[1]) / 2.0
        limits = [(-half_x, half_x), (-half_y, half_y)]
        for axis, (low, high) in enumerate(limits):
            if self.position[axis] < low:
                self.position[axis] = low
                self.velocity[axis] = abs(self.velocity[axis])
            elif self.position[axis] > high:
                self.position[axis] = high
                self.velocity[axis] = -abs(self.velocity[axis])
    


def default_people(count: int, rng: np.random.Generator, world_size: np.ndarray) -> list[PersonAgent]:
    """Create a small group of moving people for the Milestone 1 demo."""
    people: list[PersonAgent] = []
    low = np.array([-world_size[0] / 2.0, -world_size[1] / 2.0, 0.85], dtype=np.float32)
    high = np.array([world_size[0] / 2.0, world_size[1] / 2.0, 0.85], dtype=np.float32)
    for _ in range(count):
        position = _sample_open_position(rng, low, high)
        velocity = rng.normal(0.0, 0.35, size=3).astype(np.float32)
        velocity[2] = 0.0
        people.append(PersonAgent(position=position, velocity=velocity, behavior="random_walk"))
    return people

def _matchstick_ground_z(height: float, posture: str) -> float:
    """Return the root z that puts the matchstick feet on the ground plane."""
    leg_offset_z = -0.42
    leg_height = float(height) * 0.32
    if posture == "crouch":
        leg_offset_z += 0.12
        leg_height *= 0.62
    return float(-(leg_offset_z - leg_height / 2.0))

def _sample_open_position(rng: np.random.Generator, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Sample a person position away from the default central building."""
    for _ in range(100):
        position = rng.uniform(low, high).astype(np.float32)
        if abs(float(position[0])) > 1.1 or abs(float(position[1])) > 1.1:
            return position
    return rng.uniform(low, high).astype(np.float32)

def get_people_positions_velocities(people: list[PersonAgent]) -> dict[str, np.ndarray]:
    """Return each person's grouped state keyed by id."""
    return {getattr(person, "id", f"person_{idx}"): person.get_state() for idx, person in enumerate(people)}
        

def set_people_positions_velocities(people: list[PersonAgent], states: dict[str, Any]) -> None:
    """Set person states from an id-keyed mapping of [x, y, z, vx, vy, vz]."""
    people_by_id = {getattr(person, "id", f"person_{idx}"): person for idx, person in enumerate(people)}
    for person_id, state in states.items():
        if person_id not in people_by_id:
            raise KeyError(f"Unknown person id: {person_id}")
        people_by_id[person_id].set_state(state)


_LABEL_SEGMENTS: dict[str, tuple[str, ...]] = {
    "0": ("top", "upper_left", "upper_right", "lower_left", "lower_right", "bottom"),
    "1": ("upper_right", "lower_right"),
    "2": ("top", "upper_right", "middle", "lower_left", "bottom"),
    "3": ("top", "upper_right", "middle", "lower_right", "bottom"),
    "4": ("upper_left", "upper_right", "middle", "lower_right"),
    "5": ("top", "upper_left", "middle", "lower_right", "bottom"),
    "6": ("top", "upper_left", "middle", "lower_left", "lower_right", "bottom"),
    "7": ("top", "upper_right", "lower_right"),
    "8": ("top", "upper_left", "upper_right", "middle", "lower_left", "lower_right", "bottom"),
    "9": ("top", "upper_left", "upper_right", "middle", "lower_right", "bottom"),
    "_": ("bottom",),
    "p": ("top", "upper_left", "upper_right", "middle", "lower_left"),
}


def _add_person_id_label(scene: Any, gs: Any, person: PersonAgent) -> list[Any]:
    """Add a simple box-stroke id label above a person in the Genesis viewer."""
    if not hasattr(gs, "morphs") or not hasattr(gs.morphs, "Box"):
        return []
    person_id = str(getattr(person, "id", "person"))
    char_width = 0.075
    char_height = 0.14
    spacing = 0.026
    thickness = 0.012
    depth = 0.01
    total_width = len(person_id) * char_width + max(len(person_id) - 1, 0) * spacing
    origin = person.position + np.array([-total_width / 2.0, -0.08, person.height * 0.62], dtype=np.float32)
    label_entities = []

    for char_idx, char in enumerate(person_id):
        char_origin = origin + np.array([char_idx * (char_width + spacing), 0.0, 0.0], dtype=np.float32)
        for segment in _LABEL_SEGMENTS.get(char.lower(), ()):
            center, size = _label_segment_box(char_origin, segment, char_width, char_height, thickness, depth)
            morph = make_visual_morph(gs.morphs.Box, pos=center.tolist(), size=size.tolist())
            entity = _add_entity(scene, morph, name=f"{person_id}_label_{char_idx}_{segment}", gs=gs, color=(0.02, 0.02, 0.02, 1.0))
            label_entities.append(entity)
            person.entities.append(entity)
            person.entity_offsets.append(center - person.position)
    return label_entities


def _label_segment_box(
    origin: np.ndarray,
    segment: str,
    char_width: float,
    char_height: float,
    thickness: float,
    depth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the center and size for one box-stroke label segment."""
    x_mid = char_width / 2.0
    z_mid = char_height / 2.0
    x_left = thickness / 2.0
    x_right = char_width - thickness / 2.0
    z_top = char_height - thickness / 2.0
    z_bottom = thickness / 2.0
    z_upper = char_height * 0.75
    z_lower = char_height * 0.25
    vertical_size = np.array([thickness, depth, char_height / 2.0], dtype=np.float32)
    horizontal_size = np.array([char_width, depth, thickness], dtype=np.float32)
    centers = {
        "top": np.array([x_mid, 0.0, z_top], dtype=np.float32),
        "middle": np.array([x_mid, 0.0, z_mid], dtype=np.float32),
        "bottom": np.array([x_mid, 0.0, z_bottom], dtype=np.float32),
        "upper_left": np.array([x_left, 0.0, z_upper], dtype=np.float32),
        "upper_right": np.array([x_right, 0.0, z_upper], dtype=np.float32),
        "lower_left": np.array([x_left, 0.0, z_lower], dtype=np.float32),
        "lower_right": np.array([x_right, 0.0, z_lower], dtype=np.float32),
    }
    size = vertical_size if "left" in segment or "right" in segment else horizontal_size
    return origin + centers[segment], size

def _assign_ids(items: list[Any], prefix: str) -> None:
    """Attach stable ids to world objects."""
    for idx, item in enumerate(items):
        item.id = f"{prefix}_{idx}"

def main() -> None:
        # Initialize Genesis
    gs.init(backend=gs.cpu)
    # Create scene
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.0, -2.0, 1.0),
            camera_lookat=(0.0, 0.0, 0.3),
            camera_fov=45,
            max_FPS=60,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    # Add entities
    scene.add_entity(gs.morphs.Plane())
    people = [
        PersonAgent(
            position=np.array([x, y, _matchstick_ground_z(height, posture)], dtype=np.float32),
            velocity=np.zeros(3, dtype=np.float32),
            radius=radius,
            height=height,
            color=color,
            head_color=head_color,
            posture=posture,
        )
        for x, y, radius, height, posture, color, head_color in [
            (-0.8, -0.5, 0.20, 1.55, "standing", (0.95, 0.18, 0.20, 1.0), (1.0, 0.78, 0.55, 1.0)),
            (0.0, -0.7, 0.24, 1.75, "wide", (0.18, 0.55, 0.95, 1.0), (0.86, 0.62, 0.42, 1.0)),
            (0.8, -0.5, 0.18, 1.68, "reaching", (0.20, 0.72, 0.34, 1.0), (0.72, 0.48, 0.32, 1.0)),
            (-0.4, 0.35, 0.26, 1.45, "crouch", (0.85, 0.66, 0.16, 1.0), (0.95, 0.70, 0.48, 1.0)),
            (0.45, 0.35, 0.22, 1.85, "lean", (0.58, 0.28, 0.78, 1.0), (0.68, 0.46, 0.30, 1.0)),
        ]
    ]
    _assign_ids(people, "p")
    peoples = []
    for person in people:
        peoples.append(person.add_to_scene(gs, scene, name=person.id))
        _add_person_id_label(scene, gs, person)

    # scene.viewer.follow_entity(drone)
    # Build scene
    scene.build()
    rng = np.random.default_rng(0)
    world_size = np.array([4.0, 4.0, 2.0], dtype=np.float32)
    dt = 0.02
    try:
        while True:
            for person in people:
                person.update(dt, rng, world_size)
            scene.step()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import genesis as gs
    main()
