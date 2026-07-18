from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import time

from env.drone import _add_entity, make_visual_morph

def _vec3(value: Any, name: str) -> np.ndarray:
    """Convert a value to a 3D float32 vector."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 3:
        raise ValueError(f"{name} must have exactly 3 values")
    return vector.copy()


@dataclass
class BoxBuilding:
    """Axis-aligned box building used as a simple static obstacle."""

    center: np.ndarray
    size: np.ndarray
    color: tuple[float, float, float, float] = (0.55, 0.58, 0.62, 1.0)
    entity: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.center = _vec3(self.center, "center")
        self.size = _vec3(self.size, "size")

    def add_to_scene(self, gs: Any, scene: Any, name: str = "building") -> Any:
        """Add the building to a Genesis scene as a box."""
        if scene is None or gs is None or not hasattr(gs, "morphs") or not hasattr(gs.morphs, "Box"):
            return None
        morph = make_visual_morph(gs.morphs.Box, pos=self.center.tolist(), size=self.size.tolist())
        self.entity = _add_entity(scene, morph, name=name, gs=gs, color=self.color)
        return self.entity
    
def default_buildings() -> list[BoxBuilding]:
    """Return the minimal Milestone 1 building set: one box building."""
    return [BoxBuilding(center=np.array([0.0, 0.0, 1.0]), size=np.array([1.5, 1.5, 2.0]))]

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
    building1 = scene.add_entity(
        morph=gs.morphs.Box(pos=(-2.0, -1.0, 1.0), size=(1.2, 1.8, 2.0), fixed=True),
        surface=gs.surfaces.Default(color=(0.35, 0.48, 0.78, 1.0)),
        name="building_0",
    )
    building2 = scene.add_entity(
        morph=gs.morphs.Box(pos=(2.0, 1.2, 0.7), size=(2.2, 1.0, 1.4), fixed=True),
        surface=gs.surfaces.Default(color=(0.72, 0.42, 0.28, 1.0)),
        name="building_1",
    )
    # add helper to reset or get the position and size 
    scene.build()

    dt = 0.02
    try:
        while True:
            scene.step()
            time.sleep(dt)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import genesis as gs
    main()



