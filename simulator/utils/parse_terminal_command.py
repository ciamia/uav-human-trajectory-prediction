import numpy as np
from typing import Any
import select
import sys

def parse_request(request: str, drone_id=0, target_id=0) -> dict[str, Any]:
    request = request.strip()
    parts = request.replace(",", " ").split() # category = parts[0].lower()

    if not parts:
        print("Empty request")
        return None
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
            print("Mission format: mission <hover|circle|follow|land|back> <target_id>")
            return None
            raise ValueError(
                "Mission format: mission <hover|circle|follow|land|back> <target_id>"
            )

        mission_name = parts[1].lower()
        target_id = int(parts[2])

        if mission_name not in mission_names:
            print(f"Invalid mission name: {mission_name}")
            return None
            raise ValueError(f"Invalid mission name: {mission_name}")

        return {
            "which": "drone",
            "type": "mission",
            "name": mission_name,
            "target_id": target_id,
            "drone_id": drone_id,
        }
    # Drone
    if category == "drone":
        if len(parts) < 2:
            print("Drone request missing name")
            return None
            raise ValueError("Drone request missing name")

        name = parts[1].lower()
        if name in drone_requests:
            return {
                "which": "drone",
                "type": "request",
                "name": name,
                "target_id": target_id,
                "drone_id" : drone_id,
                
            }

        if name == "hover":
            return {
                "which": "drone",
                "type": "command",
                "name": "hover",
                "target_id": target_id,
                "drone_id": drone_id,
            }

        if name == "goto":
            values = [float(v) for v in parts[2:]]

            if len(values) == 3:
                cmd = np.hstack([values, 0])
            elif len(values) == 4:
                cmd = np.array(values)
            else:
                print("Drone goto format: drone goto x y z [yaw]")
                return None
                raise ValueError(
                    "Drone goto format: drone goto x y z [yaw]"
                )

            return {
                "which": "drone",
                "type": "command",
                "name": "goto",
                "command": cmd,
                "target_id": target_id,
                "drone_id" : drone_id,
            }

    # People
    if category == "people":
        if len(parts) < 2:
            print("People request missing name")
            return None
            raise ValueError("People request missing name")
        
        name = parts[2].lower()
        target_id = int(parts[1])
        if target_id < 0 and target_id >20: 
            print("People request missing id")
            return None

        if name in people_requests:
            return {
                "which": "people",
                "type": "request",
                "name": name,
                "target_id": target_id,
                "drone_id": drone_id,
            }

        if name == "goto":
            values = [float(v) for v in parts[3:]]

            if len(values) != 3:
                print("People goto format: people goto x y z")
                return None
                raise ValueError(
                    "People goto format: people goto x y z"
                )

            return {
                "which": "people",
                "type": "command",
                "name": "goto",
                "command": np.array(values),
                "target_id": target_id,
                "drone_id": drone_id,
            }

        if name == "velocity":
            values = [float(v) for v in parts[3:]]

            if len(values) != 3:
                print("People velocity format: people velocity vx vy vz")
                return None
                raise ValueError(
                    "People velocity format: people velocity vx vy vz"
                )

            return {
                "which": "people",
                "type": "command",
                "name": "velocity",
                "command": np.array(values),
                "target_id": target_id,
                "drone_id": drone_id,
            }
    print(f"Invalid request: {request}")
    return None
    raise ValueError(f"Invalid request: {request}")


def apply_request(object, request):
    if request in {"state"}:
        print(
            f"position=        {object.get_position().round(3).tolist()} \n"
            f"velocity=        {object.get_velocity().round(3).tolist()} \n"
            )
    elif request in {"pose"}:
        print(f"position={object.get_position().round(3).tolist()}\n"
            )
    elif request in {"velocity"}:
        print(
            f"velocity=        {object.get_velocity().round(3).tolist()} \n"
            )
    else:
        print(f"Please enter a valis request: state/pose/velocity")

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
      