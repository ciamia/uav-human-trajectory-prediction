# Drone Flow Matching

Simulation-only drone trajectory planning with Genesis and Conditional Flow Matching.

The current main workflow is a serving-drone scene: box buildings, moving matchstick-style people, a hover target, expert trajectory generation, Conditional Flow Matching training, MPC-style tracking, and Genesis evaluation.

Trajectory state:

```text
[x, y, z, vx, vy, vz]
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch
pip install genesis-world
pip install -e ".[dev]"
```

If you are using the local Genesis checkout, the project will prefer the official drone URDF assets at:

```text
/home/yuxia/lab_ws/Genesis/genesis/assets/urdf/drones/
```

## Drone Classes

The project keeps the lightweight planning proxy separate from the Genesis runtime:

```text
env.drone.Drone
```

Use `Drone` for expert/data generation. It stores the trajectory state `[x, y, z, vx, vy, vz]` and simple proxy state helpers. It does not own Genesis scene setup or drone URDF loading.

```text
env.drone.Drone
```

Use `Drone` for evaluation, visualization, and rollout in Genesis. It creates the Genesis scene and the `gs.morphs.Drone` entity from the configured task/drone YAML files.

## Dataset

Generate serving trajectory data:

```bash
python3 scripts/generate_serving_data.py \
  --config configs/genesis.yaml \
  --output outputs/serving_dataset.npz
```

Each sample contains drone start state, hover goal, target person state, people states, future people predictions, building geometry, command embedding, and an expert trajectory.

Dataset generation is offline and does not build a Genesis scene. The expert planner uses lightweight `Drone` state, sampled buildings/people, A* path planning, and spline refinement.

## Train

Train the serving Conditional Flow Matching model:

```bash
python3 scripts/train_serving_fm.py \
  --config configs/genesis.yaml \
  --dataset outputs/serving_dataset.npz \
  --output-dir outputs/serving_fm
```

Checkpoints are saved under:

```text
outputs/serving_fm/checkpoints/
```

The usual checkpoint path is:

```text
outputs/serving_fm/checkpoints/best.pt
```

## Sample And Track

Sample a Flow Matching trajectory, refine it, track it with the MPC controller, and roll out the Genesis drone with `Drone`:

```bash
python3 scripts/sample_serving_trajectory.py \
  --ckpt outputs/serving_fm/checkpoints/best.pt \
  --genesis-config configs/task_config.yaml
```

Run without the viewer:

```bash
python3 scripts/sample_serving_trajectory.py \
  --ckpt outputs/serving_fm/checkpoints/best.pt \
  --genesis-config configs/task_config.yaml \
  --headless
```

The sample script keeps the Genesis viewer open by default. Use `--no-keep-open` to close the viewer after the rollout.

The drone model is configured through the Genesis task config, for example:

```text
configs/task_config.yaml
```

Outputs:

```text
outputs/sample_serving_trajectory.png
outputs/sample_serving_trajectory.mp4
```

The plot compares sampled Flow Matching waypoints, the refined reference, and the MPC-tracked trajectory.

## Mission Interface

The interactive mission entry point is:

```bash
python3 scripts/mission_interface.py \
  --ckpt outputs/serving_fm/checkpoints/best.pt \
  --genesis-config configs/task_config.yaml
```

`scripts/mission_interface.py` starts a Genesis world, selects one controlled drone, listens for terminal requests, plans around the current ESDF map, and sends controller commands while people continue moving in the scene.

Useful terminal commands include:

```text
drone_id 0
drone state
drone pose
drone goto x y z yaw
drone hover
mission follow person_id
mission hover person_id
hover
q
```

For mission commands, the mission name is drone-side behavior and `person_id` selects the target person. `follow` and `hover` compute a safe hover point near the selected person, keep a safe distance from people/buildings, and replan as the target moves.

### Flow Matching In Mission Interface

The mission interface uses `planning.path_traj.TrajectorPlanner` for waypoint generation. The default realtime planner is:

```text
planner.waypoints_flow_matching -> planner.MINCO_S3NU
```

`waypoints_flow_matching` proposes collision-aware waypoints from the current drone position to the current target. `MINCO_S3NU` converts those waypoints into a smooth polynomial trajectory with zero endpoint velocity. The helper `get_realtime_traj(...)` samples that trajectory and converts flat outputs into the full controller reference:

```text
[x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
```

The realtime `follow` loop plans in a background worker. While a new segment is being generated, the controller keeps tracking the latest valid reference or hovers at the last safe goal. Stale plans are ignored when the target has moved too far from the submitted target.

To inspect the realtime reference generation without running a full mission, switch the bottom of `scripts/mission_interface.py` from `run_missions(...)` to `test_traj(...)`. The demo compares A*, RRT-Connect, and Flow Matching waypoint methods, then saves state traces to:

```text
outputs/sample_serving_trajectory_realtime_states.pdf
```

## MPC Tracking

The lightweight offline tracker is in:

```text
control/mpc_controller.py
```

It is a lightweight receding-horizon tracker using a double-integrator model and acceleration limits. It is meant for simulation and visualization only. It does not model propellers, attitude dynamics, or real motor control.

Pipeline:

```text
Flow Matching sample
-> endpoint anchoring
-> spline refinement
-> MPC tracking
-> Drone rollout
```

In `scripts/mission_interface.py`, realtime control uses the full-pose quadrotor controller exposed through `DroneMPCController`:

```text
current state: [p, q, v, w]
goal horizon:  [T, 13] = [p, q, v, w]
command:       four rotor thrusts
```

The loop samples a horizon with `build_reference(...)`, calls:

```text
controller.run_optimization(initial_state=current, goal=goal, mode="traj")
```

and converts the first four thrusts to RPMs with the Genesis drone proxy. The controller timestep is `0.05 s`; the Genesis world may step faster, so the same control command is held for the configured controller/world update ratio.

## Evaluate

Evaluate many random serving scenes:

```bash
python3 scripts/evaluate_serving_policy.py \
  --ckpt outputs/serving_fm/checkpoints/best.pt \
  --output-dir outputs/serving_eval \
  --num-scenes 100
```

Metrics include success rate, collision rate, path length, smoothness, inference time, min obstacle distance, and final hover error.

Evaluation uses the same `Drone` world path as `sample_serving_trajectory.py`, while expert fallback trajectories still use the lightweight expert planner.

## Training Curves

Plot training curves:

```bash
python3 scripts/plot_training_curves.py \
  --log outputs/train_loss.csv \
  --output outputs/training_curves.png
```

For serving training, the JSONL loss log is written under the selected serving output directory.

## Tests

```bash
pytest -q
```
