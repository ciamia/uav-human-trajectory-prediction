
# Drone Serving with Conditional Flow Matching in Genesis

## Overview

This project implements a drone-serving simulation system using:

* Genesis simulator
* Conditional Flow Matching (CFM)
* Learned trajectory generation
* Dynamic obstacle avoidance

The environment contains:

* buildings,
* moving people,
* one serving drone.

A person may issue the command:

```text id="40rzh4"
"coffee"
```

The drone must:

* launch,
* navigate safely,
* avoid collisions,
* hover next to the target person at safe distance `d_s`.

The project is initially simulation-only.

---

# High-Level Objectives

## Main Goal

Train a Conditional Flow Matching model that generates smooth collision-free drone trajectories conditioned on:

* drone start state,
* hover goal,
* people trajectories,
* obstacle map,
* command embedding.

---

# Simulation Stack

## Simulator

Use:

```text id="8e8hvk"
Genesis
```

Do NOT use:

* Gazebo,
* ROS2,
* MPC,
* SLAM,
* real hardware,

during early development.

---

# Environment

The Genesis world contains:

## Static Objects

* ground plane,
* buildings,
* walls/obstacles.

## Dynamic Objects

* people agents,
* serving drone.

---

# People Behavior

Each person:

* has position/velocity,
* occupies space,
* moves continuously.

Supported behaviors:

* random walk,
* routine following.

One person may issue:

```text id="t0g7ed"
command = "coffee"
```

which makes them the drone target.

---

# Drone Task

The drone must:

* detect/select target person,
* compute safe hover goal,
* generate trajectory,
* avoid collisions,
* hover near target.

---

# Hover Goal

The drone should:

* stop at safe distance `d_s`,
* face target person,
* maintain hover altitude,
* avoid hovering inside obstacles.

---

# Trajectory Representation

Trajectory state:

```text id="4cv7ks"
[x, y, z, vx, vy, vz]
```

Optional future extensions:

* acceleration,
* yaw,
* angular velocity.

---

# Conditional Flow Matching

The model learns:

```text id="grjlwm"
v_theta(x_t, t, c)
```

where:

* `x_t` = noisy trajectory,
* `t` = flow time,
* `c` = condition.

---

# Conditioning Information

The condition includes:

* drone start state,
* hover goal,
* target person state,
* people future predictions,
* building geometry,
* command embedding.

---

# Training Data

Training data is generated from:

* heuristic planner,
* A*,
* RRT,
* smoothed trajectories.

Expert trajectories should:

* avoid buildings,
* avoid people,
* remain smooth,
* terminate at hover goal.

---

# Loss Functions

## Base Flow Matching Loss

```text id="1fwyel"
MSE(v_theta(x_t, t, c), target_velocity)
```

## Additional Objectives

* goal loss,
* safe hover distance loss,
* smoothness loss,
* time/path loss,
* static collision loss,
* dynamic people collision loss,
* altitude bounds loss.

---

# Development Strategy

IMPORTANT:

Build incrementally.

Do NOT attempt:

* realistic quadrotor physics,
* MPC,
* SLAM,
* photorealistic humans,
* cameras,

until the minimal simulation system works.

---

# Milestones

---

# MILESTONE 1 — Genesis World

## TASK

Create the Genesis environment.

## IMPLEMENT

Create:

```text
env/people.py
env/buildings.py
env/drone.py
env/setup_genesis_world.py
```

Requirements:

* initialize Genesis scene through `Drone`,
* add ground plane,
* add box buildings,
* add moving people,
* add the Genesis drone entity,
* keep lightweight `Drone` for expert/data generation.

People:

* represented as capsules/cylinders,
* random_walk or routine behavior.

Drone:

* simple rigid proxy,
* no realistic dynamics yet.

## DONE WHEN

The following works:

```text
- 1 building
- 5 people
- 1 drone
- people move
- Genesis renders correctly
- video can be saved
```

Create:

```text
scripts/demo_world.py
```

which runs the demo.

---

# MILESTONE 2 — Coffee Target + Hover Goal

## TASK

Implement target selection and hover goal logic.

## IMPLEMENT

Create:

```text
planning/target_selection.py
planning/hover_goal.py
```

Requirements:

* select person whose command == "coffee",
* compute hover goal at distance d_s,
* avoid placing hover goal inside obstacles,
* enforce hover height.

Return:

* hover position,
* hover yaw facing target.

## DONE WHEN

Demo shows:

* one person marked as coffee target,
* hover goal visualized correctly,
* hover goal remains collision-free.

Create:

```text
scripts/demo_hover_goal.py
```

---

# MILESTONE 3 — Expert Trajectory Generator

## TASK

Generate expert trajectories.

## IMPLEMENT

Create:

```text
data/expert_generator.py
planning/refiner.py
```

Requirements:

* generate trajectories from drone start to hover goal,
* avoid buildings,
* avoid predicted people motion,
* smooth trajectories,
* output:

  * position,
  * velocity,
  * acceleration.

Use:

* A*,
* RRT,
* or heuristic waypoint planning.

## DONE WHEN

The following works:

* trajectory generated,
* avoids buildings,
* avoids people,
* trajectory smooth enough for visualization.

Create:

```text
scripts/demo_expert_trajectory.py
```

---

# MILESTONE 4 — Dataset Generation

## TASK

Generate training dataset.

## IMPLEMENT

Create:

```text
data/serving_dataset.py
scripts/generate_serving_data.py
```

Each sample should contain:

* drone start state,
* hover goal,
* target person state,
* people states,
* future people predictions,
* building geometry,
* command embedding,
* expert trajectory.

## DONE WHEN

Command works:

```bash
python scripts/generate_serving_data.py
```

and saves dataset successfully.

---

# MILESTONE 5 — Flow Matching Model

## TASK

Implement Conditional Flow Matching model.

## IMPLEMENT

Create:

```text
models/flow_transformer.py
models/condition_encoder.py
flow/flow_matching_loss.py
flow/sampler.py
```

Trajectory state:

```text
[x, y, z, vx, vy, vz]
```

Condition includes:

* drone start,
* hover goal,
* target person,
* people predictions,
* buildings,
* command embedding.

Use:

* transformer over trajectory tokens.

## DONE WHEN

The following works:

* forward pass,
* sampler,
* loss computation,
* tensor shapes correct,
* no NaNs.

Create:

```text
scripts/debug_forward_pass.py
```

---

# MILESTONE 6 — Loss Functions

## TASK

Implement trajectory objectives.

## IMPLEMENT

Add:

* goal loss,
* safe hover distance loss,
* smoothness loss,
* path/time loss,
* building collision loss,
* people collision loss,
* altitude loss.

Use soft penalties.

## DONE WHEN

Loss values:

* compute correctly,
* decrease on simple overfit test.

Create:

```text
scripts/debug_losses.py
```

---

# MILESTONE 7 — Training

## TASK

Train the Conditional Flow Matching model.

## IMPLEMENT

Create:

```text
scripts/train_serving_fm.py
```

Requirements:

* load dataset,
* train model,
* save checkpoints,
* validation loop,
* log all losses,
* periodically sample trajectories.

## DONE WHEN

Training:

* runs without crashing,
* loss decreases,
* checkpoints saved.

---

# MILESTONE 8 — Sampling Demo

## TASK

Generate trajectories in Genesis using trained model.

## IMPLEMENT

Create:

```text
scripts/sample_serving_trajectory.py
```

Requirements:

* load checkpoint,
* create Genesis world,
* sample trajectory,
* refine trajectory,
* visualize drone flight,
* save video.

## DONE WHEN

The drone:

* launches,
* flies to target,
* avoids people/buildings,
* hovers safely.

---

# MILESTONE 9 — Evaluation

## TASK

Evaluate planner over many random scenes.

## IMPLEMENT

Create:

```text
scripts/evaluate_serving_policy.py
```

Metrics:

* success rate,
* collision rate,
* path length,
* smoothness,
* inference time,
* min obstacle distance,
* final hover error.

Save:

* json metrics,
* csv metrics,
* plots.

## DONE WHEN

Evaluation runs successfully on:

* 100 scenes,
* without crashing.

---
