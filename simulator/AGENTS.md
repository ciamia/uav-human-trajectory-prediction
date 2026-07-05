# AGENTS.md

# AGENTS.md

This file defines coding and development rules for Codex and contributors.

---

# General Principles

* Keep implementations minimal and incremental.
* Prefer working code over over-engineered code.
* Avoid introducing unnecessary abstractions early.
* Prioritize readability and debuggability.
* Ensure all code runs before finishing a task.

---

# Scope Rules

Unless explicitly requested:

DO NOT implement:

* ROS2,
* MPC,
* SLAM,
* cameras,
* photorealistic rendering,
* real hardware integration,
* distributed training,
* cloud infrastructure.

The project is initially:

* simulation-only,
* Genesis-only,
* trajectory-planning-focused.

---

# Task Rules

For each task:

* Modify only files relevant to the task.
* Avoid unrelated refactors.
* Keep diffs small and reviewable.
* Add docstrings.
* Add type hints.
* Add tests when appropriate.
* Ensure imports remain clean.

---

# Development Workflow

Preferred workflow:

```text id="tjlwmw"
task
→ implement
→ run/test
→ review
→ commit
→ next task
```

Do NOT attempt to implement multiple milestones simultaneously unless explicitly requested.

---

# Simulation Rules

Use:

* Genesis simulator,
* simplified people,
* simplified drone proxy,
* simplified buildings.

Do NOT implement realistic quadrotor aerodynamics yet.

People may initially be:

* cylinders,
* capsules,
* simple rigid bodies.

Buildings may initially be:

* boxes,
* simple meshes.

---

# Drone Rules

The drone should initially:

* behave as a trajectory-following rigid body,
* use waypoint trajectories,
* not simulate low-level motor dynamics.

Trajectory state:

```text id="9e2h6y"
[x, y, z, vx, vy, vz]
```

---

# Planning Rules

Trajectory planning should prioritize:

* collision avoidance,
* smoothness,
* stable hover behavior,
* simple implementations first.

Preferred initial planners:

* A*,
* RRT,
* heuristic waypoint planning.

---

# Machine Learning Rules

Use:

* PyTorch,
* Conditional Flow Matching,
* Transformer-based trajectory models.

Model objectives:

* smooth trajectories,
* collision-free trajectories,
* hover near target person,
* maintain safe distance.

---

# Loss Rules

Implement:

* Flow Matching loss,
* goal loss,
* safe hover distance loss,
* smoothness loss,
* collision losses,
* path/time loss.

Use soft penalties first.

Avoid constrained optimization initially.

---

# Testing Rules

Every milestone should include:

* runnable demo,
* sanity checks,
* minimal tests.

Before finishing:

* verify imports,
* run scripts,
* check tensor shapes,
* check for NaNs.

---

# File Organization Rules

Preferred directories:

```text id="zqit5p"
configs/
env/
data/
models/
flow/
planning/
safety/
scripts/
tests/
```

Avoid:

* giant monolithic files,
* deeply nested abstractions,
* premature optimization.

---

# Coding Style

Use:

* explicit variable names,
* simple APIs,
* small functions,
* clear tensor shapes.

Prefer:

* clarity,
* maintainability,
* incremental correctness.

---
