"""
Human-target MPC: the drone tracks / reaches a human's PREDICTED future trajectory.

This is the "rendezvous" formulation (NOT obstacle avoidance):
the MPC reference at each horizon node j is the human's predicted position P_j,
so the cost  J = sum_j (x_j - P_j)^T Q (x_j - P_j)  drives the drone to the
human's predicted location over the horizon.

Where the human prediction comes from:
  - If a CSV file (--pred path) with columns [t, x, y] exists, it is loaded and
    interpolated. This is where Trajectron++ output plugs in.
  - Otherwise a synthetic walking human is generated so you can run a demo today.

Usage:
  ./run.sh main_human_target.py
  # or with a real prediction file:
  #   python main_human_target.py --pred human_pred.csv
"""
import argparse
import numpy as np
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from quadrotor import Quadrotor3D
from controller import Controller


def synthetic_human(t):
    """A synthetic walking human on the ground plane (z=0).

    Stand-in for Trajectron++ output. Replace by loading a real prediction.
    t can be a scalar or a numpy array. Returns (x, y) with matching shape.
    """
    x = 0.5 * t                      # walks ~0.5 m/s along +x
    y = 1.5 * np.sin(0.25 * t)       # gentle side-to-side
    return x, y


def load_prediction(csv_path):
    """Load a predicted human trajectory from CSV with columns t,x,y.

    Returns a function pred(t) -> (x, y) that linearly interpolates and
    clamps/extrapolates by holding the endpoints.
    """
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    names = data.dtype.names
    tcol = 't' if 't' in names else ('time' if 'time' in names else names[0])
    xcol = next((c for c in ('x', 'x_m', 'pos_x') if c in names), names[1])
    ycol = next((c for c in ('y', 'y_m', 'pos_y') if c in names), names[2])
    t = np.atleast_1d(data[tcol]).astype(float)
    x = np.atleast_1d(data[xcol]).astype(float)
    y = np.atleast_1d(data[ycol]).astype(float)
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]

    def pred(tq):
        tq = np.asarray(tq, dtype=float)
        xq = np.interp(tq, t, x, left=x[0], right=x[-1])
        yq = np.interp(tq, t, y, left=y[0], right=y[-1])
        return xq, yq

    return pred


def run(pred_fn, z_ref=1.0, sim_time=12.0, dt=0.1, N=10, start=(-2.0, -2.0, 1.0), save_npz=None):
    """Receding-horizon MPC that tracks the human's predicted positions."""
    quad = Quadrotor3D()
    quad.pos = np.array(start, dtype=float)

    controller = Controller(quad, t_horizon=2 * N * dt, n_nodes=N,
                            solver_options={"terminal_cost": True, "solver_type": "SQP_RTI"})
    node_dt = (2 * N * dt) / N  # time between MPC shooting nodes

    drone_path = []
    human_path = []
    time_record = []

    n_steps = int(sim_time / dt)
    for i in range(n_steps):
        t_now = i * dt

        # Build the reference over the horizon from the human PREDICTION.
        node_times = t_now + node_dt * np.arange(N + 1)
        hx, hy = pred_fn(node_times)
        goal = np.stack([hx, hy, np.full(N + 1, z_ref)], axis=1)  # (N+1, 3)

        current = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
        t0 = timeit.default_timer()
        thrust = controller.run_optimization(initial_state=current, goal=goal, mode='traj')[:4]
        time_record.append(timeit.default_timer() - t0)
        quad.update(thrust, dt)

        drone_path.append(quad.pos.copy())
        hx0, hy0 = pred_fn(t_now)
        human_path.append([float(hx0), float(hy0), 0.0])

    drone_path = np.array(drone_path)
    human_path = np.array(human_path)

    print("average solve time: {:.5f} s".format(np.mean(time_record)))
    print("max solve time:     {:.5f} s".format(np.max(time_record)))
    final_gap = np.linalg.norm(drone_path[-1, :2] - human_path[-1, :2])
    print("final horizontal drone-human gap: {:.3f} m".format(final_gap))

    if save_npz:
        gap_t = np.linalg.norm(drone_path[:, :2] - human_path[:, :2], axis=1)
        np.savez(save_npz, drone_path=drone_path, human_path=human_path,
                 time=np.arange(len(drone_path)) * dt, gap=gap_t,
                 solve_time=np.array(time_record), dt=dt, N=N, z_ref=z_ref)
        print(f"saved results to {save_npz}")

    # ---- Visualization ----
    fig = plt.figure(figsize=(14, 6))

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot(drone_path[:, 0], drone_path[:, 1], drone_path[:, 2], 'b-', label='drone (MPC)')
    ax.plot(human_path[:, 0], human_path[:, 1], human_path[:, 2], 'r--', label='human (prediction)')
    ax.scatter(*drone_path[0], c='blue', marker='o', s=40, label='drone start')
    ax.scatter(*human_path[-1], c='red', marker='*', s=120, label='human final')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.legend(); ax.set_title('3D: drone reaches human prediction')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(drone_path[:, 0], drone_path[:, 1], 'b-', label='drone (MPC)')
    ax2.plot(human_path[:, 0], human_path[:, 1], 'r--', label='human (prediction)')
    ax2.scatter(drone_path[0, 0], drone_path[0, 1], c='blue', marker='o', label='drone start')
    ax2.scatter(human_path[-1, 0], human_path[-1, 1], c='red', marker='*', s=120, label='human final')
    ax2.set_xlabel('x [m]'); ax2.set_ylabel('y [m]'); ax2.axis('equal')
    ax2.grid(True, alpha=0.3); ax2.legend(); ax2.set_title('Top view (x-y)')

    plt.tight_layout()
    plt.savefig('human_target_result.png', dpi=150, bbox_inches='tight')
    print("saved human_target_result.png")
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', default=None,
                    help='CSV with columns t,x,y of the predicted human trajectory (Trajectron output). '
                         'If omitted, a synthetic human is used.')
    ap.add_argument('--z', type=float, default=1.0, help='Reference flight height [m]')
    ap.add_argument('--sim_time', type=float, default=12.0)
    ap.add_argument('--save', default=None, help='Save results to this .npz path')
    args = ap.parse_args()

    if args.pred:
        pred_fn = load_prediction(args.pred)
        print(f"Loaded human prediction from {args.pred}")
    else:
        pred_fn = lambda t: synthetic_human(t)  # noqa: E731
        print("Using synthetic human trajectory (no --pred given)")

    run(pred_fn, z_ref=args.z, sim_time=args.sim_time, save_npz=args.save)
