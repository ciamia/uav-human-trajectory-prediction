"""
Closed-loop rolling MPC over the WHOLE video.

This is the receding-horizon version the advisor asked for: instead of a single
frozen prediction, we step through every sliding-window prediction made by
Trajectron++ along the video. At each control instant the MPC reads the *current*
window's prediction as its horizon reference, so the drone continuously chases the
human's freshly re-predicted future position.

Pipeline of time:
  Trajectron step  : dt_pred = 0.4 s   (one new prediction per step)
  MPC control step : dt_mpc  = 0.1 s   (4 control updates per Trajectron step)
  MPC horizon      : N nodes, node_dt = 0.2 s  (2 s look-ahead)

Inputs : ../vio_pipeline/vio_all_predictions.npz   (from vio_predict_all.py)
Output : ../vio_pipeline/vio_mpc_rolling.npz

Run from the cuav_nmpc folder in the `mpc` conda env (acados on the path):
    ACADOS_SOURCE_DIR=~/acados DYLD_LIBRARY_PATH=~/acados/lib:$DYLD_LIBRARY_PATH \
        ~/miniconda3/envs/mpc/bin/python vio_mpc_rolling.py
"""
import os
import numpy as np
import timeit

from quadrotor import Quadrotor3D
from controller import Controller

HERE = os.path.dirname(os.path.abspath(__file__))
PRED_NPZ = os.path.join(HERE, "..", "vio_pipeline", "vio_all_predictions.npz")
OUT_NPZ = os.path.join(HERE, "..", "vio_pipeline", "vio_mpc_rolling.npz")

Z_REF = 1.5
DT_MPC = 0.1
N = 10


def make_window_interp(t0, pred_xy, dt_pred):
    """Interpolator for one window's prediction, in ABSOLUTE time.

    pred_xy[k] is the human's predicted position at time t0 + (k+1)*dt_pred.
    Holds the endpoints outside the predicted horizon.
    """
    ph = pred_xy.shape[0]
    tp = t0 + dt_pred * np.arange(1, ph + 1)

    def f(tq):
        tq = np.asarray(tq, dtype=float)
        xq = np.interp(tq, tp, pred_xy[:, 0], left=pred_xy[0, 0], right=pred_xy[-1, 0])
        yq = np.interp(tq, tp, pred_xy[:, 1], left=pred_xy[0, 1], right=pred_xy[-1, 1])
        return xq, yq

    return f


def main():
    d = np.load(PRED_NPZ)
    track_t = d["track_t"]
    track_xy = d["track_xy"]
    win_t0 = d["win_t0"]
    pred_xy = d["pred_xy"]            # (W, PH, 2)
    dt_pred = float(d["dt"])
    W = len(win_t0)
    print(f"{W} sliding-window predictions, track {track_xy.shape[0]} steps "
          f"({track_t[-1]:.1f}s)")

    # actual human position in absolute time (for the gap / plots)
    def human_actual(tq):
        tq = np.asarray(tq, dtype=float)
        return (np.interp(tq, track_t, track_xy[:, 0]),
                np.interp(tq, track_t, track_xy[:, 1]))

    quad = Quadrotor3D()
    hx0, hy0 = human_actual(win_t0[0])
    quad.pos = np.array([float(hx0) - 3.0, float(hy0) - 3.0, Z_REF])

    # terminal_cost=True adds the advisor's terminal term (x_N - P_N)^T Q_N (x_N - P_N),
    # i.e. pin the end of the horizon to the mean final prediction, on top of the
    # stage tracking sum_i (x_i - P_i)^T Q (x_i - P_i).
    controller = Controller(quad, t_horizon=2 * N * DT_MPC, n_nodes=N,
                            solver_options={"terminal_cost": True, "solver_type": "SQP_RTI"})
    node_dt = (2 * N * DT_MPC) / N

    sub = int(round(dt_pred / DT_MPC))  # control updates per Trajectron step
    drone_path, human_path, ref_path, t_log, solve_t = [], [], [], [], []

    for w in range(W):
        t0 = win_t0[w]
        pred_fn = make_window_interp(t0, pred_xy[w], dt_pred)
        for s in range(sub):
            t_now = t0 + s * DT_MPC
            node_times = t_now + node_dt * np.arange(N + 1)
            hx, hy = pred_fn(node_times)
            goal = np.stack([hx, hy, np.full(N + 1, Z_REF)], axis=1)  # (N+1,3)

            cur = np.concatenate([quad.pos, quad.angle, quad.vel, quad.a_rate])
            tic = timeit.default_timer()
            thrust = controller.run_optimization(initial_state=cur, goal=goal, mode='traj')[:4]
            solve_t.append(timeit.default_timer() - tic)
            quad.update(thrust, DT_MPC)

            ax, ay = human_actual(t_now)
            drone_path.append(quad.pos.copy())
            human_path.append([float(ax), float(ay), 0.0])
            ref_path.append([float(hx[0]), float(hy[0]), Z_REF])
            t_log.append(t_now)

    drone_path = np.array(drone_path)
    human_path = np.array(human_path)
    ref_path = np.array(ref_path)
    t_log = np.array(t_log)
    gap = np.linalg.norm(drone_path[:, :2] - human_path[:, :2], axis=1)

    print(f"steps={len(drone_path)}  mean solve={np.mean(solve_t)*1000:.2f} ms")
    print(f"gap: start {gap[0]:.2f} m -> mean(after 3s) {gap[t_log>t_log[0]+3].mean():.2f} m "
          f"-> final {gap[-1]:.2f} m")

    np.savez(OUT_NPZ, drone_path=drone_path, human_path=human_path,
             ref_path=ref_path, time=t_log, gap=gap,
             solve_time=np.array(solve_t),
             track_t=track_t, track_xy=track_xy,
             win_t0=win_t0, pred_xy=pred_xy, dt_pred=dt_pred, z_ref=Z_REF)
    print(f"saved {OUT_NPZ}")


if __name__ == "__main__":
    main()
