import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import timeit

class Trajectory(nn.Module):
    def __init__(self, coeffs, dts):
        super(Trajectory, self).__init__()
        self.coeffs = coeffs # [n_pieces, n_coeffs, n_dims]
        self.dts = dts # [n_pieces]
        # cumulate dts
        self.ts = dts.cumsum(0).detach().numpy()
        self.ts = np.concatenate([[0], self.ts])

    def pos(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        pos = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            pos.append(self._pos(t_ - self.ts[i], self.coeffs[i]))
        pos = torch.cat(pos, 0)
        return pos
    
    def _pos(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute pos for each dimension
        T = torch.cat([t[:, None]**i for i in range(len(coeffs))], 1)
        pos = T @ coeffs
        return pos
    
    def vel(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        vel = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            vel.append(self._vel(t_ - self.ts[i], self.coeffs[i]))
        vel = torch.cat(vel, 0)
        return vel
    
    def _vel(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute vel for each dimension
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-1)], 1)
        vel = T @ (torch.arange(1, len(coeffs)).float()[:, None] * coeffs[1:])
        return vel
    
    def acc(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        acc = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            acc.append(self._acc(t_ - self.ts[i], self.coeffs[i]))
        acc = torch.cat(acc, 0)
        return acc
    
    def _acc(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute acc for each dimension
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-2)], 1)
        acc = T @ (torch.arange(2, len(coeffs)).float()[:, None] * (torch.arange(1, len(coeffs)-1).float()[:, None] * coeffs[2:]))
        return acc
    
    def jerk(self, t):
        # t [n_samples]
        # return [n_samples, n_dims]
        
        # split t into intervals based on t >= ts_i and t < ts_i+1
        jerk = []
        for i in range(len(self.ts)-1):
            t_ = t[(t >= self.ts[i]) & (t < self.ts[i+1])]
            jerk.append(self._jerk(t_ - self.ts[i], self.coeffs[i]))
        jerk = torch.cat(jerk, 0)
        return jerk
    
    def _jerk(self, t, coeffs):
        # t [n_samples]
        # coeffs [n_coeffs, n_dims]
        # return [n_samples, n_dims]
        
        # compute jerk for each dimension
        T = torch.cat([t[:, None]**i for i in range(0, len(coeffs)-3)], 1)
        jerk = T @ (torch.arange(3, len(coeffs)).float()[:, None] * (torch.arange(2, len(coeffs)-1).float()[:, None] * (torch.arange(1, len(coeffs)-2).float()[:, None] * coeffs[3:])))
        return jerk
    



class MINCO_S3NU(nn.Module):
    """
    Minimum-Control-Effort (MINCO) Polynomials
    start and end conditions
    waypoint position and time allocation parameterization
    mapping from waypoint and time parameterization to minimum energy polynomial coefficients with linear complexity
    additional tricks for efficiency, numerical stability and unconstrained optimization

    """
    # initial constraints: position, velocity, acceleration
    # polynomial order: 5 (jerk)
    def __init__(self, N):
        super(MINCO_S3NU, self).__init__()
        self.N = N

    def solve_linear(self, headPVA, tailPVA, points, t):
        # headPVA: [B, 3, 3]
        # tailPVA: [B, 3, 3]
        # points: [B, N-1, 3]
        # t: [B, N, 1]
        # return: [B, 1]

        # create a matrix T [B, N, 6] with t^0, t^1, t^2, t^3, t^4, t^5
        T = torch.cat([torch.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1)
        self.T = T

        assert headPVA.shape[0] == tailPVA.shape[0] == points.shape[0] == t.shape[0], f"Tensors must have the same batch size. Got {headPVA.shape[0]}, {tailPVA.shape[0]}, {points.shape[0]}, {t.shape[0]}"
        
        N = self.N
        B = headPVA.shape[0]
        
        A = torch.zeros(B, 6 * N, 6 * N)
        b = torch.zeros(B, 6 * N, 3)

        # inital constraints
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 2.0
        b[:, 0] = headPVA[:, 0]
        b[:, 1] = headPVA[:, 1]
        b[:, 2] = headPVA[:, 2]

        for i in range(N-1):
            # jerk continuity
            A[:, 6*i+3, 6*i+3] = 6.0
            A[:, 6*i+3, 6*i+4] = 24.0 * T[:, i, 1]
            A[:, 6*i+3, 6*i+5] = 60.0 * T[:, i, 2]
            A[:, 6*i+3, 6*i+9] = -6.0

            # snap continuity
            A[:, 6*i+4, 6*i+4] = 24.0
            A[:, 6*i+4, 6*i+5] = 120.0 * T[:, i, 1]
            A[:, 6*i+4, 6*i+10] = -24.0
            
            # position constaint
            A[:, 6*i+5, 6*i] = T[:, i, 0]
            A[:, 6*i+5, 6*i+1] = T[:, i, 1]
            A[:, 6*i+5, 6*i+2] = T[:, i, 2]
            A[:, 6*i+5, 6*i+3] = T[:, i, 3]
            A[:, 6*i+5, 6*i+4] = T[:, i, 4]
            A[:, 6*i+5, 6*i+5] = T[:, i, 5]

            b[:, 6*i+5] = points[:, i]

            # position continuity
            A[:, 6*i+6, 6*i] = T[:, i, 0]
            A[:, 6*i+6, 6*i+1] = T[:, i, 1]
            A[:, 6*i+6, 6*i+2] = T[:, i, 2]
            A[:, 6*i+6, 6*i+3] = T[:, i, 3]
            A[:, 6*i+6, 6*i+4] = T[:, i, 4]
            A[:, 6*i+6, 6*i+5] = T[:, i, 5]
            A[:, 6*i+6, 6*i+6] = -T[:, i, 0]

            # velocity continuity
            A[:, 6*i+7, 6*i+1] = T[:, i, 0]
            A[:, 6*i+7, 6*i+2] = 2.0 * T[:, i, 1]
            A[:, 6*i+7, 6*i+3] = 3.0 * T[:, i, 2]
            A[:, 6*i+7, 6*i+4] = 4.0 * T[:, i, 3]
            A[:, 6*i+7, 6*i+5] = 5.0 * T[:, i, 4]
            A[:, 6*i+7, 6*i+7] = -1.0

            # acceleration continuity
            A[:, 6*i+8, 6*i+2] = 2.0
            A[:, 6*i+8, 6*i+3] = 6.0 * T[:, i, 1]
            A[:, 6*i+8, 6*i+4] = 12.0 * T[:, i, 2]
            A[:, 6*i+8, 6*i+5] = 20.0 * T[:, i, 3]
            A[:, 6*i+8, 6*i+8] = -2.0

        # final constraints
        # position
        A[:, 6*N-3, 6*N-6] = T[:, N-1, 0]
        A[:, 6*N-3, 6*N-5] = T[:, N-1, 1]
        A[:, 6*N-3, 6*N-4] = T[:, N-1, 2]
        A[:, 6*N-3, 6*N-3] = T[:, N-1, 3]
        A[:, 6*N-3, 6*N-2] = T[:, N-1, 4]
        A[:, 6*N-3, 6*N-1] = T[:, N-1, 5]

        # velocity
        A[:, 6*N-2, 6*N-5] = T[:, N-1, 0]
        A[:, 6*N-2, 6*N-4] = 2.0 * T[:, N-1, 1]
        A[:, 6*N-2, 6*N-3] = 3.0 * T[:, N-1, 2]
        A[:, 6*N-2, 6*N-2] = 4.0 * T[:, N-1, 3]
        A[:, 6*N-2, 6*N-1] = 5.0 * T[:, N-1, 4]

        # acceleration
        A[:, 6*N-1, 6*N-4] = 2.0
        A[:, 6*N-1, 6*N-3] = 6.0 * T[:, N-1, 1]
        A[:, 6*N-1, 6*N-2] = 12.0 * T[:, N-1, 2]
        A[:, 6*N-1, 6*N-1] = 20.0 * T[:, N-1, 3]

        b[:, 6*N-3] = tailPVA[:, 0]
        b[:, 6*N-2] = tailPVA[:, 1]
        b[:, 6*N-1] = tailPVA[:, 2]

        # solve the linear system
        x = torch.linalg.solve(A, b)

        self.x = x

    def get_energy(self):
        energy = 0.0
        for i in range(self.N-1):
            energy += 36.0 * self.x[:, 6*i+3].pow(2).sum() * self.T[:, i, 1] \
                    + 144.0 * (self.x[:, 6*i+4] * self.x[:, 6*i+3]).sum() * self.T[:, i, 2] \
                    + 192.0 * self.x[:, 6*i+4].pow(2).sum() * self.T[:, i, 3] \
                    + 240.0 * (self.x[:, 6*i+5] * self.x[:, 6*i+3]).sum() * self.T[:, i, 3] \
                    + 720.0 * (self.x[:, 6*i+5] * self.x[:, 6*i+4]).sum() * self.T[:, i, 4] \
                    + 720.0 * self.x[:, 6*i+5].pow(2).sum() * self.T[:, i, 5]
        return energy

    # forward pass
    def forward(self, headPVA, tailPVA, points, t):
        self.solve_linear(headPVA, tailPVA, points, t)
        energy = self.get_energy()
        return energy
    
    def get_trajectory(self):
        trajs = []
        for i in range(self.x.shape[0]):
            coeffs = self.x[i].view(self.N, 6, 3)
            traj = Trajectory(coeffs, self.T[i, :, 1])
            trajs.append(traj)
        return trajs    



class MINCO_S3NU_PV(nn.Module):
    # initial + waypoint constraints: position, velocity
    # polynomial order: 5 (jerk)

    def __init__(self, N):
        super(MINCO_S3NU_PV, self).__init__()
        self.N = N

    def solve_linear(self, headPVA, tailPVA, pos, vel, t):
        # headPVA: [B, 3, 3]
        # tailPVA: [B, 3, 3]
        # pos: [B, N-1, 3]
        # vel: [B, N-1, 3]
        # t: [B, N, 1]
        # return: [B, 1]

        # create a matrix T [B, N, 6] with t^0, t^1, t^2, t^3, t^4, t^5
        T = torch.cat([torch.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1)
        self.T = T

        assert headPVA.shape[0] == tailPVA.shape[0] == pos.shape[0] == vel.shape[0] == t.shape[0], f"Tensors must have the same batch size. Got {headPVA.shape[0]}, {tailPVA.shape[0]}, {pos.shape[0]}, {vel.shape[0]}, {t.shape[0]}"
        
        N = self.N
        B = headPVA.shape[0]
        
        A = torch.zeros(B, 6 * N, 6 * N)
        b = torch.zeros(B, 6 * N, 3)

        # inital constraints
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 2.0
        b[:, 0] = headPVA[:, 0]
        b[:, 1] = headPVA[:, 1]
        b[:, 2] = headPVA[:, 2]

        for i in range(N-1):
            # jerk continuity
            A[:, 6*i+3, 6*i+3] = 6.0
            A[:, 6*i+3, 6*i+4] = 24.0 * T[:, i, 1]
            A[:, 6*i+3, 6*i+5] = 60.0 * T[:, i, 2]
            A[:, 6*i+3, 6*i+9] = -6.0
            
            # position constaint
            A[:, 6*i+4, 6*i] = T[:, i, 0]
            A[:, 6*i+4, 6*i+1] = T[:, i, 1]
            A[:, 6*i+4, 6*i+2] = T[:, i, 2]
            A[:, 6*i+4, 6*i+3] = T[:, i, 3]
            A[:, 6*i+4, 6*i+4] = T[:, i, 4]
            A[:, 6*i+4, 6*i+5] = T[:, i, 5]

            b[:, 6*i+4] = pos[:, i]

            # velocity constaint
            A[:, 6*i+5, 6*i+1] = T[:, i, 0]
            A[:, 6*i+5, 6*i+2] = 2.0 * T[:, i, 1]
            A[:, 6*i+5, 6*i+3] = 3.0 * T[:, i, 2]
            A[:, 6*i+5, 6*i+4] = 4.0 * T[:, i, 3]
            A[:, 6*i+5, 6*i+5] = 5.0 * T[:, i, 4]

            b[:, 6*i+5] = vel[:, i]

            # position continuity
            A[:, 6*i+6, 6*i] = T[:, i, 0]
            A[:, 6*i+6, 6*i+1] = T[:, i, 1]
            A[:, 6*i+6, 6*i+2] = T[:, i, 2]
            A[:, 6*i+6, 6*i+3] = T[:, i, 3]
            A[:, 6*i+6, 6*i+4] = T[:, i, 4]
            A[:, 6*i+6, 6*i+5] = T[:, i, 5]
            A[:, 6*i+6, 6*i+6] = -T[:, i, 0]

            # velocity continuity
            A[:, 6*i+7, 6*i+1] = T[:, i, 0]
            A[:, 6*i+7, 6*i+2] = 2.0 * T[:, i, 1]
            A[:, 6*i+7, 6*i+3] = 3.0 * T[:, i, 2]
            A[:, 6*i+7, 6*i+4] = 4.0 * T[:, i, 3]
            A[:, 6*i+7, 6*i+5] = 5.0 * T[:, i, 4]
            A[:, 6*i+7, 6*i+7] = -1.0

            # acceleration continuity
            A[:, 6*i+8, 6*i+2] = 2.0
            A[:, 6*i+8, 6*i+3] = 6.0 * T[:, i, 1]
            A[:, 6*i+8, 6*i+4] = 12.0 * T[:, i, 2]
            A[:, 6*i+8, 6*i+5] = 20.0 * T[:, i, 3]
            A[:, 6*i+8, 6*i+8] = -2.0

        # final constraints
        # position
        A[:, 6*N-3, 6*N-6] = T[:, N-1, 0]
        A[:, 6*N-3, 6*N-5] = T[:, N-1, 1]
        A[:, 6*N-3, 6*N-4] = T[:, N-1, 2]
        A[:, 6*N-3, 6*N-3] = T[:, N-1, 3]
        A[:, 6*N-3, 6*N-2] = T[:, N-1, 4]
        A[:, 6*N-3, 6*N-1] = T[:, N-1, 5]

        # velocity
        A[:, 6*N-2, 6*N-5] = T[:, N-1, 0]
        A[:, 6*N-2, 6*N-4] = 2.0 * T[:, N-1, 1]
        A[:, 6*N-2, 6*N-3] = 3.0 * T[:, N-1, 2]
        A[:, 6*N-2, 6*N-2] = 4.0 * T[:, N-1, 3]
        A[:, 6*N-2, 6*N-1] = 5.0 * T[:, N-1, 4]

        # acceleration
        A[:, 6*N-1, 6*N-4] = 2.0
        A[:, 6*N-1, 6*N-3] = 6.0 * T[:, N-1, 1]
        A[:, 6*N-1, 6*N-2] = 12.0 * T[:, N-1, 2]
        A[:, 6*N-1, 6*N-1] = 20.0 * T[:, N-1, 3]

        b[:, 6*N-3] = tailPVA[:, 0]
        b[:, 6*N-2] = tailPVA[:, 1]
        b[:, 6*N-1] = tailPVA[:, 2]

        # solve the linear system
        x = torch.linalg.solve(A, b)

        self.x = x

    def get_energy(self):
        energy = 0.0
        for i in range(self.N-1):
            energy += 36.0 * self.x[:, 6*i+3].pow(2).sum() * self.T[:, i, 1] \
                    + 144.0 * (self.x[:, 6*i+4] * self.x[:, 6*i+3]).sum() * self.T[:, i, 2] \
                    + 192.0 * self.x[:, 6*i+4].pow(2).sum() * self.T[:, i, 3] \
                    + 240.0 * (self.x[:, 6*i+5] * self.x[:, 6*i+3]).sum() * self.T[:, i, 3] \
                    + 720.0 * (self.x[:, 6*i+5] * self.x[:, 6*i+4]).sum() * self.T[:, i, 4] \
                    + 720.0 * self.x[:, 6*i+5].pow(2).sum() * self.T[:, i, 5]
        return energy

    # forward pass
    def forward(self, headPVA, tailPVA, pos, vel, t):
        self.solve_linear(headPVA, tailPVA, pos, vel, t)
        energy = self.get_energy()
        return energy
    
    def get_trajectory(self):
        trajs = []
        for i in range(self.x.shape[0]):
            coeffs = self.x[i].view(self.N, 6, 3)
            traj = Trajectory(coeffs, self.T[i, :, 1])
            trajs.append(traj)
        return trajs
    


# Monkey-patch the waypoints and headings for straight_trajectory
def straight_trajectory_with_custom_points(quad, discretization_dt, speed, p_start, p_end, yaw_start, yaw_end):
    n = 3
    pos_traj = np.stack([p_start, p_start/2 + p_end/2, p_end], axis=0)  # shape (3, 3)
    att_traj = np.zeros((n, 3))
    att_traj[0, 2] = yaw_start
    att_traj[1, 2] = (yaw_start + yaw_end)/2  # Midpoint yaw
    att_traj[2, 2] = yaw_end

    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)
    return traj, yaw, t_ref

def draw_poly(traj, u_traj, t, target_points=None, target_t=None):
    """
    Plots the generated trajectory of length n with the used keypoints.
    :param traj: Full generated reference trajectory. Numpy array of shape nx13
    :param u_traj: Generated reference inputs. Numpy array of shape nx4
    :param t: Timestamps of the references. Numpy array of length n
    :param target_points: m position keypoints used for trajectory generation. Numpy array of shape 3 x m.
    :param target_t: Timestamps of the reference position keypoints. If not passed, then they are extracted from the
    t vector, assuming constant time separation.
    """

    ders = 2
    dims = 3
    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    y_labels = [r'pos $[m]$', r'vel $[m/s]$', r'acc $[m/s^2]$', r'jer $[m/s^3]$']

    dim_legends = ['x', 'y', 'z']

    if target_t is None and target_points is not None:
        target_t = np.linspace(0, t[-1], target_points.shape[1])

    

    plt_traj = [p_traj, v_traj]

    fig = plt.figure()
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1)

        for dim in range(dims):

            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[dim])

            if d_ord == 0 and target_points is not None:
                plt.plot(target_t, target_points[dim, :], 'bo')

        plt.gca().set_xticklabels([])
        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])

    dim_legends = [['w', 'x', 'y', 'z'], ['x', 'y', 'z']]
    y_labels = [r'att $[quat]$', r'rate $[rad/s]$']
    plt_traj = [a_traj, r_traj]
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1 + ders * 2)
        for dim in range(plt_traj[d_ord].shape[1]):
            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[d_ord][dim])

        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])
        if d_ord == ders - 1:
            plt.xlabel(r'time $[s]$')
        else:
            plt.gca().set_xticklabels([])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    plt.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2])
    if target_points is not None:
        plt.plot(target_points[0, :], target_points[1, :], target_points[2, :], 'bo')
    plt.title('Target position trajectory')
    ax.set_xlabel(r'$p_x [m]$')
    ax.set_ylabel(r'$p_y [m]$')
    ax.set_zlabel(r'$p_z [m]$')

    plt.subplot(ders + 1, 2, (ders + 1) * 2)
    for i in range(u_traj.shape[1]):
        plt.plot(t, u_traj[:, i], label=r'$u_{}$'.format(i))
    plt.grid()
    plt.legend()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel(r'time $[s]$')
    plt.ylabel(r'single thrusts $[N]$')
    plt.title('Control inputs')

    plt.suptitle('Generated polynomial trajectory')

    plt.show()

        
def plot_ret(traj, u_traj, t, target_points=None, target_t=None):
    # Visualization
    # p_traj = traj[:, 0:3]
    # pl_traj = traj[:, 3:6]
    # qc_traj = traj[:, 6:9]
    # a_traj = traj[:, 9:13]
    # v_traj = traj[:, 13:16]
    # vl_traj = traj[:, 16:19]
    # wc_traj = traj[:, 19:22]
    # r_traj = traj[:, 22:25]
    # y_labels = [r'pos $[m]$', r'vel $[m/s]$', r'acc $[m/s^2]$', r'jer $[m/s^3]$',
    #             r'pos_l $[m]$', r'vel_l $[m/s]$', r'acc_l $[m/s^2]$', r'jer_l $[m/s^3]$']
    
    pos_v = np.array(traj[:, 0:3])
    pos_l = np.array(traj[:, 3:6])
    qc = np.array(traj[:, 6:9])
    qv = np.array(traj[:, 9:13])
    vel_v = np.array(traj[:, 13:16])
    vel_l = np.array(traj[:, 16:19])
    cmd = np.array(u_traj)
    # print(pos_v)
    fig = plt.figure()
    ## 3d plot
    # ax = plt.axes(projection='3d')
    # ax.plot(pv_r[:,0], pv_r[:,1], pv_r[:,2], c=[0,1,0], label='goal')
    # ax.plot(pos_v[:,0], pos_v[:,1], pos_v[:,2], c=[1,0,0], label='drone')
    # ax.plot(pos_l[:,0], pos_l[:,1], pos_l[:,2], c=[0,0,1], label='load')
    # ax.axis('equal')
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # ax.legend()

    # seperate plot
    ax1 = fig.add_subplot(2, 2, 1)  # First row, first column
    ax2 = fig.add_subplot(2, 2, 2)  # First row, second column
    ax3 = fig.add_subplot(2, 2, 3)  # Second row, first column
    ax4 = fig.add_subplot(2, 2, 4)  # Second row, second column

    ax1.plot(t, qv[:,0], 'b--', label='qv_w')
    ax1.plot(t, qv[:,1], 'r--', label='qv_x')
    ax1.plot(t, qv[:,2], 'g--', label='qv_y')
    ax1.plot(t, qv[:,3], 'k--', label='qv_z')

    ax2.plot(t, qc[:,0], 'r--', label='qc_x')
    ax2.plot(t, qc[:,1], 'g--', label='qc_y')
    ax2.plot(t, qc[:,2], 'b--', label='qc_z')

    ax3.plot(t, pos_v[:,0], 'r', label='uav_x')
    ax3.plot(t, pos_v[:,1], 'g', label='uav_y')
    ax3.plot(t, pos_v[:,2], 'b', label='uav_z')

    
    ax4.plot(t, pos_l[:,0], 'r', label='load_x')
    ax4.plot(t, pos_l[:,1], 'g', label='load_y')
    ax4.plot(t, pos_l[:,2], 'b', label='load_z')
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    # plt.show()

    fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    ax1 = fig.add_subplot(3, 1, 1)  # First row, full-width
    ax2 = fig.add_subplot(3, 1, 2)  # Second row, left
    ax3 = fig.add_subplot(3, 1, 3)  
    
    ax1.plot(t, cmd[:,0], 'b--', label='f')
    ax1.plot(t, cmd[:,1], 'r--', label='tau_x')
    ax1.plot(t, cmd[:,2], 'g--', label='tau_y')
    ax1.plot(t, cmd[:,3], 'k--', label='tau_z')

    ax2.plot(t, vel_v[:,0], 'r', label='uav_x')
    ax2.plot(t, vel_v[:,1], 'g', label='uav_y')
    ax2.plot(t, vel_v[:,2], 'b', label='uav_z')

    ax3.plot(t, vel_l[:,0], 'r', label='load_vx')
    ax3.plot(t, vel_l[:,1], 'g', label='load_vy')
    ax3.plot(t, vel_l[:,2], 'b', label='load_vz')

    ax1.legend()
    ax2.legend()
    ax3.legend()

    plt.show()

def get_full_traj(poly_coeffs, target_dt, int_dt):

    dims = poly_coeffs.shape[-1]
    full_traj = np.zeros((4, dims, 0))
    t_total = np.zeros((0,))

    if isinstance(target_dt, (float, np.floating)):
        # Adjust target_dt to make it divisible by int_dt
        target_dt = round(target_dt / int_dt) * int_dt

        # Assign target time for each keypoint using homogeneous spacing
        t_vec = np.arange(0, target_dt * (poly_coeffs.shape[0] + 1) - 1e-5, target_dt)

    else:
        # The time between each pair of points is assigned independently
        # First, also adjust each value of the target_dt vector to make it divisible by int_dt
        for i, dt in enumerate(target_dt):
            target_dt[i] = round(dt / int_dt) * int_dt

        t_vec = np.append(np.zeros(1), np.cumsum(target_dt[:-1]))

    for seg in range(len(t_vec) - 1):

        # Select time sampling (linear or quadratic) mode
        tau_dt = np.arange(t_vec[seg], t_vec[seg + 1] + 1e-5, int_dt)

        # Re-normalize time sampling vector between -1 and 1
        t1 = (tau_dt - t_vec[seg]) / (t_vec[seg + 1] - t_vec[seg]) * 2 - 1

        # Compression ratio
        compress = 2 / np.diff(t_vec)[seg]

        # Integrate current segment of trajectory
        traj = np.zeros((4, dims, len(t1)))

        for der_order in range(4):
            for i in range(dims):
                traj[der_order, i, :] = np.polyval(np.polyder(poly_coeffs[seg, :, i], der_order), t1) * (compress ** der_order)

        if seg < len(t_vec) - 2:
            # Remove last sample (will be the initial point of next segment)
            traj = traj[:, :, :-1]
            t_seg = tau_dt[:-1]
        else:
            t_seg = tau_dt

        full_traj = np.concatenate((full_traj, traj), axis=-1)
        t_total = np.concatenate((t_total, t_seg))

    # Separate into p_xyz and yaw trajectories
    yaw_traj = full_traj[:, -1, :]
    full_traj = full_traj[:, :-1, :]

    return full_traj, yaw_traj, t_total


def fit_multi_segment_polynomial_trajectory(p_targets, yaw_targets):

    p_targets = np.concatenate((p_targets, yaw_targets[np.newaxis, :]), 0)
    m = multiple_waypoints(p_targets.shape[1] - 1)

    dims = p_targets.shape[0]
    n_segments = p_targets.shape[1]

    poly_coefficients = np.zeros((n_segments - 1, 8, dims))
    for dim in range(dims):
        b = rhs_generation(p_targets[dim, :])
        poly_coefficients[:, :, dim] = np.fliplr(np.linalg.solve(m, b).reshape(n_segments - 1, 8))

    return poly_coefficients


def matrix_generation(ts):
    b = np.array([[1, ts,  ts**2, ts**3,    ts**4,    ts**5,     ts**6,     ts**7],
                  [0, 1, 2*ts,  3*ts**2,  4*ts**3,  5*ts**4,   6*ts**5,   7*ts**6],
                  [0, 0, 2,     6*ts,    12*ts**2, 20*ts**3,  30*ts**4,  42*ts**5],
                  [0, 0, 0,     6,       24*ts,    60*ts**2, 120*ts**3, 210*ts**4],
                  [0, 0, 0,     0,       24,      120*ts,    360*ts**2, 840*ts**3],
                  [0, 0, 0,     0,       0,       120,       720*ts,   2520*ts**2],
                  [0, 0, 0,     0,       0,       0,         720,      5040*ts],
                  [0, 0, 0,     0,       0,       0,         0,        5040]])

    return b


def multiple_waypoints(n_segments):

    m = np.zeros((8 * n_segments, 8 * n_segments))

    for i in range(n_segments):

        if i == 0:

            # initial condition of the first curve
            b = matrix_generation(-1.0)
            m[8 * i:8 * i + 4, 8 * i:8 * i + 8] = b[:4, :]

            # intermediary condition of the first curve
            b = matrix_generation(1.0)
            m[8 * i + 4:8 * i + 7 + 4, 8 * i:8 * i + 8] = b[:-1, :]

            # starting condition of the second curve position and derivatives
            b = matrix_generation(-1.0)
            m[8 * i + 4 + 1:8 * i + 4 + 7, 8 * (i + 1):8 * (i + 1) + 8] = -b[1:-1, :]
            m[8 * i + 4 + 7:8 * i + 4 + 8, 8 * (i + 1):8 * (i + 1) + 8] = b[0, :]

        elif i != n_segments - 1:

            # starting condition of the ith curve position and derivatives
            b = matrix_generation(1.0)
            m[8 * i + 4:8 * i + 7 + 4, 8 * i:8 * i + 8] = b[:-1, :]

            # end condition of the ith curve position and derivatives
            b = matrix_generation(-1.0)
            m[8 * i + 4 + 1:8 * i + 4 + 7, 8 * (i + 1):8 * (i + 1) + 8] = -b[1:-1, :]
            m[8 * i + 4 + 7:8 * i + 4 + 8, 8 * (i + 1):8 * (i + 1) + 8] = b[0, :]

        if i == n_segments - 1:
            # end condition of the final curve position and derivatives (4 boundary conditions)
            b = matrix_generation(1.0)
            m[8 * i + 4:8 * i + 4 + 4, 8 * i:8 * i + 8] = b[:4, :]

    return m


def fit_single_segment(p_start, p_end, v_start=None, v_end=None, a_start=None, a_end=None, j_start=None, j_end=None):

    # if v_start is None:
    #     v_start = np.array([0, 0])
    # if v_end is None:
    #     v_end = np.array([0, 0])
    # if a_start is None:
    #     a_start = np.array([0, 0])
    # if a_end is None:
    #     a_end = np.array([0, 0])
    # if j_start is None:
    #     j_start = np.array([0, 0])
    # if j_end is None:
    #     j_end = np.array([0, 0])
    if v_start is None:
        v_start = np.array([0, 0, 0])
    if v_end is None:
        v_end = np.array([0, 0, 0])
    if a_start is None:
        a_start = np.array([0, 0, 0])
    if a_end is None:
        a_end = np.array([0, 0, 0])
    if j_start is None:
        j_start = np.array([0, 0, 0])
    if j_end is None:
        j_end = np.array([0, 0, 0])

    poly_coefficients = np.zeros((8, len(p_start)))

    tf = 1
    ti = -1
    A = np.array(([
        [1 * tf ** 7,   1 * tf ** 6,   1 * tf ** 5,   1 * tf ** 4,   1 * tf ** 3,  1 * tf ** 2,  1 * tf ** 1,  1],
        [7 * tf ** 6,   6 * tf ** 5,   5 * tf ** 4,   4 * tf ** 3,   3 * tf ** 2,  2 * tf ** 1,  1,            0],
        [42 * tf ** 5,  30 * tf ** 4,  20 * tf ** 3,  12 * tf ** 2,  6 * tf ** 1,  2,            0,            0],
        [210 * tf ** 4, 120 * tf ** 3, 60 * tf ** 2,  24 * tf ** 1,  6,            0,            0,            0],
        [1 * ti ** 7,   1 * ti ** 6,   1 * ti ** 5,   1 * ti ** 4,   1 * ti ** 3,  1 * ti ** 2,  1 * ti ** 1,  1],
        [7 * ti ** 6,   6 * ti ** 5,   5 * ti ** 4,   4 * ti ** 3,   3 * ti ** 2,  2 * ti ** 1,  1,            0],
        [42 * ti ** 5,  30 * ti ** 4,  20 * ti ** 3,  12 * ti ** 2,  6 * ti ** 1,  2,            0,            0],
        [210 * ti ** 4, 120 * ti ** 3, 60 * ti ** 2,  24 * ti ** 1,  6,            0,            0,            0]]))

    A = np.tile(A[:, :, np.newaxis], (1, 1, len(p_start)))

    b = np.concatenate((p_end, v_end, a_end, j_end, p_start, v_start, a_start, j_start)).reshape(8, -1)

    for i in range(len(p_start)):
        poly_coefficients[:, i] = np.linalg.inv(A[:, :, i]).dot(np.array(b[:, i]))

    return np.expand_dims(poly_coefficients, 0)


def rhs_generation(x):
    n = x.shape[0] - 1

    big_x = np.zeros((8 * n))
    big_x[:4] = np.array([x[0], 0, 0, 0]).T
    big_x[-4:] = np.array([x[-1], 0, 0, 0]).T

    for i in range(1, n):
        big_x[8 * (i - 1) + 4:8 * (i - 1) + 8 + 4] = np.array([x[i], 0, 0, 0, 0, 0, 0, x[i]]).T

    return big_x

def flatOutputToStateControl(params, y, y_psi):
    pos = y[..., 0:3]
    vel = y[..., 3:6]
    acc = y[..., 6:9]
    jrk = y[..., 9:12]
    snp = y[..., 12:15]

    psi = y_psi[..., 0]
    psi_dot = y_psi[..., 1]
    psi_ddot = y_psi[..., 2]

    t_vec = acc + params.g_vec
    t = torch.norm(t_vec, dim=-1, keepdim=True)
    z_b = t_vec / t

    u_1 = params.m * t

    z_w = torch.stack(
        (torch.zeros_like(psi), torch.zeros_like(psi), torch.ones_like(psi)), dim=-1
    )

    x_c = torch.stack(
        (torch.cos(psi), torch.sin(psi), torch.zeros_like(psi)), dim=-1
    )
    y_c = torch.cross(z_w, x_c, dim=-1)
    x_b_ = torch.cross(y_c, z_b, dim=-1)
    x_b = x_b_ / torch.norm(x_b_, dim=-1, keepdim=True)
    y_b = torch.cross(z_b, x_b, dim=-1)

    # rotation matrix body to world
    R_b = torch.stack((x_b, y_b, z_b), dim=2)

    t_dot = torch.sum(z_b * jrk, dim=-1, keepdim=True)
    h_w = (jrk - t_dot * z_b) / t

    p = torch.sum(-h_w * y_b, dim=-1, keepdim=True)
    q = torch.sum(h_w * x_b, dim=-1, keepdim=True)
    psi_dot_vec = torch.stack(
        (torch.zeros_like(psi_dot), torch.zeros_like(psi_dot), psi_dot), dim=-1
    )
    # r = torch.sum(psi_dot_vec * z_b, dim=-1, keepdim=True)

    ### Depending on how yaw, pitch and roll is defined (following Mellinger's Phd 2011)
    # # transformation matrix from euler angles to angular velocity in body frame
    # T = torch.stack((x_c, y_b, z_w), dim=2)
    # # transformation matrix from angular velocity in body frame to euler angles
    # T_B2E = torch.bmm(torch.inverse(T), R_b)
    # r = (psi_dot.unsqueeze(-1) - T_B2E[...,2,0:1] * p - T_B2E[...,2,1:2] * q) / T_B2E[...,2,2:3]

    ### following the standart aerospace convention (roll, pitch, yaw)
    # transformation matrix from euler angles to angular velocity in body frame
    T = torch.stack((x_b, y_c, z_w), dim=2)
    # transformation matrix from angular velocity in body frame to euler angle rates
    T_B2E = torch.bmm(torch.inverse(T), R_b)
    T_B2E_ = torch.linalg.inv(torch.bmm(R_b.transpose(1, 2), T))
    # r = torch.sum(psi_dot_vec * z_b, dim=-1, keepdim=True) # wrong
    r = (
        psi_dot.unsqueeze(-1) - T_B2E[..., 2, 0:1] * p - T_B2E[..., 2, 1:2] * q
    ) / T_B2E[..., 2, 2:3]

    # r_ = 1 / torch.norm(x_b_, dim=-1, keepdim=True) * (psi_dot * torch.sum(x_c * x_b, dim=-1, keepdim=True) + q * torch.sum(y_c * z_b, dim=-1, keepdim=True))

    # angular velocity in body frame
    w_B = torch.cat((p, q, r), dim=-1)
    # angular velocity in world frame
    w_W = torch.bmm(R_b, w_B.unsqueeze(-1)).squeeze(-1)
    # euler angle rates
    eul_dot = torch.bmm(T_B2E, w_B.unsqueeze(-1)).squeeze(-1)

    w_w_zb = torch.cross(w_W, torch.cross(w_W, z_b, dim=-1), dim=-1)
    t_ddot = torch.sum(z_b * (snp - w_w_zb * t), dim=-1, keepdim=True)
    h_w_dot = (
        snp - t_ddot * z_b - 2 * torch.cross(w_W, t_dot * z_b, dim=-1)
    ) / t - w_w_zb

    p_dot = torch.sum(-h_w_dot * y_b, dim=-1, keepdim=True)
    q_dot = torch.sum(h_w_dot * x_b, dim=-1, keepdim=True)
    psi_ddot_vec = torch.stack(
        (torch.zeros_like(psi_ddot), torch.zeros_like(psi_ddot), psi_ddot), dim=-1
    )

    ### following Mellinger's Phd 2011 but using standard aerospace convention (roll, pitch, yaw)
    A = T_B2E
    b = torch.bmm(T_B2E, torch.cross(w_B, w_B, dim=-1).unsqueeze(-1)).squeeze(
        -1
    ) - torch.bmm(
        torch.inverse(T),
        (
            torch.cross(w_W, x_b, dim=-1) * eul_dot[..., 0:1]
            + torch.cross(psi_dot_vec, y_c, dim=-1) * eul_dot[..., 1:2]
        ).unsqueeze(-1),
    ).squeeze(
        -1
    )
    r_dot = (
        psi_ddot.unsqueeze(-1)
        - A[..., 2, 0:1] * p_dot
        - A[..., 2, 1:2] * q_dot
        - b[..., 2:3]
    ) / A[..., 2, 2:3]

    w_B_dot = torch.cat((p_dot, q_dot, r_dot), dim=-1)

    M = w_B_dot @ params.J + torch.cross(w_B, w_B @ params.J, dim=-1)

    u = torch.cat((u_1, M), dim=-1)

    # x = torch.cat((pos, vel, rot2quat(R_b), w_B), dim=-1)
    x = torch.cat((pos, vel, x_b, y_b, z_b, w_B), dim=-1)

    traj =  torch.cat((x, u), dim=-1)
    # return x, u, t_dot * params.m, t_ddot * params.m

    return traj


if __name__ == "__main__":

    # if 0:
        # # Define start/end positions and headings (yaw in radians)
        # # Usage:
        # p_start = np.array([0, 0, 0])
        # p_end = np.array([1, 3, 2])
        # yaw_start = np.deg2rad(50)
        # yaw_end = np.deg2rad(180)

        # traj, yaw_traj, t = straight_trajectory_with_custom_points(None, 0.1, 1, p_start, p_end, yaw_start, yaw_end)

        # # Prepare for plotting
        # N = traj.shape[2]
        # traj_for_plot = np.zeros((N, 13))
        # traj_for_plot[:, 0:3] = traj[0].T  # position
        # traj_for_plot[:, 7:10] = traj[1].T # velocity

        # # Convert yaw to quaternion (roll=0, pitch=0)
        # quats = R.from_euler('z', yaw_traj[0], degrees=False).as_quat()  # (N, 4): x, y, z, w
        # traj_for_plot[:, 3] = quats[:, 3]  # w
        # traj_for_plot[:, 4] = quats[:, 0]  # x
        # traj_for_plot[:, 5] = quats[:, 1]  # y
        # traj_for_plot[:, 6] = quats[:, 2]  # z

        # # Body rates (yaw rate)
        # traj_for_plot[:, 10] = yaw_traj[1]  # yaw rate

        # # Fake control input trajectory (zeros)
        # u_traj = np.zeros((len(t), 4))

        # # Plot
        # target_points = np.stack([p_start, p_end], axis=1)
        # draw_poly(traj_for_plot, u_traj, t, target_points=target_points)
    
    if 1:
        # inital and final position, velocity, acceleration
        headPVA = torch.tensor([[0.2, 0.2, 0.0], [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32).unsqueeze(0)
        tailPVA = torch.tensor([[0.0, 0.0, 0.0], [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32).unsqueeze(0)

        # waypoints
        pos = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]], dtype=torch.float32)
        vel = torch.tensor([[[np.sqrt(2)/2, np.sqrt(2)/2, 0.0], [-np.sqrt(2)/2, np.sqrt(2)/2, 0.0], [-np.sqrt(2)/2, -np.sqrt(2)/2, 0.0]]], dtype=torch.float32)

        # timestamps
        t = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]], dtype=torch.float32)

        start = timeit.default_timer()
        # 4 piecewise polynomial
        model = MINCO_S3NU_PV(4)

        # energy
        energy = model(headPVA, tailPVA, pos, vel, t)
        trajs = model.get_trajectory()
        ti = timeit.default_timer() - start
        print(ti)
        # sample points
        traj = trajs[0]
        t = torch.linspace(0, 1.0-0.01, 100)
        pos_ = traj.pos(t)
        vel_ = traj.vel(t)
        acc_ = traj.acc(t)
        jerk_ = traj.jerk(t)

        import matplotlib.pyplot as plt

        # plot x-y
        plt.plot(pos_[:, 0], pos_[:, 1])
        plt.scatter(pos[0, :, 0], pos[0, :, 1], c='r')
        # initial and final position
        plt.scatter(headPVA[0, 0, 0], headPVA[0, 0, 1], c='g')
        plt.scatter(tailPVA[0, 0, 0], tailPVA[0, 0, 1], c='b')
        plt.axis('equal')
        plt.show()

        plt.figure()
        plt.title("jerk_")
        plt.plot(t, vel_.norm(dim=-1))
        # plot vertical lines at t
        for i in range(t.shape[1]):
            plt.axvline(t[0,0:i].sum(), c='r')
        plt.show()
