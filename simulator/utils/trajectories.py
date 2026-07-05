""" Trajectory generation functions. For the circle, lemniscate and random trajectories.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from utils.utils import undo_quaternion_flip, rotation_matrix_to_quat, unit_quat
from utils.utils import quaternion_inverse, q_dot_q
from utils.trajectory_generator import draw_poly, plot_ret, get_full_traj, \
                                            fit_multi_segment_polynomial_trajectory, \
                                            straight_trajectory_with_custom_points
from utils.keyframe_3d_gen import random_periodical_trajectory
from configs.configuration_parameters import DirectoryConfig
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import yaml
import json
import torch
from scipy.spatial.transform import Rotation as R


def get_trajectory_X(quad, traj_X, opt_dt, speed=1, seed=0, map_limits=None, plot=False, 
                    p_start=None, p_end=None, heading=None, CUAV=False, 
                    radius=1.0, z=1.0, lin_acc=1.0, clockwise=True, yawing=False, v_max=1.0):
    """ 
    Generates a trajectory for the quadrotor, either a straight line or a circle, depending on the parameters.
    """
    if traj_X == "p2p":
        if p_start is None or p_end is None or heading is None:
            raise ValueError("For point-to-point trajectory, p_start, p_end and heading must be provided.") 
        p_start = np.array(p_start[:3])
        p_end = np.array(p_end[:3])
        yaw_start = np.deg2rad(heading[0])
        yaw_end = np.deg2rad(heading[1])
        reference_traj, t_ref, reference_u =  straight_p2p_trajectory(quad, opt_dt, speed, p_start, p_end, yaw_start, yaw_end, CUAV=CUAV)
    elif traj_X == "flyover_collect": 
        reference_traj, t_ref, reference_u =  flyover_trajectory_collect(quad, opt_dt, seed=seed, speed=speed,
                                                                 flyover_box_name=map_limits, CUAV=CUAV)
    elif traj_X == "flyover":     
        reference_traj, t_ref, reference_u =  flyover_trajectory(quad, opt_dt, seed=seed, speed=speed,
                                                         flyover_box_name=map_limits, CUAV=CUAV)  
    elif traj_X == "random":
        reference_traj, t_ref, reference_u =  random_trajectory(quad, opt_dt, seed=seed, speed=speed, map_name=map_limits,
                                                        plot=plot, CUAV=CUAV)
    elif traj_X == "loop":
        reference_traj, t_ref, reference_u =  loop_trajectory(quad, opt_dt, v_max=v_max, radius=radius, z=z,
                                                      lin_acc=lin_acc, clockwise=clockwise, map_name=map_limits,
                                                      yawing=yawing, plot=plot, CUAV=CUAV)
    elif traj_X == "lemniscate":
        reference_traj, t_ref, reference_u =  lemniscate_trajectory(quad, opt_dt, v_max=v_max, radius=radius, z=z,
                                                            lin_acc=lin_acc, clockwise=clockwise, map_name=map_limits,
                                                            yawing=yawing, plot=plot, CUAV=CUAV)
    elif traj_X == "lemniscate":
        reference_traj, t_ref, reference_u =  lemniscate_trajectory(quad, opt_dt, v_max=v_max, radius=radius, z=z,
                                                            lin_acc=lin_acc, clockwise=clockwise, map_name=map_limits,
                                                            yawing=yawing, plot=plot, CUAV=CUAV)
    elif traj_X == "trefoil_knot":
        reference_traj, t_ref, reference_u =  trefoil_knot_trajectory(quad, opt_dt, v_max=v_max, map_name=map_limits,
                                                            yawing=yawing, plot=plot, CUAV=CUAV)
    elif traj_X == "eight_shape":
        reference_traj, t_ref, reference_u =  eight_shape_trajectory(quad, opt_dt, v_max=v_max, map_name=map_limits,
                                                            yawing=yawing, plot=plot, CUAV=CUAV)
    elif traj_X == "circle":
        reference_traj, t_ref, reference_u =  circle_trajectory(quad, opt_dt, v_max=v_max, map_name=map_limits,
                                                            yawing=yawing, plot=plot, CUAV=CUAV)
    else:   
        raise ValueError("Unknown trajectory type: %s" % traj_X)
    

    return reference_traj, t_ref, reference_u
    

def p2p_traj(dt, p_start, p_end, yaw_start, yaw_end, speed=2):
    traj, yaw_traj, t = straight_trajectory_with_custom_points(None, dt, 1, p_start, p_end, yaw_start, yaw_end)

    # Prepare for plotting
    N = traj.shape[2]
    traj_for_plot = np.zeros((N, 13))
    traj_for_plot[:, 0:3] = traj[0].T  # position
    traj_for_plot[:, 7:10] = traj[1].T # velocity

    # Convert yaw to quaternion (roll=0, pitch=0)
    quats = R.from_euler('z', yaw_traj[0], degrees=False).as_quat()  # (N, 4): x, y, z, w
    traj_for_plot[:, 3] = quats[:, 3]  # w
    traj_for_plot[:, 4] = quats[:, 0]  # x
    traj_for_plot[:, 5] = quats[:, 1]  # y
    traj_for_plot[:, 6] = quats[:, 2]  # z

    # Body rates (yaw rate)
    traj_for_plot[:, 10] = yaw_traj[1]  # yaw rate

    return traj_for_plot

def waypoints_traj(dt, pos_traj, att_traj, speed=2):

    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=dt)
    # Prepare for plotting
    N = traj.shape[2]
    traj_for_plot = np.zeros((N, 13))
    traj_for_plot[:, 0:3] = traj[0].T  # position
    traj_for_plot[:, 7:10] = traj[1].T # velocity

    # Convert yaw to quaternion (roll=0, pitch=0)
    quats = R.from_euler('z', yaw_traj[0], degrees=False).as_quat()  # (N, 4): x, y, z, w
    traj_for_plot[:, 3] = quats[:, 3]  # w
    traj_for_plot[:, 4] = quats[:, 0]  # x
    traj_for_plot[:, 5] = quats[:, 1]  # y
    traj_for_plot[:, 6] = quats[:, 2]  # z

    # Body rates (yaw rate)
    traj_for_plot[:, 10] = yaw_traj[1]  # yaw rate

    return traj_for_plot

def check_trajectory(trajectory, inputs, tvec, plot=False, CUAV=False):
    """

    @param trajectory:
    @param inputs:
    @param tvec:
    @param plot:
    @return:
    """

    print(f"shape of trajectory: {trajectory.shape}")
    print(f"shape of tvec: {tvec.shape}")
    dt = np.expand_dims(np.gradient(tvec, axis=0), axis=1)
    print(f"Checking trajectory integrity with dt as: {dt[0,0]}...")
    numeric_derivative = np.gradient(trajectory, axis=0, edge_order=2) / dt
    gravity = 9.81
    

    errors = np.zeros((dt.shape[0], 3))

    num_bodyrates = []

    for i in range(dt.shape[0]):
        if CUAV:
            # For CUAV, the trajectory has 25 components: 
            numeric_velocity = numeric_derivative[i, 3:6]
            analytic_velocity = trajectory[i, 16:19]
            numeric_thrust = numeric_derivative[i, 13:16] + np.array([0.0, 0.0, gravity])
            analytic_attitude = trajectory[i, 9:13]
            numeric_bodyrates = 2.0 * q_dot_q(quaternion_inverse(trajectory[i, 9:13]), numeric_derivative[i, 9:13])[1:]
            analytic_bodyrates = trajectory[i, 22:25]
            vel_s = 16
            bodyr_s = 22
        else:
            # For quadrotor, the trajectory has 13 components: [p_xyz, q_wxyz, v_xyz, w_xyz]
            numeric_velocity = numeric_derivative[i, 0:3]
            analytic_velocity = trajectory[i, 7:10]
            numeric_thrust = numeric_derivative[i, 7:10] + np.array([0.0, 0.0, gravity])
            analytic_attitude = trajectory[i, 3:7]
            numeric_bodyrates = 2.0 * q_dot_q(quaternion_inverse(trajectory[i, 3:7]), numeric_derivative[i, 3:7])[1:]
            analytic_bodyrates = trajectory[i, 10:13]
            vel_s = 7
            bodyr_s = 10

        # 1) check if velocity is consistent with position
        errors[i, 0] = np.linalg.norm(numeric_velocity - analytic_velocity)
        if not np.allclose(analytic_velocity, numeric_velocity, atol=1e-2, rtol=1e-2):
            print(f"check CUAV trajectory {CUAV} at {i}")
            print(f"inconsistent linear velocity, error: {errors[i, 0]:.5f}")
            print(numeric_velocity)
            print(analytic_velocity)
            return False

        # 2) check if attitude is consistent with acceleration
        numeric_thrust = numeric_thrust / np.linalg.norm(numeric_thrust)
        if np.abs(np.linalg.norm(analytic_attitude) - 1.0) > 1e-6:
            print(f"quaternion does not have unit norm at {i}!")
            print(analytic_attitude)
            print(np.linalg.norm(analytic_attitude))
            # return False

        # e_z = np.array([0.0, 0.0, 1.0])
        # q_w = 1.0 + np.dot(e_z, numeric_thrust)
        # q_xyz = np.cross(e_z, numeric_thrust)
        # numeric_attitude = 0.5 * np.array([q_w] + q_xyz.tolist())
        # numeric_attitude = numeric_attitude / np.linalg.norm(numeric_attitude)
        # # the two attitudes can only differ in yaw --> check x,y component
        # q_diff = q_dot_q(quaternion_inverse(analytic_attitude), numeric_attitude)
        # errors[i, 1] = np.linalg.norm(q_diff[1:3])
        # if not np.allclose(q_diff[1:3], np.zeros(2, ), atol=1e-3, rtol=1e-3):
        #     print("Attitude and acceleration do not match!")
        #     print(analytic_attitude)
        #     print(numeric_attitude)
        #     print(q_diff)
        #     return False

        # 3) check if bodyrates agree with attitude difference
        # num_bodyrates.append(numeric_bodyrates)
        # errors[i, 2] = np.linalg.norm(numeric_bodyrates - analytic_bodyrates)
        # if not np.allclose(numeric_bodyrates, analytic_bodyrates, atol=0.05, rtol=0.05):
        #     print("inconsistent angular velocity")
        #     print(numeric_bodyrates)
        #     print(analytic_bodyrates)
        #     return False

    print("Trajectory check successful")
    print("Maximum linear velocity error: %.5f" % np.max(errors[:, 0]))
    print("Maximum attitude error: %.5f" % np.max(errors[:, 1]))
    print("Maximum angular velocity error: %.5f" % np.max(errors[:, 2]))

    if plot:
        num_bodyrates = np.stack(num_bodyrates)
        plt.figure()
        for i in range(3):
            plt.subplot(3, 2, i * 2 + 1)
            plt.plot(numeric_derivative[:, i], label='numeric')
            plt.plot(trajectory[:, vel_s + i], label='analytic')
            plt.ylabel('m/s')
            if i == 0:
                plt.title("Velocity check")
            plt.legend()

        for i in range(3):
            plt.subplot(3, 2, i * 2 + 2)
            plt.plot(num_bodyrates[:, i], label='numeric')
            plt.plot(trajectory[:, bodyr_s + i], label='analytic')
            plt.ylabel('rad/s')
            if i == 0:
                plt.title("Body rate check")
            plt.legend()
        plt.suptitle('Integrity check of reference trajectory')
        plt.show()

    return True


def minimum_snap_trajectory_generator(traj_derivatives, yaw_derivatives, t_ref, quad, map_limits, plot, adjust_pos=False):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.

    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    :param map_limits: dictionary of map limits if available, None otherwise.
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
    """

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = traj_derivatives[2, :, :].T + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    # Compute body axes
    z_b = thrust / np.sqrt(np.sum(thrust ** 2, 1))[:, np.newaxis]

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))
    for i in range(len_traj):
        f_t[i, 0] = quad.mv * z_b[i].dot(thrust[i, :].T)

    if yawing:
        # yaw is defined as the projection of the body-x axis on the horizontal plane
        x_c = np.concatenate((np.cos(yaw_derivatives[0, :])[:, np.newaxis],
                              np.sin(yaw_derivatives[0, :])[:, np.newaxis],
                              np.zeros(len_traj)[:, np.newaxis]), 1)
        y_b = np.cross(z_b, x_c)
        y_b = y_b / np.sqrt(np.sum(y_b ** 2, axis=1))[:, np.newaxis]
        x_b = np.cross(y_b, z_b)

        # Rotation matrix (from body to world)
        b_r_w = np.concatenate((x_b[:, :, np.newaxis], y_b[:, :, np.newaxis], z_b[:, :, np.newaxis]), -1)
        q = []
        for i in range(len_traj):
            # Transform to quaternion
            q.append(rotation_matrix_to_quat(b_r_w[i]))
            if i > 1:
                q[-1] = undo_quaternion_flip(q[-2], q[-1])
        q = np.stack(q)

        # Compute angular rate vector
        # Total thrust acceleration must be equal to the projection of the quadrotor acceleration into the Z body axis
        a_proj = np.zeros((len_traj, 1))

        for i in range(len_traj):
            a_proj[i, 0] = z_b[i].dot(traj_derivatives[3, :, i])

        h_omega = quad.mv / f_t * (traj_derivatives[3, :, :].T - a_proj * z_b)
        for i in range(len_traj):
            rate[i, 0] = -h_omega[i].dot(y_b[i])
            rate[i, 1] = h_omega[i].dot(x_b[i])
            rate[i, 2] = -yaw_derivatives[1, i] * np.array([0, 0, 1]).dot(z_b[i])

    else:
        # new way to compute attitude:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        e_z = np.array([[0.0, 0.0, 1.0]])
        q_w = 1.0 + np.sum(e_z * z_b, axis=1)
        q_xyz = np.cross(e_z, z_b)
        q = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
        q = q / np.sqrt(np.sum(q ** 2, 1))[:, np.newaxis]

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(q, axis=0) / discretization_dt
        w_int = np.zeros((len_traj, 3))
        for i in range(len_traj):
            w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]
        rate[:, 0] = w_int[:, 0]
        rate[:, 1] = w_int[:, 1]
        rate[:, 2] = w_int[:, 2]

        go_crazy_about_yaw = True
        if go_crazy_about_yaw:
            # print("Maximum yawrate before adaption: %.3f" % np.max(np.abs(rate[:, 2])))
            q_new = q
            yaw_corr_acc = 0.0
            for i in range(1, len_traj):
                yaw_corr = -rate[i, 2] * discretization_dt
                yaw_corr_acc += yaw_corr
                q_corr = np.array([np.cos(yaw_corr_acc / 2.0), 0.0, 0.0, np.sin(yaw_corr_acc / 2.0)])
                q_new[i, :] = q_dot_q(q[i, :], q_corr)
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]

            q_new_dot = np.gradient(q_new, axis=0) / discretization_dt
            for i in range(1, len_traj):
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q_new[i, :]), q_new_dot[i])[1:]

            q = q_new
            rate[:, 0] = w_int[:, 0]
            rate[:, 1] = w_int[:, 1]
            rate[:, 2] = w_int[:, 2]
            # print("Maximum yawrate after adaption: %.3f" % np.max(np.abs(rate[:, 2])))

    # Compute inputs
    rate_dot = np.gradient(rate, axis=0) / discretization_dt

    rate_x_Jrate = np.array([(quad.J[2] - quad.J[1]) * rate[:, 2] * rate[:, 1],
                             (quad.J[0] - quad.J[2]) * rate[:, 0] * rate[:, 2],
                             (quad.J[1] - quad.J[0]) * rate[:, 1] * rate[:, 0]]).T

    tau = rate_dot * quad.J[np.newaxis, :] + rate_x_Jrate
    b = np.concatenate((tau, f_t), axis=-1)
    a_mat = np.concatenate((quad.y_f[np.newaxis, :], -quad.x_f[np.newaxis, :],
                            quad.z_l_tau[np.newaxis, :], np.ones_like(quad.z_l_tau)[np.newaxis, :]), 0)

    reference_u = np.zeros((len_traj, 4))
    for i in range(len_traj):
        reference_u[i, :] = np.linalg.solve(a_mat, b[i, :])

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    reference_traj = np.concatenate((full_pos, q, full_vel, rate), 1)

    if adjust_pos:
        if map_limits is None:
            # Locate starting point right at x=0 and y=0.
            reference_traj[:, 0] -= reference_traj[0, 0]
            reference_traj[:, 1] -= reference_traj[0, 1]

        else:
            x_max_range = map_limits["x"][1] - map_limits["x"][0]
            y_max_range = map_limits["y"][1] - map_limits["y"][0]
            z_max_range = map_limits["z"][1] - map_limits["z"][0]

            x_center = x_max_range / 2 + map_limits["x"][0]
            y_center = y_max_range / 2 + map_limits["y"][0]
            z_center = z_max_range / 2 + map_limits["z"][0]

            # Center circle to center of map XY plane
            reference_traj[:, :3] += np.array([x_center, y_center, 0])
            reference_traj[:, 2] = z_center

    if plot:
        draw_poly(reference_traj, reference_u, t_ref)

    # Change format of reference input to motor activation, in interval [0, 1]
    reference_u = reference_u / quad.max_thrust

    return reference_traj, t_ref, reference_u

def load_map_limits_from_file(map_limits):
    import rospy
    if map_limits is not None and map_limits != "None":
        config_path = DirectoryConfig.CONFIG_DIR
        params_file = os.path.join(config_path, map_limits + '.yaml')
        try:
            with open(params_file) as file:
                limits = yaml.full_load(file)
                map_limits = {"x": [limits["x_min"], limits["x_max"]],
                              "y": [limits["y_min"], limits["y_max"]],
                              "z": [limits["z_min"], limits["z_max"]]}
                rospy.loginfo("Using world limits: " + json.dumps(limits))
        except FileNotFoundError:
            warn_msg = "Tried to load environment limits: '%s', but the file was not found. Using default limits." \
                       % map_limits
            rospy.logwarn(warn_msg)
            map_limits = None
    else:
        map_limits = None

    return map_limits

def straight_p2p_trajectory(quad, discretization_dt, speed, p_start, p_end, yaw_start, yaw_end, CUAV=False):
    n = 3
    pos_traj = np.stack([p_start, p_start/2 + p_end/2, p_end], axis=0)  # shape (3, 3)
    att_traj = np.zeros((n, 3))
    att_traj[0, 2] = yaw_start
    att_traj[1, 2] = (yaw_start + yaw_end)/2  # Midpoint yaw
    att_traj[2, 2] = yaw_end

    if speed == 0:
        raise ValueError("Speed must be non-zero for trajectory generation.")
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed
    if av_dt == 0:
        raise ValueError("Trajectory duration is zero. Check that p_start and p_end are not the same.")

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)
    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False)
    return reference_traj, t_ref, reference_u

def straight_trajectory(quad, discretization_dt, speed, CUAV=False):
    n = 3
    pos_traj = np.zeros((n, 3))
    pos_traj[:, 1] = np.linspace(-3, 3, n)
    pos_traj[:, 2] = 1
    att_traj = np.zeros_like(pos_traj)

    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)
    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False)

    # reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False)
    return reference_traj, t_ref, reference_u


def flyover_trajectory_collect(quad, discretization_dt, seed, speed, flyover_box_name, CUAV=False):
    np.random.seed(seed)
    flyover_box = load_map_limits_from_file(flyover_box_name)
    box_x_len = flyover_box['x'][1] - flyover_box['x'][0]
    box_y_len = flyover_box['y'][1] - flyover_box['y'][0]
    box_z_len = flyover_box['z'][1] - flyover_box['z'][0]
    box_center = np.array([flyover_box['x'][0] + box_x_len / 2, flyover_box['y'][0] + box_y_len / 2])
    box_diag = (box_x_len ** 2 + box_y_len ** 2) ** 0.5

    pos_traj = np.zeros((50, 3))

    for i in range(0, 50, 5):
        rand_direction = (np.random.rand(2) - 0.5) * 2
        rand_direction = rand_direction / np.linalg.norm(rand_direction)
        height_start = flyover_box['z'][0] + np.random.rand() * box_z_len
        start_point = box_center + rand_direction * 0.6 * box_diag

        direction = (box_center - start_point)
        direction = direction / np.linalg.norm(direction)

        height_flyover = flyover_box['z'][0] + np.random.rand() * box_z_len
        flyover_point = box_center - direction * 0.25 * box_diag

        height_flyover_center = 0.55
        flyover_center_point = box_center

        height_flyover2 = flyover_box['z'][0] + np.random.rand() * box_z_len
        flyover_point2 = box_center + direction * 0.25 * box_diag

        height_target = flyover_box['z'][0] + np.random.rand() * box_z_len
        target_point = box_center + direction * 0.7 * box_diag


        pos_traj[i, :2] = start_point
        pos_traj[i, 2] = height_start
        pos_traj[i + 1, :2] = flyover_point
        pos_traj[i + 1, 2] = height_flyover
        pos_traj[i + 2, :2] = flyover_center_point
        pos_traj[i + 2, 2] = height_flyover_center
        pos_traj[i + 3, :2] = flyover_point2
        pos_traj[i + 3, 2] = height_flyover2
        pos_traj[i + 4, :2] = target_point
        pos_traj[i + 4, 2] = height_target



    att_traj = np.zeros_like(pos_traj)

    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)
    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False, adjust_pos=False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False, adjust_pos=False)
    # reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False, adjust_pos=False)
    return reference_traj, t_ref, reference_u


def flyover_trajectory(quad, discretization_dt, seed, speed, flyover_box_name, CUAV=False):
    np.random.seed(seed)
    flyover_box = load_map_limits_from_file(flyover_box_name)
    box_x_len = flyover_box['x'][1] - flyover_box['x'][0]
    box_y_len = flyover_box['y'][1] - flyover_box['y'][0]
    box_z_len = flyover_box['z'][1] - flyover_box['z'][0]
    box_center = np.array([flyover_box['x'][0] + box_x_len / 2, flyover_box['y'][0] + box_y_len / 2])
    box_diag = (box_x_len ** 2 + box_y_len ** 2) ** 0.5

    pos_traj = np.zeros((17, 3))

    rand_direction = (np.random.rand(2) - 0.5) * 2
    rand_direction = rand_direction / np.linalg.norm(rand_direction)
    height = flyover_box['z'][0] + np.random.rand() * box_z_len
    start_point = box_center + rand_direction * 0.6 * box_diag

    pos_traj[0, :2] = start_point
    pos_traj[0, 2] = height

    for i in range(1, 16, 3):
        rand_direction = (np.random.rand(2) - 0.5) * 2
        rand_direction = rand_direction / np.linalg.norm(rand_direction)
        height = flyover_box['z'][0] + np.random.rand() * box_z_len
        start_point = box_center + rand_direction * 0.6 * box_diag

        direction = (box_center - start_point)
        direction = direction / np.linalg.norm(direction)

        flyover_center_point = box_center

        target_point = box_center + direction * 0.6 * box_diag

        pos_traj[i, :2] = start_point
        pos_traj[i, 2] = height
        pos_traj[i + 1, :2] = flyover_center_point
        pos_traj[i + 1, 2] = height
        pos_traj[i + 2, :2] = target_point
        pos_traj[i + 2, 2] = height

    target_point = box_center + direction * 1.2 * box_diag
    pos_traj[i + 3, :2] = target_point
    pos_traj[i + 3, 2] = height

    att_traj = np.zeros_like(pos_traj)

    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)
    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False, adjust_pos=False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False, adjust_pos=False)
    # reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False,
                                                                        #    adjust_pos=False)
    return reference_traj, t_ref, reference_u


def random_trajectory(quad, discretization_dt, seed, speed, map_name=None, plot=False, CUAV=False):
    map_limits = load_map_limits_from_file(map_name)

    # Get a random smooth position trajectory
    pos_traj, att_traj = random_periodical_trajectory(random_state=seed, map_limits=map_limits, plot=False)

    if map_limits is None:
        # Locate starting point right at x=0 and y=0.
        pos_traj[:, 0] -= pos_traj[0, 0]
        pos_traj[:, 1] -= pos_traj[0, 1]

    # Ensure no negative z position
    min_z_pos = np.min(pos_traj[:, -1])
    if min_z_pos < 0:
        pos_traj[:, -1] = pos_traj[:, -1] + 2 * min_z_pos * np.sign(min_z_pos)

    # Calculate feasible time based on the average distance between two position track points:
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed

    # Calculate the polynomial fit to the position trajectory, and compute the full kinematic reference. This
    # trajectory is sampled according to the frequency that will be needed for the MPC.
    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)

    traj, yaw, t_ref = get_full_traj(poly_pos_traj, av_dt, discretization_dt)
    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, map_limits, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, False)

    # reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, False)
    if plot:
        draw_poly(reference_traj, reference_u, t_ref, pos_traj.T)

    return reference_traj, t_ref, reference_u


def loop_trajectory(quad, discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max, map_name, plot, CUAV=False):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param map_name: Name of map to load its limits
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    # Apply map limits to radius
    map_limits = load_map_limits_from_file(map_name)
    if map_limits is not None:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]

        max_radius = min(x_max_range / 2, y_max_range / 2)

        radius = min(radius, max_radius)

    assert z > 0

    ramp_up_t = v_max / lin_acc / 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = transition_t_vec[-1] + np.arange(0, coasting_duration, discretization_dt) + discretization_dt
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = down_coasting_t_vec[-1] + np.arange(0, ramp_up_t, discretization_dt) + discretization_dt
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))
    alpha_vec = np.concatenate((
        ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end))

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * ramp_t_vec)
    coasting_alpha_dt = np.zeros_like(coasting_alpha)
    transition_alpha_dt = - alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / (2 * ramp_up_t) * transition_t_vec)
    alpha_dt = np.concatenate((
        ramp_up_alpha_dt, coasting_alpha_dt, transition_alpha_dt, coasting_alpha_dt, ramp_up_alpha_dt))

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    print(f"loop angle_vec shape is : {angle_vec.shape}")
    print(f"loop pos_traj_x shape is : {pos_traj_x.shape}")
    pos_traj_z = np.reshape(z * 0.1 *  t_ref, (1,1,-1)) # np.ones_like(pos_traj_x) * z
    print(f"loop pos_traj_z shape is : {pos_traj_z.shape}")
    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = - (radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_z = np.reshape(np.ones_like(t_ref) * z * 0.1, (1,1,-1)) # np.zeros_like(vel_traj_x)

    acc_traj_x = radius * (alpha_vec * np.cos(angle_vec) - w_vec ** 2 * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    acc_traj_y = - radius * (alpha_vec * np.sin(angle_vec) + w_vec ** 2 * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    # acc_traj_z = np.zeros_like(acc_traj_x)[np.newaxis, np.newaxis, :]
    jerk_traj_x = radius * (alpha_dt * np.cos(angle_vec) - alpha_vec * np.sin(angle_vec) * w_vec -
                            np.cos(angle_vec) * w_vec ** 3 - 2 * np.sin(angle_vec) * w_vec * alpha_vec)
    jerk_traj_y = - radius * (np.cos(angle_vec) * w_vec * alpha_vec + np.sin(angle_vec) * alpha_dt -
                              np.sin(angle_vec) * w_vec ** 3 + 2 * np.cos(angle_vec) * w_vec * alpha_vec)
    jerk_traj_x = jerk_traj_x[np.newaxis, np.newaxis, :]
    jerk_traj_y = jerk_traj_y[np.newaxis, np.newaxis, :]

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    # print(f"loop pos_traj_x shape is : {pos_traj_x.shape}")
    traj = np.concatenate((
        np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
        np.concatenate((vel_traj_x, vel_traj_y, vel_traj_z), 1),
        np.concatenate((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)), 1),
        np.concatenate((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)), 1)), 0)
    # print(f"loop traj shape is : {traj.shape}")
    yaw = np.concatenate((yaw_traj[np.newaxis, :], w_vec[np.newaxis, :]), 0)
    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, map_limits, plot)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, plot)

    return reference_traj, t_ref, reference_u

    # return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, plot)


def lemniscate_trajectory(quad, discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max, map_name, plot, CUAV=False):
    """

    :param quad:
    :param discretization_dt:
    :param radius:
    :param z:
    :param lin_acc:
    :param clockwise:
    :param yawing:
    :param v_max:
    :param map_name:
    :param plot:
    :return:
    """

    # Apply map limits to radius
    map_limits = load_map_limits_from_file(map_name)
    if map_limits is not None:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]

        max_radius = min(x_max_range / 2, y_max_range / 2)

        radius = min(radius, max_radius)

    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = transition_t_vec[-1] + np.arange(0, coasting_duration, discretization_dt) + discretization_dt
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = down_coasting_t_vec[-1] + np.arange(0, ramp_up_t, discretization_dt) + discretization_dt
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))
    alpha_vec = np.concatenate((
        ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end))

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Adaption: we achieve the highest spikes in the bodyrates when passing through the 'center' part of the figure-8
    # This leads to negative reference thrusts.
    # Let's see if we can alleviate this by adapting the z-reference in these parts to add some acceleration in the
    # z-component
    z_dim = 0.0

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * (np.sin(angle_vec) * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    pos_traj_z = - z_dim * np.cos(4.0 * angle_vec)[np.newaxis, np.newaxis, :] + z

    vel_traj_x = -radius * (w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = radius * (w_vec * np.cos(angle_vec) ** 2 - w_vec * np.sin(angle_vec) ** 2)[np.newaxis, np.newaxis, :]
    vel_traj_z = 4.0 * z_dim * w_vec * np.sin(4.0 * angle_vec)[np.newaxis, np.newaxis, :]

    acc_traj_x = -radius * (alpha_vec * np.sin(angle_vec) + w_vec ** 2 * np.cos(angle_vec))
    acc_traj_y = radius * (alpha_vec * np.cos(angle_vec) ** 2 - 2.0 * w_vec ** 2 * np.cos(angle_vec) * np.sin(
        angle_vec) - alpha_vec * np.sin(angle_vec) ** 2 - 2.0 * w_vec ** 2 * np.sin(angle_vec) * np.cos(angle_vec))
    acc_traj_z = 16.0 * z_dim * (w_vec ** 2 * np.cos(4.0 * angle_vec) + alpha_vec * np.sin(4.0 * angle_vec))
    acc_traj_x = acc_traj_x[np.newaxis, np.newaxis, :]
    acc_traj_y = acc_traj_y[np.newaxis, np.newaxis, :]
    acc_traj_z = acc_traj_z[np.newaxis, np.newaxis, :]

    traj = np.concatenate((
        np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
        np.concatenate((vel_traj_x, vel_traj_y, vel_traj_z), 1),
        np.concatenate((acc_traj_x, acc_traj_y, acc_traj_z), 1)), 0)

    yaw = np.zeros_like(traj)

    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, plot)

    return reference_traj, t_ref, reference_u

    # return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_limits, plot)

def trefoil_knot_trajectory(quad, discretization_dt, v_max, map_name=None, yawing=False, plot=False, CUAV=False):


    # ------------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------------
    A1 = 1.0       # amplitude of sin(t)
    A2 = 2.0       # amplitude of sin(2t)
    A3 = 1.0       # amplitude of sin(3t)

    F1 = 1.0       # frequency multiplier for t
    F2 = 2.0       # frequency multiplier for 2t
    F3 = 3.0       # frequency multiplier for 3t

    v_max = 1.0  # target max velocity (m/s)

    # ------------------------------------------------------------------
    # PARAMETRIC CURVE: TREFOIL KNOT
    # ------------------------------------------------------------------
    start = 0
    end = 5 * np.pi
    num_points = int((end - start) / discretization_dt) + 1  # Add 1 to include the endpoint

    t = np.linspace(start, end, num_points)
    # Unscaled velocity for rescaling
    vx_raw = A1 * F1 * np.cos(F1 * t) + A2 * F2 * np.cos(F2 * t)
    vy_raw = -A1 * F1 * np.sin(F1 * t) + A2 * F2 * np.sin(F2 * t)
    vz_raw = -A3 * F3 * np.cos(F3 * t)

    v_mag_raw = np.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)
    scale_factor = v_max / np.max(v_mag_raw)
    t_scaled = t * scale_factor

    def trefoil_knot(t):
        x = A1 * np.sin(F1 * t) + A2 * np.sin(F2 * t)
        y = A1 * np.cos(F1 * t) - A2 * np.cos(F2 * t)
        z = -A3 * np.sin(F3 * t)
        return x, y, z

    # ------------------------------------------------------------------
    # ANALYTICAL DERIVATIVES (SCALED)
    # ------------------------------------------------------------------
    pos_traj_x, pos_traj_y, pos_traj_z = trefoil_knot(t_scaled)

    vel_traj_x = scale_factor * (A1 * F1 * np.cos(F1 * t_scaled) + A2 * F2 * np.cos(F2 * t_scaled))
    vel_traj_y = scale_factor * (-A1 * F1 * np.sin(F1 * t_scaled) + A2 * F2 * np.sin(F2 * t_scaled))
    vel_traj_z = scale_factor * (-A3 * F3 * np.cos(F3 * t_scaled))

    acc_traj_x = scale_factor**2 * (-A1 * F1**2 * np.sin(F1 * t_scaled) - A2 * F2**2 * np.sin(F2 * t_scaled))
    acc_traj_y = scale_factor**2 * (-A1 * F1**2 * np.cos(F1 * t_scaled) + A2 * F2**2 * np.cos(F2 * t_scaled))
    acc_traj_z = scale_factor**2 * (A3 * F3**2 * np.sin(F3 * t_scaled))

    jx = scale_factor**3 * (-A1 * F1**3 * np.cos(F1 * t_scaled) - A2 * F2**3 * np.cos(F2 * t_scaled))
    jy = scale_factor**3 * (A1 * F1**3 * np.sin(F1 * t_scaled) - A2 * F2**3 * np.sin(F2 * t_scaled))
    jz = scale_factor**3 * (A3 * F3**3 * np.cos(F3 * t_scaled))

    # sx = scale_factor**4 * (A1 * F1**4 * np.sin(F1 * t_scaled) + A2 * F2**4 * np.sin(F2 * t_scaled))
    # sy = scale_factor**4 * (A1 * F1**4 * np.cos(F1 * t_scaled) - A2 * F2**4 * np.cos(F2 * t_scaled))
    # sz = scale_factor**4 * (-A3 * F3**4 * np.sin(F3 * t_scaled))

    # cx = scale_factor**5 * (A1 * F1**5 * np.cos(F1 * t_scaled) + A2 * F2**5 * np.cos(F2 * t_scaled))
    # cy = scale_factor**5 * (-A1 * F1**5 * np.sin(F1 * t_scaled) + A2 * F2**5 * np.sin(F2 * t_scaled))
    # cz = scale_factor**5 * (A3 * F3**5 * np.cos(F3 * t_scaled))

    # px = scale_factor**6 * (-A1 * F1**6 * np.sin(F1 * t_scaled) - A2 * F2**6 * np.sin(F2 * t_scaled))
    # py = scale_factor**6 * (-A1 * F1**6 * np.cos(F1 * t_scaled) + A2 * F2**6 * np.cos(F2 * t_scaled))
    # pz = scale_factor**6 * (-A3 * F3**6 * np.sin(F3 * t_scaled))

    print(f"shape of pos_traj_x is {pos_traj_x.shape}:")

    # traj = np.concatenate((
    #     np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
    #     np.concatenate((vel_traj_x, vel_traj_y, vel_traj_z), 1),
    #     np.concatenate((acc_traj_x, acc_traj_y, acc_traj_z), 1)), 0)

    traj = np.array([
        [pos_traj_x, pos_traj_y, pos_traj_z],
        [vel_traj_x, vel_traj_y, vel_traj_z],
        [acc_traj_x, acc_traj_y, acc_traj_z],
        [jx, jy, jz]
    ]).squeeze()
    print(f"shape of traj is {traj.shape}:")
    yaw = np.zeros_like(traj)
    t_ref = t

    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_name, plot)

    return reference_traj, t_ref, reference_u

def eight_shape_trajectory(quad, discretization_dt, v_max, map_name=None, yawing=False, plot=False, CUAV=False):
    start = 0
    end = 5 * np.pi
    num_points = int((end - start) / discretization_dt) + 1  # Add 1 to include the endpoint

    t = np.linspace(start, end, num_points)
    f = 0.5
    a = 3
    b = 3
    c = 0.1 + 0.1*t

    # yaw = np.array([0, 0, 0])

    pos_traj_x = a * np.sin(f * t)
    pos_traj_y = b * np.sin(f * t) * np.cos(f * t) + b/2
    pos_traj_z = c

    vel_traj_x = a * f * np.cos(f * t)
    vel_traj_y = b * f * (np.cos(f * t) ** 2 - np.sin(f * t) ** 2)
    vel_traj_z = 0.1

    acc_traj_x = -a * f ** 2 * np.sin(f * t)
    acc_traj_y = -2 * b * f ** 2 * np.sin(f * t) * np.cos(f * t)
    acc_traj_z = 0     

    jerk_traj_x = -a * f ** 3 * np.cos(f * t)
    jerk_traj_y = -b * f ** 3 * (np.cos(f * t) ** 2 - np.sin(f * t) ** 2)
    jerk_traj_z = 0  

    # sx = a * f ** 4 * np.sin(f * t)
    # sy = b * f ** 4 * (2 * np.sin(f * t) ** 2 - 2 * np.cos(f * t) ** 2)
    # sz = 0
    #
    # cx = a * f ** 5 * np.cos(f * t)
    # cy = -b * f ** 5 * (np.cos(f * t) ** 2 - np.sin(f * t) ** 2)
    # cz = 0    

    # px = -a * f ** 6 * np.sin(f * t)
    # py = -2 * b * f ** 6 * np.sin(f * t) * np.cos(f * t)
    # pz = 0


    # yaw = np.arctan2(vel_traj_y, pos_traj_x)

    # if yaw > 0.8:
    #     yaw -= 2 * np.pi

    traj = np.array([
        [pos_traj_x, pos_traj_y, pos_traj_z],
        [vel_traj_x, vel_traj_y, vel_traj_z],
        [acc_traj_x, acc_traj_y, acc_traj_z],
        [jerk_traj_x, jerk_traj_y, jerk_traj_z]
    ]).squeeze()

    print(f"eight_shape_trajectory : {traj.shape}:")


    yaw = np.zeros_like(traj)
    t_ref = t

    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_name, plot)

    return reference_traj, t_ref, reference_u


def circle_trajectory(quad, discretization_dt, v_max, map_name=None, yawing=False, plot=False, CUAV=False):
    
    start = 0
    end = 5 * np.pi
    num_points = int((end - start) / discretization_dt) + 1  # Add 1 to include the endpoint
    t = np.linspace(start, end, num_points)
    f = 0.2
    a_x = -3
    a_y = 3
    a_z = 1 # 0.1 + 0.1*t
    omega = 2 * np.pi * f


    # Position
    pos_traj_x = a_x * (1 - np.cos(omega * t))
    pos_traj_y = a_y * np.sin(omega * t)
    pos_traj_z = a_z * np.ones_like(t)

    # Velocity
    vel_traj_x = a_x * omega * np.sin(omega * t)
    vel_traj_y = a_y * omega * np.cos(omega * t)
    vel_traj_z = np.zeros_like(t) # np.ones_like(t) * 0.1

    # Acceleration
    acc_traj_x = a_x * omega**2 * np.cos(omega * t)
    acc_traj_y = -a_y * omega**2 * np.sin(omega * t)
    acc_traj_z = np.zeros_like(t)

    # Jerk
    jerk_traj_x = -a_x * omega**3 * np.sin(omega * t)
    jerk_traj_y = -a_y * omega**3 * np.cos(omega * t)
    jerk_traj_z = np.zeros_like(t)


    # sx = (2 * np.pi) ** 4 * f ** 4 * a_x * np.cos(omega * t)
    # sy = (2 * np.pi) ** 4 * f ** 4 * a_y * np.sin(omega * t)
    # sz = 0    

    # cx = (2 * np.pi) ** 5 * f ** 5 * a_x * np.sin(omega * t)
    # cy = (2 * np.pi) ** 5 * f ** 5 * a_y * np.cos(omega * t)
    # cz =  0   

    # px = (2 * np.pi) ** 6 * f ** 6 * a_x * np.cos(omega * t)
    # py = -(2 * np.pi) ** 6 * f ** 6 * a_y * np.sin(omega * t)
    # pz = 0    


    traj = np.array([
        [pos_traj_x, pos_traj_y, pos_traj_z],
        [vel_traj_x, vel_traj_y, vel_traj_z],
        [acc_traj_x, acc_traj_y, acc_traj_z],
        [jerk_traj_x, jerk_traj_y, jerk_traj_z]
    ]).squeeze()
    print(f"circle_trajectory is : {traj.shape}:")


    yaw = np.zeros_like(traj)
    t_ref = t

    if CUAV:
        reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False)
    else:
        reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, map_name, plot)

    return reference_traj, t_ref, reference_u

def vee_map(matrix):
    return np.array([matrix[2, 1], matrix[0, 2], matrix[1, 0]])

def vec_dot(a, b):
    return np.sum(a * b, axis=0)


def vec_cross(a, b):
    return np.cross(a, b, axis=0)

def get_load_trajX(t, traj_num):
    load_traj = {}
    if traj_num == 0:  # 8-shaped
        f = 0.5
        a = 3
        b = 3
        c = 0.1 + 0.1*t
        load_traj['b1'] = np.array([0, 0, 0])

        load_traj['xL'] = np.array([a * np.sin(f * t) ,
                                    b * np.sin(f * t) * np.cos(f * t) + b/2,
                                    c])

        load_traj['dxL'] = np.array([a * f * np.cos(f * t),
                                     b * f * (np.cos(f * t) ** 2 - np.sin(f * t) ** 2),
                                     0.5])

        load_traj['d2xL'] = np.array([-a * f ** 2 * np.sin(f * t),
                                      -2 * b * f ** 2 * np.sin(f * t) * np.cos(f * t),
                                      0])

        load_traj['d3xL'] = np.array([-a * f ** 3 * np.cos(f * t),
                                      -b * f ** 3 * (np.cos(f * t) ** 2 - np.sin(f * t) ** 2),
                                      0])

        load_traj['d4xL'] = np.array([a * f ** 4 * np.sin(f * t),
                                      b * f ** 4 * (2 * np.sin(f * t) ** 2 - 2 * np.cos(f * t) ** 2),
                                      0])

        load_traj['d5xL'] = np.array([a * f ** 5 * np.cos(f * t),
                                      -b * f ** 5 * (np.cos(f * t) ** 2 - np.sin(f * t) ** 2),
                                      0])

        load_traj['d6xL'] = np.array([-a * f ** 6 * np.sin(f * t),
                                      -2 * b * f ** 6 * np.sin(f * t) * np.cos(f * t),
                                      0])

        dx = load_traj['dxL'][0]
        dy = load_traj['dxL'][1]

        yaw = np.arctan2(dy, dx)

        if yaw > 0.8:
            yaw -= 2 * np.pi

        load_traj['yaw'] = 0

    else:  # traj_num == 1, circle
        f = 0.2
        a_x = -3
        a_y = 3
        a_z = 0.1 + 0.1*t

        load_traj['b1'] = np.array([0, 0, 0])

        load_traj['xL'] = np.array([a_x * (1 - np.cos(2 * np.pi * f * t)),
                                    a_y * np.sin(2 * np.pi * f * t),
                                    a_z])

        load_traj['dxL'] = 2 * np.pi * np.array([f * a_x * np.sin(2 * np.pi * f * t),
                                                 f * a_y * np.cos(2 * np.pi * f * t),
                                                 0.1])

        load_traj['d2xL'] = (2 * np.pi) ** 2 * np.array([f ** 2 * a_x * np.cos(2 * np.pi * f * t),
                                                         -f ** 2 * a_y * np.sin(2 * np.pi * f * t),
                                                         0])

        load_traj['d3xL'] = (2 * np.pi) ** 3 * np.array([-f ** 3 * a_x * np.sin(2 * np.pi * f * t),
                                                         -f ** 3 * a_y * np.cos(2 * np.pi * f * t),
                                                         0])

        load_traj['d4xL'] = (2 * np.pi) ** 4 * np.array([f ** 4 * a_x * np.cos(2 * np.pi * f * t),
                                                         f ** 4 * a_y * np.sin(2 * np.pi * f * t),
                                                         0])

        load_traj['d5xL'] = (2 * np.pi) ** 5 * np.array([f ** 5 * a_x * np.sin(2 * np.pi * f * t),
                                                         f ** 5 * a_y * np.cos(2 * np.pi * f * t),
                                                         f ** 5 * a_z * np.cos(2 * np.pi * f * t)])

        load_traj['d6xL'] = (2 * np.pi) ** 6 * np.array([f ** 6 * a_x * np.cos(2 * np.pi * f * t),
                                                         -f ** 6 * a_y * np.sin(2 * np.pi * f * t),
                                                         0])

        load_traj['yaw'] = 0

    return load_traj

def setup_uav(quad, p_start, p_end, yaw_start, yaw_end, discretization_dt):

    # states = quad.get_state()
    # pv = np.array(states[0])
    # pl = np.array(states[1])
    # angle = np.array(states[3])  # Quaternion is false, get eular angle
    # # Create Rotation object and extract yaw (Z `Euler angle)
    # yaw_start = angle[2]  # Yaw is the first column`
    # qv_xyzw = np.roll(qQ[:, 0], -1, axis=1) 
    # rot = Rotation.from_quat(qv_xyzw)
    # yaw_end = rot.as_euler('zyx', degrees=False)[0]  # Yaw is the first column`

    # qc0 = np.array([0,0,-1])
    # p_start = pv
    # p_end = pl - qc0 * quad.cl # xQ[:, 0]
    

    # yaw_start = 0 # Yaw is the first column`
    # yaw_end = 0 # Yaw is the first column`

    # qc0 = np.array([0,0,-1])
    # p_start = [-0.4, -0.4, 0]
    # p_end = [0,0,0.8] # xQ[:, 0]

    n = 3
    pos_traj = np.stack([p_start, p_start/2 + p_end/2, p_end], axis=0)  # shape (3, 3)
    att_traj = np.zeros((n, 3))
    att_traj[0, 2] = yaw_start
    att_traj[1, 2] = (yaw_start + yaw_end)/2  # Midpoint yaw
    att_traj[2, 2] = yaw_end

    speed = 1
    if speed == 0:
        raise ValueError("Speed must be non-zero for trajectory generation.")
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed
    if av_dt == 0:
        raise ValueError("Trajectory duration is zero. Check that p_start and p_end are not the same.")

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)

    reference_traj, t_ref, reference_u = minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, None, False, adjust_pos=False)


    return reference_traj, t_ref, reference_u

def lift_load(quad, p_start, p_end, yaw_start, yaw_end, discretization_dt):

    n = 3
    pos_traj = np.stack([p_start, p_start/2 + p_end/2, p_end], axis=0)  # shape (3, 3)
    att_traj = np.zeros((n, 3))
    att_traj[0, 2] = yaw_start
    att_traj[1, 2] = (yaw_start + yaw_end)/2  # Midpoint yaw
    att_traj[2, 2] = yaw_end

    speed = 1
    if speed == 0:
        raise ValueError("Speed must be non-zero for trajectory generation.")
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed
    av_dist = np.mean(np.sqrt(np.sum(np.diff(pos_traj, axis=0) ** 2, axis=1)))
    av_dt = av_dist / speed
    if av_dt == 0:
        raise ValueError("Trajectory duration is zero. Check that p_start and p_end are not the same.")

    poly_pos_traj = fit_multi_segment_polynomial_trajectory(pos_traj.T, att_traj[:, -1].T)
    traj, yaw, t_ref = get_full_traj(poly_pos_traj, target_dt=av_dt, int_dt=discretization_dt)

    reference_traj, t_ref, reference_u = get_nom_traj(traj, yaw, t_ref, quad, None, False, adjust_pos=False)

    return reference_traj, t_ref, reference_u

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



def get_nom_traj(traj_derivatives, yaw_derivatives, t_ref, quad, map_limits=None, plot=False, adjust_pos=True):
    """
    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    """
    # Constants
    mQ = quad.mv
    mL = quad.ml
    J = np.diag(np.array(quad.J))  # [3 x 3]
    g = quad.g[-1]
    e3 = np.reshape(np.array([0,0,1]), (3, 1))  # [3 x 1]
    l = quad.cl

    yawing = np.any(yaw_derivatives[0, :] != 0)  
    traj_len = traj_derivatives.shape[2]
    if yawing:
        yaw = yaw_derivatives[0, :]
    else:
        yaw = np.zeros_like(traj_derivatives[0, 0, :])
    # Derivative of Load Position
    xLd = traj_derivatives[0, :, :]             # [3 x N]
    vLd = traj_derivatives[1, :, :]   # [3 x N]
    aLd = traj_derivatives[2, :, :] # [3 x N]
    daLd = traj_derivatives[3, :, :]  # [3 x N]
    d2aLd = np.zeros_like(traj_derivatives[3, :, :])  # [3 x N]
    d3aLd = np.zeros_like(traj_derivatives[3, :, :])  # [3 x N]
    d4aLd = np.zeros_like(traj_derivatives[3, :, :])  # [3 x N]

    # Cable Tension and Directional Vector
    Tp = -mL * (aLd + g * e3)
    norm_Tp = np.linalg.norm(Tp, axis=0)
    p = Tp / norm_Tp
    xQ = xLd - l * p

    dTp = -mL * daLd
    dnorm_Tp = 1 / norm_Tp * vec_dot(Tp, dTp)
    dp = (dTp - p * dnorm_Tp) / norm_Tp

    d2Tp = -mL * d2aLd
    d2norm_Tp = (vec_dot(dTp, dTp) + vec_dot(Tp, d2Tp) - dnorm_Tp**2) / norm_Tp
    d2p = (d2Tp - dp * dnorm_Tp - p * d2norm_Tp - dp * dnorm_Tp) / norm_Tp

    d3Tp = -mL * d3aLd
    d3norm_Tp = (
        2 * vec_dot(d2Tp, dTp)
        + vec_dot(dTp, d2Tp)
        + vec_dot(Tp, d3Tp)
        - 3 * dnorm_Tp * d2norm_Tp
    ) / norm_Tp
    d3p = (
        d3Tp
        - d2p * dnorm_Tp
        - dp * d2norm_Tp
        - dp * d2norm_Tp
        - p * d3norm_Tp
        - d2p * dnorm_Tp
        - dp * d2norm_Tp
        - d2p * dnorm_Tp
    ) / norm_Tp

    d4Tp = -mL * d4aLd
    d4norm_Tp = (
        2 * vec_dot(d3Tp, dTp)
        + 2 * vec_dot(d2Tp, d2Tp)
        + vec_dot(d2Tp, d2Tp)
        + vec_dot(dTp, d3Tp)
        + vec_dot(dTp, d3Tp)
        + vec_dot(Tp, d4Tp)
        - 3 * d2norm_Tp**2
        - 3 * dnorm_Tp * d3norm_Tp
        - d3norm_Tp * dnorm_Tp
    ) / norm_Tp
    d4p = (
        d4Tp
        - d3p * dnorm_Tp
        - d2p * d2norm_Tp
        - d2p * d2norm_Tp
        - dp * d3norm_Tp
        - d2p * d2norm_Tp
        - dp * d3norm_Tp
        - dp * d3norm_Tp
        - p * d4norm_Tp
        - d3p * dnorm_Tp
        - d2p * d2norm_Tp
        - d2p * d2norm_Tp
        - dp * d3norm_Tp
        - d3p * dnorm_Tp
        - d2p * d2norm_Tp
        - d3p * dnorm_Tp
    ) / norm_Tp

    # Derivatives of Load Angular Velocity
    omega = vec_cross(p, dp)
    domega = vec_cross(dp, dp) + vec_cross(p, d2p)

    # Derivatives of Quadrotor's Position
    vxQ = vLd - l * dp
    axQ = aLd - l * d2p
    daxQ = daLd - l * d3p
    d2axQ = d2aLd - l * d4p

    zero_array = np.zeros_like(yaw)
    b1d = np.stack([np.cos(yaw), np.sin(yaw), zero_array], axis=0)  # shape: (3, T)

    db1d = np.zeros((3, 1))
    d2b1d = np.zeros((3, 1))

    fb3 = mQ * (axQ + g * e3) - Tp
    norm_fb3 = np.linalg.norm(fb3, axis=0)
    f = norm_fb3
    b3 = fb3 / norm_fb3
    b3_b1d = vec_cross(b3, b1d)
    norm_b3_b1d = np.linalg.norm(b3_b1d, axis=0)
    b1 = -vec_cross(b3, b3_b1d) / norm_b3_b1d
    b2 = vec_cross(b3, b1)
    R = np.array([b1, b2, b3]).transpose(1, 0, 2)

    dfb3 = mQ * daxQ - dTp
    dnorm_fb3 = vec_dot(fb3, dfb3) / norm_fb3
    db3 = (dfb3 * norm_fb3 - fb3 * dnorm_fb3) / norm_fb3**2
    db3_b1d = vec_cross(db3, b1d) + vec_cross(b3, db1d)
    dnorm_b3_b1d = vec_dot(b3_b1d, db3_b1d) / norm_b3_b1d
    db1 = (
        -vec_cross(db3, b3_b1d) - vec_cross(b3, db3_b1d) - b1 * dnorm_b3_b1d
    ) / norm_b3_b1d
    db2 = vec_cross(db3, b1) + vec_cross(b3, db1)
    dR = np.array([db1, db2, db3]).transpose(1, 0, 2)
    # R [3 x 3 x N], dR [3 x 3 x N]
    R_T = np.transpose(R, (2, 1, 0))  # [N x 3 x 3]
    dR_T = np.transpose(dR, (2, 0, 1))  # [N x 3 x 3]
    # R.T @ dR using batch matrix multiplication
    R_T_dR = np.einsum("ijk,ikl->ijl", R_T, dR_T)  # [N x 3 x 3]
    R_T_dR_T = np.transpose(R_T_dR, (1, 2, 0))  # [3 x 3 x N]
    Omega = vee_map(R_T_dR_T)

    d2fb3 = mQ * d2axQ - d2Tp
    d2norm_fb3 = (
        vec_dot(dfb3, dfb3) + vec_dot(fb3, d2fb3) - dnorm_fb3 * dnorm_fb3
    ) / norm_fb3
    d2b3 = (
        (d2fb3 * norm_fb3 + dfb3 * dnorm_fb3 - dfb3 * dnorm_fb3 - fb3 * d2norm_fb3)
        * norm_fb3**2
        - db3 * norm_fb3**2 * 2 * norm_fb3 * dnorm_fb3
    ) / norm_fb3**4
    d2b3_b1d = (
        vec_cross(d2b3, b1d)
        + vec_cross(db3, db1d)
        + vec_cross(db3, db1d)
        + vec_cross(b3, d2b1d)
    )
    d2norm_b3_b1d = (
        (vec_dot(db3_b1d, db3_b1d) + vec_dot(b3_b1d, d2b3_b1d)) * norm_b3_b1d
        - vec_dot(b3_b1d, db3_b1d) * dnorm_b3_b1d
    ) / norm_b3_b1d**2
    d2b1 = (
        (
            -vec_cross(d2b3, b3_b1d)
            - vec_cross(db3, db3_b1d)
            - vec_cross(db3, db3_b1d)
            - vec_cross(b3, d2b3_b1d)
            - db1 * dnorm_b3_b1d
            - b1 * d2norm_b3_b1d
        )
        * norm_b3_b1d
        - db1 * norm_b3_b1d * dnorm_b3_b1d
    ) / norm_b3_b1d**2
    d2b2 = (
        vec_cross(d2b3, b1)
        + vec_cross(db3, db1)
        + vec_cross(db3, db1)
        + vec_cross(b3, d2b1)
    )
    d2R = np.array([d2b1, d2b2, d2b3]).transpose(1, 0, 2)
    dR_T = np.transpose(dR, (2, 1, 0))  # [N x 3 x 3]
    dR_T_ = np.transpose(dR, (2, 0, 1))  # [N x 3 x 3]
    dR_T_dR = np.einsum("ijk,ikl->ijl", dR_T, dR_T_)  # [N x 3 x 3]
    d2R_T = np.transpose(d2R, (2, 0, 1))  # [N x 3 x 3]
    R_T_d2R = np.einsum("ijk,ikl->ijl", R_T, d2R_T)  # [N x 3 x 3]
    sum_ = (dR_T_dR + R_T_d2R).transpose(1, 2, 0)  # [3 x 3 x N]
    dOmega = vee_map(sum_)
    M = J @ dOmega + vec_cross(Omega, J @ Omega)

    print("R shape: ", R.shape)
    print("dOmega shape: ", dOmega.shape)
    print("J shape: ", J.shape)
    print("J @ dOmega shape: ", (J @ dOmega).shape)
    print("vec_cross(Omega, J @ Omega) shape: ", (vec_cross(Omega, J @ Omega)).shape)
    # Convert rotation matrix to quaternion (default format: [x, y, z, w])
    R_N =  np.transpose(R, (2, 0, 1))
    quat_xyzw = Rotation.from_matrix(R_N).as_quat()  # [x, y, z, w]
    quat_wxyz = np.roll(quat_xyzw, shift=1) # Reorder to (w, x, y, z)
    qv = quat_wxyz / np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    print("qv shape: ", qv.shape)

    if adjust_pos:
        if map_limits is None:
            # Locate starting point right at x=0 and y=0.
            print("No map limits provided, centering trajectory at (0, 0) in XY plane.")
            print(f"original UAV initial position: {xQ[:, 0]}")
            print(f"original Load initial position: {xLd[:, 0]}")
            xQ[0, :] -= xLd[0, 0]
            xQ[1, :] -= xLd[1, 0]

            xLd[0, :] -= xLd[0, 0]
            xLd[1, :] -= xLd[1, 0]
            print(f"adjusted UAV initial position: {xQ[:, 0]}")
            print(f"adjusted Load initial position: {xLd[:, 0]}")
        else:
            x_max_range = map_limits["x"][1] - map_limits["x"][0]
            y_max_range = map_limits["y"][1] - map_limits["y"][0]
            z_max_range = map_limits["z"][1] - map_limits["z"][0]

            x_center = x_max_range / 2 + map_limits["x"][0]
            y_center = y_max_range / 2 + map_limits["y"][0]
            z_center = z_max_range / 2 + map_limits["z"][0]

            # Center circle to center of map XY plane
            xQ += np.array([x_center, y_center, 0])
            xLd += np.array([x_center, y_center, 0])
            xQ[2, :] = z_center + quad.cl  # Set UAV height to the load height
            xLd[2, :] = z_center

    setup_f = True
    if setup_f:
        states = quad.get_state()
        pv = np.array(states[0])
        pl = np.array(states[1])
        angle = np.array(states[3])  # Quaternion is false, get eular angle
        # Create Rotation object and extract yaw (Z `Euler angle)
        yaw_start = angle[2]  # Yaw is the first column`
        rot = Rotation.from_quat(quat_xyzw[0,:])
        yaw_end = rot.as_euler('zyx', degrees=False)[0]  # Yaw is the first column`

        qc0 = np.array([0,0,-1])
        p_start = pv
        p_end = pl - qc0 * quad.cl # xQ[:, 0]
        setup_traj, setup_t, setup_u = setup_uav(quad, p_start, p_end, yaw_start, yaw_end, t_ref[1])

        xQ_setup = setup_traj[:, 0:3] # [N x 3]
        vQ_setup = setup_traj[:, 3:6] # [N x 3]
        qv_setup = setup_traj[:, 6:10] # [N x 4]
        Omega_setup = setup_traj[:, 10:13]

        xLd_setup = np.zeros((xQ_setup.shape[0], 3))  # [N x 3]  # [N x 3]
        vLd_setup = np.zeros((vQ_setup.shape[0], 3))  # [N x 3]
        p_setup = np.zeros((qv_setup.shape[0], 3))  # [N x 3]  # [3 x 1]
        dp_setup = np.zeros((Omega_setup.shape[0], 3))  # [N x 3]


        if np.linalg.norm(xLd[:, 0] - pl) > 0.1: # lift the load
            lift_traj, lift_t, lift_u = lift_load(quad, pl, xLd[:, 0], yaw_end, yaw_end, t_ref[1])   
            xQ_lift = lift_traj[:, 0:3] # [N x 3]
            vQ_lift = lift_traj[:, 13:16] # [N x 3]
            qv_lift = lift_traj[:, 9:13] # [N x 4]
            Omega_lift = lift_traj[:, 19:22] # [N x 3]
            xLd_lift = lift_traj[:, 3:6] # [N x 3]
            vLd_lift = lift_traj[:, 16:19]
            p_lift = lift_traj[:, 6:9] # [3 x 1]
            dp_lift = lift_traj[:, 22:25]

            xQ = np.concatenate((xQ_setup, xQ_lift, xQ.T), axis=0)
            vQ = np.concatenate((vQ_setup, vQ_lift, vxQ.T), axis=0)
            qv = np.concatenate((qv_setup, qv_lift, qv), axis=0)
            Omega = np.concatenate((Omega_setup, Omega_lift, Omega.T), axis=0)
            xLd = np.concatenate((xLd_setup, xLd_lift, xLd.T), axis=0)
            vLd = np.concatenate((vLd_setup, vLd_lift, vLd.T), axis=0)             
            p = np.concatenate((p_setup, p_lift, p.T), axis=0)
            dp = np.concatenate((dp_setup, dp_lift, dp.T), axis=0)

            u = np.concatenate((setup_u, lift_u), axis=0)
            t_ref = np.concatenate((setup_t, setup_t[-1]+ lift_t, setup_t[-1]+ lift_t[-1]+t_ref), axis=0)
            print("Lifted Load Trajectory")
        else:
            xQ = np.concatenate((xQ_setup, xQ.T), axis=0)
            vQ = np.concatenate((vQ_setup, vxQ.T), axis=0)
            qv = np.concatenate((qv_setup, qv), axis=0)
            Omega = np.concatenate((Omega_setup, Omega.T), axis=0)
            xLd = np.concatenate((xLd_setup, xLd.T), axis=0)
            vLd = np.concatenate((vLd_setup, vLd.T), axis=0)             
            p = np.concatenate((p_setup, p.T), axis=0)
            dp = np.concatenate((dp_setup, dp.T), axis=0)

            u = setup_u
            t_ref = np.concatenate((setup_t, setup_t[-1]+ t_ref), axis=0)    


    reference_traj = np.concatenate((xQ, xLd, p, qv, vQ, vLd, dp, Omega), 1)

    # print("reference_traj shape: ", reference_traj.shape)
    # print("f shape: ", np.reshape(f, (-1, 1)).shape)
    # print("M shape: ", M.shape)
    f = np.reshape(f, (-1, 1))  # [N x 1]
    # reference_u = np.concatenate((np.reshape(f, (-1, 1)), M), 1)

    # Compute inputs
    # discretization_dt = t_ref[1] - t_ref[0]
    # rate_dot = np.gradient(Omega, axis=0) / discretization_dt

    # rate_x_Jrate = np.array([(quad.J[2] - quad.J[1]) * Omega[:, 2] * Omega[:, 1],
    #                          (quad.J[0] - quad.J[2]) * Omega[:, 0] * Omega[:, 2],
    #                          (quad.J[1] - quad.J[0]) * Omega[:, 1] * Omega[:, 0]]).T
    # tau = rate_dot * quad.J[np.newaxis, :] + rate_x_Jrate
    b = np.concatenate((M.T, f), axis=-1)
    a_mat = np.concatenate((quad.y_f[np.newaxis, :], -quad.x_f[np.newaxis, :],
                            quad.z_l_tau[np.newaxis, :], np.ones_like(quad.z_l_tau)[np.newaxis, :]), 0)

    reference_u = np.zeros((traj_len, 4))
    for i in range(traj_len):
        reference_u[i, :] = np.linalg.solve(a_mat, b[i, :])

    reference_u = np.concatenate((u, reference_u), axis=0)  # Add zero column for the 13th DoF
    
    # if adjust_pos:
    #     if map_limits is None:
    #         # Locate starting point right at x=0 and y=0.
    #         print("No map limits provided, centering trajectory at (0, 0) in XY plane.")
    #         print(f"original UAV initial position: {reference_traj[0, 0:3]}")
    #         print(f"original Load initial position: {reference_traj[0, 3:6]}")
    #         reference_traj[:, 0] -= reference_traj[0, 3]
    #         reference_traj[:, 1] -= reference_traj[0, 4]

    #         reference_traj[:, 3] -= reference_traj[0, 3]
    #         reference_traj[:, 4] -= reference_traj[0, 4]
    #         print(f"adjusted UAV initial position: {reference_traj[0, 0:3]}")
    #         print(f"adjusted Load initial position: {reference_traj[0, 3:6]}")
    #     else:
    #         x_max_range = map_limits["x"][1] - map_limits["x"][0]
    #         y_max_range = map_limits["y"][1] - map_limits["y"][0]
    #         z_max_range = map_limits["z"][1] - map_limits["z"][0]

    #         x_center = x_max_range / 2 + map_limits["x"][0]
    #         y_center = y_max_range / 2 + map_limits["y"][0]
    #         z_center = z_max_range / 2 + map_limits["z"][0]

    #         # Center circle to center of map XY plane
    #         reference_traj[:, :6] += np.array([x_center, y_center, 0])
    #         reference_traj[:, 2] = z_center
    #         reference_traj[:, 5] = z_center

    if plot:
        plot_ret(reference_traj, reference_u, t_ref)

    # Change format of reference input to motor activation, in interval [0, 1]
    reference_u = reference_u / quad.max_thrust
    return reference_traj, t_ref, reference_u

    # return (
    #     xLd,  # 0
    #     vLd,  # 1
    #     aLd,  # 2
    #     -p,  # 3
    #     -dp,  # 4
    #     -d2p,  # 5
    #     omega,  # 6
    #     domega,  # 7
    #     qv,  # 8
    #     R,  # 9
    #     Omega,  # 10
    #     dOmega,   # 11
    #     daLd,  # 12
    #     d2aLd,   # 13
    #     d3p,    # 14
    #     dR,  # 15
    #     d2R,   # 16
    #     xQ,  # 17
    #     vxQ,  # 18
    #     axQ,  # 19
    #     f,  # 20
    #     M,  # 21
    # )

def loop_trajectory_test(discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max, map_name):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.

    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param map_name: Name of map to load its limits
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """

    # Apply map limits to radius
    map_limits = load_map_limits_from_file(map_name)
    if map_limits is not None:
        x_max_range = map_limits["x"][1] - map_limits["x"][0]
        y_max_range = map_limits["y"][1] - map_limits["y"][0]

        max_radius = min(x_max_range / 2, y_max_range / 2)

        radius = min(radius, max_radius)

    assert z > 0

    ramp_up_t = v_max/4  # seconds to reach maximum velocity

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = transition_t_vec[-1] + np.arange(0, coasting_duration, discretization_dt) + discretization_dt
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = down_coasting_t_vec[-1] + np.arange(0, ramp_up_t, discretization_dt) + discretization_dt
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))
    alpha_vec = np.concatenate((
        ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end))

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * ramp_t_vec)
    coasting_alpha_dt = np.zeros_like(coasting_alpha)
    transition_alpha_dt = - alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / (2 * ramp_up_t) * transition_t_vec)
    alpha_dt = np.concatenate((
        ramp_up_alpha_dt, coasting_alpha_dt, transition_alpha_dt, coasting_alpha_dt, ramp_up_alpha_dt))

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    # print(f"loop angle_vec shape is : {angle_vec.shape}")
    # print(f"loop pos_traj_x shape is : {pos_traj_x.shape}")
    pos_traj_z = np.reshape(z * 0.1 *  t_ref, (1,1,-1)) # np.ones_like(pos_traj_x) * z
    # print(f"loop pos_traj_z shape is : {pos_traj_z.shape}")
    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = - (radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_z = np.reshape(np.ones_like(t_ref) * z * 0.1, (1,1,-1)) # np.zeros_like(vel_traj_x)

    acc_traj_x = radius * (alpha_vec * np.cos(angle_vec) - w_vec ** 2 * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    acc_traj_y = - radius * (alpha_vec * np.sin(angle_vec) + w_vec ** 2 * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    # acc_traj_z = np.zeros_like(acc_traj_x)[np.newaxis, np.newaxis, :]
    jerk_traj_x = radius * (alpha_dt * np.cos(angle_vec) - alpha_vec * np.sin(angle_vec) * w_vec -
                            np.cos(angle_vec) * w_vec ** 3 - 2 * np.sin(angle_vec) * w_vec * alpha_vec)
    jerk_traj_y = - radius * (np.cos(angle_vec) * w_vec * alpha_vec + np.sin(angle_vec) * alpha_dt -
                              np.sin(angle_vec) * w_vec ** 3 + 2 * np.cos(angle_vec) * w_vec * alpha_vec)
    jerk_traj_x = jerk_traj_x[np.newaxis, np.newaxis, :]
    jerk_traj_y = jerk_traj_y[np.newaxis, np.newaxis, :]

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    # print(f"loop pos_traj_x shape is : {pos_traj_x.shape}")
    traj = np.concatenate((
        np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
        np.concatenate((vel_traj_x, vel_traj_y, vel_traj_z), 1),
        np.concatenate((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)), 1),
        np.concatenate((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)), 1)), 0)
    # print(f"loop traj shape is : {traj.shape}")
    yaw = np.concatenate((yaw_traj[np.newaxis, :], w_vec[np.newaxis, :], alpha_vec[np.newaxis, :]), 0)

    return traj, yaw, t_ref
if __name__ == "__main__":
    # p_start = np.array([0, 0, 0])
    # p_end = np.array([1, 3, 2])
    # v_start = np.array([0, 0, 0])
    # v_end = np.array([0, 0, 0])
    # a_start = np.array([0, 0, 0])
    # a_end = np.array([0, 0, 0])
    # j_start = np.array([0, 0, 0])
    # j_end = np.array([0, 0, 0])
    # heading = [50, 180]
    # traj, yaw_traj, t = straight_p2p_trajectory(None, p_start, p_end, heading[0], heading[1], 1)

    # # Fake control input trajectory (zeros, shape: len(t) x 4)
    # u_traj = np.zeros((len(t), 4))
    # N = traj.shape[2]
    # traj_for_plot = np.zeros((N, 13))
    # # Fill position, velocity, acceleration, jerk
    # traj_for_plot[:, 0:3] = traj[0].T  # position
    # traj_for_plot[:, 3:6] = traj[1].T  # acceleration
    # traj_for_plot[:, 7:10] = traj[2].T # velocity
    # traj_for_plot[:, 10:13] = traj[3].T # jerk (fill last 3 columns)

    # # quats = R.from_euler('z', yaw_traj[0], degrees=False).as_quat()  # (N, 4), (x, y, z, w)
    # # # Place as [w, x, y, z] in columns 3:7
    # # traj_for_plot[:, 3] = quats[:, 3]  # w
    # # traj_for_plot[:, 4] = quats[:, 0]  # x
    # # traj_for_plot[:, 5] = quats[:, 1]  # y
    # # traj_for_plot[:, 6] = quats[:, 2]  # z


    # # Plot
    # target_points = np.stack([p_start, p_end], axis=1)
    # draw_poly(traj_for_plot, u_traj, t, target_points=target_points)
    speed=1
    seed=0 
    map_limits=None 
    plot=False
    radius=1.0 
    z=1.0
    lin_acc=1.0
    clockwise=True
    yawing=False
    v_max=1.0
    discretization_dt=0.05
    traj, yaw, t_ref = loop_trajectory_test(discretization_dt, radius, z, lin_acc, clockwise, yawing, v_max, map_name=map_limits)
    
    # drone set up
    cl = 1 # cable length
    mv = 1 # mass of quadrotor
    qci = np.array([0.5, 0.5, 0])  # Initial 
    qc0 = np.array([0, 0, -1])  # Initial 
    pl0 = np.array([0, 0, 0])   # Initial position
    pv0 = pl0 - qci*cl  # Initial position
    vv0 = np.zeros(3)  # Initial velocity
    av0 = np.zeros(3)  # Initial acceleration
    jv0 = np.zeros(3)  # Initial jerk
    yaw0 = 0  # Initial yaw
    w0 = 0  # Initial angular velocity

    traj_p1 = traj[0, :, 0] 
    traj_v1 = traj[1, :, 0]
    traj_a1 = traj[2, :, 0]   
    traj_j1 = traj[3, :, 0]  
    traj_yaw = yaw[0, 0]  # Initial yaw
    traj_w = yaw[1, 0]  # Initial angular velocity

    setup_v = 0.5
    setup_t = np.linalg.norm(pv0 - (pl0 - qc0*cl)) / setup_v  # Time to reach first point

    setup_t_vec = np.arange(0, setup_t, discretization_dt)
    setup_p_vec = np.linspace(pv0, pl0 - qc0 * cl, setup_t_vec.shape[0])  # Linear interpolation
    setup_v_vec = np.full_like(setup_p_vec, setup_v)  # Constant velocity
    setup_a_vec = np.zeros

    
    # transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    # transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    # transition_t_vec += coasting_t_vec[-1] + discretization_dt

    # t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))

    p_start = pl0 - qci*cl  
    p_end = traj_p1 - qc0*cl
    traj_set, yaw_traj_set, t_set = straight_trajectory_with_custom_points(None, discretization_dt, setup_v, p_start, p_end, yaw0, traj_w)


    # Concatenate all sequences
    traj = np.concatenate((traj_set, traj), axis=2)
    yaw_traj = np.concatenate((yaw_traj_set[0:3, :], yaw), axis=1)  # Keep yaw rate from the main trajectory
    t = np.concatenate((t_set, t_ref + t_set[-1]))  #



    # Prepare for plotting
    N = traj.shape[2]
    traj_for_plot = np.zeros((N, 13))
    traj_for_plot[:, 0:3] = traj[0].T  # position
    traj_for_plot[:, 7:10] = traj[1].T # velocity

    # Convert yaw to quaternion (roll=0, pitch=0)
    quats = Rotation.from_euler('z', yaw_traj[0], degrees=False).as_quat()  # (N, 4): x, y, z, w
    traj_for_plot[:, 3] = quats[:, 3]  # w
    traj_for_plot[:, 4] = quats[:, 0]  # x
    traj_for_plot[:, 5] = quats[:, 1]  # y
    traj_for_plot[:, 6] = quats[:, 2]  # z

    # Body rates (yaw rate)
    traj_for_plot[:, 10] = yaw_traj[1]  # yaw rate

    # Fake control input trajectory (zeros)
    u_traj = np.zeros((len(t), 4))

    # Plot
    target_points = np.stack([p_start, p_end], axis=1)
    draw_poly(traj_for_plot, u_traj, t, target_points=target_points)
