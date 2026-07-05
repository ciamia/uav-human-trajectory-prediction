""" Implementation of the data-augmented MPC for quadrotors.

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
import torch

from multirotors.quadrotor.quad_3d_optimizer import Quad3DOptimizer
from utils.quad_3d_opt_utils import simulate_plant, uncertainty_forward_propagation
from utils.utils import make_bx_matrix
from utils.quad_param_loader import custom_quad_param_loader


class Quad3DMPC:
    def __init__(self, my_quad, t_horizon=1.0, n_nodes=5, q_cost=None, r_cost=None,
                 optimization_dt=5e-2, simulation_dt=5e-4, pre_trained_models=None, model_name="my_quad", q_mask=None,
                 solver_options=None, rdrv_d_mat=None, model_conf=None):
        """
        :param my_quad: Quadrotor3D simulator object
        :type my_quad: Quadrotor3D
        :param t_horizon: time horizon for optimization loop MPC controller
        :param n_nodes: number of MPC nodes
        :param optimization_dt: time step between two successive optimizations intended to be used.
        :param simulation_dt: discretized time-step for the quadrotor simulation
        :param pre_trained_models: additional pre-trained GP regressors to be combined with nominal model in the MPC
        :param q_cost: diagonal of Q matrix for LQR cost of MPC cost function. Must be a numpy array of length 13.
        :param r_cost: diagonal of R matrix for LQR cost of MPC cost function. Must be a numpy array of length 4.
        :param q_mask: Optional boolean mask that determines which variables from the state compute towards the
        cost function. In case no argument is passed, all variables are weighted.
        :param solver_options: Optional set of extra options dictionary for acados solver.
        :param rdrv_d_mat: 3x3 matrix that corrects the drag with a linear model according to Faessler et al. 2018. None
        if not used
        """

        if rdrv_d_mat is not None:
            # rdrv is currently not compatible with covariance mode or with GP-MPC.
            print("RDRv mode")
            self.rdrv = rdrv_d_mat
            assert pre_trained_models is None
        else:
            self.rdrv = None

        self.quad = my_quad
        self.simulation_dt = simulation_dt
        self.optimization_dt = optimization_dt

        # motor commands from last step
        self.motor_u = np.array([0., 0., 0., 0.])

        self.n_nodes = n_nodes
        self.t_horizon = t_horizon

        self.mlp = None
        self.mlp_approx = None

        # Load augmented dynamics model with GP regressor
        if isinstance(pre_trained_models, torch.nn.Module):
            self.gp_ensemble = None
            self.B_x = {}
            x_dims = len(my_quad.get_state(quaternion=True, stacked=True))
            pred_dims = [7, 8, 9] + ([10, 11, 12] if model_conf['torque_output'] else [])
            for y_dim in pred_dims:
                self.B_x[y_dim] = make_bx_matrix(x_dims, [y_dim])
                self.mlp = pre_trained_models
        else:
            self.gp_ensemble = None
            self.B_x = {}  # Selection matrix of the GP regressor-modified system states

        # For MPC optimization use
        self.quad_opt = Quad3DOptimizer(my_quad, t_horizon=t_horizon, n_nodes=n_nodes,
                                        q_cost=q_cost, r_cost=r_cost,
                                        B_x=self.B_x, gp_regressors=self.gp_ensemble,
                                        mlp_regressor=self.mlp, mlp_conf=model_conf,
                                        model_name=model_name, q_mask=q_mask,
                                        solver_options=solver_options, rdrv_d_mat=rdrv_d_mat)

    def clear(self):
        self.quad_opt.clear_acados_model()

    def get_state(self):
        """
        Returns the state of the drone, with the angle described as a wxyz quaternion
        :return: 13x1 array with the drone state: [p_xyz, a_wxyz, v_xyz, r_xyz]
        """

        x = np.expand_dims(self.quad.get_state(quaternion=True, stacked=True), 1)
        return x

    def set_reference(self, x_reference, u_reference=None):
        """
        Sets a target state for the MPC optimizer
        :param x_reference: list with 4 sub-components (position, angle quaternion, velocity, body rate). If these four
        are lists, then this means a single target point is used. If they are Nx3 and Nx4 (for quaternion) numpy arrays,
        then they are interpreted as a sequence of N tracking points.
        :param u_reference: Optional target for the optimized control inputs
        """

        if isinstance(x_reference[0], list):
            # Target state is just a point
            return self.quad_opt.set_reference_state(x_reference, u_reference)
        else:
            # Target state is a sequence
            return self.quad_opt.set_reference_trajectory(x_reference, u_reference)

    def optimize(self, use_model=0, return_x=False):
        """
        Runs MPC optimization to reach the pre-set target.
        :param use_model: Integer. Select which dynamics model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.

        :return: 4*m vector of optimized control inputs with the format: [u_1(0), u_2(0), u_3(0), u_4(0), u_1(1), ...,
        u_3(m-1), u_4(m-1)]. If return_x is True, will also return a vector of shape N+1 x 13 containing the optimized
        state prediction.
        """

        quad_current_state = self.quad.get_state(quaternion=True, stacked=True)
        quad_gp_state = self.quad.get_gp_state(quaternion=True, stacked=True)

        # Remove rate state for simplified model NLP
        out_out = self.quad_opt.run_optimization(quad_current_state, use_model=use_model, return_x=return_x,
                                                 gp_regression_state=quad_gp_state)
        return out_out

    def simulate(self, ref_u):
        """
        Runs the simulation step for the dynamics model of the quadrotor 3D.

        :param ref_u: 4-length reference vector of control inputs
        """

        # Simulate step
        self.quad.update(ref_u, self.simulation_dt)

    def simulate_plant(self, w_opt, t_horizon=None, dt_vec=None, progress_bar=False):
        """
        Given a sequence of n inputs, evaluates the simulated discrete-time plant model n steps into the future. The
        current drone state will not be changed by calling this method.
        :param w_opt: sequence of control n x m control inputs, where n is the number of steps and m is the
        dimensionality of a control input.
        :param t_horizon: time corresponding to the duration of the n control inputs. In the case that the w_opt comes
        from an MPC optimization, this parameter should be the MPC time horizon.
        :param dt_vec: a vector of timestamps, the same length as w_opt, corresponding to the total time each input is
        applied.
        :param progress_bar: boolean - whether to show a progress bar on the console or not.
        :return: the sequence of simulated quadrotor states.
        """

        if t_horizon is None and dt_vec is None:
            t_horizon = self.t_horizon

        return simulate_plant(self.quad, w_opt, simulation_dt=self.simulation_dt, simulate_func=self.simulate,
                              t_horizon=t_horizon, dt_vec=dt_vec, progress_bar=progress_bar)

    def forward_prop(self, x_0, w_opt, cov_0=None, t_horizon=None, dt=None, use_gp=False, use_model=0):
        """
        Computes the forward propagation of the state using an MPC-optimized control input sequence.
        :param x_0: Initial n-length state vector
        :param w_opt: Optimized m*4-length sequence of control inputs from MPC, with the vector format:
        [u_1(1), u_2(1), u_3(1), u_4(1), ..., u_3(m), u_4(m)]
        :param cov_0: Initial covariance estimate (default 0). Can be either a positive semi-definite matrix or a
        1D vector, which will be the diagonal of the covariance matrix. In both cases, the resulting covariance matrix
        must be nxn shape, where n is the length of x_0.
        :param t_horizon: time span of the control inputs (default is the time horizon of the MPC)
        :param dt: Optional. Vector of length m, with the corresponding integration time for every control input in
        w_opt. If none is provided, the default integration step is used.
        :param use_gp: Boolean, whether to use GP regressors when performing the integration or not.
        :param use_model: Integer. Select which dynamics model to use from the available options.
        :return: An m x n array of propagated (expected) state estimates, and an m x n x n array with the m propagated
        covariance matrices.
        """

        # Default parameters
        if dt is not None:
            assert len(dt) == len(w_opt) / 4
            t_horizon = np.sum(dt)
        if t_horizon is None:
            t_horizon = self.t_horizon
        if cov_0 is None:
            cov_0 = np.diag(np.zeros_like(np.squeeze(x_0)))
        elif len(cov_0.shape) == 1:
            cov_0 = np.diag(cov_0)
        elif len(cov_0.shape) > 2:
            TypeError("The initial covariance value must be either a 1D vector of a 2D matrix")

        gp_ensemble = self.gp_ensemble if use_gp else None

        # Compute forward propagation of state pdf
        return uncertainty_forward_propagation(x_0, w_opt, t_horizon=t_horizon, covar=cov_0, dt=dt,
                                               discrete_dynamics_f=self.quad_opt.discretize_f_and_q,
                                               dynamics_jac_f=self.quad_opt.quad_xdot_jac,
                                               B_x=self.B_x, gp_regressors=gp_ensemble,
                                               use_model=use_model, m_integration_steps=1)

    @staticmethod
    def reshape_input_sequence(u_seq):
        """
        Reshapes the an output trajectory from the 1D format: [u_0(0), u_1(0), ..., u_0(n-1), u_1(n-1), ..., u_m-1(n-1)]
        to a 2D n x m array.
        :param u_seq: 1D input sequence
        :return: 2D input sequence, were n is the number of control inputs and m is the dimension of a single input.
        """

        k = np.arange(u_seq.shape[0] / 4, dtype=int)
        u_seq = np.atleast_2d(u_seq).T if len(u_seq.shape) == 1 else u_seq
        u_seq = np.concatenate((u_seq[4 * k], u_seq[4 * k + 1], u_seq[4 * k + 2], u_seq[4 * k + 3]), 1)
        return u_seq

    def reset(self):
        return



class QuadMPC:
    def __init__(self, t_horizon, n_mpc_nodes, opt_dt, quad_name, point_reference=False, models=None,
                 model_conf=None, rdrv=None):

        quad = custom_quad_param_loader(quad_name)

        # Initialize quad MPC
        if point_reference:
            acados_config = {
                "solver_type": "SQP",
                "terminal_cost": True
            }
        else:
            acados_config = {
                "solver_type": "SQP_RTI",
                "terminal_cost": False
            }

        q_diagonal = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.01, 0.01, 0.01])
        r_diagonal = np.array([1.0, 1.0, 1.0, 1.0])

        q_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).T

        quad_mpc = Quad3DMPC(quad, t_horizon=t_horizon, optimization_dt=opt_dt, n_nodes=n_mpc_nodes,
                             pre_trained_models=models, model_conf=model_conf, model_name=quad_name,
                             solver_options=acados_config,
                             q_mask=q_mask, q_cost=q_diagonal, r_cost=r_diagonal, rdrv_d_mat=rdrv)

        self.quad_name = quad_name
        self.quad = quad
        self.quad_mpc = quad_mpc

        # Last optimization
        self.last_w = None

    def set_state(self, x):
        """
        Set quadrotor state estimate from an odometry measurement
        :param x: measured state from odometry. List with 13 components with the format: [p_xyz, q_wxyz, v_xyz, w_xyz]
        """

        self.quad.set_state(x)

    def set_gp_state(self, x):
        """
        Set a quadrotor state estimate from an odometry measurement. While the input state in the function set_state()
        will be used as initial state for the MPC optimization, the input state of this function will be used to
        evaluate the GP's. If this function is never called, then the GP's will be evaluated with the state from
        set_state() too.
        :param x: measured state from odometry. List with 13 components with the format: [p_xyz, q_wxyz, v_xyz, w_xyz]
        """

        self.quad.set_gp_state(x)

    def set_reference(self, x_ref, u_ref):
        """
        Set a reference state for the optimizer.
        :param x_ref: list with 4 sub-components (position, angle quaternion, velocity, body rate). If these four
        are lists, then this means a single target point is used. If they are Nx3 and Nx4 (for quaternion) numpy arrays,
        then they are interpreted as a sequence of N tracking points.
        :param u_ref: Optional target for the optimized control inputs
        """
        return self.quad_mpc.set_reference(x_reference=x_ref, u_reference=u_ref)

    def optimize(self, model_data):
        w_opt, x_opt = self.quad_mpc.optimize(use_model=model_data, return_x=True)

        return w_opt, x_opt
