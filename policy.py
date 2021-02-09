# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import sys
import numpy as np
import copy
from datetime import datetime
# import R.log as rlog removed as it seems unused
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(object):
    def __init__(self, dX, dU, actionBounds=None):
        """

        :param dX: scalar. dimensionality of the state
        :param dU: scalar. dimensionality of the control signal
        :param actionBounds: type R.utils.bounds
        """
        self.dX = dX
        self.dU = dU
        self.bounds = actionBounds

    def act(self, x, obs=None, time=None, noise=None):
        """
        Function to be called externally. Compute next action.
        :param x: an observation I believe?
        :param obs: also an observation?
        :param time: the time
        :param noise: noise
        :return:
            a: action to perform
            t: scalar amount of time used to compute action
        """
        start = timer()
        x = np.squeeze(x)
        assert x.shape[0] is self.dX, 'Wrong dimension state %d, should be %d' % (x.shape[0], self.dX)
        a = self._action(x, obs, time, noise)  # Compute action
        if a.ndim > 0:  # TODO: deal with the ndim==0 case
            assert a.shape[0] is self.dU, 'Wrong dimension control %d, should be %d' % (a.shape[0], self.dU)
        if self.bounds is not None:  # Bound actions, if desired
            a = np.maximum(np.minimum(a, self.bounds[1]), self.bounds[0])
        t = (timer() - start)
        return a, t

    def _action(self, x, obs, time, noise):
        """
        Abstract internal interface to compute action
        :param x:
        :param obs:
        :param time:
        :param noise:
        :return:
        """
        raise NotImplementedError('Implement in subclass')


class randomPolicy(Policy):

    def __init__(self, dX, dU, variance=1):
        Policy.__init__(self, dX=dX, dU=dU)
        self.variance = variance

    def _action(self, x, obs, time, noise):
        action = self.variance * np.random.rand(self.dU)
        return action


class sequenceActions(Policy):
    """
    This policy simply execute a pre-defined sequence of actions
    """

    def __init__(self, dX, dU, actions):
        Policy.__init__(self, dX=dX, dU=dU)
        self._actions = np.matrix(actions)
        self.n_timestep = self._actions.shape[0]
        assert self._actions.shape[1] == dU, 'Incorrect size of actions wrt n_dofs'
        self._counter = 0
        # assert "action_list" in self._data, "Incorrect/missing parameters for initialization."

    def _action(self, x, obs, time, noise):
        """
        Returns the action an agent following this policy would
        take, given a STATE, OBSERVATION, and TIMESTEP.

        Note: if time_step >= TIMEHORIZON, this function
        returns an action that does nothing.
        """
        # TODO: use time instead of _counter ?
        out = self._actions[min(self.n_timestep - 1, self._counter), :]
        out = out.ravel()
        self._counter += 1
        return out


class linearController(Policy):
    def __init__(self, dX, dU, A, B):
        """
        linear controller of the form A*x + B
        :param n_dof:
        :param A:
        :param B:
        """
        Policy.__init__(dX=dX, dU=dU)
        self.A = np.matrix(A)  # I'm confused by this. Aren't matrices necessarily 2D
        self.B = np.matrix(B)
        if self.A.ndim is 3:
            self.timevariant = True
        else:
            self.timevariant = False

    def _action(self, x, obs, time, noise):
        if self.timevariant is True:
            return self.A[time, :, :] * x + self.B[time, :]
        else:
            return self.A * x + self.B


class PID(Policy):
    """
    Proportional-integral-derivative controller.
    """

    def __init__(self, dX, dU, P, I, D, target):
        """
        :param dX: unused
        :param dU: dimensionality of state and control signal
        :param P: proportional control coeff
        :param I: integral control coeff
        :param D: derivative control coeff
        :param target: setpoint
        """
        Policy.__init__(self, dX=dX, dU=dU)
        self.n_dof = dU
        # TODO: fix dimensionality with P
        if isinstance(P, int):
            self.Kp = np.tile(P, self.n_dof)
        else:
            self.Kp = P
        if isinstance(I, int):
            self.Ki = np.tile(I, self.n_dof)
        else:
            self.Ki = I
        if isinstance(D, int):
            self.Kd = np.tile(D, self.n_dof)
        else:
            self.Kd = D
        self.target = target
        self.prev_error = 0
        self.error = 0
        # self.cum_error = 0
        # self.I_count = 0

    #    should be _action(self, x, obs, time, noise):
    # def _action(self, q, q_des):
    def _action(self, x, obs, time, noise):
        q_des = self.target
        q = x

        self.error = q_des - q
        P_value = self.Kp * self.error
        I_value = 0  # TODO: implement I and D part
        D_value = self.Kd * (self.error - self.prev_error)  # + self.D*(qd_des-qd)
        self.prev_error = self.error
        action = P_value + I_value + D_value
        return action

    def get_P(self):
        return self.Kp

    def get_I(self):
        return self.Ki

    def get_D(self):
        return self.Kd

    def get_target(self):
        return self.target


class jointsTrajectoryTrackingPID(PID):
    """
    jointsTrajectoryTrackingPID
    """

    def __init__(self, dX, dU, trajectory, P=1, I=0, D=0, id_states=np.arange(7), verbosity=1):
        """

        :param dX:
        :param dU:
        :param trajectory:
        :param P:
        :param I:
        :param D:
        :param id_states: numpy array with the index of the elements of dX that will be used in the controller
        :param verbosity:
        """
        PID.__init__(self, dX=dX, dU=dU, P=P, I=I, D=D)
        if trajectory.ndim is 1:
            # Single point
            self.trajectory = np.expand_dims(trajectory, axis=0)
        else:
            self.trajectory = trajectory
        assert self.trajectory.shape[1] == self.n_dof, \
            'Number of DOF in the trajectory does not match the number of DOF specified: %d != %d' % (
                self.trajectory.shape[1], self.n_dof)
        self.n_timestep = self.trajectory.shape[0]
        self.id_states = id_states
        self.verbosity = verbosity

    def _action(self, x, obs, time, noise):
        q = x[self.id_states]  # Only care about the position
        if time >= self.n_timestep:
            # Keep the controller stable at the last position
            action = PID._action(self, q, self.trajectory[self.n_timestep - 1, :])
        else:
            action = PID._action(self, q, self.trajectory[time, :])
            # TODO: use min(self.n_timestep-1, time)
        if self.verbosity > 1:
            print(action)
        return action


class LQR(Policy):
    def __init__(self, A, B, Q, R, actionBounds=None, horizon=10, K = None):
        '''
        :param n_dof:
        :param trajectory:
        :param horizon: Horizon
        '''
        Policy.__init__(self, dX=np.shape(A)[1], dU=np.shape(B)[1], actionBounds=actionBounds)
        from control import lqr
        # from scipy.linalg import solve_continuous_are, solve_discrete_are
        self.T = horizon
        self.A = A
        self.B = B
        self.Q = Q
        self.R = R
        # self.controller = self.compute_controller()
        # self.K = solve_continuous_are(A, B, Q, R)
        if K is None:
            self.K, S, E = lqr(A, B, Q, R)
        else:
            self.K = K

    def compute_controller(self):
        # TODO: implement me!!!
        raise NotImplementedError
        K = 0
        k = 0
        controller = linearController(A=K, B=k)
        return 0

    def get_K(self):
        return self.K

    def _action(self, x, obs, time, noise):
        u = -np.matmul(self.K, x)
        return np.array(u).squeeze()
        # return self.controller.action(x, obs, time, noise)

class Net(nn.Module):
    def __init__(self, h_layers, h_width, n_in, n_out):
        super(Net, self).__init__()
        layers = []
        layers.append(nn.Linear(n_in, h_width))
        layers.append(nn.ReLU())
        for d in range(h_layers):
            layers.append(nn.Linear(h_width, h_width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(h_width, n_out))
        self.model = nn.Sequential(*layers)

    def update_params(self, custom_weights):
        for name, param in self.model.named_parameters():
            param.data.copy_(custom_weights['model.' + name])

    def forward(self, x):
        return self.model(x)


class NN(Policy):
    def __init__(self, h_layers, h_width, n_in, n_out):
        Policy.__init__(self, dX=n_in, dU=n_out)
        self.h_layers = h_layers
        self.h_width = h_width
        self.n_in = n_in
        self.n_out = n_out
        self.model = Net(h_layers, h_width, n_in, n_out)

    def update_params(self, custom_weights):
        model_num = 0
        list_index = self.n_in * self.h_width
        custom_weight_torch = {}
        custom_weight_torch['model.0.weight'] = torch.from_numpy(custom_weights[:list_index].reshape((self.h_width, self.n_in)))
        custom_weight_torch['model.0.bias'] = torch.from_numpy(custom_weights[list_index:list_index+self.h_width])
        list_index += self.h_width
        for d in range(self.h_layers):
            model_num += 2
            custom_weight_torch['model.' + str(model_num) + '.weight'] = torch.from_numpy(custom_weights[list_index:list_index+(self.h_width*self.h_width)].reshape((self.h_width, self.h_width)))
            list_index += (self.h_width*self.h_width)
            custom_weight_torch['model.' + str(model_num) + '.bias'] = torch.from_numpy(custom_weights[list_index:list_index+self.h_width])
            list_index += self.h_width
        model_num +=2
        custom_weight_torch['model.' + str(model_num) + '.weight'] = torch.from_numpy(custom_weights[list_index:list_index+(self.h_width)])
        custom_weight_torch['model.' + str(model_num) + '.bias'] = torch.from_numpy(custom_weights[list_index+(self.h_width):])
        self.model.update_params(custom_weight_torch)

    def _action(self, x, obs, time, noise):
        action = (self.model.forward(torch.from_numpy(x).float())).detach().numpy()
        return np.clip(action, -1, 1)
