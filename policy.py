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
        :param x:
        :param obs:
        :param time:
        :param noise:
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
            a = np.maximum(np.minimum(a, self.bounds.get_max()), self.bounds.get_min())
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
        out = self._actions[min(self.n_timestep-1, self._counter), :]
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
        self.A = np.matrix(A) # I'm confused by this. Aren't matrices necessarily 2D
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
    def __init__(self, dX, dU, P, I, D, target): # added 'target' because the code was broken without it. it does nothing
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
        self.prev_error = 0
        self.error = 0
        
#    should be _action(self, x, obs, time, noise):
    def _action(self, q, q_des):
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
            'Number of DOF in the trajectory does not match the number of DOF specified: %d != %d' % (self.trajectory.shape[1], self.n_dof)
        self.n_timestep = self.trajectory.shape[0]
        self.id_states = id_states
        self.verbosity = verbosity

    def _action(self, x, obs, time, noise):
        q = x[self.id_states]  # Only care about the position
        if time >= self.n_timestep:
            # Keep the controller stable at the last position
            action = PID._action(self, q, self.trajectory[self.n_timestep-1, :])
        else:
            action = PID._action(self, q, self.trajectory[time, :])
            # TODO: use min(self.n_timestep-1, time)
        if self.verbosity > 1:
            print(action)
        return action

# class LQG(Policy):
#     def __init__(self, n_dof, A, B, Q, R, horizon=10):
#         '''
#
#         :param n_dof:
#         :param trajectory:
#         :param horizon: Horizon
#         '''
#         Policy.__init__(self, n_dof=n_dof)
#         self.T = horizon
#         self.A = A
#         self.B = B
#         self.Q = Q
#         self.R = R
#         self.controller = self.compute_controller()
#
#     def compute_controller(self):
#         # TODO: implement me!!!
#         K = 0
#         k = 0
#         controller = linearController(A=K, B=k)
#         return 0
#
#     def _action(self, x, obs, time, noise):
#         return self.controller.action(x, obs, time, noise)
