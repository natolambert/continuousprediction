import sys
import warnings
import os

# import matplotlib.cbook
#
# warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
# warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from dotmap import DotMap
from sklearn.utils import shuffle

from timeit import default_timer as timer
import matplotlib.pyplot as plt

import mujoco_py
import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
import gym
from envs import *
from gym.wrappers import Monitor

import hydra
import logging

log = logging.getLogger(__name__)

from policy import PID
from plot import plot_cf, plot_loss, setup_plotting
from dynamics_model import DynamicsModel
from reacher_pd import run_controller, create_dataset_step, create_dataset_traj


class PidPolicy:
    """
    Setup to run with a PID operating on pitch, then roll, then yaw.
    """
    def __init__(self, parameters, cfg):
        # super(PidPolicy, self).__init__(cfg)
        # cfg = cfg[cfg.policy.mode]
        self.pids = []
        self.cfg = cfg
        self.mode = self.cfg.params.mode


        self.random = False
        # assert len(cfg.params.min_pwm) == len(cfg.params.equil)
        # assert len(cfg.params.max_pwm) == len(cfg.params.equil)

        # bounds of values
        self.min_pwm = self.cfg.params.min_pwm
        self.max_pwm = self.cfg.params.max_pwm
        self.equil = self.cfg.params.equil
        self.max_int = self.cfg.params.int_max

        # how actions translate to Euler angles
        self.p_m = self.cfg.params.pitch_mult
        self.r_m = self.cfg.params.roll_mult
        self.pry = self.cfg.params.pry

        self.dt = self.cfg.params.dt
        self.numParameters = 0

        # order: pitch, roll, yaw, pitchrate, rollrate, yawRate or pitch roll yaw yawrate for hybrid or pitch roll yaw for euler
        if self.mode == 'BASIC':
            self.numpids = 2
            self.numParameters = 4
        elif self.mode == 'INTEG':
            self.numpids = 2
            self.numParameters = 6
        elif self.mode == 'EULER':
            self.numpids = 3
            self.numParameters = 9
        else:
            raise ValueError(f"Mode Not Supported {self.mode}")

        for set in parameters:
            """
            def __init__(self, desired,
                 kp, ki, kd,
                 ilimit, dt, outlimit=np.inf,
                 samplingRate=0, cutoffFreq=-1,
                 enableDFilter=False):
             """
            dX = 1
            dU = 1
            P = set[0]
            I = set[1]
            D = set[2]
            self.pids += [PID(dX, dU, P, I, D, target=0)]

    def set_params(self, parameters):
        for i, set in enumerate(parameters):
            """
            def __init__(self, desired,
                 kp, ki, kd,
                 ilimit, dt, outlimit=np.inf,
                 samplingRate=0, cutoffFreq=-1,
                 enableDFilter=False):
             """
            pid = self.pids[i]
            pid.kp = set[0]
            pid.ki = set[1]
            pid.kd = set[2]

    def get_action(self, state, metric=None):
        if self.random:
            output = np.squeeze(np.random.uniform(low=self.min_pwm, high=self.max_pwm, size=(4,)))
            self.last_action = np.array(output)
            return output, True

        # PIDs must always come in order of states then
        actions = []
        for i, pid in enumerate(self.pids):
            # pid.update(state[self.pry[i]])
            actions.append(pid._action(state[self.pry[i]], 0,0,0))


        def limit_thrust(pwm):  # Limits the thrust
            return np.clip(pwm, self.min_pwm, self.max_pwm)

        output = [0, 0, 0, 0]
        # PWM structure: 0:front right  1:front left  2:back left   3:back right
        '''Depending on which PID mode we are in, output the respective PWM values based on PID updates'''
        if self.mode == 'BASIC' or self.mode == 'INTEG':
            # output[0] = limit_thrust(self.equil[0] + self.p_m[0] * self.pids[0].out + self.r_m[0] * self.pids[1].out)
            # output[1] = limit_thrust(self.equil[1] + self.p_m[1] * self.pids[0].out + self.r_m[1] * self.pids[1].out)
            # output[2] = limit_thrust(self.equil[2] + self.p_m[2] * self.pids[0].out + self.r_m[2] * self.pids[1].out)
            # output[3] = limit_thrust(self.equil[3] + self.p_m[3] * self.pids[0].out + self.r_m[3] * self.pids[1].out)
            output[0] = limit_thrust(self.equil[0] + self.p_m[0] * actions[0] + self.r_m[0] * actions[1])
            output[1] = limit_thrust(self.equil[1] + self.p_m[1] * actions[0] + self.r_m[1] * actions[1])
            output[2] = limit_thrust(self.equil[2] + self.p_m[2] * actions[0] + self.r_m[2] * actions[1])
            output[3] = limit_thrust(self.equil[3] + self.p_m[3] * actions[0] + self.r_m[3] * actions[1])
        else:
            raise NotImplementedError("Other PID Modes not updated")

        self.last_action = np.array(output)
        return np.array(output) #, True


    def reset(self):
        self.interal = 0
        [p.reset() for p in self.pids]

    # def update(self, states):
    #     '''
    #
    #     :param states:
    #     :return:
    #     Order of states being passed: pitch, roll, yaw
    #     Updates the PID outputs based on the states being passed in (must be in the specified order above)
    #     Order of PIDs: pitch, roll, yaw, pitchRate, rollRate, yawRate
    #     '''
    #     assert len(states) == 3
    #     EulerOut = [0, 0, 0]
    #     for i in range(3):
    #         EulerOut[i] = self.pids[i].update(states[i])
    #     if self.mode == 'HYBRID':
    #         self.pids[3].update(EulerOut[2])
    #     if self.mode == 'RATE' or self.mode == 'ALL':
    #         for i in range(3):
    #             self.pids[i + 3].update(EulerOut[i])


def run_controller(env, horizon, policy, video = False):
    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    print(f"Initial RPY {observation[3:6]}")
    for i in range(horizon):
        if(video):
            env.render()
        state = observation
        action = policy.get_action(state[3:6])
        # actions = equil+

        # print(action)

        observation, reward, done, info = env.step(action)

        if done:
            logs.actions = np.array(logs.actions)
            logs.rewards = np.array(logs.rewards)
            logs.states = np.array(logs.states)
            return logs

        # Log
        # logs.times.append()
        logs.actions.append(action)
        logs.rewards.append(reward)
        logs.states.append(observation.squeeze())

    # Cluster state
    # print(f"Rollout completed, cumulative reward: {np.sum(logs.rewards)}")
    logs.actions = np.array(logs.actions)
    logs.rewards = np.array(logs.rewards)
    logs.states = np.array(logs.states)
    return logs


def collect_data(cfg, plot=True):  # Creates horizon^2/2 points
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a trajectory
    """

    env_model = cfg.env.name
    env = gym.make(env_model)
    #if (cfg.video):
        #env = Monitor(env, hydra.utils.get_original_cwd() + '/trajectories/reacher/video',
         #video_callable = lambda episode_id: episode_id==1,force=True)
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []
    if (cfg.PID_test):
        target = np.random.rand(5) * 2 - 1
    s = np.random.randint(0, 100)
    for i in range(cfg.num_trials):
        log.info('Trial %d' % i)
        env.seed(s + i)

        P = 100+np.random.rand(2) * 10000
        I = np.zeros(2)
        D = 10+np.random.rand(2) * 50000

        # Samples target uniformely from [-1, 1]
        if (not cfg.PID_test):
            # target = np.random.rand(5) * 2 - 1
            target = np.array([0, 0])

        parameters = [[P[0], 0, D[0]],
                      [P[1], 0, D[1]]]
        policy = PidPolicy(parameters, cfg.pid)

        dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy, video=cfg.video)

        flag = (abs(np.rad2deg(dotmap.states[-1][3])) < 5) and (abs(np.rad2deg(dotmap.states[-1][4])) < 5)
        # print(flag)
        while len(dotmap.states) < cfg.trial_timesteps or not flag:
            print(f"- Repeat simulation")
            env.seed(s)
            s0 = env.reset()
            P = 100+np.random.rand(2) * 10000
            I = np.zeros(2)
            D = 10+np.random.rand(2) * 50000

            # Samples target uniformely from [-1, 1]
            if (not cfg.PID_test):
                # target = np.random.rand(5) * 2 - 1
                target = np.array([0, 0])

            parameters = [[P[0], 0, D[0]],
                          [P[1], 0, D[1]]]
            policy = PidPolicy(parameters, cfg.pid)

            dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy, video=cfg.video)
            flag = (abs(np.rad2deg(dotmap.states[-1][3])) < 5) and (abs(np.rad2deg(dotmap.states[-1][4])) < 5)
            # print(flag)
            s += 1

        if plot: plot_cf(dotmap.states, dotmap.actions)


        # policy = PID(dX=2, dU=2, P=P, I=I, D=D, target=target)
        # print(type(env))
        dotmap.target = target
        dotmap.P = P
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)

    return logs


###########################################
#           Plotting / Output             #
###########################################


def log_hyperparams(cfg):
    log.info(cfg.model.str + ":")
    log.info("  hid_width: %d" % cfg.model.training.hid_width)
    log.info('  hid_depth: %d' % cfg.model.training.hid_depth)
    log.info('  epochs: %d' % cfg.model.optimizer.epochs)
    log.info('  batch size: %d' % cfg.model.optimizer.batch)
    log.info('  optimizer: %s' % cfg.model.optimizer.name)
    log.info('  learning rate: %f' % cfg.model.optimizer.lr)


###########################################
#             Main Functions              #
###########################################

@hydra.main(config_path='conf/crazyflie_pd.yaml')
def contpred(cfg):

    train = cfg.mode == 'train'

    # Collect data
    if not train:
        log.info(f"Collecting new trials")

        exper_data = collect_data(cfg, plot=cfg.plot)
        test_data = collect_data(cfg, plot=cfg.plot)

        log.info("Saving new default data")
        torch.save((exper_data, test_data),
                   hydra.utils.get_original_cwd() + '/trajectories/crazyflie/' + 'raw' + cfg.data_dir)
        log.info(f"Saved trajectories to {'/trajectories/crazyflie/' + 'raw' + cfg.data_dir}")
    # Load data
    else:
        log.info(f"Loading default data")
        # raise ValueError("Current Saved data old format")
        # Todo re-save data
        (exper_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/crazyflie/' + 'raw' + cfg.data_dir)

    if train:
        it = range(cfg.copies) if cfg.copies else [0]
        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")

        log_hyperparams(cfg)

        for i in it:
            print('Training model %d' % i)
            if traj:
                dataset = create_dataset_traj(exper_data, control_params=cfg.model.training.control_params,
                                              train_target=cfg.model.training.train_target,
                                              threshold=cfg.model.training.filter_rate,
                                              t_range=cfg.model.training.t_range)
            else:
                dataset = create_dataset_step(exper_data, delta=delta, t_range=cfg.model.training.t_range)

            model = DynamicsModel(cfg)
            train_logs, test_logs = model.train(dataset, cfg)

            setup_plotting({cfg.model.str: model})
            plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)

            log.info("Saving new default models")
            f = hydra.utils.get_original_cwd() + '/models/crazyflie/'
            if cfg.exper_dir:
                f = f + cfg.exper_dir + '/'
                if not os.path.exists(f):
                    os.mkdir(f)
            copystr = "_%d" % i if cfg.copies else ""
            f = f + cfg.model.str + copystr + '.dat'
            if cfg.model.gp:
                model._save_model(f)
            else:
                torch.save(model, f)

if __name__ == '__main__':
    sys.exit(contpred())
