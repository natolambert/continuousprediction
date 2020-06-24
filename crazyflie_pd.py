import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

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
from plot import plot_reacher, plot_loss, setup_plotting
from dynamics_model import DynamicsModel
from reacher_pd import run_controller, create_dataset_step, create_dataset_traj

def run_controller(env, horizon, policy, video = False):
    # nol 29 feb - action only acts on first 5 variables
    def obs2q(obs):
        if len(obs) < 5:
            return obs
        else:
            return obs[0:5]

    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    for i in range(horizon):
        if(video):
            env.render()
        state = observation
        us, t = policy.act(state[:2])
        actions = equil+

        # print(action)

        observation, reward, done, info = env.step(action)

        if done:
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


def collect_data(cfg, plot=False):  # Creates horizon^2/2 points
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
        if (cfg.PID_test):
            env.seed(0)
        else:
            env.seed(s + i)
        s0 = env.reset()

        P = np.random.rand(2) * 100
        I = np.zeros(2)
        D = np.random.rand(2)*10

        # Samples target uniformely from [-1, 1]
        if (not cfg.PID_test):
            # target = np.random.rand(5) * 2 - 1
            target = np.array([0,0])

        policy = PID(dX=2, dU=2, P=P, I=I, D=D, target=target)
        # print(type(env))
        dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy, video = cfg.video)

        dotmap.target = target
        dotmap.P = P / 5
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)

    if plot:
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.update_layout(
            width=1500,
            height=800,
            autosize=False,
            scene=dict(
                camera=dict(
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    eye=dict(
                        x=0,
                        y=1.0707,
                        z=1,
                    )
                ),
                aspectratio=dict(x=1, y=1, z=0.7),
                aspectmode='manual'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        for d in logs:
            states = d.states
            actions = d.actions
            plot_reacher(states, actions)

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

        exper_data = collect_data(cfg)
        test_data = collect_data(cfg)

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

            if cfg.model.training.num_traj:
                train_data = exper_data[:cfg.model.training.num_traj]
            else:
                train_data = exper_data

            if traj:
                dataset = create_dataset_traj(exper_data, control_params=cfg.model.training.control_params,
                                              train_target=cfg.model.training.train_target,
                                              threshold=cfg.model.training.filter_rate,
                                              t_range=cfg.model.training.t_range)
            else:
                dataset = create_dataset_step(train_data, delta=delta)

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
            torch.save(model, f)

if __name__ == '__main__':
    sys.exit(contpred())
