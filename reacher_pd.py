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


###########################################
#                Datasets                 #
###########################################

def create_dataset_traj(data, control_params=True, train_target=True, threshold=0.0, delta=False, t_range=0, is_lstm=False):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    threshold: the probability of dropping a given data entry
    """
    data_in, data_out = [], []
    for id, sequence in enumerate(data):
        if id % 5 == 0: print(f"- processing seq {id}")
        states = sequence.states
        if t_range > 0:
            states = states[:t_range]
        if not is_lstm:
            if id > 99:
                continue
        else:
            if id > 100-int(threshold*100):
                continue
        P = sequence.P
        D = sequence.D
        target = sequence.target
        n = states.shape[0]
        for i in range(n):  # From one state p
            for j in range(i + 1, n):
                # This creates an entry for a given state concatenated
                # with a number t of time steps as well as the PID parameters

                # The randomely continuing is something I thought of to shrink
                # the datasets while still having a large variety

                if np.random.random() < threshold and not is_lstm:
                    continue
                dat = [states[i], j - i]
                if control_params:
                    dat.extend([P, D])
                if train_target:
                    dat.append(target)
                data_in.append(np.hstack(dat))
                # data_in.append(np.hstack((states[i], j-i, target)))
                if delta:
                    data_out.append(states[j] - states[i])
                else:
                    data_out.append(states[j])
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    return data_in, data_out


def create_dataset_step(data, delta=True, t_range=0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence.states
        if t_range > 0:
            states = states[:t_range]
        for i in range(states.shape[0] - 1):
            if 'actions' in sequence.keys():
                actions = sequence.actions
                if t_range:
                    actions = actions[:t_range]
                data_in.append(np.hstack((states[i], actions[i])))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
            else:
                data_in.append(np.array(states[i]))
                if delta:
                    data_out.append(states[i + 1] - states[i])
                else:
                    data_out.append(states[i + 1])
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    return data_in, data_out


def run_controller(env, horizon, policy, video=False):
    """
    Runs a Reacher3d gym environment for horizon timesteps, making actions according to policy

    :param env: A gym object
    :param horizon: The number of states forward to look
    :param policy: A policy object (see other python file)
    """

    # WHat is going on here?
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
        if (video):
            env.render()
        state = observation
        action, t = policy.act(obs2q(state))

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
    # if (cfg.video):
    # env = Monitor(env, hydra.utils.get_original_cwd() + '/trajectories/reacher/video',
    # video_callable = lambda episode_id: episode_id==1,force=True)
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []
    if (cfg.PID_test):
        target = np.random.rand(5) * 2 - 1
    for i in range(cfg.num_trials):
        log.info('Trial %d' % i)
        if (cfg.PID_test):
            env.seed(0)
        else:
            env.seed(i)
        s0 = env.reset()

        # P = np.array([4, 4, 1, 1, 1])
        P = np.random.rand(5) * 5
        I = np.zeros(5)
        # D = np.array([0.2, 0.2, 2, 0.4, 0.4])
        D = np.random.rand(5)

        # Samples target uniformely from [-1, 1]
        if (not cfg.PID_test):
            target = np.random.rand(5) * 2 - 1

        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
        # print(type(env))
        dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy, video=cfg.video)

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

@hydra.main(config_path='conf/reacher_pd.yaml')
def contpred(cfg):
    print(cfg.pretty())

    train = cfg.mode == 'train'

    # Collect data
    if not train:
        log.info(f"Collecting new trials")

        exper_data = collect_data(cfg)
        test_data = collect_data(cfg)

        log.info("Saving new default data")
        torch.save((exper_data, test_data),
                   hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)
        log.info(f"Saved trajectories to {'/trajectories/reacher/' + 'raw' + cfg.data_dir}")
    # Load data
    else:
        log.info(f"Loading default data")
        # raise ValueError("Current Saved data old format")
        # Todo re-save data
        (exper_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    if train:
        it = range(cfg.copies) if cfg.copies else [0]
        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta
        is_lstm = cfg.model.lstm

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
                                              t_range=cfg.model.training.t_range,
                                              is_lstm = is_lstm)
            else:
                dataset = create_dataset_step(train_data, delta=delta)

            model = DynamicsModel(cfg)
            train_logs, test_logs = model.train(dataset, cfg)

            setup_plotting({cfg.model.str: model})
            plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)

            log.info("Saving new default models")
            f = hydra.utils.get_original_cwd() + '/models/reacher/'
            if cfg.exper_dir:
                f = f + cfg.exper_dir + '/'
                if not os.path.exists(f):
                    os.mkdir(f)
            copystr = "_%d" % i if cfg.copies else ""
            f = f + cfg.model.str + copystr + '.dat'
            torch.save(model, f)
        # torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless


if __name__ == '__main__':
    sys.exit(contpred())
