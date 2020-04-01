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

import hydra
import logging

log = logging.getLogger(__name__)

from policy import PID
from plot import plot_reacher, plot_loss

from dynamics_model import DynamicsModel


###########################################
#                Datasets                 #
###########################################

def create_dataset_traj(data, control_params=True, threshold=0, delta=False):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    threshold: the probability of dropping a given data entry
    """
    data_in, data_out = [], []
    for sequence in data:
        states = sequence.states
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
                if np.random.random() < threshold:
                    continue
                if control_params:
                    data_in.append(np.hstack((states[i], j - i, P, D, target)))
                else:
                    data_in.append(np.hstack((states[i], j - i)))
                # data_in.append(np.hstack((states[i], j-i, target)))
                if delta:
                    data_out.append(states[j]-states[i])
                else:
                    data_out.append(states[j])

    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset_step(data, delta=True):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for sequence in data:
        for i in range(sequence.states.shape[0] - 1):
            if 'actions' in sequence.keys():
                data_in.append(np.hstack((sequence.states[i], sequence.actions[i])))
                if delta:
                    data_out.append(sequence.states[i + 1] - sequence.states[i])
                else:
                    data_out.append(sequence.states[i + 1])
            else:
                data_in.append(np.array(sequence.states[i]))
                if delta:
                    data_out.append(sequence.states[i + 1] - sequence.states[i])
                else:
                    data_out.append(sequence.states[i + 1])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def run_controller(env, horizon, policy):
    """
    Runs a Reacher3d gym environment for horizon timesteps, making actions according to policy

    :param env: A gym object
    :param horizon: The number of states forward to look
    :param policy: A policy object (see other python file)
    """

    # WHat is going on here?
    # nol 29 feb - action only acts on first 5 variables
    def obs2q(obs):
        return obs[0:5]

    logs = DotMap()
    logs.states = []
    logs.actions = []
    logs.rewards = []
    logs.times = []

    observation = env.reset()
    for t in range(horizon):
        # env.render()
        state = observation
        action, t = policy.act(obs2q(state))

        # print(action)

        observation, reward, done, info = env.step(action)

        # Log
        # logs.times.append()
        logs.actions.append(action)
        logs.rewards.append(reward)
        logs.states.append(observation)

    # Cluster state
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
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []

    for i in range(cfg.num_trials):
        log.info('Trial %d' % i)
        env.seed(i)
        s0 = env.reset()

        # P = np.array([4, 4, 1, 1, 1])
        P = np.random.rand(5) * 5
        I = np.zeros(5)
        # D = np.array([0.2, 0.2, 2, 0.4, 0.4])
        D = np.random.rand(5)

        # Samples target uniformely from [-1, 1]
        target = np.random.rand(5) * 2 - 1

        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
        # print(type(env))

        dotmap = run_controller(env, horizon=cfg.trial_timesteps, policy=policy)
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


def collect_and_dataset(cfg):
    """
    Collects data and returns it as a dataset in the format used for training

    Params:
        cfg: the hydra configuration object thing

    Returns:
        dataset: one step dataset, a tuple (data_in, data_out)
        dataset_no_t: traj dataset, "    "       "     "      "
        training_data: an array of dotmaps, each pertaining to a test trajectory
        test_data: an array of dotmaps, each pertaining to a test trajectory
    """
    log.info('Collecting data')
    exper_data = collect_data(cfg, plot=False)  # 50
    log.info('Creating dataset')
    dataset = create_dataset_traj(exper_data, threshold=1.0)
    dataset_no_t = create_dataset_step(exper_data)  # train_data[0].states)
    return dataset, dataset_no_t, exper_data


###########################################
#           Plotting / Output             #
###########################################


def log_hyperparams(cfg):  # , configs, model_types):
    log.info("General Hyperparams:")
    log.info("  traj_len: %d" % cfg.experiment.traj_len)
    log.info('  traj_len_test: %d' % cfg.experiment.traj_len_test)
    log.info('  num_traj: %d' % cfg.experiment.num_traj)
    log.info('  num_traj_test: %d' % cfg.experiment.num_traj_test)

    # for type in model_types:
    #     conf = configs[type]
    # log.info(type)
    log.info("  hid_width: %d" % cfg.model.training.hid_width)
    log.info('  hid_depth: %d' % cfg.model.training.hid_depth)
    log.info('  epochs: %d' % cfg.model.optimizer.epochs)
    log.info('  batch size: %d' % cfg.model.optimizer.batch)
    log.info('  optimizer: %s' % cfg.model.optimizer.name)
    log.info('  learning rate: %f' % cfg.model.optimizer.lr)


###########################################
#             Main Functions              #
###########################################

@hydra.main(config_path='conf/train.yaml')
def contpred(cfg):
    log_hyperparams(cfg)  # configs, model_types)

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
        # traj_dataset = create_dataset_traj(exper_data, threshold=0.0)
        # one_step_dataset = create_dataset_step(exper_data)  # train_data[0].states)

    prob = cfg.model.prob
    traj = cfg.model.traj
    ens = cfg.model.ensemble
    delta = cfg.model.delta

    # for model_type in model_types:
    log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")
    # model_file = 'model_%s.pth.tar' % model_type

    # dataset = traj_dataset if traj else one_step_dataset

    if train:
        if traj:
            dataset = create_dataset_traj(exper_data, threshold=0.95)
        else:
            dataset = create_dataset_step(exper_data, delta=delta)

        # model = Model(cfg.model)
        # model.train(cfg, dataset)
        # loss_log = model.loss_log
        model = DynamicsModel(cfg)
        train_logs, test_logs = model.train(dataset, cfg)

        plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)
        # plot_loss_epoch(loss_log, save_loc=graph_file, show=False, s=cfg.model.str)

        log.info("Saving new default models")
        f =  hydra.utils.get_original_cwd() + '/models/reacher/'
        if cfg.exper_dir:
            f = f + cfg.exper_dir + '/'
        f = f + cfg.model.str + '.dat'
        torch.save(model, f)
        # torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless

    else:
        pass
        # TODO: Not sure what we would put in here if the point of this function is to sweep and train models
        # model_1s = torch.load(hydra.utils.get_original_cwd() + '/models/reacher/' + 'step' + cfg.model_dir)
        # model_ct = torch.load(hydra.utils.get_original_cwd() + '/models/reacher/' + 'traj' + cfg.model_dir)



if __name__ == '__main__':
    sys.exit(contpred())
