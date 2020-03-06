import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

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
# import gym
from envs import *

import hydra
import logging

log = logging.getLogger(__name__)

from policy import PID
from mbrl_resources import *
from plot import plot_reacher

from dynamics_model import DynamicsModel

###########################################
#                Datasets                 #
###########################################

def create_dataset_traj(data, control_params=True, threshold=0):
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
                data_out.append(states[j])

    # TODO we maybe don't need this, the data processor should handle it
    # print("shuffling")
    # data_in, data_out = shuffle(data_in, data_out)
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
            data_in.append(np.hstack((sequence.states[i], sequence.actions[i])))
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

def test_traj_ensemble(ensemble, test_data):
    """
    TODO: this probably doesn't belong in this file
    Tests each model in the ensemble on one test trajectory and plots the output
    """
    traj = test_data
    states = traj.states
    actions = traj.actions
    initial = states[0, :]

    model_predictions = [[] for _ in range(ensemble.n)]
    ensemble_predictions = []
    for i in range(1, states.shape[0]):
        x = np.hstack((initial, i, traj.P, traj.D, traj.target))
        ens_pred = ensemble.predict(x)
        ensemble_predictions.append(ens_pred.squeeze())
        for j in range(len(ensemble.models)):
            model = ensemble.models[j]
            model_pred = model.predict(x)
            model_predictions[j].append(model_pred.squeeze())

    ensemble_predictions = np.array(ensemble_predictions)
    model_predictions = [np.array(x) for x in model_predictions]
    # print(len(model_predictions))

    for i in range(7):
        fig, ax = plt.subplots()
        gt = states[:, i]
        plt.title("Predictions on one dimension")
        plt.xlabel("Timestep")
        plt.ylabel("State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.plot(gt, c='k', label='Groundtruth')
        plt.plot(ensemble_predictions[:, i])
        for pred in model_predictions:
            # print(pred.shape)
            plt.plot(pred[:, i], c='b')

        plt.legend()

        plt.show()


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


# def make_evaluator(train_data, test_data, type):
#     def evaluator(model):
#         dic = {type: model}
#         train_s = 0
#         num_train = len(train_data)
#         len_train = len(train_data[0].states)
#         denom_train = num_train * len_train
#         for traj in train_data:
#             outcomes = test_models(traj, dic)
#             train_s += np.sum(outcomes['mse'][type]) / denom_train
#         test_s = 0
#         num_test = len(test_data)
#         len_test = len(test_data[0].states)
#         denom_test = num_test * len_test
#         for traj in test_data:
#             outcomes = test_models(traj, dic)
#             test_s += np.sum(outcomes['mse'][type]) / denom_test
#         return (train_s, test_s)
#
#     return evaluator


###########################################
#             Main Functions              #
###########################################

@hydra.main(config_path='conf/config.yaml')
def contpred(cfg):
    log_hyperparams(cfg)  # configs, model_types)

    graph_file = 'graphs'
    os.mkdir(graph_file)

    # Collect data
    if cfg.collect_data:
        log.info(f"Collecting new trials")
        # traj_dataset, one_step_dataset, exper_data = collect_and_dataset(cfg)

        exper_data = collect_data(cfg)
        test_data = collect_data(cfg)

        if cfg.save_data:
            log.info("Saving new default data")
            torch.save((exper_data, test_data),
                       hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)
            log.info(f"Saved trajectories to {'/trajectories/reacher/' + 'raw' + cfg.data_dir}")
    # Load data
    else:
        log.info(f"Loading default data")
        # raise ValueError("Current Saved data old format")
        #Todo re-save data
        (exper_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)
        # traj_dataset = create_dataset_traj(exper_data, threshold=0.0)
        # one_step_dataset = create_dataset_step(exper_data)  # train_data[0].states)

    prob = cfg.model.prob
    traj = cfg.model.traj
    ens = cfg.model.ensemble

    # for model_type in model_types:
    log.info(f"Training model P:{[prob]}, T:{traj}, E:{ens}")
    # model_file = 'model_%s.pth.tar' % model_type

    # dataset = traj_dataset if traj else one_step_dataset

    if cfg.train_models:
        if traj:
            dataset = create_dataset_traj(exper_data, threshold=0.95)
        else:
            dataset = create_dataset_step(exper_data)

        # model = Model(cfg.model)
        # model.train(cfg, dataset)
        # loss_log = model.loss_log
        model = DynamicsModel(cfg)
        train_logs, test_logs = model.train(dataset, cfg)

        # plot_loss(loss_log, save_loc=graph_file, show=False, s=cfg.model.str)
        # plot_loss_epoch(loss_log, save_loc=graph_file, show=False, s=cfg.model.str)

        if cfg.save_models:
            # TODO: this gives a bunch of warnings, fix
            log.info("Saving new default models")
            torch.save(model,
                       hydra.utils.get_original_cwd() + '/models/reacher/' + cfg.model.str + cfg.model_dir)
        torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless

    else:
        pass
        # TODO: Not sure what we would put in here if the point of this function is to sweep and train models
        # model_1s = torch.load(hydra.utils.get_original_cwd() + '/models/reacher/' + 'step' + cfg.model_dir)
        # model_ct = torch.load(hydra.utils.get_original_cwd() + '/models/reacher/' + 'traj' + cfg.model_dir)

    # mse_t, mse_no_t, predictions_t, predictions_no_t = test_model_single(test_data[0], model, model_no_t)
    # raise ValueError("Test data needs to be regenerated")
    # for i in range(len(test_data)):
    #     test = test_data[i]
    #     file = "%s/test%d" % (graph_file, i + 1)
    #     os.mkdir(file)
    #
    #     outcomes = test_model(test, model)
    #     plot_states(test.states, outcomes['predictions'], idx_plot=[0, 1, 2, 3, 4, 5, 6], save_loc=file, show=False)
    #     plot_mse(outcomes['mse'], save_loc=file, show=False)

    # train_data_sample =

    # Blocking this since it doesn't quite work
    if False:
        # Evaluate learned model
        def augment_state(state, horizon=990):
            """
            Augment state by including time
            :param state:
            :param horizon:
            :return:
            """
            out = []
            for i in range(horizon):
                out.append(np.hstack((state, i)))
            return np.array(out)

        idx_trajectory = 0
        idx_state = 2
        state = test_data[idx_trajectory].states[idx_state]
        remaining_horizon = test_data[idx_trajectory].states.shape[0] - idx_state - 1
        groundtruth = test_data[idx_trajectory].states[idx_state:]
        pred_out = np.concatenate((state[None, :], model.predict(augment_state(state, horizon=remaining_horizon))))
        idx_plot = range(7)
        for i in idx_plot:
            plt.figure()
            h1 = plt.plot(pred_out[:, i], label='Prediction')
            h2 = plt.plot(groundtruth[:, i], c='r', label='Groundtruth')
            plt.legend()
            plt.show()


def test_sample_efficiency(cfg):
    # Collect test data
    log.info('Collecting test data')
    test_data = collect_data(nTrials=1, horizon=cfg.experiment_efficiency.traj_len_test)  # 5

    log.info('Begining loop')
    data_t = []
    data_no_t = []
    models_t = []
    models_no_t = []
    for i in range(cfg.experiment_efficiency.num_traj):
        log.info('Trajectory 1')
        new_data = collect_data(nTrials=1, horizon=cfg.experiment_efficiency.traj_len)
        data_t.extend(new_data)
        data_no_t.extend(new_data)

        dataset_t = create_dataset_traj(data_t)
        dataset_no_t = create_dataset_step(data_no_t)

        # Train t model
        n_in = dataset_t[0].shape[1]
        n_out = dataset_t[1].shape[1]
        hid_width = cfg.nn.trajectory_based.training.hid_width
        model_t = Net(structure=[n_in, hid_width, hid_width, n_out])
        model_no_t, logs = train_model(dataset_t, model,
                                       model_file, cfg.nn.trajectory_based.optimizer.lr,
                                       cfg.nn.trajectory_based.optimizer.epochs)

        # Train no t model
        n_in = dataset_no_t[0].shape[1]
        n_out = dataset_no_t[1].shape[1]
        hid_width = cfg.nn.one_step_det.training.hid_width
        model_no_t = Net(structure=[n_in, hid_width, hid_width, n_out])
        model_no_t, logs_no_t = train_model(dataset_no_t, model_no_t,
                                            model_file, cfg.nn.one_step_det.optimizer.lr,
                                            cfg.nn.one_step_det.optimizer.epochs)

        models_t.append(model_t)
        models_no_t.append(model_no_t)

    # # Plot optimization NN
    # if cfg.nn.training.plot_loss:
    #     plot_loss(logs.training_error, logs_no_t.training_error)
    #
    # if cfg.nn.training.plot_loss_epoch:
    #     plot_loss_epoch(logs.training_error_epoch,
    #         logs_no_t.training_error_epoch, cfg.nn.optimizer.epochs)

    mse_t, mse_no_t, predictions_t, predictions_no_t = test_models(test_data[0], models, models_no_t)

    plot_states(test_data[0].states, np.array(predictions_t), np.array(predictions_no_t),
                idx_plot=[0, 1, 2, 3, 4, 5, 6])

    plot_mse(mse_t, mse_no_t)


# This one can mostly be ignored; an experiment I did a while ago
# @hydra.main(config_path='conf/config.yaml')
def test_multiple_n_epochs(cfg):
    # Collect data
    log.info('Collecting data')
    train_data = collect_data(nTrials=cfg.experiment_one_model.num_traj,
                              horizon=cfg.experiment_one_model.traj_len)  # 50
    test_data = collect_data(nTrials=1, horizon=cfg.experiment_one_model.traj_len)  # 5

    # Create dataset
    log.info('Creating dataset')
    dataset = create_dataset_traj(train_data)
    dataset_no_t = create_dataset_step(train_data)  # train_data[0].states)

    # Set up the models
    n_in = dataset[0].shape[1]
    n_out = dataset[1].shape[1]
    hid_width = cfg.nn.trajectory_based.training.hid_width
    model = Net(structure=[n_in, hid_width, hid_width, n_out])

    n_in = dataset_no_t[0].shape[1]
    n_out = dataset_no_t[1].shape[1]
    hid_width = cfg.nn.one_step_det.training.hid_width
    model_no_t = Net(structure=[n_in, hid_width, hid_width, n_out])

    def loss(x, y):
        d = x - y
        norm = np.linalg.norm(d, axis=1)
        return np.sum(norm) / norm.size

    loss_t = []
    loss_no_t = []

    for n_epochs in range(cfg.nn.optimizer.epochs):
        # Train models
        model, logs = train_model(dataset, model, cfg.nn.optimizer.lr, 1)
        model_no_t, logs_no_t = train_model(dataset_no_t, model_no_t, cfg.nn.optimizer.lr, 1)

        log.info("Beginning testing of predictions")

        traj = test_data[0]
        states = traj.states
        actions = traj.actions
        initial = states[0]
        current = initial

        predictions_t = [states[0, :]]
        predictions_no_t = [states[0, :]]
        for i in range(1, states.shape[0]):
            pred_t = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            pred_no_t = model_no_t.predict(np.concatenate((current, actions[i - 1, :])))
            predictions_t.append(pred_t.squeeze())
            predictions_no_t.append(pred_no_t.squeeze())
            current = pred_no_t.squeeze()
        predictions_t = np.array(predictions_t)
        predictions_t = np.array(predictions_no_t)

        loss_t.append(loss(states, predictions_t))
        loss_no_t.append(loss(states, predictions_no_t))

    plt.figure()
    plt.title("MSE after x epochs of training")
    plt.plot(loss_t, color="blue", label="with t")
    plt.plot(loss_no_t, color="red", label="without t")
    plt.legend()
    plt.show()

    plt.figure()
    plt.title("MSE after x epochs of training, log scale")
    plt.semilogy(loss_t, color="blue", label="with t")
    plt.semilogy(loss_no_t, color="red", label="without t")
    plt.legend()
    plt.show()

    plot_states(states, np.array(predictions_t), np.array(predictions_no_t), idx_plot=[0, 1, 2, 3, 4, 5, 6])


###########################################
#               Helpers                   #
###########################################
def unpack_config_models(cfg):
    """
    Reads the config to decide which models to use
    """
    model_types = []
    if cfg.experiment.models.single.train_traj:
        model_types.append('t')
    if cfg.experiment.models.single.train_det:
        model_types.append('d')
    if cfg.experiment.models.single.train_prob:
        model_types.append('p')
    if cfg.experiment.models.single.train_prob_traj:
        model_types.append('tp')
    if cfg.experiment.models.ensemble.train_traj:
        model_types.append('te')
    if cfg.experiment.models.ensemble.train_det:
        model_types.append('de')
    if cfg.experiment.models.ensemble.train_prob:
        model_types.append('pe')
    if cfg.experiment.models.ensemble.train_prob_traj:
        model_types.append('tpe')

    return model_types


if __name__ == '__main__':
    sys.exit(contpred())
