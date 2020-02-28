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


###########################################
#            Model Training               #
###########################################

# Learn model t only
# Creating a dataset for learning different T values
def create_dataset_t_only(data):
    """
    Creates a dataset with an entry for how many timesteps in the future
    corresponding entries in the labels are
    :param data: An array of dotmaps, where each dotmap has info about a trajectory
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence.states
        for i in range(states.shape[0]):  # From one state p
            for j in range(i + 1, states.shape[0]):
                # This creates an entry for a given state concatenated with a number t of time steps
                data_in.append(np.hstack((states[i], j - i)))
                # data_in = np.vstack((data_in, np.hstack(())))
                # This creates an entry for the state t timesteps in the future
                data_out.append(states[j])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset_t_pid(data, threshold=0):
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
                data_in.append(np.hstack((states[i], j - i, P, D, target)))
                # data_in.append(np.hstack((states[i], j-i, target)))
                data_out.append(states[j])

    print("shuffling")
    data_in, data_out = shuffle(data_in, data_out)
    data_in = np.array(data_in)
    data_out = np.array(data_out)
    return data_in, data_out


def create_dataset_no_t(data):
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
            data_out.append(sequence.states[i + 1])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def collect_data(nTrials=20, horizon=150, plot=True):  # Creates horizon^2/2 points
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a trajectory
    """
    def run_controller(env, horizon, policy):
        """
        Runs a Reacher3d gym environment for horizon timesteps, making actions according to policy

        :param env: A gym object
        :param horizon: The number of states forward to look
        :param policy: A policy object (see other python file)
        """

        # WHat is going on here?
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

    env_model = 'Reacher3d-v2'
    env = gym.make(env_model)
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []

    for i in range(nTrials):
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

        dotmap = run_controller(env, horizon=horizon, policy=policy)
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
    train_data = collect_data(nTrials=cfg.experiment.num_traj, horizon=cfg.experiment.traj_len, plot=False)  # 50
    test_data = collect_data(nTrials=cfg.experiment.num_traj_test, horizon=cfg.experiment.traj_len_test,
                             plot=False)  # 5
    # test_data = train_data[:1]

    log.info('Creating dataset')
    dataset = create_dataset_t_pid(train_data, threshold=0.95)
    dataset_no_t = create_dataset_no_t(train_data)  # train_data[0].states)
    return dataset, dataset_no_t, train_data, test_data


###########################################
#           Plotting / Output             #
###########################################

label_dict = {'traj': 'Trajectory Based Deterministic',
              'det': 'One Step Deterministic',
              'prob': 'One Step Probabilistic',
              'traj_prob': 'Trajectory Based Probabilistic',
              'traj_ens': 'Trajectory Based Deterministic Ensemble',
              'det_ens': 'One Step Deterministic Ensemble',
              'prob_ens': 'One Step Probabilistic Ensemble',
              'traj_prob_ens': 'Trajectory Based Probabilistic Ensemble'}
color_dict = {'traj': 'r',
              'det': 'b',
              'prob': 'g',
              'traj_prob': 'y',
              'traj_ens': '#b53636',
              'det_ens': '#3660b5',
              'prob_ens': '#52b536',
              'traj_prob_ens': '#b5af36'}
marker_dict = {'traj': 's',
               'det': 'o',
               'prob': 'D',
               'traj_prob': 'p',
               'traj_ens': 's',
               'det_ens': 'o',
               'prob_ens': 'D',
               'traj_prob_ens': 'p', }


def plot_states(ground_truth, predictions, idx_plot=None, plot_avg=True, save_loc=None, show=True):
    """
    Plots the states given in predictions against the groundtruth. Predictions
    is a dictionary mapping model types to predictions
    """
    num = np.shape(ground_truth)[0]
    dx = np.shape(ground_truth)[1]
    if idx_plot is None:
        idx_plot = list(range(dx))

    for i in idx_plot:
        fig, ax = plt.subplots()
        gt = ground_truth[:, i]
        plt.title("Predictions on one dimension")
        plt.xlabel("Timestep")
        plt.ylabel("State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.plot(gt, c='k', label='Groundtruth')
        for key in predictions:
            # print(key)
            pred = predictions[key][:, i]
            # TODO: find a better way to do what the following line does
            chopped = np.maximum(np.minimum(pred, 3), -3)  # to keep it from messing up graphs when it diverges
            plt.plot(chopped, c=color_dict[key], label=label_dict[key], marker=marker_dict[key], markevery=50)

        plt.legend()

        if save_loc:
            plt.savefig(save_loc + "/state%d.pdf" % i)
        if show:
            plt.show()
        else:
            plt.close()

    if plot_avg:
        fig, ax = plt.subplots()
        gt = ground_truth[:, i]
        plt.title("Predictions Averaged")
        plt.xlabel("Timestep")
        plt.ylabel("Average State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        gt = np.zeros(ground_truth[:, 0:1].shape)
        for i in idx_plot:
            gt = np.hstack((gt, ground_truth[:, i:i + 1]))
        gt_avg = np.average(gt[:, 1:], axis=1)
        plt.plot(gt_avg, c='k', label='Groundtruth')

        for key in predictions:
            pred = predictions[key]
            p = np.zeros(pred[:, 0:1].shape)
            for i in idx_plot:
                p = np.hstack((p, pred[:, i:i + 1]))
            p_avg = np.average(p[:, 1:], axis=1)
            plt.plot(p_avg, c=color_dict[key], label=label_dict[key], marker=marker_dict[key], markevery=50)
        # plt.ylim(-.5, 1.5)
        plt.legend()
        if save_loc:
            plt.savefig(save_loc + "/avg_states.pdf")
        if show:
            plt.show()
        else:
            plt.close()


def plot_loss(logs, save_loc=None, show=True):
    """
    Plots the training loss of all models
    """
    for key in logs:
        plt.figure()
        log = logs[key]
        plt.plot(np.array(log.training_error[10:]), c=color_dict[key], label=label_dict[key])
        plt.title("Training Loss for %s" % label_dict[key])
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        if save_loc:
            plt.savefig("%s/loss_%s.pdf" % (save_loc, key))
        if show:
            plt.show()
        else:
            plt.close()


def plot_loss_epoch(logs, save_loc=None, show=True):
    """
    Plots the loss by epoch for each model in logs
    """
    for key in logs:
        plt.figure()
        log = logs[key]
        plt.bar(np.arange(len(log.training_error_epoch)), np.array(log.training_error_epoch))
        plt.title("Epoch Training Loss for %s" % label_dict[key])
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        if save_loc:
            plt.savefig("%s/loss_epoch_%s.pdf" % (save_loc, key))
        if show:
            plt.show()
        else:
            plt.close()


def plot_mse(MSEs, save_loc=None, show=True):
    """
    Plots MSE graphs for the sequences given given

    Parameters:
    ------------
    MSEs: a dictionary mapping model type key to an array of MSEs
    """
    # Non-log version
    fig, ax = plt.subplots()
    plt.title("MSE for a variety of models")
    plt.xlabel("Timesteps")
    plt.ylabel('Mean Square Error')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for key in MSEs:
        mse = MSEs[key]
        plt.plot(mse, color=color_dict[key], label=label_dict[key], marker=marker_dict[key], markevery=50)
    plt.legend()
    if save_loc:
        plt.savefig(save_loc + "/mse.pdf")
    if show:
        plt.show()
    else:
        plt.close()

    # Log version
    fig, ax = plt.subplots()
    plt.title("Log MSE for a variety of models")
    plt.xlabel("Timesteps")
    plt.ylabel('Mean Square Error')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for key in MSEs:
        mse = MSEs[key]
        plt.semilogy(mse, color=color_dict[key], label=label_dict[key], marker=marker_dict[key], markevery=50)
    plt.legend()
    if save_loc:
        plt.savefig(save_loc + "/mse_log.pdf")
    if show:
        plt.show()
    else:
        plt.close()


def test_models(traj, models):
    """
    Tests each of the models in the dictionary "models" on the same trajectory

    Parameters:
    ------------
    traj: the trajectory to test on
    models: a dictionary of models to test

    Returns:
    outcomes: a dictionary of MSEs and predictions. As an example of how to
              get info from this distionary, to get the MSE data from a trajectory
              -based model you'd do
                    outcomes['mse']['traj']
    """
    log.info("Beginning testing of predictions")

    MSEs = {key: [] for key in models}

    states = traj.states
    actions = traj.actions
    initial = states[0, :]

    predictions = {key: [states[0, :]] for key in models}
    currents = {key: states[0, :] for key in models}
    for i in range(1, states.shape[0]):
        groundtruth = states[i]
        for key in models:
            model = models[key]

            if key == "traj":
                prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            elif key == "det":
                prediction = model.predict(np.concatenate((currents[key], actions[i - 1, :])))
            elif key == "prob":
                prediction = model.predict(np.concatenate((currents[key], actions[i - 1, :])))
                prediction = prediction[:, :prediction.shape[1] // 2]
            elif key == 'traj_prob':
                prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
                prediction = prediction[:, :prediction.shape[1] // 2]
            elif key == "traj_ens":
                prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            elif key == "det_ens":
                prediction = model.predict(np.concatenate((currents[key], actions[i - 1, :])))
            elif key == "prob_ens":
                prediction = model.predict(np.concatenate((currents[key], actions[i - 1, :])))
                prediction = prediction[:, :prediction.shape[1] // 2]
            elif key == 'traj_prob_ens':
                prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
                prediction = prediction[:, :prediction.shape[1] // 2]

            predictions[key].append(prediction.squeeze())
            MSEs[key].append(np.square(groundtruth - prediction).mean())
            currents[key] = prediction.squeeze()
            # print(currents[key].shape)

    MSEs = {key: np.array(MSEs[key]) for key in MSEs}
    predictions = {key: np.array(predictions[key]) for key in MSEs}

    outcomes = {'mse': MSEs, 'predictions': predictions}
    return outcomes


def test_traj_ensemble(ensemble, test_data):
    """
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


def make_evaluator(train_data, test_data, type):
    def evaluator(model):
        dic = {type: model}
        train_s = 0
        num_train = len(train_data)
        len_train = len(train_data[0].states)
        denom_train = num_train * len_train
        for traj in train_data:
            outcomes = test_models(traj, dic)
            train_s += np.sum(outcomes['mse'][type]) / denom_train
        test_s = 0
        num_test = len(test_data)
        len_test = len(test_data[0].states)
        denom_test = num_test * len_test
        for traj in test_data:
            outcomes = test_models(traj, dic)
            test_s += np.sum(outcomes['mse'][type]) / denom_test
        return (train_s, test_s)

    return evaluator


###########################################
#             Main Functions              #
###########################################

@hydra.main(config_path='conf/config.yaml')
def contpred(cfg):
    COLLECT_DATA = cfg.collect_data

    model_types = unpack_config_models(cfg)

    # configs = {'traj': cfg.nn.trajectory_based,
    #            'det': cfg.nn.one_step_det,
    #            'prob': cfg.nn.one_step_prob,
    #            'traj_prob': cfg.nn.trajectory_based_prob,
    #            'traj_ens': cfg.nn.trajectory_based,
    #            'det_ens': cfg.nn.one_step_det,
    #            'prob_ens': cfg.nn.one_step_prob,
    #            'traj_prob_ens': cfg.nn.trajectory_based_prob}

    log_hyperparams(cfg)  # configs, model_types)

    # Collect data
    if cfg.collect_data:
        log.info(f"Collecting new trials")
        traj_dataset, one_step_dataset, train_data, test_data = collect_and_dataset(cfg)

        if cfg.save_data:
            log.info("Saving new default data")
            torch.save((train_data, test_data),
                       hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)
            log.info(f"Saved trajectories to {'/trajectories/reacher/' + 'raw' + cfg.data_dir}")
    else:
        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)
        traj_dataset = create_dataset_t_pid(train_data, threshold=1.0)
        one_step_dataset = create_dataset_no_t(train_data)  # train_data[0].states)

    prob = cfg.model.prob
    traj = cfg.model.traj
    ens = cfg.model.ensemble

    # for model_type in model_types:
    log.info(f"Training model P:{[prob]}, T:{traj}, E:{ens}")
    # model_file = 'model_%s.pth.tar' % model_type

    dataset = traj_dataset if traj else one_step_dataset

    # TODO INTEGRATE
    if cfg.train_models:
        model = Model(cfg.model)
        model.train(dataset)
        loss_log = model.loss_log

        # models[model_type] = model
        # loss_logs[model_type] = loss_log

        # graph_file = 'graphs'
        # os.mkdir(graph_file)
        # print(loss_log)

        plot_loss(loss_log, save_loc='loss', show=False)
        plot_loss_epoch(loss_log, save_loc='loss_epochs', show=False)
        if cfg.save_models:
            log.info("Saving new default models")
            torch.save((model, loss_log),
                       hydra.utils.get_original_cwd() + '/models/reacher/' + cfg.model.str + cfg.model_dir)

    else:
        model_1s, train_log1 = torch.load(hydra.utils.get_original_cwd() + '/models/lorenz/' + 'step' + cfg.model_dir)
        model_ct, train_log2 = torch.load(hydra.utils.get_original_cwd() + '/models/lorenz/' + 'traj' + cfg.model_dir)

    # mse_t, mse_no_t, predictions_t, predictions_no_t = test_model_single(test_data[0], model, model_no_t)
    for i in range(len(test_data)):
        test = test_data[i]
        file = "%s/test%d" % (graph_file, i + 1)
        os.mkdir(file)

        outcomes = test_models(test, models)
        plot_states(test.states, outcomes['predictions'], idx_plot=[0, 1, 2, 3, 4, 5, 6], save_loc=file, show=False)
        plot_mse(outcomes['mse'], save_loc=file, show=False)

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

        dataset_t = create_dataset_t_pid(data_t)
        dataset_no_t = create_dataset_no_t(data_no_t)

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
    dataset = create_dataset_t_pid(train_data)
    dataset_no_t = create_dataset_no_t(train_data)  # train_data[0].states)

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
        model_types.append('traj')
    if cfg.experiment.models.single.train_det:
        model_types.append('det')
    if cfg.experiment.models.single.train_prob:
        model_types.append('prob')
    if cfg.experiment.models.single.train_prob_traj:
        model_types.append('traj_prob')
    if cfg.experiment.models.ensemble.train_traj:
        model_types.append('traj_ens')
    if cfg.experiment.models.ensemble.train_det:
        model_types.append('det_ens')
    if cfg.experiment.models.ensemble.train_prob:
        model_types.append('prob_ens')
    if cfg.experiment.models.ensemble.train_prob_traj:
        model_types.append('traj_prob_ens')

    return model_types


if __name__ == '__main__':
    sys.exit(contpred())
