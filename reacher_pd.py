import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import numpy as np
from dotmap import DotMap

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
# Don't use this, use the function below it
def create_dataset_t_only(data):
    """
    Creates a dataset with an entry for how many timesteps in the future
    corresponding entries in the labels are
    :param states: An array of dotmaps, where each dotmap has info about a trajectory
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

def create_dataset_t_pid(data, probabilistic=False):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future
    :param states: A 2d np array. Each row is a state
    """
    data_in, data_out = [], []
    threshold = 0.90
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
                # NOTE: Since integral controller is not yet implemented, I am removing it here

                # The randomely continuing is something I thought of to shrink
                # the datasets while still having a large variety
                if probabilistic and np.random.random() < threshold:
                    continue


                data_in.append(np.hstack((states[i], j - i, P/5, D, target)))
                data_out.append(states[j])
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

def run_controller(env, horizon, policy):
    """
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

def collect_data(nTrials=20, horizon=150, plot=True):  # Creates horizon^2/2 points
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a sequence of steps
    """
    env_model = 'Reacher3d-v2'
    env = gym.make(env_model)
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []

    # def init_env(env):
    #     qpos = np.copy(env.init_qpos)
    #     qvel = np.copy(env.init_qvel)
    #     qpos[0:7] += np.random.normal(loc=0.5, scale=1, size=[7])
    #     qpos[-3:] += np.random.normal(loc=0, scale=1, size=[3])
    #     qvel[-3:] = 0
    #     env.goal = qpos[-3:]
    #     env.set_state(qpos, qvel)
    #     env.T = 0
    #     return env

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
        dotmap.P = P
        dotmap.I = I
        dotmap.D = D
        logs.append(dotmap)
        # print("end pos is: ", logs[i].states[-1, -3:])
        # # Visualize
        # plt.figure()
        # # h = plt.plot(logs[i].states[:, 0:7])
        # h = plt.plot(logs[i].states[:, -3:])
        # plt.legend(h)
        # plt.show()

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

def train_model(dataset, model, learning_rate, n_epochs, model_file=None, prob=False):
    """
    Wrapper for training models

    Parameters:
    ----------
    dataset: The dataset to use for training, a tuple (data_in, data_out)
    model: a PyTorch model to train
    learning_rate: the learning rate
    n_epochs: the number of epochs to train for
    model_file: the file to save this into, or None if it shouldn't be saved

    Returns:
    --------
    model: the trained model, a PyTorch model
    logs: the logs, a dotmap containing training error per iteration, training error
        per epoch, and time
    """
    p = DotMap()
    p.opt.n_epochs = n_epochs# if n_epochs else cfg.nn.optimizer.epochs  # 1000
    p.learning_rate = learning_rate
    p.useGPU = False
    if prob:
        p.criterion = Prob_Loss()
    model, logs = train_network(dataset=dataset, model=model, parameters=p)
    if model_file:
        log.info('Saving model to file: %s' % model_file)
        torch.save(model.state_dict(), model_file)
    return model, logs

def collect_and_dataset(cfg):
    log.info('Collecting data')
    train_data = collect_data(nTrials=cfg.experiment.num_traj, horizon=cfg.experiment.traj_len, plot=False)  # 50
    test_data = collect_data(nTrials=1, horizon=cfg.experiment.traj_len_test, plot=False)  # 5

    log.info('Creating dataset')
    dataset = create_dataset_t_pid(train_data, probabilistic=True)
    dataset_no_t = create_dataset_no_t(train_data)  # train_data[0].states)
    return dataset, dataset_no_t, test_data

###########################################
#               Plotting                  #
###########################################

def plot_states(ground_truth, predictions, idx_plot=None, plot_avg=True, save_loc=None):
    """
    Plots the states given in predictions against the groundtruth. Predictions
    is a dictionary mapping model types to predictions
    """
    label_dict = {"traj_based":'Trajectory Based Prediction',
                  'det': "Deterministic Prediction",
                  'prob': 'Probabilistic Prediction'}
    color_dict = {'traj_based':'r',
                  'det': 'b',
                  'prob': 'g'}

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
            pred = predictions[key][:,i]
            # TODO: find a better way to do what the following line does
            chopped = np.maximum(np.minimum(pred, 3), -3) # to keep it from messing up graphs when it diverges
            plt.plot(chopped, c=color_dict[key], label=label_dict[key])

        plt.legend()

        if save_loc:
            plt.savefig(save_loc + "/state%d.pdf" % i)
        plt.show()

    if plot_avg:
        fig, ax = plt.subplots()
        gt = ground_truth[:, i]
        plt.title("Predictions Averaged")
        plt.xlabel("Timestep")
        plt.ylabel("Average State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        gt = np.zeros(ground_truth[:,0:1].shape)
        for i in idx_plot:
            gt = np.hstack((gt, ground_truth[:, i:i+1]))
        gt_avg = np.average(gt[:,1:], axis=1)
        plt.plot(gt_avg, c='k', label='Groundtruth')

        for key in predictions:
            pred = predictions[key]
            p = np.zeros(pred[:,0:1].shape)
            for i in idx_plot:
                p = np.hstack((p, pred[:, i:i+1]))
            p_avg = np.average(p[:,1:], axis=1)
            plt.plot(p_avg, c=color_dict[key], label=label_dict[key])
        # plt.ylim(-.5, 1.5)
        plt.legend()
        if save_loc:
            plt.savefig(save_loc + "/avg_states.pdf")
        plt.show()

def plot_loss(training_error_t, training_error_no_t):
    # TODO: autosaving

    plt.figure()
    plt.plot(np.array(training_error_t))
    plt.title("Training Error with t")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.show()

    plt.figure()
    plt.plot(np.array(training_error_no_t))
    plt.title("Training Error without t")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.show()

def plot_loss_epoch(training_error_t, training_error_no_t, epochs_t, epochs_no_t):
    # TODO: autosaving

    plt.figure()
    plt.bar(np.arange(epochs_t), np.array(training_error_t))
    plt.title("Training Error with t")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.show()

    plt.figure()
    plt.bar(np.arange(epochs_no_t), np.array(training_error_no_t))
    plt.title("Training Error without t")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.show()

def plot_mse(MSEs, save_loc=None):
    """
    Plots MSE graphs for the two sequences given
    """
    label_dict = {"traj_based":'Trajectory Based',
                  'det': "Deterministic",
                  'prob': 'Probabilistic'}
    color_dict = {'traj_based':'r',
                  'det': 'b',
                  'prob': 'g'}
    marker_dict = {'traj_based':'s',
                   'det': 'o',
                   'prob': 'D'}

    # Non-log version
    fig, ax = plt.subplots()
    plt.title("MSE for deterministic and trajectory based models")
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
    plt.show()

    # Log version
    fig, ax = plt.subplots()
    plt.title("MSE for deterministic and trajectory based models")
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
    plt.show()

def test_models(traj, models_t, models_no_t):
    """
    Evaluates the models on the given states with the given inputs, needs to be updated

    Paramaters:
    -----------
    traj: a dotmap containing states and actions taken in the test
    models_t: a list of t-param models
    models_no_t: a list of step-by-step models

    Returns:
    -------
    mse_t: a 2D array such that mse_t[j][i] is the MSE of the jth t param model
        at the ith time step
    mse_no_t: a 2D array such that mse_t[j][i] is the MSE of the jth iterative model
        at the ith time step
    predictions_t: a 3D array such that predictions_t[j][i,:] is the prediction for the
        jth t-param model at the ith time step
    predictions_no_t: a 3D array such that predictions_t[j][i,:] is the prediction for the
        jth iterative model at the ith time step
    """
    log.info("Beginning testing of predictions")
    mse_t = np.zeros((len(models_t), len(traj.states)))
    mse_no_t = np.zeros((len(models_no_t), len(traj.states)))

    states = traj.states
    actions = traj.actions
    initial = states[0]
    current = initial

    predictions_t = [[states[0,:]] for model in models_t]
    predictions_no_t = [[states[0,:]] for model in models_no_t]
    for i in range(1, states.shape[0]):
        groundtruth = states[i]
        for j in range(len(models_t)):
            model = models_t[j]
            pred = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            predictions_t[j].append(pred.squeeze())
            mse_t[j][i] = np.square(groundtruth-pred).mean()


        for j in range(len(models_no_t)):
            model = models_no_t[j]
            pred = model.predict(np.concatenate((predictions_no_t[j][i-1], actions[i-1,:])))
            predictions_no_t[j].append(pred.squeeze())
            mse_no_t[j][i] = np.square(groundtruth-pred).mean()


    return mse_t, mse_no_t, predictions_t, predictions_no_t

def test_models_single(traj, models):
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
                    outcomes['mse']['traj_based']
    """
    log.info("Beginning testing of predictions")

    MSEs = {key:[] for key in models}

    states = traj.states
    actions = traj.actions
    initial = states[0,:]

    predictions = {key:[states[0,:]] for key in models}
    currents = {key: states[0,:] for key in models}
    for i in range(1, states.shape[0]):
        groundtruth = states[i]
        for key in models:
            model = models[key]

            if key == "traj_based":
                prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            elif key == "det":
                prediction = model.predict(np.concatenate((currents[key], actions[i-1,:])))
            elif key == "prob":
                pass # todo
            # TODO: ensemble versions

            predictions[key].append(prediction.squeeze())
            MSEs[key].append(np.square(groundtruth - prediction).mean())
            currents[key] = prediction.squeeze()

    MSEs = {key:np.array(MSEs[key]) for key in MSEs}
    predictions = {key:np.array(predictions[key]) for key in MSEs}

    outcomes = {'mse': MSEs, 'predictions':predictions}
    return outcomes

###########################################
#                 Misc                    #
###########################################

@hydra.main(config_path='conf/config.yaml')
def contpred(cfg):
    COLLECT_DATA = cfg.collect_data

    model_types = []
    if cfg.train_traj_based:
        model_types.append('traj_based')
    if cfg.train_det:
        model_types.append('det')
    if cfg.train_prob:
        model_types.append('prob')

    # Collect data
    if COLLECT_DATA:
        traj_dataset, one_step_dataset, test_data = collect_and_dataset(cfg)
    else:
        pass

    # TODO: redo this part so that it is easier to adjust which models
    #       you do and don't use

    models = {}

    configs = {'traj_based': cfg.nn.trajectory_based,
               'det': cfg.nn.one_step_det,
               'prob': cfg.nn.one_step_prob}

    for model_type in model_types:
        # TODO: update this to handle probabilistic loss
        log.info("Training %s model" % model_type)
        model_file = 'model_%s.pth.tar' % model_type
        dataset = traj_dataset if model_type == 'traj_based' else one_step_dataset

        n_in = dataset[0].shape[1]
        n_out = dataset[1].shape[1]
        if model_type == 'prob':
            n_out *= 2
        hid_width = configs[model_type].training.hid_width
        hid_count = configs[model_type].training.hid_depth
        struct = [n_in] + [hid_width] * hid_count + [n_out]
        model = Net(structure=struct)

        model, logs = train_model(dataset,
                                  model,
                                  configs[model_type].optimizer.lr,
                                  configs[model_type].optimizer.epochs,
                                  model_file=model_file,
                                  prob=(model_type=='prob'))

        models[model_type] = model

    # TODO: loading old models

    # Train trajectory based model
    # model_file = 'model.pth.tar'
    # n_in = dataset[0].shape[1]
    # n_out = dataset[1].shape[1]
    # hid_width = cfg.nn.trajectory_based.training.hid_width
    # model = Net(structure=[n_in, hid_width, hid_width, n_out])
    # if TRAIN_MODEL:
    #     model, logs = train_model(dataset, model,
    #             cfg.nn.trajectory_based.optimizer.lr,
    #             cfg.nn.trajectory_based.optimizer.epochs,
    #             model_file=model_file)
    #     # save.save(logs, 'logs.pkl')
    #     # TODO: save logs to file
    # else:
    #     log.info('Loading model to file: %s' % model_file)
    #     checkpoint = torch.load(model_file)
    #     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    #         model.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         model.load_state_dict(checkpoint)
    #     # logs = save.load('logs.pkl')
    #     # TODO: load logs from file
    # models["traj_based"] = model
    #
    # # Train one step deterministic model
    # model_file = 'model_no_t.pth.tar'
    # n_in = dataset_no_t[0].shape[1]
    # n_out = dataset_no_t[1].shape[1]
    # model_no_t = Net(structure=[n_in, hid_width, hid_width, n_out])
    # if TRAIN_MODEL_NO_T:
    #     model_no_t, logs_no_t = train_model(dataset_no_t, model_no_t,
    #             cfg.nn.one_step_det.optimizer.lr,
    #             cfg.nn.one_step_det.optimizer.epochs,
    #             model_file=model_file)
    # else:
    #     log.info('Loading model to file: %s' % model_file)
    #     checkpoint = torch.load(model_file)
    #     if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
    #         model_no_t.load_state_dict(checkpoint['state_dict'])
    #     else:
    #         model_no_t.load_state_dict(checkpoint)
    # models["det"] = model_no_t

    graph_file = 'graphs'
    os.mkdir(graph_file)

    # # Plot optimization NN
    # TODO: redo this with new design
    # plot_loss(logs.training_error, logs_no_t.training_error)
    # plot_loss_epoch(logs.training_error_epoch,
            # logs_no_t.training_error_epoch,
            # cfg.nn.trajectory_based.optimizer.epochs,
            # cfg.nn.one_step_det.optimizer.epochs)

    # mse_t, mse_no_t, predictions_t, predictions_no_t = test_model_single(test_data[0], model, model_no_t)
    outcomes = test_models_single(test_data[0], models)

    plot_states(test_data[0].states, outcomes['predictions'], idx_plot=[0, 1, 2, 3, 4, 5, 6], save_loc=graph_file)

    plot_mse(outcomes['mse'], save_loc=graph_file)

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

    plot_states(test_data[0].states, np.array(predictions_t), np.array(predictions_no_t), idx_plot=[0, 1, 2, 3, 4, 5, 6])

    plot_mse(mse_t, mse_no_t)


# This one can mostly be ignored; an experiment I did a while ago
# @hydra.main(config_path='conf/config.yaml')
def test_multiple_n_epochs(cfg):
    # Collect data
    log.info('Collecting data')
    train_data = collect_data(nTrials=cfg.experiment_one_model.num_traj, horizon=cfg.experiment_one_model.traj_len)  # 50
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




# train_data = collect_data(nTrials=50, horizon=300)  # 50
# dataset = create_dataset_t_pid(train_data, probabilistic=True)
if __name__ == '__main__':
    # sys.exit(test_multiple_n_epochs())
    sys.exit(contpred())


###########################################
#               Old Code                  #
###########################################
# def stateAction2forwardDyn(states, actions):
#     data_in = np.concatenate((states[:-1, :], actions[:-1, :]), axis=1)
#     data_out = states[1:, :]
#     return [data_in, data_out]


# def temp_generate_trajectories():
#     lengths = [10, 50, 100, 150, 200, 250, 300, 500]
#     for hor in lengths:
#         print("Generating length {} trajectories".format(hor))
#         data = np.array(collect_data(nTrials = 20, horizon=hor))
#         out = []
#         for trial in data:
#             out.extend(trial.states)
#         file = "trajectories/traj{}.npy".format(hor)
#         np.save(file, out)
