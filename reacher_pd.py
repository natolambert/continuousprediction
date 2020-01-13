import sys
import warnings

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import numpy as np
from dotmap import DotMap

from timeit import default_timer as timer
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import gym
from envs import *

import hydra
import logging

log = logging.getLogger(__name__)

from policy import PID
from mbrl_resources import *

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

def collect_data(nTrials=20, horizon=150): # Creates horizon^2/2 points
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
        P = np.random.rand(5)*5
        I = np.zeros(5)
        # D = np.array([0.2, 0.2, 2, 0.4, 0.4])
        D = np.random.rand(5)

        # Samples target uniformely from [-1, 1]
        target = np.random.rand(5)*2-1

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
    return logs

def train_model(dataset, model, learning_rate, n_epochs, model_file=None):
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
    model, logs = train_network(dataset=dataset, model=model, parameters=p)
    if model_file:
        log.info('Saving model to file: %s' % model_file)
        torch.save(model.state_dict(), model_file)
    return model, logs



###########################################
#               Plotting                  #
###########################################
def plot_pred(groundtruth, prediction, sorted=True):
    plt.figure()
    if sorted:
        gt = groundtruth.sort()
    else:
        gt = groundtruth
    plt.plot(gt)
    plt.plot(prediction)
    plt.show()

def plot_states(ground_truth, prediction_param, prediction_step, idx_plot=None, save=False):
    num = np.shape(ground_truth)[0]
    dx = np.shape(ground_truth)[1]
    if idx_plot is None:
        idx_plot = list(range(dx))

    for i in idx_plot:
        gt = ground_truth[:, i]
        p1 = prediction_param[:, i]
        p2 = prediction_step[:, i]
        p2_chopped = np.maximum(np.minimum(p2, 10), -10) # to keep it from diverging and messing up graphs
        plt.figure()
        plt.plot(p1, c='k', label='Prediction T-Param')
        plt.plot(p2_chopped, c='b', label='Prediction 1 Steps')
        plt.plot(gt, c='r', label='Groundtruth')
        plt.legend()
        plt.show()

def plot_average_states(ground_truth, prediction_param, prediction_step, idx_plot=None):
    num = np.shape(ground_truth)[0]
    dx = np.shape(ground_truth)[1]
    if idx_plot is None:
        idx_plot = list(range(dx))

    print(ground_truth.shape)
    print(ground_truth[:,0:].shape)

    gt = np.zeros(ground_truth[:,0:1].shape)
    p1 = np.zeros(ground_truth[:,0:1].shape)
    p2 = np.zeros(ground_truth[:,0:1].shape)
    for i in idx_plot:
        gt = np.hstack((gt, ground_truth[:, i:i+1]))
        p1 = np.hstack((p1, prediction_param[:, i:i+1]))
        p2 = np.hstack((p2, prediction_step[:, i:i+1]))
        # p2_chopped = np.maximum(np.minimum(p2, 10), -10) # to keep it from diverging and messing up graphs
    print(gt.shape)
    gt_avg = np.average(gt[:,1:], axis=1)
    p1_avg = np.average(p1[:,1:], axis=1)
    p2_avg = np.average(p2[:,1:], axis=1)
    p2_chopped = np.maximum(np.minimum(p2_avg, 10), -10) # to keep it from diverging and messing up graphs

    plt.figure()
    plt.plot(p1_avg, c='k', label='Prediction T-Param')
    plt.plot(p2_chopped, c='b', label='Prediction 1 Steps')
    plt.plot(gt_avg, c='r', label='Groundtruth')
    plt.legend()
    plt.title('Predictions, averaged')
    plt.show()

def plot_loss(training_error_t, training_error_no_t):
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

def plot_loss_epoch(training_error_t, training_error_no_t, epochs):
    plt.figure()
    plt.bar(np.arange(epochs), np.array(training_error_t))
    plt.title("Training Error with t")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.show()

    plt.figure()
    plt.bar(np.arange(epochs), np.array(training_error_no_t))
    plt.title("Training Error without t")
    plt.xlabel("epoch")
    plt.ylabel("total loss")
    plt.show()

def plot_mse(mse_t, mse_no_t):
    """
    Plots MSE graphs for the two sequences given
    """
    plt.figure()
    plt.title("MSE over time for model with and without t")
    plt.semilogy(mse_t, color='red', label='with t')
    plt.semilogy(mse_no_t, color='blue', label='without t')
    plt.legend()
    plt.show()

def test_models(traj, models_t, models_no_t):
    """
    Evaluates the models on the given states with the given inputs

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

def test_model_single(traj, model, model_no_t):
    """
    Tests one model each of t-param and iterative

    Parameters/returns:
    -------------------
    see the above method it's about the same
    """
    log.info("Beginning testing of predictions")
    mse_t = []
    mse_no_t = []

    # traj = test_data[0]
    states = traj.states
    actions = traj.actions
    initial = states[0]
    current = initial

    predictions_t = [states[0,:]]
    predictions_no_t = [states[0,:]]
    for i in range(1, states.shape[0]):
        pred_t = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
        pred_no_t = model_no_t.predict(np.concatenate((current, actions[i-1,:])))
        predictions_t.append(pred_t.squeeze())
        predictions_no_t.append(pred_no_t.squeeze())
        groundtruth = states[i]
        mse_t.append(np.square(groundtruth - pred_t).mean())
        mse_no_t.append(np.square(groundtruth - pred_no_t).mean())
        current = pred_no_t.squeeze()

    return mse_t, mse_no_t, predictions_t, predictions_no_t


###########################################
#                 Misc                    #
###########################################

@hydra.main(config_path='conf/config.yaml')
def contpred(cfg):
    COLLECT_DATA = cfg.collect_data
    CREATE_DATASET = cfg.create_dataset
    TRAIN_MODEL = cfg.train_model
    TRAIN_MODEL_NO_T = cfg.train_model_onestep

    # Collect data
    if COLLECT_DATA:
        log.info('Collecting data')
        train_data = collect_data(nTrials=cfg.experiment_one_model.num_traj, horizon=cfg.experiment_one_model.traj_len)  # 50
        test_data = collect_data(nTrials=1, horizon=cfg.experiment_one_model.traj_len_test)  # 5
    else:
        pass

    # Create dataset
    if CREATE_DATASET:
        log.info('Creating dataset')
        dataset = create_dataset_t_pid(train_data, probabilistic=True)
        dataset_no_t = create_dataset_no_t(train_data)  # train_data[0].states)
    else:
        pass

    print(dataset[0].shape)

    # Train model
    model_file = 'model.pth.tar'
    n_in = dataset[0].shape[1]
    n_out = dataset[1].shape[1]
    hid_width = cfg.nn.training.hid_width
    model = Net(structure=[n_in, hid_width, hid_width, n_out])
    if TRAIN_MODEL:
        model, logs = train_model(dataset, model, cfg.nn.optimizer.lr, cfg.nn.optimizer.epochs, model_file=model_file)
        # save.save(logs, 'logs.pkl')
        # TODO: save logs to file
    else:
        log.info('Loading model to file: %s' % model_file)
        checkpoint = torch.load(model_file)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        # logs = save.load('logs.pkl')
        # TODO: load logs from file

    # Train no t model
    model_file = 'model_no_t.pth.tar'
    n_in = dataset_no_t[0].shape[1]
    n_out = dataset_no_t[1].shape[1]
    model_no_t = Net(structure=[n_in, hid_width, hid_width, n_out])
    if TRAIN_MODEL_NO_T:
        model_no_t, logs_no_t = train_model(dataset_no_t, model_no_t, cfg.nn.optimizer.lr, cfg.nn.optimizer.epochs, model_file=model_file)
    else:
        log.info('Loading model to file: %s' % model_file)
        checkpoint = torch.load(model_file)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_no_t.load_state_dict(checkpoint['state_dict'])
        else:
            model_no_t.load_state_dict(checkpoint)

    # # Plot optimization NN
    if cfg.nn.training.plot_loss:
        plot_loss(logs.training_error, logs_no_t.training_error)

    if cfg.nn.training.plot_loss_epoch:
        plot_loss_epoch(logs.training_error_epoch,
            logs_no_t.training_error_epoch, cfg.nn.optimizer.epochs)

    mse_t, mse_no_t, predictions_t, predictions_no_t = test_model_single(test_data[0], model, model_no_t)
    # mse_t = mse_t[0]
    # mse_no_t = mse_no_t[0]
    # predictions_t = predictions_t[0]
    # predictions_no_t = predictions_no_t[0]

    # plot_states(test_data[0].states, np.array(predictions_t), np.array(predictions_no_t), idx_plot=[0, 1, 2, 3, 4, 5, 6])
    plot_average_states(test_data[0].states, np.array(predictions_t), np.array(predictions_no_t), idx_plot=[0, 1, 2, 3, 4, 5, 6])
    plot_states(test_data[0].states, np.array(predictions_t), np.array(predictions_no_t), idx_plot=[0, 1, 2, 3, 4, 5, 6])

    plot_mse(mse_t, mse_no_t)

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
        hid_width = cfg.nn.training.hid_width
        model_t = Net(structure=[n_in, hid_width, hid_width, n_out])
        model_no_t, logs = train_model(dataset_t, model,
                        model_file, cfg.nn.optimizer.lr, cfg.nn.optimizer.epochs)

        # Train no t model
        n_in = dataset_no_t[0].shape[1]
        n_out = dataset_no_t[1].shape[1]
        model_no_t = Net(structure=[n_in, hid_width, hid_width, n_out])
        model_no_t, logs_no_t = train_model(dataset_no_t, model_no_t,
                        model_file, cfg.nn.optimizer.lr, cfg.nn.optimizer.epochs)

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
    hid_width = cfg.nn.training.hid_width
    model = Net(structure=[n_in, hid_width, hid_width, n_out])
    n_in = dataset_no_t[0].shape[1]
    n_out = dataset_no_t[1].shape[1]
    model_no_t = Net(structure=[n_in, hid_width, hid_width, n_out])

    def loss(x, y):
        d = x-y
        norm = np.linalg.norm(d, axis=1)
        return np.sum(norm)/norm.size
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

        predictions_t = [states[0,:]]
        predictions_no_t = [states[0,:]]
        for i in range(1, states.shape[0]):
            pred_t = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            pred_no_t = model_no_t.predict(np.concatenate((current, actions[i-1,:])))
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
