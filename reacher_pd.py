import sys
import warnings

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

import numpy as np
from dotmap import DotMap
# import matplotlib.pyplot as plt

# import R.data as rdata
# import progressbar removed because it seemed unused
from timeit import default_timer as timer
import matplotlib.pyplot as plt
# import scipyplot as spp

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import gym
from envs import *

# from gym.monitoring import VideoRecorder
# import src.env removed because it seemed unused

import hydra
import logging

log = logging.getLogger(__name__)

from policy import PID


def stateAction2forwardDyn(states, actions):
    data_in = np.concatenate((states[:-1, :], actions[:-1, :]), axis=1)
    data_out = states[1:, :]
    return [data_in, data_out]


class Net(nn.Module):
    """
    Neural Network

    In this case this is being used as a model of the environment right?
    """

    def __init__(self, structure=[20, 100, 100, 1], tf=F.relu):
        """
        :param structure: layer sizes
        :param tf: nonlinearity function
        """
        super(Net, self).__init__()

        # TODO: parameteric NN
        # self.fc = []
        self.n_layers = len(structure) - 1
        # for idx in range(self.n_layers):
        #     self.fc[idx] = nn.Linear(structure[idx], structure[idx+1])
        self.fc1 = nn.Linear(structure[0], structure[1])
        self.fc2 = nn.Linear(structure[1], structure[2])
        self.fc3 = nn.Linear(structure[2], structure[3])
        self.tf = tf
        self._onGPU = False

    def forward(self, x):
        x = self.tf(self.fc1(x))
        x = self.tf(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, x):
        """
        wrapper to/from numpy array
        :param x:
        :return:
        """
        if self._onGPU:
            pass
        else:
            return self.forward(Variable(torch.from_numpy(np.matrix(x)).float())).data.cpu().numpy()


def train_network(dataset, model, parameters=DotMap()):
    import torch.optim as optim
    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader

    # This bit basically adds variables to the dotmap with default values
    p = DotMap()
    p.opt.n_epochs = parameters.opt.get('n_epochs', 10)
    p.opt.optimizer = optim.Adam
    p.opt.batch_size = parameters.get('batch_size', 100)
    p.criterion = nn.MSELoss()
    p.learning_rate = parameters.get('learning_rate', 0.0001)
    p.useGPU = parameters.get('useGPU', False)
    p.verbosity = parameters.get('verbosity', 1)
    p.logs = parameters.get('logs', None)

    # Init logs
    if p.logs is None:
        logs = DotMap()
        logs.training_error = []
        logs.time = None
    else:
        logs = p.logs

    # Optimizer
    optimizer = p.opt.optimizer(model.parameters(), lr=p.learning_rate)

    # Lets cudnn autotuner find optimal algorithm for hardware
    cudnn.benchmark = True

    if p.useGPU:
        model.cuda()
        p.criterion.cuda()

    # Wrapper representing map-style PyTorch dataset
    class PytorchDataset(Dataset):
        def __init__(self, dataset):
            self.inputs = torch.from_numpy(dataset[0]).float()
            self.outputs = torch.from_numpy(dataset[1]).float()
            self.n_data = dataset[0].shape[0]
            self.n_inputs = dataset[0].shape[1]
            self.n_outputs = dataset[1].shape[1]

        def __getitem__(self, index):
            # print('\tcalling Dataset:__getitem__ @ idx=%d' % index)
            input = self.inputs[index]
            output = self.outputs[index]
            return input, output

        def __len__(self):
            # print('\tcalling Dataset:__len__')
            return self.n_data

    log.info('Training NN from dataset')

    # Puts it in PyTorch dataset form and then converts to DataLoader
    #
    # DataLoader is an iterable
    dataset = PytorchDataset(dataset=dataset)  # Using PyTorch
    loader = DataLoader(dataset, batch_size=p.opt.batch_size, shuffle=True)  ##shuffle=True #False
    # pin_memory=True
    # drop_last=False

    startTime = timer()
    if logs.time is None:
        logs.time = [0]

    print("Training for %d epochs" % p.opt.n_epochs)

    for epoch in range(p.opt.n_epochs):
        log.info("Epoch %d" % (epoch))
        for i, data in enumerate(loader, 0):
            if i % 100 == 0:
                print("    Batch %d" % i)
            # Load data
            # Variable is a wrapper for Tensors with autograd
            inputs, targets = data
            if p.useGPU:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
            else:
                inputs = Variable(inputs)
                targets = Variable(targets)

            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = p.criterion(outputs, targets)
            # print(loss)

            e = loss.item()
            logs.training_error.append(e)
            # log.info('Iter %010d - %f ' % (epoch, e))
            loss.backward()
            optimizer.step()  # Does the update
            logs.time.append(timer() - logs.time[-1])

    endTime = timer()
    log.info('Optimization completed in %f[s]' % (endTime - startTime))

    return model.cpu(), logs


def plot_pred(groundtruth, prediction, sorted=True):
    plt.figure()
    if sorted:
        gt = groundtruth.sort()
    else:
        gt = groundtruth
    plt.plot(gt)
    plt.plot(prediction)
    plt.show()


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
        # print(state)
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


def collect_data(nTrials=20, horizon=1000):
    """
    Collect data for environment model
    :param nTrials:
    :param horizon:
    :return: an array of DotMaps, where each DotMap contains info about a sequence of steps
    """
    # env = gym.make('Reacher-v2')
    # I believe this model is like an arm that reaches for a point in 3D?
    env_model = 'Reacher3d-v2'
    # env_model = 'Reacher-v2'
    env = gym.make(env_model)
    # print(type(env))
    log.info('Initializing env: %s' % env_model)

    # Logs is an array of dotmaps, each dotmap contains 2d np arrays with data
    # about <horizon> steps with actions, rewards and states
    logs = []

    # def init_env(env):
    #     qpos = np.copy(env.init_qpos)
    #     qvel = np.copy(env.init_qvel)
    #     qpos[0:7] += np.random.normal(loc=0.5, scale=1, size=[7])
    #     # qpos[-3:] += np.random.normal(loc=0, scale=1, size=[3])
    #     # qvel[-3:] = 0
    #     env.goal = qpos[-3:]
    #     env.set_state(qpos, qvel)
    #     env.T = 0
    #     return env

    for i in range(nTrials):
        log.info('Trial %d' % i)
        env.seed(i)
        # The following lines are for a 3d reacher environment
        # Replacing the following because they are dim 7, but should be 5
        # P = np.array([4, 4, 1, 1, 1, 1, 2])
        # I = np.zeros(7)
        # D = [0.2, 0.2, 2, 0.4, 0.4, 0.1, 0.5]
        # target = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5]
        P = np.array([4, 4, 1, 1, 1])
        I = np.zeros(5)
        D = [0.2, 0.2, 2, 0.4, 0.4]
        target = [0.5, 0.5, 0.5, 0.5, 0.5]
        # target = env.get_body_com("target")
        policy = PID(dX=5, dU=5, P=P, I=I, D=D, target=target)
        # print(type(env))

        # Logs will be a
        logs.append(run_controller(env, horizon=horizon, policy=policy))

        # # Visualize
        # plt.figure()
        # h = plt.plot(logs[i].states[:, 0:7])
        # plt.legend(h)
        # plt.show()
    return logs


# Learn model t only
# Creating a dataset for learning different T values
def create_dataset_t_only(states):
    """
    Creates a dataset with an entry for how many timesteps in the future
    corresponding entries in the labels are
    :param states: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for i in range(states.shape[0]):  # From one state p
        for j in range(i + 1, states.shape[0]):
            # This creates an entry for a given state concatenated with a number t of time steps
            data_in.append(np.hstack((states[i], j - i)))
            # This creates an entry for the state t timesteps in the future
            data_out.append(states[j])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset_no_t(states):
    """
    Creates a dataset for learning how one state progresses to the next
    """
    data_in = []
    data_out = []
    for i in range(states.shape[0] - 1):
        data_in.append(states[i])
        data_out.append(states[i + 1])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset(data):
    dataset_in = None
    dataset_out = None
    for sequence in data:
        inputs, outputs = create_dataset_t_only(sequence.states)
        if dataset_in is None:
            dataset_in = inputs
            dataset_out = outputs
        else:
            dataset_in = np.concatenate((dataset_in, inputs), axis=0)
            dataset_out = np.concatenate((dataset_out, outputs), axis=0)
    return [dataset_in, dataset_out]


@hydra.main(config_path='config.yaml')
def contpred(cfg):
    COLLECT_DATA = cfg.collect_data
    CREATE_DATASET = cfg.create_dataset
    TRAIN_MODEL = cfg.train_model
    TRAIN_MODEL_NO_T = cfg.train_model_onestep

    # Collect data
    if COLLECT_DATA:
        log.info('Collecting data')
        train_data = collect_data(nTrials=cfg.experiment.num_traj, horizon=cfg.experiment.traj_len)  # 50
        test_data = collect_data(nTrials=1, horizon=cfg.experiment.traj_len)  # 5
    else:
        pass

    # Create dataset
    if CREATE_DATASET:
        log.info('Creating dataset')
        dataset = create_dataset(train_data)
        dataset_no_t = create_dataset_no_t(train_data[0].states)
    else:
        pass

    # Train model
    model_file = 'model.pth.tar'
    n_in = dataset[0].shape[1]
    n_out = dataset[1].shape[1]
    hid_width = cfg.nn.training.hid_width
    model = Net(structure=[n_in, hid_width, hid_width, n_out])
    if TRAIN_MODEL:
        p = DotMap()
        p.opt.n_epochs = cfg.nn.optimizer.epochs  # 1000
        p.learning_rate =cfg.nn.optimizer.lr
        p.useGPU = False
        model, logs = train_network(dataset=dataset, model=model, parameters=p)
        log.info('Saving model to file: %s' % model_file)
        torch.save(model.state_dict(), model_file)
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
    model_no_t = Net(structure=[n_in, 2000, 2000, n_out])
    if TRAIN_MODEL_NO_T:
        p = DotMap()
        p.opt.n_epochs = cfg.nn.optimizer.epochs  # 1000
        p.learning_rate = cfg.nn.optimizer.lr
        p.useGPU = False
        model_no_t, logs_no_t = train_network(dataset=dataset_no_t, model=model_no_t, parameters=p)
        log.info('Saving model to file: %s' % model_file)
        torch.save(model_no_t.state_dict(), model_file)
    else:
        log.info('Loading model to file: %s' % model_file)
        checkpoint = torch.load(model_file)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_no_t.load_state_dict(checkpoint['state_dict'])
        else:
            model_no_t.load_state_dict(checkpoint)

    # # Plot optimization NN
    if cfg.nn.training.plot_loss:
        plt.figure()
        plt.plot(np.array(logs.training_error))
        plt.title("Training Error with t")
        plt.show()

        plt.figure()
        plt.plot(np.array(logs_no_t.training_error))
        plt.title("Training Error without t")
        plt.show()

    log.info("Beginning testing of predictions")
    mse_t = []
    mse_no_t = []
    states = test_data[0].states
    initial = states[0]
    current = initial
    for i in range(1, states.shape[0]):
        pred_t = model.predict(np.hstack((initial, i)))
        pred_no_t = model_no_t.predict(current)
        groundtruth = states[i]
        mse_t.append(np.square(groundtruth - pred_t).mean())
        mse_no_t.append(np.square(groundtruth - pred_no_t).mean())
        current = pred_no_t

    plt.figure()
    plt.title("MSE over time for model with and without t")
    plt.plot(mse_t, color='red', label='with t')
    plt.plot(mse_no_t, color='blue', label='without t')
    plt.legend()
    plt.show()

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


if __name__ == '__main__':
    sys.exit(contpred())
