# Compatibility Python 2/3
# from __future__ import division, print_function, absolute_import
# from builtins import range
# from past.builtins import basestring
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
# import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

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
import save
import gym
# from gym.monitoring import VideoRecorder
# import src.env removed because it seemed unused
import scipyplot as spp

import logging
logging.basicConfig(
    filename="test.log",
    level=logging.DEBUG,
    format="%(asctime)s:%(levelname)s:%(message)s"
    )
# logger = logging.getLogger()
# logger.setLevel(logging.DEBUG)
# # To log file
# fh = logging.FileHandler('example.log')
# fh.setLevel(logging.DEBUG)
# logger.addHandler(fh)

from policy import PID
from policy import randomPolicy


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
        self.n_layers = len(structure)-1
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


# class FaceLandmarksDataset(Dataset):
#     """Face Landmarks dataset."""
#
#     def __init__(self, csv_file, root_dir, transform=None):
#         """
#         Args:
#             csv_file (string): Path to the csv file with annotations.
#             root_dir (string): Directory with all the images.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.landmarks_frame = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.landmarks_frame)
#
#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir,
#                                 self.landmarks_frame.iloc[idx, 0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample

def train_network(dataset, model, parameters=DotMap()):
    import torch.optim as optim

    # This bit basically adds variables to the dotmap with default values
    p = DotMap()
    p.opt.n_epochs = parameters.get('n_epochs', 10)
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

    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader
    from torch.utils.data.sampler import Sampler

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


    logging.info('Training NN from dataset')

    # Puts it in PyTorch dataset form and then converts to DataLoader
    #
    # DataLoader is an iterable
    dataset = PytorchDataset(dataset=dataset) # Using PyTorch
    loader = DataLoader(dataset, batch_size=p.opt.batch_size, shuffle=True)  ##shuffle=True #False
        # pin_memory=True
        # drop_last=False

    startTime = timer()
    if logs.time is None:
        logs.time = [0]

    for epoch in range(p.opt.n_epochs):
        for i, data in enumerate(loader, 0): # Not sure why it uses enumerate instead of regular interation
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

            e = loss.data[0]
            logs.training_error.append(e)
            logging.info('Iter %010d - %f ' % (epoch, e))
            loss.backward()
            optimizer.step()  # Does the update
            logs.time.append(timer() - logs.time[-1])

    endTime = timer()
    logging.info('Optimization completed in %f[s]' % (endTime - startTime))

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
        return obs[0:7]

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
        action = policy.act(obs2q(state))

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


# Is the purpose of this to collect data for the model of the environment?
def collect_data(nTrials=20, horizon=1000):
    """
    Collect data
    :param nTrials:
    :param horizon:
    :return:
    """
    # env = gym.make('Reacher-v2')
    # This makes a gym model which seems to be an abstracted environment
    # I believe this model is like an arm that reaches for a point in 3D?
    # env_model = 'Reacher3D-v0'
    env_model = 'Reacher-v2'
    env = gym.make(env_model)
    logging.info('Initializing env: %s' % env_model)

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
        logging.info('Trial %d' % i)
        env.seed(i)
        P = np.array([4, 4, 1, 1, 1, 1, 2])
        I = np.zeros(7)
        D = [0.2, 0.2, 2, 0.4, 0.4, 0.1, 0.5]
        target = [0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5]
        # policy = PID(dX=7, dU=7, P=P, I=I, D=D, target=target)
        policy = randomPolicy(dX = 7, dU = 7);
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
    data_in = []
    data_out = []
    for i in range(states.shape[0]): #From one state p
        for j in range(i+1, states.shape[0]):
            data_in.append(np.hstack((states[i], j-i)))
            data_out.append(states[j])
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


def main():

    COLLECT_DATA = False
    CREATE_DATASET = False
    TRAIN_MODEL = False

    # Collect data
    if collect_data(): # Won't this always be true?
        logging.info('Collecting data')
        train_data = collect_data(nTrials=1)  # 50
        test_data = collect_data(nTrials=1)  # 5
    else:
        pass

    # Create dataset
    if CREATE_DATASET:
        logging.info('Creating dataset')
        dataset = create_dataset(train_data)
    else:
        pass

    # Train model
    model_file = 'model.pth.tar'
    n_in = dataset[0].shape[1]
    n_out = dataset[1].shape[1]
    model = Net(structure=[n_in, 2000, 2000, n_out])
    if TRAIN_MODEL:
        p = DotMap()
        p.opt.n_epochs = 1  # 1000
        p.learning_rate = 0.000001
        p.useGPU = True
        model, logs = train_network(dataset=dataset, model=model, parameters=p)
        logging.info('Saving model to file: %s' % model_file)
        torch.save(model.state_dict(), model_file)
        save.save(logs, 'logs.pkl')
        # TODO: save logs to file
    else:
        logging.info('Loading model to file: %s' % model_file)
        checkpoint = torch.load(model_file)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        logs = save.load('logs.pkl')
        # TODO: load logs from file

    # # Plot optimization NN
    # plt.figure()
    # plt.plot(np.array(logs.training_error))

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
    main()
