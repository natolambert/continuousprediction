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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import hydra
import logging

import multiprocessing as mp

log = logging.getLogger(__name__)

def create_dataset_traj(data, control_params=True, train_target=True, threshold=0.0, delta=False, t_range=0, is_lstm=False, lstm_batch=0):
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
        if id > 99:
            continue
        P = sequence.P
        D = sequence.D
        target = sequence.target
        n = states.shape[0]

        if not is_lstm:
            for i in range(n):  # From one state p
                for j in range(i + 1, n):
                    # This creates an entry for a given state concatenated
                    # with a number t of time steps as well as the PID parameters

                    # The randomely continuing is something I thought of to shrink
                    # the datasets while still having a large variety

                    if np.random.random() < threshold:
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
        else:
            for i in range(n-lstm_batch):
                if np.random.random() < threshold:
                    continue
                for j in range(i, i+lstm_batch):
                    dat = [states[j], lstm_batch-j]
                    if control_params:
                        dat.extend([P, D])
                    if train_target:
                        dat.append(target)
                    data_in.append(np.hstack(dat))
                    # data_in.append(np.hstack((states[i], j-i, target)))
                    if delta:
                        data_out.append(states[i+lstm_batch] - states[j])
                    else:
                        data_out.append(states[i+lstm_batch])
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    return data_in, data_out


def create_dataset_step(data, delta=True, t_range=0, is_lstm = False, lstm_batch = 0):
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
        if is_lstm:
            remainder = len(data_out)%lstm_batch
            if remainder:
                data_out = data_out[:len(data_out)-remainder]
                data_in = data_in[:len(data_in)-remainder]
    data_in = np.array(data_in, dtype=np.float32)
    data_out = np.array(data_out, dtype=np.float32)

    return data_in, data_out


def obs2q(obs):
    if len(obs) < 5:
        return obs
    else:
        return obs[0:5]


class Net(nn.Module):
    """
    Deterministic Neural Network
    """

    def __init__(self, structure=[20, 100, 100, 1], tf=F.relu):
        """
        :param structure: layer sizes
        :param tf: nonlinearity function
        """
        super(Net, self).__init__()

        # TODO: parameteric NN
        fc = []
        self.n_layers = len(structure) - 1
        for i in range(self.n_layers):
            fc.append(nn.Linear(structure[i], structure[i + 1]))
        self.linears = nn.ModuleList(fc)
        self.tf = tf
        self._onGPU = False

    def testPreprocess(self, input):
        if (input.shape[1] == 26):
            inputStates = input[:, :21]
            inputActions = input[:, 21:]
            normStates = self.stateScaler.transform(inputStates)
            normActions = self.actionScaler.transform(inputActions)
            return np.hstack((normStates, normActions))
        elif (input.shape[1] == 37):
            inputStates = input[:, :21]
            inputIndex = input[:, 21]
            inputP = input[:, 22:27]
            inputD = input[:, 27:32]
            inputTargets = input[:, 32:]
            normStates = self.stateScaler.transform(inputStates)
            normIndex = self.indexScaler.transform(inputIndex)
            normP = self.PScaler.transform(inputP)
            normD = self.DScaler.transform(inputD)
            normTargets = self.targetScaler.transform(inputTargets)
            return np.hstack((normStates, normIndex, normP, normD, normTargets))
        else:
            print("Incorrect dataset length, no normalization performed")
            return input

    def testPostprocess(self, output):
        return self.outputScaler.inverse_transform(output)

    def preprocess(self, dataset, cfg):

        # Select scaling, minmax vs standard (fits to a gaussian with unit variance and 0 mean)
        # TODO: Selection should be in config
        # StandardScaler, MinMaxScaler
        # TODO: Instead of hardcoding, include in config, trajectory vs. one-step, length of different inputs, etc.
        # 26 -> one-step, 37 -> trajectory
        if (input.shape[1] == 26):
            self.stateScaler = hydra.utils.instantiate(cfg.model.preprocess.state)
            self.actionScaler = hydra.utils.instantiate(cfg.model.preprocess.action)
            self.outputScaler = hydra.utils.instantiate(cfg.model.preprocess.output)

            inputStates = input[:, :21]
            inputActions = input[:, 21:]

            stateScaler.fit(inputStates)
            actionScaler.fit(inputActions)
            outputScaler.fit(output)

            normStates = self.stateScaler.transform(inputStates)
            normActions = self.actionScaler.transform(inputActions)
            normOutput = self.outputScaler.transform(output)

            return np.hstack((normStates, normActions)), normOutput
        elif (input.shape[1] == 37):
            self.stateScaler = hydra.utils.instantiate(cfg.model.preprocess.state)
            self.indexScaler = hydra.utils.instantiate(cfg.model.preprocess.index)
            self.PScaler = hydra.utils.instantiate(cfg.model.preprocess.P)
            self.DScaler = hydra.utils.instantiate(cfg.model.preprocess.D)
            self.targetScaler = hydra.utils.instantiate(cfg.model.preprocess.target)
            self.outputScaler = hydra.utils.instantiate(cfg.model.preprocess.output)

            inputStates = input[:, :21]
            inputIndex = input[:, 21]
            inputP = input[:, 22:27]
            inputD = input[:, 27:32]
            inputTargets = input[:, 32:]

            stateScaler.fit(inputStates)
            indexScaler.fit(inputIndex)
            PScaler.fit(inputP)
            DScaler.fit(inputD)
            targetScaler.fit(inputTargets)
            outputScaler.fit(output)

            normStates = self.stateScaler.transform(inputStates)
            normIndex = self.indexScaler.transform(inputIndex)
            normP = self.PScaler.transform(inputP)
            normD = self.DScaler.transform(inputD)
            normTargets = self.targetScaler.transform(inputTargets)
            normOutput = self.outputScaler.transform(output)

            return np.hstack((normStates, normIndex, normP, normD, normTargets)), normOutput
        else:
            print("Incorrect dataset length, no normalization performed")
            return input, output

    def forward(self, x):
        for i in range(self.n_layers - 1):
            x = self.tf(self.linears[i](x))
        x = self.linears[self.n_layers - 1](x)
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
            scaledInput = self.testPreprocess(x)
            output = self.forward(Variable(torch.from_numpy(np.matrix(scaledInput)).float())).data.cpu().numpy()
            return self.testPostprocess(output)


class ProbLoss(nn.Module):
    """
    Class for probabilistic loss function
    """

    def __init__(self, size):
        super(ProbLoss, self).__init__()
        self.size = size
        self.max_logvar = torch.nn.Parameter(
            torch.tensor(1 * np.ones([1, size]), dtype=torch.float, requires_grad=True))
        self.min_logvar = torch.nn.Parameter(
            torch.tensor(-1 * np.ones([1, size]), dtype=torch.float, requires_grad=True))

    def softplus_raw(self, input):
        # Performs the elementwise softplus on the input
        # softplus(x) = 1/B * log(1+exp(B*x))
        B = torch.tensor(1, dtype=torch.float)
        return (torch.log(1 + torch.exp(input.mul_(B)))).div_(B)

        # TODO: This function has been observed outputting negative values. needs fix

    def forward(self, inputs, targets):
        # size = targets.size()[1]
        mean = inputs[:, :self.size]
        logvar = inputs[:, self.size:]

        # Caps max and min log to avoid NaNs
        logvar = self.max_logvar - self.softplus_raw(self.max_logvar - logvar)
        logvar = self.min_logvar + self.softplus_raw(logvar - self.min_logvar)

        var = torch.exp(logvar)

        diff = mean - targets
        mid = diff / var
        lg = torch.sum(torch.log(var))
        out = torch.trace(torch.mm(diff, mid.t())) + lg
        return out


class Ensemble:
    """
    A neural network ensemble
    """

    def __init__(self, structure=[20, 100, 100, 1], n=10):
        self.models = [Net(structure=structure) for _ in range(n)]
        self.n = n
        self.structure = structure

    def predict(self, x):
        predictions = np.squeeze(np.array([model.predict(x) for model in self.models]))
        return np.average(predictions, axis=0)

    def train(self, cfg, dataset, parameters=DotMap(), parallel=False):
        """
        Trains this ensemble on dataset
        """
        n = self.n

        # Partitioning data
        dataset_in = dataset[0]
        dataset_out = dataset[1]
        partition_size = dataset_in.shape[0] // self.n
        partitions_in = [dataset_in[i * partition_size:(i + 1) * partition_size, :] for i in range(n)]
        partitions_out = [dataset_out[i * partition_size:(i + 1) * partition_size, :] for i in range(n)]
        datasets = []
        for i in range(n):
            ds_in = []
            ds_out = []
            for j in range(n):
                if i == j:
                    continue
                ds_in.extend(partitions_in[j])
                ds_out.extend(partitions_out[j])
            ds_in = np.array(ds_in)
            ds_out = np.array(ds_out)
            datasets.append((ds_in, ds_out))
        # else:
        #     partition_size = min(dataset_in.shape[0]//self.n, 1000000)
        #     datasets = [(dataset_in[i*partition_size:(i+1)*partition_size,:],
        #         dataset_out[i*partition_size:(i+1)*partition_size,:]) for i in range(n)]

        print(datasets[0][0].shape)

        # Training
        if parallel:
            n = mp.cpu_count()
            pool = mp.Pool(n)
            wrapper = lambda i: train_network(datasets[i], self.models[i], cfg, parameters=parameters)
            ugh = list(range(n))
            pool.map(wrapper, ugh)
        else:
            for i in range(n):
                train_network(datasets[i], self.models[i], cfg, parameters=parameters)

        return self


class Model(object):
    """
    A wrapper class for general models, including single nets and ensembles
    """

    def __init__(self, cfg):
        self.optim_params = cfg.optimizer
        self.training_params = cfg.training
        self.prob = cfg.prob
        self.ens = cfg.ensemble
        self.traj = cfg.traj
        self.str = cfg.str

    def train(self, cfg, dataset):
        data, labels = dataset
        n_in = data.shape[1]
        n_out = labels.shape[1]
        if self.prob:
            n_out *= 2

        hid_width = self.training_params.hid_width
        hid_count = self.training_params.hid_depth
        struct = [n_in] + [hid_width] * hid_count + [n_out]

        params = DotMap()
        params.opt.n_epochs = self.optim_params.epochs
        params.opt.batch_size = self.optim_params.batch
        params.learning_rate = self.optim_params.lr
        if self.prob:
            params.criterion = ProbLoss(labels.shape[1])

        if self.ens:
            self.model = Ensemble(structure=struct, n=self.training_params.E)
            self.model.train(cfg, dataset, params)
            # TODO: thoughtfully log ensemble training
            self.loss_log = None
        else:
            self.model = Net(structure=struct)
            self.model, self.loss_log = train_network(dataset, self.model, cfg, params)

    def predict(self, x):
        if self.traj:
            return self.model.predict(x)
        else:
            raise NotImplementedError("Need to formulate as change in state")


def load_model(file):
    model, log = torch.load(file)
    return model


def train_network(dataset, model, cfg, parameters=DotMap()):
    """
    Trains model on dataset
    """
    import torch.optim as optim
    from torch.utils.data.dataset import Dataset
    from torch.utils.data import DataLoader

    # This bit basically adds variables to the dotmap with default values
    p = DotMap()
    p.opt.n_epochs = parameters.opt.get('n_epochs', 10)
    p.opt.optimizer = optim.Adam
    p.opt.batch_size = parameters.opt.get('batch_size', 100)
    p.criterion = parameters.get("criterion", nn.MSELoss())
    p.learning_rate = parameters.get('learning_rate', 0.0001)
    p.useGPU = parameters.get('useGPU', False)
    p.verbosity = parameters.get('verbosity', 1)
    p.logs = parameters.get('logs', None)
    p.evaluator = parameters.get('evaluator', None)  # A function to run on the model every 25 batches

    # Init logs
    if p.logs is None:
        logs = DotMap()
        logs.training_error = []
        logs.training_error_epoch = []
        logs.evaluations = []
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
            scaled_input, scaled_output = model.preprocess(dataset)
            self.inputs = torch.from_numpy(scaled_input).float()
            self.outputs = torch.from_numpy(scaled_output).float()
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
    # dataset = PytorchDataset(dataset=dataset)  # Using PyTorch
    # dataset = np.hstack((dataset[0], dataset[1]))
    scaled_input, scaled_output = model.preprocess(dataset, cfg)
    dataset = list(zip(scaled_input, scaled_output))
    split = cfg.model.optimizer.split
    trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size=p.opt.batch_size, shuffle=True)
    testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size=p.opt.batch_size, shuffle=True)
    # loader = DataLoader(dataset, batch_size=p.opt.batch_size, shuffle=True)  ##shuffle=True #False
    # pin_memory=True
    # drop_last=False

    startTime = timer()
    if logs.time is None:
        logs.time = [0]

    # print("Training for %d epochs" % p.opt.n_epochs)

    for epoch in range(p.opt.n_epochs):
        epoch_error = 0
        log.info("Epoch %d" % (epoch))
        for i, (inputs, targets) in enumerate(trainLoader):
            if i % 500 == 0 and i > 0:
                print("    Batch %d" % i)
            # Load data
            # Variable is a wrapper for Tensors with autograd
            # inputs, targets = data
            if p.useGPU:
                inputs = Variable(inputs.cuda())
                targets = Variable(targets.cuda())
            else:
                inputs = Variable(inputs)
                targets = Variable(targets)

            optimizer.zero_grad()
            outputs = model.forward(inputs.float())
            loss = p.criterion(outputs, targets)
            # print(loss)

            e = loss.item()
            logs.training_error.append(e)
            epoch_error += e / (len(trainLoader) * p.opt.batch_size)

            loss.backward()
            optimizer.step()  # Does the update
            logs.time.append(timer() - logs.time[-1])

            if p.evaluator is not None and i % 25 == 0:
                logs.evaluations.append(p.evaluator(model))

        test_error = torch.zeros(1)
        for i, (inputs, targets) in enumerate(testLoader):
            outputs = model.forward(inputs.float())
            loss = p.criterion(outputs, targets)

            test_error += loss.item() / (len(testLoader) * p.opt.batch_size)
        test_error = test_error

        logs.training_error_epoch.append(epoch_error)

    endTime = timer()
    log.info('Optimization completed in %f[s]' % (endTime - startTime))

    return model.cpu(), logs
