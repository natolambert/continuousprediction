import sys
import warnings
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from collections import OrderedDict
import hydra


class Net(nn.Module):
    """
    General Neural Network
    """

    def __init__(self, n_in, n_out, cfg, loss_fn, tf=nn.ReLU()):
        """
        :param structure: layer sizes
        :param tf: nonlinearity function
        """
        super(Net, self).__init__()

        self.activation = tf
        self._onGPU = False
        self.loss_fn = loss_fn

        self.n_in = n_in
        self.n_out = n_out
        self.hidden_w = cfg.model.training.hid_width
        self.cfg = cfg

        # create object nicely
        layers = []
        layers.append(('dynm_input_lin', nn.Linear(self.n_in, self.hidden_w)))
        layers.append(('dynm_input_act', self.activation))
        for d in range(cfg.model.training.hid_depth):
            layers.append(('dynm_lin_' + str(d), nn.Linear(self.hidden_w, self.hidden_w)))
            layers.append(('dynm_act_' + str(d), self.activation))

        layers.append(('dynm_out_lin', nn.Linear(self.hidden_w, self.n_out)))
        self.features = nn.Sequential(OrderedDict([*layers]))

    def forward(self, x):
        """
        Runs a forward pass of x through this network
        """
        # TODO: to make it run I had the model call .float() on inputs, but this might affect performance
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = self.features(x.float())
        return x

    def testPreprocess(self, input, cfg):
        if (cfg.model.traj):
            inputStates = input[:, :cfg.env.state_size]
            inputIndex = input[:, cfg.env.state_size]
            inputParams = input[:, cfg.env.state_size + 1:]

            inputIndex = inputIndex.reshape(-1, 1)

            normStates = self.stateScaler.transform(inputStates)
            normIndex = self.indexScaler.transform(inputIndex)
            normParams = self.paramScaler.transform(inputParams)
            return np.hstack((normStates, normIndex, normParams))
        else:
            inputStates = input[:, :cfg.env.state_size]
            normStates = self.stateScaler.transform(inputStates)
            if cfg.env.action_size > 0:
                inputActions = input[:, cfg.env.state_size:]
                normActions = self.actionScaler.transform(inputActions)
                normInput = np.hstack((normStates, normActions))
                return normInput
            else:
                normInput = normStates
                return normInput

    def testPostprocess(self, output):
        return torch.from_numpy(self.outputScaler.inverse_transform(output.detach().numpy()))

    def preprocess(self, dataset, cfg):

        # Select scaling, minmax vs standard (fits to a gaussian with unit variance and 0 mean)
        # TODO: Selection should be in config
        # StandardScaler, MinMaxScaler
        # TODO: Instead of hardcoding, include in config, trajectory vs. one-step, length of different inputs, etc.
        # 26 -> one-step, 37 -> trajectory
        input = dataset[0]
        output = dataset[1]
        if (cfg.model.traj):
            self.stateScaler = hydra.utils.instantiate(cfg.model.preprocess.state)
            self.indexScaler = hydra.utils.instantiate(cfg.model.preprocess.index)
            self.paramScaler = hydra.utils.instantiate(cfg.model.preprocess.param)
            self.outputScaler = hydra.utils.instantiate(cfg.model.preprocess.output)

            inputStates = input[:, :cfg.env.state_size]
            inputIndex = input[:, cfg.env.state_size]
            inputParams = input[:, cfg.env.state_size + 1:]

            # reshape index for one feature length
            inputIndex = inputIndex.reshape(-1, 1)

            self.stateScaler.fit(inputStates)
            self.indexScaler.fit(inputIndex)
            self.paramScaler.fit(inputParams)
            self.outputScaler.fit(output)

            normStates = self.stateScaler.transform(inputStates)
            normIndex = self.indexScaler.transform(inputIndex)
            normParams = self.paramScaler.transform(inputParams)
            normOutput = self.outputScaler.transform(output)
            normInput = np.hstack((normStates, normIndex, normParams))
            return list(zip(normInput, normOutput))
        else:
            self.stateScaler = hydra.utils.instantiate(cfg.model.preprocess.state)
            self.actionScaler = hydra.utils.instantiate(cfg.model.preprocess.action)
            self.outputScaler = hydra.utils.instantiate(cfg.model.preprocess.output)

            inputStates = input[:, :cfg.env.state_size]
            inputActions = input[:, cfg.env.state_size:]

            self.stateScaler.fit(inputStates)

            self.outputScaler.fit(output)
            normStates = self.stateScaler.transform(inputStates)
            normOutput = self.outputScaler.transform(output)

            if np.shape(inputActions)[1] > 0:
                self.actionScaler.fit(inputActions)
                normActions = self.actionScaler.transform(inputActions)
                normInput = np.hstack((normStates, normActions))
            else:
                normInput = normStates

            return list(zip(normInput, normOutput))

    def optimize(self, dataset, cfg):
        """
        Uses dataset to train this net according to the parameters in cfg
        Returns:
            train_errors: a list of average errors for each epoch on the training data
            test_errors: a list of average errors for each epoch on the test data
        """
        from torch.utils.data import DataLoader

        # Extract parameters from cfg
        lr = cfg.model.optimizer.lr
        bs = cfg.model.optimizer.batch
        split = cfg.model.optimizer.split
        epochs = cfg.model.optimizer.epochs

        # Set up the optimizer and scheduler
        # TODO: the scheduler is currently unused. Should it be doing something it isn't or removed?
        optimizer = torch.optim.Adam(self.features.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.7)

        # data preprocessing for normalization
        dataset = self.preprocess(dataset, cfg)

        if 0 < cfg.model.optimizer.max_size < len(dataset):
            import random
            dataset = random.sample(dataset, cfg.model.optimizer.max_size)

        # Puts it in PyTorch dataset form and then converts to DataLoader
        trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size=bs, shuffle=True)
        testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size=bs, shuffle=True)

        # Optimization loop
        train_errors = []
        test_errors = []
        for epoch in range(epochs):
            print("    Epoch %d" % (epoch + 1))

            train_error = 0
            test_error = 0

            # Iterate through dataset and take gradient descent steps
            for i, (inputs, targets) in enumerate(trainLoader):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs.float(), targets.float())
                train_error += loss.item() / (len(trainLoader) * bs)

                loss.backward()
                optimizer.step()  # Does the update

            # Iterate through dataset to calculate test set accuracy
            test_error = torch.zeros(1)
            for i, (inputs, targets) in enumerate(testLoader):
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs.float(), targets.float())
                test_error += loss.item() / (len(testLoader) * bs)

            train_errors.append(train_error)
            test_errors.append(test_error.item())

        return train_errors, test_errors


class DynamicsModel(object):
    """
    Wrapper class for a general dynamics model.
    The model is an ensemble of neural nets. For cases where the model should not be an ensemble it is just
    an ensemble of 1 net.
    """

    def __init__(self, cfg):
        self.ens = cfg.model.ensemble
        self.traj = cfg.model.traj
        self.prob = cfg.model.prob
        self.delta = cfg.model.delta
        self.cfg = cfg

        # Setup for data structure
        if self.ens:
            self.E = cfg.model.training.E
        else:
            self.E = 1

        if self.traj:
            self.n_in = cfg.env.state_size + (cfg.env.param_size) + 1
        else:
            self.n_in = cfg.env.state_size + cfg.env.action_size

        self.n_out = cfg.env.state_size
        if self.prob:
            # ordering matters here, because size is the number of predicted output states
            self.loss_fn = ProbLoss(self.n_out)
            self.n_out = self.n_out * 2
        else:
            self.loss_fn = nn.MSELoss()

        self.nets = [Net(self.n_in, self.n_out, cfg, self.loss_fn) for i in range(self.E)]

    def predict(self, x):
        """
        Use the model to predict values with x as input
        TODO: Fix hardcoding in this method
        TODO: particle sampling approach for probabilistic model
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        prediction = torch.zeros((x.shape[0], self.cfg.env.state_size))
        for n in self.nets:
            scaledInput = n.testPreprocess(x, self.cfg)
            if self.prob:
                prediction += n.testPostprocess(n.forward(scaledInput)[:, :self.cfg.env.state_size]) / len(self.nets)
            else:
                prediction += n.testPostprocess(n.forward(scaledInput)) / len(self.nets)
        if not self.delta:
            return prediction[:, :self.cfg.env.state_size]
        else:
            # This hardcode is the state size changing. X also includes the action / index
            return x[:, :self.cfg.env.state_size] + prediction

    def train(self, dataset, cfg):
        acctest_l = []
        acctrain_l = []

        from sklearn.model_selection import KFold  # for dataset

        if self.ens:
            # setup cross validation-ish datasets for training ensemble
            kf = KFold(n_splits=self.E)
            kf.get_n_splits(dataset)

            # iterate through the validation sets
            for (i, n), (train_idx, test_idx) in zip(enumerate(self.nets), kf.split(dataset[0])):
                print("  Model %d" % (i + 1))
                # only train on training data to ensure diversity
                sub_data = (dataset[0][train_idx], dataset[1][train_idx])
                train_e, test_e = n.optimize(sub_data, cfg)
                acctrain_l.append(train_e)
                acctest_l.append(test_e)
        else:
            train_e, test_e = self.nets[0].optimize(dataset, cfg)
            acctrain_l.append(train_e)
            acctest_l.append(test_e)

        self.acctrain, self.acctest = acctrain_l, acctest_l

        return acctrain_l, acctest_l


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
        mid = torch.div(diff, var)
        lg = torch.sum(torch.log(var))
        out = torch.trace(torch.mm(diff, mid.t())) + lg
        # same as torch.sum(((mean - targets) ** 2) / var) + lg
        return out
