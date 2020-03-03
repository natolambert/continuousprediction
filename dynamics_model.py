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
from mbrl_resources import ProbLoss


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
        x = self.features(x)
        return x

    def optimize(self, dataset, cfg):
        from torch.utils.data import DataLoader

        # This bit basically adds variables to the dotmap with default values
        lr = cfg.model.optimizer.lr
        bs = cfg.model.optimizer.batch
        split = cfg.model.optimizer.split
        epochs = cfg.model.optimizer.epochs
        # Optimizer
        # optimizer = torch.optim.Adam(super(DynamicsModel, self).parameters(), lr=lr)
        optimizer = torch.optim.Adam(self.features.parameters(), lr=lr)
        # optimizer = torch.optim.SGD(super(GeneralNN, self).parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.7)

        # Puts it in PyTorch dataset form and then converts to DataLoader
        dataset = list(zip(dataset[0], dataset[1]))
        trainLoader = DataLoader(dataset[:int(split * len(dataset))], batch_size=bs, shuffle=True)
        testLoader = DataLoader(dataset[int(split * len(dataset)):], batch_size=bs, shuffle=True)

        train_errors = []
        test_errors = []
        for epoch in range(epochs):
            train_error = 0
            test_error = 0
            # log.info("Epoch %d" % (epoch))
            for i, (inputs, targets) in enumerate(trainLoader):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, targets)
                train_error += loss.item() / (len(trainLoader) * bs)

                loss.backward()
                optimizer.step()  # Does the update

            test_error = torch.zeros(1)
            for i, (inputs, targets) in enumerate(testLoader):
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs, targets)
                test_error += loss.item() / (len(testLoader) * bs)

            train_errors.append(train_error)
            test_errors.append(test_error)

        return train_errors, test_errors


class DynamicsModel(object):
    def __init__(self, cfg):
        self.ens = cfg.model.ensemble
        self.traj = cfg.model.traj
        self.prob = cfg.model.prob

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
        prediction = torch.zeros(self.n_out)
        for n in self.nets:
            prediction += n.forward(x) / len(self.nets)
        if self.traj:
            return prediction
        else:
            return x + prediction

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
                # only train on training data to ensure diversity
                raise NotImplementedError("Fix K-folding")
                sub_data = dataset[train_idx]
                train_e, test_e = n.optimize(sub_data, cfg)
                acctrain_l.append(train_e)
                acctest_l.append(test_e)
        else:
            train_e, test_e = self.nets[0].optimize(dataset, cfg)
            acctrain_l.append(train_e)
            acctest_l.append(test_e)

        return acctrain_l, acctest_l

