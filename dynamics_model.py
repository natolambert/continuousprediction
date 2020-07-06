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
import math
import GPy

class GP(object):
    def __init__(self, n_in, n_out, cfg, loss_fn, env = "Reacher", tf=nn.ReLU()):
        self.name = 'GP'  # Default value
        self.probabilistic = True  # Default value
        # self.verbosity = parameters.get('verbosity', 3)
        # self.indent = parameters.get('indent', 0)
        self.n_inputs = n_in
        self.n_outputs = n_out
        self.kernel = 'Matern52'
        self.ARD = True
        self.fixNoise = None

        self.normalizeOutput = False
        self.t_output = None  # Store the output transformation

        self._kernel = []
        self._model = []
        self._logs = []
        self._startTime = None

    def optimize(self,  dataset, cfg):
        # logging.info('Training GP')
        # self._startTime = timer()

        # if self.normalizeOutput is True:
        #     self.t_output = train_set.normalize_output()
        # self.n_inputs = train_set.get_dim_input()
        # self.n_outputs = train_set.get_dim_output()
        # logging.info('Dataset %d -> %d with %d data' % (self.n_inputs, self.n_outputs, train_set.get_n_data()))

        if 0 < cfg.model.optimizer.max_size < len(dataset[0]):
            use = np.random.randint(0,len(dataset[0]), cfg.model.optimizer.max_size)
            d = []
            d.append(dataset[0][use])
            d.append(dataset[1][use])

        for i in range(self.n_outputs):
            # logging.info('Training covariate %d (of %d)' % (i+1, self.n_outputs))
            print('Training covariate %d (of %d)' % (i+1, self.n_outputs))
            if self.kernel == 'Matern52':
                self._kernel.append(GPy.kern.Matern52(input_dim=self.n_inputs, ARD=self.ARD))
            if self.kernel == 'Linear':
                self._kernel.append(GPy.kern.Linear(input_dim=self.n_inputs, ARD=self.ARD))

            # TODO check this line
            self._model.append(GPy.models.GPRegression(d[0], d[1][:,1].reshape(-1,1), kernel=self._kernel[i]))
            if self.fixNoise is not None:
                self._model[i].likelihood.variance.fix(self.fixNoise)
            self._model[i].optimize_restarts(num_restarts=10, verbose=False)  # , parallel=True, num_processes=5

        return 0,0
        #
        # end = timer()
        # logging.info('Training completed in %f[s]' % (end - self._startTime))

    def forward(self, x):
        n_data =np.shape(x)[0]
        mean = np.zeros((n_data, self.n_outputs))
        var = np.zeros((n_data, self.n_outputs))
        for i in range(self.n_outputs):
            t_mean, t_var = self._model[i].predict(np.array(x))
            mean[:, i] = t_mean.T
            var[:, i] = t_var.T
        if np.any(var < 0):
            # logging.warning('Variance was negative! Now it is 0, but you should be careful!')
            var[var < 0] = 0  # Make sure that variance is always positive
        return np.concatenate((mean, var))


    def get_hyperparameters(self):
        return self._model._param_array_

class Net(nn.Module):
    """
    General Neural Network
    """

    def __init__(self, n_in, n_out, cfg, loss_fn, env = "Reacher", tf=nn.ReLU()):
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
        if env == "Reacher":
            self.state_indices = cfg.model.training.state_indices
        elif env == "Lorenz":
            self.state_indices = cfg.model.training.state_indices_lorenz

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
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = self.features(x.float())
        return x

    def testPreprocess(self, input, cfg):
        if (cfg.model.traj):
            inputStates = input[:, :len(self.state_indices)]
            inputIndex = input[:, len(self.state_indices)]
            inputParams = input[:, len(self.state_indices) + 1:]

            inputIndex = inputIndex.reshape(-1, 1)

            normStates = self.stateScaler.transform(inputStates)
            normIndex = self.indexScaler.transform(inputIndex)
            normParams = self.paramScaler.transform(inputParams)
            return np.hstack((normStates, normIndex, normParams))
        else:
            inputStates = input[:, :len(self.state_indices)]
            normStates = self.stateScaler.transform(inputStates)
            if cfg.env.action_size > 0:
                inputActions = input[:, len(self.state_indices):]
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
        if cfg.model.traj:
            #no control params (state + time index)
            if np.shape(dataset[0])[1] == len(self.state_indices)+1:
                self.stateScaler = hydra.utils.instantiate(cfg.model.preprocess.state)
                self.indexScaler = hydra.utils.instantiate(cfg.model.preprocess.index)
                self.outputScaler = hydra.utils.instantiate(cfg.model.preprocess.output)

                inputStates = input[:, :len(self.state_indices)]
                inputIndex = input[:, len(self.state_indices)]

                # reshape index for one feature length
                inputIndex = inputIndex.reshape(-1, 1)

                self.stateScaler.fit(inputStates)
                self.indexScaler.fit(inputIndex)

                self.outputScaler.fit(output)

                normStates = self.stateScaler.transform(inputStates)
                normIndex = self.indexScaler.transform(inputIndex)

                normOutput = self.outputScaler.transform(output)
                normInput = np.hstack((normStates, normIndex))

            # control params
            else:
                self.stateScaler = hydra.utils.instantiate(cfg.model.preprocess.state)
                self.indexScaler = hydra.utils.instantiate(cfg.model.preprocess.index)
                self.paramScaler = hydra.utils.instantiate(cfg.model.preprocess.param)
                self.outputScaler = hydra.utils.instantiate(cfg.model.preprocess.output)

                inputStates = input[:, :len(self.state_indices)]
                inputIndex = input[:, len(self.state_indices)]
                inputParams = input[:, len(self.state_indices) + 1:]

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

            inputStates = input[:, :len(self.state_indices)]
            inputActions = input[:, len(self.state_indices):]

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
        optimizer = torch.optim.Adam(self.features.parameters(), lr=lr, weight_decay=cfg.model.optimizer.regularization)
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

            train_error = 0
            test_error = 0

            # Iterate through dataset and take gradient descent steps
            for i, (inputs, targets) in enumerate(trainLoader):
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs.float(), targets.float())
                train_error += loss.item() / (len(trainLoader))

                loss.backward()
                optimizer.step()  # Does the update

            # Iterate through dataset to calculate test set accuracy
            # test_error = torch.zeros(1)
            for i, (inputs, targets) in enumerate(testLoader):
                outputs = self.forward(inputs)
                loss = self.loss_fn(outputs.float(), targets.float())
                test_error += loss.item() / (len(testLoader))

            print(f"    Epoch {epoch + 1}, Train err: {train_error}, Test err: {test_error}")
            train_errors.append(train_error)
            test_errors.append(test_error)

        return train_errors, test_errors


class DynamicsModel(object):
    """
    Wrapper class for a general dynamics model.
    The model is an ensemble of neural nets. For cases where the model should not be an ensemble it is just
    an ensemble of 1 net.
    """

    def __init__(self, cfg, env="Reacher"):
        self.str = cfg.model.str
        self.ens = cfg.model.ensemble
        self.traj = cfg.model.traj
        self.prob = cfg.model.prob
        self.delta = cfg.model.delta
        self.train_target = cfg.model.training.train_target
        self.control_params = cfg.model.training.control_params
        if env == "Reacher":
            self.state_indices = cfg.model.training.state_indices
        elif env == "Lorenz":
            self.state_indices = cfg.model.training.state_indices_lorenz
        self.cfg = cfg

        # Setup for data structure
        if self.ens:
            self.E = cfg.model.training.E
        else:
            self.E = 1

        self.n_in = len(self.state_indices)
        if self.traj:
            self.n_in += 1
            if self.control_params:
                self.n_in += cfg.env.param_size - cfg.env.target_size
            if self.train_target:
                self.n_in += cfg.env.target_size
        else:
            self.n_in += cfg.env.action_size

        self.n_out = len(self.state_indices)
        if self.prob:
            # ordering matters here, because size is the number of predicted output states
            self.loss_fn = ProbLoss(self.n_out)
            self.n_out = self.n_out * 2
        else:
            self.loss_fn = nn.MSELoss()
        if env == "Reacher":
            if cfg.model.gp:
                self.nets = [GP(self.n_in, self.n_out, cfg, self.loss_fn) for i in range(self.E)]
            else:
                self.nets = [Net(self.n_in, self.n_out, cfg, self.loss_fn) for i in range(self.E)]
        elif env == "Lorenz":
            self.nets = [Net(self.n_in, self.n_out, cfg, self.loss_fn, env="Lorenz") for i in range(self.E)]

    def predict(self, x):
        """
        Use the model to predict values with x as input
        TODO: Fix hardcoding in this method
        TODO: particle sampling approach for probabilistic model
        """
        if type(x) == np.ndarray:
            x = torch.from_numpy(np.float64(x))
        prediction = torch.zeros((x.shape[0], len(self.state_indices)))
        for n in self.nets:
            scaledInput = n.testPreprocess(x, self.cfg)
            if self.prob:
                prediction += n.testPostprocess(n.forward(scaledInput)[:, :len(self.state_indices)]) / len(self.nets)
                # prediction += n.forward(scaledInput)[:, :len(self.state_indices)] / len(self.nets)
            else:
                prediction += n.testPostprocess(n.forward(scaledInput)) / len(self.nets)
                # prediction += n.forward(scaledInput) / len(self.nets)
        if not self.delta:
            return prediction[:, :len(self.state_indices)]
        else:
            # This hardcode is the state size changing. X also includes the action / index
            return x[:, :len(self.state_indices)] + prediction

    def train(self, dataset, cfg):
        acctest_l = []
        acctrain_l = []

        # The purpose of this line is to reform the dataset to use only the state indices requested
        if not self.train_target and not self.control_params:
            dataset = (np.hstack((dataset[0][:, self.state_indices],
                                  # dataset[0][:, (self.cfg.env.state_size - len(self.state_indices)):])),
                                  dataset[0][:, [self.cfg.env.state_size]])),
                       dataset[1][:, self.state_indices])
        else:
            dataset = (np.hstack((dataset[0][:, self.state_indices],
                              # dataset[0][:, (self.cfg.env.state_size - len(self.state_indices)):])),
                              dataset[0][:, self.cfg.env.state_size:])),
                   dataset[1][:, self.state_indices])

        if self.ens:
            from sklearn.model_selection import KFold  # for dataset

            # setup cross validation-ish datasets for training ensemble
            kf = KFold(n_splits=self.E)
            kf.get_n_splits(dataset)

            # iterate through the validation sets
            for (i, n), (train_idx, test_idx) in zip(enumerate(self.nets), kf.split(dataset[0])):
                print("  Model %d" % (i + 1))
                # only train on training data to ensure diversity
                # sub_data = (dataset[0][train_idx], dataset[1][train_idx])
                # num = len(dataset[0])
                # k = int(num * .8)
                # train_idx = np.random.choice(num, k)
                sub_data = (dataset[0][train_idx], dataset[1][train_idx])
                print('test ' + str(len(sub_data[0])))
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

    def forward(self, inputs, targets):
        # size = targets.size()[1]
        mean = inputs[:, :self.size]
        logvar = inputs[:, self.size:]

        # Caps max and min log to avoid NaNs
        # logvar = self.max_logvar - self.softplus_raw(self.max_logvar - logvar)
        # logvar = self.min_logvar + self.softplus_raw(logvar - self.min_logvar)

        logvar = torch.min(logvar, self.max_logvar)
        logvar = torch.max(logvar, self.min_logvar)

        var = torch.exp(logvar)

        diff = mean - targets
        mid = torch.div(diff, var)
        lg = torch.sum(torch.log(var))
        out = torch.trace(torch.mm(diff, mid.t())) + lg
        # same as torch.sum(((mean - targets) ** 2) / var) + lg
        return out
