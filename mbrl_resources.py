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


def obs2q(obs):
    if len(obs) < 5:
        return obs
    else:
        return obs[0:5]


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
