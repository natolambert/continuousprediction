import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt

import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
import gym
from envs import *

import hydra
import logging

log = logging.getLogger(__name__)

from plot import plot_loss, plot_efficiency
from dynamics_model import DynamicsModel
from reacher_pd import log_hyperparams, create_dataset_traj, create_dataset_step
from evaluate import test_models, num_eval


def train(cfg, exper_data):
    """
    Trains one regular model based on cfg
    """
    n = cfg.training.num_traj
    subset_data = exper_data[:n]

    prob = cfg.model.prob
    traj = cfg.model.traj
    ens = cfg.model.ensemble
    delta = cfg.model.delta

    log.info(f"Training model P:{prob}, T:{traj}, E:{ens} with n={n}")

    log_hyperparams(cfg)

    if traj:
        dataset = create_dataset_traj(subset_data, threshold=(n - 1) / n)
    else:
        dataset = create_dataset_step(subset_data, delta=delta)

    model = DynamicsModel(cfg)
    train_logs, test_logs = model.train(dataset, cfg)

    plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str + '_' + str(n), show=False)

    log.info("Saving new default models")
    f = hydra.utils.get_original_cwd() + '/models/reacher/efficiency/'
    if cfg.exper_dir:
        f = f + cfg.exper_dir + '/'
    f = f + cfg.model.str + '/'
    if not os.path.exists(f):
        os.makedirs(f)
    torch.save(model, f + str(n) + '.dat')


def plot(cfg, train_data, test_data):
    graph_file = 'Plots'
    os.mkdir(graph_file)
    models = {}

    # Load models
    f = hydra.utils.get_original_cwd() + '/models/reacher'
    if cfg.exper_dir:
        f = f + cfg.exper_dir
    for type in cfg.plotting.models:
        for n in cfg.plotting.num_traj:
            model = torch.load("%s/efficiency/%s/%d.dat" % (f, type, n))
            models[(type, n)] = model

    # Plot
    def plot_helper(data, num, graph_file):
        """
        Helper to allow plotting for both train and test data without significant code duplication
        """
        os.mkdir(graph_file)

        # Select a random subset of training data
        idx = np.random.randint(0, len(data), num)
        dat = [data[i] for i in idx]
        gt = np.array([traj.states for traj in dat])

        MSEs, predictions = test_models(dat, models)
        # Both of these are dictionaries of arrays. The keys are tuples (model_type, n) and the entries are the
        # evaluation values for the different 
        eval_data_dot = num_eval(gt, predictions, setting='dot', T_range=cfg.plotting.t_range)
        eval_data_mse = num_eval(gt, predictions, setting='mse', T_range=cfg.plotting.t_range)

        for i, id in list(enumerate(idx)):
            file = "%s/test%d" % (graph_file, i + 1)
            os.mkdir(file)

            # Plot evaluations
            evals_dot = {key: [eval_data_dot[(key, n)][i] for n in cfg.plotting.num_traj] for key in
                         cfg.plotting.models}
            evals_mse = {key: [eval_data_mse[(key, n)][i] for n in cfg.plotting.num_traj] for key in
                         cfg.plotting.models}
            plot_efficiency(evals_dot, cfg.plotting.num_traj, ylabel='Dot product similarity',
                            save_loc=file+'/efficiency_dot.pdf', show=False)
            plot_efficiency(evals_mse, cfg.plotting.num_traj, ylabel='MSE similarity',
                            save_loc=file + '/efficiency_mse.pdf', show=False, log_scale=True)

        # Plot averages
        evals_dot = {key: [np.average(eval_data_dot[(key, n)]) for n in cfg.plotting.num_traj] for key in
                     cfg.plotting.models}
        evals_mse = {key: [np.average(eval_data_mse[(key, n)]) for n in cfg.plotting.num_traj] for key in
                     cfg.plotting.models}
        plot_efficiency(evals_dot, cfg.plotting.num_traj, ylabel='Dot product similarity',
                        save_loc=graph_file + '/avg_efficiency_dot.pdf', show=False)
        plot_efficiency(evals_mse, cfg.plotting.num_traj, ylabel='MSE similarity',
                        save_loc=graph_file + '/avg_efficiency_mse.pdf', show=False, log_scale=True)




    if cfg.plotting.num_eval_train:
        log.info("Plotting train data")

        file = graph_file + "/train_data"

        plot_helper(train_data, cfg.plotting.num_eval_train, file)

    if cfg.plotting.num_eval_test:
        log.info("Plotting test data")

        file = graph_file + '/test_data'

        plot_helper(test_data, cfg.plotting.num_eval_test, file)



@hydra.main(config_path='conf/eff.yaml')
def eff(cfg):

    log.info(f"Loading default data")
    (train_data, test_data) = torch.load(
        hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    if cfg.mode == 'train':
        train(cfg, train_data)
    elif cfg.mode == 'plot':
        plot(cfg, train_data, test_data)




if __name__ == '__main__':
    sys.exit(eff())
