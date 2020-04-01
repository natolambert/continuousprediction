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

from plot import plot_loss, plot_evaluations, plot_mse
from dynamics_model import DynamicsModel
from reacher_pd import log_hyperparams, create_dataset_traj, create_dataset_step
from evaluate import test_models, num_eval


def train(cfg, exper_data):
    """
    Trains one regular model based on cfg
    """
    n = cfg.training.num_traj
    t_range = cfg.training.t_range
    subset_data = exper_data[:n]

    prob = cfg.model.prob
    traj = cfg.model.traj
    ens = cfg.model.ensemble
    delta = cfg.model.delta

    log.info(f"Training model P:{prob}, T:{traj}, E:{ens} with n={n}")

    log_hyperparams(cfg)

    if traj:
        dataset = create_dataset_traj(subset_data, threshold=(n - 1) / n, t_range=t_range)
    else:
        dataset = create_dataset_step(subset_data, delta=delta, t_range=t_range)

    model = DynamicsModel(cfg)
    train_logs, test_logs = model.train(dataset, cfg)

    plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str + '_' + str(n), show=False)

    log.info("Saving new default models")
    f = hydra.utils.get_original_cwd() + '/models/reacher/efficiency/'
    if cfg.exper_dir:
        f = f + cfg.exper_dir
    f = f + cfg.model.str
    if not os.path.exists(f):
        os.makedirs(f)
    torch.save(model, '%s/n%d_t%d.dat'%(f, n, t_range))


def plot(cfg, train_data, test_data):
    graph_file = 'Plots'
    os.mkdir(graph_file)
    models = {}

    model_keys, ns, t_ranges = cfg.plotting.models, cfg.plotting.num_traj, cfg.plotting.t_range
    if type(ns) != int and type(t_ranges) != int:
        raise ValueError('t_range and num_traj cannot both be lists')
    if type(ns) != int:
        f_names = {n: 'n%d_t%d.dat' % (n, t_ranges) for n in ns}
        x_values = ns
        xlabel = '# training trajectories'
    else:
        f_names = {t_range: 'n%d_t%d.dat' % (ns, t_range) for t_range in t_ranges}
        x_values = t_ranges
        xlabel = 'training trajectory length'

    # Load models
    f = hydra.utils.get_original_cwd() + '/models/reacher'
    if cfg.exper_dir:
        f = f + cfg.exper_dir
    for model_type in model_keys:
        for x in x_values:
            model = torch.load("%s/efficiency/%s/%s" % (f, model_type, f_names[x]))
            models[(model_type, x)] = model

    # Plot
    def plot_helper(data, num, graph_file):
        """
        Helper to allow plotting for both train and test data without significant code duplication
        """
        if not num:
            return
        os.mkdir(graph_file)

        # Select a random subset of training data
        idx = np.random.randint(0, len(data), num)
        dat = [data[i] for i in idx]
        gt = np.array([traj.states for traj in dat])

        MSEs, predictions = test_models(dat, models)
        # Both of these are dictionaries of arrays. The keys are tuples (model_type, n) and the entries are the
        # evaluation values for the different
        eval_data_dot = num_eval(gt, predictions, setting='dot', T_range=cfg.plotting.eval_t_range)
        eval_data_mse = num_eval(gt, predictions, setting='mse', T_range=cfg.plotting.eval_t_range)

        if cfg.plotting.plot_all_eval or cfg.plotting.plot_avg_eval:
            eval_file = graph_file + '/eval'
            os.mkdir(eval_file)

        if cfg.plotting.plot_all_eval:
            for i, id in list(enumerate(idx)):
                file = "%s/test%d" % (eval_file, i + 1)
                os.mkdir(file)

                # Plot evaluations
                evals_dot = {key: [eval_data_dot[(key, x)][i] for x in x_values] for key in model_keys}
                evals_mse = {key: [eval_data_mse[(key, x)][i] for x in x_values] for key in model_keys}
                evals_mse_chopped = {key: [(num if num < 10 ** 5 else float("nan")) for num in evals_mse[key]] for key in evals_mse}
                plot_evaluations(evals_dot, x_values, ylabel='Dot product similarity', xlabel=xlabel,
                                 save_loc=file+'/efficiency_dot.pdf', show=False)
                plot_evaluations(evals_mse_chopped, x_values, ylabel='MSE similarity', xlabel=xlabel,
                                 save_loc=file + '/efficiency_mse.pdf', show=False, log_scale=True)

        # Plot averages
        if cfg.plotting.plot_avg_eval:
            evals_dot = {key: [np.average(eval_data_dot[(key, x)]) for x in x_values] for key in model_keys}
            evals_mse = {key: [np.average(eval_data_mse[(key, x)]) for x in x_values] for key in model_keys}
            evals_mse_chopped = {key: [(num if num < 10 ** 5 else float("nan")) for num in evals_mse[key]] for key in
                                 evals_mse}
            plot_evaluations(evals_dot, x_values, ylabel='Dot product similarity', xlabel=xlabel,
                             save_loc=eval_file + '/avg_efficiency_dot.pdf', show=False)
            plot_evaluations(evals_mse_chopped, x_values, ylabel='MSE similarity', xlabel=xlabel,
                             save_loc=eval_file + '/avg_efficiency_mse.pdf', show=False, log_scale=True)

        # Plot states
        if cfg.plotting.plot_states:
            # TODO: this
            for i, id in list(enumerate(idx)):
                pass


        # Plot MSEs
        if cfg.plotting.plot_avg_mse:
            file = graph_file + '/mse'
            os.mkdir(file)

            MSE_avgs = {x: {key: np.mean(MSEs[(key, x)], axis=0) for key in model_keys} for x in x_values}
            for x in x_values:
                chopped = {key: [(num if num < 10 ** 5 else float("nan")) for num in MSE_avgs[x][key]] for key in MSE_avgs[x]}
                plot_mse(chopped, save_loc=file+'/avg_mse_%d.pdf'%x, show=False, log_scale=True)






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
