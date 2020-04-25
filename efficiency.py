import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import matplotlib.pyplot as plt
import itertools

import torch
import gym
from envs import *

import hydra
import logging

log = logging.getLogger(__name__)

from plot import plot_loss, plot_evaluations, plot_evaluations_3d, setup_plotting, plot_mse
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

    setup_plotting({model.str: model})
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
    if type(ns) == int:
        ns = [ns]
    if type(t_ranges) == int:
        t_ranges = [t_ranges]
    f_names = {}
    for n, t in itertools.product(ns, t_ranges):
        f_names[(n, t)] = 'n%d_t%d.dat' % (n, t)

    # Load models
    f = hydra.utils.get_original_cwd() + '/models/reacher'
    if cfg.exper_dir:
        f = f + cfg.exper_dir
    for model_type in model_keys:
        for key in f_names:
            model = torch.load("%s/efficiency/%s/%s" % (f, model_type, f_names[key]))
            models[(model_type, key)] = model

    setup_plotting(models)

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
        eval_data_dot = num_eval(gt, predictions, models, setting='dot', T_range=cfg.plotting.eval_t_range)
        eval_data_gauss = num_eval(gt, predictions, models, setting='gaussian', T_range=cfg.plotting.eval_t_range)
        eval_data_mse = num_eval(gt, predictions, models, setting='mse', T_range=cfg.plotting.eval_t_range)

        # Initialize dictionaries that will hold the data in 2d arrays that are better suited to plotting heatmaps,
        # then move the data into those dictionaries
        n_eval = gt.shape[0]
        evals_dot = {key: np.zeros((n_eval, len(ns), len(t_ranges))) for key in model_keys}
        evals_gauss = {key: np.zeros((n_eval, len(ns), len(t_ranges))) for key in model_keys}
        evals_mse = {key: np.zeros((n_eval, len(ns), len(t_ranges))) for key in model_keys}
        for (model_type, (n, t)) in eval_data_dot:
            evals_dot[model_type][:, ns.index(n), t_ranges.index(t)] = np.nan_to_num(eval_data_dot[(model_type, (n, t))])
        for (model_type, (n, t)) in eval_data_gauss:
            evals_gauss[model_type][:, ns.index(n), t_ranges.index(t)] = eval_data_gauss[(model_type, (n, t))]
        for (model_type, (n, t)) in eval_data_mse:
            dat = eval_data_mse[(model_type, (n, t))]
            evals_mse[model_type][:, ns.index(n), t_ranges.index(t)] = np.minimum(dat, 100)
            # The line above caps MSE at 100, which I found to be necessary to get good-looking heatmaps
            # TODO update that ^^^ to make it work for plotting variation over one variable at a time

        if cfg.plotting.plot_all_eval or cfg.plotting.plot_avg_eval:
            eval_file = graph_file + '/eval/'
            os.mkdir(eval_file)

        if cfg.plotting.plot_all_eval:
            for i, id in list(enumerate(idx)):
                file = "%s/test%d" % (eval_file, i + 1)
                os.mkdir(file)

                evals_dot_slice = {key: evals_dot[key][i, :, :] for key in evals_dot}
                evals_gauss_slice = {key: evals_gauss[key][i, :, :] for key in evals_gauss}
                evals_mse_slice = {key: 1 / evals_mse[key][i, :, :] for key in evals_mse}

                # Plot evaluations
                if len(ns) > 1 and len(t_ranges) > 1:
                    plot_evaluations_3d(evals_dot_slice, t_ranges, ns, ylabel='# training trajectories',
                                        xlabel='training trajectory length', zlabel='Dot product similarity',
                                        save_loc=file+'efficiency_dot', show=False)
                    plot_evaluations_3d(evals_gauss_slice, t_ranges, ns, ylabel='# training trajectories',
                                        xlabel='training trajectory length', zlabel='Gaussian similarity',
                                        save_loc=file + 'efficiency_gauss', show=False)
                    plot_evaluations_3d(evals_mse_slice, t_ranges, ns, ylabel='# training trajectories',
                                        xlabel='training trajectory length', zlabel='MSE similarity',
                                        save_loc=file + 'efficiency_mse', show=False)
                else:
                    if len(ns) > 1:
                        x_values = ns
                        xlabel = '# training trajectories'
                    else:
                        x_values = t_ranges
                        xlabel = 'training trajectory length'
                    plot_evaluations(evals_dot_slice, x_values, ylabel='Dot product similarity', xlabel=xlabel,
                                     save_loc=file+'/efficiency_dot.pdf', show=False)
                    plot_evaluations(evals_mse_slice, x_values, ylabel='MSE similarity', xlabel=xlabel,
                                     save_loc=file + '/efficiency_mse.pdf', show=False, log_scale=True)

        # Plot averages
        if cfg.plotting.plot_avg_eval:

            evals_dot_avg = {key: np.average(evals_dot[key], axis=0) for key in evals_dot}
            evals_gauss_avg = {key: np.average(evals_gauss[key], axis=0) for key in evals_gauss}
            evals_mse_avg = {key: np.average(evals_mse[key], axis=0) for key in evals_mse}

            # Plot evaluations
            if len(ns) > 1 and len(t_ranges) > 1:
                plot_evaluations_3d(evals_dot_avg, t_ranges, ns, ylabel='# training trajectories',
                                    xlabel='training trajectory length', zlabel='Dot product similarity',
                                    save_loc=eval_file + 'efficiency_dot', show=False)
                plot_evaluations_3d(evals_gauss_avg, t_ranges, ns, ylabel='# training trajectories',
                                    xlabel='training trajectory length', zlabel='Gaussian similarity',
                                    save_loc=eval_file + 'efficiency_gauss', show=False)
                plot_evaluations_3d(evals_mse_avg, t_ranges, ns, ylabel='# training trajectories',
                                    xlabel='training trajectory length', zlabel='Log mean square error',
                                    save_loc=eval_file + 'efficiency_mse', log_scale=True, show=False)
            else:
                if len(ns) > 1:
                    x_values = ns
                    xlabel = '# training trajectories'
                else:
                    x_values = t_ranges
                    xlabel = 'training trajectory length'
                plot_evaluations(evals_dot_avg, x_values, ylabel='Dot product similarity', xlabel=xlabel,
                                 save_loc=eval_file + '/efficiency_dot.pdf', show=False)
                plot_evaluations(evals_mse_avg, x_values, ylabel='MSE similarity', xlabel=xlabel,
                                 save_loc=eval_file + '/efficiency_mse.pdf', show=False, log_scale=True)

        # Plot states
        if cfg.plotting.plot_states:
            # TODO: this
            for i, id in list(enumerate(idx)):
                pass

        # Plot MSEs
        if cfg.plotting.plot_avg_mse:
            file = graph_file + '/mse'
            os.mkdir(file)

            # MSE_avgs = {x: {key: np.mean(MSEs[(key, x)], axis=0) for key in model_keys} for x in x_values}
            MSE_avgs = {key: {tup: np.average(MSEs[(key, tup)], axis=0) for tup in itertools.product(ns, t_ranges)} for key in model_keys}
            MSE_chopped = {key: {tup: [num if num < 1e5 else float('nan') for num in MSE_avgs[key][tup]] for tup in MSE_avgs[key]} for key in MSE_avgs}

            for key in model_keys:
                mses = MSE_chopped[key]
                # arbitrarily chosen color
                r = np.linspace(187/255, 109/255, len(mses))
                g = np.linspace(153/255, 36/255, len(mses))
                b = 1
                tups = list(set(mses))
                tups.sort()
                colors = {tups[i]: (r[i], g[i], b) for i in range(len(mses))}
                names = {tup: ('n: %d, t: %d' % tup) for tup in tups}
                plot_mse(mses, title='MSE efficiency for %s' % models[(key,tups[0])].cfg.model.plotting.label,
                         custom_colors=colors, custom_labels=names, show=False,
                         save_loc=file+'/%s.pdf' % models[(key, tups[0])].cfg.model.str)




            # for x in x_values:
            #     chopped = {key: [(num if num < 10 ** 5 else float("nan")) for num in MSE_avgs[x][key]] for key in MSE_avgs[x]}
            #     plot_mse(chopped, save_loc=file+'/avg_mse_%d.pdf'%x, show=False, log_scale=True)


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
