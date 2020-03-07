"""
The purpose of this file is to load in pre-loaded data and pre-trained models and evaluate them
"""

import sys

import hydra
import logging

import torch
import numpy as np

from plot import *

# from mbrl_resources import Model

log = logging.getLogger(__name__)


def test_models(test_data, models):
    """
    Tests each of the models in the dictionary "models" on each of the trajectories in test_data

    Parameters:
    ------------
    test_data: the trajectories to test on, N trajectories
    models: a dictionary of models to test, M models

    Returns:
    outcomes: a dictionary of MSEs and predictions. As an example of how to
              get info from this distionary, to get the MSE data from a trajectory
              -based model you'd do
                    outcomes['mse']['t']
    """

    log.info("Beginning testing of predictions")

    MSEs = {key: [] for key in models}

    states, actions, initials = [], [], []
    P, D, target = [], [], []

    for traj in test_data:
        states.append(traj.states)
        actions.append(traj.actions)
        initials.append(traj.states[0, :])
        P.append(traj.P)
        D.append(traj.D)
        target.append(traj.target)

    states = np.array(states)
    actions = np.array(actions)
    initials = np.array(initials)
    P_param = np.array(P)
    D_param = np.array(D)
    target = np.array(target)

    N, T, D = states.shape

    predictions = {key: [states[:, 0, :]] for key in models}
    currents = {key: states[:, 0, :] for key in models}
    for i in range(1, T):
        groundtruth = states[:, i]
        for key in models:
            model = models[key]

            if 't' in key:
                prediction = model.predict(np.hstack((initials, i*np.ones((N, 1)), P_param, D_param, target)))
                prediction = np.array(prediction.detach())
            else:
                prediction = model.predict(np.hstack((currents[key], actions[:, i - 1, :])))
                prediction = np.array(prediction.detach())
            # if 'p' in key:
            #     prediction = prediction[:, :D // 2]

            predictions[key].append(prediction.squeeze())
            MSEs[key].append(np.square(groundtruth - prediction).mean(axis=1))
            currents[key] = prediction.squeeze()
            # print(currents[key].shape)

    MSEs = {key: np.array(MSEs[key]).transpose() for key in MSEs}
    predictions = {key: np.array(predictions[key]).transpose([1, 0, 2]) for key in MSEs}

    outcomes = {'mse': MSEs, 'predictions': predictions}
    return outcomes


def unpack_config_models(cfg):
    """
    Reads the config to decide which models to use
    """
    model_types = []
    if cfg.experiment.models.single.train_traj:
        model_types.append('t')
    if cfg.experiment.models.single.train_det:
        model_types.append('d')
    if cfg.experiment.models.single.train_prob:
        model_types.append('p')
    if cfg.experiment.models.single.train_prob_traj:
        model_types.append('tp')
    if cfg.experiment.models.ensemble.train_traj:
        model_types.append('te')
    if cfg.experiment.models.ensemble.train_det:
        model_types.append('de')
    if cfg.experiment.models.ensemble.train_prob:
        model_types.append('pe')
    if cfg.experiment.models.ensemble.train_prob_traj:
        model_types.append('tpe')

    return model_types


@hydra.main(config_path='conf/config.yaml')
def evaluate(cfg):
    # print("here")
    graph_file = 'graphs'
    os.mkdir(graph_file)

    # Load test data
    log.info(f"Loading default data")
    (_, test_data) = torch.load(
        hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    # Load models
    model_types = cfg.plotting.models
    models = {}
    for model_type in model_types:
        models[model_type] = torch.load(hydra.utils.get_original_cwd() + '/models/reacher/' + model_type + cfg.model_dir)

    # Plot losses
    # TODO: this
    # log.info("Plotting loss")
    # for model_type in models:
    #     model = models[model_type]
    #     if type(model) == Model:
    #         loss_log = model.loss_log
    #         plot_loss(loss_log, save_loc=graph_file, show=False, s=model_type)
    #         plot_loss_epoch(loss_log, save_loc=graph_file, show=False, s=model_type)
    #     else:
    #         loss_log = model.acctrain
    #         test_log = model.acctest
    #         plot_loss(loss_log, save_loc=graph_file, show=False, s=model_type)


    # Evaluate models
    outcomes = test_models(test_data, models)

    # Plot shit
    MSEs, predictions = outcomes['mse'], outcomes['predictions']
    MSE_avg = {key: np.average(MSEs[key], axis=0) for key in MSEs}

    log.info("Plotting states")
    for i in range(len(test_data)):
        gt = test_data[i].states
        mse = {key: MSEs[key][i] for key in MSEs}
        pred = {key: predictions[key][i] for key in predictions}

        file = "%s/test%d" % (graph_file, i + 1)
        os.mkdir(file)

        plot_states(gt, pred, idx_plot=[0, 1, 2, 3, 4, 5, 6], save_loc=file, show=False)
        plot_mse(mse, save_loc=file, show=False)

    # plot_mse(MSE_avg, save_loc=file+"/mse_avg")





if __name__ == '__main__':
    sys.exit(evaluate())
