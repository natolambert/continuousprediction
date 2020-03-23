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
    Tests each of the models in the dictionary "models" on each of the trajectories in test_data.
    Note: this function uses Numpy arrays to handle multiple tests at once efficiently

    Parameters:
    ------------
    test_data: the trajectories to test on, N trajectories
    models: a dictionary of models to test, M models

    Returns:
     MSEs:           MSEs['x'] is a 2D array where the (i,j)th is the MSE for
                        the prediction at time j with the ith test trajectory
                        corresponding to model 'x'
     predictions:   predictions['x'] is a 3D array where the (i,j)th element
                        is an array with the predicted state at time j for the ith
                        test trajectory corresponding to model 'x'
    """

    log.info("Beginning testing of predictions")

    MSEs = {key: [] for key in models}

    states, actions, initials = [], [], []
    P, D, target = [], [], []

    # Compile the various trajectories into arrays
    for traj in test_data:
        states.append(traj.states)
        actions.append(traj.actions)
        initials.append(traj.states[0, :])
        P.append(traj.P)
        D.append(traj.D)
        target.append(traj.target)

    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)
    initials = np.array(initials)
    P_param = np.array(P)
    D_param = np.array(D)
    target = np.array(target)

    N, T, D = states.shape

    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, :]] for key in models}
    currents = {key: states[:, 0, :] for key in models}
    for i in range(1, T):
        groundtruth = states[:, i]
        for key in models:
            model = models[key]
            # Make predictions on all trajectories at once
            #
            # Commented some lines out because they seemed to break the code
            if 't' in key:
                # prediction = model.predict(np.hstack((initials, i * np.ones((N, 1)), P_param.reshape(-1, 1),
                #                                       D_param.reshape(-1, 1), target.reshape(-1, 1))))
                prediction = model.predict(np.hstack((initials, i * np.ones((N, 1)), P_param, D_param, target)))
                prediction = np.array(prediction.detach())
            else:
                # if len(np.shape(actions)) == 1:
                #     prediction = model.predict((currents[key].reshape((1, -1))))
                # else:
                #     prediction = model.predict(np.hstack((currents[key].reshape((1, -1)), actions[:, i - 1, :])))
                prediction = model.predict(np.hstack((currents[key], actions[:, i - 1, :])))
                prediction = np.array(prediction.detach())

            predictions[key].append(prediction)
            MSEs[key].append(np.square(groundtruth - prediction.squeeze()).mean(axis=1))
            currents[key] = prediction.squeeze()
            # print(currents[key].shape)

    MSEs = {key: np.array(MSEs[key]).transpose() for key in MSEs}
    if N > 1:
        predictions = {key: np.array(predictions[key]).transpose([1,0,2]) for key in predictions} # vectorized verion
    else:
        predictions = {key: np.stack(predictions[key]).squeeze() for key in predictions}

    # outcomes = {'mse': MSEs, 'predictions': predictions}
    return MSEs, predictions


def test_traj_ensemble(ensemble, test_data):
    """
    TODO: decide if this is useful or remove
    Tests each model in the ensemble on one test trajectory and plots the output
    """
    traj = test_data
    states = traj.states
    actions = traj.actions
    initial = states[0, :]

    model_predictions = [[] for _ in range(ensemble.n)]
    ensemble_predictions = []
    for i in range(1, states.shape[0]):
        x = np.hstack((initial, i, traj.P, traj.D, traj.target))
        ens_pred = ensemble.predict(x)
        ensemble_predictions.append(ens_pred.squeeze())
        for j in range(len(ensemble.models)):
            model = ensemble.models[j]
            model_pred = model.predict(x)
            model_predictions[j].append(model_pred.squeeze())

    ensemble_predictions = np.array(ensemble_predictions)
    model_predictions = [np.array(x) for x in model_predictions]
    # print(len(model_predictions))

    for i in range(7):
        fig, ax = plt.subplots()
        gt = states[:, i]
        plt.title("Predictions on one dimension")
        plt.xlabel("Timestep")
        plt.ylabel("State Value")
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        plt.plot(gt, c='k', label='Groundtruth')
        plt.plot(ensemble_predictions[:, i])
        for pred in model_predictions:
            # print(pred.shape)
            plt.plot(pred[:, i], c='b')

        plt.legend()

        plt.show()


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


@hydra.main(config_path='conf/eval.yaml')
def evaluate(cfg):
    # print("here")
    graph_file = 'Plots'
    os.mkdir(graph_file)

    # Load test data
    log.info(f"Loading default data")
    (train_data, test_data) = torch.load(
        hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    # Load models
    model_types = cfg.plotting.models
    models = {}
    for model_type in model_types:
        models[model_type] = torch.load(hydra.utils.get_original_cwd() + '/models/reacher/' + model_type + ".dat")

    log.info("Plotting states")
    mse_evald = []

    # Select a random subset of testing data
    idx = np.random.randint(0, len(test_data), cfg.plotting.num_eval)
    dat = [test_data[i] for i in idx]

    # Find MSEs and predictions from all test trajectories at once
    MSEs, predictions = test_models(dat, models)

    # Plot
    log.info("Plotting states")
    for i, id in list(enumerate(idx)):
        gt = test_data[id].states
        mse = {key: MSEs[key][i].squeeze() for key in MSEs}
        # mse_sub = {key: mse[mse[key] < 10 ** 5] for key in mse}
        mse_sub = {key: [(x if x < 10 ** 5 else float("nan")) for x in mse[key]] for key in mse}
        pred = {key: predictions[key][i] for key in predictions}

        file = "%s/test%d" % (graph_file, i + 1)
        os.mkdir(file)

        if cfg.plotting.states:
            plot_states(gt, pred, idx_plot=[0,1,2,3], save_loc=file+"/predictions", show=False)
        if cfg.plotting.mse:
            plot_mse(mse_sub, save_loc=file+"/mse.pdf", show=False)
        if cfg.plotting.sorted:
            plot_sorted(gt, pred, idx_plot=[0,1,2,3], save_loc=file+"/sorted", show=False)

        mse_evald.append(mse)

    # for i in range(cfg.plotting.num_eval):
    #     # Evaluate models
    #     idx = np.random.randint(0, len(train_data))
    #     outcomes = test_models([test_data[idx]], models)
    #
    #     # idx = np.random.randint(0, len(test_data))
    #     # outcomes = test_models([test_data[idx]], models)
    #
    #     # Plot shit
    #     # TODO account for numerical errors with predictions
    #     MSEs, predictions = outcomes['mse'], outcomes['predictions']
    #     MSE_avg = {key: np.average(MSEs[key], axis=0) for key in MSEs}
    #
    #     gt = test_data[idx].states
    #     mse = {key: MSEs[key].squeeze() for key in MSEs}
    #     mse_sub = {key: mse[key][mse[key] < 10 ** 5] for key in mse}
    #     pred = {key: predictions[key] for key in predictions}
    #
    #
    #     # plot_states(gt, pred, save_loc="Predictions; traj-" + str(idx), idx_plot=[0,1,2,3], show=False)
    #     # plot_mse(mse_sub, save_loc="Error; traj-" + str(idx), show=False)
    #     mse_evald.append(mse)
    #
    plot_mse_err(mse_evald, save_loc="Err Bar MSE of Predictions", show=False)


if __name__ == '__main__':
    sys.exit(evaluate())
