"""
The purpose of this file is to load in pre-loaded data and pre-trained models and evaluate them
"""

import sys

import hydra
import logging

import torch
import numpy as np

from plot import *

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
    # eval_indices = list(set.intersection(*[models[key].state_indices for key in models]))
    # eval_indices.sort()

    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, models[key].state_indices]] for key in models}
    currents = {key: states[:, 0, models[key].state_indices] for key in models}
    for i in range(1, T):
        groundtruth = states[:, i]
        for key in models:
            model = models[key]
            indices = model.state_indices
            traj = 't' in key or type(key) == tuple and 't' in key[0]
            # Make predictions on all trajectories at once
            if traj:
                dat = [initials[:, indices], i * np.ones((N, 1))]
                if model.control_params:
                    dat.extend([P_param, D_param])
                if model.train_target:
                    dat.append(target)
                prediction = np.array(model.predict(np.hstack(dat)).detach())
            else:
                prediction = model.predict(np.hstack((currents[key], actions[:, i - 1, :])))
                prediction = np.array(prediction.detach())

            predictions[key].append(prediction)
            MSEs[key].append(np.square(groundtruth[:, indices] - prediction.squeeze()).mean(axis=1))
            currents[key] = prediction.squeeze()

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


def find_deltas(test_data, models):
    """
    For sorted delta plots. Tests each model in 'models' on each test trajectory in 'test_data',
    finding the predicted deltas. The difference between this method and the standard test_models
    is that with this one, each prediction starts from a ground truth value

    Parameters:
        test_data: the N test trajectories to test on
        models: the M models to evaluate

    Returns:
        tbd

    """
    states, actions = [], []
    P, D, target = [], [], []

    # Compile the various trajectories into arrays
    for traj in test_data:
        states.append(traj.states)
        actions.append(traj.actions)

    # Convert to numpy arrays
    states = np.array(states)
    actions = np.array(actions)

    N, T, D = states.shape

    # Iterate through each type of model for evaluation
    deltas = {}
    for key in models:
        model = models[key]
        indices = model.state_indices
        if 't' in key:
            # This doesn't make sense for t models so not gonna bother with this
            continue
        else:
            input = np.dstack((states[:, :, indices], actions)).reshape(N*T, -1)
            prediction = model.predict(input)
            prediction = np.array(prediction.detach()).reshape(N, T, len(indices))
            delta = prediction-states[:, :, indices]
            deltas[key] = delta

    return deltas


def num_eval(gt, predictions, setting='dot', T_range=10000):
    """
    Evaluates the predictions in a way that creates one number

    Parameters:
        gt: NxTxD array of ground truth values
        predictions: a dictionary of NxTxD arrays of predictions from models
        setting: currently 'dot', 'mse'
            'dot': average over dimensions of dot product between ground truth and trajectories
            'mse': average over dimensions of MSE

    Returns:
        outputs: a dictionary of arrays of length N of evaluation
    """
    gt = gt[:, :T_range, :]
    predictions = {key: predictions[key][:, :T_range, :] for key in predictions}

    if setting == 'dot':
        N, T, D = gt.shape
        gt_norm = gt / np.linalg.norm(gt, axis=1).reshape((N, 1, D))
        predictions_norm = {key: predictions[key] / np.linalg.norm(predictions[key], axis=1).reshape((N, 1, D)) for key in predictions}
        return {key: np.sum(predictions_norm[key] * gt_norm, axis=(1, 2)) / D for key in predictions_norm}
    if setting == 'mse':
        return {key: np.mean((predictions[key]-gt)**2, axis=(1,2)) for key in predictions}
    raise ValueError("Invalid setting: " + setting)


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
    log.info("Loading models")
    model_types = cfg.plotting.models
    models = {}
    f = hydra.utils.get_original_cwd() + '/models/reacher/'
    if cfg.exper_dir:
        f = f + cfg.exper_dir + '/'
    for model_type in model_types:
        models[model_type] = torch.load(f + model_type + ".dat")

    # Plot
    def plot_helper(data, num, graph_file):
        """
        Helper to allow plotting for both train and test data without significant code duplication
        """
        os.mkdir(graph_file)
        setup_plotting(models)

        # Select a random subset of training data
        idx = np.random.randint(0, len(data), num)
        dat = [data[i] for i in idx]

        MSEs, predictions = test_models(dat, models)
        if cfg.plotting.sorted:
            deltas = find_deltas(dat, models)

        mse_evald = []
        for i, id in list(enumerate(idx)):
            gt = data[id].states
            mse = {key: MSEs[key][i].squeeze() for key in MSEs}
            mse_sub = {key: [(x if x < 10 ** 5 else float("nan")) for x in mse[key]] for key in mse}
            pred = {key: predictions[key][i] for key in predictions}

            file = "%s/test%d" % (graph_file, i + 1)
            os.mkdir(file)

            if cfg.plotting.states:
                plot_states(gt, pred, idx_plot=[0,1,2,3], save_loc=file+"/predictions", show=False)
            if cfg.plotting.mse:
                plot_mse(mse_sub, save_loc=file+"/mse.pdf", show=False)
            if cfg.plotting.sorted:
                ds = {key: deltas[key][i] for key in deltas}
                plot_sorted(gt, ds, idx_plot=[0,1,2,3], save_loc=file+"/sorted", show=False)

            mse_evald.append(mse)

        plot_mse_err(mse_evald, save_loc=("%s/Err Bar MSE of Predictions" % graph_file), show=False)

    if cfg.plotting.num_eval_train:
        log.info("Plotting train data")

        file = graph_file + "/train_data"

        plot_helper(train_data, cfg.plotting.num_eval_train, file)

    if cfg.plotting.num_eval_test:
        log.info("Plotting test data")

        file = graph_file + '/test_data'

        plot_helper(test_data, cfg.plotting.num_eval_test, file)


if __name__ == '__main__':
    sys.exit(evaluate())
