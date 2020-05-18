"""
The purpose of this file is to load in pre-loaded data and pre-trained models and evaluate them
"""

import sys

import hydra
import logging
import itertools

import torch
import numpy as np

from plot import *

log = logging.getLogger(__name__)


def test_models(test_data, models, verbose=False):
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
    P_param = P_param.reshape((len(test_data),-1))
    D_param = np.array(D)
    D_param = D_param.reshape((len(test_data),-1))
    target = np.array(target)
    target = target.reshape((len(test_data),-1))

    N, T, D = states.shape

    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, models[key].state_indices]] for key in models}
    currents = {key: states[:, 0, models[key].state_indices] for key in models}

    ind_dict = {}
    for i, key in list(enumerate(models)):
        if verbose and (i+1) % 10 == 0:
            print("    " + str(i+1))
        model = models[key]
        indices = model.state_indices
        traj = model.traj

        ind_dict[key] = indices

        for i in range(1, T):
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
            currents[key] = prediction.squeeze()

    predictions = {key: np.array(predictions[key]).transpose([1, 0, 2]) for key in predictions}
    MSEs = {key: np.square(states[:, :, ind_dict[key]] - predictions[key]).mean(axis=2) for key in predictions}

    # MSEs = {key: np.array(MSEs[key]).transpose() for key in MSEs}
    # if N > 1:
    #     predictions = {key: np.array(predictions[key]).transpose([1,0,2]) for key in predictions} # vectorized verion
    # else:
    #     predictions = {key: np.stack(predictions[key]).squeeze() for key in predictions}

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
        if 't' in key or type(key) == tuple and 't' in key[0]:
            # This doesn't make sense for t models so not gonna bother with this
            continue
        else:
            input = np.dstack((states[:, :, indices], actions)).reshape(N*T, -1)
            prediction = model.predict(input)
            prediction = np.array(prediction.detach()).reshape((N, T, len(indices)))
            delta = prediction-states[:, :, indices]
            deltas[key] = delta

    return deltas


def num_eval(gt, predictions, models, setting='gaussian', T_range=10000, verbose=False):
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

    out = {}
    for i, model_type in list(enumerate(models)):
        if (i+1) % 10 == 0 and verbose:
            print(i+1)
        gt_subset = gt[:, :, models[model_type].state_indices]
        if setting == 'dot':
            N, T, D = gt_subset.shape
            gt_norm = gt_subset / np.linalg.norm(gt_subset, axis=1).reshape((N, 1, D))
            prediction_norm = predictions[model_type] / np.linalg.norm(predictions[model_type], axis=1).reshape((N, 1, D))
            out[model_type] = np.sum(prediction_norm * gt_norm, axis=(1, 2)) / D
        elif setting == 'mse':
            out[model_type] = np.mean((predictions[model_type]-gt_subset)**2, axis=(1, 2))
        elif setting == 'gaussian':
            diff = 5 * (gt_subset - predictions[model_type])
            gauss = np.exp(-1 * np.square(diff))
            out[model_type] = np.mean(gauss, axis=(1, 2))
            # diff_dict = {key: gt_subset - predictions[key] for key in predictions}
            # gauss_dict = {key: np.exp(-1 * np.square(diff_dict[key])) for key in diff_dict}
            # out = {key: np.mean(gauss_dict[key], axis=(1, 2)) for key in gauss_dict}
        else:
            raise ValueError("Invalid setting: " + setting)
    return out


@hydra.main(config_path='conf/eval.yaml')
def evaluate(cfg):
    # print("here")
    lorenz = cfg.env == 'lorenz'
    graph_file = 'Plots'
    os.mkdir(graph_file)

    if not lorenz:
        # Load test data
        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(
            hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

        # Load models
        log.info("Loading models")
        if cfg.plotting.copies:
            model_types = list(itertools.product(cfg.plotting.models, np.arange(cfg.plotting.copies)))
        else:
            model_types = cfg.plotting.models
        models = {}
        f = hydra.utils.get_original_cwd() + '/models/reacher/'
        if cfg.exper_dir:
            f = f + cfg.exper_dir + '/'
        for model_type in model_types:
            model_str = model_type if type(model_type) == str else ('%s_%d' % model_type)
            models[model_type] = torch.load(f + model_str + ".dat")

    if lorenz:
        # Load test data
        log.info(f"Loading default data")
        (train_data, test_data) = torch.load(hydra.utils.get_original_cwd() + '/trajectories/lorenz/' + 'raw' + cfg.data_dir_lorenz)

        # Load models
        log.info("Loading models")
        model_types = cfg.plotting.models
        models = {}
        f = hydra.utils.get_original_cwd() + '/models/lorenz/'
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
        # idx = np.random.randint(0, len(data), num)
        idx = np.random.choice(np.arange(len(data)), size=num, replace=False)
        dat = [data[i] for i in idx]

        MSEs, predictions = test_models(dat, models)
        if cfg.plotting.sorted:
            deltas = find_deltas(dat, models)

        mse_evald = []
        sh = MSEs[model_types[0]][0].shape
        for i, id in list(enumerate(idx)):
            gt = data[id].states
            if cfg.plotting.copies:
                mse_all = {key: np.zeros((cfg.plotting.copies,) + sh) for key in cfg.plotting.models}
                for type, j in MSEs:
                    mse_all[type][j] = MSEs[(type, j)][i]
                mse = {key: np.median(mse_all[key], axis=0) for key in mse_all}
            else:
                mse = {key: MSEs[key][i].squeeze() for key in MSEs}
            mse_sub = {key: [(x if x < 10 ** 5 else float("nan")) for x in mse[key]] for key in mse}
            if not cfg.plotting.copies:
                pred = {key: predictions[key][i] for key in predictions}

            if cfg.plotting.all:
                file = "%s/test%d" % (graph_file, i + 1)
                os.mkdir(file)

            if cfg.plotting.states:
                plot_states(gt, pred, idx_plot=[0,1,2,3], save_loc=file+"/predictions", show=False)
            if cfg.plotting.mse:
                plot_mse(mse_sub, save_loc=file+"/mse.pdf", show=True)
            if cfg.plotting.sorted:
                ds = {key: deltas[key][i] for key in deltas}
                plot_sorted(gt, ds, idx_plot=[0,1,2,3], save_loc=file+"/sorted", show=False)

            mse_evald.append(mse)

        plot_mse_err(mse_evald, save_loc=("%s/Err Bar MSE of Predictions" % graph_file),
                     show=False, y_max=cfg.plotting.mse_y_max)

        mse_all = {}

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
