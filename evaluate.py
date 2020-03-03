"""
The purpose of this file is to load in pre-loaded data and pre-trained models and evaluate them
"""

import sys

import hydra
import logging

import torch
import numpy as np

log = logging.getLogger(__name__)

def test_models(traj, models):
    """
    Tests each of the models in the dictionary "models" on the same trajectory.
    Not to be confused with the function below, which tests a single model

    Parameters:
    ------------
    traj: the trajectory to test on
    models: a dictionary of models to test

    Returns:
    outcomes: a dictionary of MSEs and predictions. As an example of how to
              get info from this distionary, to get the MSE data from a trajectory
              -based model you'd do
                    outcomes['mse']['t']
    """
    log.info("Beginning testing of predictions")

    MSEs = {key: [] for key in models}

    states = traj.states
    actions = traj.actions
    initial = states[0, :]

    predictions = {key: [states[0, :]] for key in models}
    currents = {key: states[0, :] for key in models}
    for i in range(1, states.shape[0]):
        groundtruth = states[i]
        for key in models:
            model = models[key]

            if 't' in key:
                prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
            else:
                prediction = model.predict(np.concatenate((currents[key], actions[i - 1, :])))
            if 'p' in key:
                prediction = prediction[:, :prediction.shape[1] // 2]

            predictions[key].append(prediction.squeeze())
            MSEs[key].append(np.square(groundtruth - prediction).mean())
            currents[key] = prediction.squeeze()
            # print(currents[key].shape)

    MSEs = {key: np.array(MSEs[key]) for key in MSEs}
    predictions = {key: np.array(predictions[key]) for key in MSEs}

    outcomes = {'mse': MSEs, 'predictions': predictions}
    return outcomes


def test_model(traj, model):
    """
    Tests a single model on the trajectory given

    Parameters:
    -----------
    traj: a trajectory to test on
    model: a model object to evaluate

    Returns:
    --------
    outcomes: a dictionary of MSEs and predictions:
                {'mse': MSEs, 'predictions': predictions}
    """
    log.info("Beginning testing of predictions")

    states = traj.states
    actions = traj.actions
    initial = states[0, :]
    key = model.str

    MSEs = []
    predictions = [states[0, :]]
    current = states[0, :]
    for i in range(1, states.shape[0]):
        groundtruth = states[i]

        if 't' in key:
            prediction = model.predict(np.hstack((initial, i, traj.P, traj.D, traj.target)))
        else:
            prediction = model.predict(np.concatenate((current, actions[i - 1, :])))
        if 'p' in key:
            prediction = prediction[:, :prediction.shape[1] // 2]

        predictions.append(prediction.squeeze())
        MSEs.append(np.square(groundtruth - prediction).mean())
        current = prediction.squeeze()

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


    # Load test data
    log.info(f"Loading default data")
    (exper_data, test_data) = torch.load(
        hydra.utils.get_original_cwd() + '/trajectories/reacher/' + 'raw' + cfg.data_dir)

    # Load models
    model_types = unpack_config_models(cfg)

    # Evaluate models


    # Plot shit

    pass




if __name__=='main':
    sys.exit(evaluate())
