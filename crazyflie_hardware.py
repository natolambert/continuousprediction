import sys
import warnings
import os

import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from dotmap import DotMap

import mujoco_py
import torch

import gym
from envs import *
from gym.wrappers import Monitor

import hydra
import logging

log = logging.getLogger(__name__)

from policy import randomPolicy
from plot import plot_ss, plot_loss, setup_plotting

from dynamics_model import DynamicsModel
from reacher_pd import run_controller


###########################################
#                Datasets                 #
###########################################

def convert_pwm(packed_pwm):
    packed_pwm = packed_pwm[0]
    m1 = (packed_pwm & 0xFF) << 8
    m2 = ((packed_pwm >> 8) & 0xFF) << 8
    m3 = ((packed_pwm >> 16) & 0xFF) << 8
    m4 = ((packed_pwm >> 24) & 0xFF) << 8
    ret = np.zeros((1, 4)).squeeze()
    ret[0] = m1
    ret[1] = m2
    ret[2] = m3
    ret[3] = m4
    return ret


def create_dataset_traj(data, control_params=False, train_target=True, threshold=0.0, delta=False, t_range=0):
    """
    Creates a dataset with entries for PID parameters and number of
    timesteps in the future

    Parameters:
    -----------
    data: An array of dotmaps where each dotmap has info about a trajectory
    threshold: the probability of dropping a given data entry
    """
    data_in, data_out = [], []
    for sequence in data:
        states = sequence[0]
        # K = sequence.K
        n = states.shape[0]
        for i in range(n):  # From one state p
            for j in range(i, n):
                # This creates an entry for a given state concatenated
                # with a number t of time steps as well as the PID parameters

                dat = [states[i], j - i]
                # dat.append(K)
                data_in.append(np.hstack(dat))
                # data_in.append(np.hstack((states[i], j-i, target)))
                if delta:
                    data_out.append(states[j] - states[i])
                else:
                    data_out.append(states[j])

    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def create_dataset_step(data, delta=True, t_range=0):
    """
    Creates a dataset for learning how one state progresses to the next

    Parameters:
    -----------
    data: A 2d np array. Each row is a state
    """
    data_in = []
    data_out = []
    for sequence in data:
        states = sequence[0]
        # if t_range:
        #     states = states[:t_range]
        actions = sequence[1]
        targets = sequence[2]
        for i in range(states.shape[0] - 1):
            # if t_range:
            #     actions = actions[:t_range]
            data_in.append(np.hstack((states[i], actions[i])))

            data_out.append(targets[i])
            # if delta:
            #     data_out.append(states[i + 1] - states[i])
            # else:
            #     data_out.append(states[i + 1])
    data_in = np.array(data_in)
    data_out = np.array(data_out)

    return data_in, data_out


def log_hyperparams(cfg):
    log.info(cfg.model.str + ":")
    log.info("  hid_width: %d" % cfg.model.training.hid_width)
    log.info('  hid_depth: %d' % cfg.model.training.hid_depth)
    log.info('  epochs: %d' % cfg.model.optimizer.epochs)
    log.info('  batch size: %d' % cfg.model.optimizer.batch)
    log.info('  optimizer: %s' % cfg.model.optimizer.name)
    log.info('  learning rate: %f' % cfg.model.optimizer.lr)


###########################################
#             Main Functions              #
###########################################

def get_datasets(df):
    # split df into trajectories
    states = ['omegax_0tx', 'omegay_0tx', 'omegaz_0tx', 'pitch_0tx', 'roll_0tx', 'yaw_0tx', 'linax_0tx', 'linay_0tx',
              'linyz_0tx']  #
    actions = ['m1pwm_0tu', 'm2pwm_0tu', 'm3pwm_0tu', 'm4pwm_0tu']

    states = df[['roll', 'pitch', 'yaw']].to_numpy()
    actions = df[['pwms']].to_numpy()
    actions = np.stack([convert_pwm(a) for a in actions])

    start = 150
    l = 1000
    n = np.shape(states)[0]
    trajs = []
    for i in range(int((n - start) / l)):
        idx = i * l + start
        s = states[idx:idx + l + 1, :]
        a = actions[idx:idx + l + 1, :]
        d = states[1 + idx:idx + l + 2, :] - states[idx:idx + l + 1, :]
        trajs.append((s, a, d))
    # deltas = []
    data_train = trajs  # trajs[1::3]+trajs[2::3]
    data_test = []  # trajs[::3]
    return data_train, data_test


@hydra.main(config_path='conf/crazyflie_pd.yaml', strict=False)
def contpred(cfg):
    # Collect data
    if cfg.mode == 'collect':
        raise ValueError("This file is for validation only, not collection")
    # Load data
    else:
        log.info(f"Loading default data")
        # raise ValueError("Current Saved data old format")
        # Todo re-save data
        import pandas as pd
        raw_data = pd.read_csv(hydra.utils.get_original_cwd() + '/trajectories/crazyflie/' + 'cf2.csv',
                               error_bad_lines=False, delimiter=',')  # 'cf.csv')

    # raw_data = raw_data[::2]
    # from plot import plot_cf
    # plot_cf(raw_data[['pitch','pitch','pitch','pitch','roll','yaw']].to_numpy(), [])

    data_train, data_test = get_datasets(raw_data)
    if cfg.mode == 'train':
        it = range(cfg.copies) if cfg.copies else [0]
        prob = cfg.model.prob
        traj = cfg.model.traj
        ens = cfg.model.ensemble
        delta = cfg.model.delta

        log.info(f"Training model P:{prob}, T:{traj}, E:{ens}")

        log_hyperparams(cfg)

        for i in it:
            print('Training model %d' % i)
            if traj:
                dataset = create_dataset_traj(data_train, control_params=False,
                                              train_target=False,
                                              threshold=cfg.model.training.filter_rate,
                                              t_range=cfg.model.training.t_range)
            else:
                dataset = create_dataset_step(data_train, delta=delta)

            cfg.env.param_size = 0
            cfg.env.target_size = 0
            model = DynamicsModel(cfg, env="SS")
            train_logs, test_logs = model.train(dataset, cfg)

            # setup_plotting({cfg.model.str: model})
            # plot_loss(train_logs, test_logs, cfg, save_loc=cfg.env.name + '-' + cfg.model.str, show=False)

            log.info("Saving new default models")
            f = hydra.utils.get_original_cwd() + '/models/crazyflie_exp/'
            if cfg.exper_dir:
                f = f + cfg.exper_dir + '/'
                if not os.path.exists(f):
                    os.mkdir(f)
            copystr = "_%d" % i if cfg.copies else ""
            f = f + cfg.model.str + copystr + '.dat'
            torch.save(model, f)
        # torch.save(model, "%s_backup.dat" % cfg.model.str) # save backup regardless

    if cfg.mode == 'eval':
        model_types = ['t']  # ,'tp']
        # model_types = ['d', 'pe', 't']  # ,'tp']
        # model_types = ['p', 'tp']  # ,'tp']
        models = {}
        f = hydra.utils.get_original_cwd() + '/models/crazyflie_exp/'
        for model_type in model_types:
            models[model_type] = torch.load(f + model_type + ".dat")

        from plot import plot_mse_err, setup_plotting, plot_states
        setup_plotting(models)
        MSEs, predictions, variances = eval_exp(data_train, models)
        mse_evald = []
        sh = MSEs[model_types[0]][0].shape
        idx = np.arange(len(data_train))  # remember this line
        for i, id in list(enumerate(idx)):
            gt = data_train[id][0]
            pred = {key: predictions[key][i] for key in predictions}
            var = {key: variances[key][i] for key in variances}
            mse = {key: MSEs[key][i].squeeze() for key in MSEs}

            if True:
                # if
                plot_states(gt, pred, variances=var, idx_plot=[0, 1, 2], save_loc=f"predictions_{i}_", show=False)

            mse_evald.append(mse)

        plot_mse_err(mse_evald, save_loc=("Err Bar MSE of Predictions"),
                     show=True, legend=True)


def eval_exp(test_data, models, verbose=False, env=None, t_range=1000):
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

    # Compile the various trajectories into arrays
    for traj in test_data:
        states.append(traj[0][:t_range, :])
        actions.append(traj[1][:t_range, :])
        initials.append(traj[0][0, :])

    states = np.stack(states)
    actions = np.stack(actions)

    initials = np.array(initials)
    N, T, D = states.shape
    if len(np.shape(actions)) == 2:
        actions = np.expand_dims(actions, axis=2)
    # Iterate through each type of model for evaluation
    predictions = {key: [states[:, 0, models[key].state_indices]] for key in models}
    currents = {key: states[:, 0, models[key].state_indices] for key in models}

    from evaluate import forward_var
    variances = {key: [] for key in models}
    ind_dict = {}
    for i, key in list(enumerate(models)):
        if verbose and (i + 1) % 10 == 0:
            print("    " + str(i + 1))
        model = models[key]
        indices = model.state_indices
        traj = model.traj

        ind_dict[key] = indices

        for i in range(1, T):
            if i >= t_range:
                continue
            if traj:
                dat = [initials[:, indices], i * np.ones((N, 1))]
                prediction = np.array(model.predict(np.hstack(dat)).detach())
            else:
                if env == 'lorenz':
                    prediction = model.predict(np.array(currents[key]))
                    prediction = np.array(prediction.detach())
                else:
                    acts = actions[:, i - 1, :]
                    prediction = model.predict(np.hstack((currents[key], acts)))
                    prediction = np.array(prediction.detach())
            if model.prob:
                if traj:
                    f = np.hstack(dat)
                else:
                    f = np.hstack((currents[key], acts))
                var = forward_var(model, f).detach().numpy()
            else:
                var = np.zeros(np.shape(initials[:, indices]))

            predictions[key].append(prediction)
            currents[key] = prediction.squeeze()
            variances[key].append(var)

    variances = {key: np.stack(variances[key]).transpose([1, 0, 2]) for key in variances}
    predictions = {key: np.array(predictions[key]).transpose([1, 0, 2]) for key in predictions}

    # MSEs = {key: np.square(states[:, :, ind_dict[key]] - predictions[key]).mean(axis=2)[:, 1:] for key in predictions}

    MSEscaled = {}
    for key in predictions:
        # scaling of error
        # if env == 'crazyflie':
        #     # ind_dict[key] = [0,1,3,4,5]
        #     ind_dict[key] = [0, 1, 3, 4]
        if t_range < np.shape(states)[1]:
            l = t_range
        else:
            l = np.shape(states)[1]
        min_states = np.min(states[:, :l, ind_dict[key]], axis=(0, 1))
        max_states = np.ptp(states[:, :l, ind_dict[key]], axis=(0, 1))
        scaled_states = (states[:, :l, ind_dict[key]] - min_states) / max_states
        scaled_pred = (predictions[key][:, :, :] - min_states) / max_states
        MSEscaled[key] = np.square(scaled_states - scaled_pred).mean(axis=2)[:, 1:]

    return MSEscaled, predictions, variances


if __name__ == '__main__':
    sys.exit(contpred())
